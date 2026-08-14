"""Flask server serving the ZigLLM Studio HTML UI and REST API."""
from flask import Flask, request, jsonify, send_from_directory, Response
import os, json, threading, time, uuid, traceback
from pathlib import Path

app = Flask(__name__, static_folder=".", static_url_path="")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# In-memory store for running/training job logs
_job_logs = {}

def _log(job_id, msg):
    _job_logs.setdefault(job_id, []).append({
        "t": time.time(),
        "m": msg,
    })

def _get_log(job_id):
    return _job_logs.get(job_id, [])


# ── Static file serving ─────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(".", "index.html")


# ── Health / status ─────────────────────────────────────────────────
@app.route("/api/health")
def health():
    try:
        import torch
        cuda = torch.cuda.is_available()
        gpu = torch.cuda.get_device_name(0) if cuda else None
    except Exception:
        cuda = None
        gpu = None
    return jsonify({
        "ok": True,
        "cuda": cuda,
        "gpu": gpu,
        "version": "0.3.0",
    })


# ── Train ───────────────────────────────────────────────────────────
@app.route("/api/train", methods=["POST"])
def train():
    """Launch a training job in a background thread. Returns job_id immediately."""
    data = request.json or {}
    job_id = str(uuid.uuid4())[:8]
    _job_logs[job_id] = []

    def _run():
        try:
            from ..config import RunConfig, Architecture, RunMode, Adapter, Device
            from ..engine import Trainer

            def progress(msg):
                _log(job_id, msg)

            cfg = RunConfig(
                model_id=data.get("model_id", "Qwen/Qwen2.5-0.5B"),
                architecture=Architecture(data.get("architecture", "transformer")),
                mode=RunMode(data.get("mode", "finetune")),
                adapter=Adapter(data.get("adapter", "lora")),
                device=Device(data.get("device", "auto")),
                dataset_id=data.get("dataset_id", ""),
                dataset_split=data.get("dataset_split", "train"),
                text_column=data.get("text_column", "text"),
                seq_len=int(data.get("seq_len", 1024)),
                batch_size=int(data.get("batch_size", 1)),
                grad_accum=int(data.get("grad_accum", 8)),
                epochs=float(data.get("epochs", 1.0)),
                learning_rate=float(data.get("learning_rate", 2e-4)),
                lora_rank=int(data.get("lora_rank", 16)),
                lora_alpha=int(data.get("lora_alpha", 32)),
                lora_dropout=float(data.get("lora_dropout", 0.05)),
                gradient_checkpointing=bool(data.get("gradient_checkpointing", False)),
                flash_attention=bool(data.get("flash_attention", True)),
                torch_compile=bool(data.get("torch_compile", False)),
                num_workers=int(data.get("num_workers", 2)),
                warmup_ratio=float(data.get("warmup_ratio", 0.03)),
                streaming=bool(data.get("streaming", False)),
                compile_mode=data.get("compile_mode", "reduce-overhead"),
            )
            progress("Job started")
            result = Trainer(cfg, dataset_provider=data.get("provider", "huggingface")).run(progress=progress)
            _log(job_id, f"✓ Training complete. Output: {result}")
            _log(job_id, "JOB_DONE")
        except Exception as e:
            _log(job_id, f"✕ ERROR: {e}")
            _log(job_id, traceback.format_exc())
            _log(job_id, "JOB_DONE")

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"ok": True, "job_id": job_id})


@app.route("/api/job/<job_id>")
def job_log(job_id):
    """Stream log entries for a training job."""
    logs = _get_log(job_id)
    done = any(e["m"] == "JOB_DONE" for e in logs)
    return jsonify({"ok": True, "job_id": job_id, "logs": logs, "done": done})


# ── Benchmark ───────────────────────────────────────────────────────
@app.route("/api/benchmark", methods=["POST"])
def benchmark():
    data = request.json or {}
    try:
        from ..benchmarks import run_benchmark
        result = run_benchmark(
            data.get("name", "hellaswag"),
            data.get("model_id", "Qwen/Qwen2.5-0.5B"),
            int(data.get("limit", 0)),
            data.get("device", "auto"),
        )
        return jsonify({"ok": True, "result": result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


# ── Search datasets ─────────────────────────────────────────────────
@app.route("/api/search-datasets", methods=["POST"])
def search_datasets():
    data = request.json or {}
    try:
        from ..datasets import search_datasets
        results = search_datasets(
            query=data.get("query", ""),
            sort=data.get("sort", "downloads"),
            limit=int(data.get("limit", 20)),
            task=data.get("task", ""),
            language=data.get("language", ""),
            author=data.get("author", ""),
        )
        out = [
            {
                "id": r.id,
                "author": r.author,
                "downloads": r.downloads,
                "likes": r.likes,
                "tags": r.tags[:5],
                "description": r.description[:200],
                "url": r.url,
            }
            for r in results
        ]
        return jsonify({"ok": True, "results": out})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


# ── Create dataset ─────────────────────────────────────────────────
@app.route("/api/create-dataset/preview", methods=["POST"])
def create_dataset_preview():
    data = request.json or {}
    try:
        from ..creator import DatasetBuilder
        builder = DatasetBuilder(name=data.get("name", "preview"))
        if data.get("text"):
            lines = [l.strip() for l in data["text"].split("\n") if l.strip()]
            builder.add_texts(lines, source="text-input")
        if data.get("files"):
            builder.add_files(data["files"])
        if data.get("urls"):
            url_list = [u.strip() for u in data["urls"].split("\n") if u.strip()]
            builder.add_urls(url_list, selector=data.get("selector"))
        if data.get("dedupe"):
            builder.dedupe()
        if data.get("min_chars", 0) > 0:
            builder.filter_min_length(int(data["min_chars"]))
        if data.get("max_chars", 0) > 0:
            builder.filter_max_length(int(data["max_chars"]))
        samples = builder.preview(5)
        return jsonify({
            "ok": True,
            "count": len(builder.samples),
            "preview": samples,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/create-dataset/build", methods=["POST"])
def create_dataset_build():
    data = request.json or {}
    try:
        from ..creator import DatasetBuilder
        builder = DatasetBuilder(name=data["name"])
        if data.get("text"):
            lines = [l.strip() for l in data["text"].split("\n") if l.strip()]
            builder.add_texts(lines, source="text-input")
        if data.get("files"):
            builder.add_files(data["files"])
        if data.get("urls"):
            url_list = [u.strip() for u in data["urls"].split("\n") if u.strip()]
            builder.add_urls(url_list, selector=data.get("selector"))
        result = builder.build(
            output_path=data["output"],
            format=data.get("format", "jsonl"),
            dedupe=data.get("dedupe", True),
            min_chars=int(data.get("min_chars", 40)),
            max_chars=int(data.get("max_chars", 0)),
            split_ratios=(
                float(data.get("split_train", 0.9)),
                float(data.get("split_val", 0.05)),
                float(data.get("split_test", 0.05)),
            ),
        )
        return jsonify({"ok": True, "result": result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


# ── CSV to parquet ──────────────────────────────────────────────────
@app.route("/api/convert-csv", methods=["POST"])
def convert_csv():
    data = request.json or {}
    try:
        from ..datasets import csv_to_parquet
        result = csv_to_parquet(
            data["input"],
            data["output"],
            compression=data.get("compression", "zstd") if data.get("compression") != "none" else None,
            batch_size=int(data.get("batch_size", 100000)),
        )
        return jsonify({"ok": True, "result": result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


# ── Scrape ──────────────────────────────────────────────────────────
@app.route("/api/scrape", methods=["POST"])
def scrape():
    data = request.json or {}
    try:
        from ..datasets import scrape as do_scrape
        texts = do_scrape(data["url"], data.get("selector"))
        return jsonify({"ok": True, "count": len(texts), "texts": texts})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


# ── Build core ──────────────────────────────────────────────────────
@app.route("/api/build-core", methods=["POST"])
def build_core():
    import subprocess
    try:
        result = subprocess.run(
            ["zig", "build", "-Doptimize=ReleaseFast"],
            text=True, capture_output=True, cwd=str(PROJECT_ROOT),
        )
        ok = result.returncode == 0
        return jsonify({
            "ok": ok,
            "log": (f"✓ Built\n" if ok else "✕ Failed\n") + result.stdout + result.stderr,
        })
    except FileNotFoundError:
        return jsonify({"ok": False, "error": "Zig is not installed"}), 400


# ── Launch ──────────────────────────────────────────────────────────
def launch(host="0.0.0.0", port=7860, share=False):
    """Launch the web server.

    For Colab/Kaggle share tunnels, use ngrok or similar. Gradio's share=
    is not available with Flask.
    """
    notebook = bool(os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))
    actual_host = "0.0.0.0" if notebook else host
    print(f"ZigLLM Studio web UI: http://localhost:{port}")
    app.run(host=actual_host, port=port, debug=False, use_reloader=False)
