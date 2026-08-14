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


# ── Launch with tunnel support ──────────────────────────────────────
def launch(host="0.0.0.0", port=7860, share=False, tunnel="ngrok"):
    """Launch the web server with optional public tunnel.

    Args:
        host: Bind address (default: 0.0.0.0 for notebook environments)
        port: Local port (default: 7860)
        share: Create a public tunnel (default: False)
        tunnel: Tunnel provider: "ngrok", "cloudflared", or "localtunnel" (default: "ngrok")

    Examples:
        # Basic launch (localhost only)
        launch()

        # With ngrok tunnel (requires pyngrok)
        launch(share=True, tunnel="ngrok")

        # With cloudflared tunnel (free, no account)
        launch(share=True, tunnel="cloudflared")

        # With localtunnel (free, no account)
        launch(share=True, tunnel="localtunnel")
    """
    notebook = bool(os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))
    actual_host = "0.0.0.0" if notebook else host

    if share and notebook:
        print(f"Starting ZigLLM Studio on port {port} with {tunnel} tunnel...")
        threading.Thread(
            target=_start_tunnel,
            args=(tunnel, port),
            daemon=True,
        ).start()
    elif share:
        print("⚠ share=True only works in notebook environments (Colab/Kaggle)")

    print(f"ZigLLM Studio web UI: http://localhost:{port}")
    if share:
        print("Tunnel URL will be printed once the tunnel is established...")
    app.run(host=actual_host, port=port, debug=False, use_reloader=False)


def _start_tunnel(provider, port):
    """Start a tunnel in a background thread."""
    import time

    if provider == "ngrok":
        try:
            from pyngrok import ngrok
            tunnel = ngrok.connect(port, "http")
            print(f"✓ ngrok tunnel established: {tunnel.public_url}")
        except ImportError:
            print("✕ pyngrok not installed. Install with: pip install pyngrok")
            print("  Or set NGROK_AUTH_TOKEN env var with your ngrok token")
            print("  Alternatively, use tunnel='cloudflared' or tunnel='localtunnel'")
        except Exception as e:
            print(f"✕ ngrok tunnel failed: {e}")

    elif provider == "cloudflared":
        try:
            import subprocess
            proc = subprocess.Popen(
                ["cloudflared", "tunnel", "--url", f"http://localhost:{port}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
            )
            # Wait for tunnel URL
            for line in proc.stdout:
                if "https://" in line and "trycloudflare.com" in line:
                    import re
                    match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
                    if match:
                        print(f"✓ cloudflared tunnel established: {match.group(0)}")
                        break
        except FileNotFoundError:
            print("✕ cloudflared not installed. Install with: pip install cloudflared")
        except Exception as e:
            print(f"✕ cloudflared tunnel failed: {e}")

    elif provider == "localtunnel":
        try:
            import subprocess
            print("Starting localtunnel process...")
            proc = subprocess.Popen(
                ["lt", "--port", str(port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
            )
            print("localtunnel process started, waiting for URL...")
            for line in proc.stdout:
                print(f"[localtunnel] {line.rstrip()}")  # Log all output
                # Match any https URL from localtunnel
                if "https://" in line:
                    import re
                    # Try to extract URL
                    match = re.search(r"https://[^\s)]+", line)
                    if match:
                        url = match.group(0).rstrip()
                        print(f"\n✓ localtunnel established: {url}\n")
                        break
        except FileNotFoundError:
            print("✕ localtunnel not installed. Install with: npm install -g localtunnel")
        except Exception as e:
            print(f"✕ localtunnel failed: {e}")
            import traceback
            traceback.print_exc()

    else:
        print(f"✕ Unknown tunnel provider: {provider}")
        print("  Supported: ngrok, cloudflared, localtunnel")
