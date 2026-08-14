"""Flask server serving the ZigLLM Studio HTML UI and REST API."""
from flask import Flask, request, jsonify, send_from_directory
import os, json, threading, time, uuid, traceback, subprocess, re
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

# Detect notebook environment
_IN_NOTEBOOK = bool(os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))

def _is_notebook():
    """Detect if running inside Jupyter/IPython/Kaggle/Colab."""
    if _IN_NOTEBOOK:
        return True
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        return shell in ("ZMQInteractiveShell", "Shell")
    except Exception:
        return False

def _display_html(html):
    """Render HTML in notebook output if available."""
    if _is_notebook():
        try:
            from IPython.display import display, HTML
            display(HTML(html))
            return True
        except ImportError:
            pass
    return False


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
        "version": "0.4.0",
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


# ─ Search datasets ─────────────────────────────────────────────────
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


# ── Create dataset ────────────────────────────────────────────────
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


# ── Show URL in notebook ───────────────────────────────────────────
def _show_url(url, label="ZigLLM Studio"):
    """Display a big clickable URL card in notebook output."""
    html = f"""
    <div style="background:#0a0a0a; border:2px solid #292929; border-radius:10px; padding:18px; margin:12px 0; font-family:-apple-system,sans-serif;">
        <div style="color:#fff; font-size:15px; font-weight:600; margin-bottom:8px;">{label}</div>
        <a href="{url}" target="_blank" style="display:block; background:#fff; color:#000; text-decoration:none; padding:10px 16px; border-radius:8px; font-size:14px; font-weight:600; text-align:center; margin:8px 0;">
            🔗 Click to open →
        </a>
        <div style="color:#888; font-size:12px; font-family:monospace; word-break:break-all; margin-top:8px;">{url}</div>
    </div>"""
    _display_html(html)


# ── Launch with tunnel support ──────────────────────────────────────
def launch(host="0.0.0.0", port=7860, share=True, tunnel="cloudflared"):
    """Launch the web server with optional public tunnel.

    Args:
        host: Bind address (default: 0.0.0.0 for notebook environments)
        port: Local port (default: 7860)
        share: Create a public tunnel (default: True in notebooks, False otherwise)
        tunnel: Tunnel provider: "cloudflared", "localtunnel", "ngrok", or "kaggle"
                - "cloudflared": free, no account, no splash page (default)
                - "localtunnel": free, no account, has splash page
                - "ngrok": requires account
                - "kaggle": Kaggle-native port forwarding (prints instructions + clickable link)
    """
    notebook = bool(os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or _is_notebook()
    actual_host = "0.0.0.0" if notebook else host

    if notebook and share:
        if tunnel == "kaggle":
            # Kaggle-native: just run the server, print port-forwarding instructions
            print(f"\n{'='*60}")
            print(f"  ZigLLM Studio running on http://0.0.0.0:{port}")
            print(f"{'='*60}")
            print(f"\n  → In the right sidebar, click 'Add port' and enter {port}")
            print(f"  → Click the generated Kaggle URL to open the UI\n")
            _show_url(f"http://localhost:{port}", "ZigLLM Studio (add port in sidebar)")
        else:
            # Tunnel: start in background, then run Flask in main thread
            print(f"\nStarting ZigLLM Studio with {tunnel} tunnel on port {port}...")
            _tunnel_url = [None]

            def _tunnel_thread():
                url = _start_tunnel(tunnel, port)
                if url:
                    _tunnel_url[0] = url
                    _show_url(url, "ZigLLM Studio — Tunnel Established")

            threading.Thread(target=_tunnel_thread, daemon=True).start()

            # Wait a bit so the tunnel URL can be discovered before Flask blocks
            time.sleep(2)
            if _tunnel_url[0]:
                _show_url(_tunnel_url[0])
            elif not notebook:
                # Non-notebook: print localhost link
                _show_url(f"http://localhost:{port}")
    elif share and not notebook:
        print("⚠ share=True auto-detected as non-notebook. Use tunnel='kaggle' for Kaggle.")

    print(f"\nZigLLM Studio web UI: http://localhost:{port}")
    print(f"{'='*60}\n")
    app.run(host=actual_host, port=port, debug=False, use_reloader=False)


def _start_tunnel(provider, port):
    """Start a tunnel and return the public URL, or None on failure."""
    import shutil

    if provider == "cloudflared":
        # Auto-install cloudflared binary if missing
        cf_bin = shutil.which("cloudflared")
        if not cf_bin:
            print("Installing cloudflared binary...")
            try:
                subprocess.run(
                    ["curl", "-sSL", "-o", "/usr/local/bin/cloudflared",
                     "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"],
                    check=True, timeout=60,
                )
                os.chmod("/usr/local/bin/cloudflared", 0o755)
                cf_bin = "/usr/local/bin/cloudflared"
                print(f"✓ cloudflared installed to {cf_bin}")
            except Exception as e:
                print(f"✕ Failed to install cloudflared: {e}")
                return None

        try:
            print("Starting cloudflared tunnel (no splash page)...")
            proc = subprocess.Popen(
                [cf_bin, "tunnel", "--url", f"http://localhost:{port}", "--no-autoupdate"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
            )
            for line in proc.stdout:
                if "trycloudflare.com" in line:
                    match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
                    if match:
                        url = match.group(0)
                        print(f"\n✓ cloudflared tunnel established: {url}\n")
                        return url
                    # Fallback: extract just the hostname
                    match = re.search(r"[a-z0-9-]+\.trycloudflare\.com", line)
                    if match:
                        url = f"https://{match.group(0)}"
                        print(f"\n✓ cloudflared tunnel established: {url}\n")
                        return url
        except Exception as e:
            print(f"✕ cloudflared tunnel failed: {e}")
            return None

    elif provider == "localtunnel":
        if not shutil.which("lt"):
            print("Installing localtunnel...")
            try:
                subprocess.run(
                    ["npm", "install", "-g", "localtunnel"],
                    check=True, timeout=120,
                )
            except Exception as e:
                print(f"✕ Failed to install localtunnel: {e}")
                return None

        try:
            print("Starting localtunnel...")
            proc = subprocess.Popen(
                ["lt", "--port", str(port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
            )
            for line in proc.stdout:
                if "https://" in line and ("loca.lt" in line or "localtunnel" in line):
                    match = re.search(r"https://[^\s)]+", line)
                    if match:
                        url = match.group(0).rstrip()
                        print(f"\n✓ localtunnel established: {url}\n")
                        return url
        except Exception as e:
            print(f"✕ localtunnel failed: {e}")
            return None

    elif provider == "ngrok":
        try:
            from pyngrok import ngrok
            tunnel = ngrok.connect(port, "http")
            print(f"\n✓ ngrok tunnel established: {tunnel.public_url}\n")
            return tunnel.public_url
        except ImportError:
            print(" pyngrok not installed. Install with: pip install pyngrok")
            return None
        except Exception as e:
            print(f"✕ ngrok tunnel failed: {e}")
            return None

    elif provider == "kaggle":
        # Kaggle port forwarding doesn't need a tunnel; just prints instructions
        print(f"\n✓ Kaggle mode: Add port {port} in the right sidebar")
        print(f"  Then click the generated URL to open the UI\n")
        return None

    else:
        print(f"✕ Unknown tunnel provider: {provider}")
        return None
