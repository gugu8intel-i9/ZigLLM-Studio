from .config import RunConfig

CSS = """
:root { --ink:#f5f5f5; --muted:#929292; --line:#292929; --panel:#111; }
body,.gradio-container { background:#000 !important; color:var(--ink); }
.gradio-container { max-width:1120px !important; padding-bottom:40px !important; }
#hero { padding:34px 0 24px; border-bottom:1px solid var(--line); margin:0 0 26px; }
#hero h1 { font-size:38px; font-weight:600; letter-spacing:-1.5px; margin:0 0 8px; color:#fff; }
#hero p { color:var(--muted); font-size:15px; margin:0; }
.section { border:1px solid var(--line); border-radius:10px; padding:18px; background:var(--panel); }
.section h3 { margin-top:0; color:#fff; font-size:14px; font-weight:600; }
input,textarea,.gr-box,.gr-input,select { background:#080808 !important; border-color:#303030 !important; color:#f5f5f5 !important; }
input:focus,textarea:focus { border-color:#777 !important; box-shadow:0 0 0 1px #777 !important; }
#start,#benchmark-run { border-radius:8px; background:#fff; color:#000; border:1px solid #fff; font-weight:600; font-size:15px; }
#start:hover,#benchmark-run:hover { background:#d8d8d8; }
#log textarea { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
.badge { display:inline-block; padding:4px 9px; border-radius:4px; margin-right:6px; font-size:11px; background:#171717; color:#cfcfcf; border:1px solid #333; }
"""

def launch():
    import gradio as gr

    def start(model, arch, mode, adapter, device, provider, dataset, split, column, seq, batch, accum, epochs, lr, rank):
        from .engine import Trainer
        try:
            auto_notes=[]
            from .dependencies import ensure_dependencies
            installed=ensure_dependencies("training", qlora=False)
            import torch
            if device == "auto" and adapter == "qlora" and not torch.cuda.is_available():
                adapter = "lora"
                auto_notes.append("CUDA unavailable: switched QLoRA to LoRA so training can run on CPU.")
            elif adapter == "qlora":
                extra=ensure_dependencies("training", qlora=True)
                installed += extra
            if installed: auto_notes.append("Installed missing packages: " + ", ".join(dict.fromkeys(installed)))
            cfg=RunConfig(model_id=model,architecture=arch,mode=mode,adapter=adapter,device=device,dataset_id=dataset,dataset_split=split,text_column=column,seq_len=int(seq),batch_size=int(batch),grad_accum=int(accum),epochs=float(epochs),learning_rate=float(lr),lora_rank=int(rank))
            prefix="✓ Configuration validated\n" + ("\n".join("⚙ " + n for n in auto_notes) + "\n" if auto_notes else "") + "→ Launching training job...\n"
            return prefix + Trainer(cfg,dataset_provider=provider).run()
        except Exception as e: return "✕ ERROR: " + str(e)

    def benchmark(model, name, device, limit):
        from .benchmarks import run_benchmark
        try: return run_benchmark(name, model, int(limit), device)
        except Exception as e: return "✕ ERROR: " + str(e)

    def convert_csv(csv_file, output_path, compression, batch_size):
        from .dependencies import ensure_dependencies
        ensure_dependencies("data")
        from .datasets import csv_to_parquet
        try:
            if not csv_file: return "✕ Select a CSV file first"
            path = csv_file if isinstance(csv_file, str) else csv_file.name
            return str(csv_to_parquet(path, output_path, None if compression == "none" else compression, int(batch_size)))
        except Exception as e: return "✕ ERROR: " + str(e)

    def scrape_page(url, selector):
        from .datasets import scrape
        try:
            texts = scrape(url, selector.strip() or None)
            return f"✓ Extracted {len(texts)} text blocks\\n\\n" + "\\n\\n".join(texts)
        except Exception as e: return "✕ ERROR: " + str(e)

    def build_core():
        import subprocess
        from pathlib import Path
        try:
            project_root = Path(__file__).resolve().parent.parent
            result = subprocess.run(["zig", "build", "-Doptimize=ReleaseFast"], text=True, capture_output=True, cwd=project_root)
            return ("✓ Zig core built successfully\\n" if result.returncode == 0 else "✕ Build failed\\n") + (result.stdout + result.stderr)
        except FileNotFoundError: return "✕ Zig is not installed. Install Zig, then retry."
        except Exception as e: return "✕ ERROR: " + str(e)

    with gr.Blocks(title="ZigLLM Studio", theme=gr.themes.Base(neutral_hue="slate"), css=CSS) as app:
        gr.HTML("<div id='hero'><h1>ZigLLM Studio</h1><p>Train, fine-tune, and evaluate language models visually — optimized for Kaggle and Google Colab.</p><br><span class='badge'>Zig core</span><span class='badge'>LoRA / QLoRA</span><span class='badge'>Benchmarks</span></div>")
        with gr.Tabs():
            with gr.Tab("Training"):
                with gr.Row():
                    with gr.Column(scale=7, elem_classes="section"):
                        gr.Markdown("### 01 · Model setup")
                        model=gr.Textbox(label="Model checkpoint",value="Qwen/Qwen2.5-0.5B",info="Compatible Hugging Face checkpoint")
                        with gr.Row():
                            arch=gr.Dropdown(["transformer","looped_transformer","mamba"],value="transformer",label="Architecture")
                            mode=gr.Radio(["train","finetune"],value="finetune",label="Run mode")
                    with gr.Column(scale=5, elem_classes="section"):
                        gr.Markdown("### 02 · Compute strategy")
                        with gr.Row():
                            adapter=gr.Dropdown(["full","lora","qlora"],value="lora",label="Memory strategy")
                            device=gr.Dropdown(["auto","cuda","cpu"],value="auto",label="Device")
                with gr.Row():
                    with gr.Column(elem_classes="section"):
                        gr.Markdown("### 03 · Data source")
                        with gr.Row():
                            provider=gr.Dropdown(["huggingface","kaggle"],value="huggingface",label="Provider")
                            dataset=gr.Textbox(label="Dataset ID",placeholder="dataset/name or owner/dataset")
                        with gr.Row():
                            split=gr.Textbox(value="train",label="Split"); column=gr.Textbox(value="text",label="Text column")
                with gr.Accordion("Advanced training controls",open=True):
                    with gr.Row():
                        seq=gr.Number(value=1024,label="Sequence length"); batch=gr.Number(value=1,label="Micro-batch"); accum=gr.Number(value=8,label="Grad accumulation"); epochs=gr.Number(value=1,label="Epochs")
                    with gr.Row():
                        lr=gr.Number(value=.0002,label="Learning rate"); rank=gr.Number(value=16,label="LoRA rank")
                go=gr.Button("▶  Validate & start training",variant="primary",elem_id="start")
                output=gr.Textbox(label="Live job log",lines=7,elem_id="log")
                go.click(start,[model,arch,mode,adapter,device,provider,dataset,split,column,seq,batch,accum,epochs,lr,rank],output)
            with gr.Tab("Data tools"):
                gr.Markdown("### CSV → optimized Parquet")
                gr.Markdown("Stream large CSV exports into compressed, columnar Parquet without loading the entire file into memory.")
                csv_file=gr.File(label="CSV input", file_types=[".csv"], type="filepath")
                with gr.Row():
                    parquet_path=gr.Textbox(value="dataset.parquet", label="Output path")
                    compression=gr.Dropdown(["zstd","snappy","gzip","brotli","none"], value="zstd", label="Compression")
                    batch_size=gr.Number(value=100000, label="Read block size")
                convert_go=gr.Button("Convert to Parquet", variant="primary")
                convert_output=gr.Textbox(label="Conversion result", lines=4, elem_id="log")
                convert_go.click(convert_csv, [csv_file, parquet_path, compression, batch_size], convert_output)
                gr.Markdown("### Public web scraper")
                with gr.Row():
                    scrape_url=gr.Textbox(label="Page URL", placeholder="https://example.org/article")
                    scrape_selector=gr.Textbox(label="CSS selector (optional)", placeholder="article p")
                scrape_go=gr.Button("Scrape page")
                scrape_output=gr.Textbox(label="Extracted text", lines=8, elem_id="log")
                scrape_go.click(scrape_page, [scrape_url, scrape_selector], scrape_output)
            with gr.Tab("System"): 
                gr.Markdown("### Zig core")
                gr.Markdown("Compile the dependency-free Zig acceleration library with the same ReleaseFast settings used by the CLI.")
                core_go=gr.Button("Build Zig core", variant="primary")
                core_output=gr.Textbox(label="Build log", lines=6, elem_id="log")
                core_go.click(build_core, outputs=core_output)
            with gr.Tab("Benchmarks"):
                gr.Markdown("### Evaluate a checkpoint")
                gr.Markdown("Run standard scoring tasks directly, or get the official harness command for agent and cybersecurity benchmarks.")
                with gr.Row():
                    bm_model=gr.Textbox(label="Model checkpoint",value="Qwen/Qwen2.5-0.5B")
                    bm_name=gr.Dropdown(["swe","gmsk8","hle","cybergym","hellaswag"],value="hellaswag",label="Benchmark")
                with gr.Row():
                    bm_device=gr.Dropdown(["auto","cuda","cpu"],value="auto",label="Device")
                    bm_limit=gr.Number(value=0,label="Sample limit",info="0 = full benchmark")
                gr.Markdown("**Available:** SWE-bench · GMSK8/GSM8K · HLE · CyberGym · HellaSwag")
                bm_go=gr.Button("Run benchmark",variant="primary",elem_id="benchmark-run")
                bm_output=gr.Textbox(label="Benchmark result / harness log",lines=10,elem_id="log")
                bm_go.click(benchmark,[bm_model,bm_name,bm_device,bm_limit],bm_output)
        gr.Markdown("<center><small>ZigLLM · transparent controls for reproducible experiments</small></center>")
        import os
        # Notebook VMs cannot expose 127.0.0.1 to the user's browser. Gradio's
        # share tunnel is enabled automatically in Colab and Kaggle.
        notebook = bool(os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))
        app.launch(share=notebook, server_name="0.0.0.0" if notebook else None)
