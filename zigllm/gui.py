from .config import RunConfig

CSS = """
:root { --ink:#f5f5f5; --muted:#929292; --line:#292929; --panel:#111; }
html,body,#root,.gradio-container,.main { background:#000 !important; color:var(--ink); }
body { margin:0 !important; }
.gradio-container { width:100% !important; max-width:none !important; min-height:100vh !important; margin:0 !important; padding:0 5vw 40px !important; box-sizing:border-box !important; }
#hero { padding:34px 0 24px; border-bottom:1px solid var(--line); margin:0 0 26px; }
#hero h1 { font-size:38px; font-weight:600; letter-spacing:-1.5px; margin:0 0 8px; color:#fff; }
#hero p { color:var(--muted); font-size:15px; margin:0; }
.section { border:1px solid var(--line); border-radius:10px; padding:18px; background:var(--panel); }
.section h3 { margin-top:0; color:#fff; font-size:14px; font-weight:600; }
    .gradio-container { --body-background-fill:#000 !important; --block-background-fill:#111 !important; --block-border-color:#292929 !important; --input-background-fill:#080808 !important; --input-border-color:#303030 !important; --body-text-color:#f5f5f5 !important; --block-label-text-color:#aaa !important; --body-text-color-subdued:#888 !important; --button-primary-background-fill:#fff !important; --button-primary-text-color:#000 !important; }
.gr-block,.gr-box,.gr-panel,.gr-form,.gr-group,.form,.panel,.wrap,.block,.container,.gradio-group,
[data-testid="textbox"],[data-testid="dropdown"],[data-testid="number"],[data-testid="radio"],[data-testid="accordion"],
[data-testid="block-info"],.accordion,
div[class*="block"],div[class*="form"],div[class*="panel"],div[class*="group"] { background:#111 !important; border-color:var(--line) !important; color:var(--ink) !important; }
.gradio-container .wrap, .gradio-container .wrap > div { background:#111 !important; }
input,textarea,select,[data-testid="textbox"] input,[data-testid="dropdown"] input,[data-testid="number"] input { background:#080808 !important; border-color:#303030 !important; color:#f5f5f5 !important; }
input:focus,textarea:focus { border-color:#777 !important; box-shadow:0 0 0 1px #777 !important; }
.tabitem,.tabs,.tab-nav { background:#000 !important; border-color:var(--line) !important; }
.tab-nav button { color:#777 !important; background:#000 !important; }
.tab-nav button.selected { color:#fff !important; border-color:#fff !important; }
label,.label-wrap,.field-label { color:#aaa !important; }
#start,#benchmark-run { border-radius:8px; background:#fff; color:#000; border:1px solid #fff; font-weight:600; font-size:15px; }
#start:hover,#benchmark-run:hover { background:#d8d8d8; }
#log textarea { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
.badge { display:inline-block; padding:4px 9px; border-radius:4px; margin-right:6px; font-size:11px; background:#171717; color:#cfcfcf; border:1px solid #333; }
.dataset-row { padding:10px 12px; border-bottom:1px solid #1a1a1a; }
.dataset-row:hover { background:#1a1a1a; }
.dataset-id { color:#fff; font-weight:600; font-size:14px; }
.dataset-meta { color:#777; font-size:12px; margin-top:2px; }
.dataset-desc { color:#999; font-size:12px; margin-top:4px; }
.perf-pill { display:inline-block; padding:2px 8px; border-radius:3px; font-size:11px; margin-right:4px; background:#1a1a1a; color:#aaa; border:1px solid #292929; }
"""

def launch():
    import gradio as gr

    def start(model, arch, mode, adapter, device, provider, dataset, split, column, seq, batch, accum, epochs, lr, rank,
              grad_ckpt, flash_attn, torch_comp, num_w, warmup, stream):
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
            cfg=RunConfig(
                model_id=model, architecture=arch, mode=mode, adapter=adapter, device=device,
                dataset_id=dataset, dataset_split=split, text_column=column,
                seq_len=int(seq), batch_size=int(batch), grad_accum=int(accum),
                epochs=float(epochs), learning_rate=float(lr), lora_rank=int(rank),
                gradient_checkpointing=bool(grad_ckpt), flash_attention=bool(flash_attn),
                torch_compile=bool(torch_comp), num_workers=int(num_w),
                warmup_ratio=float(warmup), streaming=bool(stream),
            )
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
            return f"✓ Extracted {len(texts)} text blocks\n\n" + "\n\n".join(texts)
        except Exception as e: return "✕ ERROR: " + str(e)

    def build_core():
        import subprocess
        from pathlib import Path
        try:
            project_root = Path(__file__).resolve().parent.parent
            result = subprocess.run(["zig", "build", "-Doptimize=ReleaseFast"], text=True, capture_output=True, cwd=project_root)
            return ("✓ Zig core built successfully\n" if result.returncode == 0 else "✕ Build failed\n") + (result.stdout + result.stderr)
        except FileNotFoundError: return "✕ Zig is not installed. Install Zig, then retry."
        except Exception as e: return "✕ ERROR: " + str(e)

    def search_hf(query, sort, limit, task, lang, author):
        from .datasets import search_datasets
        try:
            results = search_datasets(
                query=query.strip(), sort=sort, limit=int(limit),
                task=task if task != "all" else "", language=lang.strip(),
                author=author.strip(),
            )
            if not results:
                return "No datasets found. Try a different query or filter.", ""
            # Format results for display
            lines = [f"Found {len(results)} datasets (sorted by {sort}):\n"]
            choices = []
            for i, ds in enumerate(results, 1):
                tags = ", ".join(ds.tags[:4]) if ds.tags else "—"
                desc = (ds.description[:80] + "…") if len(ds.description) > 80 else ds.description
                lines.append(f"{i}. {ds.id}")
                lines.append(f"   ↓ {ds.downloads:,}  ❤ {ds.likes}  |  {tags}")
                if desc:
                    lines.append(f"   {desc}")
                lines.append("")
                choices.append(ds.id)
            # First choice is selected by default
            default = choices[0] if choices else ""
            return "\n".join(lines), default
        except Exception as e:
            return f"✕ Search error: {e}", ""

    with gr.Blocks(title="ZigLLM Studio", theme=gr.themes.Base(neutral_hue="slate"), css=CSS) as app:
        gr.HTML("<div id='hero'><h1>ZigLLM Studio</h1><p>Train, fine-tune, and evaluate language models visually — optimized for Kaggle and Google Colab.</p><br><span class='badge'>Zig core</span><span class='badge'>LoRA / QLoRA</span><span class='badge'>Flash Attention</span><span class='badge'>Dataset Browser</span></div>")
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
                        gr.Markdown("*Use the **Dataset Browser** tab to search and pick datasets, or type an ID directly.*")
                        with gr.Row():
                            provider=gr.Dropdown(["huggingface","kaggle","local"],value="huggingface",label="Provider")
                            dataset=gr.Textbox(label="Dataset ID",placeholder="dataset/name or owner/dataset")
                        with gr.Row():
                            split=gr.Textbox(value="train",label="Split"); column=gr.Textbox(value="text",label="Text column")
                with gr.Accordion("Advanced training controls", open=False):
                    with gr.Row():
                        seq=gr.Number(value=1024,label="Sequence length"); batch=gr.Number(value=1,label="Micro-batch"); accum=gr.Number(value=8,label="Grad accumulation"); epochs=gr.Number(value=1,label="Epochs")
                    with gr.Row():
                        lr=gr.Number(value=.0002,label="Learning rate"); rank=gr.Number(value=16,label="LoRA rank")
                with gr.Accordion("⚡ Performance tuning", open=True):
                    gr.Markdown("<span class='perf-pill'>Flash Attn</span><span class='perf-pill'>Grad Checkpointing</span><span class='perf-pill'>torch.compile</span><span class='perf-pill'>Streaming</span>")
                    with gr.Row():
                        flash_attn = gr.Checkbox(label="Flash Attention 2 / SDPA", value=True, info="2-4x faster attention, lower VRAM (requires CUDA)")
                        grad_ckpt = gr.Checkbox(label="Gradient checkpointing", value=False, info="Save ~40% VRAM at ~20% compute cost")
                        torch_comp = gr.Checkbox(label="torch.compile()", value=False, info="Kernel fusion for 10-30% speedup (PyTorch 2+, CUDA)")
                        stream = gr.Checkbox(label="Stream dataset", value=False, info="Don't load full dataset into RAM (for large datasets)")
                    with gr.Row():
                        num_w = gr.Slider(0, 16, value=2, step=1, label="DataLoader workers", info="Parallel data loading processes")
                        warmup = gr.Slider(0.0, 0.3, value=0.03, step=0.01, label="Warmup ratio", info="Fraction of steps for LR warmup")
                go=gr.Button("▶  Validate & start training",variant="primary",elem_id="start")
                output=gr.Textbox(label="Live job log",lines=7,elem_id="log")
                go.click(start,[model,arch,mode,adapter,device,provider,dataset,split,column,seq,batch,accum,epochs,lr,rank,
                                grad_ckpt,flash_attn,torch_comp,num_w,warmup,stream],output)

            with gr.Tab("Dataset Browser"):
                gr.Markdown("### 🔍 Search Hugging Face Datasets")
                gr.Markdown("Browse and search any public dataset on huggingface.co/datasets. Click a result to auto-fill the Training tab.")
                with gr.Row():
                    search_query = gr.Textbox(label="Search query", placeholder="e.g. code, math, medical, alpaca, instruction…", scale=3)
                    search_sort = gr.Dropdown(["downloads", "likes", "trending", "lastModified", "createdAt"], value="downloads", label="Sort by")
                    search_limit = gr.Slider(5, 100, value=20, step=5, label="Results")
                with gr.Row():
                    search_task = gr.Dropdown(
                        ["all", "text-generation", "text-classification", "token-classification",
                         "question-answering", "summarization", "translation", "fill-mask",
                         "sentence-similarity", "table-question-answering", "zero-shot-classification"],
                        value="all", label="Task")
                    search_lang = gr.Textbox(label="Language", placeholder="e.g. en, code, fr", scale=1)
                    search_author = gr.Textbox(label="Author", placeholder="e.g. tiiuae", scale=1)
                search_go = gr.Button("🔍 Search datasets", variant="primary")
                search_results = gr.Textbox(label="Results", lines=18, elem_id="log", interactive=False)
                search_selected = gr.Textbox(label="Selected dataset ID", value="", info="This is the first result; edit or copy to the Training tab's Dataset ID field")
                search_go.click(search_hf, [search_query, search_sort, search_limit, search_task, search_lang, search_author],
                                [search_results, search_selected])

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
