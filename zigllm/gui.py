from .config import RunConfig

def launch():
    import gradio as gr
    import os

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
            default = choices[0] if choices else ""
            return "\n".join(lines), default
        except Exception as e:
            return f"✕ Search error: {e}", ""

    def dataset_preview(text_input, files, urls, selector, name, min_chars, max_chars, dedupe):
        from .creator import DatasetBuilder
        try:
            builder = DatasetBuilder(name=name or "preview")
            if text_input:
                lines = [l.strip() for l in text_input.split("\n") if l.strip()]
                builder.add_texts(lines, source="text-input")
            if files:
                file_list = files if isinstance(files, list) else [files]
                builder.add_files(file_list)
            if urls:
                url_list = [u.strip() for u in urls.split("\n") if u.strip()]
                builder.add_urls(url_list, selector=selector if selector else None)
            if dedupe:
                builder.dedupe()
            if min_chars > 0:
                builder.filter_min_length(int(min_chars))
            if max_chars > 0:
                builder.filter_max_length(int(max_chars))
            samples = builder.preview(5)
            if not samples:
                return "No samples to preview. Add some content first."
            lines = [f"Preview of first {len(samples)} samples:\n"]
            for i, s in enumerate(samples, 1):
                text = s.get("text", "")[:200]
                if len(s.get("text", "")) > 200:
                    text += "..."
                lines.append(f"{i}. {text}")
                lines.append(f"   Source: {s.get('source', 'unknown')}  Length: {len(s.get('text', ''))} chars")
                lines.append("")
            lines.append(f"Total samples before export: {len(builder.samples)}")
            return "\n".join(lines)
        except Exception as e:
            return f"✕ Preview error: {e}"

    def dataset_build(text_input, files, urls, selector, name, output, format, min_chars, max_chars, dedupe,
                      split_train, split_val, split_test, hub_upload, hub_repo, hub_private):
        from .creator import DatasetBuilder
        try:
            if not name:
                return "✕ Dataset name is required"
            if not output:
                return "✕ Output path is required"
            builder = DatasetBuilder(name=name)
            log = [f"Building dataset: {name}\n"]
            if text_input:
                lines = [l.strip() for l in text_input.split("\n") if l.strip()]
                n = builder.add_texts(lines, source="text-input")
                log.append(f"✓ Added {n} samples from text input")
            if files:
                file_list = files if isinstance(files, list) else [files]
                n = builder.add_files(file_list)
                log.append(f"✓ Ingested {n} samples from {len(file_list)} files")
            if urls:
                url_list = [u.strip() for u in urls.split("\n") if u.strip()]
                n = builder.add_urls(url_list, selector=selector if selector else None)
                log.append(f"✓ Scraped {n} samples from {len(url_list)} URLs")
            if not builder.samples:
                return "✕ No samples collected. Add content from at least one source."
            log.append(f"\nTotal samples before cleaning: {len(builder.samples)}")
            result = builder.build(
                output_path=output,
                format=format,
                dedupe=dedupe,
                min_chars=int(min_chars),
                max_chars=int(max_chars),
                split_ratios=(float(split_train), float(split_val), float(split_test)),
            )
            log.append(f"\n✓ Dataset created successfully!")
            log.append(f"  Output: {result['output']}")
            log.append(f"  Samples: {result['samples']}")
            if result['removed']:
                log.append(f"  Removed: {result['removed']}")
            log.append(f"  Format: {result['format']}")
            log.append(f"  Metadata: {result['metadata']}")
            if hub_upload:
                if not hub_repo:
                    log.append("\n⚠ Hub upload skipped: repo ID not specified")
                else:
                    try:
                        url = builder.push_to_hub(hub_repo, private=hub_private)
                        log.append(f"\n✓ Uploaded to HuggingFace Hub: {url}")
                    except Exception as e:
                        log.append(f"\n✕ Hub upload failed: {e}")
            log.append("\n💡 Use this dataset in the Training tab:")
            log.append(f"   Provider: local")
            log.append(f"   Dataset ID: {result['output']}")
            return "\n".join(log)
        except Exception as e:
            return f"✕ Build error: {e}"

    # ── Dark theme for Gradio 6.0 ────────────────────────────────────────
    theme = gr.themes.Base(
        primary_hue="slate",
        neutral_hue="slate",
        radius_size="lg",
    ).set(
        body_background_fill="#000000",
        body_text_color="#f5f5f5",
        body_text_color_subdued="#888888",
        block_background_fill="#111111",
        block_border_color="#292929",
        block_label_background_fill="#111111",
        block_label_text_color="#aaaaaa",
        block_title_text_color="#ffffff",
        input_background_fill="#080808",
        input_border_color="#303030",
        input_placeholder_color="#666666",
        button_primary_background_fill="#ffffff",
        button_primary_text_color="#000000",
        button_primary_background_fill_hover="#d8d8d8",
        button_secondary_background_fill="#1a1a1a",
        button_secondary_text_color="#cccccc",
        button_secondary_background_fill_hover="#2a2a2a",
        table_border_color="#292929",
        table_even_background_fill="#0f0f0f",
        table_odd_background_fill="#0a0a0a",
    )

    # ── CSS for Gradio 6.0 dark mode ─────────────────────────────────────
    css = """
/* Layout */
body { margin:0 !important; }
.gradio-container { width:100% !important; max-width:none !important; min-height:100vh !important; margin:0 !important; padding:0 5vw 40px !important; box-sizing:border-box !important; }

/* Hero section */
#hero { padding:34px 0 24px; border-bottom:1px solid #292929; margin:0 0 26px; }
#hero h1 { font-size:38px; font-weight:600; letter-spacing:-1.5px; margin:0 0 8px; color:#fff; }
#hero p { color:#929292; font-size:15px; margin:0; }

/* Section boxes */
.section { border:1px solid #292929 !important; border-radius:10px !important; padding:18px !important; background:#111 !important; }
.section h3 { margin-top:0; color:#fff; font-size:14px; font-weight:600; }

/* Form elements */
input, textarea, select { background:#080808 !important; border-color:#303030 !important; color:#f5f5f5 !important; }
input:focus, textarea:focus { border-color:#777 !important; box-shadow:0 0 0 1px #777 !important; }

/* Dropdown popups - CRITICAL FIX for Gradio 6.0 */
[data-testid="dropdown"] > div,
.gr-dropdown-list,
.options,
.dropdown-menu,
[role="listbox"],
[role="option"] { background:#111 !important; color:#f5f5f5 !important; }
[role="option"]:hover { background:#1a1a1a !important; }
[role="option"][aria-selected="true"] { background:#292929 !important; }

/* Tabs */
.tabitem, .tabs, .tab-nav { background:#000 !important; border-color:#292929 !important; }
.tab-nav button { color:#777 !important; background:#000 !important; }
.tab-nav button.selected { color:#fff !important; border-color:#fff !important; }

/* Labels */
label, .label-wrap, .field-label { color:#aaa !important; }

/* Buttons */
#start, #benchmark-run { border-radius:8px; background:#fff; color:#000; border:1px solid #fff; font-weight:600; font-size:15px; }
#start:hover, #benchmark-run:hover { background:#d8d8d8; }

/* Log output */
.log-output textarea { font-family:ui-monospace,SFMono-Regular,Menlo,monospace !important; background:#080808 !important; color:#f5f5f5 !important; }

/* Badges */
.badge { display:inline-block; padding:4px 9px; border-radius:4px; margin-right:6px; font-size:11px; background:#171717; color:#cfcfcf; border:1px solid #333; }
.perf-pill { display:inline-block; padding:2px 8px; border-radius:3px; font-size:11px; margin-right:4px; background:#1a1a1a; color:#aaa; border:1px solid #292929; }
"""

    # ── Build UI ────────────────────────────────────────────────────────
    with gr.Blocks(title="ZigLLM Studio") as app:
        gr.HTML("<div id='hero'><h1>ZigLLM Studio</h1><p>Train, fine-tune, and evaluate language models visually — optimized for Kaggle and Google Colab.</p><br><span class='badge'>Zig core</span><span class='badge'>LoRA / QLoRA</span><span class='badge'>Flash Attention</span><span class='badge'>Dataset Browser</span></div>")
        with gr.Tabs():
            # ── Training ────────────────────────────────────────────────
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
                output=gr.Textbox(label="Live job log",lines=7,elem_classes=["log-output"])
                go.click(start,[model,arch,mode,adapter,device,provider,dataset,split,column,seq,batch,accum,epochs,lr,rank,
                                grad_ckpt,flash_attn,torch_comp,num_w,warmup,stream],output)

            # ── Dataset Browser ─────────────────────────────────────────
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
                search_results = gr.Textbox(label="Results", lines=18, elem_classes=["log-output"], interactive=False)
                search_selected = gr.Textbox(label="Selected dataset ID", value="", info="This is the first result; edit or copy to the Training tab's Dataset ID field")
                search_go.click(search_hf, [search_query, search_sort, search_limit, search_task, search_lang, search_author],
                                [search_results, search_selected])

            # ── Data tools ──────────────────────────────────────────────
            with gr.Tab("Data tools"):
                gr.Markdown("### CSV → optimized Parquet")
                gr.Markdown("Stream large CSV exports into compressed, columnar Parquet without loading the entire file into memory.")
                csv_file=gr.File(label="CSV input", file_types=[".csv"], type="filepath")
                with gr.Row():
                    parquet_path=gr.Textbox(value="dataset.parquet", label="Output path")
                    compression=gr.Dropdown(["zstd","snappy","gzip","brotli","none"], value="zstd", label="Compression")
                    batch_size=gr.Number(value=100000, label="Read block size")
                convert_go=gr.Button("Convert to Parquet", variant="primary")
                convert_output=gr.Textbox(label="Conversion result", lines=4, elem_classes=["log-output"])
                convert_go.click(convert_csv, [csv_file, parquet_path, compression, batch_size], convert_output)
                gr.Markdown("### Public web scraper")
                with gr.Row():
                    scrape_url=gr.Textbox(label="Page URL", placeholder="https://example.org/article")
                    scrape_selector=gr.Textbox(label="CSS selector (optional)", placeholder="article p")
                scrape_go=gr.Button("Scrape page")
                scrape_output=gr.Textbox(label="Extracted text", lines=8, elem_classes=["log-output"])
                scrape_go.click(scrape_page, [scrape_url, scrape_selector], scrape_output)

            # ── System ──────────────────────────────────────────────────
            with gr.Tab("System"):
                gr.Markdown("### Zig core")
                gr.Markdown("Compile the dependency-free Zig acceleration library with the same ReleaseFast settings used by the CLI.")
                core_go=gr.Button("Build Zig core", variant="primary")
                core_output=gr.Textbox(label="Build log", lines=6, elem_classes=["log-output"])
                core_go.click(build_core, outputs=core_output)

            # ── Create Dataset ──────────────────────────────────────────
            with gr.Tab("Create Dataset"):
                gr.Markdown("### 🛠️ Build Your Own Training Dataset")
                gr.Markdown("Collect text from multiple sources, clean it, and export in a format ready for training.")

                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("#### 📥 Input Sources")
                        gr.Markdown("*Add content from any combination of these sources:*")

                        with gr.Accordion("Text Input", open=True):
                            cd_text = gr.Textbox(
                                label="Paste text (one sample per line)",
                                lines=8,
                                placeholder="The quick brown fox...\nAnother sample...\nYet another...",
                                info="Each line becomes a separate training sample"
                            )

                        with gr.Accordion("File Upload", open=False):
                            cd_files = gr.File(
                                label="Upload files (.txt, .md, .json, .jsonl)",
                                file_count="multiple",
                                file_types=[".txt", ".md", ".json", ".jsonl"],
                                type="filepath"
                            )
                            gr.Markdown("*Long files (>5000 chars) are split into paragraphs automatically.*")

                        with gr.Accordion("Web Scraping", open=False):
                            cd_urls = gr.Textbox(
                                label="URLs (one per line)",
                                lines=5,
                                placeholder="https://example.com/page1\nhttps://example.com/page2",
                                info="Public pages only — respect robots.txt and terms of service"
                            )
                            cd_selector = gr.Textbox(
                                label="CSS selector (optional)",
                                placeholder="article p",
                                info="Target specific HTML elements. Leave blank to extract all text."
                            )

                    with gr.Column(scale=1):
                        gr.Markdown("#### ️ Configuration")
                        cd_name = gr.Textbox(
                            label="Dataset name",
                            value="my-dataset",
                            placeholder="my-dataset",
                            info="Used in metadata and file naming"
                        )
                        cd_output_path = gr.Textbox(
                            label="Output path",
                            value="dataset.jsonl",
                            placeholder="dataset.jsonl",
                            info="Output file path (extension auto-applied if needed)"
                        )
                        cd_format = gr.Dropdown(
                            ["jsonl", "json", "parquet"],
                            value="jsonl",
                            label="Format",
                            info="JSONL recommended for streaming, Parquet for compression"
                        )

                        with gr.Accordion("🧹 Cleaning Options", open=False):
                            cd_min_chars = gr.Slider(
                                0, 1000, value=40, step=10,
                                label="Minimum characters",
                                info="Skip samples shorter than this (0 = no filter)"
                            )
                            cd_max_chars = gr.Slider(
                                0, 10000, value=0, step=100,
                                label="Maximum characters",
                                info="Skip samples longer than this (0 = unlimited)"
                            )
                            cd_dedupe = gr.Checkbox(
                                label="Remove duplicates",
                                value=True,
                                info="Deduplicate by text content hash"
                            )

                        with gr.Accordion("📊 Train/Val/Test Splits", open=False):
                            gr.Markdown("*Ratios should sum to ~1.0*")
                            cd_split_train = gr.Slider(0.5, 1.0, value=0.9, step=0.05, label="Train")
                            cd_split_val = gr.Slider(0.0, 0.3, value=0.05, step=0.05, label="Validation")
                            cd_split_test = gr.Slider(0.0, 0.3, value=0.05, step=0.05, label="Test")

                        with gr.Accordion("☁️ Upload to HuggingFace Hub (Optional)", open=False):
                            cd_hub_upload = gr.Checkbox(
                                label="Upload to Hub",
                                value=False,
                                info="Requires HF_TOKEN environment variable"
                            )
                            cd_hub_repo = gr.Textbox(
                                label="Hub repo ID",
                                placeholder="username/dataset-name",
                                info="e.g. myuser/my-custom-dataset"
                            )
                            cd_hub_private = gr.Checkbox(
                                label="Private dataset",
                                value=False
                            )

                with gr.Row():
                    cd_preview = gr.Button("️ Preview Samples")
                    cd_build = gr.Button("🔨 Build Dataset", variant="primary")

                cd_preview_output = gr.Textbox(
                    label="Preview (first 5 samples)",
                    lines=10,
                    interactive=False,
                    elem_classes=["log-output"]
                )
                cd_output_log = gr.Textbox(
                    label="Build Log",
                    lines=12,
                    interactive=False,
                    elem_classes=["log-output"]
                )

                cd_preview.click(
                    dataset_preview,
                    inputs=[cd_text, cd_files, cd_urls, cd_selector, cd_name, cd_min_chars, cd_max_chars, cd_dedupe],
                    outputs=cd_preview_output
                )
                cd_build.click(
                    dataset_build,
                    inputs=[cd_text, cd_files, cd_urls, cd_selector, cd_name, cd_output_path, cd_format,
                            cd_min_chars, cd_max_chars, cd_dedupe, cd_split_train, cd_split_val, cd_split_test,
                            cd_hub_upload, cd_hub_repo, cd_hub_private],
                    outputs=cd_output_log
                )

            # ─ Benchmarks ──────────────────────────────────────────────
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
                bm_output=gr.Textbox(label="Benchmark result / harness log",lines=10,elem_classes=["log-output"])
                bm_go.click(benchmark,[bm_model,bm_name,bm_device,bm_limit],bm_output)

        gr.Markdown("<center><small>ZigLLM · transparent controls for reproducible experiments</small></center>")

    # Notebook VMs cannot expose 127.0.0.1 to the user's browser. Gradio's
    # share tunnel is enabled automatically in Colab and Kaggle.
    notebook = bool(os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))
    app.launch(
        theme=theme,
        css=css,
        share=notebook,
        server_name="0.0.0.0" if notebook else None,
    )
