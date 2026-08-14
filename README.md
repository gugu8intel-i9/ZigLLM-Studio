# ZigLLM Studio

A Kaggle/Google Colab-friendly LLM training control plane with a small, dependency-free Zig core for fast validation and sizing. PyTorch/Transformers run the GPU kernels; Zig is not pretending to replace CUDA/autograd.

## Features

- **Pure HTML/Flask web UI** — no Gradio dependency, dark-themed, works in any browser. Training, evaluation, dataset browsing, and creation in one interface.
- **Create custom datasets** from text, files, and web pages — available in GUI and CLI.
- **Dataset Browser** — search and browse Hugging Face datasets directly from the UI (mirrors huggingface.co/datasets).
- **Performance optimizations**: parallel tokenization, Flash Attention 2/SDPA, gradient checkpointing, torch.compile(), streaming datasets, optimized data loading.
- Transformer, Looped Transformer and Mamba architecture choices. Transformer is native through `AutoModelForCausalLM`; Mamba and looped models accept compatible Hugging Face checkpoints and report when a specialized checkpoint is needed.
- Full training, LoRA and QLoRA (4-bit bitsandbytes on CUDA).
- Hugging Face dataset loading, Kaggle download support through `kagglehub`, local JSON/JSONL/CSV, and a polite public-page scraper.
- Benchmark tab for SWE-bench, GMSK8/GSM8K, HLE, CyberGym, and HellaSwag. GSM8K and HellaSwag can run through `lm-eval`; the other evaluations expose their official harness path because they require patches, tool traces, licensed data, or a sandbox.
- Streaming CSV → compressed Parquet conversion with PyArrow, exposed in both the GUI and CLI. It uses record batches and dictionary encoding for efficient dataset storage.
- The GUI automatically installs missing Python runtime packages. If QLoRA is selected with `device=auto` but the notebook has no CUDA GPU, it automatically falls back to LoRA instead of failing; QLoRA itself still requires CUDA.
- Zig `ReleaseFast` shared library for stable config validation, token-step arithmetic, VRAM estimation, and training step calculation.

## Colab / Kaggle quick start

```python
%cd /kaggle/working
!git clone https://github.com/gugu8intel-i9/ZigLLM-Studio.git zigllm
%cd /kaggle/working/zigllm
!python -m pip install -e '.[train,data,bench,web]'
# Optional: install Zig separately for the ReleaseFast core build.
# The Python training/UI layer does not require Zig to launch.
!python -m pip install ziglang
!zig version
!zig build -Doptimize=ReleaseFast
!zigllm gui   # launches the web UI on http://localhost:7860
```

If the repository was already cloned, do not run `git clone` again. Run:

```python
%cd /kaggle/working/zigllm
!python -m pip install -e '.[train,data,bench,web]'
!zigllm gui
```

Use a public tunnel supplied by the notebook environment if you need to access the UI remotely. In Kaggle, add Hugging Face/Kaggle credentials through the platform's secret manager, never in a notebook cell.

**Note:** `zigllm gui` launches a Flask-based web UI (not Gradio) — no Gradio dependency required. All functionality is preserved with a cleaner, more reliable interface.

## Python API

```python
from zigllm import RunConfig, Trainer
cfg = RunConfig(
    model_id="Qwen/Qwen2.5-0.5B",
    dataset_id="roneneldan/TinyStories",
    architecture="transformer", mode="finetune",
    adapter="qlora", device="cuda", seq_len=512,
)
Trainer(cfg).run()
```

CSV optimization:

```bash
pip install -e '.[data]'
zigllm csv-to-parquet data.csv data.parquet --compression zstd --batch-size 100000
```

Kaggle dataset and scraper:

```python
from zigllm.datasets import DatasetSource, fetch_dataset, scrape
kaggle = fetch_dataset(DatasetSource("kaggle", "owner/dataset-name"))
texts = scrape("https://example.org", selector="article p")
```

## Create Custom Datasets

Build training datasets from text, files, and web pages. Available via GUI (Create Dataset tab), CLI, and Python API.

### CLI

```bash
# From text samples
zigllm create-dataset --name my-dataset --output dataset.jsonl \
  --text "First sample" --text "Second sample" \
  --min-chars 40 --split-train 0.9 --split-val 0.05 --split-test 0.05

# From files
zigllm create-dataset --name file-dataset --output dataset.jsonl \
  --files data1.txt --files data2.jsonl \
  --min-chars 100

# From web scraping
zigllm create-dataset --name web-dataset --output dataset.jsonl \
  --urls "https://example.com/page1" --urls "https://example.com/page2" \
  --selector "article p"

# Optional: upload to HuggingFace Hub
zigllm create-dataset --name hub-dataset --output dataset.jsonl \
  --text "Sample" --push-hub --hub-repo username/dataset-name
```

### Python API

```python
from zigllm import DatasetBuilder, create_dataset_from_text

# Quick one-liner
result = create_dataset_from_text(
    name="my-dataset",
    texts=["Sample 1", "Sample 2", "Sample 3"],
    output_path="dataset.jsonl",
    dedupe=True,
    min_chars=40,
    split_ratios=(0.9, 0.05, 0.05)
)

# Advanced: use DatasetBuilder for more control
builder = DatasetBuilder(name="custom-dataset")
builder.add_texts(["Text 1", "Text 2"], source="manual")
builder.add_files(["data.txt", "more.jsonl"])
builder.add_urls(["https://example.com"], selector="article p")
builder.dedupe()
builder.filter_min_length(100)
result = builder.build(
    output_path="dataset.jsonl",
    format="jsonl",  # or "json", "parquet"
    split_ratios=(0.9, 0.05, 0.05)
)

# Optional: push to HuggingFace Hub
builder.push_to_hub("username/dataset-name", private=False)
```

Created datasets are immediately usable with the training pipeline:

```python
cfg = RunConfig(
    dataset_id="dataset.jsonl",  # path to your created file
    # ... other config
)
```

## Architecture and performance notes

- `zigllm/src/core.zig` builds without third-party Zig packages and is safe to compile for Linux x86_64 in notebook environments.
- **Performance optimizations enabled by default**: parallel tokenization (multi-core), `DataCollatorForLanguageModeling` for dynamic padding, Flash Attention 2 / SDPA, fused AdamW optimizer, DataLoader workers with pinned memory and prefetching.
- **Optional performance toggles** (configurable in GUI and API): gradient checkpointing (saves ~40% VRAM at ~20% compute), `torch.compile()` (10-30% speedup with kernel fusion), streaming datasets (for datasets too large for RAM).
- **Web UI**: pure HTML + Flask, no Gradio dependency. All UI components use native CSS — no Gradio version-compatibility issues. Dark-themed, responsive, works in any browser.
- Use bf16 where supported, gradient accumulation, fast tokenizers, and QLoRA for limited VRAM.
- **Dataset Browser**: search any Hugging Face dataset from the GUI, filter by task/language/author, sort by downloads/likes/trending.
- For real high-throughput production training, add FSDP or DeepSpeed and multi-GPU launch. This starter intentionally leaves those as explicit next steps rather than silently making unsafe distributed-training assumptions.
- Scraping must comply with robots.txt, terms of service, copyright and dataset licenses. The helper only handles public pages; it is not a crawler.

## Status

This is a functional starter library/control plane, not a claim that arbitrary architectures can be trained with one universal checkpoint. Looped Transformer requires a model implementation/checkpoint with repeated-block semantics; Mamba requires a Mamba checkpoint. The GUI exposes the choice and the engine keeps the loading path transparent.
