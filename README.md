# ZigLLM Studio

A Kaggle/Google Colab-friendly LLM training control plane with a small, dependency-free Zig core for fast validation and sizing. PyTorch/Transformers run the GPU kernels; Zig is not pretending to replace CUDA/autograd.

## Features

- Gradio GUI for **train** or **fine-tune**, CPU/CUDA/auto selection.
- Transformer, Looped Transformer and Mamba architecture choices. Transformer is native through `AutoModelForCausalLM`; Mamba and looped models accept compatible Hugging Face checkpoints and report when a specialized checkpoint is needed.
- Full training, LoRA and QLoRA (4-bit bitsandbytes on CUDA).
- Hugging Face dataset loading, Kaggle download support through `kagglehub`, local JSON/JSONL/CSV, and a polite public-page scraper.
- Benchmark tab for SWE-bench, GMSK8/GSM8K, HLE, CyberGym, and HellaSwag. GSM8K and HellaSwag can run through `lm-eval`; the other evaluations expose their official harness path because they require patches, tool traces, licensed data, or a sandbox.
- Streaming CSV → compressed Parquet conversion with PyArrow, exposed in both the GUI and CLI. It uses record batches and dictionary encoding for efficient dataset storage.
- Every CLI utility is also exposed in the GUI: CSV conversion, public-page scraping, and ReleaseFast Zig-core builds.
- Zig `ReleaseFast` shared library for stable config validation and token-step arithmetic.

## Colab / Kaggle quick start

```python
%cd /content
!git clone https://github.com/gugu8intel-i9/ZigLLM-Studio.git zigllm
%cd /content/zigllm
!python -m pip install -e '.[train,data,bench]'
# Optional: install Zig separately for the ReleaseFast core build.
# The Python training/UI layer does not require Zig to launch.
!python -m pip install ziglang
!zig version
!zig build -Doptimize=ReleaseFast
!zigllm gui
```

If the repository was already cloned, do not run `git clone` again. Run:

```python
%cd /content/zigllm
!python -m pip install -e '.[train,data,bench]'
!zigllm gui
```

Use a public tunnel supplied by the notebook environment if you need to access the Gradio UI remotely. In Kaggle, add Hugging Face/Kaggle credentials through the platform's secret manager, never in a notebook cell.

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

## Architecture and performance notes

- `zigllm/src/core.zig` builds without third-party Zig packages and is safe to compile for Linux x86_64 in notebook environments.
- Use bf16 where supported, gradient accumulation, fast tokenizers, pinned dataloaders (configured by Transformers), and QLoRA for limited VRAM.
- For real high-throughput production training, add FlashAttention/SDPA, FSDP or DeepSpeed, streaming datasets, checkpoint resume, and multi-GPU launch. This starter intentionally leaves those as explicit next steps rather than silently making unsafe distributed-training assumptions.
- Scraping must comply with robots.txt, terms of service, copyright and dataset licenses. The helper only handles a single public page; it is not a crawler.

## Status

This is a functional starter library/control plane, not a claim that arbitrary architectures can be trained with one universal checkpoint. Looped Transformer requires a model implementation/checkpoint with repeated-block semantics; Mamba requires a Mamba checkpoint. The GUI exposes the choice and the engine keeps the loading path transparent.
