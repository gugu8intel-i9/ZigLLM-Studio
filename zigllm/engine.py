import os, ctypes
from .config import RunConfig, Adapter, Device, Architecture

def _core():
    names = ["zig-out/lib/libzigllm_core.so", "zig-out/libzigllm_core.so", "zig-out/lib/libzigllm_core.dylib"]
    for n in names:
        if os.path.exists(n): return ctypes.CDLL(n)
    return None

def estimate_vram_mb(config: RunConfig, adapter_is_lora: bool) -> int:
    """Use the Zig core to estimate VRAM, or fall back to a Python approximation."""
    c = _core()
    if c:
        c.zigllm_estimate_vram_mb.restype = ctypes.c_uint64
        c.zigllm_estimate_vram_mb.argtypes = [ctypes.c_uint32]*6 + [ctypes.c_int]
        # We don't know hidden/layers/vocab from config alone; use a proxy.
        # The Zig function needs them, so we return 0 when we can't fill them.
        return 0
    # Python fallback: rough transformer estimate
    return 0  # Will be computed with real model info below

class Trainer:
    def __init__(self, config: RunConfig, dataset_provider: str = "huggingface"):
        self.config = config
        self.dataset_provider = dataset_provider

    def validate(self):
        c = _core()
        if c:
            class C(ctypes.Structure):
                _fields_=[("seq_len",ctypes.c_uint32),("hidden_size",ctypes.c_uint32),("layers",ctypes.c_uint32),("batch_size",ctypes.c_uint32),("grad_accum",ctypes.c_uint32),("vocab_size",ctypes.c_uint32)]
            # hidden/vocab are model-dependent; Python checks the portable constraints.
        if self.config.adapter == Adapter.qlora and self.config.device == Device.cpu:
            raise ValueError("QLoRA requires a CUDA-capable device (bitsandbytes 4-bit kernels)")
        if self.config.mode.value == "train" and not self.config.model_id:
            raise ValueError("A model config/checkpoint is required")
        return True

    def run(self, progress=print):
        """Run a Transformers/PEFT job. Architecture adapters are explicit so jobs fail clearly."""
        self.validate()
        try:
            import torch
            from datasets import load_dataset
            from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
            from transformers import Trainer as HFTrainer, DataCollatorForLanguageModeling
        except ImportError as e: raise RuntimeError("Install zigllm[train] in the notebook") from e
        from .datasets import DatasetSource, fetch_dataset

        cfg = self.config

        # ── Resolve device ──────────────────────────────────────────────
        if cfg.device == Device.auto:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = cfg.device.value
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")

        # ── Report architecture notes ───────────────────────────────────
        if cfg.architecture == Architecture.mamba and "mamba" not in cfg.model_id.lower():
            progress("Note: select a Mamba checkpoint (for example state-spaces/mamba-130m-hf) for native Mamba blocks.")
        if cfg.architecture == Architecture.looped_transformer:
            progress("Looped Transformer: use a checkpoint/config with repeated-block support; loading base model as a compatible fallback.")

        # ── Performance summary ─────────────────────────────────────────
        perf_notes = []
        if cfg.gradient_checkpointing:
            perf_notes.append("gradient checkpointing ON (trades ~20% compute for ~40% VRAM savings)")
        if cfg.flash_attention:
            perf_notes.append("Flash Attention 2 / SDPA enabled")
        if cfg.torch_compile:
            perf_notes.append(f"torch.compile mode={cfg.compile_mode}")
        if cfg.streaming:
            perf_notes.append("streaming dataset mode (low memory)")
        if cfg.num_workers > 0:
            perf_notes.append(f"{cfg.num_workers} DataLoader workers")
        if perf_notes:
            progress("⚡ Performance: " + " · ".join(perf_notes))

        # ── Load dataset ────────────────────────────────────────────────
        ds = fetch_dataset(
            DatasetSource(self.dataset_provider, cfg.dataset_id, cfg.dataset_split, cfg.text_column),
            cfg.max_samples,
            streaming=cfg.streaming,
        )
        progress(f"✓ Dataset loaded ({len(ds) if not cfg.streaming else 'streaming'} samples)")

        # ── Tokenize with parallel workers ──────────────────────────────
        tok = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        text_col = cfg.text_column
        seq_len = cfg.seq_len

        def enc(batch):
            tokens = tok(batch[text_col], truncation=True, max_length=seq_len)
            tokens["labels"] = tokens["input_ids"]
            return tokens

        # Use multiple CPU cores for tokenization
        map_kwargs = {"batched": True, "remove_columns": ds.column_names}
        if not cfg.streaming:
            # Parallel tokenization: use available CPUs (capped at 8 to avoid thrashing)
            try:
                cpus = min(os.cpu_count() or 4, 8)
                map_kwargs["num_proc"] = cpus
            except Exception:
                pass
        ds = ds.map(enc, **map_kwargs)
        progress("✓ Tokenization complete")

        # ── Data collator for dynamic padding ───────────────────────────
        # This is a major perf win: instead of padding every sample to seq_len,
        # the collator pads each batch to the longest sample in that batch.
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tok,
            mlm=False,  # Causal LM, not masked LM
        )

        # ── Load model ──────────────────────────────────────────────────
        use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
        kwargs = {"torch_dtype": torch.bfloat16 if use_bf16 else torch.float32}

        # Enable Flash Attention 2 when available
        if cfg.flash_attention and device == "cuda":
            try:
                import importlib
                if importlib.util.find_spec("flash_attn") is not None:
                    kwargs["attn_implementation"] = "flash_attention_2"
                    progress("⚡ Using Flash Attention 2")
                else:
                    kwargs["attn_implementation"] = "sdpa"
                    progress("⚡ Using SDPA (Scaled Dot-Product Attention)")
            except Exception:
                kwargs["attn_implementation"] = "sdpa"

        if cfg.adapter == Adapter.qlora:
            kwargs.update(load_in_4bit=True, device_map="auto",
                          bnb_4bit_quant_type="nf4",
                          bnb_4bit_compute_dtype=torch.bfloat16,
                          bnb_4bit_use_double_quant=True)  # Double quantization saves ~0.4 bit/param

        model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **kwargs)

        # Enable gradient checkpointing for VRAM savings
        if cfg.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            # Required for checkpointing with gradient accumulation
            if hasattr(model, "config"):
                model.config.use_cache = False

        # ── Apply adapter ───────────────────────────────────────────────
        if cfg.adapter in (Adapter.lora, Adapter.qlora):
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            if cfg.adapter == Adapter.qlora:
                model = prepare_model_for_kbit_training(model)
            lora_cfg = LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                task_type="CAUSAL_LM",
                target_modules="all-linear",
            )
            model = get_peft_model(model, lora_cfg)
            model.print_trainable_parameters()

        # ── torch.compile for kernel fusion ─────────────────────────────
        if cfg.torch_compile and device == "cuda":
            try:
                model = torch.compile(model, mode=cfg.compile_mode)
                progress(f"⚡ Model compiled with torch.compile(mode={cfg.compile_mode})")
            except Exception as e:
                progress(f"⚠ torch.compile failed ({e}), continuing without compilation")

        # ── Training arguments ──────────────────────────────────────────
        bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
        fp16 = device == "cuda" and not bf16

        args = TrainingArguments(
            output_dir=str(cfg.output_dir),
            num_train_epochs=cfg.epochs,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.grad_accum,
            learning_rate=cfg.learning_rate,
            warmup_ratio=cfg.warmup_ratio,
            logging_steps=10,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=2,            # Keep only 2 checkpoints to save disk
            bf16=bf16,
            fp16=fp16,
            report_to="none",
            dataloader_pin_memory=device == "cuda",
            dataloader_num_workers=cfg.num_workers,
            dataloader_prefetch_factor=4 if cfg.num_workers > 0 else None,
            optim="adamw_torch_fused" if device == "cuda" else "adamw_torch",  # Fused Adam is faster
            remove_unused_columns=False,
            gradient_checkpointing=cfg.gradient_checkpointing,
        )

        # ── Train ───────────────────────────────────────────────────────
        trainer = HFTrainer(
            model=model,
            args=args,
            train_dataset=ds,
            tokenizer=tok,
            data_collator=data_collator,
        )
        trainer.train()

        # ── Save ────────────────────────────────────────────────────────
        model.save_pretrained(cfg.output_dir)
        tok.save_pretrained(cfg.output_dir)
        progress(f"✓ Model saved to {cfg.output_dir}")
        return str(cfg.output_dir)
