import os, ctypes
from .config import RunConfig, Adapter, Device, Architecture

def _core():
    names = ["zig-out/lib/libzigllm_core.so", "zig-out/libzigllm_core.so", "zig-out/lib/libzigllm_core.dylib"]
    for n in names:
        if os.path.exists(n): return ctypes.CDLL(n)
    return None

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
            from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer as HFTrainer
        except ImportError as e: raise RuntimeError("Install zigllm[train] in the notebook") from e
        from .datasets import DatasetSource, fetch_dataset
        cfg=self.config
        if cfg.device == Device.auto:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = cfg.device.value
        if device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
        if cfg.architecture == Architecture.mamba and "mamba" not in cfg.model_id.lower():
            progress("Note: select a Mamba checkpoint (for example state-spaces/mamba-130m-hf) for native Mamba blocks.")
        if cfg.architecture == Architecture.looped_transformer:
            progress("Looped Transformer: use a checkpoint/config with repeated-block support; loading base model as a compatible fallback.")
        ds=fetch_dataset(DatasetSource(self.dataset_provider, cfg.dataset_id, cfg.dataset_split, cfg.text_column), cfg.max_samples)
        tok=AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)
        if tok.pad_token is None: tok.pad_token=tok.eos_token
        def enc(batch):
            tokens = tok(batch[cfg.text_column], truncation=True, max_length=cfg.seq_len)
            tokens["labels"] = tokens["input_ids"]
            return tokens
        ds=ds.map(enc, batched=True, remove_columns=ds.column_names)
        kwargs={"torch_dtype": torch.bfloat16 if device=="cuda" and torch.cuda.is_bf16_supported() else torch.float32}
        if cfg.adapter==Adapter.qlora:
            kwargs.update(load_in_4bit=True, device_map="auto", bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        model=AutoModelForCausalLM.from_pretrained(cfg.model_id, **kwargs)
        if cfg.adapter in (Adapter.lora, Adapter.qlora):
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            if cfg.adapter==Adapter.qlora: model=prepare_model_for_kbit_training(model)
            model=get_peft_model(model, LoraConfig(r=cfg.lora_rank,lora_alpha=cfg.lora_alpha,lora_dropout=cfg.lora_dropout,task_type="CAUSAL_LM",target_modules="all-linear"))
            model.print_trainable_parameters()
        args=TrainingArguments(output_dir=str(cfg.output_dir), num_train_epochs=cfg.epochs, per_device_train_batch_size=cfg.batch_size, gradient_accumulation_steps=cfg.grad_accum, learning_rate=cfg.learning_rate, logging_steps=10, save_strategy="steps", save_steps=500, bf16=device=="cuda" and torch.cuda.is_bf16_supported(), fp16=device=="cuda" and not torch.cuda.is_bf16_supported(), report_to="none")
        HFTrainer(model=model,args=args,train_dataset=ds,tokenizer=tok).train()
        model.save_pretrained(cfg.output_dir); tok.save_pretrained(cfg.output_dir)
        return str(cfg.output_dir)
