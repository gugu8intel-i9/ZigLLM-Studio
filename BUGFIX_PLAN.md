# ZigLLM Studio — Bug Fix Plan

## Summary
After thorough analysis of the entire codebase (Zig core + Python training engine + Gradio GUI + CLI + datasets), **3 bugs** were identified and fixed.

---

## Bug 1: GUI output displays literal `\n` instead of line breaks
**File:** `zigllm/gui.py` — Lines 72 and 81  
**Severity:** Medium (UI readability broken for two features)

**Problem:**  
The `scrape_page()` and `build_core()` functions use `\\n` (escaped backslash-n) in their return strings. In Python source, `\\n` produces the literal two-character string `\n` (backslash + n), not an actual newline. The Gradio textboxes show literal `\n\n` instead of line breaks.

**Root cause:** Double-escaping of newline characters.

**Fix:** Replace `\\n` with `\n` in both functions.

---

## Bug 2: Device "auto" not resolved to "cpu" when CUDA unavailable
**File:** `zigllm/engine.py` — `Trainer.run()` method  
**Severity:** Low-Medium (logic error; doesn't crash currently but semantically wrong)

**Problem:**  
The device resolution line:
```python
device = "cuda" if cfg.device.value=="auto" and torch.cuda.is_available() else cfg.device.value
```
When `device=="auto"` and CUDA is **not** available, the else-branch returns `cfg.device.value` which is still `"auto"` — not `"cpu"`. The variable stays as the unresolved string `"auto"`.

**Root cause:** The ternary only handles the success case; the fallback should be `"cpu"`, not `cfg.device.value`.

**Fix:** Use explicit if/elif/else resolution:
```python
if cfg.device == Device.auto:
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = cfg.device.value
```

---

## Bug 3: Missing `labels` column crashes training
**File:** `zigllm/engine.py` — `Trainer.run()`, `enc()` function  
**Severity:** **Critical** (training will crash immediately with `AttributeError` on `loss.backward()`)

**Problem:**  
The tokenization function:
```python
def enc(batch): return tok(batch[cfg.text_column], truncation=True, max_length=cfg.seq_len)
```
Only produces `input_ids` and `attention_mask`. For causal language modeling, HuggingFace models require a `labels` column to compute the cross-entropy loss. Without it, the model returns `loss=None`, and the HF Trainer crashes when it tries to call `.backward()` on `None`.

**Root cause:** Missing label assignment during tokenization.

**Fix:** Assign `labels = input_ids` in the tokenization function:
```python
def enc(batch):
    tokens = tok(batch[cfg.text_column], truncation=True, max_length=cfg.seq_len)
    tokens["labels"] = tokens["input_ids"]
    return tokens
```

---

## Files Modified
1. `zigllm/gui.py` — Fix escaped newlines (Bug 1)
2. `zigllm/engine.py` — Fix device resolution (Bug 2) and add labels column (Bug 3)
