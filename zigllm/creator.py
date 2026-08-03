"""Dataset creation and assembly.

Build structured training datasets from raw text, local files, and web pages.
Output JSONL / JSON / Parquet that the existing fetch_dataset("local", ...) loader
can consume directly — or push to the Hugging Face Hub.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone
import hashlib, json, os


@dataclass
class DatasetBuilder:
    """Collect, clean, and export text samples into a training-ready dataset."""
    name: str
    text_column: str = "text"
    source_column: str = "source"
    samples: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.name:
            raise ValueError("Dataset name is required")
        if not self.metadata:
            self.metadata = {
                "created": datetime.now(timezone.utc).isoformat(),
                "creator": "zigllm-studio",
                "version": "0.2.0",
            }

    # ── Ingest ──────────────────────────────────────────────────────────

    def add_text(self, text: str, source: str = "manual", split: str = "") -> bool:
        """Add a single text sample. Returns True if accepted (non-empty after strip)."""
        text = text.strip()
        if not text:
            return False
        self.samples.append({self.text_column: text, self.source_column: source, "split": split})
        return True

    def add_texts(self, texts: list[str], source: str = "manual", split: str = "") -> int:
        """Add multiple text samples. Returns number accepted."""
        count = 0
        for t in texts:
            if self.add_text(t, source=source, split=split):
                count += 1
        return count

    def add_files(self, paths: list[str], split: str = "") -> int:
        """Read .txt / .md / .json / .jsonl files and ingest their contents.

        - .txt / .md: each file becomes one sample (or one per paragraph if >5000 chars)
        - .json / .jsonl: each record's text_column value becomes a sample
        """
        count = 0
        for p_str in paths:
            p = Path(p_str)
            if not p.exists():
                continue
            suffix = p.suffix.lower()
            source = str(p)
            try:
                if suffix in (".txt", ".md", ".markdown"):
                    text = p.read_text(encoding="utf-8", errors="ignore").strip()
                    if not text:
                        continue
                    # Split long documents into paragraphs for better training granularity
                    if len(text) > 5000:
                        paragraphs = [para.strip() for para in text.split("\n\n") if para.strip()]
                        for para in paragraphs:
                            if self.add_text(para, source=source, split=split):
                                count += 1
                    else:
                        if self.add_text(text, source=source, split=split):
                            count += 1
                elif suffix == ".json":
                    data = json.loads(p.read_text(encoding="utf-8"))
                    if isinstance(data, list):
                        for item in data:
                            text = item.get(self.text_column, "") if isinstance(item, dict) else str(item)
                            if self.add_text(text, source=source, split=split):
                                count += 1
                    elif isinstance(data, dict) and self.text_column in data:
                        if self.add_text(data[self.text_column], source=source, split=split):
                            count += 1
                elif suffix == ".jsonl":
                    for line in p.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                            text = item.get(self.text_column, "") if isinstance(item, dict) else str(item)
                            if self.add_text(text, source=source, split=split):
                                count += 1
                        except json.JSONDecodeError:
                            continue
            except Exception:
                continue
        return count

    def add_urls(self, urls: list[str], selector: Optional[str] = None, split: str = "") -> int:
        """Scrape public web pages and ingest text blocks (respects robots.txt/terms)."""
        from .datasets import scrape
        count = 0
        for url in urls:
            try:
                texts = scrape(url, selector)
                for t in texts:
                    if self.add_text(t, source=url, split=split):
                        count += 1
            except Exception:
                continue
        return count

    # ── Clean ───────────────────────────────────────────────────────────

    def dedupe(self) -> int:
        """Remove duplicate samples (by text content hash). Returns number removed."""
        seen = set()
        unique = []
        removed = 0
        for s in self.samples:
            h = hashlib.sha256(s[self.text_column].encode("utf-8")).hexdigest()
            if h in seen:
                removed += 1
                continue
            seen.add(h)
            unique.append(s)
        self.samples = unique
        return removed

    def filter_min_length(self, min_chars: int) -> int:
        """Remove samples shorter than min_chars. Returns number removed."""
        before = len(self.samples)
        self.samples = [s for s in self.samples if len(s[self.text_column]) >= min_chars]
        return before - len(self.samples)

    def filter_max_length(self, max_chars: int) -> int:
        """Remove samples longer than max_chars. Returns number removed."""
        before = len(self.samples)
        self.samples = [s for s in self.samples if len(s[self.text_column]) <= max_chars]
        return before - len(self.samples)

    # ── Split ───────────────────────────────────────────────────────────

    def assign_splits(self, train: float = 0.9, val: float = 0.05, test: float = 0.05):
        """Assign train/val/test split labels based on ratios (sum should be ~1.0)."""
        total = len(self.samples)
        if total == 0:
            return
        # Shuffle deterministically so splits are reproducible
        import random
        rng = random.Random(42)
        indices = list(range(total))
        rng.shuffle(indices)
        n_train = int(total * train)
        n_val = int(total * val)
        for i, idx in enumerate(indices):
            if i < n_train:
                self.samples[idx]["split"] = "train"
            elif i < n_train + n_val:
                self.samples[idx]["split"] = "validation"
            else:
                self.samples[idx]["split"] = "test"

    # ── Preview ─────────────────────────────────────────────────────────

    def preview(self, n: int = 5) -> list[dict]:
        """Return first n samples for inspection."""
        return self.samples[:n]

    def stats(self) -> dict:
        """Return summary statistics of the current dataset."""
        if not self.samples:
            return {"count": 0, "text_column": self.text_column, "name": self.name}
        lengths = [len(s[self.text_column]) for s in self.samples]
        splits = {}
        for s in self.samples:
            sp = s.get("split", "unassigned")
            splits[sp] = splits.get(sp, 0) + 1
        sources = {}
        for s in self.samples:
            src = s.get(self.source_column, "unknown")
            sources[src] = sources.get(src, 0) + 1
        return {
            "count": len(self.samples),
            "text_column": self.text_column,
            "name": self.name,
            "avg_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "total_chars": sum(lengths),
            "splits": splits,
            "top_sources": sorted(sources.items(), key=lambda x: -x[1])[:10],
        }

    # ── Export ──────────────────────────────────────────────────────────

    def build(
        self,
        output_path: str,
        format: str = "jsonl",
        split_ratios: tuple[float, float, float] = (0.9, 0.05, 0.05),
        dedupe: bool = True,
        min_chars: int = 0,
        max_chars: int = 0,
    ) -> dict:
        """Clean and export the dataset.

        Args:
            output_path: file path (extension auto-applied if missing)
            format: "jsonl", "json", or "parquet"
            split_ratios: (train, val, test) ratios; set all to 0 to keep existing splits
            dedupe: remove duplicate texts
            min_chars: drop samples shorter than this (0 = no filter)
            max_chars: drop samples longer than this (0 = no filter)

        Returns:
            dict with output path, row counts, and stats
        """
        if not self.samples:
            raise ValueError("No samples to export. Add texts first.")

        # Cleaning pipeline
        removed = {}
        if dedupe:
            removed["duplicates"] = self.dedupe()
        if min_chars > 0:
            removed["too_short"] = self.filter_min_length(min_chars)
        if max_chars > 0:
            removed["too_long"] = self.filter_max_length(max_chars)

        # Assign splits
        if sum(split_ratios) > 0:
            self.assign_splits(*split_ratios)

        # Determine output path
        out = Path(output_path)
        if format == "jsonl" and out.suffix not in (".jsonl",):
            out = out.with_suffix(".jsonl")
        elif format == "json" and out.suffix not in (".json",):
            out = out.with_suffix(".json")
        elif format == "parquet" and out.suffix not in (".parquet",):
            out = out.with_suffix(".parquet")

        out.parent.mkdir(parents=True, exist_ok=True)

        if format == "jsonl":
            with open(out, "w", encoding="utf-8") as f:
                for sample in self.samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        elif format == "json":
            with open(out, "w", encoding="utf-8") as f:
                json.dump(self.samples, f, ensure_ascii=False, indent=2)
        elif format == "parquet":
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
            except ImportError:
                raise RuntimeError("Install pyarrow for Parquet export: pip install pyarrow")
            table = pa.Table.from_pylist(self.samples)
            pq.write_table(table, str(out), compression="zstd", use_dictionary=True)
        else:
            raise ValueError(f"Unknown format: {format}. Use jsonl, json, or parquet.")

        # Write metadata sidecar
        meta_path = out.with_suffix(".meta.json")
        meta = {
            **self.metadata,
            "name": self.name,
            "text_column": self.text_column,
            "source_column": self.source_column,
            "sample_count": len(self.samples),
            "removed": removed,
            "split_ratios": list(split_ratios),
            "splits": {},
        }
        for s in self.samples:
            sp = s.get("split", "unassigned")
            meta["splits"][sp] = meta["splits"].get(sp, 0) + 1
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        return {
            "output": str(out),
            "metadata": str(meta_path),
            "samples": len(self.samples),
            "removed": removed,
            "format": format,
            "name": self.name,
            "splits": meta["splits"],
        }

    def push_to_hub(self, repo_id: str, token: Optional[str] = None, private: bool = False) -> str:
        """Push the dataset to Hugging Face Hub.

        Requires the `huggingface_hub` and `datasets` packages.
        Returns the URL of the uploaded dataset.
        """
        try:
            from datasets import Dataset, DatasetDict
            from huggingface_hub import HfApi
        except ImportError as e:
            raise RuntimeError(
                "Install huggingface_hub and datasets: pip install huggingface_hub datasets"
            ) from e

        if not token:
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not token:
            raise RuntimeError("HuggingFace token required. Set HF_TOKEN env var or pass token=")

        # Build per-split datasets
        splits = {}
        for s in self.samples:
            sp = s.get("split", "train")
            splits.setdefault(sp, []).append(s)

        ds_dict = {}
        for sp, items in splits.items():
            ds_dict[sp] = Dataset.from_list(items)
        ds = DatasetDict(ds_dict)

        ds.push_to_hub(repo_id, token=token, private=private)
        return f"https://huggingface.co/datasets/{repo_id}"


def create_dataset_from_text(
    name: str,
    texts: list[str],
    output_path: str,
    source: str = "manual",
    format: str = "jsonl",
    dedupe: bool = True,
    min_chars: int = 40,
    max_chars: int = 0,
    split_ratios: tuple[float, float, float] = (0.9, 0.05, 0.05),
) -> dict:
    """Convenience: build a dataset from a list of text strings."""
    builder = DatasetBuilder(name=name)
    builder.add_texts(texts, source=source)
    return builder.build(
        output_path=output_path,
        format=format,
        dedupe=dedupe,
        min_chars=min_chars,
        max_chars=max_chars,
        split_ratios=split_ratios,
    )


def create_dataset_from_files(
    name: str,
    file_paths: list[str],
    output_path: str,
    format: str = "jsonl",
    dedupe: bool = True,
    min_chars: int = 40,
    split_ratios: tuple[float, float, float] = (0.9, 0.05, 0.05),
) -> dict:
    """Convenience: build a dataset by ingesting local text/json/jsonl files."""
    builder = DatasetBuilder(name=name)
    builder.add_files(file_paths)
    return builder.build(
        output_path=output_path,
        format=format,
        dedupe=dedupe,
        min_chars=min_chars,
        split_ratios=split_ratios,
    )


def create_dataset_from_urls(
    name: str,
    urls: list[str],
    output_path: str,
    selector: Optional[str] = None,
    format: str = "jsonl",
    dedupe: bool = True,
    min_chars: int = 40,
    split_ratios: tuple[float, float, float] = (0.9, 0.05, 0.05),
) -> dict:
    """Convenience: build a dataset by scraping public web pages."""
    builder = DatasetBuilder(name=name)
    builder.add_urls(urls, selector=selector)
    return builder.build(
        output_path=output_path,
        format=format,
        dedupe=dedupe,
        min_chars=min_chars,
        split_ratios=split_ratios,
    )
