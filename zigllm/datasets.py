from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse
import re

@dataclass
class DatasetSource:
    provider: str
    identifier: str
    split: str = "train"
    text_column: str = "text"

def fetch_dataset(source: DatasetSource, max_samples: int = 0):
    """Load HF datasets or a local JSON/JSONL/CSV. Kaggle requires kagglehub."""
    if source.provider == "huggingface":
        from datasets import load_dataset
        ds = load_dataset(source.identifier, split=source.split)
    elif source.provider == "kaggle":
        try:
            import kagglehub
            path = kagglehub.dataset_download(source.identifier)
        except ImportError as e:
            raise RuntimeError("Install kagglehub to download Kaggle datasets") from e
        from pathlib import Path
        from datasets import load_dataset
        root = Path(path)
        files = [str(p) for p in root.rglob("*") if p.suffix.lower() in (".json", ".jsonl", ".csv")]
        if not files: raise RuntimeError(f"No JSON/JSONL/CSV file found in Kaggle download: {path}")
        ds = load_dataset("json" if files[0].endswith((".json", ".jsonl")) else "csv", data_files=files[0], split="train")
    elif source.provider == "local":
        from datasets import load_dataset
        ds = load_dataset("json" if source.identifier.endswith((".json", ".jsonl")) else "csv", data_files=source.identifier, split="train")
    else: raise ValueError("provider must be huggingface, kaggle, or local")
    return ds.select(range(min(max_samples, len(ds)))) if max_samples else ds

def scrape(url: str, selector: Optional[str] = None) -> list[str]:
    """Respect robots.txt/terms and only scrape public pages you are allowed to use."""
    import requests
    from bs4 import BeautifulSoup
    if urlparse(url).scheme not in ("http", "https"): raise ValueError("URL must use http(s)")
    r = requests.get(url, timeout=20, headers={"User-Agent": "zigllm-dataset/0.1"}); r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    nodes = soup.select(selector) if selector else soup.find_all(["p", "article", "main"])
    out = [re.sub(r"\s+", " ", n.get_text(" ", strip=True)) for n in nodes]
    return [x for x in out if len(x) >= 40]
