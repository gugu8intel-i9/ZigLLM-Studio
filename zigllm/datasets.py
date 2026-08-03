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

def csv_to_parquet(input_path: str, output_path: str, compression: str = "zstd", batch_size: int = 100_000) -> dict:
    """Convert CSV to compressed Parquet in record batches, avoiding full-RAM loads."""
    try:
        import pyarrow as pa
        import pyarrow.csv as pacsv
        import pyarrow.parquet as pq
    except ImportError as e:
        raise RuntimeError("Install pyarrow to convert CSV: pip install pyarrow") from e
    read_options = pacsv.ReadOptions(block_size=max(1, int(batch_size)) * 1024)
    reader = pacsv.open_csv(input_path, read_options=read_options,
                            convert_options=pacsv.ConvertOptions(strings_can_be_null=True))
    writer = None; rows = 0; batches = 0
    try:
        for batch in reader:
            if writer is None:
                writer = pq.ParquetWriter(output_path, pa.schema(batch.schema), compression=compression, use_dictionary=True)
            writer.write_batch(batch); rows += batch.num_rows; batches += 1
    finally:
        if writer is not None: writer.close()
    if writer is None: raise ValueError("CSV contains no rows or header")
    return {"input": input_path, "output": output_path, "rows": rows, "batches": batches, "compression": compression}

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
