from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse, urlencode
import re

@dataclass
class DatasetSource:
    provider: str
    identifier: str
    split: str = "train"
    text_column: str = "text"

@dataclass
class DatasetResult:
    """A single dataset from the Hugging Face Hub search API."""
    id: str
    author: str
    downloads: int = 0
    likes: int = 0
    tags: list = field(default_factory=list)
    description: str = ""
    url: str = ""

    def summary(self) -> str:
        tags_str = ", ".join(self.tags[:5]) if self.tags else "—"
        desc = (self.description[:120] + "…") if len(self.description) > 120 else self.description
        return f"{self.id}\n  ↓ {self.downloads:,}  ❤ {self.likes}  Tags: {tags_str}\n  {desc}"


def search_datasets(
    query: str = "",
    sort: str = "downloads",
    direction: str = "-1",
    limit: int = 30,
    task: str = "",
    language: str = "",
    author: str = "",
) -> list[DatasetResult]:
    """Search Hugging Face datasets using the public Hub API.

    Mirrors the search on huggingface.co/datasets:
      - query: free-text search
      - sort: downloads, likes, trending, lastModified, createdAt
      - direction: -1 (descending) or 1 (ascending)
      - task: filter by task category (e.g. text-generation, summarization, question-answering)
      - language: filter by language code (e.g. en, fr, code)
      - author: filter by author/org
    """
    import requests

    params = {}
    if query:
        params["search"] = query
    if sort:
        params["sort"] = sort
    if direction:
        params["direction"] = direction
    params["limit"] = str(min(limit, 100))
    if task:
        params["task_categories"] = task
    if language:
        params["language"] = language
    if author:
        params["author"] = author

    url = f"https://huggingface.co/api/datasets?{urlencode(params)}"
    resp = requests.get(url, timeout=30, headers={"User-Agent": "zigllm-studio/0.2"})
    resp.raise_for_status()
    data = resp.json()

    results = []
    for item in data:
        tags = item.get("tags", [])
        results.append(DatasetResult(
            id=item.get("id", ""),
            author=item.get("author", item.get("id", "").split("/")[0] if "/" in item.get("id", "") else ""),
            downloads=item.get("downloads", 0),
            likes=item.get("likes", 0),
            tags=tags[:8],
            description=item.get("description", ""),
            url=f"https://huggingface.co/datasets/{item.get('id', '')}",
        ))
    return results


def get_dataset_info(dataset_id: str) -> dict:
    """Fetch metadata for a specific dataset from the Hub API."""
    import requests
    url = f"https://huggingface.co/api/datasets/{dataset_id}"
    resp = requests.get(url, timeout=20, headers={"User-Agent": "zigllm-studio/0.2"})
    resp.raise_for_status()
    data = resp.json()
    return {
        "id": data.get("id", dataset_id),
        "author": data.get("author", ""),
        "downloads": data.get("downloads", 0),
        "likes": data.get("likes", 0),
        "tags": data.get("tags", []),
        "description": data.get("description", ""),
        "url": f"https://huggingface.co/datasets/{dataset_id}",
        "siblings_count": len(data.get("siblings", [])),
    }


def list_splits(dataset_id: str) -> list[str]:
    """List available splits for a dataset via the Hub API."""
    import requests
    url = f"https://huggingface.co/api/datasets/{dataset_id}"
    resp = requests.get(url, timeout=20, headers={"User-Agent": "zigllm-studio/0.2"})
    resp.raise_for_status()
    data = resp.json()
    # The API returns parquet file info which encodes splits
    splits = set()
    for sib in data.get("siblings", []):
        fname = sib.get("rfilename", "")
        if "/" in fname:
            parts = fname.split("/")
            if len(parts) >= 2:
                splits.add(parts[0])
    return sorted(splits) if splits else ["train"]


def fetch_dataset(source: DatasetSource, max_samples: int = 0, streaming: bool = False):
    """Load HF datasets or a local JSON/JSONL/CSV. Kaggle requires kagglehub.

    When streaming=True, returns an IterableDataset that yields samples without
    downloading the full dataset — ideal for very large datasets or limited RAM.
    """
    if source.provider == "huggingface":
        from datasets import load_dataset
        ds = load_dataset(source.identifier, split=source.split, streaming=streaming)
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
    if max_samples and not streaming:
        return ds.select(range(min(max_samples, len(ds))))
    return ds


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
    r = requests.get(url, timeout=20, headers={"User-Agent": "zigllm-dataset/0.2"}); r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    nodes = soup.select(selector) if selector else soup.find_all(["p", "article", "main"])
    out = [re.sub(r"\s+", " ", n.get_text(" ", strip=True)) for n in nodes]
    return [x for x in out if len(x) >= 40]
