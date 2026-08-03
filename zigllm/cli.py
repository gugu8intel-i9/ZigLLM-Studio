import argparse

def main():
    p=argparse.ArgumentParser(prog="zigllm")
    sub=p.add_subparsers(dest="command",required=True)
    sub.add_parser("gui",help="launch the Gradio UI")
    b=sub.add_parser("build-core",help="compile the Zig acceleration library")
    a=sub.add_parser("scrape"); a.add_argument("url"); a.add_argument("--selector")
    c=sub.add_parser("csv-to-parquet", help="stream-convert CSV to compressed Parquet")
    c.add_argument("input"); c.add_argument("output"); c.add_argument("--compression", default="zstd", choices=["zstd","snappy","gzip","brotli","none"])
    c.add_argument("--batch-size", type=int, default=100000)
    # Dataset search
    s=sub.add_parser("search-datasets", help="search Hugging Face datasets")
    s.add_argument("query", nargs="?", default="", help="Search query")
    s.add_argument("--sort", default="downloads", choices=["downloads","likes","trending","lastModified","createdAt"])
    s.add_argument("--limit", type=int, default=20, help="Number of results (max 100)")
    s.add_argument("--task", default="", help="Filter by task (e.g. text-generation, summarization)")
    s.add_argument("--language", default="", help="Filter by language code (e.g. en, code)")
    s.add_argument("--author", default="", help="Filter by author/org")
    
    # Dataset creation
    cd=sub.add_parser("create-dataset", help="create a custom training dataset")
    cd.add_argument("--name", required=True, help="Dataset name")
    cd.add_argument("--output", required=True, help="Output file path")
    cd.add_argument("--format", default="jsonl", choices=["jsonl","json","parquet"], help="Output format")
    cd.add_argument("--text", action="append", help="Text to add (can be repeated)")
    cd.add_argument("--files", action="append", help="File paths to ingest (can be repeated)")
    cd.add_argument("--urls", action="append", help="URLs to scrape (can be repeated)")
    cd.add_argument("--selector", default=None, help="CSS selector for URL scraping")
    cd.add_argument("--min-chars", type=int, default=40, help="Minimum character length (default: 40)")
    cd.add_argument("--max-chars", type=int, default=0, help="Maximum character length (0=unlimited)")
    cd.add_argument("--no-dedupe", action="store_true", help="Skip deduplication")
    cd.add_argument("--split-train", type=float, default=0.9, help="Train split ratio")
    cd.add_argument("--split-val", type=float, default=0.05, help="Validation split ratio")
    cd.add_argument("--split-test", type=float, default=0.05, help="Test split ratio")
    cd.add_argument("--push-hub", action="store_true", help="Upload to HuggingFace Hub (requires HF_TOKEN)")
    cd.add_argument("--hub-repo", default="", help="Hub repo ID (e.g. username/dataset-name)")
    cd.add_argument("--private", action="store_true", help="Make Hub dataset private")
    
    args=p.parse_args()

    if args.command=="gui":
        from .gui import launch; launch()
    elif args.command=="build-core":
        import subprocess; subprocess.run(["zig","build","-Doptimize=ReleaseFast"],check=True)
    elif args.command=="scrape":
        from .datasets import scrape
        print("\n".join(scrape(args.url,args.selector)))
    elif args.command=="csv-to-parquet":
        from .datasets import csv_to_parquet
        print(csv_to_parquet(args.input, args.output, None if args.compression == "none" else args.compression, args.batch_size))
    elif args.command=="search-datasets":
        from .datasets import search_datasets
        results = search_datasets(
            query=args.query,
            sort=args.sort,
            limit=args.limit,
            task=args.task,
            language=args.language,
            author=args.author,
        )
        if not results:
            print("No datasets found.")
        else:
            print(f"Found {len(results)} datasets (sorted by {args.sort}):\n")
            for i, ds in enumerate(results, 1):
                print(f"  {i:2}. {ds.id}")
                print(f"      ↓ {ds.downloads:,}  ❤ {ds.likes}  Tags: {', '.join(ds.tags[:5])}")
                desc = (ds.description[:100] + "…") if len(ds.description) > 100 else ds.description
                if desc:
                    print(f"      {desc}")
                print()
    elif args.command=="create-dataset":
        from .creator import DatasetBuilder
        if not (args.text or args.files or args.urls):
            print("✕ No input provided. Use --text, --files, or --urls to add content.")
            return
        builder = DatasetBuilder(name=args.name)
        # Ingest inputs
        if args.text:
            n = builder.add_texts(args.text, source="cli")
            print(f"✓ Added {n} text samples")
        if args.files:
            n = builder.add_files(args.files)
            print(f"✓ Ingested {n} samples from {len(args.files)} files")
        if args.urls:
            n = builder.add_urls(args.urls, selector=args.selector)
            print(f"✓ Scraped {n} samples from {len(args.urls)} URLs")
        # Build
        result = builder.build(
            output_path=args.output,
            format=args.format,
            dedupe=not args.no_dedupe,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            split_ratios=(args.split_train, args.split_val, args.split_test),
        )
        print(f"\n✓ Dataset created: {result['output']}")
        print(f"  Samples: {result['samples']}")
        print(f"  Removed: {result['removed']}")
        print(f"  Format: {result['format']}")
        print(f"  Metadata: {result['metadata']}")
        # Optional Hub upload
        if args.push_hub:
            if not args.hub_repo:
                print("\n✕ --push-hub requires --hub-repo (e.g. username/dataset-name)")
            else:
                try:
                    url = builder.push_to_hub(args.hub_repo, private=args.private)
                    print(f"\n✓ Uploaded to HuggingFace Hub: {url}")
                except Exception as e:
                    print(f"\n✕ Hub upload failed: {e}")
