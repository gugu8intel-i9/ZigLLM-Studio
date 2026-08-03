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
