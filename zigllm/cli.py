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
