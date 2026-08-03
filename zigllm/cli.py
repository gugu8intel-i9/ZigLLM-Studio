import argparse

def main():
    p=argparse.ArgumentParser(prog="zigllm")
    sub=p.add_subparsers(dest="command",required=True)
    sub.add_parser("gui",help="launch the Gradio UI")
    b=sub.add_parser("build-core",help="compile the Zig acceleration library")
    a=sub.add_parser("scrape"); a.add_argument("url"); a.add_argument("--selector")
    args=p.parse_args()
    if args.command=="gui":
        from .gui import launch; launch()
    elif args.command=="build-core":
        import subprocess; subprocess.run(["zig","build","-Doptimize=ReleaseFast"],check=True)
    elif args.command=="scrape":
        from .datasets import scrape
        print("\n".join(scrape(args.url,args.selector)))
