import importlib.util
import subprocess
import sys

_PACKAGES = {
    "torch": "torch>=2.2",
    "transformers": "transformers>=4.40",
    "datasets": "datasets>=2.18",
    "accelerate": "accelerate>=0.28",
    "peft": "peft>=0.10",
    "gradio": "gradio>=4.0",
    "bitsandbytes": "bitsandbytes>=0.43",
    "pyarrow": "pyarrow>=15",
}

def ensure_dependencies(kind: str = "training", qlora: bool = False) -> list[str]:
    """Install missing runtime packages in notebook environments only when needed."""
    names = ["torch", "transformers", "datasets", "accelerate", "peft"] if kind == "training" else ["pyarrow"]
    if qlora: names.append("bitsandbytes")
    installed = []
    missing = [name for name in names if importlib.util.find_spec(name) is None]
    if missing:
        specs = [_PACKAGES[x] for x in missing]
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *specs])
        installed = missing
    return installed
