"""Benchmark registry and notebook-friendly launch helpers.

Some evaluations are generative task scoring (HellaSwag/GSM8K), while SWE-bench,
CyberGym and HLE require specialized harnesses and/or agent traces. This module
keeps those distinctions explicit instead of reporting misleading scores.
"""
from dataclasses import dataclass
from typing import Optional
import subprocess

@dataclass(frozen=True)
class Benchmark:
    key: str
    label: str
    kind: str
    description: str
    task: Optional[str] = None

BENCHMARKS = {
    "swe": Benchmark("swe", "SWE-bench", "agent", "Software-engineering issue resolution; requires generated patches and the SWE-bench harness."),
    "gmsk8": Benchmark("gmsk8", "GMSK8 / GSM8K", "lm_eval", "Grade-school math with exact-match answer extraction.", "gsm8k"),
    "hle": Benchmark("hle", "HLE", "agent", "Humanity's Last Exam; requires the HLE dataset, licensed assets, and a compatible evaluator."),
    "cybergym": Benchmark("cybergym", "CyberGym", "agent", "Cybersecurity benchmark; use its sandbox and safety-approved harness."),
    "hellaswag": Benchmark("hellaswag", "HellaSwag", "lm_eval", "Commonsense sentence completion.", "hellaswag"),
}

def list_benchmarks():
    return list(BENCHMARKS.values())

def run_benchmark(name: str, model_id: str, limit: int = 0, device: str = "auto") -> str:
    """Run a supported lm-eval task, or return the exact specialized next step.

    The agent-style benchmarks are intentionally not silently reduced to text
    generation: they need patches, tool traces, or a sandbox.
    """
    if name not in BENCHMARKS: raise ValueError(f"Unknown benchmark: {name}")
    b = BENCHMARKS[name]
    if b.kind == "lm_eval":
        try:
            import lm_eval  # noqa: F401
        except ImportError:
            return f"{b.label} requires lm-eval. Install with: pip install lm-eval\nThen run: {command_for(name, model_id, limit, device)}"
        cmd = command_for(name, model_id, limit, device)
        result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
        return result.stdout + ("\n" + result.stderr if result.stderr else "")
    return (f"{b.label} selected. This is a {b.kind} benchmark and needs its official harness; "
            f"the model must first produce the required outputs/traces.\n\n"
            f"Integration command:\n{command_for(name, model_id, limit, device)}")

def command_for(name: str, model_id: str, limit: int = 0, device: str = "auto") -> str:
    b = BENCHMARKS[name]
    if b.kind == "lm_eval":
        lim = f" --limit {int(limit)}" if limit else ""
        dev = "cuda" if device == "auto" else device
        return f"lm_eval --model hf --model_args pretrained={model_id} --tasks {b.task} --device {dev} --batch_size auto{lim}"
    if name == "swe": return "pip install swebench && python -m swebench.harness.run_evaluation --help"
    if name == "hle": return "pip install datasets && use the official HLE evaluator with your licensed dataset/config"
    return "Use the official CyberGym sandbox/evaluator and export its scored result"
