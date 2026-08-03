from enum import Enum
from pathlib import Path
from pydantic import BaseModel, Field, field_validator

class Architecture(str, Enum):
    transformer = "transformer"
    looped_transformer = "looped_transformer"
    mamba = "mamba"
class RunMode(str, Enum): train = "train"; finetune = "finetune"
class Adapter(str, Enum): full = "full"; lora = "lora"; qlora = "qlora"
class Device(str, Enum): auto = "auto"; cuda = "cuda"; cpu = "cpu"

class RunConfig(BaseModel):
    model_id: str = "Qwen/Qwen2.5-0.5B"
    architecture: Architecture = Architecture.transformer
    mode: RunMode = RunMode.finetune
    adapter: Adapter = Adapter.lora
    device: Device = Device.auto
    dataset_id: str = ""
    dataset_split: str = "train"
    text_column: str = "text"
    output_dir: Path = Path("outputs/zigllm")
    seq_len: int = Field(1024, ge=128, le=32768)
    batch_size: int = Field(1, ge=1, le=1024)
    grad_accum: int = Field(8, ge=1, le=4096)
    epochs: float = Field(1.0, gt=0, le=100)
    learning_rate: float = Field(2e-4, gt=0, le=1)
    lora_rank: int = Field(16, ge=1, le=256)
    lora_alpha: int = Field(32, ge=1, le=1024)
    lora_dropout: float = Field(.05, ge=0, lt=1)
    max_samples: int = Field(0, ge=0)
    @field_validator("dataset_id")
    @classmethod
    def dataset_required(cls, v, info):
        return v.strip()
    def tokens_per_step(self): return self.seq_len * self.batch_size * self.grad_accum
