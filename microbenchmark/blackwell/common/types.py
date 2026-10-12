"""Small shared benchmark records; no CUDA or family dependencies."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Variant:
    id: str
    case: str
    family: str
    parameters: dict

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Launch:
    blocks: int = 1
    repetitions: int = 1024
    count: int = 1024
    seed: int = 0
    memory_limit: int = 512 * 1024 * 1024
