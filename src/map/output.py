from dataclasses import dataclass
from enum import Enum

class DecodeMode(Enum):
    ARGMAX = "argmax"
    TOPK = "topk"

@dataclass(frozen=True)
class OutputContract:
    domain_id: str
    selector: str                 # where to read from, e.g. "z="
    label_count: int              # number of output classes
    neurons_per_label: int = 4    # redundancy
    decode_mode: DecodeMode = DecodeMode.ARGMAX