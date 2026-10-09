"""Only the 30 pinned TMA variants; other families belong in sibling packages."""
from dataclasses import asdict, dataclass
import json


@dataclass(frozen=True)
class Variant:
    id: str
    case: str
    family: str
    parameters: dict

    def to_dict(self):
        return asdict(self)



def variants():
    rows = []

    def add(case, family, suffix="", **parameters):
        rows.append(Variant(case + ("--" + suffix if suffix else ""), case, family, parameters))

    add("tma_completion_latency", "tma", transfer_bytes=16, shape=[], element_bytes=4, fine=True)
    for size in (1024, 2048, 4096, 8192, 12288, 16384):
        add("tma_bulk_throughput", "tma", str(size), transfer_bytes=size, shape=[], element_bytes=4, fine=False)
    shapes = {1: [(64,), (96,), (128,), (160,), (192,), (256,)],
              2: [(16,16), (32,16), (32,32), (64,32), (96,32), (64,64)],
              3: [(8,8,4), (8,8,8), (16,8,8), (16,16,8), (16,16,12), (16,16,16)],
              4: [(16,16,16), (64,64,1), (256,16,1), (16,256,1), (4,4,256)]}
    for dimensions, choices in shapes.items():
        case = f"tma_tensor_{dimensions}d_throughput" if dimensions != 4 else "tma_equal_bytes_shape_sweep"
        for shape in choices:
            elem = 8 if dimensions == 1 else 4
            from math import prod
            add(case, "tma", "x".join(map(str, shape)), shape=list(shape),
                transfer_bytes=prod(shape)*elem, element_bytes=elem, fine=False)
    return rows


if __name__ == "__main__":
    print(json.dumps([v.to_dict() for v in variants()], indent=2))
