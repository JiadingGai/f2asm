"""L1-enabled ordinary-load reference for the Blackwell Figure 2-style curve."""
import json

from ..common.types import Variant


def variants():
    return [Variant("memory_l1_pointer_chase", "memory_l1_pointer_chase", "pointer",
                    {"level": "l1", "fine": False, "stride": 32})]


if __name__ == "__main__":
    print(json.dumps([v.to_dict() for v in variants()], indent=2))
