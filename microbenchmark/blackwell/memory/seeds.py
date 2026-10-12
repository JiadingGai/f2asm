"""Read the pinned compiler seed; execution uses offline F2Asm-assembled SASS."""
from pathlib import Path

from .catalog import variants


def generate(variant):
    if variant not in variants():
        raise ValueError("Unknown ordinary-load variant")
    return (Path(__file__).resolve().parent / "seeds" / (variant.id + ".ptx")).read_text()
