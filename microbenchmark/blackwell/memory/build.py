"""Compile the pinned shell and encode every instruction through F2Asm, offline."""
from pathlib import Path
from types import SimpleNamespace

from ..common import build as shared
from .catalog import variants
from .protocols import inspect
from .seeds import generate


def suite():
    return SimpleNamespace(directory=Path(__file__).resolve().parent,
                           variants=variants, inspect=inspect, generate=generate)


def main():
    shared.main(suite())


if __name__ == "__main__":
    main()
