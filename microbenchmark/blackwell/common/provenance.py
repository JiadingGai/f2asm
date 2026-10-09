"""Read-only import drift check. Never refresh hashes or overwrite an imported file."""
import argparse
import json
from pathlib import Path
import subprocess

from .artifacts import sha256


def inspect(manifest, root, source_repo=None):
    root = Path(root).resolve()
    destination_changes = []
    for name, record in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Import manifest path escapes the benchmark directory")
        if not path.is_file() or sha256(path.read_bytes()) != record["sha256"]:
            destination_changes.append(name)
    source_changes = []
    if source_repo is not None:
        source_repo = Path(source_repo).resolve()
        revision = manifest["source_commit"]
        for name, expected in manifest["source_files"].items():
            path = (source_repo / name).resolve()
            if not path.is_relative_to(source_repo):
                raise ValueError("Import manifest path escapes the source repository")
            baseline = subprocess.check_output(
                ["git", "show", f"{revision}:{name}"], cwd=source_repo)
            if sha256(baseline) != expected:
                raise ValueError(f"Wrong recorded source baseline: {name}")
            if not path.is_file() or sha256(path.read_bytes()) != expected:
                source_changes.append(name)
    return {"source_commit": manifest["source_commit"],
            "destination_changes": destination_changes, "source_changes": source_changes,
            "source_checked": source_repo is not None,
            "clean": not destination_changes and not source_changes,
            "action": "Review drift before importing; never overwrite shared helpers automatically"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--source-repo", type=Path)
    args = parser.parse_args()
    result = inspect(json.loads(args.manifest.read_text()),
                     Path(__file__).resolve().parents[1], args.source_repo)
    print(json.dumps(result, indent=2))
    if not result["clean"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
