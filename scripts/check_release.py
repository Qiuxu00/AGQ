"""Check the source distribution and local documentation links; no network use."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from urllib.parse import unquote

IGNORED_PARTS = {".git", "__pycache__", ".pytest_cache", ".venv", "venv", "outputs", "local", "build", "dist"}
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".pkl", ".pickle", ".ckpt", ".safetensors", ".onnx", ".gguf", ".bin", ".npy", ".npz", ".h5", ".hdf5", ".pem", ".key", ".exe", ".dll", ".so", ".pyc", ".zip"}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".toml", ".yml", ".yaml", ".cff", ".cpp", ".cu", ".svg"}


def source_files(root):
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in IGNORED_PARTS or part.endswith(".egg-info") for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"symbolic link cannot enter source package: {relative.as_posix()}")
        if path.is_file():
            yield path


def check(root):
    root = Path(root).resolve()
    errors, total_bytes, count = [], 0, 0
    patterns = [
        re.compile(r"[A-Za-z]:[\\/](?:Development|Users|EdgeDownloads)"),
        re.compile(r"(?:autodl\.pro|seetacloud\.com)"),
        re.compile(r"(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}"),
        re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    ]
    for path in source_files(root):
        name = path.relative_to(root).as_posix()
        count += 1
        total_bytes += path.stat().st_size
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append({"file": name, "reason": "weight/binary/credential artifact prohibited"})
        if path.stat().st_size > 10_000_000:
            errors.append({"file": name, "reason": "unexpected source artifact above 10 MB"})
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8-sig")
        if any(pattern.search(text) for pattern in patterns):
            errors.append({"file": name, "reason": "private infrastructure or credential pattern"})
        if path.suffix == ".md":
            without_code = re.sub(r"```.*?```", "", text, flags=re.S)
            targets = re.findall(r"\[[^\]\n]+\]\(([^\s)]+)\)", without_code)
            targets += re.findall(r'<img\b[^>]*\bsrc="([^"]+)"', without_code)
            for target in targets:
                if re.match(r"(?:https?://|mailto:|#)", target):
                    continue
                local = unquote(target.split("#", 1)[0])
                resolved = (path.parent / local).resolve()
                if not resolved.is_relative_to(root) or not resolved.exists():
                    errors.append({"file": name, "reason": "missing or escaping local link", "link": target})
    return {"passed": not errors, "source_files": count, "source_bytes": total_bytes, "model_weights_included": False if not errors else None, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    result = check(args.root)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
