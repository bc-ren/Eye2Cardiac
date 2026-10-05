#!/usr/bin/env python3
"""Read-only, standard-library release checks; not a security certification.

Checks source syntax, local import candidates, and obvious non-public artifacts.
Dynamic imports are enumerated for human review, not claimed to be resolved.
No source module is executed, and no bytecode or report is written by default.
"""
from __future__ import annotations

import argparse
import ast
import ipaddress
import json
from pathlib import Path
import re
import sys
import tokenize


THIRD_PARTY = {
    "numpy", "pandas", "torch", "torchvision", "scipy", "sklearn", "numba",
    "PIL", "vtk", "yaml", "tqdm", "skimage", "h5py", "matplotlib", "seaborn",
    "pyarrow", "clip_modules", "timm", "einops", "safetensors", "pytest", "threadpoolctl",
}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".sh", ".ini", ".cfg", ".csv"}
DATA_SUFFIXES = {
    ".pt", ".pth", ".ckpt", ".pkl", ".pickle", ".npy", ".npz", ".parquet",
    ".h5", ".hdf5", ".nii", ".dcm", ".zip", ".tar", ".gz", ".onnx", ".safetensors",
}
PRIVATE_PATH = re.compile(r"/(?:data\d*/(?:rbc|home)(?:/|\b)|home/[^\s/]+/|Users/[^\s/]+/|Volumes/[^\s/]+/)")
IPV4 = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
KEY_PATTERNS = [
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"),
    re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16})\b"),
    re.compile(r"(?i)\b(?:password|passwd|api_key|access_token|secret_key)\s*[:=]\s*['\"][^'\"\n]{8,}['\"]"),
]


def scan(root: Path) -> dict:
    if not hasattr(sys, 'stdlib_module_names'):
        raise RuntimeError('Release checks require Python 3.10 or later (tested on Python 3.12).')
    root = root.resolve()
    findings = []
    files = []
    dynamic = []
    dependencies = set()
    ast_trees = {}
    syntax_count = 0

    def issue(path, rule, detail, line=None):
        entry = {"path": str(path.relative_to(root)), "rule": rule, "detail": detail}
        if line is not None:
            entry["line"] = line
        findings.append(entry)

    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if ".git" in relative.parts:
            continue
        if path.is_symlink():
            issue(path, "symlink", "Symlinks are not allowed in this source-only release.")
            continue
        if not path.is_file():
            continue
        files.append(path)
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo", ".nbc", ".nbi"}:
            issue(path, "generated_cache", "Generated cache must not be distributed.")
        if path.suffix.lower() in DATA_SUFFIXES:
            issue(path, "binary_or_data_artifact", "Restricted/source-external data, weights, or archives are not allowed.")
        if "SERVER_ONLY" in path.name or path.name in {".env", "id_rsa", "id_ed25519"}:
            issue(path, "private_artifact_name", "Private artifact must not be distributed.")
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {".gitignore", "LICENSE", "NOTICE", "Dockerfile", "Makefile"}:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            issue(path, "non_utf8_text", "Expected text file is not UTF-8.")
            continue
        for pattern, rule in [(PRIVATE_PATH, "private_absolute_path"), *[(p, "possible_secret") for p in KEY_PATTERNS]]:
            for match in pattern.finditer(content):
                # Deliberately never echo the potentially sensitive matching value.
                issue(path, rule, "Remove private deployment details before publishing.", content.count("\n", 0, match.start()) + 1)
        for match in IPV4.finditer(content):
            try:
                ipaddress.ip_address(match.group())
            except ValueError:
                continue
            issue(path, "ip_literal", "IP literal requires removal or explicit review.", content.count("\n", 0, match.start()) + 1)
        if path.suffix == ".py":
            try:
                with tokenize.open(path) as stream:
                    source = stream.read()
                tree = ast.parse(source, filename=str(relative))
                compile(tree, str(relative), "exec")
                syntax_count += 1
                ast_trees[path] = tree
            except (SyntaxError, ValueError, UnicodeError) as error:
                issue(path, "python_syntax", str(error))

    local_stems = {p.stem for p in ast_trees}
    stdlib = set(getattr(sys, "stdlib_module_names", ())) | {"__future__"}
    for path, tree in ast_trees.items():
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names = [node.module.split(".")[0]]
            for name in names:
                if name in stdlib or name in local_stems:
                    continue
                dependencies.add(name)
                if name not in THIRD_PARTY:
                    issue(path, "unresolved_import", f"Unknown import root {name!r}; supply local source or declare dependency.", node.lineno)
            if isinstance(node, ast.Call):
                fn = node.func
                name = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else ""
                if name in {"spec_from_file_location", "import_module", "load_module", "module"}:
                    dynamic.append({"path": str(path.relative_to(root)), "line": node.lineno, "function": name})

    return {
        "status": "PASS" if not findings else "FAIL",
        "files": len(files), "bytes": sum(p.stat().st_size for p in files),
        "python_syntax_checked": syntax_count,
        "third_party_import_roots": sorted(dependencies),
        "findings": findings, "dynamic_import_review_sites": dynamic,
        "limitations": [
            "AST import candidates do not prove runtime import resolution or dependency compatibility.",
            "Dynamic imports and configuration paths require manual review and runtime tests.",
            "Pattern scanning cannot establish absence of every identifier or secret; manually review the allowlisted files.",
            "No weights, patient records, model forward pass, or reproduction metrics were evaluated by this scanner.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    result = scan(args.root)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
