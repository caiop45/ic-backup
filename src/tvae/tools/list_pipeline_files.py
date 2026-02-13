#!/usr/bin/env python3
"""List local Python files reachable from an entrypoint via static imports."""

from __future__ import annotations

import argparse
import ast
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ImportRef:
    module: str
    level: int


def _iter_imports(source: str) -> Iterable[ImportRef]:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name:
                    yield ImportRef(module=alias.name, level=0)
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            yield ImportRef(module=node.module, level=int(node.level or 0))


def _resolve_local_module(module: str, *, roots: list[Path]) -> Path | None:
    parts = module.split(".")
    for root in roots:
        p1 = root.joinpath(*parts).with_suffix(".py")
        if p1.exists():
            return p1
        p2 = root.joinpath(*parts, "__init__.py")
        if p2.exists():
            return p2
    return None


def _resolve_relative(module: str, *, current_file: Path, level: int) -> str:
    _ = current_file
    _ = level
    return module


def collect_local_files(entry: Path, *, project_roots: list[Path]) -> list[Path]:
    entry = entry.resolve()
    visited_files: set[Path] = set()
    queue: deque[Path] = deque([entry])

    while queue:
        cur = queue.popleft().resolve()
        if cur in visited_files:
            continue
        visited_files.add(cur)

        try:
            src = cur.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            src = cur.read_text(encoding="latin-1")

        for imp in _iter_imports(src):
            mod = imp.module
            if imp.level > 0:
                mod = _resolve_relative(mod, current_file=cur, level=imp.level)
            local_path = _resolve_local_module(mod, roots=project_roots)
            if local_path is not None:
                queue.append(local_path)

    return sorted(visited_files)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "List local .py files used by a Python entrypoint via static import scan."
        )
    )
    ap.add_argument("entry", type=Path, help="Entrypoint file (e.g. train_tht_tripgen.py)")
    ap.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root for resolving imports. Default: entry dir.",
    )
    ap.add_argument(
        "--relative-to",
        type=Path,
        default=None,
        help="Base dir for relative paths. Default: project-root.",
    )

    args = ap.parse_args()
    entry = args.entry
    if not entry.exists():
        print(f"Error: entry not found: {entry}")
        return 2

    root = (args.project_root or entry.parent).resolve()
    files = collect_local_files(entry, project_roots=[root])
    base = (args.relative_to or root).resolve()

    for f in files:
        try:
            rel = f.relative_to(base)
            print(rel)
        except ValueError:
            print(f)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
