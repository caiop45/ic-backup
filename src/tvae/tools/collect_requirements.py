#!/usr/bin/env python3
"""Collect external imports from a Python entrypoint via static analysis."""

from __future__ import annotations

import argparse
import ast
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from importlib.metadata import PackageNotFoundError, packages_distributions, version
except Exception:  # pragma: no cover
    PackageNotFoundError = Exception  # type: ignore[misc,assignment]
    packages_distributions = None  # type: ignore[assignment]
    version = None  # type: ignore[assignment]

STDLIB = getattr(sys, "stdlib_module_names", set())


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


def _top_level(mod: str) -> str:
    return mod.split(".", 1)[0]


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


def _module_to_distribution(module_top: str) -> tuple[str | None, str | None]:
    if packages_distributions is None or version is None:
        return (None, None)
    mapping = packages_distributions()
    dists = mapping.get(module_top) or []
    if not dists:
        return (None, None)
    dist = sorted(dists)[0]
    try:
        return (dist, version(dist))
    except PackageNotFoundError:
        return (dist, None)


def collect_external_modules(entry: Path, *, project_roots: list[Path]) -> list[str]:
    entry = entry.resolve()
    visited_files: set[Path] = set()
    external: set[str] = set()

    queue: deque[Path] = deque([entry])
    while queue:
        cur = queue.popleft()
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
            top = _top_level(mod)

            if top in STDLIB:
                continue

            local_path = _resolve_local_module(mod, roots=project_roots)
            if local_path is not None:
                queue.append(local_path)
                continue

            external.add(top)

    return sorted(external)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("entry", type=Path, help="Python entrypoint (e.g. train_tht_tripgen.py)")
    ap.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root for resolving local imports (default: entry dir)",
    )
    ap.add_argument(
        "--write",
        type=Path,
        default=None,
        help="If set, write requirements list to this file",
    )
    ap.add_argument(
        "--with-versions",
        action="store_true",
        help="Try to include versions from installed packages",
    )
    args = ap.parse_args()

    entry = args.entry
    if not entry.exists():
        print(f"Error: entry not found: {entry}", file=sys.stderr)
        return 2

    root = (args.project_root or entry.parent).resolve()
    mods = collect_external_modules(entry, project_roots=[root])

    lines: list[str] = []
    for m in mods:
        if args.with_versions:
            dist, ver = _module_to_distribution(m)
            if dist and ver:
                lines.append(f"{dist}=={ver}")
            elif dist:
                lines.append(dist)
            else:
                lines.append(m)
        else:
            lines.append(m)

    text = "\n".join(lines) + ("\n" if lines else "")
    if args.write is not None:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(text, encoding="utf-8")
        print(f"Saved: {args.write} ({len(lines)} entries)")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
