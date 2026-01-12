#!/usr/bin/env python3
"""Coleta dependências (imports) a partir de um entrypoint Python.

Objetivo: ajudar a reconstruir um `requirements.txt` quando o ambiente foi perdido,
extraindo as bibliotecas usadas por um script e pelos módulos locais importados
no fluxo (fechamento transitivo via análise estática de `import`).

Limitações:
- Imports dinâmicos (ex: import dentro de função/try, importlib, plugins) podem não aparecer.
- O mapeamento "módulo -> pacote pip" depende do que estiver instalado no Python em execução.
"""

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
    level: int  # 0 = absoluto, >0 = relativo


def _iter_imports(source: str) -> Iterable[ImportRef]:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name:
                    yield ImportRef(module=alias.name, level=0)
        elif isinstance(node, ast.ImportFrom):
            # from .foo import bar -> module="foo", level=1
            if node.module is None:
                continue
            yield ImportRef(module=node.module, level=int(node.level or 0))


def _top_level(mod: str) -> str:
    return mod.split(".", 1)[0]


def _resolve_local_module(module: str, *, roots: list[Path]) -> Path | None:
    parts = module.split(".")
    for root in roots:
        # module.py
        p1 = root.joinpath(*parts).with_suffix(".py")
        if p1.exists():
            return p1
        # package/__init__.py
        p2 = root.joinpath(*parts, "__init__.py")
        if p2.exists():
            return p2
    return None


def _resolve_relative(module: str, *, current_file: Path, level: int) -> str:
    # Em scripts soltos, imports relativos são raros; aqui tratamos de forma prática:
    # sobe `level` diretórios a partir do arquivo atual e tenta reconstruir o módulo.
    base = current_file.parent
    for _ in range(max(0, level - 1)):
        base = base.parent
    # Se o import for "from .a.b import x", o módulo "a.b" ainda é relativo a `base`.
    # Retornamos apenas o nome do módulo (para tentar resolver por path).
    return module


def _module_to_distribution(module_top: str) -> tuple[str | None, str | None]:
    """Tenta mapear um módulo (top-level) para (dist_name, dist_version)."""
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
    """Retorna módulos top-level externos usados pelo entrypoint e seus imports locais."""
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

            # stdlib: ignora
            if top in STDLIB:
                continue

            # tenta resolver como módulo local (dentro do repo)
            local_path = _resolve_local_module(mod, roots=project_roots)
            if local_path is not None:
                queue.append(local_path)
                continue

            # se não achou localmente, considera externo
            external.add(top)

    return sorted(external)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("entry", type=Path, help="Arquivo Python de entrada (ex: train_vae_hlp_testes.py)")
    ap.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Raiz do projeto para resolver imports locais (default: diretório do entry)",
    )
    ap.add_argument(
        "--write",
        type=Path,
        default=None,
        help="Se informado, grava a lista de requirements (um por linha) nesse arquivo",
    )
    ap.add_argument(
        "--with-versions",
        action="store_true",
        help="Tenta incluir versões (requer que as libs estejam instaladas nesse Python).",
    )
    args = ap.parse_args()

    entry = args.entry
    if not entry.exists():
        print(f"Erro: arquivo não encontrado: {entry}", file=sys.stderr)
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
        print(f"Salvo: {args.write} ({len(lines)} entradas)")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

