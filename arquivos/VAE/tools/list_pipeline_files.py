#!/usr/bin/env python3
"""Lista todos os arquivos Python locais usados pela pipeline de um entrypoint.

Uso típico (a partir da raiz do projeto VAE):

    python -m tools.list_pipeline_files train_vae_hlp.py

Ou, chamando diretamente o script:

    python tools/list_pipeline_files.py train_vae_hlp.py

Ele faz uma análise estática de imports (sem executar o treino) e
imprime no terminal todos os arquivos .py do projeto que são
alcançados direta ou indiretamente a partir do entrypoint.
"""

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


def _resolve_local_module(module: str, *, roots: list[Path]) -> Path | None:
    """Tenta resolver um nome de módulo para um arquivo .py dentro do projeto."""
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
    """Ajusta imports relativos para um nome de módulo 'equivalente'."""
    # Para este projeto, basta devolver o módulo sem mexer no caminho,
    # pois a raiz do projeto já é usada como base em _resolve_local_module.
    # Mantemos a assinatura por compatibilidade/clareza.
    _ = current_file  # apenas para deixar explícito que existe
    _ = level
    return module


def collect_local_files(entry: Path, *, project_roots: list[Path]) -> list[Path]:
    """Retorna a lista de arquivos .py locais alcançados pelo entrypoint."""
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
            "Lista todos os arquivos .py locais usados pela pipeline de um "
            "arquivo de entrada (análise estática de imports)."
        )
    )
    ap.add_argument(
        "entry",
        type=Path,
        help="Arquivo Python de entrada (ex: train_vae_hlp.py ou caminhos relativos/absolutos).",
    )
    ap.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help=(
            "Raiz do projeto para resolver imports locais. "
            "Padrão: diretório do entry."
        ),
    )
    ap.add_argument(
        "--relative-to",
        type=Path,
        default=None,
        help=(
            "Diretório base para imprimir caminhos relativos. "
            "Padrão: project-root."
        ),
    )

    args = ap.parse_args()

    entry = args.entry
    if not entry.exists():
        print(f"Erro: arquivo de entrada não encontrado: {entry}")
        return 2

    root = (args.project_root or entry.parent).resolve()
    files = collect_local_files(entry, project_roots=[root])

    base = (args.relative_to or root).resolve()

    for f in files:
        try:
            rel = f.relative_to(base)
            print(rel)
        except ValueError:
            # Caso o arquivo não esteja abaixo de base, imprime caminho absoluto
            print(f)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


