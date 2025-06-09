#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gera um diagrama de fluxo de chamadas para a função `main` de um script Python.

Este script analisa um arquivo-alvo, extrai a sequência de chamadas de função
dentro de `main`, e usa o Graphviz para gerar um diagrama visual.

NOVO: Ele captura comentários marcados com um símbolo especial (ex: #@).
A busca pelo comentário é feita para cima a partir da chamada da função,
associando o primeiro que encontrar.

REQUISITO: Graphviz (comando 'dot') deve estar instalado no sistema.
"""

import ast
import os
import shutil
import subprocess
import collections
import re
import tokenize
from typing import List, Dict, Tuple

# --- CONFIGURAÇÃO ---
COMMENT_MARKER = "#@"  # Define o símbolo para os comentários que devem ser capturados
PATH_TO_MAIN_RASCUNHO = "main_rascunho.py"
OUTPUT_FILENAME = "/home/caioloss/arquivos/d_linear_modulos/save_data/fluxo_de_chamadas_com_anotacoes.png"
ALLOWED_MODULES = (
    "data_processing",
    "synthetic_data",
    "models",
    "evaluation",
    "utils",
)

def parse_main_function_calls(file_path: str) -> List[Tuple[str, str]]:
    """
    Analisa o arquivo Python e retorna uma lista de tuplas contendo
    (nome_da_chamada, comentario_associado).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo não encontrado: '{file_path}'. Verifique o caminho.")

    # 1. Usar tokenize para extrair comentários que começam com o marcador especial
    comments_map = {}
    with tokenize.open(file_path) as f:
        tokens = tokenize.generate_tokens(f.readline)
        for token in tokens:
            if token.type == tokenize.COMMENT:
                if token.string.startswith(COMMENT_MARKER):
                    line_num = token.start[0]
                    comment_text = token.string[len(COMMENT_MARKER):].strip()
                    comments_map[line_num] = comment_text

    with open(file_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    # 2. Mapear todos os imports
    imports_map: Dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name
                imports_map[name] = alias.name
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module or "."
            for alias in node.names:
                func_name = alias.asname or alias.name
                imports_map[func_name] = f"{module_name}.{alias.name}"

    main_func_node = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"), None
    )
    if main_func_node is None:
        raise RuntimeError("Função 'main' não encontrada no script.")

    # 3. Extrair todas as chamadas da função main
    class CallVisitor(ast.NodeVisitor):
        def __init__(self):
            self.calls: List[Tuple[int, str]] = []

        def visit_Call(self, node: ast.Call):
            call_name = ast.unparse(node.func)
            self.calls.append((node.lineno, call_name))
            self.generic_visit(node)

    visitor = CallVisitor()
    visitor.visit(main_func_node)

    # 4. Montar a lista final, procurando o comentário mais próximo para cima
    processed_calls = []
    sorted_calls = sorted(visitor.calls, key=lambda x: x[0])
    
    # Copiamos o mapa para poder remover comentários já utilizados
    available_comments = comments_map.copy()

    for lineno, call_name in sorted_calls:
        # Resolve o nome completo da função
        base_name = call_name.split('.')[0]
        full_name = call_name.replace(base_name, imports_map.get(base_name, base_name), 1)
        
        comment = ""
        # Inicia uma busca iterativa para cima, a partir da linha anterior à chamada
        for search_line in range(lineno - 1, 0, -1):
            if search_line in available_comments:
                # Se um comentário for encontrado, o associamos a esta chamada
                comment = available_comments[search_line]
                
                # Removemos o comentário do mapa para que não seja associado a outra função
                del available_comments[search_line]
                
                # Interrompe a busca para esta chamada, pois já encontramos o mais próximo
                break
                
        processed_calls.append((full_name, comment))
            
    return processed_calls

def build_call_graph(calls_with_comments: List[Tuple[str, str]], dot_path: str):
    """
    Constrói um arquivo .dot, usando labels HTML para incluir os comentários.
    """
    calls_by_module = collections.defaultdict(list)
    for i, (call, comment) in enumerate(calls_with_comments):
        parts = call.split('.')
        module = '.'.join(parts[:-1]) if len(parts) > 1 else "builtin_or_variable"
        calls_by_module[module].append((f"step{i+1}", call, comment))
    
    dot_lines = [
        "digraph CallFlow {",
        "    graph [rankdir=TB, splines=ortho, nodesep=1, ranksep=1.5, fontname=\"Helvetica\"];",
        "    node [shape=none, margin=0, fontname=\"Helvetica\"];",
        "    edge [color=\"#4C5B7F\"];",
        "",
        "    Start [shape=circle, style=filled, fillcolor=\"#A8E6CF\", width=0.6];",
        "    End [shape=doublecircle, style=filled, fillcolor=\"#FFD8A8\", width=0.5];",
        ""
    ]

    for module, nodes in calls_by_module.items():
        sanitized_module_id = re.sub(r'[^a-zA-Z0-9.]', '_', module).replace('.', '_')
        dot_lines.append(f"    subgraph cluster_{sanitized_module_id} {{")
        dot_lines.append(f"        label = \"Módulo: {module}\";")
        dot_lines.append("        style = \"rounded\"; color=\"#AAAAAA\";")
        
        for node_id, full_call_path, comment in nodes:
            function_name = full_call_path.split('.')[-1]
            comment_safe = comment.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

            label_html = '<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" STYLE="ROUNDED" BGCOLOR="#E6F4FF">'
            label_html += f'<TR><TD ALIGN="CENTER"><B>{function_name}</B></TD></TR>'
            if comment:
                label_html += f'<TR><TD ALIGN="LEFT" BALIGN="LEFT"><FONT POINT-SIZE="10" COLOR="#333333">{comment_safe}</FONT></TD></TR>'
            label_html += '</TABLE>>'
            
            dot_lines.append(f'        {node_id} [label={label_html}];')
        
        dot_lines.append("    }")
        dot_lines.append("")

    all_nodes = ["Start"] + [f"step{i+1}" for i in range(len(calls_with_comments))] + ["End"]
    for i in range(len(all_nodes) - 1):
        dot_lines.append(f"    {all_nodes[i]} -> {all_nodes[i+1]};")

    dot_lines.append("}")

    with open(dot_path, "w", encoding="utf-8") as f:
        f.write("\n".join(dot_lines))

def render_diagram(dot_path: str, output_path: str):
    """
    Renderiza o diagrama .dot para uma imagem PNG usando o Graphviz.
    """
    if not shutil.which("dot"):
        raise SystemExit("ERRO: O comando 'dot' do Graphviz não foi encontrado.")
    try:
        subprocess.run(["dot", "-Tpng", dot_path, "-o", output_path], check=True, capture_output=True)
        print(f"✅ Diagrama salvo com sucesso em: {output_path}")
    except subprocess.CalledProcessError as e:
        print("❌ ERRO ao renderizar o diagrama com o Graphviz.")
        print(e.stderr.decode().strip())

def main():
    """
    Função principal que orquestra a análise e a geração do diagrama.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_file_path = os.path.join(script_dir, PATH_TO_MAIN_RASCUNHO)
    
    print(f"Analisando o arquivo: '{target_file_path}'...")
    try:
        all_calls = parse_main_function_calls(target_file_path)
        print(f"Encontradas {len(all_calls)} chamadas de função no total.")

        filtered_calls = [
            item for item in all_calls if item[0].startswith(ALLOWED_MODULES)
        ]
        
        if not filtered_calls:
            print(f"Nenhuma chamada dos módulos permitidos foi encontrada. Diagrama não gerado.")
            return

        print(f"Filtrando para {len(filtered_calls)} chamadas dos módulos de interesse. Construindo o grafo...")
        dot_file = "call_flow.dot"
        build_call_graph(filtered_calls, dot_file)
        
        print("Renderizando o diagrama...")
        render_diagram(dot_file, OUTPUT_FILENAME)
        
        os.remove(dot_file)

    except (FileNotFoundError, RuntimeError) as e:
        print(f"ERRO: {e}")

if __name__ == "__main__":
    main()