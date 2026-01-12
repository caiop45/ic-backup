"""
Gera comparativos entre validação e sintético a partir dos CSVs agregados por
OD × hora × dia-da-semana e salva SOMENTE CSVs derivados (sem gerar figuras).

Entradas (nesta pasta):
- od_hora_dia_validacao.csv
- od_hora_dia_sintetico.csv

Saídas (CSVs nesta pasta):
- ./od_top{N}_participacao_validacao_vs_sintetico.csv
- ./od_top{N}_distrib_hora_validacao_vs_sintetico.csv

Adicional (variabilidade via pseudo-dias — não altera o pipeline de geração):
- variab_hora_intra_validacao.csv
- variab_hora_intra_sintetico.csv
- variab_hora_cross_val_vs_synth.csv
- variab_od_intra_validacao.csv
- variab_od_intra_sintetico.csv
- variab_od_cross_val_vs_synth.csv
- resumo_variab_hora.csv
- resumo_variab_od.csv

Observação: a “variabilidade” é estimada via pseudo-dias amostrados a partir
dos agregados (OD×hora×dia_da_semana), preservando a sazonalidade de weekday.
Não requer alterações no gerador; é uma aproximação, útil para diagnóstico
rápido quando não há cortes diários explícitos.
"""

from __future__ import annotations

from pathlib import Path
import os
import sys
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
VAE_DIR = SCRIPT_DIR.parent
if str(VAE_DIR) not in sys.path:
    sys.path.insert(0, str(VAE_DIR))

# Número de ODs do topo a considerar nos CSVs
TOP_N = 100

# Parâmetros do bloco de variabilidade (pseudo-dias)
VARIAB_ENABLE = True  # habilita a geração dos CSVs de variabilidade
VARIAB_D = 10         # número de pseudo-dias por conjunto
VARIAB_SEED = 123     # semente para reprodutibilidade

# Ajuste opcional: amostrar subset sintético para casar com total de validação (10 dias)
MATCH_SYNTH_TO_VAL = True
MATCH_MARGIN = 0.01  # 1%


def _load_inputs():
    val_path = SCRIPT_DIR / "od_hora_dia_validacao.csv"
    syn_path = SCRIPT_DIR / "od_hora_dia_sintetico.csv"
    if not val_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {val_path}")
    if not syn_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {syn_path}")
    val = pd.read_csv(val_path)
    syn = pd.read_csv(syn_path)
    # Garantias mínimas de schema
    req_cols = {"od", "hora_do_dia", "dia_da_semana", "n_viagens"}
    if not req_cols.issubset(val.columns):
        raise ValueError(f"CSV de validação não possui colunas esperadas: {req_cols}")
    if not req_cols.issubset(syn.columns):
        raise ValueError(f"CSV sintético não possui colunas esperadas: {req_cols}")
    return val, syn


def _ensure_dirs(top_n: int = TOP_N):
    # Apenas garante que a pasta de saída (save_data) exista
    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)


def plot_participacao_topN(val: pd.DataFrame, syn: pd.DataFrame, top_n: int = TOP_N) -> list[str]:
    # Totais por OD (para ordenar) e totais globais (denominador em todo o CSV)
    # Diagnósticos básicos de cardinalidade
    val_unique = int(val["od"].nunique())
    syn_unique = int(syn["od"].nunique())
    print(f"[INFO] ODs distintas | validação: {val_unique} | sintético: {syn_unique}")

    val_tot = val.groupby("od")["n_viagens"].sum().sort_values(ascending=False)
    top_ods = val_tot.head(int(top_n)).index.tolist()
    print(f"[INFO] Top solicitadas: {int(top_n)} | Top usadas: {len(top_ods)}")

    if len(top_ods) == 0:
        print(f"[INFO] Nenhuma OD encontrada em validação para top{top_n}.")
        return []

    syn_tot = syn.groupby("od")["n_viagens"].sum()

    # Denominadores: total de viagens do CSV completo (sem filtrar pelas top-N)
    val_total_all = float(val["n_viagens"].sum())
    syn_total_all = float(syn["n_viagens"].sum())
    print(f"[INFO] Totais | validação: {int(val_total_all)} | sintético: {int(syn_total_all)}")

    # Participações relativas ao total global de cada conjunto
    val_den = (val_tot.reindex(top_ods) / max(val_total_all, 1)).fillna(0.0)
    syn_den = (syn_tot.reindex(top_ods) / max(syn_total_all, 1)).fillna(0.0)

    # Soma das Top-N em % do total
    val_topn = float(val_den.sum())
    syn_topn = float(syn_den.sum())
    print(f"[INFO] Top{top_n} participação | validação: {val_topn:.2%} | sintético: {syn_topn:.2%}")

    # CSV com os valores de participação Top-N (salvo em save_data)
    # Similaridade por hora (produto escalar) para cada OD do topo
    hours = np.arange(24)
    val_hour = (
        val[val["od"].isin(top_ods)]
        .groupby(["od", "hora_do_dia"])  # soma por hora
        ["n_viagens"].sum()
        .unstack(fill_value=0)
        .reindex(columns=hours, fill_value=0)
    )
    syn_hour = (
        syn[syn["od"].isin(top_ods)]
        .groupby(["od", "hora_do_dia"])  # soma por hora
        ["n_viagens"].sum()
        .unstack(fill_value=0)
        .reindex(columns=hours, fill_value=0)
    )

    def _norm(v: np.ndarray) -> np.ndarray:
        s = float(v.sum())
        if s <= 0:
            return np.zeros_like(v, dtype=np.float64)
        return v.astype(np.float64) / s

    sim_by_od: list[float] = []
    for od in top_ods:
        v = val_hour.loc[od].to_numpy(dtype=np.int64) if od in val_hour.index else np.zeros(24, dtype=np.int64)
        s = syn_hour.loc[od].to_numpy(dtype=np.int64) if od in syn_hour.index else np.zeros(24, dtype=np.int64)
        v_n = _norm(v)
        s_n = _norm(s)
        sim = float(np.dot(v_n, s_n))
        sim_by_od.append(sim)

    df_csv = pd.DataFrame(
        {
            "od": top_ods,
            # Coluna adicional solicitada (OD = origem-destino, igual a 'od')
            "OD": top_ods,
            "n_viagens_validacao": val_tot.reindex(top_ods).fillna(0).astype(int).values,
            "n_viagens_sintetico": syn_tot.reindex(top_ods).fillna(0).astype(int).values,
            "participacao_validacao_pct": (val_den.reindex(top_ods).fillna(0.0).values * 100.0),
            "participacao_sintetico_pct": (syn_den.reindex(top_ods).fillna(0.0).values * 100.0),
            "similaridade_hora_dot": sim_by_od,
        }
    )
    out_csv = SCRIPT_DIR / f"od_top{int(top_n)}_participacao_validacao_vs_sintetico.csv"
    df_csv.to_csv(out_csv, index=False)
    print(f"[OK] CSV salvo: {out_csv}")
    return top_ods


def plot_horario_por_od(val: pd.DataFrame, syn: pd.DataFrame, ods: list[str], top_n: int = TOP_N) -> None:
    hours = np.arange(24)
    rows: list[dict] = []
    # Denominadores globais (sem normalização por OD): garantem que a soma
    # por hora de uma OD reproduza a participação global dessa OD.
    val_total_all = float(val["n_viagens"].sum())
    syn_total_all = float(syn["n_viagens"].sum())
    for od in ods:
        v = (
            val[val["od"] == od]
            .groupby("hora_do_dia")["n_viagens"].sum()
            .reindex(hours, fill_value=0)
        )
        s = (
            syn[syn["od"] == od]
            .groupby("hora_do_dia")["n_viagens"].sum()
            .reindex(hours, fill_value=0)
        )
        # Frações globais (não normalizar por OD):
        #   v_frac(h) = v(h) / total_validacao_todas_ODs
        #   s_frac(h) = s(h) / total_sintetico_todas_ODs
        v_den = v / max(val_total_all, 1)
        s_den = s / max(syn_total_all, 1)

        # Distribuições normalizadas por OD (para comparar forma horária por OD)
        v_sum = float(v.sum())
        s_sum = float(s.sum())
        v_norm = (v / v_sum) if v_sum > 0 else np.zeros_like(v, dtype=np.float64)
        s_norm = (s / s_sum) if s_sum > 0 else np.zeros_like(s, dtype=np.float64)
        sim_dot = float(np.dot(v_norm, s_norm))

        # Acumula linhas para CSV consolidado por hora (percentual global + contagens)
        rows.extend(
            {
                "od": od,
                # Coluna adicional solicitada (OD = origem-destino, igual a 'od')
                "OD": od,
                "hora_do_dia": int(h),
                "n_viagens_validacao_hora": int(v.iloc[h]),
                "n_viagens_sintetico_hora": int(s.iloc[h]),
                "validacao_pct": float(v_den.iloc[h] * 100.0),
                "sintetico_pct": float(s_den.iloc[h] * 100.0),
                # Percentual normalizado por OD (forma)
                "validacao_pct_por_od": float(v_norm.iloc[h] * 100.0),
                "sintetico_pct_por_od": float(s_norm.iloc[h] * 100.0),
                # Similaridade (produto escalar) da distribuição horária por OD (constante por od)
                "similaridade_hora_dot": sim_dot,
            }
            for h in hours
        )

    # Salva CSV consolidado de distribuições horárias das Top-N (em save_data)
    if rows:
        df_out = pd.DataFrame(rows)
        out_csv = SCRIPT_DIR / f"od_top{int(top_n)}_distrib_hora_validacao_vs_sintetico.csv"
        df_out.to_csv(out_csv, index=False)
        print(f"[OK] CSV salvo: {out_csv}")


# --------------------------------------------------------------------------- #
#                 VARIABILIDADE VIA PSEUDO-DIAS (INTRA/CROSS)                 #
# --------------------------------------------------------------------------- #

def _group_probs_by_dow(df: pd.DataFrame) -> tuple[dict[int, pd.Series], pd.Series, pd.MultiIndex]:
    """Constrói, para cada dia-da-semana, a distribuição p(od,h|dow).

    Retorna:
    - probs_by_dow: dict[dow]-> pd.Series com índice MultiIndex (od, hora_do_dia) e soma=1
    - dow_mass: pd.Series com massa total por dow (soma=1 no total)
    - full_index: união de todos os índices (od,hora) observados
    """
    if df.empty:
        return {}, pd.Series(dtype=float), pd.MultiIndex.from_arrays([[], []], names=["od", "hora_do_dia"])

    # Soma por (od, hora, dow)
    g = df.groupby(["od", "hora_do_dia", "dia_da_semana"], as_index=False)["n_viagens"].sum()
    # Massa por dow
    dow_tot = g.groupby("dia_da_semana")["n_viagens"].sum()
    total_all = float(dow_tot.sum())
    if total_all <= 0:
        return {}, pd.Series(dtype=float), pd.MultiIndex.from_arrays([[], []], names=["od", "hora_do_dia"])
    dow_mass = (dow_tot / total_all).sort_index()

    probs_by_dow: dict[int, pd.Series] = {}
    all_index = []
    for dow, sub in g.groupby("dia_da_semana"):
        key = int(dow)
        # Índice (od,h)
        idx = pd.MultiIndex.from_frame(sub[["od", "hora_do_dia"]])
        all_index.append(idx)
        counts = pd.Series(sub["n_viagens"].to_numpy(dtype=np.int64), index=idx)
        denom = float(counts.sum())
        if denom <= 0:
            continue
        probs_by_dow[key] = (counts / denom).sort_index()

    # União de todos os índices
    if all_index:
        full_index = all_index[0]
        for idx in all_index[1:]:
            full_index = full_index.union(idx)
    else:
        full_index = pd.MultiIndex.from_arrays([[], []], names=["od", "hora_do_dia"])

    # Reindexa para o mesmo universo, preenchendo 0 onde não houver prob
    for dow in list(probs_by_dow.keys()):
        probs_by_dow[dow] = probs_by_dow[dow].reindex(full_index, fill_value=0.0)

    return probs_by_dow, dow_mass, full_index


def _sample_pseudo_days(
    probs_by_dow: dict[int, pd.Series],
    dow_mass: pd.Series,
    full_index: pd.MultiIndex,
    *,
    D: int,
    total_all: int,
    seed: int = 123,
) -> list[pd.Series]:
    """Amostra D pseudo-dias como contagens inteiras por (od,hora).

    - Seleciona o dow de cada dia segundo dow_mass (empírico, soma=1).
    - Define o total do dia como N_per_day ≈ total_all / D (distribui o resto).
    - Amostra uma multinomial em p(od,h|dow) para obter contagens por (od,h).

    Retorna lista de Series (index=full_index, dtype=int64) com as contagens do dia.
    """
    rng = np.random.default_rng(seed)
    if D <= 0 or total_all <= 0 or not probs_by_dow:
        return []

    # Vetor de dows possíveis e suas probabilidades
    dows = np.array(sorted(dow_mass.index.to_numpy(dtype=np.int64)))
    probs_dow = dow_mass.reindex(dows, fill_value=0.0).to_numpy(dtype=np.float64)
    if probs_dow.sum() <= 0:
        probs_dow = np.full_like(probs_dow, 1.0 / len(probs_dow))

    # Totais por dia: divide igualmente e distribui resto
    base = total_all // D
    resto = total_all % D
    totals = np.full(D, base, dtype=np.int64)
    if resto > 0:
        totals[:resto] += 1  # primeiros ‘resto’ dias recebem +1

    pseudo_days: list[pd.Series] = []
    for i in range(D):
        dow = int(rng.choice(dows, p=probs_dow))
        p = probs_by_dow.get(dow)
        if p is None or p.sum() <= 0:
            # fallback uniforme no universo observado
            p = pd.Series(np.full(len(full_index), 1.0 / max(len(full_index), 1)), index=full_index)
        probs = p.to_numpy(dtype=np.float64)
        probs /= max(probs.sum(), 1e-12)

        n = int(totals[i])
        if n <= 0:
            pseudo_days.append(pd.Series(np.zeros(len(full_index), dtype=np.int64), index=full_index))
            continue
        counts = rng.multinomial(n=n, pvals=probs)
        pseudo_days.append(pd.Series(counts.astype(np.int64), index=full_index))
    return pseudo_days


def _day_hour_dist(day_counts: pd.Series) -> np.ndarray:
    """Distribuição por hora (24) a partir de contagens por (od,hora)."""
    if day_counts.empty:
        return np.zeros(24, dtype=np.float64)
    idx = day_counts.index
    horas = idx.get_level_values("hora_do_dia").to_numpy()
    vals = day_counts.to_numpy(dtype=np.float64)
    out = np.zeros(24, dtype=np.float64)
    for h, v in zip(horas, vals):
        if 0 <= int(h) < 24:
            out[int(h)] += float(v)
    s = out.sum()
    return out / s if s > 0 else out


def _day_od_dist(day_counts: pd.Series) -> pd.Series:
    """Distribuição por OD (Series normalizada) a partir de contagens por (od,hora)."""
    if day_counts.empty:
        return pd.Series(dtype=np.float64)
    idx = day_counts.index
    od = idx.get_level_values("od")
    agg = day_counts.to_numpy(dtype=np.float64)
    df = pd.DataFrame({"od": od, "v": agg})
    s = df.groupby("od")["v"].sum()
    denom = float(s.sum())
    return (s / denom) if denom > 0 else s


def _pairwise_inner_products(vecs: list[np.ndarray]) -> list[tuple[int, int, float]]:
    """Produto escalar entre distribuições L1-normalizadas (ou seja, no [0,1])."""
    out: list[tuple[int, int, float]] = []
    n = len(vecs)
    for i in range(n):
        vi = vecs[i]
        for j in range(i + 1, n):
            vj = vecs[j]
            sim = float(np.dot(vi, vj))
            out.append((i, j, sim))
    return out


def _pairwise_inner_products_series(vecs: list[pd.Series]) -> list[tuple[int, int, float]]:
    """Produto escalar entre séries normalizadas (reindexa na união)."""
    out: list[tuple[int, int, float]] = []
    n = len(vecs)
    # Universo comum
    union_idx = pd.Index([])
    for s in vecs:
        union_idx = union_idx.union(s.index)
    aligned = [s.reindex(union_idx, fill_value=0.0).to_numpy(dtype=np.float64) for s in vecs]
    for i in range(n):
        vi = aligned[i]
        for j in range(i + 1, n):
            vj = aligned[j]
            sim = float(np.dot(vi, vj))
            out.append((i, j, sim))
    return out


def _cross_inner_products(vecs_a: list[np.ndarray], vecs_b: list[np.ndarray]) -> list[tuple[int, int, float]]:
    out: list[tuple[int, int, float]] = []
    for i, va in enumerate(vecs_a):
        for j, vb in enumerate(vecs_b):
            out.append((i, j, float(np.dot(va, vb))))
    return out


def _cross_inner_products_series(vecs_a: list[pd.Series], vecs_b: list[pd.Series]) -> list[tuple[int, int, float]]:
    out: list[tuple[int, int, float]] = []
    # Universo comum
    union_idx = pd.Index([])
    for s in vecs_a + vecs_b:
        union_idx = union_idx.union(s.index)
    A = [s.reindex(union_idx, fill_value=0.0).to_numpy(dtype=np.float64) for s in vecs_a]
    B = [s.reindex(union_idx, fill_value=0.0).to_numpy(dtype=np.float64) for s in vecs_b]
    for i, va in enumerate(A):
        for j, vb in enumerate(B):
            out.append((i, j, float(np.dot(va, vb))))
    return out


def _write_pairs(rows: list[tuple[int, int, float]], out_path: Path, tipo: str) -> None:
    if not rows:
        pd.DataFrame({"dia_i": [], "dia_j": [], "similaridade": [], "tipo": []}).to_csv(out_path, index=False)
        print(f"[OK] CSV salvo (vazio): {out_path}")
        return
    df = pd.DataFrame(rows, columns=["dia_i", "dia_j", "similaridade"])
    df["tipo"] = tipo
    df.to_csv(out_path, index=False)
    print(f"[OK] CSV salvo: {out_path}")


def _write_summary(rows: list[tuple[int, int, float]], out_path: Path, label: str) -> None:
    sims = np.array([r[2] for r in rows], dtype=np.float64) if rows else np.array([], dtype=np.float64)
    if sims.size == 0:
        df = pd.DataFrame([{"label": label, "media": np.nan, "std": np.nan, "p10": np.nan, "p50": np.nan, "p90": np.nan}])
        df.to_csv(out_path, index=False)
        print(f"[OK] CSV salvo (vazio): {out_path}")
        return
    summary = {
        "label": label,
        "media": float(np.mean(sims)),
        "std": float(np.std(sims)),
        "p10": float(np.percentile(sims, 10)),
        "p50": float(np.percentile(sims, 50)),
        "p90": float(np.percentile(sims, 90)),
    }
    pd.DataFrame([summary]).to_csv(out_path, index=False)
    print(f"[OK] CSV salvo: {out_path}")


def _write_summary_multi(rows_map: dict[str, list[tuple[int, int, float]]], out_path: Path) -> None:
    records = []
    for label, rows in rows_map.items():
        sims = np.array([r[2] for r in rows], dtype=np.float64) if rows else np.array([], dtype=np.float64)
        if sims.size == 0:
            records.append({"label": label, "media": np.nan, "std": np.nan, "p10": np.nan, "p50": np.nan, "p90": np.nan})
        else:
            records.append({
                "label": label,
                "media": float(np.mean(sims)),
                "std": float(np.std(sims)),
                "p10": float(np.percentile(sims, 10)),
                "p50": float(np.percentile(sims, 50)),
                "p90": float(np.percentile(sims, 90)),
            })
    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"[OK] CSV salvo: {out_path}")


def compute_variability_pseudodays(val: pd.DataFrame, syn: pd.DataFrame, *, D: int = VARIAB_D, seed: int = VARIAB_SEED) -> None:
    """Calcula variabilidade intra e cruzada via pseudo-dias amostrados dos agregados.

    - Mantém o mesmo total agregado por conjunto e divide em D dias.
    - Cada pseudo-dia herda um dia-da-semana conforme a massa empírica do conjunto.
    - As contagens do dia são amostradas de uma multinomial em p(od,h|dow).
    - Força os pseudo-dias de validação e de sintético a terem os mesmos totais por dia
      (estimativa alinhada), garantindo comparabilidade.
    """
    # Preparação de probs por dow (val e synth)
    val_probs_by_dow, val_dow_mass, val_index = _group_probs_by_dow(val)
    syn_probs_by_dow, syn_dow_mass, syn_index = _group_probs_by_dow(syn)
    full_index = val_index.union(syn_index)
    # Reindexa para universo comum
    for d in list(val_probs_by_dow.keys()):
        val_probs_by_dow[d] = val_probs_by_dow[d].reindex(full_index, fill_value=0.0)
    for d in list(syn_probs_by_dow.keys()):
        syn_probs_by_dow[d] = syn_probs_by_dow[d].reindex(full_index, fill_value=0.0)

    # Totais agregados
    total_val = int(val["n_viagens"].sum())
    total_syn = int(syn["n_viagens"].sum())
    if total_val <= 0 or D <= 0:
        print("[Variab] Conjunto de validação vazio ou D<=0; pulando variabilidade.")
        return

    # Definimos os totais por dia para ambos como os do conjunto de validação
    base = total_val // D
    resto = total_val % D
    val_totals = np.full(D, base, dtype=np.int64)
    if resto > 0:
        val_totals[:resto] += 1
    syn_totals = val_totals.copy()  # força igualar total por dia

    # Amostragem
    val_days = _sample_pseudo_days(val_probs_by_dow, val_dow_mass, full_index, D=D, total_all=int(val_totals.sum()), seed=seed)
    syn_days = _sample_pseudo_days(syn_probs_by_dow, syn_dow_mass, full_index, D=D, total_all=int(syn_totals.sum()), seed=seed + 7)

    # Correção defensiva: garante exatamente os totais desejados por dia (pequenos ajustes proporcionais)
    def _renorm_to(day_counts: pd.Series, target_total: int) -> pd.Series:
        cur = int(day_counts.sum())
        if cur == target_total:
            return day_counts
        if cur <= 0:
            return pd.Series(np.zeros(len(day_counts), dtype=np.int64), index=day_counts.index)
        scale = target_total / cur
        # arredonda mantendo soma
        vals = np.floor(day_counts.to_numpy(dtype=np.float64) * scale).astype(np.int64)
        diff = target_total - int(vals.sum())
        if diff > 0:
            # adiciona +1 nos ‘diff’ maiores resíduos
            residual = (day_counts.to_numpy(dtype=np.float64) * scale) - vals
            top_idx = np.argsort(residual)[::-1][:diff]
            vals[top_idx] += 1
        elif diff < 0:
            # remove 1 onde há maior valor absoluto
            top_idx = np.argsort(vals)[::-1][: (-diff)]
            vals[top_idx] -= 1
        return pd.Series(vals, index=day_counts.index)

    val_days = [
        _renorm_to(s, int(val_totals[i])) if len(s) else pd.Series(np.zeros(len(full_index), dtype=np.int64), index=full_index)
        for i, s in enumerate(val_days)
    ]
    syn_days = [
        _renorm_to(s, int(syn_totals[i])) if len(s) else pd.Series(np.zeros(len(full_index), dtype=np.int64), index=full_index)
        for i, s in enumerate(syn_days)
    ]

    # Distribuições por dia (hora e OD)
    val_hour = [_day_hour_dist(s) for s in val_days]
    syn_hour = [_day_hour_dist(s) for s in syn_days]
    val_od = [_day_od_dist(s) for s in val_days]
    syn_od = [_day_od_dist(s) for s in syn_days]

    # Similaridades
    intra_val_h = _pairwise_inner_products(val_hour)
    intra_syn_h = _pairwise_inner_products(syn_hour)
    cross_h = _cross_inner_products(val_hour, syn_hour)

    intra_val_od = _pairwise_inner_products_series(val_od)
    intra_syn_od = _pairwise_inner_products_series(syn_od)
    cross_od = _cross_inner_products_series(val_od, syn_od)

    # Escritas
    _write_pairs(intra_val_h, SCRIPT_DIR / "variab_hora_intra_validacao.csv", "intra_val")
    _write_pairs(intra_syn_h, SCRIPT_DIR / "variab_hora_intra_sintetico.csv", "intra_syn")
    _write_pairs(cross_h, SCRIPT_DIR / "variab_hora_cross_val_vs_synth.csv", "cross")

    _write_pairs(intra_val_od, SCRIPT_DIR / "variab_od_intra_validacao.csv", "intra_val")
    _write_pairs(intra_syn_od, SCRIPT_DIR / "variab_od_intra_sintetico.csv", "intra_syn")
    _write_pairs(cross_od, SCRIPT_DIR / "variab_od_cross_val_vs_synth.csv", "cross")

    # Resumos
    _write_summary(intra_val_h + intra_syn_h + cross_h, SCRIPT_DIR / "resumo_variab_hora.csv", label=f"D={D}")
    _write_summary(intra_val_od + intra_syn_od + cross_od, SCRIPT_DIR / "resumo_variab_od.csv", label=f"D={D}")


# --------------------------------------------------------------------------- #
#        AMOSTRAGEM DE LINHAS SINTÉTICAS PARA CASAR TOTAL DA VALIDAÇÃO        #
# --------------------------------------------------------------------------- #

def sample_synth_rows_to_match_validation(
    *,
    val_daily_path: Path = SCRIPT_DIR / "od_hora_dia_validacao_10dias.csv",
    synth_path: Path = SCRIPT_DIR / "od_hora_dia_sintetico.csv",
    out_path: Path = SCRIPT_DIR / "od_hora_dia_sintetico_amostrado.csv",
    margin: float = MATCH_MARGIN,
    seed: int = VARIAB_SEED,
    max_tries: int = 50,
) -> None:
    """Seleciona linhas aleatórias (sem reposição) do CSV sintético até
    aproximar o total de viagens do CSV de validação (10 dias), com margem.
    """
    if not val_daily_path.exists():
        print(f"[Match] Arquivo de validação diária não encontrado: {val_daily_path}; pulando match.")
        return
    if not synth_path.exists():
        print(f"[Match] Arquivo sintético não encontrado: {synth_path}; pulando match.")
        return
    val = pd.read_csv(val_daily_path)
    target = int(val["n_viagens"].sum()) if not val.empty else 0
    if target <= 0:
        print(f"[Match] Total alvo inválido (={target}); pulando match.")
        return
    syn = pd.read_csv(synth_path)
    if syn.empty or "n_viagens" not in syn.columns:
        print(f"[Match] CSV sintético vazio ou sem n_viagens; pulando match.")
        return

    n = len(syn)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    best_sel = None
    best_err = float("inf")
    lo = target * (1.0 - margin)
    hi = target * (1.0 + margin)

    for _ in range(max_tries):
        rng.shuffle(idx)
        total = 0
        cut = 0
        for k in range(n):
            total += int(syn.loc[idx[k], "n_viagens"])
            if total >= lo:
                cut = k + 1
                break
        if cut == 0:
            # não alcançou o mínimo; usa tudo
            cut = n
            total = int(syn.loc[idx[:cut], "n_viagens"].sum())
        rel_err = abs(total - target) / max(target, 1)
        if lo <= total <= hi:
            best_sel = idx[:cut].copy()
            best_err = rel_err
            break
        # mantém o melhor até agora
        if rel_err < best_err:
            best_sel = idx[:cut].copy()
            best_err = rel_err

    if best_sel is None or best_sel.size == 0:
        print("[Match] Falha ao selecionar subconjunto; pulando.")
        return

    sel_df = syn.iloc[best_sel].copy().reset_index(drop=True)
    sel_total = int(sel_df["n_viagens"].sum())
    print(
        f"[Match] target={target} | selecionado={sel_total} | erro_relativo={(abs(sel_total-target)/max(target,1)):.4f} | linhas={len(sel_df)}"
    )
    sel_df.to_csv(out_path, index=False)
    print(f"[OK] CSV salvo: {out_path}")


# --------------------------------------------------------------------------- #
#    AGREGAÇÃO DE VALIDAÇÃO EM 10 DIAS CONSECUTIVOS (SEM RETREINAR MODELO)    #
# --------------------------------------------------------------------------- #

def build_validation_10days_csv_from_raw(
    *,
    days: int = 10,
    seed: int = VARIAB_SEED,
    out_path: Path = SCRIPT_DIR / "od_hora_dia_validacao_10dias.csv",
) -> None:
    """Gera `od_hora_dia_validacao_10dias.csv` a partir do parquet bruto,
    sem necessidade de treinar/rodar o pipeline do modelo.

    - Usa loader.load_real_data + split_dataset_weekly para obter o conjunto de validação.
    - Usa utils.zone_id.assign_zone_names para obter nomes de zonas.
    - Usa od_discretizer.discretize_od_pairs para obter `od_pair` (normalizado).
    - Agrega por data × od × hora_do_dia × dia_da_semana.
    """
    try:
        from data_processing.loader import load_real_data, split_dataset_weekly
        from utils.zone_id import assign_zone_names
        from od_discretizer import discretize_od_pairs
    except Exception as e:
        print(f"[DailyVal] Falha ao importar dependências: {e}")
        return

    try:
        full_df, *_ = load_real_data()
    except Exception as e:
        print(f"[DailyVal] Falha ao carregar dados reais: {e}")
        return

    try:
        train_raw, val_raw, hold_raw = split_dataset_weekly(full_df, train_frac=0.80, val_frac=0.20, datetime_col="tpep_pickup_datetime")
    except Exception as e:
        print(f"[DailyVal] Falha ao particionar semanas: {e}")
        return

    if val_raw.empty:
        print("[DailyVal] Conjunto de validação vazio; abortando agregação diária.")
        return

    try:
        val_enriched = assign_zone_names(val_raw.copy())
        disc_df, _ = discretize_od_pairs(val_enriched)
    except Exception as e:
        print(f"[DailyVal] Falha ao enriquecer/discretizar OD: {e}")
        return

    # Seleciona janela de 'days' dias consecutivos com seed determinística
    ts = pd.to_datetime(disc_df["tpep_pickup_datetime"], errors="coerce")
    dates = ts.dt.normalize()
    uniq = sorted(dates.dropna().unique())
    if not uniq:
        print("[DailyVal] Sem datas válidas no conjunto de validação.")
        return
    D = int(max(1, days))
    if len(uniq) <= D:
        chosen = uniq
    else:
        max_start = len(uniq) - D
        start = int(seed) % (max_start + 1)
        chosen = uniq[start : start + D]

    mask = dates.isin(chosen)
    sub = disc_df.loc[mask].copy()
    if sub.empty:
        print("[DailyVal] Janela selecionada resultou vazia.")
        return

    ts_sub = pd.to_datetime(sub["tpep_pickup_datetime"], errors="coerce")
    sub["data"] = ts_sub.dt.normalize().dt.date.astype(str)
    sub["hora_do_dia"] = ts_sub.dt.hour.astype(int)
    sub["dia_da_semana"] = ts_sub.dt.dayofweek.astype(int)
    sub["od"] = sub["od_pair"].astype(str)

    agg = (
        sub.groupby(["data", "od", "hora_do_dia", "dia_da_semana"]).size().reset_index(name="n_viagens")
    )
    agg = agg.sort_values(["data", "od", "dia_da_semana", "hora_do_dia"]).reset_index(drop=True)

    try:
        agg.to_csv(out_path, index=False)
        total_val = int(agg["n_viagens"].sum()) if not agg.empty else 0
        print(f"[DailyVal] CSV salvo: {out_path} | dias={len(set(agg['data']))} | total_n_viagens={total_val}")
    except Exception as e:
        print(f"[DailyVal] Falha ao salvar CSV: {e}")


# --------------------------------------------------------------------------- #
#          VARIABILIDADE REAL (10 DIAS) USANDO DIAS REAIS DE VALIDAÇÃO        #
# --------------------------------------------------------------------------- #

def compute_variability_real_days(
    *,
    val10_path: Path = SCRIPT_DIR / "od_hora_dia_validacao_10dias.csv",
    synth_path_pref: Path = SCRIPT_DIR / "od_hora_dia_sintetico_amostrado.csv",
    synth_fallback_path: Path = SCRIPT_DIR / "od_hora_dia_sintetico.csv",
    seed: int = VARIAB_SEED,
) -> None:
    """Calcula variabilidade intra/cross para 10 dias REAIS (validação) e
    10 dias SINTÉTICOS gerados a partir da distribuição sintética por DOW,
    respeitando o total por dia da validação.
    """
    if not val10_path.exists():
        print(f"[RealVar] Arquivo não encontrado: {val10_path}; abortando variabilidade real.")
        return
    syn_path = synth_path_pref if synth_path_pref.exists() else synth_fallback_path
    if not syn_path.exists():
        print(f"[RealVar] Arquivo sintético não encontrado: {syn_path}; abortando variabilidade real.")
        return

    val = pd.read_csv(val10_path)
    syn = pd.read_csv(syn_path)
    if val.empty:
        print("[RealVar] Validação diária vazia; abortando.")
        return
    if syn.empty:
        print("[RealVar] Sintético vazio; abortando.")
        return

    # Ordena dias
    days = sorted(val["data"].unique())
    D = len(days)
    if D == 0:
        print("[RealVar] Sem 'data' em validação.")
        return

    # Preparação de probs por DOW do sintético
    probs_by_dow, dow_mass, full_index = _group_probs_by_dow(syn)
    # Garante universo comum
    for dkey in list(probs_by_dow.keys()):
        probs_by_dow[dkey] = probs_by_dow[dkey].reindex(full_index, fill_value=0.0)

    rng = np.random.default_rng(seed)

    # Distribuições REAIS por dia
    val_hour_vecs: list[np.ndarray] = []
    val_od_vecs: list[pd.Series] = []
    day_totals: list[int] = []
    day_dows: list[int] = []
    for d in days:
        sub = val[val["data"] == d]
        total = int(sub["n_viagens"].sum())
        day_totals.append(total)
        # DOW do dia (assume consistente)
        if "dia_da_semana" in sub.columns and not sub["dia_da_semana"].empty:
            dow = int(sub["dia_da_semana"].iloc[0])
        else:
            dow = 0
        day_dows.append(dow)
        # Hora dist
        h = (
            sub.groupby("hora_do_dia")["n_viagens"].sum().reindex(np.arange(24), fill_value=0).to_numpy(np.int64)
        )
        h = h.astype(np.float64)
        h = h / max(h.sum(), 1e-12)
        val_hour_vecs.append(h)
        # OD dist
        od = sub.groupby("od")["n_viagens"].sum()
        denom = float(od.sum())
        od = (od / denom) if denom > 0 else od
        val_od_vecs.append(od)

    # Distribuições SINTÉTICAS por dia (amostra multinomial por DOW e total real)
    syn_hour_vecs: list[np.ndarray] = []
    syn_od_vecs: list[pd.Series] = []
    for idx, (dow, total) in enumerate(zip(day_dows, day_totals)):
        p = probs_by_dow.get(dow)
        if p is None or p.sum() <= 0:
            # fallback: usa mistura por massa dos DOWs
            if probs_by_dow:
                # média ponderada no universo full_index
                mix = sum(mass * probs_by_dow[k] for k, mass in dow_mass.items())
                p = mix.reindex(full_index, fill_value=0.0)
            else:
                p = pd.Series(np.full(len(full_index), 1.0 / max(len(full_index), 1)), index=full_index)
        probs = p.to_numpy(dtype=np.float64)
        probs /= max(probs.sum(), 1e-12)
        counts = rng.multinomial(n=int(max(total, 0)), pvals=probs)
        day_series = pd.Series(counts.astype(np.int64), index=full_index)
        # Hora dist
        syn_hour_vecs.append(_day_hour_dist(day_series))
        syn_od_vecs.append(_day_od_dist(day_series))

    # Similaridades
    intra_val_h = _pairwise_inner_products(val_hour_vecs)
    intra_syn_h = _pairwise_inner_products(syn_hour_vecs)
    cross_h = _cross_inner_products(val_hour_vecs, syn_hour_vecs)

    intra_val_od = _pairwise_inner_products_series(val_od_vecs)
    intra_syn_od = _pairwise_inner_products_series(syn_od_vecs)
    cross_od = _cross_inner_products_series(val_od_vecs, syn_od_vecs)

    # Escritas com sufixo real10
    _write_pairs(intra_val_h, SCRIPT_DIR / "variab_real10_hora_intra_validacao.csv", "intra_val")
    _write_pairs(intra_syn_h, SCRIPT_DIR / "variab_real10_hora_intra_sintetico.csv", "intra_syn")
    _write_pairs(cross_h, SCRIPT_DIR / "variab_real10_hora_cross_val_vs_synth.csv", "cross")

    _write_pairs(intra_val_od, SCRIPT_DIR / "variab_real10_od_intra_validacao.csv", "intra_val")
    _write_pairs(intra_syn_od, SCRIPT_DIR / "variab_real10_od_intra_sintetico.csv", "intra_syn")
    _write_pairs(cross_od, SCRIPT_DIR / "variab_real10_od_cross_val_vs_synth.csv", "cross")

    # Resumos multi-linha (um por tipo)
    _write_summary_multi(
        {
            "hora:intra_val": intra_val_h,
            "hora:intra_syn": intra_syn_h,
            "hora:cross": cross_h,
        },
        SCRIPT_DIR / "resumo_variab_real10_hora.csv",
    )
    _write_summary_multi(
        {
            "od:intra_val": intra_val_od,
            "od:intra_syn": intra_syn_od,
            "od:cross": cross_od,
        },
        SCRIPT_DIR / "resumo_variab_real10_od.csv",
    )


def main() -> None:
    _ensure_dirs(TOP_N)
    # 1) Gera validação diária (10 dias) se ainda não existir
    daily_path = SCRIPT_DIR / "od_hora_dia_validacao_10dias.csv"
    if not daily_path.exists():
        env_seed = int(os.environ.get("VAE_SEED", str(VARIAB_SEED))) if "os" in globals() else VARIAB_SEED
        build_validation_10days_csv_from_raw(days=10, seed=env_seed, out_path=daily_path)
    val, syn = _load_inputs()
    top_ods = plot_participacao_topN(val, syn, TOP_N)
    if top_ods:
        plot_horario_por_od(val, syn, top_ods, TOP_N)
    # Variabilidade via pseudo-dias (aproximação, sem alterar o gerador)
    if VARIAB_ENABLE:
        print(f"[Variab] Gerando variabilidade com D={VARIAB_D} pseudo-dias...")
        compute_variability_pseudodays(val, syn, D=int(VARIAB_D), seed=int(VARIAB_SEED))
    # Subconjunto sintético para casar total de n_viagens com validação (10 dias)
    if MATCH_SYNTH_TO_VAL:
        sample_synth_rows_to_match_validation()
        # Variabilidade REAL com 10 dias (validação real + sintético por DOW)
        compute_variability_real_days()


if __name__ == "__main__":
    main()
