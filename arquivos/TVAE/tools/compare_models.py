from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

DEFAULT_COLUMNS = [
    "model",
    "run_dir",
    "eval_split",
    "n_real",
    "n_synth",
    "w1_tr_te",
    "w1_tr_syn",
    "w1_te_syn",
    "g_tr_te",
    "g_tr_syn",
    "g_te_syn",
    "cov_tr_te",
    "cov_tr_syn",
    "cov_te_syn",
    "dcr_tr_syn_p05",
    "dcr_hold_syn_p05",
    "rdcr_p05",
    "dcr_rr_p05",
    "dcr_ss_p05",
]


def _load_metrics(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_metrics_table(
    baseline_metrics: Dict[str, object],
    strategy_metrics: Dict[str, object],
    *,
    baseline_label: str = "baseline",
    strategy_label: str = "strategy_a",
    baseline_run: Path | None = None,
    strategy_run: Path | None = None,
    columns: List[str] | None = None,
) -> pd.DataFrame:
    cols = columns or DEFAULT_COLUMNS
    rows = []

    def _row(label: str, run_dir: Path | None, metrics: Dict[str, object]) -> Dict[str, object]:
        row = {"model": label, "run_dir": str(run_dir) if run_dir is not None else ""}
        for key in cols:
            if key in ("model", "run_dir"):
                continue
            row[key] = metrics.get(key)
        return row

    rows.append(_row(baseline_label, baseline_run, baseline_metrics))
    rows.append(_row(strategy_label, strategy_run, strategy_metrics))

    return pd.DataFrame(rows, columns=cols)


def _write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    headers = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    def fmt(val: object) -> str:
        if isinstance(val, float):
            return f"{val:.4f}"
        return str(val)

    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[h]) for h in headers) + " |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_metrics_path(run_dir: Path, preferred: Iterable[str | Path]) -> Path:
    for name in preferred:
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No metrics file found in {run_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare baseline TVAE and Strategy A metrics")
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--strategy-a-run", type=Path, required=True)
    parser.add_argument("--baseline-metrics", type=Path, default=None)
    parser.add_argument("--strategy-metrics", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path.cwd())
    args = parser.parse_args()

    baseline_metrics_path = args.baseline_metrics or _resolve_metrics_path(
        args.baseline_run,
        [
            Path("metrics") / "metrics_tvae_order_1_hold.json",
            Path("metrics") / "metrics_tvae_order_1_val.json",
            "metrics_tvae_order_1_hold.json",
            "metrics_tvae_order_1_val.json",
            "metrics_order_1_hold.json",
            "metrics_order_1.json",
        ],
    )
    strategy_metrics_path = args.strategy_metrics or _resolve_metrics_path(
        args.strategy_a_run,
        [
            Path("metrics") / "metrics_strategy_a_hold.json",
            Path("metrics") / "metrics_strategy_a_val.json",
            "metrics_strategy_a_hold.json",
            "metrics_strategy_a.json",
        ],
    )

    baseline_metrics = _load_metrics(baseline_metrics_path)
    strategy_metrics = _load_metrics(strategy_metrics_path)

    df = build_metrics_table(
        baseline_metrics,
        strategy_metrics,
        baseline_run=args.baseline_run,
        strategy_run=args.strategy_a_run,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "metrics_table.csv"
    md_path = args.out_dir / "metrics_table.md"
    df.to_csv(csv_path, index=False)
    _write_markdown_table(df, md_path)

    print(f"[Compare] wrote {csv_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
