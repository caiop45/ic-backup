from pathlib import Path

from tools.plot_training_curves import plot_training_curves


def _write_scalars(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "step,tag,value\n",
        "0,train/loss,1.0\n",
        "1,train/loss,0.8\n",
        "0,val/loss,1.2\n",
        "1,val/loss,0.9\n",
    ]
    path.write_text("".join(lines), encoding="utf-8")


def test_plot_training_curves_creates_pngs(tmp_path):
    run_dir = tmp_path / "run"
    scalars_path = run_dir / "logs" / "scalars.csv"
    _write_scalars(scalars_path)

    out_dir = run_dir / "plots"
    created = plot_training_curves(run_dir, out_dir)

    assert created
    assert (out_dir / "train__loss.png").exists()
    assert (out_dir / "val__loss.png").exists()
