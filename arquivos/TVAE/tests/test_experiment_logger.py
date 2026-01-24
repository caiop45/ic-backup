import csv

from utils.experiment_logger import ExperimentLogger


def test_experiment_logger_writes_csv(tmp_path):
    run_dir = tmp_path / "run"
    logger = ExperimentLogger(run_dir, run_name="unit", enable_tb=False)

    for i in range(10):
        logger.log_scalar("loss", value=1.0 + i, step=i, split="train")

    logger.close()

    csv_path = run_dir / "logs" / "scalars.csv"
    assert csv_path.exists()

    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    assert len(rows) == 10
    assert rows[0]["tag"] == "train/loss"
    assert rows[0]["step"] == "0"
