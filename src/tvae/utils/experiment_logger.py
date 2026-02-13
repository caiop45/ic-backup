from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping

from tvae import config


class ExperimentLogger:
    """Lightweight scalar logger with CSV + optional TensorBoard support."""

    def __init__(self, run_dir: Path, run_name: str, enable_tb: bool | None = None) -> None:
        self.run_dir = Path(run_dir)
        self.run_name = run_name
        self.log_dir = self.run_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.log_dir / "scalars.csv"
        self.flush_every = int(getattr(config, "LOG_FLUSH_EVERY", 50))
        self.enable_tb = (
            bool(getattr(config, "LOG_ENABLE_TENSORBOARD", True))
            if enable_tb is None
            else bool(enable_tb)
        )

        self._buffer: list[tuple[int, str, float]] = []
        self._writer = None

        if self.enable_tb:
            try:
                from torch.utils.tensorboard import SummaryWriter

                tb_dir = self.log_dir / "tensorboard"
                tb_dir.mkdir(parents=True, exist_ok=True)
                self._writer = SummaryWriter(log_dir=str(tb_dir))
            except Exception as exc:  # pragma: no cover - depends on optional deps
                print(f"[Logger] TensorBoard disabled: {exc}")
                self.enable_tb = False
                self._writer = None

        self._ensure_header()

    def _ensure_header(self) -> None:
        if self.csv_path.exists():
            return
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["step", "tag", "value"])

    @staticmethod
    def _format_tag(tag: str, split: str | None) -> str:
        if split:
            prefix = f"{split}/"
            if tag.startswith(prefix):
                return tag
            return f"{split}/{tag}"
        return tag

    def log_scalar(self, tag: str, value: float, step: int, split: str | None = None) -> None:
        full_tag = self._format_tag(tag, split)
        self._buffer.append((int(step), str(full_tag), float(value)))

        if self._writer is not None:
            self._writer.add_scalar(full_tag, float(value), int(step))

        if self.flush_every > 0 and len(self._buffer) >= self.flush_every:
            self.flush()

    def log_scalars(self, values: Mapping[str, float], step: int, prefix: str = "") -> None:
        normalized = prefix.rstrip("/")
        for key, val in values.items():
            tag = f"{normalized}/{key}" if normalized else str(key)
            self.log_scalar(tag, float(val), step, split=None)

    def flush(self) -> None:
        if not self._buffer:
            if self._writer is not None:
                self._writer.flush()
            return
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("a", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerows(self._buffer)
        self._buffer.clear()
        if self._writer is not None:
            self._writer.flush()

    def close(self) -> None:
        self.flush()
        if self._writer is not None:
            self._writer.close()

    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = ["ExperimentLogger"]
