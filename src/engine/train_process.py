"""Qt-free subprocess entry point for isolated model training."""
from __future__ import annotations

import json
import logging
import sys
import threading
from pathlib import Path

from src.engine.backends import get_backend
from src.engine.trainer import TrainConfig
from src.utils.runtime_env import configure_headless_matplotlib, disable_user_site_packages


EVENT_PREFIX = "AUTOLABEL_EVENT\t"


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _emit(event_type: str, payload=None) -> None:
    message = {"type": event_type, "payload": payload}
    print(
        EVENT_PREFIX + json.dumps(message, ensure_ascii=False, default=_json_default),
        flush=True,
    )


def run_training(config_path: Path, cancel_path: Path) -> int:
    disable_user_site_packages()
    configure_headless_matplotlib()
    logging.basicConfig(level=logging.INFO)

    data = json.loads(config_path.read_text(encoding="utf-8"))
    config = TrainConfig(**data)
    trainer = get_backend(config.backend_id).create_trainer()
    monitor_stop = threading.Event()

    def monitor_cancel() -> None:
        while not monitor_stop.wait(0.2):
            if cancel_path.exists():
                trainer.request_cancel()
                return

    monitor = threading.Thread(target=monitor_cancel, daemon=True)
    monitor.start()
    try:
        trainer.train(config, on_epoch_end=lambda metrics: _emit("epoch", metrics))
        if trainer.cancelled:
            _emit("cancelled")
        else:
            _emit("finished", trainer.get_best_metrics())
        return 0
    except Exception as exc:
        logging.exception("Isolated training failed")
        _emit("error", str(exc))
        return 1
    finally:
        monitor_stop.set()
        monitor.join(timeout=1.0)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print("usage: python -m src.engine.train_process CONFIG CANCEL_FILE", file=sys.stderr)
        return 2
    return run_training(Path(args[0]), Path(args[1]))


if __name__ == "__main__":
    raise SystemExit(main())
