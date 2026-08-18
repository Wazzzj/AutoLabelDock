import sys
import json

from PyQt5.QtCore import QCoreApplication

from src.engine.trainer import TrainConfig
from src.utils.workers import TrainWorker


_APP = QCoreApplication.instance() or QCoreApplication([])


class _FakeTrainer:
    cancelled = False

    def train(self, config, on_epoch_end=None):
        pass

    def get_best_metrics(self):
        return {"metric": 1.0}


class _OrderCheckingTrainWorker(TrainWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.released = False

    def _release_trainer(self, trainer) -> None:
        self.released = True


def test_train_worker_emits_finished_after_resource_release():
    config = TrainConfig(data_yaml="data.yaml", model="model.pt", task="detect")
    worker = _OrderCheckingTrainWorker(config, trainer_cls=_FakeTrainer)
    events = []

    worker.finished_ok.connect(
        lambda metrics: events.append(("finished", worker.released, metrics))
    )

    worker.start()
    assert worker.wait(5000)
    _APP.processEvents()

    assert events == [("finished", True, {"metric": 1.0})]


def test_train_worker_does_not_mutate_backend_internals_during_release():
    loader = object()
    validator = object()
    inner = type(
        "InnerTrainer",
        (),
        {"train_loader": loader, "test_loader": loader, "validator": validator},
    )()
    model = type("Model", (), {"trainer": inner})()
    trainer = type("BackendTrainer", (), {"_model": model})()
    worker = TrainWorker(
        TrainConfig(data_yaml="data.yaml", model="model.pt", task="detect"),
        trainer_cls=_FakeTrainer,
    )
    worker._trainer = trainer

    worker._release_trainer(trainer)

    assert worker._trainer is None
    assert trainer._model is model
    assert inner.train_loader is loader
    assert inner.test_loader is loader
    assert inner.validator is validator


class _CrashAfterSaveWorker(TrainWorker):
    def _build_training_command(self, config_path, cancel_path):
        code = (
            "import os, pathlib; "
            f"p=pathlib.Path({str(self._config.project)!r})/{self._config.name!r}; "
            "(p/'weights').mkdir(parents=True, exist_ok=True); "
            "(p/'weights'/'best.pt').write_bytes(b'model'); "
            "(p/'results.csv').write_text('epoch,metrics/mAP50(B)\\n1,0.75\\n', encoding='utf-8'); "
            "os._exit(3)"
        )
        return [sys.executable, "-s", "-c", code]


def test_train_worker_recovers_completed_model_after_native_child_exit(tmp_path):
    config = TrainConfig(
        data_yaml="data.yaml",
        model="model.pt",
        task="detect",
        project=str(tmp_path),
        name="run",
    )
    worker = _CrashAfterSaveWorker(config)
    events = []
    worker.finished_ok.connect(lambda metrics: events.append(("finished", metrics)))
    worker.error.connect(lambda message: events.append(("error", message)))

    worker.start()
    assert worker.wait(10000)
    _APP.processEvents()

    assert events == [("finished", {"epoch": 1.0, "mAP50": 0.75})]


class _EventFileWorker(TrainWorker):
    def _build_training_command(self, config_path, cancel_path):
        event = json.dumps({"type": "finished", "payload": {"metric": 0.9}})
        code = (
            "from pathlib import Path; "
            f"Path({str(self._event_path)!r}).write_text("
            f"{'AUTOLABEL_EVENT' + chr(9) + event + chr(10)!r}, encoding='utf-8')"
        )
        return [sys.executable, "-s", "-c", code]


def test_train_worker_reads_events_without_child_stdout(tmp_path):
    config = TrainConfig(
        data_yaml="data.yaml",
        model="model.pt",
        task="detect",
        project=str(tmp_path),
        name="run",
    )
    best_path = tmp_path / "run" / "weights" / "best.pt"
    best_path.parent.mkdir(parents=True)
    best_path.write_bytes(b"model")
    worker = _EventFileWorker(config)
    events = []
    worker.finished_ok.connect(lambda metrics: events.append(("finished", metrics)))

    worker.start()
    assert worker.wait(10000)
    _APP.processEvents()

    assert events == [("finished", {"metric": 0.9})]
