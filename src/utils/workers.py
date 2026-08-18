"""QThread workers for training and batch inference."""
from __future__ import annotations

import csv
from dataclasses import asdict
import inspect
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from PyQt5.QtCore import QThread, pyqtSignal, QMutex

from src.core.annotation import Annotation
from src.engine.backends.base import TrainerProtocol
from src.engine.trainer import TrainConfig

logger = logging.getLogger(__name__)

_CLASSIFY_BATCH_SIZE = 16
_TRAIN_EVENT_PREFIX = "AUTOLABEL_EVENT\t"


class TrainWorker(QThread):
    """Runs backend model training in a background thread.

    Signals:
        epoch_update(dict): Emitted after each epoch with metrics dict.
        finished_ok(dict): Emitted on successful completion with best metrics.
        cancelled(): Emitted when training is cancelled by user.
        error(str): Emitted if training fails with error message.
    """

    epoch_update = pyqtSignal(dict)
    finished_ok = pyqtSignal(dict)
    cancelled = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, config: TrainConfig, trainer_cls=None, parent=None):
        super().__init__(parent)
        self._config = config
        self._trainer_cls = trainer_cls
        self._trainer: TrainerProtocol | None = None
        self._trainer_mutex = QMutex()
        self._outcome = ""
        self._outcome_payload = None
        self._cancel_requested = False
        self._cancel_path: Path | None = None
        self._event_path: Path | None = None
        self._process: subprocess.Popen | None = None
        self._process_lock = threading.Lock()
        # QThread.finished is emitted only after run() returns and its Python
        # stack has unwound. Forward the application result afterwards so the
        # GUI never refreshes models while native training resources tear down.
        self.finished.connect(self._emit_outcome)

    def cancel(self) -> None:
        """Request graceful cancellation of training."""
        self._cancel_requested = True
        if self._trainer_cls is not None:
            self._trainer_mutex.lock()
            try:
                if self._trainer:
                    self._trainer.request_cancel()
            finally:
                self._trainer_mutex.unlock()
            return
        with self._process_lock:
            cancel_path = self._cancel_path
        if cancel_path is not None:
            try:
                cancel_path.touch(exist_ok=True)
            except OSError:
                logger.exception("Failed to write training cancellation marker")

    def run(self) -> None:
        if self._trainer_cls is None:
            self._run_isolated()
        else:
            self._run_inline()

    def _run_inline(self) -> None:
        trainer = None
        self._outcome = ""
        self._outcome_payload = None
        try:
            trainer = self._trainer_cls()
            self._trainer_mutex.lock()
            try:
                self._trainer = trainer
            finally:
                self._trainer_mutex.unlock()
            trainer.train(self._config, on_epoch_end=self._on_epoch)
            if trainer.cancelled:
                self._outcome = "cancelled"
            else:
                self._outcome_payload = trainer.get_best_metrics()
                self._outcome = "finished"
        except Exception as e:
            # Broad catch intentional: uncaught exceptions in QThread silently kill the thread
            logger.exception("Training failed")
            self._outcome_payload = str(e)
            self._outcome = "error"
        finally:
            self._release_trainer(trainer)

    def _run_isolated(self) -> None:
        self._outcome = ""
        self._outcome_payload = None
        project_root = Path(__file__).resolve().parents[2]
        try:
            with tempfile.TemporaryDirectory(prefix="autolabel-train-") as temp_dir:
                temp_root = Path(temp_dir)
                config_path = temp_root / "config.json"
                cancel_path = temp_root / "cancel"
                event_path = temp_root / "events.jsonl"
                config_path.write_text(
                    json.dumps(asdict(self._config), ensure_ascii=False),
                    encoding="utf-8",
                )
                with self._process_lock:
                    self._cancel_path = cancel_path
                    self._event_path = event_path
                if self._cancel_requested:
                    cancel_path.touch(exist_ok=True)

                env = os.environ.copy()
                env["PYTHONNOUSERSITE"] = "1"
                env["MPLBACKEND"] = "Agg"
                env["PYTHONUTF8"] = "1"
                env["PYTHONIOENCODING"] = "utf-8"
                python_path = env.get("PYTHONPATH", "")
                env["PYTHONPATH"] = str(project_root) + (
                    os.pathsep + python_path if python_path else ""
                )
                creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                command = self._build_training_command(config_path, cancel_path)
                process = subprocess.Popen(
                    command,
                    cwd=str(project_root),
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=creationflags,
                )
                with self._process_lock:
                    self._process = process

                event_offset = 0
                while True:
                    event_offset = self._drain_process_events(event_path, event_offset)
                    exit_code = process.poll()
                    if exit_code is not None:
                        self._drain_process_events(event_path, event_offset)
                        break
                    time.sleep(0.1)
                self._recover_process_outcome(exit_code)
        except Exception as exc:
            logger.exception("Failed to run isolated training process")
            self._outcome = "error"
            self._outcome_payload = str(exc)
        finally:
            with self._process_lock:
                self._process = None
                self._cancel_path = None
                self._event_path = None

    def _build_training_command(self, config_path: Path, cancel_path: Path) -> list[str]:
        event_path = self._event_path
        if event_path is None:
            raise RuntimeError("training event path is not initialized")
        if getattr(sys, "frozen", False):
            return [
                sys.executable,
                "--train-process",
                str(config_path),
                str(cancel_path),
                str(event_path),
            ]
        return [
            sys.executable,
            "-s",
            "-m",
            "src.engine.train_process",
            str(config_path),
            str(cancel_path),
            str(event_path),
        ]

    def _drain_process_events(self, event_path: Path, offset: int) -> int:
        """Read newly appended training events and return the next byte offset."""
        if not event_path.exists():
            return offset
        try:
            with event_path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(offset)
                for raw_line in handle:
                    line = raw_line.rstrip()
                    if line.startswith(_TRAIN_EVENT_PREFIX):
                        self._handle_process_event(line[len(_TRAIN_EVENT_PREFIX):])
                    elif line:
                        logger.debug("[training process] %s", line)
                return handle.tell()
        except OSError:
            logger.exception("Failed to read training event file: %s", event_path)
            return offset

    def _handle_process_event(self, payload: str) -> None:
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            logger.warning("Invalid event from training process: %s", payload)
            return
        event_type = event.get("type")
        value = event.get("payload")
        if event_type == "epoch" and isinstance(value, dict):
            self._on_epoch(value)
        elif event_type == "finished":
            self._outcome = "finished"
            self._outcome_payload = value if isinstance(value, dict) else {}
        elif event_type == "cancelled":
            self._outcome = "cancelled"
            self._outcome_payload = None
        elif event_type == "error":
            self._outcome = "error"
            self._outcome_payload = str(value)

    def _recover_process_outcome(self, exit_code: int) -> None:
        best_path = (
            Path(self._config.project)
            / self._config.name
            / "weights"
            / "best.pt"
        )
        if self._outcome == "finished" and best_path.is_file():
            if exit_code != 0:
                logger.warning(
                    "Training process exited with native code %#x after saving best.pt; "
                    "recovering the completed run",
                    exit_code & 0xFFFFFFFF,
                )
            return
        if self._cancel_requested or self._outcome == "cancelled":
            self._outcome = "cancelled"
            self._outcome_payload = None
            return
        if best_path.is_file():
            self._outcome = "finished"
            self._outcome_payload = _read_saved_metrics(best_path.parents[1] / "results.csv")
            logger.warning(
                "Training process ended with code %#x after model files were saved; "
                "recovered completion from results.csv",
                exit_code & 0xFFFFFFFF,
            )
            return
        if not self._outcome or self._outcome == "finished":
            self._outcome = "error"
            self._outcome_payload = (
                f"训练子进程异常退出（代码 {exit_code & 0xFFFFFFFF:#x}），未生成 best.pt"
            )

    def _emit_outcome(self) -> None:
        outcome = self._outcome
        payload = self._outcome_payload
        self._outcome = ""
        self._outcome_payload = None
        if outcome == "finished":
            self.finished_ok.emit(payload or {})
        elif outcome == "cancelled":
            self.cancelled.emit()
        elif outcome == "error":
            self.error.emit(str(payload))

    def _release_trainer(self, trainer) -> None:
        """Release only the worker-owned reference and let Ultralytics unwind itself."""
        self._trainer_mutex.lock()
        try:
            self._trainer = None
        finally:
            self._trainer_mutex.unlock()

    def _on_epoch(self, metrics: dict) -> None:
        self.epoch_update.emit(metrics)


def _read_saved_metrics(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        logger.exception("Failed to read saved training metrics: %s", path)
        return {}
    if not rows:
        return {}
    metrics: dict[str, float] = {}
    for key, value in rows[-1].items():
        clean_key = (
            key.strip()
            .replace("metrics/", "")
            .replace("(B)", "")
            .replace("(M)", "")
            .replace("(P)", "")
        )
        try:
            metrics[clean_key] = round(float(value), 4)
        except (TypeError, ValueError):
            continue
    return metrics


class BatchPredictWorker(QThread):
    """Runs batch inference in a background thread.

    Signals:
        progress(int, int): Emitted with (current, total) after each image.
        image_done(str, object, object): Emitted with (image_path, payload, image_size).
            payload is list[Annotation] for detect/segment/pose, or tuple[str, float] | None for classify.
        finished_ok(): Emitted when all images are processed (not emitted on cancel).
        error(str): Emitted if inference fails with error message.
    """

    progress = pyqtSignal(int, int)
    image_done = pyqtSignal(str, object, object)  # (path, payload, image_size)
    finished_ok = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(
        self,
        predictor,
        image_paths: list[Path],
        conf: float = 0.5,
        iou: float = 0.45,
        project_classes: list[str] | None = None,
        class_match_mode: str = "class_id",
        kpt_labels: list[str] | None = None,
        imgsz: int | None = None,
        device: str | None = None,
        task: str = "detect",
        parent=None,
    ):
        super().__init__(parent)
        self._predictor = predictor
        self._image_paths = image_paths
        self._conf = conf
        self._iou = iou
        self._project_classes = project_classes
        self._class_match_mode = class_match_mode
        self._kpt_labels = kpt_labels
        self._imgsz = imgsz
        self._device = device
        self._task = task
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        """Request cancellation of batch processing."""
        self._cancelled.set()

    def run(self) -> None:
        total = len(self._image_paths)
        try:
            if self._task == "classify":
                self._run_classify(total)
            else:
                self._run_detect_or_pose(total)
            if not self._cancelled.is_set():
                self.finished_ok.emit()
        except Exception as e:
            # Broad catch intentional: uncaught exceptions in QThread silently kill the thread
            logger.exception("Batch inference failed")
            self.error.emit(str(e))

    def _run_classify(self, total: int) -> None:
        predict_batch = getattr(self._predictor, "predict_classify_batch", None)
        if inspect.ismethod(predict_batch):
            for batch_start in range(0, total, _CLASSIFY_BATCH_SIZE):
                if self._cancelled.is_set():
                    break
                batch_paths = self._image_paths[batch_start:batch_start + _CLASSIFY_BATCH_SIZE]
                kwargs = {
                    "project_classes": self._project_classes,
                    "filter_to_project": False,
                }
                params = inspect.signature(predict_batch).parameters
                if "imgsz" in params:
                    kwargs["imgsz"] = self._imgsz
                if "device" in params:
                    kwargs["device"] = self._device
                payloads = predict_batch(batch_paths, **kwargs)
                if len(payloads) != len(batch_paths):
                    raise ValueError(
                        "predict_classify_batch returned "
                        f"{len(payloads)} payloads for {len(batch_paths)} images"
                    )
                for offset, (img_path, payload) in enumerate(
                    zip(batch_paths, payloads), start=batch_start,
                ):
                    if self._cancelled.is_set():
                        break
                    self.image_done.emit(str(img_path), payload, (0, 0))
                    self.progress.emit(offset + 1, total)
            return

        for i, img_path in enumerate(self._image_paths):
            if self._cancelled.is_set():
                break
            payload = self._predictor.predict_classify(
                img_path,
                project_classes=self._project_classes,
                filter_to_project=False,
                imgsz=self._imgsz,
                device=self._device,
            )
            self.image_done.emit(str(img_path), payload, (0, 0))
            self.progress.emit(i + 1, total)

    def _run_detect_or_pose(self, total: int) -> None:
        for i, img_path in enumerate(self._image_paths):
            if self._cancelled.is_set():
                break
            payload, img_size = self._predictor.predict_with_size(
                img_path,
                conf=self._conf,
                iou=self._iou,
                project_classes=self._project_classes,
                class_match_mode=self._class_match_mode,
                kpt_labels=self._kpt_labels,
                imgsz=self._imgsz,
                device=self._device,
            )
            self.image_done.emit(str(img_path), payload, img_size)
            self.progress.emit(i + 1, total)


class SinglePredictWorker(QThread):
    """Runs a single-image detection/pose inference off the UI thread.

    Used for slow backends (e.g. LocateAnything) whose ``predict`` blocks for
    seconds — running that on the Qt/X event loop can stall the desktop and, on
    a single-GPU machine where the X server shares the same card, crash it.
    YOLO single-image inference stays synchronous (fast, and existing tests
    depend on it); this worker is only used when the controller knows a slow
    backend is active.

    Signals:
        done(object): emitted with the predicted ``list[Annotation]``.
        error(str): emitted with a readable message if inference fails.

    Mirrors ``BatchPredictWorker._run_detect_or_pose`` for one image but calls
    ``predict`` (not ``predict_with_size``) to match the synchronous
    single-image path in ``ModelController.predict_single``.
    """

    done = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(
        self,
        predictor,
        image_path: Path,
        conf: float = 0.5,
        iou: float = 0.45,
        project_classes: list[str] | None = None,
        class_match_mode: str = "class_id",
        imgsz: int | None = None,
        device: str | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._predictor = predictor
        self._image_path = image_path
        self._conf = conf
        self._iou = iou
        self._project_classes = project_classes
        self._class_match_mode = class_match_mode
        self._imgsz = imgsz
        self._device = device

    def run(self) -> None:
        try:
            annotations = self._predictor.predict(
                self._image_path,
                conf=self._conf,
                iou=self._iou,
                project_classes=self._project_classes,
                class_match_mode=self._class_match_mode,
                imgsz=self._imgsz,
                device=self._device,
            )
            self.done.emit(annotations)
        except Exception as e:
            # Broad catch intentional: uncaught exceptions in QThread silently
            # kill the thread. Surface a readable message to the UI instead.
            logger.exception("Single-image inference failed")
            self.error.emit(str(e))


class LocateAnythingLoadWorker(QThread):
    """Loads the LocateAnything runtime (heavy 4-bit model) off the UI thread.

    Signals:
        progress(str): short status string for the UI during load.
        loaded(object): emitted with the ready LocateAnythingPredictor.
        error(str): emitted if loading fails.

    The heavy imports (torch/transformers) all happen inside
    ``backend.load_runtime()`` — this worker never imports them itself.
    """

    progress = pyqtSignal(str)
    loaded = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, backend, parent=None):
        super().__init__(parent)
        self._backend = backend

    def run(self) -> None:
        try:
            predictor = self._backend.load_runtime(progress_cb=self._on_progress)
            self.loaded.emit(predictor)
        except Exception as e:
            # Broad catch intentional: uncaught exceptions in QThread silently kill the thread
            logger.exception("LocateAnything load failed")
            self.error.emit(str(e))

    def _on_progress(self, message: str) -> None:
        self.progress.emit(message)
