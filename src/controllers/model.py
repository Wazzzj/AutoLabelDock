"""Model controller — load, delete, import, inference."""
from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path

from PyQt5.QtWidgets import QWidget, QFileDialog, QInputDialog, QMessageBox
from PyQt5.QtGui import QImage

from src.core.project import ProjectManager
from src.core.annotation import ImageAnnotation
from src.core.label_io import load_annotation, save_annotation
from src.core.model_structure import LayerInfo, ModelStructureError, parse_model_structure
from src.engine.backends import get_backend
from src.engine.backends.base import BackendError, PredictorProtocol
from src.engine.model_manager import ModelRegistry, ModelInfo
from src.utils.workers import BatchPredictWorker, SinglePredictWorker

logger = logging.getLogger(__name__)


class ModelController:
    """Handles model lifecycle: load, delete, import, auto-label."""

    def __init__(self, parent_widget: QWidget):
        self._parent = parent_widget
        self._predictor: PredictorProtocol | None = None
        self._current_model_info: ModelInfo | None = None
        self._registry: ModelRegistry | None = None
        self._project: ProjectManager | None = None
        self._batch_worker: BatchPredictWorker | None = None

    @property
    def predictor(self) -> PredictorProtocol | None:
        return self._predictor

    @property
    def registry(self) -> ModelRegistry | None:
        return self._registry

    @property
    def current_model_info(self) -> ModelInfo | None:
        return self._current_model_info

    def current_model_imgsz(self) -> int | None:
        """Return the loaded model's preferred square inference size, if known."""
        model_info = self._current_model_info
        if model_info is not None:
            train_params = model_info.train_params if isinstance(model_info.train_params, dict) else {}
            imgsz = self._coerce_imgsz(train_params.get("imgsz"))
            if imgsz is not None:
                return imgsz

        predictor = self._predictor
        if predictor is None:
            return None
        getter = getattr(predictor, "recommended_imgsz", None)
        if callable(getter):
            try:
                return self._coerce_imgsz(getter())
            except Exception:  # noqa: BLE001 - model metadata is best-effort only
                logger.debug("Failed to read model imgsz from predictor", exc_info=True)
        return None

    @classmethod
    def _coerce_imgsz(cls, value) -> int | None:
        if value is None:
            return None
        if hasattr(value, "tolist"):
            try:
                value = value.tolist()
            except Exception:  # noqa: BLE001 - metadata can be library-specific
                return None
        if hasattr(value, "item") and not isinstance(value, (list, tuple, dict)):
            try:
                value = value.item()
            except Exception:  # noqa: BLE001 - metadata can be library-specific
                return None
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
            try:
                value = float(value)
            except ValueError:
                return None
        if isinstance(value, (list, tuple)):
            candidates = [cls._coerce_imgsz(item) for item in value]
            candidates = [item for item in candidates if item is not None]
            return max(candidates) if candidates else None
        if isinstance(value, (int, float)):
            imgsz = int(value)
            if 32 <= imgsz <= 4096:
                return imgsz
        return None

    def set_context(self, project: ProjectManager, registry: ModelRegistry) -> None:
        self._project = project
        self._registry = registry

    def unload(self) -> None:
        """Drop the current predictor and free GPU memory.

        Called before enabling a VRAM-heavy backend (e.g. LocateAnything) so a
        resident YOLO model doesn't coexist with it. Invokes the predictor's
        optional ``release()`` hook, drops the reference, then runs gc.

        IMPORTANT (out-of-process LA): ``torch.cuda.empty_cache()`` is only
        invoked **when torch is already imported** in this process. A pure-LA
        session never imports torch in the GUI process (the model lives in a
        sidecar subprocess; ``LocateAnythingPredictor.release()`` just kills that
        subprocess, which frees its own CUDA context). Importing torch here would
        re-introduce the very CUDA-in-the-GUI-process conflict that the sidecar
        architecture exists to avoid, so we must skip it on the LA-only path.
        For the YOLO path torch is already resident, so the cache release still
        happens as before.
        """
        import sys

        predictor = self._predictor
        self._predictor = None
        self._current_model_info = None
        if predictor is not None:
            release = getattr(predictor, "release", None)
            if callable(release):
                try:
                    release()
                except Exception:  # noqa: BLE001 - teardown must be best-effort
                    logger.debug("Predictor release() failed", exc_info=True)
        import gc

        gc.collect()
        # Only touch CUDA if torch is ALREADY loaded — never import it here.
        if "torch" in sys.modules:
            try:
                torch = sys.modules["torch"]
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001 - best-effort
                pass

    def set_predictor(self, predictor: PredictorProtocol | None) -> None:
        """Inject a predictor directly (e.g. a LocateAnything runtime that is
        not backed by the ModelRegistry / file path). Existing auto-label flows
        read ``self._predictor`` and work unchanged."""
        self._predictor = predictor
        self._current_model_info = None

    def load_model(self, model_id: str) -> bool:
        """Load a model for inference. Returns True on success."""
        if not self._registry or not self._project:
            return False
        model_info = self._registry.get(model_id)
        if not model_info:
            return False
        try:
            model_path = Path(model_info.path)
            if not model_path.is_absolute():
                model_path = self._project.project_dir / model_path
            if not model_path.exists():
                QMessageBox.warning(self._parent, "错误", f"模型文件不存在: {model_path}")
                return False
            backend = get_backend(model_info.backend_id)
            self.unload()
            self._predictor = backend.load_predictor(model_path, model_info)
            self._current_model_info = model_info
            logger.info(
                "Loaded model: %s from %s via %s",
                model_info.name, model_path, backend.backend_id,
            )
            return True
        except (BackendError, RuntimeError, FileNotFoundError, OSError) as e:
            logger.error("Failed to load model: %s", e, exc_info=True)
            QMessageBox.warning(self._parent, "加载失败", f"模型加载失败: {e}")
            return False

    def inspect_model_structure(self, model_path: str | Path) -> list[LayerInfo]:
        """Parse a .pt file's layer hierarchy on CPU. Thin passthrough to
        ``core.model_structure.parse_model_structure``. Raises
        ``ModelStructureError`` on bad/corrupt/non-YOLO files — callers show it
        in a QMessageBox."""
        logger.info("Inspecting model structure: %s", model_path)
        return parse_model_structure(model_path)

    def inspect_registered_model(self, model_id: str) -> list[LayerInfo]:
        """Resolve a registered model's path (mirroring ``load_model``'s
        relative→absolute logic) then parse its structure."""
        if not self._registry or not self._project:
            raise ModelStructureError("当前没有打开的项目或模型注册表")
        model_info = self._registry.get(model_id)
        if not model_info:
            raise ModelStructureError("找不到指定的模型")
        model_path = Path(model_info.path)
        if not model_path.is_absolute():
            model_path = self._project.project_dir / model_path
        return self.inspect_model_structure(model_path)

    def delete_model(self, model_id: str) -> bool:
        """Delete model from registry (not file). Returns True if deleted."""
        if not self._registry:
            return False
        model_info = self._registry.get(model_id)
        if not model_info:
            return False
        reply = QMessageBox.question(
            self._parent, "确认删除",
            f"确定要删除模型 \"{model_info.name}\" 吗？\n（模型文件不会被删除）",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._registry.remove(model_id)
            self._registry.save()
            return True
        return False

    def rename_model(self, model_id: str) -> bool:
        """Rename a model's display name via dialog. Returns True if renamed."""
        if not self._registry:
            return False
        model_info = self._registry.get(model_id)
        if not model_info:
            return False
        new_name, ok = QInputDialog.getText(
            self._parent, "重命名模型", "请输入新的模型名称:",
            text=model_info.name,
        )
        if not ok or not new_name.strip() or new_name.strip() == model_info.name:
            return False
        self._registry.rename(model_id, new_name.strip())
        self._registry.save()
        logger.info("Renamed model %s -> %s", model_id, new_name.strip())
        return True

    def import_model(self) -> ModelInfo | None:
        """Import an external model file. Returns ModelInfo or None."""
        if not self._registry or not self._project:
            return None
        file_path, _ = QFileDialog.getOpenFileName(
            self._parent, "选择模型文件", "", "模型文件 (*.pt *.onnx);;PyTorch 模型 (*.pt);;ONNX 模型 (*.onnx);;所有文件 (*)"
        )
        if not file_path:
            return None
        name, ok = QInputDialog.getText(self._parent, "模型名称", "请输入模型名称:")
        if not ok or not name.strip():
            return None
        tasks = ["detect", "segment", "obb", "classify", "pose"]
        current_task = self._project.config.task_type if self._project else "detect"
        default_idx = tasks.index(current_task) if current_task in tasks else 0
        task, ok = QInputDialog.getItem(
            self._parent, "任务类型", "选择任务类型:", tasks, default_idx, False
        )
        if not ok:
            return None
        p = Path(file_path)
        try:
            rel = p.relative_to(self._project.project_dir)
            model_path = str(rel)
        except ValueError:
            model_path = str(p)
        backend = get_backend()
        probe = backend.probe()
        model_info = ModelInfo(
            name=name.strip(),
            path=model_path,
            task=task,
            base_model="imported",
            classes=self._project.config.classes,
            backend_id=backend.backend_id,
            model_format=backend.infer_model_format(p),
            backend_version=probe.version,
            backend_runtime=probe.runtime,
            backend_metadata=probe.metadata,
        )
        self._registry.register(model_info)
        self._registry.save()
        logger.info("Imported model: %s", name.strip())
        return model_info

    def export_model_onnx(self, model_id: str) -> ModelInfo | None:
        """Export a registered PyTorch model to ONNX and register the result."""
        if not self._registry or not self._project:
            return None
        model_info = self._registry.get(model_id)
        if model_info is None:
            return None
        model_path = Path(model_info.path)
        if not model_path.is_absolute():
            model_path = self._project.project_dir / model_path
        if not model_path.exists():
            QMessageBox.warning(self._parent, "导出失败", f"模型文件不存在: {model_path}")
            return None
        if model_path.suffix.lower() != ".pt":
            QMessageBox.information(self._parent, "无法导出", "当前仅支持从 .pt 模型导出 ONNX")
            return None
        try:
            from ultralytics import YOLO

            exported = YOLO(str(model_path), task=model_info.task).export(format="onnx")
            exported_path = Path(exported) if exported else model_path.with_suffix(".onnx")
            if not exported_path.is_absolute():
                exported_path = model_path.parent / exported_path
            if not exported_path.exists():
                raise FileNotFoundError(f"未找到导出的 ONNX 文件: {exported_path}")

            try:
                rel = exported_path.relative_to(self._project.project_dir)
                registry_path = str(rel)
            except ValueError:
                registry_path = str(exported_path)

            backend = get_backend(model_info.backend_id)
            probe = backend.probe()
            onnx_info = ModelInfo(
                name=f"{model_info.name}-onnx",
                path=registry_path,
                task=model_info.task,
                base_model=model_info.name,
                classes=list(model_info.classes),
                metrics=dict(model_info.metrics),
                epochs=model_info.epochs,
                dataset_size=model_info.dataset_size,
                train_params=dict(model_info.train_params),
                backend_id=backend.backend_id,
                model_format=backend.infer_model_format(exported_path),
                backend_version=probe.version,
                backend_runtime=probe.runtime,
                backend_metadata=probe.metadata,
            )
            self._registry.register(onnx_info)
            self._registry.save()
            logger.info("Exported ONNX model: %s -> %s", model_path, exported_path)
            return onnx_info
        except Exception as e:  # noqa: BLE001 - surface exporter failures to the user
            logger.error("ONNX export failed: %s", e, exc_info=True)
            QMessageBox.warning(self._parent, "导出失败", str(e))
            return None

    def export_model_onnx_file(self, model_id: str) -> Path | None:
        """Convert a registered .pt model to ONNX and save it where the user chooses."""
        if not self._registry or not self._project:
            return None
        model_info = self._registry.get(model_id)
        if model_info is None:
            return None
        model_path = Path(model_info.path)
        if not model_path.is_absolute():
            model_path = self._project.project_dir / model_path
        if not model_path.exists():
            QMessageBox.warning(self._parent, "导出失败", f"模型文件不存在: {model_path}")
            return None
        if model_path.suffix.lower() != ".pt":
            QMessageBox.information(self._parent, "无法导出", "当前仅支持从 .pt 模型导出 ONNX")
            return None

        default_path = self._project.project_dir / f"{model_info.name}.onnx"
        target, _ = QFileDialog.getSaveFileName(
            self._parent,
            "导出 ONNX 模型",
            str(default_path),
            "ONNX 模型 (*.onnx);;所有文件 (*)",
        )
        if not target:
            return None
        target_path = Path(target)
        if target_path.suffix.lower() != ".onnx":
            target_path = target_path.with_suffix(".onnx")

        try:
            from ultralytics import YOLO

            exported = YOLO(str(model_path), task=model_info.task).export(format="onnx")
            exported_path = Path(exported) if exported else model_path.with_suffix(".onnx")
            if not exported_path.is_absolute():
                exported_path = model_path.parent / exported_path
            if not exported_path.exists():
                raise FileNotFoundError(f"未找到导出的 ONNX 文件: {exported_path}")
            if exported_path.resolve() != target_path.resolve():
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(exported_path, target_path)
            logger.info("Exported ONNX model: %s -> %s", model_path, target_path)
            return target_path
        except Exception as e:  # noqa: BLE001 - surface exporter failures to the user
            logger.error("ONNX export failed: %s", e, exc_info=True)
            QMessageBox.warning(self._parent, "导出失败", str(e))
            return None

    def export_model_pt(self, model_id: str) -> Path | None:
        """Copy a registered .pt model to a user-selected destination."""
        if not self._registry or not self._project:
            return None
        model_info = self._registry.get(model_id)
        if model_info is None:
            return None
        model_path = Path(model_info.path)
        if not model_path.is_absolute():
            model_path = self._project.project_dir / model_path
        if not model_path.exists():
            QMessageBox.warning(self._parent, "导出失败", f"模型文件不存在: {model_path}")
            return None
        if model_path.suffix.lower() != ".pt":
            QMessageBox.information(self._parent, "无法导出", "当前仅支持导出已注册的 .pt 模型")
            return None

        default_path = self._project.project_dir / f"{model_info.name}.pt"
        target, _ = QFileDialog.getSaveFileName(
            self._parent,
            "导出 PT 模型",
            str(default_path),
            "PyTorch 模型 (*.pt);;所有文件 (*)",
        )
        if not target:
            return None
        target_path = Path(target)
        if target_path.suffix.lower() != ".pt":
            target_path = target_path.with_suffix(".pt")
        try:
            if model_path.resolve() == target_path.resolve():
                QMessageBox.information(self._parent, "无需导出", "源文件和目标文件相同")
                return None
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(model_path, target_path)
            logger.info("Exported PT model: %s -> %s", model_path, target_path)
            return target_path
        except OSError as e:
            logger.error("PT export failed: %s", e, exc_info=True)
            QMessageBox.warning(self._parent, "导出失败", str(e))
            return None

    def predict_single(
        self,
        img_path: Path,
        classes: list[str],
        conf: float = 0.5,
        iou: float = 0.45,
        class_match_mode: str = "class_id",
        imgsz: int | None = None,
        device: str | None = None,
        filter_to_project: bool = True,
    ) -> list | None:
        """执行单张图片预测。

        Returns:
            list: 推理成功，可能为空列表。
            None: 推理失败或模型未加载。
        """

        if not self._predictor:
            QMessageBox.information(
                self._parent,
                "无法自动标注",
                "请先在模型面板中加载一个模型。",
            )
            return None

        try:
            return self._predictor.predict(
                img_path,
                conf=conf,
                iou=iou,
                project_classes=classes,
                class_match_mode=class_match_mode,
                imgsz=imgsz,
                device=device,
                filter_to_project=filter_to_project,
            )

        except Exception as error:
            logger.error(
                "Auto-label failed: %s",
                error,
                exc_info=True,
            )

            error_message = str(error).strip() or type(error).__name__

            QMessageBox.warning(
                self._parent,
                "自动标注失败",
                "自动标注过程中发生错误：\n\n"
                f"{error_message}",
            )

            return None

    def predict_native_plot(
        self,
        img_path: Path,
        conf: float = 0.5,
        iou: float = 0.45,
        imgsz: int | None = None,
        device: str | None = None,
        retain_highest_confidence_roi: bool = False,
    ) -> tuple[QImage, str] | None:
        """Run the loaded Ultralytics model directly and return result.plot()."""
        if not self._predictor:
            QMessageBox.information(self._parent, "提示", "请先在模型面板中加载一个模型")
            return None
        predict_native = getattr(self._predictor, "predict_native", None)
        if not callable(predict_native):
            QMessageBox.warning(self._parent, "推理失败", "当前后端不支持 YOLO 原生推理显示")
            return None
        try:
            results = predict_native(
                img_path,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=device,
            )
            if not results:
                return None
            result = results[0]
            result = self._retain_highest_confidence_native_roi(
                result, retain_highest_confidence_roi,
            )
            plotted = self._plot_result_for_display(result)
            qimage = self._array_to_qimage(plotted)
            summary = self._native_result_summary(result)
            return qimage, summary
        except (RuntimeError, OSError, ValueError) as e:
            logger.error("Native YOLO predict failed: %s", e, exc_info=True)
            QMessageBox.warning(self._parent, "推理失败", str(e))
            return None

    def predict_native_save(
        self,
        image_paths: list[Path],
        output_parent: Path,
        conf: float = 0.5,
        iou: float = 0.45,
        imgsz: int | None = None,
        device: str | None = None,
        retain_highest_confidence_roi: bool = False,
    ) -> tuple[Path, int] | None:
        """Run native YOLO prediction for many images and save plotted results."""
        if not self._predictor:
            QMessageBox.information(self._parent, "提示", "请先在模型面板中加载一个模型")
            return None
        if not image_paths:
            return None
        predict_native = getattr(self._predictor, "predict_native", None)
        if not callable(predict_native):
            QMessageBox.warning(self._parent, "推理失败", "当前后端不支持 YOLO 原生批量保存")
            return None
        output_parent.mkdir(parents=True, exist_ok=True)
        model_name = (
            self._current_model_info.name if self._current_model_info is not None else "model"
        )
        safe_model_name = "".join(
            ch if ch.isalnum() or ch in ("-", "_") else "_"
            for ch in model_name.strip()
        ) or "model"
        run_name = f"{safe_model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            results = []
            result_dir = output_parent / run_name
            result_dir.mkdir(parents=True, exist_ok=True)
            used_names: set[str] = set()
            for image_path in image_paths:
                image_results = predict_native(
                    image_path,
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    device=device,
                )
                if image_results:
                    results.extend(image_results)
                    for result in image_results:
                        result = self._retain_highest_confidence_native_roi(
                            result, retain_highest_confidence_roi,
                        )
                        output_path = self._unique_prediction_path(
                            result_dir,
                            Path(getattr(result, "path", None) or image_path).name,
                            used_names,
                        )
                        qimage = self._array_to_qimage(
                            self._plot_result_for_display(result)
                        )
                        qimage.save(str(output_path))
            return result_dir, len(results or [])
        except Exception as e:  # noqa: BLE001 - backend runtimes raise library-specific errors
            logger.error("Native YOLO batch predict failed: %s", e, exc_info=True)
            QMessageBox.warning(self._parent, "推理失败", str(e))
            return None

    @staticmethod
    def _plot_result_for_display(result):
        """Render native YOLO output using Ultralytics' default visualization."""
        return result.plot()

    @staticmethod
    def _retain_highest_confidence_native_roi(result, enabled: bool):
        """Slice an Ultralytics result to its highest-confidence spatial ROI."""
        if not enabled:
            return result
        obb = getattr(result, "obb", None)
        boxes = obb if obb is not None and len(obb) > 0 else getattr(result, "boxes", None)
        confidences = getattr(boxes, "conf", None) if boxes is not None else None
        if confidences is None or len(confidences) <= 1:
            return result
        scores = confidences.tolist() if hasattr(confidences, "tolist") else list(confidences)
        scores = [float(score.item()) if hasattr(score, "item") else float(score) for score in scores]
        highest_index = max(range(len(scores)), key=scores.__getitem__)
        # Results slicing keeps boxes/OBB, masks and keypoints aligned.
        return result[highest_index:highest_index + 1]

    @staticmethod
    def _unique_prediction_path(
        result_dir: Path,
        filename: str,
        used_names: set[str],
    ) -> Path:
        """Return a non-conflicting output path for flattened directory predictions."""
        candidate = Path(filename)
        stem = candidate.stem or "prediction"
        suffix = candidate.suffix or ".jpg"
        name = f"{stem}{suffix}"
        index = 1
        while name.casefold() in used_names or (result_dir / name).exists():
            name = f"{stem}_{index}{suffix}"
            index += 1
        used_names.add(name.casefold())
        return result_dir / name

    @staticmethod
    def _array_to_qimage(image_array) -> QImage:
        """Convert Ultralytics result.plot() numpy image to a detached QImage."""
        if image_array is None:
            return QImage()
        if getattr(image_array, "ndim", 0) == 2:
            h, w = image_array.shape
            return QImage(
                image_array.data, w, h, w, QImage.Format_Grayscale8,
            ).copy()
        h, w, channels = image_array.shape
        if channels >= 3:
            # Ultralytics plot returns BGR ndarray by default; Qt expects RGB.
            rgb = image_array[:, :, :3][:, :, ::-1].copy()
            return QImage(
                rgb.data, w, h, 3 * w, QImage.Format_RGB888,
            ).copy()
        raise ValueError("Unsupported prediction image format")

    @staticmethod
    def _native_result_summary(result) -> str:
        boxes = getattr(result, "boxes", None)
        if boxes is not None and getattr(boxes, "cls", None) is not None:
            return f"YOLO 原生推理: {len(boxes.cls)} 个目标"
        probs = getattr(result, "probs", None)
        if probs is not None:
            names = getattr(result, "names", None) or {}
            cls_id = int(probs.top1)
            class_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
            conf = float(probs.top1conf.item())
            return f"YOLO 原生分类: {class_name} ({conf:.2f})"
        return "YOLO 原生推理完成"

    def create_single_predict_worker(
        self,
        img_path: Path,
        classes: list[str],
        conf: float = 0.5,
        iou: float = 0.45,
        class_match_mode: str = "class_id",
        imgsz: int | None = None,
        device: str | None = None,
    ) -> SinglePredictWorker | None:
        """Build a background worker for single-image detection/pose/OBB inference.

        Used for slow backends (e.g. LocateAnything) so ``predict`` does not
        block the Qt/X event loop. Returns ``None`` (and warns the user) when no
        predictor is loaded — mirroring ``predict_single``'s guard so the
        caller can bail without special-casing. The caller owns connecting the
        worker's ``done`` / ``error`` signals and starting it.
        """
        if not self._predictor:
            QMessageBox.information(self._parent, "提示", "请先在模型面板中加载一个模型")
            return None
        return SinglePredictWorker(
            predictor=self._predictor,
            image_path=img_path,
            conf=conf,
            iou=iou,
            project_classes=classes,
            class_match_mode=class_match_mode,
            imgsz=imgsz,
            device=device,
        )

    def predict_single_classify(
        self,
        img_path: Path,
        classes: list[str],
        filter_to_project: bool = False,
        imgsz: int | None = None,
        device: str | None = None,
    ) -> tuple[str, float] | None:
        """Run classify inference, returning (class_name, conf) or None.

        Returns the raw model class name without filtering — the caller
        (MainWindow) routes the result through ProjectController.register_auto_class.
        """
        if not self._predictor:
            QMessageBox.information(self._parent, "提示", "请先在模型面板中加载一个模型")
            return None
        try:
            return self._predictor.predict_classify(
                img_path,
                project_classes=classes,
                filter_to_project=filter_to_project,
                imgsz=imgsz,
                device=device,
            )
        except (RuntimeError, OSError, ValueError) as e:
            logger.error("Classify failed: %s", e, exc_info=True)
            QMessageBox.warning(self._parent, "自动标注失败", str(e))
            return None
