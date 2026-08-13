"""Project controller — create, open, export, class management."""
from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from PyQt5.QtWidgets import QWidget, QFileDialog, QMessageBox, QListWidgetItem

from src.core.annotation import ImageAnnotation
from src.core.config import AppConfig
from src.core.project import IMAGE_EXTENSIONS, ProjectManager
from src.core.label_io import load_annotation, save_annotation
from src.core.backup import BackupManager
from src.ui.dialogs import (
    NewProjectDialog,
    ExportDialog,
    ClassManagerDialog,
    ImportDialog,
)

logger = logging.getLogger(__name__)


_IMAGENET_ID_RE = re.compile(r"^n\d{8}$")
_MAX_CLASS_NAME_LEN = 64


@dataclass
class RegistrationResult:
    """Outcome of ProjectController.register_auto_class."""
    action: Literal[
        "registered",
        "existing",
        "rejected_blacklist",
        "rejected_disabled",
        "rejected_invalid",
    ]
    applied_name: str | None
    reason: str


@dataclass
class ClassPreviewItem:
    """A model class diffed against the current project's class list."""
    model_name: str
    is_blacklisted: bool
    default_checked: bool


class ProjectController:
    """Handles project lifecycle: create, open, export, class management."""

    def __init__(self, app_config: AppConfig, config_path: Path, parent_widget: QWidget):
        self._app_config = app_config
        self._config_path = config_path
        self._parent = parent_widget
        self._project: ProjectManager | None = None
        self._backup_mgr: BackupManager | None = None

    @property
    def project(self) -> ProjectManager | None:
        return self._project

    @property
    def backup_manager(self) -> BackupManager | None:
        return self._backup_mgr

    def create_project(self) -> ProjectManager | None:
        """Show new project dialog and create. Returns ProjectManager or None."""
        dlg = NewProjectDialog(self._parent)
        if not dlg.exec_():
            return None
        name, proj_dir, image_dir, classes, task_type = dlg.get_values()
        if not name or not proj_dir:
            return None
        try:
            pm = ProjectManager.create(
                proj_dir, name,
                image_dir=image_dir or "images",
                classes=classes or None,
                task_type=task_type,
                create_dirs=True,
            )
            moved_count, moved_label_count = self._move_root_images_to_image_dir(pm)
            if moved_count:
                logger.info("Moved %d root images into %s", moved_count, pm.image_root())
            if moved_label_count:
                logger.info(
                    "Moved %d root label JSONs into %s",
                    moved_label_count,
                    pm.config.label_dir,
                )
            imported_count, imported_format = self.import_discovered_obb_sidecars(pm)
            if imported_count:
                logger.info(
                    "Auto-imported %d OBB annotations from %s sidecars",
                    imported_count,
                    imported_format,
                )
            self._project = pm
            self._add_recent(pm)
            return pm
        except (OSError, ValueError) as e:
            logger.error("Failed to create project: %s", e, exc_info=True)
            QMessageBox.warning(self._parent, "错误", f"创建项目失败: {e}")
            return None

    def _move_root_images_to_image_dir(self, project: ProjectManager) -> tuple[int, int]:
        project_dir = project.project_dir
        image_dir = project.image_root()
        image_dir.mkdir(parents=True, exist_ok=True)

        try:
            image_dir_resolved = image_dir.resolve()
        except OSError:
            image_dir_resolved = image_dir

        moved_count = 0
        moved_label_count = 0
        for src in sorted(project_dir.iterdir()):
            if not src.is_file() or src.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                if src.parent.resolve() == image_dir_resolved:
                    continue
            except OSError:
                pass
            dst = self._unique_destination(image_dir / src.name)
            shutil.move(str(src), str(dst))
            moved_count += 1

            label_src = src.with_suffix(".json")
            if not label_src.exists():
                continue
            label_dst = project.label_path_for(dst)
            label_dst = self._unique_destination(label_dst)
            label_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(label_src), str(label_dst))
            moved_label_count += 1
        return moved_count, moved_label_count

    @staticmethod
    def _unique_destination(path: Path) -> Path:
        if not path.exists():
            return path
        stem = path.stem
        suffix = path.suffix
        parent = path.parent
        index = 1
        while True:
            candidate = parent / f"{stem}_{index}{suffix}"
            if not candidate.exists():
                return candidate
            index += 1

    def open_project_dialog(self) -> ProjectManager | None:
        """Show file dialog and open project. Returns ProjectManager or None."""
        path, _ = QFileDialog.getOpenFileName(
            self._parent, "打开项目", "", "项目文件 (project.json)"
        )
        if not path:
            return None
        return self.open_project(Path(path).parent)

    def open_project(self, project_dir: Path) -> ProjectManager | None:
        """Open a project from directory. Returns ProjectManager or None."""
        try:
            pm = ProjectManager.open(project_dir)
            imported_count, imported_format = self.import_discovered_obb_sidecars(pm)
            if imported_count:
                logger.info(
                    "Auto-imported %d OBB annotations from %s sidecars",
                    imported_count,
                    imported_format,
                )
            self._project = pm
            self._add_recent(pm)
            return pm
        except (FileNotFoundError, json.JSONDecodeError, KeyError, OSError, ValueError) as e:
            logger.error("Failed to open project: %s", e, exc_info=True)
            QMessageBox.warning(self._parent, "打开失败", f"无法打开项目: {e}")
            return None

    def open_recent(self, item: QListWidgetItem) -> ProjectManager | None:
        """Open a project from the recent list. Returns ProjectManager or None."""
        project_dir = Path(item.text())
        pm = self.open_project(project_dir)
        if pm is None:
            # Remove invalid entry
            self._app_config.recent_projects = [
                p for p in self._app_config.recent_projects if p != item.text()
            ]
            self._app_config.save(self._config_path)
        return pm

    def export(self, project: ProjectManager) -> Path | None:
        """Show export dialog and export selected data version with images."""
        dlg = ExportDialog(
            self._parent,
            data_versions=project.list_data_folders(),
            active_data_version=project.config.active_data_folder,
        )
        if not dlg.exec_():
            return None

        fmt, out_dir, only_confirmed, data_version = dlg.get_values()
        if not out_dir:
            return None
        output_root = Path(out_dir)
        try:
            # Auto-backup before export
            self.create_backup()
            from src.core.formats import get_export_registry

            registry = get_export_registry()
            info = registry.get(fmt)
            if info is None:
                raise ValueError(f"Unknown export format: {fmt}")

            image_count, annotations, source_annotations = self._prepare_format_export(
                project,
                output_root,
                data_version,
                fmt=fmt,
                only_confirmed=only_confirmed,
            )
            labels_root = output_root / "labels"
            labels_root.mkdir(parents=True, exist_ok=True)
            export_output = output_root if fmt == "YOLO" else labels_root
            export_annotations = source_annotations if fmt in {"ImageFolder", "CSV"} else annotations
            registry.export(
                fmt,
                export_annotations,
                export_output,
                classes=project.config.classes,
                only_confirmed=only_confirmed,
                source_image_dir=project.image_root(),
                task_type=project.config.task_type,
            )
            version_text = data_version or "全部数据"
            QMessageBox.information(
                self._parent,
                "导出完成",
                f"已导出 {fmt}（{version_text}），图片 {image_count} 张。\n{output_root}",
            )
            logger.info("Exported %s (%s) to %s", fmt, version_text, output_root)
            return output_root
        except (OSError, ValueError, KeyError) as e:
            logger.error("Export failed: %s", e, exc_info=True)
            QMessageBox.warning(self._parent, "导出失败", str(e))
            raise

    @staticmethod
    def import_discovered_obb_sidecars(project: ProjectManager) -> tuple[int, str]:
        """Best-effort wrapper so malformed optional sidecars do not block a project."""
        try:
            return ProjectController._import_discovered_obb_sidecars(project)
        except (OSError, ValueError, KeyError) as exc:
            logger.warning("Failed to auto-import OBB sidecars: %s", exc, exc_info=True)
            return 0, ""

    @staticmethod
    def _import_discovered_obb_sidecars(project: ProjectManager) -> tuple[int, str]:
        """Convert matching project-local OBB sidecars into internal JSON labels.

        Existing internal annotations are never overwritten. When XML and YOLO
        OBB files coexist, the source matching more currently unlabeled images
        wins; XML wins ties because it commonly contains richer object metadata.
        """
        if project.config.task_type != "obb":
            return 0, ""

        missing_by_stem = {
            image_path.stem.casefold(): image_path
            for image_path in project.list_images()
            if load_annotation(project.label_path_for(image_path)) is None
        }
        if not missing_by_stem:
            return 0, ""

        source_dirs: list[Path] = []
        for candidate in (
            project.project_dir,
            project.image_root(),
            project.project_dir / "labels",
            project.project_dir / "label",
            project.project_dir / "annotations",
        ):
            if candidate.exists() and candidate.is_dir() and candidate not in source_dirs:
                source_dirs.append(candidate)

        candidates: list[tuple[int, int, str, Path]] = []
        for source_dir in source_dirs:
            xml_stems = {path.stem.casefold() for path in source_dir.glob("*.xml")}
            txt_stems = {
                path.stem.casefold()
                for path in source_dir.glob("*.txt")
                if path.name.casefold() != "classes.txt"
            }
            xml_matches = len(xml_stems & missing_by_stem.keys())
            txt_matches = len(txt_stems & missing_by_stem.keys())
            if xml_matches:
                candidates.append((xml_matches, 1, "VOC-OBB", source_dir))
            if txt_matches:
                candidates.append((txt_matches, 0, "YOLO", source_dir))

        if not candidates:
            return 0, ""

        _matches, _xml_priority, fmt, source_dir = max(
            candidates,
            key=lambda item: (item[0], item[1]),
        )
        if fmt == "VOC-OBB":
            from src.core.formats.voc_obb import import_voc_obb

            imported = import_voc_obb(source_dir, classes=project.config.classes or None)
        else:
            from src.core.formats.yolo import import_yolo_auto

            imported = import_yolo_auto(
                source_dir,
                classes=None,
                task_type="obb",
            )

        selected: list[tuple[Path, ImageAnnotation]] = []
        new_classes: list[str] = []
        known_classes = set(project.config.classes)
        for image_annotation in imported:
            image_path = missing_by_stem.get(Path(image_annotation.image_path).stem.casefold())
            if image_path is None:
                continue
            selected.append((image_path, image_annotation))
            for annotation in image_annotation.annotations:
                if annotation.class_name and annotation.class_name not in known_classes:
                    known_classes.add(annotation.class_name)
                    new_classes.append(annotation.class_name)

        if new_classes:
            project.config.classes.extend(new_classes)
            project.save()

        imported_count = 0
        for image_path, image_annotation in selected:
            for annotation in image_annotation.annotations:
                class_id = project.config.get_class_id(annotation.class_name)
                if class_id >= 0:
                    annotation.class_id = class_id
            image_annotation.image_path = image_path.name
            save_annotation(image_annotation, project.label_path_for(image_path))
            imported_count += 1

        return imported_count, fmt

    def _prepare_format_export(
        self,
        project: ProjectManager,
        output_root: Path,
        data_version: str,
        fmt: str = "",
        only_confirmed: bool = False,
    ) -> tuple[int, list, list]:
        image_root = project.image_root()
        scan_root = image_root / Path(data_version) if data_version else image_root
        images_out = output_root / "images"
        images_out.mkdir(parents=True, exist_ok=True)

        image_count = 0
        annotations = []
        source_annotations = []

        for image_path in project.list_images(data_folder=data_version):
            try:
                rel = image_path.resolve().relative_to(scan_root.resolve())
            except (OSError, ValueError):
                rel = Path(image_path.name)

            label_path = project.label_path_for(image_path)
            ia = load_annotation(label_path)
            if only_confirmed and not self._has_format_exportable_annotations(
                ia,
                fmt,
                only_confirmed,
                project.config.task_type,
            ):
                continue

            image_dst = images_out / rel
            if self._copy_export_file(image_path, image_dst):
                image_count += 1

            if ia is None:
                continue
            exported_image_path = (Path("images") / rel).as_posix()
            annotations.append(replace(ia, image_path=exported_image_path))
            source_annotations.append(replace(ia, image_path=str(image_path)))

        return image_count, annotations, source_annotations

    @staticmethod
    def _has_format_exportable_annotations(
        ia,
        fmt: str,
        only_confirmed: bool,
        task_type: str = "detect",
    ) -> bool:
        if ia is None:
            return False
        if fmt in {"ImageFolder", "CSV"}:
            return bool(ia.image_tags) and (not only_confirmed or ia.image_tags_confirmed)
        for ann in ia.annotations:
            if only_confirmed and not ann.confirmed:
                continue
            if fmt == "YOLO":
                if task_type == "segment":
                    if len(ann.polygon) >= 3 or ann.bbox is not None:
                        return True
                    continue
                if task_type == "obb":
                    if len(ann.polygon) == 4 or (ann.bbox is not None and not ann.polygon):
                        return True
                    continue
                if ann.bbox is not None:
                    return True
                continue
            if fmt == "COCO" and ann.bbox is not None:
                return True
            if fmt == "iSAT" and (len(ann.polygon) >= 3 or ann.bbox is not None):
                return True
            if fmt == "labelme" and (ann.bbox is not None or ann.polygon or ann.keypoints):
                return True
        return False

    @staticmethod
    def _copy_export_file(src: Path, dst: Path) -> bool:
        try:
            if src.resolve() == dst.resolve():
                return False
        except OSError:
            pass
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True

    def import_annotations(self, project: ProjectManager) -> int | None:
        """Show import dialog and import annotations.

        Returns number of images imported, or None if cancelled/failed.
        New class names found in imported data are auto-added to project classes.
        """
        dlg = ImportDialog(self._parent)
        if not dlg.exec_():
            return None
        fmt, path, conflict_mode = dlg.get_values()
        if not fmt or not path:
            return None

        try:
            # Auto-backup before import
            self.create_backup()

            from src.core.formats import get_import_registry
            info = get_import_registry().get(fmt)
            if info is None:
                QMessageBox.warning(self._parent, "导入失败", f"未知的导入格式: {fmt}")
                return None

            if info.is_full_import:
                # Importer manages images + labels directly (e.g. ImageFolder)
                result = info.import_fn(Path(path), project)
                imported_count = result.get("imported", 0)
                skipped_count = result.get("skipped", 0)
                new_classes = result.get("classes", [])
                msg = f"成功导入 {imported_count} 张图片"
                if skipped_count:
                    msg += f"，跳过 {skipped_count} 张"
                if new_classes:
                    msg += f"\n类别: {', '.join(new_classes)}"
                QMessageBox.information(self._parent, "导入完成", msg)
                logger.info("Imported %s from %s: %d ok, %d skipped",
                            fmt, path, imported_count, skipped_count)
                return imported_count

            imported = self._invoke_importer(
                fmt,
                path,
                project.config.classes,
                task_type=project.config.task_type,
            )
            if not imported:
                QMessageBox.information(self._parent, "提示", "未找到可导入的标注")
                return 0

            # Build lookup of existing project images by stem
            image_by_stem: dict[str, Path] = {
                p.stem: p for p in project.list_images()
            }

            # Collect new classes (preserve order, dedupe)
            existing_classes = set(project.config.classes)
            new_classes: list[str] = []
            for ia in imported:
                for ann in ia.annotations:
                    if ann.class_name and ann.class_name not in existing_classes:
                        existing_classes.add(ann.class_name)
                        new_classes.append(ann.class_name)
                for tag in ia.image_tags:
                    if tag and tag not in existing_classes:
                        existing_classes.add(tag)
                        new_classes.append(tag)
            if new_classes:
                project.config.classes.extend(new_classes)
                project.save()

            # Re-resolve class_id against (possibly updated) project classes
            imported_count = 0
            skipped_count = 0
            for ia in imported:
                stem = Path(ia.image_path).stem
                matched = image_by_stem.get(stem)
                if matched is None:
                    skipped_count += 1
                    continue

                label_path = project.label_path_for(matched)
                existing = load_annotation(label_path)

                # Re-map class_id to current project classes
                for ann in ia.annotations:
                    cid = project.config.get_class_id(ann.class_name)
                    if cid >= 0:
                        ann.class_id = cid

                # Determine image_size (prefer existing, else imported, else load from disk)
                img_size = (0, 0)
                if existing and existing.image_size != (0, 0):
                    img_size = existing.image_size
                elif ia.image_size != (0, 0):
                    img_size = ia.image_size
                else:
                    from src.utils.image import get_image_size
                    try:
                        img_size = get_image_size(matched)
                    except (OSError, ValueError):
                        img_size = (0, 0)

                # Apply conflict resolution
                if conflict_mode == "skip" and existing and existing.annotations:
                    skipped_count += 1
                    continue
                elif conflict_mode == "overwrite" or existing is None:
                    new_ia = ia
                    new_ia.image_path = matched.name
                    new_ia.image_size = img_size
                elif conflict_mode == "merge":
                    existing.annotations.extend(ia.annotations)
                    if existing.image_size == (0, 0):
                        existing.image_size = img_size
                    if ia.image_tags and not existing.image_tags:
                        existing.image_tags = list(ia.image_tags)
                        existing.image_tags_confirmed = ia.image_tags_confirmed
                        existing.image_tags_source = ia.image_tags_source
                    new_ia = existing
                else:
                    # Fallback: treat as overwrite when no existing
                    new_ia = ia
                    new_ia.image_path = matched.name
                    new_ia.image_size = img_size

                save_annotation(new_ia, label_path)
                imported_count += 1

            msg = f"成功导入 {imported_count} 个图片的标注"
            if skipped_count:
                msg += f"，跳过 {skipped_count} 个"
            if new_classes:
                msg += f"\n自动添加了 {len(new_classes)} 个新类别: {', '.join(new_classes)}"
            QMessageBox.information(self._parent, "导入完成", msg)
            logger.info("Imported %s from %s: %d ok, %d skipped", fmt, path, imported_count, skipped_count)
            return imported_count

        except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
            logger.error("Import failed: %s", e, exc_info=True)
            QMessageBox.warning(self._parent, "导入失败", str(e))
            return None

    def _invoke_importer(
        self,
        fmt: str,
        path: str,
        classes: list[str],
        task_type: str | None = None,
    ) -> list:
        """Invoke the appropriate importer. Each importer has a different signature."""
        from src.core.formats import get_import_registry
        from src.core.formats.yolo import import_yolo_auto
        from src.core.formats.coco import import_coco
        from src.core.formats.labelme import import_labelme
        from src.core.formats.isat import import_isat
        from src.core.formats.voc_obb import import_voc_obb

        registry = get_import_registry()
        info = registry.get(fmt)
        if info is None:
            raise ValueError(f"未知的导入格式: {fmt}")

        p = Path(path)
        if fmt == "YOLO":
            has_external_metadata = any(
                candidate.exists()
                for candidate in [
                    p / "data.yaml",
                    p.parent / "data.yaml",
                    p / "classes.txt",
                    p.parent / "classes.txt",
                ]
            )
            return import_yolo_auto(
                p,
                classes=None if has_external_metadata else (classes or None),
                task_type=task_type,
            )
        elif fmt == "COCO":
            return import_coco(p, classes=classes or None)
        elif fmt == "labelme":
            return import_labelme(p)
        elif fmt == "iSAT":
            return import_isat(p)
        elif fmt == "VOC-OBB":
            return import_voc_obb(p, classes=classes or None)
        else:
            raise ValueError(f"未实现的导入格式: {fmt}")

    def manage_classes(self, project: ProjectManager) -> bool:
        """Show class manager dialog. Returns True if classes were changed."""
        dlg = ClassManagerDialog(
            project.config.classes,
            project.config.class_colors,
            self._parent,
        )
        if dlg.exec_():
            self.create_backup()  # Auto-backup before class changes
            project.config.classes = dlg.get_classes()
            project.save()
            return True
        return False

    def register_auto_class(
        self, raw_name: str, *, force: bool = False,
    ) -> RegistrationResult:
        """Register a class produced by an auto-label prediction.

        - ``force=True`` skips the project-level ``auto_register_classes`` gate
          (used by the batch pre-dialog where the user has explicitly approved
          a list of new classes). It still applies name validation.
        - Idempotent: returns ``"existing"`` when the class is already known.
        - On success, persists ``project.json`` immediately so worker threads
          and subsequent calls see the updated class list.
        """
        if self._project is None:
            return RegistrationResult(
                action="rejected_invalid",
                applied_name=None,
                reason="no project loaded",
            )
        ok, name, reason_kind = self._validate_class_name(raw_name)
        if not ok:
            if reason_kind == "rejected_blacklist":
                return RegistrationResult(
                    action="rejected_blacklist",
                    applied_name=None,
                    reason=f"模型类名 '{name}' 疑似 ImageNet ID，已跳过",
                )
            return RegistrationResult(
                action="rejected_invalid",
                applied_name=None,
                reason="模型类名为空或过长",
            )
        if name in self._project.config.classes:
            return RegistrationResult(
                action="existing",
                applied_name=name,
                reason=f"类别 '{name}' 已存在",
            )
        if not force and not self._project.config.auto_register_classes:
            return RegistrationResult(
                action="rejected_disabled",
                applied_name=None,
                reason="未开启自动登记",
            )
        self._project.add_class(name)
        self._project.save()
        logger.info("Auto-registered class: %s", name)
        return RegistrationResult(
            action="registered",
            applied_name=name,
            reason=f"已新增类别 '{name}'",
        )

    @staticmethod
    def _validate_class_name(raw: str) -> tuple[bool, str, str | None]:
        name = (raw or "").strip()
        if not name or len(name) > _MAX_CLASS_NAME_LEN:
            return False, name, "rejected_invalid"
        if _IMAGENET_ID_RE.match(name):
            return False, name, "rejected_blacklist"
        return True, name, None

    def preview_model_classes(self, predictor) -> list[ClassPreviewItem]:
        """Diff predictor.model.names against project.classes.

        Returns one ClassPreviewItem per model class that is *not* in
        ``project.classes``. Already-registered classes are excluded — the
        dialog should only ask the user about new ones.
        """
        if predictor is None or self._project is None:
            return []
        model = getattr(predictor, "model", None)
        names = getattr(model, "names", None) if model is not None else None
        if not names:
            return []
        if isinstance(names, dict):
            iterable = names.values()
        elif isinstance(names, (list, tuple)):
            iterable = names
        else:
            return []
        existing = set(self._project.config.classes)
        items: list[ClassPreviewItem] = []
        seen: set[str] = set()
        for raw in iterable:
            ok, name, reason = self._validate_class_name(raw)
            if not ok and reason == "rejected_invalid":
                continue
            if name in existing or name in seen:
                continue
            seen.add(name)
            is_black = reason == "rejected_blacklist"
            items.append(
                ClassPreviewItem(
                    model_name=name,
                    is_blacklisted=is_black,
                    default_checked=not is_black,
                )
            )
        return items

    def _add_recent(self, pm: ProjectManager) -> None:
        self._app_config.add_recent_project(str(pm.project_dir))
        self._app_config.save(self._config_path)
        self._backup_mgr = BackupManager(pm.project_dir)

    def create_backup(self) -> Path | None:
        """Create a manual backup of the current project. Returns backup path."""
        if self._backup_mgr and self._project:
            return self._backup_mgr.create_backup(self._project.config.label_dir)
        return None

    def list_backups(self) -> list[dict]:
        """List available backups for the current project."""
        if self._backup_mgr:
            return self._backup_mgr.list_backups()
        return []

    def restore_backup(self, backup_name: str) -> bool:
        """Restore a backup by name. Returns True on success."""
        if self._backup_mgr and self._project:
            return self._backup_mgr.restore_backup(backup_name, self._project.config.label_dir)
        return False
