"""Project configuration and management."""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.utils.colors import assign_color

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
IGNORED_IMAGE_TREE_DIR_NAMES = {"model_predictions"}


@dataclass
class ProjectConfig:
    """Project configuration stored in project.json."""

    name: str
    image_dir: str  # relative to project dir
    label_dir: str  # relative to project dir
    classes: list[str]
    class_colors: dict[str, str] = field(default_factory=dict)
    keypoint_templates: dict[str, dict] = field(default_factory=dict)
    default_model: str = ""
    auto_label_conf: float = 0.5
    auto_label_iou: float = 0.45
    created_at: str = ""
    version: str = "1.0"
    task_type: str = "detect"  # "detect" | "segment" | "pose" | "classify"
    auto_register_classes: bool = True
    # Project-level registry of known user tags. Per-image tag selections
    # live on ImageAnnotation.tags; this list is just the autocomplete source.
    tags: list[str] = field(default_factory=list)
    # Relative folder under image_dir used as the active data version/subset.
    # Empty means "all images under image_dir".
    active_data_folder: str = ""
    # Explicit registry of user-created data versions. Kept alongside the
    # filesystem scan so empty versions are visible immediately and persistently.
    data_folders: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "image_dir": self.image_dir,
            "label_dir": self.label_dir,
            "classes": self.classes,
            "class_colors": self.class_colors,
            "keypoint_templates": self.keypoint_templates,
            "default_model": self.default_model,
            "auto_label_conf": self.auto_label_conf,
            "auto_label_iou": self.auto_label_iou,
            "created_at": self.created_at,
            "version": self.version,
            "task_type": self.task_type,
            "auto_register_classes": self.auto_register_classes,
            "tags": self.tags,
            "active_data_folder": self.active_data_folder,
            "data_folders": self.data_folders,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ProjectConfig:
        return cls(
            name=d["name"],
            image_dir=d["image_dir"],
            label_dir=d["label_dir"],
            classes=d["classes"],
            class_colors=d.get("class_colors", {}),
            keypoint_templates=d.get("keypoint_templates", {}),
            default_model=d.get("default_model", ""),
            auto_label_conf=d.get("auto_label_conf", 0.5),
            auto_label_iou=d.get("auto_label_iou", 0.45),
            created_at=d.get("created_at", ""),
            version=d.get("version", "1.0"),
            task_type=d.get("task_type", "detect"),
            auto_register_classes=d.get("auto_register_classes", True),
            tags=list(d.get("tags", [])),
            active_data_folder=d.get("active_data_folder", ""),
            data_folders=list(d.get("data_folders", [])),
        )

    def get_class_color(self, class_name: str) -> str:
        """Get color for a class. Uses custom color if set, otherwise auto-assigns from palette."""
        if class_name in self.class_colors:
            return self.class_colors[class_name]
        idx = self.classes.index(class_name) if class_name in self.classes else 0
        return assign_color(idx)

    def get_class_id(self, class_name: str) -> int:
        """Get class index. Returns -1 if not found."""
        try:
            return self.classes.index(class_name)
        except ValueError:
            return -1


class ProjectManager:
    """Manages a project directory and its configuration."""

    def __init__(self, project_dir: Path, config: ProjectConfig):
        self.project_dir = project_dir
        self.config = config

    @classmethod
    def create(
        cls,
        project_dir: Path | str,
        name: str,
        image_dir: str = "images",
        classes: list[str] | None = None,
        task_type: str = "detect",
        create_dirs: bool = True,
    ) -> ProjectManager:
        """Create a new project.

        Args:
            project_dir: Path where the project will be created.
            name: Project name.
            image_dir: Either a relative subdir name (created inside project_dir)
                       or an absolute path to an existing image directory.
            classes: Initial class list.
            task_type: Task type - "detect", "segment", "pose", or "classify".
        """
        project_dir = Path(project_dir)
        project_dir.mkdir(parents=True, exist_ok=True)

        image_path = Path(image_dir)
        if image_path.is_absolute():
            # External image directory — store relative to project or absolute
            if create_dirs and not image_path.exists():
                image_path.mkdir(parents=True, exist_ok=True)
            try:
                rel = image_path.relative_to(project_dir)
                image_dir_str = str(rel)
            except ValueError:
                # Outside project dir — store absolute path
                image_dir_str = str(image_path)
        else:
            image_dir_str = image_dir
            if create_dirs and image_dir_str not in {"", "."}:
                (project_dir / image_dir_str).mkdir(exist_ok=True)

        label_dir = "labels"
        if create_dirs:
            (project_dir / label_dir).mkdir(exist_ok=True)

        config = ProjectConfig(
            name=name,
            image_dir=image_dir_str,
            label_dir=label_dir,
            classes=classes or [],
            created_at=datetime.now().isoformat(timespec="seconds"),
            task_type=task_type,
        )
        pm = cls(project_dir, config)
        pm.save()
        return pm

    @classmethod
    def open(cls, project_dir: Path | str) -> ProjectManager:
        """Open an existing project."""
        project_dir = Path(project_dir)
        config_path = project_dir / "project.json"
        if not config_path.exists():
            raise FileNotFoundError(f"No project.json found in {project_dir}")
        data = json.loads(config_path.read_text(encoding="utf-8"))
        config = ProjectConfig.from_dict(data)
        return cls(project_dir, config)

    def save(self) -> None:
        """Save project config to project.json."""
        path = self.project_dir / "project.json"
        path.write_text(
            json.dumps(self.config.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def image_root(self) -> Path:
        """Return the resolved image directory for this project."""
        img_dir = Path(self.config.image_dir)
        if not img_dir.is_absolute():
            img_dir = self.project_dir / img_dir
        return img_dir

    def _normalize_data_folder(self, folder: str | None) -> str:
        if not folder:
            return ""
        value = str(folder).replace("\\", "/").strip("/")
        parts = [p for p in value.split("/") if p and p not in {".", ".."}]
        return "/".join(parts)

    def _is_ignored_data_folder(self, folder: str | None) -> bool:
        normalized = self._normalize_data_folder(folder)
        if not normalized:
            return False
        return normalized.split("/", 1)[0].casefold() in IGNORED_IMAGE_TREE_DIR_NAMES

    def _is_ignored_image_path(self, path: Path, image_root: Path) -> bool:
        try:
            rel = path.relative_to(image_root)
        except ValueError:
            return False
        return any(part.casefold() in IGNORED_IMAGE_TREE_DIR_NAMES for part in rel.parts[:-1])

    def list_data_folders(self) -> list[str]:
        """List folders under image_dir that can be used as data versions."""
        img_dir = self.image_root()
        folders: set[str] = set()
        for folder in self.config.data_folders:
            normalized = self._normalize_data_folder(folder)
            if normalized and not self._is_ignored_data_folder(normalized):
                folders.add(normalized)
        if not img_dir.exists():
            return sorted(folders, key=lambda s: (s.count("/"), s.lower()))
        label_dir = Path(self.config.label_dir)
        if not label_dir.is_absolute():
            label_dir = self.project_dir / label_dir
        for p in img_dir.rglob("*"):
            if not p.is_dir():
                continue
            try:
                p.resolve().relative_to(label_dir.resolve())
                continue
            except ValueError:
                pass
            rel = p.relative_to(img_dir).as_posix()
            if (
                rel
                and not self._is_ignored_data_folder(rel)
                and not any(part.startswith(".") for part in rel.split("/"))
            ):
                folders.add(rel)
        return sorted(folders, key=lambda s: (s.count("/"), s.lower()))

    def create_data_folder(self, folder: str) -> str:
        """Create a data-version folder under image_dir and return its normalized name."""
        name = self._normalize_data_folder(folder)
        if not name:
            raise ValueError("folder name is empty")
        target = self.image_root() / Path(name)
        target.mkdir(parents=True, exist_ok=True)
        if name not in self.config.data_folders:
            self.config.data_folders.append(name)
            self.config.data_folders = self.list_data_folders()
        return name

    def rename_data_folder(self, old: str, new: str) -> str:
        """Rename a data-version folder under image_dir."""
        old_name = self._normalize_data_folder(old)
        new_name = self._normalize_data_folder(new)
        if not old_name or not new_name:
            raise ValueError("folder name is empty")
        root = self.image_root()
        src = root / Path(old_name)
        dst = root / Path(new_name)
        if not src.exists() or not src.is_dir():
            raise FileNotFoundError(src)
        if dst.exists():
            raise FileExistsError(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)

        label_root = Path(self.config.label_dir)
        if not label_root.is_absolute():
            label_root = self.project_dir / label_root
        old_label = label_root / Path(old_name)
        new_label = label_root / Path(new_name)
        if old_label.exists():
            new_label.parent.mkdir(parents=True, exist_ok=True)
            if new_label.exists():
                raise FileExistsError(new_label)
            old_label.rename(new_label)
        if self.config.active_data_folder == old_name:
            self.config.active_data_folder = new_name
        updated: list[str] = []
        for folder in self.config.data_folders:
            normalized = self._normalize_data_folder(folder)
            if normalized == old_name:
                updated.append(new_name)
            elif normalized.startswith(old_name + "/"):
                updated.append(new_name + normalized[len(old_name):])
            elif normalized:
                updated.append(normalized)
        if new_name not in updated:
            updated.append(new_name)
        self.config.data_folders = sorted(set(updated), key=lambda s: (s.count("/"), s.lower()))
        return new_name

    def delete_data_folder(self, folder: str) -> None:
        """Delete an empty data-version folder and its empty mirrored label folder."""
        name = self._normalize_data_folder(folder)
        if not name:
            raise ValueError("folder name is empty")
        target = self.image_root() / Path(name)
        label_root = Path(self.config.label_dir)
        if not label_root.is_absolute():
            label_root = self.project_dir / label_root
        label_target = label_root / Path(name)
        if label_target.exists():
            for child in label_target.rglob("*"):
                if child.is_file():
                    raise OSError(f"label folder is not empty: {label_target}")
        target.rmdir()
        if label_target.exists():
            for child in sorted(label_target.rglob("*"), reverse=True):
                if child.is_dir():
                    child.rmdir()
            label_target.rmdir()
        if self.config.active_data_folder == name:
            self.config.active_data_folder = ""
        self.config.data_folders = [
            f for f in self.config.data_folders
            if f != name and not f.startswith(name + "/")
        ]

    def move_images_to_folder(self, paths, folder: str) -> tuple[list[Path], list[Path]]:
        """Move images and their mirrored label JSONs into a data-version folder.

        Returns:
            (new_paths, skipped_paths). A path is skipped when the destination
            image already exists or the source image is outside image_dir.
        """
        folder_name = self._normalize_data_folder(folder)
        root = self.image_root()
        target_dir = root / Path(folder_name) if folder_name else root
        target_dir.mkdir(parents=True, exist_ok=True)

        moved: list[Path] = []
        skipped: list[Path] = []
        for raw_path in paths:
            src = Path(raw_path)
            try:
                src.resolve().relative_to(root.resolve())
            except (OSError, ValueError):
                skipped.append(src)
                continue
            dst = target_dir / src.name
            if src.resolve() == dst.resolve():
                skipped.append(src)
                continue
            if dst.exists():
                skipped.append(src)
                continue

            old_label = self.label_path_for(src)
            new_label = self.label_path_for(dst)
            if old_label.exists() and new_label.exists():
                skipped.append(src)
                continue

            src.rename(dst)
            if old_label.exists():
                new_label.parent.mkdir(parents=True, exist_ok=True)
                old_label.rename(new_label)
            moved.append(dst)
        return moved, skipped

    def import_images_to_folder(self, sources, folder: str) -> tuple[list[Path], list[Path]]:
        """Copy image files or directories into a data-version folder.

        Directories are scanned recursively. When a source image has a same-stem
        JSON next to it, the JSON is copied to the mirrored label folder using
        the same relative path. An existing mirrored label is preserved and
        does not prevent a missing image from being imported.

        Returns:
            (new_paths, skipped_paths). A path is skipped when the destination
            image already exists.
        """
        folder_name = self._normalize_data_folder(folder)
        root = self.image_root()
        target_dir = root / Path(folder_name) if folder_name else root
        target_dir.mkdir(parents=True, exist_ok=True)

        imported: list[Path] = []
        skipped: list[Path] = []
        seen: set[Path] = set()
        for raw_source in sources:
            source = Path(raw_source)
            if source.is_dir():
                candidates = sorted(
                    p for p in source.rglob("*")
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                )
                base_dir = source
            elif source.is_file() and source.suffix.lower() in IMAGE_EXTENSIONS:
                candidates = [source]
                base_dir = source.parent
            else:
                skipped.append(source)
                continue

            for src in candidates:
                try:
                    resolved = src.resolve()
                except OSError:
                    skipped.append(src)
                    continue
                if resolved in seen:
                    continue
                seen.add(resolved)

                rel = src.relative_to(base_dir)
                dst = target_dir / rel
                label_src = src.with_suffix(".json")
                label_dst = self.label_path_for(dst)

                try:
                    same_file = src.resolve() == dst.resolve()
                except OSError:
                    same_file = False
                if same_file or dst.exists():
                    skipped.append(src)
                    continue

                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                if label_src.exists() and not label_dst.exists():
                    label_dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(label_src, label_dst)
                imported.append(dst)

        if folder_name and folder_name not in self.config.data_folders:
            self.config.data_folders.append(folder_name)
            self.config.data_folders = self.list_data_folders()
        return imported, skipped

    def list_images(self, data_folder: str | None = None) -> list[Path]:
        """List image files in the active data-version folder, sorted by path."""
        img_dir = self.image_root()
        if not img_dir.exists():
            return []
        folder = self._normalize_data_folder(
            self.config.active_data_folder if data_folder is None else data_folder
        )
        if self._is_ignored_data_folder(folder):
            return []
        scan_root = img_dir / Path(folder) if folder else img_dir
        if not scan_root.exists():
            return []
        return sorted(
            p for p in scan_root.rglob("*")
            if (
                p.is_file()
                and p.suffix.lower() in IMAGE_EXTENSIONS
                and not self._is_ignored_image_path(p, img_dir)
            )
        )

    def label_path_for(self, image_path: Path | str) -> Path:
        """Get the label JSON path for a given image."""
        image_path = Path(image_path)
        label_dir = Path(self.config.label_dir)
        if not label_dir.is_absolute():
            label_dir = self.project_dir / label_dir
        try:
            rel = image_path.resolve().relative_to(self.image_root().resolve())
        except (OSError, ValueError):
            rel = Path(image_path.name)
        if len(rel.parts) <= 1:
            return label_dir / (image_path.stem + ".json")
        return label_dir / rel.with_suffix(".json")

    def delete_images(self, paths) -> tuple[int, int]:
        """Delete image files and their corresponding label JSONs from disk.

        Args:
            paths: Iterable of image paths (absolute) to delete.

        Returns:
            (images_deleted, labels_deleted) — count of files that actually existed
            and were removed.
        """
        img_deleted = 0
        lbl_deleted = 0
        for img_path in paths:
            img_path = Path(img_path)
            try:
                if img_path.exists():
                    img_path.unlink()
                    img_deleted += 1
            except OSError:
                logger.warning("Failed to delete image: %s", img_path, exc_info=True)
            label_path = self.label_path_for(img_path)
            try:
                if label_path.exists():
                    label_path.unlink()
                    lbl_deleted += 1
            except OSError:
                logger.warning("Failed to delete label: %s", label_path, exc_info=True)
        return img_deleted, lbl_deleted

    def add_class(self, class_name: str) -> None:
        """Add a class if it doesn't exist."""
        if class_name not in self.config.classes:
            self.config.classes.append(class_name)

    def remove_class(self, class_name: str) -> None:
        """Remove a class."""
        if class_name in self.config.classes:
            self.config.classes.remove(class_name)
            self.config.class_colors.pop(class_name, None)
