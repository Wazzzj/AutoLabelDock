"""Helpers for detecting image/JSON data directories inside a project."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.core.project import IMAGE_EXTENSIONS, ProjectManager


COMMON_IMAGE_DIRS = (
    ".",
    "images",
    "image",
    "imgs",
    "img",
    "raw",
    "raw_data",
    "data",
    "dataset",
)
COMMON_LABEL_DIRS = (
    ".",
    "labels",
    "label",
    "annotations",
    "annotation",
    "json",
    "jsons",
)
IGNORED_DIR_NAMES = {
    ".backups",
    ".git",
    "__pycache__",
    "datasets",
    "models",
    "runs",
}


@dataclass(frozen=True)
class ProjectDataCandidate:
    """A possible image + JSON label directory pairing."""

    image_dir: str
    label_dir: str
    image_count: int
    matched_json_count: int
    label_count: int

    @property
    def score(self) -> tuple[int, int, int]:
        return (self.matched_json_count, self.image_count, self.label_count)


def _as_config_path(project_dir: Path, path: Path) -> str:
    try:
        rel = path.resolve().relative_to(project_dir.resolve())
        text = rel.as_posix()
        return text if text else "."
    except ValueError:
        return str(path)


def _resolve_config_dir(project_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_dir / path


def _iter_images(directory: Path) -> list[Path]:
    if not directory.exists() or not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _label_dirs_for(project_dir: Path, image_dir: Path) -> list[Path]:
    dirs: list[Path] = [image_dir]
    dirs.extend(project_dir / name for name in COMMON_LABEL_DIRS if name != ".")
    if image_dir.parent != project_dir:
        dirs.extend(image_dir.parent / name for name in COMMON_LABEL_DIRS if name != ".")

    seen: set[Path] = set()
    unique: list[Path] = []
    for path in dirs:
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _image_dirs_for(project_dir: Path) -> list[Path]:
    dirs = [project_dir / name for name in COMMON_IMAGE_DIRS]
    if project_dir.exists():
        for child in sorted(project_dir.iterdir()):
            if (
                child.is_dir()
                and child.name not in IGNORED_DIR_NAMES
                and not child.name.startswith(".")
            ):
                dirs.append(child)

    seen: set[Path] = set()
    unique: list[Path] = []
    for path in dirs:
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def find_project_data_candidates(project: ProjectManager) -> list[ProjectDataCandidate]:
    """Find project-local folders that contain images and same-stem JSON files."""
    project_dir = project.project_dir
    candidates: list[ProjectDataCandidate] = []

    for image_dir in _image_dirs_for(project_dir):
        images = _iter_images(image_dir)
        if not images:
            continue
        image_stems = {p.stem for p in images}
        for label_dir in _label_dirs_for(project_dir, image_dir):
            if not label_dir.exists() or not label_dir.is_dir():
                continue
            label_files = [p for p in label_dir.glob("*.json") if p.name != "project.json"]
            if not label_files:
                continue
            label_stems = {p.stem for p in label_files}
            matched = len(image_stems & label_stems)
            if matched == 0:
                continue
            candidates.append(
                ProjectDataCandidate(
                    image_dir=_as_config_path(project_dir, image_dir),
                    label_dir=_as_config_path(project_dir, label_dir),
                    image_count=len(images),
                    matched_json_count=matched,
                    label_count=len(label_files),
                )
            )

    return sorted(candidates, key=lambda c: c.score, reverse=True)


def best_project_data_candidate(project: ProjectManager) -> ProjectDataCandidate | None:
    candidates = find_project_data_candidates(project)
    return candidates[0] if candidates else None


def current_project_data_candidate(project: ProjectManager) -> ProjectDataCandidate | None:
    """Summarize the currently configured image_dir/label_dir pairing."""
    image_dir = _resolve_config_dir(project.project_dir, project.config.image_dir)
    label_dir = _resolve_config_dir(project.project_dir, project.config.label_dir)
    images = _iter_images(image_dir)
    if not images or not label_dir.exists() or not label_dir.is_dir():
        return None
    image_stems = {p.stem for p in images}
    label_files = [p for p in label_dir.glob("*.json") if p.name != "project.json"]
    label_stems = {p.stem for p in label_files}
    matched = len(image_stems & label_stems)
    return ProjectDataCandidate(
        image_dir=project.config.image_dir,
        label_dir=project.config.label_dir,
        image_count=len(images),
        matched_json_count=matched,
        label_count=len(label_files),
    )
