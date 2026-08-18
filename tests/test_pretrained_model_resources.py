from pathlib import Path

from src.core.resources import PRETRAINED_MODEL_DIR, resolve_pretrained_model_path


def test_bundled_pretrained_model_name_resolves_to_dedicated_directory():
    resolved = resolve_pretrained_model_path("yolo11n.pt")

    assert resolved == PRETRAINED_MODEL_DIR / "yolo11n.pt"
    assert resolved.is_file()


def test_unknown_official_model_name_remains_available_for_auto_download():
    model_name = "not-bundled-yolo-model.pt"

    assert resolve_pretrained_model_path(model_name) == Path(model_name)
