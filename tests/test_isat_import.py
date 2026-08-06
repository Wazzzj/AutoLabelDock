import json
import tempfile
import unittest
from pathlib import Path

from src.core.annotation import Annotation, ImageAnnotation
from src.core.formats import get_export_registry, get_import_registry
from src.core.formats.isat import export_isat, import_isat, import_isat_file
from src.core.annotation_classes import merged_project_annotation_classes
from src.core.label_io import load_annotation
from src.core.project import ProjectManager


def _isat_data():
    return {
        "info": {
            "description": "ISAT",
            "folder": "C:/external/images",
            "name": "sample.jpeg",
            "width": 200,
            "height": 100,
            "depth": 3,
            "note": "",
        },
        "objects": [
            {
                "category": "glue",
                "group": 1,
                "segmentation": [[20, 10], [80, 10], [80, 60], [20, 60]],
                "bbox": [19.5, 9.5, 80.5, 60.5],
                "area": 3000,
                "layer": 1,
                "iscrowd": False,
                "note": "",
            }
        ],
    }


class TestISATImport(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self._temp_dir.name)

    def tearDown(self):
        self._temp_dir.cleanup()

    def _write_sample(self, data=None):
        json_path = self.temp_path / "sample.json"
        json_path.write_text(json.dumps(data or _isat_data()), encoding="utf-8")
        return json_path

    def test_import_isat_file_preserves_polygon_and_bbox(self):
        image_annotation = import_isat_file(self._write_sample())

        self.assertIsNotNone(image_annotation)
        self.assertEqual(image_annotation.image_path, "sample.jpeg")
        self.assertEqual(image_annotation.image_size, (200, 100))
        self.assertEqual(len(image_annotation.annotations), 1)
        annotation = image_annotation.annotations[0]
        self.assertEqual(annotation.class_name, "glue")
        self.assertEqual(annotation.source, "isat")
        self.assertTrue(annotation.confirmed)
        expected_polygon = [
            (0.1, 0.1),
            (0.4, 0.1),
            (0.4, 0.6),
            (0.1, 0.6),
        ]
        for actual, expected in zip(annotation.polygon, expected_polygon):
            for actual_value, expected_value in zip(actual, expected):
                self.assertAlmostEqual(actual_value, expected_value)
        for actual, expected in zip(
            annotation.bbox,
            (0.25, 0.35, 0.305, 0.51),
        ):
            self.assertAlmostEqual(actual, expected)

    def test_import_isat_falls_back_to_polygon_bbox_and_skips_background(self):
        data = _isat_data()
        data["objects"][0].pop("bbox")
        data["objects"].append(
            {
                "category": "__background__",
                "segmentation": [[0, 0], [10, 0], [10, 10]],
                "bbox": [0, 0, 10, 10],
            }
        )

        image_annotation = import_isat_file(self._write_sample(data))

        self.assertIsNotNone(image_annotation)
        self.assertEqual(len(image_annotation.annotations), 1)
        for actual, expected in zip(
            image_annotation.annotations[0].bbox,
            (0.25, 0.35, 0.3, 0.5),
        ):
            self.assertAlmostEqual(actual, expected)

    def test_import_isat_directory_ignores_other_json_formats(self):
        nested = self.temp_path / "nested"
        nested.mkdir()
        (nested / "sample.json").write_text(
            json.dumps(_isat_data()),
            encoding="utf-8",
        )
        (self.temp_path / "labelme.json").write_text(
            json.dumps({"imagePath": "other.jpg", "shapes": []}),
            encoding="utf-8",
        )
        (self.temp_path / "broken.json").write_text("{", encoding="utf-8")

        imported = import_isat(self.temp_path)

        self.assertEqual(
            [annotation.image_path for annotation in imported],
            ["sample.jpeg"],
        )

    def test_load_annotation_recognizes_isat_sidecar(self):
        image_annotation = load_annotation(self._write_sample())

        self.assertIsNotNone(image_annotation)
        self.assertEqual(image_annotation.annotations[0].source, "isat")

    def test_isat_importer_is_registered(self):
        info = get_import_registry().get("iSAT")

        self.assertIsNotNone(info)
        self.assertEqual(info.label, "iSAT (json)")
        self.assertFalse(info.input_is_file)

    def test_export_isat_preserves_polygon_bbox_and_subdirectory(self):
        output_dir = self.temp_path / "labels"
        export_isat(
            [
                ImageAnnotation(
                    image_path="images/batch_a/sample.jpeg",
                    image_size=(200, 100),
                    annotations=[
                        Annotation(
                            class_name="glue",
                            class_id=0,
                            bbox=(0.25, 0.35, 0.305, 0.51),
                            polygon=[
                                (0.1, 0.1),
                                (0.4, 0.1),
                                (0.4, 0.6),
                                (0.1, 0.6),
                            ],
                        )
                    ],
                )
            ],
            output_dir,
        )

        json_path = output_dir / "batch_a" / "sample.json"
        data = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(data["info"], {
            "description": "ISAT",
            "folder": "images/batch_a",
            "name": "sample.jpeg",
            "width": 200,
            "height": 100,
            "depth": 3,
            "note": "",
        })
        self.assertEqual(data["objects"][0]["segmentation"], [
            [20.0, 10.0],
            [80.0, 10.0],
            [80.0, 60.0],
            [20.0, 60.0],
        ])
        self.assertEqual(data["objects"][0]["bbox"], [19.5, 9.5, 80.5, 60.5])
        self.assertEqual(data["objects"][0]["area"], 3000.0)

        imported = import_isat_file(json_path)
        self.assertIsNotNone(imported)
        self.assertEqual(imported.image_size, (200, 100))
        self.assertEqual(imported.annotations[0].polygon, [
            (0.1, 0.1),
            (0.4, 0.1),
            (0.4, 0.6),
            (0.1, 0.6),
        ])

    def test_export_isat_converts_bbox_to_polygon_and_filters_unconfirmed(self):
        output_dir = self.temp_path / "labels"
        export_isat(
            [
                ImageAnnotation(
                    image_path="images/sample.jpg",
                    image_size=(100, 80),
                    annotations=[
                        Annotation(
                            class_name="confirmed",
                            class_id=0,
                            bbox=(0.5, 0.5, 0.4, 0.5),
                            confirmed=True,
                        ),
                        Annotation(
                            class_name="pending",
                            class_id=1,
                            bbox=(0.2, 0.2, 0.1, 0.1),
                            confirmed=False,
                        ),
                    ],
                )
            ],
            output_dir,
            only_confirmed=True,
        )

        data = json.loads((output_dir / "sample.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["objects"]), 1)
        self.assertEqual(data["objects"][0]["category"], "confirmed")
        self.assertEqual(data["objects"][0]["segmentation"], [
            [30.0, 20.0],
            [70.0, 20.0],
            [70.0, 60.0],
            [30.0, 60.0],
        ])

    def test_isat_exporter_is_registered(self):
        info = get_export_registry().get("iSAT")

        self.assertIsNotNone(info)
        self.assertEqual(info.label, "iSAT (json)")
        self.assertFalse(info.needs_classes)

    def test_project_class_discovery_reads_isat_sidecar(self):
        project = ProjectManager.create(
            self.temp_path,
            "iSAT project",
            task_type="segment",
        )
        (project.image_root() / "sample.jpeg").write_bytes(b"image placeholder")
        project.label_path_for(project.image_root() / "sample.jpeg").write_text(
            json.dumps(_isat_data()),
            encoding="utf-8",
        )

        self.assertEqual(
            merged_project_annotation_classes(project),
            ["glue"],
        )


if __name__ == "__main__":
    unittest.main()
