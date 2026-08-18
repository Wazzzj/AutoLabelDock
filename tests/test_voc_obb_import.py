import math
import xml.etree.ElementTree as ET

from src.core.annotation import Annotation, ImageAnnotation
from src.core.formats import get_export_registry
from src.core.formats.voc_obb import export_rolabelimg_obb, import_voc_obb


def test_voc_obb_import_converts_radians_to_normalized_corners(tmp_path):
    (tmp_path / "sample.xml").write_text(
        """<annotation>
  <filename>sample.jpeg</filename>
  <size><width>200</width><height>100</height><depth>3</depth></size>
  <object><name>label</name><type>robndbox</type><robndbox>
    <cx>100</cx><cy>50</cy><w>80</w><h>20</h><angle>0.5</angle>
  </robndbox></object>
</annotation>""",
        encoding="utf-8",
    )

    imported = import_voc_obb(tmp_path)
    annotation = imported[0].annotations[0]

    assert imported[0].image_path == "sample.jpeg"
    assert imported[0].image_size == (200, 100)
    assert annotation.class_name == "label"
    assert len(annotation.polygon) == 4
    expected_x = (100 - 40 * math.cos(0.5) + 10 * math.sin(0.5)) / 200
    expected_y = (50 - 40 * math.sin(0.5) - 10 * math.cos(0.5)) / 100
    assert abs(annotation.polygon[0][0] - expected_x) < 1e-9
    assert abs(annotation.polygon[0][1] - expected_y) < 1e-9
    assert annotation.bbox is not None


def test_voc_obb_import_prefers_sidecar_stem_when_filename_is_stale(tmp_path):
    (tmp_path / "part_1.xml").write_text(
        """<annotation>
  <filename>part_0.jpeg</filename>
  <size><width>200</width><height>100</height><depth>3</depth></size>
  <object><name>label</name><robndbox>
    <cx>100</cx><cy>50</cy><w>80</w><h>20</h><angle>0</angle>
  </robndbox></object>
</annotation>""",
        encoding="utf-8",
    )

    imported = import_voc_obb(tmp_path)

    assert imported[0].image_path == "part_1"


def test_rolabelimg_obb_export_writes_robndbox_and_round_trips(tmp_path):
    center_x, center_y = 100, 50
    width, height, angle = 80, 20, 0.4
    corners = [
        (
            (center_x + dx * math.cos(angle) - dy * math.sin(angle)) / 200,
            (center_y + dx * math.sin(angle) + dy * math.cos(angle)) / 100,
        )
        for dx, dy in (
            (-width / 2, -height / 2),
            (width / 2, -height / 2),
            (width / 2, height / 2),
            (-width / 2, height / 2),
        )
    ]
    export_rolabelimg_obb(
        [
            ImageAnnotation(
                image_path="images/batch/sample.jpg",
                image_size=(200, 100),
                annotations=[
                    Annotation(
                        class_name="part",
                        class_id=0,
                        polygon=corners,
                        confirmed=True,
                    )
                ],
            )
        ],
        tmp_path,
        only_confirmed=True,
        task_type="obb",
    )

    xml_path = tmp_path / "batch" / "sample.xml"
    root = ET.parse(xml_path).getroot()
    assert root.findtext("filename") == "sample.jpg"
    assert root.findtext("object/name") == "part"
    assert root.find("object/robndbox") is not None

    imported = import_voc_obb(tmp_path)
    assert len(imported) == 1
    assert imported[0].annotations[0].class_name == "part"
    for actual, expected in zip(imported[0].annotations[0].polygon, corners):
        assert abs(actual[0] - expected[0]) < 1e-6
        assert abs(actual[1] - expected[1]) < 1e-6


def test_obb_export_formats_are_split_in_registry():
    registry = get_export_registry()

    rolabelimg = registry.get("RoLabelImg-OBB")
    xanylabeling = registry.get("X-AnyLabeling-OBB")
    assert rolabelimg.label == "roLabelImg OBB (xml)"
    assert xanylabeling.label == "X-AnyLabeling OBB (json)"
    assert rolabelimg.task_types == xanylabeling.task_types == frozenset({"obb"})
