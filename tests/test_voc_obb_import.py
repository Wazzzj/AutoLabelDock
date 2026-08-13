import math

from src.core.formats.voc_obb import import_voc_obb


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
