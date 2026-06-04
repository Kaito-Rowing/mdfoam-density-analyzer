from __future__ import annotations

import math
from pathlib import Path

import pytest

from mdfoam_analyzer.openfoam import (
    OpenFoamParseError,
    discover_density_fields,
    equivalent_radius,
    numeric_time_dirs,
    read_field_info,
    read_mesh_volumes,
    read_scalar_internal_field,
    strip_comments,
)


FIXTURES = Path(__file__).resolve().parent / "fixtures"
CASE = FIXTURES / "minimal_parent" / "case001"
MAIN = CASE / "main"


def test_strip_comments_removes_block_and_line_comments() -> None:
    clean = strip_comments("value 1; // ignored\n/* ignored */ value2 2;")

    assert "ignored" not in clean
    assert "value 1;" in clean
    assert "value2 2;" in clean


def test_reads_scalar_field_info_and_values() -> None:
    field_path = MAIN / "0" / "rhoM_water"

    info = read_field_info(field_path)

    assert info.field_class == "volScalarField"
    assert info.location == "0"
    assert info.object_name == "rhoM_water"
    assert info.value_count == 1
    assert read_scalar_internal_field(field_path) == [1000.0]


def test_rejects_mismatched_nonuniform_scalar_count(tmp_path: Path) -> None:
    field_path = tmp_path / "rhoM_water"
    field_path.write_text(
        """
FoamFile
{
    class volScalarField;
    object rhoM_water;
}
internalField nonuniform List<scalar>
2
(
1
)
;
""",
        encoding="utf-8",
    )

    with pytest.raises(OpenFoamParseError, match="declared 2 values"):
        read_scalar_internal_field(field_path)


def test_discovers_numeric_time_dirs_and_density_fields() -> None:
    time_dirs = numeric_time_dirs(MAIN)

    assert [time for time, _ in time_dirs] == [0.0, 1.0, 2.0, 3.0]
    assert discover_density_fields(MAIN) == ["rhoM_water"]


def test_reads_single_cell_cube_mesh_volume_and_center() -> None:
    mesh = read_mesh_volumes(MAIN / "constant" / "polyMesh")

    assert mesh.volumes == [pytest.approx(1.0)]
    assert mesh.is_constant is True
    assert mesh.unique_volume_count == 1
    assert mesh.total_volume == pytest.approx(1.0)
    assert mesh.cell_centers == [(0.5, 0.5, 0.5)]
    assert mesh.point_bounds == ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
    assert mesh.neighbour_pairs == []


def test_equivalent_radius_formula() -> None:
    volume = 4.0 / 3.0 * math.pi * 8.0

    assert equivalent_radius(volume) == pytest.approx(2.0)
    assert equivalent_radius(0.0) == 0.0
