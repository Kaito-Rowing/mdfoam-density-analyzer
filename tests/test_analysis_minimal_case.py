from __future__ import annotations

import math
from pathlib import Path

import pytest

from mdfoam_analyzer.analysis import (
    AnalysisSettings,
    analyze_case,
    discover_cases,
    discover_fields_for_cases,
    find_evaporation_time,
)
from mdfoam_analyzer.openfoam import equivalent_radius


FIXTURES = Path(__file__).resolve().parent / "fixtures"
PARENT = FIXTURES / "minimal_parent"
CASE = PARENT / "case001"


def _write_minimal_field(path: Path, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(str(value) for value in values)
    path.write_text(
        f"""
FoamFile
{{
    version 2.0;
    format ascii;
    class volScalarField;
    location "{path.parent.name}";
    object rhoM_water;
}}
dimensions [1 -3 0 0 0 0 0];
internalField nonuniform List<scalar>
{len(values)}
(
{body}
)
;
boundaryField
{{
}}
""",
        encoding="utf-8",
    )


def test_discovers_parent_and_single_case() -> None:
    assert discover_cases(PARENT) == [CASE.resolve()]
    assert discover_cases(CASE) == [CASE.resolve()]


def test_discovers_fields_for_minimal_case() -> None:
    assert discover_fields_for_cases([CASE]) == ["rhoM_water"]


def test_analyzes_minimal_case_without_real_simulation_data() -> None:
    result = analyze_case(CASE, AnalysisSettings(consecutive_zero_count=3))

    assert result.status == "ok"
    assert result.error == ""
    assert result.case_name == "case001"
    assert result.time_count == 4
    assert result.field_class == "volScalarField"
    assert result.volume_mode == "mesh constant cell volume"
    assert result.evaporation_time == 1.0

    first = result.rows[0]
    assert first.time == 0.0
    assert first.volume == pytest.approx(1.0)
    assert first.equivalent_radius == pytest.approx(equivalent_radius(1.0))
    assert first.selected_cell_count == 1
    assert first.total_cell_count == 1
    assert first.contact_angle_deg is None
    assert first.contact_radius is None

    assert [row.volume for row in result.rows] == [1.0, 0.0, 0.0, 0.0]
    assert result.max_volume == pytest.approx(1.0)
    assert result.final_volume == 0.0


def test_fallback_manual_cell_volume_when_mesh_is_unavailable(tmp_path: Path) -> None:
    case_dir = tmp_path / "fallback_case"
    _write_minimal_field(case_dir / "main" / "0" / "rhoM_water", [1000.0, 0.0])

    result = analyze_case(
        case_dir,
        AnalysisSettings(
            density_threshold=500.0,
            manual_cell_volume=0.25,
            consecutive_zero_count=1,
        ),
    )

    assert result.status == "ok"
    assert result.volume_mode == "manual constant cell volume"
    assert result.rows[0].volume == pytest.approx(0.25)
    assert result.rows[0].equivalent_radius == pytest.approx(equivalent_radius(0.25))
    assert result.rows[0].selected_cell_count == 1
    assert result.rows[0].total_cell_count == 2
    assert result.rows[0].contact_angle_deg is None
    assert result.rows[0].contact_radius is None


def test_evaporation_time_uses_first_time_in_consecutive_zero_run() -> None:
    rows = analyze_case(CASE, AnalysisSettings(consecutive_zero_count=3)).rows

    assert find_evaporation_time(rows, zero_tolerance=0.0, consecutive_zero_count=3) == 1.0
    assert find_evaporation_time(rows, zero_tolerance=0.0, consecutive_zero_count=4) is None


def test_equivalent_radius_matches_sphere_volume_identity() -> None:
    sphere_volume = 4.0 / 3.0 * math.pi * 2.0**3

    assert equivalent_radius(sphere_volume) == pytest.approx(2.0)
