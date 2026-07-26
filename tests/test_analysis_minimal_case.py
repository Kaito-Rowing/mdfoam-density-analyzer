from __future__ import annotations

import csv
import math
from pathlib import Path
import shutil

import pytest

from mdfoam_analyzer.analysis import (
    AnalysisLayoutProfile,
    AnalysisSettings,
    analyze_case,
    detect_analysis_layout,
    detect_batch_layout,
    discover_cases,
    discover_fields_for_cases,
    find_evaporation_time,
    write_summary_csv,
    write_timeseries_csv,
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


def test_discovers_and_analyzes_direct_openfoam_case_root(tmp_path: Path) -> None:
    direct_case = tmp_path / "v12"
    shutil.copytree(CASE / "main", direct_case)

    assert discover_cases(direct_case) == [direct_case.resolve()]
    assert discover_cases(tmp_path) == [direct_case.resolve()]
    assert discover_fields_for_cases([direct_case]) == ["rhoM_water"]

    result = analyze_case(
        direct_case,
        AnalysisSettings(consecutive_zero_count=3),
    )

    assert result.status == "ok"
    assert result.case_name == "v12"
    assert result.time_count == 4
    assert result.evaporation_time == 1.0


def test_discovers_multiple_cases_with_differently_named_data_roots(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "batch"
    conventional_case = parent / "main1"
    named_case = parent / "mainv12"
    shutil.copytree(CASE / "main", conventional_case / "main")
    shutil.copytree(CASE / "main", named_case / "mainv12")

    cases = discover_cases(parent)

    assert cases == [conventional_case.resolve(), named_case.resolve()]
    assert discover_fields_for_cases(cases) == ["rhoM_water"]
    results = [
        analyze_case(case, AnalysisSettings(consecutive_zero_count=3))
        for case in cases
    ]
    assert [result.case_name for result in results] == ["main1", "mainv12"]
    assert [result.status for result in results] == ["ok", "ok"]


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


def test_mismatched_time_is_skipped_and_breaks_zero_sequence(
    tmp_path: Path,
) -> None:
    case = tmp_path / "case_with_fragment"
    shutil.copytree(CASE, case)
    _write_minimal_field(case / "main" / "2" / "rhoM_water", [0.0, 0.0])
    _write_minimal_field(case / "main" / "4" / "rhoM_water", [0.0])

    result = analyze_case(
        case,
        AnalysisSettings(consecutive_zero_count=2),
    )

    assert result.status == "ok"
    assert [row.time for row in result.rows] == [0.0, 1.0, 3.0, 4.0]
    assert result.evaporation_time == 3.0
    assert len(result.warnings) == 1
    assert len(result.skipped_times) == 1
    skipped = result.skipped_times[0]
    assert skipped.time == 2.0
    assert skipped.density_count == 2
    assert skipped.expected_cell_count == 1

    summary_path = tmp_path / "summary.csv"
    timeseries_path = tmp_path / "timeseries.csv"
    write_summary_csv(summary_path, [result])
    write_timeseries_csv(timeseries_path, [result])
    with summary_path.open(encoding="utf-8-sig", newline="") as handle:
        summary = next(csv.DictReader(handle))
    with timeseries_path.open(encoding="utf-8-sig", newline="") as handle:
        timeseries = list(csv.DictReader(handle))
    assert summary["warning_count"] == "1"
    assert summary["skipped_time_count"] == "1"
    assert summary["skipped_times"] == "2"
    assert [float(row["time"]) for row in timeseries] == [0.0, 1.0, 3.0, 4.0]


def test_all_mismatched_times_return_error_with_batch_profile(
    tmp_path: Path,
) -> None:
    case = tmp_path / "all_fragments"
    shutil.copytree(CASE, case)
    for time_name in ("0", "1", "2", "3"):
        _write_minimal_field(
            case / "main" / time_name / "rhoM_water",
            [0.0, 0.0],
        )
    profile = detect_analysis_layout(CASE, AnalysisSettings())

    result = analyze_case(case, AnalysisSettings(), layout_profile=profile)

    assert result.status == "error"
    assert result.rows == []
    assert len(result.skipped_times) == 4
    assert "No valid rhoM_water" in result.error


def test_batch_layout_uses_first_analyzable_case_and_rejects_cell_mismatch() -> None:
    profile = detect_batch_layout([CASE], AnalysisSettings())

    assert profile.mode == "reconstructed"
    assert profile.expected_total_cells == 1
    assert profile.processor_count == 0
    assert Path(profile.source_case) == CASE.resolve()

    incompatible = AnalysisLayoutProfile(
        mode="reconstructed",
        expected_total_cells=2,
        processor_count=0,
        source_case="synthetic-first-case",
    )
    result = analyze_case(CASE, AnalysisSettings(), layout_profile=incompatible)

    assert result.status == "error"
    assert "Batch layout mismatch" in result.error


def test_evaporation_time_uses_first_time_in_consecutive_zero_run() -> None:
    rows = analyze_case(CASE, AnalysisSettings(consecutive_zero_count=3)).rows

    assert find_evaporation_time(rows, zero_tolerance=0.0, consecutive_zero_count=3) == 1.0
    assert find_evaporation_time(rows, zero_tolerance=0.0, consecutive_zero_count=4) is None


def test_equivalent_radius_matches_sphere_volume_identity() -> None:
    sphere_volume = 4.0 / 3.0 * math.pi * 2.0**3

    assert equivalent_radius(sphere_volume) == pytest.approx(2.0)
