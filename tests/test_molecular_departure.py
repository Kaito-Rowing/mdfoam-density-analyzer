from __future__ import annotations

from pathlib import Path
import math
import shutil

import numpy as np
import pytest

from mdfoam_analyzer.analysis import AnalysisSettings, analyze_case
from mdfoam_analyzer.molecular_departure import (
    _Frame,
    _Geometry,
    _aggregate_height_bins,
    _detect_events,
    _height_bin_edges,
    analyze_molecular_departures,
    largest_seeded_periodic_cluster,
    read_label_field,
    read_lagrangian_positions_with_cells,
    read_molecule_id_list,
    write_departure_events_csv,
    write_departure_height_bins_csv,
)


FIXTURES = Path(__file__).resolve().parent / "fixtures"
MINIMAL_CASE = FIXTURES / "minimal_parent" / "case001"
SAMPLE_CASE = Path(__file__).resolve().parents[1] / "run001_x001"


def _frame(
    time: float,
    entries: list[tuple[int, tuple[float, float, float], bool]],
    geometry: _Geometry | None = None,
) -> _Frame:
    geometry = geometry or _Geometry(
        z_base=0.0,
        z_top=2.0,
        sphere_center=(0.0, 0.0, 1.0),
        sphere_radius=1.0,
        contact_radius=1.0,
    )
    return _Frame(
        time=time,
        time_name=str(time),
        positions=np.asarray([item[1] for item in entries], dtype=float).reshape((-1, 3)),
        orig_proc_ids=np.zeros(len(entries), dtype=np.int64),
        orig_ids=np.asarray([item[0] for item in entries], dtype=np.int64),
        liquid_mask=np.asarray([item[2] for item in entries], dtype=bool),
        cluster_size=sum(item[2] for item in entries),
        geometry=geometry,
        cluster_fallback=False,
        point_bounds=((-5.0, 5.0), (-5.0, 5.0), (-1.0, 3.0)),
    )


def _write_cloud(
    time_dir: Path,
    rows: list[tuple[tuple[float, float, float], int, int, int, int]],
) -> None:
    cloud = time_dir / "lagrangian" / "moleculeCloud"
    cloud.mkdir(parents=True, exist_ok=True)
    positions = "\n".join(
        f"({x} {y} {z}) {cell}" for (x, y, z), cell, _, _, _ in rows
    )
    (cloud / "positions").write_text(
        f"FoamFile {{ class Cloud<passivePositionParticle>; }}\n"
        f"{len(rows)}\n(\n{positions}\n)\n",
        encoding="utf-8",
    )
    for name, field_index in (("id", 2), ("origId", 3), ("origProcId", 4)):
        values = "\n".join(str(row[field_index]) for row in rows)
        (cloud / name).write_text(
            f"FoamFile {{ class labelField; }}\n"
            f"{len(rows)}\n(\n{values}\n)\n",
            encoding="utf-8",
        )


def test_reads_positions_compact_labels_and_id_list(tmp_path: Path) -> None:
    positions = tmp_path / "positions"
    positions.write_text(
        "2\n(\n(1 2 3) 4\n(5 6 7) 8\n)\n",
        encoding="utf-8",
    )
    labels = tmp_path / "origProcId"
    labels.write_text("2{3}\n", encoding="utf-8")
    id_list = tmp_path / "idList"
    id_list.write_text("idList 2 ( water Pt1 );\n", encoding="utf-8")

    parsed = read_lagrangian_positions_with_cells(positions)

    np.testing.assert_allclose(parsed.positions, [[1, 2, 3], [5, 6, 7]])
    np.testing.assert_array_equal(parsed.cell_indices, [4, 8])
    np.testing.assert_array_equal(read_label_field(labels), [3, 3])
    assert read_molecule_id_list(id_list) == ["water", "Pt1"]


def test_largest_seeded_cluster_connects_across_periodic_boundary() -> None:
    positions = np.asarray(
        [
            [0.05, 1.0, 0.0],
            [9.95, 1.0, 0.0],
            [5.0, 1.0, 0.0],
        ]
    )

    mask, size, fallback = largest_seeded_periodic_cluster(
        positions,
        np.asarray([True, False, False]),
        cutoff=0.2,
        point_bounds=((0.0, 10.0), (0.0, 2.0), (-1.0, 1.0)),
    )

    np.testing.assert_array_equal(mask, [True, True, False])
    assert size == 2
    assert fallback is False

    empty_mask, empty_size, no_seed = largest_seeded_periodic_cluster(
        positions,
        np.zeros(3, dtype=bool),
        cutoff=0.2,
        point_bounds=((0.0, 10.0), (0.0, 2.0), (-1.0, 1.0)),
    )
    np.testing.assert_array_equal(empty_mask, [False, False, False])
    assert empty_size == 0
    assert no_seed is True


def test_event_detection_handles_confirmation_reentry_and_escape() -> None:
    frames = [
        _frame(
            0.0,
            [
                (1, (0.0, 0.0, 0.4), True),
                (2, (0.0, 0.0, 0.8), False),
                (3, (0.0, 0.0, 0.2), True),
            ],
        ),
        _frame(
            1.0,
            [
                (1, (0.0, 0.0, 0.5), False),
                (2, (0.0, 0.0, 0.8), True),
            ],
        ),
        _frame(
            2.0,
            [
                (1, (0.0, 0.0, 0.6), False),
                (2, (0.0, 0.0, 0.9), False),
            ],
        ),
        _frame(
            3.0,
            [
                (1, (0.0, 0.0, 0.7), False),
                (2, (0.0, 0.0, 0.8), True),
            ],
        ),
        _frame(
            4.0,
            [
                (1, (0.0, 0.0, 0.8), False),
                (2, (0.0, 0.0, 0.9), False),
            ],
        ),
    ]

    events = _detect_events(
        frames,
        species_id=0,
        species_name="water",
        confirmation_frames=3,
        evaporation_time=2.5,
    )

    assert len(events) == 3
    by_id = {}
    for event in events:
        by_id.setdefault(event.orig_id, []).append(event)
    assert by_id[1][0].confirmed is True
    assert by_id[1][0].confirmation_time == 3.0
    assert by_id[2][0].confirmed is False
    assert "unconfirmed_short_excursion" in by_id[2][0].quality_flags
    assert by_id[3][0].confirmed is True
    assert "escaped_domain" in by_id[3][0].quality_flags
    assert all(event.first_outside_time <= 2.5 for event in events)

    bins = _aggregate_height_bins(
        frames,
        events,
        bin_count=2,
        evaporation_time=2.5,
    )
    assert bins[0].area_time_exposure == pytest.approx(5.0 * math.pi)
    assert sum(item.raw_count for item in bins) == 3
    assert sum(item.confirmed_count for item in bins) == 2


def test_equal_surface_area_bins_use_fitted_spherical_band_area() -> None:
    expected = np.linspace(0.0, 1.0, 5)
    np.testing.assert_allclose(_height_bin_edges(4, "equal_height"), expected)
    np.testing.assert_allclose(
        _height_bin_edges(4, "equal_surface_area"),
        expected,
    )

    frames = [
        _frame(0.0, [(1, (0.0, 0.0, 0.5), True)]),
        _frame(1.0, [(1, (0.0, 0.0, 0.5), True)]),
    ]
    bins = _aggregate_height_bins(
        frames,
        [],
        bin_count=4,
        evaporation_time=None,
        bin_mode="equal_surface_area",
    )

    assert [item.area_time_exposure for item in bins] == pytest.approx(
        [math.pi] * 4
    )

    with pytest.raises(ValueError, match="Unsupported departure bin mode"):
        _height_bin_edges(4, "unsupported")


def test_high_cadence_event_uses_sphere_interface_intersection() -> None:
    geometry = _Geometry(
        z_base=-1.0,
        z_top=1.0,
        sphere_center=(0.0, 0.0, 0.0),
        sphere_radius=1.0,
        contact_radius=1.0,
    )
    frames = [
        _frame(0.0, [(1, (0.0, 0.0, 0.5), True)], geometry),
        _frame(1.0e-14, [(1, (0.0, 0.0, 1.5), False)], geometry),
    ]

    events = _detect_events(
        frames,
        species_id=0,
        species_name="water",
        confirmation_frames=1,
        evaporation_time=None,
    )

    assert len(events) == 1
    assert events[0].position_method == "interface_intersection"
    assert events[0].departure_z == pytest.approx(1.0)
    assert events[0].normalized_height == pytest.approx(1.0)


def test_analyze_case_integrates_lagrangian_departure_inputs(
    tmp_path: Path,
) -> None:
    case = tmp_path / "case001"
    shutil.copytree(MINIMAL_CASE, case)
    (case / "main" / "constant" / "idList").write_text(
        "idList 2 ( water Pt1 );\n",
        encoding="utf-8",
    )
    _write_cloud(
        case / "main" / "0",
        [((0.5, 0.5, 0.5), 0, 0, 100, 0)],
    )
    for time_name in ("1", "2", "3"):
        _write_cloud(case / "main" / time_name, [])

    result = analyze_case(
        case,
        AnalysisSettings(departure_enabled=True),
    )

    assert result.status == "ok"
    assert result.departure_result is not None
    assert result.departure_result.status == "ok"
    assert result.departure_result.bin_mode == "equal_height"
    assert result.departure_result.raw_event_count == 1
    assert result.departure_result.confirmed_event_count == 1
    relative_paths = {item.relative_path for item in result.input_files}
    assert "main/constant/idList" in relative_paths
    assert "main/0/lagrangian/moleculeCloud/origId" in relative_paths
    assert "main/0/lagrangian/moleculeCloud/origProcId" in relative_paths

    events_csv = tmp_path / "events.csv"
    bins_csv = tmp_path / "bins.csv"
    write_departure_events_csv(events_csv, [result])
    write_departure_height_bins_csv(bins_csv, [result])
    assert "position_method" in events_csv.read_text(encoding="utf-8-sig")
    assert "area_time_exposure_m2_s" in bins_csv.read_text(
        encoding="utf-8-sig"
    )
    assert "bin_mode" in bins_csv.read_text(encoding="utf-8-sig")


def test_processor_lagrangian_frames_are_aggregated(tmp_path: Path) -> None:
    case = tmp_path / "processor_case"
    processor = case / "main" / "processor0"
    shutil.copytree(MINIMAL_CASE / "main" / "constant", processor / "constant")
    for time_name in ("0", "1", "2", "3"):
        source = MINIMAL_CASE / "main" / time_name / "rhoM_water"
        destination = processor / time_name / "rhoM_water"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    (case / "main" / "constant").mkdir(parents=True)
    (case / "main" / "constant" / "idList").write_text(
        "idList 1 ( water );\n",
        encoding="utf-8",
    )
    _write_cloud(
        processor / "0",
        [((0.5, 0.5, 0.5), 0, 0, 100, 0)],
    )
    for time_name in ("1", "2", "3"):
        _write_cloud(processor / time_name, [])

    result = analyze_case(
        case,
        AnalysisSettings(departure_enabled=True),
    )

    assert result.status == "ok"
    assert result.departure_result is not None
    assert result.departure_result.raw_event_count == 1
    assert any(
        "processor0/0/lagrangian" in item.relative_path
        for item in result.input_files
    )


def test_departure_analysis_honors_stop_request(tmp_path: Path) -> None:
    case = tmp_path / "case001"
    shutil.copytree(MINIMAL_CASE, case)
    (case / "main" / "constant" / "idList").write_text(
        "idList 1 ( water );\n",
        encoding="utf-8",
    )
    _write_cloud(
        case / "main" / "0",
        [((0.5, 0.5, 0.5), 0, 0, 100, 0)],
    )

    result = analyze_molecular_departures(
        case,
        AnalysisSettings(departure_enabled=True),
        evaporation_time=None,
        stop_requested=lambda: True,
    )

    assert result.status == "stopped"


@pytest.mark.skipif(
    not SAMPLE_CASE.is_dir(),
    reason="local sample case is not available",
)
def test_sample_case_departure_regression() -> None:
    result = analyze_case(
        SAMPLE_CASE,
        AnalysisSettings(departure_enabled=True),
    )

    assert result.status == "ok"
    assert result.departure_result is not None
    departure = result.departure_result
    assert departure.status == "ok"
    assert departure.frame_count == 127
    assert departure.raw_event_count > 1200
    assert departure.confirmed_event_count > 1200
    assert departure.confirmed_event_count <= departure.raw_event_count
    assert sum(item.raw_count for item in departure.height_bins) == (
        departure.raw_event_count - departure.excluded_normalized_height_count
    )
