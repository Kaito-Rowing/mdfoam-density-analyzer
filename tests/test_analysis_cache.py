from __future__ import annotations

import os
from pathlib import Path
import shutil

import numpy as np
import pytest

import mdfoam_analyzer.analysis as analysis
from mdfoam_analyzer.analysis import AnalysisSettings, analyze_case
from mdfoam_analyzer.analysis_cache import AnalysisCacheSession


FIXTURES = Path(__file__).resolve().parent / "fixtures"
SOURCE_CASE = FIXTURES / "minimal_parent" / "case001"


def _copy_case(tmp_path: Path) -> Path:
    case = tmp_path / "case001"
    shutil.copytree(SOURCE_CASE, case)
    return case


def _fail(*args, **kwargs):
    raise AssertionError("uncached parser or calculation was called")


def test_complete_cache_hit_matches_uncached_result_and_skips_calculation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _copy_case(tmp_path)
    cache_root = tmp_path / "cache"
    first = analyze_case(
        case,
        AnalysisSettings(),
        cache_session=AnalysisCacheSession(cache_root),
    )

    monkeypatch.setattr(analysis, "read_scalar_internal_field", _fail)
    monkeypatch.setattr(analysis, "read_mesh_volumes", _fail)
    monkeypatch.setattr(analysis, "_volume_for_time", _fail)
    second = analyze_case(
        case,
        AnalysisSettings(),
        cache_session=AnalysisCacheSession(cache_root),
    )

    assert second == first


def test_threshold_change_reuses_density_and_mesh_but_recalculates_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _copy_case(tmp_path)
    cache_root = tmp_path / "cache"
    analyze_case(
        case,
        AnalysisSettings(density_threshold=500.0),
        cache_session=AnalysisCacheSession(cache_root),
    )
    original_volume_for_time = analysis._volume_for_time
    calls = 0

    def count_volume_calls(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_volume_for_time(*args, **kwargs)

    monkeypatch.setattr(analysis, "read_scalar_internal_field", _fail)
    monkeypatch.setattr(analysis, "read_mesh_volumes", _fail)
    monkeypatch.setattr(analysis, "_volume_for_time", count_volume_calls)
    result = analyze_case(
        case,
        AnalysisSettings(density_threshold=700.0),
        cache_session=AnalysisCacheSession(cache_root),
    )

    assert result.status == "ok"
    assert calls == 4


def test_same_size_and_mtime_content_change_invalidates_result(
    tmp_path: Path,
) -> None:
    case = _copy_case(tmp_path)
    cache_root = tmp_path / "cache"
    first = analyze_case(
        case,
        AnalysisSettings(),
        cache_session=AnalysisCacheSession(cache_root),
    )
    field = case / "main" / "0" / "rhoM_water"
    original_stat = field.stat()
    original = field.read_text(encoding="utf-8")
    changed = original.replace("\n1000\n", "\n0000\n")
    assert len(changed.encode("utf-8")) == len(original.encode("utf-8"))
    field.write_text(changed, encoding="utf-8")
    os.utime(field, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    second = analyze_case(
        case,
        AnalysisSettings(),
        cache_session=AnalysisCacheSession(cache_root),
    )

    assert first.rows[0].volume == 1.0
    assert second.rows[0].volume == 0.0


def test_added_time_directory_invalidates_complete_result(tmp_path: Path) -> None:
    case = _copy_case(tmp_path)
    cache_root = tmp_path / "cache"
    first = analyze_case(
        case,
        AnalysisSettings(),
        cache_session=AnalysisCacheSession(cache_root),
    )
    new_field = case / "main" / "4" / "rhoM_water"
    new_field.parent.mkdir()
    shutil.copy2(case / "main" / "3" / "rhoM_water", new_field)

    second = analyze_case(
        case,
        AnalysisSettings(),
        cache_session=AnalysisCacheSession(cache_root),
    )

    assert first.time_count == 4
    assert second.time_count == 5


def test_corrupt_result_cache_falls_back_to_correct_partial_cache(
    tmp_path: Path,
) -> None:
    case = _copy_case(tmp_path)
    cache_root = tmp_path / "cache"
    expected = analyze_case(
        case,
        AnalysisSettings(),
        cache_session=AnalysisCacheSession(cache_root),
    )
    result_entry = next((cache_root / "result").iterdir())
    (result_entry / "float_rows.npy").write_bytes(b"corrupt")
    logs: list[str] = []

    actual = analyze_case(
        case,
        AnalysisSettings(),
        cache_session=AnalysisCacheSession(cache_root, log=logs.append),
        log=logs.append,
    )

    assert actual == expected
    assert any("recalculating" in message for message in logs)


def test_algorithm_version_change_invalidates_only_complete_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _copy_case(tmp_path)
    cache_root = tmp_path / "cache"
    analyze_case(
        case,
        AnalysisSettings(),
        cache_session=AnalysisCacheSession(cache_root),
    )
    monkeypatch.setattr(
        analysis,
        "ANALYSIS_ALGORITHM_VERSION",
        analysis.ANALYSIS_ALGORITHM_VERSION + 1,
    )
    monkeypatch.setattr(analysis, "read_scalar_internal_field", _fail)
    monkeypatch.setattr(analysis, "read_mesh_volumes", _fail)

    result = analyze_case(
        case,
        AnalysisSettings(),
        cache_session=AnalysisCacheSession(cache_root),
    )

    assert result.status == "ok"


def test_processor_case_complete_cache_hit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = tmp_path / "processor_case"
    source_mesh = SOURCE_CASE / "main" / "constant" / "polyMesh"
    source_field = SOURCE_CASE / "main" / "0" / "rhoM_water"
    for processor_name in ("processor0", "processor1"):
        processor = case / "main" / processor_name
        shutil.copytree(source_mesh, processor / "constant" / "polyMesh")
        field = processor / "0" / "rhoM_water"
        field.parent.mkdir(parents=True)
        shutil.copy2(source_field, field)
    cache_root = tmp_path / "cache"
    first = analyze_case(
        case,
        AnalysisSettings(),
        cache_session=AnalysisCacheSession(cache_root),
    )

    monkeypatch.setattr(analysis, "read_scalar_internal_field", _fail)
    monkeypatch.setattr(analysis, "read_mesh_volumes", _fail)
    second = analyze_case(
        case,
        AnalysisSettings(),
        cache_session=AnalysisCacheSession(cache_root),
    )

    assert second == first


def test_stopped_and_error_results_are_not_cached(tmp_path: Path) -> None:
    case = _copy_case(tmp_path)
    stopped_root = tmp_path / "stopped-cache"
    stopped = analyze_case(
        case,
        AnalysisSettings(),
        stop_requested=lambda: True,
        cache_session=AnalysisCacheSession(stopped_root),
    )
    error_root = tmp_path / "error-cache"
    error = analyze_case(
        case,
        AnalysisSettings(density_field="missing"),
        cache_session=AnalysisCacheSession(error_root),
    )

    assert stopped.status == "stopped"
    assert error.status == "error"
    assert not (stopped_root / "result").exists()
    assert not (error_root / "result").exists()


def test_partial_entries_are_rolled_back_after_analysis_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _copy_case(tmp_path)
    cache_root = tmp_path / "cache"
    original_reader = analysis.read_scalar_internal_field
    calls = 0

    def fail_after_first(path: Path) -> list[float]:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("forced parser failure")
        return original_reader(path)

    monkeypatch.setattr(analysis, "read_scalar_internal_field", fail_after_first)
    result = analyze_case(
        case,
        AnalysisSettings(),
        cache_session=AnalysisCacheSession(cache_root),
    )

    assert result.status == "error"
    assert not list((cache_root / "density").glob("*"))
    assert not list((cache_root / "mesh").glob("*"))
    assert not (cache_root / "result").exists()


def test_lru_prune_removes_oldest_entry_first(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    session = AnalysisCacheSession(cache_root, max_bytes=10**9)
    session.store_density("old", list(np.linspace(0.0, 1.0, 128)))
    session.store_density("new", list(np.linspace(1.0, 2.0, 128)))
    old_entry = cache_root / "density" / "old"
    new_entry = cache_root / "density" / "new"
    os.utime(old_entry, (1, 1))
    os.utime(new_entry, (2, 2))
    new_size = sum(path.stat().st_size for path in new_entry.iterdir())
    session.max_bytes = new_size + 1

    session.prune()

    assert not old_entry.exists()
    assert new_entry.exists()
