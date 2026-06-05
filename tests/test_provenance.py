from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import shutil

import pytest

from mdfoam_analyzer import __version__
from mdfoam_analyzer.analysis import AnalysisSettings, analyze_case
from mdfoam_analyzer.provenance import (
    MANIFEST_FORMAT,
    PROJECT_FORMAT,
    ProvenanceError,
    RunContext,
    apply_remote_input_paths,
    load_analysis_settings,
    save_analysis_settings,
    write_analysis_manifest,
)


FIXTURES = Path(__file__).resolve().parent / "fixtures"
CASE = FIXTURES / "minimal_parent" / "case001"


def test_analysis_settings_round_trip(tmp_path: Path) -> None:
    settings = AnalysisSettings(
        density_field="rhoN_custom",
        density_threshold=432.1,
        zero_tolerance=1.0e-30,
        consecutive_zero_count=5,
        manual_cell_volume=2.0e-27,
        dx=1.0e-9,
        dy=2.0e-9,
        dz=3.0e-9,
        contact_fit_lower=0.25,
        contact_fit_upper=0.85,
        contact_unwrap_xy=False,
        contact_average_percent=65.0,
    )
    path = tmp_path / "mdfoam_project.json"

    save_analysis_settings(path, settings)

    assert load_analysis_settings(path) == settings
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["format"] == PROJECT_FORMAT
    assert payload["schema_version"] == 1
    assert payload["app_version"] == __version__
    assert payload["analysis_settings"] == asdict(settings)


@pytest.mark.parametrize(
    "payload",
    [
        "{bad json",
        json.dumps({"format": "other", "schema_version": 1}),
        json.dumps(
            {
                "format": PROJECT_FORMAT,
                "schema_version": 999,
                "analysis_settings": {},
            }
        ),
    ],
)
def test_analysis_settings_reject_invalid_documents(
    tmp_path: Path, payload: str
) -> None:
    path = tmp_path / "bad.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ProvenanceError):
        load_analysis_settings(path)


def test_analysis_records_used_files_and_mesh_statistics() -> None:
    result = analyze_case(CASE, AnalysisSettings())

    relative_paths = {item.relative_path for item in result.input_files}
    assert relative_paths == {
        "main/constant/polyMesh/points",
        "main/constant/polyMesh/faces",
        "main/constant/polyMesh/owner",
        "main/constant/polyMesh/neighbour",
        "main/0/rhoM_water",
        "main/1/rhoM_water",
        "main/2/rhoM_water",
        "main/3/rhoM_water",
    }
    assert all(item.size > 0 for item in result.input_files)
    assert all(item.mtime > 0 for item in result.input_files)
    assert result.mesh_statistics is not None
    assert result.mesh_statistics.mesh_count == 1
    assert result.mesh_statistics.cell_count == 1
    assert result.mesh_statistics.total_cell_volume == pytest.approx(1.0)
    assert result.mesh_statistics.cell_centers_available is True


def test_manual_volume_statistics_do_not_claim_mesh_files(tmp_path: Path) -> None:
    case = tmp_path / "manual"
    field = case / "main" / "0" / "rhoM_water"
    field.parent.mkdir(parents=True)
    field.write_text(
        """
FoamFile
{
    class volScalarField;
}
internalField nonuniform List<scalar>
2
(
1000
0
)
;
""",
        encoding="utf-8",
    )

    result = analyze_case(
        case,
        AnalysisSettings(manual_cell_volume=0.25),
    )

    assert [item.relative_path for item in result.input_files] == [
        "main/0/rhoM_water"
    ]
    assert result.mesh_statistics is not None
    assert result.mesh_statistics.volume_mode == "manual constant cell volume"
    assert result.mesh_statistics.cell_count == 2
    assert result.mesh_statistics.cell_centers_available is False


def test_processor_mesh_statistics_are_aggregated(tmp_path: Path) -> None:
    case = tmp_path / "processor_case"
    source_mesh = CASE / "main" / "constant" / "polyMesh"
    source_field = CASE / "main" / "0" / "rhoM_water"
    for processor_name in ("processor0", "processor1"):
        processor = case / "main" / processor_name
        shutil.copytree(source_mesh, processor / "constant" / "polyMesh")
        field = processor / "0" / "rhoM_water"
        field.parent.mkdir(parents=True)
        shutil.copy2(source_field, field)

    result = analyze_case(case, AnalysisSettings())

    assert result.status == "ok"
    assert result.mesh_statistics is not None
    assert result.mesh_statistics.mesh_count == 2
    assert result.mesh_statistics.cell_count == 2
    assert result.mesh_statistics.total_cell_volume == pytest.approx(2.0)
    assert (
        result.mesh_statistics.volume_mode
        == "processor mesh constant cell volumes"
    )
    assert len(result.input_files) == 10


def test_manifest_contains_context_summary_and_no_ssh_secrets(
    tmp_path: Path,
) -> None:
    settings = AnalysisSettings()
    result = analyze_case(CASE, settings)
    remote_metadata = {
        "files": [
            {
                "relative_path": item.relative_path,
                "remote_path": f"/remote/case001/{item.relative_path}",
                "size": item.size,
                "mtime": 123,
            }
            for item in result.input_files
        ]
    }
    remote_manifest = tmp_path / ".mdfoam_remote_manifest.json"
    remote_manifest.write_text(json.dumps(remote_metadata), encoding="utf-8")
    apply_remote_input_paths(result, remote_manifest, "/remote/case001")

    path = tmp_path / "analysis_manifest.json"
    write_analysis_manifest(
        path,
        RunContext(
            input_mode="ssh",
            selected_root="/remote",
            analysis_settings=settings,
            remote_host="cluster.example",
            remote_port=22,
            remote_username="researcher",
        ),
        [result],
    )

    text = path.read_text(encoding="utf-8")
    payload = json.loads(text)
    assert payload["format"] == MANIFEST_FORMAT
    assert payload["app_version"] == __version__
    assert payload["run_context"]["remote"] == {
        "host": "cluster.example",
        "port": 22,
        "username": "researcher",
    }
    assert payload["cases"][0]["source_case_path"] == "/remote/case001"
    assert payload["cases"][0]["mesh_source"].startswith("/remote/case001/")
    assert payload["cases"][0]["result_summary"]["time_count"] == 4
    assert payload["cases"][0]["mesh_statistics"]["cell_count"] == 1
    assert "secret" not in text.lower()
    assert "key_path" not in text
    assert all(
        item["source_path"].startswith("/remote/case001/")
        for item in payload["cases"][0]["input_files"]
    )
