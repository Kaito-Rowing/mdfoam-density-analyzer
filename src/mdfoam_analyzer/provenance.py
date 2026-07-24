from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from . import __version__
from .analysis import AnalysisSettings, CaseResult


PROJECT_FORMAT = "mdfoam-analysis-settings"
MANIFEST_FORMAT = "mdfoam-analysis-manifest"
SCHEMA_VERSION = 1


class ProvenanceError(ValueError):
    pass


@dataclass(frozen=True)
class RunContext:
    input_mode: str
    selected_root: str
    analysis_settings: AnalysisSettings = AnalysisSettings()
    remote_host: str | None = None
    remote_port: int | None = None
    remote_username: str | None = None


def save_analysis_settings(path: Path, settings: AnalysisSettings) -> None:
    payload = {
        "format": PROJECT_FORMAT,
        "schema_version": SCHEMA_VERSION,
        "app_version": __version__,
        "saved_at_utc": _utc_now(),
        "analysis_settings": asdict(settings),
    }
    _write_json_atomic(path, payload)


def load_analysis_settings(path: Path) -> AnalysisSettings:
    payload = _read_json_object(path)
    _validate_document(payload, PROJECT_FORMAT)
    raw_settings = payload.get("analysis_settings")
    if not isinstance(raw_settings, dict):
        raise ProvenanceError("analysis_settings must be a JSON object")

    defaults = AnalysisSettings()
    values: dict[str, Any] = {}
    for item in fields(AnalysisSettings):
        value = raw_settings.get(item.name, getattr(defaults, item.name))
        values[item.name] = _validate_setting(item.name, value)
    return AnalysisSettings(**values)


def write_analysis_manifest(
    path: Path,
    run_context: RunContext,
    results: list[CaseResult],
) -> None:
    payload = {
        "format": MANIFEST_FORMAT,
        "schema_version": SCHEMA_VERSION,
        "app_version": __version__,
        "generated_at_utc": _utc_now(),
        "analysis_settings": asdict(run_context.analysis_settings),
        "run_context": _run_context_payload(run_context),
        "cases": [_case_payload(result) for result in results],
    }
    _write_json_atomic(path, payload)


def apply_remote_input_paths(
    result: CaseResult,
    remote_manifest_path: Path,
    remote_case: str,
) -> None:
    payload = _read_json_object(remote_manifest_path)
    remote_files = {
        item.get("relative_path"): item
        for item in payload.get("files", [])
        if isinstance(item, dict) and isinstance(item.get("relative_path"), str)
    }
    result.source_case_path = remote_case
    if result.mesh_source not in {"", "manual cell volume"}:
        try:
            relative_mesh = (
                Path(result.mesh_source)
                .resolve()
                .relative_to(result.case_dir.resolve())
                .as_posix()
            )
            result.mesh_source = (
                f"{remote_case.rstrip('/')}/{relative_mesh}"
            )
        except (OSError, ValueError):
            pass
    for record in result.input_files:
        remote = remote_files.get(record.relative_path)
        if remote is None:
            continue
        remote_path = remote.get("remote_path")
        if isinstance(remote_path, str):
            record.source_path = remote_path
        size = remote.get("size")
        if isinstance(size, int) and not isinstance(size, bool):
            record.size = size
        mtime = remote.get("mtime")
        if isinstance(mtime, (int, float)) and not isinstance(mtime, bool):
            record.mtime = float(mtime)


def _run_context_payload(context: RunContext) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "input_mode": context.input_mode,
        "selected_root": context.selected_root,
    }
    if context.input_mode == "ssh":
        payload["remote"] = {
            "host": context.remote_host,
            "port": context.remote_port,
            "username": context.remote_username,
        }
    return payload


def _case_payload(result: CaseResult) -> dict[str, Any]:
    mesh = asdict(result.mesh_statistics) if result.mesh_statistics is not None else None
    departure = result.departure_result
    return {
        "case_name": result.case_name,
        "source_case_path": result.source_case_path,
        "status": result.status,
        "error": result.error,
        "field_class": result.field_class,
        "mesh_source": result.mesh_source,
        "volume_mode": result.volume_mode,
        "input_files": [asdict(item) for item in result.input_files],
        "mesh_statistics": mesh,
        "molecular_departure": (
            None
            if departure is None
            else {
                "status": departure.status,
                "species_name": departure.species_name,
                "bin_mode": departure.bin_mode,
                "species_id": departure.species_id,
                "frame_count": departure.frame_count,
                "raw_event_count": departure.raw_event_count,
                "confirmed_event_count": departure.confirmed_event_count,
                "excluded_normalized_height_count": (
                    departure.excluded_normalized_height_count
                ),
                "fallback_cluster_frame_count": (
                    departure.fallback_cluster_frame_count
                ),
                "warning": departure.warning,
                "error": departure.error,
                "height_bins": [
                    asdict(item) for item in departure.height_bins
                ],
            }
        ),
        "result_summary": {
            "time_count": result.time_count,
            "max_volume": result.max_volume,
            "final_volume": result.final_volume,
            "evaporation_time": result.evaporation_time,
            "initial_contact_angle_deg": result.initial_contact_angle_deg,
            "final_valid_contact_angle_deg": result.final_valid_contact_angle_deg,
            "average_contact_angle_deg": result.average_contact_angle_deg,
            "initial_contact_radius": result.initial_contact_radius,
            "final_valid_contact_radius": result.final_valid_contact_radius,
        },
    }


def _validate_document(payload: dict[str, Any], expected_format: str) -> None:
    if payload.get("format") != expected_format:
        raise ProvenanceError(f"Unsupported file format: {payload.get('format')!r}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ProvenanceError(
            f"Unsupported schema version: {payload.get('schema_version')!r}"
        )


def _validate_setting(name: str, value: Any) -> Any:
    if name in {"density_field", "departure_species"}:
        if not isinstance(value, str) or not value:
            raise ProvenanceError(f"{name} must be a non-empty string")
        return value
    if name == "departure_bin_mode":
        if value not in {"equal_height", "equal_surface_area"}:
            raise ProvenanceError(
                "departure_bin_mode must be equal_height or equal_surface_area"
            )
        return value
    if name in {"contact_unwrap_xy", "departure_enabled"}:
        if not isinstance(value, bool):
            raise ProvenanceError(f"{name} must be boolean")
        return value
    if name in {
        "consecutive_zero_count",
        "departure_confirmation_frames",
        "departure_height_bins",
    }:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ProvenanceError(f"{name} must be an integer")
        return value
    if name in {"manual_cell_volume", "dx", "dy", "dz"} and value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ProvenanceError(f"{name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ProvenanceError(f"{name} must be finite")
    return numeric


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProvenanceError(f"Cannot read JSON file: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProvenanceError("The JSON root must be an object")
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except (OSError, TypeError, ValueError) as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise ProvenanceError(f"Cannot write JSON file: {exc}") from exc


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
