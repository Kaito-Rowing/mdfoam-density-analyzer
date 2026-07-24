from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import csv
import math
import platform
import sys
from typing import TYPE_CHECKING

import numpy as np

from .analysis_cache import AnalysisCacheSession, CacheCancelled, FileFingerprint
from .openfoam import (
    MeshVolumeInfo,
    OpenFoamParseError,
    discover_density_fields,
    equivalent_radius,
    numeric_time_dirs,
    read_field_info,
    read_mesh_volumes,
    read_scalar_internal_field,
)

if TYPE_CHECKING:
    from .molecular_departure import MolecularDepartureResult


ANALYSIS_ALGORITHM_VERSION = 1
DENSITY_PARSER_VERSION = 1
MESH_PARSER_VERSION = 1
MESH_FILE_NAMES = ("points", "faces", "owner", "neighbour")


@dataclass(frozen=True)
class AnalysisSettings:
    density_field: str = "rhoM_water"
    density_threshold: float = 500.0
    zero_tolerance: float = 0.0
    consecutive_zero_count: int = 3
    manual_cell_volume: float | None = None
    dx: float | None = None
    dy: float | None = None
    dz: float | None = None
    contact_fit_lower: float = 0.5
    contact_fit_upper: float = 1.0
    contact_unwrap_xy: bool = True
    contact_average_percent: float = 100.0
    departure_enabled: bool = False
    departure_species: str = "water"
    departure_cutoff: float = 4.0e-10
    departure_confirmation_frames: int = 3
    departure_height_bins: int = 10
    departure_bin_mode: str = "equal_height"

    def fallback_cell_volume(self) -> float | None:
        if self.manual_cell_volume and self.manual_cell_volume > 0:
            return self.manual_cell_volume
        if self.dx and self.dy and self.dz and self.dx > 0 and self.dy > 0 and self.dz > 0:
            return self.dx * self.dy * self.dz
        return None


@dataclass
class TimeResult:
    time: float
    volume: float
    equivalent_radius: float
    selected_cell_count: int
    total_cell_count: int
    contact_angle_deg: float | None = None
    contact_radius: float | None = None
    contact_fit_point_count: int = 0


@dataclass
class InputFileRecord:
    relative_path: str
    source_path: str
    size: int
    mtime: float


@dataclass(frozen=True)
class MeshStatistics:
    mesh_count: int
    cell_count: int
    volume_mode: str
    min_cell_volume: float
    max_cell_volume: float
    total_cell_volume: float
    unique_volume_count: int
    point_bounds: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ] | None
    cell_centers_available: bool


@dataclass(frozen=True)
class ContactFitDiagnostics:
    raw_points: np.ndarray
    points: np.ndarray
    fit_mask: np.ndarray
    z_base: float | None
    sphere_center: tuple[float, float, float] | None
    sphere_radius: float | None
    contact_angle_deg: float | None
    contact_radius: float | None
    fit_point_count: int
    failure_reason: str = ""


@dataclass
class CaseResult:
    case_name: str
    case_dir: Path
    status: str
    rows: list[TimeResult] = field(default_factory=list)
    evaporation_time: float | None = None
    error: str = ""
    mesh_source: str = ""
    volume_mode: str = ""
    field_class: str = ""
    contact_average_percent: float = 100.0
    source_case_path: str = ""
    input_files: list[InputFileRecord] = field(default_factory=list)
    mesh_statistics: MeshStatistics | None = None
    departure_result: MolecularDepartureResult | None = None

    @property
    def time_count(self) -> int:
        return len(self.rows)

    @property
    def max_volume(self) -> float:
        return max((row.volume for row in self.rows), default=0.0)

    @property
    def final_volume(self) -> float:
        return self.rows[-1].volume if self.rows else 0.0

    @property
    def initial_contact_angle_deg(self) -> float | None:
        return self.rows[0].contact_angle_deg if self.rows else None

    @property
    def final_valid_contact_angle_deg(self) -> float | None:
        for row in reversed(self.rows):
            if row.contact_angle_deg is not None:
                return row.contact_angle_deg
        return None

    @property
    def average_contact_angle_deg(self) -> float | None:
        if not self.rows:
            return None
        percent = max(0.0, min(100.0, self.contact_average_percent))
        limit = max(1, math.ceil(len(self.rows) * percent / 100.0))
        values = [
            row.contact_angle_deg
            for row in self.rows[:limit]
            if row.contact_angle_deg is not None
        ]
        if not values:
            return None
        return sum(values) / len(values)

    @property
    def initial_contact_radius(self) -> float | None:
        return self.rows[0].contact_radius if self.rows else None

    @property
    def final_valid_contact_radius(self) -> float | None:
        for row in reversed(self.rows):
            if row.contact_radius is not None:
                return row.contact_radius
        return None


def discover_cases(parent: Path) -> list[Path]:
    parent = parent.resolve()
    if (parent / "main").is_dir():
        return [parent]

    cases = [
        child
        for child in parent.iterdir()
        if child.is_dir() and (child / "main").is_dir()
    ]
    return sorted(cases, key=lambda item: item.name)


def discover_fields_for_cases(cases: list[Path]) -> list[str]:
    names: set[str] = set()
    for case in cases:
        names.update(discover_density_fields(case / "main"))
    if "rhoM_water" in names:
        return ["rhoM_water"] + sorted(name for name in names if name != "rhoM_water")
    return sorted(names)


def _cache_input_paths(main_dir: Path, density_field: str) -> list[Path]:
    reconstructed = [
        time_dir / density_field
        for _, time_dir in numeric_time_dirs(main_dir)
        if (time_dir / density_field).is_file()
    ]
    paths: list[Path] = []
    if reconstructed:
        paths.extend(reconstructed)
        mesh_dir = main_dir / "constant" / "polyMesh"
        paths.extend(
            mesh_dir / name
            for name in MESH_FILE_NAMES
            if (mesh_dir / name).is_file()
        )
    else:
        for processor in sorted(
            path for path in main_dir.glob("processor*") if path.is_dir()
        ):
            density_paths = [
                time_dir / density_field
                for _, time_dir in numeric_time_dirs(processor)
                if (time_dir / density_field).is_file()
            ]
            if not density_paths:
                continue
            paths.extend(density_paths)
            mesh_dir = processor / "constant" / "polyMesh"
            paths.extend(
                mesh_dir / name
                for name in MESH_FILE_NAMES
                if (mesh_dir / name).is_file()
            )
    return sorted({path.resolve() for path in paths}, key=str)


def _result_cache_key(
    cache_session: AnalysisCacheSession,
    case_dir: Path,
    settings: AnalysisSettings,
    fingerprints: dict[Path, FileFingerprint],
) -> str:
    return cache_session.key(
        {
            "algorithm_version": ANALYSIS_ALGORITHM_VERSION,
            "environment": {
                "python_implementation": platform.python_implementation(),
                "python_version": list(sys.version_info[:3]),
                "numpy_version": np.__version__,
            },
            "case_path": str(case_dir.resolve()),
            "settings": asdict(settings),
            "inputs": [
                {
                    "relative_path": path.relative_to(case_dir).as_posix(),
                    "sha256": fingerprint.digest,
                }
                for path, fingerprint in sorted(
                    fingerprints.items(),
                    key=lambda item: str(item[0]),
                )
            ],
        }
    )


def _read_density_values(
    path: Path,
    cache_session: AnalysisCacheSession | None,
    fingerprints: dict[Path, FileFingerprint],
) -> list[float]:
    if cache_session is None:
        return read_scalar_internal_field(path)
    resolved = path.resolve()
    fingerprint = fingerprints.get(resolved)
    if fingerprint is None:
        fingerprint = cache_session.fingerprint(resolved)
        fingerprints[resolved] = fingerprint
    key = cache_session.key(
        {
            "parser_version": DENSITY_PARSER_VERSION,
            "sha256": fingerprint.digest,
        }
    )
    cached = cache_session.load_density(key)
    if cached is not None:
        return cached
    values = read_scalar_internal_field(path)
    cache_session.store_density(key, values)
    return values


def _mesh_cache_payload(
    mesh_info: MeshVolumeInfo,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    centers = (
        np.asarray(mesh_info.cell_centers, dtype=np.float64)
        if mesh_info.cell_centers is not None
        else np.empty((0, 3), dtype=np.float64)
    )
    pairs = (
        np.asarray(mesh_info.neighbour_pairs, dtype=np.int64).reshape((-1, 2))
        if mesh_info.neighbour_pairs is not None
        else np.empty((0, 2), dtype=np.int64)
    )
    bounds = (
        np.asarray(mesh_info.point_bounds, dtype=np.float64)
        if mesh_info.point_bounds is not None
        else np.empty((0, 2), dtype=np.float64)
    )
    return (
        {
            "is_constant": mesh_info.is_constant,
            "unique_volume_count": mesh_info.unique_volume_count,
            "min_volume": mesh_info.min_volume,
            "max_volume": mesh_info.max_volume,
            "total_volume": mesh_info.total_volume,
            "has_cell_centers": mesh_info.cell_centers is not None,
            "has_point_bounds": mesh_info.point_bounds is not None,
            "has_neighbour_pairs": mesh_info.neighbour_pairs is not None,
        },
        {
            "volumes.npy": np.asarray(mesh_info.volumes, dtype=np.float64),
            "cell_centers.npy": centers,
            "point_bounds.npy": bounds,
            "neighbour_pairs.npy": pairs,
        },
    )


def _restore_mesh_info(
    cached: tuple[dict[str, object], dict[str, np.ndarray]],
    poly_mesh_dir: Path,
) -> MeshVolumeInfo:
    metadata, arrays = cached
    volumes = arrays["volumes.npy"]
    centers = arrays["cell_centers.npy"]
    bounds = arrays["point_bounds.npy"]
    pairs = arrays["neighbour_pairs.npy"]
    if volumes.dtype != np.float64 or volumes.ndim != 1 or len(volumes) == 0:
        raise ValueError("invalid cached mesh volumes")
    if centers.dtype != np.float64 or centers.ndim != 2 or centers.shape[1:] != (3,):
        raise ValueError("invalid cached cell centers")
    if bounds.dtype != np.float64 or bounds.ndim != 2 or bounds.shape[1:] != (2,):
        raise ValueError("invalid cached point bounds")
    if pairs.dtype != np.int64 or pairs.ndim != 2 or pairs.shape[1:] != (2,):
        raise ValueError("invalid cached neighbour pairs")

    has_centers = metadata.get("has_cell_centers") is True
    has_bounds = metadata.get("has_point_bounds") is True
    has_pairs = metadata.get("has_neighbour_pairs") is True
    if has_centers and len(centers) != len(volumes):
        raise ValueError("cached cell center count mismatch")
    if has_bounds and bounds.shape != (3, 2):
        raise ValueError("cached point bounds shape mismatch")
    return MeshVolumeInfo(
        volumes=volumes.tolist(),
        source=str(poly_mesh_dir),
        is_constant=bool(metadata["is_constant"]),
        unique_volume_count=int(metadata["unique_volume_count"]),
        min_volume=float(metadata["min_volume"]),
        max_volume=float(metadata["max_volume"]),
        total_volume=float(metadata["total_volume"]),
        cell_centers=(
            [tuple(float(value) for value in row) for row in centers]
            if has_centers
            else None
        ),
        point_bounds=(
            tuple(tuple(float(value) for value in row) for row in bounds)
            if has_bounds
            else None
        ),
        neighbour_pairs=(
            [tuple(int(value) for value in row) for row in pairs]
            if has_pairs
            else None
        ),
    )


def _case_result_cache_payload(
    result: CaseResult,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    float_rows = np.asarray(
        [
            [
                row.time,
                row.volume,
                row.equivalent_radius,
                np.nan if row.contact_angle_deg is None else row.contact_angle_deg,
                np.nan if row.contact_radius is None else row.contact_radius,
            ]
            for row in result.rows
        ],
        dtype=np.float64,
    ).reshape((-1, 5))
    count_rows = np.asarray(
        [
            [
                row.selected_cell_count,
                row.total_cell_count,
                row.contact_fit_point_count,
            ]
            for row in result.rows
        ],
        dtype=np.int64,
    ).reshape((-1, 3))
    mesh = asdict(result.mesh_statistics) if result.mesh_statistics is not None else None
    return (
        {
            "case_name": result.case_name,
            "status": result.status,
            "evaporation_time": result.evaporation_time,
            "error": result.error,
            "mesh_source": result.mesh_source,
            "volume_mode": result.volume_mode,
            "field_class": result.field_class,
            "contact_average_percent": result.contact_average_percent,
            "source_case_path": result.source_case_path,
            "mesh_statistics": mesh,
            "input_relative_paths": [
                record.relative_path for record in result.input_files
            ],
        },
        {
            "float_rows.npy": float_rows,
            "count_rows.npy": count_rows,
        },
    )


def _restore_case_result(
    cached: tuple[dict[str, object], dict[str, np.ndarray]],
    case_dir: Path,
    settings: AnalysisSettings,
    fingerprints: dict[Path, FileFingerprint],
) -> CaseResult:
    metadata, arrays = cached
    float_rows = arrays["float_rows.npy"]
    count_rows = arrays["count_rows.npy"]
    if (
        float_rows.dtype != np.float64
        or float_rows.ndim != 2
        or float_rows.shape[1:] != (5,)
        or count_rows.dtype != np.int64
        or count_rows.ndim != 2
        or count_rows.shape[1:] != (3,)
        or len(float_rows) != len(count_rows)
    ):
        raise ValueError("invalid cached case rows")
    rows = [
        TimeResult(
            time=float(values[0]),
            volume=float(values[1]),
            equivalent_radius=float(values[2]),
            selected_cell_count=int(counts[0]),
            total_cell_count=int(counts[1]),
            contact_angle_deg=None if np.isnan(values[3]) else float(values[3]),
            contact_radius=None if np.isnan(values[4]) else float(values[4]),
            contact_fit_point_count=int(counts[2]),
        )
        for values, counts in zip(float_rows, count_rows)
    ]
    raw_mesh = metadata.get("mesh_statistics")
    mesh_statistics = None
    if raw_mesh is not None:
        if not isinstance(raw_mesh, dict):
            raise ValueError("invalid cached mesh statistics")
        raw_bounds = raw_mesh.get("point_bounds")
        point_bounds = (
            tuple(tuple(float(value) for value in row) for row in raw_bounds)
            if raw_bounds is not None
            else None
        )
        mesh_statistics = MeshStatistics(
            mesh_count=int(raw_mesh["mesh_count"]),
            cell_count=int(raw_mesh["cell_count"]),
            volume_mode=str(raw_mesh["volume_mode"]),
            min_cell_volume=float(raw_mesh["min_cell_volume"]),
            max_cell_volume=float(raw_mesh["max_cell_volume"]),
            total_cell_volume=float(raw_mesh["total_cell_volume"]),
            unique_volume_count=int(raw_mesh["unique_volume_count"]),
            point_bounds=point_bounds,
            cell_centers_available=bool(raw_mesh["cell_centers_available"]),
        )
    relative_paths = metadata.get("input_relative_paths")
    if not isinstance(relative_paths, list) or not all(
        isinstance(item, str) for item in relative_paths
    ):
        raise ValueError("invalid cached input path list")
    input_records: list[InputFileRecord] = []
    for relative_path in relative_paths:
        path = (case_dir / Path(relative_path)).resolve()
        fingerprint = fingerprints.get(path)
        if fingerprint is None:
            raise ValueError(f"cached input is no longer present: {relative_path}")
        input_records.append(
            InputFileRecord(
                relative_path=relative_path,
                source_path=str(path),
                size=fingerprint.size,
                mtime=fingerprint.mtime,
            )
        )
    return CaseResult(
        case_name=case_dir.name,
        case_dir=case_dir,
        status=str(metadata["status"]),
        rows=rows,
        evaporation_time=(
            None
            if metadata.get("evaporation_time") is None
            else float(metadata["evaporation_time"])
        ),
        error=str(metadata.get("error", "")),
        mesh_source=str(metadata.get("mesh_source", "")),
        volume_mode=str(metadata.get("volume_mode", "")),
        field_class=str(metadata.get("field_class", "")),
        contact_average_percent=settings.contact_average_percent,
        source_case_path=str(case_dir),
        input_files=input_records,
        mesh_statistics=mesh_statistics,
    )


def analyze_case(
    case_dir: Path,
    settings: AnalysisSettings,
    stop_requested=lambda: False,
    log=lambda message: None,
    cache_session: AnalysisCacheSession | None = None,
) -> CaseResult:
    case_dir = case_dir.resolve()
    result = CaseResult(
        case_name=case_dir.name,
        case_dir=case_dir,
        status="running",
        contact_average_percent=settings.contact_average_percent,
        source_case_path=str(case_dir.resolve()),
    )
    main_dir = case_dir / "main"
    if not main_dir.is_dir():
        result.status = "error"
        result.error = "main directory not found"
        return result

    if cache_session is not None:
        cache_session.begin_case()
    try:
        fingerprints: dict[Path, FileFingerprint] = {}
        result_cache_key: str | None = None
        cache_hits_before = cache_session.hit_counts() if cache_session else (0, 0)
        use_complete_result_cache = (
            cache_session is not None and not settings.departure_enabled
        )
        if cache_session is not None:
            input_paths = _cache_input_paths(main_dir, settings.density_field)
            fingerprints = cache_session.fingerprints(input_paths, stop_requested)
            if stop_requested():
                result.status = "stopped"
                cache_session.rollback_case()
                return result
            result_cache_key = _result_cache_key(
                cache_session,
                case_dir,
                settings,
                fingerprints,
            )
            cached_result = (
                cache_session.load_result(result_cache_key)
                if use_complete_result_cache
                else None
            )
            if cached_result is not None:
                try:
                    restored = _restore_case_result(
                        cached_result,
                        case_dir,
                        settings,
                        fingerprints,
                    )
                except Exception as exc:
                    cache_session.invalidate("result", result_cache_key, exc)
                else:
                    log(f"{case_dir.name}: local analysis cache hit")
                    cache_session.commit_case()
                    return restored
            log(f"{case_dir.name}: local analysis cache miss")

        reconstructed = _analyze_reconstructed(
            main_dir,
            settings,
            stop_requested,
            log,
            cache_session,
            fingerprints,
        )
        if reconstructed:
            rows, mesh_info, field_class, input_paths = reconstructed
            result.rows = rows
            result.mesh_source = mesh_info.source
            result.volume_mode = _volume_mode_label(mesh_info)
            result.field_class = field_class
            result.mesh_statistics = _mesh_statistics([mesh_info], result.volume_mode)
        else:
            rows, source, field_class, input_paths, mesh_infos = _analyze_processors(
                main_dir,
                settings,
                stop_requested,
                log,
                cache_session,
                fingerprints,
            )
            result.rows = rows
            result.mesh_source = source
            result.volume_mode = "processor meshes"
            result.field_class = field_class
            if mesh_infos:
                result.mesh_statistics = _mesh_statistics(
                    mesh_infos,
                    _processor_volume_mode(mesh_infos),
                )

        result.input_files = _input_file_records(case_dir, input_paths)

        if stop_requested():
            result.status = "stopped"
        elif not result.rows:
            result.status = "error"
            result.error = f"No readable {settings.density_field} time fields found"
        else:
            result.evaporation_time = find_evaporation_time(
                result.rows,
                settings.zero_tolerance,
                settings.consecutive_zero_count,
            )
            result.status = "ok"
            if settings.departure_enabled:
                from .molecular_departure import analyze_molecular_departures

                departure_result = analyze_molecular_departures(
                    case_dir,
                    settings,
                    result.evaporation_time,
                    stop_requested=stop_requested,
                    log=log,
                )
                result.departure_result = departure_result
                if departure_result.warning:
                    log(
                        f"{case_dir.name}: molecular departure warning: "
                        f"{departure_result.warning}"
                    )
                if departure_result.status == "error":
                    log(
                        f"{case_dir.name}: molecular departure error: "
                        f"{departure_result.error}"
                    )
                result.input_files = _merge_input_file_records(
                    result.input_files,
                    _input_file_records(
                        case_dir,
                        departure_result.input_paths,
                    ),
                )
                if departure_result.status == "stopped":
                    result.status = "stopped"
            if (
                result.status == "ok"
                and use_complete_result_cache
                and cache_session is not None
                and result_cache_key is not None
            ):
                metadata, arrays = _case_result_cache_payload(result)
                cache_session.store_result(result_cache_key, metadata, arrays)
                density_before, mesh_before = cache_hits_before
                density_after, mesh_after = cache_session.hit_counts()
                if density_after > density_before or mesh_after > mesh_before:
                    log(
                        f"{case_dir.name}: local analysis cache partial hit "
                        f"(density={density_after - density_before}, "
                        f"mesh={mesh_after - mesh_before})"
                    )
    except CacheCancelled:
        result.status = "stopped"
    except Exception as exc:
        result.status = "error"
        result.error = str(exc)
    if cache_session is not None:
        if result.status == "ok":
            cache_session.commit_case()
        else:
            cache_session.rollback_case()
    return result


def _analyze_reconstructed(
    main_dir: Path,
    settings: AnalysisSettings,
    stop_requested,
    log,
    cache_session: AnalysisCacheSession | None = None,
    fingerprints: dict[Path, FileFingerprint] | None = None,
) -> tuple[list[TimeResult], MeshVolumeInfo, str, list[Path]] | None:
    fingerprints = fingerprints or {}
    time_dirs = [
        (time_value, time_dir)
        for time_value, time_dir in numeric_time_dirs(main_dir)
        if (time_dir / settings.density_field).is_file()
    ]
    if not time_dirs:
        return None

    first_field = time_dirs[0][1] / settings.density_field
    field_info = read_field_info(first_field)
    if not field_info.is_cell_field:
        raise OpenFoamParseError(
            f"{settings.density_field} is {field_info.field_class}; expected volScalarField"
        )

    poly_mesh_dir = main_dir / "constant" / "polyMesh"
    mesh_info = _read_or_build_volumes(
        poly_mesh_dir,
        settings,
        field_info.value_count,
        cache_session,
        fingerprints,
    )
    input_paths = _mesh_input_paths(poly_mesh_dir, mesh_info)
    rows: list[TimeResult] = []
    for time_value, time_dir in time_dirs:
        if stop_requested():
            break
        density_path = time_dir / settings.density_field
        densities = _read_density_values(
            density_path,
            cache_session,
            fingerprints,
        )
        input_paths.append(density_path)
        rows.append(
            _volume_for_time(
                time_value,
                densities,
                mesh_info,
                settings,
            )
        )
    log(f"{main_dir.parent.name}: 再構成済み時刻フィールド={len(rows)}")
    return rows, mesh_info, field_info.field_class or "", input_paths


def _analyze_processors(
    main_dir: Path,
    settings: AnalysisSettings,
    stop_requested,
    log,
    cache_session: AnalysisCacheSession | None = None,
    fingerprints: dict[Path, FileFingerprint] | None = None,
) -> tuple[list[TimeResult], str, str, list[Path], list[MeshVolumeInfo]]:
    fingerprints = fingerprints or {}
    processors = sorted(path for path in main_dir.glob("processor*") if path.is_dir())
    if not processors:
        return [], "", "", [], []

    proc_data = []
    input_paths: list[Path] = []
    mesh_infos: list[MeshVolumeInfo] = []
    all_times: set[float] = set()
    field_class = ""
    for processor in processors:
        time_dirs = {
            time_value: time_dir
            for time_value, time_dir in numeric_time_dirs(processor)
            if (time_dir / settings.density_field).is_file()
        }
        if not time_dirs:
            continue
        first_field = next(iter(time_dirs.values())) / settings.density_field
        field_info = read_field_info(first_field)
        if not field_info.is_cell_field:
            raise OpenFoamParseError(
                f"{first_field}: {field_info.field_class}; expected volScalarField"
            )
        field_class = field_info.field_class or ""
        mesh_info = _read_or_build_volumes(
            processor / "constant" / "polyMesh",
            settings,
            field_info.value_count,
            cache_session,
            fingerprints,
        )
        proc_data.append((processor, time_dirs, mesh_info))
        mesh_infos.append(mesh_info)
        input_paths.extend(
            _mesh_input_paths(processor / "constant" / "polyMesh", mesh_info)
        )
        all_times.update(time_dirs.keys())

    rows: list[TimeResult] = []
    for time_value in sorted(all_times):
        if stop_requested():
            break
        volume = 0.0
        selected = 0
        total = 0
        contact_points: list[tuple[float, float, float]] = []
        point_bounds = _combined_point_bounds(
            mesh_info.point_bounds for _, _, mesh_info in proc_data
        )
        for _, time_dirs, mesh_info in proc_data:
            time_dir = time_dirs.get(time_value)
            if not time_dir:
                continue
            density_path = time_dir / settings.density_field
            densities = _read_density_values(
                density_path,
                cache_session,
                fingerprints,
            )
            input_paths.append(density_path)
            stats = _selected_volume_stats(
                densities,
                mesh_info.volumes,
                settings,
                mesh_info.cell_centers,
            )
            volume += stats.volume
            selected += stats.selected_count
            total += stats.total_count
            contour_points = density_contour_points(
                densities,
                mesh_info,
                settings.density_threshold,
            )
            if contour_points:
                contact_points.extend(contour_points)
        contact_angle, contact_radius, fit_count = _contact_metrics(
            contact_points,
            point_bounds,
            settings,
        )
        rows.append(
            TimeResult(
                time_value,
                volume,
                equivalent_radius(volume),
                selected,
                total,
                contact_angle,
                contact_radius,
                fit_count,
            )
        )

    log(f"{main_dir.parent.name}: processor分割時刻フィールド={len(rows)}")
    return (
        rows,
        f"{len(proc_data)} processor meshes",
        field_class,
        input_paths,
        mesh_infos,
    )


def _read_or_build_volumes(
    poly_mesh_dir: Path,
    settings: AnalysisSettings,
    expected_cells: int | None,
    cache_session: AnalysisCacheSession | None = None,
    fingerprints: dict[Path, FileFingerprint] | None = None,
) -> MeshVolumeInfo:
    fingerprints = fingerprints or {}
    mesh_paths = [
        poly_mesh_dir / name
        for name in MESH_FILE_NAMES
        if (poly_mesh_dir / name).is_file()
    ]
    mesh_key: str | None = None
    if cache_session is not None and mesh_paths:
        mesh_key = cache_session.key(
            {
                "parser_version": MESH_PARSER_VERSION,
                "expected_cells": expected_cells,
                "files": [
                    {
                        "name": path.name,
                        "sha256": fingerprints[path.resolve()].digest,
                    }
                    for path in mesh_paths
                ],
            }
        )
        cached_mesh = cache_session.load_mesh(mesh_key)
        if cached_mesh is not None:
            try:
                mesh_info = _restore_mesh_info(cached_mesh, poly_mesh_dir)
                if expected_cells is not None and len(mesh_info.volumes) != expected_cells:
                    raise ValueError("cached mesh cell count mismatch")
            except Exception as exc:
                cache_session.invalidate("mesh", mesh_key, exc)
            else:
                return mesh_info

    try:
        mesh_info = read_mesh_volumes(poly_mesh_dir)
        if expected_cells is not None and len(mesh_info.volumes) != expected_cells:
            raise OpenFoamParseError(
                f"{poly_mesh_dir}: mesh cells={len(mesh_info.volumes)}, field cells={expected_cells}"
            )
        if cache_session is not None and mesh_key is not None:
            metadata, arrays = _mesh_cache_payload(mesh_info)
            cache_session.store_mesh(mesh_key, metadata, arrays)
        return mesh_info
    except Exception as mesh_error:
        fallback = settings.fallback_cell_volume()
        if fallback is None or expected_cells is None:
            raise OpenFoamParseError(
                f"Cell volumes unavailable from mesh and no manual volume was provided: {mesh_error}"
            ) from mesh_error
        volumes = [fallback] * expected_cells
        return MeshVolumeInfo(
            volumes=volumes,
            source="manual cell volume",
            is_constant=True,
            unique_volume_count=1,
            min_volume=fallback,
            max_volume=fallback,
            total_volume=fallback * expected_cells,
        )


def _volume_mode_label(mesh_info: MeshVolumeInfo) -> str:
    if mesh_info.source == "manual cell volume":
        return "manual constant cell volume"
    if mesh_info.is_constant:
        return "mesh constant cell volume"
    return f"mesh per-cell volumes ({mesh_info.unique_volume_count} unique)"


def _processor_volume_mode(mesh_infos: list[MeshVolumeInfo]) -> str:
    manual_count = sum(
        info.source == "manual cell volume" for info in mesh_infos
    )
    if manual_count == len(mesh_infos):
        return "processor manual constant cell volumes"
    if manual_count:
        return "processor mixed mesh and manual cell volumes"
    if all(info.is_constant for info in mesh_infos):
        return "processor mesh constant cell volumes"
    return "processor mesh per-cell volumes"


def _mesh_input_paths(poly_mesh_dir: Path, mesh_info: MeshVolumeInfo) -> list[Path]:
    if mesh_info.source == "manual cell volume":
        return []
    return [
        path
        for path in (
            poly_mesh_dir / "points",
            poly_mesh_dir / "faces",
            poly_mesh_dir / "owner",
            poly_mesh_dir / "neighbour",
        )
        if path.is_file()
    ]


def _input_file_records(case_dir: Path, paths: list[Path]) -> list[InputFileRecord]:
    records: list[InputFileRecord] = []
    seen: set[Path] = set()
    resolved_case = case_dir.resolve()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        stat = resolved.stat()
        records.append(
            InputFileRecord(
                relative_path=resolved.relative_to(resolved_case).as_posix(),
                source_path=str(resolved),
                size=stat.st_size,
                mtime=stat.st_mtime,
            )
        )
    return sorted(records, key=lambda item: item.relative_path)


def _merge_input_file_records(
    left: list[InputFileRecord],
    right: list[InputFileRecord],
) -> list[InputFileRecord]:
    records = {record.relative_path: record for record in left}
    records.update({record.relative_path: record for record in right})
    return [records[key] for key in sorted(records)]


def _mesh_statistics(
    mesh_infos: list[MeshVolumeInfo],
    volume_mode: str,
) -> MeshStatistics:
    all_volumes = [volume for info in mesh_infos for volume in info.volumes]
    return MeshStatistics(
        mesh_count=len(mesh_infos),
        cell_count=len(all_volumes),
        volume_mode=volume_mode,
        min_cell_volume=min(all_volumes),
        max_cell_volume=max(all_volumes),
        total_cell_volume=sum(all_volumes),
        unique_volume_count=len({round(value, 34) for value in all_volumes}),
        point_bounds=_combined_point_bounds(
            info.point_bounds for info in mesh_infos
        ),
        cell_centers_available=all(
            info.cell_centers is not None for info in mesh_infos
        ),
    )


@dataclass(frozen=True)
class _SelectedVolumeStats:
    volume: float
    selected_count: int
    total_count: int
    selected_centers: list[tuple[float, float, float]] | None


def _volume_for_time(
    time_value: float,
    densities: list[float],
    mesh_info: MeshVolumeInfo,
    settings: AnalysisSettings,
) -> TimeResult:
    stats = _selected_volume_stats(
        densities,
        mesh_info.volumes,
        settings,
        mesh_info.cell_centers,
    )
    contact_points = density_contour_points(densities, mesh_info, settings.density_threshold)
    contact_angle, contact_radius, fit_count = _contact_metrics(
        contact_points,
        mesh_info.point_bounds,
        settings,
    )
    return TimeResult(
        time=time_value,
        volume=stats.volume,
        equivalent_radius=equivalent_radius(stats.volume),
        selected_cell_count=stats.selected_count,
        total_cell_count=stats.total_count,
        contact_angle_deg=contact_angle,
        contact_radius=contact_radius,
        contact_fit_point_count=fit_count,
    )


def _selected_volume_stats(
    densities: list[float],
    cell_volumes: list[float],
    settings: AnalysisSettings,
    cell_centers: list[tuple[float, float, float]] | None = None,
) -> _SelectedVolumeStats:
    densities = _expanded_densities(densities, len(cell_volumes))
    if cell_centers is not None and len(cell_centers) != len(cell_volumes):
        raise OpenFoamParseError(
            f"Cell center count {len(cell_centers)} does not match cell volume count {len(cell_volumes)}"
        )

    selected_centers: list[tuple[float, float, float]] | None
    selected_centers = [] if cell_centers is not None else None
    selected_volume = 0.0
    selected_count = 0
    for cell_index, (density, cell_volume) in enumerate(zip(densities, cell_volumes)):
        if density >= settings.density_threshold:
            selected_volume += cell_volume
            selected_count += 1
            if selected_centers is not None and cell_centers is not None:
                selected_centers.append(cell_centers[cell_index])
    return _SelectedVolumeStats(
        volume=selected_volume,
        selected_count=selected_count,
        total_count=len(cell_volumes),
        selected_centers=selected_centers,
    )


def _expanded_densities(densities: list[float], cell_count: int) -> list[float]:
    if len(densities) == 1 and cell_count != 1:
        return densities * cell_count
    if len(densities) != cell_count:
        raise OpenFoamParseError(
            f"Density count {len(densities)} does not match cell volume count {cell_count}"
        )
    return densities


def density_contour_points(
    densities: list[float],
    mesh_info: MeshVolumeInfo,
    threshold: float,
) -> list[tuple[float, float, float]] | None:
    if mesh_info.cell_centers is None or mesh_info.neighbour_pairs is None:
        return None

    densities = _expanded_densities(densities, len(mesh_info.volumes))
    centers = np.asarray(mesh_info.cell_centers, dtype=float)
    contour_points: list[tuple[float, float, float]] = []
    for owner_cell, neighbour_cell in mesh_info.neighbour_pairs:
        owner_value = densities[owner_cell]
        neighbour_value = densities[neighbour_cell]
        delta_owner = owner_value - threshold
        delta_neighbour = neighbour_value - threshold
        if delta_owner == 0.0 and delta_neighbour == 0.0:
            continue
        if delta_owner * delta_neighbour > 0.0:
            continue

        denominator = neighbour_value - owner_value
        if denominator == 0.0:
            continue
        fraction = (threshold - owner_value) / denominator
        if fraction < -1.0e-12 or fraction > 1.0 + 1.0e-12:
            continue
        fraction = max(0.0, min(1.0, fraction))
        point = centers[owner_cell] + fraction * (
            centers[neighbour_cell] - centers[owner_cell]
        )
        contour_points.append((float(point[0]), float(point[1]), float(point[2])))
    return contour_points


def _combined_point_bounds(
    bounds_list,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None:
    valid_bounds = [bounds for bounds in bounds_list if bounds is not None]
    if not valid_bounds:
        return None
    return tuple(
        (
            min(bounds[axis][0] for bounds in valid_bounds),
            max(bounds[axis][1] for bounds in valid_bounds),
        )
        for axis in range(3)
    )


def _contact_metrics(
    selected_centers: list[tuple[float, float, float]] | None,
    point_bounds: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ] | None,
    settings: AnalysisSettings,
) -> tuple[float | None, float | None, int]:
    diagnostics = contact_fit_diagnostics(selected_centers, point_bounds, settings)
    return (
        diagnostics.contact_angle_deg,
        diagnostics.contact_radius,
        diagnostics.fit_point_count,
    )


def contact_fit_diagnostics(
    selected_centers: list[tuple[float, float, float]] | None,
    point_bounds: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ] | None,
    settings: AnalysisSettings,
) -> ContactFitDiagnostics:
    if not selected_centers:
        empty = np.empty((0, 3), dtype=float)
        return ContactFitDiagnostics(empty, empty, np.zeros(0, dtype=bool), None, None, None, None, None, 0, "液滴点がありません")

    raw_points = np.asarray(selected_centers, dtype=float)
    if len(raw_points) < 4:
        return ContactFitDiagnostics(
            raw_points,
            raw_points,
            np.zeros(len(raw_points), dtype=bool),
            None,
            None,
            None,
            None,
            None,
            0,
            "液滴点が4点未満です",
        )

    points = raw_points.copy()
    if settings.contact_unwrap_xy and point_bounds is not None:
        points = _unwrap_xy(points, point_bounds)

    z_values = points[:, 2]
    z_min = float(np.min(z_values))
    z_max = float(np.max(z_values))
    z_height = z_max - z_min
    if z_height <= 0.0:
        return ContactFitDiagnostics(raw_points, points, np.zeros(len(points), dtype=bool), None, None, None, None, None, 0, "z方向高さが0です")

    z_base_threshold = z_min + 0.05 * z_height
    base_z = z_values[z_values <= z_base_threshold]
    z_base = float(np.mean(base_z)) if len(base_z) > 0 else z_min

    fit_lower, fit_upper = _normalized_fit_range(settings)
    z_fit_lower = z_min + fit_lower * z_height
    z_fit_upper = z_min + fit_upper * z_height
    fit_mask = (z_values >= z_fit_lower) & (z_values <= z_fit_upper)
    fit_count = int(np.count_nonzero(fit_mask))
    if fit_count < 4:
        return ContactFitDiagnostics(raw_points, points, fit_mask, z_base, None, None, None, None, fit_count, "fit点が4点未満です")

    sphere = _fit_sphere_algebraic(points[fit_mask])
    if sphere is None:
        return ContactFitDiagnostics(raw_points, points, fit_mask, z_base, None, None, None, None, fit_count, "球フィットに失敗しました")

    x_center, y_center, z_center, radius = sphere
    if radius < 1.0e-12:
        return ContactFitDiagnostics(raw_points, points, fit_mask, z_base, (x_center, y_center, z_center), radius, None, None, fit_count, "球半径が小さすぎます")

    cos_arg = (z_base - z_center) / radius
    cos_arg = max(-1.0, min(1.0, cos_arg))
    theta_rad = math.acos(cos_arg)
    contact_angle_deg = math.degrees(theta_rad)
    contact_radius = radius * math.sin(theta_rad)
    return ContactFitDiagnostics(
        raw_points,
        points,
        fit_mask,
        z_base,
        (x_center, y_center, z_center),
        radius,
        contact_angle_deg,
        contact_radius,
        fit_count,
    )


def _normalized_fit_range(settings: AnalysisSettings) -> tuple[float, float]:
    lower = max(0.0, min(1.0, settings.contact_fit_lower))
    upper = max(0.0, min(1.0, settings.contact_fit_upper))
    if lower > upper:
        lower, upper = upper, lower
    return lower, upper


def _fit_sphere_algebraic(
    points: np.ndarray,
) -> tuple[float, float, float, float] | None:
    if len(points) < 4:
        return None
    try:
        x_values = points[:, 0]
        y_values = points[:, 1]
        z_values = points[:, 2]
        matrix = np.c_[x_values, y_values, z_values, np.ones(len(points))]
        vector = -(x_values**2 + y_values**2 + z_values**2)
        coefficients, *_ = np.linalg.lstsq(matrix, vector, rcond=None)
        d_value, e_value, f_value, g_value = coefficients
        x_center = float(-d_value / 2.0)
        y_center = float(-e_value / 2.0)
        z_center = float(-f_value / 2.0)
        radius_term = x_center**2 + y_center**2 + z_center**2 - float(g_value)
        if not math.isfinite(radius_term):
            return None
        radius = math.sqrt(max(0.0, radius_term))
        return x_center, y_center, z_center, radius
    except Exception:
        return None


def _unwrap_xy(
    points: np.ndarray,
    point_bounds: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ],
) -> np.ndarray:
    unwrapped = points.copy()
    for axis in (0, 1):
        minimum, maximum = point_bounds[axis]
        period = maximum - minimum
        if period <= 0.0:
            continue
        center = _circular_mean(unwrapped[:, axis], minimum, period)
        unwrapped[:, axis] = center + (
            (unwrapped[:, axis] - center + period / 2.0) % period - period / 2.0
        )
    return unwrapped


def _circular_mean(values: np.ndarray, minimum: float, period: float) -> float:
    angles = 2.0 * math.pi * ((values - minimum) % period) / period
    sin_mean = float(np.mean(np.sin(angles)))
    cos_mean = float(np.mean(np.cos(angles)))
    if abs(sin_mean) + abs(cos_mean) < 1.0e-12:
        return float(np.mean(values))
    angle = math.atan2(sin_mean, cos_mean)
    if angle < 0.0:
        angle += 2.0 * math.pi
    return minimum + period * angle / (2.0 * math.pi)


def find_evaporation_time(
    rows: list[TimeResult],
    zero_tolerance: float,
    consecutive_zero_count: int,
) -> float | None:
    if consecutive_zero_count <= 0:
        consecutive_zero_count = 1

    zero_run_start: float | None = None
    zero_run_count = 0
    for row in rows:
        if row.volume <= zero_tolerance:
            if zero_run_count == 0:
                zero_run_start = row.time
            zero_run_count += 1
            if zero_run_count >= consecutive_zero_count:
                return zero_run_start
        else:
            zero_run_start = None
            zero_run_count = 0
    return None


def write_summary_csv(path: Path, results: list[CaseResult]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "case",
                "status",
                "time_count",
                "max_volume",
                "final_volume",
                "evaporation_time",
                "initial_contact_angle_deg",
                "final_valid_contact_angle_deg",
                "average_contact_angle_deg",
                "contact_average_percent",
                "initial_contact_radius",
                "final_valid_contact_radius",
                "mesh_source",
                "volume_mode",
                "field_class",
                "departure_status",
                "departure_raw_event_count",
                "departure_confirmed_event_count",
                "departure_excluded_height_count",
                "error",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.case_name,
                    result.status,
                    result.time_count,
                    result.max_volume,
                    result.final_volume,
                    "" if result.evaporation_time is None else result.evaporation_time,
                    _csv_optional(result.initial_contact_angle_deg),
                    _csv_optional(result.final_valid_contact_angle_deg),
                    _csv_optional(result.average_contact_angle_deg),
                    result.contact_average_percent,
                    _csv_optional(result.initial_contact_radius),
                    _csv_optional(result.final_valid_contact_radius),
                    result.mesh_source,
                    result.volume_mode,
                    result.field_class,
                    (
                        ""
                        if result.departure_result is None
                        else result.departure_result.status
                    ),
                    (
                        ""
                        if result.departure_result is None
                        else result.departure_result.raw_event_count
                    ),
                    (
                        ""
                        if result.departure_result is None
                        else result.departure_result.confirmed_event_count
                    ),
                    (
                        ""
                        if result.departure_result is None
                        else result.departure_result.excluded_normalized_height_count
                    ),
                    result.error,
                ]
            )


def write_timeseries_csv(path: Path, results: list[CaseResult]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "case",
                "time",
                "volume",
                "equivalent_radius",
                "selected_cell_count",
                "total_cell_count",
                "contact_angle_deg",
                "contact_radius",
                "contact_fit_point_count",
            ]
        )
        for result in results:
            for row in result.rows:
                writer.writerow(
                    [
                        result.case_name,
                        row.time,
                        row.volume,
                        row.equivalent_radius,
                        row.selected_cell_count,
                        row.total_cell_count,
                        _csv_optional(row.contact_angle_deg),
                        _csv_optional(row.contact_radius),
                        row.contact_fit_point_count,
                    ]
                )


def _csv_optional(value: float | None) -> float | str:
    return "" if value is None else value
