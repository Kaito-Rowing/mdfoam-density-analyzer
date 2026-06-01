from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import csv

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

    @property
    def time_count(self) -> int:
        return len(self.rows)

    @property
    def max_volume(self) -> float:
        return max((row.volume for row in self.rows), default=0.0)

    @property
    def final_volume(self) -> float:
        return self.rows[-1].volume if self.rows else 0.0


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


def analyze_case(
    case_dir: Path,
    settings: AnalysisSettings,
    stop_requested=lambda: False,
    log=lambda message: None,
) -> CaseResult:
    result = CaseResult(case_name=case_dir.name, case_dir=case_dir, status="running")
    main_dir = case_dir / "main"
    if not main_dir.is_dir():
        result.status = "error"
        result.error = "main directory not found"
        return result

    try:
        reconstructed = _analyze_reconstructed(main_dir, settings, stop_requested, log)
        if reconstructed:
            rows, mesh_info, field_class = reconstructed
            result.rows = rows
            result.mesh_source = mesh_info.source
            result.volume_mode = _volume_mode_label(mesh_info)
            result.field_class = field_class
        else:
            rows, source, field_class = _analyze_processors(main_dir, settings, stop_requested, log)
            result.rows = rows
            result.mesh_source = source
            result.volume_mode = "processor meshes"
            result.field_class = field_class

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
    except Exception as exc:
        result.status = "error"
        result.error = str(exc)
    return result


def _analyze_reconstructed(
    main_dir: Path,
    settings: AnalysisSettings,
    stop_requested,
    log,
) -> tuple[list[TimeResult], MeshVolumeInfo, str] | None:
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

    mesh_info = _read_or_build_volumes(main_dir / "constant" / "polyMesh", settings, field_info.value_count)
    rows: list[TimeResult] = []
    for time_value, time_dir in time_dirs:
        if stop_requested():
            break
        densities = read_scalar_internal_field(time_dir / settings.density_field)
        rows.append(_volume_for_time(time_value, densities, mesh_info.volumes, settings))
    log(f"{main_dir.parent.name}: 再構成済み時刻フィールド={len(rows)}")
    return rows, mesh_info, field_info.field_class or ""


def _analyze_processors(
    main_dir: Path,
    settings: AnalysisSettings,
    stop_requested,
    log,
) -> tuple[list[TimeResult], str, str]:
    processors = sorted(path for path in main_dir.glob("processor*") if path.is_dir())
    if not processors:
        return [], "", ""

    proc_data = []
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
        )
        proc_data.append((processor, time_dirs, mesh_info))
        all_times.update(time_dirs.keys())

    rows: list[TimeResult] = []
    for time_value in sorted(all_times):
        if stop_requested():
            break
        volume = 0.0
        selected = 0
        total = 0
        for _, time_dirs, mesh_info in proc_data:
            time_dir = time_dirs.get(time_value)
            if not time_dir:
                continue
            densities = read_scalar_internal_field(time_dir / settings.density_field)
            row = _volume_for_time(time_value, densities, mesh_info.volumes, settings)
            volume += row.volume
            selected += row.selected_cell_count
            total += row.total_cell_count
        rows.append(TimeResult(time_value, volume, equivalent_radius(volume), selected, total))

    log(f"{main_dir.parent.name}: processor分割時刻フィールド={len(rows)}")
    return rows, f"{len(proc_data)} processor meshes", field_class


def _read_or_build_volumes(
    poly_mesh_dir: Path,
    settings: AnalysisSettings,
    expected_cells: int | None,
) -> MeshVolumeInfo:
    try:
        mesh_info = read_mesh_volumes(poly_mesh_dir)
        if expected_cells is not None and len(mesh_info.volumes) != expected_cells:
            raise OpenFoamParseError(
                f"{poly_mesh_dir}: mesh cells={len(mesh_info.volumes)}, field cells={expected_cells}"
            )
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


def _volume_for_time(
    time_value: float,
    densities: list[float],
    cell_volumes: list[float],
    settings: AnalysisSettings,
) -> TimeResult:
    if len(densities) == 1 and len(cell_volumes) != 1:
        densities = densities * len(cell_volumes)
    if len(densities) != len(cell_volumes):
        raise OpenFoamParseError(
            f"Density count {len(densities)} does not match cell volume count {len(cell_volumes)}"
        )

    selected_volume = 0.0
    selected_count = 0
    for density, cell_volume in zip(densities, cell_volumes):
        if density >= settings.density_threshold:
            selected_volume += cell_volume
            selected_count += 1
    return TimeResult(
        time=time_value,
        volume=selected_volume,
        equivalent_radius=equivalent_radius(selected_volume),
        selected_cell_count=selected_count,
        total_cell_count=len(cell_volumes),
    )


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
                "mesh_source",
                "volume_mode",
                "field_class",
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
                    result.mesh_source,
                    result.volume_mode,
                    result.field_class,
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
                    ]
                )
