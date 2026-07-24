from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
import csv
import math
import re
from typing import Callable

import numpy as np

from .openfoam import (
    MeshVolumeInfo,
    NUMBER_RE,
    OpenFoamParseError,
    numeric_time_dirs,
    read_mesh_volumes,
    read_scalar_internal_field,
    strip_comments,
)


LAGRANGIAN_FILE_NAMES = ("positions", "id", "origId", "origProcId")
LOW_CADENCE_RECOMMENDATION = 1.0e-13


@dataclass(frozen=True)
class DepartureEvent:
    event_index: int
    orig_proc_id: int
    orig_id: int
    species_id: int
    species_name: str
    last_liquid_time: float
    first_outside_time: float
    event_time: float
    time_uncertainty: float
    last_liquid_x: float
    last_liquid_y: float
    last_liquid_z: float
    first_outside_x: float | None
    first_outside_y: float | None
    first_outside_z: float | None
    departure_x: float
    departure_y: float
    departure_z: float
    position_method: str
    height: float | None
    normalized_height: float | None
    normalized_radius: float | None
    cluster_size: int
    confirmed: bool
    confirmation_time: float | None
    confirmation_frames_observed: int
    quality_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class DepartureHeightBin:
    bin_index: int
    eta_lower: float
    eta_upper: float
    raw_count: int
    confirmed_count: int
    area_time_exposure: float
    raw_rate: float | None
    confirmed_rate: float | None


@dataclass
class MolecularDepartureResult:
    status: str
    species_name: str
    bin_mode: str = "equal_height"
    species_id: int | None = None
    frame_count: int = 0
    events: list[DepartureEvent] = field(default_factory=list)
    height_bins: list[DepartureHeightBin] = field(default_factory=list)
    excluded_normalized_height_count: int = 0
    fallback_cluster_frame_count: int = 0
    warning: str = ""
    error: str = ""
    input_paths: list[Path] = field(default_factory=list)

    @property
    def raw_event_count(self) -> int:
        return len(self.events)

    @property
    def confirmed_event_count(self) -> int:
        return sum(event.confirmed for event in self.events)


@dataclass(frozen=True)
class LagrangianPositions:
    positions: np.ndarray
    cell_indices: np.ndarray


@dataclass(frozen=True)
class _Geometry:
    z_base: float | None
    z_top: float | None
    sphere_center: tuple[float, float, float] | None
    sphere_radius: float | None
    contact_radius: float | None

    @property
    def height(self) -> float | None:
        if self.z_base is None or self.z_top is None:
            return None
        height = self.z_top - self.z_base
        return height if height > 0.0 else None


@dataclass(frozen=True)
class _Frame:
    time: float
    time_name: str
    positions: np.ndarray
    orig_proc_ids: np.ndarray
    orig_ids: np.ndarray
    liquid_mask: np.ndarray
    cluster_size: int
    geometry: _Geometry
    cluster_fallback: bool
    point_bounds: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ] | None


@dataclass(frozen=True)
class _Observation:
    frame_index: int
    position: np.ndarray
    liquid: bool
    cluster_size: int


def read_lagrangian_positions_with_cells(path: Path) -> LagrangianPositions:
    declared_count, body = _read_counted_body(path)
    pattern = (
        rf"\(\s*({NUMBER_RE})\s+({NUMBER_RE})\s+({NUMBER_RE})\s*\)"
        r"\s+(-?\d+)"
    )
    rows = re.findall(pattern, body)
    if len(rows) != declared_count:
        raise OpenFoamParseError(
            f"{path}: declared {declared_count} positions, parsed {len(rows)}"
        )
    positions = np.asarray(
        [(float(x), float(y), float(z)) for x, y, z, _ in rows],
        dtype=float,
    ).reshape((-1, 3))
    cell_indices = np.asarray([int(cell) for *_, cell in rows], dtype=np.int64)
    return LagrangianPositions(positions, cell_indices)


def read_label_field(path: Path) -> np.ndarray:
    text = strip_comments(path.read_text(errors="ignore"))
    compact = re.search(r"\b(\d+)\s*\{\s*(-?\d+)\s*\}\s*;?\s*$", text, re.S)
    if compact:
        return np.full(
            int(compact.group(1)),
            int(compact.group(2)),
            dtype=np.int64,
        )
    declared_count, body = _read_counted_body_text(text, path)
    values = np.asarray(
        [int(item) for item in re.findall(r"-?\d+", body)],
        dtype=np.int64,
    )
    if len(values) != declared_count:
        raise OpenFoamParseError(
            f"{path}: declared {declared_count} labels, parsed {len(values)}"
        )
    return values


def read_molecule_id_list(path: Path) -> list[str]:
    text = strip_comments(path.read_text(errors="ignore"))
    match = re.search(r"\bidList\s+\d+\s*\((.*?)\)\s*;", text, re.S)
    if not match:
        raise OpenFoamParseError(f"Cannot parse idList: {path}")
    return [token for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", match.group(1))]


def largest_seeded_periodic_cluster(
    positions: np.ndarray,
    seed_mask: np.ndarray,
    cutoff: float,
    point_bounds: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ] | None,
) -> tuple[np.ndarray, int, bool]:
    positions = np.asarray(positions, dtype=float)
    seed_mask = np.asarray(seed_mask, dtype=bool)
    count = len(positions)
    if count == 0:
        return np.zeros(0, dtype=bool), 0, False
    if len(seed_mask) != count:
        raise ValueError("seed_mask length does not match positions")
    if cutoff <= 0.0 or not math.isfinite(cutoff):
        raise ValueError("cluster cutoff must be positive and finite")

    parent = np.arange(count, dtype=np.int64)
    component_sizes = np.ones(count, dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if component_sizes[left_root] < component_sizes[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        component_sizes[left_root] += component_sizes[right_root]

    x_min, x_period, x_bin_count = _periodic_axis_bins(
        point_bounds, 0, cutoff, positions[:, 0]
    )
    y_min, y_period, y_bin_count = _periodic_axis_bins(
        point_bounds, 1, cutoff, positions[:, 1]
    )
    z_min = float(np.min(positions[:, 2]))
    x_bins = _periodic_bin_indices(
        positions[:, 0], x_min, x_period, x_bin_count
    )
    y_bins = _periodic_bin_indices(
        positions[:, 1], y_min, y_period, y_bin_count
    )
    z_bins = np.floor((positions[:, 2] - z_min) / cutoff).astype(np.int64)

    buckets: dict[tuple[int, int, int], list[int]] = {}
    for index, key in enumerate(zip(x_bins, y_bins, z_bins)):
        buckets.setdefault(tuple(int(value) for value in key), []).append(index)

    cutoff_squared = cutoff * cutoff
    for key, indices in buckets.items():
        x_bin, y_bin, z_bin = key
        neighbour_keys = {
            (
                (x_bin + dx) % x_bin_count,
                (y_bin + dy) % y_bin_count,
                z_bin + dz,
            )
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
        }
        for neighbour_key in neighbour_keys:
            neighbour_indices = buckets.get(neighbour_key)
            if not neighbour_indices:
                continue
            for left in indices:
                for right in neighbour_indices:
                    if right <= left:
                        continue
                    delta = positions[right] - positions[left]
                    if x_period > 0.0:
                        delta[0] = (
                            (delta[0] + x_period / 2.0) % x_period
                            - x_period / 2.0
                        )
                    if y_period > 0.0:
                        delta[1] = (
                            (delta[1] + y_period / 2.0) % y_period
                            - y_period / 2.0
                        )
                    if float(np.dot(delta, delta)) <= cutoff_squared:
                        union(left, right)

    roots = np.asarray([find(index) for index in range(count)], dtype=np.int64)
    unique_roots, sizes = np.unique(roots, return_counts=True)
    seeded_roots = set(roots[seed_mask].tolist())
    candidates = [
        (int(size), int(root))
        for root, size in zip(unique_roots, sizes)
        if int(root) in seeded_roots
    ]
    if not candidates:
        return np.zeros(count, dtype=bool), 0, True
    selected_size, selected_root = max(candidates)
    return roots == selected_root, selected_size, False


def analyze_molecular_departures(
    case_dir: Path,
    settings,
    evaporation_time: float | None,
    stop_requested: Callable[[], bool] = lambda: False,
    log: Callable[[str], None] = lambda message: None,
) -> MolecularDepartureResult:
    result = MolecularDepartureResult(
        status="running",
        species_name=settings.departure_species,
        bin_mode=settings.departure_bin_mode,
    )
    try:
        main_dir = case_dir / "main"
        id_list_path = main_dir / "constant" / "idList"
        species_names: list[str]
        if id_list_path.is_file():
            species_names = read_molecule_id_list(id_list_path)
            result.input_paths.append(id_list_path)
        elif settings.departure_species == "water":
            species_names = ["water"]
            result.warning = "constant/idList is missing; water was assumed to be id=0."
        else:
            raise OpenFoamParseError(
                "constant/idList is required for a non-default molecule species"
            )
        try:
            species_id = species_names.index(settings.departure_species)
        except ValueError as exc:
            raise OpenFoamParseError(
                f"Molecule species {settings.departure_species!r} is not in idList"
            ) from exc
        result.species_id = species_id

        frame_sources = _discover_frame_sources(
            main_dir,
            settings.density_field,
        )
        if not frame_sources:
            result.status = "no_data"
            result.error = "No readable Lagrangian moleculeCloud frames were found."
            return result

        frames: list[_Frame] = []
        for time_value, time_name, parts in frame_sources:
            if stop_requested():
                result.status = "stopped"
                return result
            frame, input_paths = _load_frame(
                time_value,
                time_name,
                parts,
                species_id,
                settings,
            )
            frames.append(frame)
            result.input_paths.extend(input_paths)
        result.frame_count = len(frames)
        result.fallback_cluster_frame_count = sum(
            frame.cluster_fallback for frame in frames
        )

        result.events = _detect_events(
            frames,
            species_id,
            settings.departure_species,
            max(1, int(settings.departure_confirmation_frames)),
            evaporation_time,
        )
        result.height_bins = _aggregate_height_bins(
            frames,
            result.events,
            max(1, int(settings.departure_height_bins)),
            evaporation_time,
            settings.departure_bin_mode,
        )
        result.excluded_normalized_height_count = sum(
            event.normalized_height is None
            or event.normalized_height < 0.0
            or event.normalized_height > 1.0
            for event in result.events
        )
        warning_parts = [result.warning] if result.warning else []
        time_steps = [
            right.time - left.time
            for left, right in zip(frames, frames[1:])
            if right.time > left.time
        ]
        if time_steps and float(np.median(time_steps)) > LOW_CADENCE_RECOMMENDATION:
            warning_parts.append(
                "The Lagrangian write interval is coarse for interface crossing; "
                "departure positions use the last liquid-frame position. "
                "A representative validation interval of about 1e-13 s or less "
                "is recommended."
            )
        if result.fallback_cluster_frame_count:
            warning_parts.append(
                f"{result.fallback_cluster_frame_count} frame(s) had no "
                "density-seeded cluster and were treated as having no liquid "
                "cluster."
            )
        result.warning = " ".join(warning_parts)
        result.status = "ok"
        log(
            f"{case_dir.name}: molecular departures raw={result.raw_event_count}, "
            f"confirmed={result.confirmed_event_count}, frames={result.frame_count}"
        )
        return result
    except Exception as exc:
        result.status = "error"
        result.error = str(exc)
        return result


def write_departure_events_csv(
    path: Path,
    case_results,
) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "case",
                "event_index",
                "orig_proc_id",
                "orig_id",
                "species_id",
                "species_name",
                "last_liquid_time",
                "first_outside_time",
                "event_time",
                "time_uncertainty",
                "last_liquid_x",
                "last_liquid_y",
                "last_liquid_z",
                "first_outside_x",
                "first_outside_y",
                "first_outside_z",
                "departure_x",
                "departure_y",
                "departure_z",
                "position_method",
                "height",
                "normalized_height",
                "normalized_radius",
                "cluster_size",
                "confirmed",
                "confirmation_time",
                "confirmation_frames_observed",
                "quality_flags",
            ]
        )
        for case_result in case_results:
            departure = getattr(case_result, "departure_result", None)
            if departure is None:
                continue
            for event in departure.events:
                writer.writerow(
                    [
                        case_result.case_name,
                        event.event_index,
                        event.orig_proc_id,
                        event.orig_id,
                        event.species_id,
                        event.species_name,
                        event.last_liquid_time,
                        event.first_outside_time,
                        event.event_time,
                        event.time_uncertainty,
                        event.last_liquid_x,
                        event.last_liquid_y,
                        event.last_liquid_z,
                        _csv_optional(event.first_outside_x),
                        _csv_optional(event.first_outside_y),
                        _csv_optional(event.first_outside_z),
                        event.departure_x,
                        event.departure_y,
                        event.departure_z,
                        event.position_method,
                        _csv_optional(event.height),
                        _csv_optional(event.normalized_height),
                        _csv_optional(event.normalized_radius),
                        event.cluster_size,
                        int(event.confirmed),
                        _csv_optional(event.confirmation_time),
                        event.confirmation_frames_observed,
                        ";".join(event.quality_flags),
                    ]
                )


def write_departure_height_bins_csv(
    path: Path,
    case_results,
) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "case",
                "bin_mode",
                "bin_index",
                "eta_lower",
                "eta_upper",
                "raw_count",
                "confirmed_count",
                "area_time_exposure_m2_s",
                "raw_rate_1_per_m2_s",
                "confirmed_rate_1_per_m2_s",
            ]
        )
        for case_result in case_results:
            departure = getattr(case_result, "departure_result", None)
            if departure is None:
                continue
            for item in departure.height_bins:
                writer.writerow(
                    [
                        case_result.case_name,
                        departure.bin_mode,
                        item.bin_index,
                        item.eta_lower,
                        item.eta_upper,
                        item.raw_count,
                        item.confirmed_count,
                        item.area_time_exposure,
                        _csv_optional(item.raw_rate),
                        _csv_optional(item.confirmed_rate),
                    ]
                )


def _discover_frame_sources(
    main_dir: Path,
    density_field: str,
) -> list[tuple[float, str, list[tuple[Path, MeshVolumeInfo]]]]:
    reconstructed_mesh_dir = main_dir / "constant" / "polyMesh"
    reconstructed_mesh = (
        read_mesh_volumes(reconstructed_mesh_dir)
        if reconstructed_mesh_dir.is_dir()
        else None
    )
    reconstructed: list[
        tuple[float, str, list[tuple[Path, MeshVolumeInfo]]]
    ] = []
    if reconstructed_mesh is not None:
        for time_value, time_dir in numeric_time_dirs(main_dir):
            if _frame_part_available(time_dir, density_field):
                reconstructed.append(
                    (time_value, time_dir.name, [(time_dir, reconstructed_mesh)])
                )
    if reconstructed:
        return reconstructed

    processors: list[tuple[Path, MeshVolumeInfo]] = []
    for processor in sorted(
        path for path in main_dir.glob("processor*") if path.is_dir()
    ):
        mesh_dir = processor / "constant" / "polyMesh"
        if mesh_dir.is_dir():
            processors.append((processor, read_mesh_volumes(mesh_dir)))
    by_time: dict[float, tuple[str, list[tuple[Path, MeshVolumeInfo]]]] = {}
    for processor, mesh_info in processors:
        for time_value, time_dir in numeric_time_dirs(processor):
            if not _frame_part_available(time_dir, density_field):
                continue
            time_name, parts = by_time.setdefault(
                time_value, (time_dir.name, [])
            )
            parts.append((time_dir, mesh_info))
            by_time[time_value] = (time_name, parts)
    return [
        (time_value, time_name, parts)
        for time_value, (time_name, parts) in sorted(by_time.items())
    ]


def _frame_part_available(time_dir: Path, density_field: str) -> bool:
    cloud_dir = time_dir / "lagrangian" / "moleculeCloud"
    return (time_dir / density_field).is_file() and all(
        (cloud_dir / file_name).is_file()
        for file_name in LAGRANGIAN_FILE_NAMES
    )


def _load_frame(
    time_value: float,
    time_name: str,
    parts: list[tuple[Path, MeshVolumeInfo]],
    species_id: int,
    settings,
) -> tuple[_Frame, list[Path]]:
    selected_positions: list[np.ndarray] = []
    selected_proc_ids: list[np.ndarray] = []
    selected_orig_ids: list[np.ndarray] = []
    selected_seed_masks: list[np.ndarray] = []
    contour_points: list[tuple[float, float, float]] = []
    bounds_list = []
    input_paths: list[Path] = []

    for time_dir, mesh_info in parts:
        cloud_dir = time_dir / "lagrangian" / "moleculeCloud"
        positions_field = read_lagrangian_positions_with_cells(
            cloud_dir / "positions"
        )
        molecule_ids = read_label_field(cloud_dir / "id")
        orig_ids = read_label_field(cloud_dir / "origId")
        orig_proc_ids = read_label_field(cloud_dir / "origProcId")
        count = len(positions_field.positions)
        if not (
            len(molecule_ids)
            == len(orig_ids)
            == len(orig_proc_ids)
            == count
        ):
            raise OpenFoamParseError(
                f"{cloud_dir}: Lagrangian field lengths do not match"
            )
        densities = np.asarray(
            read_scalar_internal_field(time_dir / settings.density_field),
            dtype=float,
        )
        if len(densities) == 1 and len(mesh_info.volumes) != 1:
            densities = np.full(len(mesh_info.volumes), densities[0])
        if len(densities) != len(mesh_info.volumes):
            raise OpenFoamParseError(
                f"{time_dir}: density count does not match mesh cells"
            )
        species_mask = molecule_ids == species_id
        species_positions = positions_field.positions[species_mask]
        species_cells = positions_field.cell_indices[species_mask]
        valid_cells = (species_cells >= 0) & (species_cells < len(densities))
        seed_mask = np.zeros(len(species_positions), dtype=bool)
        seed_mask[valid_cells] = (
            densities[species_cells[valid_cells]]
            >= settings.density_threshold
        )
        selected_positions.append(species_positions)
        selected_proc_ids.append(orig_proc_ids[species_mask])
        selected_orig_ids.append(orig_ids[species_mask])
        selected_seed_masks.append(seed_mask)

        from .analysis import density_contour_points

        part_contours = density_contour_points(
            densities.tolist(),
            mesh_info,
            settings.density_threshold,
        )
        if part_contours:
            contour_points.extend(part_contours)
        if mesh_info.point_bounds is not None:
            bounds_list.append(mesh_info.point_bounds)
        input_paths.extend(
            [
                time_dir / settings.density_field,
                *(cloud_dir / name for name in LAGRANGIAN_FILE_NAMES),
            ]
        )
        input_paths.extend(_mesh_input_paths(mesh_info))

    positions = _concatenate(selected_positions, (0, 3), float)
    proc_ids = _concatenate(selected_proc_ids, (0,), np.int64)
    orig_ids = _concatenate(selected_orig_ids, (0,), np.int64)
    seed_mask = _concatenate(selected_seed_masks, (0,), bool)
    if len(orig_ids):
        identity_pairs = np.column_stack((proc_ids, orig_ids))
        if len(np.unique(identity_pairs, axis=0)) != len(identity_pairs):
            raise OpenFoamParseError(
                f"{time_name}: duplicate (origProcId, origId) identities"
            )
    point_bounds = _combined_point_bounds(bounds_list)
    liquid_mask, cluster_size, cluster_fallback = (
        largest_seeded_periodic_cluster(
            positions,
            seed_mask,
            settings.departure_cutoff,
            point_bounds,
        )
    )
    geometry = _fit_geometry(contour_points, point_bounds, settings)
    return (
        _Frame(
            time=time_value,
            time_name=time_name,
            positions=positions,
            orig_proc_ids=proc_ids,
            orig_ids=orig_ids,
            liquid_mask=liquid_mask,
            cluster_size=cluster_size,
            geometry=geometry,
            cluster_fallback=cluster_fallback,
            point_bounds=point_bounds,
        ),
        input_paths,
    )


def _fit_geometry(
    contour_points: list[tuple[float, float, float]],
    point_bounds,
    settings,
) -> _Geometry:
    from .analysis import contact_fit_diagnostics

    diagnostics = contact_fit_diagnostics(
        contour_points or None,
        point_bounds,
        settings,
    )
    z_top = (
        float(np.max(np.asarray(contour_points, dtype=float)[:, 2]))
        if contour_points
        else None
    )
    return _Geometry(
        z_base=diagnostics.z_base,
        z_top=z_top,
        sphere_center=diagnostics.sphere_center,
        sphere_radius=diagnostics.sphere_radius,
        contact_radius=diagnostics.contact_radius,
    )


def _detect_events(
    frames: list[_Frame],
    species_id: int,
    species_name: str,
    confirmation_frames: int,
    evaporation_time: float | None,
) -> list[DepartureEvent]:
    tracks: dict[tuple[int, int], list[_Observation]] = {}
    for frame_index, frame in enumerate(frames):
        for index in range(len(frame.positions)):
            key = (
                int(frame.orig_proc_ids[index]),
                int(frame.orig_ids[index]),
            )
            tracks.setdefault(key, []).append(
                _Observation(
                    frame_index=frame_index,
                    position=frame.positions[index],
                    liquid=bool(frame.liquid_mask[index]),
                    cluster_size=frame.cluster_size,
                )
            )

    events: list[DepartureEvent] = []
    for (orig_proc_id, orig_id), observations in sorted(tracks.items()):
        has_entered_liquid = False
        previous: _Observation | None = None
        for observation_index, observation in enumerate(observations):
            if observation.liquid:
                has_entered_liquid = True
                previous = observation
                continue
            if (
                has_entered_liquid
                and previous is not None
                and previous.liquid
            ):
                event = _build_event(
                    len(events),
                    orig_proc_id,
                    orig_id,
                    species_id,
                    species_name,
                    previous,
                    observation,
                    observations,
                    observation_index,
                    frames,
                    confirmation_frames,
                    evaporation_time,
                )
                if event is not None:
                    events.append(event)
            previous = observation

        if (
            has_entered_liquid
            and previous is not None
            and previous.liquid
            and previous.frame_index < len(frames) - 1
        ):
            first_outside_index = previous.frame_index + 1
            first_outside_time = frames[first_outside_index].time
            if (
                evaporation_time is None
                or first_outside_time <= evaporation_time
            ):
                events.append(
                    _build_escape_event(
                        len(events),
                        orig_proc_id,
                        orig_id,
                        species_id,
                        species_name,
                        previous,
                        first_outside_index,
                        frames,
                    )
                )
    events.sort(
        key=lambda event: (
            event.event_time,
            event.orig_proc_id,
            event.orig_id,
        )
    )
    return [
        replace(event, event_index=index)
        for index, event in enumerate(events)
    ]


def _build_event(
    event_index: int,
    orig_proc_id: int,
    orig_id: int,
    species_id: int,
    species_name: str,
    last_liquid: _Observation,
    first_outside: _Observation,
    observations: list[_Observation],
    first_outside_observation_index: int,
    frames: list[_Frame],
    confirmation_frames: int,
    evaporation_time: float | None,
) -> DepartureEvent | None:
    outside_time = frames[first_outside.frame_index].time
    if evaporation_time is not None and outside_time > evaporation_time:
        return None

    observed_count = 0
    confirmation_time = None
    confirmed = False
    quality_flags: list[str] = []
    if first_outside.frame_index != last_liquid.frame_index + 1:
        quality_flags.append("observation_gap")
    expected_frame = first_outside.frame_index
    for later in observations[first_outside_observation_index:]:
        if later.frame_index != expected_frame:
            quality_flags.append("observation_gap")
            break
        if later.liquid:
            break
        observed_count += 1
        if observed_count >= confirmation_frames:
            confirmed = True
            confirmation_time = frames[later.frame_index].time
            break
        expected_frame += 1

    last_observation = observations[-1]
    if (
        not confirmed
        and last_observation.frame_index < len(frames) - 1
        and all(
            not item.liquid
            for item in observations[first_outside_observation_index:]
        )
    ):
        confirmed = True
        confirmation_time = frames[last_observation.frame_index + 1].time
        quality_flags.append("escaped_domain")
    if not confirmed:
        quality_flags.append("unconfirmed_short_excursion")

    return _event_from_transition(
        event_index,
        orig_proc_id,
        orig_id,
        species_id,
        species_name,
        last_liquid,
        first_outside.frame_index,
        first_outside.position,
        frames,
        confirmed,
        confirmation_time,
        observed_count,
        quality_flags,
    )


def _build_escape_event(
    event_index: int,
    orig_proc_id: int,
    orig_id: int,
    species_id: int,
    species_name: str,
    last_liquid: _Observation,
    first_outside_index: int,
    frames: list[_Frame],
) -> DepartureEvent:
    return _event_from_transition(
        event_index,
        orig_proc_id,
        orig_id,
        species_id,
        species_name,
        last_liquid,
        first_outside_index,
        None,
        frames,
        True,
        frames[first_outside_index].time,
        0,
        ["escaped_domain"],
    )


def _event_from_transition(
    event_index: int,
    orig_proc_id: int,
    orig_id: int,
    species_id: int,
    species_name: str,
    last_liquid: _Observation,
    first_outside_index: int,
    first_outside_position: np.ndarray | None,
    frames: list[_Frame],
    confirmed: bool,
    confirmation_time: float | None,
    confirmation_frames_observed: int,
    quality_flags: list[str],
) -> DepartureEvent:
    liquid_frame = frames[last_liquid.frame_index]
    outside_time = frames[first_outside_index].time
    liquid_time = liquid_frame.time
    geometry = liquid_frame.geometry
    position = last_liquid.position
    delta_time = outside_time - liquid_time
    departure_position = position
    position_method = "last_liquid_frame"
    if (
        first_outside_position is not None
        and delta_time <= LOW_CADENCE_RECOMMENDATION
    ):
        intersection = _segment_sphere_intersection(
            position,
            first_outside_position,
            geometry,
            liquid_frame.point_bounds,
        )
        if intersection is not None:
            departure_position = intersection
            position_method = "interface_intersection"
            quality_flags.append("interface_intersection")
        else:
            quality_flags.append("interface_intersection_unavailable")
    height = (
        float(departure_position[2] - geometry.z_base)
        if geometry.z_base is not None
        else None
    )
    geometry_height = geometry.height
    normalized_height = (
        height / geometry_height
        if height is not None and geometry_height is not None
        else None
    )
    normalized_radius = _normalized_radius(
        departure_position,
        geometry,
        liquid_frame.point_bounds,
    )
    if normalized_height is None:
        quality_flags.append("height_geometry_missing")
    elif normalized_height < 0.0 or normalized_height > 1.0:
        quality_flags.append("normalized_height_out_of_range")
    if normalized_radius is None:
        quality_flags.append("radius_geometry_missing")
    if delta_time > LOW_CADENCE_RECOMMENDATION:
        quality_flags.append("low_time_resolution")
    return DepartureEvent(
        event_index=event_index,
        orig_proc_id=orig_proc_id,
        orig_id=orig_id,
        species_id=species_id,
        species_name=species_name,
        last_liquid_time=liquid_time,
        first_outside_time=outside_time,
        event_time=(liquid_time + outside_time) / 2.0,
        time_uncertainty=max(0.0, delta_time / 2.0),
        last_liquid_x=float(position[0]),
        last_liquid_y=float(position[1]),
        last_liquid_z=float(position[2]),
        first_outside_x=(
            None
            if first_outside_position is None
            else float(first_outside_position[0])
        ),
        first_outside_y=(
            None
            if first_outside_position is None
            else float(first_outside_position[1])
        ),
        first_outside_z=(
            None
            if first_outside_position is None
            else float(first_outside_position[2])
        ),
        departure_x=float(departure_position[0]),
        departure_y=float(departure_position[1]),
        departure_z=float(departure_position[2]),
        position_method=position_method,
        height=height,
        normalized_height=normalized_height,
        normalized_radius=normalized_radius,
        cluster_size=last_liquid.cluster_size,
        confirmed=confirmed,
        confirmation_time=confirmation_time,
        confirmation_frames_observed=confirmation_frames_observed,
        quality_flags=tuple(dict.fromkeys(quality_flags)),
    )


def _aggregate_height_bins(
    frames: list[_Frame],
    events: list[DepartureEvent],
    bin_count: int,
    evaporation_time: float | None,
    bin_mode: str = "equal_height",
) -> list[DepartureHeightBin]:
    eta_edges = _height_bin_edges(bin_count, bin_mode)
    raw_counts = np.zeros(bin_count, dtype=np.int64)
    confirmed_counts = np.zeros(bin_count, dtype=np.int64)
    for event in events:
        eta = event.normalized_height
        if eta is None or eta < 0.0 or eta > 1.0:
            continue
        index = min(
            bin_count - 1,
            int(np.searchsorted(eta_edges, eta, side="right") - 1),
        )
        raw_counts[index] += 1
        if event.confirmed:
            confirmed_counts[index] += 1

    exposure = np.zeros(bin_count, dtype=float)
    for left, right in zip(frames, frames[1:]):
        interval_end = right.time
        if evaporation_time is not None:
            if left.time >= evaporation_time:
                break
            interval_end = min(interval_end, evaporation_time)
        delta_time = interval_end - left.time
        height = left.geometry.height
        radius = left.geometry.sphere_radius
        if delta_time <= 0.0 or height is None or radius is None:
            continue
        for index in range(bin_count):
            eta_width = eta_edges[index + 1] - eta_edges[index]
            exposure[index] += (
                2.0 * math.pi * radius * height * eta_width * delta_time
            )

    result: list[DepartureHeightBin] = []
    for index in range(bin_count):
        value = float(exposure[index])
        result.append(
            DepartureHeightBin(
                bin_index=index,
                eta_lower=float(eta_edges[index]),
                eta_upper=float(eta_edges[index + 1]),
                raw_count=int(raw_counts[index]),
                confirmed_count=int(confirmed_counts[index]),
                area_time_exposure=value,
                raw_rate=(
                    float(raw_counts[index]) / value if value > 0.0 else None
                ),
                confirmed_rate=(
                    float(confirmed_counts[index]) / value
                    if value > 0.0
                    else None
                ),
            )
        )
    return result


def _height_bin_edges(bin_count: int, bin_mode: str) -> np.ndarray:
    if bin_count <= 0:
        raise ValueError("height bin count must be positive")
    if bin_mode == "equal_height":
        return np.linspace(0.0, 1.0, bin_count + 1)
    if bin_mode == "equal_surface_area":
        # For a fitted sphere, the area of a horizontal spherical band is
        # dA = 2*pi*R*dz. Therefore equal cumulative surface area maps to equal
        # normalized-height intervals. Keep this as a separate mode so the
        # physical intent is explicit and can support non-spherical surfaces.
        area_fractions = np.linspace(0.0, 1.0, bin_count + 1)
        return area_fractions
    raise ValueError(f"Unsupported departure bin mode: {bin_mode}")


def _normalized_radius(
    position: np.ndarray,
    geometry: _Geometry,
    point_bounds,
) -> float | None:
    if (
        geometry.sphere_center is None
        or geometry.contact_radius is None
        or geometry.contact_radius <= 0.0
    ):
        return None
    delta_x = float(position[0] - geometry.sphere_center[0])
    delta_y = float(position[1] - geometry.sphere_center[1])
    if point_bounds is not None:
        x_period = point_bounds[0][1] - point_bounds[0][0]
        y_period = point_bounds[1][1] - point_bounds[1][0]
        if x_period > 0.0:
            delta_x = (delta_x + x_period / 2.0) % x_period - x_period / 2.0
        if y_period > 0.0:
            delta_y = (delta_y + y_period / 2.0) % y_period - y_period / 2.0
    return math.hypot(delta_x, delta_y) / geometry.contact_radius


def _segment_sphere_intersection(
    start: np.ndarray,
    end: np.ndarray,
    geometry: _Geometry,
    point_bounds,
) -> np.ndarray | None:
    if geometry.sphere_center is None or geometry.sphere_radius is None:
        return None
    center = np.asarray(geometry.sphere_center, dtype=float)
    start_unwrapped = np.asarray(start, dtype=float).copy()
    end_unwrapped = np.asarray(end, dtype=float).copy()
    for axis in (0, 1):
        if point_bounds is None:
            continue
        period = point_bounds[axis][1] - point_bounds[axis][0]
        if period <= 0.0:
            continue
        start_unwrapped[axis] = center[axis] + (
            (start_unwrapped[axis] - center[axis] + period / 2.0) % period
            - period / 2.0
        )
        end_unwrapped[axis] = start_unwrapped[axis] + (
            (end_unwrapped[axis] - start[axis] + period / 2.0) % period
            - period / 2.0
        )
    direction = end_unwrapped - start_unwrapped
    a_value = float(np.dot(direction, direction))
    if a_value <= 0.0:
        return None
    offset = start_unwrapped - center
    b_value = 2.0 * float(np.dot(offset, direction))
    c_value = float(np.dot(offset, offset)) - geometry.sphere_radius**2
    discriminant = b_value * b_value - 4.0 * a_value * c_value
    if discriminant < 0.0:
        return None
    root = math.sqrt(max(0.0, discriminant))
    candidates = [
        (-b_value - root) / (2.0 * a_value),
        (-b_value + root) / (2.0 * a_value),
    ]
    valid = [value for value in candidates if 0.0 <= value <= 1.0]
    if not valid:
        return None
    parameter = min(valid)
    return start_unwrapped + parameter * direction


def _periodic_axis_bins(
    point_bounds,
    axis: int,
    cutoff: float,
    values: np.ndarray,
) -> tuple[float, float, int]:
    if point_bounds is not None:
        minimum, maximum = point_bounds[axis]
        period = maximum - minimum
    else:
        minimum = float(np.min(values))
        period = float(np.max(values) - minimum)
    if period <= 0.0:
        return minimum, 0.0, 1
    return minimum, period, max(1, int(math.floor(period / cutoff)))


def _periodic_bin_indices(
    values: np.ndarray,
    minimum: float,
    period: float,
    bin_count: int,
) -> np.ndarray:
    if period <= 0.0:
        return np.zeros(len(values), dtype=np.int64)
    return (
        np.floor(((values - minimum) % period) / period * bin_count)
        .astype(np.int64)
        .clip(0, bin_count - 1)
    )


def _combined_point_bounds(bounds_list):
    valid = [bounds for bounds in bounds_list if bounds is not None]
    if not valid:
        return None
    return tuple(
        (
            min(bounds[axis][0] for bounds in valid),
            max(bounds[axis][1] for bounds in valid),
        )
        for axis in range(3)
    )


def _mesh_input_paths(mesh_info: MeshVolumeInfo) -> list[Path]:
    source = Path(mesh_info.source)
    if not source.is_dir():
        return []
    return [
        source / name
        for name in ("points", "faces", "owner", "neighbour")
        if (source / name).is_file()
    ]


def _concatenate(values, empty_shape, dtype):
    nonempty = [np.asarray(value, dtype=dtype) for value in values if len(value)]
    if not nonempty:
        return np.empty(empty_shape, dtype=dtype)
    return np.concatenate(nonempty, axis=0)


def _read_counted_body(path: Path) -> tuple[int, str]:
    return _read_counted_body_text(
        strip_comments(path.read_text(errors="ignore")),
        path,
    )


def _read_counted_body_text(text: str, path: Path) -> tuple[int, str]:
    match = re.search(r"\b(\d+)\s*\((.*)\)\s*;?\s*$", text, re.S)
    if not match:
        raise OpenFoamParseError(f"Cannot parse counted OpenFOAM list: {path}")
    return int(match.group(1)), match.group(2)


def _csv_optional(value):
    return "" if value is None else value
