from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re

import numpy as np

from .analysis import AnalysisSettings, ContactFitDiagnostics, contact_fit_diagnostics
from .openfoam import (
    NUMBER_RE,
    OpenFoamParseError,
    numeric_time_dirs,
    read_mesh_volumes,
    read_scalar_internal_field,
    strip_comments,
)


@dataclass(frozen=True)
class ParticleCloud:
    positions: np.ndarray
    ids: np.ndarray | None
    id_warning: str = ""


@dataclass(frozen=True)
class VisualizationFrame:
    time: float
    time_name: str
    particles: ParticleCloud
    selected_centers: np.ndarray
    contact: ContactFitDiagnostics
    point_bounds: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ] | None


def case_time_dirs(case_dir: Path) -> list[tuple[float, Path]]:
    main_dir = case_dir / "main"
    return numeric_time_dirs(main_dir)


def find_time_dir(case_dir: Path, time_value: float) -> Path | None:
    for value, path in case_time_dirs(case_dir):
        if value == time_value:
            return path
    return None


def load_visualization_frame(
    case_dir: Path,
    time_value: float,
    settings: AnalysisSettings,
) -> VisualizationFrame:
    time_dir = find_time_dir(case_dir, time_value)
    if time_dir is None:
        raise OpenFoamParseError(f"時刻ディレクトリが見つかりません: {time_value}")

    mesh_info = read_mesh_volumes(case_dir / "main" / "constant" / "polyMesh")
    density_path = time_dir / settings.density_field
    selected_centers: list[tuple[float, float, float]] = []
    if density_path.is_file() and mesh_info.cell_centers is not None:
        densities = _expanded_densities(
            read_scalar_internal_field(density_path),
            len(mesh_info.cell_centers),
        )
        selected_centers = [
            center
            for density, center in zip(densities, mesh_info.cell_centers)
            if density >= settings.density_threshold
        ]

    contact = contact_fit_diagnostics(selected_centers, mesh_info.point_bounds, settings)
    return VisualizationFrame(
        time=time_value,
        time_name=time_dir.name,
        particles=read_particle_cloud(time_dir),
        selected_centers=np.asarray(selected_centers, dtype=float),
        contact=contact,
        point_bounds=mesh_info.point_bounds,
    )


def read_particle_cloud(time_dir: Path) -> ParticleCloud:
    cloud_dir = time_dir / "lagrangian" / "moleculeCloud"
    positions_path = cloud_dir / "positions"
    if not positions_path.is_file():
        return ParticleCloud(np.empty((0, 3), dtype=float), None, "positionsがありません")

    positions = read_lagrangian_positions(positions_path)
    ids_path = cloud_dir / "id"
    if not ids_path.is_file():
        return ParticleCloud(positions, None, "idがないため単色表示")

    ids = read_label_field(ids_path)
    if len(ids) != len(positions):
        return ParticleCloud(
            positions,
            None,
            f"id数({len(ids)})とpositions数({len(positions)})が一致しないため単色表示",
        )
    return ParticleCloud(positions, ids)


def read_lagrangian_positions(path: Path) -> np.ndarray:
    declared_count, body = _read_counted_body(path)
    pattern = rf"\(\s*({NUMBER_RE})\s+({NUMBER_RE})\s+({NUMBER_RE})\s*\)\s+\d+"
    values = [(float(x), float(y), float(z)) for x, y, z in re.findall(pattern, body)]
    if len(values) != declared_count:
        raise OpenFoamParseError(
            f"{path}: declared {declared_count} positions, parsed {len(values)}"
        )
    return np.asarray(values, dtype=float)


def read_label_field(path: Path) -> np.ndarray:
    declared_count, body = _read_counted_body(path)
    values = [int(item) for item in re.findall(r"-?\d+", body)]
    if len(values) != declared_count:
        raise OpenFoamParseError(
            f"{path}: declared {declared_count} labels, parsed {len(values)}"
        )
    return np.asarray(values, dtype=int)


def replicate_xy(
    positions: np.ndarray,
    point_bounds: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ] | None,
    tile_count: int,
    ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    if tile_count <= 1 or len(positions) == 0 or point_bounds is None:
        return positions, ids

    tile_count = max(1, min(16, int(tile_count)))
    x_period = point_bounds[0][1] - point_bounds[0][0]
    y_period = point_bounds[1][1] - point_bounds[1][0]
    if x_period <= 0.0 or y_period <= 0.0:
        return positions, ids

    start = -(tile_count // 2)
    offsets = [(ix * x_period, iy * y_period, 0.0) for ix in range(start, start + tile_count) for iy in range(start, start + tile_count)]
    replicated = np.concatenate([positions + np.asarray(offset) for offset in offsets], axis=0)
    replicated_ids = np.tile(ids, len(offsets)) if ids is not None else None
    return replicated, replicated_ids


def downsample_points(
    positions: np.ndarray,
    ids: np.ndarray | None,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    if max_points <= 0 or len(positions) <= max_points:
        return positions, ids
    indices = np.linspace(0, len(positions) - 1, max_points, dtype=int)
    return positions[indices], ids[indices] if ids is not None else None


def read_remote_case_from_manifest(case_dir: Path) -> str | None:
    manifest_path = case_dir / ".mdfoam_remote_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    remote_case = data.get("remote_case")
    return str(remote_case) if remote_case else None


def _read_counted_body(path: Path) -> tuple[int, str]:
    text = strip_comments(path.read_text(errors="ignore"))
    match = re.search(r"\b(\d+)\s*\((.*)\)\s*;?\s*$", text, flags=re.S)
    if not match:
        raise OpenFoamParseError(f"Cannot parse counted OpenFOAM list: {path}")
    return int(match.group(1)), match.group(2)


def _expanded_densities(densities: list[float], cell_count: int) -> list[float]:
    if len(densities) == 1 and cell_count != 1:
        return densities * cell_count
    if len(densities) != cell_count:
        raise OpenFoamParseError(
            f"Density count {len(densities)} does not match cell count {cell_count}"
        )
    return densities
