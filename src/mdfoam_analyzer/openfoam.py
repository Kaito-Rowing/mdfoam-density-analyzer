from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re


NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


class OpenFoamParseError(RuntimeError):
    pass


@dataclass(frozen=True)
class FieldInfo:
    path: Path
    name: str
    field_class: str | None
    location: str | None
    object_name: str | None
    dimensions: str | None
    value_count: int | None

    @property
    def is_cell_field(self) -> bool:
        return self.field_class == "volScalarField"


@dataclass(frozen=True)
class MeshVolumeInfo:
    volumes: list[float]
    source: str
    is_constant: bool
    unique_volume_count: int
    min_volume: float
    max_volume: float
    total_volume: float
    cell_centers: list[tuple[float, float, float]] | None = None
    point_bounds: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ] | None = None
    neighbour_pairs: list[tuple[int, int]] | None = None


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//.*", "", text)


def foam_header_value(text: str, key: str) -> str | None:
    match = re.search(rf"\b{re.escape(key)}\s+([^;{{}}]+);", text)
    return match.group(1).strip().strip('"') if match else None


def read_field_info(path: Path) -> FieldInfo:
    text = path.read_text(errors="ignore")
    clean = strip_comments(text)
    return FieldInfo(
        path=path,
        name=path.name,
        field_class=foam_header_value(clean, "class"),
        location=foam_header_value(clean, "location"),
        object_name=foam_header_value(clean, "object"),
        dimensions=foam_header_value(clean, "dimensions"),
        value_count=_read_declared_internal_count(clean),
    )


def _read_declared_internal_count(clean_text: str) -> int | None:
    match = re.search(
        rf"internalField\s+nonuniform\s+List<scalar>\s+(\d+)\s*\(",
        clean_text,
        flags=re.S,
    )
    return int(match.group(1)) if match else None


def read_scalar_internal_field(path: Path) -> list[float]:
    text = strip_comments(path.read_text(errors="ignore"))

    uniform = re.search(rf"internalField\s+uniform\s+({NUMBER_RE})\s*;", text)
    if uniform:
        return [float(uniform.group(1))]

    nonuniform = re.search(
        rf"internalField\s+nonuniform\s+List<scalar>\s+(\d+)\s*\((.*?)\)\s*;",
        text,
        flags=re.S,
    )
    if not nonuniform:
        raise OpenFoamParseError(f"internalField not found: {path}")

    declared_count = int(nonuniform.group(1))
    values = [float(item) for item in re.findall(NUMBER_RE, nonuniform.group(2))]
    if len(values) != declared_count:
        raise OpenFoamParseError(
            f"{path}: declared {declared_count} values, parsed {len(values)}"
        )
    return values


def numeric_time_dirs(root: Path) -> list[tuple[float, Path]]:
    result: list[tuple[float, Path]] = []
    if not root.exists():
        return result
    for child in root.iterdir():
        if not child.is_dir():
            continue
        try:
            time_value = float(child.name)
        except ValueError:
            continue
        result.append((time_value, child))
    result.sort(key=lambda item: item[0])
    return result


def is_openfoam_data_dir(path: Path) -> bool:
    """Return whether *path* looks like an OpenFOAM case data directory."""
    if not path.is_dir():
        return False

    if (path / "constant" / "polyMesh").is_dir():
        return True
    if (path / "constant").is_dir() and (path / "system").is_dir():
        return True

    for _, time_dir in numeric_time_dirs(path):
        if any(
            field.is_file() and field.name.startswith(("rhoM", "rhoN"))
            for field in time_dir.iterdir()
        ):
            return True

    for processor in path.glob("processor*"):
        if not processor.is_dir():
            continue
        if (processor / "constant" / "polyMesh").is_dir():
            return True
        for _, time_dir in numeric_time_dirs(processor):
            if any(
                field.is_file() and field.name.startswith(("rhoM", "rhoN"))
                for field in time_dir.iterdir()
            ):
                return True
    return False


def resolve_case_data_dir(case_dir: Path) -> Path | None:
    """Resolve either ``case/main`` or a directly selected OpenFOAM case root."""
    main_dir = case_dir / "main"
    if main_dir.is_dir():
        return main_dir
    if is_openfoam_data_dir(case_dir):
        return case_dir
    return None


def discover_density_fields(main_dir: Path) -> list[str]:
    names: set[str] = set()
    for _, time_dir in numeric_time_dirs(main_dir):
        for field in time_dir.iterdir():
            if field.is_file() and field.name.startswith(("rhoM", "rhoN")):
                names.add(field.name)
        if names:
            break

    if not names:
        for processor in sorted(main_dir.glob("processor*")):
            for _, time_dir in numeric_time_dirs(processor):
                for field in time_dir.iterdir():
                    if field.is_file() and field.name.startswith(("rhoM", "rhoN")):
                        names.add(field.name)
                if names:
                    break
            if names:
                break

    return sorted(names)


def read_mesh_cell_count(poly_mesh_dir: Path) -> int:
    owner = _read_labels(poly_mesh_dir / "owner")
    neighbour_path = poly_mesh_dir / "neighbour"
    neighbour = _read_labels(neighbour_path) if neighbour_path.exists() else []
    if not owner and not neighbour:
        raise OpenFoamParseError(f"No mesh cells found: {poly_mesh_dir}")
    return max(owner + neighbour) + 1


def read_mesh_volumes(poly_mesh_dir: Path) -> MeshVolumeInfo:
    points = _read_points(poly_mesh_dir / "points")
    faces = _read_faces(poly_mesh_dir / "faces")
    owner = _read_labels(poly_mesh_dir / "owner")
    neighbour_path = poly_mesh_dir / "neighbour"
    neighbour = _read_labels(neighbour_path) if neighbour_path.exists() else []

    if len(faces) != len(owner):
        raise OpenFoamParseError(
            f"{poly_mesh_dir}: faces ({len(faces)}) and owner ({len(owner)}) differ"
        )

    cell_count = max(owner + neighbour) + 1 if owner or neighbour else 0
    volumes = [0.0] * cell_count
    cell_point_ids: list[set[int]] = [set() for _ in range(cell_count)]
    for face_index, face in enumerate(faces):
        center, area_vector = _face_center_area_vector(face, points)
        contribution = _dot(center, area_vector) / 3.0
        owner_cell = owner[face_index]
        volumes[owner_cell] += contribution
        cell_point_ids[owner_cell].update(face)
        if face_index < len(neighbour):
            neighbour_cell = neighbour[face_index]
            volumes[neighbour_cell] -= contribution
            cell_point_ids[neighbour_cell].update(face)

    abs_volumes = [abs(value) for value in volumes]
    if not abs_volumes:
        raise OpenFoamParseError(f"No cell volumes could be computed from {poly_mesh_dir}")

    cell_centers = [_cell_center(point_ids, points) for point_ids in cell_point_ids]
    point_bounds = _point_bounds(points)
    min_volume = min(abs_volumes)
    max_volume = max(abs_volumes)
    tolerance = max(max_volume * 1e-9, 1e-300)
    rounded = {round(value, 34) for value in abs_volumes}
    return MeshVolumeInfo(
        volumes=abs_volumes,
        source=str(poly_mesh_dir),
        is_constant=(max_volume - min_volume) <= tolerance,
        unique_volume_count=len(rounded),
        min_volume=min_volume,
        max_volume=max_volume,
        total_volume=sum(abs_volumes),
        cell_centers=cell_centers,
        point_bounds=point_bounds,
        neighbour_pairs=[(owner[index], neighbour[index]) for index in range(len(neighbour))],
    )


def _cell_center(
    point_ids: set[int],
    points: list[tuple[float, float, float]],
) -> tuple[float, float, float]:
    if not point_ids:
        return (0.0, 0.0, 0.0)
    coords = [points[index] for index in point_ids]
    return tuple(sum(point[axis] for point in coords) / len(coords) for axis in range(3))


def _point_bounds(
    points: list[tuple[float, float, float]],
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    return tuple(
        (min(point[axis] for point in points), max(point[axis] for point in points))
        for axis in range(3)
    )


def _read_list_body(path: Path) -> tuple[int, str]:
    text = strip_comments(path.read_text(errors="ignore"))
    match = re.search(r"\b(\d+)\s*\((.*)\)\s*$", text, flags=re.S)
    if not match:
        raise OpenFoamParseError(f"Cannot parse OpenFOAM list: {path}")
    return int(match.group(1)), match.group(2)


def _read_labels(path: Path) -> list[int]:
    declared_count, body = _read_list_body(path)
    values = [int(item) for item in re.findall(r"-?\d+", body)]
    if len(values) != declared_count:
        raise OpenFoamParseError(
            f"{path}: declared {declared_count} labels, parsed {len(values)}"
        )
    return values


def _read_points(path: Path) -> list[tuple[float, float, float]]:
    declared_count, body = _read_list_body(path)
    pattern = rf"\(\s*({NUMBER_RE})\s+({NUMBER_RE})\s+({NUMBER_RE})\s*\)"
    values = [
        (float(x), float(y), float(z))
        for x, y, z in re.findall(pattern, body)
    ]
    if len(values) != declared_count:
        raise OpenFoamParseError(
            f"{path}: declared {declared_count} points, parsed {len(values)}"
        )
    return values


def _read_faces(path: Path) -> list[list[int]]:
    declared_count, body = _read_list_body(path)
    faces: list[list[int]] = []
    for match in re.finditer(r"(\d+)\s*\(([^()]*)\)", body):
        face_size = int(match.group(1))
        point_ids = [int(item) for item in match.group(2).split()]
        if len(point_ids) != face_size:
            raise OpenFoamParseError(f"{path}: malformed face {len(faces)}")
        faces.append(point_ids)
    if len(faces) != declared_count:
        raise OpenFoamParseError(
            f"{path}: declared {declared_count} faces, parsed {len(faces)}"
        )
    return faces


def _face_center_area_vector(
    face: list[int], points: list[tuple[float, float, float]]
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    polygon = [points[index] for index in face]
    center = tuple(sum(point[axis] for point in polygon) / len(polygon) for axis in range(3))
    area = (0.0, 0.0, 0.0)
    for index, point in enumerate(polygon):
        next_point = polygon[(index + 1) % len(polygon)]
        area = _add(area, _cross(_sub(point, center), _sub(next_point, center)))
    return center, _scale(area, 0.5)


def _sub(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def _add(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def _scale(vector: tuple[float, float, float], scale: float) -> tuple[float, float, float]:
    return (vector[0] * scale, vector[1] * scale, vector[2] * scale)


def _cross(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _dot(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def equivalent_radius(volume: float) -> float:
    return (3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0) if volume > 0 else 0.0
