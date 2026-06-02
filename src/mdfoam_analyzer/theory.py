from __future__ import annotations

from dataclasses import dataclass, field
import csv
import math
from pathlib import Path
from typing import Iterable

from .analysis import CaseResult
from .openfoam import equivalent_radius


BOLTZMANN_CONSTANT = 1.380649e-23
DEFAULT_ALPHA_VALUES = (0.8, 0.9, 1.0)
XLSX_REFERENCE_THETA_DEG = 76.10000489560471
XLSX_REFERENCE_RHO_L = 956.2
XLSX_REFERENCE_RHO_V = 0.9409227266221003


@dataclass(frozen=True)
class TheorySettings:
    rho_v: float = XLSX_REFERENCE_RHO_V
    rho_l: float = XLSX_REFERENCE_RHO_L
    temperature: float = 300.0
    molecule_mass: float = 2.9915e-26
    v0_source: str = "max_volume"
    theta_source: str = "average"
    fixed_theta_deg: float = XLSX_REFERENCE_THETA_DEG
    fit_percent: float = 100.0
    fit_nonzero_only: bool = True
    fit_alpha_min: float = 0.0
    fit_alpha_max: float = 1.0


THEORY_PRESETS: dict[str, TheorySettings] = {
    "xlsx準拠": TheorySettings(),
    "水 300K 参考": TheorySettings(rho_v=0.0256, rho_l=1000.0),
}


@dataclass(frozen=True)
class SphericalCapGeometry:
    contact_radius: float
    height: float
    surface_area: float


@dataclass(frozen=True)
class TheoryCurve:
    alpha_e: float
    times: list[float]
    volumes: list[float]
    equivalent_radii: list[float]
    evaporated_masses: list[float]


@dataclass(frozen=True)
class TheoryFitResult:
    alpha_e: float | None = None
    sse: float | None = None
    rmse: float | None = None
    r2: float | None = None
    point_count: int = 0
    status: str = "not fitted"
    boundary: str = ""


@dataclass(frozen=True)
class TheoryComparison:
    case_name: str
    v0: float | None
    theta_deg: float | None
    times: list[float]
    md_evaporated_masses: list[float]
    md_equivalent_radii: list[float]
    curves: dict[float, TheoryCurve] = field(default_factory=dict)
    fit: TheoryFitResult = field(default_factory=TheoryFitResult)
    fit_curve: TheoryCurve | None = None
    status: str = "not calculated"


def spherical_cap_geometry(volume: float, theta_deg: float) -> SphericalCapGeometry:
    if volume <= 0.0:
        return SphericalCapGeometry(0.0, 0.0, 0.0)

    height_to_radius = height_to_contact_radius_ratio(theta_deg)
    radius = (6.0 * volume / (math.pi * height_to_radius * (height_to_radius**2 + 3.0))) ** (1.0 / 3.0)
    height = height_to_radius * radius
    surface_area = math.pi * (height**2 + radius**2)
    return SphericalCapGeometry(radius, height, surface_area)


def height_to_contact_radius_ratio(theta_deg: float) -> float:
    theta_rad = math.radians(theta_deg)
    sin_theta = math.sin(theta_rad)
    if abs(sin_theta) < 1.0e-12:
        raise ValueError("contact angle must not have zero sine")
    ratio = (1.0 - math.cos(theta_rad)) / sin_theta
    if ratio <= 0.0 or not math.isfinite(ratio):
        raise ValueError("contact angle must describe a positive spherical cap")
    return ratio


def thermal_speed(settings: TheorySettings) -> float:
    _validate_physical_settings(settings)
    return math.sqrt(BOLTZMANN_CONSTANT * settings.temperature / (2.0 * math.pi * settings.molecule_mass))


def evaporation_flux(settings: TheorySettings, alpha_e: float = 1.0) -> float:
    return alpha_e * settings.rho_v * thermal_speed(settings)


def effective_theta_deg(result: CaseResult, settings: TheorySettings) -> float | None:
    if settings.theta_source == "fixed":
        return settings.fixed_theta_deg
    return result.average_contact_angle_deg


def integrate_theory_curve(
    times: Iterable[float],
    initial_volume: float,
    theta_deg: float,
    settings: TheorySettings,
    alpha_e: float,
) -> TheoryCurve:
    output_times = list(times)
    if not output_times:
        return TheoryCurve(alpha_e, [], [], [], [])
    _validate_physical_settings(settings)

    current_volume = max(0.0, initial_volume)
    max_evaporated_mass = settings.rho_l * current_volume
    current_evaporated_mass = 0.0
    volumes = [current_volume]
    evaporated_masses = [current_evaporated_mass]

    for previous_time, current_time in zip(output_times, output_times[1:]):
        dt = max(0.0, current_time - previous_time)
        current_volume, current_evaporated_mass = _integrate_interval(
            current_volume,
            current_evaporated_mass,
            dt,
            theta_deg,
            settings,
            alpha_e,
            max_evaporated_mass,
        )
        volumes.append(current_volume)
        evaporated_masses.append(current_evaporated_mass)

    return TheoryCurve(
        alpha_e=alpha_e,
        times=output_times,
        volumes=volumes,
        equivalent_radii=[equivalent_radius(volume) for volume in volumes],
        evaporated_masses=evaporated_masses,
    )


def build_theory_comparison(
    result: CaseResult,
    settings: TheorySettings,
    alpha_values: Iterable[float] = DEFAULT_ALPHA_VALUES,
) -> TheoryComparison:
    times = [row.time for row in result.rows]
    md_radii = [row.equivalent_radius for row in result.rows]
    if not result.rows:
        return TheoryComparison(result.case_name, None, None, [], [], [], status="no time rows")
    try:
        _validate_physical_settings(settings)
    except ValueError as exc:
        return TheoryComparison(result.case_name, None, None, times, [], md_radii, status=str(exc))

    theta_deg = effective_theta_deg(result, settings)
    if theta_deg is None:
        return TheoryComparison(result.case_name, None, None, times, [], md_radii, status="contact angle is missing")

    v0 = _effective_initial_volume(result, settings)
    if v0 <= 0.0:
        return TheoryComparison(result.case_name, v0, theta_deg, times, [], md_radii, status="initial volume is zero")

    md_initial_volume = result.rows[0].volume
    md_evaporated_masses = [settings.rho_l * (md_initial_volume - row.volume) for row in result.rows]
    try:
        curves = {
            float(alpha): integrate_theory_curve(times, v0, theta_deg, settings, float(alpha))
            for alpha in alpha_values
        }
        fit = _fit_alpha(result, settings, v0, theta_deg, md_evaporated_masses)
        fit_curve = (
            integrate_theory_curve(times, v0, theta_deg, settings, fit.alpha_e)
            if fit.alpha_e is not None
            else None
        )
    except ValueError as exc:
        return TheoryComparison(
            result.case_name,
            v0,
            theta_deg,
            times,
            md_evaporated_masses,
            md_radii,
            status=str(exc),
        )

    return TheoryComparison(
        case_name=result.case_name,
        v0=v0,
        theta_deg=theta_deg,
        times=times,
        md_evaporated_masses=md_evaporated_masses,
        md_equivalent_radii=md_radii,
        curves=curves,
        fit=fit,
        fit_curve=fit_curve,
        status="ok",
    )


def fit_quality_metrics(target: Iterable[float], predicted: Iterable[float]) -> tuple[float, float, float | None]:
    target_values = list(target)
    predicted_values = list(predicted)
    if len(target_values) != len(predicted_values):
        raise ValueError("target and predicted length mismatch")
    if not target_values:
        return 0.0, 0.0, None

    residuals = [predicted - target for target, predicted in zip(target_values, predicted_values)]
    sse = sum(residual * residual for residual in residuals)
    rmse = math.sqrt(sse / len(target_values))
    mean_target = sum(target_values) / len(target_values)
    sst = sum((target - mean_target) ** 2 for target in target_values)
    if sst <= 0.0:
        r2 = 1.0 if sse <= 1.0e-300 else 0.0
    else:
        r2 = 1.0 - sse / sst
    return sse, rmse, r2


def write_theory_summary_csv(
    path: Path,
    results: list[CaseResult],
    settings: TheorySettings,
    alpha_values: Iterable[float] = DEFAULT_ALPHA_VALUES,
) -> None:
    alpha_values = list(alpha_values)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "case",
                "V0",
                "theta_deg",
                "fit_alpha_e",
                "fit_sse",
                "fit_rmse",
                "fit_r2",
                "fit_point_count",
            "fit_status",
            "fit_boundary",
            "status",
            ]
        )
        for result in results:
            comparison = build_theory_comparison(result, settings, alpha_values)
            writer.writerow(
                [
                    comparison.case_name,
                    _csv_optional(comparison.v0),
                    _csv_optional(comparison.theta_deg),
                    _csv_optional(comparison.fit.alpha_e),
                    _csv_optional(comparison.fit.sse),
                    _csv_optional(comparison.fit.rmse),
                    _csv_optional(comparison.fit.r2),
                    comparison.fit.point_count,
                    comparison.fit.status,
                    comparison.fit.boundary,
                    comparison.fit.status if comparison.status == "ok" else comparison.status,
                ]
            )


def write_theory_timeseries_csv(
    path: Path,
    results: list[CaseResult],
    settings: TheorySettings,
    alpha_values: Iterable[float] = DEFAULT_ALPHA_VALUES,
) -> None:
    alpha_values = [float(alpha) for alpha in alpha_values]
    header = ["case", "time", "EM_MD", "R_eq_MD"]
    for alpha in alpha_values:
        suffix = _alpha_suffix(alpha)
        header.extend([f"EM_theory_alpha_{suffix}", f"V_theory_alpha_{suffix}", f"R_eq_theory_alpha_{suffix}"])
    header.extend(["EM_theory_fit", "V_theory_fit", "R_eq_theory_fit"])

    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for result in results:
            comparison = build_theory_comparison(result, settings, alpha_values)
            for index, time_value in enumerate(comparison.times):
                row: list[float | str] = [
                    comparison.case_name,
                    time_value,
                    _indexed_optional(comparison.md_evaporated_masses, index),
                    _indexed_optional(comparison.md_equivalent_radii, index),
                ]
                for alpha in alpha_values:
                    curve = comparison.curves.get(alpha)
                    row.extend(_curve_values(curve, index))
                row.extend(_curve_values(comparison.fit_curve, index))
                writer.writerow(row)


def _integrate_interval(
    volume: float,
    evaporated_mass: float,
    dt: float,
    theta_deg: float,
    settings: TheorySettings,
    alpha_e: float,
    max_evaporated_mass: float,
) -> tuple[float, float]:
    if dt <= 0.0 or volume <= 0.0 or alpha_e <= 0.0:
        return max(0.0, volume), min(max_evaporated_mass, evaporated_mass)

    first_rate = _mass_rate(volume, theta_deg, settings, alpha_e)
    if first_rate <= 0.0:
        return max(0.0, volume), min(max_evaporated_mass, evaporated_mass)
    estimated_empty_time = settings.rho_l * volume / first_rate
    substeps = max(1, min(500, math.ceil(dt / max(estimated_empty_time / 20.0, 1.0e-300))))
    step_dt = dt / substeps

    current_volume = volume
    current_evaporated_mass = evaporated_mass
    for _ in range(substeps):
        if current_volume <= 0.0:
            break
        current_volume, current_evaporated_mass = _rk4_step(
            current_volume,
            current_evaporated_mass,
            step_dt,
            theta_deg,
            settings,
            alpha_e,
        )
        if current_volume <= 0.0 or current_evaporated_mass >= max_evaporated_mass:
            current_volume = 0.0
            current_evaporated_mass = max_evaporated_mass
            break

    return max(0.0, current_volume), min(max_evaporated_mass, current_evaporated_mass)


def _rk4_step(
    volume: float,
    evaporated_mass: float,
    dt: float,
    theta_deg: float,
    settings: TheorySettings,
    alpha_e: float,
) -> tuple[float, float]:
    def derivative(state_volume: float) -> tuple[float, float]:
        if state_volume <= 0.0:
            return 0.0, 0.0
        rate = _mass_rate(state_volume, theta_deg, settings, alpha_e)
        return -rate / settings.rho_l, rate

    k1_v, k1_em = derivative(volume)
    k2_v, k2_em = derivative(volume + 0.5 * dt * k1_v)
    k3_v, k3_em = derivative(volume + 0.5 * dt * k2_v)
    k4_v, k4_em = derivative(volume + dt * k3_v)

    next_volume = volume + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
    next_evaporated_mass = evaporated_mass + (dt / 6.0) * (k1_em + 2.0 * k2_em + 2.0 * k3_em + k4_em)
    return next_volume, next_evaporated_mass


def _mass_rate(volume: float, theta_deg: float, settings: TheorySettings, alpha_e: float) -> float:
    if volume <= 0.0:
        return 0.0
    geometry = spherical_cap_geometry(volume, theta_deg)
    thermal_speed = math.sqrt(BOLTZMANN_CONSTANT * settings.temperature / (2.0 * math.pi * settings.molecule_mass))
    flux = alpha_e * settings.rho_v * thermal_speed
    return flux * geometry.surface_area


def _fit_alpha(
    result: CaseResult,
    settings: TheorySettings,
    v0: float,
    theta_deg: float,
    md_evaporated_masses: list[float],
) -> TheoryFitResult:
    indices = _fit_indices(result, settings)
    if len(indices) < 2:
        return TheoryFitResult(point_count=len(indices), status="not enough fit points")

    lower = max(0.0, settings.fit_alpha_min)
    upper = max(lower, settings.fit_alpha_max)
    if lower == upper:
        alpha = lower
    else:
        alpha = _minimize_alpha_sse(result, settings, v0, theta_deg, md_evaporated_masses, indices, lower, upper)

    curve = integrate_theory_curve([row.time for row in result.rows], v0, theta_deg, settings, alpha)
    target = [md_evaporated_masses[index] for index in indices]
    predicted = [curve.evaporated_masses[index] for index in indices]
    sse, rmse, r2 = fit_quality_metrics(target, predicted)
    boundary = _fit_boundary(alpha, lower, upper)
    status = "ok"
    if boundary == "lower":
        status = "ok (lower bound)"
    elif boundary == "upper":
        status = "ok (upper bound)"
    return TheoryFitResult(alpha, sse, rmse, r2, len(indices), status, boundary)


def _minimize_alpha_sse(
    result: CaseResult,
    settings: TheorySettings,
    v0: float,
    theta_deg: float,
    md_evaporated_masses: list[float],
    indices: list[int],
    lower: float,
    upper: float,
) -> float:
    objective_cache: dict[float, float] = {}

    def objective(alpha: float) -> float:
        if alpha not in objective_cache:
            curve = integrate_theory_curve([row.time for row in result.rows], v0, theta_deg, settings, alpha)
            residuals = [
                curve.evaporated_masses[index] - md_evaporated_masses[index]
                for index in indices
            ]
            objective_cache[alpha] = sum(residual * residual for residual in residuals)
        return objective_cache[alpha]

    grid_count = 65
    grid = [lower + (upper - lower) * index / (grid_count - 1) for index in range(grid_count)]
    best_index = min(range(grid_count), key=lambda index: objective(grid[index]))
    left = grid[max(0, best_index - 1)]
    right = grid[min(grid_count - 1, best_index + 1)]
    if left == right:
        return grid[best_index]

    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    c = right - inv_phi * (right - left)
    d = left + inv_phi * (right - left)
    for _ in range(80):
        if objective(c) < objective(d):
            right = d
            d = c
            c = right - inv_phi * (right - left)
        else:
            left = c
            c = d
            d = left + inv_phi * (right - left)
    candidates = [left, (left + right) / 2.0, right, grid[best_index]]
    return min(candidates, key=objective)


def _fit_indices(result: CaseResult, settings: TheorySettings) -> list[int]:
    indices = list(range(len(result.rows)))
    if settings.fit_nonzero_only:
        indices = [index for index in indices if result.rows[index].volume > 0.0]
    percent = max(0.0, min(100.0, settings.fit_percent))
    if percent < 100.0 and indices:
        limit = max(1, math.ceil(len(indices) * percent / 100.0))
        indices = indices[:limit]
    return indices


def _effective_initial_volume(result: CaseResult, settings: TheorySettings) -> float:
    if settings.v0_source == "first_volume":
        return result.rows[0].volume if result.rows else 0.0
    return result.max_volume


def _validate_physical_settings(settings: TheorySettings) -> None:
    if settings.rho_v < 0.0:
        raise ValueError("rho_v must be non-negative")
    if settings.rho_l <= 0.0:
        raise ValueError("rho_l must be positive")
    if settings.temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if settings.molecule_mass <= 0.0:
        raise ValueError("molecule mass must be positive")
    if settings.theta_source not in ("average", "fixed"):
        raise ValueError("theta_source must be average or fixed")
    if settings.theta_source == "fixed":
        height_to_contact_radius_ratio(settings.fixed_theta_deg)


def _fit_boundary(alpha: float, lower: float, upper: float) -> str:
    tolerance = max(1.0e-8, abs(upper - lower) * 1.0e-5)
    if alpha <= lower + tolerance:
        return "lower"
    if alpha >= upper - tolerance:
        return "upper"
    return ""


def _curve_values(curve: TheoryCurve | None, index: int) -> list[float | str]:
    if curve is None:
        return ["", "", ""]
    return [
        _indexed_optional(curve.evaporated_masses, index),
        _indexed_optional(curve.volumes, index),
        _indexed_optional(curve.equivalent_radii, index),
    ]


def _indexed_optional(values: list[float], index: int) -> float | str:
    return values[index] if index < len(values) else ""


def _csv_optional(value: float | None) -> float | str:
    return "" if value is None else value


def _alpha_suffix(alpha: float) -> str:
    return f"{alpha:g}".replace("-", "m").replace(".", "p")
