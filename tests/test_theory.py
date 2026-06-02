from __future__ import annotations

from pathlib import Path
import math
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mdfoam_analyzer.analysis import CaseResult, TimeResult
from mdfoam_analyzer.theory import (
    THEORY_PRESETS,
    TheorySettings,
    build_theory_comparison,
    evaporation_flux,
    fit_quality_metrics,
    integrate_theory_curve,
    spherical_cap_geometry,
)


class TheoryCalculationTests(unittest.TestCase):
    def test_spherical_cap_geometry_for_90_degrees(self) -> None:
        radius = 2.0
        height = 2.0
        volume = math.pi * height * (height**2 + 3.0 * radius**2) / 6.0

        geometry = spherical_cap_geometry(volume, 90.0)

        self.assertAlmostEqual(geometry.contact_radius, radius)
        self.assertAlmostEqual(geometry.height, height)
        self.assertAlmostEqual(geometry.surface_area, 8.0 * math.pi)

    def test_zero_alpha_keeps_volume_constant(self) -> None:
        settings = TheorySettings()
        curve = integrate_theory_curve([0.0, 1.0, 2.0], 1.0e-24, 90.0, settings, 0.0)

        self.assertEqual(curve.volumes, [1.0e-24, 1.0e-24, 1.0e-24])
        self.assertEqual(curve.evaporated_masses, [0.0, 0.0, 0.0])

    def test_xlsx_reference_preset_matches_old_theory_curve(self) -> None:
        settings = THEORY_PRESETS["xlsx準拠"]
        theta = 76.10000489560471
        volume = 4.42e-26

        geometry = spherical_cap_geometry(volume, theta)
        curve = integrate_theory_curve([0.0, 1.0e-10], volume, theta, settings, 1.0)

        self.assertAlmostEqual(evaporation_flux(settings, 1.0), 139.676, delta=1.0e-3)
        self.assertAlmostEqual(geometry.surface_area, 4.8755e-17, delta=1.0e-21)
        self.assertAlmostEqual(curve.evaporated_masses[1], 6.78e-25, delta=2.0e-27)
        self.assertAlmostEqual(curve.volumes[1], 4.35e-26, delta=2.0e-29)

    def test_large_alpha_clamps_volume_at_zero(self) -> None:
        settings = TheorySettings(rho_v=1.0e20)
        curve = integrate_theory_curve([0.0, 1.0], 1.0e-27, 90.0, settings, 1.0)

        self.assertTrue(all(volume >= 0.0 for volume in curve.volumes))
        self.assertAlmostEqual(curve.volumes[-1], 0.0)
        self.assertLessEqual(curve.evaporated_masses[-1], settings.rho_l * 1.0e-27)

    def test_fit_recovers_synthetic_alpha(self) -> None:
        settings = TheorySettings()
        times = [index * 1.0e-10 for index in range(8)]
        curve = integrate_theory_curve(times, 4.0e-26, 90.0, settings, 0.9)
        rows = [
            TimeResult(time, volume, 0.0, 1, 1, contact_angle_deg=90.0)
            for time, volume in zip(curve.times, curve.volumes)
        ]
        result = CaseResult("synthetic", Path(), "ok", rows=rows)

        comparison = build_theory_comparison(result, settings, [0.8, 0.9, 1.0])

        self.assertEqual(comparison.fit.status, "ok")
        self.assertIsNotNone(comparison.fit.alpha_e)
        self.assertAlmostEqual(comparison.fit.alpha_e or 0.0, 0.9, places=4)
        self.assertIsNotNone(comparison.fit.r2)
        self.assertGreater(comparison.fit.r2 or 0.0, 0.999)

    def test_fixed_theta_source_does_not_require_contact_angle_rows(self) -> None:
        settings = TheorySettings(theta_source="fixed", fixed_theta_deg=76.10000489560471)
        result = CaseResult(
            "fixed-theta",
            Path(),
            "ok",
            rows=[
                TimeResult(0.0, 4.42e-26, 0.0, 1, 1),
                TimeResult(1.0e-10, 4.35e-26, 0.0, 1, 1),
            ],
        )

        comparison = build_theory_comparison(result, settings, [1.0])

        self.assertEqual(comparison.status, "ok")
        self.assertAlmostEqual(comparison.theta_deg or 0.0, 76.10000489560471)

    def test_fit_reports_upper_boundary(self) -> None:
        settings = TheorySettings(rho_v=0.0256, rho_l=1000.0)
        result = CaseResult(
            "boundary",
            Path(),
            "ok",
            rows=[
                TimeResult(0.0, 4.42e-26, 0.0, 1, 1, contact_angle_deg=76.10000489560471),
                TimeResult(1.0e-10, 3.0e-26, 0.0, 1, 1, contact_angle_deg=76.10000489560471),
                TimeResult(2.0e-10, 2.0e-26, 0.0, 1, 1, contact_angle_deg=76.10000489560471),
            ],
        )

        comparison = build_theory_comparison(result, settings, [1.0])

        self.assertEqual(comparison.fit.boundary, "upper")
        self.assertIn("upper bound", comparison.fit.status)

    def test_fit_quality_metrics_for_perfect_match(self) -> None:
        sse, rmse, r2 = fit_quality_metrics([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])

        self.assertEqual(sse, 0.0)
        self.assertEqual(rmse, 0.0)
        self.assertEqual(r2, 1.0)


if __name__ == "__main__":
    unittest.main()
