from __future__ import annotations

import math

import pytest

from mdfoam_analyzer.analysis import AnalysisSettings, contact_fit_diagnostics


def test_contact_fit_recovers_synthetic_hemisphere_geometry() -> None:
    root_three = math.sqrt(3.0)
    points = [
        (2.0, 0.0, 0.0),
        (-2.0, 0.0, 0.0),
        (0.0, 2.0, 0.0),
        (0.0, -2.0, 0.0),
        (root_three, 0.0, 1.0),
        (-root_three, 0.0, 1.0),
        (0.0, root_three, 1.0),
        (0.0, -root_three, 1.0),
        (0.0, 0.0, 2.0),
    ]

    diagnostics = contact_fit_diagnostics(
        points,
        ((-2.0, 2.0), (-2.0, 2.0), (0.0, 2.0)),
        AnalysisSettings(contact_fit_lower=0.0, contact_fit_upper=1.0),
    )

    assert diagnostics.failure_reason == ""
    assert diagnostics.fit_point_count == len(points)
    assert diagnostics.sphere_center == pytest.approx((0.0, 0.0, 0.0), abs=1.0e-12)
    assert diagnostics.sphere_radius == pytest.approx(2.0)
    assert diagnostics.contact_angle_deg == pytest.approx(90.0)
    assert diagnostics.contact_radius == pytest.approx(2.0)


def test_contact_fit_reports_blank_metrics_for_too_few_points() -> None:
    diagnostics = contact_fit_diagnostics(
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        ((0.0, 1.0), (0.0, 1.0), (0.0, 0.0)),
        AnalysisSettings(),
    )

    assert diagnostics.fit_point_count == 0
    assert diagnostics.contact_angle_deg is None
    assert diagnostics.contact_radius is None
    assert diagnostics.sphere_center is None
