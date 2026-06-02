from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mdfoam_analyzer.analysis import (
    AnalysisSettings,
    CaseResult,
    TimeResult,
    analyze_case,
    density_contour_points,
)
from mdfoam_analyzer.openfoam import MeshVolumeInfo


class ContactContourTests(unittest.TestCase):
    def test_average_contact_angle_uses_configured_front_percent(self) -> None:
        result = CaseResult(
            case_name="case",
            case_dir=Path(),
            status="ok",
            contact_average_percent=50.0,
            rows=[
                TimeResult(0.0, 1.0, 1.0, 1, 1, 80.0),
                TimeResult(1.0, 1.0, 1.0, 1, 1, None),
                TimeResult(2.0, 1.0, 1.0, 1, 1, 100.0),
                TimeResult(3.0, 1.0, 1.0, 1, 1, 120.0),
            ],
        )

        self.assertEqual(result.average_contact_angle_deg, 80.0)

    def test_density_contour_points_interpolate_between_neighboring_cell_centers(self) -> None:
        mesh_info = MeshVolumeInfo(
            volumes=[1.0, 1.0],
            source="test",
            is_constant=True,
            unique_volume_count=1,
            min_volume=1.0,
            max_volume=1.0,
            total_volume=2.0,
            cell_centers=[(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)],
            point_bounds=((0.0, 2.0), (0.0, 0.0), (0.0, 0.0)),
            neighbour_pairs=[(0, 1)],
        )

        self.assertEqual(density_contour_points([1000.0, 0.0], mesh_info, 500.0), [(1.0, 0.0, 0.0)])

    @unittest.skipUnless((ROOT / "run001_x001").is_dir(), "local sample case is not available")
    def test_sample_contact_angle_uses_contour_points(self) -> None:
        result = analyze_case(ROOT / "run001_x001", AnalysisSettings())
        row = next(item for item in result.rows if item.time == 1e-10)

        self.assertAlmostEqual(row.contact_angle_deg or 0.0, 81.10057426317968)
        self.assertAlmostEqual(row.contact_radius or 0.0, 2.8254863800372293e-09)
        self.assertEqual(row.contact_fit_point_count, 131)


if __name__ == "__main__":
    unittest.main()
