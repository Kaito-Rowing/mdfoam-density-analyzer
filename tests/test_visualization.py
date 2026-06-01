from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mdfoam_analyzer.analysis import AnalysisSettings, analyze_case
from mdfoam_analyzer.visualization import (
    downsample_points_by_id,
    id_draw_order,
    load_visualization_frame,
    plot_bounds_from_point_bounds,
    read_label_field,
    read_lagrangian_positions,
    replicate_xy,
)


class VisualizationParserTests(unittest.TestCase):
    def test_reads_lagrangian_positions_and_id_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            positions = base / "positions"
            positions.write_text(
                """
FoamFile
{
    class Cloud<passivePositionParticle>;
}
3
(
(1 2 3) 0
(4 5 6) 1
(7 8 9) 2
)
""",
                encoding="utf-8",
            )
            ids = base / "id"
            ids.write_text(
                """
FoamFile
{
    class labelField;
}
3
(
1
0
1
)
""",
                encoding="utf-8",
            )
            np.testing.assert_allclose(
                read_lagrangian_positions(positions),
                np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float),
            )
            np.testing.assert_array_equal(read_label_field(ids), np.asarray([1, 0, 1]))

    def test_replicates_particles_in_xy_tiles(self) -> None:
        positions = np.asarray([[0.5, 0.5, 1.0], [1.5, 1.5, 2.0]])
        ids = np.asarray([0, 1])
        bounds = ((0.0, 2.0), (0.0, 2.0), (0.0, 3.0))
        replicated, replicated_ids = replicate_xy(positions, bounds, 3, ids)
        self.assertEqual(len(replicated), 18)
        self.assertEqual(len(replicated_ids), 18)
        self.assertTrue(any(np.allclose(point, [-1.5, -1.5, 1.0]) for point in replicated))
        self.assertTrue(any(np.allclose(point, [2.5, 2.5, 1.0]) for point in replicated))

    def test_downsamples_by_id_and_preserves_both_ids(self) -> None:
        positions = np.column_stack(
            [
                np.arange(1000, dtype=float),
                np.zeros(1000),
                np.r_[np.zeros(800), np.ones(200)],
            ]
        )
        ids = np.r_[np.zeros(800, dtype=int), np.ones(200, dtype=int)]
        result = downsample_points_by_id(positions, ids, 100)
        self.assertEqual(result.original_count, 1000)
        self.assertEqual(result.displayed_count, 100)
        self.assertTrue(result.was_downsampled)
        unique, counts = np.unique(result.ids, return_counts=True)
        self.assertEqual(set(unique.tolist()), {0, 1})
        self.assertGreater(counts[unique.tolist().index(0)], counts[unique.tolist().index(1)])

    def test_downsample_zero_keeps_all_points(self) -> None:
        positions = np.asarray([[0, 0, 0], [1, 1, 1]], dtype=float)
        ids = np.asarray([0, 1])
        result = downsample_points_by_id(positions, ids, 0)
        self.assertFalse(result.was_downsampled)
        self.assertEqual(result.displayed_count, 2)
        np.testing.assert_array_equal(result.ids, ids)

    def test_id_draw_order_uses_mean_z(self) -> None:
        positions = np.asarray([[0, 0, 10], [0, 0, 12], [0, 0, -1], [0, 0, 0]], dtype=float)
        ids = np.asarray([5, 5, 2, 2])
        self.assertEqual(id_draw_order(positions, ids), [2, 5])

    def test_plot_bounds_expand_for_periodic_tiles(self) -> None:
        bounds = plot_bounds_from_point_bounds(((0.0, 2.0), (10.0, 14.0), (-1.0, 1.0)), 3)
        assert bounds is not None
        self.assertLess(bounds.x[0], -2.0)
        self.assertGreater(bounds.x[1], 4.0)
        self.assertLess(bounds.y[0], 6.0)
        self.assertGreater(bounds.y[1], 18.0)

    def test_plot_bounds_caps_tile_count_at_16(self) -> None:
        bounds16 = plot_bounds_from_point_bounds(((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), 16)
        bounds99 = plot_bounds_from_point_bounds(((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), 99)
        self.assertEqual(bounds16, bounds99)


class VisualizationSampleTests(unittest.TestCase):
    @unittest.skipUnless((ROOT / "run001_x001").is_dir(), "local sample case is not available")
    def test_sample_frame_matches_analysis_contact_metrics(self) -> None:
        case_dir = ROOT / "run001_x001"
        settings = AnalysisSettings()
        result = analyze_case(case_dir, settings)
        row = next(item for item in result.rows if item.time == 1e-10)
        frame = load_visualization_frame(case_dir, 1e-10, settings)
        self.assertEqual(len(frame.particles.positions), 7058)
        self.assertEqual(set(frame.particles.ids.tolist()), {0, 1})
        self.assertAlmostEqual(frame.contact.contact_angle_deg or 0.0, row.contact_angle_deg or 0.0)
        self.assertAlmostEqual(frame.contact.contact_radius or 0.0, row.contact_radius or 0.0)
        self.assertEqual(frame.contact.fit_point_count, row.contact_fit_point_count)


if __name__ == "__main__":
    unittest.main()
