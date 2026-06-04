from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from PySide6.QtWidgets import QApplication
except ImportError as exc:
    raise unittest.SkipTest(f"PySide6 GUI runtime is unavailable: {exc}") from exc

from mdfoam_analyzer.analysis import CaseResult, TimeResult
from mdfoam_analyzer.gui import CombinedPlotWidget, GraphPngPreviewDialog, MainWindow, PlotWidget
from mdfoam_analyzer.theory import evaporation_flux


class GuiTheorySettingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_xlsx_preset_is_default_and_preserves_small_molecule_mass(self) -> None:
        window = MainWindow()
        try:
            settings = window.theory_settings()

            self.assertAlmostEqual(settings.rho_l, 956.2)
            self.assertAlmostEqual(settings.rho_v, 0.9409227266221003)
            self.assertAlmostEqual(settings.molecule_mass, 2.9915e-26)
            self.assertAlmostEqual(evaporation_flux(settings, 1.0), 139.676, delta=1.0e-3)
            self.assertIn("139.676", window.theory_diagnostics_label.text())
        finally:
            window.close()

    def test_reference_water_preset_updates_material_values(self) -> None:
        window = MainWindow()
        try:
            window.theory_preset_combo.setCurrentText("水 300K 参考")
            settings = window.theory_settings()

            self.assertAlmostEqual(settings.rho_l, 1000.0)
            self.assertAlmostEqual(settings.rho_v, 0.0256)
            self.assertAlmostEqual(settings.molecule_mass, 2.9915e-26)
        finally:
            window.close()

    def test_time_plots_are_clipped_at_evaporation_time(self) -> None:
        window = MainWindow()
        try:
            result = CaseResult(
                case_name="case001",
                case_dir=Path(),
                status="ok",
                evaporation_time=2.0,
                rows=[
                    TimeResult(0.0, 3.0, 3.0, 3, 3, contact_angle_deg=80.0, contact_radius=1.0),
                    TimeResult(1.0, 2.0, 2.0, 2, 3, contact_angle_deg=81.0, contact_radius=1.1),
                    TimeResult(2.0, 1.0, 1.0, 1, 3, contact_angle_deg=82.0, contact_radius=1.2),
                    TimeResult(3.0, 0.0, 0.0, 0, 3, contact_angle_deg=83.0, contact_radius=1.3),
                ],
            )
            window.results.append(result)
            window.add_result_row(result)
            visual_refresh_count = 0

            def fake_refresh_visualization() -> None:
                nonlocal visual_refresh_count
                visual_refresh_count += 1

            window.refresh_visualization = fake_refresh_visualization
            window.table.setCurrentCell(0, 0)
            window.update_selected_case_plots()

            self.assertEqual(window.volume_plot._last_plot[4], [0.0, 1.0, 2.0])
            self.assertEqual(visual_refresh_count, 0)
            window.tabs.setCurrentWidget(window.radius_plot)
            self.assertEqual(window.radius_plot._last_plot[4], [0.0, 1.0, 2.0])
            window.tabs.setCurrentWidget(window.contact_angle_plot)
            self.assertEqual(window.contact_angle_plot._last_plot[4], [0.0, 1.0, 2.0])
            window.tabs.setCurrentWidget(window.contact_radius_plot)
            self.assertEqual(window.contact_radius_plot._last_plot[4], [0.0, 1.0, 2.0])

            window.tabs.setCurrentWidget(window.theory_tab)
            for plot in (window.theory_em_plot, window.theory_radius_plot):
                self.assertIsNotNone(plot._last_series)
                for series in plot._last_series[3]:
                    self.assertTrue(all(time <= 2.0 for time in series.x))

            export_plot = window._plot_for_export("volume_time", result, window.volume_plot)
            try:
                self.assertEqual(export_plot._last_plot[4], [0.0, 1.0, 2.0])
                self.assertEqual(window._suggested_png_filename("volume_time", result), "case001_volume_time.png")
            finally:
                export_plot.close()
            combined_export = window._plot_for_export("theory_combined", result, window.theory_em_plot)
            try:
                self.assertIsInstance(combined_export, CombinedPlotWidget)
                for source_plot in combined_export.source_plots:
                    for series in source_plot._last_series[3]:
                        self.assertTrue(all(time <= 2.0 for time in series.x))
                self.assertEqual(window._suggested_png_filename("theory_combined", result), "case001_theory_combined.png")
                with tempfile.TemporaryDirectory() as directory:
                    output_path = Path(directory) / "case001_theory_combined.png"
                    combined_export.save_png(output_path)
                    self.assertTrue(output_path.is_file())
                    self.assertGreater(output_path.stat().st_size, 0)
            finally:
                combined_export.close()
            window.tabs.setCurrentWidget(window.visual_tab)
            self.assertEqual(visual_refresh_count, 1)
        finally:
            window.close()

    def test_theory_png_preview_selects_one_graph_and_quality(self) -> None:
        first = PlotWidget()
        second = PlotWidget()
        try:
            first.plot_xy("first", "x", "y", [0.0, 1.0], [1.0, 2.0])
            second.plot_xy("second", "x", "y", [0.0, 1.0], [2.0, 3.0])

            dialog = GraphPngPreviewDialog(
                [
                    ("first graph", first, "first.png"),
                    ("second graph", second, "second.png"),
                    ("combined", [first, second], "combined.png"),
                ],
                ".",
                suggested_filename="theory.png",
            )
            try:
                self.assertIsNotNone(dialog.source_combo)
                self.assertEqual(dialog.preview_plot._last_plot[1], "first")
                dialog.source_combo.setCurrentIndex(1)
                self.assertEqual(dialog.preview_plot._last_plot[1], "second")
                dialog.source_combo.setCurrentIndex(2)
                self.assertIsInstance(dialog.preview_plot, CombinedPlotWidget)
                dialog.quality_combo.setCurrentText("高 600dpi")
                self.assertEqual(dialog.preview_plot.settings.dpi, 600)
                with tempfile.TemporaryDirectory() as directory:
                    output_path = Path(directory) / "combined.png"
                    dialog.preview_plot.save_png(output_path)
                    self.assertTrue(output_path.is_file())
                    self.assertGreater(output_path.stat().st_size, 0)
            finally:
                dialog.close()
        finally:
            first.close()
            second.close()

    def test_graph_axes_are_fixed_across_cases_and_manual_mode_overrides(self) -> None:
        window = MainWindow()
        try:
            first = CaseResult(
                case_name="case_low",
                case_dir=Path(),
                status="ok",
                evaporation_time=2.0,
                rows=[
                    TimeResult(0.0, 1.0, 1.0, 1, 1, contact_angle_deg=10.0),
                    TimeResult(1.0, 1.0, 1.0, 1, 1, contact_angle_deg=140.0),
                    TimeResult(3.0, 1.0, 1.0, 1, 1, contact_angle_deg=1000.0),
                ],
            )
            second = CaseResult(
                case_name="case_mid",
                case_dir=Path(),
                status="ok",
                evaporation_time=2.0,
                rows=[
                    TimeResult(0.0, 1.0, 1.0, 1, 1, contact_angle_deg=70.0),
                    TimeResult(1.0, 1.0, 1.0, 1, 1, contact_angle_deg=90.0),
                    TimeResult(3.0, 1.0, 1.0, 1, 1, contact_angle_deg=2000.0),
                ],
            )
            for result in (first, second):
                window.results.append(result)
                window.add_result_row(result)
            window.update_common_axis_ranges()
            window.tabs.setCurrentWidget(window.contact_angle_plot)

            window.table.setCurrentCell(0, 0)
            window.update_selected_case_plots()
            first_limits = (
                window.contact_angle_plot.settings.x_min,
                window.contact_angle_plot.settings.x_max,
                window.contact_angle_plot.settings.y_min,
                window.contact_angle_plot.settings.y_max,
            )
            self.assertLessEqual(first_limits[2], 10.0)
            self.assertGreaterEqual(first_limits[3], 140.0)
            self.assertLess(first_limits[3], 1000.0)

            window.table.setCurrentCell(1, 0)
            window.update_selected_case_plots()
            second_limits = (
                window.contact_angle_plot.settings.x_min,
                window.contact_angle_plot.settings.x_max,
                window.contact_angle_plot.settings.y_min,
                window.contact_angle_plot.settings.y_max,
            )
            self.assertEqual(first_limits, second_limits)

            window.graph_axis_mode_combo.setCurrentText("手動固定")
            window.graph_y_min_spin.setValue(50.0)
            window.graph_y_max_spin.setValue(150.0)
            self.assertEqual(window.contact_angle_plot.settings.axis_mode, "manual_fixed")
            self.assertEqual(window.contact_angle_plot.settings.y_min, 50.0)
            self.assertEqual(window.contact_angle_plot.settings.y_max, 150.0)
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
