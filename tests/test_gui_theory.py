from __future__ import annotations

import os
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, call, patch

from PIL import Image


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from PySide6.QtWidgets import QApplication, QDialog, QFileDialog
except ImportError as exc:
    raise unittest.SkipTest(f"PySide6 GUI runtime is unavailable: {exc}") from exc

from mdfoam_analyzer.analysis import (
    AnalysisLayoutProfile,
    AnalysisSettings,
    CaseResult,
    TimeResult,
)
from mdfoam_analyzer.gui import (
    AnalyzerWorker,
    PNG_PREVIEW_PIXELS_PER_INCH,
    CombinedPlotWidget,
    GraphSettings,
    GraphPngPreviewDialog,
    MainWindow,
    PlotSeries,
    PlotWidget,
)
from mdfoam_analyzer.molecular_departure import (
    DepartureHeightBin,
    MolecularDepartureResult,
)
from mdfoam_analyzer.provenance import RunContext
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

    def test_loading_analysis_settings_only_updates_analysis_controls(self) -> None:
        window = MainWindow()
        try:
            original_source = window.source_combo.currentData()
            settings = AnalysisSettings(
                density_field="rhoN_custom",
                density_threshold=321.0,
                zero_tolerance=1.0e-20,
                consecutive_zero_count=7,
                manual_cell_volume=2.0e-27,
                dx=1.0e-9,
                dy=2.0e-9,
                dz=3.0e-9,
                contact_fit_lower=0.2,
                contact_fit_upper=0.8,
                contact_unwrap_xy=False,
                contact_average_percent=55.0,
                departure_enabled=True,
                departure_species="water",
                departure_cutoff=5.0e-10,
                departure_confirmation_frames=4,
                departure_height_bins=12,
                departure_bin_mode="equal_surface_area",
            )

            window.apply_analysis_settings(settings)

            self.assertEqual(window.settings(), settings)
            self.assertEqual(window.source_combo.currentData(), original_source)
            self.assertIsNone(window.worker)
            self.assertIsNone(window.thread)
        finally:
            window.close()

    def test_csv_export_includes_analysis_manifest(self) -> None:
        window = MainWindow()
        try:
            result = CaseResult(
                case_name="case001",
                case_dir=Path(),
                status="ok",
                rows=[TimeResult(0.0, 1.0, 1.0, 1, 1)],
                source_case_path="/data/case001",
            )
            settings = AnalysisSettings()
            window.results = [result]
            window._last_run_settings = settings
            window._last_run_context = RunContext(
                "local", "/data", analysis_settings=settings
            )
            with tempfile.TemporaryDirectory() as temp_dir:
                output = Path(temp_dir)
                window._choose_output_directory = lambda *_: output

                window.export_csv()

                manifest = json.loads(
                    (output / "analysis_manifest.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(
                    manifest["analysis_settings"]["density_field"],
                    "rhoM_water",
                )
                self.assertEqual(
                    manifest["cases"][0]["result_summary"]["time_count"],
                    1,
                )
                self.assertTrue(
                    (output / "mdfoam_departure_events.csv").is_file()
                )
                self.assertTrue(
                    (output / "mdfoam_departure_height_bins.csv").is_file()
                )
        finally:
            window.close()

    def test_local_worker_detects_each_case_layout_independently(self) -> None:
        cases = [Path("case_a"), Path("case_b")]
        settings = AnalysisSettings()
        profiles = [
            AnalysisLayoutProfile(
                mode="reconstructed",
                expected_total_cells=3825,
                processor_count=0,
                source_case="case_a",
            ),
            AnalysisLayoutProfile(
                mode="reconstructed",
                expected_total_cells=4352,
                processor_count=0,
                source_case="case_b",
            ),
        ]
        received_profiles = []

        def fake_analyze_case(case, run_settings, **kwargs):
            received_profiles.append(kwargs.get("layout_profile"))
            return CaseResult(Path(case).name, Path(case), "ok")

        with (
            patch(
                "mdfoam_analyzer.gui.detect_analysis_layout",
                side_effect=profiles,
            ) as detect,
            patch(
                "mdfoam_analyzer.gui.analyze_case",
                side_effect=fake_analyze_case,
            ),
        ):
            worker = AnalyzerWorker(cases, settings)
            worker.run()

        self.assertEqual(
            detect.call_args_list,
            [call(cases[0], settings), call(cases[1], settings)],
        )
        self.assertEqual(received_profiles, profiles)

    def test_departure_distribution_plot_switches_between_count_and_rate(self) -> None:
        window = MainWindow()
        try:
            result = CaseResult(
                case_name="case001",
                case_dir=Path(),
                status="ok",
                rows=[TimeResult(0.0, 1.0, 1.0, 1, 1)],
                departure_result=MolecularDepartureResult(
                    status="ok",
                    species_name="water",
                    height_bins=[
                        DepartureHeightBin(
                            0,
                            0.0,
                            0.5,
                            4,
                            3,
                            2.0,
                            2.0,
                            1.5,
                        )
                    ],
                ),
            )

            window._plot_departure_result(
                window.departure_distribution_plot,
                result,
                "departure_height_distribution",
            )
            count_series = window.departure_distribution_plot._last_series
            self.assertIsNotNone(count_series)
            self.assertEqual(count_series[3][0].y, [4.0])

            window.departure_intensity_combo.setCurrentIndex(
                window.departure_intensity_combo.findData("rate")
            )
            window._plot_departure_result(
                window.departure_distribution_plot,
                result,
                "departure_height_distribution",
            )
            rate_series = window.departure_distribution_plot._last_series
            self.assertIsNotNone(rate_series)
            self.assertEqual(rate_series[3][0].y, [2.0])
        finally:
            window.close()

    def test_invalid_settings_file_does_not_change_current_values(self) -> None:
        window = MainWindow()
        try:
            window.threshold_spin.setValue(777.0)
            with tempfile.TemporaryDirectory() as temp_dir:
                path = Path(temp_dir) / "invalid.json"
                path.write_text("{invalid", encoding="utf-8")
                with (
                    patch(
                        "mdfoam_analyzer.gui.QFileDialog.getOpenFileName",
                        return_value=(str(path), "JSON (*.json)"),
                    ),
                    patch("mdfoam_analyzer.gui.QMessageBox.warning"),
                ):
                    window.load_analysis_settings_file()

            self.assertEqual(window.threshold_spin.value(), 777.0)
        finally:
            window.close()

    def test_ui_language_switches_without_changing_internal_settings(self) -> None:
        window = MainWindow()
        try:
            expected_titles = {
                "en": "mdFOAM Density Analyzer",
                "zh": "mdFOAM 密度分析器",
                "es": "Analizador de densidad mdFOAM",
                "hi": "mdFOAM घनत्व विश्लेषक",
            }
            for language, title in expected_titles.items():
                window.apply_language(language)
                self.assertEqual(window.windowTitle(), title)
                self.assertEqual(window.language_combo.findData(language), window.language_combo.currentIndex())
                self.assertEqual(window.source_combo.currentData(), "local")
                self.assertEqual(window.theory_v0_source_combo.currentData(), "max_volume")
                self.assertEqual(window.theory_theta_source_combo.currentData(), "average")
                self.assertEqual(window.graph_axis_mode_combo.currentData(), "auto_fixed")
                self.assertEqual(window.graph_aspect_combo.currentData(), "auto")

            window.apply_language("en")
            self.assertEqual(window.workflow_tabs.tabText(0), "Input")
            self.assertEqual(window.export_csv_button.text(), "Export CSV")
            self.assertEqual(
                window.save_settings_button.text(),
                "Save analysis settings",
            )
            self.assertEqual(
                window.load_settings_button.text(),
                "Load analysis settings",
            )
            self.assertEqual(
                window.export_manifest_button.text(),
                "Save analysis record",
            )
            self.assertEqual(window.table.horizontalHeaderItem(0).text(), "Case")
            self.assertEqual(window.tabs.tabText(4), "Evaporation time")
            window.theory_theta_source_combo.setCurrentIndex(window.theory_theta_source_combo.findData("fixed"))
            self.assertEqual(window.theory_settings().theta_source, "fixed")
            window.graph_axis_mode_combo.setCurrentIndex(window.graph_axis_mode_combo.findData("manual_fixed"))
            window.on_graph_settings_changed()
            self.assertEqual(window.current_plot_widget().settings.axis_mode, "manual_fixed")
        finally:
            window.close()

    def test_dashboard_sidebar_controls_pages_and_log_drawer(self) -> None:
        window = MainWindow()
        try:
            self.assertTrue(window.workflow_tabs.tabBar().isHidden())
            self.assertTrue(window.input_nav_button.isChecked())
            self.assertEqual(window.workflow_tabs.currentIndex(), 0)
            self.assertTrue(window.log_group.isHidden())

            window.settings_nav_button.click()
            self.assertEqual(window.workflow_tabs.currentIndex(), 1)
            self.assertTrue(window.settings_nav_button.isChecked())

            window.workflow_tabs.setCurrentIndex(2)
            self.assertTrue(window.results_nav_button.isChecked())
            self.assertEqual(window.page_title_label.text(), window.workflow_tabs.tabText(2))

            window.log_toggle_button.click()
            self.assertFalse(window.log_group.isHidden())
            window.log_toggle_button.click()
            self.assertTrue(window.log_group.isHidden())
        finally:
            window.close()

    def test_dashboard_control_panels_collapse_and_expand(self) -> None:
        window = MainWindow()
        try:
            self.assertFalse(window.graph_settings_group.isChecked())
            self.assertEqual(window.graph_settings_group.maximumHeight(), 34)
            window.graph_settings_group.setChecked(True)
            self.assertGreater(window.graph_settings_group.maximumHeight(), 1000)

            self.assertFalse(window.visual_settings_group.isChecked())
            window.visual_settings_group.setChecked(True)
            self.assertGreater(window.visual_settings_group.maximumHeight(), 1000)
        finally:
            window.close()

    def test_dashboard_can_shrink_to_small_desktop_size(self) -> None:
        window = MainWindow()
        try:
            window.resize(1024, 720)
            window.show()
            self.app.processEvents()
            self.assertLessEqual(window.width(), 1024)
            self.assertLessEqual(window.height(), 720)
            self.assertTrue(window.workflow_tabs.isVisible())
        finally:
            window.close()

    def test_input_and_settings_use_compact_responsive_layout(self) -> None:
        window = MainWindow()
        try:
            window.resize(1400, 900)
            window.show()
            self.app.processEvents()
            self.assertLessEqual(window.input_content.maximumWidth(), 1120)
            self.assertEqual(window.source_stack.maximumHeight(), 150)
            self.assertLessEqual(window.threshold_spin.maximumWidth(), 360)
            self.assertLessEqual(window.theory_rho_l_spin.maximumWidth(), 360)
            advanced_index = window.settings_grid.indexOf(window.advanced_group)
            self.assertEqual(window.settings_grid.getItemPosition(advanced_index)[:2], (0, 1))

            window.resize(1024, 720)
            self.app.processEvents()
            advanced_index = window.settings_grid.indexOf(window.advanced_group)
            theory_index = window.settings_grid.indexOf(window.theory_group)
            self.assertEqual(window.settings_grid.getItemPosition(advanced_index)[:2], (1, 0))
            self.assertEqual(window.settings_grid.getItemPosition(theory_index)[:2], (3, 0))

            window.source_combo.setCurrentIndex(window.source_combo.findData("ssh"))
            self.assertGreater(window.source_stack.maximumHeight(), 1000)
        finally:
            window.close()

    def test_dashboard_kpis_follow_selected_result(self) -> None:
        window = MainWindow()
        try:
            result = CaseResult(
                case_name="case_dashboard",
                case_dir=Path(),
                status="ok",
                evaporation_time=2.5,
                rows=[TimeResult(0.0, 3.0, 1.0, 1, 1, contact_angle_deg=82.0)],
            )
            window.results.append(result)
            window.add_result_row(result)
            window.table.setCurrentCell(0, 0)
            window.update_selected_case_plots()

            self.assertEqual(window.kpi_case_value.text(), "case_dashboard")
            self.assertEqual(window.kpi_volume_value.text(), "3")
            self.assertEqual(window.kpi_evaporation_value.text(), "2.5")
            self.assertEqual(window.kpi_contact_value.text(), "82")
        finally:
            window.close()

    def test_result_table_shows_completed_with_warning(self) -> None:
        window = MainWindow()
        try:
            result = CaseResult(
                case_name="case_warning",
                case_dir=Path(),
                status="ok",
                rows=[TimeResult(0.0, 1.0, 1.0, 1, 1)],
                warnings=["Skipped one incomplete time field"],
            )

            window.add_result_row(result)

            self.assertEqual(window.table.item(0, 14).text(), "完了（警告あり）")
            self.assertIn(
                "Skipped one incomplete time field",
                window.table.item(0, 15).text(),
            )
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
                md_series = next(series for series in plot._last_series[3] if series.label == "MD")
                self.assertEqual(md_series.color, "#f2f5f3")
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
                self.assertEqual(dialog.preview_plot.figure.get_facecolor()[:3], (1.0, 1.0, 1.0))
                self.assertEqual(dialog.preview_plot.figure.axes[0].get_facecolor()[:3], (1.0, 1.0, 1.0))
                expected_size = (
                    round(dialog.width_spin.value() * PNG_PREVIEW_PIXELS_PER_INCH),
                    round(dialog.height_spin.value() * PNG_PREVIEW_PIXELS_PER_INCH),
                )
                self.assertEqual(
                    (dialog.preview_plot.width(), dialog.preview_plot.height()),
                    expected_size,
                )
                dialog.resize(1400, 900)
                self.app.processEvents()
                self.assertEqual(
                    (dialog.preview_plot.width(), dialog.preview_plot.height()),
                    expected_size,
                )
                dialog.width_spin.setValue(9.0)
                self.assertEqual(
                    dialog.preview_plot.width(),
                    round(9.0 * PNG_PREVIEW_PIXELS_PER_INCH),
                )
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
                    with Image.open(output_path) as image:
                        self.assertEqual(image.convert("RGB").getpixel((0, 0)), (255, 255, 255))
            finally:
                dialog.close()
        finally:
            first.close()
            second.close()

    def test_png_export_uses_print_theme_and_restores_dark_canvas(self) -> None:
        plot = PlotWidget()
        try:
            plot.plot_xy("title", "x", "y", [0.0, 1.0], [1.0, 2.0])
            self.assertNotEqual(plot.figure.get_facecolor()[:3], (1.0, 1.0, 1.0))
            with tempfile.TemporaryDirectory() as directory:
                output_path = Path(directory) / "plot.png"
                plot.save_png(output_path)
                with Image.open(output_path) as image:
                    self.assertEqual(image.convert("RGB").getpixel((0, 0)), (255, 255, 255))
            self.assertFalse(plot.light_theme)
            self.assertNotEqual(plot.figure.get_facecolor()[:3], (1.0, 1.0, 1.0))
        finally:
            plot.close()

    def test_png_preview_updates_point_text_and_grid_settings_immediately(self) -> None:
        plot = PlotWidget()
        try:
            plot.plot_xy("original", "time", "volume", [0.0, 1.0], [1.0, 2.0])
            dialog = GraphPngPreviewDialog(plot, ".")
            try:
                previous_size = dialog.point_size_spin.value()
                dialog.point_size_spin.stepUp()
                self.app.processEvents()

                self.assertEqual(dialog.preview_plot.settings.point_size, previous_size + 1.0)
                axis = dialog.preview_plot.figure.axes[0]
                self.assertEqual(axis.collections[0].get_sizes()[0], previous_size + 1.0)

                dialog.title_edit.setText("custom title")
                dialog.x_label_edit.setText("custom x")
                dialog.y_label_edit.setText("custom y")
                dialog.title_font_size_spin.setValue(19)
                dialog.axis_label_font_size_spin.setValue(15)
                dialog.font_size_spin.setValue(12)
                dialog.grid_alpha_spin.setValue(0.25)
                dialog.grid_line_width_spin.setValue(1.7)
                dialog.grid_line_style_combo.setCurrentText("破線")
                dialog.x_tick_rotation_spin.setValue(30.0)
                self.app.processEvents()

                axis = dialog.preview_plot.figure.axes[0]
                self.assertEqual(axis.get_title(), "custom title")
                self.assertEqual(axis.get_xlabel(), "custom x")
                self.assertEqual(axis.get_ylabel(), "custom y")
                self.assertEqual(axis.title.get_fontsize(), 19)
                self.assertEqual(axis.xaxis.label.get_fontsize(), 15)
                self.assertEqual(axis.get_xticklabels()[0].get_fontsize(), 12)
                self.assertEqual(axis.get_xticklabels()[0].get_rotation(), 30.0)
                visible_grid_lines = [line for line in axis.get_xgridlines() if line.get_visible()]
                self.assertTrue(visible_grid_lines)
                self.assertEqual(visible_grid_lines[0].get_alpha(), 0.25)
                self.assertEqual(visible_grid_lines[0].get_linewidth(), 1.7)
                self.assertEqual(visible_grid_lines[0].get_linestyle(), "--")
            finally:
                dialog.close()
        finally:
            plot.close()

    def test_png_preview_updates_series_legend_marker_and_line_style(self) -> None:
        plot = PlotWidget()
        try:
            plot.plot_series(
                "series",
                "x",
                "y",
                [
                    PlotSeries("points", [0.0, 1.0], [1.0, 2.0], marker="s"),
                    PlotSeries("line", [0.0, 1.0], [2.0, 3.0], style="line", linestyle="--"),
                ],
            )
            dialog = GraphPngPreviewDialog(plot, ".")
            try:
                dialog.marker_combo.setCurrentText("^")
                dialog.marker_override_check.setChecked(True)
                dialog.line_width_override_check.setChecked(True)
                dialog.line_width_spin.setValue(3.0)
                dialog.line_style_combo.setCurrentText("点線")
                dialog.legend_font_size_spin.setValue(14)
                dialog.legend_location_combo.setCurrentText("左上")
                self.app.processEvents()

                axis = dialog.preview_plot.figure.axes[0]
                self.assertEqual(axis.collections[0].get_paths()[0].vertices.shape[0], 4)
                self.assertEqual(axis.lines[0].get_linewidth(), 3.0)
                self.assertEqual(axis.lines[0].get_linestyle(), ":")
                legend = axis.get_legend()
                self.assertIsNotNone(legend)
                self.assertEqual(legend.get_texts()[0].get_fontsize(), 14)
                self.assertEqual(legend._loc, 2)

                dialog.legend_check.setChecked(False)
                self.app.processEvents()
                self.assertIsNone(dialog.preview_plot.figure.axes[0].get_legend())
            finally:
                dialog.close()
        finally:
            plot.close()

    def test_png_preview_settings_fit_panel_explain_controls_and_reset(self) -> None:
        plot = PlotWidget()
        try:
            plot.plot_xy("original", "time", "volume", [0.0, 1.0], [1.0, 2.0])
            dialog = GraphPngPreviewDialog(plot, ".")
            try:
                dialog.resize(900, 650)
                dialog.show()
                self.app.processEvents()

                settings_widget = dialog.settings_scroll.widget()
                viewport = dialog.settings_scroll.viewport()
                self.assertEqual(
                    dialog.settings_scroll.horizontalScrollBar().maximum(),
                    0,
                )
                self.assertLessEqual(
                    settings_widget.minimumSizeHint().width(),
                    viewport.width(),
                )

                explained_controls = (
                    dialog.title_edit,
                    dialog.x_label_edit,
                    dialog.y_label_edit,
                    dialog.color_button,
                    dialog.point_size_spin,
                    dialog.alpha_spin,
                    dialog.font_size_spin,
                    dialog.marker_combo,
                    dialog.aspect_combo,
                    dialog.title_check,
                    dialog.axis_label_check,
                    dialog.tick_label_check,
                    dialog.grid_check,
                    dialog.axis_auto_check,
                    dialog.x_min_spin,
                    dialog.x_max_spin,
                    dialog.y_min_spin,
                    dialog.y_max_spin,
                    dialog.x_log_check,
                    dialog.y_log_check,
                    dialog.x_tick_rotation_spin,
                    dialog.width_spin,
                    dialog.height_spin,
                    dialog.quality_combo,
                    dialog.transparent_check,
                )
                self.assertTrue(all(control.toolTip() for control in explained_controls))
                self.assertIn("#", dialog.color_button.text())
                self.assertTrue(dialog.point_size_spin.isEnabled())
                self.assertFalse(dialog.line_width_override_check.isEnabled())
                self.assertFalse(dialog.legend_check.isEnabled())

                original_size = dialog.preview_plot.settings.point_size
                dialog.point_size_spin.setValue(original_size + 10.0)
                dialog.title_edit.setText("changed")
                self.assertNotEqual(
                    dialog.preview_plot.settings.point_size,
                    original_size,
                )
                dialog.reset_settings()
                self.assertEqual(
                    dialog.preview_plot.settings.point_size,
                    original_size,
                )
                self.assertEqual(dialog.title_edit.text(), "original")
                self.assertEqual(
                    dialog.preview_plot.figure.axes[0].get_title(),
                    "original",
                )
            finally:
                dialog.close()
        finally:
            plot.close()

    def test_all_case_png_preview_returns_settings_without_saving_one_file(
        self,
    ) -> None:
        plot = PlotWidget()
        try:
            plot.plot_xy("case_a: volume", "time", "volume", [0.0, 1.0], [1.0, 2.0])
            dialog = GraphPngPreviewDialog(
                plot,
                ".",
                configure_only=True,
            )
            try:
                dialog.point_size_spin.setValue(31.0)
                dialog.quality_combo.setCurrentIndex(
                    dialog.quality_combo.findData(600)
                )
                dialog.axis_auto_check.setChecked(True)
                with patch.object(
                    QFileDialog,
                    "getSaveFileName",
                    side_effect=AssertionError("single-file dialog must not open"),
                ):
                    dialog.save_png()

                self.assertEqual(dialog.result(), QDialog.Accepted)
                self.assertIsNone(dialog.saved_path)
                self.assertIsNotNone(dialog.configured_settings)
                self.assertEqual(dialog.configured_settings.point_size, 31.0)
                self.assertEqual(dialog.configured_settings.dpi, 600)
                self.assertTrue(dialog.configured_settings.axis_auto)
                self.assertEqual(
                    dialog.configured_settings.axis_mode,
                    "per_plot_auto",
                )
                self.assertIsNone(dialog.configured_settings.title_text)
            finally:
                dialog.close()
        finally:
            plot.close()

    def test_all_case_png_preview_accepts_combined_plot(self) -> None:
        first = PlotWidget()
        second = PlotWidget()
        combined = None
        try:
            first.plot_xy("first", "x", "y", [0.0], [1.0])
            second.plot_xy("second", "x", "y", [0.0], [2.0])
            combined = CombinedPlotWidget([first, second])
            dialog = GraphPngPreviewDialog(
                combined,
                ".",
                configure_only=True,
            )
            try:
                self.assertIsInstance(dialog.preview_plot, CombinedPlotWidget)
                dialog.height_spin.setValue(12.0)
                dialog.save_png()
                self.assertEqual(dialog.result(), QDialog.Accepted)
                self.assertEqual(dialog.configured_settings.image_height, 12.0)
            finally:
                dialog.close()
        finally:
            if combined is not None:
                combined.close()
            first.close()
            second.close()

    def test_all_case_png_export_applies_preview_settings_to_every_case(
        self,
    ) -> None:
        window = MainWindow()
        try:
            for name, volume in (("case_a", 1.0), ("case_b", 2.0)):
                result = CaseResult(
                    case_name=name,
                    case_dir=Path(),
                    status="ok",
                    rows=[TimeResult(0.0, volume, volume, 1, 1)],
                )
                window.results.append(result)
                window.add_result_row(result)
            window.table.setCurrentCell(0, 0)
            window.tabs.setCurrentWidget(window.volume_plot)
            window.update_selected_case_plots()

            configured = GraphSettings(
                point_color="#123456",
                point_size=27.0,
                image_width=9.0,
                image_height=6.0,
                dpi=600,
                title_text=None,
            )
            settings_dialog = Mock()
            settings_dialog.exec.return_value = QDialog.Accepted
            settings_dialog.configured_settings = configured
            saved: list[tuple[Path, GraphSettings, str]] = []

            def capture_save(plot: PlotWidget, path: Path) -> None:
                saved.append(
                    (
                        Path(path),
                        GraphSettings(**vars(plot.settings)),
                        plot._last_plot[1],
                    )
                )

            with tempfile.TemporaryDirectory() as directory:
                output_dir = Path(directory) / "all_png"
                with (
                    patch(
                        "mdfoam_analyzer.gui.GraphPngPreviewDialog",
                        return_value=settings_dialog,
                    ) as preview_class,
                    patch.object(
                        window,
                        "_choose_output_directory",
                        return_value=output_dir,
                    ),
                    patch.object(
                        PlotWidget,
                        "save_png",
                        autospec=True,
                        side_effect=capture_save,
                    ),
                ):
                    window.export_all_png()

            self.assertEqual(len(saved), 2)
            self.assertEqual(
                [item[0].name for item in saved],
                ["case_a_volume_time.png", "case_b_volume_time.png"],
            )
            self.assertEqual([item[2] for item in saved], ["case_a: 体積-時間", "case_b: 体積-時間"])
            self.assertTrue(all(item[1].point_color == "#123456" for item in saved))
            self.assertTrue(all(item[1].point_size == 27.0 for item in saved))
            self.assertTrue(all(item[1].dpi == 600 for item in saved))
            self.assertTrue(preview_class.call_args.kwargs["configure_only"])
        finally:
            window.close()

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
