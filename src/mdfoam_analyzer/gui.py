from __future__ import annotations

from pathlib import Path
import sys
import traceback

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtGui import QDoubleValidator
from PySide6.QtGui import QKeySequence

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from .analysis import (
    AnalysisSettings,
    CaseResult,
    analyze_case,
    discover_cases,
    discover_fields_for_cases,
    write_summary_csv,
    write_timeseries_csv,
)


class ScientificDoubleSpinBox(QDoubleSpinBox):
    def __init__(self) -> None:
        super().__init__()
        self.validator = QDoubleValidator(0.0, 1.0e100, 16, self)
        self.validator.setNotation(QDoubleValidator.ScientificNotation)

    def textFromValue(self, value: float) -> str:
        return f"{value:.8g}"

    def valueFromText(self, text: str) -> float:
        try:
            return max(0.0, float(text.strip()))
        except ValueError:
            return 0.0

    def validate(self, text: str, position: int):
        return self.validator.validate(text, position)


class ResultsTable(QTableWidget):
    def keyPressEvent(self, event) -> None:
        if event.matches(QKeySequence.Copy):
            self.copy_selection()
            return
        super().keyPressEvent(event)

    def copy_selection(self) -> None:
        indexes = self.selectedIndexes()
        if not indexes:
            current = self.currentIndex()
            if current.isValid():
                indexes = [current]
            else:
                return

        rows = sorted({index.row() for index in indexes})
        columns = sorted({index.column() for index in indexes})
        selected = {(index.row(), index.column()) for index in indexes}
        lines: list[str] = []
        for row in rows:
            values: list[str] = []
            for column in columns:
                item = self.item(row, column)
                values.append(item.text() if item and (row, column) in selected else "")
            lines.append("\t".join(values))
        QApplication.clipboard().setText("\n".join(lines))


class AnalyzerWorker(QObject):
    progress = Signal(int, int)
    log = Signal(str)
    case_finished = Signal(object)
    finished = Signal()

    def __init__(self, cases: list[Path], settings: AnalysisSettings) -> None:
        super().__init__()
        self.cases = cases
        self.settings = settings
        self._stop_requested = False

    @Slot()
    def run(self) -> None:
        try:
            total = len(self.cases)
            for index, case in enumerate(self.cases, start=1):
                if self._stop_requested:
                    self.log.emit("残りのケースを解析せずに停止しました。")
                    break
                self.log.emit(f"{case.name} を解析中...")
                result = analyze_case(
                    case,
                    self.settings,
                    stop_requested=lambda: self._stop_requested,
                    log=self.log.emit,
                )
                self.case_finished.emit(result)
                self.progress.emit(index, total)
        except Exception:
            self.log.emit(traceback.format_exc())
        finally:
            self.finished.emit()

    @Slot()
    def stop(self) -> None:
        self._stop_requested = True


class PlotWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.figure = Figure(figsize=(5, 3), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

    def clear(self, title: str) -> None:
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.set_title(title)
        axis.grid(True, alpha=0.3)
        self.canvas.draw_idle()

    def plot_xy(self, title: str, x_label: str, y_label: str, x: list[float], y: list[float]) -> None:
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.scatter(x, y, s=18)
        axis.set_title(title)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.grid(True, alpha=0.3)
        self.canvas.draw_idle()

    def plot_bar(self, title: str, labels: list[str], values: list[float]) -> None:
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.bar(labels, values)
        axis.set_title(title)
        axis.set_ylabel("蒸発完了時間 [s]")
        axis.tick_params(axis="x", rotation=45)
        axis.grid(True, axis="y", alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def save_png(self, path: Path) -> None:
        self.figure.savefig(path, dpi=180)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("mdFOAM 密度解析アプリ")
        self.resize(1300, 820)

        self.cases: list[Path] = []
        self.results: list[CaseResult] = []
        self.worker: AnalyzerWorker | None = None
        self.thread: QThread | None = None

        self._build_ui()
        self._connect_signals()
        self.folder_edit.setText(str(Path.cwd()))
        self.load_folder(Path.cwd())

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)

        top_row = QHBoxLayout()
        self.folder_edit = QLabel()
        self.folder_edit.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.browse_button = QPushButton("フォルダ選択...")
        self.refresh_button = QPushButton("更新")
        top_row.addWidget(QLabel("親フォルダ:"))
        top_row.addWidget(self.folder_edit, 1)
        top_row.addWidget(self.browse_button)
        top_row.addWidget(self.refresh_button)
        root_layout.addLayout(top_row)

        splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(splitter, 1)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        splitter.addWidget(left_panel)

        self.case_list = QListWidget()
        self.case_list.setSelectionMode(QAbstractItemView.SingleSelection)
        left_layout.addWidget(QLabel("ケース一覧"))
        left_layout.addWidget(self.case_list, 1)

        settings_group = QGroupBox("解析設定")
        settings_layout = QGridLayout(settings_group)
        self.field_combo = QComboBox()
        self.threshold_spin = self._scientific_spin(500.0)
        self.zero_spin = self._scientific_spin(0.0)
        self.zero_count_spin = QSpinBox()
        self.zero_count_spin.setRange(1, 999)
        self.zero_count_spin.setValue(3)
        self.cell_volume_spin = self._scientific_spin(0.0)
        self.dx_spin = self._scientific_spin(0.0)
        self.dy_spin = self._scientific_spin(0.0)
        self.dz_spin = self._scientific_spin(0.0)

        settings_layout.addWidget(QLabel("密度フィールド"), 0, 0)
        settings_layout.addWidget(self.field_combo, 0, 1)
        settings_layout.addWidget(QLabel("密度しきい値"), 1, 0)
        settings_layout.addWidget(self.threshold_spin, 1, 1)
        settings_layout.addWidget(QLabel("0判定許容値"), 2, 0)
        settings_layout.addWidget(self.zero_spin, 2, 1)
        settings_layout.addWidget(QLabel("連続ゼロ数"), 3, 0)
        settings_layout.addWidget(self.zero_count_spin, 3, 1)
        settings_layout.addWidget(QLabel("セル体積"), 4, 0)
        settings_layout.addWidget(self.cell_volume_spin, 4, 1)
        settings_layout.addWidget(QLabel("dx"), 5, 0)
        settings_layout.addWidget(self.dx_spin, 5, 1)
        settings_layout.addWidget(QLabel("dy"), 6, 0)
        settings_layout.addWidget(self.dy_spin, 6, 1)
        settings_layout.addWidget(QLabel("dz"), 7, 0)
        settings_layout.addWidget(self.dz_spin, 7, 1)
        left_layout.addWidget(settings_group)

        button_row = QHBoxLayout()
        self.run_button = QPushButton("解析実行")
        self.stop_button = QPushButton("停止")
        self.stop_button.setEnabled(False)
        button_row.addWidget(self.run_button)
        button_row.addWidget(self.stop_button)
        left_layout.addLayout(button_row)

        export_row = QHBoxLayout()
        self.export_csv_button = QPushButton("CSV出力")
        self.export_png_button = QPushButton("PNG出力")
        export_row.addWidget(self.export_csv_button)
        export_row.addWidget(self.export_png_button)
        left_layout.addLayout(export_row)

        self.progress = QProgressBar()
        left_layout.addWidget(self.progress)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        left_layout.addWidget(QLabel("ログ"))
        left_layout.addWidget(self.log_box, 1)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self.table = ResultsTable(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["ケース", "時刻数", "最大体積", "最終体積", "蒸発完了時刻", "状態", "エラー"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionsClickable(True)
        self.table.horizontalHeader().sectionClicked.connect(self.select_table_column)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setStyleSheet(
            """
            QTableWidget::item {
                padding-left: 6px;
                padding-right: 6px;
            }
            QTableWidget::item:selected {
                background: #e7f0fb;
                color: #111111;
            }
            QTableWidget::item:focus {
                outline: 0;
                border: none;
            }
            """
        )
        right_layout.addWidget(self.table, 1)

        self.tabs = QTabWidget()
        self.volume_plot = PlotWidget()
        self.radius_plot = PlotWidget()
        self.evap_plot = PlotWidget()
        self.tabs.addTab(self.volume_plot, "体積-時間")
        self.tabs.addTab(self.radius_plot, "等価半径-時間")
        self.tabs.addTab(self.evap_plot, "蒸発完了時刻")
        right_layout.addWidget(self.tabs, 2)
        self.volume_plot.clear("体積-時間")
        self.radius_plot.clear("等価半径-時間")
        self.evap_plot.clear("蒸発完了時刻")

    def _connect_signals(self) -> None:
        self.browse_button.clicked.connect(self.choose_folder)
        self.refresh_button.clicked.connect(lambda: self.load_folder(Path(self.folder_edit.text())))
        self.run_button.clicked.connect(self.start_analysis)
        self.stop_button.clicked.connect(self.stop_analysis)
        self.export_csv_button.clicked.connect(self.export_csv)
        self.export_png_button.clicked.connect(self.export_png)
        self.table.itemSelectionChanged.connect(self.update_selected_case_plots)

    @Slot(int)
    def select_table_column(self, column: int) -> None:
        self.table.clearSelection()
        for row in range(self.table.rowCount()):
            item = self.table.item(row, column)
            if item:
                item.setSelected(True)
        if self.table.rowCount() > 0:
            self.table.setCurrentCell(0, column)

    def _scientific_spin(self, value: float) -> QDoubleSpinBox:
        spin = ScientificDoubleSpinBox()
        spin.setDecimals(16)
        spin.setRange(0.0, 1.0e100)
        spin.setSingleStep(1.0)
        spin.setValue(value)
        return spin

    @Slot()
    def choose_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "親フォルダを選択", self.folder_edit.text())
        if path:
            self.load_folder(Path(path))

    def load_folder(self, path: Path) -> None:
        self.folder_edit.setText(str(path))
        self.case_list.clear()
        self.field_combo.clear()
        self.results.clear()
        self.table.setRowCount(0)
        try:
            self.cases = discover_cases(path)
            fields = discover_fields_for_cases(self.cases)
            if not fields:
                fields = ["rhoM_water"]
            self.field_combo.addItems(fields)
            default_index = self.field_combo.findText("rhoM_water")
            if default_index >= 0:
                self.field_combo.setCurrentIndex(default_index)
            for case in self.cases:
                item = QListWidgetItem(case.name)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked)
                item.setData(Qt.UserRole, str(case))
                self.case_list.addItem(item)
            self.log(f"{path} から {len(self.cases)} ケースを読み込みました。")
            self.log(f"密度フィールド: {', '.join(fields)}")
        except Exception as exc:
            self.cases = []
            self.log(f"フォルダ読み込みに失敗しました: {exc}")

    def selected_cases(self) -> list[Path]:
        cases: list[Path] = []
        for index in range(self.case_list.count()):
            item = self.case_list.item(index)
            if item.checkState() == Qt.Checked:
                cases.append(Path(item.data(Qt.UserRole)))
        return cases

    def settings(self) -> AnalysisSettings:
        return AnalysisSettings(
            density_field=self.field_combo.currentText() or "rhoM_water",
            density_threshold=self.threshold_spin.value(),
            zero_tolerance=self.zero_spin.value(),
            consecutive_zero_count=self.zero_count_spin.value(),
            manual_cell_volume=self.cell_volume_spin.value() or None,
            dx=self.dx_spin.value() or None,
            dy=self.dy_spin.value() or None,
            dz=self.dz_spin.value() or None,
        )

    @Slot()
    def start_analysis(self) -> None:
        cases = self.selected_cases()
        if not cases:
            QMessageBox.warning(self, "ケースなし", "少なくとも1つのケースを選択してください。")
            return

        self.results.clear()
        self.table.setRowCount(0)
        self.progress.setRange(0, len(cases))
        self.progress.setValue(0)
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.log("解析を開始しました。")

        self.thread = QThread(self)
        self.worker = AnalyzerWorker(cases, self.settings())
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_progress)
        self.worker.log.connect(self.log)
        self.worker.case_finished.connect(self.on_case_finished)
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    @Slot()
    def stop_analysis(self) -> None:
        if self.worker:
            self.worker.stop()
            self.log("停止を要求しました。")
        self.stop_button.setEnabled(False)

    @Slot(int, int)
    def on_progress(self, current: int, total: int) -> None:
        self.progress.setRange(0, total)
        self.progress.setValue(current)

    @Slot(object)
    def on_case_finished(self, result: CaseResult) -> None:
        self.results.append(result)
        self.add_result_row(result)
        self.update_evap_plot()
        if len(self.results) == 1:
            self.table.selectRow(0)

    @Slot()
    def on_analysis_finished(self) -> None:
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.worker = None
        self.thread = None
        self.log("解析が完了しました。")

    def add_result_row(self, result: CaseResult) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        values = [
            result.case_name,
            str(result.time_count),
            _fmt(result.max_volume),
            _fmt(result.final_volume),
            "" if result.evaporation_time is None else _fmt(result.evaporation_time),
            _status_label(result.status),
            result.error,
        ]
        for column, value in enumerate(values):
            item = QTableWidgetItem(value)
            item.setData(Qt.UserRole, len(self.results) - 1)
            self.table.setItem(row, column, item)

    @Slot()
    def update_selected_case_plots(self) -> None:
        current_row = self.table.currentRow()
        if current_row < 0:
            return
        item = self.table.item(current_row, 0)
        if item is None:
            return
        result_index = item.data(Qt.UserRole)
        if result_index is None or result_index >= len(self.results):
            return
        result = self.results[result_index]
        times = [row.time for row in result.rows]
        volumes = [row.volume for row in result.rows]
        radii = [row.equivalent_radius for row in result.rows]
        self.volume_plot.plot_xy(f"{result.case_name}: 体積-時間", "時間 [s]", "体積 [m^3]", times, volumes)
        self.radius_plot.plot_xy(
            f"{result.case_name}: 等価半径-時間",
            "時間 [s]",
            "等価半径 [m]",
            times,
            radii,
        )

    def update_evap_plot(self) -> None:
        labels = [result.case_name for result in self.results if result.evaporation_time is not None]
        values = [result.evaporation_time for result in self.results if result.evaporation_time is not None]
        if labels:
            self.evap_plot.plot_bar("蒸発完了時刻", labels, values)
        else:
            self.evap_plot.clear("蒸発完了時刻")

    @Slot()
    def export_csv(self) -> None:
        if not self.results:
            QMessageBox.information(self, "結果なし", "出力前に解析を実行してください。")
            return
        directory = QFileDialog.getExistingDirectory(self, "出力フォルダを選択", self.folder_edit.text())
        if not directory:
            return
        out_dir = Path(directory)
        write_summary_csv(out_dir / "mdfoam_summary.csv", self.results)
        write_timeseries_csv(out_dir / "mdfoam_timeseries.csv", self.results)
        self.log(f"CSVを出力しました: {out_dir}")

    @Slot()
    def export_png(self) -> None:
        tab = self.tabs.currentWidget()
        if not isinstance(tab, PlotWidget):
            return
        path, _ = QFileDialog.getSaveFileName(self, "現在のグラフを保存", self.folder_edit.text(), "PNG (*.png)")
        if not path:
            return
        tab.save_png(Path(path))
        self.log(f"PNGを出力しました: {path}")

    @Slot(str)
    def log(self, message: str) -> None:
        self.log_box.append(message)


def _fmt(value: float) -> str:
    return f"{value:.8g}"


def _status_label(status: str) -> str:
    labels = {
        "ok": "完了",
        "error": "エラー",
        "stopped": "停止",
        "running": "実行中",
    }
    return labels.get(status, status)


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
