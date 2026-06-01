from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import traceback

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
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
from matplotlib.animation import PillowWriter
from matplotlib.figure import Figure
from matplotlib import rcParams
import numpy as np

from .analysis import (
    AnalysisSettings,
    CaseResult,
    analyze_case,
    discover_cases,
    discover_fields_for_cases,
    write_summary_csv,
    write_timeseries_csv,
)
from .cache import clear_cache
from .remote import (
    RemoteError,
    RemoteProfile,
    SshConnection,
    discover_remote_cases,
    discover_remote_fields_for_cases,
    list_remote_dirs,
    normalize_remote_path,
    remote_join,
    remote_name,
    sync_remote_case,
    sync_remote_lagrangian_time,
    validate_private_key_path,
)
from .visualization import (
    PlotBounds,
    VisualizationFrame,
    case_time_dirs,
    downsample_points_by_id,
    id_draw_order,
    load_visualization_frame,
    plot_bounds_from_point_bounds,
    read_remote_case_from_manifest,
    replicate_xy,
)


rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

THREE_D_AUTO_MAX_POINTS = 50_000


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

    def __init__(
        self,
        cases: list[Path] | list[str],
        settings: AnalysisSettings,
        remote_profile: RemoteProfile | None = None,
    ):
        super().__init__()
        self.cases = cases
        self.settings = settings
        self.remote_profile = remote_profile
        self._stop_requested = False

    @Slot()
    def run(self) -> None:
        try:
            if self.remote_profile is not None:
                self._run_remote()
                return

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

    def _run_remote(self) -> None:
        profile = self.remote_profile
        if profile is None:
            return
        total = len(self.cases)
        with SshConnection(profile) as connection:
            sftp = connection.sftp
            for index, case in enumerate(self.cases, start=1):
                if self._stop_requested:
                    self.log.emit("残りのケースを解析せずに停止しました。")
                    break
                remote_case = str(case)
                self.log.emit(f"{remote_name(remote_case)} をSFTP同期中...")
                try:
                    local_case = sync_remote_case(
                        sftp,
                        profile,
                        remote_case,
                        self.settings.density_field,
                        stop_requested=lambda: self._stop_requested,
                        log=self.log.emit,
                    )
                    if self._stop_requested:
                        self.progress.emit(index, total)
                        break
                    self.log.emit(f"{remote_name(remote_case)} を解析中...")
                    result = analyze_case(
                        local_case,
                        self.settings,
                        stop_requested=lambda: self._stop_requested,
                        log=self.log.emit,
                    )
                except Exception as exc:
                    result = CaseResult(
                        case_name=remote_name(remote_case),
                        case_dir=Path(),
                        status="error",
                        error=str(exc),
                    )
                self.case_finished.emit(result)
                self.progress.emit(index, total)

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


class VisualizationPlotWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.figure = Figure(figsize=(6, 4), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

    def clear(self, message: str = "可視化するケースと時刻を選択してください") -> None:
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes)
        axis.set_axis_off()
        self.canvas.draw_idle()

    def draw_frame(
        self,
        frame: VisualizationFrame,
        mode: str,
        projection: str,
        periodic_enabled: bool,
        tile_count: int,
        point_size: float,
        max_points: int,
        show_legend: bool,
        show_title: bool,
        show_axis_labels: bool,
        show_axis_ticks: bool,
        show_info_text: bool,
        show_grid: bool,
        show_liquid: bool,
        show_fit: bool,
    ) -> None:
        self.figure.clear()
        is_3d = mode.startswith("3D")
        particles = frame.particles.positions
        particle_ids = frame.particles.ids
        downsample = None
        if periodic_enabled:
            normalized_tile_count = max(1, min(16, int(tile_count)))
            full_periodic_count = len(particles) * normalized_tile_count * normalized_tile_count
            if is_3d:
                effective_max_points = max_points if max_points > 0 else THREE_D_AUTO_MAX_POINTS
                if effective_max_points > 0 and full_periodic_count > effective_max_points:
                    tile_total = normalized_tile_count * normalized_tile_count
                    base_limit = max(1, (effective_max_points + tile_total - 1) // tile_total)
                    base_downsample = downsample_points_by_id(particles, particle_ids, base_limit)
                    particles, particle_ids = replicate_xy(
                        base_downsample.positions,
                        frame.point_bounds,
                        normalized_tile_count,
                        base_downsample.ids,
                    )
                    final_downsample = downsample_points_by_id(particles, particle_ids, effective_max_points)
                    particles = final_downsample.positions
                    particle_ids = final_downsample.ids
                    downsample = final_downsample.__class__(
                        particles,
                        particle_ids,
                        full_periodic_count,
                        len(particles),
                        True,
                    )
                else:
                    particles, particle_ids = replicate_xy(
                        particles,
                        frame.point_bounds,
                        normalized_tile_count,
                        particle_ids,
                    )
            else:
                particles, particle_ids = replicate_xy(
                    particles,
                    frame.point_bounds,
                    normalized_tile_count,
                    particle_ids,
                )
        effective_max_points = max_points
        if is_3d and effective_max_points <= 0:
            effective_max_points = THREE_D_AUTO_MAX_POINTS
        if downsample is None:
            downsample = downsample_points_by_id(particles, particle_ids, effective_max_points)
            particles = downsample.positions
            particle_ids = downsample.ids
        bounds = plot_bounds_from_point_bounds(
            frame.point_bounds,
            tile_count if periodic_enabled else 1,
        )

        if is_3d:
            axis = self.figure.add_subplot(111, projection="3d")
            self._draw_3d(axis, frame, particles, particle_ids, point_size, show_legend, show_liquid, show_fit, bounds)
        else:
            axis = self.figure.add_subplot(111)
            self._draw_2d(axis, frame, particles, particle_ids, projection, point_size, show_legend, show_liquid, show_fit, bounds)
        self._apply_visibility_options(
            axis,
            show_title,
            show_axis_labels,
            show_axis_ticks,
            show_info_text,
            show_grid,
            frame,
            downsample,
            is_3d,
        )
        self.canvas.draw_idle()
        return downsample

    def save_png(self, path: Path) -> None:
        self.figure.savefig(path, dpi=180)

    def _draw_2d(
        self,
        axis,
        frame: VisualizationFrame,
        particles: np.ndarray,
        particle_ids: np.ndarray | None,
        projection: str,
        point_size: float,
        show_legend: bool,
        show_liquid: bool,
        show_fit: bool,
        bounds: PlotBounds | None,
    ) -> None:
        axis_index, labels = _projection_axes(projection)
        self._scatter_by_id_2d(axis, particles, particle_ids, axis_index, point_size, show_legend)

        if show_liquid and len(frame.selected_centers) > 0:
            axis.scatter(
                frame.selected_centers[:, axis_index[0]],
                frame.selected_centers[:, axis_index[1]],
                s=max(point_size * 0.7, 2.0),
                c="#4c78a8",
                alpha=0.28,
                label="液滴セル",
                zorder=10,
            )

        contact = frame.contact
        if show_fit and len(contact.points) > 0 and len(contact.fit_mask) == len(contact.points):
            fit_points = contact.points[contact.fit_mask]
            if len(fit_points) > 0:
                axis.scatter(
                    fit_points[:, axis_index[0]],
                    fit_points[:, axis_index[1]],
                    s=max(point_size * 1.5, 8.0),
                    facecolors="none",
                    edgecolors="#d62728",
                    linewidths=0.8,
                    label="fit点",
                    zorder=12,
                )
            self._draw_fit_geometry_2d(axis, contact, projection, axis_index)

        axis.set_title(f"{frame.time_name}: {projection} 可視化")
        axis.set_xlabel(labels[0])
        axis.set_ylabel(labels[1])
        axis.set_aspect("equal", adjustable="box")
        self._apply_2d_bounds(axis, projection, bounds)
        if show_legend:
            axis.legend(loc="best", fontsize=8, markerscale=1.8)

    def _draw_3d(
        self,
        axis,
        frame: VisualizationFrame,
        particles: np.ndarray,
        particle_ids: np.ndarray | None,
        point_size: float,
        show_legend: bool,
        show_liquid: bool,
        show_fit: bool,
        bounds: PlotBounds | None,
    ) -> None:
        self._scatter_by_id_3d(axis, particles, particle_ids, point_size, show_legend)
        if show_liquid and len(frame.selected_centers) > 0:
            centers = frame.selected_centers
            axis.scatter(centers[:, 0], centers[:, 1], centers[:, 2], s=max(point_size * 0.5, 1.0), c="#4c78a8", alpha=0.12, label="液滴セル")
        contact = frame.contact
        if show_fit and len(contact.points) > 0 and len(contact.fit_mask) == len(contact.points):
            fit_points = contact.points[contact.fit_mask]
            if len(fit_points) > 0:
                axis.scatter(fit_points[:, 0], fit_points[:, 1], fit_points[:, 2], s=max(point_size * 1.4, 7.0), c="#d62728", alpha=0.85, label="fit点")
            if contact.sphere_center is not None and contact.sphere_radius is not None:
                self._draw_sphere_wire(axis, contact.sphere_center, contact.sphere_radius)
            if contact.z_base is not None and frame.point_bounds is not None:
                x0, x1 = frame.point_bounds[0]
                y0, y1 = frame.point_bounds[1]
                axis.plot([x0, x1], [y0, y0], [contact.z_base, contact.z_base], color="#333333", linestyle="--", linewidth=1.0, label="基板高さ")
        axis.set_title(f"{frame.time_name}: 3D概観")
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.set_zlabel("z [m]")
        self._apply_3d_bounds(axis, bounds, frame)
        if show_legend:
            axis.legend(loc="best", fontsize=8, markerscale=1.8)

    def _scatter_by_id_2d(
        self,
        axis,
        particles: np.ndarray,
        particle_ids: np.ndarray | None,
        axis_index: tuple[int, int],
        point_size: float,
        show_legend: bool,
    ) -> None:
        if len(particles) == 0:
            return
        if particle_ids is None:
            axis.scatter(particles[:, axis_index[0]], particles[:, axis_index[1]], s=point_size, c="#666666", alpha=0.7, label="粒子")
            return
        ordered_ids = id_draw_order(particles, particle_ids)
        cmap = self.figure.colormaps["tab20"] if hasattr(self.figure, "colormaps") else None
        for index, id_value in enumerate(ordered_ids):
            mask = particle_ids == id_value
            is_top = index == len(ordered_ids) - 1
            color = ("#b0b0b0" if not is_top else (f"C{index % 10}" if cmap is None else cmap(index % 20)))
            label = f"id={id_value}" if show_legend and len(ordered_ids) <= 20 else None
            axis.scatter(
                particles[mask, axis_index[0]],
                particles[mask, axis_index[1]],
                s=point_size * (1.25 if is_top else 0.35),
                color=color,
                alpha=0.88 if is_top else 0.12,
                label=label,
                zorder=3 + index,
            )

    def _scatter_by_id_3d(
        self,
        axis,
        particles: np.ndarray,
        particle_ids: np.ndarray | None,
        point_size: float,
        show_legend: bool,
    ) -> None:
        if len(particles) == 0:
            return
        if particle_ids is None:
            axis.scatter(particles[:, 0], particles[:, 1], particles[:, 2], s=point_size, c="#666666", alpha=0.7, label="粒子")
            return
        ordered_ids = id_draw_order(particles, particle_ids)
        for index, id_value in enumerate(ordered_ids):
            mask = particle_ids == id_value
            label = f"id={id_value}" if show_legend and len(ordered_ids) <= 20 else None
            is_top = index == len(ordered_ids) - 1
            axis.scatter(
                particles[mask, 0],
                particles[mask, 1],
                particles[mask, 2],
                s=point_size * (1.35 if is_top else 0.18),
                color=f"C{index % 10}" if is_top else "#b0b0b0",
                alpha=0.9 if is_top else 0.025,
                depthshade=False,
                label=label,
            )

    def _draw_fit_geometry_2d(
        self,
        axis,
        contact,
        projection: str,
        axis_index: tuple[int, int],
    ) -> None:
        center = contact.sphere_center
        radius = contact.sphere_radius
        if center is None or radius is None:
            return
        angles = np.linspace(0.0, 2.0 * np.pi, 240)
        if projection in ("xz", "yz"):
            horizontal_axis = axis_index[0]
            x_values = center[horizontal_axis] + radius * np.cos(angles)
            z_values = center[2] + radius * np.sin(angles)
            axis.plot(x_values, z_values, color="#d62728", linewidth=1.0, linestyle="--", label="球fit")
            if contact.z_base is not None:
                axis.axhline(contact.z_base, color="#333333", linewidth=1.0, linestyle="--", label="基板高さ")
            if contact.z_base is not None and contact.contact_radius is not None:
                axis.plot(
                    [center[horizontal_axis] - contact.contact_radius, center[horizontal_axis] + contact.contact_radius],
                    [contact.z_base, contact.z_base],
                    color="#2ca02c",
                    linewidth=2.0,
                    label="接触半径",
                )
        elif projection == "xy":
            x_values = center[0] + radius * np.cos(angles)
            y_values = center[1] + radius * np.sin(angles)
            axis.plot(x_values, y_values, color="#d62728", linewidth=1.0, linestyle=":", label="球fit投影")
            if contact.contact_radius is not None:
                axis.plot(
                    center[0] + contact.contact_radius * np.cos(angles),
                    center[1] + contact.contact_radius * np.sin(angles),
                    color="#2ca02c",
                    linewidth=1.2,
                    label="接触円",
                )

    def _draw_sphere_wire(self, axis, center: tuple[float, float, float], radius: float) -> None:
        u = np.linspace(0, 2 * np.pi, 24)
        v = np.linspace(0, np.pi, 12)
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
        axis.plot_wireframe(x, y, z, color="#d62728", linewidth=0.35, alpha=0.35)

    def _set_3d_equal(self, axis, frame: VisualizationFrame) -> None:
        arrays = [frame.particles.positions]
        if len(frame.selected_centers) > 0:
            arrays.append(frame.selected_centers)
        points = np.concatenate([item for item in arrays if len(item) > 0], axis=0) if any(len(item) > 0 for item in arrays) else np.empty((0, 3))
        if len(points) == 0:
            return
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        center = (mins + maxs) / 2.0
        radius = max(float(np.max(maxs - mins)) / 2.0, 1.0e-30)
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)

    def _apply_2d_bounds(self, axis, projection: str, bounds: PlotBounds | None) -> None:
        if bounds is None:
            return
        if projection == "yz":
            axis.set_xlim(bounds.y)
            axis.set_ylim(bounds.z)
        elif projection == "xy":
            axis.set_xlim(bounds.x)
            axis.set_ylim(bounds.y)
        else:
            axis.set_xlim(bounds.x)
            axis.set_ylim(bounds.z)

    def _apply_3d_bounds(self, axis, bounds: PlotBounds | None, frame: VisualizationFrame) -> None:
        if bounds is None:
            self._set_3d_equal(axis, frame)
            return
        axis.set_xlim(bounds.x)
        axis.set_ylim(bounds.y)
        axis.set_zlim(bounds.z)
        try:
            axis.set_box_aspect(
                (
                    bounds.x[1] - bounds.x[0],
                    bounds.y[1] - bounds.y[0],
                    bounds.z[1] - bounds.z[0],
                )
            )
        except Exception:
            pass

    def _apply_visibility_options(
        self,
        axis,
        show_title: bool,
        show_axis_labels: bool,
        show_axis_ticks: bool,
        show_info_text: bool,
        show_grid: bool,
        frame: VisualizationFrame,
        downsample,
        is_3d: bool,
    ) -> None:
        if not show_title:
            axis.set_title("")
        if not show_axis_labels:
            axis.set_xlabel("")
            axis.set_ylabel("")
            if is_3d and hasattr(axis, "set_zlabel"):
                axis.set_zlabel("")
        if not show_axis_ticks:
            axis.set_xticks([])
            axis.set_yticks([])
            if is_3d and hasattr(axis, "set_zticks"):
                axis.set_zticks([])
        axis.grid(show_grid, alpha=0.25)
        if show_info_text:
            self._add_info_text(axis, frame, downsample)

    def _add_info_text(self, axis, frame: VisualizationFrame, downsample) -> None:
        contact = frame.contact
        angle = "-" if contact.contact_angle_deg is None else f"{contact.contact_angle_deg:.4g} deg"
        radius = "-" if contact.contact_radius is None else f"{contact.contact_radius:.4g} m"
        text = f"粒子 {len(frame.particles.positions)} / 液滴セル {len(frame.selected_centers)} / fit点 {contact.fit_point_count}\n接触角 {angle} / 接触半径 {radius}"
        if downsample.was_downsampled:
            text += f"\n表示用間引き: {downsample.original_count} -> {downsample.displayed_count}"
        if frame.particles.id_warning:
            text += f"\n{frame.particles.id_warning}"
        if contact.failure_reason:
            text += f"\n{contact.failure_reason}"
        if hasattr(axis, "text2D"):
            axis.text2D(0.01, 0.99, text, transform=axis.transAxes, va="top", fontsize=9)
        else:
            axis.text(0.01, 0.99, text, transform=axis.transAxes, va="top", fontsize=9)


def _projection_axes(projection: str) -> tuple[tuple[int, int], tuple[str, str]]:
    if projection == "yz":
        return (1, 2), ("y [m]", "z [m]")
    if projection == "xy":
        return (0, 1), ("x [m]", "y [m]")
    return (0, 2), ("x [m]", "z [m]")


APP_SETTINGS_DIR = (
    Path(os.environ["APPDATA"]) / "mdfoam-density-analyzer"
    if os.environ.get("APPDATA")
    else Path.home() / ".mdfoam-density-analyzer"
)
PROFILE_PATH = APP_SETTINGS_DIR / "ssh_profile.json"
KEYRING_SERVICE = "mdfoam-density-analyzer"


def _load_profile_settings() -> dict[str, object]:
    if not PROFILE_PATH.is_file():
        return {}
    try:
        return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_profile_settings(settings: dict[str, object]) -> None:
    APP_SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    PROFILE_PATH.write_text(
        json.dumps(settings, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _secret_key(profile_name: str, host: str, username: str) -> str:
    return f"{profile_name or 'default'}:{username}@{host}"


def _read_saved_secret(profile_name: str, host: str, username: str) -> str:
    if not host or not username:
        return ""
    try:
        import keyring

        return keyring.get_password(KEYRING_SERVICE, _secret_key(profile_name, host, username)) or ""
    except Exception:
        return ""


def _save_secret(profile_name: str, host: str, username: str, secret: str) -> str | None:
    if not host or not username or not secret:
        return None
    try:
        import keyring

        keyring.set_password(KEYRING_SERVICE, _secret_key(profile_name, host, username), secret)
        return None
    except Exception as exc:
        return f"資格情報の保存に失敗しました: {exc}"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("mdFOAM 密度解析アプリ")
        self.resize(1360, 900)

        self.cases: list[Path] = []
        self.local_folder_path = Path.cwd()
        self.remote_cases: list[str] = []
        self.loaded_source = ""
        self.remote_browser_connection: SshConnection | None = None
        self.remote_browser_path = ""
        self._last_visual_downsample_message = ""
        self.results: list[CaseResult] = []
        self.worker: AnalyzerWorker | None = None
        self.thread: QThread | None = None

        self._build_ui()
        self._load_ssh_profile()
        self._set_source_mode("ローカル")
        self._connect_signals()
        self.folder_edit.setText(str(Path.cwd()))
        self.load_folder(Path.cwd())

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)

        self.workflow_tabs = QTabWidget()
        root_layout.addWidget(self.workflow_tabs, 1)

        input_tab = QWidget()
        input_layout = QVBoxLayout(input_tab)
        self.workflow_tabs.addTab(input_tab, "入力")

        source_group = QGroupBox("入力元")
        source_layout = QVBoxLayout(source_group)
        source_row = QHBoxLayout()
        self.source_combo = QComboBox()
        self.source_combo.addItems(["ローカル", "SSH"])
        self.folder_edit = QLabel()
        self.folder_edit.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.refresh_button = QPushButton("更新")
        source_row.addWidget(QLabel("入力元"))
        source_row.addWidget(self.source_combo)
        source_row.addWidget(QLabel("選択中"))
        source_row.addWidget(self.folder_edit, 1)
        source_row.addWidget(self.refresh_button)
        source_layout.addLayout(source_row)
        input_layout.addWidget(source_group)

        self.source_stack = QStackedWidget()
        input_layout.addWidget(self.source_stack, 2)

        local_panel = QWidget()
        local_layout = QVBoxLayout(local_panel)
        local_group = QGroupBox("ローカルフォルダ")
        local_group_layout = QHBoxLayout(local_group)
        self.browse_button = QPushButton("フォルダを選択")
        local_group_layout.addWidget(QLabel("解析対象ケースを含むフォルダを選択します。"))
        local_group_layout.addStretch(1)
        local_group_layout.addWidget(self.browse_button)
        local_layout.addWidget(local_group)
        local_layout.addStretch(1)
        self.source_stack.addWidget(local_panel)

        ssh_panel = QWidget()
        ssh_panel_layout = QVBoxLayout(ssh_panel)
        ssh_splitter = QSplitter(Qt.Horizontal)
        ssh_panel_layout.addWidget(ssh_splitter, 1)

        self.ssh_group = QGroupBox("SSH/SFTP接続")
        ssh_form = QFormLayout(self.ssh_group)
        ssh_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.profile_edit = QLineEdit()
        self.host_edit = QLineEdit()
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(22)
        self.username_edit = QLineEdit()
        self.key_path_edit = QLineEdit()
        self.key_browse_button = QPushButton("参照")
        self.secret_edit = QLineEdit()
        self.secret_edit.setEchoMode(QLineEdit.Password)
        self.remote_path_edit = QLineEdit()
        self.save_credentials_check = QCheckBox("資格情報を保存")

        key_row = QHBoxLayout()
        key_row.addWidget(self.key_path_edit, 1)
        key_row.addWidget(self.key_browse_button)
        ssh_form.addRow("プロファイル", self.profile_edit)
        ssh_form.addRow("ホスト", self.host_edit)
        ssh_form.addRow("ポート", self.port_spin)
        ssh_form.addRow("ユーザー", self.username_edit)
        ssh_form.addRow("OpenSSH秘密鍵", key_row)
        ssh_form.addRow("パスフレーズ/パスワード", self.secret_edit)
        ssh_form.addRow("リモートパス", self.remote_path_edit)
        ssh_form.addRow("", self.save_credentials_check)
        ssh_splitter.addWidget(self.ssh_group)

        browser_group = QGroupBox("リモートフォルダ")
        browser_layout = QVBoxLayout(browser_group)
        browser_button_row = QHBoxLayout()
        self.connect_remote_button = QPushButton("接続/更新")
        self.remote_up_button = QPushButton("上へ")
        self.remote_open_button = QPushButton("開く")
        self.remote_select_button = QPushButton("このフォルダを選択")
        self.clear_cache_button = QPushButton("キャッシュ削除")
        browser_button_row.addWidget(self.connect_remote_button)
        browser_button_row.addWidget(self.remote_up_button)
        browser_button_row.addWidget(self.remote_open_button)
        browser_button_row.addWidget(self.remote_select_button)
        browser_button_row.addStretch(1)
        browser_button_row.addWidget(self.clear_cache_button)
        browser_layout.addLayout(browser_button_row)
        self.remote_dir_list = QListWidget()
        self.remote_dir_list.setMinimumHeight(240)
        browser_layout.addWidget(self.remote_dir_list, 1)
        ssh_splitter.addWidget(browser_group)
        ssh_splitter.setStretchFactor(0, 1)
        ssh_splitter.setStretchFactor(1, 2)
        self.source_stack.addWidget(ssh_panel)

        case_group = QGroupBox("ケース一覧")
        case_layout = QVBoxLayout(case_group)

        self.case_list = QListWidget()
        self.case_list.setSelectionMode(QAbstractItemView.SingleSelection)
        case_layout.addWidget(self.case_list, 1)
        input_layout.addWidget(case_group, 3)

        settings_tab = QWidget()
        settings_tab_layout = QVBoxLayout(settings_tab)
        self.workflow_tabs.addTab(settings_tab, "解析設定")

        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_content = QWidget()
        settings_content_layout = QVBoxLayout(settings_content)
        settings_scroll.setWidget(settings_content)
        settings_tab_layout.addWidget(settings_scroll, 1)

        basic_group = QGroupBox("基本設定")
        basic_layout = QFormLayout(basic_group)
        basic_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.field_combo = QComboBox()
        self.field_combo.setMinimumWidth(260)
        self.threshold_spin = self._scientific_spin(500.0)
        self.zero_spin = self._scientific_spin(0.0)
        self.zero_count_spin = QSpinBox()
        self.zero_count_spin.setRange(1, 999)
        self.zero_count_spin.setValue(3)
        basic_layout.addRow("密度フィールド", self.field_combo)
        basic_layout.addRow("密度しきい値", self.threshold_spin)
        basic_layout.addRow("0判定許容値", self.zero_spin)
        basic_layout.addRow("連続ゼロ数", self.zero_count_spin)
        settings_content_layout.addWidget(basic_group)

        advanced_group = QGroupBox("詳細設定")
        advanced_layout = QFormLayout(advanced_group)
        advanced_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.cell_volume_spin = self._scientific_spin(0.0)
        self.dx_spin = self._scientific_spin(0.0)
        self.dy_spin = self._scientific_spin(0.0)
        self.dz_spin = self._scientific_spin(0.0)
        self.contact_fit_lower_spin = self._fraction_spin(0.5)
        self.contact_fit_upper_spin = self._fraction_spin(1.0)
        self.contact_unwrap_check = QCheckBox("有効")
        self.contact_unwrap_check.setChecked(True)
        advanced_layout.addRow("セル体積 fallback", self.cell_volume_spin)
        advanced_layout.addRow("dx fallback", self.dx_spin)
        advanced_layout.addRow("dy fallback", self.dy_spin)
        advanced_layout.addRow("dz fallback", self.dz_spin)
        advanced_layout.addRow("接触角fit下限", self.contact_fit_lower_spin)
        advanced_layout.addRow("接触角fit上限", self.contact_fit_upper_spin)
        advanced_layout.addRow("xy周期補正", self.contact_unwrap_check)
        settings_content_layout.addWidget(advanced_group)
        settings_content_layout.addStretch(1)

        run_group = QGroupBox("実行")
        run_layout = QVBoxLayout(run_group)
        button_row = QHBoxLayout()
        self.run_button = QPushButton("解析実行")
        self.stop_button = QPushButton("停止")
        self.stop_button.setEnabled(False)
        button_row.addWidget(self.run_button)
        button_row.addWidget(self.stop_button)
        button_row.addStretch(1)
        run_layout.addLayout(button_row)
        self.progress = QProgressBar()
        run_layout.addWidget(self.progress)
        settings_tab_layout.addWidget(run_group)

        results_tab = QWidget()
        results_layout = QVBoxLayout(results_tab)
        self.workflow_tabs.addTab(results_tab, "結果")

        export_row = QHBoxLayout()
        self.export_csv_button = QPushButton("CSV出力")
        self.export_png_button = QPushButton("PNG出力")
        export_row.addStretch(1)
        export_row.addWidget(self.export_csv_button)
        export_row.addWidget(self.export_png_button)
        results_layout.addLayout(export_row)

        self.table = ResultsTable(0, 11)
        self.table.setHorizontalHeaderLabels(
            [
                "ケース",
                "時刻数",
                "最大体積",
                "最終体積",
                "蒸発完了時刻",
                "初期接触角",
                "最終有効接触角",
                "初期接触半径",
                "最終有効接触半径",
                "状態",
                "エラー",
            ]
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

        self.tabs = QTabWidget()
        self.volume_plot = PlotWidget()
        self.radius_plot = PlotWidget()
        self.contact_angle_plot = PlotWidget()
        self.contact_radius_plot = PlotWidget()
        self.evap_plot = PlotWidget()
        self.visual_plot = VisualizationPlotWidget()
        self.tabs.addTab(self.volume_plot, "体積-時間")
        self.tabs.addTab(self.radius_plot, "等価半径-時間")
        self.tabs.addTab(self.contact_angle_plot, "接触角-時間")
        self.tabs.addTab(self.contact_radius_plot, "接触半径-時間")
        self.tabs.addTab(self.evap_plot, "蒸発完了時刻")
        visual_tab = QWidget()
        visual_layout = QVBoxLayout(visual_tab)
        visual_top_row = QHBoxLayout()
        self.visual_case_label = QLabel("ケース: -")
        self.visual_time_label = QLabel("時刻: -")
        self.visual_prev_button = QPushButton("前")
        self.visual_next_button = QPushButton("次")
        self.visual_time_slider = QSlider(Qt.Horizontal)
        self.visual_time_slider.setEnabled(False)
        visual_top_row.addWidget(self.visual_case_label)
        visual_top_row.addWidget(self.visual_prev_button)
        visual_top_row.addWidget(self.visual_time_slider, 1)
        visual_top_row.addWidget(self.visual_next_button)
        visual_top_row.addWidget(self.visual_time_label)
        visual_layout.addLayout(visual_top_row)

        visual_options_row = QHBoxLayout()
        self.visual_mode_combo = QComboBox()
        self.visual_mode_combo.addItems(["2D診断", "3D概観"])
        self.visual_projection_combo = QComboBox()
        self.visual_projection_combo.addItems(["xz", "yz", "xy"])
        self.visual_periodic_check = QCheckBox("粒子周期表示")
        self.visual_tile_spin = QSpinBox()
        self.visual_tile_spin.setRange(1, 16)
        self.visual_tile_spin.setValue(1)
        self.visual_point_size_spin = QDoubleSpinBox()
        self.visual_point_size_spin.setRange(0.1, 50.0)
        self.visual_point_size_spin.setDecimals(1)
        self.visual_point_size_spin.setValue(6.0)
        self.visual_max_points_spin = QSpinBox()
        self.visual_max_points_spin.setRange(0, 10_000_000)
        self.visual_max_points_spin.setValue(0)
        self.visual_max_points_spin.setToolTip("0は2Dでは全表示、3D周期表示では自動上限を使います。")
        self.visual_legend_check = QCheckBox("凡例")
        self.visual_legend_check.setChecked(True)
        self.visual_title_check = QCheckBox("タイトル")
        self.visual_title_check.setChecked(True)
        self.visual_axis_label_check = QCheckBox("軸ラベル")
        self.visual_axis_label_check.setChecked(True)
        self.visual_axis_tick_check = QCheckBox("軸目盛")
        self.visual_axis_tick_check.setChecked(True)
        self.visual_info_check = QCheckBox("情報")
        self.visual_info_check.setChecked(True)
        self.visual_grid_check = QCheckBox("グリッド")
        self.visual_grid_check.setChecked(True)
        self.visual_liquid_check = QCheckBox("液滴セル")
        self.visual_liquid_check.setChecked(True)
        self.visual_fit_check = QCheckBox("fit診断")
        self.visual_fit_check.setChecked(True)
        visual_options_row.addWidget(QLabel("表示"))
        visual_options_row.addWidget(self.visual_mode_combo)
        visual_options_row.addWidget(QLabel("投影"))
        visual_options_row.addWidget(self.visual_projection_combo)
        visual_options_row.addWidget(self.visual_periodic_check)
        visual_options_row.addWidget(QLabel("NxN"))
        visual_options_row.addWidget(self.visual_tile_spin)
        visual_options_row.addWidget(QLabel("点"))
        visual_options_row.addWidget(self.visual_point_size_spin)
        visual_options_row.addWidget(QLabel("最大表示点数"))
        visual_options_row.addWidget(self.visual_max_points_spin)
        visual_options_row.addWidget(self.visual_legend_check)
        visual_options_row.addWidget(self.visual_title_check)
        visual_options_row.addWidget(self.visual_axis_label_check)
        visual_options_row.addWidget(self.visual_axis_tick_check)
        visual_options_row.addWidget(self.visual_info_check)
        visual_options_row.addWidget(self.visual_grid_check)
        visual_options_row.addWidget(self.visual_liquid_check)
        visual_options_row.addWidget(self.visual_fit_check)
        visual_options_row.addStretch(1)
        visual_layout.addLayout(visual_options_row)

        visual_export_row = QHBoxLayout()
        self.visual_range_start_slider = QSlider(Qt.Horizontal)
        self.visual_range_end_slider = QSlider(Qt.Horizontal)
        self.visual_range_start_slider.setEnabled(False)
        self.visual_range_end_slider.setEnabled(False)
        self.visual_range_label = QLabel("GIF範囲: -")
        self.visual_fps_spin = QSpinBox()
        self.visual_fps_spin.setRange(1, 60)
        self.visual_fps_spin.setValue(8)
        self.visual_png_button = QPushButton("PNG保存")
        self.visual_gif_button = QPushButton("GIF保存")
        visual_export_row.addWidget(QLabel("開始"))
        visual_export_row.addWidget(self.visual_range_start_slider, 1)
        visual_export_row.addWidget(QLabel("終了"))
        visual_export_row.addWidget(self.visual_range_end_slider, 1)
        visual_export_row.addWidget(self.visual_range_label)
        visual_export_row.addWidget(QLabel("FPS"))
        visual_export_row.addWidget(self.visual_fps_spin)
        visual_export_row.addWidget(self.visual_png_button)
        visual_export_row.addWidget(self.visual_gif_button)
        visual_layout.addLayout(visual_export_row)
        visual_layout.addWidget(self.visual_plot, 1)
        self.visual_plot.clear()
        self.tabs.addTab(visual_tab, "可視化")
        results_splitter = QSplitter(Qt.Vertical)
        results_splitter.addWidget(self.table)
        results_splitter.addWidget(self.tabs)
        results_splitter.setStretchFactor(0, 1)
        results_splitter.setStretchFactor(1, 2)
        results_layout.addWidget(results_splitter, 1)
        self.volume_plot.clear("体積-時間")
        self.radius_plot.clear("等価半径-時間")
        self.contact_angle_plot.clear("接触角-時間")
        self.contact_radius_plot.clear("接触半径-時間")
        self.evap_plot.clear("蒸発完了時刻")

        log_group = QGroupBox("ログ")
        log_layout = QVBoxLayout(log_group)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(140)
        log_layout.addWidget(self.log_box)
        root_layout.addWidget(log_group)

    def _connect_signals(self) -> None:
        self.source_combo.currentTextChanged.connect(self._set_source_mode)
        self.browse_button.clicked.connect(self.choose_folder)
        self.refresh_button.clicked.connect(self.refresh_source)
        self.key_browse_button.clicked.connect(self.choose_private_key)
        self.connect_remote_button.clicked.connect(self.connect_remote_browser)
        self.remote_up_button.clicked.connect(self.remote_go_parent)
        self.remote_open_button.clicked.connect(self.remote_open_selected)
        self.remote_dir_list.itemDoubleClicked.connect(lambda _: self.remote_open_selected())
        self.remote_select_button.clicked.connect(self.remote_select_current)
        self.clear_cache_button.clicked.connect(self.clear_remote_cache)
        self.run_button.clicked.connect(self.start_analysis)
        self.stop_button.clicked.connect(self.stop_analysis)
        self.export_csv_button.clicked.connect(self.export_csv)
        self.export_png_button.clicked.connect(self.export_png)
        self.table.itemSelectionChanged.connect(self.update_selected_case_plots)
        self.visual_time_slider.valueChanged.connect(self.on_visual_time_changed)
        self.visual_prev_button.clicked.connect(lambda: self.set_visual_time_index(self.visual_time_slider.value() - 1))
        self.visual_next_button.clicked.connect(lambda: self.set_visual_time_index(self.visual_time_slider.value() + 1))
        self.visual_range_start_slider.valueChanged.connect(self.update_visual_range_label)
        self.visual_range_end_slider.valueChanged.connect(self.update_visual_range_label)
        self.visual_mode_combo.currentTextChanged.connect(lambda _: self.refresh_visualization())
        self.visual_projection_combo.currentTextChanged.connect(lambda _: self.refresh_visualization())
        self.visual_periodic_check.stateChanged.connect(lambda _: self.refresh_visualization())
        self.visual_tile_spin.valueChanged.connect(lambda _: self.refresh_visualization())
        self.visual_point_size_spin.valueChanged.connect(lambda _: self.refresh_visualization())
        self.visual_max_points_spin.valueChanged.connect(lambda _: self.refresh_visualization())
        self.visual_legend_check.stateChanged.connect(lambda _: self.refresh_visualization())
        self.visual_title_check.stateChanged.connect(lambda _: self.refresh_visualization())
        self.visual_axis_label_check.stateChanged.connect(lambda _: self.refresh_visualization())
        self.visual_axis_tick_check.stateChanged.connect(lambda _: self.refresh_visualization())
        self.visual_info_check.stateChanged.connect(lambda _: self.refresh_visualization())
        self.visual_grid_check.stateChanged.connect(lambda _: self.refresh_visualization())
        self.visual_liquid_check.stateChanged.connect(lambda _: self.refresh_visualization())
        self.visual_fit_check.stateChanged.connect(lambda _: self.refresh_visualization())
        self.visual_png_button.clicked.connect(self.export_visual_png)
        self.visual_gif_button.clicked.connect(self.export_visual_gif)

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

    def _fraction_spin(self, value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(3)
        spin.setRange(0.0, 1.0)
        spin.setSingleStep(0.05)
        spin.setValue(value)
        return spin

    @Slot(str)
    def _set_source_mode(self, mode: str) -> None:
        is_remote = mode == "SSH"
        self.source_stack.setCurrentIndex(1 if is_remote else 0)
        if is_remote:
            self.folder_edit.setText(self.remote_path_edit.text())
            if self.loaded_source != "SSH":
                self._clear_loaded_cases()
        else:
            self.folder_edit.setText(str(self.local_folder_path))
            if self.loaded_source not in ("", "ローカル"):
                self.load_folder(self.local_folder_path)

    def _clear_loaded_cases(self) -> None:
        self.case_list.clear()
        self.field_combo.clear()
        self.field_combo.addItem("rhoM_water")
        self.results.clear()
        self.table.setRowCount(0)
        self.cases = []
        self.remote_cases = []
        self.loaded_source = ""
        self.update_visual_controls(None)

    def _load_ssh_profile(self) -> None:
        profile = _load_profile_settings()
        self.profile_edit.setText(str(profile.get("name", "default")))
        self.host_edit.setText(str(profile.get("host", "")))
        self.port_spin.setValue(int(profile.get("port", 22) or 22))
        self.username_edit.setText(str(profile.get("username", "")))
        self.key_path_edit.setText(str(profile.get("key_path", "")))
        self.remote_path_edit.setText(str(profile.get("remote_path", "")))
        self.save_credentials_check.setChecked(bool(profile.get("save_credentials", False)))
        if self.save_credentials_check.isChecked():
            self.secret_edit.setText(
                _read_saved_secret(
                    self.profile_edit.text(),
                    self.host_edit.text(),
                    self.username_edit.text(),
                )
            )

    def _save_ssh_profile(self) -> None:
        profile = {
            "name": self.profile_edit.text().strip() or "default",
            "host": self.host_edit.text().strip(),
            "port": self.port_spin.value(),
            "username": self.username_edit.text().strip(),
            "key_path": self.key_path_edit.text().strip(),
            "remote_path": self.remote_path_edit.text().strip(),
            "save_credentials": self.save_credentials_check.isChecked(),
        }
        _save_profile_settings(profile)
        if self.save_credentials_check.isChecked():
            warning = _save_secret(
                str(profile["name"]),
                str(profile["host"]),
                str(profile["username"]),
                self.secret_edit.text(),
            )
            if warning:
                self.log(warning)

    def _remote_profile(self) -> RemoteProfile:
        profile = RemoteProfile(
            name=self.profile_edit.text().strip() or "default",
            host=self.host_edit.text().strip(),
            port=self.port_spin.value(),
            username=self.username_edit.text().strip(),
            key_path=self.key_path_edit.text().strip(),
            secret=self.secret_edit.text(),
            remote_path=normalize_remote_path(self.remote_path_edit.text()),
        )
        if not profile.host:
            raise RemoteError("SSHホストを入力してください。")
        if not profile.username:
            raise RemoteError("SSHユーザー名を入力してください。")
        if not profile.remote_path:
            raise RemoteError("リモートパスを入力してください。")
        validate_private_key_path(profile.key_path)
        return profile

    @Slot()
    def refresh_source(self) -> None:
        if self.source_combo.currentText() == "SSH":
            self.connect_remote_browser()
        else:
            self.load_folder(self.local_folder_path)

    @Slot()
    def choose_private_key(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "OpenSSH秘密鍵を選択", self.key_path_edit.text())
        if not path:
            return
        if path.lower().endswith(".ppk"):
            QMessageBox.warning(
                self,
                "鍵形式の変換が必要です",
                ".ppk は直接読み込めません。PuTTYgenでOpenSSH形式の秘密鍵に変換してから指定してください。",
            )
            return
        self.key_path_edit.setText(path)

    @Slot()
    def connect_remote_browser(self) -> None:
        try:
            profile = self._remote_profile()
            self._save_ssh_profile()
            if self.remote_browser_connection is not None:
                self.remote_browser_connection.close()
            self.remote_browser_connection = SshConnection(profile)
            self.remote_browser_connection.connect()
            self.remote_browser_path = profile.remote_path
            self._load_remote_dir(self.remote_browser_path)
            self.load_remote_folder(self.remote_browser_path)
            self.log(f"SSH接続しました: {profile.username}@{profile.host}:{self.remote_browser_path}")
        except Exception as exc:
            self.log(f"SSH接続/リモート読み込みに失敗しました: {exc}")
            QMessageBox.warning(self, "SSH接続エラー", str(exc))

    def _load_remote_dir(self, path: str) -> None:
        if self.remote_browser_connection is None or self.remote_browser_connection.sftp is None:
            raise RemoteError("SSH接続がありません。")
        path = normalize_remote_path(path)
        self.remote_dir_list.clear()
        for name, full_path in list_remote_dirs(self.remote_browser_connection.sftp, path):
            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, full_path)
            self.remote_dir_list.addItem(item)
        self.remote_browser_path = path
        self.remote_path_edit.setText(path)
        self.folder_edit.setText(path)

    @Slot()
    def remote_go_parent(self) -> None:
        try:
            parent = remote_join(self.remote_browser_path, "..")
            self._load_remote_dir(parent)
        except Exception as exc:
            self.log(f"リモートフォルダを開けません: {exc}")

    @Slot()
    def remote_open_selected(self) -> None:
        item = self.remote_dir_list.currentItem()
        if item is None:
            return
        try:
            self._load_remote_dir(str(item.data(Qt.UserRole)))
        except Exception as exc:
            self.log(f"リモートフォルダを開けません: {exc}")

    @Slot()
    def remote_select_current(self) -> None:
        try:
            self.load_remote_folder(self.remote_browser_path)
        except Exception as exc:
            self.log(f"リモートケース読み込みに失敗しました: {exc}")

    @Slot()
    def clear_remote_cache(self) -> None:
        try:
            clear_cache()
            self.log("SSH解析キャッシュを削除しました。")
        except Exception as exc:
            self.log(f"キャッシュ削除に失敗しました: {exc}")

    @Slot()
    def choose_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "親フォルダを選択", str(self.local_folder_path))
        if path:
            self.load_folder(Path(path))

    def load_folder(self, path: Path) -> None:
        self.local_folder_path = path
        self.folder_edit.setText(str(path))
        self.case_list.clear()
        self.field_combo.clear()
        self.results.clear()
        self.table.setRowCount(0)
        self.update_visual_controls(None)
        self.remote_cases = []
        self.loaded_source = "ローカル"
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

    def load_remote_folder(self, path: str) -> None:
        if self.remote_browser_connection is None or self.remote_browser_connection.sftp is None:
            raise RemoteError("SSH接続がありません。")
        path = normalize_remote_path(path)
        self.folder_edit.setText(path)
        self.remote_path_edit.setText(path)
        self.case_list.clear()
        self.field_combo.clear()
        self.results.clear()
        self.table.setRowCount(0)
        self.update_visual_controls(None)
        self.cases = []
        self.loaded_source = "SSH"
        self.remote_cases = discover_remote_cases(self.remote_browser_connection.sftp, path)
        fields = discover_remote_fields_for_cases(self.remote_browser_connection.sftp, self.remote_cases)
        if not fields:
            fields = ["rhoM_water"]
        self.field_combo.addItems(fields)
        default_index = self.field_combo.findText("rhoM_water")
        if default_index >= 0:
            self.field_combo.setCurrentIndex(default_index)
        for case in self.remote_cases:
            item = QListWidgetItem(remote_name(case))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            item.setData(Qt.UserRole, case)
            self.case_list.addItem(item)
        self.log(f"{path} から {len(self.remote_cases)} リモートケースを読み込みました。")
        self.log(f"密度フィールド: {', '.join(fields)}")

    def selected_cases(self) -> list[Path] | list[str]:
        if self.source_combo.currentText() == "SSH":
            remote_cases: list[str] = []
            for index in range(self.case_list.count()):
                item = self.case_list.item(index)
                if item.checkState() == Qt.Checked:
                    remote_cases.append(str(item.data(Qt.UserRole)))
            return remote_cases

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
            contact_fit_lower=self.contact_fit_lower_spin.value(),
            contact_fit_upper=self.contact_fit_upper_spin.value(),
            contact_unwrap_xy=self.contact_unwrap_check.isChecked(),
        )

    def _local_dialog_start_dir(self) -> str:
        if self.local_folder_path.exists():
            return str(self.local_folder_path)
        return str(Path.cwd())

    @Slot()
    def start_analysis(self) -> None:
        cases = self.selected_cases()
        if not cases:
            QMessageBox.warning(self, "ケースなし", "少なくとも1つのケースを選択してください。")
            return
        remote_profile = None
        if self.source_combo.currentText() == "SSH":
            try:
                remote_profile = self._remote_profile()
                self._save_ssh_profile()
            except Exception as exc:
                QMessageBox.warning(self, "SSH設定エラー", str(exc))
                return

        self.results.clear()
        self.table.setRowCount(0)
        self.update_visual_controls(None)
        self.progress.setRange(0, len(cases))
        self.progress.setValue(0)
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.log("解析を開始しました。")
        self.workflow_tabs.setCurrentIndex(2)

        self.thread = QThread(self)
        self.worker = AnalyzerWorker(cases, self.settings(), remote_profile)
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
            _fmt_optional(result.initial_contact_angle_deg),
            _fmt_optional(result.final_valid_contact_angle_deg),
            _fmt_optional(result.initial_contact_radius),
            _fmt_optional(result.final_valid_contact_radius),
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
        angle_points = [
            (row.time, row.contact_angle_deg)
            for row in result.rows
            if row.contact_angle_deg is not None
        ]
        radius_points = [
            (row.time, row.contact_radius)
            for row in result.rows
            if row.contact_radius is not None
        ]
        self.contact_angle_plot.plot_xy(
            f"{result.case_name}: 接触角-時間",
            "時間 [s]",
            "接触角 [deg]",
            [point[0] for point in angle_points],
            [point[1] for point in angle_points],
        )
        self.contact_radius_plot.plot_xy(
            f"{result.case_name}: 接触半径-時間",
            "時間 [s]",
            "接触半径 [m]",
            [point[0] for point in radius_points],
            [point[1] for point in radius_points],
        )
        self.update_visual_controls(result)

    def current_result(self) -> CaseResult | None:
        current_row = self.table.currentRow()
        if current_row < 0:
            return None
        item = self.table.item(current_row, 0)
        if item is None:
            return None
        result_index = item.data(Qt.UserRole)
        if result_index is None or result_index >= len(self.results):
            return None
        return self.results[result_index]

    def update_visual_controls(self, result: CaseResult | None = None) -> None:
        result = result or self.current_result()
        if result is None or not result.rows:
            self.visual_case_label.setText("ケース: -")
            self.visual_time_label.setText("時刻: -")
            self.visual_range_label.setText("GIF範囲: -")
            self.visual_time_slider.setEnabled(False)
            self.visual_range_start_slider.setEnabled(False)
            self.visual_range_end_slider.setEnabled(False)
            self.visual_plot.clear()
            return

        count = len(result.rows)
        self.visual_case_label.setText(f"ケース: {result.case_name}")
        for slider in (self.visual_time_slider, self.visual_range_start_slider, self.visual_range_end_slider):
            slider.blockSignals(True)
            slider.setRange(0, count - 1)
            slider.setEnabled(True)
            slider.blockSignals(False)
        self.visual_range_start_slider.blockSignals(True)
        self.visual_range_end_slider.blockSignals(True)
        self.visual_range_start_slider.setValue(0)
        self.visual_range_end_slider.setValue(count - 1)
        self.visual_range_start_slider.blockSignals(False)
        self.visual_range_end_slider.blockSignals(False)
        if self.visual_time_slider.value() >= count:
            self.visual_time_slider.setValue(count - 1)
        self.update_visual_range_label()
        self.refresh_visualization()

    def set_visual_time_index(self, index: int) -> None:
        if not self.visual_time_slider.isEnabled():
            return
        index = max(self.visual_time_slider.minimum(), min(self.visual_time_slider.maximum(), index))
        self.visual_time_slider.setValue(index)

    @Slot(int)
    def on_visual_time_changed(self, index: int) -> None:
        self.refresh_visualization()

    @Slot()
    def update_visual_range_label(self) -> None:
        result = self.current_result()
        if result is None or not result.rows:
            self.visual_range_label.setText("GIF範囲: -")
            return
        start, end = self._visual_range_indices()
        self.visual_range_label.setText(
            f"GIF範囲: {result.rows[start].time:.4g} - {result.rows[end].time:.4g}"
        )

    def refresh_visualization(self) -> None:
        result = self.current_result()
        if result is None or not result.rows or not self.visual_time_slider.isEnabled():
            return
        self._apply_visual_defaults()
        index = max(0, min(self.visual_time_slider.value(), len(result.rows) - 1))
        row = result.rows[index]
        self.visual_time_label.setText(f"時刻: {row.time:.8g}")
        try:
            frame = self._load_visual_frame(result, row.time)
            self._draw_visual_frame(frame)
        except Exception as exc:
            self.visual_plot.clear(f"可視化データを読み込めません: {exc}")
            self.log(f"可視化データを読み込めません: {exc}")

    def _apply_visual_defaults(self) -> None:
        if self.visual_mode_combo.currentText().startswith("3D") and self.visual_periodic_check.isChecked():
            if self.visual_max_points_spin.value() == 0:
                self.visual_max_points_spin.blockSignals(True)
                self.visual_max_points_spin.setSpecialValueText(f"自動({THREE_D_AUTO_MAX_POINTS})")
                self.visual_max_points_spin.blockSignals(False)

    def _draw_visual_frame(self, frame: VisualizationFrame) -> None:
        downsample = self.visual_plot.draw_frame(
            frame,
            self.visual_mode_combo.currentText(),
            self.visual_projection_combo.currentText(),
            self.visual_periodic_check.isChecked(),
            self.visual_tile_spin.value(),
            self.visual_point_size_spin.value(),
            self.visual_max_points_spin.value(),
            self.visual_legend_check.isChecked(),
            self.visual_title_check.isChecked(),
            self.visual_axis_label_check.isChecked(),
            self.visual_axis_tick_check.isChecked(),
            self.visual_info_check.isChecked(),
            self.visual_grid_check.isChecked(),
            self.visual_liquid_check.isChecked(),
            self.visual_fit_check.isChecked(),
        )
        if downsample.was_downsampled:
            message = f"表示用間引き: {downsample.original_count} -> {downsample.displayed_count}"
            if message != self._last_visual_downsample_message:
                self.log(message)
                self._last_visual_downsample_message = message

    def _load_visual_frame(self, result: CaseResult, time_value: float) -> VisualizationFrame:
        time_dir = self._ensure_visualization_files(result, time_value)
        return load_visualization_frame(result.case_dir, time_value, self.settings())

    def _ensure_visualization_files(self, result: CaseResult, time_value: float) -> Path | None:
        time_dirs = case_time_dirs(result.case_dir)
        time_dir = next((path for value, path in time_dirs if value == time_value), None)
        if time_dir is None:
            return None
        positions_path = time_dir / "lagrangian" / "moleculeCloud" / "positions"
        id_path = time_dir / "lagrangian" / "moleculeCloud" / "id"
        if positions_path.is_file() and id_path.is_file():
            return time_dir

        remote_case = read_remote_case_from_manifest(result.case_dir)
        if remote_case is None:
            return time_dir

        profile = self._remote_profile()
        if self.remote_browser_connection is not None and self.remote_browser_connection.sftp is not None:
            sync_remote_lagrangian_time(
                self.remote_browser_connection.sftp,
                profile,
                remote_case,
                result.case_dir,
                time_dir.name,
                log=self.log,
            )
        else:
            with SshConnection(profile) as connection:
                sync_remote_lagrangian_time(
                    connection.sftp,
                    profile,
                    remote_case,
                    result.case_dir,
                    time_dir.name,
                    log=self.log,
                )
        return time_dir

    def _visual_range_indices(self) -> tuple[int, int]:
        start = self.visual_range_start_slider.value()
        end = self.visual_range_end_slider.value()
        if start > end:
            start, end = end, start
        return start, end

    @Slot()
    def export_visual_png(self) -> None:
        result = self.current_result()
        if result is None:
            QMessageBox.information(self, "ケースなし", "可視化する結果ケースを選択してください。")
            return
        path, _ = QFileDialog.getSaveFileName(self, "可視化PNGを保存", self._local_dialog_start_dir(), "PNG (*.png)")
        if not path:
            return
        self.visual_plot.save_png(Path(path))
        self.log(f"可視化PNGを保存しました: {path}")

    @Slot()
    def export_visual_gif(self) -> None:
        result = self.current_result()
        if result is None or not result.rows:
            QMessageBox.information(self, "ケースなし", "可視化する結果ケースを選択してください。")
            return
        path, _ = QFileDialog.getSaveFileName(self, "可視化GIFを保存", self._local_dialog_start_dir(), "GIF (*.gif)")
        if not path:
            return
        start, end = self._visual_range_indices()
        rows = result.rows[start : end + 1]
        writer = PillowWriter(fps=self.visual_fps_spin.value())
        try:
            with writer.saving(self.visual_plot.figure, path, dpi=120):
                for row in rows:
                    frame = self._load_visual_frame(result, row.time)
                    self._draw_visual_frame(frame)
                    self.visual_plot.canvas.draw()
                    writer.grab_frame()
                    QApplication.processEvents()
            self.log(f"可視化GIFを保存しました: {path}")
        except Exception as exc:
            QMessageBox.warning(self, "GIF保存エラー", str(exc))
            self.log(f"可視化GIF保存に失敗しました: {exc}")

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
        directory = QFileDialog.getExistingDirectory(self, "出力フォルダを選択", self._local_dialog_start_dir())
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
        path, _ = QFileDialog.getSaveFileName(self, "現在のグラフを保存", self._local_dialog_start_dir(), "PNG (*.png)")
        if not path:
            return
        tab.save_png(Path(path))
        self.log(f"PNGを出力しました: {path}")

    @Slot(str)
    def log(self, message: str) -> None:
        self.log_box.append(message)

    def closeEvent(self, event) -> None:
        if self.remote_browser_connection is not None:
            self.remote_browser_connection.close()
        super().closeEvent(event)


def _fmt(value: float) -> str:
    return f"{value:.8g}"


def _fmt_optional(value: float | None) -> str:
    return "" if value is None else _fmt(value)


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
