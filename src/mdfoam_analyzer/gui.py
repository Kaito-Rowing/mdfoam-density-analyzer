from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sys
import traceback

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QFormLayout,
    QGridLayout,
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
    QSizePolicy,
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
from PySide6.QtGui import QColor
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
    TimeResult,
    analyze_case,
    detect_analysis_layout,
    discover_cases,
    discover_fields_for_cases,
    write_summary_csv,
    write_timeseries_csv,
)
from .analysis_cache import AnalysisCacheSession
from .cache import clear_cache, clear_local_analysis_cache
from .molecular_departure import (
    write_departure_events_csv,
    write_departure_height_bins_csv,
)
from .provenance import (
    ProvenanceError,
    RunContext,
    apply_remote_input_paths,
    load_analysis_settings,
    save_analysis_settings,
    write_analysis_manifest,
)
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
from .theory import (
    DEFAULT_ALPHA_VALUES,
    THEORY_PRESETS,
    TheoryComparison,
    TheorySettings,
    build_theory_comparison,
    evaporation_flux,
    height_to_contact_radius_ratio,
    spherical_cap_geometry,
    write_theory_summary_csv,
    write_theory_timeseries_csv,
)
from .theme import APP_STYLESHEET, COLORS
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

LANGUAGES: dict[str, str] = {
    "ja": "日本語",
    "en": "English",
    "zh": "中文",
    "es": "Español",
    "hi": "हिन्दी",
}

TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": {
        "mdFOAM 密度解析アプリ": "mdFOAM Density Analyzer",
        "言語": "Language",
        "入力": "Input",
        "入力元": "Input source",
        "ローカル": "Local",
        "選択中": "Selected",
        "更新": "Refresh",
        "ローカルフォルダ": "Local folder",
        "フォルダを選択": "Select folder",
        "解析対象ケースを含むフォルダを選択します。": "Select a folder that contains cases to analyze.",
        "SSH/SFTP接続": "SSH/SFTP connection",
        "参照": "Browse",
        "資格情報を保存": "Save credentials",
        "プロファイル": "Profile",
        "ホスト": "Host",
        "ポート": "Port",
        "ユーザー": "User",
        "OpenSSH秘密鍵": "OpenSSH private key",
        "パスフレーズ/パスワード": "Passphrase/password",
        "リモートパス": "Remote path",
        "リモートフォルダ": "Remote folder",
        "接続/更新": "Connect/refresh",
        "上へ": "Up",
        "開く": "Open",
        "このフォルダを選択": "Select this folder",
        "キャッシュ削除": "Clear cache",
        "ローカル解析キャッシュ削除": "Clear local analysis cache",
        "ケース一覧": "Cases",
        "解析設定": "Analysis settings",
        "基本設定": "Basic settings",
        "密度フィールド": "Density field",
        "密度しきい値": "Density threshold",
        "0判定許容値": "Zero tolerance",
        "連続ゼロ数": "Consecutive zeros",
        "詳細設定": "Advanced settings",
        "セル体積 fallback": "Cell volume fallback",
        "接触角fit下限": "Contact angle fit lower",
        "接触角fit上限": "Contact angle fit upper",
        "平均接触角の対象範囲": "Average contact angle range",
        "xy周期補正": "xy periodic correction",
        "有効": "Enabled",
        "蒸発係数 / 理論比較": "Evaporation coefficient / theory",
        "物性値プリセット": "Material preset",
        "飽和蒸気密度 rho_v [kg/m^3]": "Saturated vapor density rho_v [kg/m^3]",
        "液体密度 rho_l [kg/m^3]": "Liquid density rho_l [kg/m^3]",
        "温度 T [K]": "Temperature T [K]",
        "分子1個の質量 m [kg]": "Mass per molecule m [kg]",
        "理論初期体積 V0": "Theory initial volume V0",
        "接触角ソース": "Contact angle source",
        "固定theta [deg]": "Fixed theta [deg]",
        "fit対象範囲": "Fit range",
        "非ゼロ体積のみfit": "Fit non-zero volumes only",
        "fit alpha_e 下限": "fit alpha_e lower",
        "fit alpha_e 上限": "fit alpha_e upper",
        "計算確認": "Calculation check",
        "最大体積": "Maximum volume",
        "先頭時刻体積": "First time volume",
        "平均接触角": "Average contact angle",
        "固定theta": "Fixed theta",
        "実行": "Run",
        "解析実行": "Run analysis",
        "停止": "Stop",
        "結果": "Results",
        "CSV出力": "Export CSV",
        "PNG出力": "Export PNG",
        "全ケースPNG出力": "Export all case PNGs",
        "グラフ表示設定": "Graph display settings",
        "色": "Color",
        "点色": "Point color",
        "点サイズ": "Point size",
        "透明度": "Opacity",
        "文字": "Text",
        "マーカー": "Marker",
        "縦横": "Aspect",
        "自動": "Auto",
        "等倍": "Equal",
        "タイトル": "Title",
        "軸ラベル": "Axis labels",
        "目盛": "Ticks",
        "グリッド": "Grid",
        "軸対象": "Axis target",
        "軸モード": "Axis mode",
        "現在グラフ": "Current graph",
        "自動固定": "Auto fixed",
        "手動固定": "Manual fixed",
        "x最小": "x min",
        "x最大": "x max",
        "y最小": "y min",
        "y最大": "y max",
        "x対数": "x log",
        "y対数": "y log",
        "PNG幅[in]": "PNG width [in]",
        "PNG高さ[in]": "PNG height [in]",
        "画質": "Quality",
        "透明背景": "Transparent background",
        "低 150dpi": "Low 150dpi",
        "標準 300dpi": "Standard 300dpi",
        "高 600dpi": "High 600dpi",
        "ケース": "Case",
        "時刻": "Time",
        "GIF範囲": "GIF range",
        "可視化するケースと時刻を選択してください": "Select a case and time to visualize",
        "時刻数": "Time count",
        "最大体積": "Max volume",
        "最終体積": "Final volume",
        "蒸発完了時刻": "Evaporation time",
        "初期接触角": "Initial contact angle",
        "最終有効接触角": "Final valid contact angle",
        "初期接触半径": "Initial contact radius",
        "最終有効接触半径": "Final valid contact radius",
        "推定alpha_e": "Estimated alpha_e",
        "fit状態": "fit status",
        "状態": "Status",
        "エラー": "Error",
        "エラー / 警告": "Error / warning",
        "体積-時間": "Volume-Time",
        "等価半径-時間": "Equivalent radius-Time",
        "接触角-時間": "Contact angle-Time",
        "接触半径-時間": "Contact radius-Time",
        "蒸発量 EM": "Evaporated mass EM",
        "理論/MD 等価半径": "Theory/MD equivalent radius",
        "上下2枚": "Two stacked graphs",
        "保存する理論グラフ": "Theory graph to save",
        "全ケースPNG出力するグラフを選択してください。": "Select the graph to export as PNG for all cases.",
        "保存先フォルダ": "Destination folder",
        "親フォルダを選択": "Select parent folder",
        "現在のグラフを保存": "Save current graph",
        "蒸発量 EM-時間": "Evaporated mass EM-Time",
        "理論/MD 等価半径-時間": "Theory/MD equivalent radius-Time",
        "fit推定": "fit estimate",
        "可視化": "Visualization",
        "ケース: -": "Case: -",
        "時刻: -": "Time: -",
        "前": "Prev",
        "次": "Next",
        "表示": "View",
        "2D診断": "2D diagnostics",
        "3D概観": "3D overview",
        "投影": "Projection",
        "粒子周期表示": "Periodic particles",
        "点": "Points",
        "最大表示点数": "Max displayed points",
        "凡例": "Legend",
        "軸目盛": "Axis ticks",
        "情報": "Info",
        "液滴セル": "Droplet cells",
        "fit診断": "fit diagnostics",
        "GIF範囲: -": "GIF range: -",
        "開始": "Start",
        "終了": "End",
        "PNG保存": "Save PNG",
        "GIF保存": "Save GIF",
        "ログ": "Log",
        "PNG出力プレビュー": "PNG export preview",
        "保存するグラフ": "Graph to save",
        "軸": "Axis",
        "幅 [in]": "Width [in]",
        "高さ [in]": "Height [in]",
        "PNG": "PNG",
        "保存": "Save",
        "キャンセル": "Cancel",
        "グラフ": "Graph",
    },
    "zh": {
        "mdFOAM 密度解析アプリ": "mdFOAM 密度分析器",
        "言語": "语言",
        "入力": "输入",
        "入力元": "输入源",
        "ローカル": "本地",
        "選択中": "当前选择",
        "更新": "刷新",
        "ローカルフォルダ": "本地文件夹",
        "フォルダを選択": "选择文件夹",
        "解析対象ケースを含むフォルダを選択します。": "选择包含待分析案例的文件夹。",
        "SSH/SFTP接続": "SSH/SFTP 连接",
        "参照": "浏览",
        "資格情報を保存": "保存凭据",
        "プロファイル": "配置文件",
        "ホスト": "主机",
        "ポート": "端口",
        "ユーザー": "用户",
        "OpenSSH秘密鍵": "OpenSSH 私钥",
        "パスフレーズ/パスワード": "密码短语/密码",
        "リモートパス": "远程路径",
        "リモートフォルダ": "远程文件夹",
        "接続/更新": "连接/刷新",
        "上へ": "向上",
        "開く": "打开",
        "このフォルダを選択": "选择此文件夹",
        "キャッシュ削除": "清除缓存",
        "ローカル解析キャッシュ削除": "清除本地分析缓存",
        "ケース一覧": "案例列表",
        "解析設定": "分析设置",
        "基本設定": "基本设置",
        "密度フィールド": "密度场",
        "密度しきい値": "密度阈值",
        "0判定許容値": "零判定容差",
        "連続ゼロ数": "连续零数量",
        "詳細設定": "高级设置",
        "セル体積 fallback": "单元体积备用值",
        "接触角fit下限": "接触角拟合下限",
        "接触角fit上限": "接触角拟合上限",
        "平均接触角の対象範囲": "平均接触角范围",
        "xy周期補正": "xy 周期修正",
        "有効": "启用",
        "蒸発係数 / 理論比較": "蒸发系数 / 理论比较",
        "物性値プリセット": "物性预设",
        "飽和蒸気密度 rho_v [kg/m^3]": "饱和蒸气密度 rho_v [kg/m^3]",
        "液体密度 rho_l [kg/m^3]": "液体密度 rho_l [kg/m^3]",
        "温度 T [K]": "温度 T [K]",
        "分子1個の質量 m [kg]": "单个分子质量 m [kg]",
        "理論初期体積 V0": "理论初始体积 V0",
        "接触角ソース": "接触角来源",
        "固定theta [deg]": "固定 theta [deg]",
        "fit対象範囲": "拟合范围",
        "非ゼロ体積のみfit": "仅拟合非零体积",
        "fit alpha_e 下限": "fit alpha_e 下限",
        "fit alpha_e 上限": "fit alpha_e 上限",
        "計算確認": "计算检查",
        "最大体積": "最大体积",
        "先頭時刻体積": "首个时刻体积",
        "平均接触角": "平均接触角",
        "固定theta": "固定 theta",
        "実行": "执行",
        "解析実行": "运行分析",
        "停止": "停止",
        "結果": "结果",
        "CSV出力": "导出 CSV",
        "PNG出力": "导出 PNG",
        "全ケースPNG出力": "导出所有案例 PNG",
        "グラフ表示設定": "图表显示设置",
        "色": "颜色",
        "点色": "点颜色",
        "点サイズ": "点大小",
        "透明度": "透明度",
        "文字": "文字",
        "マーカー": "标记",
        "縦横": "纵横比",
        "自動": "自动",
        "等倍": "等比例",
        "タイトル": "标题",
        "軸ラベル": "轴标签",
        "目盛": "刻度",
        "グリッド": "网格",
        "軸対象": "轴目标",
        "軸モード": "轴模式",
        "現在グラフ": "当前图表",
        "自動固定": "自动固定",
        "手動固定": "手动固定",
        "x最小": "x 最小",
        "x最大": "x 最大",
        "y最小": "y 最小",
        "y最大": "y 最大",
        "x対数": "x 对数",
        "y対数": "y 对数",
        "PNG幅[in]": "PNG 宽度 [in]",
        "PNG高さ[in]": "PNG 高度 [in]",
        "画質": "质量",
        "透明背景": "透明背景",
        "低 150dpi": "低 150dpi",
        "標準 300dpi": "标准 300dpi",
        "高 600dpi": "高 600dpi",
        "ケース": "案例",
        "時刻": "时刻",
        "GIF範囲": "GIF 范围",
        "可視化するケースと時刻を選択してください": "请选择要可视化的案例和时刻",
        "時刻数": "时刻数",
        "最大体積": "最大体积",
        "最終体積": "最终体积",
        "蒸発完了時刻": "蒸发完成时刻",
        "初期接触角": "初始接触角",
        "最終有効接触角": "最终有效接触角",
        "初期接触半径": "初始接触半径",
        "最終有効接触半径": "最终有效接触半径",
        "推定alpha_e": "估计 alpha_e",
        "fit状態": "fit 状态",
        "状態": "状态",
        "エラー": "错误",
        "エラー / 警告": "错误 / 警告",
        "体積-時間": "体积-时间",
        "等価半径-時間": "等效半径-时间",
        "接触角-時間": "接触角-时间",
        "接触半径-時間": "接触半径-时间",
        "蒸発量 EM": "蒸发量 EM",
        "理論/MD 等価半径": "理论/MD 等效半径",
        "上下2枚": "上下两图",
        "保存する理論グラフ": "要保存的理论图表",
        "全ケースPNG出力するグラフを選択してください。": "请选择要为所有案例导出 PNG 的图表。",
        "保存先フォルダ": "保存目标文件夹",
        "親フォルダを選択": "选择父文件夹",
        "現在のグラフを保存": "保存当前图表",
        "蒸発量 EM-時間": "蒸发量 EM-时间",
        "理論/MD 等価半径-時間": "理论/MD 等效半径-时间",
        "fit推定": "fit 估计",
        "可視化": "可视化",
        "ケース: -": "案例: -",
        "時刻: -": "时刻: -",
        "前": "前",
        "次": "后",
        "表示": "显示",
        "2D診断": "2D 诊断",
        "3D概観": "3D 概览",
        "投影": "投影",
        "粒子周期表示": "粒子周期显示",
        "点": "点",
        "最大表示点数": "最大显示点数",
        "凡例": "图例",
        "軸目盛": "轴刻度",
        "情報": "信息",
        "液滴セル": "液滴单元",
        "fit診断": "fit 诊断",
        "GIF範囲: -": "GIF 范围: -",
        "開始": "开始",
        "終了": "结束",
        "PNG保存": "保存 PNG",
        "GIF保存": "保存 GIF",
        "ログ": "日志",
        "PNG出力プレビュー": "PNG 导出预览",
        "保存するグラフ": "要保存的图表",
        "軸": "轴",
        "幅 [in]": "宽度 [in]",
        "高さ [in]": "高度 [in]",
        "PNG": "PNG",
        "保存": "保存",
        "キャンセル": "取消",
        "グラフ": "图表",
    },
    "es": {
        "mdFOAM 密度解析アプリ": "Analizador de densidad mdFOAM",
        "言語": "Idioma",
        "入力": "Entrada",
        "入力元": "Fuente de entrada",
        "ローカル": "Local",
        "選択中": "Seleccionado",
        "更新": "Actualizar",
        "ローカルフォルダ": "Carpeta local",
        "フォルダを選択": "Seleccionar carpeta",
        "解析対象ケースを含むフォルダを選択します。": "Seleccione una carpeta que contenga los casos a analizar.",
        "SSH/SFTP接続": "Conexión SSH/SFTP",
        "参照": "Examinar",
        "資格情報を保存": "Guardar credenciales",
        "プロファイル": "Perfil",
        "ホスト": "Host",
        "ポート": "Puerto",
        "ユーザー": "Usuario",
        "OpenSSH秘密鍵": "Clave privada OpenSSH",
        "パスフレーズ/パスワード": "Frase/contraseña",
        "リモートパス": "Ruta remota",
        "リモートフォルダ": "Carpeta remota",
        "接続/更新": "Conectar/actualizar",
        "上へ": "Arriba",
        "開く": "Abrir",
        "このフォルダを選択": "Seleccionar esta carpeta",
        "キャッシュ削除": "Borrar caché",
        "ローカル解析キャッシュ削除": "Borrar caché de análisis local",
        "ケース一覧": "Casos",
        "解析設定": "Ajustes de análisis",
        "基本設定": "Ajustes básicos",
        "密度フィールド": "Campo de densidad",
        "密度しきい値": "Umbral de densidad",
        "0判定許容値": "Tolerancia de cero",
        "連続ゼロ数": "Ceros consecutivos",
        "詳細設定": "Ajustes avanzados",
        "セル体積 fallback": "Volumen de celda alternativo",
        "接触角fit下限": "Límite inferior fit ángulo",
        "接触角fit上限": "Límite superior fit ángulo",
        "平均接触角の対象範囲": "Rango de ángulo medio",
        "xy周期補正": "Corrección periódica xy",
        "有効": "Activado",
        "蒸発係数 / 理論比較": "Coef. evaporación / teoría",
        "物性値プリセット": "Preajuste de material",
        "飽和蒸気密度 rho_v [kg/m^3]": "Densidad vapor saturado rho_v [kg/m^3]",
        "液体密度 rho_l [kg/m^3]": "Densidad líquida rho_l [kg/m^3]",
        "温度 T [K]": "Temperatura T [K]",
        "分子1個の質量 m [kg]": "Masa por molécula m [kg]",
        "理論初期体積 V0": "Volumen inicial teórico V0",
        "接触角ソース": "Fuente de ángulo",
        "固定theta [deg]": "theta fijo [deg]",
        "fit対象範囲": "Rango de fit",
        "非ゼロ体積のみfit": "Fit solo volumen no cero",
        "fit alpha_e 下限": "fit alpha_e inferior",
        "fit alpha_e 上限": "fit alpha_e superior",
        "計算確認": "Comprobación",
        "最大体積": "Volumen máximo",
        "先頭時刻体積": "Volumen inicial",
        "平均接触角": "Ángulo de contacto medio",
        "固定theta": "theta fijo",
        "実行": "Ejecución",
        "解析実行": "Ejecutar análisis",
        "停止": "Detener",
        "結果": "Resultados",
        "CSV出力": "Exportar CSV",
        "PNG出力": "Exportar PNG",
        "全ケースPNG出力": "Exportar PNG de todos",
        "グラフ表示設定": "Ajustes de gráfico",
        "色": "Color",
        "点色": "Color de puntos",
        "点サイズ": "Tamaño de punto",
        "透明度": "Opacidad",
        "文字": "Texto",
        "マーカー": "Marcador",
        "縦横": "Aspecto",
        "自動": "Auto",
        "等倍": "Igual",
        "タイトル": "Título",
        "軸ラベル": "Etiquetas de eje",
        "目盛": "Marcas",
        "グリッド": "Cuadrícula",
        "軸対象": "Objetivo de eje",
        "軸モード": "Modo de eje",
        "現在グラフ": "Gráfico actual",
        "自動固定": "Fijo automático",
        "手動固定": "Fijo manual",
        "x最小": "x mín",
        "x最大": "x máx",
        "y最小": "y mín",
        "y最大": "y máx",
        "x対数": "x log",
        "y対数": "y log",
        "PNG幅[in]": "Ancho PNG [in]",
        "PNG高さ[in]": "Alto PNG [in]",
        "画質": "Calidad",
        "透明背景": "Fondo transparente",
        "低 150dpi": "Baja 150dpi",
        "標準 300dpi": "Estándar 300dpi",
        "高 600dpi": "Alta 600dpi",
        "ケース": "Caso",
        "時刻": "Tiempo",
        "GIF範囲": "Rango GIF",
        "可視化するケースと時刻を選択してください": "Seleccione un caso y un tiempo para visualizar",
        "時刻数": "Núm. tiempos",
        "最大体積": "Volumen máx.",
        "最終体積": "Volumen final",
        "蒸発完了時刻": "Tiempo evaporación",
        "初期接触角": "Ángulo inicial",
        "最終有効接触角": "Ángulo válido final",
        "初期接触半径": "Radio contacto inicial",
        "最終有効接触半径": "Radio contacto final",
        "推定alpha_e": "alpha_e estimado",
        "fit状態": "estado fit",
        "状態": "Estado",
        "エラー": "Error",
        "エラー / 警告": "Error / advertencia",
        "体積-時間": "Volumen-Tiempo",
        "等価半径-時間": "Radio equivalente-Tiempo",
        "接触角-時間": "Ángulo-Tiempo",
        "接触半径-時間": "Radio contacto-Tiempo",
        "蒸発量 EM": "Masa evaporada EM",
        "理論/MD 等価半径": "Radio equivalente teoría/MD",
        "上下2枚": "Dos gráficos apilados",
        "保存する理論グラフ": "Gráfico teórico a guardar",
        "全ケースPNG出力するグラフを選択してください。": "Seleccione el gráfico que se exportará como PNG para todos los casos.",
        "保存先フォルダ": "Carpeta de destino",
        "親フォルダを選択": "Seleccionar carpeta principal",
        "現在のグラフを保存": "Guardar gráfico actual",
        "蒸発量 EM-時間": "Masa evaporada EM-Tiempo",
        "理論/MD 等価半径-時間": "Radio equivalente teoría/MD-Tiempo",
        "fit推定": "estimación fit",
        "可視化": "Visualización",
        "ケース: -": "Caso: -",
        "時刻: -": "Tiempo: -",
        "前": "Anterior",
        "次": "Siguiente",
        "表示": "Vista",
        "2D診断": "Diagnóstico 2D",
        "3D概観": "Vista 3D",
        "投影": "Proyección",
        "粒子周期表示": "Partículas periódicas",
        "点": "Puntos",
        "最大表示点数": "Máx. puntos",
        "凡例": "Leyenda",
        "軸目盛": "Marcas de eje",
        "情報": "Información",
        "液滴セル": "Celdas de gota",
        "fit診断": "diagnóstico fit",
        "GIF範囲: -": "Rango GIF: -",
        "開始": "Inicio",
        "終了": "Fin",
        "PNG保存": "Guardar PNG",
        "GIF保存": "Guardar GIF",
        "ログ": "Registro",
        "PNG出力プレビュー": "Vista previa PNG",
        "保存するグラフ": "Gráfico a guardar",
        "軸": "Eje",
        "幅 [in]": "Ancho [in]",
        "高さ [in]": "Alto [in]",
        "PNG": "PNG",
        "保存": "Guardar",
        "キャンセル": "Cancelar",
        "グラフ": "Gráfico",
    },
    "hi": {
        "mdFOAM 密度解析アプリ": "mdFOAM घनत्व विश्लेषक",
        "言語": "भाषा",
        "入力": "इनपुट",
        "入力元": "इनपुट स्रोत",
        "ローカル": "स्थानीय",
        "選択中": "चयनित",
        "更新": "रीफ्रेश",
        "ローカルフォルダ": "स्थानीय फ़ोल्डर",
        "フォルダを選択": "फ़ोल्डर चुनें",
        "解析対象ケースを含むフォルダを選択します。": "विश्लेषण केस वाले फ़ोल्डर को चुनें।",
        "SSH/SFTP接続": "SSH/SFTP कनेक्शन",
        "参照": "ब्राउज़",
        "資格情報を保存": "क्रेडेंशियल सहेजें",
        "プロファイル": "प्रोफ़ाइल",
        "ホスト": "होस्ट",
        "ポート": "पोर्ट",
        "ユーザー": "उपयोगकर्ता",
        "OpenSSH秘密鍵": "OpenSSH निजी कुंजी",
        "パスフレーズ/パスワード": "पासफ्रेज़/पासवर्ड",
        "リモートパス": "रिमोट पथ",
        "リモートフォルダ": "रिमोट फ़ोल्डर",
        "接続/更新": "कनेक्ट/रीफ्रेश",
        "上へ": "ऊपर",
        "開く": "खोलें",
        "このフォルダを選択": "यह फ़ोल्डर चुनें",
        "キャッシュ削除": "कैश हटाएं",
        "ローカル解析キャッシュ削除": "स्थानीय विश्लेषण कैश हटाएं",
        "ケース一覧": "केस",
        "解析設定": "विश्लेषण सेटिंग",
        "基本設定": "मूल सेटिंग",
        "密度フィールド": "घनत्व फ़ील्ड",
        "密度しきい値": "घनत्व सीमा",
        "0判定許容値": "शून्य सहनशीलता",
        "連続ゼロ数": "लगातार शून्य",
        "詳細設定": "उन्नत सेटिंग",
        "セル体積 fallback": "सेल आयतन fallback",
        "接触角fit下限": "संपर्क कोण fit निचली सीमा",
        "接触角fit上限": "संपर्क कोण fit ऊपरी सीमा",
        "平均接触角の対象範囲": "औसत संपर्क कोण रेंज",
        "xy周期補正": "xy आवधिक सुधार",
        "有効": "सक्षम",
        "蒸発係数 / 理論比較": "वाष्पीकरण गुणांक / सिद्धांत",
        "物性値プリセット": "गुण preset",
        "飽和蒸気密度 rho_v [kg/m^3]": "संतृप्त वाष्प घनत्व rho_v [kg/m^3]",
        "液体密度 rho_l [kg/m^3]": "द्रव घनत्व rho_l [kg/m^3]",
        "温度 T [K]": "तापमान T [K]",
        "分子1個の質量 m [kg]": "प्रति अणु द्रव्यमान m [kg]",
        "理論初期体積 V0": "सैद्धांतिक आरंभिक आयतन V0",
        "接触角ソース": "संपर्क कोण स्रोत",
        "固定theta [deg]": "स्थिर theta [deg]",
        "fit対象範囲": "fit रेंज",
        "非ゼロ体積のみfit": "केवल गैर-शून्य आयतन fit",
        "fit alpha_e 下限": "fit alpha_e निचली सीमा",
        "fit alpha_e 上限": "fit alpha_e ऊपरी सीमा",
        "計算確認": "गणना जांच",
        "最大体積": "अधिकतम आयतन",
        "先頭時刻体積": "पहला समय आयतन",
        "平均接触角": "औसत संपर्क कोण",
        "固定theta": "स्थिर theta",
        "実行": "चलाएं",
        "解析実行": "विश्लेषण चलाएं",
        "停止": "रोकें",
        "結果": "परिणाम",
        "CSV出力": "CSV निर्यात",
        "PNG出力": "PNG निर्यात",
        "全ケースPNG出力": "सभी केस PNG निर्यात",
        "グラフ表示設定": "ग्राफ़ प्रदर्शन सेटिंग",
        "色": "रंग",
        "点色": "बिंदु रंग",
        "点サイズ": "बिंदु आकार",
        "透明度": "अपारदर्शिता",
        "文字": "पाठ",
        "マーカー": "मार्कर",
        "縦横": "आस्पेक्ट",
        "自動": "स्वतः",
        "等倍": "समान",
        "タイトル": "शीर्षक",
        "軸ラベル": "अक्ष लेबल",
        "目盛": "टिक",
        "グリッド": "ग्रिड",
        "軸対象": "अक्ष लक्ष्य",
        "軸モード": "अक्ष मोड",
        "現在グラフ": "वर्तमान ग्राफ़",
        "自動固定": "स्वतः स्थिर",
        "手動固定": "मैनुअल स्थिर",
        "x最小": "x न्यून",
        "x最大": "x अधिक",
        "y最小": "y न्यून",
        "y最大": "y अधिक",
        "x対数": "x log",
        "y対数": "y log",
        "PNG幅[in]": "PNG चौड़ाई [in]",
        "PNG高さ[in]": "PNG ऊंचाई [in]",
        "画質": "गुणवत्ता",
        "透明背景": "पारदर्शी पृष्ठभूमि",
        "低 150dpi": "कम 150dpi",
        "標準 300dpi": "मानक 300dpi",
        "高 600dpi": "उच्च 600dpi",
        "ケース": "केस",
        "時刻": "समय",
        "GIF範囲": "GIF रेंज",
        "可視化するケースと時刻を選択してください": "विज़ुअलाइज़ करने के लिए केस और समय चुनें",
        "時刻数": "समय संख्या",
        "最大体積": "अधिकतम आयतन",
        "最終体積": "अंतिम आयतन",
        "蒸発完了時刻": "वाष्पीकरण समय",
        "初期接触角": "आरंभिक संपर्क कोण",
        "最終有効接触角": "अंतिम वैध संपर्क कोण",
        "初期接触半径": "आरंभिक संपर्क त्रिज्या",
        "最終有効接触半径": "अंतिम वैध संपर्क त्रिज्या",
        "推定alpha_e": "अनुमानित alpha_e",
        "fit状態": "fit स्थिति",
        "状態": "स्थिति",
        "エラー": "त्रुटि",
        "エラー / 警告": "त्रुटि / चेतावनी",
        "体積-時間": "आयतन-समय",
        "等価半径-時間": "समतुल्य त्रिज्या-समय",
        "接触角-時間": "संपर्क कोण-समय",
        "接触半径-時間": "संपर्क त्रिज्या-समय",
        "蒸発量 EM": "वाष्पित द्रव्यमान EM",
        "理論/MD 等価半径": "सिद्धांत/MD समतुल्य त्रिज्या",
        "上下2枚": "दो स्टैक किए ग्राफ़",
        "保存する理論グラフ": "सहेजने वाला सिद्धांत ग्राफ़",
        "全ケースPNG出力するグラフを選択してください。": "सभी केस के लिए PNG निर्यात करने वाला ग्राफ़ चुनें।",
        "保存先フォルダ": "गंतव्य फ़ोल्डर",
        "親フォルダを選択": "पैरेंट फ़ोल्डर चुनें",
        "現在のグラフを保存": "वर्तमान ग्राफ़ सहेजें",
        "蒸発量 EM-時間": "वाष्पित द्रव्यमान EM-समय",
        "理論/MD 等価半径-時間": "सिद्धांत/MD समतुल्य त्रिज्या-समय",
        "fit推定": "fit अनुमान",
        "可視化": "विज़ुअलाइज़ेशन",
        "ケース: -": "केस: -",
        "時刻: -": "समय: -",
        "前": "पिछला",
        "次": "अगला",
        "表示": "दृश्य",
        "2D診断": "2D निदान",
        "3D概観": "3D अवलोकन",
        "投影": "प्रोजेक्शन",
        "粒子周期表示": "आवधिक कण",
        "点": "बिंदु",
        "最大表示点数": "अधिकतम बिंदु",
        "凡例": "लीजेंड",
        "軸目盛": "अक्ष टिक",
        "情報": "जानकारी",
        "液滴セル": "बूंद सेल",
        "fit診断": "fit निदान",
        "GIF範囲: -": "GIF रेंज: -",
        "開始": "आरंभ",
        "終了": "समाप्त",
        "PNG保存": "PNG सहेजें",
        "GIF保存": "GIF सहेजें",
        "ログ": "लॉग",
        "PNG出力プレビュー": "PNG निर्यात पूर्वावलोकन",
        "保存するグラフ": "सहेजने वाला ग्राफ़",
        "軸": "अक्ष",
        "幅 [in]": "चौड़ाई [in]",
        "高さ [in]": "ऊंचाई [in]",
        "PNG": "PNG",
        "保存": "सहेजें",
        "キャンセル": "रद्द",
        "グラフ": "ग्राफ़",
    },
}

TRANSLATIONS["en"].update(
    {
        "解析設定を保存": "Save analysis settings",
        "解析設定を読込": "Load analysis settings",
        "解析記録を保存": "Save analysis record",
        "解析設定を保存しました": "Analysis settings saved",
        "解析設定を読み込みました": "Analysis settings loaded",
        "解析記録を保存しました": "Analysis record saved",
        "解析を実行してから保存してください。": "Run the analysis before saving.",
        "解析記録の保存に失敗しました": "Failed to save analysis record",
    }
)
TRANSLATIONS["zh"].update(
    {
        "解析設定を保存": "保存分析设置",
        "解析設定を読込": "加载分析设置",
        "解析記録を保存": "保存分析记录",
        "解析設定を保存しました": "已保存分析设置",
        "解析設定を読み込みました": "已加载分析设置",
        "解析記録を保存しました": "已保存分析记录",
        "解析を実行してから保存してください。": "请先运行分析再保存。",
        "解析記録の保存に失敗しました": "保存分析记录失败",
    }
)
TRANSLATIONS["es"].update(
    {
        "解析設定を保存": "Guardar ajustes de análisis",
        "解析設定を読込": "Cargar ajustes de análisis",
        "解析記録を保存": "Guardar registro de análisis",
        "解析設定を保存しました": "Ajustes de análisis guardados",
        "解析設定を読み込みました": "Ajustes de análisis cargados",
        "解析記録を保存しました": "Registro de análisis guardado",
        "解析を実行してから保存してください。": "Ejecute el análisis antes de guardar.",
        "解析記録の保存に失敗しました": "No se pudo guardar el registro de análisis",
    }
)
TRANSLATIONS["hi"].update(
    {
        "解析設定を保存": "विश्लेषण सेटिंग सहेजें",
        "解析設定を読込": "विश्लेषण सेटिंग लोड करें",
        "解析記録を保存": "विश्लेषण रिकॉर्ड सहेजें",
        "解析設定を保存しました": "विश्लेषण सेटिंग सहेजी गई",
        "解析設定を読み込みました": "विश्लेषण सेटिंग लोड की गई",
        "解析記録を保存しました": "विश्लेषण रिकॉर्ड सहेजा गया",
        "解析を実行してから保存してください。": "सहेजने से पहले विश्लेषण चलाएं।",
        "解析記録の保存に失敗しました": "विश्लेषण रिकॉर्ड सहेजा नहीं जा सका",
    }
)

TRANSLATIONS["en"].update(
    {
        "分子離脱解析": "Molecular departure analysis",
        "分子種": "Molecule species",
        "クラスタ距離": "Cluster cutoff",
        "確定連続時刻数": "Confirmation frames",
        "高さビン数": "Height bins",
        "高さビン方式": "Height binning",
        "高さ等間隔": "Equal height",
        "球面表面積等分": "Equal fitted surface area",
        "分布表示": "Distribution metric",
        "イベント件数": "Event count",
        "面積時間あたり": "Per area-time",
        "分子離脱高さ分布": "Molecular departure height",
        "分子離脱 時刻-高さ": "Molecular departure time-height",
    }
)
TRANSLATIONS["zh"].update(
    {
        "分子離脱解析": "分子脱离分析",
        "分子種": "分子种类",
        "クラスタ距離": "团簇截断距离",
        "確定連続時刻数": "确认连续时刻数",
        "高さビン数": "高度分箱数",
        "高さビン方式": "高度分箱方式",
        "高さ等間隔": "等高度",
        "球面表面積等分": "拟合球面等面积",
        "分布表示": "分布指标",
        "イベント件数": "事件数",
        "面積時間あたり": "单位面积时间",
        "分子離脱高さ分布": "分子脱离高度分布",
        "分子離脱 時刻-高さ": "分子脱离 时间-高度",
    }
)
TRANSLATIONS["es"].update(
    {
        "分子離脱解析": "Análisis de salida molecular",
        "分子種": "Especie molecular",
        "クラスタ距離": "Corte de clúster",
        "確定連続時刻数": "Fotogramas de confirmación",
        "高さビン数": "Intervalos de altura",
        "高さビン方式": "División de altura",
        "高さ等間隔": "Altura uniforme",
        "球面表面積等分": "Área esférica uniforme",
        "分布表示": "Métrica de distribución",
        "イベント件数": "Número de eventos",
        "面積時間あたり": "Por área-tiempo",
        "分子離脱高さ分布": "Altura de salida molecular",
        "分子離脱 時刻-高さ": "Salida molecular tiempo-altura",
    }
)
TRANSLATIONS["hi"].update(
    {
        "分子離脱解析": "आणविक निर्गमन विश्लेषण",
        "分子種": "अणु प्रजाति",
        "クラスタ距離": "क्लस्टर कटऑफ",
        "確定連続時刻数": "पुष्टि फ्रेम",
        "高さビン数": "ऊंचाई बिन",
        "高さビン方式": "ऊंचाई बिन विधि",
        "高さ等間隔": "समान ऊंचाई",
        "球面表面積等分": "समान गोलाकार क्षेत्रफल",
        "分布表示": "वितरण माप",
        "イベント件数": "घटना संख्या",
        "面積時間あたり": "क्षेत्रफल-समय प्रति",
        "分子離脱高さ分布": "आणविक निर्गमन ऊंचाई",
        "分子離脱 時刻-高さ": "आणविक निर्गमन समय-ऊंचाई",
    }
)

TRANSLATIONS["en"].update(
    {
        "グラフ内容": "Plot content",
        "グラフタイトル": "Plot title",
        "x軸ラベル": "x-axis label",
        "y軸ラベル": "y-axis label",
        "目盛文字": "Tick text",
        "文字サイズ": "Font sizes",
        "タイトル文字": "Title",
        "軸ラベル文字": "Axis labels",
        "凡例文字": "Legend",
        "凡例": "Legend",
        "全系列で統一": "Apply to all series",
        "線と凡例": "Lines and legend",
        "線幅を統一": "Uniform line width",
        "線幅": "Line width",
        "線種": "Line style",
        "元の線種": "Original styles",
        "実線": "Solid",
        "破線": "Dashed",
        "点線": "Dotted",
        "一点鎖線": "Dash-dot",
        "凡例位置": "Legend position",
        "自動配置": "Best",
        "右上": "Upper right",
        "右下": "Lower right",
        "左上": "Upper left",
        "左下": "Lower left",
        "グリッド設定": "Grid settings",
        "グリッド透明度": "Grid opacity",
        "グリッド線幅": "Grid line width",
        "グリッド線種": "Grid line style",
        "x目盛回転 [deg]": "x tick rotation [deg]",
        "変更は右のプレビューへすぐ反映されます。各項目にカーソルを合わせると説明を表示します。": "Changes appear immediately in the preview. Hover over a control for details.",
        "対象": "Target",
        "点と基本表示": "Points and basic display",
        "点の不透明度": "Point opacity",
        "縦横比": "Aspect ratio",
        "表示する要素": "Visible elements",
        "軸範囲を自動設定": "Automatic axis range",
        "設定をリセット": "Reset settings",
    }
)
TRANSLATIONS["zh"].update(
    {
        "グラフ内容": "图表内容",
        "グラフタイトル": "图表标题",
        "x軸ラベル": "x 轴标签",
        "y軸ラベル": "y 轴标签",
        "目盛文字": "刻度文字",
        "文字サイズ": "字体大小",
        "タイトル文字": "标题",
        "軸ラベル文字": "轴标签",
        "凡例文字": "图例",
        "凡例": "图例",
        "全系列で統一": "应用于所有系列",
        "線と凡例": "线条和图例",
        "線幅を統一": "统一线宽",
        "線幅": "线宽",
        "線種": "线型",
        "元の線種": "原始线型",
        "実線": "实线",
        "破線": "虚线",
        "点線": "点线",
        "一点鎖線": "点划线",
        "凡例位置": "图例位置",
        "自動配置": "自动",
        "右上": "右上",
        "右下": "右下",
        "左上": "左上",
        "左下": "左下",
        "グリッド設定": "网格设置",
        "グリッド透明度": "网格透明度",
        "グリッド線幅": "网格线宽",
        "グリッド線種": "网格线型",
        "x目盛回転 [deg]": "x 刻度旋转 [deg]",
        "変更は右のプレビューへすぐ反映されます。各項目にカーソルを合わせると説明を表示します。": "更改会立即显示在右侧预览中。将鼠标悬停在控件上可查看说明。",
        "対象": "目标",
        "点と基本表示": "点和基本显示",
        "点の不透明度": "点不透明度",
        "縦横比": "纵横比",
        "表示する要素": "显示元素",
        "軸範囲を自動設定": "自动轴范围",
        "設定をリセット": "重置设置",
    }
)
TRANSLATIONS["es"].update(
    {
        "グラフ内容": "Contenido del gráfico",
        "グラフタイトル": "Título del gráfico",
        "x軸ラベル": "Etiqueta del eje x",
        "y軸ラベル": "Etiqueta del eje y",
        "目盛文字": "Texto de marcas",
        "文字サイズ": "Tamaños de fuente",
        "タイトル文字": "Título",
        "軸ラベル文字": "Etiquetas de eje",
        "凡例文字": "Leyenda",
        "凡例": "Leyenda",
        "全系列で統一": "Aplicar a todas las series",
        "線と凡例": "Líneas y leyenda",
        "線幅を統一": "Ancho de línea uniforme",
        "線幅": "Ancho de línea",
        "線種": "Estilo de línea",
        "元の線種": "Estilos originales",
        "実線": "Continua",
        "破線": "Discontinua",
        "点線": "Punteada",
        "一点鎖線": "Punto-raya",
        "凡例位置": "Posición de leyenda",
        "自動配置": "Óptima",
        "右上": "Superior derecha",
        "右下": "Inferior derecha",
        "左上": "Superior izquierda",
        "左下": "Inferior izquierda",
        "グリッド設定": "Ajustes de cuadrícula",
        "グリッド透明度": "Opacidad de cuadrícula",
        "グリッド線幅": "Ancho de cuadrícula",
        "グリッド線種": "Estilo de cuadrícula",
        "x目盛回転 [deg]": "Rotación de marcas x [deg]",
        "変更は右のプレビューへすぐ反映されます。各項目にカーソルを合わせると説明を表示します。": "Los cambios aparecen de inmediato en la vista previa. Pase el cursor sobre un control para ver su descripción.",
        "対象": "Objetivo",
        "点と基本表示": "Puntos y visualización básica",
        "点の不透明度": "Opacidad de puntos",
        "縦横比": "Relación de aspecto",
        "表示する要素": "Elementos visibles",
        "軸範囲を自動設定": "Rango de ejes automático",
        "設定をリセット": "Restablecer ajustes",
    }
)
TRANSLATIONS["hi"].update(
    {
        "グラフ内容": "ग्राफ़ सामग्री",
        "グラフタイトル": "ग्राफ़ शीर्षक",
        "x軸ラベル": "x-अक्ष लेबल",
        "y軸ラベル": "y-अक्ष लेबल",
        "目盛文字": "टिक टेक्स्ट",
        "文字サイズ": "फ़ॉन्ट आकार",
        "タイトル文字": "शीर्षक",
        "軸ラベル文字": "अक्ष लेबल",
        "凡例文字": "लीजेंड",
        "凡例": "लीजेंड",
        "全系列で統一": "सभी श्रृंखलाओं पर लागू",
        "線と凡例": "रेखाएँ और लीजेंड",
        "線幅を統一": "समान रेखा चौड़ाई",
        "線幅": "रेखा चौड़ाई",
        "線種": "रेखा शैली",
        "元の線種": "मूल शैलियाँ",
        "実線": "ठोस",
        "破線": "डैश",
        "点線": "बिंदु",
        "一点鎖線": "डैश-बिंदु",
        "凡例位置": "लीजेंड स्थान",
        "自動配置": "सर्वोत्तम",
        "右上": "ऊपरी दायाँ",
        "右下": "निचला दायाँ",
        "左上": "ऊपरी बायाँ",
        "左下": "निचला बायाँ",
        "グリッド設定": "ग्रिड सेटिंग",
        "グリッド透明度": "ग्रिड अपारदर्शिता",
        "グリッド線幅": "ग्रिड रेखा चौड़ाई",
        "グリッド線種": "ग्रिड रेखा शैली",
        "x目盛回転 [deg]": "x टिक घुमाव [deg]",
        "変更は右のプレビューへすぐ反映されます。各項目にカーソルを合わせると説明を表示します。": "बदलाव तुरंत दाईं ओर पूर्वावलोकन में दिखते हैं। विवरण के लिए नियंत्रण पर कर्सर रखें।",
        "対象": "लक्ष्य",
        "点と基本表示": "बिंदु और मूल प्रदर्शन",
        "点の不透明度": "बिंदु अपारदर्शिता",
        "縦横比": "आस्पेक्ट अनुपात",
        "表示する要素": "दिखने वाले तत्व",
        "軸範囲を自動設定": "स्वचालित अक्ष सीमा",
        "設定をリセット": "सेटिंग रीसेट करें",
    }
)

THREE_D_AUTO_MAX_POINTS = 50_000
QUALITY_DPI_OPTIONS = {
    "低 150dpi": 150,
    "標準 300dpi": 300,
    "高 600dpi": 600,
}
DEFAULT_QUALITY_LABEL = "標準 300dpi"
PNG_PREVIEW_PIXELS_PER_INCH = 72


def _tr(text: str, language: str = "ja") -> str:
    if language == "ja":
        return text
    return TRANSLATIONS.get(language, {}).get(text, text)


def _combo_set_items(combo: QComboBox, items: list[tuple[str, object]], language: str = "ja") -> None:
    current_data = combo.currentData()
    combo.blockSignals(True)
    combo.clear()
    for label, data in items:
        combo.addItem(_tr(label, language), data)
    if current_data is not None:
        index = combo.findData(current_data)
        if index >= 0:
            combo.setCurrentIndex(index)
    combo.blockSignals(False)


def _combo_data(combo: QComboBox, default: str = "") -> str:
    data = combo.currentData()
    return default if data is None else str(data)


@dataclass
class GraphSettings:
    point_color: str = COLORS["accent"]
    point_size: float = 18.0
    point_alpha: float = 0.9
    font_size: int = 10
    title_font_size: int | None = None
    axis_label_font_size: int | None = None
    legend_font_size: int | None = None
    title_text: str | None = None
    x_label_text: str | None = None
    y_label_text: str | None = None
    title_visible: bool = True
    axis_labels_visible: bool = True
    tick_labels_visible: bool = True
    legend_visible: bool = True
    legend_location: str = "best"
    grid_visible: bool = True
    grid_alpha: float | None = None
    grid_line_width: float = 0.8
    grid_line_style: str = "-"
    axis_mode: str = "auto_fixed"
    axis_auto: bool = True
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0
    aspect: str = "auto"
    x_log: bool = False
    y_log: bool = False
    marker: str = "o"
    marker_override: str | None = None
    line_width: float | None = None
    line_style: str = "source"
    x_tick_rotation: float | None = None
    image_width: float = 8.0
    image_height: float = 5.0
    dpi: int = 300
    transparent: bool = False


@dataclass
class PlotSeries:
    label: str
    x: list[float]
    y: list[float]
    style: str = "scatter"
    color: str | None = None
    marker: str | None = None
    linestyle: str = "-"
    linewidth: float = 1.5


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


class SignedScientificDoubleSpinBox(QDoubleSpinBox):
    def __init__(self) -> None:
        super().__init__()
        self.validator = QDoubleValidator(-1.0e100, 1.0e100, 16, self)
        self.validator.setNotation(QDoubleValidator.ScientificNotation)
        self.setDecimals(16)

    def textFromValue(self, value: float) -> str:
        return f"{value:.8g}"

    def valueFromText(self, text: str) -> float:
        try:
            return float(text.strip())
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


class CollapsibleGroupBox(QGroupBox):
    def __init__(self, title: str, expanded: bool = True) -> None:
        super().__init__(title)
        self.setCheckable(True)
        self.setChecked(expanded)
        self.toggled.connect(self._sync_visibility)

    @Slot(bool)
    def _sync_visibility(self, expanded: bool) -> None:
        layout = self.layout()
        if layout is not None:
            self._set_layout_visible(layout, expanded)
        self.setMaximumHeight(16_777_215 if expanded else 34)

    def _set_layout_visible(self, layout, visible: bool) -> None:
        for index in range(layout.count()):
            item = layout.itemAt(index)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.setVisible(visible)
            elif child_layout is not None:
                self._set_layout_visible(child_layout, visible)


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
            cache_session = AnalysisCacheSession(log=self.log.emit)
            for index, case in enumerate(self.cases, start=1):
                if self._stop_requested:
                    self.log.emit("残りのケースを解析せずに停止しました。")
                    break
                case_path = Path(case)
                self.log.emit(f"{case_path.name} を解析中...")
                try:
                    layout_profile = detect_analysis_layout(
                        case_path,
                        self.settings,
                    )
                    self.log.emit(
                        f"{case_path.name} のセル構成: "
                        f"{layout_profile.mode}, "
                        f"総セル数={layout_profile.expected_total_cells}, "
                        f"processor数={layout_profile.processor_count}"
                    )
                    result = analyze_case(
                        case_path,
                        self.settings,
                        stop_requested=lambda: self._stop_requested,
                        log=self.log.emit,
                        cache_session=cache_session,
                        layout_profile=layout_profile,
                    )
                except Exception as exc:
                    message = (
                        f"{case_path.name} のセル構成を判定できませんでした: {exc}"
                    )
                    self.log.emit(message)
                    result = CaseResult(
                        case_name=case_path.name,
                        case_dir=case_path,
                        status="error",
                        error=message,
                        contact_average_percent=(
                            self.settings.contact_average_percent
                        ),
                        source_case_path=str(case_path),
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
                        include_lagrangian=self.settings.departure_enabled,
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
                    apply_remote_input_paths(
                        result,
                        local_case / ".mdfoam_remote_manifest.json",
                        remote_case,
                    )
                except Exception as exc:
                    result = CaseResult(
                        case_name=remote_name(remote_case),
                        case_dir=Path(),
                        status="error",
                        error=str(exc),
                        contact_average_percent=self.settings.contact_average_percent,
                        source_case_path=remote_case,
                    )
                self.case_finished.emit(result)
                self.progress.emit(index, total)

    @Slot()
    def stop(self) -> None:
        self._stop_requested = True


class PlotWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.figure = Figure(figsize=(5, 3), tight_layout=True, facecolor=COLORS["surface"])
        self.canvas = FigureCanvas(self.figure)
        self.settings = GraphSettings()
        self.light_theme = False
        self._last_plot: tuple[str, str, str, str, list, list] | None = None
        self._last_series: tuple[str, str, str, list[PlotSeries]] | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

    def clear(self, title: str) -> None:
        self._last_plot = ("clear", title, "", "", [], [])
        self._last_series = None
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        self._apply_common_style(axis, "", "", title)
        self.canvas.draw_idle()

    def plot_xy(self, title: str, x_label: str, y_label: str, x: list[float], y: list[float]) -> None:
        self._last_plot = ("xy", title, x_label, y_label, list(x), list(y))
        self._last_series = None
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.scatter(
            x,
            y,
            s=self.settings.point_size,
            c=self.settings.point_color,
            alpha=self.settings.point_alpha,
            marker=self.settings.marker_override or self.settings.marker,
        )
        self._apply_common_style(axis, x_label, y_label, title)
        self.canvas.draw_idle()

    def plot_series(self, title: str, x_label: str, y_label: str, series_list: list[PlotSeries]) -> None:
        copied_series = [
            PlotSeries(
                item.label,
                list(item.x),
                list(item.y),
                item.style,
                item.color,
                item.marker,
                item.linestyle,
                item.linewidth,
            )
            for item in series_list
        ]
        self._last_plot = ("series", title, x_label, y_label, [], [])
        self._last_series = (title, x_label, y_label, copied_series)
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        for item in copied_series:
            if not item.x or not item.y:
                continue
            if item.style == "line":
                axis.plot(
                    item.x,
                    item.y,
                    label=item.label,
                    color=self._display_series_color(item.color),
                    linestyle=(
                        item.linestyle
                        if self.settings.line_style == "source"
                        else self.settings.line_style
                    ),
                    linewidth=(
                        self.settings.line_width
                        if self.settings.line_width is not None
                        else item.linewidth
                    ),
                    alpha=self.settings.point_alpha,
                )
            else:
                axis.scatter(
                    item.x,
                    item.y,
                    label=item.label,
                    s=self.settings.point_size,
                    c=self._display_series_color(item.color) or self.settings.point_color,
                    alpha=self.settings.point_alpha,
                    marker=self.settings.marker_override or item.marker or self.settings.marker,
                )
        if self.settings.legend_visible and any(item.label for item in copied_series):
            axis.legend(
                fontsize=self.settings.legend_font_size or max(6, self.settings.font_size - 1),
                loc=self.settings.legend_location,
            )
        self._apply_common_style(axis, x_label, y_label, title)
        self.canvas.draw_idle()

    def plot_bar(self, title: str, labels: list[str], values: list[float]) -> None:
        self._last_plot = ("bar", title, "", "蒸発完了時間 [s]", list(labels), list(values))
        self._last_series = None
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.bar(labels, values, color=self.settings.point_color, alpha=self.settings.point_alpha)
        self._apply_common_style(axis, "", "蒸発完了時間 [s]", title, is_bar=True)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def redraw(self) -> None:
        if self._last_plot is None:
            return
        kind, title, x_label, y_label, x, y = self._last_plot
        if kind == "xy":
            self.plot_xy(title, x_label, y_label, x, y)
        elif kind == "bar":
            self.plot_bar(title, x, y)
        elif kind == "series" and self._last_series is not None:
            self.plot_series(*self._last_series)
        elif kind == "clear":
            self.clear(title)

    def save_png(self, path: Path) -> None:
        original_theme = self.light_theme
        self.light_theme = True
        try:
            self.redraw()
            self.figure.set_size_inches(self.settings.image_width, self.settings.image_height, forward=False)
            self.figure.tight_layout()
            self.figure.savefig(
                path,
                dpi=self.settings.dpi,
                transparent=self.settings.transparent,
                facecolor="white",
                edgecolor="white",
                bbox_inches="tight",
            )
        finally:
            self.light_theme = original_theme
            self.redraw()

    def copy_plot_state_from(self, other: "PlotWidget") -> None:
        self.settings = GraphSettings(**vars(other.settings))
        self._last_plot = None
        if other._last_plot is not None:
            kind, title, x_label, y_label, x, y = other._last_plot
            self._last_plot = (kind, title, x_label, y_label, list(x), list(y))
        self._last_series = None
        if other._last_series is not None:
            title, x_label, y_label, series = other._last_series
            self._last_series = (
                title,
                x_label,
                y_label,
                [
                    PlotSeries(
                        item.label,
                        list(item.x),
                        list(item.y),
                        item.style,
                        item.color,
                        item.marker,
                        item.linestyle,
                        item.linewidth,
                    )
                    for item in series
                ],
            )
        self.redraw()

    def _apply_common_style(
        self,
        axis,
        x_label: str,
        y_label: str,
        title: str = "",
        is_bar: bool = False,
    ) -> None:
        settings = self.settings
        background = "#ffffff" if self.light_theme else COLORS["surface"]
        foreground = "#111111" if self.light_theme else COLORS["muted"]
        spine_color = "#111111" if self.light_theme else COLORS["border"]
        grid_color = "#c8c8c8" if self.light_theme else COLORS["grid"]
        display_title = settings.title_text if settings.title_text is not None else title
        display_x_label = settings.x_label_text if settings.x_label_text is not None else x_label
        display_y_label = settings.y_label_text if settings.y_label_text is not None else y_label
        title_font_size = settings.title_font_size or settings.font_size + 1
        axis_label_font_size = settings.axis_label_font_size or settings.font_size
        self.figure.set_facecolor(background)
        axis.set_facecolor(background)
        axis.tick_params(colors=foreground)
        axis.xaxis.label.set_color(foreground)
        axis.yaxis.label.set_color(foreground)
        axis.title.set_color("#111111" if self.light_theme else COLORS["text"])
        for spine in axis.spines.values():
            spine.set_color(spine_color)
        legend = axis.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor("#ffffff" if self.light_theme else COLORS["surface_alt"])
            legend.get_frame().set_edgecolor(spine_color)
            for text in legend.get_texts():
                text.set_color("#111111" if self.light_theme else COLORS["text"])
        if settings.title_visible and display_title:
            axis.set_title(display_title, fontsize=title_font_size)
        axis.set_xlabel(
            display_x_label if settings.axis_labels_visible else "",
            fontsize=axis_label_font_size,
        )
        axis.set_ylabel(
            display_y_label if settings.axis_labels_visible else "",
            fontsize=axis_label_font_size,
        )
        if not is_bar:
            axis.set_xscale("log" if settings.x_log else "linear")
        axis.set_yscale("log" if settings.y_log else "linear")
        axis.tick_params(
            axis="both",
            which="both",
            labelsize=settings.font_size,
            labelbottom=settings.tick_labels_visible,
            labelleft=settings.tick_labels_visible,
        )
        rotation = settings.x_tick_rotation
        if rotation is None:
            rotation = 45.0 if is_bar else 0.0
        axis.tick_params(
            axis="x",
            labelrotation=rotation if settings.tick_labels_visible else 0.0,
        )
        axis.grid(
            settings.grid_visible,
            color=grid_color,
            alpha=(
                settings.grid_alpha
                if settings.grid_alpha is not None
                else (0.65 if self.light_theme else 0.45)
            ),
            linewidth=settings.grid_line_width,
            linestyle=settings.grid_line_style,
        )
        if settings.axis_mode in ("auto_fixed", "manual_fixed") or not settings.axis_auto:
            if not is_bar and settings.x_min < settings.x_max:
                axis.set_xlim(settings.x_min, settings.x_max)
            if settings.y_min < settings.y_max:
                axis.set_ylim(settings.y_min, settings.y_max)
        if settings.aspect == "equal":
            axis.set_aspect("equal", adjustable="box")
        else:
            axis.set_aspect("auto")

    def _display_series_color(self, color: str | None) -> str | None:
        if self.light_theme and color == COLORS["md_series"]:
            return "#111111"
        return color


class CombinedPlotWidget(QWidget):
    def __init__(
        self,
        source_plots: list[PlotWidget],
        owns_source_plots: bool = False,
        light_theme: bool = False,
    ) -> None:
        super().__init__()
        self.figure = Figure(figsize=(8, 5), tight_layout=True, facecolor=COLORS["surface"])
        self.canvas = FigureCanvas(self.figure)
        self.source_plots = source_plots
        self.owns_source_plots = owns_source_plots
        self.light_theme = light_theme
        self.settings = GraphSettings(**vars(source_plots[0].settings)) if source_plots else GraphSettings()
        self.settings.image_height = min(30.0, max(self.settings.image_height, self.settings.image_height * max(1, len(source_plots))))
        self._last_plot = source_plots[0]._last_plot if source_plots else None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.redraw()

    def redraw(self) -> None:
        self.figure.clear()
        if not self.source_plots:
            axis = self.figure.add_subplot(111)
            axis.set_axis_off()
            self.canvas.draw_idle()
            return
        axes = self.figure.subplots(len(self.source_plots), 1, squeeze=False)
        for axis, plot in zip(axes[:, 0], self.source_plots):
            _draw_plot_on_axis(plot, axis, self._settings_for_source_plot(plot), self.light_theme)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _settings_for_source_plot(self, plot: PlotWidget) -> GraphSettings:
        settings = GraphSettings(**vars(self.settings))
        settings.axis_mode = plot.settings.axis_mode
        settings.axis_auto = plot.settings.axis_auto
        settings.x_min = plot.settings.x_min
        settings.x_max = plot.settings.x_max
        settings.y_min = plot.settings.y_min
        settings.y_max = plot.settings.y_max
        settings.x_log = plot.settings.x_log
        settings.y_log = plot.settings.y_log
        return settings

    def save_png(self, path: Path) -> None:
        original_theme = self.light_theme
        self.light_theme = True
        try:
            self.redraw()
            self.figure.set_size_inches(self.settings.image_width, self.settings.image_height, forward=False)
            self.figure.tight_layout()
            self.figure.savefig(
                path,
                dpi=self.settings.dpi,
                transparent=self.settings.transparent,
                facecolor="white",
                edgecolor="white",
                bbox_inches="tight",
            )
        finally:
            self.light_theme = original_theme
            self.redraw()

    def closeEvent(self, event) -> None:
        if self.owns_source_plots:
            for plot in self.source_plots:
                plot.close()
        super().closeEvent(event)


class GraphPngPreviewDialog(QDialog):
    def __init__(
        self,
        source_plot: PlotWidget | list[tuple[str, PlotWidget | list[PlotWidget]] | tuple[str, PlotWidget | list[PlotWidget], str]],
        start_dir: str,
        parent: QWidget | None = None,
        suggested_filename: str = "graph.png",
    ) -> None:
        super().__init__(parent)
        self.language = getattr(parent, "language", "ja")
        self.setWindowTitle(_tr("PNG出力プレビュー", self.language))
        self.resize(1100, 760)
        self.start_dir = start_dir
        self.suggested_filename = suggested_filename
        self.saved_path: Path | None = None
        if isinstance(source_plot, list):
            self.source_options = [
                (item[0], item[1] if isinstance(item[1], list) else [item[1]], item[2] if len(item) >= 3 else suggested_filename)
                for item in source_plot
            ]
        else:
            self.source_options = [(_tr("グラフ", self.language), [source_plot], suggested_filename)]

        self.preview_widgets: list[PlotWidget | CombinedPlotWidget] = []
        for _label, plots, _filename in self.source_options:
            if len(plots) == 1:
                preview = PlotWidget()
                preview.copy_plot_state_from(plots[0])
                preview.light_theme = True
                preview.redraw()
            else:
                preview = CombinedPlotWidget(plots, light_theme=True)
            self.preview_widgets.append(preview)
        self._initial_preview_settings = [
            GraphSettings(**vars(widget.settings)) for widget in self.preview_widgets
        ]
        self.preview_plot: PlotWidget | CombinedPlotWidget = self.preview_widgets[0]

        layout = QVBoxLayout(self)
        body = QHBoxLayout()
        layout.addLayout(body, 1)

        settings_group = QGroupBox(_tr("グラフ表示設定", self.language))
        settings_layout = QVBoxLayout(settings_group)
        self.settings_scroll = QScrollArea()
        self.settings_scroll.setWidgetResizable(True)
        self.settings_scroll.setFrameShape(QFrame.NoFrame)
        self.settings_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.settings_scroll.setMinimumWidth(360)
        self.settings_scroll.setMaximumWidth(520)
        self.settings_scroll.setWidget(settings_group)
        body.addWidget(self.settings_scroll, 0)
        help_hint = QLabel(
            _tr(
                "変更は右のプレビューへすぐ反映されます。各項目にカーソルを合わせると説明を表示します。",
                self.language,
            )
        )
        help_hint.setWordWrap(True)
        help_hint.setProperty("role", "muted")
        settings_layout.addWidget(help_hint)
        self.preview_stack = QStackedWidget()
        self.preview_scroll_areas: list[QScrollArea] = []
        for widget in self.preview_widgets:
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(False)
            scroll_area.setAlignment(Qt.AlignCenter)
            scroll_area.setWidget(widget)
            self.preview_scroll_areas.append(scroll_area)
            self.preview_stack.addWidget(scroll_area)
        body.addWidget(self.preview_stack, 1)

        self.source_combo: QComboBox | None = None
        if len(self.source_options) > 1:
            self.source_combo = QComboBox()
            self.source_combo.addItems([label for label, _plots, _filename in self.source_options])
            source_group = QGroupBox(_tr("保存するグラフ", self.language))
            source_layout = self._form_layout(source_group)
            source_layout.addRow(_tr("対象", self.language), self.source_combo)
            settings_layout.addWidget(source_group)

        content_group = QGroupBox(_tr("グラフ内容", self.language))
        content_layout = self._form_layout(content_group)
        self.title_edit = QLineEdit()
        self.title_edit.setClearButtonEnabled(True)
        self.x_label_edit = QLineEdit()
        self.x_label_edit.setClearButtonEnabled(True)
        self.y_label_edit = QLineEdit()
        self.y_label_edit.setClearButtonEnabled(True)
        content_layout.addRow(_tr("グラフタイトル", self.language), self.title_edit)
        content_layout.addRow(_tr("x軸ラベル", self.language), self.x_label_edit)
        content_layout.addRow(_tr("y軸ラベル", self.language), self.y_label_edit)
        settings_layout.addWidget(content_group)

        appearance_group = QGroupBox(_tr("点と基本表示", self.language))
        appearance_layout = self._form_layout(appearance_group)
        self.color_button = QPushButton(_tr("色", self.language))
        self.point_size_spin = self._double_spin(1.0, 200.0, 1, self.preview_plot.settings.point_size)
        self.point_size_spin.setSuffix(" pt²")
        self.alpha_spin = self._double_spin(0.05, 1.0, 2, self.preview_plot.settings.point_alpha, 0.05)
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(6, 40)
        self.font_size_spin.setValue(self.preview_plot.settings.font_size)
        self.font_size_spin.setKeyboardTracking(True)
        self.font_size_spin.setSuffix(" pt")
        self.title_font_size_spin = QSpinBox()
        self.title_font_size_spin.setRange(6, 72)
        self.title_font_size_spin.setSuffix(" pt")
        self.axis_label_font_size_spin = QSpinBox()
        self.axis_label_font_size_spin.setRange(6, 72)
        self.axis_label_font_size_spin.setSuffix(" pt")
        self.legend_font_size_spin = QSpinBox()
        self.legend_font_size_spin.setRange(6, 72)
        self.legend_font_size_spin.setSuffix(" pt")
        self.marker_combo = QComboBox()
        self.marker_combo.addItems(["o", "s", "^", "D", "x", "+", "."])
        self.marker_combo.setCurrentText(self.preview_plot.settings.marker)
        self.marker_override_check = QCheckBox(_tr("全系列で統一", self.language))
        self.aspect_combo = QComboBox()
        _combo_set_items(self.aspect_combo, [("自動", "auto"), ("等倍", "equal")], self.language)
        index = self.aspect_combo.findData(self.preview_plot.settings.aspect)
        self.aspect_combo.setCurrentIndex(index if index >= 0 else 0)
        appearance_layout.addRow(_tr("点色", self.language), self.color_button)
        appearance_layout.addRow(_tr("点サイズ", self.language), self.point_size_spin)
        appearance_layout.addRow(_tr("点の不透明度", self.language), self.alpha_spin)
        appearance_layout.addRow(_tr("目盛文字", self.language), self.font_size_spin)
        appearance_layout.addRow(_tr("マーカー", self.language), self.marker_combo)
        appearance_layout.addRow("", self.marker_override_check)
        appearance_layout.addRow(_tr("縦横比", self.language), self.aspect_combo)
        settings_layout.addWidget(appearance_group)

        font_group = QGroupBox(_tr("文字サイズ", self.language))
        font_layout = self._form_layout(font_group)
        font_layout.addRow(_tr("タイトル文字", self.language), self.title_font_size_spin)
        font_layout.addRow(_tr("軸ラベル文字", self.language), self.axis_label_font_size_spin)
        font_layout.addRow(_tr("凡例文字", self.language), self.legend_font_size_spin)
        settings_layout.addWidget(font_group)

        visibility_group = QGroupBox(_tr("表示する要素", self.language))
        visibility_layout = QGridLayout(visibility_group)
        self.title_check = QCheckBox(_tr("タイトル", self.language))
        self.axis_label_check = QCheckBox(_tr("軸ラベル", self.language))
        self.tick_label_check = QCheckBox(_tr("目盛", self.language))
        self.legend_check = QCheckBox(_tr("凡例", self.language))
        self.grid_check = QCheckBox(_tr("グリッド", self.language))
        visibility_layout.addWidget(self.title_check, 0, 0)
        visibility_layout.addWidget(self.axis_label_check, 0, 1)
        visibility_layout.addWidget(self.tick_label_check, 1, 0)
        visibility_layout.addWidget(self.legend_check, 1, 1)
        visibility_layout.addWidget(self.grid_check, 2, 0)
        settings_layout.addWidget(visibility_group)

        line_group = QGroupBox(_tr("線と凡例", self.language))
        line_layout = self._form_layout(line_group)
        self.line_width_override_check = QCheckBox(_tr("線幅を統一", self.language))
        self.line_width_spin = self._double_spin(0.1, 20.0, 1, 1.5, 0.1)
        self.line_width_spin.setSuffix(" pt")
        self.line_style_combo = QComboBox()
        _combo_set_items(
            self.line_style_combo,
            [
                ("元の線種", "source"),
                ("実線", "-"),
                ("破線", "--"),
                ("点線", ":"),
                ("一点鎖線", "-."),
            ],
            self.language,
        )
        self.legend_location_combo = QComboBox()
        _combo_set_items(
            self.legend_location_combo,
            [
                ("自動配置", "best"),
                ("右上", "upper right"),
                ("右下", "lower right"),
                ("左上", "upper left"),
                ("左下", "lower left"),
            ],
            self.language,
        )
        line_layout.addRow("", self.line_width_override_check)
        line_layout.addRow(_tr("線幅", self.language), self.line_width_spin)
        line_layout.addRow(_tr("線種", self.language), self.line_style_combo)
        line_layout.addRow(_tr("凡例位置", self.language), self.legend_location_combo)
        settings_layout.addWidget(line_group)

        grid_group = QGroupBox(_tr("グリッド設定", self.language))
        grid_layout = self._form_layout(grid_group)
        self.grid_alpha_spin = self._double_spin(0.0, 1.0, 2, 0.65, 0.05)
        self.grid_line_width_spin = self._double_spin(0.1, 5.0, 1, 0.8, 0.1)
        self.grid_line_width_spin.setSuffix(" pt")
        self.grid_line_style_combo = QComboBox()
        _combo_set_items(
            self.grid_line_style_combo,
            [("実線", "-"), ("破線", "--"), ("点線", ":"), ("一点鎖線", "-.")],
            self.language,
        )
        grid_layout.addRow(_tr("グリッド透明度", self.language), self.grid_alpha_spin)
        grid_layout.addRow(_tr("グリッド線幅", self.language), self.grid_line_width_spin)
        grid_layout.addRow(_tr("グリッド線種", self.language), self.grid_line_style_combo)
        settings_layout.addWidget(grid_group)

        axis_group = QGroupBox(_tr("軸", self.language))
        axis_layout = self._form_layout(axis_group)
        self.axis_auto_check = QCheckBox(_tr("軸範囲を自動設定", self.language))
        self.x_min_spin = self._signed_spin(self.preview_plot.settings.x_min)
        self.x_max_spin = self._signed_spin(self.preview_plot.settings.x_max)
        self.y_min_spin = self._signed_spin(self.preview_plot.settings.y_min)
        self.y_max_spin = self._signed_spin(self.preview_plot.settings.y_max)
        self.x_log_check = QCheckBox(_tr("x対数", self.language))
        self.y_log_check = QCheckBox(_tr("y対数", self.language))
        self.x_tick_rotation_spin = self._double_spin(-180.0, 180.0, 0, 0.0, 5.0)
        self.x_tick_rotation_spin.setSuffix("°")
        axis_layout.addRow("", self.axis_auto_check)
        axis_layout.addRow(_tr("x最小", self.language), self.x_min_spin)
        axis_layout.addRow(_tr("x最大", self.language), self.x_max_spin)
        axis_layout.addRow(_tr("y最小", self.language), self.y_min_spin)
        axis_layout.addRow(_tr("y最大", self.language), self.y_max_spin)
        axis_layout.addRow("", self.x_log_check)
        axis_layout.addRow("", self.y_log_check)
        axis_layout.addRow(_tr("x目盛回転 [deg]", self.language), self.x_tick_rotation_spin)
        settings_layout.addWidget(axis_group)

        output_group = QGroupBox(_tr("PNG", self.language))
        output_layout = self._form_layout(output_group)
        self.width_spin = self._double_spin(1.0, 30.0, 1, self.preview_plot.settings.image_width)
        self.height_spin = self._double_spin(1.0, 30.0, 1, self.preview_plot.settings.image_height)
        self.quality_combo = QComboBox()
        for label, dpi in QUALITY_DPI_OPTIONS.items():
            self.quality_combo.addItem(_tr(label, self.language), dpi)
        quality_index = self.quality_combo.findData(self.preview_plot.settings.dpi)
        self.quality_combo.setCurrentIndex(quality_index if quality_index >= 0 else 1)
        self.transparent_check = QCheckBox(_tr("透明背景", self.language))
        output_layout.addRow(_tr("幅 [in]", self.language), self.width_spin)
        output_layout.addRow(_tr("高さ [in]", self.language), self.height_spin)
        output_layout.addRow(_tr("画質", self.language), self.quality_combo)
        output_layout.addRow("", self.transparent_check)
        settings_layout.addWidget(output_group)
        settings_layout.addStretch(1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Reset | QDialogButtonBox.Cancel
        )
        buttons.button(QDialogButtonBox.Save).setText(_tr("保存", self.language))
        buttons.button(QDialogButtonBox.Reset).setText(_tr("設定をリセット", self.language))
        buttons.button(QDialogButtonBox.Cancel).setText(_tr("キャンセル", self.language))
        layout.addWidget(buttons)

        self._load_controls_from_preview()
        self._install_control_help(buttons)
        self._connect_controls()
        self._update_preview_canvas_size()
        buttons.button(QDialogButtonBox.Save).clicked.connect(self.save_png)
        buttons.button(QDialogButtonBox.Reset).clicked.connect(self.reset_settings)
        buttons.rejected.connect(self.reject)

    @staticmethod
    def _form_layout(group: QGroupBox) -> QFormLayout:
        form = QFormLayout(group)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        return form

    def _install_control_help(self, buttons: QDialogButtonBox) -> None:
        help_items: list[tuple[QWidget, str]] = [
            (
                self.title_edit,
                "PNGに表示するグラフタイトルを書き換えます。空欄にするとタイトル文字を消せます。",
            ),
            (
                self.x_label_edit,
                "横軸の名称と単位を書き換えます。例: 時間 [s]",
            ),
            (
                self.y_label_edit,
                "縦軸の名称と単位を書き換えます。例: 体積 [m³]",
            ),
            (
                self.color_button,
                "単一系列の点・棒の色を選びます。複数系列固有の色は維持されます。",
            ),
            (
                self.point_size_spin,
                "散布点の面積をpt²で変更します。値を大きくすると点が大きくなります。",
            ),
            (
                self.alpha_spin,
                "点・線・棒の不透明度です。1.00で完全に不透明、0に近いほど薄くなります。",
            ),
            (
                self.font_size_spin,
                "x軸・y軸の目盛数値やカテゴリ名の文字サイズを変更します。",
            ),
            (
                self.marker_combo,
                "散布点の形を選びます。○、□、△、×などを切り替えられます。",
            ),
            (
                self.marker_override_check,
                "オンにすると、複数系列が持つ個別マーカーも上の形へ統一します。",
            ),
            (
                self.aspect_combo,
                "自動は領域に合わせて縦横を伸縮し、等倍はx方向とy方向を同じ縮尺で表示します。",
            ),
            (
                self.title_font_size_spin,
                "グラフタイトルだけの文字サイズを変更します。",
            ),
            (
                self.axis_label_font_size_spin,
                "x軸・y軸ラベルだけの文字サイズを変更します。",
            ),
            (
                self.legend_font_size_spin,
                "凡例内の系列名の文字サイズを変更します。",
            ),
            (
                self.title_check,
                "グラフタイトルをPNGに表示するか切り替えます。",
            ),
            (
                self.axis_label_check,
                "x軸・y軸ラベルをPNGに表示するか切り替えます。",
            ),
            (
                self.tick_label_check,
                "軸の目盛数値やカテゴリ名を表示するか切り替えます。",
            ),
            (
                self.legend_check,
                "複数系列の凡例を表示するか切り替えます。系列がないグラフには影響しません。",
            ),
            (
                self.grid_check,
                "プロット領域の補助グリッド線を表示するか切り替えます。",
            ),
            (
                self.line_width_override_check,
                "オンにすると、複数系列が持つ個別の線幅を指定値へ統一します。",
            ),
            (
                self.line_width_spin,
                "折れ線の太さを変更します。「線幅を統一」がオンのときに有効です。",
            ),
            (
                self.line_style_combo,
                "折れ線を元の線種・実線・破線・点線・一点鎖線から選びます。",
            ),
            (
                self.legend_location_combo,
                "凡例を置く位置を選びます。自動配置はデータとの重なりが少ない位置を探します。",
            ),
            (
                self.grid_alpha_spin,
                "グリッド線の不透明度です。0で透明、1で完全に不透明になります。",
            ),
            (
                self.grid_line_width_spin,
                "グリッド線の太さを変更します。",
            ),
            (
                self.grid_line_style_combo,
                "グリッド線を実線・破線・点線・一点鎖線から選びます。",
            ),
            (
                self.axis_auto_check,
                "オンではデータ全体が収まる軸範囲を使用します。オフにすると下の最小・最大値を編集できます。",
            ),
            (
                self.x_min_spin,
                "手動設定時のx軸下限です。x最大より小さい値を指定してください。",
            ),
            (
                self.x_max_spin,
                "手動設定時のx軸上限です。x最小より大きい値を指定してください。",
            ),
            (
                self.y_min_spin,
                "手動設定時のy軸下限です。y最大より小さい値を指定してください。",
            ),
            (
                self.y_max_spin,
                "手動設定時のy軸上限です。y最小より大きい値を指定してください。",
            ),
            (
                self.x_log_check,
                "x軸を対数目盛にします。表示範囲とデータには正の値が必要です。",
            ),
            (
                self.y_log_check,
                "y軸を対数目盛にします。表示範囲とデータには正の値が必要です。",
            ),
            (
                self.x_tick_rotation_spin,
                "x軸の目盛文字を指定角度だけ回転します。長いケース名の重なり回避に便利です。",
            ),
            (
                self.width_spin,
                "保存するPNGの横幅をインチ単位で指定します。プレビューの横幅も同じ比率で変わります。",
            ),
            (
                self.height_spin,
                "保存するPNGの高さをインチ単位で指定します。プレビューの高さも同じ比率で変わります。",
            ),
            (
                self.quality_combo,
                "保存時の解像度をdpiで選びます。高dpiほど精細ですが、ファイルサイズと保存時間が増えます。",
            ),
            (
                self.transparent_check,
                "オンにすると保存PNGの背景を透明にします。プレビューは視認性のため白背景で表示します。",
            ),
        ]
        if self.source_combo is not None:
            help_items.append(
                (
                    self.source_combo,
                    "プレビューして保存するグラフを切り替えます。グラフごとの表示設定は個別に保持されます。",
                )
            )
        for widget, description in help_items:
            self._set_control_help(widget, description)
            if isinstance(
                widget,
                (QComboBox, QDoubleSpinBox, QLineEdit, QSpinBox, QPushButton),
            ):
                widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self._set_control_help(
            buttons.button(QDialogButtonBox.Save),
            "現在プレビューしている内容と表示設定でPNGを保存します。",
        )
        self._set_control_help(
            buttons.button(QDialogButtonBox.Reset),
            "現在選択中のグラフだけ、PNGプレビューを開いた時点の設定へ戻します。",
        )
        self._set_control_help(
            buttons.button(QDialogButtonBox.Cancel),
            "PNGを保存せずにプレビュー画面を閉じます。",
        )

    @staticmethod
    def _set_control_help(widget: QWidget, description: str) -> None:
        widget.setToolTip(description)
        widget.setStatusTip(description)
        widget.setWhatsThis(description)
        widget.setAccessibleDescription(description)

    def _double_spin(
        self,
        minimum: float,
        maximum: float,
        decimals: int,
        value: float,
        step: float = 1.0,
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setValue(value)
        spin.setKeyboardTracking(True)
        return spin

    def _signed_spin(self, value: float) -> QDoubleSpinBox:
        spin = SignedScientificDoubleSpinBox()
        spin.setRange(-1.0e100, 1.0e100)
        spin.setSingleStep(1.0)
        spin.setValue(value)
        spin.setKeyboardTracking(True)
        return spin

    def _load_controls_from_preview(self) -> None:
        settings = self.preview_plot.settings
        controls = (
            self.title_edit,
            self.x_label_edit,
            self.y_label_edit,
            self.point_size_spin,
            self.alpha_spin,
            self.font_size_spin,
            self.title_font_size_spin,
            self.axis_label_font_size_spin,
            self.legend_font_size_spin,
            self.marker_combo,
            self.marker_override_check,
            self.aspect_combo,
            self.title_check,
            self.axis_label_check,
            self.tick_label_check,
            self.legend_check,
            self.grid_check,
            self.line_width_override_check,
            self.line_width_spin,
            self.line_style_combo,
            self.legend_location_combo,
            self.grid_alpha_spin,
            self.grid_line_width_spin,
            self.grid_line_style_combo,
            self.axis_auto_check,
            self.x_min_spin,
            self.x_max_spin,
            self.y_min_spin,
            self.y_max_spin,
            self.x_log_check,
            self.y_log_check,
            self.x_tick_rotation_spin,
            self.width_spin,
            self.height_spin,
            self.quality_combo,
            self.transparent_check,
        )
        for control in controls:
            control.blockSignals(True)
        original_title, original_x_label, original_y_label = self._original_labels()
        self.title_edit.setText(
            settings.title_text if settings.title_text is not None else original_title
        )
        self.x_label_edit.setText(
            settings.x_label_text if settings.x_label_text is not None else original_x_label
        )
        self.y_label_edit.setText(
            settings.y_label_text if settings.y_label_text is not None else original_y_label
        )
        self._set_color_button(settings.point_color)
        self.point_size_spin.setValue(settings.point_size)
        self.alpha_spin.setValue(settings.point_alpha)
        self.font_size_spin.setValue(settings.font_size)
        self.title_font_size_spin.setValue(settings.title_font_size or settings.font_size + 1)
        self.axis_label_font_size_spin.setValue(settings.axis_label_font_size or settings.font_size)
        self.legend_font_size_spin.setValue(
            settings.legend_font_size or max(6, settings.font_size - 1)
        )
        self.marker_combo.setCurrentText(settings.marker_override or settings.marker)
        self.marker_override_check.setChecked(settings.marker_override is not None)
        aspect_index = self.aspect_combo.findData(settings.aspect)
        self.aspect_combo.setCurrentIndex(aspect_index if aspect_index >= 0 else 0)
        self.title_check.setChecked(settings.title_visible)
        self.axis_label_check.setChecked(settings.axis_labels_visible)
        self.tick_label_check.setChecked(settings.tick_labels_visible)
        self.legend_check.setChecked(settings.legend_visible)
        self.grid_check.setChecked(settings.grid_visible)
        self.line_width_override_check.setChecked(settings.line_width is not None)
        self.line_width_spin.setValue(
            settings.line_width if settings.line_width is not None else 1.5
        )
        line_style_index = self.line_style_combo.findData(settings.line_style)
        self.line_style_combo.setCurrentIndex(line_style_index if line_style_index >= 0 else 0)
        legend_location_index = self.legend_location_combo.findData(settings.legend_location)
        self.legend_location_combo.setCurrentIndex(
            legend_location_index if legend_location_index >= 0 else 0
        )
        self.grid_alpha_spin.setValue(
            settings.grid_alpha if settings.grid_alpha is not None else 0.65
        )
        self.grid_line_width_spin.setValue(settings.grid_line_width)
        grid_style_index = self.grid_line_style_combo.findData(settings.grid_line_style)
        self.grid_line_style_combo.setCurrentIndex(
            grid_style_index if grid_style_index >= 0 else 0
        )
        self.axis_auto_check.setChecked(settings.axis_auto)
        self.x_min_spin.setValue(settings.x_min)
        self.x_max_spin.setValue(settings.x_max)
        self.y_min_spin.setValue(settings.y_min)
        self.y_max_spin.setValue(settings.y_max)
        self.x_log_check.setChecked(settings.x_log)
        self.y_log_check.setChecked(settings.y_log)
        self.x_tick_rotation_spin.setValue(
            settings.x_tick_rotation
            if settings.x_tick_rotation is not None
            else (45.0 if self._plot_kind() == "bar" else 0.0)
        )
        self.width_spin.setValue(settings.image_width)
        self.height_spin.setValue(settings.image_height)
        quality_index = self.quality_combo.findData(settings.dpi)
        self.quality_combo.setCurrentIndex(quality_index if quality_index >= 0 else 1)
        self.transparent_check.setChecked(settings.transparent)
        for control in controls:
            control.blockSignals(False)
        self._update_axis_spin_enabled()
        self._update_line_controls_enabled()

    def _plot_kind(self) -> str:
        return (
            self.preview_plot._last_plot[0]
            if self.preview_plot._last_plot is not None
            else "xy"
        )

    def _original_labels(self) -> tuple[str, str, str]:
        if isinstance(self.preview_plot, CombinedPlotWidget):
            return "", "", ""
        if self.preview_plot._last_plot is None:
            return "", "", ""
        _kind, title, x_label, y_label, _x, _y = self.preview_plot._last_plot
        return title, x_label, y_label

    def _connect_controls(self) -> None:
        if self.source_combo is not None:
            self.source_combo.currentIndexChanged.connect(self.on_source_changed)
        self.color_button.clicked.connect(self.choose_color)
        for widget in (self.title_edit, self.x_label_edit, self.y_label_edit):
            widget.textChanged.connect(lambda _: self.apply_settings())
        for widget in (
            self.point_size_spin,
            self.alpha_spin,
            self.font_size_spin,
            self.title_font_size_spin,
            self.axis_label_font_size_spin,
            self.legend_font_size_spin,
            self.marker_combo,
            self.marker_override_check,
            self.aspect_combo,
            self.title_check,
            self.axis_label_check,
            self.tick_label_check,
            self.legend_check,
            self.grid_check,
            self.line_width_override_check,
            self.line_width_spin,
            self.line_style_combo,
            self.legend_location_combo,
            self.grid_alpha_spin,
            self.grid_line_width_spin,
            self.grid_line_style_combo,
            self.axis_auto_check,
            self.x_min_spin,
            self.x_max_spin,
            self.y_min_spin,
            self.y_max_spin,
            self.x_log_check,
            self.y_log_check,
            self.x_tick_rotation_spin,
            self.width_spin,
            self.height_spin,
            self.quality_combo,
            self.transparent_check,
        ):
            if isinstance(widget, QComboBox):
                widget.currentTextChanged.connect(lambda _: self.apply_settings())
            elif isinstance(widget, QCheckBox):
                widget.stateChanged.connect(lambda _: self.apply_settings())
            else:
                widget.valueChanged.connect(lambda _: self.apply_settings())

    @Slot()
    def choose_color(self) -> None:
        color = QColorDialog.getColor(QColor(self.preview_plot.settings.point_color), self, _tr("点色", self.language))
        if not color.isValid():
            return
        self.preview_plot.settings.point_color = color.name()
        self._set_color_button(color.name())
        self.preview_plot.redraw()

    @Slot(int)
    def on_source_changed(self, index: int) -> None:
        if index < 0 or index >= len(self.source_options):
            return
        self.preview_stack.setCurrentIndex(index)
        self.preview_plot = self.preview_widgets[index]
        self._load_controls_from_preview()
        self._update_preview_canvas_size()
        self.preview_plot.redraw()

    @Slot()
    def reset_settings(self) -> None:
        index = self.source_combo.currentIndex() if self.source_combo is not None else 0
        if index < 0 or index >= len(self._initial_preview_settings):
            return
        self.preview_plot.settings = GraphSettings(
            **vars(self._initial_preview_settings[index])
        )
        self._load_controls_from_preview()
        self._update_preview_canvas_size()
        self.preview_plot.redraw()

    @Slot()
    def apply_settings(self) -> None:
        settings = self.preview_plot.settings
        settings.point_size = self.point_size_spin.value()
        settings.point_alpha = self.alpha_spin.value()
        settings.font_size = self.font_size_spin.value()
        settings.title_font_size = self.title_font_size_spin.value()
        settings.axis_label_font_size = self.axis_label_font_size_spin.value()
        settings.legend_font_size = self.legend_font_size_spin.value()
        original_title, original_x_label, original_y_label = self._original_labels()
        settings.title_text = self._text_override(self.title_edit.text(), original_title)
        settings.x_label_text = self._text_override(self.x_label_edit.text(), original_x_label)
        settings.y_label_text = self._text_override(self.y_label_edit.text(), original_y_label)
        marker_value = self.marker_combo.currentText()
        settings.marker = marker_value
        settings.marker_override = (
            marker_value if self.marker_override_check.isChecked() else None
        )
        settings.aspect = _combo_data(self.aspect_combo, "auto")
        settings.title_visible = self.title_check.isChecked()
        settings.axis_labels_visible = self.axis_label_check.isChecked()
        settings.tick_labels_visible = self.tick_label_check.isChecked()
        settings.legend_visible = self.legend_check.isChecked()
        settings.grid_visible = self.grid_check.isChecked()
        settings.line_width = (
            self.line_width_spin.value()
            if self.line_width_override_check.isChecked()
            else None
        )
        settings.line_style = _combo_data(self.line_style_combo, "source")
        settings.legend_location = _combo_data(self.legend_location_combo, "best")
        settings.grid_alpha = self.grid_alpha_spin.value()
        settings.grid_line_width = self.grid_line_width_spin.value()
        settings.grid_line_style = _combo_data(self.grid_line_style_combo, "-")
        settings.axis_auto = self.axis_auto_check.isChecked()
        settings.x_min = self.x_min_spin.value()
        settings.x_max = self.x_max_spin.value()
        settings.y_min = self.y_min_spin.value()
        settings.y_max = self.y_max_spin.value()
        settings.x_log = self.x_log_check.isChecked()
        settings.y_log = self.y_log_check.isChecked()
        settings.x_tick_rotation = self.x_tick_rotation_spin.value()
        settings.image_width = self.width_spin.value()
        settings.image_height = self.height_spin.value()
        settings.dpi = int(self.quality_combo.currentData() or QUALITY_DPI_OPTIONS[DEFAULT_QUALITY_LABEL])
        settings.transparent = self.transparent_check.isChecked()
        self._update_axis_spin_enabled()
        self._update_line_controls_enabled()
        self.preview_plot.figure.set_size_inches(settings.image_width, settings.image_height, forward=False)
        self._update_preview_canvas_size()
        self.preview_plot.redraw()

    @staticmethod
    def _text_override(value: str, original: str) -> str | None:
        text = value.strip()
        return None if text == original else text

    def _update_preview_canvas_size(self) -> None:
        settings = self.preview_plot.settings
        width = max(1, round(settings.image_width * PNG_PREVIEW_PIXELS_PER_INCH))
        height = max(1, round(settings.image_height * PNG_PREVIEW_PIXELS_PER_INCH))
        self.preview_plot.figure.set_dpi(PNG_PREVIEW_PIXELS_PER_INCH)
        self.preview_plot.figure.set_size_inches(settings.image_width, settings.image_height, forward=False)
        self.preview_plot.canvas.setFixedSize(width, height)
        self.preview_plot.setFixedSize(width, height)

    def _update_axis_spin_enabled(self) -> None:
        plot_kind = self._plot_kind()
        x_axis_available = plot_kind in ("xy", "series")
        manual_axis = not self.axis_auto_check.isChecked()
        self.x_min_spin.setEnabled(manual_axis and x_axis_available)
        self.x_max_spin.setEnabled(manual_axis and x_axis_available)
        self.y_min_spin.setEnabled(manual_axis)
        self.y_max_spin.setEnabled(manual_axis)
        self.x_log_check.setEnabled(x_axis_available)

    def _update_line_controls_enabled(self) -> None:
        plots = (
            self.preview_plot.source_plots
            if isinstance(self.preview_plot, CombinedPlotWidget)
            else [self.preview_plot]
        )
        has_scatter = False
        has_line = False
        has_legend = False
        for plot in plots:
            kind = plot._last_plot[0] if plot._last_plot is not None else "clear"
            if kind == "xy":
                has_scatter = True
            elif kind == "series" and plot._last_series is not None:
                series_list = plot._last_series[3]
                has_scatter = has_scatter or any(
                    item.style != "line" for item in series_list
                )
                has_line = has_line or any(
                    item.style == "line" for item in series_list
                )
                has_legend = has_legend or any(item.label for item in series_list)

        self.point_size_spin.setEnabled(has_scatter)
        self.marker_combo.setEnabled(has_scatter)
        self.marker_override_check.setEnabled(has_scatter and has_legend)
        self.line_width_override_check.setEnabled(has_line)
        self.line_width_spin.setEnabled(
            has_line and self.line_width_override_check.isChecked()
        )
        self.line_style_combo.setEnabled(has_line)
        self.legend_check.setEnabled(has_legend)
        self.legend_location_combo.setEnabled(
            has_legend and self.legend_check.isChecked()
        )
        self.legend_font_size_spin.setEnabled(
            has_legend and self.legend_check.isChecked()
        )
        grid_enabled = self.grid_check.isChecked()
        self.grid_alpha_spin.setEnabled(grid_enabled)
        self.grid_line_width_spin.setEnabled(grid_enabled)
        self.grid_line_style_combo.setEnabled(grid_enabled)

    def _set_color_button(self, color: str) -> None:
        self.color_button.setText(color.upper())
        self.color_button.setStyleSheet(f"background-color: {color};")

    @Slot()
    def save_png(self) -> None:
        index = self.source_combo.currentIndex() if self.source_combo is not None else 0
        filename = self.source_options[index][2] if 0 <= index < len(self.source_options) else self.suggested_filename
        start_path = str(Path(self.start_dir) / filename)
        path, _ = QFileDialog.getSaveFileName(self, _tr("現在のグラフを保存", self.language), start_path, "PNG (*.png)")
        if not path:
            return
        output_path = Path(path)
        if output_path.suffix.lower() != ".png":
            output_path = output_path.with_suffix(".png")
        self.preview_plot.save_png(output_path)
        self.saved_path = output_path
        self.accept()


class VisualizationPlotWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.figure = Figure(figsize=(6, 4), tight_layout=True, facecolor=COLORS["surface"])
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

    def clear(self, message: str = "可視化するケースと時刻を選択してください") -> None:
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.set_facecolor(COLORS["surface"])
        axis.text(
            0.5,
            0.5,
            message,
            color=COLORS["muted"],
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
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
        is_3d = mode == "3d" or mode.startswith("3D")
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
        self._apply_dark_axis_style(axis, is_3d)
        self.canvas.draw_idle()
        return downsample

    def _apply_dark_axis_style(self, axis, is_3d: bool) -> None:
        self.figure.set_facecolor(COLORS["surface"])
        axis.set_facecolor(COLORS["surface"])
        axis.tick_params(colors=COLORS["muted"])
        axis.xaxis.label.set_color(COLORS["muted"])
        axis.yaxis.label.set_color(COLORS["muted"])
        axis.title.set_color(COLORS["text"])
        if is_3d and hasattr(axis, "zaxis"):
            axis.zaxis.label.set_color(COLORS["muted"])
            axis.zaxis.set_tick_params(colors=COLORS["muted"])
            for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
                pane.set_facecolor(COLORS["surface_alt"])
                pane.set_edgecolor(COLORS["border"])
        for spine in axis.spines.values():
            spine.set_color(COLORS["border"])

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
        self.language = "ja"
        self.setStyleSheet(APP_STYLESHEET)
        self.setWindowTitle(_tr("mdFOAM 密度解析アプリ", self.language))
        self.resize(1360, 900)

        self.cases: list[Path] = []
        self.local_folder_path = Path.cwd()
        self.remote_cases: list[str] = []
        self.loaded_source = ""
        self.remote_browser_connection: SshConnection | None = None
        self.remote_browser_path = ""
        self._last_visual_downsample_message = ""
        self.results: list[CaseResult] = []
        self._last_run_settings: AnalysisSettings | None = None
        self._last_run_context: RunContext | None = None
        self._theory_comparison_cache: dict[tuple[int, TheorySettings], TheoryComparison] = {}
        self._auto_axis_ranges: dict[str, tuple[float, float, float, float]] = {}
        self.worker: AnalyzerWorker | None = None
        self.thread: QThread | None = None
        self._theory_refresh_timer = QTimer(self)
        self._theory_refresh_timer.setSingleShot(True)
        self._theory_refresh_timer.setInterval(250)
        self._theory_refresh_timer.timeout.connect(self.refresh_theory_outputs)

        self._build_ui()
        self._configure_spinbox_input_behavior()
        self._load_ssh_profile()
        self._set_source_mode("local")
        self._connect_signals()
        self.folder_edit.setText(str(Path.cwd()))
        self.load_folder(Path.cwd())

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.sidebar = QFrame()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(232)
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(18, 24, 18, 18)
        sidebar_layout.setSpacing(8)
        brand_title = QLabel("mdFOAM")
        brand_title.setObjectName("brandTitle")
        brand_subtitle = QLabel("DENSITY ANALYZER")
        brand_subtitle.setObjectName("brandSubtitle")
        sidebar_layout.addWidget(brand_title)
        sidebar_layout.addWidget(brand_subtitle)
        sidebar_layout.addSpacing(24)

        self.workflow_nav_group = QButtonGroup(self)
        self.workflow_nav_group.setExclusive(True)
        self.input_nav_button = QPushButton("入力")
        self.settings_nav_button = QPushButton("解析設定")
        self.results_nav_button = QPushButton("結果")
        self.workflow_nav_buttons = [
            self.input_nav_button,
            self.settings_nav_button,
            self.results_nav_button,
        ]
        for index, button in enumerate(self.workflow_nav_buttons):
            button.setCheckable(True)
            button.setProperty("nav", True)
            self.workflow_nav_group.addButton(button, index)
            sidebar_layout.addWidget(button)
        self.input_nav_button.setChecked(True)
        sidebar_layout.addStretch(1)
        self.log_toggle_button = QPushButton("ログ")
        self.log_toggle_button.setCheckable(True)
        self.log_toggle_button.setProperty("nav", True)
        sidebar_layout.addWidget(self.log_toggle_button)

        self.language_label = QLabel("言語")
        self.language_combo = QComboBox()
        for code, label in LANGUAGES.items():
            self.language_combo.addItem(label, code)
        sidebar_layout.addWidget(self.language_label)
        sidebar_layout.addWidget(self.language_combo)
        root_layout.addWidget(self.sidebar)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(24, 18, 24, 18)
        content_layout.setSpacing(12)
        page_header = QHBoxLayout()
        page_header_text = QVBoxLayout()
        self.page_title_label = QLabel("入力")
        self.page_title_label.setObjectName("pageTitle")
        self.page_subtitle_label = QLabel("解析対象とデータソースを選択")
        self.page_subtitle_label.setObjectName("pageSubtitle")
        page_header_text.addWidget(self.page_title_label)
        page_header_text.addWidget(self.page_subtitle_label)
        page_header.addLayout(page_header_text)
        page_header.addStretch(1)
        content_layout.addLayout(page_header)

        self.workflow_tabs = QTabWidget()
        self.workflow_tabs.tabBar().hide()
        self.workflow_tabs.setDocumentMode(True)
        self.workflow_tabs.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        content_layout.addWidget(self.workflow_tabs, 1)

        input_tab = QWidget()
        input_outer_layout = QHBoxLayout(input_tab)
        input_outer_layout.setContentsMargins(0, 0, 0, 0)
        input_outer_layout.addStretch(1)
        self.input_content = QWidget()
        self.input_content.setMaximumWidth(1120)
        input_layout = QVBoxLayout(self.input_content)
        self.input_layout = input_layout
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(12)
        input_outer_layout.addWidget(self.input_content, 1)
        input_outer_layout.addStretch(1)
        self.workflow_tabs.addTab(input_tab, "入力")

        source_group = QGroupBox("入力元")
        source_layout = QVBoxLayout(source_group)
        source_row = QHBoxLayout()
        self.source_combo = QComboBox()
        self.source_combo.setMaximumWidth(200)
        _combo_set_items(self.source_combo, [("ローカル", "local"), ("SSH", "ssh")], self.language)
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
        input_layout.addWidget(self.source_stack)

        local_panel = QWidget()
        local_layout = QVBoxLayout(local_panel)
        local_group = QGroupBox("ローカルフォルダ")
        local_group_layout = QHBoxLayout(local_group)
        self.browse_button = QPushButton("フォルダを選択")
        self.clear_local_cache_button = QPushButton("ローカル解析キャッシュ削除")
        local_group_layout.addWidget(QLabel("解析対象ケースを含むフォルダを選択します。"))
        local_group_layout.addStretch(1)
        local_group_layout.addWidget(self.clear_local_cache_button)
        local_group_layout.addWidget(self.browse_button)
        local_layout.addWidget(local_group)
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
        for widget in (
            self.profile_edit,
            self.host_edit,
            self.port_spin,
            self.username_edit,
            self.key_path_edit,
            self.secret_edit,
            self.remote_path_edit,
        ):
            widget.setMaximumWidth(360)
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
        input_layout.addWidget(case_group, 1)

        settings_tab = QWidget()
        settings_tab_layout = QVBoxLayout(settings_tab)
        self.workflow_tabs.addTab(settings_tab, "解析設定")

        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_canvas = QWidget()
        settings_canvas_layout = QHBoxLayout(settings_canvas)
        settings_canvas_layout.setContentsMargins(0, 0, 0, 0)
        settings_canvas_layout.addStretch(1)
        self.settings_content = QWidget()
        self.settings_content.setMaximumWidth(1120)
        self.settings_grid = QGridLayout(self.settings_content)
        self.settings_grid.setContentsMargins(0, 0, 0, 0)
        self.settings_grid.setHorizontalSpacing(14)
        self.settings_grid.setVerticalSpacing(14)
        settings_canvas_layout.addWidget(self.settings_content, 1)
        settings_canvas_layout.addStretch(1)
        settings_scroll.setWidget(settings_canvas)
        settings_tab_layout.addWidget(settings_scroll, 1)

        self.basic_group = QGroupBox("基本設定")
        basic_layout = QFormLayout(self.basic_group)
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
        self.settings_grid.addWidget(self.basic_group, 0, 0)

        self.fallback_group = QGroupBox("セル体積 fallback")
        fallback_layout = QFormLayout(self.fallback_group)
        fallback_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.cell_volume_spin = self._scientific_spin(0.0)
        self.dx_spin = self._scientific_spin(0.0)
        self.dy_spin = self._scientific_spin(0.0)
        self.dz_spin = self._scientific_spin(0.0)
        fallback_layout.addRow("セル体積 fallback", self.cell_volume_spin)
        fallback_layout.addRow("dx fallback", self.dx_spin)
        fallback_layout.addRow("dy fallback", self.dy_spin)
        fallback_layout.addRow("dz fallback", self.dz_spin)
        self.settings_grid.addWidget(self.fallback_group, 1, 0)

        self.advanced_group = QGroupBox("詳細設定")
        advanced_layout = QFormLayout(self.advanced_group)
        advanced_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.contact_fit_lower_spin = self._fraction_spin(0.5)
        self.contact_fit_upper_spin = self._fraction_spin(1.0)
        self.contact_average_percent_spin = QDoubleSpinBox()
        self.contact_average_percent_spin.setDecimals(1)
        self.contact_average_percent_spin.setRange(1.0, 100.0)
        self.contact_average_percent_spin.setSingleStep(5.0)
        self.contact_average_percent_spin.setSuffix(" %")
        self.contact_average_percent_spin.setValue(100.0)
        self.contact_unwrap_check = QCheckBox("有効")
        self.contact_unwrap_check.setChecked(True)
        advanced_layout.addRow("接触角fit下限", self.contact_fit_lower_spin)
        advanced_layout.addRow("接触角fit上限", self.contact_fit_upper_spin)
        advanced_layout.addRow("平均接触角の対象範囲", self.contact_average_percent_spin)
        advanced_layout.addRow("xy周期補正", self.contact_unwrap_check)
        self.settings_grid.addWidget(self.advanced_group, 0, 1)

        self.departure_group = QGroupBox("分子離脱解析")
        departure_layout = QFormLayout(self.departure_group)
        departure_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.departure_enabled_check = QCheckBox("有効")
        self.departure_species_edit = QLineEdit("water")
        self.departure_cutoff_nm_spin = QDoubleSpinBox()
        self.departure_cutoff_nm_spin.setDecimals(3)
        self.departure_cutoff_nm_spin.setRange(0.01, 10.0)
        self.departure_cutoff_nm_spin.setSingleStep(0.05)
        self.departure_cutoff_nm_spin.setSuffix(" nm")
        self.departure_cutoff_nm_spin.setValue(0.40)
        self.departure_confirmation_spin = QSpinBox()
        self.departure_confirmation_spin.setRange(1, 100)
        self.departure_confirmation_spin.setValue(3)
        self.departure_height_bins_spin = QSpinBox()
        self.departure_height_bins_spin.setRange(1, 100)
        self.departure_height_bins_spin.setValue(10)
        self.departure_bin_mode_combo = QComboBox()
        _combo_set_items(
            self.departure_bin_mode_combo,
            [
                ("高さ等間隔", "equal_height"),
                ("球面表面積等分", "equal_surface_area"),
            ],
            self.language,
        )
        self.departure_bin_mode_combo.setToolTip(
            "球フィットでは球面帯面積 dA=2πRdz のため、"
            "表面積等分の境界は高さ等間隔と一致します。"
        )
        self.departure_intensity_combo = QComboBox()
        _combo_set_items(
            self.departure_intensity_combo,
            [("イベント件数", "count"), ("面積時間あたり", "rate")],
            self.language,
        )
        departure_layout.addRow("分子離脱解析", self.departure_enabled_check)
        departure_layout.addRow("分子種", self.departure_species_edit)
        departure_layout.addRow("クラスタ距離", self.departure_cutoff_nm_spin)
        departure_layout.addRow("確定連続時刻数", self.departure_confirmation_spin)
        departure_layout.addRow("高さビン数", self.departure_height_bins_spin)
        departure_layout.addRow("高さビン方式", self.departure_bin_mode_combo)
        departure_layout.addRow("分布表示", self.departure_intensity_combo)
        self.settings_grid.addWidget(self.departure_group, 1, 1)

        self.theory_group = QGroupBox("蒸発係数 / 理論比較")
        theory_layout = QFormLayout(self.theory_group)
        theory_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        default_theory = THEORY_PRESETS["xlsx準拠"]
        self.theory_preset_combo = QComboBox()
        self.theory_preset_combo.addItems(list(THEORY_PRESETS.keys()))
        self.theory_rho_v_spin = self._scientific_spin(default_theory.rho_v)
        self.theory_rho_l_spin = self._scientific_spin(default_theory.rho_l)
        self.theory_temperature_spin = self._scientific_spin(default_theory.temperature)
        self.theory_molecule_mass_spin = self._scientific_spin(default_theory.molecule_mass)
        self.theory_v0_source_combo = QComboBox()
        _combo_set_items(self.theory_v0_source_combo, [("最大体積", "max_volume"), ("先頭時刻体積", "first_volume")], self.language)
        self.theory_theta_source_combo = QComboBox()
        _combo_set_items(self.theory_theta_source_combo, [("平均接触角", "average"), ("固定theta", "fixed")], self.language)
        self.theory_fixed_theta_spin = QDoubleSpinBox()
        self.theory_fixed_theta_spin.setDecimals(6)
        self.theory_fixed_theta_spin.setRange(0.001, 179.999)
        self.theory_fixed_theta_spin.setSingleStep(0.1)
        self.theory_fixed_theta_spin.setValue(default_theory.fixed_theta_deg)
        self.theory_fit_percent_spin = QDoubleSpinBox()
        self.theory_fit_percent_spin.setDecimals(1)
        self.theory_fit_percent_spin.setRange(1.0, 100.0)
        self.theory_fit_percent_spin.setSingleStep(5.0)
        self.theory_fit_percent_spin.setSuffix(" %")
        self.theory_fit_percent_spin.setValue(100.0)
        self.theory_fit_nonzero_check = QCheckBox("有効")
        self.theory_fit_nonzero_check.setChecked(True)
        self.theory_fit_alpha_min_spin = QDoubleSpinBox()
        self.theory_fit_alpha_min_spin.setDecimals(4)
        self.theory_fit_alpha_min_spin.setRange(0.0, 10.0)
        self.theory_fit_alpha_min_spin.setSingleStep(0.05)
        self.theory_fit_alpha_min_spin.setValue(0.0)
        self.theory_fit_alpha_max_spin = QDoubleSpinBox()
        self.theory_fit_alpha_max_spin.setDecimals(4)
        self.theory_fit_alpha_max_spin.setRange(0.0, 10.0)
        self.theory_fit_alpha_max_spin.setSingleStep(0.05)
        self.theory_fit_alpha_max_spin.setValue(1.0)
        self.theory_diagnostics_label = QLabel()
        self.theory_diagnostics_label.setWordWrap(True)
        self.theory_diagnostics_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        theory_layout.addRow("物性値プリセット", self.theory_preset_combo)
        theory_layout.addRow("飽和蒸気密度 rho_v [kg/m^3]", self.theory_rho_v_spin)
        theory_layout.addRow("液体密度 rho_l [kg/m^3]", self.theory_rho_l_spin)
        theory_layout.addRow("温度 T [K]", self.theory_temperature_spin)
        theory_layout.addRow("分子1個の質量 m [kg]", self.theory_molecule_mass_spin)
        theory_layout.addRow("理論初期体積 V0", self.theory_v0_source_combo)
        theory_layout.addRow("接触角ソース", self.theory_theta_source_combo)
        theory_layout.addRow("固定theta [deg]", self.theory_fixed_theta_spin)
        theory_layout.addRow("fit対象範囲", self.theory_fit_percent_spin)
        theory_layout.addRow("非ゼロ体積のみfit", self.theory_fit_nonzero_check)
        theory_layout.addRow("fit alpha_e 下限", self.theory_fit_alpha_min_spin)
        theory_layout.addRow("fit alpha_e 上限", self.theory_fit_alpha_max_spin)
        theory_layout.addRow("計算確認", self.theory_diagnostics_label)
        self.settings_grid.addWidget(self.theory_group, 2, 0, 1, 2)
        self.settings_grid.setColumnStretch(0, 1)
        self.settings_grid.setColumnStretch(1, 1)
        self.settings_grid.setRowStretch(2, 1)

        compact_fields = (
            self.field_combo,
            self.threshold_spin,
            self.zero_spin,
            self.zero_count_spin,
            self.cell_volume_spin,
            self.dx_spin,
            self.dy_spin,
            self.dz_spin,
            self.contact_fit_lower_spin,
            self.contact_fit_upper_spin,
            self.contact_average_percent_spin,
            self.departure_species_edit,
            self.departure_cutoff_nm_spin,
            self.departure_confirmation_spin,
            self.departure_height_bins_spin,
            self.departure_bin_mode_combo,
            self.departure_intensity_combo,
            self.theory_preset_combo,
            self.theory_rho_v_spin,
            self.theory_rho_l_spin,
            self.theory_temperature_spin,
            self.theory_molecule_mass_spin,
            self.theory_v0_source_combo,
            self.theory_theta_source_combo,
            self.theory_fixed_theta_spin,
            self.theory_fit_percent_spin,
            self.theory_fit_alpha_min_spin,
            self.theory_fit_alpha_max_spin,
        )
        for widget in compact_fields:
            widget.setMinimumWidth(180)
            widget.setMaximumWidth(360)

        run_group = QGroupBox("実行")
        run_group.setMaximumWidth(1120)
        run_layout = QVBoxLayout(run_group)
        project_row = QHBoxLayout()
        self.save_settings_button = QPushButton("解析設定を保存")
        self.load_settings_button = QPushButton("解析設定を読込")
        project_row.addWidget(self.save_settings_button)
        project_row.addWidget(self.load_settings_button)
        project_row.addStretch(1)
        run_layout.addLayout(project_row)
        button_row = QHBoxLayout()
        self.run_button = QPushButton("解析実行")
        self.run_button.setProperty("variant", "primary")
        self.stop_button = QPushButton("停止")
        self.stop_button.setProperty("variant", "danger")
        self.stop_button.setEnabled(False)
        button_row.addWidget(self.run_button)
        button_row.addWidget(self.stop_button)
        button_row.addStretch(1)
        run_layout.addLayout(button_row)
        self.progress = QProgressBar()
        run_layout.addWidget(self.progress)
        run_row = QHBoxLayout()
        run_row.addStretch(1)
        run_row.addWidget(run_group, 1)
        run_row.addStretch(1)
        settings_tab_layout.addLayout(run_row)

        results_tab = QWidget()
        results_layout = QVBoxLayout(results_tab)
        self.workflow_tabs.addTab(results_tab, "結果")

        export_row = QHBoxLayout()
        self.export_manifest_button = QPushButton("解析記録を保存")
        self.export_manifest_button.setEnabled(False)
        self.export_csv_button = QPushButton("CSV出力")
        self.export_png_button = QPushButton("PNG出力")
        self.export_all_png_button = QPushButton("全ケースPNG出力")
        export_row.addStretch(1)
        export_row.addWidget(self.export_manifest_button)
        export_row.addWidget(self.export_csv_button)
        export_row.addWidget(self.export_png_button)
        export_row.addWidget(self.export_all_png_button)
        results_layout.addLayout(export_row)

        self.kpi_cards_layout = QHBoxLayout()
        self.kpi_case_value = self._add_kpi_card(self.kpi_cards_layout, "ケース", "-")
        self.kpi_volume_value = self._add_kpi_card(self.kpi_cards_layout, "最大体積", "-")
        self.kpi_evaporation_value = self._add_kpi_card(self.kpi_cards_layout, "蒸発完了時刻", "-")
        self.kpi_contact_value = self._add_kpi_card(self.kpi_cards_layout, "平均接触角", "-")
        results_layout.addLayout(self.kpi_cards_layout)

        graph_settings_group = CollapsibleGroupBox("グラフ表示設定", expanded=False)
        self.graph_settings_group = graph_settings_group
        graph_settings_layout = QVBoxLayout(graph_settings_group)
        graph_row1 = QHBoxLayout()
        self.graph_color_button = QPushButton("色")
        self.graph_point_size_spin = QDoubleSpinBox()
        self.graph_point_size_spin.setRange(1.0, 200.0)
        self.graph_point_size_spin.setDecimals(1)
        self.graph_alpha_spin = QDoubleSpinBox()
        self.graph_alpha_spin.setRange(0.05, 1.0)
        self.graph_alpha_spin.setSingleStep(0.05)
        self.graph_alpha_spin.setDecimals(2)
        self.graph_font_size_spin = QSpinBox()
        self.graph_font_size_spin.setRange(6, 40)
        self.graph_marker_combo = QComboBox()
        self.graph_marker_combo.addItems(["o", "s", "^", "D", "x", "+", "."])
        self.graph_aspect_combo = QComboBox()
        _combo_set_items(self.graph_aspect_combo, [("自動", "auto"), ("等倍", "equal")], self.language)
        self.graph_title_check = QCheckBox("タイトル")
        self.graph_axis_label_check = QCheckBox("軸ラベル")
        self.graph_tick_label_check = QCheckBox("目盛")
        self.graph_grid_check = QCheckBox("グリッド")
        graph_row1.addWidget(QLabel("点色"))
        graph_row1.addWidget(self.graph_color_button)
        graph_row1.addWidget(QLabel("点サイズ"))
        graph_row1.addWidget(self.graph_point_size_spin)
        graph_row1.addWidget(QLabel("透明度"))
        graph_row1.addWidget(self.graph_alpha_spin)
        graph_row1.addWidget(QLabel("文字"))
        graph_row1.addWidget(self.graph_font_size_spin)
        graph_row1.addWidget(QLabel("マーカー"))
        graph_row1.addWidget(self.graph_marker_combo)
        graph_row1.addWidget(QLabel("縦横"))
        graph_row1.addWidget(self.graph_aspect_combo)
        graph_row1.addWidget(self.graph_title_check)
        graph_row1.addWidget(self.graph_axis_label_check)
        graph_row1.addWidget(self.graph_tick_label_check)
        graph_row1.addWidget(self.graph_grid_check)
        graph_row1.addStretch(1)
        graph_settings_layout.addLayout(graph_row1)

        graph_row2 = QHBoxLayout()
        self.graph_axis_target_combo = QComboBox()
        self.graph_axis_target_combo.addItem(self.t("現在グラフ"), "")
        self.graph_axis_mode_combo = QComboBox()
        _combo_set_items(self.graph_axis_mode_combo, [("自動固定", "auto_fixed"), ("手動固定", "manual_fixed")], self.language)
        self.graph_x_min_spin = self._signed_scientific_spin(0.0)
        self.graph_x_max_spin = self._signed_scientific_spin(1.0)
        self.graph_y_min_spin = self._signed_scientific_spin(0.0)
        self.graph_y_max_spin = self._signed_scientific_spin(1.0)
        self.graph_x_log_check = QCheckBox("x対数")
        self.graph_y_log_check = QCheckBox("y対数")
        graph_row2.addWidget(QLabel("軸対象"))
        graph_row2.addWidget(self.graph_axis_target_combo)
        graph_row2.addWidget(QLabel("軸モード"))
        graph_row2.addWidget(self.graph_axis_mode_combo)
        graph_row2.addWidget(QLabel("x最小"))
        graph_row2.addWidget(self.graph_x_min_spin)
        graph_row2.addWidget(QLabel("x最大"))
        graph_row2.addWidget(self.graph_x_max_spin)
        graph_row2.addWidget(QLabel("y最小"))
        graph_row2.addWidget(self.graph_y_min_spin)
        graph_row2.addWidget(QLabel("y最大"))
        graph_row2.addWidget(self.graph_y_max_spin)
        graph_row2.addWidget(self.graph_x_log_check)
        graph_row2.addWidget(self.graph_y_log_check)
        graph_row2.addStretch(1)
        graph_settings_layout.addLayout(graph_row2)

        graph_row3 = QHBoxLayout()
        self.graph_width_spin = QDoubleSpinBox()
        self.graph_width_spin.setRange(1.0, 30.0)
        self.graph_width_spin.setDecimals(1)
        self.graph_height_spin = QDoubleSpinBox()
        self.graph_height_spin.setRange(1.0, 30.0)
        self.graph_height_spin.setDecimals(1)
        self.graph_quality_combo = QComboBox()
        for label, dpi in QUALITY_DPI_OPTIONS.items():
            self.graph_quality_combo.addItem(_tr(label, self.language), dpi)
        self.graph_transparent_check = QCheckBox("透明背景")
        graph_row3.addWidget(QLabel("PNG幅[in]"))
        graph_row3.addWidget(self.graph_width_spin)
        graph_row3.addWidget(QLabel("PNG高さ[in]"))
        graph_row3.addWidget(self.graph_height_spin)
        graph_row3.addWidget(QLabel("画質"))
        graph_row3.addWidget(self.graph_quality_combo)
        graph_row3.addWidget(self.graph_transparent_check)
        graph_row3.addStretch(1)
        graph_settings_layout.addLayout(graph_row3)
        results_layout.addWidget(graph_settings_group)

        self.table = ResultsTable(0, 16)
        self.table.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.table_header_sources = [
            "ケース",
            "時刻数",
            "最大体積",
            "最終体積",
            "蒸発完了時刻",
            "初期接触角",
            "最終有効接触角",
            "平均接触角",
            "初期接触半径",
            "最終有効接触半径",
            "推定alpha_e",
            "fit RMSE",
            "fit R^2",
            "fit状態",
            "状態",
            "エラー / 警告",
        ]
        self.table.setHorizontalHeaderLabels(self.table_header_sources)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionsClickable(True)
        self.table.horizontalHeader().sectionClicked.connect(self.select_table_column)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)

        self.tabs = QTabWidget()
        self.tabs.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.volume_plot = PlotWidget()
        self.radius_plot = PlotWidget()
        self.contact_angle_plot = PlotWidget()
        self.contact_radius_plot = PlotWidget()
        self.evap_plot = PlotWidget()
        self.departure_distribution_plot = PlotWidget()
        self.departure_time_height_plot = PlotWidget()
        self.visual_plot = VisualizationPlotWidget()
        self.tabs.addTab(self.volume_plot, "体積-時間")
        self.tabs.addTab(self.radius_plot, "等価半径-時間")
        self.tabs.addTab(self.contact_angle_plot, "接触角-時間")
        self.tabs.addTab(self.contact_radius_plot, "接触半径-時間")
        self.tabs.addTab(self.evap_plot, "蒸発完了時刻")
        self.tabs.addTab(self.departure_distribution_plot, "分子離脱高さ分布")
        self.tabs.addTab(self.departure_time_height_plot, "分子離脱 時刻-高さ")

        self.theory_tab = QWidget()
        theory_tab_layout = QVBoxLayout(self.theory_tab)
        theory_controls_row = QHBoxLayout()
        self.theory_show_md_check = QCheckBox("MD")
        self.theory_show_md_check.setChecked(True)
        self.theory_alpha_checks: dict[float, QCheckBox] = {}
        theory_controls_row.addWidget(self.theory_show_md_check)
        for alpha in DEFAULT_ALPHA_VALUES:
            checkbox = QCheckBox(f"alpha_e={alpha:g}")
            checkbox.setChecked(True)
            self.theory_alpha_checks[float(alpha)] = checkbox
            theory_controls_row.addWidget(checkbox)
        self.theory_show_fit_check = QCheckBox("fit推定")
        self.theory_show_fit_check.setChecked(True)
        self.theory_fit_label = QLabel("推定alpha_e: -")
        self.theory_fit_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        theory_controls_row.addWidget(self.theory_show_fit_check)
        theory_controls_row.addStretch(1)
        theory_controls_row.addWidget(self.theory_fit_label)
        theory_tab_layout.addLayout(theory_controls_row)
        self.theory_em_plot = PlotWidget()
        self.theory_radius_plot = PlotWidget()
        theory_splitter = QSplitter(Qt.Vertical)
        theory_splitter.addWidget(self.theory_em_plot)
        theory_splitter.addWidget(self.theory_radius_plot)
        theory_splitter.setStretchFactor(0, 1)
        theory_splitter.setStretchFactor(1, 1)
        theory_tab_layout.addWidget(theory_splitter, 1)
        self.tabs.addTab(self.theory_tab, "蒸発係数 / 理論比較")

        self.visual_tab = QWidget()
        visual_layout = QVBoxLayout(self.visual_tab)
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

        self.visual_settings_group = CollapsibleGroupBox("表示", expanded=False)
        visual_settings_layout = QVBoxLayout(self.visual_settings_group)
        visual_options_row = QHBoxLayout()
        self.visual_mode_combo = QComboBox()
        _combo_set_items(self.visual_mode_combo, [("2D診断", "2d"), ("3D概観", "3d")], self.language)
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
        visual_settings_layout.addLayout(visual_options_row)

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
        visual_settings_layout.addLayout(visual_export_row)
        visual_layout.addWidget(self.visual_settings_group)
        visual_layout.addWidget(self.visual_plot, 1)
        self.visual_plot.clear()
        self.tabs.addTab(self.visual_tab, "可視化")
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
        self.departure_distribution_plot.clear("分子離脱高さ分布")
        self.departure_time_height_plot.clear("分子離脱 時刻-高さ")
        self.theory_em_plot.clear("蒸発量 EM-時間")
        self.theory_radius_plot.clear("理論/MD 等価半径-時間")

        self.log_group = QGroupBox("ログ")
        log_layout = QVBoxLayout(self.log_group)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(140)
        log_layout.addWidget(self.log_box)
        self.log_group.setVisible(False)
        content_layout.addWidget(self.log_group)
        root_layout.addWidget(content, 1)
        self.graph_settings_group._sync_visibility(False)
        self.visual_settings_group._sync_visibility(False)

    def _add_kpi_card(self, layout: QHBoxLayout, label: str, value: str) -> QLabel:
        card = QFrame()
        card.setProperty("kpi", True)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(14, 10, 14, 10)
        label_widget = QLabel(label)
        label_widget.setProperty("kpiLabel", True)
        value_widget = QLabel(value)
        value_widget.setProperty("kpiValue", True)
        value_widget.setTextInteractionFlags(Qt.TextSelectableByMouse)
        card_layout.addWidget(label_widget)
        card_layout.addWidget(value_widget)
        layout.addWidget(card, 1)
        return value_widget

    def _configure_spinbox_input_behavior(self) -> None:
        for spin in self.findChildren(QSpinBox):
            spin.setKeyboardTracking(False)
        for spin in self.findChildren(QDoubleSpinBox):
            spin.setKeyboardTracking(False)

    def _connect_signals(self) -> None:
        self.workflow_nav_group.idClicked.connect(self.workflow_tabs.setCurrentIndex)
        self.workflow_tabs.currentChanged.connect(self._workflow_page_changed)
        self.log_toggle_button.toggled.connect(self.log_group.setVisible)
        self.language_combo.currentIndexChanged.connect(lambda _: self.apply_language(_combo_data(self.language_combo, "ja")))
        self.source_combo.currentIndexChanged.connect(lambda _: self._set_source_mode(_combo_data(self.source_combo, "local")))
        self.browse_button.clicked.connect(self.choose_folder)
        self.clear_local_cache_button.clicked.connect(self.clear_local_cache)
        self.refresh_button.clicked.connect(self.refresh_source)
        self.key_browse_button.clicked.connect(self.choose_private_key)
        self.connect_remote_button.clicked.connect(self.connect_remote_browser)
        self.remote_up_button.clicked.connect(self.remote_go_parent)
        self.remote_open_button.clicked.connect(self.remote_open_selected)
        self.remote_dir_list.itemDoubleClicked.connect(lambda _: self.remote_open_selected())
        self.remote_select_button.clicked.connect(self.remote_select_current)
        self.clear_cache_button.clicked.connect(self.clear_remote_cache)
        self.save_settings_button.clicked.connect(self.save_analysis_settings_file)
        self.load_settings_button.clicked.connect(self.load_analysis_settings_file)
        self.run_button.clicked.connect(self.start_analysis)
        self.stop_button.clicked.connect(self.stop_analysis)
        self.export_manifest_button.clicked.connect(self.export_analysis_manifest)
        self.export_csv_button.clicked.connect(self.export_csv)
        self.export_png_button.clicked.connect(self.export_png)
        self.export_all_png_button.clicked.connect(self.export_all_png)
        self.table.itemSelectionChanged.connect(self.update_selected_case_plots)
        self.tabs.currentChanged.connect(self.on_result_tab_changed)
        self.departure_intensity_combo.currentIndexChanged.connect(
            lambda _: (
                self.update_common_axis_ranges(),
                self.refresh_current_result_tab(),
            )
        )
        self.theory_preset_combo.currentTextChanged.connect(self.apply_theory_preset)
        for widget in (
            self.theory_show_md_check,
            self.theory_show_fit_check,
            *self.theory_alpha_checks.values(),
        ):
            widget.stateChanged.connect(lambda _: self.refresh_theory_plot_visibility())
        for widget in (
            self.theory_rho_v_spin,
            self.theory_rho_l_spin,
            self.theory_temperature_spin,
            self.theory_molecule_mass_spin,
            self.theory_v0_source_combo,
            self.theory_theta_source_combo,
            self.theory_fixed_theta_spin,
            self.theory_fit_percent_spin,
            self.theory_fit_nonzero_check,
            self.theory_fit_alpha_min_spin,
            self.theory_fit_alpha_max_spin,
        ):
            if isinstance(widget, QComboBox):
                widget.currentTextChanged.connect(lambda _: self.schedule_theory_outputs_refresh())
            elif isinstance(widget, QCheckBox):
                widget.stateChanged.connect(lambda _: self.schedule_theory_outputs_refresh())
            else:
                widget.valueChanged.connect(lambda _: self.schedule_theory_outputs_refresh())
        self.graph_color_button.clicked.connect(self.choose_graph_color)
        for widget in (
            self.graph_point_size_spin,
            self.graph_alpha_spin,
            self.graph_font_size_spin,
            self.graph_marker_combo,
            self.graph_aspect_combo,
            self.graph_title_check,
            self.graph_axis_label_check,
            self.graph_tick_label_check,
            self.graph_grid_check,
            self.graph_axis_mode_combo,
            self.graph_x_min_spin,
            self.graph_x_max_spin,
            self.graph_y_min_spin,
            self.graph_y_max_spin,
            self.graph_x_log_check,
            self.graph_y_log_check,
            self.graph_width_spin,
            self.graph_height_spin,
            self.graph_quality_combo,
            self.graph_transparent_check,
        ):
            if isinstance(widget, QComboBox):
                widget.currentTextChanged.connect(lambda _: self.on_graph_settings_changed())
            elif isinstance(widget, QCheckBox):
                widget.stateChanged.connect(lambda _: self.on_graph_settings_changed())
            else:
                widget.valueChanged.connect(lambda _: self.on_graph_settings_changed())
        self.graph_axis_target_combo.currentIndexChanged.connect(lambda _: self.load_graph_settings_from_current_plot())
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
        self.load_graph_settings_from_current_plot()
        self.refresh_theory_outputs()
        self._workflow_page_changed(self.workflow_tabs.currentIndex())

    @Slot(int)
    def _workflow_page_changed(self, index: int) -> None:
        if 0 <= index < len(self.workflow_nav_buttons):
            self.workflow_nav_buttons[index].setChecked(True)
            self.page_title_label.setText(self.workflow_tabs.tabText(index))
        subtitles = [
            "解析対象ケースを含むフォルダを選択します。",
            "解析設定",
            "結果",
        ]
        if 0 <= index < len(subtitles):
            self.page_subtitle_label.setText(self.t(subtitles[index]))

    def t(self, text: str) -> str:
        return _tr(text, self.language)

    def apply_language(self, language: str) -> None:
        if language not in LANGUAGES:
            language = "ja"
        self.language = language
        language_index = self.language_combo.findData(language)
        if language_index >= 0 and self.language_combo.currentIndex() != language_index:
            self.language_combo.blockSignals(True)
            self.language_combo.setCurrentIndex(language_index)
            self.language_combo.blockSignals(False)
        self.setWindowTitle(self.t("mdFOAM 密度解析アプリ"))
        self._translate_static_widgets(self)
        _combo_set_items(self.source_combo, [("ローカル", "local"), ("SSH", "ssh")], self.language)
        _combo_set_items(self.theory_v0_source_combo, [("最大体積", "max_volume"), ("先頭時刻体積", "first_volume")], self.language)
        _combo_set_items(self.theory_theta_source_combo, [("平均接触角", "average"), ("固定theta", "fixed")], self.language)
        _combo_set_items(self.graph_aspect_combo, [("自動", "auto"), ("等倍", "equal")], self.language)
        _combo_set_items(self.graph_axis_mode_combo, [("自動固定", "auto_fixed"), ("手動固定", "manual_fixed")], self.language)
        _combo_set_items(
            self.departure_bin_mode_combo,
            [
                ("高さ等間隔", "equal_height"),
                ("球面表面積等分", "equal_surface_area"),
            ],
            self.language,
        )
        _combo_set_items(
            self.departure_intensity_combo,
            [("イベント件数", "count"), ("面積時間あたり", "rate")],
            self.language,
        )
        _combo_set_items(self.visual_mode_combo, [("2D診断", "2d"), ("3D概観", "3d")], self.language)
        self._set_quality_combo_items(self.graph_quality_combo)
        self.table.setHorizontalHeaderLabels([self.t(header) for header in self.table_header_sources])
        self._translate_tabs(self.workflow_tabs)
        self._translate_tabs(self.tabs)
        self._workflow_page_changed(self.workflow_tabs.currentIndex())
        self.load_graph_settings_from_current_plot()
        self.update_visual_controls()
        self.refresh_current_result_tab()

    def _set_quality_combo_items(self, combo: QComboBox) -> None:
        current_dpi = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        for label, dpi in QUALITY_DPI_OPTIONS.items():
            combo.addItem(self.t(label), dpi)
        index = combo.findData(current_dpi if current_dpi is not None else QUALITY_DPI_OPTIONS[DEFAULT_QUALITY_LABEL])
        combo.setCurrentIndex(index if index >= 0 else 1)
        combo.blockSignals(False)

    def _translate_static_widgets(self, root: QWidget) -> None:
        source_keys = set()
        for translations in TRANSLATIONS.values():
            source_keys.update(translations.keys())
        widgets = [
            *root.findChildren(QLabel),
            *root.findChildren(QPushButton),
            *root.findChildren(QCheckBox),
            *root.findChildren(QGroupBox),
        ]
        for widget in widgets:
            if isinstance(widget, QLabel):
                text = widget.text()
            else:
                text = widget.title() if isinstance(widget, QGroupBox) else widget.text()
            source = widget.property("i18n_source_text")
            if source is None and text in source_keys:
                source = text
                widget.setProperty("i18n_source_text", source)
            if not source:
                continue
            translated = self.t(str(source))
            if isinstance(widget, QGroupBox):
                widget.setTitle(translated)
            else:
                widget.setText(translated)

    def _translate_tabs(self, tabs: QTabWidget) -> None:
        source_keys = set()
        for translations in TRANSLATIONS.values():
            source_keys.update(translations.keys())
        for index in range(tabs.count()):
            source = tabs.tabBar().tabData(index)
            if source is None:
                text = tabs.tabText(index)
                if text not in source_keys:
                    continue
                source = text
                tabs.tabBar().setTabData(index, source)
            tabs.setTabText(index, self.t(str(source)))

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
        spin.setDecimals(323)
        spin.setRange(0.0, 1.0e100)
        spin.setSingleStep(1.0)
        spin.setValue(value)
        return spin

    def _signed_scientific_spin(self, value: float) -> QDoubleSpinBox:
        spin = SignedScientificDoubleSpinBox()
        spin.setRange(-1.0e100, 1.0e100)
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

    def _clear_results(self) -> None:
        self.results.clear()
        self._last_run_settings = None
        self._last_run_context = None
        if hasattr(self, "export_manifest_button"):
            self.export_manifest_button.setEnabled(False)
        self._theory_comparison_cache.clear()
        self._auto_axis_ranges.clear()
        self.table.setRowCount(0)
        self._update_kpi_cards(None)

    def _set_source_mode(self, mode: str) -> None:
        is_remote = mode == "ssh"
        self.source_stack.setCurrentIndex(1 if is_remote else 0)
        self.source_stack.setMaximumHeight(16_777_215 if is_remote else 150)
        self.input_layout.setStretch(1, 2 if is_remote else 0)
        self.input_layout.setStretch(2, 1)
        if is_remote:
            self.folder_edit.setText(self.remote_path_edit.text())
            if self.loaded_source != "SSH":
                self._clear_loaded_cases()
        else:
            self.folder_edit.setText(str(self.local_folder_path))
            if self.loaded_source not in ("", "local"):
                self.load_folder(self.local_folder_path)

    def _clear_loaded_cases(self) -> None:
        self.case_list.clear()
        self.field_combo.clear()
        self.field_combo.addItem("rhoM_water")
        self._clear_results()
        self.cases = []
        self.remote_cases = []
        self.loaded_source = ""
        self.update_visual_controls(None)
        self.update_theory_plots(None)

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
        if _combo_data(self.source_combo, "local") == "ssh":
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
    def clear_local_cache(self) -> None:
        try:
            clear_local_analysis_cache()
            self.log("ローカル解析キャッシュを削除しました。")
        except Exception as exc:
            self.log(f"ローカル解析キャッシュの削除に失敗しました: {exc}")

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
        self._clear_results()
        self.update_visual_controls(None)
        self.remote_cases = []
        self.loaded_source = "local"
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
        self._clear_results()
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
        if _combo_data(self.source_combo, "local") == "ssh":
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
            contact_average_percent=self.contact_average_percent_spin.value(),
            departure_enabled=self.departure_enabled_check.isChecked(),
            departure_species=(
                self.departure_species_edit.text().strip() or "water"
            ),
            departure_cutoff=self.departure_cutoff_nm_spin.value() / 1.0e9,
            departure_confirmation_frames=self.departure_confirmation_spin.value(),
            departure_height_bins=self.departure_height_bins_spin.value(),
            departure_bin_mode=_combo_data(
                self.departure_bin_mode_combo,
                "equal_height",
            ),
        )

    def apply_analysis_settings(self, settings: AnalysisSettings) -> None:
        field_index = self.field_combo.findText(settings.density_field)
        if field_index < 0:
            self.field_combo.addItem(settings.density_field)
            field_index = self.field_combo.findText(settings.density_field)
        self.field_combo.setCurrentIndex(field_index)
        self.threshold_spin.setValue(settings.density_threshold)
        self.zero_spin.setValue(settings.zero_tolerance)
        self.zero_count_spin.setValue(settings.consecutive_zero_count)
        self.cell_volume_spin.setValue(settings.manual_cell_volume or 0.0)
        self.dx_spin.setValue(settings.dx or 0.0)
        self.dy_spin.setValue(settings.dy or 0.0)
        self.dz_spin.setValue(settings.dz or 0.0)
        self.contact_fit_lower_spin.setValue(settings.contact_fit_lower)
        self.contact_fit_upper_spin.setValue(settings.contact_fit_upper)
        self.contact_unwrap_check.setChecked(settings.contact_unwrap_xy)
        self.contact_average_percent_spin.setValue(
            settings.contact_average_percent
        )
        self.departure_enabled_check.setChecked(settings.departure_enabled)
        self.departure_species_edit.setText(settings.departure_species)
        self.departure_cutoff_nm_spin.setValue(
            settings.departure_cutoff / 1.0e-9
        )
        self.departure_confirmation_spin.setValue(
            settings.departure_confirmation_frames
        )
        self.departure_height_bins_spin.setValue(
            settings.departure_height_bins
        )
        bin_mode_index = self.departure_bin_mode_combo.findData(
            settings.departure_bin_mode
        )
        self.departure_bin_mode_combo.setCurrentIndex(
            bin_mode_index if bin_mode_index >= 0 else 0
        )

    @Slot()
    def save_analysis_settings_file(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            self.t("解析設定を保存"),
            str(Path(self._local_dialog_start_dir()) / "mdfoam_project.json"),
            "JSON (*.json)",
        )
        if not path:
            return
        output_path = _ensure_suffix(Path(path), ".json")
        try:
            save_analysis_settings(output_path, self.settings())
        except ProvenanceError as exc:
            QMessageBox.warning(self, self.t("解析設定を保存"), str(exc))
            return
        self.log(f"{self.t('解析設定を保存しました')}: {output_path}")

    @Slot()
    def load_analysis_settings_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            self.t("解析設定を読込"),
            self._local_dialog_start_dir(),
            "JSON (*.json)",
        )
        if not path:
            return
        try:
            loaded = load_analysis_settings(Path(path))
        except ProvenanceError as exc:
            QMessageBox.warning(self, self.t("解析設定を読込"), str(exc))
            return
        self.apply_analysis_settings(loaded)
        self.log(f"{self.t('解析設定を読み込みました')}: {path}")

    def theory_settings(self) -> TheorySettings:
        v0_source = _combo_data(self.theory_v0_source_combo, "max_volume")
        theta_source = _combo_data(self.theory_theta_source_combo, "average")
        return TheorySettings(
            rho_v=self.theory_rho_v_spin.value(),
            rho_l=self.theory_rho_l_spin.value(),
            temperature=self.theory_temperature_spin.value(),
            molecule_mass=self.theory_molecule_mass_spin.value(),
            v0_source=v0_source,
            theta_source=theta_source,
            fixed_theta_deg=self.theory_fixed_theta_spin.value(),
            fit_percent=self.theory_fit_percent_spin.value(),
            fit_nonzero_only=self.theory_fit_nonzero_check.isChecked(),
            fit_alpha_min=self.theory_fit_alpha_min_spin.value(),
            fit_alpha_max=self.theory_fit_alpha_max_spin.value(),
        )

    @Slot(str)
    def apply_theory_preset(self, name: str) -> None:
        preset = THEORY_PRESETS.get(name)
        if preset is None:
            return
        widgets = (
            self.theory_rho_v_spin,
            self.theory_rho_l_spin,
            self.theory_temperature_spin,
            self.theory_molecule_mass_spin,
        )
        for widget in widgets:
            widget.blockSignals(True)
        self.theory_rho_v_spin.setValue(preset.rho_v)
        self.theory_rho_l_spin.setValue(preset.rho_l)
        self.theory_temperature_spin.setValue(preset.temperature)
        self.theory_molecule_mass_spin.setValue(preset.molecule_mass)
        for widget in widgets:
            widget.blockSignals(False)
        self.refresh_theory_outputs()

    def update_theory_control_state(self) -> None:
        self.theory_fixed_theta_spin.setEnabled(_combo_data(self.theory_theta_source_combo, "average") == "fixed")

    def update_theory_diagnostics(self, result: CaseResult | None = None) -> None:
        result = result or self.current_result()
        settings = self.theory_settings()
        try:
            flux = evaporation_flux(settings, 1.0)
        except ValueError as exc:
            self.theory_diagnostics_label.setText(str(exc))
            return

        parts = [f"J(alpha_e=1): {flux:.6g} kg/(m2*s)"]
        theta = self._diagnostic_theta(result, settings)
        if theta is not None:
            try:
                ratio = height_to_contact_radius_ratio(theta)
                parts.append(f"theta: {theta:.6g} deg")
                parts.append(f"h/r: {ratio:.6g}")
            except ValueError as exc:
                parts.append(str(exc))
        if result is not None and result.rows and theta is not None:
            v0 = self._diagnostic_v0(result, settings)
            try:
                geometry = spherical_cap_geometry(v0, theta)
                parts.append(f"V0: {v0:.6g} m^3")
                parts.append(f"S0: {geometry.surface_area:.6g} m^2")
            except ValueError as exc:
                parts.append(str(exc))
        self.theory_diagnostics_label.setText(" / ".join(parts))

    def _diagnostic_theta(self, result: CaseResult | None, settings: TheorySettings) -> float | None:
        if settings.theta_source == "fixed":
            return settings.fixed_theta_deg
        if result is None:
            return None
        return result.average_contact_angle_deg

    def _diagnostic_v0(self, result: CaseResult, settings: TheorySettings) -> float:
        if settings.v0_source == "first_volume":
            return result.rows[0].volume if result.rows else 0.0
        return result.max_volume

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
        if _combo_data(self.source_combo, "local") == "ssh":
            try:
                remote_profile = self._remote_profile()
                self._save_ssh_profile()
            except Exception as exc:
                QMessageBox.warning(self, "SSH設定エラー", str(exc))
                return

        self._clear_results()
        run_settings = self.settings()
        if remote_profile is None:
            run_context = RunContext(
                input_mode="local",
                selected_root=str(self.local_folder_path.resolve()),
                analysis_settings=run_settings,
            )
        else:
            run_context = RunContext(
                input_mode="ssh",
                selected_root=normalize_remote_path(self.remote_path_edit.text()),
                analysis_settings=run_settings,
                remote_host=remote_profile.host,
                remote_port=remote_profile.port,
                remote_username=remote_profile.username,
            )
        self._last_run_settings = run_settings
        self._last_run_context = run_context
        self.update_visual_controls(None)
        self.update_theory_plots(None)
        self.progress.setRange(0, len(cases))
        self.progress.setValue(0)
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.log("解析を開始しました。")
        self.workflow_tabs.setCurrentIndex(2)

        self.thread = QThread(self)
        self.worker = AnalyzerWorker(cases, run_settings, remote_profile)
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

    def current_plot_widget(self) -> PlotWidget | None:
        tab = self.tabs.currentWidget()
        if isinstance(tab, PlotWidget):
            return tab
        if tab is self.theory_tab:
            target = self.graph_axis_target_combo.currentData() if hasattr(self, "graph_axis_target_combo") else ""
            return self.theory_radius_plot if target == "theory_equivalent_radius" else self.theory_em_plot
        return None

    @Slot()
    def choose_graph_color(self) -> None:
        plot = self.current_plot_widget()
        if plot is None:
            return
        color = QColorDialog.getColor(QColor(plot.settings.point_color), self, "点色")
        if not color.isValid():
            return
        plot.settings.point_color = color.name()
        self._set_graph_color_button(plot.settings.point_color)
        plot.redraw()

    @Slot()
    def on_result_tab_changed(self) -> None:
        self.refresh_current_result_tab()
        self.load_graph_settings_from_current_plot()

    def load_graph_settings_from_current_plot(self) -> None:
        is_theory_tab = self.tabs.currentWidget() is self.theory_tab
        current_target = self.graph_axis_target_combo.currentData()
        self.graph_axis_target_combo.blockSignals(True)
        self.graph_axis_target_combo.clear()
        if is_theory_tab:
            self.graph_axis_target_combo.addItem(self.t("蒸発量 EM"), "theory_evaporated_mass")
            self.graph_axis_target_combo.addItem(self.t("理論/MD 等価半径"), "theory_equivalent_radius")
            index = self.graph_axis_target_combo.findData(current_target)
            self.graph_axis_target_combo.setCurrentIndex(index if index >= 0 else 0)
            self.graph_axis_target_combo.setEnabled(True)
        else:
            self.graph_axis_target_combo.addItem(self.t("現在グラフ"), "")
            self.graph_axis_target_combo.setEnabled(False)
        self.graph_axis_target_combo.blockSignals(False)
        plot = self.current_plot_widget()
        enabled = plot is not None
        for widget in (
            self.graph_color_button,
            self.graph_point_size_spin,
            self.graph_alpha_spin,
            self.graph_font_size_spin,
            self.graph_marker_combo,
            self.graph_aspect_combo,
            self.graph_title_check,
            self.graph_axis_label_check,
            self.graph_tick_label_check,
            self.graph_grid_check,
            self.graph_axis_mode_combo,
            self.graph_x_min_spin,
            self.graph_x_max_spin,
            self.graph_y_min_spin,
            self.graph_y_max_spin,
            self.graph_x_log_check,
            self.graph_y_log_check,
            self.graph_width_spin,
            self.graph_height_spin,
            self.graph_quality_combo,
            self.graph_transparent_check,
        ):
            widget.blockSignals(True)
            widget.setEnabled(enabled)
        if plot is not None:
            settings = plot.settings
            self._set_graph_color_button(settings.point_color)
            self.graph_point_size_spin.setValue(settings.point_size)
            self.graph_alpha_spin.setValue(settings.point_alpha)
            self.graph_font_size_spin.setValue(settings.font_size)
            self.graph_marker_combo.setCurrentText(settings.marker)
            aspect_index = self.graph_aspect_combo.findData(settings.aspect)
            self.graph_aspect_combo.setCurrentIndex(aspect_index if aspect_index >= 0 else 0)
            self.graph_title_check.setChecked(settings.title_visible)
            self.graph_axis_label_check.setChecked(settings.axis_labels_visible)
            self.graph_tick_label_check.setChecked(settings.tick_labels_visible)
            self.graph_grid_check.setChecked(settings.grid_visible)
            axis_mode_index = self.graph_axis_mode_combo.findData(settings.axis_mode)
            self.graph_axis_mode_combo.setCurrentIndex(axis_mode_index if axis_mode_index >= 0 else 0)
            self.graph_x_min_spin.setValue(settings.x_min)
            self.graph_x_max_spin.setValue(settings.x_max)
            self.graph_y_min_spin.setValue(settings.y_min)
            self.graph_y_max_spin.setValue(settings.y_max)
            self.graph_x_log_check.setChecked(settings.x_log)
            self.graph_y_log_check.setChecked(settings.y_log)
            self.graph_width_spin.setValue(settings.image_width)
            self.graph_height_spin.setValue(settings.image_height)
            quality_index = self.graph_quality_combo.findData(settings.dpi)
            self.graph_quality_combo.setCurrentIndex(quality_index if quality_index >= 0 else 1)
            self.graph_transparent_check.setChecked(settings.transparent)
        for widget in (
            self.graph_color_button,
            self.graph_point_size_spin,
            self.graph_alpha_spin,
            self.graph_font_size_spin,
            self.graph_marker_combo,
            self.graph_aspect_combo,
            self.graph_title_check,
            self.graph_axis_label_check,
            self.graph_tick_label_check,
            self.graph_grid_check,
            self.graph_axis_mode_combo,
            self.graph_x_min_spin,
            self.graph_x_max_spin,
            self.graph_y_min_spin,
            self.graph_y_max_spin,
            self.graph_x_log_check,
            self.graph_y_log_check,
            self.graph_width_spin,
            self.graph_height_spin,
            self.graph_quality_combo,
            self.graph_transparent_check,
        ):
            widget.blockSignals(False)
        self.update_axis_spin_enabled()

    def on_graph_settings_changed(self) -> None:
        plot = self.current_plot_widget()
        if plot is None:
            return
        settings = plot.settings
        settings.point_size = self.graph_point_size_spin.value()
        settings.point_alpha = self.graph_alpha_spin.value()
        settings.font_size = self.graph_font_size_spin.value()
        settings.marker = self.graph_marker_combo.currentText()
        settings.aspect = _combo_data(self.graph_aspect_combo, "auto")
        settings.title_visible = self.graph_title_check.isChecked()
        settings.axis_labels_visible = self.graph_axis_label_check.isChecked()
        settings.tick_labels_visible = self.graph_tick_label_check.isChecked()
        settings.grid_visible = self.graph_grid_check.isChecked()
        settings.axis_mode = _combo_data(self.graph_axis_mode_combo, "auto_fixed")
        settings.axis_auto = settings.axis_mode != "manual_fixed"
        settings.x_min = self.graph_x_min_spin.value()
        settings.x_max = self.graph_x_max_spin.value()
        settings.y_min = self.graph_y_min_spin.value()
        settings.y_max = self.graph_y_max_spin.value()
        settings.x_log = self.graph_x_log_check.isChecked()
        settings.y_log = self.graph_y_log_check.isChecked()
        settings.image_width = self.graph_width_spin.value()
        settings.image_height = self.graph_height_spin.value()
        settings.dpi = int(self.graph_quality_combo.currentData() or QUALITY_DPI_OPTIONS[DEFAULT_QUALITY_LABEL])
        settings.transparent = self.graph_transparent_check.isChecked()
        self.update_axis_spin_enabled()
        kind = self._current_graph_kind()
        if kind is not None:
            self._apply_axis_settings_for_kind(plot, kind)
        plot.redraw()

    def update_axis_spin_enabled(self) -> None:
        plot = self.current_plot_widget()
        plot_kind = plot._last_plot[0] if plot is not None and plot._last_plot is not None else "xy"
        has_plot = plot is not None
        x_axis_available = has_plot and plot_kind in ("xy", "series")
        manual_axis = has_plot and _combo_data(self.graph_axis_mode_combo, "auto_fixed") == "manual_fixed"
        self.graph_x_min_spin.setEnabled(manual_axis and x_axis_available)
        self.graph_x_max_spin.setEnabled(manual_axis and x_axis_available)
        self.graph_y_min_spin.setEnabled(manual_axis)
        self.graph_y_max_spin.setEnabled(manual_axis)
        self.graph_x_log_check.setEnabled(x_axis_available)
        self.graph_y_log_check.setEnabled(has_plot)

    def _set_graph_color_button(self, color: str) -> None:
        self.graph_color_button.setStyleSheet(f"background-color: {color};")

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
        self.export_manifest_button.setEnabled(True)
        self.add_result_row(result)
        self.update_common_axis_ranges()
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

    def _theory_comparison(self, result: CaseResult) -> TheoryComparison:
        key = (id(result), self.theory_settings())
        comparison = self._theory_comparison_cache.get(key)
        if comparison is None:
            if len(self._theory_comparison_cache) > 512:
                self._theory_comparison_cache.clear()
            comparison = build_theory_comparison(result, key[1], DEFAULT_ALPHA_VALUES)
            self._theory_comparison_cache[key] = comparison
        return comparison

    def add_result_row(self, result: CaseResult) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        comparison = self._theory_comparison(result)
        messages = ([result.error] if result.error else []) + list(result.warnings)
        values = [
            result.case_name,
            str(result.time_count),
            _fmt(result.max_volume),
            _fmt(result.final_volume),
            "" if result.evaporation_time is None else _fmt(result.evaporation_time),
            _fmt_optional(result.initial_contact_angle_deg),
            _fmt_optional(result.final_valid_contact_angle_deg),
            _fmt_optional(result.average_contact_angle_deg),
            _fmt_optional(result.initial_contact_radius),
            _fmt_optional(result.final_valid_contact_radius),
            _fmt_optional(comparison.fit.alpha_e),
            _fmt_optional(comparison.fit.rmse),
            _fmt_optional(comparison.fit.r2),
            comparison.fit.status,
            _status_label(result.status, bool(result.warnings)),
            " | ".join(messages),
        ]
        for column, value in enumerate(values):
            item = QTableWidgetItem(value)
            item.setData(Qt.UserRole, len(self.results) - 1)
            self.table.setItem(row, column, item)

    @Slot()
    def update_selected_case_plots(self) -> None:
        self.refresh_current_result_tab()

    def refresh_current_result_tab(self) -> None:
        result = self.current_result()
        self._update_kpi_cards(result)
        tab = self.tabs.currentWidget()
        if tab is self.volume_plot:
            if result is None:
                self.volume_plot.clear("体積-時間")
            else:
                self._plot_standard_result_for_export(self.volume_plot, result, "volume_time")
        elif tab is self.radius_plot:
            if result is None:
                self.radius_plot.clear("等価半径-時間")
            else:
                self._plot_standard_result_for_export(self.radius_plot, result, "equivalent_radius_time")
        elif tab is self.contact_angle_plot:
            if result is None:
                self.contact_angle_plot.clear("接触角-時間")
            else:
                self._plot_standard_result_for_export(self.contact_angle_plot, result, "contact_angle_time")
        elif tab is self.contact_radius_plot:
            if result is None:
                self.contact_radius_plot.clear("接触半径-時間")
            else:
                self._plot_standard_result_for_export(self.contact_radius_plot, result, "contact_radius_time")
        elif tab is self.evap_plot:
            self.update_evap_plot()
        elif tab is self.departure_distribution_plot:
            if result is None:
                self.departure_distribution_plot.clear("分子離脱高さ分布")
            else:
                self._plot_departure_result(
                    self.departure_distribution_plot,
                    result,
                    "departure_height_distribution",
                )
        elif tab is self.departure_time_height_plot:
            if result is None:
                self.departure_time_height_plot.clear("分子離脱 時刻-高さ")
            else:
                self._plot_departure_result(
                    self.departure_time_height_plot,
                    result,
                    "departure_time_height",
                )
        elif tab is self.theory_tab:
            self.update_theory_plots(result)
        elif tab is self.visual_tab:
            self.update_visual_controls(result)

    def _update_kpi_cards(self, result: CaseResult | None) -> None:
        if result is None:
            values = ("-", "-", "-", "-")
        else:
            values = (
                result.case_name,
                _fmt(result.max_volume),
                _fmt_optional(result.evaporation_time),
                _fmt_optional(result.average_contact_angle_deg),
            )
        for label, value in zip(
            (
                self.kpi_case_value,
                self.kpi_volume_value,
                self.kpi_evaporation_value,
                self.kpi_contact_value,
            ),
            values,
        ):
            label.setText(value or "-")

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

    @Slot()
    def schedule_theory_outputs_refresh(self) -> None:
        self._theory_refresh_timer.start()

    @Slot()
    def refresh_theory_plot_visibility(self) -> None:
        self.update_common_axis_ranges()
        if self.tabs.currentWidget() is self.theory_tab:
            self.update_theory_plots()

    @Slot()
    def refresh_theory_outputs(self) -> None:
        self._theory_refresh_timer.stop()
        self.update_theory_control_state()
        self.update_theory_diagnostics()
        self.update_theory_table_columns()
        self.update_common_axis_ranges()
        if self.tabs.currentWidget() is self.theory_tab:
            self.update_theory_plots()

    def update_theory_table_columns(self) -> None:
        if not self.results:
            return
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item is None:
                continue
            result_index = item.data(Qt.UserRole)
            if result_index is None or result_index >= len(self.results):
                continue
            comparison = self._theory_comparison(self.results[result_index])
            values = [
                _fmt_optional(comparison.fit.alpha_e),
                _fmt_optional(comparison.fit.rmse),
                _fmt_optional(comparison.fit.r2),
                comparison.fit.status,
            ]
            for column, value in zip((10, 11, 12, 13), values):
                table_item = self.table.item(row, column)
                if table_item is None:
                    table_item = QTableWidgetItem(value)
                    self.table.setItem(row, column, table_item)
                else:
                    table_item.setText(value)
                table_item.setData(Qt.UserRole, result_index)

    def update_theory_plots(self, result: CaseResult | None = None) -> None:
        result = result or self.current_result()
        self.update_theory_diagnostics(result)
        if result is None or not result.rows:
            self.theory_fit_label.setText("推定alpha_e: -")
            self.theory_em_plot.clear("蒸発量 EM-時間")
            self.theory_radius_plot.clear("理論/MD 等価半径-時間")
            return

        comparison = self._theory_comparison(result)
        self._update_theory_fit_label(comparison)
        if comparison.status != "ok":
            self.theory_em_plot.clear(f"{result.case_name}: {comparison.status}")
            self.theory_radius_plot.clear(f"{result.case_name}: {comparison.status}")
            return

        em_series: list[PlotSeries] = []
        radius_series: list[PlotSeries] = []
        if self.theory_show_md_check.isChecked():
            em_times, md_evaporated_masses = _clip_xy_to_evaporation(
                result,
                comparison.times,
                comparison.md_evaporated_masses,
            )
            radius_times, md_equivalent_radii = _clip_xy_to_evaporation(
                result,
                comparison.times,
                comparison.md_equivalent_radii,
            )
            em_series.append(
                PlotSeries(
                    "MD",
                    em_times,
                    md_evaporated_masses,
                    style="scatter",
                    color=COLORS["md_series"],
                    marker="o",
                )
            )
            radius_series.append(
                PlotSeries(
                    "MD",
                    radius_times,
                    md_equivalent_radii,
                    style="scatter",
                    color=COLORS["md_series"],
                    marker="o",
                )
            )

        alpha_colors = {
            0.8: "#1f77b4",
            0.9: "#ff7f0e",
            1.0: "#2ca02c",
        }
        for alpha, checkbox in self.theory_alpha_checks.items():
            if not checkbox.isChecked():
                continue
            curve = comparison.curves.get(alpha)
            if curve is None:
                continue
            label = f"alpha_e={alpha:g}"
            color = alpha_colors.get(alpha)
            em_times, evaporated_masses = _clip_xy_to_evaporation(result, curve.times, curve.evaporated_masses)
            radius_times, equivalent_radii = _clip_xy_to_evaporation(result, curve.times, curve.equivalent_radii)
            em_series.append(
                PlotSeries(label, em_times, evaporated_masses, style="line", color=color)
            )
            radius_series.append(
                PlotSeries(label, radius_times, equivalent_radii, style="line", color=color)
            )

        if self.theory_show_fit_check.isChecked() and comparison.fit_curve is not None:
            fit_label = "fit"
            if comparison.fit.alpha_e is not None:
                fit_label = f"fit alpha_e={comparison.fit.alpha_e:.4g}"
                if comparison.fit.boundary:
                    fit_label += f" ({comparison.fit.boundary})"
            em_times, evaporated_masses = _clip_xy_to_evaporation(
                result,
                comparison.fit_curve.times,
                comparison.fit_curve.evaporated_masses,
            )
            radius_times, equivalent_radii = _clip_xy_to_evaporation(
                result,
                comparison.fit_curve.times,
                comparison.fit_curve.equivalent_radii,
            )
            em_series.append(
                PlotSeries(
                    fit_label,
                    em_times,
                    evaporated_masses,
                    style="line",
                    color="#d62728",
                    linestyle="--",
                    linewidth=1.8,
                )
            )
            radius_series.append(
                PlotSeries(
                    fit_label,
                    radius_times,
                    equivalent_radii,
                    style="line",
                    color="#d62728",
                    linestyle="--",
                    linewidth=1.8,
                )
            )

        self._apply_axis_settings_for_kind(self.theory_em_plot, "theory_evaporated_mass")
        self._apply_axis_settings_for_kind(self.theory_radius_plot, "theory_equivalent_radius")
        self.theory_em_plot.plot_series(
            f"{result.case_name}: 蒸発量 EM-時間",
            "時間 [s]",
            "蒸発量 EM [kg]",
            em_series,
        )
        self.theory_radius_plot.plot_series(
            f"{result.case_name}: 理論/MD 等価半径-時間",
            "時間 [s]",
            "等価半径 [m]",
            radius_series,
        )

    def _update_theory_fit_label(self, comparison: TheoryComparison) -> None:
        if comparison.status != "ok":
            self.theory_fit_label.setText(f"理論比較: {comparison.status}")
            return
        fit = comparison.fit
        if fit.alpha_e is None:
            self.theory_fit_label.setText(
                f"V0: {_fmt_optional(comparison.v0)} / theta: {_fmt_optional(comparison.theta_deg)} / {fit.status}"
            )
            return
        self.theory_fit_label.setText(
            " / ".join(
                [
                    f"V0: {_fmt_optional(comparison.v0)}",
                    f"theta: {_fmt_optional(comparison.theta_deg)} deg",
                    f"alpha_e: {_fmt(fit.alpha_e)}",
                    f"RMSE: {_fmt_optional(fit.rmse)}",
                    f"R^2: {_fmt_optional(fit.r2)}",
                    f"fit点: {fit.point_count}",
                    f"fit状態: {fit.status}",
                ]
            )
        )

    def update_visual_controls(self, result: CaseResult | None = None) -> None:
        result = result or self.current_result()
        if result is None or not result.rows:
            self.visual_case_label.setText(self.t("ケース: -"))
            self.visual_time_label.setText(self.t("時刻: -"))
            self.visual_range_label.setText(self.t("GIF範囲: -"))
            self.visual_time_slider.setEnabled(False)
            self.visual_range_start_slider.setEnabled(False)
            self.visual_range_end_slider.setEnabled(False)
            self.visual_plot.clear(self.t("可視化するケースと時刻を選択してください"))
            return

        count = len(result.rows)
        self.visual_case_label.setText(f"{self.t('ケース')}: {result.case_name}")
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
            self.visual_range_label.setText(self.t("GIF範囲: -"))
            return
        start, end = self._visual_range_indices()
        self.visual_range_label.setText(
            f"{self.t('GIF範囲')}: {result.rows[start].time:.4g} - {result.rows[end].time:.4g}"
        )

    def refresh_visualization(self) -> None:
        result = self.current_result()
        if result is None or not result.rows or not self.visual_time_slider.isEnabled():
            return
        self._apply_visual_defaults()
        index = max(0, min(self.visual_time_slider.value(), len(result.rows) - 1))
        row = result.rows[index]
        self.visual_time_label.setText(f"{self.t('時刻')}: {row.time:.8g}")
        try:
            frame = self._load_visual_frame(result, row.time)
            self._draw_visual_frame(frame)
        except Exception as exc:
            self.visual_plot.clear(f"可視化データを読み込めません: {exc}")
            self.log(f"可視化データを読み込めません: {exc}")

    def _apply_visual_defaults(self) -> None:
        if _combo_data(self.visual_mode_combo, "2d") == "3d" and self.visual_periodic_check.isChecked():
            if self.visual_max_points_spin.value() == 0:
                self.visual_max_points_spin.blockSignals(True)
                self.visual_max_points_spin.setSpecialValueText(f"自動({THREE_D_AUTO_MAX_POINTS})")
                self.visual_max_points_spin.blockSignals(False)

    def _draw_visual_frame(self, frame: VisualizationFrame) -> None:
        downsample = self.visual_plot.draw_frame(
            frame,
            _combo_data(self.visual_mode_combo, "2d"),
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

    def _current_graph_kind(self) -> str | None:
        tab = self.tabs.currentWidget()
        if tab is self.volume_plot:
            return "volume_time"
        if tab is self.radius_plot:
            return "equivalent_radius_time"
        if tab is self.contact_angle_plot:
            return "contact_angle_time"
        if tab is self.contact_radius_plot:
            return "contact_radius_time"
        if tab is self.evap_plot:
            return "evaporation_time_all_cases"
        if tab is self.departure_distribution_plot:
            return "departure_height_distribution"
        if tab is self.departure_time_height_plot:
            return "departure_time_height"
        if tab is self.theory_tab:
            target = self.graph_axis_target_combo.currentData() if hasattr(self, "graph_axis_target_combo") else ""
            return "theory_equivalent_radius" if target == "theory_equivalent_radius" else "theory_evaporated_mass"
        return None

    def update_common_axis_ranges(self) -> None:
        ranges: dict[str, tuple[float, float, float, float]] = {}
        for kind in (
            "volume_time",
            "equivalent_radius_time",
            "contact_angle_time",
            "contact_radius_time",
            "evaporation_time_all_cases",
            "departure_height_distribution",
            "departure_time_height",
            "theory_evaporated_mass",
            "theory_equivalent_radius",
        ):
            axis_range = self._auto_axis_range_for_kind(kind)
            if axis_range is not None:
                ranges[kind] = axis_range
        self._auto_axis_ranges = ranges

    def _auto_axis_range_for_kind(self, kind: str) -> tuple[float, float, float, float] | None:
        x_values: list[float] = []
        y_values: list[float] = []
        if kind in ("volume_time", "equivalent_radius_time", "contact_angle_time", "contact_radius_time"):
            for result in self.results:
                rows = _rows_until_evaporation(result)
                for row in rows:
                    if kind == "volume_time":
                        x_values.append(row.time)
                        y_values.append(row.volume)
                    elif kind == "equivalent_radius_time":
                        x_values.append(row.time)
                        y_values.append(row.equivalent_radius)
                    elif kind == "contact_angle_time" and row.contact_angle_deg is not None:
                        x_values.append(row.time)
                        y_values.append(row.contact_angle_deg)
                    elif kind == "contact_radius_time" and row.contact_radius is not None:
                        x_values.append(row.time)
                        y_values.append(row.contact_radius)
        elif kind == "evaporation_time_all_cases":
            y_values = [result.evaporation_time for result in self.results if result.evaporation_time is not None]
            x_values = list(range(len(y_values)))
        elif kind in ("departure_height_distribution", "departure_time_height"):
            for result in self.results:
                departure = result.departure_result
                if departure is None or departure.status != "ok":
                    continue
                if kind == "departure_height_distribution":
                    rate_mode = (
                        _combo_data(self.departure_intensity_combo, "count")
                        == "rate"
                    )
                    for item in departure.height_bins:
                        center = (item.eta_lower + item.eta_upper) / 2.0
                        values = (
                            (item.raw_rate, item.confirmed_rate)
                            if rate_mode
                            else (
                                float(item.raw_count),
                                float(item.confirmed_count),
                            )
                        )
                        for value in values:
                            if value is not None:
                                x_values.append(center)
                                y_values.append(value)
                else:
                    for event in departure.events:
                        eta = event.normalized_height
                        if eta is not None and 0.0 <= eta <= 1.0:
                            x_values.append(event.event_time)
                            y_values.append(eta)
        elif kind in ("theory_evaporated_mass", "theory_equivalent_radius"):
            value_kind = "em" if kind == "theory_evaporated_mass" else "radius"
            for result in self.results:
                comparison = self._theory_comparison(result)
                if comparison.status != "ok":
                    continue
                for series in self._theory_export_series(result, comparison, value_kind):
                    x_values.extend(series.x)
                    y_values.extend(series.y)
        return _padded_axis_range(x_values, y_values)

    def _apply_axis_settings_for_kind(self, plot: PlotWidget, kind: str) -> None:
        if plot.settings.axis_mode == "auto_fixed":
            axis_range = self._auto_axis_ranges.get(kind)
            if axis_range is not None:
                plot.settings.x_min, plot.settings.x_max, plot.settings.y_min, plot.settings.y_max = axis_range
            plot.settings.axis_auto = False
        elif plot.settings.axis_mode == "manual_fixed":
            plot.settings.axis_auto = False

    def _current_png_context(self) -> tuple[str, PlotWidget] | None:
        tab = self.tabs.currentWidget()
        if tab is self.volume_plot:
            return "volume_time", self.volume_plot
        if tab is self.radius_plot:
            return "equivalent_radius_time", self.radius_plot
        if tab is self.contact_angle_plot:
            return "contact_angle_time", self.contact_angle_plot
        if tab is self.contact_radius_plot:
            return "contact_radius_time", self.contact_radius_plot
        if tab is self.evap_plot:
            return "evaporation_time_all_cases", self.evap_plot
        if tab is self.departure_distribution_plot:
            return "departure_height_distribution", self.departure_distribution_plot
        if tab is self.departure_time_height_plot:
            return "departure_time_height", self.departure_time_height_plot
        return None

    def _theory_png_options(self, result: CaseResult | None = None) -> list[tuple[str, PlotWidget | list[PlotWidget], str]]:
        return [
            (self.t("蒸発量 EM"), self.theory_em_plot, self._suggested_png_filename("theory_evaporated_mass", result)),
            (self.t("理論/MD 等価半径"), self.theory_radius_plot, self._suggested_png_filename("theory_equivalent_radius", result)),
            (self.t("上下2枚"), [self.theory_em_plot, self.theory_radius_plot], self._suggested_png_filename("theory_combined", result)),
        ]

    def _suggested_png_filename(self, kind: str, result: CaseResult | None) -> str:
        if kind == "evaporation_time_all_cases":
            return "evaporation_time_all_cases.png"
        case_name = _safe_filename(result.case_name) if result is not None else "selected_case"
        suffixes = {
            "volume_time": "volume_time",
            "equivalent_radius_time": "equivalent_radius_time",
            "contact_angle_time": "contact_angle_time",
            "contact_radius_time": "contact_radius_time",
            "departure_height_distribution": "departure_height_distribution",
            "departure_time_height": "departure_time_height",
            "theory": "theory",
            "theory_evaporated_mass": "theory_evaporated_mass",
            "theory_equivalent_radius": "theory_equivalent_radius",
            "theory_combined": "theory_combined",
        }
        return f"{case_name}_{suffixes.get(kind, 'graph')}.png"

    def _choose_output_directory(self, title: str, suggested_name: str) -> Path | None:
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        layout = QVBoxLayout(dialog)
        row = QHBoxLayout()
        path_edit = QLineEdit(str(Path(self._local_dialog_start_dir()) / suggested_name))
        browse_button = QPushButton(self.t("参照"))
        row.addWidget(path_edit, 1)
        row.addWidget(browse_button)
        layout.addWidget(QLabel(self.t("保存先フォルダ")))
        layout.addLayout(row)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("OK")
        buttons.button(QDialogButtonBox.Cancel).setText(self.t("キャンセル"))
        layout.addWidget(buttons)

        def browse() -> None:
            directory = QFileDialog.getExistingDirectory(dialog, self.t("親フォルダを選択"), self._local_dialog_start_dir())
            if directory:
                path_edit.setText(str(Path(directory) / suggested_name))

        browse_button.clicked.connect(browse)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        if dialog.exec() != QDialog.Accepted:
            return None
        text = path_edit.text().strip()
        if not text:
            return None
        return Path(text)

    def _selected_theory_bulk_kind(self) -> str | None:
        dialog = QDialog(self)
        dialog.setWindowTitle(self.t("保存する理論グラフ"))
        layout = QVBoxLayout(dialog)
        combo = QComboBox()
        combo.addItem(self.t("蒸発量 EM"), "theory_evaporated_mass")
        combo.addItem(self.t("理論/MD 等価半径"), "theory_equivalent_radius")
        combo.addItem(self.t("上下2枚"), "theory_combined")
        layout.addWidget(QLabel(self.t("全ケースPNG出力するグラフを選択してください。")))
        layout.addWidget(combo)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("OK")
        buttons.button(QDialogButtonBox.Cancel).setText(self.t("キャンセル"))
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.Accepted:
            return None
        return str(combo.currentData())

    def _plot_for_export(self, kind: str, result: CaseResult | None, base_plot: PlotWidget) -> PlotWidget | CombinedPlotWidget:
        plot = PlotWidget()
        plot.settings = GraphSettings(**vars(base_plot.settings))
        if kind == "evaporation_time_all_cases":
            labels = [item.case_name for item in self.results if item.evaporation_time is not None]
            values = [item.evaporation_time for item in self.results if item.evaporation_time is not None]
            self._apply_axis_settings_for_kind(plot, kind)
            if labels:
                plot.plot_bar("蒸発完了時刻", labels, values)
            else:
                plot.clear("蒸発完了時刻")
            return plot
        if result is None:
            plot.clear("ケースが選択されていません")
            return plot
        if kind in ("volume_time", "equivalent_radius_time", "contact_angle_time", "contact_radius_time"):
            self._plot_standard_result_for_export(plot, result, kind)
        elif kind in ("departure_height_distribution", "departure_time_height"):
            self._plot_departure_result(plot, result, kind)
        elif kind == "theory_combined":
            return self._plot_combined_theory_result_for_export(result, base_plot)
        elif kind in ("theory_evaporated_mass", "theory_equivalent_radius"):
            self._plot_theory_result_for_export(plot, result, kind)
        else:
            plot.clear("保存対象外のグラフです")
        return plot

    def _plot_combined_theory_result_for_export(self, result: CaseResult, base_plot: PlotWidget) -> CombinedPlotWidget:
        em_plot = PlotWidget()
        radius_plot = PlotWidget()
        em_plot.settings = GraphSettings(**vars(self.theory_em_plot.settings))
        radius_plot.settings = GraphSettings(**vars(self.theory_radius_plot.settings))
        self._plot_theory_result_for_export(em_plot, result, "theory_evaporated_mass")
        self._plot_theory_result_for_export(radius_plot, result, "theory_equivalent_radius")
        combined = CombinedPlotWidget([em_plot, radius_plot], owns_source_plots=True)
        combined.settings = GraphSettings(**vars(base_plot.settings))
        combined.settings.image_height = min(30.0, max(combined.settings.image_height, combined.settings.image_height * 2.0))
        combined.redraw()
        return combined

    def _plot_standard_result_for_export(self, plot: PlotWidget, result: CaseResult, kind: str) -> None:
        rows = _rows_until_evaporation(result)
        times = [row.time for row in rows]
        self._apply_axis_settings_for_kind(plot, kind)
        if kind == "volume_time":
            plot.plot_xy(f"{result.case_name}: 体積-時間", "時間 [s]", "体積 [m^3]", times, [row.volume for row in rows])
        elif kind == "equivalent_radius_time":
            plot.plot_xy(
                f"{result.case_name}: 等価半径-時間",
                "時間 [s]",
                "等価半径 [m]",
                times,
                [row.equivalent_radius for row in rows],
            )
        elif kind == "contact_angle_time":
            points = [(row.time, row.contact_angle_deg) for row in rows if row.contact_angle_deg is not None]
            plot.plot_xy(
                f"{result.case_name}: 接触角-時間",
                "時間 [s]",
                "接触角 [deg]",
                [point[0] for point in points],
                [point[1] for point in points],
            )

        elif kind == "contact_radius_time":
            points = [
                (row.time, row.contact_radius)
                for row in rows
                if row.contact_radius is not None
            ]
            plot.plot_xy(
                f"{result.case_name}: 接触半径-時間",
                "時間 [s]",
                "接触半径 [m]",
                [point[0] for point in points],
                [point[1] for point in points],
            )

    def _plot_departure_result(
        self,
        plot: PlotWidget,
        result: CaseResult,
        kind: str,
    ) -> None:
        departure = result.departure_result
        self._apply_axis_settings_for_kind(plot, kind)
        if departure is None:
            plot.clear(f"{result.case_name}: 分子離脱解析は無効です")
            return
        if departure.status != "ok":
            message = departure.error or departure.status
            plot.clear(f"{result.case_name}: {message}")
            return

        excluded = departure.excluded_normalized_height_count
        if kind == "departure_height_distribution":
            bin_mode_label = (
                "球面表面積等分"
                if departure.bin_mode == "equal_surface_area"
                else "高さ等間隔"
            )
            rate_mode = (
                _combo_data(self.departure_intensity_combo, "count") == "rate"
            )
            centers = [
                (item.eta_lower + item.eta_upper) / 2.0
                for item in departure.height_bins
            ]
            raw_values = [
                (
                    float("nan")
                    if item.raw_rate is None
                    else item.raw_rate
                )
                if rate_mode
                else float(item.raw_count)
                for item in departure.height_bins
            ]
            confirmed_values = [
                (
                    float("nan")
                    if item.confirmed_rate is None
                    else item.confirmed_rate
                )
                if rate_mode
                else float(item.confirmed_count)
                for item in departure.height_bins
            ]
            y_label = (
                "イベント率 [1/(m^2 s)]"
                if rate_mode
                else "イベント件数"
            )
            plot.plot_series(
                f"{result.case_name}: 分子離脱高さ分布 "
                f"({bin_mode_label}, 範囲外 {excluded})",
                "正規化高さ eta",
                y_label,
                [
                    PlotSeries(
                        "全離脱",
                        centers,
                        raw_values,
                        style="scatter",
                        color="#7f8c8d",
                        marker="o",
                    ),
                    PlotSeries(
                        "確定離脱",
                        centers,
                        confirmed_values,
                        style="scatter",
                        color="#d62728",
                        marker="s",
                    ),
                ],
            )
            return

        raw_events = [
            event
            for event in departure.events
            if not event.confirmed
            and event.normalized_height is not None
            and 0.0 <= event.normalized_height <= 1.0
        ]
        confirmed_events = [
            event
            for event in departure.events
            if event.confirmed
            and event.normalized_height is not None
            and 0.0 <= event.normalized_height <= 1.0
        ]
        plot.plot_series(
            f"{result.case_name}: 分子離脱 時刻-高さ (範囲外 {excluded})",
            "時刻 [s]",
            "正規化高さ eta",
            [
                PlotSeries(
                    "未確定離脱",
                    [event.event_time for event in raw_events],
                    [float(event.normalized_height) for event in raw_events],
                    style="scatter",
                    color="#7f8c8d",
                    marker="o",
                ),
                PlotSeries(
                    "確定離脱",
                    [event.event_time for event in confirmed_events],
                    [
                        float(event.normalized_height)
                        for event in confirmed_events
                    ],
                    style="scatter",
                    color="#d62728",
                    marker="s",
                ),
            ],
        )
    def _plot_theory_result_for_export(self, plot: PlotWidget, result: CaseResult, kind: str) -> None:
        comparison = self._theory_comparison(result)
        self._apply_axis_settings_for_kind(plot, kind)
        if comparison.status != "ok":
            plot.clear(f"{result.case_name}: {comparison.status}")
            return
        series = self._theory_export_series(result, comparison, "em" if kind == "theory_evaporated_mass" else "radius")
        if kind == "theory_evaporated_mass":
            plot.plot_series(f"{result.case_name}: 蒸発量 EM-時間", "時間 [s]", "蒸発量 EM [kg]", series)
        else:
            plot.plot_series(f"{result.case_name}: 理論/MD 等価半径-時間", "時間 [s]", "等価半径 [m]", series)

    def _theory_export_series(self, result: CaseResult, comparison: TheoryComparison, value_kind: str) -> list[PlotSeries]:
        series_list: list[PlotSeries] = []
        if self.theory_show_md_check.isChecked():
            values = comparison.md_evaporated_masses if value_kind == "em" else comparison.md_equivalent_radii
            x_values, y_values = _clip_xy_to_evaporation(result, comparison.times, values)
            series_list.append(
                PlotSeries("MD", x_values, y_values, style="scatter", color=COLORS["md_series"], marker="o")
            )
        alpha_colors = {0.8: "#1f77b4", 0.9: "#ff7f0e", 1.0: "#2ca02c"}
        for alpha, checkbox in self.theory_alpha_checks.items():
            if not checkbox.isChecked():
                continue
            curve = comparison.curves.get(alpha)
            if curve is None:
                continue
            values = curve.evaporated_masses if value_kind == "em" else curve.equivalent_radii
            x_values, y_values = _clip_xy_to_evaporation(result, curve.times, values)
            series_list.append(PlotSeries(f"alpha_e={alpha:g}", x_values, y_values, style="line", color=alpha_colors.get(alpha)))
        if self.theory_show_fit_check.isChecked() and comparison.fit_curve is not None:
            label = "fit"
            if comparison.fit.alpha_e is not None:
                label = f"fit alpha_e={comparison.fit.alpha_e:.4g}"
                if comparison.fit.boundary:
                    label += f" ({comparison.fit.boundary})"
            values = comparison.fit_curve.evaporated_masses if value_kind == "em" else comparison.fit_curve.equivalent_radii
            x_values, y_values = _clip_xy_to_evaporation(result, comparison.fit_curve.times, values)
            series_list.append(PlotSeries(label, x_values, y_values, style="line", color="#d62728", linestyle="--", linewidth=1.8))
        return series_list

    @Slot()
    def export_visual_png(self) -> None:
        result = self.current_result()
        if result is None:
            QMessageBox.information(self, "ケースなし", "可視化する結果ケースを選択してください。")
            return
        filename = f"{_safe_filename(result.case_name)}_visualization.png"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "可視化PNGを保存",
            str(Path(self._local_dialog_start_dir()) / filename),
            "PNG (*.png)",
        )
        if not path:
            return
        output_path = _ensure_suffix(Path(path), ".png")
        self.visual_plot.save_png(output_path)
        self.log(f"可視化PNGを保存しました: {output_path}")

    @Slot()
    def export_visual_gif(self) -> None:
        result = self.current_result()
        if result is None or not result.rows:
            QMessageBox.information(self, "ケースなし", "可視化する結果ケースを選択してください。")
            return
        filename = f"{_safe_filename(result.case_name)}_visualization.gif"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "可視化GIFを保存",
            str(Path(self._local_dialog_start_dir()) / filename),
            "GIF (*.gif)",
        )
        if not path:
            return
        output_path = _ensure_suffix(Path(path), ".gif")
        start, end = self._visual_range_indices()
        rows = result.rows[start : end + 1]
        writer = PillowWriter(fps=self.visual_fps_spin.value())
        try:
            with writer.saving(self.visual_plot.figure, str(output_path), dpi=120):
                for row in rows:
                    frame = self._load_visual_frame(result, row.time)
                    self._draw_visual_frame(frame)
                    self.visual_plot.canvas.draw()
                    writer.grab_frame()
                    QApplication.processEvents()
            self.log(f"可視化GIFを保存しました: {output_path}")
        except Exception as exc:
            QMessageBox.warning(self, "GIF保存エラー", str(exc))
            self.log(f"可視化GIF保存に失敗しました: {exc}")

    def update_evap_plot(self) -> None:
        labels = [result.case_name for result in self.results if result.evaporation_time is not None]
        values = [result.evaporation_time for result in self.results if result.evaporation_time is not None]
        self._apply_axis_settings_for_kind(self.evap_plot, "evaporation_time_all_cases")
        if labels:
            self.evap_plot.plot_bar("蒸発完了時刻", labels, values)
        else:
            self.evap_plot.clear("蒸発完了時刻")

    def _write_current_manifest(self, path: Path) -> None:
        if self._last_run_settings is None or self._last_run_context is None:
            raise ProvenanceError("No completed analysis context is available")
        write_analysis_manifest(
            path,
            self._last_run_context,
            self.results,
        )

    @Slot()
    def export_analysis_manifest(self) -> None:
        if not self.results:
            QMessageBox.information(
                self,
                self.t("解析記録を保存"),
                self.t("解析を実行してから保存してください。"),
            )
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            self.t("解析記録を保存"),
            str(Path(self._local_dialog_start_dir()) / "analysis_manifest.json"),
            "JSON (*.json)",
        )
        if not path:
            return
        output_path = _ensure_suffix(Path(path), ".json")
        try:
            self._write_current_manifest(output_path)
        except ProvenanceError as exc:
            QMessageBox.warning(self, self.t("解析記録を保存"), str(exc))
            return
        self.log(f"{self.t('解析記録を保存しました')}: {output_path}")

    @Slot()
    def export_csv(self) -> None:
        if not self.results:
            QMessageBox.information(self, "結果なし", "出力前に解析を実行してください。")
            return
        out_dir = self._choose_output_directory("CSV出力フォルダ", f"mdfoam_csv_{_timestamp()}")
        if out_dir is None:
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        write_summary_csv(out_dir / "mdfoam_summary.csv", self.results)
        write_timeseries_csv(out_dir / "mdfoam_timeseries.csv", self.results)
        write_departure_events_csv(
            out_dir / "mdfoam_departure_events.csv",
            self.results,
        )
        write_departure_height_bins_csv(
            out_dir / "mdfoam_departure_height_bins.csv",
            self.results,
        )
        theory_settings = self.theory_settings()
        write_theory_summary_csv(out_dir / "mdfoam_theory_summary.csv", self.results, theory_settings, DEFAULT_ALPHA_VALUES)
        write_theory_timeseries_csv(out_dir / "mdfoam_theory_timeseries.csv", self.results, theory_settings, DEFAULT_ALPHA_VALUES)
        try:
            self._write_current_manifest(out_dir / "analysis_manifest.json")
        except ProvenanceError as exc:
            QMessageBox.warning(self, self.t("解析記録を保存"), str(exc))
            self.log(f"{self.t('解析記録の保存に失敗しました')}: {exc}")
        self.log(f"CSVを出力しました: {out_dir}")

    @Slot()
    def export_png(self) -> None:
        tab = self.tabs.currentWidget()
        if isinstance(tab, PlotWidget):
            context = self._current_png_context()
            if context is None:
                return
            kind, plot = context
            filename = self._suggested_png_filename(kind, self.current_result())
            dialog = GraphPngPreviewDialog(plot, self._local_dialog_start_dir(), self, filename)
        elif tab is self.theory_tab:
            result = self.current_result()
            filename = self._suggested_png_filename("theory_evaporated_mass", result)
            dialog = GraphPngPreviewDialog(self._theory_png_options(result), self._local_dialog_start_dir(), self, filename)
        else:
            return
        if dialog.exec() == QDialog.Accepted and dialog.saved_path is not None:
            self.log(f"PNGを出力しました: {dialog.saved_path}")

    @Slot()
    def export_all_png(self) -> None:
        if not self.results:
            QMessageBox.information(self, "結果なし", "出力前に解析を実行してください。")
            return
        tab = self.tabs.currentWidget()
        context = self._current_png_context()
        if context is not None:
            kind, base_plot = context
        elif tab is self.theory_tab:
            kind = self._selected_theory_bulk_kind()
            if kind is None:
                return
            base_plot = self.theory_radius_plot if kind == "theory_equivalent_radius" else self.theory_em_plot
        else:
            QMessageBox.information(self, "対象外", "可視化タブは全ケースPNG出力の対象外です。")
            return
        out_dir = self._choose_output_directory("PNG出力フォルダ", f"mdfoam_png_{_timestamp()}")
        if out_dir is None:
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        saved_count = 0
        try:
            if kind == "evaporation_time_all_cases":
                plot = self._plot_for_export(kind, None, base_plot)
                plot.save_png(out_dir / "evaporation_time_all_cases.png")
                plot.close()
                saved_count = 1
            else:
                for result in self.results:
                    plot = self._plot_for_export(kind, result, base_plot)
                    plot.save_png(out_dir / self._suggested_png_filename(kind, result))
                    plot.close()
                    saved_count += 1
        except Exception as exc:
            QMessageBox.warning(self, "PNG出力エラー", str(exc))
            self.log(f"全ケースPNG出力に失敗しました: {exc}")
            return
        self.log(f"全ケースPNGを出力しました: {out_dir} ({saved_count}枚)")

    @Slot(str)
    def log(self, message: str) -> None:
        self.log_box.append(message)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "settings_grid"):
            self._update_settings_grid(event.size().width())

    def _update_settings_grid(self, window_width: int) -> None:
        compact = window_width < 1180
        positions = (
            (
                (self.basic_group, 0, 0),
                (self.advanced_group, 1, 0),
                (self.fallback_group, 2, 0),
                (self.theory_group, 3, 0),
                (self.departure_group, 4, 0),
            )
            if compact
            else (
                (self.basic_group, 0, 0),
                (self.advanced_group, 0, 1),
                (self.fallback_group, 1, 0),
                (self.departure_group, 1, 1),
                (self.theory_group, 2, 0),
            )
        )
        for widget, row, column in positions:
            self.settings_grid.addWidget(widget, row, column)
        self.settings_grid.setColumnStretch(0, 1)
        self.settings_grid.setColumnStretch(1, 0 if compact else 1)

    def closeEvent(self, event) -> None:
        if self.remote_browser_connection is not None:
            self.remote_browser_connection.close()
        super().closeEvent(event)


def _fmt(value: float) -> str:
    return f"{value:.8g}"


def _padded_axis_range(
    x_values: list[float],
    y_values: list[float],
) -> tuple[float, float, float, float] | None:
    x_range = _padded_value_range(x_values)
    y_range = _padded_value_range(y_values)
    if x_range is None or y_range is None:
        return None
    return x_range[0], x_range[1], y_range[0], y_range[1]


def _padded_value_range(values: list[float]) -> tuple[float, float] | None:
    finite_values = [float(value) for value in values if np.isfinite(value)]
    if not finite_values:
        return None
    minimum = min(finite_values)
    maximum = max(finite_values)
    if minimum == maximum:
        padding = max(abs(minimum) * 0.05, 1.0e-12)
    else:
        padding = (maximum - minimum) * 0.05
    lower = minimum - padding
    upper = maximum + padding
    if minimum >= 0.0 and lower < 0.0:
        lower = 0.0
    if lower == upper:
        upper = lower + max(abs(lower) * 0.1, 1.0e-12)
    return lower, upper


def _draw_plot_on_axis(
    plot: PlotWidget,
    axis,
    settings: GraphSettings,
    light_theme: bool = False,
) -> None:
    if plot._last_plot is None:
        axis.set_axis_off()
        return
    kind, title, x_label, y_label, x, y = plot._last_plot
    original_settings = plot.settings
    original_theme = plot.light_theme
    plot.settings = settings
    plot.light_theme = light_theme
    try:
        if kind == "xy":
            axis.scatter(
                x,
                y,
                s=settings.point_size,
                c=settings.point_color,
                alpha=settings.point_alpha,
                marker=settings.marker_override or settings.marker,
            )
            plot._apply_common_style(axis, x_label, y_label, title)
        elif kind == "bar":
            axis.bar(x, y, color=settings.point_color, alpha=settings.point_alpha)
            plot._apply_common_style(axis, "", y_label, title, is_bar=True)
        elif kind == "series" and plot._last_series is not None:
            title, x_label, y_label, series_list = plot._last_series
            for item in series_list:
                if not item.x or not item.y:
                    continue
                if item.style == "line":
                    axis.plot(
                        item.x,
                        item.y,
                        label=item.label,
                        color=plot._display_series_color(item.color),
                        linestyle=(
                            item.linestyle
                            if settings.line_style == "source"
                            else settings.line_style
                        ),
                        linewidth=(
                            settings.line_width
                            if settings.line_width is not None
                            else item.linewidth
                        ),
                        alpha=settings.point_alpha,
                    )
                else:
                    axis.scatter(
                        item.x,
                        item.y,
                        label=item.label,
                        s=settings.point_size,
                        c=plot._display_series_color(item.color) or settings.point_color,
                        alpha=settings.point_alpha,
                        marker=settings.marker_override or item.marker or settings.marker,
                    )
            if settings.legend_visible and any(item.label for item in series_list):
                axis.legend(
                    fontsize=settings.legend_font_size or max(6, settings.font_size - 1),
                    loc=settings.legend_location,
                )
            plot._apply_common_style(axis, x_label, y_label, title)
        elif kind == "clear":
            plot._apply_common_style(axis, "", "", title)
    finally:
        plot.settings = original_settings
        plot.light_theme = original_theme


def _quality_label_for_dpi(dpi: int) -> str:
    for label, value in QUALITY_DPI_OPTIONS.items():
        if value == dpi:
            return label
    return DEFAULT_QUALITY_LABEL


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_filename(value: str) -> str:
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", value.strip())
    sanitized = re.sub(r"\s+", "_", sanitized)
    sanitized = sanitized.strip("._")
    return sanitized or "case"


def _ensure_suffix(path: Path, suffix: str) -> Path:
    return path if path.suffix.lower() == suffix.lower() else path.with_suffix(suffix)


def _rows_until_evaporation(result: CaseResult) -> list[TimeResult]:
    if result.evaporation_time is None:
        return list(result.rows)
    return [row for row in result.rows if row.time <= result.evaporation_time]


def _clip_xy_to_evaporation(
    result: CaseResult,
    x_values: list[float],
    y_values: list[float],
) -> tuple[list[float], list[float]]:
    if result.evaporation_time is None:
        return list(x_values), list(y_values)
    clipped = [
        (x_value, y_value)
        for x_value, y_value in zip(x_values, y_values)
        if x_value <= result.evaporation_time
    ]
    if not clipped:
        return [], []
    x_clipped, y_clipped = zip(*clipped)
    return list(x_clipped), list(y_clipped)


def _fmt_optional(value: float | None) -> str:
    return "" if value is None else _fmt(value)


def _status_label(status: str, has_warnings: bool = False) -> str:
    if status == "ok" and has_warnings:
        return "完了（警告あり）"
    labels = {
        "ok": "完了",
        "error": "エラー",
        "stopped": "停止",
        "running": "実行中",
    }
    return labels.get(status, status)


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLESHEET)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
