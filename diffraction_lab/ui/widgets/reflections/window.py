"""ViewHKL-style reflection viewer window."""

from __future__ import annotations

from pathlib import Path

from decologr import Decologr as log
from picoui.icons import IconRegistry
from picoui.theme import ThemeManager, ThemeType
from picoui.widget.viewer_chrome import icon_from_registry, icon_label, standard_zoom_actions
from PySide6.QtCore import Qt, QSettings, QSize
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTableView,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from molib.xtal.reflections.errors import (
    ReflectionLoadError,
    UnsupportedReflectionFormatError,
)
from molib.xtal.reflections.loader import ReflectionFileLoader
from molib.xtal.reflections.model import ReflectionData
from molib.xtal.reflections.quantity import DisplayQuantity, available_quantities
from molib.xtal.reflections.zone import ZonePreset, ZoneSpec
from diffraction_lab.ui.widgets.reflections.controller import ViewerFilterState, prepare_view
from diffraction_lab.ui.widgets.reflections.plot import PlotPointCloud, ReciprocalSpacePlot
from diffraction_lab.ui.widgets.reflections.summary import (
    format_crystal_toolbar_lines,
    format_selection_details,
    format_summary_text,
)
from diffraction_lab.ui.widgets.reflections.table_model import (
    OBSERVATION_ID_ROLE,
    ReflectionTableModel,
)


class RecentFilesManager:
    """Persist recently opened reflection paths in ``QSettings``."""

    MAX_RECENT = 10
    ORG = "ElMo"
    APP = "ReflectionViewer"
    KEY = "recent_files"

    def __init__(self) -> None:
        self._settings = QSettings(self.ORG, self.APP)

    def _read(self) -> list[str]:
        raw = self._settings.value(self.KEY)
        if raw is None or raw == "":
            return []
        if isinstance(raw, list):
            return [str(p) for p in raw if p and str(p).strip()]
        if isinstance(raw, str):
            return [raw] if raw.strip() else []
        try:
            return [str(p) for p in raw if p and str(p).strip()]
        except TypeError:
            return []

    def _write(self, paths: list[str]) -> None:
        self._settings.setValue(self.KEY, paths)
        self._settings.sync()

    def add(self, file_path: str | Path) -> None:
        """Move *file_path* to the front of the recent list."""
        if not file_path:
            return
        resolved = str(Path(file_path).expanduser().resolve())
        paths = self._read()
        if resolved in paths:
            paths.remove(resolved)
        paths.insert(0, resolved)
        self._write(paths[: self.MAX_RECENT])

    def get(self) -> list[str]:
        """Return existing recent paths, pruning missing entries."""
        paths = self._read()
        existing = [p for p in paths if Path(p).exists()]
        if len(existing) != len(paths):
            self._write(existing)
        return existing

    def clear(self) -> None:
        """Clear the recent-files list."""
        self._write([])


class ReflectionViewerWindow(QMainWindow):
    """Dedicated window for inspecting crystallographic reflections."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(None)
        self.setWindowTitle("Reflection Viewer")
        self.setMinimumSize(1000, 700)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

        self._loader = ReflectionFileLoader()
        self._data: ReflectionData | None = None
        self._path: Path | None = None
        self._selected_id: int | None = None
        self._updating = False
        self._recent_files = RecentFilesManager()

        self._filter = ViewerFilterState(zone=ZoneSpec.from_preset(ZonePreset.HK0))

        try:
            ThemeManager.initialize(ThemeType.AUTO)
        except Exception as ex:
            log.warning(f"ThemeManager.initialize failed: {ex}")

        self.setWindowIcon(icon_from_registry(IconRegistry.CHART_SCATTER_PLOT))
        self._build_actions()
        self._build_menus()
        self._build_toolbar()
        self._build_central_ui()
        self._apply_widget_styles()
        self._wire_actions()
        self._refresh_empty()
        log.info("ReflectionViewerWindow initialized")

    def _build_actions(self) -> None:
        """Create QActions shared by menus and the toolbar."""
        self._act_open = QAction(
            icon_from_registry(IconRegistry.FOLDER_OPENED),
            "&Open…",
            self,
        )
        self._act_open.setShortcut(QKeySequence.StandardKey.Open)
        self._act_open.setStatusTip("Open an MTZ or structure-factor CIF file")

        self._act_reload = QAction(
            icon_from_registry(IconRegistry.MDI_REFRESH),
            "&Reload",
            self,
        )
        self._act_reload.setShortcut(QKeySequence.StandardKey.Refresh)
        self._act_reload.setStatusTip("Reload the current reflection file")

        (
            self._act_zoom_in,
            self._act_zoom_out,
            self._act_reset_zoom,
        ) = standard_zoom_actions(
            self,
            on_zoom_in=lambda: self._plot.zoom_in(),
            on_zoom_out=lambda: self._plot.zoom_out(),
            on_reset=lambda: self._plot.reset_view(),
            zoom_in_tip="Zoom in (scroll to zoom · drag to pan)",
            zoom_out_tip="Zoom out (scroll to zoom · drag to pan)",
            reset_tip="Fit the reciprocal-space plot to the window",
        )

        self._act_axes = QAction(
            icon_from_registry(IconRegistry.AXIS_X_ARROW),
            "Show &axes",
            self,
        )
        self._act_axes.setCheckable(True)
        self._act_axes.setChecked(True)
        self._act_axes.setStatusTip("Toggle reciprocal-space axes")

        self._act_symmetry = QAction(
            icon_from_registry(IconRegistry.AUTO_ROTATE),
            "Show &symmetry axes",
            self,
        )
        self._act_symmetry.setCheckable(True)
        self._act_symmetry.setChecked(True)
        self._act_symmetry.setStatusTip("Toggle symmetry-axis overlays")

        self._act_contours = QAction(
            icon_from_registry(IconRegistry.CIRCLE_OUTLINE),
            "Show resolution &contours",
            self,
        )
        self._act_contours.setCheckable(True)
        self._act_contours.setChecked(True)
        self._act_contours.setStatusTip("Toggle resolution contour rings")

        self._act_summary = QAction(
            icon_from_registry(IconRegistry.INFO),
            "Show Summary &Tab",
            self,
        )
        self._act_summary.setStatusTip("Switch to the Summary tab")

        self._act_quit = QAction(
            icon_from_registry(IconRegistry.EXIT),
            "&Quit",
            self,
        )
        self._act_quit.setShortcut(QKeySequence.StandardKey.Quit)
        self._act_quit.setStatusTip("Close the Reflection Viewer")

    def _build_menus(self) -> None:
        """Build File / View menus (DiffractionLab-style chrome)."""
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self._act_open)
        self._recent_menu = QMenu("Open &Recent", self)
        file_menu.addMenu(self._recent_menu)
        self._recent_menu.aboutToShow.connect(self._refresh_recent_files_menu)
        self._refresh_recent_files_menu()
        file_menu.addAction(self._act_reload)
        file_menu.addSeparator()
        file_menu.addAction(self._act_quit)

        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self._act_zoom_in)
        view_menu.addAction(self._act_zoom_out)
        view_menu.addAction(self._act_reset_zoom)
        view_menu.addSeparator()
        view_menu.addAction(self._act_axes)
        view_menu.addAction(self._act_symmetry)
        view_menu.addAction(self._act_contours)
        view_menu.addSeparator()
        view_menu.addAction(self._act_summary)

    def _refresh_recent_files_menu(self) -> None:
        """Rebuild the Open Recent submenu from persisted paths."""
        self._recent_menu.clear()
        paths = self._recent_files.get()
        if not paths:
            empty = QAction("(No recent files)", self)
            empty.setEnabled(False)
            self._recent_menu.addAction(empty)
            return
        for index, path in enumerate(paths):
            label = path if len(path) < 80 else f"…{path[-77:]}"
            if index < 9:
                label = f"&{index + 1}  {label}"
            action = QAction(label, self)
            action.setStatusTip(path)
            action.triggered.connect(
                lambda checked=False, p=path: self.load_path(p)
            )
            self._recent_menu.addAction(action)
        self._recent_menu.addSeparator()
        clear_action = QAction("Clear Recent", self)
        clear_action.triggered.connect(self._clear_recent_files)
        self._recent_menu.addAction(clear_action)

    def _clear_recent_files(self) -> None:
        """Clear persisted recent files and refresh the submenu."""
        self._recent_files.clear()
        self._refresh_recent_files_menu()
        self._status.showMessage("Recent files cleared.", 3000)

    def _build_toolbar(self) -> None:
        """Build the primary tools toolbar."""
        toolbar = QToolBar("Reflection tools")
        toolbar.setObjectName("reflectionViewerToolBar")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(20, 20))
        self.addToolBar(toolbar)
        toolbar.addAction(self._act_open)
        toolbar.addAction(self._act_reload)
        toolbar.addSeparator()
        toolbar.addAction(self._act_zoom_in)
        toolbar.addAction(self._act_zoom_out)
        toolbar.addAction(self._act_reset_zoom)
        toolbar.addSeparator()
        toolbar.addAction(self._act_axes)
        toolbar.addAction(self._act_symmetry)
        toolbar.addAction(self._act_contours)
        toolbar.addSeparator()
        toolbar.addAction(self._act_summary)

    def _build_central_ui(self) -> None:
        """Build metadata bar, plot, and side panel."""
        meta_bar = QToolBar("Crystal metadata")
        meta_bar.setObjectName("reflectionViewerMetaBar")
        meta_bar.setMovable(False)
        meta_bar.setFloatable(False)
        meta_bar.setIconSize(QSize(16, 16))
        self.addToolBarBreak()
        self.addToolBar(meta_bar)
        meta_host = QWidget()
        meta_layout = QVBoxLayout(meta_host)
        meta_layout.setContentsMargins(8, 2, 8, 2)
        meta_layout.setSpacing(2)
        sg_host, self._meta_space_group = icon_label(
            IconRegistry.SYMMETRY_MATES_TAB, "Space group: —"
        )
        cell_host, self._meta_unit_cell = icon_label(
            IconRegistry.CUBE, "Unit cell (file): —"
        )
        merged_host, self._meta_merged = icon_label(
            IconRegistry.DATABASE, "Merged/unmerged: —"
        )
        res_h_host, self._meta_res_header = icon_label(
            IconRegistry.RULER, "Resolution (file header): —"
        )
        res_c_host, self._meta_res_computed = icon_label(
            IconRegistry.MDI_MAGNIFY, "Resolution (computed from indices): —"
        )
        for host, label in (
            (sg_host, self._meta_space_group),
            (cell_host, self._meta_unit_cell),
            (merged_host, self._meta_merged),
            (res_h_host, self._meta_res_header),
            (res_c_host, self._meta_res_computed),
        ):
            label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            meta_layout.addWidget(host)
        meta_bar.addWidget(meta_host)

        central = QWidget(self)
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        self._plot = ReciprocalSpacePlot()
        splitter.addWidget(self._plot)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        form = QFormLayout()

        self._zone_combo = QComboBox()
        for preset in (
            ZonePreset.HK0,
            ZonePreset.H0L,
            ZonePreset.OKL,
            ZonePreset.HHL,
            ZonePreset.CUSTOM,
        ):
            self._zone_combo.addItem(preset.value, preset)
        form.addRow("Zone", self._zone_combo)

        self._level_spin = QDoubleSpinBox()
        self._level_spin.setDecimals(0)
        self._level_spin.setRange(-100, 100)
        self._level_spin.setValue(0)
        form.addRow("Zone level", self._level_spin)

        self._expand_symmetry_cb = QCheckBox("Expand by symmetry")
        self._expand_symmetry_cb.setChecked(True)
        self._expand_symmetry_cb.setToolTip(
            "Map observations into this zone with space-group operations "
            "(and Friedel mates). Needed when a zone is sparsely observed "
            "(e.g. h0l on a spindle-aligned 0kl-rich scan)."
        )
        form.addRow(self._expand_symmetry_cb)

        self._nz_x = QDoubleSpinBox()
        self._nz_y = QDoubleSpinBox()
        self._nz_z = QDoubleSpinBox()
        for spin, val in ((self._nz_x, 0), (self._nz_y, 0), (self._nz_z, 1)):
            spin.setDecimals(0)
            spin.setRange(-20, 20)
            spin.setValue(val)
        normal_row = QHBoxLayout()
        normal_row.addWidget(self._nz_x)
        normal_row.addWidget(self._nz_y)
        normal_row.addWidget(self._nz_z)
        form.addRow("Custom normal", normal_row)

        self._quantity_combo = QComboBox()
        form.addRow("Quantity", self._quantity_combo)

        self._column_combo = QComboBox()
        form.addRow("Column", self._column_combo)

        self._dmin_spin = QDoubleSpinBox()
        self._dmin_spin.setDecimals(2)
        self._dmin_spin.setRange(0.0, 1000.0)
        self._dmin_spin.setSpecialValueText("none")
        self._dmin_spin.setValue(0.0)
        form.addRow("High-res cutoff (Å)", self._dmin_spin)

        self._dmax_spin = QDoubleSpinBox()
        self._dmax_spin.setDecimals(2)
        self._dmax_spin.setRange(0.0, 1000.0)
        self._dmax_spin.setSpecialValueText("none")
        self._dmax_spin.setValue(0.0)
        form.addRow("Low-res cutoff (Å)", self._dmax_spin)

        self._axes_cb = QCheckBox("Show axes")
        self._axes_cb.setIcon(icon_from_registry(IconRegistry.AXIS_X_ARROW))
        self._axes_cb.setChecked(True)
        self._symmetry_cb = QCheckBox("Show symmetry axes")
        self._symmetry_cb.setIcon(icon_from_registry(IconRegistry.AUTO_ROTATE))
        self._symmetry_cb.setChecked(True)
        self._contours_cb = QCheckBox("Show resolution contours")
        self._contours_cb.setIcon(icon_from_registry(IconRegistry.CIRCLE_OUTLINE))
        self._contours_cb.setChecked(True)
        form.addRow(self._axes_cb)
        form.addRow(self._symmetry_cb)
        form.addRow(self._contours_cb)

        self._marker_spin = QDoubleSpinBox()
        self._marker_spin.setRange(0.2, 5.0)
        self._marker_spin.setSingleStep(0.1)
        self._marker_spin.setValue(1.0)
        form.addRow("Marker scale", self._marker_spin)

        self._contrast_spin = QDoubleSpinBox()
        self._contrast_spin.setRange(0.2, 5.0)
        self._contrast_spin.setSingleStep(0.1)
        self._contrast_spin.setValue(1.0)
        form.addRow("Contrast", self._contrast_spin)

        apply_btn = QPushButton("Apply filters")
        apply_btn.setIcon(icon_from_registry(IconRegistry.FILTER))
        apply_btn.clicked.connect(self._apply_filters)
        form.addRow(apply_btn)

        right_layout.addLayout(form)

        self._details = QTextEdit()
        self._details.setReadOnly(True)
        self._details.setMaximumHeight(180)
        selection_header = QWidget()
        selection_row = QHBoxLayout(selection_header)
        selection_row.setContentsMargins(0, 0, 0, 0)
        selection_row.setSpacing(4)
        selection_icon = QLabel()
        selection_icon.setPixmap(
            icon_from_registry(IconRegistry.PICK_ATOM).pixmap(16, 16)
        )
        selection_row.addWidget(selection_icon)
        selection_row.addWidget(QLabel("Selection"))
        selection_row.addStretch(1)
        right_layout.addWidget(selection_header)
        right_layout.addWidget(self._details)

        tabs = QTabWidget()
        self._table_model = ReflectionTableModel(self)
        self._table = QTableView()
        self._table.setModel(self._table_model)
        self._table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self._table.setSortingEnabled(False)
        tabs.addTab(
            self._table,
            icon_from_registry(IconRegistry.TABLE_GRID),
            "Reflections",
        )

        self._summary = QTextEdit()
        self._summary.setReadOnly(True)
        tabs.addTab(
            self._summary, icon_from_registry(IconRegistry.INFO), "Summary"
        )
        self._tabs = tabs
        right_layout.addWidget(tabs, stretch=1)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        self._status = QStatusBar()
        self.setStatusBar(self._status)

    def _apply_widget_styles(self) -> None:
        """Apply ThemeManager colors to the plot host when available."""
        try:
            colors = ThemeManager.css_colors()
        except Exception:
            return
        base = colors.get("base", "#1e1e22")
        mid = colors.get("mid", "#3f4042")
        text = colors.get("text", "#e1e5e9")
        self._plot.setStyleSheet(
            f"background-color: {base}; border: 1px solid {mid}; color: {text};"
        )

    def _wire_actions(self) -> None:
        self._act_open.triggered.connect(self.open_file_dialog)
        self._act_reload.triggered.connect(self.reload)
        self._act_summary.triggered.connect(lambda: self._tabs.setCurrentIndex(1))
        self._act_quit.triggered.connect(self.close)

        self._act_axes.toggled.connect(self._on_axes_action_toggled)
        self._act_symmetry.toggled.connect(self._on_symmetry_action_toggled)
        self._act_contours.toggled.connect(self._on_contours_action_toggled)

        self._plot.reflection_selected.connect(self._on_plot_selected)
        self._table.selectionModel().selectionChanged.connect(self._on_table_selected)
        self._axes_cb.toggled.connect(self._on_axes_checkbox_toggled)
        self._symmetry_cb.toggled.connect(self._on_symmetry_checkbox_toggled)
        self._contours_cb.toggled.connect(self._on_contours_checkbox_toggled)
        self._marker_spin.valueChanged.connect(self._plot.set_marker_scale)
        self._contrast_spin.valueChanged.connect(self._plot.set_contrast)
        self._zone_combo.currentIndexChanged.connect(self._on_zone_preset_changed)

    def _sync_overlay_checkbox(self, checkbox: QCheckBox, checked: bool) -> None:
        if checkbox.isChecked() != checked:
            checkbox.blockSignals(True)
            checkbox.setChecked(checked)
            checkbox.blockSignals(False)

    def _sync_overlay_action(self, action: QAction, checked: bool) -> None:
        if action.isChecked() != checked:
            action.blockSignals(True)
            action.setChecked(checked)
            action.blockSignals(False)

    def _on_axes_action_toggled(self, checked: bool) -> None:
        self._sync_overlay_checkbox(self._axes_cb, checked)
        self._plot.set_show_axes(checked)

    def _on_symmetry_action_toggled(self, checked: bool) -> None:
        self._sync_overlay_checkbox(self._symmetry_cb, checked)
        self._plot.set_show_symmetry_axes(checked)

    def _on_contours_action_toggled(self, checked: bool) -> None:
        self._sync_overlay_checkbox(self._contours_cb, checked)
        self._plot.set_show_contours(checked)

    def _on_axes_checkbox_toggled(self, checked: bool) -> None:
        self._sync_overlay_action(self._act_axes, checked)
        self._plot.set_show_axes(checked)

    def _on_symmetry_checkbox_toggled(self, checked: bool) -> None:
        self._sync_overlay_action(self._act_symmetry, checked)
        self._plot.set_show_symmetry_axes(checked)

    def _on_contours_checkbox_toggled(self, checked: bool) -> None:
        self._sync_overlay_action(self._act_contours, checked)
        self._plot.set_show_contours(checked)

    def _refresh_empty(self) -> None:
        self._summary.setPlainText("No reflection file loaded.")
        self._details.setPlainText("No reflection selected.")
        self._status.showMessage(
            "Open an MTZ or structure-factor CIF file.  "
            "Scroll to zoom · drag to pan."
        )
        self._table_model.set_data(None)
        self._plot.set_point_cloud(PlotPointCloud())
        self._update_crystal_toolbar(None)

    def _update_crystal_toolbar(self, data: ReflectionData | None) -> None:
        """Refresh crystallographic metadata labels on the second toolbar."""
        lines = format_crystal_toolbar_lines(data)
        self._meta_space_group.setText(lines["space_group"])
        self._meta_unit_cell.setText(lines["unit_cell"])
        self._meta_merged.setText(lines["merged"])
        self._meta_res_header.setText(lines["res_header"])
        self._meta_res_computed.setText(lines["res_computed"])
        self._meta_space_group.setToolTip(lines["space_group"])
        self._meta_unit_cell.setToolTip(lines["unit_cell"])
        self._meta_merged.setToolTip(lines["merged"])
        self._meta_res_header.setToolTip(lines["res_header"])
        self._meta_res_computed.setToolTip(lines["res_computed"])

    def open_file_dialog(self) -> None:
        """Prompt for a reflection file and load it."""
        start_dir = ""
        if self._path is not None:
            start_dir = str(self._path.parent)
        else:
            recent = self._recent_files.get()
            if recent:
                start_dir = str(Path(recent[0]).parent)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Reflection File",
            start_dir,
            "Reflection files (*.mtz *.cif *.fcf *.hkl);;MTZ (*.mtz);;"
            "SF-CIF (*.cif *.fcf);;All files (*)",
        )
        if path:
            self.load_path(Path(path))

    def load_path(self, path: Path | str) -> bool:
        """Load reflections from *path*. Return True on success."""
        p = Path(path).expanduser()
        if not p.is_file():
            QMessageBox.warning(self, "Open failed", f"File not found:\n{p}")
            self._refresh_recent_files_menu()
            return False
        try:
            data = self._loader.load(p)
        except UnsupportedReflectionFormatError as ex:
            QMessageBox.warning(self, "Unsupported format", str(ex))
            return False
        except ReflectionLoadError as ex:
            QMessageBox.critical(self, "Load failed", str(ex))
            return False
        except Exception as ex:
            QMessageBox.critical(self, "Load failed", f"Unexpected error: {ex}")
            log.error(f"Reflection load failed: {ex}")
            return False

        self._data = data
        self._path = p.resolve()
        self._selected_id = None
        self.setWindowTitle(f"Reflection Viewer — {p.name}")
        self._recent_files.add(self._path)
        self._refresh_recent_files_menu()
        self._populate_quantity_combos()
        self._apply_filters()
        return True

    def reload(self) -> None:
        """Reload the current file."""
        if self._path is None:
            self.open_file_dialog()
            return
        self.load_path(self._path)

    def _populate_quantity_combos(self) -> None:
        self._updating = True
        self._quantity_combo.clear()
        self._column_combo.clear()
        if self._data is None:
            self._updating = False
            return
        for q in available_quantities(self._data):
            self._quantity_combo.addItem(q.value, q)
        for label in self._data.column_labels():
            self._column_combo.addItem(label, label)
        for prefer in (DisplayQuantity.F, DisplayQuantity.I, DisplayQuantity.COLUMN):
            idx = self._quantity_combo.findData(prefer)
            if idx >= 0:
                self._quantity_combo.setCurrentIndex(idx)
                break
        self._updating = False

    def _on_zone_preset_changed(self, _index: int = 0) -> None:
        preset = self._zone_combo.currentData()
        custom = preset == ZonePreset.CUSTOM
        for w in (self._nz_x, self._nz_y, self._nz_z):
            w.setEnabled(custom)

    def _read_filter_state(self) -> ViewerFilterState:
        preset_data = self._zone_combo.currentData()
        if isinstance(preset_data, ZonePreset):
            preset = preset_data
        else:
            preset = ZonePreset(str(preset_data))
        level = int(self._level_spin.value())
        if preset == ZonePreset.CUSTOM:
            zone = ZoneSpec(
                normal=(
                    int(self._nz_x.value()),
                    int(self._nz_y.value()),
                    int(self._nz_z.value()),
                ),
                level=level,
                preset=ZonePreset.CUSTOM,
            )
        else:
            zone = ZoneSpec.from_preset(preset, level=level)

        quantity_data = self._quantity_combo.currentData()
        if isinstance(quantity_data, DisplayQuantity):
            quantity = quantity_data
        elif quantity_data:
            quantity = DisplayQuantity(str(quantity_data))
        else:
            quantity = DisplayQuantity.COLUMN
        column = self._column_combo.currentData()
        d_min = self._dmin_spin.value() or None
        d_max = self._dmax_spin.value() or None
        if d_min is not None and d_min <= 0:
            d_min = None
        if d_max is not None and d_max <= 0:
            d_max = None
        return ViewerFilterState(
            zone=zone,
            quantity=quantity,
            column_label=column,
            d_min=d_min,
            d_max=d_max,
            expand_symmetry=bool(self._expand_symmetry_cb.isChecked()),
        )

    def _apply_filters(self) -> None:
        if self._data is None:
            self._refresh_empty()
            return
        self._filter = self._read_filter_state()
        prepared = prepare_view(self._data, self._filter)
        self._plot.set_point_cloud(prepared.cloud)
        if self._selected_id is not None:
            self._plot.set_selected_observation(self._selected_id)
        value_cols = []
        if prepared.quantity is not None:
            value_cols.append(prepared.quantity.value_label)
            if prepared.quantity.sigma_label:
                value_cols.append(prepared.quantity.sigma_label)
        self._table_model.set_data(
            self._data,
            row_indices=prepared.row_indices,
            value_columns=value_cols or None,
        )
        self._summary.setPlainText(
            format_summary_text(
                self._data,
                displayed_count=prepared.displayed_count,
                selection=prepared.quantity,
            )
        )
        self._details.setPlainText(
            format_selection_details(
                self._data,
                self._selected_id,
                selection=prepared.quantity,
            )
        )
        self._update_crystal_toolbar(self._data)
        zone_name = getattr(
            self._filter.zone.preset, "value", str(self._filter.zone.preset)
        )
        self._status.showMessage(
            f"{self._path.name if self._path else ''}  |  "
            f"displayed {prepared.displayed_count} / {self._data.n_reflections}  |  "
            f"zone {zone_name} level {self._filter.zone.level}  |  "
            f"scroll to zoom · drag to pan"
        )

    def _on_plot_selected(self, observation_id: int) -> None:
        self._selected_id = int(observation_id)
        self._plot.set_selected_observation(self._selected_id)
        row = self._table_model.row_for_observation_id(self._selected_id)
        if row is not None:
            self._updating = True
            self._table.selectRow(row)
            self._table.scrollTo(self._table_model.index(row, 0))
            self._updating = False
        prepared_q = None
        if self._data is not None:
            from molib.xtal.reflections.quantity import resolve_quantity

            prepared_q = resolve_quantity(
                self._data,
                self._filter.quantity,
                column_label=self._filter.column_label,
            )
        self._details.setPlainText(
            format_selection_details(
                self._data, self._selected_id, selection=prepared_q
            )
        )

    def _on_table_selected(self, *_args) -> None:
        if self._updating:
            return
        indexes = self._table.selectionModel().selectedRows()
        if not indexes:
            return
        obs = indexes[0].data(OBSERVATION_ID_ROLE)
        if obs is None:
            return
        self._on_plot_selected(int(obs))
