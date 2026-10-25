"""Specs for the ProcessingLab window's controls, in one place.

Every spin box, button, check box, combo box and text field of
:class:`~diffraction_lab.processinglab_window.ProcessingLabWindow` is
described here: range, default, label, tooltip, placeholder and the slot it
calls. The window keeps them as ``self.specs`` (one group per tab,
``self.specs.spots.threshold``) and builds the widgets with PicoUI's
``create_*_from_spec`` helpers, or the few below where PicoUI has no
equivalent (editable combo boxes, an "unset" minimum, text fields).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from picoui.specs.widgets import ButtonSpec, CheckBoxSpec, ComboBoxSpec, DoubleSpinBoxSpec, SpinBoxSpec
from picoui.widget.helper import (
    create_button_from_spec,
    create_checkbox_from_spec,
    create_combo_box,
    create_double_spinbox_from_spec,
)
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QLineEdit, QPlainTextEdit, QPushButton

from diffraction_lab.processinglab_run import REFINABLE_PARAMETERS

if TYPE_CHECKING:
    from diffraction_lab.processinglab_window import ProcessingLabWindow

#: Largest frame number the frame boxes accept.
MAX_FRAME = 999_999

REFINE_LABELS = {
    "orientation": "Orientation",
    "cell": "Unit cell",
    "distance": "Distance",
    "beam": "Beam centre",
    "tilt_twist": "Tilt / twist",
    "mosaicity": "Mosaicity",
}


@dataclass
class OptionalDoubleSpinBoxSpec(DoubleSpinBoxSpec):
    """A double spin box whose minimum means "not set", shown as *unset_text* ("all data")."""

    unset_text: str = ""


@dataclass
class EditableComboBoxSpec(ComboBoxSpec):
    """A combo box the user can also type into (e.g. a space group)."""

    editable: bool = True


@dataclass
class TextFieldSpec:
    """A one-line (or, with *multiline*, plain-text) entry field."""

    placeholder: str = ""
    tooltip: str = ""
    multiline: bool = False
    max_height: int | None = None


def button(label: str, slot: Callable | None = None, tooltip: str = "") -> ButtonSpec:
    """A push button that is not the dialog's default button (Enter stays free)."""
    return ButtonSpec(label=label, slot=slot, tooltip=tooltip, default=False, checkable=False)


def frame_spec(tooltip: str = "") -> SpinBoxSpec:
    """A frame-number box: 1 … :data:`MAX_FRAME`, step 1."""
    return SpinBoxSpec(min_val=1, max_val=MAX_FRAME, step=1, tooltip=tooltip)


# ------------------------------------------------------------------ builders


def create_checkbox(spec: CheckBoxSpec) -> QCheckBox:
    """PicoUI's check box, with the box left of its label as elsewhere in DiffractionLab."""
    check = create_checkbox_from_spec(spec)
    check.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
    return check


def create_button(spec: ButtonSpec) -> QPushButton:
    """PicoUI's button from *spec*."""
    return create_button_from_spec(spec)


def create_optional_double_spinbox(spec: OptionalDoubleSpinBoxSpec) -> QDoubleSpinBox:
    """PicoUI's double spin box, showing *spec.unset_text* at its minimum."""
    spin = create_double_spinbox_from_spec(spec)
    if spec.unset_text:
        spin.setSpecialValueText(spec.unset_text)
    return spin


def create_editable_combo(spec: EditableComboBoxSpec) -> QComboBox:
    """PicoUI's combo box, editable if *spec.editable*."""
    combo = create_combo_box(spec=spec)
    combo.setEditable(spec.editable)
    return combo


def create_text_field(spec: TextFieldSpec) -> QLineEdit | QPlainTextEdit:
    """A line edit (or plain-text edit) from *spec*."""
    edit: QLineEdit | QPlainTextEdit = QPlainTextEdit() if spec.multiline else QLineEdit()
    edit.setPlaceholderText(spec.placeholder)
    if spec.tooltip:
        edit.setToolTip(spec.tooltip)
    if spec.max_height is not None:
        edit.setMaximumHeight(spec.max_height)
    return edit


# ------------------------------------------------------------------ the window's specs


@dataclass
class SetupSpecs:
    """1. Setup."""

    frames: SpinBoxSpec
    work_folder: TextFieldSpec
    browse: ButtonSpec
    new_session: ButtonSpec
    reveal: ButtonSpec
    run_all: ButtonSpec


@dataclass
class SpotsSpecs:
    """2. Find spots."""

    frames: SpinBoxSpec
    threshold: DoubleSpinBoxSpec
    max_spots: SpinBoxSpec
    find: ButtonSpec
    show_spots: CheckBoxSpec


@dataclass
class IndexSpecs:
    """3. Index."""

    threshold: DoubleSpinBoxSpec
    max_cell_edge: OptionalDoubleSpinBoxSpec
    index: ButtonSpec
    reindex: ButtonSpec


@dataclass
class StrategySpecs:
    """4. Strategy."""

    space_group: EditableComboBoxSpec
    target: DoubleSpinBoxSpec
    anomalous: CheckBoxSpec
    within: CheckBoxSpec
    resolution: OptionalDoubleSpinBoxSpec
    calculate: ButtonSpec
    use_frames: ButtonSpec


@dataclass
class IntegrateSpecs:
    """5. Integrate."""

    space_group: EditableComboBoxSpec
    integrate: ButtonSpec
    untrusted: TextFieldSpec
    exclude_resolution: TextFieldSpec
    exclude_ice: CheckBoxSpec
    report: ButtonSpec
    view_mtz: ButtonSpec
    show_defpix: CheckBoxSpec
    refine: dict[str, CheckBoxSpec]
    rerun_space_group: TextFieldSpec
    rerun_cell: TextFieldSpec
    rerun: ButtonSpec


@dataclass
class SymmetrySpecs:
    """6. Symmetry."""

    run: ButtonSpec
    use_space_group: ButtonSpec
    log: ButtonSpec


@dataclass
class ScaleSpecs:
    """7. Scale."""

    resolution: OptionalDoubleSpinBoxSpec
    anomalous: CheckBoxSpec
    run: ButtonSpec
    view_merged: ButtonSpec
    log: ButtonSpec


@dataclass
class ProcessingLabSpecs:
    """Every control of the ProcessingLab window, one group per tab."""

    setup: SetupSpecs
    spots: SpotsSpecs
    index: IndexSpecs
    strategy: StrategySpecs
    integrate: IntegrateSpecs
    symmetry: SymmetrySpecs
    scale: ScaleSpecs


def processinglab_specs(w: ProcessingLabWindow) -> ProcessingLabSpecs:
    """The specs of window *w*, with its slots.

    :param w: The window (its methods are the slots).
    :returns: One spec per control.
    """
    # Slots read ``w._s`` when clicked: the window replaces its session.
    from diffraction_lab.processinglab_window import _open  # the window's "open in Finder" helper

    return ProcessingLabSpecs(
        setup=SetupSpecs(
            frames=frame_spec(),
            work_folder=TextFieldSpec(tooltip="Sessions are created here as session_YYYYMMDD_HHMMSS"),
            browse=button("Browse…", w._on_browse_work),
            new_session=button("New session", w._on_new_session),
            reveal=button("Reveal", lambda: _open(w._s.folder)),
            run_all=button("Run all steps with these settings", w._on_run_all,
                           "Find spots → index (suggested solution) → integrate (its first space group) "
                           "→ Pointless → Aimless"),
        ),
        spots=SpotsSpecs(
            frames=frame_spec(),
            threshold=DoubleSpinBoxSpec(min_val=1, max_val=100, step=1, value=5),
            max_spots=SpinBoxSpec(min_val=50, max_val=100_000, step=10, value=20_000),
            find=button("Find spots", w._on_find_spots),
            show_spots=CheckBoxSpec(label="Show spots on the image (current frame)",
                                    slot=lambda _state: w._update_spot_overlay()),
        ),
        index=IndexSpecs(
            threshold=DoubleSpinBoxSpec(min_val=1.0, max_val=200.0, step=1.0, value=15.0),
            max_cell_edge=OptionalDoubleSpinBoxSpec(min_val=0.0, max_val=2000.0, step=1.0, value=0.0,
                                                    suffix=" Å", unset_text="200 Å (default)",
                                                    tooltip="Longest cell edge searched; raise it for very "
                                                            "long axes"),
            index=button("Index", lambda: w._on_index(None)),
            reindex=button("Use the highlighted solution (re-index)", w._on_reindex_selected,
                           "Re-run AUTOINDEX choosing this lattice (AUTOINDEX CHARACTER n)"),
        ),
        strategy=StrategySpecs(
            space_group=EditableComboBoxSpec(tooltip="Higher symmetry needs less rotation: use the space "
                                                     "group you expect."),
            target=DoubleSpinBoxSpec(min_val=10.0, max_val=100.0, step=1.0, value=99.0, decimals=1, suffix=" %"),
            anomalous=CheckBoxSpec(label="Anomalous (both Friedel mates of each pair)"),
            within=CheckBoxSpec(label="Within Setup's frames to process (otherwise plan on a full turn)",
                                checked_state=True),
            resolution=OptionalDoubleSpinBoxSpec(min_val=0.0, max_val=20.0, step=1.0, value=0.0, suffix=" Å",
                                                 unset_text="detector edge"),
            calculate=button("Calculate strategy", w._on_strategy),
            use_frames=button("Integrate these frames", w._on_use_strategy_frames,
                              "Set Setup's frames to process to the frames the strategy chose"),
        ),
        integrate=IntegrateSpecs(
            space_group=EditableComboBoxSpec(tooltip="Initial space group: any group of the chosen lattice. "
                                                     "Pointless (Symmetry tab) tests it after integration."),
            integrate=button("Integrate", w._on_integrate),
            untrusted=TextFieldSpec(placeholder="One per line, XDS pixels (x along fast, from 1), e.g.\n"
                                                "RECTANGLE 1200 1300 1150 1250\n"
                                                "ELLIPSE x1 x2 y1 y2   ·   QUADRILATERAL x1 y1 … x4 y4",
                                    multiline=True, max_height=72),
            exclude_resolution=TextFieldSpec(placeholder="e.g. 3.93 3.87, 2.70 2.64 (Å)"),
            exclude_ice=CheckBoxSpec(label="Exclude the usual ice-ring ranges"),
            report=button("Open report", lambda: _open(w._s.integrated.directory / "process.md"
                                                       if w._s.integrated else None)),
            view_mtz=button("Open MTZ in viewer",
                            lambda: w._view_mtz(w._s.integrated.mtz if w._s.integrated else None)),
            show_defpix=CheckBoxSpec(label="Show the ignored pixels on the image (magenta)",
                                     tooltip="DEFPIX's control image, defpix.cbf, over the main window's image",
                                     slot=w._update_defpix_overlay),
            refine={name: CheckBoxSpec(label=REFINE_LABELS[name], checked_state=name in ("orientation", "mosaicity"))
                    for name in REFINABLE_PARAMETERS},
            rerun_space_group=TextFieldSpec(placeholder="optional; filled from Pointless when it is safe"),
            rerun_cell=TextFieldSpec(placeholder="optional a b c α β γ (default: the refined cell)"),
            rerun=button("Re-integrate", w._on_reintegrate),
        ),
        symmetry=SymmetrySpecs(
            run=button("Run Pointless", w._on_pointless),
            use_space_group=button("Re-integrate in this space group…", w._on_use_pointless_space_group),
            log=button("Open log", lambda: _open(w._s.pointless.log if w._s.pointless else None)),
        ),
        scale=ScaleSpecs(
            resolution=OptionalDoubleSpinBoxSpec(min_val=0.0, max_val=20.0, step=1.0, value=0.0, suffix=" Å",
                                                 unset_text="all data"),
            anomalous=CheckBoxSpec(label="Anomalous (keep I+ and I− separate)"),
            run=button("Run Aimless", w._on_aimless),
            view_merged=button("Open merged MTZ in viewer",
                               lambda: w._view_mtz(w._s.aimless.hklout if w._s.aimless else None)),
            log=button("Open log", lambda: _open(w._s.aimless.log if w._s.aimless else None)),
        ),
    )
