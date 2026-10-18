"""
Loading CBF, TIFF and H5 files in the style of Mosflm
Mark Brooks (c) 2025 Princeton, U.S.A.
"""

import argparse
import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# When this file is run as a script, Python puts ``imosflm/`` on sys.path and
# ``imosflm.py`` would shadow the ``imosflm`` package. Prefer the repo root.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path[:] = [p for p in sys.path if Path(p).resolve() != _HERE]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import h5py
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSlider,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtGui import (
    QAction,
    QColor,
    QIcon,
    QImage,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtCore import Qt, QPoint, QSettings, QSize

from diffraction_lab.beam import BeamParameters
from diffraction_lab.detector import DetectorParameters
from diffraction_lab.image_canvas import DiffractionImageCanvas, build_resolution_rings
from diffraction_lab.indexing import IndexingGeometry, as_float
from diffraction_lab.processing_wizard import ProcessingWizard
from diffraction_lab.session import (
    DiffractionLabSession,
    ViewerState,
    load_session,
    save_session,
)
from diffraction_lab.session.io import SESSION_SUFFIX
from diffraction_lab.session.model import ProcessingState

from PIL import Image, ImageOps
import fabio  # Import FabIO
import matplotlib.pyplot as plt
import cv2
import numpy as np

try:
    from picoui.icons import IconRegistry
    from picoui.theme import ThemeManager, ThemeType
    from picoui.widget.viewer_chrome import icon_label, standard_zoom_actions

    _HAS_PICOUI = True
except ImportError:  # pragma: no cover - optional when run outside ElMo venv
    IconRegistry = None  # type: ignore[assignment, misc]
    ThemeManager = None  # type: ignore[assignment, misc]
    ThemeType = None  # type: ignore[assignment, misc]
    icon_label = None  # type: ignore[assignment, misc]
    standard_zoom_actions = None  # type: ignore[assignment, misc]
    _HAS_PICOUI = False


def _registry_icon(icon_key: str) -> QIcon:
    """Return an IconRegistry icon, or an empty icon if PicoUI is unavailable."""
    if not _HAS_PICOUI or not icon_key:
        return QIcon()
    return IconRegistry.get_icon_safe(icon_key)


def _icon_key(attr: str, fallback: str = "") -> str:
    """Resolve an IconRegistry attribute name when PicoUI is available."""
    if not _HAS_PICOUI:
        return fallback
    return getattr(IconRegistry, attr, fallback) or fallback


def _readout_widget(icon_key: str, text: str) -> tuple[QWidget, QLabel]:
    """Build a small icon + text readout; return ``(host, text_label)``."""
    if _HAS_PICOUI and icon_label is not None:
        return icon_label(icon_key, text, margin_right=8)
    host = QWidget()
    row = QHBoxLayout(host)
    row.setContentsMargins(0, 0, 8, 0)
    row.setSpacing(4)
    if icon_key:
        icon = QLabel()
        icon.setPixmap(_registry_icon(icon_key).pixmap(16, 16))
        row.addWidget(icon)
    label = QLabel(text)
    row.addWidget(label)
    return host, label


def _theme_colors() -> dict[str, str]:
    """Return ThemeManager CSS colors, with dark fallbacks when unavailable."""
    if _HAS_PICOUI:
        try:
            return dict(ThemeManager.css_colors())
        except Exception:
            pass
    return {
        "base": "#202124",
        "button": "#3f4042",
        "window": "#202124",
        "text": "#e1e5e9",
        "mid": "#3f4042",
        "highlight": "#8ab4f7",
        "highlighted_text": "#202124",
        "alternate": "#ffffff",
    }


def is_readable_hdf5(path: Path) -> bool:
    """Return True if *path* opens as a complete HDF5 file."""
    try:
        with h5py.File(path, "r"):
            return True
    except OSError as ex:
        logging.warning("HDF5 not readable yet (%s): %s", path, ex)
        return False


def _is_broken_symlink(path: Path) -> bool:
    """Return True if *path* is a symlink whose target is missing."""
    try:
        return path.is_symlink() and not path.exists()
    except OSError:
        return False


def find_external_data_file(master_dir: Path, filename: str) -> Path | None:
    """Locate an externally linked NeXus data file near a master HDF5.

    Handles common browser download quirks:

    - files left in ``name.h5.download/name.h5``
    - Safari/Chrome renames such as ``name-2.h5`` / ``name (1).h5`` when a
      broken symlink already occupies ``name.h5``
    """
    stem = Path(filename).stem
    suffix = Path(filename).suffix  # typically .h5

    candidates: list[Path] = [
        master_dir / filename,
        master_dir / f"{filename}.download" / filename,
        master_dir / f"{stem}.download" / filename,
        master_dir / stem / filename,
    ]
    # Browser duplicate-name renames: stem-2.h5, stem-3.h5, stem (1).h5, …
    for n in range(2, 10):
        candidates.append(master_dir / f"{stem}-{n}{suffix}")
        candidates.append(master_dir / f"{stem} ({n - 1}){suffix}")

    try:
        for sub in master_dir.iterdir():
            if not sub.is_dir():
                continue
            candidates.append(sub / filename)
            for n in range(2, 6):
                candidates.append(sub / f"{stem}-{n}{suffix}")
    except OSError:
        pass

    # Exact-name recursive matches, then stem-* renames in the master directory.
    try:
        candidates.extend(sorted(master_dir.glob(f"**/{filename}")))
        candidates.extend(sorted(master_dir.glob(f"{stem}-*{suffix}")))
    except OSError:
        pass

    seen: set[Path] = set()
    readable: list[Path] = []
    for cand in candidates:
        try:
            if _is_broken_symlink(cand):
                continue
            if not cand.is_file():
                continue
            resolved = cand.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if is_readable_hdf5(resolved):
                readable.append(resolved)
        except OSError:
            continue

    if not readable:
        return None

    # Prefer the canonical filename if it is a real, readable file; otherwise
    # the largest readable candidate (complete re-download vs partial).
    for path in readable:
        if path.name == filename and not path.is_symlink():
            return path
    for path in readable:
        if path.name == filename:
            return path
    return max(readable, key=lambda p: p.stat().st_size)


def ensure_nexus_external_links(
    master_path: Path,
) -> tuple[dict[str, Path], list[str], list[str]]:
    """Resolve ``/entry/data`` ExternalLinks for a NeXus master file.

    Creates or repairs sibling symlinks when a linked file is found under an
    alternate name/location (browser ``*.download`` folders or ``*-2.h5``
    renames).

    :returns: ``(resolved, missing_filenames, incomplete_filenames)``
    """
    master_path = Path(master_path).resolve()
    master_dir = master_path.parent
    resolved: dict[str, Path] = {}
    missing: list[str] = []
    incomplete: list[str] = []

    with h5py.File(master_path, "r") as handle:
        if "entry/data" not in handle:
            return resolved, missing, incomplete
        group = handle["entry/data"]
        for name in group.keys():
            link = group.get(name, getlink=True)
            if not isinstance(link, h5py.ExternalLink):
                continue
            target_name = str(link.filename)
            expected = master_dir / target_name
            found = find_external_data_file(master_dir, target_name)
            if found is None:
                # Distinguish "present but truncated" from truly missing when
                # only a broken symlink / partial download occupies the name.
                partial = None
                for cand in (
                    expected,
                    master_dir / f"{target_name}.download" / target_name,
                ):
                    try:
                        if cand.is_file() and not is_readable_hdf5(cand):
                            partial = cand
                            break
                    except OSError:
                        continue
                if partial is not None:
                    incomplete.append(target_name)
                    logging.error(
                        "Incomplete/truncated data file for %s at %s",
                        name,
                        partial,
                    )
                else:
                    missing.append(target_name)
                    logging.error(
                        "Missing external data file for %s: expected %s next to %s",
                        name,
                        target_name,
                        master_dir,
                    )
                continue

            # Repair broken / stale symlink so h5py ExternalLinks also work.
            try:
                if _is_broken_symlink(expected) or (
                    expected.is_symlink() and expected.resolve() != found
                ):
                    expected.unlink()
                    logging.info("Removed stale symlink %s", expected)
                if not expected.exists():
                    expected.symlink_to(found)
                    logging.info("Created symlink %s -> %s", expected, found)
                elif expected.resolve() != found and expected.name != found.name:
                    # Canonical name already exists as a different real file;
                    # open the resolved path directly.
                    logging.info(
                        "Using %s for link %s (canonical name occupied)",
                        found,
                        name,
                    )
            except OSError as ex:
                logging.warning(
                    "Could not repair symlink %s -> %s (%s); "
                    "will open data file directly",
                    expected,
                    found,
                    ex,
                )
            resolved[name] = found
    return resolved, missing, incomplete


def list_entry_data_names(master_path: Path) -> list[str]:
    """List ``/entry/data`` image link names without opening external files."""
    names: list[str] = []
    with h5py.File(master_path, "r") as handle:
        if "entry/data" not in handle:
            return names
        group = handle["entry/data"]
        for name in group.keys():
            link = group.get(name, getlink=True)
            if isinstance(link, h5py.ExternalLink):
                names.append(name)
            elif isinstance(link, h5py.HardLink):
                # Skip non-image auxiliaries such as omega.
                if name.lower().startswith("data"):
                    names.append(name)
    return names

# Set environment variables
os.environ["QT_LOGGING_RULES"] = "*.debug=false"
os.environ['MAINTAINER'] = 'mosflm@mrc-lmb.cam.ac.uk'
os.environ['MOSFLM_VERSION_REQUIRED'] = '7.4'
os.environ['TKIMAGELOAD'] = '0'
os.environ['EXPERTDETECTORSETTINGS'] = '0'
os.environ['SPIRAL'] = '0'
os.environ['HDF5file'] = '0'

# Debugging setup
debugging = os.getenv('MOSFLM_DEBUG', '0') == '1'

# CCP4i2 mode setup
ccp4i2 = os.getenv('CCP4I2', '0') == '1'
if ccp4i2:
    logging.debug("ccp4i2 mode: ON")

# Fastload mode setup
fastload = os.getenv('FASTLOAD', '0') == '1'
if fastload:
    logging.debug("fastload mode: ON")


def calculate_two_theta_and_resolution(pixel_x, pixel_y, metadata):
    """
    Compute the scattering angle (theta) and resolution at a given pixel.

    Parameters:
        pixel_x (float): X-coordinate of the pixel.
        pixel_y (float): Y-coordinate of the pixel.
        metadata (dict): Dictionary containing extracted instrument parameters.

    Returns:
        tuple: (theta (float), resolution (float))
            - theta: Scattering angle in degrees.
            - resolution: Resolution in Ångstroms.

    Raises:
        ValueError: If any required metadata key is missing or if detector_distance is zero.
    """
    logging.info("----------------------------------------")
    logging.info(f"Computing theta and resolution for pixel \tx: {pixel_x:.0f}, y: {pixel_y:.0f}")

    required_keys = ["beam_centre_x", "beam_centre_y", "detector_distance",
                     "incident_wavelength", "detector_x_pixel_size"]

    # Ensure all required keys exist
    for key in required_keys:
        if key not in metadata:
            logging.error("Missing required metadata key: %s", key)
            raise ValueError(f"Missing required metadata key: {key}")

    # Extract parameters
    beam_center_x = metadata["beam_centre_x"]
    beam_center_y = metadata["beam_centre_y"]
    detector_distance = metadata["detector_distance"]
    wavelength = metadata["incident_wavelength"]
    pixel_size = metadata["detector_x_pixel_size"]

    logging.debug("Metadata extracted: Beam Center (%s, %s), Detector Distance: %s mm, Wavelength: %s Å, Pixel Size: %s mm",
                  beam_center_x, beam_center_y, detector_distance, wavelength, pixel_size)

    if detector_distance == 0:
        logging.critical("Detector distance is zero! This will cause a division error.")
        raise ValueError("Detector distance cannot be zero.")

    # Convert pixel coordinates to mm distances from beam center
    delta_x = (pixel_x - beam_center_x) * pixel_size
    delta_y = (pixel_y - beam_center_y) * pixel_size
    logging.debug("Pixel offsets from beam center: ΔX = %s mm, ΔY = %s mm", delta_x, delta_y)

    # Compute R (distance from beam center to the pixel in mm)
    R = np.hypot(delta_x, delta_y)
    logging.info(f"Computed radial distance: \t\t\t\tR = \t{R:.0f} mm")

    # Compute theta in radians
    two_theta_rad = calculate_two_theta_radians(R, detector_distance)
    two_theta_deg = np.degrees(two_theta_rad)
    logging.info(f"Computed scattering angle: \t\t\tθ = \t{two_theta_deg:.2f}°")

    # Compute resolution using Bragg's Law
    if two_theta_rad == 0:
        resolution = float('inf')
        logging.warning("Theta is zero, resolution is set to infinity.")
    else:
        resolution = wavelength / (2 * np.sin(two_theta_rad / 2 )) # two_theta / 2 = theta
    logging.info(f"Computed resolution: \t\t\t\t\t{resolution:.2f} Å")
    return two_theta_deg, resolution


def calculate_two_theta_radians(R, detector_distance):
    """ calculate_two_theta """
    logging.info(f"calculate_two_theta")
    two_theta_rad = np.arctan(R / detector_distance)
    return two_theta_rad


def calculate_radius_pixels(resolution, incident_wavelength, detector_distance, x_pixel_size):
    """Calculate the radius in pixels for a given resolution."""
    two_theta_rad = np.arcsin(incident_wavelength / (2 * resolution))
    radius_mm = detector_distance * np.tan(two_theta_rad * 2)
    radius_pixels = int(radius_mm / x_pixel_size)  # Convert to pixels
    return radius_pixels  # Fix: Return the correct variable

def add_resolution_rings(image, beam_center, detector_distance, x_pixel_size, incident_wavelength, show_rings=True):
    """Display the image with resolution rings using OpenCV and PIL."""
    try:
        image = image.convert("RGB")  # Ensure it's RGB mode
        img_np = np.array(image)  # Convert to NumPy array for OpenCV processing

        if show_rings:
            # Define resolution values in Ångströms
            resolutions = [1.0, 2.0, 3.0]
            radii = []

            # Compute radii for resolution rings
            for resolution in resolutions:
                radius_pixels = calculate_radius_pixels(resolution, incident_wavelength, detector_distance, x_pixel_size)
                radii.append(radius_pixels)

            # Draw resolution rings using OpenCV
            center = (int(beam_center[0]), int(beam_center[1]))
            for i, radius in enumerate(radii):
                cv2.circle(img_np, center, radius, (255, 0, 0), 2)  # Draw red rings
                label_position = (center[0] + radius + 5, center[1])
                cv2.putText(img_np, f"{resolutions[i]} Ang", label_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)

        # Convert back to PIL for displaying or saving
        result_image = Image.fromarray(img_np)
        return result_image  # You can save or display it using result_image.show()
    except Exception as ex:
        print(f"Error: {ex} occurred")
        return image.convert("RGB") if hasattr(image, "convert") else image


def add_indexing_spot_markers(
    image,
    spots_px: list[tuple[float, float, bool]] | None,
    *,
    cross_half: int = 8,
) -> Image.Image:
    """Draw indexing spots on *image* (Mosflm colours).

    ``predicted`` True → blue (indexed/predicted fulls); False → red (found).
    """
    image = image.convert("RGB")
    if not spots_px:
        return image
    img_np = np.array(image)
    height, width = img_np.shape[:2]
    for x_px, y_px, predicted in spots_px:
        x = int(round(float(x_px)))
        y = int(round(float(y_px)))
        if x < 0 or y < 0 or x >= width or y >= height:
            continue
        # OpenCV BGR: Mosflm found=red, predicted fulls=blue
        color = (255, 0, 0) if predicted else (0, 0, 255)
        cv2.drawMarker(
            img_np,
            (x, y),
            color,
            markerType=cv2.MARKER_CROSS,
            markerSize=cross_half * 2,
            thickness=2,
            line_type=cv2.LINE_AA,
        )
    return Image.fromarray(img_np)


def traverse_hdf5(group, path, beam_x=None, beam_y=None, detector_distance=None,
                  incident_wavelength=None, x_pixel_size=None, y_pixel_size=None):
    """Recursively print attributes and datasets in an HDF5 group."""
    # Print attributes of the current group
    if group.attrs:
        logging.info(f"\nAttributes in {path}:")
        for attr, value in group.attrs.items():
            logging.info(f"  {attr}: {value}")

    # Iterate through datasets and subgroups
    for name, item in group.items():
        full_path = f"{path}/{name}"

        if isinstance(item, h5py.Group):  # If it's a subgroup, recurse
            logging.info(f"\n--- Entering {full_path} ---")
            beam_x, beam_y, detector_distance, incident_wavelength, x_pixel_size, y_pixel_size = \
                traverse_hdf5(item, full_path, beam_x, beam_y, detector_distance,
                              incident_wavelength, x_pixel_size, y_pixel_size)

        elif isinstance(item, h5py.Dataset):  # If it's a dataset, print its value
            try:
                value = item[()]  # Extract dataset value

                if full_path == r"/entry/instrument/detector/beam_centre_x":
                    beam_x = value
                elif full_path == r"/entry/instrument/detector/beam_centre_y":
                    beam_y = value
                elif full_path == r"/entry/instrument/detector/detector_distance":
                    detector_distance = value
                elif full_path == r"/entry/instrument/beam/incident_wavelength":
                    incident_wavelength = value
                elif full_path == r"/entry/instrument/detector/x_pixel_size":
                    x_pixel_size = value
                elif full_path == r"/entry/instrument/detector/y_pixel_size":
                    y_pixel_size = value
                logging.info(f"{full_path}: {value}")

            except Exception as e:
                logging.info(f"Could not read {full_path}: {e}")

    return beam_x, beam_y, detector_distance, incident_wavelength, x_pixel_size, y_pixel_size


def pil_to_qpixmap(image):
    """
    Convert a PIL image (in RGB mode) to a QPixmap.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    width, height = image.size
    data = image.tobytes("raw", "RGB")
    qimage = QImage(data, width, height, 3 * width, QImage.Format_RGB888)
    return QPixmap.fromImage(qimage)


class RecentFilesManager:
    """Persist recently opened image paths in ``QSettings``."""

    MAX_RECENT = 10
    ORG = "DiffractionLab"
    APP = "preferences"
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


class RecentSessionsManager:
    """Persist recently opened Session paths in ``QSettings``."""

    MAX_RECENT = 10
    ORG = "DiffractionLab"
    APP = "preferences"
    KEY = "recent_sessions"

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
        """Move *file_path* to the front of the recent sessions list."""
        if not file_path:
            return
        resolved = str(Path(file_path).expanduser().resolve())
        paths = self._read()
        if resolved in paths:
            paths.remove(resolved)
        paths.insert(0, resolved)
        self._write(paths[: self.MAX_RECENT])

    def get(self) -> list[str]:
        """Return existing recent session paths, pruning missing entries."""
        paths = self._read()
        existing = [p for p in paths if Path(p).exists()]
        if len(existing) != len(paths):
            self._write(existing)
        return existing

    def clear(self) -> None:
        """Clear the recent-sessions list."""
        self._write([])


class DiffractionLabApp(QMainWindow):
    """DiffractionLab App"""
    def __init__(self):
        super().__init__()
        self.beam_parameters = BeamParameters()
        self.detector = DetectorParameters()
        self.setWindowTitle("DiffractionLab")
        self.setGeometry(100, 100, 900, 1000)
        self.current_image = None
        self.current_file = None
        self._raw_frame: np.ndarray | None = None
        self.dataset = None
        self._h5_master = None
        self._h5_data = None
        self._external_data_paths: dict[str, Path] = {}
        self.num_frames = 0
        self.slice_size = 10  # Number of frames to load at once
        self.image_inverted = False
        self.show_rings = True  # Default to showing rings
        self.show_indexing_spots = True
        self._recent_files = RecentFilesManager()
        self._recent_sessions = RecentSessionsManager()
        self._session_path: Path | None = None
        self._session_dirty = False
        self._processing_wizard: ProcessingWizard | None = None
        self._indexing_spot_overlay: list[tuple[float, float, bool]] | None = None
        self._indexing_solution = None

        if _HAS_PICOUI:
            self.setWindowIcon(_registry_icon(IconRegistry.DATASET_PROCESSING))

        self._build_actions()
        self._build_menus()
        self._build_toolbar()
        self._build_central_ui()
        self._apply_widget_styles()

        status = QStatusBar()
        self.setStatusBar(status)
        status.showMessage("Open a CBF, HDF5, TIFF, or image file to begin.")

    def _build_actions(self) -> None:
        """Create QActions backed by IconRegistry icons."""
        self._act_open = QAction(
            _registry_icon(
                IconRegistry.FOLDER_OPENED if _HAS_PICOUI else ""
            ),
            "&Open…",
            self,
        )
        self._act_open.setShortcut("Ctrl+O")
        self._act_open.setStatusTip("Open a diffraction image or HDF5 dataset")
        self._act_open.triggered.connect(self.open_file)

        self._act_new_session = QAction("&New Session", self)
        self._act_new_session.setShortcut("Ctrl+N")
        self._act_new_session.setStatusTip("Clear the viewer and start a new Session")
        self._act_new_session.triggered.connect(self.new_session)

        self._act_open_session = QAction("Open &Session…", self)
        self._act_open_session.setShortcut("Ctrl+Shift+O")
        self._act_open_session.setStatusTip(
            "Open a DiffractionLab Session (.dlsession)"
        )
        self._act_open_session.triggered.connect(self.open_session_dialog)

        self._act_save_session = QAction("&Save Session", self)
        self._act_save_session.setShortcut("Ctrl+S")
        self._act_save_session.setStatusTip("Save the current Session")
        self._act_save_session.setEnabled(False)
        self._act_save_session.triggered.connect(self.save_session_action)

        self._act_save_session_as = QAction("Save Session &As…", self)
        self._act_save_session_as.setShortcut("Ctrl+Shift+S")
        self._act_save_session_as.setStatusTip(
            "Save the current Session under a new name"
        )
        self._act_save_session_as.triggered.connect(self.save_session_as_action)

        self._act_save_spotfile = QAction("Save Spot &File…", self)
        self._act_save_spotfile.setStatusTip(
            "Save the Mosflm spot list (.spt) from Find spots to a chosen path"
        )
        self._act_save_spotfile.triggered.connect(self.save_spotfile_action)

        self._act_reload = QAction(
            _registry_icon(_icon_key("MDI_REFRESH", _icon_key("REFRESH"))),
            "&Reload",
            self,
        )
        self._act_reload.setShortcut(QKeySequence.StandardKey.Refresh)
        self._act_reload.setStatusTip("Reload the current image or HDF5 dataset")
        self._act_reload.triggered.connect(self.reload_file)

        self._act_invert = QAction(
            _registry_icon(
                IconRegistry.BACKGROUND_TOGGLE if _HAS_PICOUI else ""
            ),
            "&Invert",
            self,
        )
        self._act_invert.setStatusTip("Invert the current image")
        self._act_invert.triggered.connect(self.invert_image)

        self._act_histogram = QAction(
            _registry_icon(
                IconRegistry.CHART_SCATTER_PLOT if _HAS_PICOUI else ""
            ),
            "&Histogram",
            self,
        )
        self._act_histogram.setStatusTip("Show intensity histogram")
        self._act_histogram.triggered.connect(self.show_histogram)

        self._act_rings = QAction(
            _registry_icon(_icon_key("CIRCLE_OUTLINE", _icon_key("CLIP_SLICE"))),
            "Show &rings",
            self,
        )
        self._act_rings.setCheckable(True)
        self._act_rings.setChecked(self.show_rings)
        self._act_rings.setStatusTip("Toggle resolution rings")
        self._act_rings.toggled.connect(self._on_rings_action_toggled)

        self._act_indexing_spots = QAction(
            _registry_icon(_icon_key("SPOTS", _icon_key("CHART_SCATTER_PLOT"))),
            "Show indexing &spots",
            self,
        )
        self._act_indexing_spots.setCheckable(True)
        self._act_indexing_spots.setChecked(self.show_indexing_spots)
        self._act_indexing_spots.setStatusTip(
            "Toggle matched-spot markers from the Processing window"
        )
        self._act_indexing_spots.toggled.connect(
            self._on_indexing_spots_action_toggled
        )

        self._act_index = QAction(
            _registry_icon(
                IconRegistry.DATASET_PROCESSING if _HAS_PICOUI else ""
            ),
            "&Process…",
            self,
        )
        self._act_index.setShortcut("Ctrl+I")
        self._act_index.setStatusTip(
            "Open Processing (find spots, index, integrate → MTZ)"
        )
        self._act_index.triggered.connect(self.open_indexing_dialog)

        self._act_terminal = QAction(
            _registry_icon(_icon_key("TERMINAL_TAB")),
            "Show &Terminal",
            self,
        )
        self._act_terminal.setShortcut("Ctrl+`")
        self._act_terminal.setStatusTip(
            "Open the JDXI-style terminal that tails DIALS log output"
        )
        self._act_terminal.triggered.connect(self.show_terminal_viewer)

        if _HAS_PICOUI and standard_zoom_actions is not None:
            (
                self._act_zoom_in,
                self._act_zoom_out,
                self._act_reset_zoom,
            ) = standard_zoom_actions(
                self,
                on_zoom_in=self._zoom_in,
                on_zoom_out=self._zoom_out,
                on_reset=self._reset_zoom,
                zoom_in_tip="Zoom in (scroll to zoom · drag to pan)",
                zoom_out_tip="Zoom out (scroll to zoom · drag to pan)",
                reset_tip="Fit the diffraction image to the window",
            )
        else:
            self._act_zoom_in = QAction("Zoom &In", self)
            self._act_zoom_in.setShortcut("Ctrl+=")
            self._act_zoom_in.setStatusTip(
                "Zoom in (scroll to zoom · drag to pan)"
            )
            self._act_zoom_in.triggered.connect(self._zoom_in)

            self._act_zoom_out = QAction("Zoom &Out", self)
            self._act_zoom_out.setShortcut("Ctrl+-")
            self._act_zoom_out.setStatusTip(
                "Zoom out (scroll to zoom · drag to pan)"
            )
            self._act_zoom_out.triggered.connect(self._zoom_out)

            self._act_reset_zoom = QAction("&Reset Zoom", self)
            self._act_reset_zoom.setShortcut("Ctrl+0")
            self._act_reset_zoom.setStatusTip(
                "Fit the diffraction image to the window"
            )
            self._act_reset_zoom.triggered.connect(self._reset_zoom)

        self._act_quit = QAction(
            _registry_icon(IconRegistry.EXIT if _HAS_PICOUI else ""),
            "&Quit",
            self,
        )
        self._act_quit.setShortcut("Ctrl+Q")
        self._act_quit.setStatusTip("Quit DiffractionLab")
        self._act_quit.triggered.connect(self.close)

    def _build_menus(self) -> None:
        """Build File / Process menus with IconRegistry actions."""
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self._act_new_session)
        file_menu.addAction(self._act_open)
        self._recent_menu = QMenu("Open &Recent", self)
        file_menu.addMenu(self._recent_menu)
        self._recent_menu.aboutToShow.connect(self._refresh_recent_files_menu)
        self._refresh_recent_files_menu()
        file_menu.addAction(self._act_reload)
        file_menu.addSeparator()
        file_menu.addAction(self._act_open_session)
        self._recent_sessions_menu = QMenu("Open Recent &Session", self)
        file_menu.addMenu(self._recent_sessions_menu)
        self._recent_sessions_menu.aboutToShow.connect(
            self._refresh_recent_sessions_menu
        )
        self._refresh_recent_sessions_menu()
        file_menu.addAction(self._act_save_session)
        file_menu.addAction(self._act_save_session_as)
        file_menu.addAction(self._act_save_spotfile)
        file_menu.addSeparator()
        file_menu.addAction(self._act_invert)
        file_menu.addAction(self._act_histogram)
        file_menu.addAction(self._act_rings)
        file_menu.addAction(self._act_indexing_spots)
        file_menu.addSeparator()
        file_menu.addAction(self._act_quit)

        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self._act_zoom_in)
        view_menu.addAction(self._act_zoom_out)
        view_menu.addAction(self._act_reset_zoom)
        view_menu.addSeparator()
        view_menu.addAction(self._act_rings)
        view_menu.addAction(self._act_indexing_spots)
        view_menu.addSeparator()
        view_menu.addAction(self._act_terminal)

        process_menu = self.menuBar().addMenu("&Process")
        process_menu.addAction(self._act_index)
        process_menu.addAction(self._act_save_spotfile)

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
                lambda checked=False, p=path: self.open_path(p)
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
        self.statusBar().showMessage("Recent files cleared.", 3000)

    def _refresh_recent_sessions_menu(self) -> None:
        """Rebuild the Open Recent Session submenu from persisted paths."""
        self._recent_sessions_menu.clear()
        paths = self._recent_sessions.get()
        if not paths:
            empty = QAction("(No recent sessions)", self)
            empty.setEnabled(False)
            self._recent_sessions_menu.addAction(empty)
            return
        for index, path in enumerate(paths):
            label = path if len(path) < 80 else f"…{path[-77:]}"
            if index < 9:
                label = f"&{index + 1}  {label}"
            action = QAction(label, self)
            action.setStatusTip(path)
            action.triggered.connect(
                lambda checked=False, p=path: self.open_session_path(p)
            )
            self._recent_sessions_menu.addAction(action)
        self._recent_sessions_menu.addSeparator()
        clear_action = QAction("Clear Recent Sessions", self)
        clear_action.triggered.connect(self._clear_recent_sessions)
        self._recent_sessions_menu.addAction(clear_action)

    def _clear_recent_sessions(self) -> None:
        """Clear persisted recent sessions and refresh the submenu."""
        self._recent_sessions.clear()
        self._refresh_recent_sessions_menu()
        self.statusBar().showMessage("Recent sessions cleared.", 3000)

    def _mark_session_dirty(self) -> None:
        """Mark the Session as modified and update Save enablement / title."""
        self._session_dirty = True
        self._update_session_actions()

    def _update_session_actions(self) -> None:
        """Enable Save when a session path is known; refresh window title."""
        self._act_save_session.setEnabled(self._session_path is not None)
        title = "DiffractionLab"
        if self._session_path is not None:
            title = f"DiffractionLab — {self._session_path.name}"
            if self._session_dirty:
                title += " *"
        elif self._session_dirty:
            title = "DiffractionLab *"
        self.setWindowTitle(title)

    def to_viewer_state(self) -> ViewerState:
        """Capture viewer prefs into a :class:`ViewerState` for Session save."""
        dataset_name = None
        if hasattr(self, "dataset_combo") and self.dataset_combo.count():
            dataset_name = self.dataset_combo.currentText() or None
        frame_index = 0
        if hasattr(self, "frame_slider"):
            frame_index = int(self.frame_slider.value())
        contrast = 100
        if hasattr(self, "contrast_slider"):
            contrast = int(self.contrast_slider.value())
        return ViewerState(
            image_path=str(self.current_file) if self.current_file else None,
            dataset_name=dataset_name,
            frame_index=frame_index,
            show_rings=bool(self.show_rings),
            show_indexing_spots=bool(self.show_indexing_spots),
            image_inverted=bool(self.image_inverted),
            contrast=contrast,
        )

    def apply_viewer_state(self, state: ViewerState) -> list[str]:
        """Apply viewer prefs from *state* (image must already be loaded).

        :returns: Warnings for missing or skipped fields.
        """
        warnings: list[str] = []
        if state.dataset_name and hasattr(self, "dataset_combo"):
            idx = self.dataset_combo.findText(state.dataset_name)
            if idx >= 0:
                self.dataset_combo.setCurrentIndex(idx)
            else:
                warnings.append(f"Dataset not found: {state.dataset_name}")
        if hasattr(self, "frame_slider") and self.num_frames > 0:
            frame = max(0, min(int(state.frame_index), self.num_frames - 1))
            self.frame_slider.setValue(frame)
        if hasattr(self, "contrast_slider"):
            contrast = max(50, min(300, int(state.contrast)))
            self.contrast_slider.setValue(contrast)
        if bool(state.image_inverted) != bool(self.image_inverted):
            # invert_image toggles; call only when needed
            if hasattr(self, "invert_image"):
                self.invert_image()
            else:
                self.image_inverted = bool(state.image_inverted)
        self.show_rings = bool(state.show_rings)
        if hasattr(self, "rings_checkbox"):
            self.rings_checkbox.blockSignals(True)
            self.rings_checkbox.setChecked(self.show_rings)
            self.rings_checkbox.blockSignals(False)
        if hasattr(self, "_act_rings"):
            self._act_rings.blockSignals(True)
            self._act_rings.setChecked(self.show_rings)
            self._act_rings.blockSignals(False)
        if hasattr(self, "image_canvas"):
            self.image_canvas.set_rings_visible(self.show_rings)
        self.show_indexing_spots = bool(state.show_indexing_spots)
        if hasattr(self, "_act_indexing_spots"):
            self._act_indexing_spots.blockSignals(True)
            self._act_indexing_spots.setChecked(self.show_indexing_spots)
            self._act_indexing_spots.blockSignals(False)
        if hasattr(self, "image_canvas"):
            self.image_canvas.set_spots_visible(self.show_indexing_spots)
        return warnings

    def gather_session(self) -> DiffractionLabSession:
        """Build a :class:`DiffractionLabSession` from the current UI."""
        processing: ProcessingState | None = None
        if self._processing_wizard is not None:
            processing = self._processing_wizard.to_processing_state()
        return DiffractionLabSession(
            viewer=self.to_viewer_state(),
            processing=processing,
        )

    def new_session(self) -> None:
        """Clear viewer and Processing; start an untitled Session."""
        if self._session_dirty and self._session_path is not None:
            reply = QMessageBox.question(
                self,
                "New Session",
                "Discard unsaved Session changes?",
                QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if reply != QMessageBox.StandardButton.Discard:
                return
        self._close_indexing_dialog()
        self._clear_viewer()
        self._session_path = None
        self._session_dirty = False
        self._update_session_actions()
        self.statusBar().showMessage("New Session.", 3000)

    def _close_indexing_dialog(self) -> None:
        """Close and discard the Processing dialog if open."""
        dialog = self._processing_wizard
        if dialog is None:
            return
        dialog.close()
        dialog.deleteLater()
        self._processing_wizard = None

    def _clear_viewer(self) -> None:
        """Unload the current image / HDF5 dataset from the viewer."""
        self.current_file = None
        self.current_image = None
        self._raw_frame = None
        self.dataset = None
        self._h5_master = None
        self._h5_data = None
        self._external_data_paths = {}
        self.num_frames = 0
        self.beam_parameters = BeamParameters()
        self._indexing_spot_overlay = None
        self._indexing_solution = None
        self.image_inverted = False
        if hasattr(self, "dataset_combo"):
            self.dataset_combo.blockSignals(True)
            self.dataset_combo.clear()
            self.dataset_combo.blockSignals(False)
        if hasattr(self, "frame_slider"):
            self.frame_slider.setMinimum(0)
            self.frame_slider.setMaximum(0)
            self.frame_slider.setValue(0)
        if hasattr(self, "detector_distance_label"):
            self.detector_distance_label.setText("distance: N/A")
        if hasattr(self, "beam_x_label"):
            self.beam_x_label.setText("beam_x: N/A")
        if hasattr(self, "beam_y_label"):
            self.beam_y_label.setText("beam_y: N/A")
        if hasattr(self, "image_canvas"):
            self.image_canvas.set_spot_overlay(None)
            self.image_canvas.set_resolution_rings(None)
            self.image_canvas.set_beam_center(None)
            self.image_canvas.set_image_pixmap(QPixmap(), reset_view=True)
            self.image_canvas.set_placeholder("No image loaded")

    def open_session_dialog(self) -> None:
        """Prompt for a ``.dlsession`` file and open it."""
        start_dir = ""
        recent = self._recent_sessions.get()
        if recent:
            start_dir = str(Path(recent[0]).parent)
        elif self._recent_files.get():
            start_dir = str(Path(self._recent_files.get()[0]).parent)
        chosen, _ = QFileDialog.getOpenFileName(
            self,
            "Open Session",
            start_dir,
            f"DiffractionLab Session (*{SESSION_SUFFIX} *.json);;All files (*)",
        )
        if chosen:
            self.open_session_path(chosen)

    def open_session_path(self, file_path: str | Path) -> None:
        """Load a Session from *file_path* and restore viewer + Processing."""
        path = Path(file_path).expanduser()
        try:
            session = load_session(path)
        except Exception as ex:
            QMessageBox.warning(self, "Open Session", str(ex))
            return

        warnings: list[str] = []
        image = session.viewer.image_path
        if image:
            image_path = Path(image)
            if image_path.is_file():
                self.open_path(image_path)
                warnings.extend(self.apply_viewer_state(session.viewer))
            else:
                warnings.append(f"Image not found: {image_path}")
                self.apply_viewer_state(session.viewer)
        else:
            warnings.extend(self.apply_viewer_state(session.viewer))

        if session.processing is not None:
            if self.current_file is None and session.processing.master_path:
                master = Path(session.processing.master_path)
                if master.is_file():
                    self.open_path(master)
                    warnings.extend(self.apply_viewer_state(session.viewer))
            if self.current_image is not None or self.dataset is not None:
                self.open_indexing_dialog()
                dialog = self._processing_wizard
                if dialog is not None:
                    # Refresh defaults from the loaded image, then overlay session.
                    dialog._populate_defaults()
                    warnings.extend(dialog.apply_processing_state(session.processing))

        self._session_path = path.resolve()
        self._session_dirty = False
        self._recent_sessions.add(self._session_path)
        self._refresh_recent_sessions_menu()
        self._update_session_actions()
        msg = f"Opened Session {self._session_path.name}"
        if warnings:
            msg += f" ({len(warnings)} warning(s))"
            self.statusBar().showMessage(msg, 8000)
            QMessageBox.information(
                self,
                "Open Session",
                msg + ":\n\n" + "\n".join(warnings[:12]),
            )
        else:
            self.statusBar().showMessage(msg, 5000)

    def save_session_action(self) -> None:
        """Save to the current Session path, or prompt if untitled."""
        if self._session_path is None:
            self.save_session_as_action()
            return
        self._write_session(self._session_path)

    def save_session_as_action(self) -> None:
        """Prompt for a Session path and save."""
        start = ""
        if self._session_path is not None:
            start = str(self._session_path)
        elif self.current_file:
            start = str(Path(self.current_file).with_suffix(SESSION_SUFFIX))
        chosen, _ = QFileDialog.getSaveFileName(
            self,
            "Save Session As",
            start,
            f"DiffractionLab Session (*{SESSION_SUFFIX});;JSON (*.json)",
        )
        if not chosen:
            return
        path = Path(chosen)
        if path.suffix.lower() not in (SESSION_SUFFIX, ".json"):
            path = path.with_suffix(SESSION_SUFFIX)
        self._write_session(path)

    def save_spotfile_action(self) -> None:
        """Prompt for a path and save the current Mosflm ``.spt`` spot list."""
        dialog = self._processing_wizard
        if dialog is None or dialog.current_mosflm_spotfile() is None:
            QMessageBox.information(
                self,
                "Save Spot File",
                "No Mosflm spot file is available yet.\n\n"
                "Open Process…, run Find spots on the MOSFLM path "
                "(not DIALS), then use Save Spot File again.",
            )
            return
        start = ""
        if self.current_file:
            start = str(Path(self.current_file).with_suffix(".spt"))
        else:
            src = dialog.current_mosflm_spotfile()
            if src is not None:
                start = str(src.with_suffix(".spt"))
        chosen, _ = QFileDialog.getSaveFileName(
            self,
            "Save Spot File",
            start,
            "Mosflm spot list (*.spt);;All files (*)",
        )
        if not chosen:
            return
        try:
            saved = dialog.save_mosflm_spotfile(Path(chosen))
        except Exception as ex:
            QMessageBox.warning(self, "Save Spot File", str(ex))
            return
        self.statusBar().showMessage(f"Saved spot file {saved}", 5000)

    def _write_session(self, path: Path) -> None:
        """Serialize the current UI to *path*."""
        try:
            session = self.gather_session()
            saved = save_session(path, session)
        except Exception as ex:
            QMessageBox.warning(self, "Save Session", str(ex))
            return
        self._session_path = saved
        self._session_dirty = False
        self._recent_sessions.add(saved)
        self._refresh_recent_sessions_menu()
        self._update_session_actions()
        self.statusBar().showMessage(f"Saved Session {saved.name}", 5000)

    def _build_toolbar(self) -> None:
        """Build a themed toolbar with IconRegistry actions."""
        toolbar = QToolBar("Main")
        toolbar.setObjectName("qtmosflmMainToolBar")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(20, 20))
        self.addToolBar(toolbar)
        toolbar.addAction(self._act_open)
        toolbar.addAction(self._act_reload)
        toolbar.addSeparator()
        toolbar.addAction(self._act_invert)
        toolbar.addAction(self._act_histogram)
        toolbar.addAction(self._act_rings)
        toolbar.addAction(self._act_indexing_spots)
        toolbar.addSeparator()
        toolbar.addAction(self._act_zoom_in)
        toolbar.addAction(self._act_zoom_out)
        toolbar.addAction(self._act_reset_zoom)
        toolbar.addSeparator()
        toolbar.addAction(self._act_index)

    def _build_central_ui(self) -> None:
        """Build the central image viewer and controls."""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Resolution / beam readout (icon + label pairs)
        resolution_layout = QHBoxLayout()
        resolution_layout.setSpacing(12)
        res_host, self.resolution_label = _readout_widget(
            _icon_key("MDI_MAGNIFY", _icon_key("SEARCH")),
            "Resolution: N/A",
        )
        beam_x_host, self.beam_x_label = _readout_widget(
            _icon_key("CROSSHAIRS_GPS", _icon_key("RECENTER_VIEW")),
            "beam_x: N/A",
        )
        beam_y_host, self.beam_y_label = _readout_widget(
            _icon_key("IMAGE_FILTER_CENTER", _icon_key("RECENTER_VIEW")),
            "beam_y: N/A",
        )
        distance_host, self.detector_distance_label = _readout_widget(
            _icon_key("RULER"),
            "distance: N/A",
        )
        x_pix_host, self.x_pixel_label = _readout_widget(
            _icon_key("AXIS_X_ARROW", _icon_key("ARROW_RIGHT")),
            "x pixel: N/A",
        )
        y_pix_host, self.y_pixel_label = _readout_widget(
            _icon_key("AXIS_Y_ARROW", _icon_key("ALIGN_LEFT")),
            "y pixel: N/A",
        )
        theta_host, self.two_theta_label = _readout_widget(
            _icon_key("ANGLE_ACUTE", _icon_key("AUTO_ROTATE")),
            "two theta: N/A degrees",
        )
        wavelength_host, self.wavelength_label = _readout_widget(
            _icon_key("WAVELENGTH", "mdi.sine-wave"),
            "wavelength: N/A Å",
        )
        for host in (
            res_host,
            beam_x_host,
            beam_y_host,
            distance_host,
            x_pix_host,
            y_pix_host,
            theta_host,
            wavelength_host,
        ):
            resolution_layout.addWidget(host)
        resolution_layout.addStretch(1)
        layout.addLayout(resolution_layout)

        # Dataset selection
        dataset_layout = QHBoxLayout()
        dataset_label = QLabel("Dataset")
        if _HAS_PICOUI:
            dataset_icon = QLabel()
            dataset_icon.setPixmap(
                _registry_icon(IconRegistry.DATASETS).pixmap(16, 16)
            )
            dataset_layout.addWidget(dataset_icon)
        dataset_layout.addWidget(dataset_label)
        self.dataset_combo = QComboBox()
        self.dataset_combo.currentIndexChanged.connect(self.change_dataset)
        dataset_layout.addWidget(self.dataset_combo, stretch=1)
        layout.addLayout(dataset_layout)

        # Image display (shared ZoomableCanvas look-and-feel with Reflection Viewer)
        self.image_canvas = DiffractionImageCanvas()
        self.image_canvas.setMinimumHeight(700)
        self.image_canvas.set_placeholder("No image loaded")
        self.image_canvas.cursor_moved.connect(self._on_image_cursor_moved)
        layout.addWidget(self.image_canvas, stretch=1)
        # Keep attribute for older call sites / styles that still say image_label.
        self.image_label = self.image_canvas

        # Frame slider
        frame_layout = QHBoxLayout()
        if _HAS_PICOUI:
            frame_icon = QLabel()
            frame_icon.setPixmap(
                _registry_icon(IconRegistry.CAMERA).pixmap(16, 16)
            )
            frame_layout.addWidget(frame_icon)
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setFixedHeight(20)
        self.frame_slider.valueChanged.connect(self.update_frame)
        frame_layout.addWidget(self.frame_slider)
        self.frame_label = QLabel("Frame: 0")
        frame_layout.addWidget(self.frame_label)
        layout.addLayout(frame_layout)

        # Buttons (icons mirror toolbar actions)
        button_layout = QHBoxLayout()
        layout.addLayout(button_layout)

        open_button = QPushButton("Open")
        open_button.setIcon(_registry_icon(
            IconRegistry.FOLDER_OPENED if _HAS_PICOUI else ""
        ))
        open_button.clicked.connect(self.open_file)
        button_layout.addWidget(open_button)

        invert_button = QPushButton("Invert")
        invert_button.setIcon(_registry_icon(
            IconRegistry.BACKGROUND_TOGGLE if _HAS_PICOUI else ""
        ))
        invert_button.clicked.connect(self.invert_image)
        button_layout.addWidget(invert_button)

        self.rings_checkbox = QCheckBox("Show Rings")
        self.rings_checkbox.setIcon(
            _registry_icon(_icon_key("CIRCLE_OUTLINE", _icon_key("CLIP_SLICE")))
        )
        self.rings_checkbox.setChecked(self.show_rings)
        self.rings_checkbox.stateChanged.connect(self.toggle_rings)
        button_layout.addWidget(self.rings_checkbox)

        self.indexing_spots_checkbox = QCheckBox("Show indexing spots")
        self.indexing_spots_checkbox.setIcon(
            _registry_icon(_icon_key("SPOTS", _icon_key("CHART_SCATTER_PLOT")))
        )
        self.indexing_spots_checkbox.setChecked(self.show_indexing_spots)
        self.indexing_spots_checkbox.setStatusTip(
            "Toggle matched-spot markers from the Processing window"
        )
        self.indexing_spots_checkbox.stateChanged.connect(
            self.toggle_indexing_spots
        )
        button_layout.addWidget(self.indexing_spots_checkbox)

        zoom_in_button = QPushButton("Zoom In")
        zoom_in_button.setIcon(_registry_icon(_icon_key("ZOOM_IN")))
        zoom_in_button.clicked.connect(self._zoom_in)
        button_layout.addWidget(zoom_in_button)

        zoom_out_button = QPushButton("Zoom Out")
        zoom_out_button.setIcon(_registry_icon(_icon_key("ZOOM_OUT")))
        zoom_out_button.clicked.connect(self._zoom_out)
        button_layout.addWidget(zoom_out_button)

        reset_zoom_button = QPushButton("Reset Zoom")
        reset_zoom_button.setIcon(
            _registry_icon(_icon_key("FIT_TO_PAGE", _icon_key("RECENTER_VIEW")))
        )
        reset_zoom_button.clicked.connect(self._reset_zoom)
        button_layout.addWidget(reset_zoom_button)

        histogram_button = QPushButton("Histogram")
        histogram_button.setIcon(_registry_icon(
            IconRegistry.CHART_SCATTER_PLOT if _HAS_PICOUI else ""
        ))
        histogram_button.clicked.connect(self.show_histogram)
        button_layout.addWidget(histogram_button)

        index_button = QPushButton("Process")
        index_button.setIcon(_registry_icon(
            IconRegistry.DATASET_PROCESSING if _HAS_PICOUI else ""
        ))
        index_button.clicked.connect(self.open_indexing_dialog)
        button_layout.addWidget(index_button)

        # Contrast slider
        contrast_layout = QHBoxLayout()
        if _HAS_PICOUI:
            contrast_icon = QLabel()
            contrast_icon.setPixmap(
                _registry_icon(IconRegistry.COLOR).pixmap(16, 16)
            )
            contrast_layout.addWidget(contrast_icon)
        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setFixedHeight(20)
        self.contrast_slider.setRange(50, 300)
        self.contrast_slider.setValue(100)
        self.contrast_slider.valueChanged.connect(self.update_contrast)
        contrast_layout.addWidget(self.contrast_slider)
        self.contrast_label = QLabel("Contrast: 100")
        contrast_layout.addWidget(self.contrast_label)
        layout.addLayout(contrast_layout)

    def _apply_widget_styles(self) -> None:
        """Apply ThemeManager-derived colors to key widgets."""
        colors = _theme_colors()
        self.image_canvas.setStyleSheet(
            f"DiffractionImageCanvas {{"
            f" background-color: {colors['base']};"
            f" color: {colors['text']};"
            f" border: 1px solid {colors['mid']};"
            f" border-radius: 4px;"
            f"}}"
        )
        self.image_canvas.set_background(colors["base"])

    def closeEvent(self, event) -> None:  # noqa: N802
        """Release HDF5 handles when the window closes."""
        self._close_hdf5_handles()
        super().closeEvent(event)

    def _on_rings_action_toggled(self, checked: bool) -> None:
        """Keep toolbar action and checkbox in sync."""
        self.show_rings = bool(checked)
        if self.rings_checkbox.isChecked() != checked:
            self.rings_checkbox.blockSignals(True)
            self.rings_checkbox.setChecked(checked)
            self.rings_checkbox.blockSignals(False)
        self.image_canvas.set_rings_visible(self.show_rings)

    def _on_indexing_spots_action_toggled(self, checked: bool) -> None:
        """Keep toolbar action and checkbox in sync for indexing spots."""
        self.show_indexing_spots = bool(checked)
        if self.indexing_spots_checkbox.isChecked() != checked:
            self.indexing_spots_checkbox.blockSignals(True)
            self.indexing_spots_checkbox.setChecked(checked)
            self.indexing_spots_checkbox.blockSignals(False)
        self.image_canvas.set_spots_visible(self.show_indexing_spots)
        dialog = self._processing_wizard
        if dialog is not None and hasattr(dialog, "show_spots_checkbox"):
            box = dialog.show_spots_checkbox
            if box.isChecked() != self.show_indexing_spots:
                box.blockSignals(True)
                box.setChecked(self.show_indexing_spots)
                box.blockSignals(False)

    def show_terminal_viewer(self) -> None:
        """Open the JDXI-style DIALS terminal (via the Processing window)."""
        if self._processing_wizard is None:
            self.open_indexing_dialog()
        dialog = self._processing_wizard
        if dialog is not None:
            dialog.show_terminal_viewer()

    def open_indexing_dialog(self) -> None:
        """Open the Processing dialog for the loaded image / dataset."""
        if self.current_image is None and self.dataset is None:
            QMessageBox.information(
                self,
                "Process",
                "Open a diffraction image or HDF5 / NeXus master first.",
            )
            return
        if self._processing_wizard is None:
            try:
                self._processing_wizard = ProcessingWizard(self)
                self._processing_wizard.destroyed.connect(
                    self._on_indexing_dialog_destroyed
                )
            except Exception as ex:
                QMessageBox.critical(
                    self,
                    "Process",
                    f"Could not open Processing dialog:\n{ex}",
                )
                return
        self._processing_wizard.show()
        self._processing_wizard.raise_()
        self._processing_wizard.activateWindow()

    def _on_indexing_dialog_destroyed(self, *_args) -> None:
        self._processing_wizard = None

    def set_indexing_spot_overlay(
        self,
        spots_px: list[tuple[float, float, bool]] | None,
        *,
        solution=None,
    ) -> None:
        """Show or clear indexing spot markers on the current image.

        Each entry is ``(x_px, y_px, predicted)`` using Mosflm colours:
        found/observed (False) are red; predicted/indexed (True) are blue.
        """
        self._indexing_spot_overlay = spots_px
        self._indexing_solution = solution
        self.image_canvas.set_spot_overlay(
            spots_px, visible=self.show_indexing_spots
        )
        if spots_px is not None:
            n_pred = sum(1 for *_xy, predicted in spots_px if predicted)
            n_found = sum(1 for *_xy, predicted in spots_px if not predicted)
            n_total = len(spots_px)
            label = ""
            if solution is not None:
                label = f"#{getattr(solution, 'number', '?')} {getattr(solution, 'lattice', '')} "
            self.statusBar().showMessage(
                f"Indexing overlay {label}: "
                f"{n_pred} predicted (blue) / {n_found} found (red) "
                f"({n_total} markers)"
                + ("" if self.show_indexing_spots else " (hidden)"),
                5000,
            )

    def toggle_indexing_spots(self, state) -> None:
        """Toggle visibility of indexing spot markers."""
        checked = state == Qt.CheckState.Checked or state is True or state == 2
        self.show_indexing_spots = bool(checked)
        self.image_canvas.set_spots_visible(self.show_indexing_spots)
        logging.info(
            "Toggle indexing spots: %s",
            "ON" if self.show_indexing_spots else "OFF",
        )
        if self._act_indexing_spots.isChecked() != self.show_indexing_spots:
            self._act_indexing_spots.blockSignals(True)
            self._act_indexing_spots.setChecked(self.show_indexing_spots)
            self._act_indexing_spots.blockSignals(False)
        dialog = self._processing_wizard
        if dialog is not None and hasattr(dialog, "show_spots_checkbox"):
            box = dialog.show_spots_checkbox
            if box.isChecked() != self.show_indexing_spots:
                box.blockSignals(True)
                box.setChecked(self.show_indexing_spots)
                box.blockSignals(False)
        if self.current_image is not None:
            self.show_image(self.current_image)

    def build_indexing_geometry(self) -> IndexingGeometry:
        """Build :class:`IndexingGeometry` from the currently loaded image."""
        if self.current_file is None:
            raise ValueError("No file loaded")
        beam = self.beam_parameters
        if beam.x is None or beam.y is None:
            raise ValueError("Beam centre not available")
        if beam.incident_wavelength is None:
            raise ValueError("Wavelength not available")

        pixel_mm = as_float(self.detector.x_pixel_size)
        beam_x_px = as_float(beam.x)
        beam_y_px = as_float(beam.y)
        distance_mm = as_float(beam.detector_distance)
        wavelength = as_float(beam.incident_wavelength)

        if self.dataset is not None:
            height = int(self.dataset.shape[-2])
            width = int(self.dataset.shape[-1])
        elif self.current_image is not None:
            width, height = self.current_image.size
        else:
            raise ValueError("No image dimensions available")

        path = Path(self.current_file)
        phi0, phi1 = self.phi_range_for_frame(int(self.frame_slider.value()))
        return IndexingGeometry(
            directory=path.parent,
            template=path.name,
            image_number=max(1, int(self.frame_slider.value()) + 1),
            beam_x_mm=beam_x_px * pixel_mm,
            beam_y_mm=beam_y_px * pixel_mm,
            distance_mm=distance_mm,
            wavelength_a=wavelength,
            pixel_size_mm=pixel_mm,
            image_width=width,
            image_height=height,
            phi_start=phi0,
            phi_end=phi1,
            is_hdf5_master=path.suffix.lower() == ".h5",
        )

    def phi_range_for_frame(self, frame_index: int) -> tuple[float, float]:
        """Return (phi_start, phi_end) in degrees for *frame_index*."""
        omegas = self._read_omega_array()
        if omegas is not None and 0 <= frame_index < len(omegas):
            phi0 = float(omegas[frame_index])
            if frame_index + 1 < len(omegas):
                phi1 = float(omegas[frame_index + 1])
            elif len(omegas) > 1:
                step = float(omegas[1] - omegas[0])
                phi1 = phi0 + step
            else:
                phi1 = phi0 + 0.5
            return phi0, phi1
        # Fallback 0.5° wedges from frame 0.
        width = 0.5
        phi0 = frame_index * width
        return phi0, phi0 + width

    def _read_omega_array(self) -> np.ndarray | None:
        """Read ``/entry/data/omega`` or ``/entry/sample/transformations/omega``."""
        handle = self._h5_master
        if handle is None and self.current_file:
            try:
                with h5py.File(self.current_file, "r") as f:
                    return self._omega_from_handle(f)
            except Exception:
                return None
        if handle is None:
            return None
        try:
            return self._omega_from_handle(handle)
        except Exception:
            return None

    @staticmethod
    def _omega_from_handle(handle: h5py.File) -> np.ndarray | None:
        for path in (
            "/entry/data/omega",
            "/entry/sample/transformations/omega",
            "/entry/instrument/detector/transformations/omega",
        ):
            if path in handle:
                return np.asarray(handle[path][()], dtype=np.float64)
        return None

    def load_frames_for_indexing(
        self, indices: list[int]
    ) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
        """Load raw detector frames and phi wedges for indexing."""
        if self.dataset is None:
            raise ValueError("No dataset loaded")
        frames: list[np.ndarray] = []
        phis: list[tuple[float, float]] = []
        n = int(self.dataset.shape[0])
        for idx in indices:
            if idx < 0 or idx >= n:
                raise IndexError(f"Frame {idx} out of range (0–{n - 1})")
            frame = np.asarray(self.dataset[idx], dtype=np.float64)
            if frame.ndim != 2:
                raise ValueError(f"Frame {idx} has unexpected shape {frame.shape}")
            frames.append(frame)
            phis.append(self.phi_range_for_frame(idx))
        return frames, phis

    def open_file(self) -> None:
        """Prompt for an image / HDF5 file and open it."""
        start_dir = ""
        recent = self._recent_files.get()
        if recent:
            start_dir = str(Path(recent[0]).parent)
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            start_dir,
            "Image Files (*.jpg *.png *.gif *.tif *.tiff *.cbf *.h5)",
        )
        if file_path:
            self.open_path(file_path)

    def reload_file(self) -> None:
        """Reload the current file, or prompt to open one if none is loaded."""
        if not self.current_file:
            self.open_file()
            return
        path = Path(self.current_file)
        if not path.is_file():
            QMessageBox.warning(
                self,
                "Reload failed",
                f"File not found:\n{path}",
            )
            return
        frame = int(self.frame_slider.value()) if hasattr(self, "frame_slider") else 0
        self.open_path(path)
        if hasattr(self, "frame_slider") and self.num_frames > 0:
            self.frame_slider.setValue(min(frame, max(0, self.num_frames - 1)))

    def open_path(self, file_path: str | Path) -> None:
        """Open *file_path* and add it to the recent-files list."""
        path = Path(file_path).expanduser()
        if not path.is_file():
            QMessageBox.warning(
                self,
                "Open failed",
                f"File not found:\n{path}",
            )
            self._refresh_recent_files_menu()
            return
        file_path = str(path.resolve())
        self.current_file = file_path
        logging.info("file_path: %s", file_path)
        self.statusBar().showMessage(f"Loaded {file_path}", 5000)
        self._recent_files.add(file_path)
        self._refresh_recent_files_menu()
        if file_path.lower().endswith(".cbf"):
            self.display_cbf_image(file_path)
        elif file_path.lower().endswith(".h5"):
            self.extract_instrument_metadata(file_path)
            self.load_hdf5_image(file_path)
        else:
            self.display_image(file_path)
        self._mark_session_dirty()

    def display_image(self, file_path):
        """ display_image """
        logging.info(f"display_image: {file_path}")
        try:
            image = Image.open(file_path)
            self.current_image = image
            # Prefer float raster for intensity histogram; fall back to L.
            try:
                self._raw_frame = np.asarray(image.convert("F"), dtype=np.float64)
            except Exception:
                self._raw_frame = np.asarray(
                    image.convert("L"), dtype=np.float64
                )
            self.show_image(image)
        except Exception as ex:
            self._raw_frame = None
            logging.error(f"Error displaying image: {ex}")

    def parse_hdf5(self, group, path):
        """Parse HDF5 instrument group and update beam / detector parameters."""
        try:
            (
                beam_x,
                beam_y,
                detector_distance,
                incident_wavelength,
                x_pixel_size,
                y_pixel_size,
            ) = traverse_hdf5(group, path)
            beam = self.beam_parameters
            if beam_x is not None:
                beam.x = as_float(beam_x)
            if beam_y is not None:
                beam.y = as_float(beam_y)
            if detector_distance is not None:
                beam.detector_distance = as_float(detector_distance)
            if incident_wavelength is not None:
                beam.incident_wavelength = as_float(incident_wavelength)
            if x_pixel_size is not None:
                self.detector.x_pixel_size = as_float(x_pixel_size)
            if y_pixel_size is not None:
                self.detector.y_pixel_size = as_float(y_pixel_size)
        except Exception as ex:
            logging.error(f"Error parsing hdf5: {ex}")

    def traverse_hdf5_old(self, group, path):
        """Recursively print attributes and datasets in an HDF5 group."""
        if group.attrs:
            logging.info(f"\nAttributes in {path}:")
            for attr, value in group.attrs.items():
                logging.info(f"  {attr}: {value}")

        # Iterate through datasets and subgroups
        for name, item in group.items():
            full_path = f"{path}/{name}"

            if isinstance(item, h5py.Group):  # If it's a subgroup, recurse
                logging.info(f"\n--- Entering {full_path} ---")
                self.parse_hdf5(item, full_path)

            elif isinstance(item, h5py.Dataset):  # If it's a dataset, print its value
                try:
                    value = item[()]  # Extract dataset value
                    beam = self.beam_parameters

                    if full_path == r"/entry/instrument/detector/beam_centre_x":
                        beam.x = as_float(value)
                    elif full_path == r"/entry/instrument/detector/beam_centre_y":
                        beam.y = as_float(value)
                    elif full_path == r"/entry/instrument/detector/detector_distance":
                        beam.detector_distance = as_float(value)
                    elif full_path == r"/entry/instrument/beam/incident_wavelength":
                        beam.incident_wavelength = as_float(value)
                    elif full_path == r"/entry/instrument/detector/x_pixel_size":
                        self.detector.x_pixel_size = as_float(value)
                    elif full_path == r"/entry/instrument/detector/y_pixel_size":
                        self.detector.y_pixel_size = as_float(value)
                    logging.info(f"{full_path}: {value}")

                except Exception as e:
                    logging.info(f"Could not read {full_path}: {e}")

    def extract_instrument_metadata_old(self, hdf5_path):
        """Recursively extract and print all metadata from /entry/instrument in an HDF5 file."""
        try:
            with h5py.File(hdf5_path, "r") as f:
                instrument_path = "/entry/instrument"

                if instrument_path in f:
                    logging.info(f"\n--- Extracting Metadata from {instrument_path} ---")
                    self.parse_hdf5(f[instrument_path], instrument_path)
                else:
                    logging.info(f"Path '{instrument_path}' not found in the HDF5 file.")

        except Exception as ex:
            logging.info(f"Error: {ex} occurred")

    def extract_instrument_metadata(self, hdf5_path):
        """Recursively extract and print all metadata from /entry/instrument in an HDF5 file."""
        if not os.path.isfile(hdf5_path):
            logging.error(f"File '{hdf5_path}' does not exist or is not a file.")
            return

        try:
            with h5py.File(hdf5_path, "r") as f:
                instrument_path = "/entry/instrument"

                if instrument_path in f:
                    logging.info(f"\n--- Extracting Metadata from {instrument_path} ---")
                    self.parse_hdf5(f[instrument_path], instrument_path)
                else:
                    logging.warning(f"Path '{instrument_path}' not found in the HDF5 file.")
        except Exception as ex:
            logging.error(f"Error opening or parsing HDF5 file '{hdf5_path}': {ex}")

    def display_cbf_image(self, file_path):
        """ display_cbf_image """
        try:
            cbf_image = fabio.open(file_path)
            data = cbf_image.data
            self._raw_frame = np.asarray(data, dtype=np.float64)
            image = Image.fromarray(data)
            self.current_image = image.convert("L")
            self.show_image(self.current_image)
        except Exception as e:
            self._raw_frame = None
            logging.error(f"Failed to load CBF image: {e}")

    def _close_hdf5_handles(self) -> None:
        """Close any open NeXus master / data file handles."""
        self.dataset = None
        self._raw_frame = None
        for attr in ("_h5_data", "_h5_master"):
            handle = getattr(self, attr, None)
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass
                setattr(self, attr, None)

    def _open_external_dataset(self, link_name: str) -> h5py.Dataset:
        """Open one ``/entry/data`` image dataset, following ExternalLinks safely."""
        if self._h5_master is None:
            raise ValueError("No master HDF5 file is open.")

        # Prefer a previously resolved absolute data-file path.
        data_path = self._external_data_paths.get(link_name)
        if data_path is not None:
            if self._h5_data is not None:
                try:
                    self._h5_data.close()
                except Exception:
                    pass
                self._h5_data = None
            self._h5_data = h5py.File(data_path, "r")
            # NeXus data files usually expose the stack at /data
            for candidate in ("/data", "data", f"/entry/data/{link_name}"):
                if candidate in self._h5_data:
                    return self._h5_data[candidate]
            raise ValueError(f"No image dataset found in {data_path}")

        # Hard-linked dataset inside the master (rare for Eiger masters).
        path = f"/entry/data/{link_name}"
        if path not in self._h5_master:
            raise ValueError(f"Dataset path '{path}' not found.")
        return self._h5_master[path]

    def load_hdf5_image(self, image_path):
        """Load and display an HDF5 / NeXus image stack with metadata extraction."""
        master_path = Path(image_path).resolve()
        try:
            self._close_hdf5_handles()
            resolved, missing, incomplete = ensure_nexus_external_links(master_path)
            self._external_data_paths = resolved

            problems: list[str] = []
            if missing:
                problems.append(
                    "Missing next to the master:\n"
                    + "\n".join(f"  • {name}" for name in missing)
                )
            if incomplete:
                problems.append(
                    "Present but incomplete/truncated (download still running?):\n"
                    + "\n".join(f"  • {name}" for name in incomplete)
                )
            if problems:
                message = (
                    "This NeXus master file links to external data chunk files.\n\n"
                    + "\n\n".join(problems)
                    + f"\n\nMaster: {master_path}\n\n"
                    "Eiger masters expect files like `…_000001.h5` beside the "
                    "master. Browsers may leave them in `name.h5.download/` or "
                    "rename completed downloads to `name-2.h5` when a broken "
                    "symlink still occupies the original name."
                )
                if not resolved:
                    QMessageBox.critical(self, "HDF5 data files not ready", message)
                    self.statusBar().showMessage(
                        "HDF5 data files missing or still downloading.", 8000
                    )
                    return
                QMessageBox.warning(
                    self,
                    "Some HDF5 data files not ready",
                    message + "\n\nLoading the complete chunks only.",
                )

            datasets = [name for name in list_entry_data_names(master_path) if name in resolved]
            if not datasets:
                datasets = list(resolved.keys())
            if not datasets:
                raise ValueError("No complete image datasets found under '/entry/data'")

            self._h5_master = h5py.File(master_path, "r")

            if self.beam_parameters.center is None:
                raise ValueError(
                    "Beam center coordinates not found in instrument metadata."
                )

            self.dataset_combo.blockSignals(True)
            self.dataset_combo.clear()
            self.dataset_combo.addItems(datasets)
            self.dataset_combo.blockSignals(False)

            self.dataset = self._open_external_dataset(datasets[0])
            self.num_frames = int(self.dataset.shape[0])
            self.frame_slider.setMinimum(0)
            self.frame_slider.setMaximum(max(self.num_frames - 1, 0))
            self.display_hdf5_image(0)
            self.statusBar().showMessage(
                f"Loaded {datasets[0]} ({self.num_frames} frames) from {master_path.name}",
                6000,
            )
            logging.info(
                "Loaded NeXus master %s with datasets %s",
                master_path,
                datasets,
            )
        except Exception as ex:
            logging.exception("Failed to load HDF5 image %s", image_path)
            QMessageBox.critical(
                self,
                "HDF5 load failed",
                f"Could not load {image_path}:\n\n{ex}",
            )
            self.statusBar().showMessage(f"HDF5 load failed: {ex}", 8000)

    def change_dataset(self, index):
        """Change the dataset based on the combo box selection."""
        dataset_name = self.dataset_combo.itemText(index)
        if not dataset_name or self._h5_master is None:
            return
        try:
            self.dataset = self._open_external_dataset(dataset_name)
            self.num_frames = int(self.dataset.shape[0])
            self.frame_slider.setMaximum(max(self.num_frames - 1, 0))
            self.display_hdf5_image(0)
            self.statusBar().showMessage(
                f"Dataset {dataset_name}: {self.num_frames} frames", 4000
            )
        except Exception as ex:
            logging.exception("Failed to switch dataset to %s", dataset_name)
            QMessageBox.warning(
                self,
                "Dataset switch failed",
                f"Could not open dataset '{dataset_name}':\n\n{ex}",
            )

    def update_frame(self, start_index):
        """ Update frame and apply current contrast level """
        if self.dataset is not None:
            # Display the selected frame
            self.display_hdf5_image(start_index)

            # Apply contrast from the slider value
            contrast_value = self.contrast_slider.value()  # Get current slider value
            self.update_contrast(contrast_value)  # Apply contrast adjustment

            # Update UI
            self.frame_label.setText(f"Frame: {start_index}")
            logging.info(f"Frame {start_index} updated with contrast level {contrast_value}.")
            if self._processing_wizard is not None and self._processing_wizard.isVisible():
                self._processing_wizard.refresh_overlay_for_current_frame()

    def display_hdf5_image(self, start_index):
        """Display a slice of frames from the HDF5 dataset with improved contrast and no white borders."""
        try:
            if self.dataset is None:
                raise ValueError("No dataset loaded.")

            # Select frame slice
            end_index = min(start_index + self.slice_size, self.num_frames)
            frame_data = np.array(self.dataset[start_index:end_index, :, :])  # Convert to NumPy array
            # Keep the slider frame (not the CLAHE average) for intensity histogram.
            self._raw_frame = np.asarray(
                self.dataset[start_index], dtype=np.float64
            )

            # Ensure grayscale format
            if frame_data.ndim == 3:
                gray = frame_data
            else:
                gray = cv2.cvtColor(frame_data, cv2.COLOR_RGB2GRAY)

            # Mask non-white areas (considering white as near 255)
            mask = gray < 240  # Pixels below 240 are considered part of the image
            valid_pixels = frame_data[mask]

            # Compute statistics for contrast enhancement
            mean_val = np.mean(valid_pixels)
            std_val = np.std(valid_pixels)
            lower_bound = mean_val - 5 * std_val
            upper_bound = mean_val + 5 * std_val

            # Clip and normalize pixel values
            frame_clipped = np.clip(frame_data, lower_bound, upper_bound)
            p_low, p_high = np.percentile(frame_clipped[mask], (2, 98))  # Percentile-based contrast adjustment
            frame_normalized = np.clip((frame_clipped - p_low) / (p_high - p_low) * 255, 0, 255).astype(np.uint8)

            # Apply Adaptive Histogram Equalization (CLAHE)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_frames = np.array([clahe.apply(frame) for frame in frame_normalized])

            # Apply Gamma Correction
            gamma = 1.2  # Adjustable
            gamma_corrected = np.power(enhanced_frames / 255.0, gamma) * 255
            gamma_corrected = gamma_corrected.astype(np.uint8)

            # Compute the average processed frame
            final_frame = np.mean(gamma_corrected, axis=0).astype(np.uint8)

            # Convert to PIL image for display
            self.current_image = Image.fromarray(final_frame)
            if self.image_inverted:
                self.do_invert_image()
            contrast_value = self.contrast_slider.value()  # Get current slider value
            self.update_contrast(contrast_value)  # Apply contrast adjustment
            self.show_image(self.current_image)

            logging.info(f"Frames {start_index} to {end_index} displayed successfully.")

        except Exception as ex:
            self._raw_frame = None
            logging.error(f"Error in display_hdf5_image: {ex}")

    def show_image(self, image): # png_rings
        """Display the image; rings and spots are painted by the zoomable canvas."""
        try:
            logging.info(f"show_image received image: mode={image.mode}, size={image.size}")
            pixmap = pil_to_qpixmap(image.convert("RGB"))
            self.image_canvas.set_spot_overlay(
                self._indexing_spot_overlay,
                visible=self.show_indexing_spots,
            )
            self._sync_resolution_rings_to_canvas()
            # Keep zoom when refreshing the same-sized frame; reset on new size.
            self.image_canvas.set_image_pixmap(pixmap, reset_view=None)

        except Exception as ex:
            logging.error(f"Error in show_image: {ex}")

    def _sync_resolution_rings_to_canvas(self) -> None:
        """Push beam marker and resolution rings onto the zoomable canvas."""
        beam = self.beam_parameters
        center = beam.center
        self.image_canvas.set_beam_center(center)
        self._update_beam_geometry_labels()
        rings = []
        try:
            if (
                center is not None
                and beam.incident_wavelength is not None
                and self.detector.x_pixel_size is not None
            ):
                rings = build_resolution_rings(
                    beam_x=float(center[0]),
                    beam_y=float(center[1]),
                    wavelength_a=float(beam.incident_wavelength),
                    distance_mm=float(beam.detector_distance),
                    pixel_size_mm=float(self.detector.x_pixel_size),
                )
        except Exception as ex:
            logging.warning("Could not build resolution rings: %s", ex)
            rings = []
        self.image_canvas.set_resolution_rings(rings, visible=self.show_rings)

    def _update_beam_geometry_labels(self) -> None:
        """Refresh beam centre and detector-distance readouts from current parameters."""
        beam = self.beam_parameters
        if hasattr(self, "beam_x_label"):
            if beam.x is not None:
                self.beam_x_label.setText(f"beam_x: {beam.x:.0f}")
            else:
                self.beam_x_label.setText("beam_x: N/A")
        if hasattr(self, "beam_y_label"):
            if beam.y is not None:
                self.beam_y_label.setText(f"beam_y: {beam.y:.0f}")
            else:
                self.beam_y_label.setText("beam_y: N/A")
        if hasattr(self, "detector_distance_label"):
            self.detector_distance_label.setText(
                f"distance: {beam.detector_distance:.1f} mm"
            )

    def _zoom_in(self) -> None:
        self.image_canvas.zoom_in()

    def _zoom_out(self) -> None:
        self.image_canvas.zoom_out()

    def _reset_zoom(self) -> None:
        self.image_canvas.reset_view()

    def draw_resolution_rings(self, image):  # qt_rings
        """Display the image with resolution rings."""
        logging.info(f"show_image: {image}")
        try:
            pixmap = pil_to_qpixmap(annotated_image)
            if self.show_rings:
                # Draw resolution rings
                painter = QPainter(pixmap)
                pen = QPen(QColor(255, 0, 0), 2)  # Red color, 2 pixels wide
                painter.setPen(pen)
                font = painter.font()
                font.setPointSize(60)
                painter.setFont(font)

                beam = self.beam_parameters
                beam_center = beam.center
                if (
                    beam_center is None
                    or beam.incident_wavelength is None
                ):
                    painter.end()
                    return

                # Calculate radii for resolution rings
                resolutions = [1.0, 2.0, 3.0]  # Ångströms
                radii = []
                for resolution in resolutions:
                    radius_pixels = calculate_radius_pixels(
                        resolution,
                        beam.incident_wavelength,
                        beam.detector_distance,
                        self.detector.x_pixel_size,
                    )
                    radii.append(radius_pixels)

                center = QPoint(int(beam_center[0]), int(beam_center[1]))

                for i, radius in enumerate(radii):
                    painter.drawEllipse(center, int(radius), int(radius))
                    # Draw label
                    label_pos = QPoint(center.x() + int(radius) + 5, center.y())
                    painter.drawText(label_pos, f"{resolutions[i]} Å")

                painter.end()

            self.image_canvas.set_image_pixmap(pixmap, reset_view=None)
        except Exception as ex:
            logging.error(f"Error: {ex} occurred")

    def invert_image(self):
        """Invert the image."""
        logging.info(f"invert_image")
        if not self.image_inverted:
            self.image_inverted = True
        else:
            self.image_inverted = False

        if self.current_image:
            try:
                self.do_invert_image()
            except Exception as ex:
                logging.error(f"Error: {ex} occurred")

    def do_invert_image(self):
        """Perform image inversion."""
        logging.info(f"do_invert_image")
        if self.current_image:
            try:
                inverted_image = ImageOps.invert(self.current_image.convert("RGB"))
                self.current_image = inverted_image
                self.show_image(inverted_image)
            except Exception as ex:
                logging.error(f"Error: {ex} occurred")

    def update_contrast(self, value):
        """Update contrast using NumPy for better performance."""
        logging.info(f"update_contrast: {value}")
        if self.current_image:
            try:
                contrast_factor = value / 100.0

                # Convert PIL image to NumPy array
                img_array = np.array(self.current_image.convert("RGB"), dtype=np.float32)

                # Apply contrast adjustment
                mean = np.mean(img_array, axis=(0, 1), keepdims=True)  # Compute mean per channel
                adjusted_array = mean + contrast_factor * (img_array - mean)  # Contrast formula

                # Clip values to ensure valid range
                adjusted_array = np.clip(adjusted_array, 0, 255).astype(np.uint8)

                # Convert back to PIL image
                adjusted_image = Image.fromarray(adjusted_array)

                # Update the UI
                self.current_image = adjusted_image
                self.show_image(adjusted_image)
                self.contrast_label.setText(f"Contrast: {value}")
            except Exception as ex:
                logging.error(f"Error in update_contrast: {ex}")

    def show_histogram(self):
        """Show an intensity histogram of the current raw detector frame.

        Excludes PAD dead/hot sentinels (Eiger ``2**16-1`` / ``2**32-1``) the
        same way as :func:`imosflm.spotfile.find_spots_in_frame`, so saturated
        mask pixels do not pin the percentile range near 65535.
        """
        logging.info("show_histogram")
        frame = self._raw_frame
        if frame is None:
            QMessageBox.information(
                self,
                "Histogram",
                "No raw detector frame is available to histogram.",
            )
            return

        data = np.asarray(frame, dtype=np.float64).ravel()
        total = int(data.size)
        finite_positive = data[np.isfinite(data) & (data > 0.0)]
        # Trusted ADU: skip gaps (≤0) and saturated / mask sentinels.
        valid = finite_positive[
            (finite_positive < 60000.0) & (finite_positive < 1.0e8)
        ]
        n_sat = int(finite_positive.size - valid.size)
        if valid.size == 0:
            QMessageBox.information(
                self,
                "Histogram",
                (
                    "No trusted pixels to histogram "
                    f"({total:,} total, {finite_positive.size:,} positive, "
                    f"{n_sat:,} saturated/mask ≥ 60000)."
                ),
            )
            return

        if valid.size >= 20:
            vmin, vmax = np.percentile(valid, (1.0, 99.0))
        else:
            vmin, vmax = float(np.min(valid)), float(np.max(valid))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = float(np.min(valid)), float(np.max(valid))
        if vmax <= vmin:
            vmax = vmin + 1.0

        counts, bin_edges = np.histogram(valid, bins=256, range=(vmin, vmax))
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        widths = np.diff(bin_edges)

        fig = plt.figure("Histogram")
        fig.clf()
        ax = fig.add_subplot(111)
        ax.bar(centers, counts, width=widths, align="center", color="#4C78A8")
        sat_note = f", excluded {n_sat:,} sat/mask" if n_sat else ""
        ax.set_title(
            f"Intensity histogram  ({valid.size:,} / {total:,} pixels{sat_note})"
        )
        ax.set_xlabel("Intensity (ADU)")
        ax.set_ylabel("Pixel count")
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        plt.show()

    def resizeEvent(self, event):
        """Keep the zoomable canvas sized with the window."""
        super().resizeEvent(event)

    def _on_image_cursor_moved(self, x_dataset: float, y_dataset: float) -> None:
        """Update pixel / resolution readouts and status bar from the canvas cursor."""
        beam = self.beam_parameters
        center = beam.center
        if self.current_image is None or center is None:
            return
        self.x_pixel_label.setText(f"x pixel: {x_dataset:.0f}")
        self.y_pixel_label.setText(f"y pixel: {y_dataset:.0f}")
        self._update_beam_geometry_labels()
        metadata = {
            "beam_centre_x": center[0],
            "beam_centre_y": center[1],
            "detector_distance": beam.detector_distance,
            "incident_wavelength": beam.incident_wavelength,
            "detector_x_pixel_size": self.detector.x_pixel_size,
        }
        two_theta, resolution = calculate_two_theta_and_resolution(
            x_dataset, y_dataset, metadata
        )
        self.two_theta_label.setText(f"two theta: {two_theta:0.2f} °")
        self.wavelength_label.setText(f"wavelength: {beam.incident_wavelength:0.2f} Ang.")
        self.resolution_label.setText(f"Resolution: {resolution:0.2f} Å")

        mode = "hand" if self.image_canvas.is_panning() else "crosshair"
        parts = [
            mode,
            f"x={x_dataset:.0f}",
            f"y={y_dataset:.0f}",
            f"d={resolution:0.2f} Å",
            f"2θ={two_theta:0.2f}°",
        ]
        nearest = self.image_canvas.nearest_spot()
        if nearest is not None:
            sx, sy, dist = nearest
            _t, spot_res = calculate_two_theta_and_resolution(sx, sy, metadata)
            parts.append(
                f"nearest spot {dist:0.1f} px → d={spot_res:0.2f} Å"
            )
        self.statusBar().showMessage("  ·  ".join(parts))

    def toggle_rings(self, state):
        """Toggle the visibility of rings on the image."""
        checked = state == Qt.CheckState.Checked.value or state == Qt.CheckState.Checked
        self.show_rings = bool(checked)
        logging.info(f"Toggle rings: {'ON' if self.show_rings else 'OFF'}")
        if self._act_rings.isChecked() != self.show_rings:
            self._act_rings.blockSignals(True)
            self._act_rings.setChecked(self.show_rings)
            self._act_rings.blockSignals(False)
        self.image_canvas.set_rings_visible(self.show_rings)


def parse_arguments():
    """ parse_arguments """
    logging.info("parse_arguments")     
    parser = argparse.ArgumentParser(description='DiffractionLab Application')
    parser.add_argument('--ccp4i2', action='store_true', help='Autosaves an iMosflm session file upon exit')
    parser.add_argument('--debug', '-d', action='store_true', help='Creates a large output file for debugging purposes')
    parser.add_argument('--expert', '-e', action='store_true', help='Permits access to advanced detector settings')
    parser.add_argument('--fastload', '-f', action='store_true', help='Attempt to speed loading of many images offline')
    parser.add_argument('--version', '-v', action='store_true', help='Displays the program version and required Mosflm version, then exits')
    # Remove the explicit --help argument
    return parser.parse_args()


def setup_logging():
    """Set up logging configuration"""
    try:
        # Create logs directory in user's home directory
        log_dir = Path.home() / ".qtmosflm" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Log file path
        log_file = log_dir / "qtmosflm.log"
        logging.debug(f"Setting up logging to: {log_file}")

        # Reset root handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        # Configure rotating file logging
        file_handler = RotatingFileHandler(
            str(log_file),
            maxBytes=1024 * 1024,  # 1MB per file
            backupCount=5,  # Keep 5 backup files
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(filename)-14s| %(lineno)-5s| %(levelname)-8s| %(message)-24s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)

        # Configure console logging
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            "%(filename)-14s| %(lineno)-5s| %(levelname)-8s| %(message)-24s"
        )
        console_handler.setFormatter(console_formatter)

        # Configure root logger
        logging.root.setLevel(logging.DEBUG)
        logging.root.addHandler(file_handler)
        logging.root.addHandler(console_handler)

        logging.info("Logging setup complete")
        logging.info("QtMosflm starting up...")
        logging.debug(f"Log file: {log_file}")
        return log_file

    except Exception as ex:
        logging.debug(f"Error setting up logging: {str(ex)}")
        raise


def main():
    """ main entry point"""
    setup_logging()
    args = parse_arguments()

    # Set environment variables based on arguments
    if args.ccp4i2:
        os.environ['CCP4I2'] = '1'
        logging.debug("* CCP4i2 mode on - be VERY sure you want this")

    if args.debug:
        os.environ['MOSFLM_DEBUG'] = '1'
        logging.debug("* full debugging turned on")

    if args.expert:
        os.environ['EXPERTDETECTORSETTINGS'] = '1'
        logging.debug("* Expert mode on - be VERY sure you want this")

    if args.fastload:
        os.environ['FASTLOAD'] = '1'
        logging.debug("* FASTLOAD mode on")

    if args.version:
        logging.debug(f"* {os.environ.get('IMOSFLM_VERSION', 'Unknown version')}")
        sys.exit()

    # Initialize and run the application
    app = QApplication(sys.argv)
    app.setApplicationName("DiffractionLab")
    app.setOrganizationName("PyMosflm")

    if _HAS_PICOUI:
        try:
            ThemeManager.initialize(ThemeType.AUTO)
            logging.info("ThemeManager initialized (AUTO)")
        except Exception as ex:
            logging.warning("ThemeManager.initialize failed: %s", ex)
    else:
        logging.warning(
            "picoui not installed — running without ThemeManager / IconRegistry"
        )

    window = DiffractionLabApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
