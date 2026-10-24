"""Tests for DEFPIX support: pixel exclusions for ``ProcessingLab run`` and the ignored-pixel overlay."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PySide6")

from diffraction_lab.processing_sweep import Frames  # noqa: E402
from diffraction_lab.processinglab_format import format_record  # noqa: E402
from diffraction_lab.processinglab_run import (  # noqa: E402
    PixelExclusions,
    RunOptions,
    build_run_argv,
    load_run_record,
)

EXE = Path("/opt/ProcessingLab")


def _app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _write_defpix_cbf(path: Path, shape: tuple[int, int] = (6, 8)) -> np.ndarray:
    """A small defpix.cbf: −3 where ignored, −1 where the detector masks."""
    import fabio.cbfimage

    data = np.full(shape, 20, dtype=np.int32)
    data[0, :] = -1
    data[2, 3] = data[-1, -2] = -3
    fabio.cbfimage.CbfImage(data=data).write(str(path))
    return data == -3


def test_pixel_exclusions_parse_and_argv() -> None:
    ex = PixelExclusions.parse("RECTANGLE 1200 1300 1150 1250\n# a comment\n\nell 1, 21, 1, 11\n"
                               "QUAD 1 1 11 1 11 11 1 11", "3.93 3.87, 2.70-2.64", ice=True)
    assert ex.untrusted == (("RECTANGLE", (1200.0, 1300.0, 1150.0, 1250.0)),
                            ("ELLIPSE", (1.0, 21.0, 1.0, 11.0)),
                            ("QUADRILATERAL", (1.0, 1.0, 11.0, 1.0, 11.0, 11.0, 1.0, 11.0)))
    assert ex.resolution == ((3.93, 3.87), (2.70, 2.64)) and ex.ice and ex
    assert ex.argv[:6] == ["--untrusted", "RECTANGLE", "1200", "1300", "1150", "1250"]
    assert ex.argv[-7:] == ["--exclude-resolution", "3.93", "3.87", "--exclude-resolution", "2.7", "2.64",
                            "--exclude-ice"]
    assert not PixelExclusions.parse("", "")
    for regions, resolution, message in [("CIRCLE 1 2 3", "", "line 1: 'CIRCLE'"),
                                         ("\nRECTANGLE 1 2 3", "", "line 2: RECTANGLE needs 4"),
                                         ("RECTANGLE 1 2 a 4", "", "numbers"),
                                         ("", "3.9", "needs two values"),
                                         ("", "0 3.9", "positive")]:
        with pytest.raises(ValueError, match=message):
            PixelExclusions.parse(regions, resolution)


def test_build_run_argv_with_exclusions() -> None:
    ex = PixelExclusions.parse("RECTANGLE 1 10 1 20", "", ice=True)
    argv = build_run_argv(EXE, RunOptions(image=Path("/d/x_0001.cbf"), frames=Frames(1, 30), exclusions=ex,
                                          output_dir=Path("out")))
    assert argv == [str(EXE), "run", "--image", "/d/x_0001.cbf", "--range", "1:30", "--untrusted", "RECTANGLE",
                    "1", "10", "1", "20", "--exclude-ice", "--output-dir", "out"]
    # No exclusions: no extra arguments.
    assert "--exclude-ice" not in build_run_argv(EXE, RunOptions(image=Path("x.cbf"), exclusions=PixelExclusions()))


def test_run_record_defpix(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "input.json").write_text("{}")
    (run / "process.json").write_text(json.dumps({
        "schema_version": 2, "status": "completed", "error": None,
        "integration": {"counts": {"integrated_reflections": 16425},
                        "defpix": {"frames_sampled": [1, 15, 30], "block_px": 8,
                                   "pixels": {"trusted": 5538830, "detector_masked": 527917, "untrusted_region": 0,
                                              "excluded_resolution": 0, "shaded": 150863, "defective": 2,
                                              "hot": 2}}},
        "outputs": [{"kind": "defpix_image", "path": "defpix.cbf"}],
    }))
    rec = load_run_record(run)
    assert rec.defpix_image == run / "defpix.cbf"
    assert rec.defpix_pixels["shaded"] == 150863 and rec.defpix_pixels["defective"] == 2
    html = format_record(rec)
    assert "Ignored pixels" in html and "150,863 shaded, 2 defective, 2 hot" in html


def test_canvas_pixel_mask() -> None:
    from PySide6.QtGui import QPixmap

    from diffraction_lab.image_canvas import DiffractionImageCanvas

    _app()
    canvas = DiffractionImageCanvas()
    pixmap = QPixmap(8, 6)
    canvas.set_image_pixmap(pixmap)
    mask = np.zeros((6, 8), dtype=bool)
    mask[2, 3] = True
    canvas.set_pixel_mask(mask)
    assert canvas.has_pixel_mask()
    image = canvas._pixel_mask
    assert (image.width(), image.height()) == (8, 6)
    assert image.pixelColor(3, 2).alpha() > 0 and image.pixelColor(0, 0).alpha() == 0
    canvas.grab()  # paints with the mask
    canvas.set_pixel_mask(None)
    assert not canvas.has_pixel_mask()


def test_main_window_defpix_overlay(tmp_path: Path) -> None:
    from PySide6.QtGui import QAction, QPixmap

    from diffraction_lab.image_canvas import DiffractionImageCanvas
    from diffraction_lab.main import DiffractionLabApp

    _app()
    canvas = DiffractionImageCanvas()
    canvas.set_image_pixmap(QPixmap(8, 6))
    messages: list[str] = []
    action = QAction("defpix")
    action.setCheckable(True)  # as in the main window
    fake = SimpleNamespace(image_canvas=canvas, _act_defpix=action,
                           statusBar=lambda: SimpleNamespace(showMessage=lambda m, *_: messages.append(m)))
    expected = _write_defpix_cbf(tmp_path / "defpix.cbf")
    assert DiffractionLabApp.set_defpix_overlay(fake, tmp_path / "defpix.cbf") == 2
    assert canvas.has_pixel_mask() and fake._act_defpix.isEnabled() and fake._act_defpix.isChecked()
    assert "2 ignored pixels" in messages[-1]
    assert expected.sum() == 2
    # Another image size: refused, nothing drawn over the wrong image.
    _write_defpix_cbf(tmp_path / "other.cbf", (5, 5))
    with pytest.raises(ValueError, match="5×5 pixels, the image 8×6"):
        DiffractionLabApp.set_defpix_overlay(fake, tmp_path / "other.cbf")
    assert DiffractionLabApp.set_defpix_overlay(fake, None) == 0
    assert not canvas.has_pixel_mask() and not fake._act_defpix.isEnabled()


def test_window_exclusions_and_overlay(tmp_path: Path, monkeypatch) -> None:
    from PySide6.QtWidgets import QMessageBox, QWidget

    from diffraction_lab.processinglab_run import RunRecord
    from diffraction_lab.processinglab_window import ProcessingLabWindow

    _app()
    shown: list[Path | None] = []

    class Viewer(QWidget):
        def set_indexing_spot_overlay(self, *_args, **_kw) -> None:
            pass

        def set_defpix_overlay(self, path: Path | None) -> int:
            shown.append(path)
            return 0 if path is None else 2

    viewer = Viewer()
    window = ProcessingLabWindow(viewer)
    window.untrusted_edit.setPlainText("RECTANGLE 1 10 1 20")
    window.exclude_res_edit.setText("3.93 3.87")
    window.exclude_ice_check.setChecked(True)
    ex = window._exclusions()
    assert ex == PixelExclusions((("RECTANGLE", (1.0, 10.0, 1.0, 20.0)),), ((3.93, 3.87),), True)
    warnings: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", lambda _p, _t, text: warnings.append(text))
    window.untrusted_edit.setPlainText("BOX 1 2 3 4")
    assert window._exclusions() is None and "BOX" in warnings[-1]

    defpix = tmp_path / "defpix.cbf"
    window._s.integrated = RunRecord(directory=tmp_path, status="completed", error=None, schema_version=2,
                                     image=None, frames=None, space_group_number=None, space_group_symbol=None,
                                     cell=None, wavelength_a=None, distance_mm=None, beam_mm=None,
                                     tilt_twist_mrad=None, mosaicity_deg=None, outputs={"defpix_image": defpix})
    window._show_all()
    assert window.show_defpix_check.isEnabled() and shown == []
    window.show_defpix_check.setChecked(True)
    assert shown == [defpix] and "2 ignored pixels" in window.status_label.text()
    window.show_defpix_check.setChecked(False)
    assert shown == [defpix, None]
    window.show_defpix_check.setChecked(True)
    window.close()  # clears what it showed
    assert shown[-1] is None
