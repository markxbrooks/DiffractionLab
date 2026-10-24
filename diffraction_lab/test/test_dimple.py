"""Tests for running CCP4 Dimple and the Dimple tab of the processing windows."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from diffraction_lab.dimple_runner import (
    DimpleOptions,
    dimple_command,
    find_dimple,
    new_output_dir,
    parse_dimple_log,
    run_dimple,
)

DATA = Path(__file__).parent / "data" / "dimple"
CCP4_DATA = Path(os.environ.get("CCP4", "/Applications/ccp4-9")) / "examples" / "data"


def _blob_run(tmp_path: Path) -> Path:
    """The blob fixture as an output directory, with the files Dimple writes beside its log."""
    out = tmp_path / "dimple_blob"
    out.mkdir()
    shutil.copy(DATA / "blob" / "dimple.log", out)
    for name in ("final.pdb", "final.mtz", "coot.sh", "blob1-coot.py", "blob1v1.png", "blob1v2.png"):
        (out / name).write_text("")
    return out


def test_options_argv() -> None:
    assert DimpleOptions().argv == ["-f", "png"]
    opts = DimpleOptions(slow=2, mr_when_r=0.5, free_r_flags="-", resolution=1.8, anode=True, images=False,
                         no_hetatm=True, libin=Path("/l/lig.cif"), icolumn="I", extra=("--jelly", "0"))
    assert opts.argv == ["-s", "-s", "-M", "0.5", "-R", "-", "--reso", "1.8", "--anode", "--no-hetatm",
                         "--libin", "/l/lig.cif", "-I", "I", "--jelly", "0"]


def test_command_sources_ccp4_setup(tmp_path: Path) -> None:
    exe = tmp_path / "bin" / "dimple"
    exe.parent.mkdir()
    exe.write_text("")
    cmd = dimple_command(exe, Path("/d/a b.mtz"), Path("/d/m.pdb"), Path("/o/run"), DimpleOptions(images=False))
    assert cmd == f"exec {exe} '/d/a b.mtz' /d/m.pdb /o/run"
    (exe.parent / "ccp4.setup-sh").write_text("")
    assert dimple_command(exe, Path("/d/x.mtz"), Path("/d/m.pdb"), Path("/o"), DimpleOptions()).startswith(
        f". {exe.parent / 'ccp4.setup-sh'} && exec {exe} -f png")


def test_new_output_dir(tmp_path: Path) -> None:
    first = new_output_dir(tmp_path)
    assert first.parent == tmp_path and first.name.startswith("dimple_") and not first.exists()
    first.mkdir()
    assert new_output_dir(tmp_path) != first  # never reused


def test_parse_blob_run(tmp_path: Path) -> None:
    result = parse_dimple_log(_blob_run(tmp_path))
    assert result.succeeded and result.error is None and result.molecular_replacement is None
    assert [s.name for s in result.refinement] == ["rigid", "jelly", "restr"]
    final = result.final
    assert (final.r_start, final.r_end, final.r_free_start, final.r_free_end) == (0.2663, 0.2517, 0.284, 0.2914)
    (blob,) = result.blobs
    assert (blob.x, blob.y, blob.z, blob.score) == (16.14, 38.55, 22.94, 444.9)
    assert blob.coot_script.name == "blob1-coot.py" and [p.name for p in blob.images] == ["blob1v1.png",
                                                                                          "blob1v2.png"]
    assert result.coot_script.name == "coot.sh" and result.final_mtz.name == "final.mtz"
    assert "find-blobs" in [s.name for s in result.steps]


def test_parse_failed_run(tmp_path: Path) -> None:
    out = tmp_path / "failed"
    shutil.copytree(DATA / "failed", out)
    result = parse_dimple_log(out / "dimple.log")
    assert not result.succeeded
    assert result.error == "INPUT ERROR: No scattering in coordinate file"
    assert result.molecular_replacement == "phaser" and result.refinement == ()
    # Without a step error, the screen summary says why.
    (out / "dimple.log").write_text((out / "dimple.log").read_text().replace("error: INPUT ERROR", "note: x")
                                    .replace("exit_status: 65", "exit_status: 0"))
    assert parse_dimple_log(out).error == "Error: No phaser solution.."


def test_run_dimple_refusals(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="MTZ not found"):
        run_dimple(tmp_path / "x.mtz", tmp_path / "m.pdb", tmp_path / "out")


@pytest.mark.skipif(find_dimple() is None or not (CCP4_DATA / "insulin.mtz").is_file(),
                    reason="needs CCP4 Dimple and its insulin example")
def test_run_dimple_insulin(tmp_path: Path) -> None:
    lines: list[str] = []
    result = run_dimple(CCP4_DATA / "insulin.mtz", CCP4_DATA / "insulin.pdb", tmp_path / "run",
                        options=DimpleOptions(images=False), on_line=lines.append)
    assert result.succeeded, result.error
    assert result.final.r_free_end is not None and result.final.r_free_end < 0.25
    assert any("refmac5 restr" in line for line in lines)  # streamed as it ran


# ------------------------------------------------------------------ the tab

def test_tab_in_windows_and_result(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from diffraction_lab import dimple_tab as tab_module
    from diffraction_lab.dimple_tab import DimpleTab, format_dimple
    from diffraction_lab.processing_window_base import ProcessingEngine, ProcessingWindowBase

    _app = QApplication.instance() or QApplication([])
    mtz = tmp_path / "aimless.mtz"
    mtz.write_text("")
    window = ProcessingWindowBase(None, ProcessingEngine.PROCESSINGLAB)
    started: list[tuple] = []
    monkeypatch.setattr(window, "start_job", lambda message, fn, done, on_line=None: started.append(
        (message, fn, done, on_line)) or True)
    monkeypatch.setattr(tab_module, "find_dimple", lambda: Path("/ccp4/bin/dimple"))
    tab = DimpleTab(window, mtz_provider=lambda: mtz, folder_provider=lambda: tmp_path / "05-dimple")
    assert not tab.run_btn.isEnabled()
    tab.fill_from_window()
    assert tab.mtz_edit.text() == str(mtz)
    model = tmp_path / "model.pdb"
    model.write_text("")
    tab.model_edit.setText(str(model))
    assert tab.run_btn.isEnabled() and "05-dimple" in tab.folder_label.text()
    tab.slow_spin.setValue(1)
    tab.free_edit.setText("-")
    tab.images_check.setChecked(False)
    assert tab.options().argv == ["-s", "-R", "-"]

    calls: list[dict] = []
    monkeypatch.setattr(tab_module, "run_dimple", lambda *a, **kw: calls.append({"args": a, **kw}) or
                        parse_dimple_log(_blob_run(tmp_path)))
    tab.run_btn.click()
    message, fn, done, on_line = started[-1]
    assert "model.pdb against aimless.mtz" in message and not tab.run_btn.isEnabled()
    result = fn(lambda _line: None)
    assert calls[0]["args"][2].parent == tmp_path / "05-dimple"
    done(result)
    assert tab.blob_list.count() == 1 and "Blob 1 at (16.1, 38.5, 22.9)" in tab.blob_list.item(0).text()
    assert tab.coot_btn.isEnabled() and tab.coot_blob_btn.isEnabled() and tab.run_btn.isEnabled()
    html = format_dimple(result)
    assert "0.2517 / 0.2914" in html and "1 unmodelled blob" in html

    # A failure is reported in the tab, not raised in the job thread.
    warnings: list[str] = []
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "warning", lambda _p, _t, text: warnings.append(text))
    monkeypatch.setattr(tab_module, "run_dimple", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")))
    tab.run_btn.click()
    _m, fn, done, _l = started[-1]
    done(fn(lambda _line: None))
    assert warnings == ["boom"] and "Dimple failed" in tab.result_label.text()


def test_processing_windows_have_dimple_tab(tmp_path: Path) -> None:
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from diffraction_lab.processing_wizard import ProcessingWizard

    _app = QApplication.instance() or QApplication([])
    wizard = ProcessingWizard(None)
    tabs = [wizard.pipeline_tabs.tabText(i) for i in range(wizard.pipeline_tabs.count())]
    assert tabs.index("8. Dimple") == tabs.index("7. Scale") + 1
    assert wizard._dimple_mtz() is None and wizard._dimple_folder().name == "dimple"
