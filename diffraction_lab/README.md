# DiffractionLab

PySide6 diffraction-image viewer and interactive indexer, inspired by **iMosflm**.

DiffractionLab opens crystallographic images (**CBF**, **HDF5 / NeXus**, **TIFF**, and common raster formats), supports zoom/pan and resolution-ring overlays, and runs indexing from an Index window:

- **Mosflm DPS** for formats Mosflm can process
- **DIALS** automatically for **Eiger / HDF5 masters** that Mosflm cannot process

## Run

From the repository root (with dependencies installed):

```sh
python imosflm/qtmosflm.py
```

Optional flags:

| Flag | Purpose |
|------|---------|
| `--debug` / `-d` | Large debug log under `~/.qtmosflm/logs/` |
| `--expert` / `-e` | Advanced detector settings |
| `--fastload` / `-f` | Faster offline multi-image load (experimental) |
| `--version` / `-v` | Print version information |

## Image viewer

- **Open** CBF, HDF5/NeXus masters, TIFF, or other images
- Frame slider for multi-image / HDF5 datasets
- Zoom in / out / reset (toolbar, View menu, shortcuts `Ctrl+=`, `Ctrl+-`, `Ctrl+0`); mouse wheel zoom and hand-drag pan
- Contrast slider and **Histogram**
- **Show rings** — resolution rings from beam centre / distance / wavelength
- Live readout: resolution, `beam_x` / `beam_y`, pixel coordinates, two-theta
- **Show indexing spots** — overlay from the Index window on the current frame

## Processing window

Open via **Process…** (toolbar, Process menu, or `Ctrl+I`). Pipeline steps are tabs: **Setup → Find spots → Index → Integrate → MTZ**.

### Mosflm path (CBF and similar)

1. **Find spots** on the selected frame range  
2. **Run indexing** (Mosflm `autoindex dps`)  
3. Inspect solutions, refine unit cell, estimate mosaicity  

### DIALS path (HDF5 → MTZ)

When the loaded dataset is an HDF5 / NeXus master, Processing becomes a **DIALS panel** (Mosflm-only fields are hidden):

| UI section | Purpose |
|------------|---------|
| Master `.h5` + work directory | Browse / Reveal; default `<master_dir>/<stem>/YYYY-MM-DD-HH-MM-SS-dials/` beside the master |
| DIALS settings | SpecEditor for `nproc`, `d_min`, min spot size, gain, known space group / unit cell (live phil preview) |
| Frames Start/End | Contiguous 1-based DIALS image range (Step ignored) |
| Pipeline checklist | Import → Find spots → Index/refine → Integrate → Export MTZ |
| Step buttons | Find spots → Index → Integrate with DIALS |
| **Run full DIALS pipeline → MTZ** | One-shot `import → find_spots → index → refine → integrate → export` |
| Integrate first/last | Contiguous wedge passed as `scan_range` to `dials.integrate` |
| MTZ tab | Reveal / Copy / **Open in Reflection File Viewer**; **Summary vs resolution** plot from `dials.integrate.log` |
| Progress / log | Indeterminate bar; **Show terminal** (JDXI-style live tail of `dials.session.log`); Open last `dials.*.log` |

Typical interactive flow:

1. Open a master `.h5`  
2. **Process…** (`Ctrl+I`)  
3. Adjust work dir / settings / image wedge on the **Setup** tab  
4. **Find spots** → check overlays on the image  
5. **Index** → inspect cell in the table  
6. **Integrate** (or **Run full pipeline**)  
7. **MTZ** tab → **Open in Reflection File Viewer**  

DIALS stdout streams live into a JDXI Editor-like **Terminal** window (dark monospace viewer with fast/slow refresh). It opens automatically when a DIALS step starts, or via **View → Show Terminal** (`Ctrl+\``) / **Show terminal** in Processing.

Strong-spot centroids and indexed predictions are cached as `.npz` in the work directory and drawn on the **currently displayed frame** as you move the frame slider.

To open MTZs in ElMo Reflection Viewer, DiffractionLab looks for `../ElMo/.venv/bin/python` (or set `ELMO_PYTHON` / `ELMO`).

Tutorial flow mirrored by the engine:  
[DIALS processing in detail](https://dials.github.io/documentation/tutorials/processing_in_detail_betalactamase_dui.html).

## Requirements

- Python 3.9+
- `PySide6`, `numpy`, `h5py`, `fabio`, `matplotlib`, `Pillow`, OpenCV (`cv2`)
- Optional: `pycbf`; optional [PicoUI](https://github.com/markxbrooks) for toolbar icons

### External tools

| Need | Tool |
|------|------|
| Mosflm DPS | **Mosflm** (CCP4) |
| HDF5 find-spots / index / integrate | **DIALS** (`dials.import`, …) via CCP4 or PATH |

Discovery order for DIALS: `$DIALS_DIR`, `$CCP4`, `PATH`, then common install prefixes (e.g. `/Applications/ccp4-9`). Source CCP4 in your shell before launching if tools are not on `PATH`:

```sh
source /Applications/ccp4-9/bin/ccp4.setup-sh
python imosflm/qtmosflm.py
```

## Package layout

| Module | Role |
|--------|------|
| `qtmosflm.py` | Main window / image viewer |
| `image_canvas.py` | Zoomable diffraction canvas & overlays |
| `indexing_dialog.py` | Index UI (Mosflm + DIALS HDF5→MTZ panel) |
| `dials_engine.py` | DIALS subprocess pipeline + `.npz` caches |
| `terminal_log_viewer.py` | JDXI-style dark terminal that tails DIALS logs |
| `reflection_viewer_launch.py` | Launch ElMo Reflection Viewer on an MTZ |
| `mosflm_client.py` | Mosflm process I/O |
| `spotfile.py` | Mosflm spotfile I/O / peak find |
| `indexing.py` | Geometry helpers shared by indexing |

## License

MIT — see the repository root `LICENSE`.
