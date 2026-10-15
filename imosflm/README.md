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

## Index window

Open via **Index…** (toolbar, Process menu, or `Ctrl+I`).

### Mosflm path (CBF and similar)

1. **Find spots** on the selected frame range  
2. **Run indexing** (Mosflm `autoindex dps`)  
3. Inspect solutions, refine unit cell, estimate mosaicity  

### DIALS path (HDF5 / NeXus)

When the loaded dataset is an HDF5 master, the Index UI switches to DIALS:

1. **Find spots (DIALS)** — `dials.import` + `dials.find_spots` on a short wedge  
2. **Run DIALS indexing** — `dials.index` (fft3d) with automatic refine  
3. **Integrate with DIALS** — `dials.integrate` + `dials.export` → MTZ  

Strong-spot centroids and indexed predictions are cached as `.npz` under `~/.qtmosflm/dials/` and drawn on the **currently displayed frame** as you move the frame slider.

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
| `indexing_dialog.py` | Index UI (Mosflm + DIALS) |
| `dials_engine.py` | DIALS subprocess pipeline + `.npz` caches |
| `mosflm_client.py` | Mosflm process I/O |
| `spotfile.py` | Mosflm spotfile I/O / peak find |
| `indexing.py` | Geometry helpers shared by indexing |

## License

MIT — see the repository root `LICENSE`.
