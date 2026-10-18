# DiffractionLab

![DiffractionLab screenshot](resources/DiffractionLabLogo.png)

**Interactive X-ray diffraction imaging, indexing, and processing with Python and Qt**

DiffractionLab is a PySide6 desktop application for inspecting crystallographic diffraction images and running interactive indexing workflows. It combines the classic **MOSFLM DPS** indexing workflow (CCP4 F77 `ipmosflm` or **ProcessingLab** `ProcessingLab`) with **DIALS** processing for HDF5 / NeXus datasets.

The repository contains:

- **DiffractionLab** — a PySide6 diffraction-image viewer and indexer, currently implemented under `imosflm/`.
- **Legacy iMosflm Tcl sources** — retained under `src/` as reference / legacy UI code.

![DiffractionLab screenshot](resources/screenshot2.png)

## Features

### Diffraction-image viewer

- Load and display **CBF**, **HDF5 / NeXus**, **TIFF**, and common raster formats such as **JPEG** and **PNG**.
- Navigate multi-image datasets with a frame slider.
- Zoom and pan using the mouse, toolbar, View menu, and keyboard shortcuts.
- Adjust contrast and inspect the intensity histogram.
- Display resolution rings calculated from beam centre, detector distance, and wavelength.
- View live readouts for resolution, beam coordinates, pixel coordinates, and two-theta.
- Overlay found spots and indexed predictions on the currently displayed frame.

### Indexing with MOSFLM

For CBF and other formats supported by MOSFLM, the Processing window provides an iMosflm-style workflow:

1. Find spots over the selected frame range.
2. Run MOSFLM `autoindex dps`.
3. Inspect indexing solutions.
4. Refine the unit cell and estimate mosaicity.

### Indexing and processing with DIALS (HDF5 → MTZ)

When an HDF5 / NeXus master is opened, the Processing window becomes a DIALS panel (tabs: Setup → Find spots → Index → Integrate → MTZ):

1. **Import and find spots** — `dials.import` and `dials.find_spots` on a contiguous image wedge.
2. **Index and refine** — `dials.index` (`fft3d`) with automatic refine.
3. **Integrate and export** — `dials.integrate` (with `scan_range`) and `dials.export` → MTZ.

The panel also exposes work-directory browse/reveal (default `<master_dir>/<stem>/YYYY-MM-DD-HH-MM-SS-dials/` beside the HDF5 master), DIALS settings (`nproc`, `d_min`, gain, min spot size, optional known symmetry), a pipeline checklist, **Run full pipeline → MTZ**, a JDXI-style live **Terminal** for DIALS output (**View → Show Terminal**), and post-MTZ actions (Reveal, Copy path, **Open in Reflection File Viewer**, **Summary vs resolution**).

Working files default to `~/.qtmosflm/dials/<dataset_stem>/`. Spot/prediction overlays update with the frame slider.

See [imosflm/README.md](diffraction_lab/README.md) for the full Index UI reference, and the [DIALS processing-in-detail tutorial](https://dials.github.io/documentation/tutorials/processing_in_detail_betalactamase_dui.html).

## Requirements

- Python 3.9+
- PySide6
- NumPy
- h5py
- fabio
- Matplotlib
- Pillow
- OpenCV (`cv2`)

Optional dependencies:

- `pycbf` for enhanced CBF handling.
- [PicoUI](https://github.com/markxbrooks) (`picoui`) for themed toolbar icons.
- [SchedArray](https://github.com/markxbrooks) — SQLite job scheduler for run / log / store / retrieve of DIALS and Pointless steps (Processing **Jobs** tab). Also needs `sqlalchemy>=2` and [decologr](https://github.com/markxbrooks).

### External crystallography tools

| Task | Tool | Notes |
|---|---|---|
| MOSFLM DPS indexing / integrate | **MOSFLM** (CCP4 `ipmosflm`) or **ProcessingLab** (`ProcessingLab`) | Non-HDF5; see env vars below |
| HDF5 / NeXus spot finding, indexing, and integration | **DIALS** (CCP4 or standalone) | Used for Eiger and other compatible HDF5 / NeXus datasets |
| Job history / local queue | **SchedArray** (optional) | Persists CLI jobs; live logs still tailed by the Terminal viewer |

**Backend selection (non-HDF5):**

| Env | Effect |
|---|---|
| `DIFFRACTIONLAB_BACKEND=processinglab` | Force ProcessingLab `ProcessingLab` |
| `DIFFRACTIONLAB_BACKEND=mosflm` | Force F77 `ipmosflm` (MOSFLMSOCKET) |
| `DIFFRACTIONLAB_PREFER_PROCESSINGLAB=1` | Prefer ProcessingLab when both are installed |
| `INTEGRATIONLAB` / `PROCESSINGLAB_HOME` | Path to `ProcessingLab` binary or ProcessingLab root |
| `MOSFLM_EXEC` | Path to F77 `ipmosflm` |

Default: use F77 Mosflm when found; otherwise fall back to ProcessingLab. ProcessingLab does not implement `MOSFLMSOCKET` (mosaicity estimate / index2 refine stay Mosflm-only).

DiffractionLab searches for DIALS commands on `PATH`, under `$DIALS_DIR` or `$CCP4`, and in common CCP4 installation prefixes (for example, `/Applications/ccp4-9`). Spot / prediction overlay export may use CCP4's Python when DIALS cannot be imported in the active virtual environment.

Ensure that CCP4 or a standalone DIALS installation is available in your shell when using indexing or HDF5 processing.

If `dials.import` fails with `FileNotFoundError: .../ccp4-9/libtbx_env` while importing from
`…/projects/cctbx_project/libtbx/…`, a development CCTBX checkout is on `PYTHONPATH` and is
shadowing CCP4’s `libtbx`. DiffractionLab strips those paths for DIALS child processes; for a
manual shell fix: `unset PYTHONPATH LIBTBX_BUILD` (or remove `cctbx_project` from `PYTHONPATH`)
before running `dials.import`.

### SchedArray job backend (optional)

When SchedArray is installed, DIALS and Pointless CLI steps are submitted through a **per-master** SQLite scheduler beside the HDF5 stem directory:

`…/<master_stem>/schedarray_scheduler.db`

Example: `thaumatin_9_1_master.h5` → `…/thaumatin_9_1_master/schedarray_scheduler.db`.

Stdout is written live to the usual work-dir log files so the Terminal viewer can tail them. The Processing dialog **Jobs** tab lists, filters, cancels, and reopens those jobs (MxPanDDA Tracked Jobs UI pattern).

Local monorepo install:

```sh
pip install -e ../decologr
pip install -e ../SchedArray
```

Disable SchedArray and fall back to in-process `Popen` with:

```sh
export USE_SCHEDARRAY=false
```

## Installation

Clone the repository and install the Python dependencies:

```sh
git clone https://github.com/markxbrooks/pymosflm.git
cd pymosflm

pip install numpy h5py fabio matplotlib pillow opencv-python-headless PySide6
```

The repository URL and directory above retain the current `pymosflm` name. Update them if the GitHub repository is renamed.

If you use CCP4, source its environment before launching the application. For example, on macOS:

```sh
source /Applications/ccp4-9/bin/ccp4.setup-sh
```

## Usage

From the repository root, run:

```sh
python imosflm/main.py
```

Optional command-line flags:

| Flag | Purpose |
|---|---|
| `--debug` / `-d` | Write a detailed debug log under `~/.qtmosflm/logs/` |
| `--expert` / `-e` | Enable advanced detector settings |
| `--fastload` / `-f` | Enable experimental faster offline multi-image loading |
| `--version` / `-v` | Print version information |

### Typical workflow

1. Open a diffraction image or an HDF5 / NeXus master.
2. Adjust contrast and zoom; optionally enable resolution rings.
3. Open **Process…** from the toolbar or Process menu (or press `Ctrl+I`).
4. Run the available indexing workflow:
   - **CBF and other MOSFLM-supported images:** find spots and run MOSFLM DPS autoindexing.
   - **HDF5 / NeXus masters:** find spots with DIALS, index and refine, then optionally integrate and export to MTZ.
5. Enable **Show indexing spots** to overlay found spots and predictions. For multi-image datasets, move the frame slider to inspect the overlay on each frame.

## Package layout

The current application code remains under `imosflm/`:

| Module | Role |
|---|---|
| `qtmosflm.py` | Main window and image viewer entry point |
| `image_canvas.py` | Zoomable diffraction canvas and overlays |
| `indexing_dialog.py` | Indexing UI for MOSFLM and DIALS |
| `terminal_log_viewer.py` | JDXI-style live terminal for DIALS logs |
| `dials_engine.py` | DIALS subprocess pipeline and `.npz` caches |
| `mosflm_client.py` | MOSFLM process I/O |
| `spotfile.py` | MOSFLM spot-file I/O and peak finding |
| `indexing.py` | Geometry helpers shared by indexing workflows |
| `src/` | Classic iMosflm Tcl sources retained for reference |

## Roadmap

- Improve HDF5 frame loading and caching performance.
- Expand DIALS settings, including gain, `d_min`, and space-group / unit-cell constraints.
- Add deeper MOSFLM session and strategy integration.
- Improve alignment with CCP4i2 and established crystallographic processing pipelines.

DiffractionLab is under active development. Contributions, bug reports, and feature requests are welcome.

## Contributing

Please submit issues or pull requests for bug fixes, improvements, and new features.

## License

Released under the MIT License. See [`LICENSE`](LICENSE).
