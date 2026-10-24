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

### The ProcessingLab Processing window (ProcessingLab → Pointless → Aimless)

**Process → Process with ProcessingLab…** (`Ctrl+Shift+I`) opens a processing window that uses only ProcessingLab and CCP4, for any image format. The DIALS / MOSFLM Processing window (**Process…**, `Ctrl+I`) is unchanged. The tabs:

1. **Setup:** the sweep and the frames to process, the work folder, which programs were found, and a pipeline checklist. **Run all steps** chains the steps below, each with its suggested choice.
2. **Find spots:** ProcessingLab FINDSPOTS over a frame wedge (default: the first 50 frames), with a σ threshold and spot limit. **Show spots on the image** overlays the spots on whichever frame the main window shows.
3. **Index:** ProcessingLab DPS autoindexing, with a spot I/σ threshold and maximum cell edge.
   - **Solutions:** all 44 lattice solutions are listed with their penalty, cell, metric deviation and possible space groups. The chosen one is in bold; ★ marks the last before the largest penalty jump.
   - **Choosing another:** highlight a row and **re-index** to use that lattice (`AUTOINDEX CHARACTER n`).
4. **Integrate:** ProcessingLab integration in an **initial space group** chosen from the solution's groups, refining everything.
   - **Result:** the refined cell, detector, mosaicity and statistics, and the report.
   - **Re-integrate:** re-integration from the refined unit cell and detector parameters, choosing what is refined again. It can also take a space group (filled in from Pointless when that's safe) or a cell.
   - **Live charts:** while it integrates, the tab charts the **average spot profile** (one panel per φ slice) and the **resolution bins**: completeness and I/σ, and CC½ and R<sub>meas</sub>, against d. Both update after every batch of frames (25 by default). They are drawn from the `PROFILE_MAP` and `RESOLUTION_BINS` blocks in ProcessingLab's output, the same blocks ProcessingLab's `doc/read_profile_maps.py` and `doc/resolution_bins.py` plot.
5. **Symmetry:** CCP4 Pointless on the integrated reflections.
   - **Results:** the Laue and space group with their confidences and the reindex operator, and the **L-test for twinning** (⟨|L|⟩, ⟨L²⟩, twin fraction and verdict).
   - **When it can't decide:** it says when only the point group could be determined.
   - **Re-integrating in its group:** **Re-integrate in this space group** is available only when Pointless found a space group without reindexing.
6. **Scale:** CCP4 Aimless on Pointless's output, with an optional high-resolution cutoff and an anomalous option. It shows the overall, inner and outer statistics, the resolution estimates and the merged MTZ.

Each step is a ProcessingLab run in a session folder (`<work>/session_YYYYMMDD_HHMMSS/`): `01-findspots/`, `02-index/`, `03-integrate/` and `04-ccp4/` (Pointless and Aimless). Each has its own `process.json` and `process.md`, and a repeated step gets a new numbered folder. ProcessingLab's output streams into the log pane. The steps are also a Python API: `diffraction_lab.processinglab_stages`.

### Processing with ProcessingLab (any format → MTZ → Pointless / Aimless → re-run)

The **ProcessingLab** tab of the Processing window drives `ProcessingLab run` on the open image (CBF, SMV or an HDF5 / NeXus master):

1. **Run ProcessingLab:** find spots, autoindex and integrate the chosen frames into a new `processing_YYYYMMDD_HHMMSS/` folder. The work folder defaults to `<image_dir>/<stem>-processinglab/`, or `~/.qtmosflm/processinglab/<stem>/` when the image folder is read-only. A space group and cell can be given up front. The Result box shows the run record: the refined cell, space group, distance, beam, tilt/twist and mosaicity, what was refined or held fixed, the integrated reflections and the unscaled statistics. **Open run…** loads an earlier run.
2. **Run Pointless → Aimless:** run on the run's `process.mtz`, into the run's `ccp4/` folder (`pointless.mtz`, `aimless.mtz`, both logs). It shows:
   - Pointless's solution, reindex operator and space-group and Laue-group confidences. When only the point group can be determined (too few axial reflections for screw axes), it says so.
   - Aimless's overall, inner and outer statistics: resolution, R<sub>meas</sub>, CC½, ⟨I/σ⟩, completeness and multiplicity, plus the CC½ resolution estimate.
3. **Re-run with refined parameters:** runs `ProcessingLab run --from` on the run, starting from its refined unit cell, detector distance, beam centre, tilt/twist and orientation, with no autoindexing.
   - **What is refined:** ticked parameters are refined again and the rest are held fixed. The default is orientation and mosaicity.
   - **Space group:** filled in from Pointless only when Pointless found a space group and its reindex operator is `[h,k,l]`, because `--spacegroup` does not reindex.
   - **Cell:** a different cell can be entered; ProcessingLab rebuilds the orientation matrix for it.

ProcessingLab's output streams into the tab's log. The underlying API is `diffraction_lab.processinglab_run` (`run_processinglab`, `load_run_record`, `rerun_options`) and `diffraction_lab.aimless_runner` (`scale_and_merge`).

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
| `ProcessingLab run` (find spots → index → integrate, re-run with `--from`) | **ProcessingLab** with `run --refine` | The ProcessingLab tab; any format |
| Symmetry, scaling and merging | **CCP4 Pointless** and **Aimless** | Found on `PATH`, `$CBIN`, `$CCP4/bin` or `/Applications/ccp4-9/bin` |
| Job history / local queue | **SchedArray** (optional) | Persists CLI jobs; live logs still tailed by the Terminal viewer |

**Backend selection (non-HDF5):**

| Env | Effect |
|---|---|
| `DIFFRACTIONLAB_BACKEND=processinglab` | Force ProcessingLab `ProcessingLab` |
| `DIFFRACTIONLAB_BACKEND=mosflm` | Force F77 `ipmosflm` (MOSFLMSOCKET) |
| `DIFFRACTIONLAB_PREFER_PROCESSINGLAB=1` | Prefer ProcessingLab when both are installed |
| `PROCESSINGLAB` / `PROCESSINGLAB_HOME` | Path to `ProcessingLab` binary or ProcessingLab root |
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
   - **Any format, with ProcessingLab:** in the **ProcessingLab** tab, run ProcessingLab, then Pointless → Aimless, then re-run with the refined cell and detector parameters.
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
| `processinglab_run.py` | `ProcessingLab run` / `--from` command lines, live log, `input.json` / `process.json` records |
| `processinglab_panel.py` | ProcessingLab tab: run → Pointless / Aimless → re-run from refined values |
| `processing_sweep.py` | `ProcessingSweep` (image, `frames`, HDF5 data files) and `Frames` (`start`, `end`): the sweep ProcessingLab processes |
| `processinglab_stages.py` | ProcessingLab step by step: find spots, index (all lattice solutions), integrate |
| `processinglab_window.py` | The ProcessingLab Processing window (spots → index → integrate → Pointless → Aimless) |
| `processinglab_charts.py` | Live integration charts: streaming `PROFILE_MAP` / `RESOLUTION_BINS` parser, Qt signals, matplotlib canvases |
| `pointless_runner.py`, `aimless_runner.py` | CCP4 Pointless and Aimless runners and log parsers |
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
