# pymosflm (qtMosflm)

**Python port of the famous crystallographic application**

`pymosflm` is an experimental Python-based implementation inspired by **iMosflm / MOSFLM**, for viewing crystallographic diffraction images and driving interactive indexing. It opens **CBF**, **HDF5 / NeXus**, and **TIFF** (and other common image formats), with zoom/pan, resolution rings, and spot overlays.

For traditional image formats, indexing uses **Mosflm DPS**. For **Eiger / HDF5 masters** that Mosflm cannot process, qtMosflm switches to a **DIALS** backend automatically.

![Screenshot](resources/screenshot1.png)

## Features

### Image viewer
- Load and display **CBF**, **HDF5 / NeXus**, **TIFF**, and other images (**JPEG**, **PNG**, …).
- Zoom / pan (mouse wheel, hand drag, toolbar Zoom In / Out / Reset).
- Contrast control and intensity histogram.
- Resolution-ring overlay and live beam / resolution / two-theta readout.
- Indexing-spot overlay on the current frame (toggle from toolbar or Index window).

### Indexing — Mosflm DPS
- Spot finding and **autoindex dps** for formats Mosflm can process.
- Solution table, unit-cell refine, and mosaicity estimation (iMosflm-style workflow).

### Indexing & processing — DIALS (HDF5 / NeXus)
When you open an HDF5 / NeXus master, the Index window uses **DIALS** instead of Mosflm:

1. **Find spots** — `dials.import` + `dials.find_spots` on a short image wedge.
2. **Index** — `dials.index` (fft3d) with automatic **refine**.
3. **Integrate** — `dials.integrate` + `dials.export` → MTZ.

DIALS strong-spot centroids and indexed predictions are cached as NumPy `.npz` files and drawn on the **currently displayed frame** as you scrub the frame slider. Working files land under `~/.qtmosflm/dials/`.

Pipeline reference: [DIALS processing in detail (DUI tutorial)](https://dials.github.io/documentation/tutorials/processing_in_detail_betalactamase_dui.html).

## Requirements

- Python 3.9+
- **PySide6**, `numpy`, `h5py`, `fabio`, `matplotlib`, `Pillow`, OpenCV (`cv2`)
- Optional: `pycbf` for enhanced CBF handling
- Optional UI polish: [PicoUI](https://github.com/markxbrooks) (`picoui`) for themed toolbar icons

### External crystallography tools

| Task | Tool | Notes |
|------|------|--------|
| Mosflm DPS indexing | **Mosflm** (CCP4) | Used for non-HDF5 datasets |
| HDF5 spot-find / index / integrate | **DIALS** (CCP4 or standalone) | Required for Eiger / NeXus masters |

qtMosflm looks for `dials.*` on `PATH`, or under `$DIALS_DIR`, `$CCP4`, or common CCP4 install prefixes (e.g. `/Applications/ccp4-9`). Spot/prediction overlay export uses CCP4’s Python when the venv cannot import `dials`.

## Installation

```sh
git clone https://github.com/markxbrooks/pymosflm.git
cd pymosflm

pip install numpy h5py fabio matplotlib pillow opencv-python-headless PySide6
```

Ensure CCP4 (or a DIALS install) is sourced in your shell if you need indexing / HDF5 processing, for example:

```sh
source /Applications/ccp4-9/bin/ccp4.setup-sh   # macOS CCP4 example
```

## Usage

```sh
python imosflm/qtmosflm.py
```

1. **Open** a diffraction image or HDF5 / NeXus master.
2. Adjust contrast, zoom, and (optionally) show resolution rings.
3. **Index…** (toolbar or Process menu):
   - CBF / traditional images → Mosflm spot finding + DPS.
   - HDF5 master → DIALS find spots → index/refine → optional integrate to MTZ.
4. Toggle **Show indexing spots** to overlay found / matched spots on the image; change frames to update the overlay for that image number.

## Roadmap
- Faster HDF5 frame loading and caching.
- Richer DIALS settings UI (gain, `d_min`, space group / cell constraints).
- Deeper Mosflm session / strategy integration.
- Closer alignment with CCP4i2 / existing crystallographic pipelines.

This project is in active development—stay tuned for updates.

## Contributions

Contributions are welcome! Feel free to submit issues, feature requests, or pull requests.

## License

`pymosflm` is released under the MIT License.
