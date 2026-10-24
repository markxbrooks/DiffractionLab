"""The sweep being processed: its images and their frames.

A :class:`ProcessingSweep` is one rotation sweep as ProcessingLab sees it: an
HDF5/NeXus master with every frame of every data file, or a CBF/SMV sweep
from the opened frame through the consecutively numbered files after it.
Its :class:`Frames` (``frames.start`` … ``frames.end``, inclusive) are the
frame numbers ProcessingLab uses (``--range start:end``).

The image viewer counts frames one HDF5 data file at a time; use
:meth:`ProcessingSweep.sweep_frame` to turn a viewer position into a sweep
frame.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

_HDF5_SUFFIXES = {".h5", ".nxs"}
_LAST_NUMBER = re.compile(r"(\d+)(?=\D*$)")


@dataclass(frozen=True, slots=True)
class Frames:
    """An inclusive range of frame numbers.

    :param start: First frame.
    :param end: Last frame (``>= start``).
    :raises ValueError: If ``end < start`` or ``start < 0``.
    """

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"frames must satisfy 0 <= start <= end, got {self.start}–{self.end}")

    @classmethod
    def between(cls, a: int, b: int) -> Frames:
        """The frames between *a* and *b*, in either order."""
        return cls(min(int(a), int(b)), max(int(a), int(b)))

    @property
    def count(self) -> int:
        """Number of frames."""
        return self.end - self.start + 1

    def __contains__(self, frame: object) -> bool:
        return isinstance(frame, int) and self.start <= frame <= self.end

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.start, self.end + 1))

    def clamp(self, other: Frames) -> Frames:
        """These frames limited to *other*.

        :raises ValueError: If they do not overlap.
        """
        return Frames(max(self.start, other.start), min(self.end, other.end))

    def first(self, n: int) -> Frames:
        """The first *n* frames (all of them if there are fewer)."""
        return Frames(self.start, min(self.start + max(int(n), 1) - 1, self.end))

    @property
    def range_argument(self) -> str:
        """ProcessingLab's ``--range`` value, ``"start:end"``."""
        return f"{self.start}:{self.end}"

    def __str__(self) -> str:
        return f"{self.start}–{self.end}"


@dataclass(frozen=True, slots=True)
class DataFile:
    """One data file of an HDF5/NeXus master and the sweep frames it holds."""

    name: str
    frames: Frames


@dataclass(frozen=True, slots=True)
class ProcessingSweep:
    """One sweep: the image ProcessingLab is given and every frame it covers.

    :param image: HDF5/NeXus master, or the first frame of a CBF/SMV sweep.
    :param frames: The sweep's frames.
    :param data_files: For a master, its data files in sweep order.
    """

    image: Path
    frames: Frames
    data_files: tuple[DataFile, ...] = ()

    @property
    def is_hdf5(self) -> bool:
        """Whether the sweep is an HDF5/NeXus master."""
        return self.image.suffix.lower() in _HDF5_SUFFIXES

    @classmethod
    def from_image(cls, image: Path | str) -> ProcessingSweep | None:
        """The sweep containing *image*, as ProcessingLab will read it.

        HDF5/NeXus: every frame of every data file (a missing data file ends
        the sweep). CBF/SMV: the opened frame and the consecutively numbered
        files after it; an image without a frame number is a one-frame sweep.

        :param image: A master, or one frame.
        :returns: The sweep, or ``None`` if a master cannot be read.
        """
        image = Path(image).expanduser()
        if image.suffix.lower() in _HDF5_SUFFIXES:
            data_files = _hdf5_data_files(image)
            if not data_files:
                return None
            return cls(image, Frames(data_files[0].frames.start, data_files[-1].frames.end), data_files)
        m = _LAST_NUMBER.search(image.stem)
        if m is None:
            return cls(image, Frames(1, 1))
        digits = m.group(1)
        start = int(digits)
        prefix, suffix = image.stem[:m.start()], image.stem[m.end():] + image.suffix
        end = start
        while (image.parent / f"{prefix}{end + 1:0{len(digits)}d}{suffix}").is_file():
            end += 1
        return cls(image, Frames(start, end))

    def spot_wedge(self, n: int = 50) -> Frames:
        """The default spot-search wedge: the first *n* frames."""
        return self.frames.first(n)

    def data_file(self, name: str) -> DataFile | None:
        """The data file called *name* (e.g. ``"data_000002"``), if any."""
        return next((d for d in self.data_files if d.name == name), None)

    def sweep_frame(self, viewer_index: int, data_file: str | None = None) -> int:
        """The sweep frame at a 0-based image-viewer position.

        :param viewer_index: The viewer's frame slider value.
        :param data_file: The HDF5 data file shown (its first frame is the
            offset), or ``None`` for the first.
        :returns: The sweep frame number.
        """
        shown = self.data_file(data_file) if data_file else None
        first = shown.frames.start if shown is not None else self.frames.start
        return first + int(viewer_index)


def _hdf5_data_files(master: Path) -> tuple[DataFile, ...]:
    """A master's ``/entry/data/data_*`` files and their sweep frames (1-based)."""
    import h5py

    files: list[DataFile] = []
    try:
        with h5py.File(master, "r") as f:
            data = f["/entry/data"]
            start = 1
            for name in sorted(n for n in data.keys() if n.startswith("data_")):
                try:
                    n = int(data[name].shape[0])
                except (KeyError, OSError):
                    break  # a missing data file ends the sweep
                files.append(DataFile(name, Frames(start, start + n - 1)))
                start += n
    except (OSError, KeyError):
        return ()
    return tuple(files)
