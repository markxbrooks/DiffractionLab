"""Drive ``ProcessingLab run`` and read its run records.

``ProcessingLab run`` finds spots, autoindexes and integrates a sweep into a
new output directory, writing ``input.json`` (resolved inputs) and
``process.json`` (results, schema version 2; see ProcessingLab's
``processing-lab/docs/RUN_RECORDS.md``). ``--from RUN`` re-processes from a
completed run's refined model, with ``--refine`` choosing what is refined.

This module builds those command lines, runs them with a live log, and loads
the records into :class:`RunRecord`. The older CONTROL-script client is
:mod:`diffraction_lab.processinglab_client`.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from diffraction_lab.processinglab_client import find_integrationlab

logger = logging.getLogger(__name__)

#: Parameters ``--refine`` accepts (besides ``all`` and ``none``).
REFINABLE_PARAMETERS: tuple[str, ...] = (
    "orientation",
    "cell",
    "distance",
    "beam",
    "tilt_twist",
    "mosaicity",
)

_OUTPUT_DIR_LINE = re.compile(r"^\s*Output directory:\s*(.+?)\s*$")


@dataclass(frozen=True)
class RunOptions:
    """Options for one ``ProcessingLab run`` invocation.

    :param image: HDF5/NeXus master or one frame of a CBF/SMV sweep.
        Optional with ``from_run`` (its images are reused).
    :param frame_range: Inclusive ``(first, last)`` frames, or ``None`` for
        every frame (or the previous run's range with ``from_run``).
    :param output_dir: New directory for the run (must not exist), or
        ``None`` to let ProcessingLab create ``processing_YYYYMMDD_HHMMSS``
        under ``parent_dir``.
    :param parent_dir: Working directory for the process; where the default
        output directory is created.
    :param from_run: A completed run directory (or its ``process.json``) to
        re-process from.
    :param space_group: ``--spacegroup`` (number or symbol), or ``None``.
    :param cell: ``--cell`` (a, b, c, α, β, γ), or ``None``.
    :param refine: ``--refine`` parameter names, or ``None`` for
        ProcessingLab's default policy.
    :param xds_inp: ``--xds-inp`` path, or ``None``.
    :param xparm: ``--xparm`` (an XPARM.XDS / GXPARM.XDS with orientation),
        or ``None``.
    """

    image: Path | None = None
    frame_range: tuple[int, int] | None = None
    output_dir: Path | None = None
    parent_dir: Path | None = None
    from_run: Path | None = None
    space_group: str | None = None
    cell: tuple[float, float, float, float, float, float] | None = None
    refine: tuple[str, ...] | None = None
    xds_inp: Path | None = None
    xparm: Path | None = None


@dataclass(frozen=True)
class RunRecord:
    """The parts of a run's ``input.json`` / ``process.json`` DiffractionLab shows.

    Lengths in Å or mm, angles in degrees, tilt/twist in mrad. ``None`` where
    the run did not get that far.
    """

    directory: Path
    status: str
    error: str | None
    schema_version: int
    image: Path | None
    frame_range: tuple[int, int] | None
    space_group_number: int | None
    space_group_symbol: str | None
    cell: tuple[float, float, float, float, float, float] | None
    wavelength_a: float | None
    distance_mm: float | None
    beam_mm: tuple[float, float] | None
    tilt_twist_mrad: tuple[float, float] | None
    mosaicity_deg: float | None
    refinement: dict[str, str] = field(default_factory=dict)
    integrated_reflections: int | None = None
    quality_total: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Path] = field(default_factory=dict)

    @property
    def completed(self) -> bool:
        """Whether the run finished (not whether its result is validated)."""
        return self.status == "completed"

    @property
    def mtz(self) -> Path | None:
        """The unmerged MTZ, if the run wrote one."""
        return self.outputs.get("mtz")

    @property
    def can_rerun(self) -> bool:
        """Whether ``--from`` can use this run (completed with a refined model)."""
        return self.completed and self.cell is not None


class ProcessingLabError(RuntimeError):
    """``ProcessingLab run`` refused its inputs or failed; ``str()`` says why."""

    def __init__(self, message: str, *, run_dir: Path | None = None) -> None:
        super().__init__(message)
        self.run_dir = run_dir


def build_run_argv(executable: Path, options: RunOptions) -> list[str]:
    """Command line for ``ProcessingLab run`` with *options*.

    :param executable: The ``ProcessingLab`` binary.
    :param options: What to run.
    :returns: ``argv`` (executable first).
    :raises ValueError: If neither an image nor ``from_run`` is given, or a
        ``refine`` name is unknown.
    """
    if options.image is None and options.from_run is None and options.xds_inp is None:
        raise ValueError("ProcessingLab run needs an image, an XDS.INP or a run to re-process")
    argv = [str(executable), "run"]
    if options.from_run is not None:
        argv += ["--from", str(options.from_run)]
    if options.image is not None:
        argv += ["--image", str(options.image)]
    if options.xds_inp is not None:
        argv += ["--xds-inp", str(options.xds_inp)]
    if options.xparm is not None:
        argv += ["--xparm", str(options.xparm)]
    if options.frame_range is not None:
        first, last = options.frame_range
        argv += ["--range", f"{int(first)}:{int(last)}"]
    if options.space_group:
        argv += ["--spacegroup", str(options.space_group).strip()]
    if options.cell is not None:
        argv += ["--cell", *(f"{v:g}" for v in options.cell)]
    if options.refine is not None:
        unknown = [n for n in options.refine if n not in REFINABLE_PARAMETERS]
        if unknown:
            raise ValueError(f"unknown refine parameter(s): {', '.join(unknown)}")
        argv += ["--refine", ",".join(options.refine) if options.refine else "none"]
    if options.output_dir is not None:
        argv += ["--output-dir", str(options.output_dir)]
    return argv


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _cell_tuple(cell: Any) -> tuple[float, float, float, float, float, float] | None:
    try:
        return tuple(float(cell[k]) for k in ("a", "b", "c", "alpha", "beta", "gamma"))  # type: ignore[return-value]
    except (KeyError, TypeError, ValueError):
        return None


def load_run_record(run_dir: Path | str) -> RunRecord:
    """Load a run's ``input.json`` and ``process.json``.

    Reads schema version 2 (``experiment`` holds the final model) and falls
    back to the version 1 ``refinement`` layout.

    :param run_dir: The run directory, or its ``process.json``.
    :returns: The record.
    :raises FileNotFoundError: If ``process.json`` is missing.
    :raises ValueError: If a record is not valid JSON.
    """
    path = Path(run_dir).expanduser()
    directory = path.parent if path.name == "process.json" else path
    try:
        process = json.loads((directory / "process.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError as ex:
        raise ValueError(f"{directory / 'process.json'}: {ex}") from ex
    try:
        inputs = json.loads((directory / "input.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        inputs = {}

    image: Path | None = None
    images = (inputs.get("inputs") or {}).get("images") or []
    if images and images[0].get("path"):
        image = Path(images[0]["path"])
    frame_range: tuple[int, int] | None = None
    fr = ((inputs.get("experiment") or {}).get("frame_range") or {}).get("value")
    if isinstance(fr, dict) and "first" in fr and "last" in fr:
        frame_range = (int(fr["first"]), int(fr["last"]))

    experiment = process.get("experiment") or {}
    refinement = process.get("refinement") or {}
    if experiment:
        crystal = experiment.get("crystal") or {}
        detector = experiment.get("detector") or {}
        sg = crystal.get("space_group") or {}
        cell = _cell_tuple(crystal.get("unit_cell"))
        beam = detector.get("beam_centre_mm") or {}
        tt = detector.get("tilt_twist_mrad") or {}
        sg_number = int(sg["number"]) if isinstance(sg.get("number"), int) else None
        sg_symbol = sg.get("symbol")
        wavelength = _number(experiment.get("wavelength_angstrom"))
        distance = _number(detector.get("distance_mm"))
        beam_mm = (float(beam["x"]), float(beam["y"])) if "x" in beam and "y" in beam else None
        tilt = (float(tt["tilt"]), float(tt["twist"])) if "tilt" in tt and "twist" in tt else None
        mosaicity = _number(experiment.get("mosaicity_deg"))
    else:
        # Version 1: final values in ``refinement``.
        cell = _cell_tuple(refinement.get("unit_cell"))
        sg_number = refinement.get("space_group_number")
        sg_symbol = None
        wavelength = None
        distance = _number(refinement.get("detector_distance_mm"))
        bc = refinement.get("beam_centre_mm")
        beam_mm = (float(bc[0]), float(bc[1])) if isinstance(bc, list) and len(bc) == 2 else None
        tv = refinement.get("tilt_twist_mrad")
        tilt = (float(tv[0]), float(tv[1])) if isinstance(tv, list) and len(tv) == 2 else None
        mosaicity = _number(refinement.get("mosaicity_deg"))

    integration = process.get("integration") or {}
    counts = integration.get("counts") or {}
    integrated = counts.get("integrated_reflections", integration.get("reflections"))
    quality = process.get("quality") or {}

    outputs: dict[str, Path] = {}
    for item in process.get("outputs") or []:
        kind, rel = item.get("kind"), item.get("path")
        if kind and rel:
            p = Path(rel)
            outputs.setdefault(kind, p if p.is_absolute() else directory / p)

    return RunRecord(
        directory=directory,
        status=str(process.get("status") or "unknown"),
        error=process.get("error"),
        schema_version=int(process.get("schema_version") or 1),
        image=image,
        frame_range=frame_range,
        space_group_number=sg_number,
        space_group_symbol=sg_symbol,
        cell=cell,
        wavelength_a=wavelength,
        distance_mm=distance,
        beam_mm=beam_mm,
        tilt_twist_mrad=tilt,
        mosaicity_deg=mosaicity,
        refinement=dict(refinement.get("parameters") or {}),
        integrated_reflections=int(integrated) if isinstance(integrated, int) else None,
        quality_total=dict(quality.get("total") or {}),
        outputs=outputs,
    )


def run_processinglab(
    options: RunOptions,
    *,
    executable: Path | None = None,
    on_line: Callable[[str], None] | None = None,
) -> RunRecord:
    """Run ``ProcessingLab run`` and return its record.

    :param options: What to run.
    :param executable: The binary (default: :func:`find_integrationlab`).
    :param on_line: Called with each line of output as it arrives.
    :returns: The completed run's record.
    :raises FileNotFoundError: If ProcessingLab is not installed.
    :raises ProcessingLabError: If it refused the inputs (nothing created) or
        the run failed (``run_dir`` set; its ``process.json`` has the error).
    """
    argv = build_run_argv(_executable(executable), options)
    cwd = Path(options.parent_dir or Path.cwd()).expanduser()
    return execute(argv, cwd=cwd, output_dir=options.output_dir, on_line=on_line)


def run_control_stage(
    image: Path,
    control_lines: Sequence[str],
    output_dir: Path,
    *,
    script_path: Path | None = None,
    executable: Path | None = None,
    on_line: Callable[[str], None] | None = None,
) -> RunRecord:
    """Run a CONTROL script (``ProcessingLab --image … --output-dir … script``).

    For one processing stage at a time (FINDSPOTS, AUTOINDEX): the stage gets
    the same run records as ``run`` (``input.json``, ``process.json``).

    :param image: HDF5 master or one frame of a CBF/SMV sweep.
    :param control_lines: CONTROL keywords (``GO`` / ``EXIT`` included).
    :param output_dir: New directory for the stage (must not exist).
    :param script_path: Where to write the script (default: beside
        *output_dir*, as ``<name>.in``).
    :param executable: The binary (default: :func:`find_integrationlab`).
    :param on_line: Called with each line of output.
    :returns: The completed stage's record.
    :raises FileNotFoundError: If ProcessingLab is not installed.
    :raises ProcessingLabError: If the stage was refused or failed.
    """
    exe = _executable(executable)
    output_dir = Path(output_dir).expanduser().resolve()
    script = Path(script_path or output_dir.with_name(f"{output_dir.name}.in"))
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("\n".join(control_lines).rstrip() + "\n", encoding="utf-8")
    argv = [str(exe), "--image", str(image), "--output-dir", str(output_dir), str(script)]
    return execute(argv, cwd=output_dir.parent, output_dir=output_dir, on_line=on_line)


def _executable(executable: Path | None) -> Path:
    exe = executable or find_integrationlab()
    if exe is None:
        raise FileNotFoundError(
            "ProcessingLab not found: install it (make install), put it on PATH, "
            "or set $PROCESSINGLAB"
        )
    return Path(exe)


def execute(
    argv: Sequence[str],
    *,
    cwd: Path,
    output_dir: Path | None = None,
    on_line: Callable[[str], None] | None = None,
) -> RunRecord:
    """Run a ProcessingLab command line and load the run it reports.

    :param argv: The command line.
    :param cwd: Working directory (created if needed).
    :param output_dir: The ``--output-dir`` given, if any (used when the
        output does not name the directory).
    :param on_line: Called with each line of output.
    :returns: The completed run's record.
    :raises ProcessingLabError: If it refused the inputs or the run failed.
    """
    cwd = Path(cwd).expanduser()
    cwd.mkdir(parents=True, exist_ok=True)
    logger.info("Running %s in %s", " ".join(argv), cwd)
    lines: list[str] = []
    run_dir: Path | None = None
    with subprocess.Popen(
        list(argv),
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as proc:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            lines.append(line)
            if (m := _OUTPUT_DIR_LINE.match(line)) is not None:
                run_dir = Path(m.group(1))
            if on_line is not None:
                on_line(line)
        returncode = proc.wait()

    if run_dir is None and output_dir is not None and (cwd / output_dir).is_dir():
        run_dir = (cwd / output_dir).resolve()
    if run_dir is None:
        # Refused before anything was created: the message is the last line.
        message = next((ln for ln in reversed(lines) if ln.strip()), f"exit {returncode}")
        raise ProcessingLabError(message.strip())
    try:
        record = load_run_record(run_dir)
    except FileNotFoundError as ex:
        raise ProcessingLabError(f"ProcessingLab exited {returncode} without records in {run_dir}",
                                 run_dir=run_dir) from ex
    if not record.completed:
        raise ProcessingLabError(record.error or f"ProcessingLab exited {returncode}", run_dir=run_dir)
    return record


def rerun_options(
    record: RunRecord,
    *,
    refine: Sequence[str] | None = None,
    space_group: str | None = None,
    cell: tuple[float, float, float, float, float, float] | None = None,
    output_dir: Path | None = None,
) -> RunOptions:
    """Options to re-process *record* from its refined model (``--from``).

    The images and frame range come from the record. ``refine`` names what
    is refined (the rest are held at the refined values); ``None`` keeps
    ProcessingLab's default (orientation refined, mosaicity re-estimated).

    :param record: A completed run.
    :param refine: Parameters to refine, or ``None``.
    :param space_group: Override the space group, or ``None``.
    :param cell: Override the cell, or ``None``.
    :param output_dir: New output directory, or ``None`` for a timestamped
        one beside the original run.
    :returns: The options.
    :raises ValueError: If the run cannot seed a re-run.
    """
    if not record.can_rerun:
        raise ValueError(f"{record.directory} is {record.status}; only a completed run can be re-processed")
    return RunOptions(
        from_run=record.directory,
        parent_dir=record.directory.parent,
        output_dir=output_dir,
        space_group=space_group,
        cell=cell,
        refine=tuple(refine) if refine is not None else None,
    )


def latest_run_dir(parent: Path | str) -> Path | None:
    """The newest run directory (with a ``process.json``) under *parent*."""
    try:
        runs = [p for p in Path(parent).expanduser().iterdir()
                if p.is_dir() and (p / "process.json").is_file()]
    except OSError:
        return None
    return max(runs, key=lambda p: (p / "process.json").stat().st_mtime, default=None)
