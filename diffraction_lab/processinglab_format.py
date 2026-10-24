"""Display helpers for ProcessingLab results.

Formats a run record (:func:`format_record`) and Pointless / Aimless results
(:func:`format_scaling`) as HTML for the ProcessingLab window, parses a
typed unit cell (:func:`parse_cell_text`) and chooses where a dataset's runs
are kept (:func:`default_work_parent`).
"""

from __future__ import annotations

import logging
from pathlib import Path

from diffraction_lab.aimless_runner import ScaleMergeResult
from diffraction_lab.processinglab_run import RunRecord

logger = logging.getLogger(__name__)


def parse_cell_text(text: str) -> tuple[float, float, float, float, float, float] | None:
    """Parse an optional ``a b c α β γ`` (spaces or commas) cell entry.

    :param text: The entry; blank means none.
    :returns: Six values, or ``None`` when blank.
    :raises ValueError: If it is not six positive numbers.
    """
    tokens = text.replace(",", " ").split()
    if not tokens:
        return None
    try:
        values = tuple(float(t) for t in tokens)
    except ValueError as ex:
        raise ValueError(f"cell must be six numbers, got {text!r}") from ex
    if len(values) != 6 or any(v <= 0 for v in values):
        raise ValueError(f"cell must be six positive numbers, got {text!r}")
    return values  # type: ignore[return-value]


def default_work_parent(image: Path) -> Path:
    """Where runs for *image* go: ``<image_dir>/<stem>-processinglab/``, or
    ``~/.qtmosflm/processinglab/<stem>/`` when the image folder is read-only.

    :param image: The open image (any frame, or an HDF5 master).
    :returns: The folder (created).
    """
    stem = image.stem.removesuffix("_master")
    beside = image.parent / f"{stem}-processinglab"
    try:
        beside.mkdir(parents=True, exist_ok=True)
        probe = beside / ".write-test"
        probe.touch()
        probe.unlink()
        return beside
    except OSError:
        home = Path.home() / ".qtmosflm" / "processinglab" / stem
        home.mkdir(parents=True, exist_ok=True)
        return home


def format_record(record: RunRecord) -> str:
    """HTML summary of a run for the Result box."""

    def num(v: float | None, fmt: str) -> str:
        return "—" if v is None else format(v, fmt)

    rows = [("Status", record.status + (f" — {record.error}" if record.error else ""))]
    if record.space_group_number is not None:
        symbol = f" ({record.space_group_symbol})" if record.space_group_symbol else ""
        rows.append(("Space group", f"{record.space_group_number}{symbol}"))
    if record.cell is not None:
        a, b, c, al, be, ga = record.cell
        rows.append(("Cell", f"{a:.3f} {b:.3f} {c:.3f} Å, {al:.2f} {be:.2f} {ga:.2f}°"))
    if record.distance_mm is not None:
        beam = "—" if record.beam_mm is None else f"{record.beam_mm[0]:.3f} {record.beam_mm[1]:.3f}"
        rows.append(("Detector", f"distance {record.distance_mm:.3f} mm, beam {beam} mm"))
    if record.tilt_twist_mrad is not None:
        rows.append(("Tilt / twist", f"{record.tilt_twist_mrad[0]:.2f} / {record.tilt_twist_mrad[1]:.2f} mrad"))
    if record.mosaicity_deg is not None:
        rows.append(("Mosaicity", f"{record.mosaicity_deg:.3f}°"))
    if record.refinement:
        refined = [k for k, v in record.refinement.items() if v != "fixed"]
        fixed = [k for k, v in record.refinement.items() if v == "fixed"]
        rows.append(("Refined", ", ".join(refined) or "nothing"))
        rows.append(("Held fixed", ", ".join(fixed) or "nothing"))
    if record.integrated_reflections is not None:
        rows.append(("Integrated", f"{record.integrated_reflections} reflections"))
    if record.defpix_pixels:
        d = record.defpix_pixels
        ignored = [f"{d[k]:,} {label}" for k, label in (
            ("shaded", "shaded"), ("defective", "defective"), ("hot", "hot"),
            ("untrusted_region", "in untrusted regions"), ("excluded_resolution", "in excluded resolution"),
        ) if d.get(k)]
        rows.append(("Ignored pixels", ", ".join(ignored) or "none (besides the detector's gaps)"))
    q = record.quality_total
    if q:
        rows.append((
            "Unscaled",
            f"R<sub>meas</sub> {num(q.get('r_meas_pct'), '.1f')}%, I/σ {num(q.get('mean_i_over_sigma'), '.2f')}, "
            f"CC½ {num(q.get('cc_half_pct'), '.1f')}%, completeness {num(q.get('completeness_pct'), '.1f')}%",
        ))
    rows.append(("Folder", str(record.directory)))
    return "<table>" + "".join(f"<tr><td><b>{k}</b>&nbsp;</td><td>{v}</td></tr>" for k, v in rows) + "</table>"


def format_scaling(result: ScaleMergeResult) -> str:
    """HTML summary of Pointless + Aimless for the scaling box."""
    p, s = result.pointless, result.aimless.summary
    kind = p.solution_kind or "space group"
    lines = [
        f"<b>Pointless:</b> {kind} {p.space_group or '—'}, reindex {p.reindex_operator or '—'}, "
        f"space-group confidence {'—' if p.confidence is None else f'{p.confidence:.2f}'}"
        + ("" if p.laue_confidence is None else f", Laue-group confidence {p.laue_confidence:.2f}"),
    ]
    if kind == "point group":
        lines.append("<i>Only the point group was determined (too few axial reflections for screw axes).</i>")

    def cell(label: str, shell: str, fmt: str) -> str:
        v = s.value(label, shell)
        return "—" if v is None else format(v, fmt)

    table = [("Resolution (Å)", "High resolution limit", ".2f"),
             ("R<sub>meas</sub>", "Rmeas (all I+ & I-)", ".3f"),
             ("CC½", "Mn(I) half-set correlation CC(1/2)", ".3f"),
             ("⟨I/σ⟩", "Mean((I)/sd(I))", ".1f"),
             ("Completeness (%)", "Completeness", ".1f"),
             ("Multiplicity", "Multiplicity", ".1f")]
    rows = "".join(
        f"<tr><td>{name}</td><td align=right>{cell(label, 'overall', f)}</td>"
        f"<td align=right>{cell(label, 'inner', f)}</td><td align=right>{cell(label, 'outer', f)}</td></tr>"
        for name, label, f in table
    )
    lines.append(f"<b>Aimless</b> ({s.space_group or '—'}):<table><tr><th></th><th>Overall</th>"
                 f"<th>Inner</th><th>Outer</th></tr>{rows}</table>")
    if (cc := s.resolution_estimates.get("half-dataset correlation CC(1/2) >  0.30")) is not None:
        lines.append(f"Resolution estimate (CC½ &gt; 0.3): {cc:.2f} Å")
    return "<br>".join(lines)
