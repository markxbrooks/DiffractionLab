"""The processing steps of a ProcessingLab run.

Mirrors ProcessingLab's ``RunStep`` (``processing-lab/include/mosflm/io/RunStep.hpp``):
``process.json`` (schema 3) names each stage by a step's value
(``"find_spots"``, ``"indexing"``, ``"defpix"``, ``"integrate"``,
``"strategy"``), and logs show its label (``"RunStep.DEFPIX"``). Schema 1–2
stage names (``findspots``, ``autoindex``, ``process``) are read too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

#: Schema 1–2 stage names and the steps they were.
_LEGACY_NAMES = {"findspots": "find_spots", "autoindex": "indexing", "process": "integrate"}


class RunStep(Enum):
    """A step of a ProcessingLab run, in run order.

    ``DEFPIX`` (the pixels integration ignores) runs at the start of
    ``PROCESS``, before ``INTEGRATE``.
    """

    FIND_SPOTS = "find_spots"
    INDEXING = "indexing"
    DEFPIX = "defpix"
    INTEGRATE = "integrate"
    STRATEGY = "strategy"

    @property
    def label(self) -> str:
        """``"RunStep.FIND_SPOTS"``, … as ProcessingLab logs it."""
        return f"{type(self).__name__}.{self.name}"

    @property
    def keyword(self) -> str:
        """The CONTROL keyword that runs it (``DEFPIX`` runs inside ``PROCESS``)."""
        return {RunStep.FIND_SPOTS: "FINDSPOTS", RunStep.INDEXING: "AUTOINDEX", RunStep.DEFPIX: "PROCESS",
                RunStep.INTEGRATE: "PROCESS", RunStep.STRATEGY: "STRATEGY"}[self]

    @classmethod
    def parse(cls, name: str) -> RunStep | None:
        """The step called *name*: its value, name or label (any case), or a schema 1–2 stage name.

        :returns: The step, or ``None`` if *name* is not one.
        """
        key = str(name).strip().lower().removeprefix("runstep.")
        key = _LEGACY_NAMES.get(key, key)
        return next((step for step in cls if step.value == key), None)


@dataclass(frozen=True)
class StageRecord:
    """One entry of ``process.json`` ``stages``.

    :param name: As recorded.
    :param step: The step, or ``None`` for a name this version doesn't know.
    :param status: ``pending``, ``running``, ``completed``, ``failed`` or ``skipped``.
    :param duration_s: Seconds it took, or ``None``.
    :param message: Human-readable summary, or ``None``.
    :param error: Why it failed, or ``None``.
    :param results: Machine-readable summary (e.g. ``{"spots": 1500}``).
    """

    name: str
    step: RunStep | None
    status: str
    duration_s: float | None = None
    message: str | None = None
    error: str | None = None
    results: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        """``RunStep.…`` label, or the recorded name."""
        return self.step.label if self.step is not None else self.name

    @classmethod
    def from_json(cls, stage: dict[str, Any]) -> StageRecord:
        """Read one ``stages`` entry."""
        name = str(stage.get("name") or "")
        duration = stage.get("duration_s")
        return cls(
            name=name,
            step=RunStep.parse(name),
            status=str(stage.get("status") or "unknown"),
            duration_s=float(duration) if isinstance(duration, (int, float)) else None,
            message=stage.get("message"),
            error=stage.get("error"),
            results=dict(stage.get("results") or {}),
        )
