"""DiffractionLab entry point (alias for ``qtmosflm``).

ElMo and other launchers look for ``diffraction_lab/diffraction_lab.py``
(or legacy ``imosflm/…``). This module runs the same ``main()`` as
:mod:`qtmosflm`.
"""

from __future__ import annotations

import runpy
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def main() -> None:
    """Run the DiffractionLab / qtMosflm GUI."""
    runpy.run_path(str(_HERE / "qtmosflm.py"), run_name="__main__")


if __name__ == "__main__":
    main()
