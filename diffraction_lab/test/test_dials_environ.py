"""Sanitise CCP4/DIALS child environments (dev cctbx on PYTHONPATH)."""

from __future__ import annotations

import os

from diffraction_lab.dials_engine import (
    _is_dev_cctbx_path,
    dials_subprocess_environ,
)


def test_is_dev_cctbx_path() -> None:
    assert _is_dev_cctbx_path("/Users/brooks/projects/cctbx_project")
    assert _is_dev_cctbx_path("/Users/me/cctbx_project/libtbx")
    assert not _is_dev_cctbx_path("/Applications/ccp4-9/lib/python3.9/site-packages")


def test_dials_subprocess_environ_strips_cctbx(monkeypatch) -> None:
    monkeypatch.setenv(
        "PYTHONPATH",
        "/Users/brooks/projects/cctbx_project:/opt/other",
    )
    monkeypatch.setenv("LIBTBX_BUILD", "")
    env = dials_subprocess_environ()
    assert "/Users/brooks/projects/cctbx_project" not in env.get("PYTHONPATH", "")
    assert env.get("PYTHONPATH") == "/opt/other"
    assert "LIBTBX_BUILD" not in env


def test_dials_subprocess_environ_clears_empty_pythonpath(monkeypatch) -> None:
    monkeypatch.setenv("PYTHONPATH", "/Users/brooks/projects/cctbx_project")
    env = dials_subprocess_environ()
    assert "PYTHONPATH" not in env
