"""Socket client for CCP4 / F77 ``ipmosflm`` (MOSFLMSOCKET protocol).

Mirrors the launch / command path used by ``src/mosflm.tcl``:
listen on localhost → start ``ipmosflm MOSFLMSOCKET <port>`` → send
line-oriented keywords → receive one-line XML responses and ``<done>``.

ProcessingLab's ``IntegrationLab`` does **not** implement MOSFLMSOCKET; see
:mod:`diffraction_lab.processinglab_client` for CONTROL-script driving.
"""

from __future__ import annotations

import logging
import os
import select
import shutil
import socket
import subprocess
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


logger = logging.getLogger(__name__)


def find_ipmosflm() -> Path | None:
    """Locate the Mosflm binary (``ipmosflm``), matching iMosflm's search."""
    for key in ("MOSFLM_EXEC", "ELMO_IPMOSFLM"):
        value = os.environ.get(key)
        if value:
            path = Path(value).expanduser()
            if path.is_file():
                return path.resolve()
    which = shutil.which("ipmosflm")
    if which:
        return Path(which).resolve()
    for candidate in (
        Path("/Applications/ccp4-9/bin/ipmosflm"),
        Path("/Applications/ccp4-8/bin/ipmosflm"),
        Path(os.environ.get("CBIN", "")) / "ipmosflm",
        Path(os.environ.get("CCP4", "")) / "bin" / "ipmosflm",
    ):
        try:
            if candidate.is_file():
                return candidate.resolve()
        except OSError:
            continue
    return None


@dataclass
class MosflmMessage:
    """One framed message from Mosflm."""

    kind: str  # "xml" | "done" | "banner" | "other"
    text: str
    root: ET.Element | None = None

    @property
    def doctype(self) -> str | None:
        if self.root is None:
            return None
        return self.root.tag


@dataclass
class MosflmClient:
    """Blocking socket client for a single Mosflm process."""

    executable: Path
    work_dir: Path
    timeout_s: float = 120.0
    _server: socket.socket | None = field(default=None, init=False, repr=False)
    _conn: socket.socket | None = field(default=None, init=False, repr=False)
    _proc: subprocess.Popen | None = field(default=None, init=False, repr=False)
    _buffer: str = field(default="", init=False, repr=False)

    def start(self) -> None:
        """Start Mosflm and wait for its outbound socket connection."""
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("127.0.0.1", 0))
        self._server.listen(1)
        port = self._server.getsockname()[1]
        logger.info("Starting %s MOSFLMSOCKET %s in %s", self.executable, port, self.work_dir)
        self._proc = subprocess.Popen(
            [str(self.executable), "MOSFLMSOCKET", str(port)],
            cwd=str(self.work_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        self._server.settimeout(self.timeout_s)
        self._conn, addr = self._server.accept()
        self._conn.settimeout(self.timeout_s)
        logger.info("Mosflm connected from %s", addr)
        # Drain startup banner / processor_information.
        self.wait_for(doctypes=("processor_information",), also_done=True, timeout_s=10.0)

    def close(self) -> None:
        """Shut down Mosflm and release sockets."""
        try:
            if self._conn is not None:
                try:
                    self.send("exit")
                except OSError:
                    pass
                try:
                    self._conn.close()
                except OSError:
                    pass
        finally:
            self._conn = None
            if self._server is not None:
                try:
                    self._server.close()
                except OSError:
                    pass
                self._server = None
            if self._proc is not None and self._proc.poll() is None:
                try:
                    self._proc.terminate()
                    self._proc.wait(timeout=5)
                except Exception:
                    try:
                        self._proc.kill()
                    except Exception:
                        pass
            self._proc = None

    def __enter__(self) -> "MosflmClient":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def send(self, command: str) -> None:
        """Send one Mosflm keyword line."""
        if self._conn is None:
            raise RuntimeError("Mosflm client is not connected")
        line = command.rstrip("\n") + "\n"
        logger.debug("→ %s", command)
        self._conn.sendall(line.encode("ascii", errors="replace"))

    def send_many(self, commands: Iterable[str]) -> None:
        """Send several command lines."""
        for command in commands:
            if command is None:
                continue
            text = str(command).strip()
            if text:
                self.send(text)

    def _read_line(self, timeout_s: float | None = None) -> str | None:
        deadline = time.time() + (timeout_s if timeout_s is not None else self.timeout_s)
        assert self._conn is not None
        while time.time() < deadline:
            if "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                return line.strip("\r")
            remaining = max(0.0, deadline - time.time())
            ready, _, _ = select.select([self._conn], [], [], min(remaining, 0.25))
            if not ready:
                continue
            chunk = self._conn.recv(65536)
            if not chunk:
                return None
            self._buffer += chunk.decode("utf-8", errors="replace")
        raise TimeoutError("Timed out waiting for Mosflm output")

    def recv_message(self, timeout_s: float | None = None) -> MosflmMessage | None:
        """Receive the next framed Mosflm message."""
        while True:
            line = self._read_line(timeout_s=timeout_s)
            if line is None:
                return None
            if not line:
                continue
            if line == "<done>":
                return MosflmMessage(kind="done", text=line)
            if line.startswith("<?xml") or line.startswith("<") and "version=" not in line[:40]:
                # XML payload (often one line). Banner looks like <Mosflm version …>
                if line.startswith("<Mosflm "):
                    return MosflmMessage(kind="banner", text=line)
                try:
                    root = ET.fromstring(line)
                    return MosflmMessage(kind="xml", text=line, root=root)
                except ET.ParseError:
                    return MosflmMessage(kind="other", text=line)
            return MosflmMessage(kind="other", text=line)

    def wait_for(
        self,
        *,
        doctypes: Iterable[str],
        also_done: bool = False,
        timeout_s: float | None = None,
    ) -> list[MosflmMessage]:
        """Collect messages until one of *doctypes* (and optionally ``done``) arrives."""
        wanted = set(doctypes)
        collected: list[MosflmMessage] = []
        deadline = time.time() + (timeout_s if timeout_s is not None else self.timeout_s)
        saw_target = False
        while time.time() < deadline:
            remaining = max(0.05, deadline - time.time())
            msg = self.recv_message(timeout_s=remaining)
            if msg is None:
                break
            collected.append(msg)
            logger.debug("← %s %s", msg.kind, msg.doctype or msg.text[:80])
            if msg.kind == "xml" and msg.doctype in wanted:
                saw_target = True
                if not also_done:
                    return collected
            if also_done and msg.kind == "done" and (saw_target or not wanted):
                return collected
            if msg.kind == "done" and not wanted and also_done:
                return collected
        if saw_target:
            return collected
        raise TimeoutError(
            f"Timed out waiting for Mosflm XML {sorted(wanted)}; "
            f"got {[m.doctype or m.kind for m in collected]}"
        )

    def go(self, *, expect: Iterable[str], timeout_s: float | None = None) -> list[MosflmMessage]:
        """Send ``go`` and wait for the named response document type(s)."""
        self.send("go")
        return self.wait_for(doctypes=expect, also_done=True, timeout_s=timeout_s)
