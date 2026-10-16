"""JDXI Editor-like terminal / log viewer for DiffractionLab.

Tails a logfile into a read-only monospace ``QTextEdit`` (same pattern as
JDXI's ``LogViewer``), and also accepts live ``append_text`` pushes from
streaming subprocesses.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

_STYLE = """
QMainWindow, QWidget {
    background-color: #1e1e1e;
    color: #d4d4d4;
}
QTextEdit {
    background-color: #0d0d0d;
    color: #d4d4d4;
    border: 1px solid #3c3c3c;
    selection-background-color: #264f78;
}
QPushButton {
    background-color: #2d2d2d;
    color: #d4d4d4;
    border: 1px solid #3c3c3c;
    padding: 4px 10px;
}
QPushButton:hover {
    background-color: #3a3a3a;
}
QPushButton:checked {
    background-color: #0e639c;
    border-color: #1177bb;
}
QLabel {
    color: #9cdcfe;
}
"""


class TerminalLogViewer(QMainWindow):
    """Read-only terminal-style log viewer with file tailing and live append."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        log_file: Path | None = None,
        title: str = "DiffractionLab Terminal",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(900, 420)
        self.setStyleSheet(_STYLE)

        self.log_file: Path | None = Path(log_file) if log_file else None
        self.last_position = 0
        self.fast_refresh_mode = True

        main = QWidget()
        layout = QVBoxLayout(main)

        self.path_label = QLabel(self._path_caption())
        self.path_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.path_label.setWordWrap(True)
        layout.addWidget(self.path_label)

        self.log_text = QTextEdit()
        self.log_text.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.log_text.setReadOnly(True)
        font = QFont("Menlo")
        if not font.exactMatch():
            font = QFont("Courier New")
        font.setPointSize(11)
        self.log_text.setFont(font)
        layout.addWidget(self.log_text)

        buttons = QHBoxLayout()
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear_log)
        buttons.addWidget(clear_btn)

        self.refresh_mode_button = QPushButton("Fast Refresh (500ms)")
        self.refresh_mode_button.setCheckable(True)
        self.refresh_mode_button.setChecked(True)
        self.refresh_mode_button.clicked.connect(self.toggle_refresh_mode)
        buttons.addWidget(self.refresh_mode_button)

        reload_btn = QPushButton("Reload file")
        reload_btn.clicked.connect(self.load_log_file)
        buttons.addWidget(reload_btn)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.setCentralWidget(main)

        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_log_file)
        self.refresh_timer.start(500)

        if self.log_file is not None:
            self.load_log_file()

    def _path_caption(self) -> str:
        if self.log_file is None:
            return "No log file — waiting for live output…"
        return f"Tailing: {self.log_file}"

    def set_log_file(self, path: Path | None, *, reload: bool = True) -> None:
        """Point the viewer at *path* and optionally reload from the start."""
        self.log_file = Path(path) if path else None
        self.path_label.setText(self._path_caption())
        self.last_position = 0
        if reload:
            self.load_log_file()

    def append_text(self, text: str) -> None:
        """Append live text (e.g. from a streaming subprocess)."""
        if not text:
            return
        self.log_text.moveCursor(QTextCursor.MoveOperation.End)
        self.log_text.insertPlainText(text)
        self.log_text.moveCursor(QTextCursor.MoveOperation.End)

    def load_log_file(self) -> None:
        """Load the entire logfile into the text widget."""
        if self.log_file is None:
            return
        if not self.log_file.exists():
            self.log_text.setPlainText(
                f"Log file not found yet.\n\nExpected: {self.log_file}\n"
            )
            self.last_position = 0
            return
        try:
            content = self.log_file.read_text(encoding="utf-8", errors="replace")
            self.log_text.setPlainText(content)
            self.last_position = len(content.encode("utf-8", errors="replace"))
            # Prefer byte offset from tell() after read:
            with open(self.log_file, "rb") as handle:
                handle.seek(0, 2)
                self.last_position = handle.tell()
            self.log_text.moveCursor(QTextCursor.MoveOperation.End)
        except OSError as ex:
            self.log_text.setPlainText(f"Error reading log file: {ex}\n")

    def refresh_log_file(self) -> None:
        """Append any new bytes from the logfile (JDXI-style tail)."""
        if self.log_file is None or not self.log_file.exists():
            return
        try:
            with open(self.log_file, "rb") as handle:
                handle.seek(self.last_position)
                new_bytes = handle.read()
                if not new_bytes:
                    return
                self.last_position = handle.tell()
            text = new_bytes.decode("utf-8", errors="replace")
            self.append_text(text)
        except OSError:
            pass

    def toggle_refresh_mode(self) -> None:
        """Toggle between fast (500ms) and slow (30s) file polling."""
        self.fast_refresh_mode = not self.fast_refresh_mode
        if self.fast_refresh_mode:
            self.refresh_timer.setInterval(500)
            self.refresh_mode_button.setText("Fast Refresh (500ms)")
            self.refresh_mode_button.setChecked(True)
        else:
            self.refresh_timer.setInterval(30000)
            self.refresh_mode_button.setText("Slow Refresh (30s)")
            self.refresh_mode_button.setChecked(False)
        self.refresh_timer.start()

    def clear_log(self) -> None:
        """Clear the on-screen buffer (does not truncate the file)."""
        self.log_text.clear()
        if self.log_file is not None and self.log_file.exists():
            try:
                with open(self.log_file, "rb") as handle:
                    handle.seek(0, 2)
                    self.last_position = handle.tell()
            except OSError:
                self.last_position = 0
        else:
            self.last_position = 0

    def closeEvent(self, event) -> None:  # noqa: N802
        self.refresh_timer.stop()
        event.accept()
