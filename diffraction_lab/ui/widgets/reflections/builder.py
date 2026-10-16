"""
build_reflection_viewer
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget


def build_reflection_viewer(parent: QWidget) -> "ReflectionViewerWindow":
    """build reflection viewer"""
    from diffraction_lab.ui.widgets.reflections.window import ReflectionViewerWindow
    window = ReflectionViewerWindow(parent)
    window.setAttribute(Qt.WA_DeleteOnClose)
    window.show()
    window.raise_()
    window.activateWindow()
    if window._data is None:
        # Offer a file dialog when nothing is preloaded.
        window.open_file_dialog()
    return window
