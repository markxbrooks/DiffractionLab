"""Qt table model over :class:`~molib.xtal.reflections.model.ReflectionData`."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt

from molib.xtal.reflections.model import ReflectionData

OBSERVATION_ID_ROLE = Qt.ItemDataRole.UserRole + 1


class ReflectionTableModel(QAbstractTableModel):
    """Tabular view of filtered reflections with stable observation ids."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._data: ReflectionData | None = None
        self._indices: np.ndarray = np.zeros(0, dtype=np.int32)
        self._extra_columns: list[str] = []
        self._headers: list[str] = ["id", "H", "K", "L", "d (Å)"]

    def set_data(
        self,
        data: ReflectionData | None,
        *,
        row_indices: Sequence[int] | np.ndarray | None = None,
        value_columns: Sequence[str] | None = None,
    ) -> None:
        """Replace backing data and visible row index list."""
        self.beginResetModel()
        self._data = data
        if data is None:
            self._indices = np.zeros(0, dtype=np.int32)
            self._extra_columns = []
            self._headers = ["id", "H", "K", "L", "d (Å)"]
        else:
            if row_indices is None:
                self._indices = np.arange(data.n_reflections, dtype=np.int32)
            else:
                self._indices = np.asarray(row_indices, dtype=np.int32).ravel()
            self._extra_columns = list(value_columns or data.column_labels()[:4])
            self._headers = ["id", "H", "K", "L", "d (Å)"] + self._extra_columns
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        if parent.isValid():
            return 0
        return int(self._indices.shape[0])

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        if parent.isValid():
            return 0
        return len(self._headers)

    def headerData(  # noqa: N802
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            if 0 <= section < len(self._headers):
                return self._headers[section]
            return None
        if 0 <= section < self.rowCount():
            return str(section + 1)
        return None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or self._data is None:
            return None
        row = index.row()
        col = index.column()
        if row < 0 or row >= self._indices.shape[0]:
            return None
        obs_id = int(self._indices[row])
        if role == OBSERVATION_ID_ROLE:
            return obs_id
        if role not in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            return None

        data = self._data
        if col == 0:
            return obs_id
        if col == 1:
            return int(data.h[obs_id])
        if col == 2:
            return int(data.k[obs_id])
        if col == 3:
            return int(data.l[obs_id])
        if col == 4:
            if data.d_spacing is None:
                return ""
            d = float(data.d_spacing[obs_id])
            return f"{d:.3f}" if np.isfinite(d) else ""
        extra_i = col - 5
        if 0 <= extra_i < len(self._extra_columns):
            label = self._extra_columns[extra_i]
            if data.get_missing(label)[obs_id]:
                return ""
            return f"{float(data.get_values(label)[obs_id]):.4g}"
        return None

    def observation_id_at(self, row: int) -> int | None:
        """Return observation id for a table row."""
        if row < 0 or row >= self._indices.shape[0]:
            return None
        return int(self._indices[row])

    def row_for_observation_id(self, observation_id: int) -> int | None:
        """Return table row for an observation id, if visible."""
        matches = np.nonzero(self._indices == int(observation_id))[0]
        if matches.size == 0:
            return None
        return int(matches[0])

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:  # noqa: N802
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
