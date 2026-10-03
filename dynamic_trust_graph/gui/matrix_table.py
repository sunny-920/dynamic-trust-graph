from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QTableWidget, QTableWidgetItem


class MatrixTable(QTableWidget):
    cellValueEdited = Signal(str, str, float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._nodes: list[str] = []
        self._updating = False
        self._decimals = 2
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.itemChanged.connect(self._handle_item_changed)

    def set_matrix(
        self,
        nodes: list[str],
        matrix: np.ndarray,
        editable: bool = False,
        decimals: int = 2,
    ) -> None:
        self._updating = True
        self._nodes = list(nodes)
        self._decimals = decimals
        self.clear()
        self.setRowCount(len(nodes))
        self.setColumnCount(len(nodes))
        self.setHorizontalHeaderLabels(nodes)
        self.setVerticalHeaderLabels(nodes)

        for row, source in enumerate(nodes):
            for col, _target in enumerate(nodes):
                value = float(matrix[row, col])
                item = QTableWidgetItem(f"{value:.{decimals}f}")
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                flags = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
                if editable and source != nodes[col]:
                    flags |= Qt.ItemFlag.ItemIsEditable
                item.setFlags(flags)
                self.setItem(row, col, item)

        self._updating = False

    def _handle_item_changed(self, item: QTableWidgetItem) -> None:
        if self._updating:
            return
        row = item.row()
        col = item.column()
        if row < 0 or col < 0 or row >= len(self._nodes) or col >= len(self._nodes):
            return

        try:
            value = float(item.text())
        except ValueError:
            value = 0.0
        value = float(np.clip(value, 0.0, 1.0))

        self._updating = True
        item.setText(f"{value:.{self._decimals}f}")
        self._updating = False
        self.cellValueEdited.emit(self._nodes[row], self._nodes[col], value)
