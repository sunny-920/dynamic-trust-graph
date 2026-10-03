from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class ControlPanel(QWidget):
    addNodeRequested = Signal(str)
    removeNodeRequested = Signal(str)
    addEdgeRequested = Signal(str, str)
    removeEdgeRequested = Signal(str, str)
    randomGraphRequested = Signal(int, int)
    randomInteractionRequested = Signal()
    echoInteractionRequested = Signal()
    clearInteractionRequested = Signal()
    reportMisinformationRequested = Signal(str)
    rewardTrustRequested = Signal(str)
    randomTrustRequested = Signal()
    resetTrustRequested = Signal()
    weightsChanged = Signal(float, float, float)
    thresholdChanged = Signal(float)
    trustGateThresholdChanged = Signal(float)
    simulateRequested = Signal(str, float, bool)
    pathRequested = Signal(str, str, float)
    loadSampleRequested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._syncing = False
        self._weights = {"alpha": 0.3, "beta": 0.4, "gamma": 0.3}
        self._build_ui()

    def set_nodes(self, nodes: list[str]) -> None:
        combos = [
            self.remove_node_combo,
            self.edge_source_combo,
            self.edge_target_combo,
            self.trust_node_combo,
            self.diffusion_source_combo,
            self.path_source_combo,
            self.path_target_combo,
        ]
        previous = {combo: combo.currentText() for combo in combos}

        for combo in combos:
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(nodes)
            if previous[combo] in nodes:
                combo.setCurrentText(previous[combo])
            combo.blockSignals(False)

        if nodes:
            if self.path_target_combo.currentText() == self.path_source_combo.currentText():
                target_index = 1 if len(nodes) > 1 else 0
                self.path_target_combo.setCurrentIndex(target_index)

    def threshold(self) -> float:
        return float(self.threshold_spin.value())

    def trust_gate_threshold(self) -> float:
        return float(self.trust_gate_spin.value())

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        title = QLabel("動態信任圖模型")
        title.setObjectName("PanelTitle")
        layout.addWidget(title)

        layout.addWidget(self._build_node_group())
        layout.addWidget(self._build_edge_group())
        layout.addWidget(self._build_random_graph_group())
        layout.addWidget(self._build_interaction_group())
        layout.addWidget(self._build_trust_group())
        layout.addWidget(self._build_weight_group())
        layout.addWidget(self._build_analysis_group())

        sample_button = QPushButton("載入預設範例")
        sample_button.clicked.connect(self.loadSampleRequested.emit)
        layout.addWidget(sample_button)
        layout.addStretch(1)

    def _build_node_group(self) -> QGroupBox:
        group = QGroupBox("節點")
        grid = QGridLayout(group)
        self.node_input = QLineEdit()
        self.node_input.setPlaceholderText("例如 I")
        add_button = QPushButton("+ 新增節點")
        add_button.clicked.connect(self._emit_add_node)
        self.remove_node_combo = QComboBox()
        remove_button = QPushButton("刪除節點")
        remove_button.clicked.connect(
            lambda: self.removeNodeRequested.emit(self.remove_node_combo.currentText())
        )

        grid.addWidget(self.node_input, 0, 0)
        grid.addWidget(add_button, 0, 1)
        grid.addWidget(self.remove_node_combo, 1, 0)
        grid.addWidget(remove_button, 1, 1)
        return group

    def _build_edge_group(self) -> QGroupBox:
        group = QGroupBox("有向邊")
        grid = QGridLayout(group)
        self.edge_source_combo = QComboBox()
        self.edge_target_combo = QComboBox()
        add_button = QPushButton("+ 新增邊")
        remove_button = QPushButton("刪除邊")
        add_button.clicked.connect(self._emit_add_edge)
        remove_button.clicked.connect(self._emit_remove_edge)

        grid.addWidget(QLabel("From"), 0, 0)
        grid.addWidget(self.edge_source_combo, 0, 1)
        grid.addWidget(QLabel("To"), 1, 0)
        grid.addWidget(self.edge_target_combo, 1, 1)
        grid.addWidget(add_button, 2, 0)
        grid.addWidget(remove_button, 2, 1)
        return group

    def _build_random_graph_group(self) -> QGroupBox:
        group = QGroupBox("隨機生成圖")
        layout = QVBoxLayout(group)
        ranges = [
            (5, 10, "5-10 個節點"),
            (10, 30, "10-30 個節點"),
            (30, 50, "30-50 個節點"),
        ]
        for minimum, maximum, label in ranges:
            button = QPushButton(label)
            button.clicked.connect(
                lambda _checked=False, low=minimum, high=maximum: self.randomGraphRequested.emit(
                    low,
                    high,
                )
            )
            layout.addWidget(button)
        return group

    def _build_interaction_group(self) -> QGroupBox:
        group = QGroupBox("互動矩陣")
        layout = QGridLayout(group)
        random_button = QPushButton("隨機填入")
        echo_button = QPushButton("回聲室範例")
        clear_button = QPushButton("清空")
        random_button.clicked.connect(self.randomInteractionRequested.emit)
        echo_button.clicked.connect(self.echoInteractionRequested.emit)
        clear_button.clicked.connect(self.clearInteractionRequested.emit)
        layout.addWidget(random_button, 0, 0)
        layout.addWidget(echo_button, 0, 1)
        layout.addWidget(clear_button, 1, 0, 1, 2)
        return group

    def _build_trust_group(self) -> QGroupBox:
        group = QGroupBox("信任更新")
        layout = QVBoxLayout(group)
        self.trust_node_combo = QComboBox()
        report_button = QPushButton("通報假訊息 -0.20")
        reward_button = QPushButton("通報優良互動 +0.05")
        random_button = QPushButton("隨機產生信任值")
        reset_button = QPushButton("重置信任")
        report_button.clicked.connect(
            lambda: self.reportMisinformationRequested.emit(self.trust_node_combo.currentText())
        )
        reward_button.clicked.connect(
            lambda: self.rewardTrustRequested.emit(self.trust_node_combo.currentText())
        )
        random_button.clicked.connect(self.randomTrustRequested.emit)
        reset_button.clicked.connect(self.resetTrustRequested.emit)
        layout.addWidget(self.trust_node_combo)
        layout.addWidget(report_button)
        layout.addWidget(reward_button)
        layout.addWidget(random_button)
        layout.addWidget(reset_button)
        return group

    def _build_weight_group(self) -> QGroupBox:
        group = QGroupBox("Score 權重")
        form = QFormLayout(group)
        self.alpha_slider = self._make_weight_slider("alpha")
        self.beta_slider = self._make_weight_slider("beta")
        self.gamma_slider = self._make_weight_slider("gamma")
        self.alpha_label = QLabel()
        self.beta_label = QLabel()
        self.gamma_label = QLabel()
        self.trust_gate_spin = QDoubleSpinBox()
        self.trust_gate_spin.setRange(0.0, 1.0)
        self.trust_gate_spin.setSingleStep(0.05)
        self.trust_gate_spin.setDecimals(2)
        self.trust_gate_spin.setValue(0.20)
        self.trust_gate_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.trust_gate_spin.valueChanged.connect(
            lambda value: self.trustGateThresholdChanged.emit(float(value))
        )
        trust_gate_minus_button = QPushButton("-")
        trust_gate_plus_button = QPushButton("+")
        trust_gate_minus_button.setFixedWidth(34)
        trust_gate_plus_button.setFixedWidth(34)
        trust_gate_minus_button.clicked.connect(lambda: self._step_trust_gate_threshold(-0.05))
        trust_gate_plus_button.clicked.connect(lambda: self._step_trust_gate_threshold(0.05))
        trust_gate_layout = QHBoxLayout()
        trust_gate_layout.addWidget(trust_gate_minus_button)
        trust_gate_layout.addWidget(self.trust_gate_spin)
        trust_gate_layout.addWidget(trust_gate_plus_button)
        form.addRow(self.alpha_label, self.alpha_slider)
        form.addRow(self.beta_label, self.beta_slider)
        form.addRow(self.gamma_label, self.gamma_slider)
        form.addRow("信任限流閥值", trust_gate_layout)
        self._sync_slider_values()
        return group

    def _build_analysis_group(self) -> QGroupBox:
        group = QGroupBox("擴散與路徑")
        layout = QVBoxLayout(group)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setValue(0.40)
        self.threshold_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.threshold_spin.valueChanged.connect(lambda value: self.thresholdChanged.emit(float(value)))
        threshold_minus_button = QPushButton("-")
        threshold_plus_button = QPushButton("+")
        threshold_minus_button.setFixedWidth(34)
        threshold_plus_button.setFixedWidth(34)
        threshold_minus_button.clicked.connect(lambda: self._step_threshold(-0.05))
        threshold_plus_button.clicked.connect(lambda: self._step_threshold(0.05))
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(threshold_minus_button)
        threshold_layout.addWidget(self.threshold_spin)
        threshold_layout.addWidget(threshold_plus_button)

        self.diffusion_source_combo = QComboBox()
        simulate_button = QPushButton("模擬擴散")
        simulate_button.clicked.connect(self._emit_simulate)

        self.path_source_combo = QComboBox()
        self.path_target_combo = QComboBox()
        self.path_source_combo.currentTextChanged.connect(self._ensure_path_target_differs)
        path_button = QPushButton("搜尋高風險路徑")
        path_button.clicked.connect(self._emit_path)

        form = QFormLayout()
        form.addRow("門檻", threshold_layout)
        form.addRow("擴散起點", self.diffusion_source_combo)
        layout.addLayout(form)
        layout.addWidget(simulate_button)

        path_layout = QHBoxLayout()
        path_layout.addWidget(self.path_source_combo)
        path_layout.addWidget(self.path_target_combo)
        layout.addLayout(path_layout)
        layout.addWidget(path_button)
        return group

    def _make_weight_slider(self, key: str) -> QSlider:
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 100)
        slider.valueChanged.connect(lambda value, name=key: self._weight_slider_changed(name, value))
        return slider

    def _weight_slider_changed(self, changed_key: str, value: int) -> None:
        if self._syncing:
            return
        new_value = max(0.0, min(1.0, value / 100.0))
        other_keys = [key for key in self._weights if key != changed_key]
        remaining = 1.0 - new_value
        other_total = sum(self._weights[key] for key in other_keys)

        self._weights[changed_key] = new_value
        if other_total <= 0:
            for key in other_keys:
                self._weights[key] = remaining / len(other_keys)
        else:
            for key in other_keys:
                self._weights[key] = self._weights[key] / other_total * remaining

        self._sync_slider_values()
        self.weightsChanged.emit(
            self._weights["alpha"],
            self._weights["beta"],
            self._weights["gamma"],
        )

    def _sync_slider_values(self) -> None:
        self._syncing = True
        self.alpha_slider.setValue(round(self._weights["alpha"] * 100))
        self.beta_slider.setValue(round(self._weights["beta"] * 100))
        self.gamma_slider.setValue(round(self._weights["gamma"] * 100))
        self.alpha_label.setText(f"α 關係 {self._weights['alpha']:.2f}")
        self.beta_label.setText(f"β 互動 {self._weights['beta']:.2f}")
        self.gamma_label.setText(f"γ 信任 {self._weights['gamma']:.2f}")
        self._syncing = False

    def _emit_add_node(self) -> None:
        node = self.node_input.text().strip()
        self.addNodeRequested.emit(node)
        self.node_input.clear()

    def _emit_add_edge(self) -> None:
        self.addEdgeRequested.emit(
            self.edge_source_combo.currentText(),
            self.edge_target_combo.currentText(),
        )

    def _emit_remove_edge(self) -> None:
        self.removeEdgeRequested.emit(
            self.edge_source_combo.currentText(),
            self.edge_target_combo.currentText(),
        )

    def _emit_simulate(self) -> None:
        self.simulateRequested.emit(
            self.diffusion_source_combo.currentText(),
            float(self.threshold_spin.value()),
            False,
        )

    def _emit_path(self) -> None:
        self.pathRequested.emit(
            self.path_source_combo.currentText(),
            self.path_target_combo.currentText(),
            float(self.threshold_spin.value()),
        )

    def _step_threshold(self, delta: float) -> None:
        next_value = max(0.0, min(1.0, self.threshold_spin.value() + delta))
        self.threshold_spin.setValue(round(next_value, 2))

    def _step_trust_gate_threshold(self, delta: float) -> None:
        next_value = max(0.0, min(1.0, self.trust_gate_spin.value() + delta))
        self.trust_gate_spin.setValue(round(next_value, 2))

    def _ensure_path_target_differs(self) -> None:
        if self.path_target_combo.currentText() != self.path_source_combo.currentText():
            return
        if self.path_target_combo.count() <= 1:
            return
        next_index = (self.path_target_combo.currentIndex() + 1) % self.path_target_combo.count()
        self.path_target_combo.setCurrentIndex(next_index)
