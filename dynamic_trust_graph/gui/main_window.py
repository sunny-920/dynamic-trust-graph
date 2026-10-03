from __future__ import annotations

import networkx as nx

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from dynamic_trust_graph.core.centrality import (
    degree_metrics,
    high_risk_nodes,
    pagerank_scores,
    push_rankings,
    scc_lookup,
    strongly_connected_component_analysis,
    top_pagerank,
)
from dynamic_trust_graph.core.diffusion import DiffusionResult, simulate_bfs_diffusion
from dynamic_trust_graph.core.graph_model import DynamicTrustGraph
from dynamic_trust_graph.core.path_analysis import PathResult, find_high_risk_path
from dynamic_trust_graph.gui.control_panel import ControlPanel
from dynamic_trust_graph.gui.graph_canvas import GraphCanvas
from dynamic_trust_graph.gui.matrix_table import MatrixTable


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.model = DynamicTrustGraph.sample()
        self.diffusion_result: DiffusionResult | None = None
        self.static_result: DiffusionResult | None = None
        self.dynamic_result: DiffusionResult | None = None
        self.path_result: PathResult | None = None
        self.threshold = 0.40
        self._updating_trust_table = False

        self.setWindowTitle("動態信任圖模型：假訊息傳播分析")
        self.resize(1480, 900)
        self._build_ui()
        self._connect_signals()
        self._apply_style()
        self._refresh_all()

    def _build_ui(self) -> None:
        root = QSplitter(Qt.Orientation.Horizontal)
        self.control_panel = ControlPanel()
        control_scroll = QScrollArea()
        control_scroll.setWidgetResizable(True)
        control_scroll.setWidget(self.control_panel)
        control_scroll.setMinimumWidth(330)
        control_scroll.setMaximumWidth(390)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_relation_tab(), "關係圖")
        self.tabs.addTab(self._build_interaction_tab(), "互動矩陣")
        self.tabs.addTab(self._build_trust_tab(), "信任圖")
        self.tabs.addTab(self._build_score_tab(), "Score 分析")
        self.tabs.addTab(self._build_graph_analysis_tab(), "圖形分析")
        self.tabs.addTab(self._build_push_tab(), "推播順序")

        root.addWidget(control_scroll)
        root.addWidget(self.tabs)
        root.setStretchFactor(0, 0)
        root.setStretchFactor(1, 1)
        self.setCentralWidget(root)
        self.setStatusBar(QStatusBar())

    def _build_relation_tab(self) -> QWidget:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        self.relation_canvas = GraphCanvas("有向社群關係圖")
        self.relation_matrix_table = MatrixTable()
        self.degree_table = QTableWidget()
        self._configure_table(self.degree_table)

        side = QWidget()
        side_layout = QVBoxLayout(side)
        relation_note = QLabel("關係權重：單向資訊流 R=0.7；雙向互相關注/互傳 R=1.0。")
        relation_note.setWordWrap(True)
        side_layout.addWidget(relation_note)
        side_layout.addWidget(QLabel("關係矩陣 R"))
        side_layout.addWidget(self.relation_matrix_table, 3)
        side_layout.addWidget(QLabel("入度 / 出度"))
        side_layout.addWidget(self.degree_table, 2)

        layout.addWidget(self.relation_canvas, 3)
        layout.addWidget(side, 2)
        return tab

    def _build_interaction_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        note = QLabel(
            "互動值說明\n"
            "無互動 0.00-0.20：幾乎沒有互動\n"
            "低互動 0.21-0.40：偶爾按讚或觀看\n"
            "中互動 0.41-0.70：常按讚、偶爾留言\n"
            "高互動 0.71-1.00：常留言、分享、點擊"
        )
        note.setObjectName("HelpNote")
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addWidget(QLabel("互動矩陣 I"))
        self.interaction_matrix_table = MatrixTable()
        layout.addWidget(self.interaction_matrix_table)
        return tab

    def _build_trust_tab(self) -> QWidget:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        self.trust_canvas = GraphCanvas("信任圖")
        self.trust_node_table = QTableWidget()
        self._configure_table(self.trust_node_table)

        side = QWidget()
        side_layout = QVBoxLayout(side)
        trust_note = QLabel(
            "信任值說明\n"
            "高信任 0.80-1.00：經常傳好訊息，幾乎不傳假訊息\n"
            "普通信任 0.50-0.79：好訊息與一般訊息為主\n"
            "低信任 0.20-0.49：曾多次傳假訊息\n"
            "高風險 0.00-0.19：經常傳假訊息"
        )
        trust_note.setObjectName("HelpNote")
        trust_note.setWordWrap(True)
        side_layout.addWidget(trust_note)
        side_layout.addWidget(QLabel("節點信任值 T"))
        side_layout.addWidget(self.trust_node_table)
        layout.addWidget(self.trust_canvas, 3)
        layout.addWidget(side, 2)
        return tab

    def _build_score_tab(self) -> QWidget:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        self.score_canvas = GraphCanvas("Score 圖與 PageRank")
        self.score_matrix_table = MatrixTable()
        self.pagerank_table = QTableWidget()
        self.scc_table = QTableWidget()
        self._configure_table(self.pagerank_table)
        self._configure_table(self.scc_table)

        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.addWidget(QLabel("Score 矩陣 S"))
        side_layout.addWidget(self.score_matrix_table, 2)
        side_layout.addWidget(QLabel("PageRank 影響力"))
        side_layout.addWidget(self.pagerank_table, 1)
        side_layout.addWidget(QLabel("SCC 回聲室"))
        side_layout.addWidget(self.scc_table, 1)

        layout.addWidget(self.score_canvas, 3)
        layout.addWidget(side, 2)
        return tab

    def _build_push_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel("推播順序列表：每個節點依指向它的來源節點 Score 由高到低排序"))
        self.push_table = QTableWidget()
        self._configure_table(self.push_table)
        self.push_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.push_table, 3)
        layout.addWidget(QLabel("擴散與路徑結果"))
        self.analysis_text = QTextEdit()
        self.analysis_text.setReadOnly(True)
        self.analysis_text.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.analysis_text.setFont(QFont("Consolas", 10))
        layout.addWidget(self.analysis_text, 2)
        return tab

    def _build_graph_analysis_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        graph_grid = QGridLayout()
        self.top_influence_canvas = GraphCanvas("Top 3 高影響力節點")
        self.high_risk_canvas = GraphCanvas("高風險節點")
        self.scc_canvas = GraphCanvas("SCC 回聲室分析")
        graph_grid.addWidget(self.top_influence_canvas, 0, 0)
        graph_grid.addWidget(self.high_risk_canvas, 0, 1)
        graph_grid.addWidget(self.scc_canvas, 0, 2)
        layout.addLayout(graph_grid, 2)
        self.graph_analysis_text = QTextEdit()
        self.graph_analysis_text.setReadOnly(True)
        self.graph_analysis_text.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.graph_analysis_text.setFont(QFont("Consolas", 10))
        layout.addWidget(self.graph_analysis_text, 1)
        return tab

    def _configure_table(self, table: QTableWidget) -> None:
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().setVisible(False)
        table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def _connect_signals(self) -> None:
        panel = self.control_panel
        panel.addNodeRequested.connect(self._add_node)
        panel.removeNodeRequested.connect(self._remove_node)
        panel.addEdgeRequested.connect(self._add_edge)
        panel.removeEdgeRequested.connect(self._remove_edge)
        panel.randomGraphRequested.connect(self._generate_random_graph)
        panel.randomInteractionRequested.connect(self._random_interaction)
        panel.echoInteractionRequested.connect(self._echo_interaction)
        panel.clearInteractionRequested.connect(self._clear_interaction)
        panel.reportMisinformationRequested.connect(self._report_misinformation)
        panel.rewardTrustRequested.connect(self._reward_trust)
        panel.randomTrustRequested.connect(self._random_trust)
        panel.resetTrustRequested.connect(self._reset_trust)
        panel.weightsChanged.connect(self._set_weights)
        panel.thresholdChanged.connect(self._set_threshold)
        panel.trustGateThresholdChanged.connect(self._set_trust_gate_threshold)
        panel.simulateRequested.connect(self._simulate_diffusion)
        panel.pathRequested.connect(self._find_path)
        panel.loadSampleRequested.connect(self._load_sample)
        self.interaction_matrix_table.cellValueEdited.connect(self._set_interaction_cell)
        self.trust_node_table.itemChanged.connect(self._set_trust_node_item)

    def _apply_style(self) -> None:
        QApplication.instance().setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #f8fafc;
                color: #111827;
                font-family: "Microsoft JhengHei", "Segoe UI", sans-serif;
            }
            QGroupBox {
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 12px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QPushButton {
                background: #ffffff;
                border: 1px solid #94a3b8;
                border-radius: 5px;
                padding: 6px 9px;
            }
            QPushButton:hover {
                background: #e0f2fe;
                border-color: #0284c7;
            }
            QLineEdit, QComboBox, QDoubleSpinBox, QTextEdit, QTableWidget {
                background: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 4px;
            }
            QTabWidget::pane {
                border: 1px solid #cbd5e1;
                background: #ffffff;
            }
            QTabBar::tab {
                background: #e2e8f0;
                padding: 8px 12px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background: #ffffff;
                border: 1px solid #cbd5e1;
                border-bottom: 1px solid #ffffff;
            }
            QLabel#PanelTitle {
                font-size: 20px;
                font-weight: 700;
                padding: 4px 0 8px 0;
            }
            QLabel#HelpNote {
                background: #f1f5f9;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                padding: 10px 12px;
                color: #334155;
            }
            """
        )

    def _refresh_all(self) -> None:
        nodes = self.model.nodes
        relation = self.model.relation_matrix
        interaction = self.model.interaction
        trust = self.model.trust
        score = self.model.score_matrix
        threshold = self.threshold

        self.control_panel.set_nodes(nodes)
        self.relation_matrix_table.set_matrix(nodes, relation, editable=False, decimals=1)
        self.interaction_matrix_table.set_matrix(nodes, interaction, editable=True, decimals=2)
        self.score_matrix_table.set_matrix(nodes, score, editable=False, decimals=2)

        trust_node_scores = self.model.node_trust_scores
        low_trust_nodes = self._low_trust_nodes(trust_node_scores)
        effective_relation = self._effective_relation(relation, low_trust_nodes)
        pagerank = pagerank_scores(nodes, score)
        sccs = strongly_connected_component_analysis(nodes, effective_relation, interaction, trust)
        scc_map = scc_lookup(sccs)
        degree = degree_metrics(nodes, relation)

        relation_edges = list(self.model.graph.edges())
        trust_edges = self.model.trust_graph_edges
        score_edges = self._score_edges(score, threshold)
        score_weights = {
            edge: float(score[nodes.index(edge[0]), nodes.index(edge[1])]) for edge in score_edges
        }

        highlight_nodes = set()
        highlight_edges = set()
        if self.diffusion_result:
            highlight_nodes.update(self.diffusion_result.infected)
            highlight_edges.update(self.diffusion_result.propagation_edges)
        if self.path_result and self.path_result.success:
            highlight_nodes.update(self.path_result.path)
            highlight_edges.update(self.path_result.edges)

        self.relation_canvas.draw_graph(
            nodes,
            relation_edges,
            title="有向社群關係圖",
            edge_mode="relation",
        )
        self.trust_canvas.draw_graph(
            nodes,
            trust_edges,
            title=f"信任完全圖：節點平均信任值，低於 {self.model.trust_gate_threshold:.2f} 移除對外邊",
            node_label_values=trust_node_scores,
            node_value_threshold=self.model.trust_gate_threshold,
            edge_mode="trust_node",
        )
        self.score_canvas.draw_graph(
            nodes,
            score_edges,
            weights=score_weights,
            title=f"Score 圖 S[u][v] >= {threshold:.2f}",
            node_scores=pagerank,
            highlight_nodes=highlight_nodes,
            highlight_edges=highlight_edges,
            muted_nodes=low_trust_nodes,
            scc_lookup=scc_map,
            edge_mode="score",
        )

        self._fill_degree_table(nodes, degree, pagerank, scc_map)
        self._fill_trust_node_table(nodes, trust_node_scores)
        self._fill_pagerank_table(pagerank)
        self._fill_scc_table(sccs)
        self._fill_push_table(score)
        self._update_result_graphs(score, pagerank, sccs)
        self._update_analysis_text(score, pagerank, sccs)
        self._update_graph_analysis_text(pagerank, sccs)

    def _score_edges(self, score, threshold: float) -> list[tuple[str, str]]:
        edges: list[tuple[str, str]] = []
        for row, source in enumerate(self.model.nodes):
            for col, target in enumerate(self.model.nodes):
                if source != target and float(score[row, col]) >= threshold:
                    edges.append((source, target))
        return edges

    def _low_trust_nodes(self, trust_node_scores: dict[str, float]) -> set[str]:
        return {
            node
            for node, value in trust_node_scores.items()
            if value < self.model.trust_gate_threshold
        }

    def _effective_relation(self, relation, low_trust_nodes: set[str]):
        effective_relation = relation.copy()
        for node in low_trust_nodes:
            if node in self.model.nodes:
                effective_relation[self.model.nodes.index(node), :] = 0.0
        return effective_relation

    def _relation_edges_from_matrix(self, relation) -> list[tuple[str, str]]:
        edges: list[tuple[str, str]] = []
        for row, source in enumerate(self.model.nodes):
            for col, target in enumerate(self.model.nodes):
                if source != target and float(relation[row, col]) > 0.0:
                    edges.append((source, target))
        return edges

    def _update_result_graphs(self, score, pagerank, sccs) -> None:
        nodes = self.model.nodes
        relation = self.model.relation_matrix
        score_edges = self._score_edges(score, self.threshold)
        score_weights = {
            edge: float(score[nodes.index(edge[0]), nodes.index(edge[1])]) for edge in score_edges
        }
        top_nodes = {node for node, _value in top_pagerank(pagerank, 3)}
        risk_nodes = {
            node for node, _value in high_risk_nodes(nodes, score, pagerank, limit=5)
        }
        low_trust_nodes = self._low_trust_nodes(self.model.node_trust_scores)
        effective_relation = self._effective_relation(relation, low_trust_nodes)
        effective_relation_edges = self._relation_edges_from_matrix(effective_relation)

        self.top_influence_canvas.draw_graph(
            nodes,
            score_edges,
            weights=score_weights,
            title="Top 3 高影響力節點",
            node_scores=pagerank,
            highlight_nodes=top_nodes,
            muted_nodes=low_trust_nodes,
            edge_mode="score",
        )
        self.high_risk_canvas.draw_graph(
            nodes,
            score_edges,
            weights=score_weights,
            title="高風險節點",
            node_scores=pagerank,
            highlight_nodes=risk_nodes,
            muted_nodes=low_trust_nodes,
            edge_mode="score",
        )
        self.scc_canvas.draw_graph(
            nodes,
            effective_relation_edges,
            title="SCC 回聲室分析",
            scc_lookup=scc_lookup(sccs),
            muted_nodes=low_trust_nodes,
            edge_mode="relation",
        )

    def _fill_degree_table(
        self,
        nodes: list[str],
        degree: dict[str, dict[str, int]],
        pagerank: dict[str, float],
        scc_map: dict[str, int],
    ) -> None:
        headers = ["節點", "Out-degree", "In-degree", "PageRank", "SCC"]
        self.degree_table.setColumnCount(len(headers))
        self.degree_table.setHorizontalHeaderLabels(headers)
        self.degree_table.setRowCount(len(nodes))
        for row, node in enumerate(nodes):
            values = [
                node,
                str(degree[node]["out_degree"]),
                str(degree[node]["in_degree"]),
                f"{pagerank.get(node, 0.0):.4f}",
                str(scc_map.get(node, "-")),
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.degree_table.setItem(row, col, item)

    def _fill_trust_node_table(
        self,
        nodes: list[str],
        trust_node_scores: dict[str, float],
    ) -> None:
        headers = ["節點", "信任值"]
        self._updating_trust_table = True
        self.trust_node_table.clear()
        self.trust_node_table.setColumnCount(len(headers))
        self.trust_node_table.setHorizontalHeaderLabels(headers)
        self.trust_node_table.setRowCount(len(nodes))

        for row, node in enumerate(nodes):
            node_item = QTableWidgetItem(node)
            node_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            node_item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            self.trust_node_table.setItem(row, 0, node_item)

            value = trust_node_scores.get(node, 0.0)
            value_item = QTableWidgetItem(f"{value:.2f}")
            value_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            value_item.setFlags(
                Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsEditable
            )
            self.trust_node_table.setItem(row, 1, value_item)

        self._updating_trust_table = False

    def _fill_pagerank_table(self, pagerank: dict[str, float]) -> None:
        ranking = top_pagerank(pagerank, limit=len(pagerank))
        headers = ["排名", "節點", "PageRank"]
        self.pagerank_table.setColumnCount(len(headers))
        self.pagerank_table.setHorizontalHeaderLabels(headers)
        self.pagerank_table.setRowCount(len(ranking))
        for row, (node, value) in enumerate(ranking, start=1):
            for col, text in enumerate([str(row), node, f"{value:.5f}"]):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.pagerank_table.setItem(row - 1, col, item)

    def _fill_scc_table(self, sccs) -> None:
        headers = ["SCC", "節點", "Avg I", "Avg T", "狀態"]
        self.scc_table.setColumnCount(len(headers))
        self.scc_table.setHorizontalHeaderLabels(headers)
        self.scc_table.setRowCount(len(sccs))
        for row, summary in enumerate(sccs):
            status = "高風險回聲室" if summary.high_risk_echo_chamber else (
                "潛在回聲室" if summary.potential_echo_chamber else "一般群體"
            )
            values = [
                str(summary.component_id),
                ", ".join(summary.nodes),
                f"{summary.avg_interaction:.2f}",
                f"{summary.avg_trust:.2f}",
                status,
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.scc_table.setItem(row, col, item)

    def _fill_push_table(self, score) -> None:
        rankings = push_rankings(self.model.nodes, score, self.threshold)
        headers = ["接收節點 / 推播來源", "順位", "Score"]
        rows: list[tuple[str, str, str, float | None, bool]] = []
        for target in self.model.nodes:
            sources = rankings.get(target, [])
            rows.append(
                (
                    f"{target} 的推播來源順序（指向 {target}，Score 由高到低，門檻 >= {self.threshold:.2f}）",
                    "",
                    "",
                    None,
                    True,
                )
            )
            if not sources:
                rows.append(("無符合門檻的推播來源", "-", "-", None, False))
                continue
            for rank, (source, value) in enumerate(sources, start=1):
                rows.append((source, str(rank), f"{value:.2f}", float(value), False))

        self.push_table.setColumnCount(len(headers))
        self.push_table.setHorizontalHeaderLabels(headers)
        self.push_table.clearSpans()
        self.push_table.setRowCount(len(rows))
        self.push_table.verticalHeader().setVisible(False)
        self.push_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        for row, (node_text, rank, score_text, value, is_group_header) in enumerate(rows):
            if is_group_header:
                item = QTableWidgetItem(node_text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
                item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
                self.push_table.setItem(row, 0, item)
                self.push_table.setSpan(row, 0, 1, len(headers))
                continue

            values = [node_text, rank, score_text]
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
                self.push_table.setItem(row, col, item)
        self.push_table.resizeRowsToContents()

    def _update_analysis_text(self, score, pagerank, sccs) -> None:
        lines: list[str] = []
        lines.append("推播門檻")
        lines.append(f"Threshold = {self.threshold:.2f}")
        lines.append(f"信任限流閥值 = {self.model.trust_gate_threshold:.2f}")

        if self.static_result:
            static_count = self.static_result.infected_count
            lines.append("")
            lines.append("Static Mode 擴散結果")
            lines.append(f"Static Mode 感染節點數：{static_count}")
            lines.append(f"Static 傳播層數：{self.static_result.depth}")
            lines.append(
                "Static 是否傳遍全圖："
                + ("是" if static_count == len(self.model.nodes) else "否")
            )
            lines.append("")
            lines.append("BFS 層級")
            selected = self.static_result
            lines.append("目前顯示：Static")
            for index, layer in enumerate(selected.layers):
                lines.append(f"Layer {index}: {', '.join(layer)}")
            if selected.propagation_edges:
                edges = " | ".join(f"{source}->{target}" for source, target in selected.propagation_edges)
                lines.append(f"傳播路徑：{edges}")

        if self.path_result:
            lines.append("")
            lines.append("Dijkstra 高風險路徑")
            lines.append(self.path_result.message)
            if self.path_result.success:
                score_text = " -> ".join(f"{score:.2f}" for score in self.path_result.edge_scores)
                lines.append(f"邊分數：{score_text}")
                lines.append(f"總成本：{self.path_result.total_cost:.3f}")

        self.analysis_text.setPlainText("\n".join(lines))

    def _update_graph_analysis_text(self, pagerank, sccs) -> None:
        score = self.model.score_matrix
        degree = degree_metrics(self.model.nodes, score)
        lines: list[str] = []
        lines.append("Top 3 高影響力節點")
        for index, (node, value) in enumerate(top_pagerank(pagerank, 3), start=1):
            lines.append(f"{index}. {node}: PageRank = {value:.5f}")

        lines.append("")
        lines.append("高風險節點")
        for index, (node, risk) in enumerate(
            high_risk_nodes(self.model.nodes, score, pagerank, 5),
            start=1,
        ):
            lines.append(
                f"{index}. {node}: risk={risk:.3f}, "
                f"out={degree[node]['out_degree']}, in={degree[node]['in_degree']}"
            )

        lines.append("")
        lines.append("SCC 回聲室分析")
        for summary in sccs:
            status = "高風險回聲室" if summary.high_risk_echo_chamber else (
                "潛在回聲室" if summary.potential_echo_chamber else "一般群體"
            )
            lines.append(
                f"SCC {summary.component_id}: {', '.join(summary.nodes)} | "
                f"avg I={summary.avg_interaction:.2f}, avg T={summary.avg_trust:.2f} | {status}"
            )
        self.graph_analysis_text.setPlainText("\n".join(lines))

    def _clear_transient_results(self) -> None:
        self.diffusion_result = None
        self.static_result = None
        self.dynamic_result = None
        self.path_result = None

    def _add_node(self, node: str) -> None:
        if self.model.add_node(node):
            self._clear_transient_results()
            self.statusBar().showMessage(f"已新增節點 {node.strip()}", 3000)
        self._refresh_all()

    def _remove_node(self, node: str) -> None:
        if self.model.remove_node(node):
            self._clear_transient_results()
            self.statusBar().showMessage(f"已刪除節點 {node}", 3000)
        self._refresh_all()

    def _add_edge(self, source: str, target: str) -> None:
        if self.model.add_edge(source, target):
            self._clear_transient_results()
            self.statusBar().showMessage(f"已新增有向邊 {source}->{target}", 3000)
        elif source == target:
            self.statusBar().showMessage("新增邊失敗：起點與終點需不同", 3000)
        self._refresh_all()

    def _remove_edge(self, source: str, target: str) -> None:
        if self.model.remove_edge(source, target):
            self._clear_transient_results()
            self.statusBar().showMessage(f"已刪除有向邊 {source}->{target}", 3000)
        self._refresh_all()

    def _generate_random_graph(self, min_nodes: int, max_nodes: int) -> None:
        current_weights = self.model.weights
        generated = DynamicTrustGraph.random_scenario(min_nodes, max_nodes)
        generated.set_weights(
            current_weights.alpha,
            current_weights.beta,
            current_weights.gamma,
        )
        generated.set_trust_gate_threshold(self.control_panel.trust_gate_threshold())
        self.model = generated
        self._clear_transient_results()

        low_trust_count = len(self._low_trust_nodes(self.model.node_trust_scores))
        isolated_count = sum(
            1 for node in self.model.nodes if self.model.graph.degree(node) == 0
        )
        cycle_group_count = sum(
            1
            for component in nx.strongly_connected_components(self.model.graph)
            if len(component) > 1
        )
        self.statusBar().showMessage(
            f"已生成 {len(self.model.nodes)} 個節點、"
            f"{self.model.graph.number_of_edges()} 條邊，"
            f"{isolated_count} 個孤立點、"
            f"{cycle_group_count} 個循環群、"
            f"{low_trust_count} 個低信任節點",
            5000,
        )
        self._refresh_all()

    def _random_interaction(self) -> None:
        self.model.randomize_interaction()
        self._clear_transient_results()
        self.statusBar().showMessage("已隨機產生互動矩陣", 3000)
        self._refresh_all()

    def _echo_interaction(self) -> None:
        self.model.build_echo_chamber_interaction()
        self._clear_transient_results()
        self.statusBar().showMessage("已產生回聲室互動範例", 3000)
        self._refresh_all()

    def _clear_interaction(self) -> None:
        self.model.clear_interaction()
        self._clear_transient_results()
        self.statusBar().showMessage("已清空互動矩陣", 3000)
        self._refresh_all()

    def _report_misinformation(self, node: str) -> None:
        self.model.report_misinformation(node)
        self._clear_transient_results()
        self.statusBar().showMessage(f"{node} 的對外信任已 -0.20", 3000)
        self._refresh_all()

    def _reward_trust(self, node: str) -> None:
        self.model.reward_good_interaction(node)
        self._clear_transient_results()
        self.statusBar().showMessage(f"{node} 的對外信任已 +0.05", 3000)
        self._refresh_all()

    def _random_trust(self) -> None:
        self.model.randomize_node_trust()
        self._clear_transient_results()
        self.statusBar().showMessage("已隨機產生各節點信任值", 3000)
        self._refresh_all()

    def _reset_trust(self) -> None:
        self.model.reset_trust()
        self._clear_transient_results()
        self.statusBar().showMessage("已重置信任值", 3000)
        self._refresh_all()

    def _set_weights(self, alpha: float, beta: float, gamma: float) -> None:
        self.model.set_weights(alpha, beta, gamma)
        self._clear_transient_results()
        self._refresh_all()

    def _set_threshold(self, threshold: float) -> None:
        self.threshold = float(threshold)
        self._refresh_all()

    def _set_trust_gate_threshold(self, threshold: float) -> None:
        self.model.set_trust_gate_threshold(threshold)
        self._clear_transient_results()
        self._refresh_all()

    def _simulate_diffusion(self, source: str, threshold: float, dynamic_mode: bool) -> None:
        if source not in self.model.nodes:
            return
        self.path_result = None
        relation = self.model.relation_matrix
        interaction = self.model.interaction
        trust = self.model.trust
        weights = self.model.weights
        self.static_result = simulate_bfs_diffusion(
            self.model.nodes,
            relation,
            interaction,
            trust,
            weights,
            source,
            threshold,
            dynamic_mode=False,
            decay_factor=self.model.decay_factor,
            trust_gate_threshold=self.model.trust_gate_threshold,
        )
        self.dynamic_result = None
        self.diffusion_result = self.static_result
        self.statusBar().showMessage("已完成 BFS 擴散模擬", 3000)
        self._refresh_all()

    def _find_path(self, source: str, target: str, threshold: float) -> None:
        self.diffusion_result = None
        self.path_result = find_high_risk_path(
            self.model.nodes,
            self.model.score_matrix,
            source,
            target,
            threshold,
        )
        self.statusBar().showMessage(self.path_result.message, 3000)
        self._refresh_all()

    def _set_interaction_cell(self, source: str, target: str, value: float) -> None:
        self.model.set_interaction(source, target, value)
        self._clear_transient_results()
        self._refresh_all()

    def _set_trust_node_item(self, item: QTableWidgetItem) -> None:
        if self._updating_trust_table or item.column() != 1:
            return

        node_item = self.trust_node_table.item(item.row(), 0)
        if node_item is None:
            return

        try:
            value = float(item.text())
        except ValueError:
            value = 0.0
        value = max(0.0, min(1.0, value))

        self._updating_trust_table = True
        item.setText(f"{value:.2f}")
        self._updating_trust_table = False

        self.model.set_node_trust(node_item.text(), value)
        self._clear_transient_results()
        self._refresh_all()

    def _load_sample(self) -> None:
        self.model = DynamicTrustGraph.sample()
        self.model.set_trust_gate_threshold(self.control_panel.trust_gate_threshold())
        self._clear_transient_results()
        self.threshold = self.control_panel.threshold()
        self.statusBar().showMessage("已載入預設社群圖", 3000)
        self._refresh_all()
