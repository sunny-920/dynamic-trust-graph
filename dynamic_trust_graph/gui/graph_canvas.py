from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import QVBoxLayout, QWidget

plt.rcParams["font.sans-serif"] = [
    "Microsoft JhengHei",
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

LOW_TRUST_NODE_COLOR = "#4b5563"


class GraphCanvas(QWidget):
    def __init__(self, title: str = "", parent=None) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=(6, 4), dpi=110)
        self.canvas = FigureCanvas(self.figure)
        self.title = title
        self._axis = None
        self._pan_state = None
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_button_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.canvas.mpl_connect("button_release_event", self._on_button_release)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

    def draw_graph(
        self,
        nodes: list[str],
        edges: Iterable[tuple[str, str]],
        weights: dict[tuple[str, str], float] | None = None,
        title: str | None = None,
        node_scores: dict[str, float] | None = None,
        highlight_nodes: set[str] | None = None,
        highlight_edges: set[tuple[str, str]] | None = None,
        muted_nodes: set[str] | None = None,
        scc_lookup: dict[str, int] | None = None,
        node_label_values: dict[str, float] | None = None,
        node_value_threshold: float | None = None,
        edge_mode: str = "relation",
    ) -> None:
        self._pan_state = None
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        self._axis = axis
        axis.set_title(title or self.title, fontsize=12, pad=10)
        axis.axis("off")

        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        edge_list = list(edges)
        graph.add_edges_from(edge_list)

        if not nodes:
            axis.text(0.5, 0.5, "尚無節點", ha="center", va="center", fontsize=12)
            self.canvas.draw_idle()
            return

        position_graph = graph.copy()
        if not edge_list and len(nodes) > 1:
            for source, target in zip(nodes, nodes[1:]):
                position_graph.add_edge(source, target)
        layout_iterations = 25 if len(nodes) > 30 else 35 if len(nodes) > 15 else 50
        positions = nx.spring_layout(
            position_graph,
            seed=19,
            k=1.1,
            iterations=layout_iterations,
        )

        highlight_nodes = highlight_nodes or set()
        highlight_edges = highlight_edges or set()
        muted_nodes = muted_nodes or set()
        weights = weights or {}
        scc_lookup = scc_lookup or {}
        node_label_values = node_label_values or {}

        node_colors = []
        muted_label_nodes = set(muted_nodes)
        palette = plt.get_cmap("tab10")
        for node in nodes:
            if node in muted_nodes:
                node_colors.append(LOW_TRUST_NODE_COLOR)
            elif node in highlight_nodes:
                node_colors.append("#ef4444")
            elif edge_mode == "trust_node" and node in node_label_values:
                node_colors.append(
                    self._trust_node_color(node_label_values[node], node_value_threshold)
                )
                if (
                    node_value_threshold is not None
                    and node_label_values[node] < node_value_threshold
                ):
                    muted_label_nodes.add(node)
            elif node in scc_lookup:
                node_colors.append(palette((scc_lookup[node] - 1) % 10))
            else:
                node_colors.append("#d1d5db")

        if node_scores:
            max_score = max(node_scores.values()) if node_scores else 0.0
            node_sizes = [
                600 + (2200 * node_scores.get(node, 0.0) / max_score if max_score else 0)
                for node in nodes
            ]
        else:
            node_sizes = [900 for _node in nodes]

        nx.draw_networkx_nodes(
            graph,
            positions,
            node_color=node_colors,
            node_size=node_sizes,
            edgecolors="#111827",
            linewidths=1.0,
            ax=axis,
        )
        labels = {
            node: f"{node}\n{node_label_values[node]:.2f}" if node in node_label_values else node
            for node in nodes
        }
        normal_labels = {
            node: label for node, label in labels.items() if node not in muted_label_nodes
        }
        muted_labels = {
            node: label for node, label in labels.items() if node in muted_label_nodes
        }
        if normal_labels:
            nx.draw_networkx_labels(
                graph,
                positions,
                labels=normal_labels,
                font_size=10,
                font_weight="bold",
                font_color="#111827",
                ax=axis,
            )
        if muted_labels:
            nx.draw_networkx_labels(
                graph,
                positions,
                labels=muted_labels,
                font_size=10,
                font_weight="bold",
                font_color="#f8fafc",
                ax=axis,
            )

        edge_colors = []
        edge_widths = []
        for edge in edge_list:
            value = weights.get(edge, 1.0)
            if edge in highlight_edges:
                edge_colors.append("#dc2626")
                edge_widths.append(4.0)
            elif edge_mode == "trust":
                edge_colors.append(self._trust_color(value))
                edge_widths.append(1.2 + 3.0 * value)
            elif edge_mode == "trust_node":
                edge_colors.append("#94a3b8")
                edge_widths.append(0.8)
            elif edge_mode == "score":
                edge_colors.append(self._score_color(value))
                edge_widths.append(1.0 + 3.2 * value)
            else:
                edge_colors.append("#64748b")
                edge_widths.append(1.8)

        if edge_list:
            nx.draw_networkx_edges(
                graph,
                positions,
                edge_color=edge_colors,
                width=edge_widths,
                arrows=True,
                arrowsize=18,
                arrowstyle="-|>",
                connectionstyle="arc3,rad=0.08",
                min_source_margin=16,
                min_target_margin=18,
                ax=axis,
            )

        label_limit = 40 if edge_mode == "score" else 18
        should_show_weight_labels = bool(weights) and len(edge_list) <= label_limit
        if edge_list and should_show_weight_labels:
            labels = {edge: f"{weights.get(edge, 0.0):.2f}" for edge in edge_list}
            nx.draw_networkx_edge_labels(
                graph,
                positions,
                edge_labels=labels,
                font_size=7 if edge_mode == "score" and len(edge_list) > 18 else 8,
                rotate=False,
                bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.75},
                ax=axis,
            )

        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _on_scroll(self, event) -> None:
        if self._axis is None or event.inaxes is not self._axis:
            return
        if event.xdata is None or event.ydata is None:
            return

        scale = 0.80 if event.button == "up" else 1.25
        x_min, x_max = self._axis.get_xlim()
        y_min, y_max = self._axis.get_ylim()
        new_width = (x_max - x_min) * scale
        new_height = (y_max - y_min) * scale

        x_ratio = (event.xdata - x_min) / (x_max - x_min)
        y_ratio = (event.ydata - y_min) / (y_max - y_min)
        self._axis.set_xlim(
            event.xdata - new_width * x_ratio,
            event.xdata + new_width * (1.0 - x_ratio),
        )
        self._axis.set_ylim(
            event.ydata - new_height * y_ratio,
            event.ydata + new_height * (1.0 - y_ratio),
        )
        self.canvas.draw_idle()

    def _on_button_press(self, event) -> None:
        if event.button != 2 or self._axis is None or event.inaxes is not self._axis:
            return
        if event.xdata is None or event.ydata is None:
            return
        self._pan_state = (
            event.xdata,
            event.ydata,
            self._axis.get_xlim(),
            self._axis.get_ylim(),
        )

    def _on_mouse_move(self, event) -> None:
        if self._pan_state is None or self._axis is None or event.inaxes is not self._axis:
            return
        if event.xdata is None or event.ydata is None:
            return

        start_x, start_y, start_xlim, start_ylim = self._pan_state
        delta_x = event.xdata - start_x
        delta_y = event.ydata - start_y
        self._axis.set_xlim(start_xlim[0] - delta_x, start_xlim[1] - delta_x)
        self._axis.set_ylim(start_ylim[0] - delta_y, start_ylim[1] - delta_y)
        self.canvas.draw_idle()

    def _on_button_release(self, event) -> None:
        if event.button == 2:
            self._pan_state = None

    @staticmethod
    def _trust_color(value: float) -> str:
        if value >= 0.6:
            return "#b91c1c"
        if value >= 0.4:
            return "#f59e0b"
        return "#94a3b8"

    @staticmethod
    def _trust_node_color(value: float, threshold: float | None = None) -> str:
        if threshold is not None and value < threshold:
            return LOW_TRUST_NODE_COLOR
        if value >= 0.8:
            return "#16a34a"
        if value >= 0.5:
            return "#facc15"
        if value >= 0.2:
            return "#fb923c"
        return "#94a3b8"

    @staticmethod
    def _score_color(value: float) -> str:
        if value >= 0.75:
            return "#be123c"
        if value >= 0.5:
            return "#f97316"
        return "#64748b"
