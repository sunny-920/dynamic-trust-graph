from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np


@dataclass
class PathResult:
    success: bool
    source: str
    target: str
    path: list[str]
    edge_scores: list[float]
    total_cost: float
    message: str

    @property
    def edges(self) -> list[tuple[str, str]]:
        return list(zip(self.path, self.path[1:]))


def build_cost_graph(
    nodes: list[str],
    score: np.ndarray,
    threshold: float,
    epsilon: float = 0.0001,
) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    for row, source in enumerate(nodes):
        for col, target in enumerate(nodes):
            if source == target:
                continue
            edge_score = float(score[row, col])
            if edge_score >= threshold:
                graph.add_edge(
                    source,
                    target,
                    score=edge_score,
                    cost=1.0 / (edge_score + epsilon),
                )
    return graph


def find_high_risk_path(
    nodes: list[str],
    score: np.ndarray,
    source: str,
    target: str,
    threshold: float = 0.4,
    epsilon: float = 0.0001,
) -> PathResult:
    if source not in nodes or target not in nodes or source == target:
        return PathResult(
            success=False,
            source=source,
            target=target,
            path=[],
            edge_scores=[],
            total_cost=0.0,
            message="請選擇不同且存在的起點與終點。",
        )

    graph = build_cost_graph(nodes, score, threshold, epsilon)
    try:
        path = nx.dijkstra_path(graph, source, target, weight="cost")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return PathResult(
            success=False,
            source=source,
            target=target,
            path=[],
            edge_scores=[],
            total_cost=0.0,
            message="目前無高風險傳播路徑。",
        )

    edge_scores = [
        float(score[nodes.index(edge_source), nodes.index(edge_target)])
        for edge_source, edge_target in zip(path, path[1:])
    ]
    total_cost = sum(1.0 / (edge_score + epsilon) for edge_score in edge_scores)
    return PathResult(
        success=True,
        source=source,
        target=target,
        path=path,
        edge_scores=edge_scores,
        total_cost=float(total_cost),
        message=" -> ".join(path),
    )
