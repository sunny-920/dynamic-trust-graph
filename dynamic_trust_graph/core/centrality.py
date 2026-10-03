from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np


@dataclass
class SCCSummary:
    component_id: int
    nodes: list[str]
    avg_interaction: float
    avg_trust: float
    potential_echo_chamber: bool
    high_risk_echo_chamber: bool


def build_score_graph(nodes: list[str], score: np.ndarray, threshold: float = 0.0) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    for row, source in enumerate(nodes):
        for col, target in enumerate(nodes):
            if source == target:
                continue
            edge_score = float(score[row, col])
            if edge_score > threshold:
                graph.add_edge(source, target, score=edge_score)
    return graph


def pagerank_scores(nodes: list[str], score: np.ndarray) -> dict[str, float]:
    if not nodes:
        return {}
    graph = build_score_graph(nodes, score, threshold=0.0)
    try:
        ranks = nx.pagerank(graph, weight="score")
    except ModuleNotFoundError as exc:
        if exc.name != "scipy":
            raise
        ranks = _pagerank_numpy_fallback(nodes, score)
    except nx.PowerIterationFailedConvergence:
        ranks = _pagerank_numpy_fallback(nodes, score)
    return _zero_non_transmitters(nodes, score, ranks)


def _pagerank_numpy_fallback(
    nodes: list[str],
    score: np.ndarray,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1.0e-8,
) -> dict[str, float]:
    size = len(nodes)
    if size == 0:
        return {}

    weights = np.array(score, dtype=float, copy=True)
    weights = np.clip(weights, 0.0, None)
    np.fill_diagonal(weights, 0.0)

    transition = np.zeros((size, size), dtype=float)
    row_sums = weights.sum(axis=1)
    for row in range(size):
        if row_sums[row] > 0:
            transition[row, :] = weights[row, :] / row_sums[row]
        else:
            transition[row, :] = 1.0 / size

    rank = np.full(size, 1.0 / size, dtype=float)
    teleport = np.full(size, (1.0 - damping) / size, dtype=float)
    for _iteration in range(max_iter):
        next_rank = teleport + damping * transition.T.dot(rank)
        if np.linalg.norm(next_rank - rank, ord=1) < size * tol:
            rank = next_rank
            break
        rank = next_rank

    rank_sum = float(rank.sum())
    if rank_sum > 0:
        rank = rank / rank_sum
    return {node: float(rank[index]) for index, node in enumerate(nodes)}


def _zero_non_transmitters(
    nodes: list[str],
    score: np.ndarray,
    ranks: dict[str, float],
) -> dict[str, float]:
    adjusted = dict(ranks)
    for row, node in enumerate(nodes):
        if float(np.sum(score[row, :])) <= 0.0:
            adjusted[node] = 0.0

    total = float(sum(adjusted.values()))
    if total > 0:
        return {node: float(value / total) for node, value in adjusted.items()}
    return {node: 0.0 for node in nodes}


def top_pagerank(pagerank: dict[str, float], limit: int = 3) -> list[tuple[str, float]]:
    positive_items = [(node, value) for node, value in pagerank.items() if value > 0]
    return sorted(positive_items, key=lambda item: (-item[1], item[0]))[:limit]


def degree_metrics(nodes: list[str], relation: np.ndarray) -> dict[str, dict[str, int]]:
    metrics: dict[str, dict[str, int]] = {}
    for index, node in enumerate(nodes):
        metrics[node] = {
            "out_degree": int(np.count_nonzero(relation[index, :] > 0)),
            "in_degree": int(np.count_nonzero(relation[:, index] > 0)),
        }
    return metrics


def strongly_connected_component_analysis(
    nodes: list[str],
    relation: np.ndarray,
    interaction: np.ndarray,
    trust: np.ndarray,
) -> list[SCCSummary]:
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    for row, source in enumerate(nodes):
        for col, target in enumerate(nodes):
            if relation[row, col] > 0:
                graph.add_edge(source, target)

    components = list(nx.strongly_connected_components(graph))
    order = {node: index for index, node in enumerate(nodes)}
    components.sort(key=lambda group: (-len(group), min(order[node] for node in group)))

    summaries: list[SCCSummary] = []
    for component_id, component in enumerate(components, start=1):
        ordered_nodes = sorted(component, key=lambda node: order[node])
        indices = [order[node] for node in ordered_nodes]
        pair_values_interaction: list[float] = []
        pair_values_trust: list[float] = []

        for row in indices:
            for col in indices:
                if row == col:
                    continue
                pair_values_interaction.append(float(interaction[row, col]))
                pair_values_trust.append(float(trust[row, col]))

        avg_interaction = (
            float(np.mean(pair_values_interaction)) if pair_values_interaction else 0.0
        )
        avg_trust = float(np.mean(pair_values_trust)) if pair_values_trust else 0.0
        potential = len(ordered_nodes) >= 3
        high_risk = potential and avg_interaction >= 0.7 and avg_trust >= 0.6
        summaries.append(
            SCCSummary(
                component_id=component_id,
                nodes=ordered_nodes,
                avg_interaction=avg_interaction,
                avg_trust=avg_trust,
                potential_echo_chamber=potential,
                high_risk_echo_chamber=high_risk,
            )
        )
    return summaries


def scc_lookup(sccs: list[SCCSummary]) -> dict[str, int]:
    lookup: dict[str, int] = {}
    for summary in sccs:
        for node in summary.nodes:
            lookup[node] = summary.component_id
    return lookup


def push_rankings(
    nodes: list[str],
    score: np.ndarray,
    threshold: float = 0.4,
) -> dict[str, list[tuple[str, float]]]:
    rankings: dict[str, list[tuple[str, float]]] = {}
    for col, target in enumerate(nodes):
        sources: list[tuple[str, float]] = []
        for row, source in enumerate(nodes):
            if source == target:
                continue
            edge_score = float(score[row, col])
            if edge_score >= threshold:
                sources.append((source, edge_score))
        rankings[target] = sorted(sources, key=lambda item: (-item[1], item[0]))
    return rankings


def high_risk_nodes(
    nodes: list[str],
    effective_score: np.ndarray,
    pagerank: dict[str, float],
    limit: int = 5,
) -> list[tuple[str, float]]:
    if not nodes:
        return []

    degree = degree_metrics(nodes, effective_score)
    max_page = max(pagerank.values()) if pagerank else 1.0
    max_in = max((item["in_degree"] for item in degree.values()), default=1) or 1
    max_out = max((item["out_degree"] for item in degree.values()), default=1) or 1
    scores: list[tuple[str, float]] = []

    for node in nodes:
        page_component = pagerank.get(node, 0.0) / max_page if max_page else 0.0
        out_component = degree[node]["out_degree"] / max_out
        in_component = degree[node]["in_degree"] / max_in
        risk = 0.55 * page_component + 0.3 * out_component + 0.15 * in_component
        if risk > 0:
            scores.append((node, float(risk)))

    return sorted(scores, key=lambda item: (-item[1], item[0]))[:limit]
