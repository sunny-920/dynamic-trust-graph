from __future__ import annotations

import copy
from typing import Iterable

import networkx as nx
import numpy as np

from dynamic_trust_graph.core.matrix_model import (
    ScoreWeights,
    build_relation_matrix,
    calculate_score_matrix,
    clamp_probability,
    create_default_trust_matrix,
)
from dynamic_trust_graph.core.trust_update import decay_outgoing_trust, reward_outgoing_trust
from dynamic_trust_graph.utils.sample_data import load_sample_data


def _node_label(index: int) -> str:
    label = ""
    value = index + 1
    while value:
        value, remainder = divmod(value - 1, 26)
        label = chr(ord("A") + remainder) + label
    return label


def _balanced_groups(nodes: list[str], group_count: int) -> list[list[str]]:
    base_size, remainder = divmod(len(nodes), group_count)
    groups: list[list[str]] = []
    start = 0
    for group_index in range(group_count):
        size = base_size + (1 if group_index < remainder else 0)
        groups.append(nodes[start : start + size])
        start += size
    return groups


class DynamicTrustGraph:
    def __init__(
        self,
        nodes: Iterable[str] | None = None,
        edges: Iterable[tuple[str, str]] | None = None,
        interaction: np.ndarray | None = None,
        trust: np.ndarray | None = None,
        weights: ScoreWeights | None = None,
    ) -> None:
        self.graph = nx.DiGraph()
        self.nodes: list[str] = []
        self.weights = (weights or ScoreWeights()).normalized()
        self.decay_factor = 0.2
        self.reward_value = 0.05
        self.trust_gate_threshold = 0.2
        self.interaction = np.zeros((0, 0), dtype=float)
        self.trust = np.zeros((0, 0), dtype=float)

        for node in nodes or []:
            self.add_node(node)
        for source, target in edges or []:
            self.add_edge(source, target)

        if interaction is not None:
            self.set_interaction_matrix(interaction)
        if trust is not None:
            self.set_trust_matrix(trust)

    @classmethod
    def sample(cls) -> "DynamicTrustGraph":
        nodes, edges, interaction, trust = load_sample_data()
        return cls(nodes=nodes, edges=edges, interaction=interaction, trust=trust)

    @classmethod
    def random_scenario(
        cls,
        min_nodes: int,
        max_nodes: int,
        seed: int | None = None,
    ) -> "DynamicTrustGraph":
        if min_nodes < 2 or max_nodes < min_nodes:
            raise ValueError("Invalid random graph node range.")

        rng = np.random.default_rng(seed)
        node_count = int(rng.integers(min_nodes, max_nodes + 1))
        nodes = [_node_label(index) for index in range(node_count)]

        if node_count <= 10:
            isolated_count = 1
            group_count = 2
            edge_factor = rng.uniform(1.3, 1.8)
        elif node_count <= 30:
            isolated_count = max(2, node_count // 10)
            group_count = int(rng.integers(3, 6))
            edge_factor = rng.uniform(1.5, 2.1)
        else:
            isolated_count = max(3, node_count // 10)
            group_count = int(rng.integers(5, 9))
            edge_factor = rng.uniform(1.7, 2.3)

        active_nodes = nodes[:-isolated_count]
        isolated_nodes = set(nodes[-isolated_count:])
        group_count = min(group_count, max(2, len(active_nodes) // 2))
        groups = _balanced_groups(active_nodes, group_count)
        group_of = {
            node: group_index
            for group_index, group in enumerate(groups)
            for node in group
        }
        edges: set[tuple[str, str]] = set()
        protected_sink_nodes: set[str] = set()

        for group_index, group in enumerate(groups):
            pattern = group_index % 4
            if pattern == 0:
                # Closed directed cycle with a few hub shortcuts.
                for index, source in enumerate(group):
                    edges.add((source, group[(index + 1) % len(group)]))
                hub = group[0]
                for target in group[2:]:
                    if rng.random() < 0.55:
                        edges.add((hub, target))
            elif pattern == 1:
                # One-way chain with dead ends and no route back.
                for source, target in zip(group, group[1:]):
                    edges.add((source, target))
                if len(group) >= 4:
                    edges.add((group[0], group[-1]))
                protected_sink_nodes.add(group[-1])
            elif pattern == 2:
                # Inward hub: many nodes can reach the hub, but the hub cannot spread out.
                hub = group[0]
                for source in group[1:]:
                    edges.add((source, hub))
                if len(group) >= 3:
                    edges.add((group[-1], group[-2]))
                protected_sink_nodes.add(hub)
            else:
                # Small echo chamber with a dense local SCC.
                for index, source in enumerate(group):
                    edges.add((source, group[(index + 1) % len(group)]))
                    if len(group) > 2:
                        edges.add((source, group[(index - 1) % len(group)]))

        bridge_sources: list[str] = []
        if len(groups) >= 2 and node_count > 10:
            edges.add((groups[0][-1], groups[1][0]))
        if len(groups) >= 4:
            blocked_source = groups[2][-1]
            edges.add((blocked_source, groups[3][0]))
            bridge_sources.append(blocked_source)
        if len(groups) >= 6:
            edges.add((groups[4][-1], groups[5][0]))

        target_edges = max(len(edges), round(len(active_nodes) * edge_factor))
        attempts = 0
        while len(edges) < target_edges and attempts < target_edges * 30:
            attempts += 1
            group = groups[int(rng.integers(0, len(groups)))]
            if len(group) < 2:
                continue
            source, target = rng.choice(group, size=2, replace=False)
            if str(source) in protected_sink_nodes:
                continue
            edges.add((str(source), str(target)))

        trust_values = rng.uniform(0.30, 0.72, size=node_count)
        node_index = {node: index for index, node in enumerate(nodes)}
        active_indices = np.array([node_index[node] for node in active_nodes], dtype=int)
        shuffled_indices = rng.permutation(active_indices)
        low_count = max(1, len(active_nodes) // 10)
        high_count = max(1, len(active_nodes) // 7)
        trust_values[shuffled_indices[:low_count]] = rng.uniform(0.05, 0.18, size=low_count)
        trust_values[
            shuffled_indices[low_count : low_count + high_count]
        ] = rng.uniform(0.80, 0.94, size=high_count)

        barrier_nodes: set[str] = set(bridge_sources)
        for group_index, group in enumerate(groups):
            if group_index % 3 == 1 and group:
                barrier_nodes.add(group[len(group) // 2])
        for node in barrier_nodes:
            trust_values[node_index[node]] = rng.uniform(0.22, 0.30)

        interaction = rng.uniform(0.01, 0.12, size=(node_count, node_count))
        for group in groups:
            for source in group:
                for target in group:
                    if source != target:
                        interaction[node_index[source], node_index[target]] = rng.uniform(
                            0.08,
                            0.24,
                        )

        for source, target in edges:
            if source in barrier_nodes or rng.random() < 0.22:
                low, high = (0.06, 0.24)
            elif group_of[source] == group_of[target]:
                low, high = (0.55, 0.92)
            else:
                low, high = (0.32, 0.62)
            interaction[node_index[source], node_index[target]] = rng.uniform(low, high)

        for node in isolated_nodes:
            index = node_index[node]
            interaction[index, :] = rng.uniform(0.0, 0.06, size=node_count)
            interaction[:, index] = rng.uniform(0.0, 0.06, size=node_count)
        np.fill_diagonal(interaction, 0.0)

        trust = np.zeros((node_count, node_count), dtype=float)
        for row, value in enumerate(trust_values):
            trust[row, :] = value
            trust[row, row] = 0.0

        return cls(
            nodes=nodes,
            edges=sorted(edges),
            interaction=interaction,
            trust=trust,
        )

    def clone(self) -> "DynamicTrustGraph":
        cloned = DynamicTrustGraph(
            nodes=self.nodes,
            edges=list(self.graph.edges()),
            interaction=self.interaction.copy(),
            trust=self.trust.copy(),
            weights=copy.copy(self.weights),
        )
        cloned.decay_factor = self.decay_factor
        cloned.reward_value = self.reward_value
        cloned.trust_gate_threshold = self.trust_gate_threshold
        return cloned

    @property
    def relation_matrix(self) -> np.ndarray:
        return build_relation_matrix(self.nodes, self.graph.edges())

    @property
    def score_matrix(self) -> np.ndarray:
        return calculate_score_matrix(
            self.relation_matrix,
            self.interaction,
            self.trust,
            self.weights,
            trust_gate_threshold=self.trust_gate_threshold,
        )

    @property
    def node_trust_scores(self) -> dict[str, float]:
        if not self.nodes:
            return {}
        scores: dict[str, float] = {}
        for row, node in enumerate(self.nodes):
            values = [float(self.trust[row, col]) for col in range(len(self.nodes)) if col != row]
            scores[node] = float(np.mean(values)) if values else 0.0
        return scores

    @property
    def trust_graph_edges(self) -> list[tuple[str, str]]:
        node_scores = self.node_trust_scores
        edges: list[tuple[str, str]] = []
        for source in self.nodes:
            if node_scores.get(source, 0.0) < self.trust_gate_threshold:
                continue
            for target in self.nodes:
                if source != target:
                    edges.append((source, target))
        return edges

    def add_node(self, node: str) -> bool:
        node = node.strip()
        if not node or node in self.nodes:
            return False

        old_nodes = list(self.nodes)
        old_interaction = self.interaction.copy()
        old_trust = self.trust.copy()
        self.nodes.append(node)
        self.graph.add_node(node)
        self._resize_matrices(old_nodes, old_interaction, old_trust)
        return True

    def remove_node(self, node: str) -> bool:
        if node not in self.nodes:
            return False

        old_nodes = list(self.nodes)
        old_interaction = self.interaction.copy()
        old_trust = self.trust.copy()
        self.nodes.remove(node)
        self.graph.remove_node(node)
        self._resize_matrices(old_nodes, old_interaction, old_trust)
        return True

    def add_edge(self, source: str, target: str) -> bool:
        source = source.strip()
        target = target.strip()
        if not source or not target or source == target:
            return False
        if source not in self.nodes:
            self.add_node(source)
        if target not in self.nodes:
            self.add_node(target)
        if self.graph.has_edge(source, target):
            return False
        self.graph.add_edge(source, target)
        return True

    def remove_edge(self, source: str, target: str) -> bool:
        if self.graph.has_edge(source, target):
            self.graph.remove_edge(source, target)
            return True
        return False

    def set_interaction(self, source: str, target: str, value: float) -> None:
        if source not in self.nodes or target not in self.nodes:
            return
        row = self.nodes.index(source)
        col = self.nodes.index(target)
        self.interaction[row, col] = 0.0 if row == col else clamp_probability(value)

    def set_trust(self, source: str, target: str, value: float) -> None:
        if source not in self.nodes or target not in self.nodes:
            return
        row = self.nodes.index(source)
        col = self.nodes.index(target)
        self.trust[row, col] = 0.0 if row == col else clamp_probability(value)

    def set_node_trust(self, node: str, value: float) -> None:
        if node not in self.nodes:
            return
        row = self.nodes.index(node)
        self.trust[row, :] = clamp_probability(value)
        self.trust[row, row] = 0.0

    def set_interaction_matrix(self, interaction: np.ndarray) -> None:
        matrix = np.array(interaction, dtype=float, copy=True)
        if matrix.shape != (len(self.nodes), len(self.nodes)):
            raise ValueError("Interaction matrix shape does not match node count.")
        self.interaction = np.clip(matrix, 0.0, 1.0)
        np.fill_diagonal(self.interaction, 0.0)

    def set_trust_matrix(self, trust: np.ndarray) -> None:
        matrix = np.array(trust, dtype=float, copy=True)
        if matrix.shape != (len(self.nodes), len(self.nodes)):
            raise ValueError("Trust matrix shape does not match node count.")
        self.trust = np.clip(matrix, 0.0, 1.0)
        np.fill_diagonal(self.trust, 0.0)

    def randomize_interaction(self, seed: int | None = None) -> None:
        rng = np.random.default_rng(seed)
        self.interaction = rng.uniform(0.0, 1.0, size=(len(self.nodes), len(self.nodes)))
        np.fill_diagonal(self.interaction, 0.0)

    def build_echo_chamber_interaction(self, seed: int | None = None) -> None:
        rng = np.random.default_rng(seed)
        size = len(self.nodes)
        interaction = rng.uniform(0.0, 0.3, size=(size, size))
        np.fill_diagonal(interaction, 0.0)

        preferred_groups = [["D", "E", "F"], ["C", "G", "H"]]
        groups = [group for group in preferred_groups if all(node in self.nodes for node in group)]
        if not groups and size >= 3:
            midpoint = max(3, size // 2)
            groups = [self.nodes[:midpoint]]

        for group in groups:
            for source in group:
                for target in group:
                    if source == target:
                        continue
                    interaction[self.nodes.index(source), self.nodes.index(target)] = rng.uniform(
                        0.7,
                        1.0,
                    )

        self.interaction = interaction

    def clear_interaction(self) -> None:
        self.interaction = np.zeros((len(self.nodes), len(self.nodes)), dtype=float)

    def randomize_node_trust(self, seed: int | None = None) -> None:
        rng = np.random.default_rng(seed)
        size = len(self.nodes)
        values = rng.uniform(0.0, 1.0, size=size)
        trust = np.zeros((size, size), dtype=float)
        for row, value in enumerate(values):
            trust[row, :] = value
            trust[row, row] = 0.0
        self.trust = trust

    def reset_trust(self) -> None:
        self.trust = create_default_trust_matrix(len(self.nodes))

    def report_misinformation(self, node: str) -> None:
        self.trust = decay_outgoing_trust(
            self.nodes,
            self.trust,
            node,
            decay_factor=self.decay_factor,
        )

    def reward_good_interaction(self, node: str) -> None:
        self.trust = reward_outgoing_trust(
            self.nodes,
            self.trust,
            node,
            reward_value=self.reward_value,
        )

    def set_weights(self, alpha: float, beta: float, gamma: float) -> None:
        self.weights = ScoreWeights(alpha, beta, gamma).normalized()

    def set_trust_gate_threshold(self, threshold: float) -> None:
        self.trust_gate_threshold = clamp_probability(threshold)

    def _resize_matrices(
        self,
        old_nodes: list[str],
        old_interaction: np.ndarray,
        old_trust: np.ndarray,
    ) -> None:
        size = len(self.nodes)
        interaction = np.zeros((size, size), dtype=float)
        trust = create_default_trust_matrix(size)
        new_index = {node: pos for pos, node in enumerate(self.nodes)}

        for old_row, source in enumerate(old_nodes):
            if source not in new_index:
                continue
            for old_col, target in enumerate(old_nodes):
                if target not in new_index:
                    continue
                row = new_index[source]
                col = new_index[target]
                interaction[row, col] = old_interaction[old_row, old_col]
                trust[row, col] = old_trust[old_row, old_col]

        np.fill_diagonal(interaction, 0.0)
        np.fill_diagonal(trust, 0.0)
        self.interaction = interaction
        self.trust = trust
