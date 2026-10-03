from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

from dynamic_trust_graph.core.matrix_model import ScoreWeights, calculate_score_matrix
from dynamic_trust_graph.core.trust_update import decay_outgoing_trust


@dataclass
class DiffusionResult:
    source: str
    dynamic_mode: bool
    threshold: float
    layers: list[list[str]]
    infected: set[str]
    propagation_edges: list[tuple[str, str]]
    paths: dict[str, list[str]]
    final_trust: np.ndarray
    final_score: np.ndarray

    @property
    def infected_count(self) -> int:
        return len(self.infected)

    @property
    def depth(self) -> int:
        return max(0, len(self.layers) - 1)


def simulate_bfs_diffusion(
    nodes: list[str],
    relation: np.ndarray,
    interaction: np.ndarray,
    trust: np.ndarray,
    weights: ScoreWeights,
    source: str,
    threshold: float = 0.4,
    dynamic_mode: bool = False,
    decay_factor: float = 0.2,
    trust_gate_threshold: float = 0.2,
) -> DiffusionResult:
    if source not in nodes:
        raise ValueError(f"Unknown source node: {source}")

    current_trust = np.array(trust, dtype=float, copy=True)
    current_score = calculate_score_matrix(
        relation,
        interaction,
        current_trust,
        weights,
        trust_gate_threshold=trust_gate_threshold,
    )
    source_index = nodes.index(source)
    infected = {source}
    paths: dict[str, list[str]] = {source: [source]}
    queue: deque[tuple[str, int]] = deque([(source, 0)])
    layer_map: dict[int, list[str]] = defaultdict(list)
    layer_map[0].append(source)
    propagation_edges: list[tuple[str, str]] = []

    while queue:
        sender, depth = queue.popleft()
        row = nodes.index(sender)
        candidate_indices = [
            col
            for col, score in enumerate(current_score[row, :])
            if col != row and score >= threshold
        ]
        candidate_indices.sort(key=lambda col: (-current_score[row, col], nodes[col]))

        new_targets: list[str] = []
        for col in candidate_indices:
            target = nodes[col]
            if target in infected:
                continue
            infected.add(target)
            paths[target] = paths[sender] + [target]
            propagation_edges.append((sender, target))
            queue.append((target, depth + 1))
            layer_map[depth + 1].append(target)
            new_targets.append(target)

        if dynamic_mode and new_targets:
            current_trust = decay_outgoing_trust(
                nodes,
                current_trust,
                sender,
                decay_factor=decay_factor,
            )
            current_score = calculate_score_matrix(
                relation,
                interaction,
                current_trust,
                weights,
                trust_gate_threshold=trust_gate_threshold,
            )

    layers = [layer_map[index] for index in sorted(layer_map)]
    if not layers:
        layers = [[nodes[source_index]]]

    return DiffusionResult(
        source=source,
        dynamic_mode=dynamic_mode,
        threshold=threshold,
        layers=layers,
        infected=infected,
        propagation_edges=propagation_edges,
        paths=paths,
        final_trust=current_trust,
        final_score=current_score,
    )
