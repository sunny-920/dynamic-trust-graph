from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ScoreWeights:
    alpha: float = 0.3
    beta: float = 0.4
    gamma: float = 0.3

    def normalized(self) -> "ScoreWeights":
        values = np.array([self.alpha, self.beta, self.gamma], dtype=float)
        values = np.clip(values, 0.0, 1.0)
        total = float(values.sum())
        if total <= 0:
            return ScoreWeights()
        values = values / total
        return ScoreWeights(float(values[0]), float(values[1]), float(values[2]))


def normalize_weights(alpha: float, beta: float, gamma: float) -> ScoreWeights:
    return ScoreWeights(alpha, beta, gamma).normalized()


def build_relation_matrix(nodes: list[str], edges: Iterable[tuple[str, str]]) -> np.ndarray:
    index = {node: pos for pos, node in enumerate(nodes)}
    relation = np.zeros((len(nodes), len(nodes)), dtype=float)
    edge_set = {
        (source, target)
        for source, target in edges
        if source in index and target in index and source != target
    }
    for source, target in edge_set:
        if source in index and target in index and source != target:
            relation[index[source], index[target]] = (
                1.0 if (target, source) in edge_set else 0.7
            )
    return relation


def create_default_trust_matrix(size: int, default_value: float = 0.5) -> np.ndarray:
    trust = np.full((size, size), default_value, dtype=float)
    np.fill_diagonal(trust, 0.0)
    return trust


def calculate_score_matrix(
    relation: np.ndarray,
    interaction: np.ndarray,
    trust: np.ndarray,
    weights: ScoreWeights,
    trust_gate_threshold: float = 0.2,
) -> np.ndarray:
    weights = weights.normalized()
    score = (
        weights.alpha * relation.astype(float)
        + weights.beta * interaction.astype(float)
        + weights.gamma * trust.astype(float)
    )
    trust = trust.astype(float)
    size = trust.shape[0]
    for row in range(size):
        row_values = [float(trust[row, col]) for col in range(size) if col != row]
        node_trust = float(np.mean(row_values)) if row_values else 0.0
        if node_trust < trust_gate_threshold:
            score[row, :] = 0.0
    score = np.clip(score, 0.0, 1.0)
    np.fill_diagonal(score, 0.0)
    return score


def clamp_probability(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))
