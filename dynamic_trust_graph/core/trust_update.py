from __future__ import annotations

import numpy as np

from dynamic_trust_graph.core.matrix_model import clamp_probability


def decay_outgoing_trust(
    nodes: list[str],
    trust: np.ndarray,
    sender: str,
    decay_factor: float = 0.2,
) -> np.ndarray:
    updated = np.array(trust, dtype=float, copy=True)
    if sender not in nodes:
        return updated

    row = nodes.index(sender)
    updated[row, :] = np.clip(updated[row, :] - decay_factor, 0.0, 1.0)
    updated[row, row] = 0.0
    return updated


def reward_outgoing_trust(
    nodes: list[str],
    trust: np.ndarray,
    sender: str,
    reward_value: float = 0.05,
) -> np.ndarray:
    updated = np.array(trust, dtype=float, copy=True)
    if sender not in nodes:
        return updated

    row = nodes.index(sender)
    updated[row, :] = [clamp_probability(value + reward_value) for value in updated[row, :]]
    updated[row, row] = 0.0
    return updated
