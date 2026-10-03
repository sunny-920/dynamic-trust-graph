from __future__ import annotations

import numpy as np

from dynamic_trust_graph.core.matrix_model import create_default_trust_matrix


SAMPLE_NODES = ["A", "B", "C", "D", "E"]

SAMPLE_EDGES = [
    ("A", "B"),
    ("A", "C"),
    ("B", "C"),
    ("C", "B"),
    ("B", "D"),
    ("C", "D"),
    ("C", "E"),
    ("D", "E"),
    ("E", "D"),
    ("E", "B"),
]


def load_sample_data() -> tuple[list[str], list[tuple[str, str]], np.ndarray, np.ndarray]:
    nodes = list(SAMPLE_NODES)
    edges = list(SAMPLE_EDGES)
    size = len(nodes)
    interaction = np.array(
        [
            [0.00, 0.82, 0.35, 0.14, 0.47],
            [0.22, 0.00, 0.76, 0.58, 0.31],
            [0.18, 0.69, 0.00, 0.84, 0.62],
            [0.09, 0.41, 0.27, 0.00, 0.88],
            [0.24, 0.73, 0.52, 0.66, 0.00],
        ],
        dtype=float,
    )
    trust = create_default_trust_matrix(size)
    index = {node: pos for pos, node in enumerate(nodes)}

    trust_values = {
        "A": 0.82,
        "B": 0.57,
        "C": 0.34,
        "D": 0.08,
        "E": 0.71,
    }
    for source, value in trust_values.items():
        row = index[source]
        trust[row, :] = value
        trust[row, row] = 0.0

    np.fill_diagonal(interaction, 0.0)
    np.fill_diagonal(trust, 0.0)
    return nodes, edges, interaction, trust
