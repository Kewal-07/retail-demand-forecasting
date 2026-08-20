"""MinT hierarchical reconciliation. See CLAUDE.md Section 6D and Section 9.

Applied to point forecasts only -- reconciling a full predictive
distribution has no clean standard method and the conformal guarantee
does not survive it.

mint_reconcile() uses identity error covariance (the OLS-style special
case of MinT): no residual history is passed to this function to
estimate a richer covariance, so this is the simplification that's
actually computable from (point_forecasts, summing_matrix) alone.

point_forecasts' dict key order must match the row order
build_summing_matrix() used to build summing_matrix -- both preserve
insertion order: hierarchy_map's keys (upper levels, in the order given)
followed by the leaf series (bottom level, first-seen order).
"""
import numpy as np


def build_summing_matrix(hierarchy_map: dict) -> np.ndarray:
    all_nodes = list(hierarchy_map.keys())
    for children in hierarchy_map.values():
        for child in children:
            if child not in all_nodes:
                all_nodes.append(child)

    bottom_level = [n for n in all_nodes if n not in hierarchy_map]
    upper_level = [n for n in all_nodes if n in hierarchy_map]
    ordered_nodes = upper_level + bottom_level

    def _leaves_under(node: str) -> list[str]:
        if node not in hierarchy_map:
            return [node]
        leaves: list[str] = []
        for child in hierarchy_map[node]:
            leaves.extend(_leaves_under(child))
        return leaves

    S = np.zeros((len(ordered_nodes), len(bottom_level)))
    for i, node in enumerate(ordered_nodes):
        for leaf in _leaves_under(node):
            S[i, bottom_level.index(leaf)] = 1
    return S


def mint_reconcile(
    point_forecasts: dict[str, np.ndarray], summing_matrix: np.ndarray
) -> dict[str, np.ndarray]:
    keys = list(point_forecasts.keys())
    Y = np.stack([np.asarray(point_forecasts[k], dtype="float64") for k in keys], axis=0)

    S = summing_matrix
    projection = np.linalg.inv(S.T @ S) @ S.T  # (n_bottom, n_series)
    reconciled_bottom = projection @ Y
    reconciled_all = S @ reconciled_bottom  # coherent by construction

    return {k: reconciled_all[i] for i, k in enumerate(keys)}
