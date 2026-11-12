"""
src/brain_network.py - Functional Brain Network & ENCS Extraction
Implements proportional thresholding (50%), graph binarization,
small-world graph metric evaluations (clustering coefficient, path length),
and Epileptic Network Connectivity Strength (ENCS) computation.
"""

from typing import Dict, Tuple, List, Optional
import numpy as np
from config import PROPORTIONAL_THRESHOLD


def binarize_network_proportional(
    wpli_matrix: np.ndarray,
    threshold: float = PROPORTIONAL_THRESHOLD
) -> np.ndarray:
    """
    Binarizes the functional connectivity matrix using proportional thresholding.
    Preserves top (threshold * 100)% of upper-triangular edge weights as 1, remainder 0.
    
    Args:
        wpli_matrix: (N, N) symmetric connectivity matrix
        threshold: Proportional threshold (e.g. 0.50 for top 50%)
        
    Returns:
        binary_adj: (N, N) symmetric binary adjacency matrix
    """
    N = wpli_matrix.shape[0]
    if N <= 1:
        return np.zeros((N, N), dtype=int)

    # Get upper triangular indices (excluding main diagonal)
    triu_indices = np.triu_indices(N, k=1)
    weights = wpli_matrix[triu_indices]
    n_edges_total = len(weights)
    
    if n_edges_total == 0:
        return np.zeros((N, N), dtype=int)

    # Number of edges to retain
    n_retain = int(np.round(threshold * n_edges_total))
    n_retain = max(1, min(n_retain, n_edges_total))

    # Find cutoff value
    sorted_weights = np.sort(weights)[::-1]
    cutoff = sorted_weights[n_retain - 1]

    # Construct symmetric binary adjacency matrix
    binary_adj = np.zeros((N, N), dtype=int)
    mask = (wpli_matrix >= cutoff) & (wpli_matrix > 0)
    np.fill_diagonal(mask, False)

    binary_adj[mask] = 1
    return binary_adj


def compute_encs(binary_adj: np.ndarray) -> np.ndarray:
    """
    Computes the Epileptic Network Connectivity Strength (ENCS) for each node.
    ENCS_i = sum_j (A_ij)  (Node degree in the binarized network)
    
    Args:
        binary_adj: (N, N) binary adjacency matrix
        
    Returns:
        encs: (N,) array containing connection count for each channel
    """
    return np.sum(binary_adj, axis=1).astype(float)


def compute_clustering_coefficient(binary_adj: np.ndarray) -> float:
    """
    Computes average network clustering coefficient C.
    """
    N = binary_adj.shape[0]
    if N < 3:
        return 0.0

    degrees = np.sum(binary_adj, axis=1)
    # A^3 diagonal gives 2 * number of triangles per node
    adj_cubed = np.linalg.matrix_power(binary_adj, 3)
    triangles = np.diag(adj_cubed) / 2.0

    possible_triangles = degrees * (degrees - 1) / 2.0
    valid = possible_triangles > 0

    local_cc = np.zeros(N)
    local_cc[valid] = triangles[valid] / possible_triangles[valid]

    return float(np.mean(local_cc))


def compute_characteristic_path_length(binary_adj: np.ndarray) -> float:
    """
    Computes average shortest path length L using BFS / Floyd-Warshall.
    """
    N = binary_adj.shape[0]
    if N <= 1:
        return 0.0

    # Shortest paths via BFS for unweighted graphs
    dist_matrix = np.full((N, N), np.inf)
    np.fill_diagonal(dist_matrix, 0)

    for i in range(N):
        queue = [i]
        visited = {i: 0}
        while queue:
            curr = queue.pop(0)
            curr_dist = visited[curr]
            neighbors = np.where(binary_adj[curr] == 1)[0]
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited[neighbor] = curr_dist + 1
                    dist_matrix[i, neighbor] = curr_dist + 1
                    queue.append(neighbor)

    # Average over all reachable pairs (excluding diagonal and infinite distances)
    finite_mask = np.isfinite(dist_matrix) & ~np.eye(N, dtype=bool)
    if not np.any(finite_mask):
        return 0.0
    return float(np.mean(dist_matrix[finite_mask]))


def analyze_threshold_curve(
    wpli_matrix: np.ndarray,
    thresholds: Optional[List[float]] = None
) -> Dict[str, List[float]]:
    """
    Evaluates clustering coefficient and path length across thresholds (0.1 to 1.0)
    matching Figure 2 in the paper.
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.1, 1.01, 0.1)]

    cc_list = []
    cpl_list = []

    for t in thresholds:
        bin_adj = binarize_network_proportional(wpli_matrix, threshold=t)
        cc = compute_clustering_coefficient(bin_adj)
        cpl = compute_characteristic_path_length(bin_adj)
        cc_list.append(cc)
        cpl_list.append(cpl)

    return {
        "thresholds": thresholds,
        "clustering_coefficients": cc_list,
        "characteristic_path_lengths": cpl_list
    }
