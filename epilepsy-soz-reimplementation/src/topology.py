"""
src/topology.py - Topological Data Analysis (TDA) Pipeline
Implements Takens Delay Embedding, Vietoris-Rips Simplex Complex construction (H0, H1, H2),
Persistence Diagram extraction, and Persistence Entropy (PE) calculation.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from ripser import ripser
from config import (
    TAKENS_DELAY,
    TAKENS_DIMENSION,
    HOMOLOGY_DIMENSIONS,
    MAX_TDA_POINTS
)


def takens_embedding(
    signal_1d: np.ndarray,
    delay: int = TAKENS_DELAY,
    dimension: int = TAKENS_DIMENSION
) -> np.ndarray:
    """
    Constructs high-dimensional phase space trajectory via Takens' delay embedding theorem.
    v(t) = [x(t), x(t + tau), x(t + 2*tau), ..., x(t + (m-1)*tau)]
    
    Args:
        signal_1d: 1D time series of shape (N,)
        delay: Time delay parameter tau (default 28)
        dimension: Embedding dimension m (default 4)
        
    Returns:
        point_cloud: Array of shape (K, dimension), where K = N - (m - 1) * delay
    """
    n_samples = len(signal_1d)
    max_lag = (dimension - 1) * delay
    if n_samples <= max_lag:
        raise ValueError(f"Signal length ({n_samples}) is too short for delay {delay} and dimension {dimension}")

    k_points = n_samples - max_lag
    point_cloud = np.zeros((k_points, dimension), dtype=np.float64)

    for i in range(dimension):
        point_cloud[:, i] = signal_1d[i * delay : i * delay + k_points]

    return point_cloud


def subsample_point_cloud(
    points: np.ndarray,
    max_points: int = MAX_TDA_POINTS
) -> np.ndarray:
    """
    Subsamples phase space point cloud uniformly in time to ensure fast Vietoris-Rips filtration
    while preserving dynamical manifold geometry.
    """
    n_pts = points.shape[0]
    if n_pts <= max_points:
        return points
    indices = np.linspace(0, n_pts - 1, max_points, dtype=int)
    return points[indices]


def compute_persistence_diagrams(
    point_cloud: np.ndarray,
    max_dim: int = 2
) -> List[np.ndarray]:
    """
    Computes Vietoris-Rips persistence diagrams for homology dimensions 0, 1, ..., max_dim.
    
    Args:
        point_cloud: Array of shape (K, m)
        max_dim: Maximum homology dimension (2 for H0, H1, H2)
        
    Returns:
        diagrams: List of arrays [D_0, D_1, D_2] where each D_k has shape (num_features, 2) [birth, death]
    """
    # Ripser returns a dict with 'dgms' containing [H0, H1, H2]
    result = ripser(point_cloud, maxdim=max_dim)
    return result["dgms"]


def compute_persistence_entropy_for_diagram(diagram: np.ndarray) -> float:
    """
    Computes Shannon Persistence Entropy for a single persistence diagram D.
    
    PE = - sum (P_i * log2(P_i)), where P_i = p_i / sum(p_j), p_i = death_i - birth_i
    
    Args:
        diagram: (num_features, 2) array of [birth, death] pairs
        
    Returns:
        entropy: Scalar persistence entropy value
    """
    if len(diagram) == 0:
        return 0.0

    # Filter out infinite death points (essential classes)
    finite_mask = np.isfinite(diagram[:, 1]) & (diagram[:, 1] > diagram[:, 0])
    finite_bars = diagram[finite_mask]

    if len(finite_bars) == 0:
        return 0.0

    lifespans = finite_bars[:, 1] - finite_bars[:, 0]
    total_life = np.sum(lifespans)

    if total_life <= 1e-12:
        return 0.0

    probabilities = lifespans / total_life
    # Ignore zero probabilities
    valid_p = probabilities[probabilities > 0]
    entropy = -np.sum(valid_p * np.log2(valid_p))
    return float(entropy)


def compute_persistence_entropy(
    signal_1d: np.ndarray,
    delay: int = TAKENS_DELAY,
    dimension: int = TAKENS_DIMENSION,
    max_dim: int = 2,
    max_points: int = MAX_TDA_POINTS
) -> Dict[str, float]:
    """
    Full pipeline to compute Persistence Entropy for a 1D channel time series.
    
    Returns:
        dict with:
            - 'pe_h0': Persistence entropy for H0 (connected components)
            - 'pe_h1': Persistence entropy for H1 (1D loops)
            - 'pe_h2': Persistence entropy for H2 (2D voids)
            - 'pe_total': Sum / aggregate persistence entropy
            - 'pe_mean': Mean persistence entropy across dimensions
    """
    # 1. Takens phase space embedding
    pt_cloud = takens_embedding(signal_1d, delay=delay, dimension=dimension)
    
    # 2. Subsample for speed
    pt_cloud_sub = subsample_point_cloud(pt_cloud, max_points=max_points)
    
    # 3. Persistent Homology
    diagrams = compute_persistence_diagrams(pt_cloud_sub, max_dim=max_dim)
    
    # 4. Compute Persistence Entropy per dimension
    pe_h0 = compute_persistence_entropy_for_diagram(diagrams[0]) if len(diagrams) > 0 else 0.0
    pe_h1 = compute_persistence_entropy_for_diagram(diagrams[1]) if len(diagrams) > 1 else 0.0
    pe_h2 = compute_persistence_entropy_for_diagram(diagrams[2]) if len(diagrams) > 2 else 0.0

    pe_total = pe_h0 + pe_h1 + pe_h2
    pe_mean = pe_total / max(1, len(diagrams))

    return {
        "pe_h0": pe_h0,
        "pe_h1": pe_h1,
        "pe_h2": pe_h2,
        "pe_total": pe_total,
        "pe_mean": pe_mean
    }


def compute_ami(signal_1d: np.ndarray, max_lag: int = 50, n_bins: int = 32) -> Tuple[np.ndarray, int]:
    """
    Average Mutual Information (AMI) parameter verification for optimal delay selection.
    
    Returns:
        ami_values: Array of AMI for lags 1 to max_lag
        optimal_tau: First local minimum or peak lag
    """
    ami_vals = []
    for lag in range(1, max_lag + 1):
        x = signal_1d[:-lag]
        y = signal_1d[lag:]
        
        # 2D joint histogram
        hist_2d, _, _ = np.histogram2d(x, y, bins=n_bins)
        p_xy = hist_2d / np.sum(hist_2d)
        
        p_x = np.sum(p_xy, axis=1)
        p_y = np.sum(p_xy, axis=0)
        
        # Outer product p(x)*p(y)
        p_x_p_y = np.outer(p_x, p_y)
        
        # Non-zero mask
        mask = (p_xy > 0) & (p_x_p_y > 0)
        ami = np.sum(p_xy[mask] * np.log2(p_xy[mask] / p_x_p_y[mask]))
        ami_vals.append(ami)

    ami_arr = np.array(ami_vals)
    # Optimal lag: first local minimum
    diffs = np.diff(ami_arr)
    local_mins = np.where((diffs[:-1] < 0) & (diffs[1:] > 0))[0] + 1
    optimal_tau = local_mins[0] + 1 if len(local_mins) > 0 else int(np.argmin(ami_arr) + 1)
    
    return ami_arr, optimal_tau


def compute_fnn(
    signal_1d: np.ndarray,
    delay: int = TAKENS_DELAY,
    max_dim: int = 8,
    rtol: float = 15.0,
    atol: float = 2.0
) -> Tuple[np.ndarray, int]:
    """
    False Nearest Neighbors (FNN) parameter verification for optimal embedding dimension.
    """
    from sklearn.neighbors import NearestNeighbors
    
    fnn_ratios = []
    std_data = np.std(signal_1d)

    for m in range(1, max_dim + 1):
        try:
            emb_m = takens_embedding(signal_1d, delay=delay, dimension=m)
            emb_next = takens_embedding(signal_1d, delay=delay, dimension=m + 1)
        except ValueError:
            break

        n_pts = min(len(emb_m), len(emb_next))
        emb_m = emb_m[:n_pts]
        emb_next = emb_next[:n_pts]

        nn = NearestNeighbors(n_neighbors=2, algorithm="auto").fit(emb_m)
        distances, indices = nn.kneighbors(emb_m)

        dist_m = distances[:, 1]
        nn_idx = indices[:, 1]

        # Distance in m+1 dimension
        dist_next = np.abs(emb_next[:, -1] - emb_next[nn_idx, -1])
        total_dist_next = np.sqrt(dist_m**2 + dist_next**2)

        # FNN criteria
        criterion1 = (dist_next / np.maximum(dist_m, 1e-12)) > rtol
        criterion2 = (total_dist_next / max(std_data, 1e-12)) > atol
        is_false_nn = criterion1 | criterion2

        fnn_ratio = np.mean(is_false_nn)
        fnn_ratios.append(fnn_ratio)

    fnn_arr = np.array(fnn_ratios)
    optimal_dim = int(np.where(fnn_arr < 0.05)[0][0] + 1) if np.any(fnn_arr < 0.05) else int(np.argmin(fnn_arr) + 1)

    return fnn_arr, optimal_dim
