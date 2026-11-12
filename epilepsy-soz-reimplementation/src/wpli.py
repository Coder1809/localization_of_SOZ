"""
src/wpli.py - Weighted Phase Lag Index (WPLI) Functional Brain Network
Calculates pairwise phase synchronization using the Weighted Phase Lag Index (Vinck et al., 2011; Fang et al., 2024).
"""

from typing import Optional
import numpy as np
import scipy.signal as signal


def compute_analytic_signal(data: np.ndarray) -> np.ndarray:
    """
    Computes analytic signal using the Hilbert transform along the time axis (axis=1).
    
    Args:
        data: (n_channels, n_samples)
        
    Returns:
        complex array of shape (n_channels, n_samples)
    """
    return signal.hilbert(data, axis=-1)


def compute_pairwise_wpli(
    window_data: np.ndarray,
    analytic_signals: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Computes the Weighted Phase Lag Index (WPLI) connectivity matrix for a multi-channel epoch.
    
    WPLI = |E[Im(X * Y*)]| / E[|Im(X * Y*)|]
    
    Args:
        window_data: (n_channels, n_samples) real-valued filtered signal
        analytic_signals: optional precomputed Hilbert transform of window_data
        
    Returns:
        wpli_matrix: (n_channels, n_channels) symmetric matrix with values in [0, 1] and diagonal=0
    """
    if analytic_signals is None:
        analytic_signals = compute_analytic_signal(window_data)
        
    n_channels, n_samples = analytic_signals.shape
    wpli_matrix = np.zeros((n_channels, n_channels), dtype=np.float64)

    # Compute pairwise cross-spectra imaginary parts efficiently
    # For channel i and channel j: Im(Z_i * conj(Z_j)) = Im(Z_i)*Re(Z_j) - Re(Z_i)*Im(Z_j)
    re_part = np.real(analytic_signals) # (n_channels, n_samples)
    im_part = np.imag(analytic_signals) # (n_channels, n_samples)

    for i in range(n_channels):
        # Broadcast channel i with all channels j > i
        # cross_im has shape (n_channels - i - 1, n_samples)
        cross_im = im_part[i:i+1, :] * re_part[i+1:, :] - re_part[i:i+1, :] * im_part[i+1:, :]
        
        # Numerator: |sum(Im)|
        sum_im = np.sum(cross_im, axis=1)
        num = np.abs(sum_im)
        
        # Denominator: sum(|Im|)
        denom = np.sum(np.abs(cross_im), axis=1)
        
        # Handle zero denominator to prevent NaN
        valid = denom > 1e-12
        wpli_vals = np.zeros_like(num)
        wpli_vals[valid] = num[valid] / denom[valid]
        
        # Assign to upper and lower triangular matrices
        wpli_matrix[i, i+1:] = wpli_vals
        wpli_matrix[i+1:, i] = wpli_vals

    np.fill_diagonal(wpli_matrix, 0.0)
    return wpli_matrix


def compute_multi_window_wpli(
    windows: list,
) -> np.ndarray:
    """
    Computes WPLI matrices for a list of time windows.
    
    Args:
        windows: List of (n_channels, n_samples) arrays
        
    Returns:
        wpli_matrices: (n_windows, n_channels, n_channels)
    """
    matrices = []
    for win in windows:
        mat = compute_pairwise_wpli(win)
        matrices.append(mat)
    return np.array(matrices)
