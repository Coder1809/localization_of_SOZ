"""
src/features.py - Feature Extraction, Fusion, and Sample Balancing
Coordinates WPLI/ENCS network feature extraction and Takens/PE topological feature extraction,
applying the paper's 5x focal oversampling and multi-modal fusion.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from config import (
    TARGET_BAND,
    WINDOW_LENGTH_SEC,
    WINDOW_SHIFT_SEC,
    SOZ_OVERSAMPLE_SLICES,
    PROPORTIONAL_THRESHOLD,
    TARGET_SAMPLING_RATE
)
from src.preprocessing import filter_frequency_band, segment_into_windows
from src.wpli import compute_pairwise_wpli
from src.brain_network import binarize_network_proportional, compute_encs
from src.topology import compute_persistence_entropy


def extract_window_features(
    window_data_gamma: np.ndarray,
    compute_tda: bool = True,
    channels_to_compute_tda: Optional[List[int]] = None,
    proportional_threshold: float = PROPORTIONAL_THRESHOLD
) -> Dict[str, np.ndarray]:
    """
    Extracts ENCS and PE features for a single 30s multi-channel epoch.
    """
    n_channels = window_data_gamma.shape[0]

    # 1. WPLI Connectivity Matrix
    wpli_matrix = compute_pairwise_wpli(window_data_gamma)

    # 2. Proportional Thresholding & ENCS
    binary_adj = binarize_network_proportional(wpli_matrix, threshold=proportional_threshold)
    encs = compute_encs(binary_adj)

    result = {
        "encs": encs,
        "wpli_matrix": wpli_matrix,
        "binary_adj": binary_adj
    }

    if compute_tda:
        pe_h0 = np.zeros(n_channels)
        pe_h1 = np.zeros(n_channels)
        pe_h2 = np.zeros(n_channels)
        pe_total = np.zeros(n_channels)

        target_channels = channels_to_compute_tda if channels_to_compute_tda is not None else range(n_channels)

        for ch_idx in target_channels:
            ch_signal = window_data_gamma[ch_idx, :]
            pe_dict = compute_persistence_entropy(ch_signal)
            pe_h0[ch_idx] = pe_dict["pe_h0"]
            pe_h1[ch_idx] = pe_dict["pe_h1"]
            pe_h2[ch_idx] = pe_dict["pe_h2"]
            pe_total[ch_idx] = pe_dict["pe_total"]

        result.update({
            "pe_h0": pe_h0,
            "pe_h1": pe_h1,
            "pe_h2": pe_h2,
            "pe_total": pe_total
        })

    return result


def extract_recording_dataset(
    recording_dict: Dict[str, Any],
    preprocessed_data: np.ndarray,
    feature_mode: str = "encs_pe",
    soz_slices: int = SOZ_OVERSAMPLE_SLICES,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Extracts balanced feature matrix X and label vector y for a recording:
    - Gamma band filtering
    - 5 overlapping slices (30s window, 25s shift)
    - 5 slices extracted for SOZ channels (5x augmentation)
    - 1 slice extracted for non-SOZ channels
    - Assembles final feature dataset matching paper Table 3 & 4 sample counts.
    
    Args:
        recording_dict: Info dict from dataset loader
        preprocessed_data: (n_channels, n_samples) 200 Hz CAR filtered data
        feature_mode: 'encs' (Table 3) or 'encs_pe' (Table 4)
        soz_slices: Number of slices for SOZ channels (default 5)
        verbose: Print progress
        
    Returns:
        dict containing:
            - 'X': (n_samples, n_features) feature matrix
            - 'y': (n_samples,) binary label vector (1=SOZ, 0=Non-SOZ)
            - 'feature_names': list of feature column names
            - 'sample_channel_names': list of channel names for each sample
            - 'channel_labels': original channel ground-truth labels
            - 'n_positives': count of positive samples
            - 'n_negatives': count of negative samples
            - 'total_epochs': total sample count
    """
    labels = recording_dict["labels"] # (n_channels,) 1=SOZ, 0=Non-SOZ
    channel_names = recording_dict["channel_names"]
    n_channels = len(labels)
    soz_indices = np.where(labels == 1)[0]
    non_soz_indices = np.where(labels == 0)[0]

    # 1. Filter into Gamma band (30-60 Hz)
    gamma_data = filter_frequency_band(preprocessed_data, band_name=TARGET_BAND)

    # Dynamic focal balancing: ensure positive sample count approximates negative count
    n_soz = len(soz_indices)
    n_non_soz = len(non_soz_indices)
    if soz_slices is None or soz_slices == 5:
        target_slices = max(5, int(np.ceil(n_non_soz / max(1, n_soz))))
    else:
        target_slices = int(soz_slices)

    # 2. Extract distinct windows across recording
    distinct_windows = segment_into_windows(
        gamma_data,
        srate=TARGET_SAMPLING_RATE,
        window_sec=WINDOW_LENGTH_SEC,
        shift_sec=WINDOW_SHIFT_SEC,
        max_windows=None
    )
    n_distinct = len(distinct_windows)

    compute_tda = ("pe" in feature_mode.lower())

    # 3. Extract features for distinct windows
    window_features = []
    for w_idx in range(n_distinct):
        if verbose and w_idx < 5:
            print(f"  Extracting features for window {w_idx + 1}/{min(5, n_distinct)}...")
        # Window 0 requires all channels; subsequent windows only need SOZ channels
        channels_tda = None if w_idx == 0 else soz_indices
        win_feat = extract_window_features(
            distinct_windows[w_idx],
            compute_tda=compute_tda,
            channels_to_compute_tda=channels_tda,
            proportional_threshold=PROPORTIONAL_THRESHOLD
        )
        window_features.append(win_feat)

    # 4. Construct balanced dataset
    X_list = []
    y_list = []
    sample_channels = []

    # Non-SOZ channels (Window 0)
    w0 = window_features[0]
    for ch_idx in non_soz_indices:
        feat_vector = _build_feature_vector(w0, ch_idx, feature_mode)
        X_list.append(feat_vector)
        y_list.append(0)
        sample_channels.append(channel_names[ch_idx])

    # SOZ channels (multi-window samples matching target_slices)
    for s_idx in range(target_slices):
        w_feat = window_features[s_idx % n_distinct]
        for ch_idx in soz_indices:
            feat_vector = _build_feature_vector(w_feat, ch_idx, feature_mode)
            X_list.append(feat_vector)
            y_list.append(1)
            sample_channels.append(f"{channel_names[ch_idx]}_w{s_idx+1}")

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=int)

    feature_names = ["ENCS"]
    if compute_tda:
        feature_names.extend(["PE_H0", "PE_H1", "PE_H2", "PE_total"])

    return {
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "sample_channel_names": sample_channels,
        "channel_names": channel_names,
        "channel_labels": labels,
        "n_positives": int(np.sum(y == 1)),
        "n_negatives": int(np.sum(y == 0)),
        "total_epochs": len(y),
        "window_features": window_features
    }


def _build_feature_vector(
    feat_dict: Dict[str, np.ndarray],
    ch_idx: int,
    feature_mode: str
) -> List[float]:
    """Helper to assemble feature vector for channel ch_idx."""
    encs_val = float(feat_dict["encs"][ch_idx])
    if "pe" in feature_mode.lower():
        pe_h0 = float(feat_dict["pe_h0"][ch_idx])
        pe_h1 = float(feat_dict["pe_h1"][ch_idx])
        pe_h2 = float(feat_dict["pe_h2"][ch_idx])
        pe_tot = float(feat_dict["pe_total"][ch_idx])
        return [encs_val, pe_h0, pe_h1, pe_h2, pe_tot]
    else:
        return [encs_val]
