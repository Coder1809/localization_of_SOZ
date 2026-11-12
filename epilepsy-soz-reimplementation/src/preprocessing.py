"""
src/preprocessing.py - Signal Preprocessing Pipeline
Implements downsampling (200 Hz), 60 Hz notch filtering, 0.5-80 Hz bandpass filtering,
Common Average Referencing (CAR), frequency band decomposition, and window segmentation.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.signal as signal
from config import (
    TARGET_SAMPLING_RATE,
    NOTCH_FREQUENCY,
    NOTCH_Q,
    BANDPASS_LOW,
    BANDPASS_HIGH,
    FILTER_ORDER,
    FREQUENCY_BANDS,
    TARGET_BAND,
    WINDOW_LENGTH_SEC,
    WINDOW_SHIFT_SEC
)


def resample_signal(
    data: np.ndarray,
    orig_srate: float,
    target_srate: float = TARGET_SAMPLING_RATE
) -> np.ndarray:
    """
    Resamples multi-channel signal to the target sampling rate using Fourier method.
    
    Args:
        data: (n_channels, n_samples)
        orig_srate: Original sampling rate in Hz
        target_srate: Desired sampling rate in Hz (default 200 Hz)
    """
    if np.isclose(orig_srate, target_srate):
        return data
    n_samples_orig = data.shape[1]
    n_samples_target = int(round(n_samples_orig * target_srate / orig_srate))
    resampled = signal.resample(data, n_samples_target, axis=1)
    return resampled


def apply_notch_filter(
    data: np.ndarray,
    srate: float = TARGET_SAMPLING_RATE,
    freq: float = NOTCH_FREQUENCY,
    q: float = NOTCH_Q
) -> np.ndarray:
    """
    Applies an IIR notch filter at the specified frequency (default 60 Hz) to eliminate line noise.
    """
    b, a = signal.iirnotch(w0=freq, Q=q, fs=srate)
    filtered = signal.filtfilt(b, a, data, axis=1)
    return filtered


def apply_bandpass_filter(
    data: np.ndarray,
    srate: float = TARGET_SAMPLING_RATE,
    lowcut: float = BANDPASS_LOW,
    highcut: float = BANDPASS_HIGH,
    order: int = FILTER_ORDER
) -> np.ndarray:
    """
    Applies a zero-phase Butterworth bandpass filter (default 0.5 - 80 Hz).
    """
    nyq = 0.5 * srate
    low = max(lowcut / nyq, 0.0001)
    high = min(highcut / nyq, 0.9999)
    b, a = signal.butter(order, [low, high], btype="bandpass")
    filtered = signal.filtfilt(b, a, data, axis=1)
    return filtered


def apply_car_rereferencing(data: np.ndarray) -> np.ndarray:
    """
    Applies Common Average Reference (CAR) re-referencing.
    Subtracts the average of all retained good channels at each time point.
    """
    car_mean = np.mean(data, axis=0, keepdims=True)
    return data - car_mean


def filter_frequency_band(
    data: np.ndarray,
    band_name: str = TARGET_BAND,
    srate: float = TARGET_SAMPLING_RATE,
    order: int = FILTER_ORDER
) -> np.ndarray:
    """
    Filters preprocessed data into a specific EEG frequency band (e.g. gamma: 30-60 Hz).
    """
    if band_name not in FREQUENCY_BANDS:
        raise ValueError(f"Unknown frequency band: {band_name}. Supported: {list(FREQUENCY_BANDS.keys())}")
    lowcut, highcut = FREQUENCY_BANDS[band_name]
    return apply_bandpass_filter(data, srate=srate, lowcut=lowcut, highcut=highcut, order=order)


def preprocess_recording(
    raw_data: np.ndarray,
    orig_srate: float,
    target_srate: float = TARGET_SAMPLING_RATE
) -> np.ndarray:
    """
    Executes the full paper preprocessing pipeline:
    1. Downsample to 200 Hz
    2. 60 Hz notch filter
    3. 0.5 - 80 Hz bandpass filter
    4. Common average re-referencing (CAR)
    
    Returns:
        preprocessed_data: (n_channels, n_samples_200hz)
    """
    # 1. Resample to 200 Hz
    data_200 = resample_signal(raw_data, orig_srate=orig_srate, target_srate=target_srate)
    
    # 2. 60 Hz Notch filter
    data_notched = apply_notch_filter(data_200, srate=target_srate, freq=NOTCH_FREQUENCY)
    
    # 3. 0.5 - 80 Hz Bandpass filter
    data_bandpass = apply_bandpass_filter(data_notched, srate=target_srate, lowcut=BANDPASS_LOW, highcut=BANDPASS_HIGH)
    
    # 4. Common Average Reference (CAR)
    data_car = apply_car_rereferencing(data_bandpass)
    
    return data_car


def segment_into_windows(
    data: np.ndarray,
    srate: float = TARGET_SAMPLING_RATE,
    window_sec: float = WINDOW_LENGTH_SEC,
    shift_sec: float = WINDOW_SHIFT_SEC,
    max_windows: Optional[int] = None
) -> List[np.ndarray]:
    """
    Segments data into overlapping windows of duration `window_sec` with step `shift_sec`.
    
    Args:
        data: (n_channels, n_samples)
        srate: Sampling rate (200 Hz)
        window_sec: Window length in seconds (30.0 s)
        shift_sec: Window shift in seconds (25.0 s)
        max_windows: Maximum number of windows to extract (optional)
        
    Returns:
        List of 2D arrays, each of shape (n_channels, window_samples)
    """
    n_samples = data.shape[1]
    win_len = int(round(window_sec * srate))
    win_shift = int(round(shift_sec * srate))

    if n_samples < win_len:
        # If recording is shorter than window, pad with zero-mean edge replication
        pad_width = win_len - n_samples
        padded_data = np.pad(data, ((0, 0), (0, pad_width)), mode="reflect")
        return [padded_data]

    windows = []
    start = 0
    while start + win_len <= n_samples:
        win_data = data[:, start : start + win_len]
        windows.append(win_data)
        start += win_shift
        if max_windows and len(windows) >= max_windows:
            break

    return windows
