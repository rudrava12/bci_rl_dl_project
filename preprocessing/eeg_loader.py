"""EEG data loading and preprocessing pipeline."""

from typing import Tuple
import mne
import numpy as np
import logging

logger = logging.getLogger(__name__)


def load_eeg(file_path: str, l_freq: float = 1.0, h_freq: float = 40.0) -> mne.io.BaseRaw:
    """
    Load and preprocess raw EEG data from EDF file.
    
    Args:
        file_path: Path to EDF file
        l_freq: Low frequency cutoff (Hz)
        h_freq: High frequency cutoff (Hz)
        
    Returns:
        Preprocessed MNE raw object
        
    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file format is invalid
    """
    try:
        logger.info(f"Loading EEG from: {file_path}")
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        logger.info(f"EEG loaded: {raw.info['nchan']} channels, {len(raw)} samples")
        
        # Bandpass filter
        raw.filter(l_freq, h_freq, verbose=False)
        logger.info(f"Applied bandpass filter: {l_freq}-{h_freq} Hz")
        
        return raw
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading EEG: {str(e)}")
        raise ValueError(f"Failed to load EEG file: {str(e)}")


def segment_data(
    raw: mne.io.BaseRaw | np.ndarray, 
    window_size: int = 128,
    overlap: float = 0.5
) -> np.ndarray:
    """
    Segment continuous EEG data into windows.
    
    Args:
        raw: MNE raw object or numpy array of shape (n_channels, n_samples)
        window_size: Size of each segment (samples)
        overlap: Overlap ratio between windows (0-1)
        
    Returns:
        Segmented data of shape (n_segments, n_channels, window_size)
    """
    # Extract data if MNE object
    if isinstance(raw, mne.io.BaseRaw):
        data = raw.get_data()
    else:
        data = raw
    
    if data.ndim != 2:
        raise ValueError(f"Expected 2D data, got shape {data.shape}")
    
    n_channels, n_samples = data.shape
    stride = int(window_size * (1 - overlap))
    
    segments = []
    for i in range(0, n_samples - window_size, stride):
        segment = data[:, i:i + window_size]
        segments.append(segment)
    
    result = np.array(segments)
    logger.info(f"Created {len(segments)} segments of shape {result.shape[1:]}")
    return result


def create_labels(num_samples: int, num_classes: int = 3) -> np.ndarray:
    """
    Create placeholder labels for EEG segments.
    
    Args:
        num_samples: Number of samples to generate
        num_classes: Number of classes
        
    Returns:
        Random labels array
    """
    return np.random.randint(0, num_classes, num_samples)


def normalize_eeg(data: np.ndarray, axis: Tuple[int, ...] = (1, 2)) -> np.ndarray:
    """
    Normalize EEG data using z-score normalization.
    
    Args:
        data: EEG data array
        axis: Axes along which to normalize
        
    Returns:
        Normalized data
    """
    mean = np.mean(data, axis=axis, keepdims=True)
    std = np.std(data, axis=axis, keepdims=True)
    return (data - mean) / (std + 1e-8)
