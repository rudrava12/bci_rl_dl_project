"""BCI Preprocessing package."""

from .eeg_loader import load_eeg, segment_data, normalize_eeg, create_labels

__all__ = ['load_eeg', 'segment_data', 'normalize_eeg', 'create_labels']
