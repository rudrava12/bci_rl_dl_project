"""Unit tests for EEG data loading and preprocessing."""

import numpy as np
from preprocessing.eeg_loader import segment_data


def test_segment_data():
    """Test EEG data segmentation."""
    # Create mock EEG data: 64 channels, 1024 samples
    mock_data = np.random.randn(64, 1024)
    
    segments = segment_data(mock_data, window_size=128)
    
    assert segments.ndim == 3, "Segments should be 3D array"
    assert segments.shape[1] == 64, "Should have 64 channels"
    assert segments.shape[2] == 128, "Window size should be 128"
    assert len(segments) > 0, "Should have at least one segment"
    
    print(f"✓ Segmentation test passed: {len(segments)} segments created")


def test_segment_shape():
    """Test segment shape consistency."""
    mock_data = np.random.randn(64, 512)
    segments = segment_data(mock_data, window_size=64)
    
    # Each segment should have shape (64, 64)
    for seg in segments:
        assert seg.shape == (64, 64), f"Expected shape (64, 64), got {seg.shape}"
    
    print(f"✓ Shape test passed: all segments have correct shape")


if __name__ == "__main__":
    test_segment_data()
    test_segment_shape()
    print("\n✅ All tests passed!")