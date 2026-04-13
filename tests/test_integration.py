"""Integration tests for the BCI system."""

import torch
import numpy as np
import tempfile
from pathlib import Path

from models.cnn_model import EEG_CNN
from rl.dqn_agent import DQNAgent
from rl.replay_buffer import ReplayBuffer
from environment.env import BCIEnv
from preprocessing.eeg_loader import normalize_eeg
from config.config import Config


def test_cnn_forward_pass():
    """Test CNN forward pass."""
    cnn = EEG_CNN()
    
    # Random input: (batch=2, channels=64, samples=128)
    x = torch.randn(2, 64, 128)
    
    logits, features = cnn(x)
    
    assert logits.shape == (2, 3), f"Expected logits shape (2, 3), got {logits.shape}"
    assert features.shape == (2, 64), f"Expected features shape (2, 64), got {features.shape}"
    
    print("✓ CNN forward pass test passed")


def test_dqn_agent():
    """Test DQN agent."""
    agent = DQNAgent(state_size=64, action_size=3)
    
    # Test action selection
    state = np.random.randn(64)
    action = agent.choose_action(state)
    
    assert 0 <= action < 3, f"Invalid action: {action}"
    
    # Test training
    batch = [
        (np.random.randn(64), i % 3, 1.0, np.random.randn(64), 0)
        for i in range(32)
    ]
    loss = agent.train(batch)
    
    assert isinstance(loss, float), f"Expected float loss, got {type(loss)}"
    
    print("✓ DQN agent test passed")


def test_replay_buffer():
    """Test replay buffer."""
    buffer = ReplayBuffer(capacity=100)
    
    # Store experiences
    for i in range(50):
        exp = (np.random.randn(64), i % 3, 1.0, np.random.randn(64), 0)
        buffer.store(exp)
    
    assert buffer.size() == 50, f"Expected size 50, got {buffer.size()}"
    
    # Sample batch
    batch = buffer.sample(32)
    assert len(batch) == 32, f"Expected batch size 32, got {len(batch)}"
    
    # Test clear
    buffer.clear()
    assert buffer.size() == 0, f"Expected size 0 after clear, got {buffer.size()}"
    
    print("✓ Replay buffer test passed")


def test_bci_environment():
    """Test BCI environment."""
    env = BCIEnv(state_size=3)
    
    # Test reset
    state = env.reset()
    assert state.shape == (3,), f"Expected state shape (3,), got {state.shape}"
    
    # Test step
    for _ in range(5):
        action = np.random.randint(0, 3)
        next_state, reward, done = env.step(action)
        
        assert next_state.shape == (3,), f"Expected state shape (3,), got {next_state.shape}"
        assert isinstance(reward, float), f"Expected float reward, got {type(reward)}"
        assert isinstance(done, bool), f"Expected bool done, got {type(done)}"
    
    print("✓ BCI environment test passed")


def test_data_normalization():
    """Test data normalization."""
    # Random data: (10 segments, 64 channels, 128 samples)
    data = np.random.randn(10, 64, 128)
    
    normalized = normalize_eeg(data)
    
    # Check mean and std
    assert np.isclose(normalized.mean(), 0, atol=1e-6), "Mean should be close to 0"
    assert np.isclose(normalized.std(), 1, atol=1e-1), "Std should be close to 1"
    
    print("✓ Data normalization test passed")


def test_model_persistence():
    """Test model saving and loading."""
    cnn = EEG_CNN()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "model.pth"
        
        # Save
        torch.save(cnn.state_dict(), path)
        assert path.exists(), "Model file not saved"
        
        # Load
        cnn_loaded = EEG_CNN()
        cnn_loaded.load_state_dict(torch.load(path))
        
        # Test they produce same output
        x = torch.randn(1, 64, 128)
        out1, feat1 = cnn(x)
        out2, feat2 = cnn_loaded(x)
        
        assert torch.allclose(out1, out2, atol=1e-5), "Outputs don't match after loading"
    
    print("✓ Model persistence test passed")


def test_config():
    """Test configuration system."""
    config = Config()
    
    # Check all configs exist
    assert config.cnn is not None
    assert config.dqn is not None
    assert config.data is not None
    assert config.training is not None
    assert config.paths is not None
    assert config.env is not None
    
    # Check conversions
    config_dict = config.to_dict()
    assert isinstance(config_dict, dict)
    assert len(config_dict) > 0
    
    print("✓ Configuration test passed")


def test_integrated_pipeline():
    """Test full integrated pipeline."""
    config = Config()
    
    # Initialize components
    cnn = EEG_CNN()
    agent = DQNAgent(state_size=config.dqn.state_size, action_size=config.dqn.action_size)
    buffer = ReplayBuffer()
    env = BCIEnv(state_size=config.env.state_size)
    
    # Mock EEG data: (64 channels, 256 samples)
    mock_eeg = np.random.randn(64, 256)
    
    # Segmentation
    num_segments = (256 - 128) // 128 + 1
    segments = []
    for i in range(0, mock_eeg.shape[1] - 128, 128):
        segments.append(mock_eeg[:, i:i+128])
    
    assert len(segments) > 0, "No segments created"
    
    # CNN forward pass
    segment_tensor = torch.FloatTensor(np.array([segments[0]]))
    logits, features = cnn(segment_tensor)
    
    assert logits.shape == (1, 3)
    assert features.shape == (1, 64)
    
    # RL loop
    state = features.detach().numpy()[0]
    for _ in range(10):
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        buffer.store((state, action, reward, next_state, float(done)))
        state = next_state
    
    # Train agent
    if buffer.size() >= 32:
        batch = buffer.sample(32)
        loss = agent.train(batch)
        assert isinstance(loss, float)
    
    print("✓ Integrated pipeline test passed")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("RUNNING INTEGRATION TESTS")
    print("="*60 + "\n")
    
    test_cnn_forward_pass()
    test_dqn_agent()
    test_replay_buffer()
    test_bci_environment()
    test_data_normalization()
    test_model_persistence()
    test_config()
    test_integrated_pipeline()
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED")
    print("="*60 + "\n")
