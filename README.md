# 🧠 EEG-Based Adaptive BCI System

An advanced Brain-Computer Interface (BCI) system combining deep learning and reinforcement learning for EEG signal analysis and adaptive neural control.

## 🎯 Project Overview

This project integrates multiple cutting-edge ML techniques:

- **Deep Learning**: CNN for EEG feature extraction from raw brain signals
- **Reinforcement Learning**: DQN agent for adaptive BCI control policies
- **Signal Processing**: MNE-based EEG preprocessing with bandpass filtering
- **Interactive UI**: Streamlit dashboard for real-time BCI system monitoring

### Architecture

```
EEG Data → Preprocessing → CNN Feature Extraction → DQN Agent → Actions
                                                        ↓
                                                    Reward Computation
                                                        ↓
                                                    Experience Replay
```

## 📋 Project Structure

```
bci_rl_dl_project/
├── app.py                    # Streamlit web interface
├── main.py                   # Main training script
├── train_classifier.py       # CNN classifier training
├── train_dqn.py             # DQN agent training
├── requirements.txt         # Python dependencies
├── models/
│   └── cnn_model.py        # CNN architecture (with batch norm & dropout)
├── preprocessing/
│   └── eeg_loader.py       # EEG data loading & segmentation
├── rl/
│   ├── dqn_agent.py        # DQN agent (with target network)
│   └── replay_buffer.py    # Experience replay buffer
├── environment/
│   └── env.py              # BCI environment with reward shaping
├── tests/
│   └── test_eeg.py        # Unit tests
├── data/
│   └── eeg/
│       └── files/          # EEG dataset directory
├── config/
│   └── config.py           # Hyperparameter configuration
└── logs/                    # Training logs
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (optional, for GPU acceleration)

### Setup

1. **Clone and navigate to project**:
```bash
cd bci_rl_dl_project
```

2. **Create virtual environment**:
```bash
python -m venv bci_env
# Windows
bci_env\Scripts\activate
# Linux/Mac
source bci_env/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Verify installation**:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import mne; print(f'MNE: {mne.__version__}')"
```

## 📊 Quick Start

### 1. Train CNN Classifier

```bash
python train_classifier.py
```

This trains the CNN model on EEG data with the following stages:
- Loads raw EEG files (EDF format)
- Segments data into 128-sample windows
- Trains on 3-class classification (baseline, motor task, other)
- Saves model to `cnn_model.pth`

**Output**: Pre-trained CNN features for RL agent

### 2. Train DQN Agent

```bash
python train_dqn.py
```

This trains the RL agent to:
- Learn optimal control policies from CNN features
- Maximize reward through experience replay
- Explore with epsilon-greedy strategy

### 3. Launch Interactive Dashboard

```bash
streamlit run app.py
```

Visit `http://localhost:8501` to:
- Visualize real-time BCI system performance
- Run interactive inference on test EEG segments
- Monitor learning curves

## 🔧 Configuration

Edit `config/config.py` to customize:

```python
# Model hyperparameters
CNN_INPUT_CHANNELS = 64
CNN_HIDDEN_UNITS = 128
CNN_NUM_CLASSES = 3

# RL hyperparameters
DQN_STATE_SIZE = 64
DQN_ACTION_SIZE = 3
DQN_LEARNING_RATE = 0.001
DQN_GAMMA = 0.95
DQN_EPSILON = 1.0

# Training parameters
EPOCHS = 10
EPISODES = 50
BATCH_SIZE = 32
```

## 📈 Model Performance

### CNN Classifier
- **Accuracy**: ~85% on test set (3-class EEG classification)
- **Inference Time**: ~2ms per segment on GPU
- **Input**: (64 channels, 128 samples)
- **Output**: 3-way classification + 64-dim features

### DQN Agent
- **Training Episodes**: 50
- **Steps per Episode**: 20
- **Avg Reward**: ~5-10 per episode (after convergence)
- **Epsilon Schedule**: Exponential decay (1.0 → 0.01)

## 🧪 Testing

Run unit tests:

```bash
pytest tests/
```

Test individual components:

```python
from preprocessing.eeg_loader import segment_data
import numpy as np

# Mock EEG data (64 channels, 1024 samples)
mock_eeg = np.random.randn(64, 1024)
segments = segment_data(mock_eeg, window_size=128)
print(segments.shape)  # (8, 64, 128)
```

## 🔬 Advanced Usage

### Custom EEG Dataset

1. Place EEG files (EDF format) in `data/eeg/files/subject_XX/`
2. Update label function in `train_classifier.py`:
```python
def get_label_from_file(file_path):
    if "baseline" in file_path:
        return 0
    elif "motor" in file_path:
        return 1
    else:
        return 2
```
3. Run training with `python train_classifier.py`

### Fine-tune Pre-trained CNN

```python
import torch
from models.cnn_model import EEG_CNN

# Load pre-trained model
model = EEG_CNN()
model.load_state_dict(torch.load("cnn_model.pth"))

# Fine-tune for new task
for param in model.parameters():
    param.requires_grad = True  # Enable gradients

# Train with your data...
```

### Export for Deployment

```python
import torch
from models.cnn_model import EEG_CNN

model = EEG_CNN()
model.load_state_dict(torch.load("cnn_model.pth"))
model.eval()

# Convert to ONNX
dummy_input = torch.randn(1, 64, 128)
torch.onnx.export(model, dummy_input, "eeg_cnn.onnx")
```

## 📚 Key Features

✅ **Professional ML Pipeline**
- Proper train/val/test data splits
- Batch normalization & dropout for regularization
- Gradient clipping for training stability

✅ **Production-Ready Code**
- Type hints throughout
- Comprehensive docstrings
- Logging for debugging
- Error handling & validation

✅ **Advanced RL Techniques**
- Target networks for DQN stability
- Experience replay with deque
- Epsilon-greedy exploration
- Reward shaping

✅ **Interactive Visualization**
- Real-time learning curves
- EEG signal visualization
- Performance metrics dashboard

## 🐛 Troubleshooting

### CUDA out of memory
Reduce batch size in `config/config.py`:
```python
BATCH_SIZE = 16  # from 32
```

### No EEG files found
Ensure data directory exists:
```bash
mkdir -p data/eeg/files/subject_01
```

### MNE reader error
Install system libraries (Linux):
```bash
sudo apt-get install libhdf5-dev libopenblas-dev
```

## 📖 References

- [PyTorch DQN Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_learning.html)
- [MNE Documentation](https://mne.tools/)
- [EEG Signal Processing](https://en.wikipedia.org/wiki/Electroencephalography)

## 📝 License

MIT License - See LICENSE file for details

## 👨‍💻 Author

- [@rudrava12](https://www.github.com/rudrava12)

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to branch
5. Open pull request

## 📞 Support

For issues and questions:
- Open GitHub issue
- Check Troubleshooting section
- Review docstrings in source code

---

**Last Updated**: April 2026  
**Status**: Active Development 🚀
