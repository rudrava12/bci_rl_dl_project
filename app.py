"""Simplified Streamlit interface for BCI system."""

import streamlit as st
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import logging

from models.cnn_model import EEG_CNN
from rl.dqn_agent import DQNAgent
from rl.replay_buffer import ReplayBuffer
from environment.env import BCIEnv
from preprocessing.eeg_loader import load_eeg, segment_data, normalize_eeg
from config.config import Config

# Setup
st.set_page_config(page_title="🧠 BCI System", layout="centered")

logger = logging.getLogger(__name__)
config = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================
# TITLE & STATUS
# ============================================

st.title("🧠 BCI System")
st.markdown(f"**Device:** {device} | **PyTorch:** {torch.__version__} | **Status:** ✅ Ready")
st.markdown("---")


# ============================================
# LOAD DATA & MODELS
# ============================================

@st.cache_resource
def load_models():
    """Load CNN and DQN models."""
    cnn = EEG_CNN().to(device)
    cnn.eval()
    
    agent = DQNAgent(state_size=config.dqn.state_size, action_size=config.dqn.action_size)
    
    try:
        if os.path.exists(config.paths.cnn_model_path):
            cnn.load_state_dict(torch.load(config.paths.cnn_model_path, map_location=device))
    except Exception as e:
        st.warning(f"⚠️ Could not load CNN: {e}")
    
    try:
        if os.path.exists(config.paths.dqn_model_path):
            agent.load(config.paths.dqn_model_path)
    except Exception as e:
        st.warning(f"⚠️ Could not load DQN: {e}")
    
    return cnn, agent


@st.cache_data
def load_real_eeg():
    """Load real EEG data."""
    base_path = config.paths.data_dir
    all_segments = []
    
    if not os.path.exists(base_path):
        return None
    
    files_found = 0
    for subject in os.listdir(base_path):
        subject_path = os.path.join(base_path, subject)
        
        if not os.path.isdir(subject_path):
            continue
        
        for file in os.listdir(subject_path):
            if file.endswith(".edf"):
                try:
                    file_path = os.path.join(subject_path, file)
                    raw = load_eeg(file_path)
                    segments = segment_data(raw, window_size=config.data.window_size)
                    
                    if config.data.normalize:
                        segments = normalize_eeg(segments)
                    
                    all_segments.append(segments)
                    files_found += 1
                    
                    if files_found >= 5:
                        break
                except Exception as e:
                    continue
        
        if files_found >= 5:
            break
    
    if not all_segments:
        return None
    
    return np.concatenate(all_segments, axis=0)


# Load models and data
cnn, agent = load_models()
real_eeg = load_real_eeg()


# ============================================
# MAIN CONTENT
# ============================================

st.subheader("📊 System Info")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("CNN Status", "✅ Ready")

with col2:
    st.metric("DQN Status", "✅ Ready")

with col3:
    eeg_count = len(real_eeg) if real_eeg is not None else 0
    st.metric("EEG Segments", f"{eeg_count}")


st.markdown("---")


# ============================================
# SIMULATION
# ============================================

st.subheader("🎯 Run Simulation")

if st.button("▶️ Run 10 Episodes"):
    if real_eeg is None:
        st.error("❌ No EEG data found")
    else:
        with st.spinner("Running simulation..."):
            env = BCIEnv(state_size=config.dqn.state_size)
            agent.epsilon = 0
            
            rewards = []
            
            for ep in range(10):
                state = env.reset()
                episode_reward = 0
                
                for step in range(20):
                    action = agent.choose_action(state)
                    state, reward, done = env.step(action)
                    episode_reward += reward
                    if done:
                        break
                
                rewards.append(episode_reward)
            
            # Display results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Reward", f"{np.mean(rewards):.2f}")
            with col2:
                st.metric("Max Reward", f"{np.max(rewards):.2f}")
            with col3:
                st.metric("Min Reward", f"{np.min(rewards):.2f}")
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(rewards, marker='o', linewidth=2, color='steelblue')
            ax.set_xlabel("Episode")
            ax.set_ylabel("Total Reward")
            ax.set_title("Simulation Results")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)


st.markdown("---")


# ============================================
# INFERENCE
# ============================================

st.subheader("🔍 Run Inference")

if st.button("▶️ Predict Action"):
    if real_eeg is None:
        st.error("❌ No EEG data found")
    else:
        idx = np.random.randint(0, len(real_eeg))
        eeg_segment = real_eeg[idx:idx+1]
        
        eeg_tensor = torch.FloatTensor(eeg_segment).to(device)
        
        with torch.no_grad():
            logits, features = cnn(eeg_tensor)
        
        state = features.cpu().numpy()[0]
        agent.epsilon = 0
        action = agent.choose_action(state)
        
        action_names = ["🎯 Focus", "😌 Relax", "😐 Neutral"]
        confidence = float(torch.softmax(logits, dim=1)[0, action].cpu().numpy())
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Predicted Action", action_names[action])
            st.metric("Confidence", f"{confidence:.1%}")
        
        with col2:
            logits_np = logits.detach().cpu().numpy()[0]
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ['green' if i == action else 'lightgray' for i in range(3)]
            ax.bar(action_names, logits_np, color=colors)
            ax.set_ylabel("Score")
            ax.set_title("Action Scores")
            st.pyplot(fig)


st.markdown("---")
st.caption("Last Updated: April 2026 | Status: ✅ Active")
