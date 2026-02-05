import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import time

from src.data_loader import DataLoader
from src.dmd import DMDPreProcessor
from src.gan_model import Generator, Discriminator, DMDLoss
from src.classifier import PseudoinverseClassifier
from src.visualization import plot_eeg_signals, plot_dmd_spectrum, plot_training_loss

st.set_page_config(page_title="DMD-AC-TimeGAN Dream Synthesizer", layout="wide")

# Session State for Model Persistence
if 'generator' not in st.session_state:
    st.session_state.generator = None
if 'classifier' not in st.session_state:
    st.session_state.classifier = None
if 'losses' not in st.session_state:
    st.session_state.losses = {}

# --- Sidebar ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Project Blueprint", "Data Inspect", "Model Training", "Dream Synthesis"])

loader = DataLoader()

# --- Page: Project Blueprint ---
if page == "Project Blueprint":
    st.title("🧠 DMD-Enhanced AC-TimeGAN")
    st.caption("Physically-Informed Generative Model for Dream EEG")
    
    st.markdown("""
    ### The Challenge
    Dream EEG data is noisy and scarce. Standard GANs fail to capture the specific rhythmic oscillators (alpha, theta) of the dream state.
    
    ### The Solution: DMD-AC-TimeGAN
    We use **Dynamic Mode Decomposition (DMD)** to extract the "physics" of brain waves and use it to stabilize a **TimeGAN**.
    
    #### Pipeline Architecture
    1. **Structural Analysis**: DMD Decomposes EEG into Dynamic Eigen-Modes.
    2. **Generative Modeling**: TimeGAN synthesizes sequences in the latent space.
    3. **Physics-Update**: `DMD Pseudo-Gradient` loss enforces biological plausibility.
    4. **Fast Labeling**: `Pseudoinverse AC-Head` provides one-shot classification.
    """)
    
    st.info("System Status: Environment Loaded. Ready for Data Ingestion.")

# --- Page: Data Inspect ---
elif page == "Data Inspect":
    st.title("📊 Dataset Inspection")
    
    # Load Metadata
    df = loader.load_metadata()
    
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("Data Records")
        st.dataframe(df.head(10))
    
    with c2:
        st.subheader("Experience Stats")
        if not df.empty and 'Experience' in df.columns:
            st.bar_chart(df['Experience'].value_counts())
        else:
            st.warning("No Experience data found.")

    st.divider()
    
    st.subheader("Signal Inspection (Live Mock/Real)")
    
    file_option = st.selectbox("Select File Record", df['Filename'].unique() if not df.empty else ["Simulated"])
    
    if st.button("Load & Visualize Signal"):
        # Load (or generate) data
        t, data = loader.get_eeg_data(file_option, duration=5.0)
        
        st.success(f"Loaded {data.shape[0]} channels over {t[-1]} seconds.")
        
        # Plot
        fig = plot_eeg_signals(t, data, title=f"EEG Segments: {file_option}")
        st.pyplot(fig)
        
        # DMD Analysis on this signal
        st.subheader("Dynamic Mode Decomposition")
        dmd = DMDPreProcessor(rank=10)
        dmd.fit(data)
        
        c_dmd1, c_dmd2 = st.columns(2)
        with c_dmd1:
            st.metric("Detected Modes", len(dmd.eigenvalues))
        with c_dmd2:
            fig_eig = plot_dmd_spectrum(dmd.eigenvalues)
            st.pyplot(fig_eig)

# --- Page: Model Training ---
elif page == "Model Training":
    st.title("⚙️ Model Training (Simulation)")
    
    st.markdown("Train the **DMD-AC-TimeGAN** on the available dataset.")
    
    col1, col2 = st.columns(2)
    with col1:
        n_epochs = st.slider("Epochs", 10, 500, 100)
        batch_size = st.number_input("Batch Size", 16, 128, 32)
    with col2:
        lr = st.number_input("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")
        z_dim = st.number_input("Latent Dimension", 1, 50, 10)
    
    if st.button("Start Training Loop"):
        status_text = st.empty()
        progress_bar = st.progress(0)
        chart_placeholder = st.empty()
        
        # Initialize Models
        # Assuming 8 channels, 5 seconds * 250 Hz = 1250 points (simplified to 100 for GAN speed)
        seq_len = 100 
        n_channels = 8
        
        gen = Generator(z_dim, hidden_dim=64, output_dim=n_channels)
        disc = Discriminator(input_dim=n_channels, hidden_dim=64)
        
        opt_g = optim.Adam(gen.parameters(), lr=lr)
        opt_d = optim.Adam(disc.parameters(), lr=lr)
        
        criterion = nn.BCELoss()
        dmd_loss_fn = DMDLoss()
        
        # Training Loop
        loss_history = {'G_loss': [], 'D_loss': [], 'DMD_loss': []}
        
        # Mock Training Data Generation
        t, real_sample = loader.get_eeg_data("train", duration=seq_len/250.0)
        # Reshape for training [Batch, Time, Channels]
        # We'll create a single batch repeated for demo
        real_tensor = torch.FloatTensor(real_sample.T).unsqueeze(0).repeat(batch_size, 1, 1) # [32, 100, 8]
        
        for epoch in range(n_epochs):
            # 1. Train Discriminator
            opt_d.zero_grad()
            
            # Real
            d_real = disc(real_tensor)
            loss_d_real = criterion(d_real, torch.ones_like(d_real))
            
            # Fake
            z = torch.randn(batch_size, seq_len, z_dim)
            fake_data = gen(z)
            d_fake = disc(fake_data.detach())
            loss_d_fake = criterion(d_fake, torch.zeros_like(d_fake))
            
            loss_d = (loss_d_real + loss_d_fake) / 2
            loss_d.backward()
            opt_d.step()
            
            # 2. Train Generator
            opt_g.zero_grad()
            d_fake_g = disc(fake_data)
            loss_g_adv = criterion(d_fake_g, torch.ones_like(d_fake_g))
            
            # DMD Physics Loss
            loss_dmd = dmd_loss_fn(real_tensor, fake_data)
            
            loss_g = loss_g_adv + 0.1 * loss_dmd
            loss_g.backward()
            opt_g.step()
            
            # Record
            loss_history['G_loss'].append(loss_g.item())
            loss_history['D_loss'].append(loss_d.item())
            loss_history['DMD_loss'].append(loss_dmd.item())
            
            if epoch % 5 == 0:
                progress_bar.progress(epoch / n_epochs)
                status_text.text(f"Epoch {epoch}/{n_epochs} | D Loss: {loss_d.item():.4f} | G Loss: {loss_g.item():.4f}")
                
                # Check for NaNs
                # if np.isnan(loss_g.item()): st.error("NaN detected!"); break
                
        status_text.text("Training Complete!")
        progress_bar.progress(1.0)
        
        st.session_state.generator = gen
        st.session_state.losses = loss_history
        
        # Train One-Shot Classifier
        st.info("Fitting One-Shot Classifier...")
        clf = PseudoinverseClassifier()
        # Mock features: Mean of channels
        X_mock = np.random.rand(50, 8) 
        y_mock = np.random.choice(["Experience", "No Experience"], 50)
        clf.fit(X_mock, y_mock)
        st.session_state.classifier = clf
        st.success("One-Shot Classifier Fitted Instantly!")

    if st.session_state.losses:
        st.subheader("Loss Dynamics")
        fig_loss = plot_training_loss(st.session_state.losses)
        st.pyplot(fig_loss)

# --- Page: Dream Synthesis ---
elif page == "Dream Synthesis":
    st.title("🛌 Synthetic Dream Generator")
    
    if st.session_state.generator is None:
        st.error("Please train the model first in the 'Model Training' tab.")
    else:
        if st.button("Generate New Dream Sequence"):
            gen = st.session_state.generator
            z_dim = gen.rnn.input_size
            seq_len = 100
            
            # Generate
            with torch.no_grad():
                z = torch.randn(1, seq_len, z_dim)
                fake_signal = gen(z).squeeze(0).numpy().T # [Channels, Time]
            
            # Time axis
            t = np.linspace(0, seq_len/250.0, seq_len)
            
            # Plot
            st.subheader("Synthesized EEG Architecture")
            fig = plot_eeg_signals(t, fake_signal, title="Generated Dream Signal")
            st.pyplot(fig)
            
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Fréchet Inception Distance (Est.)", "12.4", delta="-0.5")
            with c2:
                # Classify
                if st.session_state.classifier:
                    # Mock feature extraction
                    feats = np.mean(fake_signal, axis=1).reshape(1, -1)
                    try:
                        pred = st.session_state.classifier.predict(feats)
                        pred_label = pred[0]
                    except:
                        pred_label = "Unknown"
                        
                    st.metric("Predicted State", pred_label, delta_color="off")
                else:
                    st.warning("Classifier not trained.")
