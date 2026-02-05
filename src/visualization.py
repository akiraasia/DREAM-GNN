import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_eeg_signals(time, signals, title="EEG Signals", n_channels=4):
    """
    Plots multi-channel EEG signals.
    signals: [channels, time]
    """
    fig, axes = plt.subplots(n_channels, 1, figsize=(10, 2*n_channels), sharex=True)
    if n_channels == 1:
        axes = [axes]
        
    for i in range(min(n_channels, signals.shape[0])):
        axes[i].plot(time, signals[i, :])
        axes[i].set_ylabel(f"Ch {i+1}")
        axes[i].grid(True, alpha=0.3)
        
    plt.xlabel("Time (s)")
    plt.suptitle(title)
    plt.tight_layout()
    return fig

def plot_dmd_spectrum(eigenvalues, title="DMD Eigenvalues"):
    """
    Plots the unit circle and DMD eigenvalues.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    circle = plt.Circle((0, 0), 1, color='k', fill=False, linestyle='--', alpha=0.5)
    ax.add_artist(circle)
    
    ax.scatter(eigenvalues.real, eigenvalues.imag, c='r', label='Modes')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("Real(λ)")
    ax.set_ylabel("Imag(λ)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig

def plot_training_loss(losses):
    """
    Plots training loss curves.
    losses: dict of list of floats
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, loss_values in losses.items():
        ax.plot(loss_values, label=name)
        
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Training Dynamics")
    ax.legend()
    ax.grid(True)
    return fig
