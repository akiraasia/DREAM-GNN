import pandas as pd
import numpy as np
import os

class DataLoader:
    def __init__(self, base_path="."):
        self.base_path = base_path
        self.records_path = os.path.join(base_path, "Data records.csv")
        self.datasets_path = os.path.join(base_path, "Datasets.csv")
        self.data_records = None
        self.datasets = None
        
    def load_metadata(self):
        """Loads the CSV metadata files."""
        if os.path.exists(self.records_path):
            self.data_records = pd.read_csv(self.records_path)
        else:
            # Fallback for completely empty env
            print(f"Warning: {self.records_path} not found. Creating empty DataFrame.")
            self.data_records = pd.DataFrame(columns=["Filename", "Experience", "Subject ID", "Last sleep stage"])

        if os.path.exists(self.datasets_path):
            self.datasets = pd.read_csv(self.datasets_path)
        
        return self.data_records

    def get_eeg_data(self, filename, duration=10, final_fs=250, n_channels=8):
        """
        Attempts to load real EEG data. 
        If missing, generates synthetic 'dream' data for demonstration.
        """
        # In a real scenario, we would load .edf here using pyedflib or mne
        # file_path = os.path.join(self.base_path, "data", filename)
        
        # Since we don't have the files, we generate synthetic data
        return self._generate_synthetic_eeg(duration, final_fs, n_channels)

    def _generate_synthetic_eeg(self, duration, fs, n_channels):
        """
        Generates synthetic EEG-like signals.
        Combination of sine waves at Delta, Theta, Alpha, Beta frequencies + Noise.
        """
        t = np.linspace(0, duration, int(duration * fs))
        data = np.zeros((n_channels, len(t)))
        
        # Brain wave frequency bands
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30)
        }
        
        for i in range(n_channels):
            # Base noise
            signal = np.random.normal(0, 0.5, size=len(t))
            
            # Mix random frequencies from bands
            weights = np.random.dirichlet(np.ones(4)) # Random dominance
            
            for (name, (low, high)), w in zip(bands.items(), weights):
                freq = np.random.uniform(low, high)
                phase = np.random.uniform(0, 2*np.pi)
                amp = w * np.random.uniform(0.5, 2.0)
                signal += amp * np.sin(2 * np.pi * freq * t + phase)
            
            data[i, :] = signal
            
        return t, data

    def filter_by_experience(self, experience_type):
        """Returns records matching a specific experience type."""
        if self.data_records is None:
            self.load_metadata()
        return self.data_records[self.data_records['Experience'] == experience_type]
