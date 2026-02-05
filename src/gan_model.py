import torch
import torch.nn as nn
import numpy as np

class Generator(nn.Module):
    def __init__(self, z_dim, hidden_dim, output_dim, num_layers=2):
        super(Generator, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.rnn = nn.GRU(z_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, z):
        # z: [batch, seq_len, z_dim]
        # h0 initialized to 0
        out, _ = self.rnn(z)
        out = self.linear(out)
        return torch.tanh(out) # Normalize to [-1, 1] usually

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super(Discriminator, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        out, _ = self.rnn(x)
        # Use the last hidden state or average? Use last time step output
        out = self.linear(out[:, -1, :])
        return torch.sigmoid(out)

class DMDLoss(nn.Module):
    def __init__(self):
        super(DMDLoss, self).__init__()
        
    def forward(self, real_data, fake_data):
        """
        Computes the structural difference between Real and Fake metrics using 
        Mock DMD logic differentiable in PyTorch?
        
        True Differentiable SVD is expensive: torch.linalg.svd
        For this prototype, we compute a simplified 'structural' loss based on 
        frequency components (FFT) which is a proxy for the oscillatory modes 
        DMD captures, or we try small SVD.
        
        Let's try a simplified correlation matrix loss which captures spatial 
        structures, as SVD is eigenvalues of correlation matrix.
        """
        # Data: [Batch, Time, Channels] -> Permute to [Batch, Channels, Time] for covariance
        real_perm = real_data.permute(0, 2, 1)
        fake_perm = fake_data.permute(0, 2, 1)
        
        # Covariance matrices (Spatial structure)
        real_cov = torch.matmul(real_perm, real_perm.transpose(1, 2))
        fake_cov = torch.matmul(fake_perm, fake_perm.transpose(1, 2))
        
        loss_cov = torch.mean((real_cov - fake_cov) ** 2)
        
        # Temporal structure (FFT) - approximates the 'eigenvalues' part of DMD
        real_fft = torch.fft.rfft(real_data, dim=1).abs()
        fake_fft = torch.fft.rfft(fake_data, dim=1).abs()
        
        loss_fft = torch.mean((real_fft - fake_fft) ** 2)
        
        return loss_cov + loss_fft
