import numpy as np
from scipy.linalg import svd

class DMDPreProcessor:
    def __init__(self, rank=None):
        self.rank = rank
        self.modes = None
        self.eigenvalues = None
        self.amplitudes = None

    def fit(self, X):
        """
        Compute DMD for data matrix X (n_channels, n_timesteps).
        """
        # 1. Create snapshot matrices
        X1 = X[:, :-1]
        X2 = X[:, 1:]
        
        # 2. SVD of X1
        U, s, Vh = svd(X1, full_matrices=False)
        
        if self.rank is None:
            # Simple rank truncation based on energy if not specified
            # Keep 95% energy for example, or just use all
            r = len(s)
        else:
            r = min(self.rank, len(s))
            
        U_r = U[:, :r]
        s_r = s[:r]
        V_r = Vh[:r, :] # Vh is already transposed in scipy svd
        
        # 3. Compute Atilde (low-rank linear operator)
        # Atilde = U' * X2 * V * S^-1
        # Note: V_r.conj().T is V from the paper definitions usually
        Sr_inv = np.diag(1./s_r)
        
        # X2 @ V_r.conj().T @ Sr_inv
        # In python: X2 @ V_r.T @ Sr_inv
        
        Atilde = U_r.conj().T @ X2 @ V_r.conj().T @ Sr_inv
        
        # 4. Eigendecomposition of Atilde
        lambdas, W = np.linalg.eig(Atilde)
        
        # 5. Compute DMD Modes
        # Phi = X2 * V * S^-1 * W
        phi = X2 @ V_r.conj().T @ Sr_inv @ W
        
        self.eigenvalues = lambdas
        self.modes = phi
        
        return self.modes, self.eigenvalues

    def reconstruct(self, t):
        """
        Reconstruct signal dynamics (Optional, for viz).
        """
        if self.modes is None:
            raise ValueError("Fit the model first.")
        
        # Dynamics: x(t) = Phi * exp(Omega * t) * b
        # Simply returning modes for now as that's what we need for GAN features
        return self.modes
