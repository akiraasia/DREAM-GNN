import numpy as np
import torch

class PseudoinverseClassifier:
    def __init__(self):
        self.W = None
        self.labels = None

    def fit(self, X, y):
        """
        One-Shot learning using Moore-Penrose Pseudoinverse.
        X: Feature matrix [n_samples, n_features]
        y: Labels [n_samples] (will be one-hot encoded)
        
        W = Y * X_pseudo_inverse
        """
        # Convert inputs to numpy if they are tensors
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        
        # Flatten time series features: [Batch, Time, Channels] -> [Batch, Time*Channels]
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
            
        # One-hot encode y
        unique_labels = np.unique(y)
        self.labels = unique_labels
        
        n_samples = len(y)
        n_classes = len(unique_labels)
        Y_onehot = np.zeros((n_classes, n_samples))
        
        for i, label in enumerate(y):
            idx = np.where(unique_labels == label)[0][0]
            Y_onehot[idx, i] = 1.0
            
        # Add bias term to X
        X_design = np.vstack([X.T, np.ones(n_samples)]) # [features+1, samples]
        
        # Compute Pseudoinverse of X_design
        # W = Y @ pinv(X_design)
        # Dimensions: [classes, samples] @ [samples, features+1] = [classes, features+1]
        
        X_pinv = np.linalg.pinv(X_design)
        self.W = Y_onehot @ X_pinv
        
        return self

    def predict(self, X):
        if self.W is None:
            raise ValueError("Model not fitted.")
            
        # Convert inputs
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
            
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
            
        n_samples = X.shape[0]
        # Add bias
        X_design = np.vstack([X.T, np.ones(n_samples)])
        
        # Linear projection
        # Y_pred = W @ X
        scores = self.W @ X_design # [classes, samples]
        
        # Argmax
        predictions_idx = np.argmax(scores, axis=0)
        return self.labels[predictions_idx]
