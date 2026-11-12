"""
src/classifier.py - Dimension Reduction (PCA) and Support Vector Machine (SVM)
Implements PCA feature transformation and SVM with RBF kernel (C = 0.5)
strictly preventing data leakage during training and cross-validation.
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from config import SVM_KERNEL, SVM_C, RANDOM_STATE, PCA_N_COMPONENTS


class SOZClassifierPipeline:
    """
    Standardized classification pipeline:
    1. StandardScaler (fit on train fold only)
    2. PCA (fit on train fold only)
    3. SVC(kernel='rbf', C=0.5, probability=True)
    """

    def __init__(
        self,
        c: float = SVM_C,
        kernel: str = SVM_KERNEL,
        pca_components: Optional[int] = PCA_N_COMPONENTS,
        use_pca: bool = True,
        random_state: int = RANDOM_STATE
    ):
        self.c = c
        self.kernel = kernel
        self.pca_components = pca_components
        self.use_pca = use_pca
        self.random_state = random_state

        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCA] = None
        self.svm: Optional[SVC] = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Fits StandardScaler, PCA, and SVM on training data only.
        """
        # 1. Feature scaling
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)

        # 2. PCA projection
        if self.use_pca and X_train.shape[1] > 1:
            n_comp = self.pca_components
            if n_comp is None or n_comp > X_train.shape[1]:
                n_comp = min(X_train.shape[1], X_train.shape[0])
            self.pca = PCA(n_components=n_comp, random_state=self.random_state)
            X_proj = self.pca.fit_transform(X_scaled)
        else:
            self.pca = None
            X_proj = X_scaled

        # 3. Fit SVM
        self.svm = SVC(
            C=self.c,
            kernel=self.kernel,
            probability=True,
            random_state=self.random_state
        )
        self.svm.fit(X_proj, y_train)
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Transforms test features using fitted train scaler/PCA and returns binary predictions.
        """
        X_proj = self._transform(X_test)
        return self.svm.predict(X_proj)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Returns class probabilities for test samples [P(y=0), P(y=1)].
        """
        X_proj = self._transform(X_test)
        return self.svm.predict_proba(X_proj)

    def decision_function(self, X_test: np.ndarray) -> np.ndarray:
        """
        Returns decision boundary distance values for ROC-AUC computation.
        """
        X_proj = self._transform(X_test)
        return self.svm.decision_function(X_proj)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        """Transforms input using fitted training scaler and PCA."""
        if self.scaler is None or self.svm is None:
            raise RuntimeError("Pipeline must be fitted before calling transform/predict.")
        X_scaled = self.scaler.transform(X)
        if self.pca is not None:
            return self.pca.transform(X_scaled)
        return X_scaled

# =============================================================================
# Endsem Proposed Model: Bidirectional Gated Recurrent Unit (Bi-GRU)
# =============================================================================
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader

    class BiGRUSOZModel(nn.Module):
        """
        Bidirectional GRU with Global Temporal Mean Pooling and MLP Head
        As formulated in Minor Project Report Section 4.6.2 and Algorithm 1.
        """
        def __init__(
            self,
            input_dim: int = 5,
            hidden_dim: int = 16,
            num_layers: int = 2,
            dropout: float = 0.2
        ):
            super().__init__()
            self.gru = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2, 16),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(16, 1)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: (batch_size, seq_len, input_dim) tensor
            Returns:
                logits: (batch_size,) unnormalized logits
            """
            out, _ = self.gru(x)
            # Global Temporal Mean Pooling across temporal sequence (T micro-windows)
            pooled = torch.mean(out, dim=1)
            logits = self.mlp(pooled)
            return logits.squeeze(-1)

    class BiGRUPipeline:
        """
        Endsem Deep Sequential Classification Pipeline with Dynamic Focal Loss.
        """
        def __init__(
            self,
            input_dim: int = 5,
            hidden_dim: int = 16,
            num_layers: int = 2,
            dropout: float = 0.2,
            lr: float = 0.01,
            weight_decay: float = 1e-3,
            epochs: int = 80,
            batch_size: int = 32,
            device: str = "cpu"
        ):
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.dropout = dropout
            self.lr = lr
            self.weight_decay = weight_decay
            self.epochs = epochs
            self.batch_size = batch_size
            self.device = torch.device(device)
            self.model: Optional[BiGRUSOZModel] = None

        def fit(self, X_train: np.ndarray, y_train: np.ndarray):
            """
            Fits Bi-GRU with dynamic class-weighted binary cross-entropy / focal loss.
            """
            # Ensure 3D input: (N_samples, T_windows, N_features)
            if X_train.ndim == 2:
                X_train = np.expand_dims(X_train, axis=1)

            n_pos = max(int(np.sum(y_train == 1)), 1)
            n_neg = max(int(np.sum(y_train == 0)), 1)
            pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=self.device)

            self.model = BiGRUSOZModel(
                input_dim=X_train.shape[2],
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=self.dropout
            ).to(self.device)

            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )

            dataset = TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32)
            )
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            self.model.train()
            for epoch in range(self.epochs):
                for batch_x, batch_y in loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    optimizer.zero_grad()
                    logits = self.model(batch_x)
                    loss = criterion(logits, batch_y)
                    loss.backward()
                    optimizer.step()

            return self

        def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
            if X_test.ndim == 2:
                X_test = np.expand_dims(X_test, axis=1)

            self.model.eval()
            with torch.no_grad():
                x_tensor = torch.tensor(X_test, dtype=torch.float32, device=self.device)
                logits = self.model(x_tensor)
                probs = torch.sigmoid(logits).cpu().numpy()
            return np.column_stack([1.0 - probs, probs])

        def predict(self, X_test: np.ndarray, threshold: float = 0.5) -> np.ndarray:
            probs = self.predict_proba(X_test)[:, 1]
            return (probs >= threshold).astype(int)

except ImportError:
    pass
