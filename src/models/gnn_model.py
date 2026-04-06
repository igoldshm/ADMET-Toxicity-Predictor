"""
Graph Neural Network Model for Tox21
=====================================
Implements an AttentiveFP-inspired GNN with multi-task binary classification
heads — one per Tox21 assay.

Architecture:
    Atom Embedding → Graph Attention Layers → Readout (global attention pool)
    → Shared Dense Trunk → Per-task Sigmoid Heads

Why GNNs for toxicity?
    Fingerprints hash circular atom environments into fixed-length bit-vectors,
    inevitably colliding distinct environments.  GNNs operate directly on the
    molecular graph, preserving exact topology and enabling the network to learn
    arbitrary subgraph patterns — crucial for detecting reactive motifs like
    epoxides (3-membered ring + O) or Michael acceptors (C=C adjacent to C=O).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import deepchem as dc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import AttentiveFP, global_add_pool, global_mean_pool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GNNConfig:
    """Hyperparameters for the multi-task GNN."""
    # Architecture
    node_in_feats: int    = 39     # from MolGraphConvFeaturizer
    edge_in_feats: int    = 10
    hidden_channels: int  = 200
    num_layers: int       = 2
    num_timesteps: int    = 2      # AttentiveFP readout steps
    dropout: float        = 0.2

    # Training
    n_tasks: int          = 12
    learning_rate: float  = 1e-3
    weight_decay: float   = 1e-5
    batch_size: int       = 64
    n_epochs: int         = 50
    patience: int         = 10     # early stopping

    # Optimiser
    use_class_weights: bool = True

    # Checkpointing
    checkpoint_dir: Path  = Path("checkpoints")


# ---------------------------------------------------------------------------
# Model Definition
# ---------------------------------------------------------------------------

class MultiTaskAttentiveFP(nn.Module):
    """
    Multi-task AttentiveFP classifier for 12 Tox21 endpoints.

    Each task has its own binary classification head; the backbone is shared
    so the model can exploit cross-task signal (e.g. both NR-ER and NR-ER-LBD
    target estrogen receptor — correlated labels).

    Parameters
    ----------
    config : GNNConfig
    """

    def __init__(self, config: GNNConfig):
        super().__init__()
        self.config = config

        # Atom embedding projection
        self.atom_proj = nn.Sequential(
            nn.Linear(config.node_in_feats, config.hidden_channels),
            nn.LayerNorm(config.hidden_channels),
            nn.GELU(),
        )

        # AttentiveFP message-passing backbone (PyG implementation)
        self.gnn = AttentiveFP(
            in_channels=config.hidden_channels,
            hidden_channels=config.hidden_channels,
            out_channels=config.hidden_channels,
            edge_dim=config.edge_in_feats,
            num_layers=config.num_layers,
            num_timesteps=config.num_timesteps,
            dropout=config.dropout,
        )

        # Shared dense trunk
        self.trunk = nn.Sequential(
            nn.Linear(config.hidden_channels, config.hidden_channels // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_channels // 2, config.hidden_channels // 4),
            nn.GELU(),
        )

        # Per-task classification heads
        head_in = config.hidden_channels // 4
        self.task_heads = nn.ModuleList([
            nn.Linear(head_in, 1) for _ in range(config.n_tasks)
        ])

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x          : (N, node_in_feats) atom features
        edge_index : (2, E) bond connectivity
        edge_attr  : (E, edge_in_feats) bond features
        batch      : (N,) molecule assignment per atom

        Returns
        -------
        logits : (batch_size, n_tasks) — apply sigmoid for probabilities
        """
        x = self.atom_proj(x)
        x = self.gnn(x, edge_index, edge_attr, batch)  # → (batch_size, hidden)
        x = self.trunk(x)                               # → (batch_size, hidden//4)

        logits = torch.cat(
            [head(x) for head in self.task_heads], dim=1
        )  # → (batch_size, n_tasks)
        return logits

    def predict_proba(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> np.ndarray:
        """Returns probabilities ∈ [0,1] per task as numpy array."""
        self.eval()
        with torch.no_grad():
            logits = self(x, edge_index, edge_attr, batch)
            probs  = torch.sigmoid(logits)
        return probs.cpu().numpy()


# ---------------------------------------------------------------------------
# DeepChem-compatible GNN wrapper (uses dc.models.AttentiveFPModel)
# ---------------------------------------------------------------------------

class DeepChemGNNTrainer:
    """
    Thin wrapper around DeepChem's AttentiveFPModel for Tox21 multi-task
    classification.  Using DeepChem's model directly ensures compatibility
    with their Dataset, Splitter, and Transformer pipeline.

    The custom MultiTaskAttentiveFP above is used when SHAP / attention
    visualisations require direct PyTorch access.
    """

    def __init__(self, config: Optional[GNNConfig] = None):
        self.config = config or GNNConfig()
        self.model: Optional[dc.models.AttentiveFPModel] = None

    def build(self, n_tasks: int) -> "DeepChemGNNTrainer":
        """Instantiate the DeepChem model."""
        self.config.n_tasks = n_tasks
        self.model = dc.models.AttentiveFPModel(
            n_tasks=n_tasks,
            mode="classification",
            num_layers=self.config.num_layers,
            num_timesteps=self.config.num_timesteps,
            graph_feat_size=self.config.hidden_channels,
            dropout=self.config.dropout,
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            model_dir=str(self.config.checkpoint_dir),
        )
        logger.info("AttentiveFP model built — %d tasks", n_tasks)
        return self

    def train(
        self,
        train_dataset: dc.data.Dataset,
        valid_dataset: dc.data.Dataset,
        transformers: List,
        n_epochs: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Train with early stopping on validation AUC.

        Returns
        -------
        Dict mapping task name → best validation AUC.
        """
        if self.model is None:
            raise RuntimeError("Call .build() before .train()")

        epochs   = n_epochs or self.config.n_epochs
        patience = self.config.patience
        best_auc = 0.0
        no_improve = 0

        metric = dc.metrics.Metric(
            dc.metrics.roc_auc_score,
            np.mean,
            mode="classification",
        )

        logger.info("Starting training — max epochs=%d | patience=%d", epochs, patience)

        for epoch in range(1, epochs + 1):
            self.model.fit(train_dataset, nb_epoch=1, deterministic=False)
            val_scores = self.model.evaluate(valid_dataset, [metric], transformers)
            val_auc    = val_scores["mean-roc_auc_score"]

            logger.info("Epoch %3d | val_AUC=%.4f", epoch, val_auc)

            if val_auc > best_auc + 1e-4:
                best_auc   = val_auc
                no_improve = 0
                self.model.save_checkpoint()
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        # Restore best checkpoint
        self.model.restore()
        return {"best_val_auc": best_auc}

    def evaluate(
        self,
        dataset: dc.data.Dataset,
        transformers: List,
    ) -> Dict[str, float]:
        """Return AUC, F1, and accuracy on the given dataset."""
        metrics = [
            dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean,   mode="classification"),
            dc.metrics.Metric(dc.metrics.f1_score,       np.mean,   mode="classification"),
            dc.metrics.Metric(dc.metrics.accuracy_score, np.mean,   mode="classification"),
        ]
        scores = self.model.evaluate(dataset, metrics, transformers)
        return {
            "auc":      scores.get("mean-roc_auc_score", 0.0),
            "f1":       scores.get("mean-f1_score",       0.0),
            "accuracy": scores.get("mean-accuracy_score", 0.0),
        }

    def predict(self, dataset: dc.data.Dataset) -> np.ndarray:
        """Return raw probabilities of shape (n_samples, n_tasks)."""
        if self.model is None:
            raise RuntimeError("Model not built or loaded.")
        return self.model.predict(dataset)
