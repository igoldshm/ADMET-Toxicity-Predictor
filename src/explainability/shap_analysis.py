"""
SHAP-Based Explainability
==========================
Explains GNN and fingerprint-based model predictions using SHAP
(SHapley Additive exPlanations).

For fingerprint models:
    Uses TreeExplainer (exact, fast) for RF/XGBoost.
    Uses KernelExplainer (model-agnostic, slower) for others.
    SHAP values map directly to individual ECFP bits → atom environments.

For GNN models:
    Uses GNNExplainer (DeepChem/PyG implementation) which masks
    edges and nodes to maximise prediction faithfulness.
    Produces per-atom importance scores — visualised on 2D structure.

Why SHAP over feature importance?
    TreeShap values satisfy three axioms (efficiency, symmetry, dummy) that
    raw Gini importance violates.  They're directional (positive = pushes
    toward toxicity) and additive, enabling faithful local explanations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ECFP bit → atom environment decoder
# ---------------------------------------------------------------------------

def decode_ecfp_bit(
    smiles: str,
    bit_idx: int,
    radius: int = 2,
    n_bits: int = 2048,
) -> Optional[List[int]]:
    """
    Return the atom indices responsible for setting ECFP bit `bit_idx`.

    Useful for mapping high-SHAP bits back to chemical substructures.
    """
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    bit_info: Dict[int, List[Tuple]] = {}
    AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=radius, nBits=n_bits, bitInfo=bit_info
    )
    if bit_idx not in bit_info:
        return None
    # Each entry: (center_atom_idx, radius_used)
    return [center for center, _ in bit_info[bit_idx]]


# ---------------------------------------------------------------------------
# Fingerprint model explainer
# ---------------------------------------------------------------------------

class FingerprintSHAPExplainer:
    """
    Computes SHAP values for tree-based (RF, XGB) fingerprint classifiers.

    Parameters
    ----------
    model : fitted sklearn/xgboost estimator
        Must have .predict_proba().
    background_data : np.ndarray
        Background dataset for KernelExplainer (use a subsample of training data).
    use_tree_explainer : bool
        True for RF/XGBoost (exact), False for others (KernelExplainer approximation).
    """

    def __init__(
        self,
        model,
        background_data: np.ndarray,
        use_tree_explainer: bool = True,
        task_idx: int = 0,
    ):
        self.model     = model
        self.task_idx  = task_idx

        if use_tree_explainer:
            self.explainer = shap.TreeExplainer(
                model,
                data=background_data,
                feature_perturbation="interventional",
                model_output="probability",
            )
        else:
            predict_fn = lambda x: model.predict_proba(x)[:, 1]
            self.explainer = shap.KernelExplainer(
                predict_fn,
                shap.kmeans(background_data, 50),
            )

    def explain(self, X: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values for molecules in X.

        Returns
        -------
        shap_values : np.ndarray of shape (n_samples, n_features)
            Positive values → push prediction toward toxic.
            Negative values → push prediction toward non-toxic.
        """
        vals = self.explainer.shap_values(X)
        # TreeExplainer returns list[array] for binary classification
        if isinstance(vals, list):
            vals = vals[1]   # positive class
        return vals

    def top_toxic_bits(
        self,
        shap_values: np.ndarray,
        n: int = 20,
    ) -> pd.DataFrame:
        """
        Return the top-n ECFP bits by mean absolute SHAP value across all molecules.

        High positive mean SHAP → structural environment consistently
        associated with toxicity prediction.
        """
        mean_shap = shap_values.mean(axis=0)
        abs_shap  = np.abs(shap_values).mean(axis=0)

        df = pd.DataFrame({
            "bit":       np.arange(len(mean_shap)),
            "mean_shap": mean_shap,
            "abs_shap":  abs_shap,
        })
        df = df.nlargest(n, "abs_shap").reset_index(drop=True)
        return df

    def plot_summary(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        max_display: int = 20,
        save_path: Optional[Path] = None,
    ) -> None:
        """Beeswarm summary plot — one dot per molecule per feature."""
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(
            shap_values, X,
            max_display=max_display,
            show=False,
        )
        plt.title("SHAP Feature Importance — ECFP bits → Toxicity", fontsize=13)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("SHAP summary plot saved to %s", save_path)
        plt.show()

    def plot_waterfall(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        mol_idx: int = 0,
        save_path: Optional[Path] = None,
    ) -> None:
        """Single-molecule waterfall explaining one prediction."""
        base_val = self.explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = base_val[1]

        exp = shap.Explanation(
            values=shap_values[mol_idx],
            base_values=base_val,
            data=X[mol_idx],
        )
        fig, ax = plt.subplots(figsize=(12, 5))
        shap.waterfall_plot(exp, max_display=15, show=False)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()


# ---------------------------------------------------------------------------
# Atom-level GNN explanation
# ---------------------------------------------------------------------------

class GNNAtomExplainer:
    """
    Wraps DeepChem's GNNExplainer to produce per-atom importance scores.
    Colours atoms by importance on a 2D structure depiction.

    Requires: deepchem >= 2.7, torch_geometric
    """

    def __init__(self, dc_model, task_idx: int = 0):
        self.dc_model  = dc_model
        self.task_idx  = task_idx

    def explain_molecule(
        self,
        smiles: str,
        save_path: Optional[Path] = None,
    ) -> Optional[np.ndarray]:
        """
        Run GNNExplainer on a single SMILES string.

        Returns atom importance scores (higher = more influential).
        Saves highlighted 2D depiction if save_path is provided.
        """
        try:
            from deepchem.models.torch_models.gnn_explainer import GNNExplainer

            explainer   = GNNExplainer(self.dc_model, num_hops=2)
            node_mask, edge_mask = explainer.explain_molecule(smiles)
            atom_importance = node_mask.cpu().numpy().flatten()

            if save_path:
                self._draw_highlighted(smiles, atom_importance, save_path)

            return atom_importance

        except ImportError:
            logger.warning("GNNExplainer not available — install deepchem[torch]")
            return None

    @staticmethod
    def _draw_highlighted(
        smiles: str,
        atom_importance: np.ndarray,
        save_path: Path,
    ) -> None:
        """Render molecule with atoms coloured by importance (blue→red)."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return

        norm = (atom_importance - atom_importance.min()) / (
            atom_importance.max() - atom_importance.min() + 1e-8
        )

        # Map importance → (R, G, B) using a warm colormap
        atom_colors: Dict[int, Tuple[float, float, float]] = {}
        for i, imp in enumerate(norm):
            atom_colors[i] = (imp, 0.2 * (1 - imp), 1.0 - imp)   # red for high importance

        drawer = rdMolDraw2D.MolDraw2DCairo(600, 400)
        drawer.drawOptions().addAtomIndices = False
        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer, mol,
            highlightAtoms=list(range(len(norm))),
            highlightAtomColors=atom_colors,
        )
        drawer.FinishDrawing()
        with open(save_path, "wb") as f:
            f.write(drawer.GetDrawingText())
        logger.info("Atom importance image saved to %s", save_path)
