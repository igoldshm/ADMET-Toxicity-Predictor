"""
Tox21 Dataset Loader
====================
Handles loading, preprocessing, and splitting of the Tox21 dataset
using DeepChem's built-in loaders with additional data quality checks.

Tox21 contains 12 biological assays for ~8,000 compounds:
    Nuclear Receptor Signaling:
        NR-AR, NR-AR-LBD, NR-AhR, NR-Aromatase,
        NR-ER, NR-ER-LBD, NR-PPAR-gamma
    Stress Response Pathways:
        SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import deepchem as dc
import numpy as np
import pandas as pd
from deepchem.data import Dataset

logger = logging.getLogger(__name__)

# All 12 Tox21 assay targets
TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]

# Task biological context for human-readable reporting
TASK_DESCRIPTIONS = {
    "NR-AR":        "Androgen Receptor — hormonal disruption",
    "NR-AR-LBD":    "Androgen Receptor Ligand Binding Domain",
    "NR-AhR":       "Aryl Hydrocarbon Receptor — dioxin-response pathway",
    "NR-Aromatase": "Aromatase inhibition — estrogen biosynthesis",
    "NR-ER":        "Estrogen Receptor alpha — endocrine disruption",
    "NR-ER-LBD":    "Estrogen Receptor LBD",
    "NR-PPAR-gamma":"Peroxisome Proliferator-Activated Receptor gamma",
    "SR-ARE":       "Antioxidant Response Element — oxidative stress",
    "SR-ATAD5":     "DNA damage response / genotoxicity surrogate",
    "SR-HSE":       "Heat Shock Element — proteotoxic stress",
    "SR-MMP":       "Mitochondrial membrane potential — mitotoxicity",
    "SR-p53":       "p53 activation — DNA damage / genotoxicity",
}


@dataclass
class Tox21DataConfig:
    """Configuration for dataset loading and splitting."""
    tasks: List[str] = field(default_factory=lambda: TOX21_TASKS)
    featurizer_type: str = "ECFP"          # ECFP | GraphConv | MACCSKeys | RDKit
    splitter_type: str = "scaffold"        # scaffold | random | stratified
    frac_train: float = 0.8
    frac_valid: float = 0.1
    frac_test: float = 0.1
    seed: int = 42
    data_dir: Optional[Path] = None


@dataclass
class Tox21DataBundle:
    """Container returned after loading and splitting."""
    train: Dataset
    valid: Dataset
    test: Dataset
    transformers: List
    config: Tox21DataConfig
    class_imbalance: Dict[str, float] = field(default_factory=dict)

    @property
    def n_tasks(self) -> int:
        return len(self.config.tasks)

    @property
    def feature_dim(self) -> int:
        return self.train.X.shape[1] if len(self.train.X.shape) > 1 else 0


class Tox21Loader:
    """
    Loads the Tox21 benchmark dataset through DeepChem.

    Supports multiple featurization strategies so downstream experiments
    can compare ECFP fingerprints, graph convolution features, and MACCS keys
    on identical train/valid/test splits.

    Usage
    -----
    >>> loader = Tox21Loader(Tox21DataConfig(featurizer_type="ECFP"))
    >>> bundle = loader.load()
    >>> print(f"Train size: {len(bundle.train)}")
    """

    FEATURIZER_MAP = {
        "ECFP":       dc.feat.CircularFingerprint(size=2048, radius=2),
        "MACCS":      dc.feat.MACCSKeysFingerprint(),
        "RDKit":      dc.feat.RDKitDescriptors(),
        "GraphConv":  dc.feat.MolGraphConvFeaturizer(use_edges=True),
        "AttentiveFP": dc.feat.MolGraphConvFeaturizer(use_edges=True, use_chirality=True),
    }

    SPLITTER_MAP = {
        "scaffold":   dc.splits.ScaffoldSplitter(),
        "random":     dc.splits.RandomSplitter(),
        "stratified": dc.splits.RandomStratifiedSplitter(),
        "fingerprint": dc.splits.FingerprintSplitter(),
    }

    def __init__(self, config: Optional[Tox21DataConfig] = None):
        self.config = config or Tox21DataConfig()
        self._validate_config()

    def _validate_config(self) -> None:
        if self.config.featurizer_type not in self.FEATURIZER_MAP:
            raise ValueError(
                f"Unknown featurizer '{self.config.featurizer_type}'. "
                f"Choose from: {list(self.FEATURIZER_MAP.keys())}"
            )
        total = self.config.frac_train + self.config.frac_valid + self.config.frac_test
        if not abs(total - 1.0) < 1e-6:
            raise ValueError(f"Split fractions must sum to 1.0, got {total:.3f}")

    def load(self) -> Tox21DataBundle:
        """
        Download (if needed) and split the Tox21 dataset.

        Returns
        -------
        Tox21DataBundle
            Populated with train/valid/test splits and class imbalance stats.
        """
        logger.info(
            "Loading Tox21 | featurizer=%s | splitter=%s",
            self.config.featurizer_type,
            self.config.splitter_type,
        )

        featurizer = self.FEATURIZER_MAP[self.config.featurizer_type]
        splitter   = self.SPLITTER_MAP[self.config.splitter_type]

        # DeepChem handles caching automatically via data_dir
        tasks, datasets, transformers = dc.molnet.load_tox21(
            featurizer=featurizer,
            splitter=splitter,
            reload=True,
            data_dir=str(self.config.data_dir) if self.config.data_dir else None,
            save_dir=str(self.config.data_dir) if self.config.data_dir else None,
        )

        train, valid, test = datasets

        logger.info(
            "Split sizes — train: %d | valid: %d | test: %d",
            len(train), len(valid), len(test),
        )

        imbalance = self._compute_class_imbalance(train, tasks)

        return Tox21DataBundle(
            train=train,
            valid=valid,
            test=test,
            transformers=transformers,
            config=self.config,
            class_imbalance=imbalance,
        )

    @staticmethod
    def _compute_class_imbalance(
        dataset: Dataset,
        tasks: List[str],
    ) -> Dict[str, float]:
        """
        Compute positive-class ratio per task.

        Heavy imbalance (< 5 % positives) is common in Tox21 and mandates
        balanced accuracy / AUC-ROC as primary metrics rather than accuracy.
        """
        y = dataset.y  # shape: (n_samples, n_tasks)
        w = dataset.w  # sample weights; 0 = missing label

        ratios: Dict[str, float] = {}
        for i, task in enumerate(tasks):
            mask    = w[:, i] != 0
            labels  = y[mask, i]
            n_pos   = labels.sum()
            n_total = mask.sum()
            ratio   = float(n_pos / n_total) if n_total > 0 else 0.0
            ratios[task] = ratio

            if ratio < 0.05:
                logger.warning(
                    "Task '%s': severe class imbalance — %.1f%% positives",
                    task, ratio * 100,
                )
        return ratios

    def get_smiles(self, split: str = "train") -> List[str]:
        """
        Convenience method to retrieve raw SMILES strings from a split.
        Useful for chemistry validation and visualisation.
        """
        bundle = self.load()
        dataset_map = {"train": bundle.train, "valid": bundle.valid, "test": bundle.test}
        if split not in dataset_map:
            raise ValueError(f"split must be one of {list(dataset_map.keys())}")
        return dataset_map[split].ids.tolist()  # DeepChem stores SMILES in .ids

    def summary(self, bundle: Tox21DataBundle) -> pd.DataFrame:
        """Return a DataFrame summarising per-task statistics."""
        rows = []
        for task in bundle.config.tasks:
            rows.append({
                "task":        task,
                "description": TASK_DESCRIPTIONS.get(task, ""),
                "pos_ratio":   round(bundle.class_imbalance.get(task, 0.0), 4),
                "is_severely_imbalanced": bundle.class_imbalance.get(task, 1.0) < 0.05,
            })
        return pd.DataFrame(rows)
