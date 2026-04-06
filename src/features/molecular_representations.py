"""
Molecular Representations
=========================
Compares three complementary ways to encode chemical structure for ML:

    1. ECFP Fingerprints  — circular, atom-environment hashing (Morgan algorithm)
    2. MACCS Keys         — 166 expert-curated structural keys
    3. Graph Tensors      — atom/bond features for message-passing networks

Each representation captures different chemical information:
- Fingerprints are sparse, interpretable, fast
- Graph features preserve full topology, enable attention / pooling

Design note: keeping representations in one module makes ablation studies
(which repr → which AUC?) clean and reproducible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import CalcTPSA

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Atom-level features used in graph featurisation
# ---------------------------------------------------------------------------

ATOM_FEATURES = {
    "atomic_num":     list(range(1, 119)),   # H → Og
    "degree":         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "implicit_valence": [0, 1, 2, 3, 4, 5, 6],
    "formal_charge":  [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    "num_Hs":         [0, 1, 2, 3, 4],
    "hybridization":  [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
}

BOND_FEATURES = {
    "bond_type": [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    "is_conjugated": [False, True],
    "is_in_ring":    [False, True],
    "stereo": [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
    ],
}


@dataclass
class FingerprintConfig:
    ecfp_radius: int = 2
    ecfp_size: int = 2048
    fcfp: bool = False   # feature-based (pharmacophore) ECFP variant


class MolecularRepresentation:
    """
    Featurises molecules into vectors suitable for classical ML or GNNs.

    All methods accept a SMILES string and return numpy arrays so they are
    framework-agnostic (sklearn, PyTorch, TensorFlow).

    Parameters
    ----------
    fp_config : FingerprintConfig
        Controls ECFP radius and bit-vector size.
    """

    def __init__(self, fp_config: Optional[FingerprintConfig] = None):
        self.fp_config = fp_config or FingerprintConfig()

    # ------------------------------------------------------------------
    # 1. ECFP Fingerprints (Morgan Algorithm)
    # ------------------------------------------------------------------

    def ecfp(self, smiles: str) -> Optional[np.ndarray]:
        """
        Extended-Connectivity Fingerprint (ECFP).

        Iteratively encodes atomic environments up to `radius` bonds out,
        then folds into a bit-vector of length `size`.  The resulting dense
        binary vector is the industry standard for ligand-based virtual
        screening (LBVS).

        Key limitation: bit collisions lose structural information at high radius
        combined with small vector size.  ECFP4 (radius=2, 2048 bits) is the
        pragmatic default.
        """
        mol = self._parse(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=self.fp_config.ecfp_radius,
            nBits=self.fp_config.ecfp_size,
            useFeatures=self.fp_config.fcfp,
        )
        return np.array(fp)

    # ------------------------------------------------------------------
    # 2. MACCS Keys
    # ------------------------------------------------------------------

    def maccs_keys(self, smiles: str) -> Optional[np.ndarray]:
        """
        Molecular ACCess System (MACCS) keys — 166 expert-curated SMARTS queries.

        Each bit encodes the presence of a specific structural feature:
        e.g. bit 65 = nitro group, bit 163 = thiocarbonyl.

        Advantages: human-interpretable, compact, widely validated.
        Limitation: fixed vocabulary — cannot encode novel scaffolds.
        """
        mol = self._parse(smiles)
        if mol is None:
            return None
        fp = MACCSkeys.GenMACCSKeys(mol)
        return np.array(fp)

    # ------------------------------------------------------------------
    # 3. RDKit Physicochemical Descriptors (Lipinski + more)
    # ------------------------------------------------------------------

    def physicochemical(self, smiles: str) -> Optional[Dict[str, float]]:
        """
        Compute 12 physicochemical descriptors relevant to ADMET:

        MW      — Molecular Weight
        LogP    — Wildman-Crippen predicted lipophilicity
        HBD     — H-bond donors (Lipinski rule ≤ 5)
        HBA     — H-bond acceptors (Lipinski rule ≤ 10)
        TPSA    — Topological Polar Surface Area (absorption/BBB)
        RotBonds— Rotatable bonds (flexibility → oral bioavailability)
        ArRings — Aromatic rings (CYP450 metabolism liability)
        HeavyAt — Heavy atom count
        Fsp3    — Fraction sp3 carbons (3D character, lower clearance)
        NCharge — Net formal charge (affects permeability)
        MR      — Molar refractivity (polarizability)
        FracCSP3— Fraction C-sp3 (Pfizer escape-from-flatland metric)
        """
        mol = self._parse(smiles)
        if mol is None:
            return None

        return {
            "MW":       Descriptors.MolWt(mol),
            "LogP":     Descriptors.MolLogP(mol),
            "HBD":      rdMolDescriptors.CalcNumHBD(mol),
            "HBA":      rdMolDescriptors.CalcNumHBA(mol),
            "TPSA":     CalcTPSA(mol),
            "RotBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "ArRings":  rdMolDescriptors.CalcNumAromaticRings(mol),
            "HeavyAt":  mol.GetNumHeavyAtoms(),
            "Fsp3":     rdMolDescriptors.CalcFractionCSP3(mol),
            "NCharge":  Chem.rdmolops.GetFormalCharge(mol),
            "MR":       Descriptors.MolMR(mol),
        }

    # ------------------------------------------------------------------
    # 4. Graph Tensor Representation
    # ------------------------------------------------------------------

    def graph_features(self, smiles: str) -> Optional[Dict]:
        """
        Encode molecule as (node_features, edge_index, edge_features) tensors
        suitable for PyTorch Geometric / DGL message-passing networks.

        Node features: one-hot atom type + degree + hybridisation + ...
        Edge features: bond type + conjugation + ring membership + stereo
        """
        mol = self._parse(smiles)
        if mol is None:
            return None

        node_feats = np.array([
            self._atom_features(atom) for atom in mol.GetAtoms()
        ], dtype=np.float32)

        edge_index = []
        edge_feats = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            feat  = self._bond_features(bond)
            # undirected graph: add both directions
            edge_index += [[i, j], [j, i]]
            edge_feats  += [feat, feat]

        return {
            "node_features": node_feats,
            "edge_index":    np.array(edge_index, dtype=np.int64).T if edge_index else np.empty((2, 0), dtype=np.int64),
            "edge_features": np.array(edge_feats, dtype=np.float32) if edge_feats else np.empty((0, len(self._bond_features(None))), dtype=np.float32),
            "num_nodes":     mol.GetNumAtoms(),
            "smiles":        smiles,
        }

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def batch_ecfp(self, smiles_list: List[str]) -> np.ndarray:
        """Featurise a list of SMILES → 2-D ECFP matrix, padding NaN for failures."""
        vecs = [self.ecfp(s) for s in smiles_list]
        size = self.fp_config.ecfp_size
        out  = np.zeros((len(vecs), size), dtype=np.float32)
        for i, v in enumerate(vecs):
            if v is not None:
                out[i] = v
        return out

    def batch_physicochemical(self, smiles_list: List[str]) -> pd.DataFrame:
        """Return a tidy DataFrame of physicochemical descriptors."""
        rows = [self.physicochemical(s) or {} for s in smiles_list]
        return pd.DataFrame(rows, index=smiles_list)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse(smiles: str) -> Optional[Chem.Mol]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning("RDKit could not parse SMILES: %s", smiles)
        return mol

    @staticmethod
    def _one_hot(value, choices: list) -> List[int]:
        encoding = [int(value == c) for c in choices]
        encoding.append(int(value not in choices))  # "other" bucket
        return encoding

    def _atom_features(self, atom: Chem.Atom) -> List[float]:
        feats = []
        feats += self._one_hot(atom.GetAtomicNum(),     ATOM_FEATURES["atomic_num"])
        feats += self._one_hot(atom.GetDegree(),        ATOM_FEATURES["degree"])
        feats += self._one_hot(atom.GetImplicitValence(), ATOM_FEATURES["implicit_valence"])
        feats += self._one_hot(atom.GetFormalCharge(),  ATOM_FEATURES["formal_charge"])
        feats += self._one_hot(atom.GetTotalNumHs(),    ATOM_FEATURES["num_Hs"])
        feats += self._one_hot(atom.GetHybridization(), ATOM_FEATURES["hybridization"])
        feats.append(float(atom.GetIsAromatic()))
        feats.append(float(atom.IsInRing()))
        return feats

    def _bond_features(self, bond: Optional[Chem.Bond]) -> List[float]:
        if bond is None:
            # Return zero vector of correct length for empty edge case
            return [0.0] * (len(BOND_FEATURES["bond_type"]) + 1 +
                            len(BOND_FEATURES["stereo"]) + 1 + 2)
        feats  = self._one_hot(bond.GetBondType(),    BOND_FEATURES["bond_type"])
        feats += self._one_hot(bond.GetStereo(),      BOND_FEATURES["stereo"])
        feats.append(float(bond.GetIsConjugated()))
        feats.append(float(bond.IsInRing()))
        return feats
