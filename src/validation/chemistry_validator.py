"""
Layer 2: Chemistry Intuition Validation
========================================
This module implements rule-based chemistry knowledge checks that run
*after* the GNN makes predictions — flagging discrepancies between
ML confidence and known medicinal-chemistry red flags.

Rationale
---------
ML models learn statistical correlations in training data.  Tox21 is an
*in vitro* panel — it does not capture every mechanism of in vivo toxicity.
Reactive electrophiles (e.g. aldehydes, epoxides) may score "safe" because
the assay cells were not exposed long enough, or because training data for
that motif is sparse.

This validator surfaces those cases explicitly so a human reviewer can
decide whether additional experiments are warranted.  It does NOT override
the model — it augments it with domain expertise.

Structural Alert Sources
------------------------
• Brenk et al. (2008) — SMARTS-based structural alerts
• PAINS filters (Baell & Holloway, 2010) — 480 pan-assay interference series
• FAF-Drugs4 reactive group library
• Gleeson et al. (2011) — metabolic soft spots
• Internal heuristics derived from bench experience

Each alert includes:
    smarts        : SMARTS pattern for substructure matching
    name          : Human-readable flag name
    severity      : 'high' | 'medium' | 'low'
    mechanism     : Mechanistic explanation (written for README excerpts)
    implication   : What the discrepancy means for compound prioritisation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, FilterCatalog, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalogParams

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Alert severity
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    HIGH   = "high"
    MEDIUM = "medium"
    LOW    = "low"


# ---------------------------------------------------------------------------
# Structural alert library
# ---------------------------------------------------------------------------

@dataclass
class StructuralAlert:
    name:        str
    smarts:      str
    severity:    Severity
    mechanism:   str       # mechanistic explanation
    implication: str       # what it means for the prediction


# Core library — selected for mechanistic clarity and literature support
STRUCTURAL_ALERTS: List[StructuralAlert] = [

    # ── Reactive Electrophiles ──────────────────────────────────────────────

    StructuralAlert(
        name="Aldehyde",
        smarts="[CX3H1](=O)",
        severity=Severity.HIGH,
        mechanism=(
            "Aldehydes are potent reactive electrophiles capable of forming "
            "covalent Schiff-base adducts with lysine residues and N-terminal "
            "amines of proteins.  This off-target covalent reactivity can trigger "
            "immune-mediated toxicity (haptenisation) and is a well-documented "
            "mechanism for idiosyncratic drug reactions."
        ),
        implication=(
            "If the model predicts low toxicity, the aldehyde motif may be "
            "underrepresented in Tox21 training data.  A GSH (glutathione) "
            "trapping assay and Ames test are recommended before deprioritising."
        ),
    ),

    StructuralAlert(
        name="Epoxide",
        smarts="[C;r3]1[O;r3][C;r3]1",
        severity=Severity.HIGH,
        mechanism=(
            "Epoxides are strained three-membered rings that react readily with "
            "nucleophilic sites on DNA (guanine N7) and proteins (cysteine, "
            "lysine, histidine).  Arene epoxides formed by CYP450 activation "
            "of aromatic rings are a canonical genotoxic mechanism — see "
            "benzo[a]pyrene-7,8-dihydrodiol-9,10-epoxide as the paradigm case."
        ),
        implication=(
            "A model 'safe' prediction for an epoxide-containing compound "
            "warrants Ames mutagenicity testing and CYP450 phenotyping.  The "
            "Tox21 assays are primarily cell-based and may not capture "
            "epoxide-mediated DNA alkylation efficiently."
        ),
    ),

    StructuralAlert(
        name="Michael Acceptor (α,β-unsaturated carbonyl)",
        smarts="[C,c]=[C,c]-[C,c](=[O,N,S])",
        severity=Severity.HIGH,
        mechanism=(
            "α,β-Unsaturated carbonyls (enones, acrylamides, maleimides) undergo "
            "1,4-addition (Michael reaction) with biological thiols — principally "
            "cysteine residues in catalytic sites and glutathione.  This "
            "electrophilic reactivity is the basis of FDA-approved covalent drugs "
            "(e.g. ibrutinib) but also responsible for off-target toxicity when "
            "reactivity is non-selective."
        ),
        implication=(
            "A 'safe' ML prediction should be supplemented with intrinsic "
            "reactivity data: GSH half-life assay, thiol reactivity index (TRI), "
            "and time-dependent inhibition (TDI) of CYP3A4."
        ),
    ),

    StructuralAlert(
        name="Acyl Halide",
        smarts="[C;!R](=O)[F,Cl,Br,I]",
        severity=Severity.HIGH,
        mechanism=(
            "Acyl halides are among the most reactive electrophilic functional "
            "groups — orders of magnitude more reactive than simple alkyl halides. "
            "They acylate amine, hydroxyl, and thiol nucleophiles rapidly, "
            "forming stable amide, ester, or thioester bonds with proteins "
            "and glutathione."
        ),
        implication=(
            "This motif is rarely present as such in final drug candidates; "
            "it is more commonly a synthetic intermediate.  Its presence in a "
            "screening compound likely indicates a false positive hit or "
            "unstable compound requiring re-synthesis."
        ),
    ),

    # ── Nitroaromatics & Quinones ───────────────────────────────────────────

    StructuralAlert(
        name="Nitroaromatic",
        smarts="[n,nH,N](-[!H])-[!H].[cH1]1[cH1][cH1][cH1][cH1][cH1]1",
        severity=Severity.HIGH,
        mechanism=(
            "Nitroarenes are bioactivated by intestinal and hepatic "
            "nitroreductases to reactive hydroxylamine intermediates and "
            "ultimately nitroso compounds.  These electrophilic species "
            "alkylate DNA and haemoglobin, forming characteristic DNA adducts "
            "detected in the Ames test and comet assay.  The para-nitroaniline "
            "structural class has produced numerous examples of genotoxic "
            "carcinogens in chronic bioassays."
        ),
        implication=(
            "Industry consensus (ICH M7) classifies nitroaromatics as "
            "structural alerts for mutagenicity.  Any ML 'safe' prediction "
            "for this class should be treated with significant scepticism."
        ),
    ),

    StructuralAlert(
        name="Nitroaromatic (simple)",
        smarts="[c][N+](=O)[O-]",
        severity=Severity.HIGH,
        mechanism="See Nitroaromatic entry — direct SMARTS match for nitro group on aromatic ring.",
        implication="Standard industry structural alert.  Requires mutagenicity confirmation.",
    ),

    StructuralAlert(
        name="Quinone",
        smarts="O=C1C=CC(=O)C=C1",
        severity=Severity.HIGH,
        mechanism=(
            "Quinones are redox-active compounds that enter futile redox cycles "
            "with cellular reductases (NQO1, CYP450 reductase), generating "
            "reactive oxygen species (ROS) — superoxide, hydrogen peroxide, "
            "and hydroxyl radical.  They also arylate thiols directly through "
            "Michael addition.  Doxorubicin-induced cardiomyopathy is the "
            "clinical paradigm for quinone toxicity."
        ),
        implication=(
            "The SR-ARE assay in Tox21 is sensitive to oxidative stress but "
            "may underestimate quinone toxicity if NQO1 expression in the "
            "test cell line is high (protective detoxification).  Additional "
            "cell lines expressing low NQO1 are advisable."
        ),
    ),

    # ── Metabolic Liabilities ───────────────────────────────────────────────

    StructuralAlert(
        name="Thiophene (CYP bioactivation risk)",
        smarts="c1ccsc1",
        severity=Severity.MEDIUM,
        mechanism=(
            "Thiophenes are oxidised by CYP450 to reactive thiophene-S-oxides "
            "and epoxides, which can react with hepatic proteins.  "
            "Tienilic acid (withdrawn: immune hepatitis) and sudoxicam "
            "(withdrawn: hepatotoxicity) are canonical examples.  "
            "The risk is highly dependent on the substitution pattern "
            "and the specific CYP isoform responsible."
        ),
        implication=(
            "A 'safe' Tox21 prediction does not rule out metabolic activation "
            "risk.  CYP450 phenotyping and covalent binding studies "
            "(radiolabelled protein binding assay) are appropriate follow-ups."
        ),
    ),

    StructuralAlert(
        name="Aniline (N-hydroxylation risk)",
        smarts="[NH2][c]",
        severity=Severity.MEDIUM,
        mechanism=(
            "Primary anilines undergo N-hydroxylation by CYP1A2 and NAT enzymes "
            "to form arylhydroxylamines — electrophiles that form adducts with "
            "DNA and haemoglobin.  The canonical carcinogen 4-aminobiphenyl and "
            "numerous occupational carcinogens (benzidine-based dyes) carry this "
            "structural feature.  Dapsone's adverse haemolytic toxicity also "
            "proceeds via N-hydroxylation."
        ),
        implication=(
            "CYP1A2 phenotyping and haemoglobin adduct studies are recommended "
            "before clinical development of primary aniline-containing compounds."
        ),
    ),

    # ── PAINS / Assay Interference ─────────────────────────────────────────

    StructuralAlert(
        name="Catechol (oxidation / chelation artefact)",
        smarts="[OH]c1ccccc1[OH]",
        severity=Severity.MEDIUM,
        mechanism=(
            "Catechols (ortho-dihydroxybenzenes) autooxidise in cell culture "
            "medium and assay buffers, generating ortho-quinones and H₂O₂ "
            "as byproducts.  This oxidative byproduct, not the parent compound, "
            "may be responsible for observed biological activity.  Catechols "
            "also form stable chelates with metal ions (Fe²⁺, Zn²⁺) present "
            "in enzyme active sites, creating assay artefacts."
        ),
        implication=(
            "A 'toxic' Tox21 prediction for a catechol may reflect assay "
            "interference rather than intrinsic toxicity.  Confirm with "
            "aerobic vs. anaerobic conditions and metal-chelation controls."
        ),
    ),

    StructuralAlert(
        name="Rhodanine (PAINS scaffold)",
        smarts="O=C1NC(=S)SC1",
        severity=Severity.MEDIUM,
        mechanism=(
            "Rhodanines are a notorious pan-assay interference (PAINS) scaffold.  "
            "They chelate metals, react with protein thiols, and undergo Michael "
            "addition.  The literature contains many examples of rhodanine 'hits' "
            "that failed to confirm in orthogonal assays — a disproportionate "
            "fraction of published rhodanine bioactivity is likely artifactual."
        ),
        implication=(
            "Rhodanine-containing compounds should be deprioritised regardless "
            "of ML prediction; even a 'toxic' prediction may be meaningless "
            "because the compound likely interferes with the assay readout itself."
        ),
    ),

    # ── Unstable Bonds ─────────────────────────────────────────────────────

    StructuralAlert(
        name="Peroxide Bond (O–O)",
        smarts="[OX2][OX2]",
        severity=Severity.HIGH,
        mechanism=(
            "Organic peroxides are thermally and photochemically labile.  "
            "Homolytic cleavage produces alkoxy radicals that initiate lipid "
            "peroxidation chains, damage mitochondrial membranes, and trigger "
            "apoptosis.  Endoperoxides like artemisinins are deliberately "
            "exploited for anti-malarial activity but are toxic at higher doses."
        ),
        implication="Peroxide-containing compounds require stability studies before biological assessment.",
    ),

    StructuralAlert(
        name="Diazo Group",
        smarts="[N;X1]=[N;X2]",
        severity=Severity.HIGH,
        mechanism=(
            "Diazonium and diazo compounds are highly electrophilic and form "
            "covalent bonds with nucleophilic sites in DNA, constituting direct "
            "alkylating agents.  Streptomycin analogue streptozotocin (a "
            "nitrosamide/diazo species) is a direct-acting carcinogen used "
            "experimentally to induce diabetes via selective β-cell destruction."
        ),
        implication="This motif is a definitive structural alert for genotoxicity.",
    ),
]

# Build SMARTS pattern index for performance
_COMPILED_ALERTS = [
    (alert, Chem.MolFromSmarts(alert.smarts))
    for alert in STRUCTURAL_ALERTS
    if Chem.MolFromSmarts(alert.smarts) is not None
]


# ---------------------------------------------------------------------------
# Prediction discrepancy dataclass
# ---------------------------------------------------------------------------

@dataclass
class PredictionDiscrepancy:
    """Represents a case where chemistry flags conflict with ML prediction."""
    smiles:          str
    task:            str
    model_prob:      float              # raw model probability
    model_label:     str               # "TOXIC" | "SAFE"
    alerts_triggered: List[StructuralAlert]
    discrepancy_type: str              # "false_safe" | "false_toxic" | "aligned"
    max_severity:    Severity
    reviewer_note:   str               # auto-generated narrative


# ---------------------------------------------------------------------------
# Main validator
# ---------------------------------------------------------------------------

class ChemistryValidator:
    """
    Layer 2 post-hoc chemistry validation.

    Screens model predictions against structural alerts and PAINS filters,
    then categorises discrepancies and generates reviewer notes.

    Parameters
    ----------
    toxicity_threshold : float
        Probability above which a prediction is classified as toxic. Default 0.5.
    safe_concern_threshold : float
        When model_prob < this AND high-severity alert fires → "false_safe" flag.
        Set lower (e.g. 0.3) to be more aggressive.

    Usage
    -----
    >>> validator = ChemistryValidator()
    >>> discrepancies = validator.validate_batch(smiles_list, predictions, task_name="NR-AhR")
    >>> report = validator.generate_report(discrepancies)
    >>> print(report)
    """

    # RDKit PAINS catalog
    _pains_params = FilterCatalogParams()
    _pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    _pains_catalog = FilterCatalog.FilterCatalog(_pains_params)

    def __init__(
        self,
        toxicity_threshold: float = 0.5,
        safe_concern_threshold: float = 0.35,
    ):
        self.tox_thresh  = toxicity_threshold
        self.safe_thresh = safe_concern_threshold

    # ------------------------------------------------------------------

    def screen_molecule(self, smiles: str) -> List[StructuralAlert]:
        """
        Run all structural alert checks on a single SMILES string.

        Returns the list of triggered alerts, sorted by severity.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning("Could not parse SMILES: %s", smiles)
            return []

        triggered = []

        # Custom SMARTS alerts
        for alert, pattern in _COMPILED_ALERTS:
            if mol.HasSubstructMatch(pattern):
                triggered.append(alert)

        # RDKit PAINS catalog
        if self._pains_catalog.HasMatch(mol):
            entry = self._pains_catalog.GetFirstMatch(mol)
            triggered.append(StructuralAlert(
                name=f"PAINS: {entry.GetDescription()}",
                smarts="",
                severity=Severity.MEDIUM,
                mechanism=(
                    "This compound matches a Pan-Assay INterference Compound (PAINS) "
                    "structural pattern.  PAINS compounds produce spurious activity "
                    "readouts in biochemical screens through non-specific mechanisms: "
                    "aggregation, redox cycling, fluorescence interference, or "
                    "metal chelation.  The biological activity they display rarely "
                    "translates to genuine on-target effects."
                ),
                implication=(
                    "Both 'toxic' and 'safe' ML predictions for PAINS compounds "
                    "may reflect assay interference rather than true toxicity. "
                    "Orthogonal assay formats (e.g. bioluminescence instead of "
                    "fluorescence) and counter-screens are mandatory."
                ),
            ))

        # Sort: HIGH first
        severity_order = {Severity.HIGH: 0, Severity.MEDIUM: 1, Severity.LOW: 2}
        triggered.sort(key=lambda a: severity_order[a.severity])

        return triggered

    def validate_molecule(
        self,
        smiles: str,
        model_prob: float,
        task: str = "unspecified",
    ) -> PredictionDiscrepancy:
        """
        Compare model probability against chemistry screen for one molecule.
        """
        alerts   = self.screen_molecule(smiles)
        label    = "TOXIC" if model_prob >= self.tox_thresh else "SAFE"
        n_high   = sum(1 for a in alerts if a.severity == Severity.HIGH)
        max_sev  = alerts[0].severity if alerts else Severity.LOW

        # Classify discrepancy type
        if label == "SAFE" and n_high >= 1:
            disc_type = "false_safe"
        elif label == "TOXIC" and not alerts:
            disc_type = "uncertain_toxic"   # model flags; chemistry doesn't — investigate
        elif label == "TOXIC" and alerts:
            disc_type = "aligned_toxic"
        else:
            disc_type = "aligned_safe"

        note = self._generate_reviewer_note(smiles, model_prob, label, alerts, task, disc_type)

        return PredictionDiscrepancy(
            smiles=smiles,
            task=task,
            model_prob=round(model_prob, 4),
            model_label=label,
            alerts_triggered=alerts,
            discrepancy_type=disc_type,
            max_severity=max_sev,
            reviewer_note=note,
        )

    def validate_batch(
        self,
        smiles_list: List[str],
        model_probs: List[float],
        task: str = "unspecified",
    ) -> List[PredictionDiscrepancy]:
        """Validate a list of SMILES strings against model probabilities."""
        results = []
        for smiles, prob in zip(smiles_list, model_probs):
            results.append(self.validate_molecule(smiles, prob, task))
        return results

    def generate_report(
        self,
        discrepancies: List[PredictionDiscrepancy],
        include_aligned: bool = False,
    ) -> pd.DataFrame:
        """
        Tabulate discrepancies for human review.

        Parameters
        ----------
        include_aligned : bool
            If True, include cases where model and chemistry agree.
        """
        rows = []
        for d in discrepancies:
            if not include_aligned and d.discrepancy_type in ("aligned_safe",):
                continue
            rows.append({
                "smiles":            d.smiles,
                "task":              d.task,
                "model_prob":        d.model_prob,
                "model_label":       d.model_label,
                "discrepancy_type":  d.discrepancy_type,
                "max_severity":      d.max_severity.value,
                "n_alerts":          len(d.alerts_triggered),
                "alert_names":       " | ".join(a.name for a in d.alerts_triggered),
                "reviewer_note":     d.reviewer_note,
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            sev_order = {"high": 0, "medium": 1, "low": 2, "": 3}
            df["_sev_rank"] = df["max_severity"].map(sev_order).fillna(3)
            df = df.sort_values(["_sev_rank", "model_prob"]).drop(columns="_sev_rank")
        return df

    # ------------------------------------------------------------------
    # Reviewer note generation
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_reviewer_note(
        smiles: str,
        model_prob: float,
        label: str,
        alerts: List[StructuralAlert],
        task: str,
        disc_type: str,
    ) -> str:
        """
        Auto-generate a human-readable reviewer note.

        This text is suitable for direct inclusion in notebooks and READMEs —
        structured to mirror the style of a medicinal chemist's annotation.
        """
        if disc_type == "false_safe":
            top_alert = alerts[0]
            return (
                f"⚠️  CHEMISTRY FLAG — MODEL PREDICTS SAFE ({model_prob:.0%} probability), "
                f"CHEMISTRY RAISES CONCERN\n\n"
                f"Task: {task}\n"
                f"Alert triggered: {top_alert.name} (severity: {top_alert.severity.value.upper()})\n\n"
                f"Mechanism: {top_alert.mechanism}\n\n"
                f"Implication for this prediction: {top_alert.implication}\n\n"
                f"Recommendation: Do not rely on the model's 'safe' classification for this "
                f"compound without orthogonal experimental data.  The structural alert "
                f"indicates a toxicophore that Tox21 in vitro assays may not fully capture."
            )

        elif disc_type == "uncertain_toxic":
            return (
                f"🔍  CHEMISTRY QUERY — MODEL PREDICTS TOXIC ({model_prob:.0%} probability), "
                f"NO STRUCTURAL ALERTS DETECTED\n\n"
                f"Task: {task}\n"
                f"No classical structural alerts were identified in this molecule.\n\n"
                f"Possible explanations for the model's toxicity prediction:\n"
                f"  1. The model identified a non-obvious structural pattern "
                f"correlated with toxicity in training data.\n"
                f"  2. The compound may produce a reactive metabolite not "
                f"captured by intact-structure alerts.\n"
                f"  3. The prediction may be a false positive — particularly "
                f"likely if the compound is structurally novel relative to training data.\n\n"
                f"Recommendation: Review SHAP values for this compound to identify "
                f"which structural features drove the prediction, then assess "
                f"chemical plausibility."
            )

        elif disc_type == "aligned_toxic":
            names = ", ".join(a.name for a in alerts[:3])
            return (
                f"✅  CHEMISTRY ALIGNED — MODEL PREDICTS TOXIC ({model_prob:.0%}) "
                f"AND STRUCTURAL ALERTS CONFIRM\n\n"
                f"Task: {task}\n"
                f"Alerts: {names}\n"
                f"The model prediction is supported by known structural toxicophores.  "
                f"High confidence in toxicity classification."
            )

        else:  # aligned_safe
            return (
                f"✅  CHEMISTRY ALIGNED — MODEL PREDICTS SAFE ({model_prob:.0%}) "
                f"AND NO STRUCTURAL ALERTS DETECTED\n\n"
                f"Task: {task}\n"
                f"No structural liabilities identified.  "
                f"Standard follow-up assays recommended before advancing."
            )
