# ADMET Toxicity Predictor
### *Two-Layer Validation: Graph Neural Networks + Chemistry Intuition*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![DeepChem](https://img.shields.io/badge/DeepChem-2.7+-green.svg)](https://deepchem.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-pytest-passing-brightgreen.svg)](#testing)

> **A graph neural network that predicts drug toxicity across 12 biological assays - then has its predictions interrogated by the developer's chemistry intuition.**

---

## What This Is

This project predicts **Absorption, Distribution, Metabolism, Excretion, and Toxicity (ADMET)** properties of drug candidate molecules. It is built on the **Tox21 dataset**, the industry-standard benchmark used by the FDA and NIH for computational toxicology, and implements an **AttentiveFP Graph Neural Network** via DeepChem.

But the model is half the story.

The distinguishing feature of this project is a **two-layer validation framework** that doesn't treat the model as a black box oracle. Layer 1 is standard ML evaluation. Layer 2 applies medicinal chemistry knowledge to audit the model's predictions, flagging cases where the model's statistical confidence conflicts with known chemical toxicophores.

**This project exists to demonstrate that the most dangerous prediction an AIDD model can make is a false negative on toxicity**, and that catching those requires domain knowledge, not just a better AUC score.

---

## The Two Layers

### Layer 1: Standard ML Validation

```
Tox21 Dataset (8,014 compounds, 12 assays)
    ↓
Scaffold Split  →  Train (80%) / Valid (10%) / Test (10%)
    ↓
Three featurisation strategies compared:
    • ECFP4 fingerprints (circular Morgan, 2048 bits)
    • MACCS keys (166 expert-curated structural keys)
    • Graph tensors (atom/bond features for GNN)
    ↓
Models trained:
    • AttentiveFP GNN (primary model)
    • Random Forest on ECFP (baseline)
    • XGBoost on ECFP (baseline)
    • Logistic Regression on ECFP (linear baseline)
    ↓
Evaluation: AUC-ROC, F1, balanced accuracy per task
```

Scaffold splitting is intentional: it tests generalisation to structurally **novel** scaffolds, not just interpolation within known chemical space. A model that scores well on scaffold splits is a model that can handle actual drug discovery, where new chemical series are always being explored.

### Layer 2: Chemistry Intuition Validation

After the model produces predictions, I manually reviewed cases where model confidence and structural chemistry disagreed. The validator implements rule-based screening using curated SMARTS patterns derived from:

- Brenk et al. (2008) structural alerts
- PAINS filters (Baell & Holloway, 2010), 480 pan-assay interference series
- FAF-Drugs4 reactive group library
- ICH M7 mutagenicity structural alerts

**Discrepancy categories surfaced:**

| Category | Meaning | Action |
|---|---|---|
| `false_safe` | Model says safe; chemistry flags a known toxicophore | **High priority review, do not advance** |
| `uncertain_toxic` | Model says toxic; no structural alerts | Investigate SHAP values; may be a false positive |
| `aligned_toxic` | Model and chemistry both flag toxic | High confidence; deprioritise |
| `aligned_safe` | Model says safe; no alerts | Standard follow-up assays |

---

## Chemistry Validation Examples

These are real cases drawn from running the validator on Tox21 test-set predictions. The format mirrors what a medicinal chemist would write in a compound review.

---

**Case 1 - False Safe: Aldehyde**

> The model assigned a toxicity probability of **0.09** to this benzaldehyde derivative for the NR-AhR assay, classifying it as non-toxic.
>
> However, **the aldehyde functional group (`[CX3H1](=O)`)** present in this structure is a reactive electrophile capable of forming covalent Schiff-base adducts with lysine residues and N-terminal amines of proteins. This off-target covalent reactivity can trigger immune-mediated toxicity through haptenisation — a mechanism the model's Tox21 training data likely underrepresents, as *in vitro* cell viability assays are poorly sensitive to delayed hypersensitivity.
>
> **Recommendation:** Do not advance based on the model's 'safe' classification. A GSH trapping assay and modified Ames test are required before deprioritisation.

---

**Case 2 - False Safe: Nitroaromatic**

> The model assigned a toxicity probability of **0.14** to this 4-substituted nitrobenzene derivative for the SR-p53 assay.
>
> The **nitroaromatic motif (`[c][N+](=O)[O-]`)** is a Category 1 structural alert under ICH M7 guidelines. Nitroarenes are bioactivated by intestinal and hepatic nitroreductases to reactive hydroxylamine intermediates and ultimately nitroso species, which alkylate DNA forming characteristic adducts. The SR-p53 assay is designed to detect p53 pathway activation from DNA damage — if this assay is returning low signal for a nitroarene, it may indicate cytotoxicity at the test concentration is masking the genotoxic response, or that the test cell line has atypical nitroreductase expression.
>
> **Recommendation:** Reject the 'safe' classification. Run Ames mutagenicity (TA98/TA100) and in vitro micronucleus assay per ICH M7 guidelines.

---

**Case 3 - Uncertain Toxic: No Structural Alerts**

> The model assigned a toxicity probability of **0.81** to aspirin (acetylsalicylic acid) for the NR-AR assay, despite this compound having no classical structural alerts.
>
> No reactive functional groups, PAINS patterns, or structural toxicophores were identified by the chemistry validator. Possible explanations for this prediction: (1) the model identified a non-obvious substructure correlated with NR-AR activity in training data — the carboxylic acid or ester moiety may overlap with training examples that were weakly androgenic; (2) the prediction reflects statistical noise near the scaffold boundary in the scaffold split.
>
> SHAP analysis for this compound identified ECFP bits corresponding to the **carboxylic acid environment** and the **ester group** as primary drivers of the positive prediction — not a toxicophore, but a pharmacophore that happens to correlate weakly with NR-AR in this dataset.
>
> **Recommendation:** Flag as likely false positive. Confirm with counter-screen using a structurally matched inactive compound. The SHAP-driven mechanistic explanation does not support this as a genuine androgen receptor disruptor.

---

**Case 4 - PAINS Interference: Rhodanine**

> The model assigned a toxicity probability of **0.72** to a rhodanine-containing compound for SR-ARE.
>
> This compound matches a **PAINS structural pattern (Baell & Holloway, 2010)**. The rhodanine scaffold is notorious for producing assay artefacts through three mechanisms: metal chelation (Zn²⁺ in SR-ARE transcription factors), thiol reactivity (Michael addition), and non-specific aggregation. The 'toxic' prediction may reflect assay interference, not genuine SR-ARE pathway activation.
>
> **Recommendation:** Discard this compound from the hit list regardless of ML prediction. Both 'toxic' and 'safe' ML classifications are unreliable for PAINS compounds because the compound likely interferes with the assay readout itself. Orthogonal biophysical assays (SPR, ITC) are required to establish genuine binding.

---

## Repository Structure

```
admet-toxicity-predictor/
│
├── src/
│   ├── data/
│   │   └── tox21_loader.py          # Tox21 loading, splitting, class imbalance stats
│   │
│   ├── features/
│   │   └── molecular_representations.py  # ECFP, MACCS, RDKit descriptors, graph tensors
│   │
│   ├── models/
│   │   ├── gnn_model.py             # AttentiveFP GNN (multi-task, DeepChem)
│   │   └── baseline_models.py       # RF, XGBoost, LR, NaiveBayes baselines
│   │
│   ├── explainability/
│   │   └── shap_analysis.py         # TreeSHAP + GNNExplainer, atom-level visualisation
│   │
│   ├── validation/
│   │   └── chemistry_validator.py   # Layer 2 — structural alert screening + reviewer notes
│   │
│   └── pipeline.py                  # End-to-end orchestration script
│
├── tests/
│   └── test_chemistry_validator.py  # pytest suite for the chemistry validation layer
│
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb    # Dataset statistics, class imbalance plots
│   └── 02_chemistry_validation.ipynb   # Interactive validation case walkthrough
│
├── results/                             # Auto-generated on pipeline run
│   ├── figures/                         # SHAP summary plots, ROC curves
│   └── reports/                         # CSVs: leaderboard, chemistry flags, SHAP bits
│
├── requirements.txt
└── README.md
```

---

## Technical Design Decisions

### Why AttentiveFP?

AttentiveFP (Xiong et al., 2020) uses graph attention mechanisms that assign importance weights to individual atoms and bonds during message passing. This gives two practical advantages for toxicity prediction:

1. **Motif sensitivity:** The attention heads learn to focus on toxicophoric substructures, the model can weight an epoxide ring differently from a saturated carbon even if they appear in similar molecular contexts.

2. **Explainability bridge:** Attention weights provide an initial qualitative explanation of which atoms drove the prediction, which then gets validated by the Layer 2 chemistry screen.

### Why SHAP over Model-Internal Attention?

Attention weights tell us which atoms were attended to. SHAP values tell us the counterfactual contribution of each feature to the output, answering *"how much would the prediction change if this atom environment were absent?"* These are complementary, not redundant. High attention + low SHAP = the atom was important for the computation but didn't directionally drive the toxicity prediction. High SHAP = directional importance.

For fingerprint baselines, TreeSHAP (exact, O(TLD) complexity) is used rather than permutation SHAP, making explanation tractable for RandomForest with 300 trees.

### Why Scaffold Split?

Random splits allow test molecules to share scaffolds (core ring systems) with training molecules, inflating apparent generalisation. Scaffold splits assign all molecules of a given scaffold exclusively to one partition. This is harder — and more realistic — because it tests whether the model has learned transferable chemical principles rather than memorised scaffold-specific patterns.

### Class Imbalance Handling

Tox21 is heavily imbalanced, some assays have < 5% positive labels. The pipeline handles this via:

- `class_weight="balanced"` in all sklearn baselines
- `scale_pos_weight` in XGBoost
- Balanced accuracy and AUC-ROC (not accuracy) as primary metrics
- Missing label masking (`w=0` entries excluded per-task)

---

## Molecular Representations Compared

| Representation | Dimension | Captures | Best For |
|---|---|---|---|
| **ECFP4** | 2048 bits (sparse) | Circular atom environments to radius 2 | Fast baselines, virtual screening |
| **MACCS Keys** | 166 bits | Expert-curated functional groups | Interpretability, pharmacophore matching |
| **RDKit Descriptors** | ~200 continuous | Lipinski properties, TPSA, Fsp3, etc. | ADMET rule-of-5 prediction |
| **Graph Tensors** | Variable (topology) | Full molecular graph, bond types, chirality | GNN input — best structural fidelity |

---

## Structural Alert Library

The validator screens for the following chemical liabilities:

| Alert | Severity | Mechanism |
|---|---|---|
| Aldehyde | 🔴 High | Schiff-base protein adducts; haptenisation |
| Epoxide | 🔴 High | DNA alkylation; CYP-bioactivated arene oxides |
| Michael acceptor (α,β-unsat. carbonyl) | 🔴 High | 1,4-thiol addition; GSH depletion |
| Acyl halide | 🔴 High | Rapid non-selective acylation of nucleophiles |
| Nitroaromatic | 🔴 High | Nitroreductase activation → hydroxylamine → DNA adducts |
| Quinone | 🔴 High | Redox cycling → ROS; Michael addition to thiols |
| Peroxide (O–O) | 🔴 High | Lipid peroxidation chain initiation |
| Diazo group | 🔴 High | Direct-acting alkylating agent |
| Thiophene | 🟡 Medium | CYP-bioactivated to reactive S-oxide/epoxide |
| Primary aniline | 🟡 Medium | CYP1A2/NAT N-hydroxylation → arylhydroxylamine |
| Catechol | 🟡 Medium | Autooxidation → ortho-quinone; metal chelation artefact |
| Rhodanine | 🟡 Medium | PAINS scaffold; assay interference |
| PAINS (480 patterns) | 🟡 Medium | Pan-assay interference — RDKit FilterCatalog |

---

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/admet-toxicity-predictor
cd admet-toxicity-predictor

# Create environment (conda recommended for RDKit)
conda create -n admet python=3.10
conda activate admet
conda install -c conda-forge rdkit

# Install dependencies
pip install -r requirements.txt
```

### Run the full pipeline

```bash
python src/pipeline.py \
    --featurizer GraphConv \
    --splitter scaffold \
    --epochs 50 \
    --output-dir results/
```

### Run only chemistry validation

```python
from src.validation.chemistry_validator import ChemistryValidator

validator = ChemistryValidator(toxicity_threshold=0.5)

# Single molecule
disc = validator.validate_molecule(
    smiles="O=CC1=CC=CC=C1",   # benzaldehyde
    model_prob=0.09,
    task="NR-AhR",
)
print(disc.reviewer_note)

# Batch
disc_list = validator.validate_batch(smiles_list, model_probs, task="NR-AhR")
report_df = validator.generate_report(disc_list)
print(report_df[["smiles", "discrepancy_type", "alert_names"]].to_string())
```

### Run tests

```bash
pytest tests/ -v --tb=short
```

---

## Results (Expected Ranges)

Results vary by run due to scaffold split stochasticity. Representative ranges from five independent runs:

| Model | Featurisation | Mean AUC (12 tasks) | Mean F1 |
|---|---|---|---|
| AttentiveFP GNN | Graph tensors | **0.82 – 0.86** | 0.61 – 0.68 |
| Random Forest | ECFP4 | 0.78 – 0.82 | 0.57 – 0.63 |
| XGBoost | ECFP4 | 0.79 – 0.83 | 0.58 – 0.65 |
| Logistic Regression | ECFP4 | 0.72 – 0.76 | 0.49 – 0.55 |
| Naive Bayes | ECFP4 | 0.66 – 0.71 | 0.44 – 0.50 |

*GNN improvement over RF baseline is most pronounced on tasks with complex, non-linear SAR: NR-AhR (+4-6% AUC) and SR-ARE (+3-5% AUC).  Tasks with simpler SAR (NR-AR-LBD) show smaller GNN advantage.*

---

## Key Limitations (Honest Assessment)

**1. In vitro ≠ In vivo.** Tox21 measures cell-based responses in immortalised human cell lines. It does not model pharmacokinetics, bioactivation in primary hepatocytes, or systemic multi-organ toxicity. A compound clearing the Tox21 panel is nowhere near "proven safe."

**2. Structural alerts are necessary, not sufficient.** The validator flags known liabilities but cannot enumerate all mechanisms of toxicity. A clean alert screen does not mean a compound is safe — it means it doesn't contain *known* flags.

**3. Scaffold split creates distribution shift.** The GNN generalises imperfectly to genuinely novel scaffolds. Performance on internal proprietary datasets will typically be lower than on Tox21 benchmark splits.

**4. SHAP explains the model, not the chemistry.** A high SHAP value for an ECFP bit tells you the model weighted that structural environment heavily. It does not confirm the structural environment is mechanistically relevant — that still requires a chemist.

---

## References

- Xiong Z. et al. (2020). Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph Attention Mechanism. *J. Med. Chem.*, 63(16), 8749-8760.
- Baell J. & Holloway G. (2010). New Substructure Filters for Removal of Pan Assay Interference Compounds (PAINS). *J. Med. Chem.*, 53(7), 2719-2740.
- Brenk R. et al. (2008). Lessons Learnt from Assembling Screening Libraries for Drug Discovery. *ChemMedChem*, 3(3), 435-444.
- Lundberg S. & Lee S.I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*, 30.
- Ramsundar B. et al. (2019). *Deep Learning for the Life Sciences*. O'Reilly Media. (DeepChem textbook)
- ICH M7(R1) Guideline (2017). Assessment and Control of DNA Reactive (Mutagenic) Impurities in Pharmaceuticals.

---

## License

MIT License — see [LICENSE](LICENSE).

---

*Built as a demonstration that drug discovery AI is most useful when it knows the limits of its own knowledge.*
