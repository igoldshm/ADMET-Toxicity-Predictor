"""
Main Pipeline — ADMET Toxicity Predictor
==========================================
Orchestrates the full pipeline:

    1. Load Tox21 with scaffold split
    2. Train GNN (AttentiveFP) + baselines (RF, XGB, LR, NB)
    3. Evaluate all models on held-out test set
    4. Generate SHAP explanations
    5. Run Layer 2 chemistry validation on discrepant predictions
    6. Export report artefacts

Run:
    python src/pipeline.py --featurizer GraphConv --epochs 50 --output-dir results/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ADMET Toxicity Predictor — full training & validation pipeline"
    )
    parser.add_argument(
        "--featurizer", default="GraphConv",
        choices=["GraphConv", "ECFP", "MACCS", "RDKit", "AttentiveFP"],
        help="Molecular featurisation for GNN training (default: GraphConv)",
    )
    parser.add_argument(
        "--splitter", default="scaffold",
        choices=["scaffold", "random", "stratified"],
        help="Train/valid/test splitting strategy (default: scaffold)",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Max GNN training epochs (default: 50)",
    )
    parser.add_argument(
        "--hidden", type=int, default=200,
        help="GNN hidden layer width (default: 200)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results"),
        help="Directory for all outputs (default: results/)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--skip-baselines", action="store_true",
        help="Skip baseline model training (faster iteration)",
    )
    parser.add_argument(
        "--n-shap-samples", type=int, default=100,
        help="Number of test-set molecules for SHAP analysis (default: 100)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:

    # ── 0. Setup ─────────────────────────────────────────────────────────

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figures").mkdir(exist_ok=True)
    (args.output_dir / "reports").mkdir(exist_ok=True)

    logger.info("=" * 70)
    logger.info("ADMET Toxicity Predictor — pipeline start")
    logger.info("=" * 70)

    # ── 1. Load data ─────────────────────────────────────────────────────

    from src.data.tox21_loader import Tox21DataConfig, Tox21Loader, TOX21_TASKS

    config = Tox21DataConfig(
        featurizer_type=args.featurizer,
        splitter_type=args.splitter,
        seed=args.seed,
        data_dir=args.output_dir / "data_cache",
    )
    loader = Tox21Loader(config)
    bundle = loader.load()

    summary_df = loader.summary(bundle)
    logger.info("\nDataset summary:\n%s", summary_df.to_string(index=False))
    summary_df.to_csv(args.output_dir / "reports" / "dataset_summary.csv", index=False)

    # ── 2. Train GNN ─────────────────────────────────────────────────────

    from src.models.gnn_model import DeepChemGNNTrainer, GNNConfig

    gnn_config = GNNConfig(
        hidden_channels=args.hidden,
        n_epochs=args.epochs,
        n_tasks=bundle.n_tasks,
        checkpoint_dir=args.output_dir / "checkpoints" / "gnn",
    )
    gnn_trainer = DeepChemGNNTrainer(gnn_config)
    gnn_trainer.build(n_tasks=bundle.n_tasks)

    logger.info("Training GNN ...")
    train_result = gnn_trainer.train(
        bundle.train, bundle.valid, bundle.transformers, n_epochs=args.epochs
    )
    logger.info("Best val AUC: %.4f", train_result["best_val_auc"])

    gnn_test_scores = gnn_trainer.evaluate(bundle.test, bundle.transformers)
    logger.info(
        "GNN test — AUC: %.4f | F1: %.4f | Acc: %.4f",
        gnn_test_scores["auc"], gnn_test_scores["f1"], gnn_test_scores["accuracy"]
    )

    # ── 3. Baseline models ────────────────────────────────────────────────

    baseline_leaderboard = None

    if not args.skip_baselines:
        from src.data.tox21_loader import Tox21DataConfig, Tox21Loader
        from src.models.baseline_models import BaselineEvaluator

        # Reload with ECFP for classical models
        ecfp_bundle = Tox21Loader(
            Tox21DataConfig(featurizer_type="ECFP", splitter_type=args.splitter, seed=args.seed)
        ).load()

        evaluator = BaselineEvaluator(tasks=TOX21_TASKS, seed=args.seed)
        baseline_results = evaluator.run(
            ecfp_bundle.train.X, ecfp_bundle.train.y, ecfp_bundle.train.w,
            ecfp_bundle.test.X,  ecfp_bundle.test.y,  ecfp_bundle.test.w,
        )
        baseline_leaderboard = evaluator.leaderboard(baseline_results)
        logger.info("\nBaseline leaderboard:\n%s", baseline_leaderboard.to_string())
        baseline_leaderboard.to_csv(
            args.output_dir / "reports" / "baseline_leaderboard.csv", index=False
        )

    # ── 4. SHAP explanations ──────────────────────────────────────────────

    # Note: full SHAP on GNN uses GNNExplainer; here we do fingerprint SHAP
    # as a reproducible complement that runs without GPU.
    logger.info("Generating SHAP explanations on ECFP baseline (RandomForest) ...")

    try:
        from sklearn.ensemble import RandomForestClassifier
        from src.explainability.shap_analysis import FingerprintSHAPExplainer

        ecfp_bundle = Tox21Loader(
            Tox21DataConfig(featurizer_type="ECFP", splitter_type=args.splitter, seed=args.seed)
        ).load()

        task_idx = 2   # NR-AhR — interesting & well-represented task
        task_name = TOX21_TASKS[task_idx]

        train_mask = ecfp_bundle.train.w[:, task_idx] != 0
        X_tr = ecfp_bundle.train.X[train_mask]
        y_tr = ecfp_bundle.train.y[train_mask, task_idx].astype(int)
        X_te = ecfp_bundle.test.X

        rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                    n_jobs=-1, random_state=args.seed)
        rf.fit(X_tr, y_tr)

        shap_explainer = FingerprintSHAPExplainer(
            rf, background_data=X_tr[:200], use_tree_explainer=True
        )
        n_shap   = min(args.n_shap_samples, len(X_te))
        shap_vals = shap_explainer.explain(X_te[:n_shap])

        top_bits = shap_explainer.top_toxic_bits(shap_vals, n=20)
        top_bits.to_csv(
            args.output_dir / "reports" / f"shap_top_bits_{task_name}.csv", index=False
        )

        shap_explainer.plot_summary(
            shap_vals, X_te[:n_shap],
            save_path=args.output_dir / "figures" / f"shap_summary_{task_name}.png"
        )
        logger.info("SHAP analysis complete for task: %s", task_name)

    except Exception as exc:
        logger.warning("SHAP analysis skipped: %s", exc)

    # ── 5. Chemistry validation (Layer 2) ─────────────────────────────────

    logger.info("Running Layer 2 chemistry validation ...")

    from src.validation.chemistry_validator import ChemistryValidator

    test_smiles   = bundle.test.ids.tolist()          # SMILES stored in .ids
    gnn_probs     = gnn_trainer.predict(bundle.test)  # shape: (n_test, n_tasks)

    validator     = ChemistryValidator(toxicity_threshold=0.5, safe_concern_threshold=0.3)
    all_discrepancies = []

    for task_idx_v, task_name_v in enumerate(TOX21_TASKS):
        task_probs = gnn_probs[:, task_idx_v].tolist()
        disc_list  = validator.validate_batch(test_smiles, task_probs, task=task_name_v)
        all_discrepancies.extend(disc_list)

    report_df = validator.generate_report(all_discrepancies, include_aligned=False)
    report_path = args.output_dir / "reports" / "chemistry_validation_report.csv"
    report_df.to_csv(report_path, index=False)

    n_false_safe     = (report_df["discrepancy_type"] == "false_safe").sum()
    n_uncertain_tox  = (report_df["discrepancy_type"] == "uncertain_toxic").sum()

    logger.info(
        "Chemistry validation complete — "
        "%d false-safe flags | %d uncertain-toxic flags",
        n_false_safe, n_uncertain_tox,
    )

    # ── 6. Final summary ──────────────────────────────────────────────────

    summary = {
        "gnn_test_auc":      gnn_test_scores["auc"],
        "gnn_test_f1":       gnn_test_scores["f1"],
        "gnn_test_accuracy": gnn_test_scores["accuracy"],
        "n_test_molecules":  len(test_smiles),
        "n_chemistry_flags_false_safe":    int(n_false_safe),
        "n_chemistry_flags_uncertain_tox": int(n_uncertain_tox),
    }

    if baseline_leaderboard is not None:
        best_baseline = baseline_leaderboard.iloc[0]
        summary["best_baseline_name"] = best_baseline["model"]
        summary["best_baseline_auc"]  = float(best_baseline["mean_AUC"])

    summary_path = args.output_dir / "pipeline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 70)
    logger.info("Pipeline complete.  Results in: %s", args.output_dir)
    logger.info("=" * 70)
    logger.info(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    run(args)
