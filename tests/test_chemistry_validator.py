"""
Tests for ChemistryValidator — Layer 2 validation.

These tests verify that known toxic motifs are correctly flagged
regardless of the ML model's prediction.  They serve as a regression
suite: if structural alert SMARTS are modified, these tests catch regressions.

Run:
    pytest tests/test_chemistry_validator.py -v
"""

import pytest
from src.validation.chemistry_validator import ChemistryValidator, Severity


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def validator():
    return ChemistryValidator(toxicity_threshold=0.5, safe_concern_threshold=0.3)


# Known toxic motif SMILES → expected alert name fragment
KNOWN_TOXIC_SMILES = [
    # Aldehyde
    ("O=CC1=CC=CC=C1",           "Aldehyde",      Severity.HIGH),
    # Epoxide
    ("C1CO1",                     "Epoxide",       Severity.HIGH),
    # Nitroaromatic
    ("c1ccc([N+](=O)[O-])cc1",    "Nitroaromatic", Severity.HIGH),
    # Michael acceptor (trans-4-phenyl-3-buten-2-one)
    ("O=C(/C=C/c1ccccc1)C",       "Michael",       Severity.HIGH),
    # Thiophene
    ("c1ccsc1",                    "Thiophene",     Severity.MEDIUM),
    # Aniline
    ("Nc1ccccc1",                  "Aniline",       Severity.MEDIUM),
]

CLEAN_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",   # aspirin
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",   # testosterone
    "OC(=O)Cc1ccccc1",         # phenylacetic acid
]


# ---------------------------------------------------------------------------
# Structural alert detection
# ---------------------------------------------------------------------------

class TestStructuralAlerts:

    @pytest.mark.parametrize("smiles,alert_fragment,severity", KNOWN_TOXIC_SMILES)
    def test_known_alert_detected(self, validator, smiles, alert_fragment, severity):
        alerts = validator.screen_molecule(smiles)
        alert_names = [a.name for a in alerts]
        matched = any(alert_fragment.lower() in n.lower() for n in alert_names)
        assert matched, (
            f"Expected alert containing '{alert_fragment}' for SMILES '{smiles}'. "
            f"Got: {alert_names}"
        )

    @pytest.mark.parametrize("smiles,alert_fragment,expected_sev", KNOWN_TOXIC_SMILES)
    def test_severity_correct(self, validator, smiles, alert_fragment, expected_sev):
        alerts = validator.screen_molecule(smiles)
        matched = [a for a in alerts if alert_fragment.lower() in a.name.lower()]
        if matched:
            assert matched[0].severity == expected_sev

    @pytest.mark.parametrize("smiles", CLEAN_SMILES)
    def test_clean_compounds_no_high_severity(self, validator, smiles):
        alerts = validator.screen_molecule(smiles)
        high_alerts = [a for a in alerts if a.severity == Severity.HIGH]
        assert len(high_alerts) == 0, (
            f"Unexpected HIGH severity alert for clean compound '{smiles}': "
            f"{[a.name for a in high_alerts]}"
        )

    def test_invalid_smiles_returns_empty(self, validator):
        alerts = validator.screen_molecule("NOT_A_SMILES!!!")
        assert alerts == []


# ---------------------------------------------------------------------------
# Discrepancy classification
# ---------------------------------------------------------------------------

class TestDiscrepancyClassification:

    def test_false_safe_flagged(self, validator):
        """Aldehyde compound predicted safe → should be 'false_safe'."""
        smiles    = "O=CC1=CC=CC=C1"   # benzaldehyde
        model_prob = 0.1                # model says safe
        disc = validator.validate_molecule(smiles, model_prob, task="NR-AhR")
        assert disc.discrepancy_type == "false_safe"
        assert disc.model_label == "SAFE"

    def test_aligned_toxic_classified(self, validator):
        """Nitroaromatic predicted toxic — chemistry agrees → 'aligned_toxic'."""
        smiles    = "c1ccc([N+](=O)[O-])cc1"
        model_prob = 0.85
        disc = validator.validate_molecule(smiles, model_prob, task="SR-p53")
        assert disc.discrepancy_type == "aligned_toxic"
        assert disc.model_label == "TOXIC"

    def test_uncertain_toxic_for_clean_compound(self, validator):
        """Clean compound predicted toxic → 'uncertain_toxic'."""
        smiles    = "CC(=O)Oc1ccccc1C(=O)O"   # aspirin
        model_prob = 0.75
        disc = validator.validate_molecule(smiles, model_prob, task="NR-AR")
        assert disc.discrepancy_type == "uncertain_toxic"

    def test_aligned_safe_for_clean_compound(self, validator):
        """Clean compound predicted safe → 'aligned_safe'."""
        smiles    = "OC(=O)Cc1ccccc1"   # phenylacetic acid
        model_prob = 0.15
        disc = validator.validate_molecule(smiles, model_prob, task="NR-AR")
        assert disc.discrepancy_type == "aligned_safe"


# ---------------------------------------------------------------------------
# Batch validation
# ---------------------------------------------------------------------------

class TestBatchValidation:

    def test_batch_returns_correct_length(self, validator):
        smiles_list = [s for s, _, _ in KNOWN_TOXIC_SMILES]
        probs       = [0.1] * len(smiles_list)   # all predicted safe
        disc_list   = validator.validate_batch(smiles_list, probs, task="test")
        assert len(disc_list) == len(smiles_list)

    def test_report_excludes_aligned_safe_by_default(self, validator):
        smiles_list = CLEAN_SMILES
        probs       = [0.1] * len(smiles_list)
        disc_list   = validator.validate_batch(smiles_list, probs)
        report      = validator.generate_report(disc_list, include_aligned=False)
        # aligned_safe should be excluded
        if not report.empty:
            assert "aligned_safe" not in report["discrepancy_type"].values

    def test_report_non_empty_for_toxic_motifs(self, validator):
        smiles_list = [s for s, _, _ in KNOWN_TOXIC_SMILES]
        probs       = [0.05] * len(smiles_list)   # all predicted safe → false_safe
        disc_list   = validator.validate_batch(smiles_list, probs)
        report      = validator.generate_report(disc_list)
        assert len(report) > 0


# ---------------------------------------------------------------------------
# Reviewer note quality
# ---------------------------------------------------------------------------

class TestReviewerNotes:

    def test_false_safe_note_contains_warning(self, validator):
        disc = validator.validate_molecule("O=CC1=CC=CC=C1", 0.1, task="NR-AhR")
        assert "CHEMISTRY FLAG" in disc.reviewer_note or "⚠️" in disc.reviewer_note

    def test_note_contains_mechanism(self, validator):
        disc = validator.validate_molecule("O=CC1=CC=CC=C1", 0.1, task="NR-AhR")
        assert len(disc.reviewer_note) > 100, "Reviewer note should be substantive"

    def test_aligned_toxic_note_positive(self, validator):
        disc = validator.validate_molecule(
            "c1ccc([N+](=O)[O-])cc1", 0.9, task="SR-p53"
        )
        assert "✅" in disc.reviewer_note or "ALIGNED" in disc.reviewer_note
