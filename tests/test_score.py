# ─────────────────────────────────────────────────────────────
# tests/test_score.py
# Unit tests for score.py feature engineering pipeline
# Run with: pytest tests/
# ─────────────────────────────────────────────────────────────

import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from score import build_features, load_cancellations


# ─────────────────────────────────────────────────────────────
# Fixtures — synthetic data shared across tests
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def sample_orders():
    """Minimal order table with two customers."""
    return pd.DataFrame({
        "Customer ID":  ["C001", "C001", "C001", "C002", "C002"],
        "Invoice":      ["INV001", "INV002", "INV003", "INV004", "INV005"],
        "order_ts":     pd.to_datetime([
            "2010-09-01", "2010-10-01", "2010-11-01",
            "2010-10-15", "2010-11-15",
        ]),
        "total_revenue": [100.0, 200.0, 150.0, 300.0, 250.0],
        "n_items":       [3, 5, 4, 6, 7],
    })


@pytest.fixture
def sample_cancellations():
    """Cancellations for C001 only — C002 has none."""
    return pd.DataFrame({
        "Customer ID": ["C001", "C001"],
        "Invoice":     ["CINV001", "CINV002"],
        "InvoiceDate": pd.to_datetime(["2010-10-05", "2010-11-10"]),
        "Quantity":    [-2, -3],
        "Price":       [10.0, 15.0],
    })


@pytest.fixture
def snapshot_date():
    return pd.Timestamp("2010-11-30")


# ─────────────────────────────────────────────────────────────
# Test 1 — Correct columns returned
# ─────────────────────────────────────────────────────────────

def test_build_features_returns_correct_columns(
    sample_orders, sample_cancellations, snapshot_date
):
    """build_features should return all 14 expected feature columns."""
    df, feature_cols = build_features(
        sample_orders, sample_cancellations, snapshot_date
    )

    expected_cols = [
        "recency_days",
        "orders_7d",  "revenue_7d",  "items_7d",
        "orders_30d", "revenue_30d", "items_30d",
        "orders_90d", "revenue_90d", "items_90d",
        "lifetime_orders", "lifetime_revenue",
        "cancel_count", "cancellation_rate",
    ]

    assert set(expected_cols).issubset(set(feature_cols)), (
        f"Missing columns: {set(expected_cols) - set(feature_cols)}"
    )
    assert len(feature_cols) == 14


# ─────────────────────────────────────────────────────────────
# Test 2 — Cancellation rate calculation
# ─────────────────────────────────────────────────────────────

def test_build_features_cancellation_rate_calculation(
    sample_orders, sample_cancellations, snapshot_date
):
    """cancellation_rate should equal cancel_count / lifetime_orders."""
    df, _ = build_features(
        sample_orders, sample_cancellations, snapshot_date
    )

    c001 = df[df["Customer ID"] == "C001"].iloc[0]

    assert c001["cancel_count"] == 2
    assert c001["lifetime_orders"] == 3
    assert abs(c001["cancellation_rate"] - (2 / 3)) < 1e-6


# ─────────────────────────────────────────────────────────────
# Test 3 — Leakage safety
# ─────────────────────────────────────────────────────────────

def test_build_features_leakage_safety(sample_orders):
    """Cancellations after snapshot_date must not be included."""
    snapshot = pd.Timestamp("2010-10-31")

    # One cancellation before snapshot, one after
    cancellations = pd.DataFrame({
        "Customer ID": ["C001", "C001"],
        "Invoice":     ["CINV001", "CINV002"],
        "InvoiceDate": pd.to_datetime(["2010-10-05", "2010-11-10"]),
        "Quantity":    [-2, -3],
        "Price":       [10.0, 15.0],
    })

    df, _ = build_features(sample_orders, cancellations, snapshot)

    c001 = df[df["Customer ID"] == "C001"].iloc[0]

    # Only the pre-snapshot cancellation should be counted
    assert c001["cancel_count"] == 1, (
        f"Expected 1 cancellation before snapshot, got {c001['cancel_count']}"
    )


# ─────────────────────────────────────────────────────────────
# Test 4 — Customers with no cancellations get zero not NaN
# ─────────────────────────────────────────────────────────────

def test_customers_with_no_cancellations_get_zero(
    sample_orders, sample_cancellations, snapshot_date
):
    """Customers with no cancellation history should get 0, not NaN."""
    df, _ = build_features(
        sample_orders, sample_cancellations, snapshot_date
    )

    c002 = df[df["Customer ID"] == "C002"].iloc[0]

    assert c002["cancel_count"] == 0
    assert c002["cancellation_rate"] == 0.0
    assert not np.isnan(c002["cancel_count"])
    assert not np.isnan(c002["cancellation_rate"])


# ─────────────────────────────────────────────────────────────
# Test 5 — load_cancellations raises on missing file
# ─────────────────────────────────────────────────────────────

def test_load_cancellations_raises_on_missing_file(tmp_path, monkeypatch):
    """load_cancellations should raise FileNotFoundError if parquet missing."""
    import score
    monkeypatch.setattr(
        score,
        "Path",
        lambda *args: tmp_path / "nonexistent.parquet"
    )

    with pytest.raises(FileNotFoundError):
        load_cancellations()