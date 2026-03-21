# ─────────────────────────────────────────────────────────────
# score.py
# ─────────────────────────────────────────────────────────────
# Weekly customer churn scoring script for the engagement-risk-ml
# project. Loads a trained model, builds snapshot features from
# order history, scores all active customers, and outputs a
# ranked CSV of churn probabilities for CRM/outreach targeting.
#
# Usage:
#   python src/score.py
#
# The script will prompt you to configure all parameters
# interactively before running.
# ─────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from datetime import datetime, timedelta


# ─────────────────────────────────────────────────────────────
# Helper: Prompt Utilities
# ─────────────────────────────────────────────────────────────

def prompt_int(message, default, min_val=1, max_val=365):
    """Prompt user for an integer with a default and validation."""
    while True:
        raw = input(f"{message} [default: {default}]: ").strip()
        if raw == "":
            return default
        try:
            val = int(raw)
            if min_val <= val <= max_val:
                return val
            print(f"  Please enter a value between {min_val} and {max_val}.")
        except ValueError:
            print("  Invalid input — please enter a whole number.")


def prompt_float(message, default, min_val=0.0, max_val=1.0):
    """Prompt user for a float with a default and validation."""
    while True:
        raw = input(f"{message} [default: {default}]: ").strip()
        if raw == "":
            return default
        try:
            val = float(raw)
            if min_val <= val <= max_val:
                return val
            print(f"  Please enter a value between {min_val} and {max_val}.")
        except ValueError:
            print("  Invalid input — please enter a decimal number.")


def prompt_choice(message, choices):
    """Prompt user to select from a numbered list of choices."""
    print(f"\n{message}")
    for i, (label, description) in enumerate(choices, 1):
        print(f"  [{i}] {label} — {description}")
    while True:
        raw = input("Enter choice number: ").strip()
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(choices):
                return idx
            print(f"  Please enter a number between 1 and {len(choices)}.")
        except ValueError:
            print("  Invalid input — please enter a number.")


# ─────────────────────────────────────────────────────────────
# Step 1 — Configuration
# ─────────────────────────────────────────────────────────────

def configure():
    """Interactively collect all scoring parameters from the user."""

    print("\n" + "=" * 60)
    print(" engagement-risk-ml — Weekly Churn Scoring Script")
    print("=" * 60)
    print("\nThis script scores all active customers by their predicted")
    print("churn risk and outputs a ranked CSV for CRM targeting.\n")
    print("Press Enter to accept defaults or type a new value.\n")

    # ── Model selection ───────────────────────────────────────
    model_idx = prompt_choice(
        "Select model:",
        [
            ("XGBoost",
             "Highest predictive performance — recommended for deployment"),
            ("Logistic Regression (L1)",
             "Interpretable baseline — preferred for regulated environments"),
        ]
    )
    model_paths = [
        Path("models/churn_model_xgb.pkl"),
        Path("models/churn_model_l1.pkl"),
    ]
    model_names = ["XGBoost (Tuned)", "Logistic Regression L1 (Tuned)"]
    model_path = model_paths[model_idx]
    model_name = model_names[model_idx]

    # ── Prediction horizon ────────────────────────────────────
    print("\n── Prediction Horizon ───────────────────────────────────")
    print("How many days forward should the model predict churn?")
    print("A customer is labeled 'at risk' if they make no purchase")
    print("within this window after the snapshot date.")
    horizon_days = prompt_int(
        "Prediction horizon (days)", default=30, min_val=1, max_val=180
    )

    # ── Recency threshold ─────────────────────────────────────
    print("\n── Recency Heuristic Threshold ──────────────────────────")
    print("The recency threshold defines the baseline heuristic:")
    print("customers with no purchase within this many days are")
    print("flagged as at-risk by the simple recency rule.")
    print("This is used for baseline comparison only — not for")
    print("the ML model's predictions.")
    recency_threshold = prompt_int(
        "Recency threshold (days)", default=30, min_val=1, max_val=365
    )

    # ── Feature windows ───────────────────────────────────────
    print("\n── Feature Windows ──────────────────────────────────────")
    print("Rolling window sizes (in days) used to compute behavioral")
    print("features. Must be 3 values separated by commas.")
    print("Example: 7,30,90 computes features over the last 7, 30,")
    print("and 90 days prior to the snapshot date.")
    while True:
        raw = input("Feature windows [default: 7,30,90]: ").strip()
        if raw == "":
            windows = (7, 30, 90)
            break
        try:
            windows = tuple(sorted(int(w.strip()) for w in raw.split(",")))
            if len(windows) == 3 and all(w >= 1 for w in windows):
                break
            print("  Please enter exactly 3 positive integers separated by commas.")
        except ValueError:
            print("  Invalid input — example format: 7,30,90")

    # ── Classification threshold ──────────────────────────────
    print("\n── Classification Threshold ─────────────────────────────")
    threshold_idx = prompt_choice(
        "Select business scenario (determines classification threshold):",
        [
            ("Balanced — Max F1 (t=0.21)",
             "Flags ~92% of customers — best when no strong cost asymmetry"),
            ("High Recall — Broad Outreach (t=0.33)",
             "Catches ~90% of churners — best for low-cost channels (email, SMS)"),
            ("High Precision — Targeted Outreach (t=0.73)",
             "85%+ hit rate — best for expensive interventions (calls, discounts)"),
            ("Custom threshold",
             "Enter your own value between 0.0 and 1.0"),
        ]
    )
    preset_thresholds = [0.21, 0.33, 0.73]
    if threshold_idx < 3:
        threshold = preset_thresholds[threshold_idx]
        print(f"  Using threshold: {threshold}")
    else:
        threshold = prompt_float(
            "Enter custom threshold", default=0.33,
            min_val=0.0, max_val=1.0
        )

    # ── Snapshot date ─────────────────────────────────────────
    print("\n── Snapshot Date ────────────────────────────────────────")
    print("The snapshot date is the 'as of' date for feature computation.")
    print("Features are computed from orders at or before this date.")
    print("Default is today's date.")
    while True:
        raw = input(
            f"Snapshot date (YYYY-MM-DD) [default: today "
            f"{datetime.today().strftime('%Y-%m-%d')}]: "
        ).strip()
        if raw == "":
            snapshot_date = pd.Timestamp(datetime.today().date())
            break
        try:
            snapshot_date = pd.Timestamp(raw)
            break
        except ValueError:
            print("  Invalid date format — please use YYYY-MM-DD.")

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" Configuration Summary")
    print("=" * 60)
    print(f"  Model:               {model_name}")
    print(f"  Snapshot date:       {snapshot_date.date()}")
    print(f"  Prediction horizon:  {horizon_days} days")
    print(f"  Recency threshold:   {recency_threshold} days")
    print(f"  Feature windows:     {windows[0]}d, {windows[1]}d, {windows[2]}d")
    print(f"  Classification threshold: {threshold}")

    confirm = input("\nProceed with these settings? (y/n) [default: y]: ").strip()
    if confirm.lower() == "n":
        print("Restarting configuration...\n")
        return configure()

    return {
        "model_path":        model_path,
        "model_name":        model_name,
        "snapshot_date":     snapshot_date,
        "horizon_days":      horizon_days,
        "recency_threshold": recency_threshold,
        "windows":           windows,
        "threshold":         threshold,
    }


# ─────────────────────────────────────────────────────────────
# Step 2 — Load Data
# ─────────────────────────────────────────────────────────────

def load_orders():
    """Load processed order table from parquet."""
    orders_path = Path("data/processed/orders.parquet")
    if not orders_path.exists():
        raise FileNotFoundError(
            f"Order data not found at {orders_path}.\n"
            "Please run notebooks 01 and 02 first to generate "
            "the processed data files."
        )
    orders = pd.read_parquet(orders_path)
    orders["order_ts"] = pd.to_datetime(orders["order_ts"])
    print(f"\n  Loaded {len(orders):,} orders for "
          f"{orders['Customer ID'].nunique():,} customers.")
    return orders


# ─────────────────────────────────────────────────────────────
# Step 3 — Build Features
# ─────────────────────────────────────────────────────────────

def build_features(orders, snapshot_date, windows=(7, 30, 90)):
    """
    Build leakage-safe rolling behavioral features for all active
    customers as of snapshot_date. Mirrors the feature engineering
    pipeline from notebook 02 exactly.
    """
    snap_orders = orders[orders["order_ts"] <= snapshot_date].copy()

    if snap_orders.empty:
        raise ValueError(
            f"No orders found at or before {snapshot_date.date()}. "
            "Check your snapshot date."
        )

    features = []

    for customer_id, hist in snap_orders.groupby("Customer ID"):
        feat = {
            "Customer ID":  customer_id,
            "snapshot_date": snapshot_date,
        }

        last_order_date = hist["order_ts"].max()
        feat["recency_days"] = (snapshot_date - last_order_date).days

        for w in windows:
            start = snapshot_date - pd.Timedelta(days=w)
            win = hist[hist["order_ts"] > start]
            feat[f"orders_{w}d"]  = len(win)
            feat[f"revenue_{w}d"] = win["total_revenue"].sum()
            feat[f"items_{w}d"]   = win["n_items"].sum()

        feat["lifetime_orders"]  = len(hist)
        feat["lifetime_revenue"] = hist["total_revenue"].sum()

        features.append(feat)

    feature_cols = (
        ["recency_days"] +
        [f"orders_{w}d"  for w in windows] +
        [f"revenue_{w}d" for w in windows] +
        [f"items_{w}d"   for w in windows] +
        ["lifetime_orders", "lifetime_revenue"]
    )

    df = pd.DataFrame(features)
    print(f"  Built features for {len(df):,} active customers.")
    return df, feature_cols


# ─────────────────────────────────────────────────────────────
# Step 4 — Score Customers
# ─────────────────────────────────────────────────────────────

def score_customers(df, feature_cols, model_path, model_name,
                    threshold, recency_threshold):
    """
    Load model, generate churn probabilities, apply threshold,
    and assign risk tiers.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}.\n"
            "Please run the modeling notebooks to generate "
            "and save the model artifacts."
        )

    print(f"\n  Loading model: {model_name}")
    model = joblib.load(model_path)

    X = df[feature_cols].copy()

    # Align features to the exact schema/order the model was trained on.
    # This ensures CustomerID is dropped and features are in the correct order for sklearn/XGBoost.
    X = X[model.feature_names_in_]

    proba = model.predict_proba(X)[:, 1]
    prediction = (proba >= threshold).astype(int)

    # Recency heuristic baseline for comparison
    recency_flag = (df["recency_days"] > recency_threshold).astype(int)

    # Risk tier assignment based on probability
    def assign_tier(p):
        if p >= 0.73:
            return "High"
        elif p >= 0.33:
            return "Medium"
        else:
            return "Low"

    results = df[["Customer ID", "snapshot_date", "recency_days"]].copy()
    results["churn_probability"]  = proba.round(4)
    results["churn_prediction"]   = prediction
    results["risk_tier"]          = [assign_tier(p) for p in proba]
    results["recency_flag"]       = recency_flag
    results["model_used"]         = model_name
    results["threshold_used"]     = threshold

    # Sort by churn probability descending — highest risk first
    results = results.sort_values(
        "churn_probability", ascending=False
    ).reset_index(drop=True)
    results.index += 1  # 1-based rank
    results.index.name = "rank"

    return results


# ─────────────────────────────────────────────────────────────
# Step 5 — Save Output
# ─────────────────────────────────────────────────────────────

def save_output(results, config):
    """
    Save scored customer list to a timestamped CSV in reports/.
    Each weekly run produces a uniquely named file so historical
    runs are preserved.
    """
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"churn_scores_{config['snapshot_date'].date()}_{timestamp}.csv"
    out_path = out_dir / filename

    results.to_csv(out_path)
    return out_path


# ─────────────────────────────────────────────────────────────
# Step 6 — Print Summary Report
# ─────────────────────────────────────────────────────────────

def print_summary(results, config, out_path):
    """Print a concise summary of scoring results to the terminal."""

    total      = len(results)
    n_flagged  = results["churn_prediction"].sum()
    n_high     = (results["risk_tier"] == "High").sum()
    n_medium   = (results["risk_tier"] == "Medium").sum()
    n_low      = (results["risk_tier"] == "Low").sum()
    n_recency  = results["recency_flag"].sum()

    print("\n" + "=" * 60)
    print(" Scoring Complete — Summary Report")
    print("=" * 60)
    print(f"  Snapshot date:       {config['snapshot_date'].date()}")
    print(f"  Model:               {config['model_name']}")
    print(f"  Threshold:           {config['threshold']}")
    print(f"  Horizon:             {config['horizon_days']} days")
    print(f"  Feature windows:     {config['windows']}")
    print(f"\n  Total customers scored:    {total:,}")
    print(f"  Flagged as at-risk:        {n_flagged:,} "
          f"({n_flagged/total:.1%})")
    print(f"\n  Risk Tier Breakdown:")
    print(f"    High  (p >= 0.73):       {n_high:,} "
          f"({n_high/total:.1%})")
    print(f"    Medium(p >= 0.33):       {n_medium:,} "
          f"({n_medium/total:.1%})")
    print(f"    Low   (p <  0.33):       {n_low:,} "
          f"({n_low/total:.1%})")
    print(f"\n  Recency heuristic flags:   {n_recency:,} "
          f"({n_recency/total:.1%}) "
          f"[>{config['recency_threshold']}d since last order]")
    print(f"\n  Output saved to: {out_path}")
    print("=" * 60)
    print("\n  Top 10 highest-risk customers:")
    print(results[[
        "Customer ID", "churn_probability",
        "risk_tier", "recency_days"
    ]].head(10).to_string())
    print("\n")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    # Step 1 — Configure
    config = configure()

    print("\n── Running Scoring Pipeline ─────────────────────────────")

    # Step 2 — Load data
    print("\n[1/4] Loading order data...")
    orders = load_orders()

    # Step 3 — Build features
    print("\n[2/4] Building snapshot features...")
    feature_df, feature_cols = build_features(
        orders,
        config["snapshot_date"],
        config["windows"]
    )

    # Step 4 — Score customers
    print("\n[3/4] Scoring customers...")
    results = score_customers(
        feature_df,
        feature_cols,
        config["model_path"],
        config["model_name"],
        config["threshold"],
        config["recency_threshold"]
    )

    # Step 5 — Save output
    print("\n[4/4] Saving output...")
    out_path = save_output(results, config)

    # Step 6 — Print summary
    print_summary(results, config, out_path)


if __name__ == "__main__":
    main()