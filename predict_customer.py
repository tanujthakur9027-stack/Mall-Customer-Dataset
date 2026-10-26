import argparse
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "mall_customer_analysis"
MODEL_PATH = OUTPUT_DIR / "kmeans_model.joblib"
SCALER_PATH = OUTPUT_DIR / "scaler.joblib"
FEATURES = ["Annual Income (k$)", "Spending Score (1-100)"]

CLUSTER_NAMES = {
    0: "Cluster 0 (Middle-of-the-Road)",
    1: "Cluster 1 (Target VIP)",
    2: "Cluster 2 (Careless Youth)",
    3: "Cluster 3 (Frugal Saver)",
    4: "Cluster 4 (Budget-Conscious)"
}


def load_artifacts():
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        from mall_customer_analysis import main as run_train
        run_train()
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def predict_cluster(income: float, spending_score: float, verbose: bool = False):
    # Auto-normalize: if user typed 55000 instead of 55, convert to 55
    if income >= 1000:
        income = income / 1000.0

    model, scaler = load_artifacts()
    input_df = pd.DataFrame([[income, spending_score]], columns=FEATURES)
    scaled_input = scaler.transform(input_df)
    scaled_df = pd.DataFrame(scaled_input, columns=FEATURES)

    cluster_id = int(model.predict(scaled_df)[0])

    if not verbose:
        # Simple one-line clean output requested by user
        print(f"\n--> {CLUSTER_NAMES[cluster_id]}")
        return cluster_id

    # Verbose mode
    print(f"\nCluster: Cluster {cluster_id}")
    print(f"Persona: {CLUSTER_NAMES[cluster_id]}")
    return cluster_id


def main():
    parser = argparse.ArgumentParser(description="Mall Customer Cluster Predictor")
    parser.add_argument("args", nargs="*", type=float, help="Income and Spending Score (e.g. 55 65)")
    parser.add_argument("--income", type=float, default=None)
    parser.add_argument("--spending", type=float, default=None)
    parser.add_argument("--verbose", "-v", action="store_true", help="Show extra details")
    parsed = parser.parse_args()

    # Direct arguments: python predict_customer.py 55 65
    if len(parsed.args) >= 2:
        income = parsed.args[0]
        spending = parsed.args[1]
        predict_cluster(income, spending, parsed.verbose)
        return

    # Flag arguments: python predict_customer.py --income 55 --spending 65
    if parsed.income is not None and parsed.spending is not None:
        predict_cluster(parsed.income, parsed.spending, parsed.verbose)
        return

    # Interactive mode
    print("MALL CUSTOMER PREDICTOR")
    inc_input = input("Enter Annual Income (e.g. 55 or 55000): ").strip()
    spd_input = input("Enter Spending Score (1-100): ").strip()
    try:
        inc = float(inc_input)
        spd = float(spd_input)
        predict_cluster(inc, spd, parsed.verbose)
    except ValueError:
        print("Error: Please enter valid numbers.")


if __name__ == "__main__":
    main()
