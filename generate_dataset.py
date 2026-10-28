from pathlib import Path

import numpy as np
import pandas as pd


# Project directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATASET_PATH = DATA_DIR / "business_loan_data.csv"


def generate_dataset(n_samples=5000, random_state=42):
    """Generate synthetic business-loan data for project demonstration."""
    rng = np.random.default_rng(random_state)

    annual_revenue = rng.uniform(150_000, 5_000_000, n_samples).round(2)
    annual_profit = (
        annual_revenue * rng.uniform(-0.15, 0.35, n_samples)
    ).round(2)

    loan_amount = (
        annual_revenue * rng.uniform(0.05, 0.80, n_samples)
    ).round(2)

    credit_score = rng.integers(300, 851, n_samples)
    business_age_years = rng.integers(0, 31, n_samples)
    existing_debt = (
        annual_revenue * rng.uniform(0.0, 1.2, n_samples)
    ).round(2)

    late_payments = rng.poisson(1.5, n_samples)
    debt_to_revenue = existing_debt / np.maximum(annual_revenue, 1)
    loan_to_revenue = loan_amount / np.maximum(annual_revenue, 1)
    profit_margin = annual_profit / np.maximum(annual_revenue, 1)

    # Synthetic risk score used to generate demonstration labels.
    risk_score = (
        0.35 * (650 - credit_score) / 100
        + 0.25 * debt_to_revenue
        + 0.20 * loan_to_revenue
        + 0.15 * late_payments
        - 0.10 * business_age_years / 10
        - 0.15 * profit_margin
        + rng.normal(0, 0.25, n_samples)
    )

    # Convert risk scores to synthetic default outcomes.
    default_probability = 1 / (1 + np.exp(-risk_score))
    loan_default = rng.binomial(1, default_probability)

    dataset = pd.DataFrame(
        {
            "annual_revenue": annual_revenue,
            "annual_profit": annual_profit,
            "loan_amount": loan_amount,
            "credit_score": credit_score,
            "business_age_years": business_age_years,
            "existing_debt": existing_debt,
            "late_payments": late_payments,
            "loan_default": loan_default,
        }
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(DATASET_PATH, index=False)

    print("Synthetic business-loan dataset created successfully.")
    print(f"Dataset location: {DATASET_PATH}")
    print(f"Total records: {len(dataset):,}")
    print(f"Default rate: {dataset['loan_default'].mean():.2%}")
    print("\nFirst five records:")
    print(dataset.head())


if __name__ == "__main__":
    generate_dataset()