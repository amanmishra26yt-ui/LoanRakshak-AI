from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "business_loan_data.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "loan_default_model.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.pkl"


# --------------------------------------------------
# 2. Features and target
# --------------------------------------------------
FEATURES = [
    "annual_revenue",
    "annual_profit",
    "loan_amount",
    "credit_score",
    "business_age_years",
    "existing_debt",
    "late_payments",
]

TARGET = "loan_default"


def main():
    # Check the dataset
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Run 'python generate_dataset.py' first."
        )

    data = pd.read_csv(DATA_PATH)

    required_columns = FEATURES + [TARGET]
    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing columns: {missing_columns}"
        )

    X = data[FEATURES]
    y = data[TARGET].astype(int)

    if y.nunique() < 2:
        raise ValueError(
            "The target must contain both default and non-default examples."
        )

    # --------------------------------------------------
    # 3. Split data into training and testing sets
    # --------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # --------------------------------------------------
    # 4. Build and train the model
    # --------------------------------------------------
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=12,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print("Training LoanRakshak AI model...")
    model.fit(X_train, y_train)

    # --------------------------------------------------
    # 5. Evaluate the model
    # --------------------------------------------------
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)

    if y_test.nunique() == 2:
        roc_auc = roc_auc_score(y_test, probabilities)
    else:
        roc_auc = None

    print("\nMODEL EVALUATION")
    print("-" * 45)
    print(f"Training records: {len(X_train):,}")
    print(f"Testing records:  {len(X_test):,}")
    print(f"Accuracy:         {accuracy:.2%}")
    print(
        f"ROC-AUC:          {roc_auc:.4f}"
        if roc_auc is not None
        else "ROC-AUC:          Not available"
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["No Default", "Default"],
            zero_division=0,
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    # --------------------------------------------------
    # 6. Save the trained model and metadata
    # --------------------------------------------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    metadata = {
        "model_name": "Random Forest Classifier",
        "project_name": "LoanRakshak AI",
        "dataset_file": DATA_PATH.name,
        "rows": int(len(data)),
        "features": FEATURES,
        "target": TARGET,
        "accuracy": float(accuracy),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "default_rate": float(y.mean()),
        "synthetic_data": True,
    }

    joblib.dump(model, MODEL_PATH)
    joblib.dump(metadata, METADATA_PATH)

    print("\nMODEL FILES SAVED SUCCESSFULLY")
    print(f"Model:    {MODEL_PATH}")
    print(f"Metadata: {METADATA_PATH}")
    print("\nImportant: These are demonstration results from synthetic data.")


if __name__ == "__main__":
    main()