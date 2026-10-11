
import csv
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# PATHS AND SETTINGS
# ============================================================

BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR.parent / "dataset" / "sequence_data.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_FILE = MODEL_DIR / "sequence_classifier.joblib"

FRAMES_PER_SAMPLE = 30
FEATURES_PER_FRAME = 21 * 3
EXPECTED_FEATURES = FRAMES_PER_SAMPLE * FEATURES_PER_FRAME


# ============================================================
# LOAD DATA
# ============================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

data = pd.read_csv(DATA_FILE)

if "label" not in data.columns:
    raise ValueError("Dataset must contain a 'label' column.")

if data.empty:
    raise ValueError("Dataset is empty. Record some sequences first.")

X = data.drop(columns=["label"])
y = data["label"].astype(str)

if X.shape[1] != EXPECTED_FEATURES:
    raise ValueError(
        f"Expected {EXPECTED_FEATURES} features per sequence, "
        f"but found {X.shape[1]}. Check the collector settings."
    )

if X.isnull().any().any():
    raise ValueError("Dataset contains missing coordinate values.")

if not np.isfinite(X.to_numpy(dtype=float)).all():
    raise ValueError("Dataset contains invalid numeric values.")

print("Dataset loaded successfully.")
print(f"Sequences: {len(data)}")
print(f"Features per sequence: {X.shape[1]}")
print("\nSamples per sign:")
print(y.value_counts())


# ============================================================
# CHECK CLASS COUNT
# ============================================================

class_counts = y.value_counts()

if len(class_counts) < 2:
    print(
        "\nOnly one sign label exists. A classifier cannot learn "
        "to distinguish signs with one class."
    )
    print(
        "Collect sequences for at least one additional, "
        "verified ISL sign before training."
    )
    raise SystemExit(0)

if class_counts.min() < 2:
    raise ValueError(
        "Each sign needs at least two samples for a stratified split."
    )


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# TRAIN MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

print("\nTraining sequence classifier...")
model.fit(X_train, y_train)


# ============================================================
# EVALUATE
# ============================================================

predictions = model.predict(X_test)

print("\nTest accuracy:", round(accuracy_score(y_test, predictions), 4))
print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(model, MODEL_FILE)

print(f"\nModel saved to: {MODEL_FILE}")
print("Training complete.")
