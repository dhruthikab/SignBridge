import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).parent

DATA_FILE = (
    BASE_DIR.parent
    / "dataset"
    / "landmark_data.csv"
)

MODEL_DIR = BASE_DIR / "models"

MODEL_FILE = MODEL_DIR / "sign_classifier.joblib"


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.2


# ============================================================
# CHECK DATASET
# ============================================================

if not DATA_FILE.exists():

    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_FILE}\n\n"
        "Collect some sign data first."
    )


print()
print("======================================")
print(" SignBridge - ML Training")
print("======================================")
print()

print(f"Dataset: {DATA_FILE}")


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

print(f"Total samples: {len(df)}")


# ============================================================
# CHECK LABELS
# ============================================================

if "label" not in df.columns:

    raise ValueError(
        "Dataset does not contain a 'label' column."
    )


labels = df["label"].unique()

print()
print("Signs found:")

for label in labels:
    count = (df["label"] == label).sum()
    print(f"  {label}: {count} samples")


if len(labels) < 2:

    raise ValueError(
        "\nAt least 2 different signs are required "
        "before training the classifier.\n\n"
        "Currently found: "
        + ", ".join(labels)
    )


# ============================================================
# EXTRACT LANDMARK FEATURES
# ============================================================

feature_columns = []

for i in range(21):

    feature_columns.append(f"x{i}")
    feature_columns.append(f"y{i}")
    feature_columns.append(f"z{i}")


missing_columns = [
    column
    for column in feature_columns
    if column not in df.columns
]


if missing_columns:

    raise ValueError(
        "Missing landmark columns:\n"
        + "\n".join(missing_columns)
    )


X_raw = df[feature_columns].values.astype(np.float32)

y = df["label"].values


# ============================================================
# NORMALIZE LANDMARKS
# ============================================================

def normalize_landmarks(samples):

    normalized_samples = []

    for sample in samples:

        # Convert:
        # [x0,y0,z0,x1,y1,z1,...]
        # into:
        # 21 x 3

        landmarks = sample.reshape(21, 3)

        # ----------------------------------------------------
        # Step 1: Move wrist to origin
        # ----------------------------------------------------

        wrist = landmarks[0].copy()

        landmarks = landmarks - wrist


        # ----------------------------------------------------
        # Step 2: Scale based on maximum distance
        # ----------------------------------------------------

        distances = np.linalg.norm(
            landmarks,
            axis=1
        )

        max_distance = np.max(distances)

        if max_distance > 0:

            landmarks = (
                landmarks / max_distance
            )


        # ----------------------------------------------------
        # Flatten back to 63 features
        # ----------------------------------------------------

        normalized_samples.append(
            landmarks.flatten()
        )


    return np.array(
        normalized_samples,
        dtype=np.float32
    )


X = normalize_landmarks(X_raw)


print()
print(f"Features per sample: {X.shape[1]}")


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)


print()
print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")


# ============================================================
# RANDOM FOREST CLASSIFIER
# ============================================================

print()
print("Training Random Forest...")


classifier = RandomForestClassifier(
    n_estimators=200,
    random_state=RANDOM_STATE,
    class_weight="balanced"
)


classifier.fit(
    X_train,
    y_train
)


print("Training complete!")


# ============================================================
# PREDICTIONS
# ============================================================

y_pred = classifier.predict(X_test)


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print()
print("======================================")
print(" MODEL RESULTS")
print("======================================")

print()
print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("Classification Report:")
print()

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("Confusion Matrix:")
print()

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


model_data = {
    "classifier": classifier,
    "labels": list(labels),
    "feature_count": X.shape[1]
}


joblib.dump(
    model_data,
    MODEL_FILE
)


print()
print("======================================")
print(" MODEL SAVED")
print("======================================")

print()
print(f"Saved to:")
print(MODEL_FILE)

print()
print("Training finished successfully!")