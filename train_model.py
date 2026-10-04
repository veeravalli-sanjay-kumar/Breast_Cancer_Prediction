import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# 1. LOAD DATASET
# ============================================================

data = pd.read_csv("breast-cancer.csv")

print("========================================")
print("BREAST CANCER DECISION TREE")
print("========================================")

print("\nDataset shape:", data.shape)


# ============================================================
# 2. SET CORRECT COLUMN NAMES
# ============================================================

data.columns = [
    "age",
    "menopause",
    "tumor-size",
    "inv-nodes",
    "node-caps",
    "deg-malig",
    "breast",
    "breast-quad",
    "irradiat",
    "class"
]

print("\nColumns:")
print(data.columns.tolist())


# ============================================================
# 3. REPLACE MISSING VALUES
# ============================================================

data = data.replace("?", np.nan)


# ============================================================
# 4. SEPARATE INPUT AND TARGET
# ============================================================

X = data.drop("class", axis=1)

y = data["class"]


# ============================================================
# 5. IMPORTANT:
#    ALL INPUT FEATURES ARE CATEGORICAL
# ============================================================

categorical_columns = [
    "age",
    "menopause",
    "tumor-size",
    "inv-nodes",
    "node-caps",
    "deg-malig",
    "breast",
    "breast-quad",
    "irradiat"
]

print("\nCategorical features:")
print(categorical_columns)


# ============================================================
# 6. CATEGORICAL PREPROCESSING
# ============================================================

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),

    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore")
    )
])


# ============================================================
# 7. PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer([
    (
        "categorical",
        categorical_pipeline,
        categorical_columns
    )
])


# ============================================================
# 8. DECISION TREE
# ============================================================

decision_tree = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=5,
    random_state=42
)


# ============================================================
# 9. COMPLETE PIPELINE
# ============================================================

model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),

    (
        "classifier",
        decision_tree
    )
])


# ============================================================
# 10. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 11. TRAIN
# ============================================================

print("\nTraining Decision Tree...")

model.fit(X_train, y_train)

print("Training completed!")


# ============================================================
# 12. PREDICTION
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 13. ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n========================================")
print("MODEL EVALUATION")
print("========================================")

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)


# ============================================================
# 14. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ============================================================
# 15. CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# 16. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    "model.pkl"
)

print("\n========================================")
print("MODEL SAVED")
print("========================================")

print("Saved as: model.pkl")