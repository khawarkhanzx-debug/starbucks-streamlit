import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def prepare(raw):
    df = raw.copy()
    df.columns = df.columns.str.strip()
    if df.columns.duplicated().any():
        raise ValueError("Duplicate column names found.")
    target = next((c for c in ["type", "Beverage_category", "Category"] if c in df), None)
    if target is None:
        raise ValueError("CSV needs a Beverage_category, type or Category column.")
    df = df.loc[df[target].notna()].copy()
    y = df[target].astype(str)
    X = pd.DataFrame(index=df.index)
    for col in df.columns:
        if col == target or col in ["Beverage", "Beverage_prep"]:
            continue
        values = pd.to_numeric(
            df[col].astype(str).str.replace("%", "", regex=False)
            .str.replace("g", "", regex=False).str.strip(), errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)
        if values.notna().any():
            X[col] = values
    if X.empty or X.shape[1] == 0:
        raise ValueError("No usable numerical features found.")
    if y.nunique() < 2 or y.value_counts().min() < 2:
        raise ValueError("At least two categories with two rows each are required.")
    return X, y, target


def train(raw):
    X, y, target = prepare(raw)
    n_test = max(int(np.ceil(len(y) * .2)), y.nunique())
    if len(y) - n_test < y.nunique():
        raise ValueError("Too few rows for a stratified train/test split.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=n_test, random_state=42, stratify=y
    )
    pipeline = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                             DecisionTreeClassifier(max_depth=5, random_state=42))
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    return pipeline, X, {
        "target": target, "accuracy": accuracy_score(y_test, pred),
        "report": pd.DataFrame(classification_report(y_test, pred, output_dict=True,
                                                     zero_division=0)).T,
        "confusion": confusion_matrix(y_test, pred, labels=pipeline.classes_),
        "test_rows": len(y_test),
    }
