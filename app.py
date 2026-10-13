from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from modeling import train

st.set_page_config(page_title="Starbucks EDA & Predictor", page_icon="☕", layout="wide")
st.title("☕ Starbucks EDA & Drink Category Predictor")
st.caption("Educational analysis of your dataset. Not an official Starbucks application.")
local = Path(__file__).parent / "data" / "starbucks.csv"
uploaded = None
if not local.exists():
    uploaded = st.sidebar.file_uploader("Upload starbucks.csv", type=["csv"])
if uploaded is None and not local.exists():
    st.info("Upload your original Starbucks menu CSV to explore it and train the decision tree.")
    st.stop()
try:
    df = pd.read_csv(uploaded if uploaded is not None else local)
    df.columns = df.columns.str.strip()
except Exception as exc:
    st.error(f"Cannot read CSV: {exc}")
    st.stop()

prediction, eda, performance = st.tabs(["Predict category", "Explore data", "Model results"])
with eda:
    a, b, c = st.columns(3)
    a.metric("Rows", len(df))
    b.metric("Columns", len(df.columns))
    c.metric("Duplicate rows", int(df.duplicated().sum()))
    st.dataframe(df, use_container_width=True)
    st.subheader("Missing values")
    st.dataframe(df.isna().sum().rename("Missing rows"))
    if not df.empty:
        st.subheader("Statistical summary")
        st.dataframe(df.describe(include="all").T)
    numeric = df.select_dtypes("number")
    if len(numeric.columns):
        selected = st.selectbox("Numerical feature", numeric.columns)
        fig, ax = plt.subplots()
        ax.hist(numeric[selected].dropna(), bins=20, edgecolor="white")
        ax.set_xlabel(selected)
        ax.set_ylabel("Number of rows")
        st.pyplot(fig)
        plt.close(fig)

@st.cache_resource
def fit(data):
    return train(data)

try:
    pipeline, X, results = fit(df)
except ValueError as exc:
    prediction.warning(str(exc))
    performance.warning("Model cannot be trained on this CSV.")
    st.stop()
with prediction:
    st.write("Enter every model feature. Defaults use dataset medians.")
    with st.form("prediction"):
        inputs = {}
        columns = st.columns(3)
        for i, col in enumerate(X.columns):
            with columns[i % 3]:
                inputs[col] = st.number_input(col, value=float(X[col].median()), key=col)
        submit = st.form_submit_button("Predict category", type="primary")
    if submit:
        values = pd.DataFrame([inputs], columns=X.columns)
        st.success(f"Predicted category: {pipeline.predict(values)[0]}")
        for col in X:
            if inputs[col] < X[col].min() or inputs[col] > X[col].max():
                st.warning(f"{col} is outside the dataset's observed range.")
with performance:
    st.metric("Held-out test accuracy", f"{results['accuracy']:.1%}")
    st.caption(f"Test rows: {results['test_rows']}. Stratified split, seed 42, tree depth 5. "
               "Median imputation is fitted on training rows only. Accuracy describes this "
               "dataset split and does not guarantee performance on new drinks.")
    st.dataframe(results["report"])
    st.subheader("Confusion matrix")
    st.dataframe(pd.DataFrame(results["confusion"], index=pipeline.classes_,
                             columns=pipeline.classes_))
    st.caption("Rows: actual category; columns: predicted category.")
    tree = pipeline.named_steps["decisiontreeclassifier"]
    st.subheader("Feature importance")
    st.bar_chart(pd.Series(tree.feature_importances_, index=X.columns).sort_values(ascending=False))
    with st.expander("View decision tree"):
        fig, ax = plt.subplots(figsize=(24, 12))
        plot_tree(tree, feature_names=list(X.columns), class_names=list(pipeline.classes_),
                  filled=True, rounded=True, fontsize=7, ax=ax)
        st.pyplot(fig)
        plt.close(fig)

