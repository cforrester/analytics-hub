# app/explainability.py

import streamlit as st
from sklearn.pipeline import Pipeline
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

@st.cache_resource
def get_explainer(_model, background):
    from shap import TreeExplainer
    return TreeExplainer(
        _model,
        data=background,
        feature_perturbation="interventional"
    )

@st.cache_data
def get_shap_values(_explainer, background):
    return _explainer(background)

def run():
    st.header("Model Explainability")

    # Retrieve model and training data
    model = st.session_state.get("model")
    X_train = st.session_state.get("model_X_train")
    if model is None or X_train is None:
        st.info("Train a model first in Predictive Modeling.")
        return

    # If Pipeline, extract final estimator and transform inputs
    if isinstance(model, Pipeline):
        final_estimator = model.steps[-1][1]
        preproc = Pipeline(model.steps[:-1])
        X_transformed = preproc.transform(X_train)
    else:
        final_estimator = model
        X_transformed = X_train

    # Sample background for SHAP
    sample_size = min(100, X_transformed.shape[0])
    if hasattr(X_transformed, "iloc"):
        background = X_transformed.sample(n=sample_size, random_state=42)
    else:
        background = pd.DataFrame(
            X_transformed,
            columns=list(X_train.columns),
        ).sample(n=sample_size, random_state=42)

    # Build explainer and compute SHAP values
    explainer = get_explainer(final_estimator, background)
    shap_values = get_shap_values(explainer, background)

    # Get numpy array of SHAP values
    raw = shap_values.values if hasattr(shap_values, "values") else np.array(shap_values)
    arr = np.array(raw)

    # Select correct slice: multiclass vs single-output
    if arr.ndim == 3:
        # arr shape: (samples, classes, features) → average over samples & classes
        importance = np.abs(arr).mean(axis=(0, 1))
    else:
        # arr shape: (samples, features)
        importance = np.abs(arr).mean(axis=0)

    # Build feature name list from background columns
    feature_names = background.columns.tolist()

    # Pair and sort
    pairs = list(zip(feature_names, importance.tolist()))
    pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)

    # Create DataFrame from pairs to ensure matching lengths
    imp_df = pd.DataFrame(pairs_sorted, columns=["feature", "importance"])

    # --- Global Feature Importance Plot ---
    st.subheader("Global Feature Importance")
    top_n = min(10, len(imp_df))
    top_df = imp_df.head(top_n)
    features = top_df["feature"].iloc[::-1].tolist()
    importances = top_df["importance"].iloc[::-1].tolist()
    y_pos = np.arange(len(features))

    fig, ax = plt.subplots(figsize=(8, len(features) * 0.4))
    ax.barh(y_pos, importances, edgecolor="k")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    st.pyplot(fig)

    # --- Local Explanation ---
    st.subheader("Local Explanation")
    idx = st.number_input(
        "Index of instance to explain",
        min_value=0,
        max_value=background.shape[0] - 1,
        value=0
    )
    if st.button("Explain Instance"):
        single = background.iloc[[idx]]
        single_shap = explainer(single)
        single_arr = np.array(
            single_shap.values if hasattr(single_shap, "values")
            else single_shap
        )
        if single_arr.ndim == 3:
            # shape (1, classes, features) → pick class 1
            vals = single_arr[0, 1, :]
        else:
            # shape (1, features)
            vals = single_arr[0]

        st.write("Features for this instance:")
        st.write(single)
        st.write("SHAP contribution values:")
        for feat, sv in zip(feature_names, vals):
            st.write(f"{feat}: {float(sv):.4f}")
