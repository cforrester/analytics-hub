# app/causal.py

import networkx as nx
# Restore DoWhy’s missing d_separated hook
from networkx.algorithms.d_separation import is_d_separator
nx.algorithms.d_separated = is_d_separator

import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer
from dowhy import CausalModel

def run(df):
    st.header("Causal Inference Analysis")
    if df is None:
        st.info("Please upload a dataset to use this feature.")
        return

    # Work on a copy so original df stays intact
    df_proc = df.copy()
    cols = df_proc.columns.tolist()

    # Sidebar inputs
    treatment      = st.selectbox("Select Treatment (independent variable)", cols)
    outcome        = st.selectbox("Select Outcome (dependent variable)",
                                  [c for c in cols if c != treatment])
    common_causes = st.multiselect("Select potential confounders (common causes)",
                                   [c for c in cols if c not in [treatment, outcome]])

    # Auto-encode non-numeric columns to integer codes
    encode_cols = [treatment, outcome] + common_causes
    for col in encode_cols:
        if not pd.api.types.is_numeric_dtype(df_proc[col]):
            df_proc[col] = pd.Categorical(df_proc[col]).codes

    # Impute missing values (median strategy) for the relevant columns
    if encode_cols:
        imputer = SimpleImputer(strategy="median")
        df_proc[encode_cols] = imputer.fit_transform(df_proc[encode_cols])

    if st.button("Estimate Causal Effect"):
        try:
            model = CausalModel(
                data=df_proc,             # use the cleaned, encoded DataFrame
                treatment=treatment,
                outcome=outcome,
                common_causes=common_causes or None
            )
        except Exception as e:
            st.error(f"Failed to construct CausalModel: {e}")
            return

        try:
            identified_estimand = model.identify_effect()
            causal_estimate    = model.estimate_effect(
                identified_estimand,
                method_name="backdoor.propensity_score_matching"
            )
        except Exception as e:
            st.error(f"Estimation failed: {e}")
            return

        st.subheader("Estimated Causal Effect")
        st.write(causal_estimate)
        st.write(f"Effect of {treatment} on {outcome}: {causal_estimate.value}")
