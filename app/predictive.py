# app/predictive.py

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

def run(df):
    st.header("Predictive Modeling")
    if df is None:
        st.info("Please upload a dataset to use this feature.")
        return

    # Select target column and problem type
    columns = df.columns.tolist()
    target = st.selectbox("Select the target (outcome) column", columns)
    problem_type = st.radio("Problem Type", options=["Classification", "Regression"], index=0)

    if st.button("Train Model"):
        try:
            X = df.drop(columns=[target])
            y = df[target]
        except KeyError:
            st.error("Target column not found in dataset.")
            return

        # One-hot encode categorical features
        X = pd.get_dummies(X, drop_first=True)

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Build pipeline: impute missing values, then fit model
        if problem_type == "Classification":
            # Encode target if categorical
            if y.dtype == object or y.dtype.name == "category":
                y_train = pd.factorize(y_train)[0]
                y_test = pd.factorize(y_test)[0]

            model = make_pipeline(
                SimpleImputer(strategy="median"),
                RandomForestClassifier(random_state=42)
            )
        else:
            model = make_pipeline(
                SimpleImputer(strategy="median"),
                RandomForestRegressor(random_state=42)
            )

        # Train the model
        model.fit(X_train, y_train)

        # Predict on test set
        y_pred = model.predict(X_test)

        # Compute metrics
        metrics = {}
        if problem_type == "Classification":
            metrics["Accuracy"] = accuracy_score(y_test, y_pred)
        else:
            mse = mean_squared_error(y_test, y_pred)
            metrics["MSE"] = mse
            metrics["RMSE"] = np.sqrt(mse)
            metrics["R²"] = r2_score(y_test, y_pred)

        # Display results
        st.subheader("Model Training Results")
        for name, value in metrics.items():
            st.write(f"{name}: {value}")

        # Save model and training data for explainability
        st.session_state.model = model
        st.session_state.model_X_train = X_train

        st.success("Model trained and saved for explainability.")
