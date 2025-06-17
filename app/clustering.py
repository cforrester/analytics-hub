import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest

def run(df):
    st.header("Clustering and Anomaly Detection")
    if df is None:
        st.info("Please upload a dataset to use this feature.")
        return
    # Feature selection for clustering
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        st.error("No numeric features available for clustering.")
        return
    features = st.multiselect("Select features for clustering", numeric_cols, default=numeric_cols)
    algorithm = st.selectbox("Choose algorithm", ["K-Means", "DBSCAN", "Isolation Forest"])
    if st.button("Run"):
        data = df[features].dropna()
        if data.shape[0] == 0:
            st.error("Selected features have no valid rows (all NaNs?).")
            return
        if algorithm == "K-Means":
            k = st.sidebar.slider("Number of clusters (k)", min_value=2, max_value=10, value=3)
            model = KMeans(n_clusters=k, random_state=0)
            labels = model.fit_predict(data)
            st.write(f"Cluster centers (means): {model.cluster_centers_}")
            st.write(f"Cluster labels counts: {np.unique(labels, return_counts=True)}")
            # If 2D, plot clusters
            if data.shape[1] == 2:
                fig, ax = plt.subplots()
                ax.scatter(data.iloc[:,0], data.iloc[:,1], c=labels, cmap='viridis', alpha=0.7)
                ax.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], marker='X', s=100, c='red')
                ax.set_title("K-Means Clustering (2D visualization)")
                st.pyplot(fig)
        elif algorithm == "DBSCAN":
            eps = st.sidebar.slider("eps (neighborhood radius)", min_value=0.1, max_value=10.0, value=0.5, step=0.1)
            min_samples = st.sidebar.slider("min_samples", min_value=1, max_value=20, value=5)
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(data)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            st.write(f"Estimated number of clusters: {n_clusters}")
            st.write(f"Estimated number of noise (outliers): {n_noise}")
            # Plot if 2D
            if data.shape[1] == 2:
                fig, ax = plt.subplots()
                ax.scatter(data.iloc[:,0], data.iloc[:,1], c=labels, cmap='plasma', alpha=0.8)
                ax.set_title("DBSCAN Clustering (2D visualization)")
                st.pyplot(fig)
        else:  # Isolation Forest
            model = IsolationForest(random_state=0, contamination='auto')
            preds = model.fit_predict(data)
            outliers = (preds == -1)
            num_outliers = np.sum(outliers)
            st.write(f"Number of anomalies detected: {num_outliers}")
            if data.shape[1] == 2:
                fig, ax = plt.subplots()
                ax.scatter(data.iloc[:,0], data.iloc[:,1], c=outliers, cmap='coolwarm')
                ax.set_title("Isolation Forest Anomaly Detection (2D visualization)")
                st.pyplot(fig)
