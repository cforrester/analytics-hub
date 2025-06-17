import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def run(df):
    st.header("Automated Exploratory Data Analysis")
    if df is None:
        st.info("Please upload a dataset to get started.")
        return
    # Basic dataset info
    st.subheader("Dataset Overview")
    st.write("Shape of dataset:", df.shape)
    st.write(df.describe(include='all').T)  # summary statistics (transpose for readability)
    # Show sample data
    st.write("First 5 rows:", df.head())
    # Column selection for distributions
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        sel_col = st.selectbox("Select a numeric column to visualize distribution", numeric_cols)
        fig, ax = plt.subplots()
        sns.histplot(df[sel_col].dropna(), kde=True, ax=ax)
        ax.set_title(f'Distribution of {sel_col}')
        st.pyplot(fig)
    # Correlation matrix
    if len(numeric_cols) > 1:
        if st.checkbox("Show correlation matrix"):
            corr = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(6,5))
            sns.heatmap(corr, annot=True, cmap="YlGnBu", ax=ax)
            ax.set_title("Correlation Matrix")
            st.pyplot(fig)
    # TODO: Could add more EDA visuals (boxplots, pairplots) as needed.
