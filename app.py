# app.py

import streamlit as st
import os
import pandas as pd
from io import BytesIO
from PIL import Image
import PyPDF2

# Import tabular modules
from app import eda, predictive, time_series, clustering, causal, optimization, explainability
# Import image‐analysis dispatcher
import app.image_analysis as image_analysis
# Import PDF‐analysis dispatcher
import app.pdf as pdf_analysis
# Import pickle‐analysis dispatcher
import app.pickle as pickle_analysis

# Page config
st.set_page_config(page_title="Universal Prescriptive Analytics Engine", layout="wide")
st.title("Universal Prescriptive Analytics Engine")
st.sidebar.title("Navigation")

# Ensure data/ folder exists
os.makedirs("data", exist_ok=True)

# Initialize session state
if "uploaded_bytes" not in st.session_state:
    st.session_state.uploaded_bytes = None
    st.session_state.uploaded_name  = None
if "df_raw" not in st.session_state:
    st.session_state.df_raw = None

# File uploader
uploaded_file = st.sidebar.file_uploader(
    "Upload a file (CSV, image, PDF, or pickle)",
    type=None,
    accept_multiple_files=False,
)
if uploaded_file:
    raw = uploaded_file.read()
    st.session_state.uploaded_bytes = raw
    st.session_state.uploaded_name  = uploaded_file.name
    fname = st.session_state.uploaded_name.lower()

    # Reset CSV if not .csv
    if not fname.endswith(".csv"):
        st.session_state.df_raw = None

    # 1) CSV
    if fname.endswith(".csv"):
        try:
            df_loaded = pd.read_csv(BytesIO(raw))
            st.session_state.df_raw = df_loaded
            st.sidebar.success(f"Loaded CSV: {uploaded_file.name}")
        except Exception as e:
            st.sidebar.error(f"CSV parse failed: {e}")

    # 2) Image
    elif fname.endswith((".png", ".jpg", ".jpeg", ".gif")):
        try:
            img = Image.open(BytesIO(raw))
            st.image(img, caption=uploaded_file.name, use_container_width=True)
            with open(f"data/{uploaded_file.name}", "wb") as f:
                f.write(raw)
        except Exception as e:
            st.sidebar.error(f"Image preview failed: {e}")

    # 3) PDF
    elif fname.endswith(".pdf"):
        try:
            reader = PyPDF2.PdfReader(BytesIO(raw))
            txt = reader.pages[0].extract_text() or ""
            st.text_area(f"First page of {uploaded_file.name}", txt, height=200)
            with open(f"data/{uploaded_file.name}", "wb") as f:
                f.write(raw)
        except Exception as e:
            st.sidebar.error(f"PDF parse failed: {e}")

    # 4) Pickle
    elif fname.endswith((".pkl", ".pickle")):
        st.sidebar.info("Pickle file uploaded. Choose an analysis module below.")
        with open(f"data/{uploaded_file.name}", "wb") as f:
            f.write(raw)

    else:
        st.sidebar.info("Unsupported file type. Upload CSV, image, PDF, or pickle.")
else:
    if st.session_state.df_raw is None:
        st.sidebar.info("Upload a CSV, image, PDF, or pickle to get started.")

# Only show “Fields to exclude” for CSV
if st.session_state.df_raw is not None:
    cols = st.session_state.df_raw.columns.tolist()
    exclude = st.sidebar.multiselect("Fields to exclude", cols)
    df = (
        st.session_state.df_raw.drop(columns=exclude)
        if exclude
        else st.session_state.df_raw
    )
else:
    df = None

# Define module lists
csv_modules = [
    "Automated EDA",
    "Predictive Modeling",
    "Time Series Forecasting",
    "Clustering & Anomaly Detection",
    "Causal Inference",
    "Prescriptive Optimization",
    "Explainability",
]
image_modules = [
    "Metadata & Color Histograms",
    "Grayscale & Edges",
    "Blur & Sharpen",
    "Color Quantization",
    "ImageNet Classification",
    "Object Detection",
    "OCR",
    "Anomaly Detection",
]
pdf_modules = [
    "Document Metadata",
    "Extract Text",
    "Word Frequency",
    "Named Entity Recognition",
    "Summarization",
    "Extract Tables",
    "Page Thumbnails",
    "OCR on Pages",
]
pickle_modules = [
    "Metadata",
    "Disassemble",
    "List Globals",
    "Safe Preview",
    "Opcode Stats",
    "Protocol Versions",
    "Find Large Objects",
    "Detect Dangerous Objects",
]

# Pick dropdown options based on uploaded file type
if df is not None:
    options = csv_modules
elif st.session_state.uploaded_name and st.session_state.uploaded_name.lower().endswith(
    (".png", ".jpg", ".jpeg", ".gif")
):
    options = image_modules
elif st.session_state.uploaded_name and st.session_state.uploaded_name.lower().endswith(".pdf"):
    options = pdf_modules
elif st.session_state.uploaded_name and st.session_state.uploaded_name.lower().endswith(
    (".pkl", ".pickle")
):
    options = pickle_modules
else:
    options = []

# Sidebar dropdown
if options:
    page = st.sidebar.selectbox("Select Feature Module", options)
else:
    page = None

# Dispatch to the correct module
if page in csv_modules:
    {
        "Automated EDA": eda,
        "Predictive Modeling": predictive,
        "Time Series Forecasting": time_series,
        "Clustering & Anomaly Detection": clustering,
        "Causal Inference": causal,
        "Prescriptive Optimization": optimization,
        "Explainability": explainability,
    }[page].run(df)

elif page in image_modules:
    image_analysis.run_feature(page, st.session_state.uploaded_bytes, st.session_state.uploaded_name)

elif page in pdf_modules:
    pdf_analysis.run_feature(page, st.session_state.uploaded_bytes, st.session_state.uploaded_name)

elif page in pickle_modules:
    pickle_analysis.run_feature(page, st.session_state.uploaded_bytes, st.session_state.uploaded_name)
