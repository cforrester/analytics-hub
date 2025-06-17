# Universal Prescriptive Analytics Engine

A Dockerized Streamlit application for performing prescriptive analytics on a wide range of data types: CSV, images, PDFs, and Python pickle files. Features include automated EDA, predictive modeling, time series forecasting, clustering & anomaly detection, causal inference, prescriptive optimization, explainability, image analysis, PDF analysis, and secure pickle inspection.

## Requirements

- Docker & Docker Compose
- (Optional) Python 3.10+ for local development
- Git

## Installation

1. Clone the repository
   git clone https://github.com/cforrester/analytics-hub.git
   cd analytics-hub

2. Build and run with Docker Compose
   docker compose up -d --build

3. Open your browser to http://localhost:8501

## Usage

1. Use the sidebar to Upload a file (CSV, image, PDF, or pickle).
2. Select a Feature Module from the dropdown.
3. Interact with the inputs and view results.

## Module Reference

### CSV Modules
- Automated EDA: summary stats, histograms, correlations, missing-value report
- Predictive Modeling: preprocessing, train/test split, pipelines, performance metrics
- Time Series Forecasting: date parsing, ARIMA/Prophet models, forecast plots
- Clustering & Anomaly Detection: K-means, DBSCAN, PCA visualization, outlier detection
- Causal Inference: causal graph building, backdoor adjustment, DoWhy estimation
- Prescriptive Optimization: linear program formulation, maximize/minimize objectives, optimal solutions
- Explainability: SHAP global and local explanations

### Image Modules
- Metadata & Color Histograms: EXIF data, RGB/Hue histograms
- Grayscale & Edges: convert to grayscale, Canny edge detection
- Blur & Sharpen: Gaussian blur, unsharp masking
- Color Quantization: K-means color reduction
- ImageNet Classification: pretrained ResNet50 top-1 predictions
- Object Detection: YOLOv5 bounding boxes + detection table
- OCR: text extraction via Tesseract
- Anomaly Detection: autoencoder reconstruction error heatmap

### PDF Modules
- Document Metadata: title, author, creation date, page count
- Extract Text: full text or per-page preview
- Word Frequency: tokenization, top-N word counts
- Named Entity Recognition: SpaCy NER pipeline
- Summarization: Gensim or transformer-based summaries
- Extract Tables: Tabula table extraction to DataFrame
- Page Thumbnails: PNG previews of all pages
- OCR on Pages: OCR each page to text

### Pickle Modules
- Metadata: MD5/SHA-256 hashes, protocol version
- Disassemble: opcode listing via pickletools.dis
- List Globals: referenced modules and class names
- Safe Preview: restricted unpickle of primitives and whitelisted classes
- Opcode Stats: frequency of each pickle opcode
- Protocol Versions: counts of PROTO versions
- Find Large Objects: identify embedded constants >1 KB
- Detect Dangerous Objects: flag GLOBAL or call opcodes that may execute code

## Development

- Source code lives in the app/ directory.
- After making changes, rebuild and restart:
    docker compose build streamlit_app
    docker compose up -d

- To view logs:
    docker compose logs -f streamlit_app
