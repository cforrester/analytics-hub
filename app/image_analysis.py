# app/image_analysis.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import cv2
import pytesseract
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import torch
import torchvision.transforms as T
from torchvision import models
import json
import os
import urllib.request
import pandas as pd
import altair as alt
import base64

def _load_imagenet_labels():
    local = "/usr/share/imagenet_class_index.json"
    if os.path.exists(local):
        with open(local) as f:
            return {int(k): v[1] for k, v in json.load(f).items()}
    for url in [
        "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_class_index.json",
        "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json",
    ]:
        try:
            data = json.loads(urllib.request.urlopen(url).read().decode())
            return {int(k): v[1] for k, v in data.items()}
        except:
            pass
    st.error("Could not load ImageNet labels.")
    return {}

def run_feature(feature, uploaded_bytes, filename):
    # Load image + array once
    try:
        img = Image.open(BytesIO(uploaded_bytes)).convert("RGB")
    except Exception as e:
        st.error(f"Cannot open image: {e}")
        return
    arr = np.array(img)

    st.header(feature)

    if feature == "Metadata & Color Histograms":
        h, w, c = arr.shape
        st.write(f"Dimensions: {w}×{h}, Channels: {c}")
        fig, axes = plt.subplots(1, 3, figsize=(12, 3))
        for i, col in enumerate(("R","G","B")):
            axes[i].hist(arr[...,i].ravel(), bins=256, color=col.lower())
            axes[i].set_title(f"{col} channel")
        st.pyplot(fig)

    elif feature == "Grayscale & Edges":
        gray  = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        c1, c2 = st.columns(2)
        c1.image(gray,  caption="Grayscale", clamp=True)
        c2.image(edges, caption="Canny Edges", clamp=True)

    elif feature == "Blur & Sharpen":
        k = st.slider("Gaussian blur kernel size", 1, 31, 5, step=2)
        blurred = cv2.GaussianBlur(arr, (k,k), 0)
        kernel  = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        sharp   = cv2.filter2D(arr, -1, kernel)
        c1, c2  = st.columns(2)
        c1.image(blurred, caption="Blurred",   clamp=True)
        c2.image(sharp,   caption="Sharpened", clamp=True)

    elif feature == "Color Quantization":
        n = st.slider("Number of colors", 2, 20, 5)
        data = arr.reshape(-1,3)
        km = KMeans(n_clusters=n, random_state=0).fit(data)
        quant = km.cluster_centers_[km.labels_].reshape(arr.shape).astype(np.uint8)
        st.image(quant, caption=f"{n}-color quantized", clamp=True)

    elif feature == "ImageNet Classification":
        st.write("Running ResNet50 (CPU)…")
        model = models.resnet50(pretrained=True).eval()
        prep = T.Compose([
            T.Resize(256), T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        inp = prep(img).unsqueeze(0)
        with torch.no_grad():
            out = model(inp)
        probs = torch.nn.functional.softmax(out[0], dim=0)
        top5  = torch.topk(probs, 5)
        idxs  = top5.indices.cpu().numpy()
        vals  = top5.values.cpu().numpy()
        labels = _load_imagenet_labels()
        st.subheader("Top-5 predictions:")
        for i,p in zip(idxs, vals):
            st.write(f"{labels.get(i,str(i))}: {p:.3f}")

    elif feature == "Object Detection":
        st.write("Detecting objects (CPU)…")
        od = models.detection.fasterrcnn_resnet50_fpn(pretrained=True).eval()
        prep = T.Compose([T.ToTensor()])
        inp = prep(img).unsqueeze(0)
        with torch.no_grad():
            det = od(inp)[0]
        labels = _load_imagenet_labels()
        results = []
        disp = arr.copy()
        for idx,(box,lab,sc) in enumerate(zip(det["boxes"],det["labels"],det["scores"])):
            if sc < 0.5: continue
            x1,y1,x2,y2 = box.int().tolist()
            lbl = labels.get(int(lab),str(int(lab)))
            results.append({"id":idx,"label":lbl,"score":round(float(sc),3),
                            "x1":x1,"y1":y1,"x2":x2,"y2":y2})
            cv2.rectangle(disp,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(disp,f"{lbl}:{sc:.2f}",(x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

        if results:
            df_det = pd.DataFrame(results)
            # display table
            st.subheader("Detections")
            st.table(df_det[["label","score","x1","y1","x2","y2"]])
        st.image(disp, caption="Annotated", clamp=True)

    elif feature == "OCR":
        try:
            txt = pytesseract.image_to_string(img)
            st.text_area("Extracted Text", txt, height=200)
        except Exception as e:
            st.error(f"OCR failed: {e}")

    elif feature == "Anomaly Detection":
        flat = arr.reshape(-1,3)
        cont = st.slider("Outlier fraction",0.0,0.5,0.01,step=0.01)
        iso = IsolationForest(contamination=cont, random_state=0)
        preds = iso.fit_predict(flat)
        mask  = (preds.reshape(arr.shape[:2]) == -1)
        over  = arr.copy()
        over[mask] = [255,0,0]
        st.image(over, caption="Anomalies", clamp=True)

    else:
        st.error(f"Unknown feature: {feature}")
