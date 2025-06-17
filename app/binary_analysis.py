# app/binary_analysis.py

import streamlit as st
import magic
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import zlib
import gzip
import lzma
import re
import hashlib
import math

def _shannon_entropy(data: bytes) -> float:
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())

def run(data_bytes: bytes, filename: str):
    st.header("Binary Data Analysis")
    st.write(f"Filename: {filename}, Size: {len(data_bytes)} bytes")

    # 1) File type identification
    if st.checkbox("Identify file type (magic/mime)"):
        try:
            fmt = magic.from_buffer(data_bytes)
            mime = magic.from_buffer(data_bytes, mime=True)
            st.write(f"Description: **{fmt}**")
            st.write(f"MIME type: **{mime}**")
        except Exception as e:
            st.error(f"magic analysis failed: {e}")

    # 2) Global Shannon entropy
    if st.checkbox("Compute global entropy"):
        ent = _shannon_entropy(data_bytes)
        st.write(f"Global Shannon entropy: **{ent:.4f} bits per byte**")

    # 3) Sliding‐window entropy
    if st.checkbox("Sliding‐window entropy"):
        window = st.number_input("Window size (bytes)", min_value=64, max_value=65536, value=1024, step=64)
        step = st.number_input("Step size (bytes)", min_value=1, max_value=window, value=window//2, step=1)
        data = data_bytes
        ents = []
        positions = []
        for i in range(0, len(data) - window + 1, step):
            w = data[i : i + window]
            ents.append(_shannon_entropy(w))
            positions.append(i)
        fig, ax = plt.subplots()
        ax.plot(positions, ents)
        ax.set_xlabel("Offset (bytes)")
        ax.set_ylabel("Entropy")
        ax.set_title("Sliding‐window entropy")
        st.pyplot(fig)

    # 4) Byte‐value histogram
    if st.checkbox("Show byte‐value histogram"):
        arr = np.frombuffer(data_bytes, dtype=np.uint8)
        counts, bins = np.histogram(arr, bins=256, range=(0,255))
        fig, ax = plt.subplots()
        ax.bar(bins[:-1], counts, width=1.0)
        ax.set_xlabel("Byte value")
        ax.set_ylabel("Frequency")
        ax.set_title("Byte‐value histogram")
        st.pyplot(fig)

    # 5) Compression ratios
    if st.checkbox("Compute compression ratios"):
        raw_len = len(data_bytes)
        gz = gzip.compress(data_bytes)
        zl = zlib.compress(data_bytes)
        lz = lzma.compress(data_bytes)
        st.write(f"GZIP:   {len(gz)/raw_len:.3f}")
        st.write(f"Zlib:   {len(zl)/raw_len:.3f}")
        st.write(f"LZMA:   {len(lz)/raw_len:.3f}")

    # 6) Extract printable strings
    if st.checkbox("Extract ASCII/UTF-8 strings"):
        minlen = st.number_input("Min string length", min_value=4, max_value=64, value=8, step=1)
        pattern = re.compile(rb"[ -~]{" + str(minlen).encode() + rb",}")
        strings = pattern.findall(data_bytes)
        text_strings = [s.decode(errors="ignore") for s in strings]
        st.write(f"Found {len(text_strings)} strings of length ≥ {minlen}")
        st.dataframe(text_strings, height=200)

    # 7) Cryptographic hashes
    if st.checkbox("Compute cryptographic hashes"):
        md5  = hashlib.md5(data_bytes).hexdigest()
        sha1 = hashlib.sha1(data_bytes).hexdigest()
        sha256 = hashlib.sha256(data_bytes).hexdigest()
        st.write(f"MD5:    {md5}")
        st.write(f"SHA-1:  {sha1}")
        st.write(f"SHA-256:{sha256}")

    # 8) Byte‐image visualization
    if st.checkbox("Visualize raw bytes as grayscale image"):
        width = st.number_input("Image width (pixels)", min_value=16, max_value=2048, value=256, step=16)
        arr = np.frombuffer(data_bytes, dtype=np.uint8)
        height = math.ceil(len(arr) / width)
        padded = np.pad(arr, (0, width * height - len(arr)), mode="constant", constant_values=0)
        img_array = padded.reshape((height, width))
        fig, ax = plt.subplots()
        ax.imshow(img_array, cmap="gray", aspect="auto")
        ax.set_axis_off()
        st.pyplot(fig)

    # 9) Header/footer carving basic
    if st.checkbox("Scan for known magic signatures"):
        sigs = {
            "JPEG": b"\xff\xd8\xff",
            "PNG":  b"\x89PNG\r\n\x1a\n",
            "GIF":  b"GIF8",
            "PDF":  b"%PDF-",
            "ZIP":  b"PK\x03\x04",
        }
        results = []
        for name, sig in sigs.items():
            idxs = [m.start() for m in re.finditer(re.escape(sig), data_bytes)]
            if idxs:
                results.append((name, sig, idxs[:5]))
        if results:
            for name, sig, offs in results:
                st.write(f"{name} signature found at offsets: {offs}")
        else:
            st.write("No known signatures found.")
