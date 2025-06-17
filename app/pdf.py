# app/pdf.py

import streamlit as st
import pdfplumber
import PyPDF2
import pandas as pd
import numpy as np
from io import BytesIO
from PIL import Image
import pytesseract
import spacy
import matplotlib.pyplot as plt

# For extractive summarization
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# Cache loading of spaCy model
@st.cache_resource
def get_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("spaCy model 'en_core_web_sm' not found. Please install it.")
        return None

def run_feature(feature: str, pdf_bytes: bytes, filename: str):
    st.header(f"PDF Analysis: {feature}")

    # Load PDF pages and full text
    try:
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            pages = pdf.pages
            full_text = "\n".join(p.extract_text() or "" for p in pages)
    except Exception as e:
        st.error(f"Failed to open PDF: {e}")
        return

    # 1) Document Metadata
    if feature == "Document Metadata":
        try:
            reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
            meta = reader.metadata or {}
            df_meta = pd.DataFrame.from_dict(meta, orient="index", columns=["Value"])
            st.table(df_meta)
        except Exception as e:
            st.error(f"Metadata extraction failed: {e}")

    # 2) Extract Text
    elif feature == "Extract Text":
        st.text_area("Full text", full_text, height=400)

    # 3) Word Frequency
    elif feature == "Word Frequency":
        words = [w.lower() for w in full_text.split() if w.isalpha()]
        freq = pd.Series(words).value_counts().head(20)
        st.bar_chart(freq)
        st.table(freq.rename_axis("word").reset_index(name="count"))

    # 4) Named Entity Recognition
    elif feature == "Named Entity Recognition":
        nlp = get_nlp()
        if nlp is not None:
            doc = nlp(full_text)
            ents = [(ent.text, ent.label_) for ent in doc.ents]
            if ents:
                df_ents = pd.DataFrame(ents, columns=["Entity", "Label"])
                st.table(df_ents)
            else:
                st.write("No entities found.")

    # 5) Summarization (using sumy LexRank)
    elif feature == "Summarization":
        # Let user choose number of sentences
        sentences_count = st.slider("Number of sentences", 1, 10, 3)
        try:
            parser = PlaintextParser.from_string(full_text, Tokenizer("english"))
            summarizer = LexRankSummarizer()
            summary_sentences = summarizer(parser.document, sentences_count)
            summary_text = " ".join(str(s) for s in summary_sentences)
            st.write(summary_text)
        except Exception as e:
            st.error(f"Summarization failed: {e}")

    # 6) Extract Tables
    elif feature == "Extract Tables":
        tables = []
        for i, page in enumerate(pages):
            for table in page.extract_tables():
                df_table = pd.DataFrame(table[1:], columns=table[0])
                tables.append((i + 1, df_table))
        if tables:
            for page_num, df_table in tables:
                st.subheader(f"Page {page_num} Table")
                st.dataframe(df_table)
        else:
            st.write("No tables found in this PDF.")

    # 7) Page Thumbnails
    elif feature == "Page Thumbnails":
        thumbs = []
        captions = []
        for i, page in enumerate(pages[:5]):  # limit to first 5 pages
            im = page.to_image(resolution=100).original.convert("RGB")
            thumbs.append(im)
            captions.append(f"Page {i + 1}")
        if thumbs:
            st.image(thumbs, caption=captions, use_column_width=True)
        else:
            st.write("Could not render page thumbnails.")

    # 8) OCR on Pages
    elif feature == "OCR on Pages":
        for i, page in enumerate(pages[:5], start=1):
            st.subheader(f"Page {i}")
            im = page.to_image(resolution=200).original.convert("RGB")
            try:
                text = pytesseract.image_to_string(im)
                st.text_area("", text, height=200)
            except Exception as e:
                st.error(f"OCR failed on page {i}: {e}")

    else:
        st.error(f"Unknown PDF analysis feature: {feature}")
