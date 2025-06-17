FROM python:3.11-slim
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libgl1-mesa-glx \
      libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
# for Tesseract OCR
RUN apt-get update && apt-get install -y tesseract-ocr
RUN pip install -U pip && pip install -r requirements.txt
COPY . .
EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
