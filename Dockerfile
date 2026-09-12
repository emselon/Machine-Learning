FROM python:3.11-slim

WORKDIR /app

# matplotlib cần lib hệ thống để render ảnh không cần GUI.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

EXPOSE 8000

CMD ["sh", "-c", "python src/train_compare_svm.py && uvicorn src.api:app --host 0.0.0.0 --port 8000"]