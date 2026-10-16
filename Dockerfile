FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# CPU wheels are smaller than the default CUDA-enabled PyTorch downloads.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.10.0 torchvision==0.25.0
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY src ./src
COPY frontend ./frontend
COPY models ./models
COPY data/processed/agrisense_metadata_clean.csv ./data/processed/agrisense_metadata_clean.csv
EXPOSE 8000
# Honor Render's PORT; one worker avoids a second ~200MB model instance.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
