FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

COPY src/ ./src/
COPY configs/ ./configs/

RUN mkdir -p artifacts

EXPOSE 8000

CMD ["uvicorn", "src.bank_segmentation.api:app", "--host", "0.0.0.0", "--port", "8000"]