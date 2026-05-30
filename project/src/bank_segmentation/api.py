from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field

from .models import SegmentationModel
from .data import load_and_clean_data

from dotenv import load_dotenv

from contextlib import asynccontextmanager

# Загружаем .env
load_dotenv()


# --- Настройка логирования ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("bank_segmentation_api")

SERVICE_CONFIG_PATH = Path("configs/service.yaml")
if SERVICE_CONFIG_PATH.exists():
    with open(SERVICE_CONFIG_PATH, 'r') as f:
        SERVICE_CFG = yaml.safe_load(f)
else:
    SERVICE_CFG = {
        "paths": {"artifacts_dir": "artifacts", "model_file": "best_model.pkl", "scaler_file": "scaler.pkl"},
        "server": {"host": "0.0.0.0", "port": 8000}
    }

ARTIFACTS_DIR = Path(SERVICE_CFG['paths']['artifacts_dir'])
MODEL_PATH = ARTIFACTS_DIR / SERVICE_CFG['paths']['model_file']
SCALER_PATH = ARTIFACTS_DIR / SERVICE_CFG['paths']['scaler_file']

# --- Инициализация FastAPI ---
app = FastAPI(
    title="Bank Customer Segmentation API",
    version="1.0.0",
    description="API для сегментации клиентов банка и оценки кредитного риска.",
    docs_url="/docs",
    redoc_url="/redoc",
)

# --- Prometheus Metrics (Наблюдаемость) ---
REQUEST_COUNT = Counter(
    'http_requests_total', 
    'Total HTTP Requests', 
    ['method', 'endpoint', 'status']
)
REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds', 
    'HTTP Request Latency', 
    ['method', 'endpoint']
)

# --- Пути к артефактам ---
# Предполагаем, что артефакты лежат в папке artifacts в корне проекта (откуда запускается uvicorn)
ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "best_model.pkl"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"

# Глобальная переменная для модели
model_wrapper: Optional[SegmentationModel] = None

# --- Pydantic Models ---

class ClientData(BaseModel):
    """
    Входные данные клиента.
    Поля должны соответствовать исходным признакам датасета (до лог-трансформации).
    Примеры значений взяты из среднего по датасету.
    """
    BALANCE: float = Field(..., ge=0, description="Остаток на счете")
    PURCHASES: float = Field(..., ge=0, description="Сумма покупок")
    ONEOFF_PURCHASES: float = Field(..., ge=0, description="Разовые покупки")
    INSTALLMENTS_PURCHASES: float = Field(..., ge=0, description="Покупки в рассрочку")
    CASH_ADVANCE: float = Field(..., ge=0, description="Снятие наличных")
    PAYMENTS: float = Field(..., ge=0, description="Сумма платежей")
    MINIMUM_PAYMENTS: float = Field(..., ge=0, description="Минимальный платеж")
    CASH_ADVANCE_TRX: int = Field(..., ge=0, description="Количество транзакций снятия наличных")
    CREDIT_LIMIT: float = Field(..., ge=0, description="Кредитный лимит")
    PURCHASES_TRX: int = Field(..., ge=0, description="Количество транзакций покупок")
    PURCHASES_FREQUENCY: float = Field(..., ge=0, le=1, description="Частота покупок")
    ONEOFF_PURCHASES_FREQUENCY: float = Field(..., ge=0, le=1, description="Частота разовых покупок")
    PURCHASES_INSTALLMENTS_FREQUENCY: float = Field(..., ge=0, le=1, description="Частота покупок в рассрочку")
    CASH_ADVANCE_FREQUENCY: float = Field(..., ge=0, le=1, description="Частота снятия наличных")
    PRC_FULL_PAYMENT: float = Field(..., ge=0, le=1, description="Доля полных оплат")
    TENURE: int = Field(..., ge=0, description="Срок обслуживания (месяцы)")

class PredictionResult(BaseModel):
    client_index: int
    cluster_id: int
    typology: str
    risk_level: str
    recommendation: str
    description: str

class ErrorResponse(BaseModel):
    detail: str

# --- Lifecycle Events ---



# ... (остальные импорты)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Код, который выполняется при ЗАПУСКЕ (Startup)
    logger.info("Загрузка модели сегментации...")
    
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        logger.error(f"Артефакты не найдены! Проверьте пути: {MODEL_PATH}, {SCALER_PATH}")
        raise FileNotFoundError("Model artifacts not found. Please run 'bank-cli train' first.")
    
    try:
        global model_wrapper
        model_wrapper = SegmentationModel()
        model_wrapper.load(str(MODEL_PATH), str(SCALER_PATH))
        logger.info("Модель успешно загружена и готова к работе.")
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {e}")
        raise e
        
    yield # Здесь приложение работает
    
    # Код, который выполняется при ОСТАНОВКЕ (Shutdown) - можно оставить пустым
    logger.info("Сервис остановлен.")

# При создании приложения передаем lifespan
app = FastAPI(
    title=SERVICE_CFG['server'].get('title', "Bank Customer Segmentation API"),
    version=SERVICE_CFG['server'].get('version', "1.0.0"),
    description="API для сегментации клиентов банка и оценки кредитного риска.",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan, # <--- ДОБАВЬТЕ ЭТУ СТРОКУ
)

# УДАЛИТЕ старый декоратор @app.on_event("startup") и функцию load_model полностью

@app.get("/", tags=["System"])
def root():
    return {
        "message": "Welcome to Bank Segmentation API",
        "docs": "/docs",
        "health": "/health"
    }

# --- Middleware для Метрик ---

@app.middleware("http")
async def add_metrics(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    
    # Запись метрик
    REQUEST_LATENCY.labels(
        method=request.method, 
        endpoint=request.url.path
    ).observe(duration)
    
    REQUEST_COUNT.labels(
        method=request.method, 
        endpoint=request.url.path, 
        status=response.status_code
    ).inc()
    
    return response

# --- Endpoints ---

@app.get("/health", tags=["System"], summary="Health Check")
def health_check():
    """
    Проверка работоспособности сервиса и наличия загруженной модели.
    """
    if model_wrapper is None or not model_wrapper.is_fitted:
        raise HTTPException(status_code=503, detail="Service Unavailable: Model not loaded")
    
    return {
        "status": "ok",
        "service": "bank-segmentation",
        "version": "1.0.0",
        "model_loaded": True
    }

@app.get("/metrics", tags=["System"], summary="Prometheus Metrics")
def metrics():
    """
    Эндпоинт для сбора метрик Prometheus.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post(
    "/predict", 
    response_model=PredictionResult, 
    tags=["Prediction"],
    summary="Предсказание сегмента для одного клиента",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input data"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
def predict_client(client: ClientData):
    """
    Принимает финансовые данные клиента в формате JSON и возвращает:
    - ID кластера
    - Типологию клиента
    - Уровень риска
    - Рекомендацию
    
    Пример тела запроса см. в схеме ClientData.
    """
    if model_wrapper is None:
        raise HTTPException(status_code=500, detail="Model is not initialized")

    start_time = time.time()
    
    try:
        # 1. Преобразование Pydantic модели в DataFrame
        # dict() возвращает словарь, создаем DataFrame с одной строкой
        df_input = pd.DataFrame([client.model_dump()])
        
        # 2. Предсказание
        # model.predict возвращает список словарей
        results = model_wrapper.predict(df_input)
        
        if not results:
            raise HTTPException(status_code=500, detail="Prediction returned empty result")
            
        result = results[0]
        
        # 3. Логирование
        latency = time.time() - start_time
        logger.info(
            f"Prediction made for client. "
            f"Cluster: {result['cluster_id']}, "
            f"Risk: {result['risk_level']}, "
            f"Latency: {latency:.4f}s"
        )
        
        return result
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Internal error during prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

# Импорт Response для metrics endpoint
from starlette.responses import Response