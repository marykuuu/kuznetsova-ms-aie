import pytest
from pathlib import Path
import sys
import os

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fastapi.testclient import TestClient
from bank_segmentation.api import app

@pytest.fixture
def client():
    """Фикстура для создания тестового клиента."""
    # Перед тестами убедимся, что модель загружена (или замокаем это)
    # Для простоты, предположим, что артефакты уже есть в artifacts/
    # Если нет, тесты могут упасть. В реальном проекте лучше моковать загрузку модели.
    with TestClient(app) as c:
        yield c

def test_health_check(client):
    """Проверяет health-check эндпоинт."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict_endpoint_success(client):
    """Проверяет успешное предсказание."""
    # Пример данных клиента
    client_data = {
        "BALANCE": 4000,
        "PURCHASES": 1000,
        "ONEOFF_PURCHASES": 500,
        "INSTALLMENTS_PURCHASES": 500,
        "CASH_ADVANCE": 0,
        "PAYMENTS": 1000,
        "MINIMUM_PAYMENTS": 100,
        "CASH_ADVANCE_TRX": 0,
        "CREDIT_LIMIT": 5000,
        "PURCHASES_TRX": 10,
        "PURCHASES_FREQUENCY": 0.5,
        "ONEOFF_PURCHASES_FREQUENCY": 0.2,
        "PURCHASES_INSTALLMENTS_FREQUENCY": 0.3,
        "CASH_ADVANCE_FREQUENCY": 0,
        "PRC_FULL_PAYMENT": 0.8,
        "TENURE": 12
    }
    
    response = client.post("/predict", json=client_data)
    
    assert response.status_code == 200
    
    data = response.json()
    assert "cluster_id" in data
    assert "typology" in data
    assert "risk_level" in data
    assert "recommendation" in data
    assert "description" in data

def test_predict_endpoint_validation_error(client):
    """Проверяет валидацию входных данных."""
    # Неполные данные
    incomplete_data = {
        "BALANCE": 4000,
        # Остальные поля отсутствуют
    }
    
    response = client.post("/predict", json=incomplete_data)
    
    assert response.status_code == 422 # Validation Error