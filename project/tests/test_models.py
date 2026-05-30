import pandas as pd
import numpy as np
import pytest
from pathlib import Path
import sys
import os
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bank_segmentation.models import SegmentationModel, CLUSTER_PROFILES

def get_sample_data():
    """Вспомогательная функция для создания тестовых данных."""
    data = {
        'BALANCE': [100, 200, 300, 400, 500] * 2,
        'PURCHASES': [10, 20, 30, 40, 50] * 2,
        'CASH_ADVANCE': [5, 10, 15, 20, 25] * 2,
        'INSTALLMENTS_PURCHASES': [2, 4, 6, 8, 10] * 2,
        'PURCHASES_FREQUENCY': [0.1, 0.2, 0.3, 0.4, 0.5] * 2,
        'CASH_ADVANCE_FREQUENCY': [0.05, 0.1, 0.15, 0.2, 0.25] * 2,
        'PRC_FULL_PAYMENT': [0.8, 0.7, 0.6, 0.5, 0.4] * 2,
        'CREDIT_LIMIT': [1000, 2000, 3000, 4000, 5000] * 2,
        'TENURE': [12, 24, 36, 48, 60] * 2,
        'ONEOFF_PURCHASES': [1, 2, 3, 4, 5] * 2,
        'PAYMENTS': [10, 20, 30, 40, 50] * 2,
        'MINIMUM_PAYMENTS': [1, 2, 3, 4, 5] * 2,
        'CASH_ADVANCE_TRX': [1, 2, 3, 4, 5] * 2,
        'PURCHASES_TRX': [1, 2, 3, 4, 5] * 2,
    }
    return pd.DataFrame(data)

def test_segmentation_model_fit_and_predict():
    """Проверяет, что модель обучается и делает предсказания."""
    df = get_sample_data()
    
    model = SegmentationModel()
    model.fit(df)
    
    assert model.is_fitted, "Model should be fitted"
    
    predictions = model.predict(df.head(3))
    
    assert len(predictions) == 3, "Should have 3 predictions"
    
    for pred in predictions:
        assert 'cluster_id' in pred
        assert 'typology' in pred
        assert 'risk_level' in pred
        assert 'recommendation' in pred
        assert 'description' in pred
        assert isinstance(pred['cluster_id'], int)
        assert pred['typology'] in [p['typology'] for p in CLUSTER_PROFILES.values()]

def test_segmentation_model_predict_without_fit_raises_error():
    """Проверяет, что predict без fit вызывает ошибку."""
    df = get_sample_data()
    
    model = SegmentationModel()
    
    with pytest.raises(Exception, match="Модель не обучена"):
        model.predict(df)

def test_segmentation_model_save_and_load(tmp_path):
    """Проверяет сохранение и загрузку модели."""
    df = get_sample_data()
    
    model = SegmentationModel()
    model.fit(df)
    
    model_path = tmp_path / "model.pkl"
    scaler_path = tmp_path / "scaler.pkl"
    
    model.save(str(model_path), str(scaler_path))
    
    assert model_path.exists(), "Model file should exist"
    assert scaler_path.exists(), "Scaler file should exist"
    
    # Загружаем в новую инстанцию
    new_model = SegmentationModel()
    new_model.load(str(model_path), str(scaler_path))
    
    assert new_model.is_fitted, "Loaded model should be fitted"
    
    # Предсказания должны совпадать
    original_preds = model.predict(df.head(1))
    loaded_preds = new_model.predict(df.head(1))
    
    assert original_preds[0]['cluster_id'] == loaded_preds[0]['cluster_id']