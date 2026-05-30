import pandas as pd
import numpy as np
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bank_segmentation.features import FeatureEngineer, SELECTED_FEATURES, LOG_COLS

def test_feature_engineer_fit_transform_shape():
    """Проверяет, что выходной массив имеет правильную форму."""
    data = {
        'BALANCE': [100, 200, 300, 400, 500],
        'PURCHASES': [10, 20, 30, 40, 50],
        'CASH_ADVANCE': [5, 10, 15, 20, 25],
        'INSTALLMENTS_PURCHASES': [2, 4, 6, 8, 10],
        'PURCHASES_FREQUENCY': [0.1, 0.2, 0.3, 0.4, 0.5],
        'CASH_ADVANCE_FREQUENCY': [0.05, 0.1, 0.15, 0.2, 0.25],
        'PRC_FULL_PAYMENT': [0.8, 0.7, 0.6, 0.5, 0.4],
        'CREDIT_LIMIT': [1000, 2000, 3000, 4000, 5000],
        'TENURE': [12, 24, 36, 48, 60],
        # Добавим другие колонки из LOG_COLS, чтобы избежать ошибок
        'ONEOFF_PURCHASES': [1, 2, 3, 4, 5],
        'PAYMENTS': [10, 20, 30, 40, 50],
        'MINIMUM_PAYMENTS': [1, 2, 3, 4, 5],
        'CASH_ADVANCE_TRX': [1, 2, 3, 4, 5],
        'PURCHASES_TRX': [1, 2, 3, 4, 5],
    }
    df = pd.DataFrame(data)
    
    fe = FeatureEngineer()
    scaled_data = fe.fit_transform(df)
    
    assert scaled_data.shape[0] == df.shape[0], "Number of rows should match"
    assert scaled_data.shape[1] == len(SELECTED_FEATURES), f"Number of features should be {len(SELECTED_FEATURES)}"

def test_feature_engineer_transform_without_fit_raises_error():
    """Проверяет, что transform без fit вызывает ошибку."""
    data = {
        'BALANCE': [100, 200],
        'PURCHASES': [10, 20],
        'CASH_ADVANCE': [5, 10],
        'INSTALLMENTS_PURCHASES': [2, 4],
        'PURCHASES_FREQUENCY': [0.1, 0.2],
        'CASH_ADVANCE_FREQUENCY': [0.05, 0.1],
        'PRC_FULL_PAYMENT': [0.8, 0.7],
        'CREDIT_LIMIT': [1000, 2000],
        'TENURE': [12, 24],
        'ONEOFF_PURCHASES': [1, 2],
        'PAYMENTS': [10, 20],
        'MINIMUM_PAYMENTS': [1, 2],
        'CASH_ADVANCE_TRX': [1, 2],
        'PURCHASES_TRX': [1, 2],
    }
    df = pd.DataFrame(data)
    
    fe = FeatureEngineer()
    
    with pytest.raises(Exception, match="FeatureEngineer не был обучен"):
        fe.transform(df)

def test_feature_engineer_log_transform_applied():
    """Проверяет, что логарифмирование применяется к нужным колонкам."""
    data = {
        'BALANCE': [100, 200],
        'PURCHASES': [10, 20],
        'CASH_ADVANCE': [5, 10],
        'INSTALLMENTS_PURCHASES': [2, 4],
        'PURCHASES_FREQUENCY': [0.1, 0.2],
        'CASH_ADVANCE_FREQUENCY': [0.05, 0.1],
        'PRC_FULL_PAYMENT': [0.8, 0.7],
        'CREDIT_LIMIT': [1000, 2000],
        'TENURE': [12, 24],
        'ONEOFF_PURCHASES': [1, 2],
        'PAYMENTS': [10, 20],
        'MINIMUM_PAYMENTS': [1, 2],
        'CASH_ADVANCE_TRX': [1, 2],
        'PURCHASES_TRX': [1, 2],
    }
    df = pd.DataFrame(data)
    
    fe = FeatureEngineer()
    df_log = fe.apply_log_transform(df)
    
    # Проверяем, что значения изменились (логарифмирование)
    for col in LOG_COLS:
        if col in df.columns:
            assert not np.allclose(df[col].values, df_log[col].values), f"Column {col} should be log-transformed"