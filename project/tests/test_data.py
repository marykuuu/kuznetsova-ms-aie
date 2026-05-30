import pandas as pd
import numpy as np
import pytest
from pathlib import Path
import sys
import os

# Добавляем src в путь, чтобы импорты работали при запуске pytest из корня project/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bank_segmentation.data import load_and_clean_data, FILL_MEDIAN_COLS

def test_load_and_clean_data_removes_cust_id():
    """Проверяет, что столбец CUST_ID удаляется."""
    data = {
        'CUST_ID': [1, 2, 3],
        'BALANCE': [100, 200, 300],
        'PURCHASES': [50, 60, 70]
    }
    df = pd.DataFrame(data)
    
    # Сохраняем во временный CSV для теста
    temp_csv = "temp_test_data.csv"
    df.to_csv(temp_csv, index=False)
    
    try:
        cleaned_df = load_and_clean_data(file_path=temp_csv)
        assert 'CUST_ID' not in cleaned_df.columns, "CUST_ID should be removed"
        assert 'BALANCE' in cleaned_df.columns
    finally:
        if os.path.exists(temp_csv):
            os.remove(temp_csv)

def test_load_and_clean_data_fills_median():
    """Проверяет, что пропуски в указанных колонках заполняются медианой."""
    data = {
        'CREDIT_LIMIT': [1000, np.nan, 3000],
        'MINIMUM_PAYMENTS': [100, 200, np.nan],
        'OTHER_COL': [1, 2, 3]
    }
    df = pd.DataFrame(data)
    
    cleaned_df = load_and_clean_data(df=df)
    
    # Проверяем, что пропусков в целевых колонках нет
    for col in FILL_MEDIAN_COLS:
        if col in cleaned_df.columns:
            assert cleaned_df[col].isnull().sum() == 0, f"Nulls in {col} should be filled"
            
    # Проверяем конкретные значения медианы
    assert cleaned_df['CREDIT_LIMIT'].iloc[1] == 2000.0 # Медиана [1000, 3000]
    assert cleaned_df['MINIMUM_PAYMENTS'].iloc[2] == 150.0 # Медиана [100, 200]

def test_load_and_clean_data_handles_remaining_nulls():
    """Проверяет, что остальные пропуски заполняются нулями."""
    data = {
        'COL_A': [1, np.nan, 3],
        'COL_B': [np.nan, 2, 3]
    }
    df = pd.DataFrame(data)
    
    cleaned_df = load_and_clean_data(df=df)
    
    assert cleaned_df.isnull().sum().sum() == 0, "All nulls should be filled with 0"
    assert cleaned_df['COL_A'].iloc[1] == 0
    assert cleaned_df['COL_B'].iloc[0] == 0