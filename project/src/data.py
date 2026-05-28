import pandas as pd
import numpy as np

# Путь к данным по умолчанию
DEFAULT_DATA_PATH = "data/credit_cards_dataset.csv"

def load_data(path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """
    Загружает датасет.
    
    Args:
        path: Путь к csv файлу.
        
    Returns:
        pd.DataFrame: Загруженный датафрейм.
    """
    try:
        df = pd.read_csv(path)
        print(f"Датасет успешно загружен! Размер: {df.shape}")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл не найден по пути: {path}")

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Выполняет базовую очистку данных:
    1. Удаляет идентификатор клиента CUST_ID.
    2. Заполняет пропуски в CREDIT_LIMIT медианой.
    3. Заполняет пропуски в MINIMUM_PAYMENTS медианой.
    
    Args:
        df: Исходный датафрейм.
        
    Returns:
        pd.DataFrame: Очищенный датафрейм.
    """
    df_clean = df.copy()
    
    if 'CUST_ID' in df_clean.columns:
        df_clean.drop('CUST_ID', axis=1, inplace=True)
        
    if 'CREDIT_LIMIT' in df_clean.columns:
        df_clean['CREDIT_LIMIT'] = df_clean['CREDIT_LIMIT'].fillna(df_clean['CREDIT_LIMIT'].median(), inplace=True)
        
    if 'MINIMUM_PAYMENTS' in df_clean.columns:
        df_clean['MINIMUM_PAYMENTS'] = df_clean['MINIMUM_PAYMENTS'].fillna(df_clean['MINIMUM_PAYMENTS'].median(), inplace=True)
        
    if df_clean.isnull().sum().sum() > 0:
        print("Внимание: остались пропуски в данных:")
        print(df_clean.isnull().sum())
        
    return df_clean