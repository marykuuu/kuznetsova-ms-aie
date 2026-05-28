import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Список признаков, выбранных на основе анализа корреляций
SELECTED_FEATURES = [
    'PURCHASES',
    'CASH_ADVANCE',
    'INSTALLMENTS_PURCHASES',

    'PURCHASES_FREQUENCY',
    'CASH_ADVANCE_FREQUENCY',

    'PRC_FULL_PAYMENT',

    'CREDIT_LIMIT',
    'TENURE'
]

# Признаки, имеющие сильное правостороннее skewness, которые стоит логарифмировать
LOG_COLUMNS = [
    'BALANCE',
    'PURCHASES',
    'ONEOFF_PURCHASES',
    'INSTALLMENTS_PURCHASES',
    'CASH_ADVANCE',
    'PAYMENTS',
    'MINIMUM_PAYMENTS',
    'CASH_ADVANCE_TRX',
    'CREDIT_LIMIT',
    'PURCHASES_TRX'
]

def select_and_transform_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Отбирает целевые признаки и применяет логарифмическое преобразование 
    к skewed признакам.
    
    Args:
        df: Очищенный датафрейм.
        
    Returns:
        pd.DataFrame: Датафрейм с отобранными и преобразованными признаками.
    """
    df_proc = df.copy()
    
    existing_features = [col for col in SELECTED_FEATURES if col in df_proc.columns]
    df_selected = df_proc[existing_features].copy()
    
    for col in LOG_COLUMNS:
        if col in df_selected.columns:
            df_selected[col] = np.log1p(df_selected[col])
            
    return df_selected

def scale_features(df: pd.DataFrame, scaler: StandardScaler = None, fit: bool = True) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Масштабирует признаки с помощью StandardScaler.
    
    Args:
        df: Датафрейм с признаками (после отбора и трансформации).
        scaler: Экземпляр StandardScaler. Если None, создается новый.
        fit: Если True, скалер обучается (fit) на данных. Если False, только трансформирует.
             Используется False при инференсе (предсказании на новых данных).
        
    Returns:
        tuple[pd.DataFrame, StandardScaler]: Масштабированный датафрейм и объект скалера.
    """
    if scaler is None:
        scaler = StandardScaler()
        
    columns_to_scale = df.columns
    
    if fit:
        scaled_data = scaler.fit_transform(df[columns_to_scale])
    else:
        scaled_data = scaler.transform(df[columns_to_scale])
        
    df_scaled = pd.DataFrame(scaled_data, columns=columns_to_scale, index=df.index)
    
    return df_scaled, scaler

def save_scaler(scaler: StandardScaler, path: str = "artifacts/scaler.pkl"):
    """
    Сохраняет объект StandardScaler в файл.
    
    Args:
        scaler: Объект StandardScaler.
        path: Путь для сохранения файла.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)
    print(f"Scaler saved to {path}")

def load_scaler(path: str = "artifacts/scaler.pkl") -> StandardScaler:
    """
    Загружает объект StandardScaler из файла.
    
    Args:
        path: Путь к файлу скалера.
        
    Returns:
        StandardScaler: Загруженный объект скалера.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scaler file not found at {path}")
    scaler = joblib.load(path)
    return scaler

def get_preprocessed_and_scaled_data(df: pd.DataFrame, fit_scaler: bool = True, scaler_path: str = "artifacts/scaler.pkl") -> tuple[pd.DataFrame, StandardScaler]:
    """
    Полный пайплайн обработки признаков:
    1. Отбор признаков.
    2. Логарифмирование.
    3. Масштабирование (StandardScaler).
    4. Сохранение скалера (если fit_scaler=True).
    
    Args:
        df: Очищенный датафрейм (после clean_data).
        fit_scaler: Обучать ли скалер заново.
        scaler_path: Путь для сохранения/загрузки скалера.
        
    Returns:
        tuple[pd.DataFrame, StandardScaler]: Готовый датафрейм и скалер.
    """
    df_transformed = select_and_transform_features(df)
    
    if fit_scaler:
        df_scaled, scaler = scale_features(df_transformed, fit=True)
        save_scaler(scaler, path=scaler_path)
    else:
        scaler = load_scaler(path=scaler_path)
        df_scaled, _ = scale_features(df_transformed, scaler=scaler, fit=False)
        
    return df_scaled, scaler