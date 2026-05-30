import pandas as pd
import numpy as np
import logging
import yaml
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("configs/training.yaml")

def load_data_config():
    if not CONFIG_PATH.exists():
        return {"fill_median_cols": ["CREDIT_LIMIT", "MINIMUM_PAYMENTS"]}
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)['data']

DATA_CFG = load_data_config()
FILL_MEDIAN_COLS = DATA_CFG['fill_median_cols']

DEFAULT_DATA_PATH = "data/credit_cards_dataset.csv"

def load_and_clean_data(file_path: Optional[str] = DEFAULT_DATA_PATH, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Загружает данные из CSV или принимает готовый DataFrame.
    Выполняет базовую очистку:
    1. Удаляет CUST_ID.
    2. Заполняет пропуски в CREDIT_LIMIT и MINIMUM_PAYMENTS медианой.
    
    Args:
        file_path: Путь к CSV файлу (если df не передан).
        df: Готовый DataFrame (если file_path не передан).
        
    Returns:
        pd.DataFrame: Очищенный датафрейм.
    """
    if df is None:
        if file_path is None:
            raise ValueError("Необходимо указать либо file_path, либо передать df.")
        logger.info(f"Загрузка данных из {file_path}")
        df = pd.read_csv(file_path)
    
    df_clean = df.copy()
    
    if 'CUST_ID' in df_clean.columns:
        df_clean.drop(['CUST_ID'], axis=1, inplace=True)
        logger.info("Столбец CUST_ID удален.")
    

    for col in FILL_MEDIAN_COLS:
        if col in df_clean.columns:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
            logger.info(f"Пропуски в {col} заполнены медианой ({median_val:.2f}).")
            

    nulls = df_clean.isnull().sum().sum()
    if nulls > 0:
        logger.warning(f"В данных осталось {nulls} пропусков. Они будут заполнены нулями для безопасности.")
        df_clean.fillna(0, inplace=True)
        
    return df_clean

def preprocess_input_json(input_data: dict) -> pd.DataFrame:
    """
    Преобразует JSON от API в DataFrame для обработки.
    """
    df = pd.DataFrame([input_data])

    if 'CUST_ID' in df.columns:
        df.drop(['CUST_ID'], axis=1, inplace=True)
    return df