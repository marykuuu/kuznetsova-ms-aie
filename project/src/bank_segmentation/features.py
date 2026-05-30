import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import logging
import yaml
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("configs/training.yaml")

def load_feature_config():
    if not CONFIG_PATH.exists():
        logger.warning(f"Config file {CONFIG_PATH} not found. Using empty lists.")
        return {"log_cols": [], "selected_features": []}
    with open(CONFIG_PATH, 'r') as f:
        cfg = yaml.safe_load(f)
        return cfg['features']

FEATURE_CFG = load_feature_config()
LOG_COLS = FEATURE_CFG['log_cols']
SELECTED_FEATURES = FEATURE_CFG['selected_features']

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def apply_log_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Применяет log1p к финансовым колонкам для уменьшения асимметрии.
        """
        df_proc = df.copy()
        cols_to_log = [col for col in LOG_COLS if col in df_proc.columns]
        
        for col in cols_to_log:
            df_proc[col] = np.log1p(df_proc[col])
            
        return df_proc

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        1. Логарифмирует данные.
        2. Отбирает нужные признаки.
        3. Масштабирует (StandardScaler).
        """
        logger.info("Fit transform features...")
        

        df_log = self.apply_log_transform(df)
        
        missing_feats = [f for f in SELECTED_FEATURES if f not in df_log.columns]
        if missing_feats:
            raise ValueError(f"Отсутствуют необходимые признаки: {missing_feats}")
            
        df_selected = df_log[SELECTED_FEATURES]
        
        scaled_data = self.scaler.fit_transform(df_selected)
        self.is_fitted = True
        
        logger.info("Features fitted and transformed.")
        return scaled_data

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Трансформирует новые данные (использует уже обученный scaler).
        """
        if not self.is_fitted:
            raise Exception("FeatureEngineer не был обучен. Сначала вызовите fit_transform.")
            
        logger.info("Transform new data...")
        
        df_log = self.apply_log_transform(df)
        
        df_selected = df_log[SELECTED_FEATURES]
        
        scaled_data = self.scaler.transform(df_selected)
        
        return scaled_data

    def save_scaler(self, path: str):
        joblib.dump(self.scaler, path)
        logger.info(f"Scaler saved to {path}")

    def load_scaler(self, path: str):
        self.scaler = joblib.load(path)
        self.is_fitted = True
        logger.info(f"Scaler loaded from {path}")