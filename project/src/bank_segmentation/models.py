import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import joblib
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, List
from .features import FeatureEngineer

logger = logging.getLogger(__name__)


CONFIG_PATH = Path("configs/training.yaml")

def load_training_config():
    if not CONFIG_PATH.exists():
        logger.warning(f"Config file {CONFIG_PATH} not found. Using defaults.")
        return {
            "model": {"params": {"n_clusters": 5, "init": "k-means++", "n_init": 10, "random_state": 42}},
            "features": {"selected_features": []} # Fallback
        }
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

# Загружаем конфиг при импорте модуля
TRAINING_CFG = load_training_config()
BEST_PARAMS = TRAINING_CFG['model']['params']
SELECTED_FEATURES = TRAINING_CFG['features']['selected_features']

# Описание кластеров на основе анализа
CLUSTER_PROFILES = {
    0: {
        "typology": "Новые клиенты",
        "risk_level": "Средний",
        "recommendation": "Увеличить лимит при хорошей истории",
        "description": "Низкий кредитный лимит, часто снимают наличные, редко гасят долг полностью."
    },
    1: {
        "typology": "Наличные",
        "risk_level": "Высокий",
        "recommendation": "Осторожно с кредитованием",
        "description": "Используют карту как источник наличных, высокие лимиты, но платят только минимум."
    },
    2: {
        "typology": "Ответственные клиенты",
        "risk_level": "Низкий",
        "recommendation": "Одобрить кредит, увеличить лимит",
        "description": "Активные покупки, рассрочки, почти всегда гасят долг полностью. Идеальные клиенты."
    },
    3: {
        "typology": "Любители рассрочки",
        "risk_level": "Средний",
        "recommendation": "Предлагать рассрочки",
        "description": "Лояльные клиенты, покупают в рассрочку, но платят только минимум. Приносят доход процентами."
    },
    4: {
        "typology": "Активные пользователи",
        "risk_level": "Средний/Высокий",
        "recommendation": "Индивидуальный подход",
        "description": "Максимально используют карту (покупки + наличные), высокий лимит, но долги не гасят полностью."
    }
}

class SegmentationModel:
    def __init__(self):
        self.model = KMeans(**BEST_PARAMS)
        self.feature_engineer = FeatureEngineer()
        self.is_fitted = False

    def fit(self, df: pd.DataFrame):
        """
        Обучает пайплайн: FeatureEngineer + KMeans.
        """
        logger.info("Обучение модели сегментации...")
        
        X_scaled = self.feature_engineer.fit_transform(df)
        
        self.model.fit(X_scaled)
        self.is_fitted = True
        
        logger.info(f"Модель обучена. Центры кластеров: {self.model.cluster_centers_.shape}")
        
        # Опционально: здесь можно пересчитать CLUSTER_PROFILES на основе реальных центров,
        # чтобы сопоставить label 0 с правильным описанием.
        self._map_clusters_to_profiles(df, X_scaled)

    def _map_clusters_to_profiles(self, df_raw: pd.DataFrame, X_scaled: np.ndarray):
        """
        Этот метод помогает понять, какой физический смысл имеет label 0, 1 и т.д.
        В продакшене вы можете сохранить этот маппинг в файл.
        """
        labels = self.model.predict(X_scaled)
        df_temp = df_raw.copy()
        df_temp['cluster'] = labels
        
        # Выводим средние значения для каждого кластера в лог для проверки
        means = df_temp.groupby('cluster')[SELECTED_FEATURES].mean()
        logger.info(f"Centroids analysis:\n{means}")
        # На основе этого лога вы можете поменять ключи в CLUSTER_PROFILES выше, если нужно.

    def predict(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Предсказывает сегмент для новых данных.
        """
        if not self.is_fitted:
            raise Exception("Модель не обучена!")
            
        X_scaled = self.feature_engineer.transform(df)
        
        clusters = self.model.predict(X_scaled)
        
        results = []
        for i, cluster_id in enumerate(clusters):

            profile = CLUSTER_PROFILES.get(cluster_id, {
                "typology": "Неизвестно",
                "risk_level": "Unknown",
                "recommendation": "Manual Review",
                "description": "Cluster not mapped"
            })
            
            result = {
                "client_index": i,
                "cluster_id": int(cluster_id),
                "typology": profile['typology'],
                "risk_level": profile['risk_level'],
                "recommendation": profile['recommendation'],
                "description": profile['description']
            }
            results.append(result)
            
        return results

    def save(self, model_path: str, scaler_path: str):
        joblib.dump(self.model, model_path)
        self.feature_engineer.save_scaler(scaler_path)
        logger.info("Model and Scaler saved.")

    def load(self, model_path: str, scaler_path: str):
        self.model = joblib.load(model_path)
        self.feature_engineer.load_scaler(scaler_path)
        self.is_fitted = True
        logger.info("Model and Scaler loaded.")