import os
import sys
import logging
import yaml
from pathlib import Path
import pandas as pd
import joblib

# Добавляем корень проекта в путь, чтобы импорты из src работали корректно
# Предполагается, что train.py запускается из корня проекта: python -m src.train
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

from .data import load_and_clean_data
from .models import SegmentationModel

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("train.log")
    ]
)
logger = logging.getLogger(__name__)

# Пути к данным и артефактам
DATA_PATH = "data/credit_cards_dataset.csv"
ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
CONFIG_PATH = os.path.join(ARTIFACTS_DIR, "config.yaml")

def ensure_dir_exists(dir_path: str):
    """Создает директорию, если она не существует."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        logger.info(f"Директория {dir_path} создана.")

def save_config(config: dict, path: str):
    """Сохраняет конфигурацию в YAML файл."""
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Конфигурация сохранена в {path}")

def main():
    logger.info("="*50)
    logger.info("Запуск процесса обучения модели сегментации")
    logger.info("="*50)

    # 1. Подготовка директории для артефактов
    ensure_dir_exists(ARTIFACTS_DIR)

    # 2. Загрузка и очистка данных
    try:
        logger.info(f"Загрузка данных из {DATA_PATH}...")
        df_raw = load_and_clean_data(file_path=DATA_PATH)
        logger.info(f"Данные загружены. Размер: {df_raw.shape}")
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных: {e}")
        sys.exit(1)

    # 3. Инициализация и обучение модели
    try:
        logger.info("Инициализация модели сегментации...")
        model_wrapper = SegmentationModel()
        
        logger.info("Начало обучения (fit)...")
        model_wrapper.fit(df_raw)
        logger.info("Обучение завершено успешно.")
    except Exception as e:
        logger.error(f"Ошибка при обучении модели: {e}")
        sys.exit(1)

    # 4. Сохранение артефактов
    try:
        logger.info("Сохранение модели и скейлера...")
        model_wrapper.save(MODEL_PATH, SCALER_PATH)
        
        # Сохраняем конфиг с параметрами модели и списком признаков
        config = {
            "model_type": "KMeans",
            "parameters": BEST_PARAMS,
            "features": SELECTED_FEATURES,
            "input_shape": list(df_raw.shape),
            "description": "Модель сегментации клиентов банка на основе транзакционных данных."
        }
        save_config(config, CONFIG_PATH)
        
        logger.info("Все артефакты успешно сохранены.")
    except Exception as e:
        logger.error(f"Ошибка при сохранении артефактов: {e}")
        sys.exit(1)

    # 5. Демонстрация работы на примере (инференс)
    try:
        logger.info("Тестовый инференс на первых 5 клиентах...")
        sample_data = df_raw.head(5)
        predictions = model_wrapper.predict(sample_data)
        
        for pred in predictions:
            logger.info(f"Клиент {pred['client_index']}: Кластер {pred['cluster_id']} | "
                        f"Тип: {pred['typology']} | Риск: {pred['risk_level']}")
                        
    except Exception as e:
        logger.error(f"Ошибка при тестовом инференсе: {e}")

    logger.info("="*50)
    logger.info("Процесс обучения завершен.")
    logger.info("="*50)

if __name__ == "__main__":
    main()