import os
import sys
import logging
import yaml
from pathlib import Path
import pandas as pd
import joblib
from dotenv import load_dotenv

# Добавляем корень проекта в PYTHONPATH, если запускаем скрипт напрямую
if __name__ == "__main__":
    project_root = str(Path(__file__).resolve().parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

load_dotenv()

# --- Конфигурация путей из .env ---
DATA_PATH = os.getenv("DATA_PATH", "data/credit_cards_dataset.csv")
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "artifacts")
MODEL_NAME = os.getenv("MODEL_NAME", "best_model.pkl")
SCALER_NAME = os.getenv("SCALER_NAME", "scaler.pkl")

# Формируем полные пути
MODEL_PATH = Path(ARTIFACTS_DIR) / MODEL_NAME
SCALER_PATH = Path(ARTIFACTS_DIR) / SCALER_NAME
# Путь для отчета о последнем обучении (в artifacts, а не в configs!)
LAST_RUN_CONFIG_PATH = Path(ARTIFACTS_DIR) / "training_report.yaml"

# Путь к основному конфигу обучения (источник истины)
TRAINING_CONFIG_PATH = Path("configs/training.yaml")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        # logging.FileHandler("train.log") 
    ]
)
logger = logging.getLogger(__name__)

from bank_segmentation.data import load_and_clean_data
from bank_segmentation.models import SegmentationModel

def load_training_config():
    """Загружает гиперпараметры и списки признаков из YAML конфига."""
    if not TRAINING_CONFIG_PATH.exists():
        logger.error(f"Файл конфигурации не найден: {TRAINING_CONFIG_PATH}")
        raise FileNotFoundError(f"Config file not found: {TRAINING_CONFIG_PATH}")
    
    with open(TRAINING_CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
        
    return config

def ensure_dir_exists(dir_path: Path):
    """Создает директорию, если она не существует."""
    dir_path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Директория проверена/создана: {dir_path}")

def save_training_report(config: dict, path: Path):
    """Сохраняет отчет о параметрах последнего запуска в artifacts."""
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Отчет об обучении сохранен в {path}")

def main():
    logger.info("="*50)
    logger.info("Запуск процесса обучения модели сегментации")
    logger.info("="*50)

    # 0. Загрузка конфигурации обучения
    try:
        training_cfg = load_training_config()
        model_params = training_cfg['model']['params']
        selected_features = training_cfg['features']['selected_features']
        logger.info(f"Конфигурация загружена из {TRAINING_CONFIG_PATH}")
        logger.info(f"Модель: KMeans, Кластеров: {model_params.get('n_clusters')}")
    except Exception as e:
        logger.error(f"Ошибка при чтении конфигурации: {e}")
        sys.exit(1)

    # 1. Подготовка директории для артефактов
    ensure_dir_exists(Path(ARTIFACTS_DIR))

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
        logger.error(f"Ошибка при обучении модели: {e}", exc_info=True)
        sys.exit(1)

    # 4. Сохранение артефактов
    try:
        logger.info(f"Сохранение модели в {MODEL_PATH} и скалера в {SCALER_PATH}...")
        model_wrapper.save(str(MODEL_PATH), str(SCALER_PATH))
        
        # Сохраняем ОТЧЕТ о том, с какими параметрами прошло это обучение.
        # Мы НЕ перезаписываем configs/training.yaml!
        report_config = {
            "model_type": "KMeans",
            "parameters": model_params,
            "features": selected_features,
            "input_shape": list(df_raw.shape),
            "description": "Отчет о последнем запуске обучения.",
            "timestamp": str(pd.Timestamp.now()) # Можно добавить время для истории
        }
        save_training_report(report_config, LAST_RUN_CONFIG_PATH)
        
        logger.info("Артефакты и отчет успешно сохранены.")
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