from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
import logging

from dotenv import load_dotenv

load_dotenv() 

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Импорт компонентов проекта
# Так как мы находимся внутри пакета bank_segmentation, используем относительные импорты
from .models import SegmentationModel
from .data import load_and_clean_data

app = typer.Typer(help="CLI для сервиса сегментации клиентов банка")

# Константы путей (относительно корня проекта, откуда запускается команда)
ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "best_model.pkl"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
DEFAULT_DATA_PATH = Path("data/credit_cards_dataset.csv")


def _ensure_artifacts_dir():
    """Создает директорию artifacts, если она не существует."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


@app.command()
def train(
    data_path: str = typer.Option(str(DEFAULT_DATA_PATH), help="Путь к CSV файлу с данными."),
):
    """
    Обучить модель сегментации и сохранить артефакты (модель + скалер).
    """
    data_file = Path(data_path)
    if not data_file.exists():
        typer.echo(f"[Ошибка] Файл данных не найден: {data_file}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"[INFO] Загрузка данных из {data_file}...")
    try:
        df = load_and_clean_data(file_path=str(data_file))
        typer.echo(f"[INFO] Данные загружены. Размер: {df.shape}")
    except Exception as e:
        typer.echo(f"[Ошибка] Не удалось загрузить данные: {e}", err=True)
        raise typer.Exit(code=1)

    typer.echo("[INFO] Инициализация и обучение модели...")
    try:
        model_wrapper = SegmentationModel()
        model_wrapper.fit(df)
        typer.echo("[INFO] Обучение завершено успешно.")
    except Exception as e:
        typer.echo(f"[Ошибка] Ошибка при обучении: {e}", err=True)
        raise typer.Exit(code=1)

    _ensure_artifacts_dir()
    
    typer.echo(f"[INFO] Сохранение артефактов в {ARTIFACTS_DIR}...")
    try:
        model_wrapper.save(str(MODEL_PATH), str(SCALER_PATH))
        typer.echo("[INFO] Модель и скалер сохранены.")
    except Exception as e:
        typer.echo(f"[Ошибка] Не удалось сохранить модель: {e}", err=True)
        raise typer.Exit(code=1)

    typer.echo("\n[Успех] Модель готова к использованию!")
    typer.echo(f"   Модель: {MODEL_PATH}")
    typer.echo(f"   Скалер: {SCALER_PATH}")


@app.command()
def predict(
    input_file: str = typer.Argument(..., help="Путь к JSON файлу с данными клиента."),
):
    """
    Предсказать сегмент для клиента по JSON файлу.
    
    Пример JSON:
    {
      "BALANCE": 4000,
      "PURCHASES": 1000,
      ...
    }
    """
    input_path = Path(input_file)
    if not input_path.exists():
        typer.echo(f"[Ошибка] Файл ввода не найден: {input_file}", err=True)
        raise typer.Exit(code=1)

    # Проверка наличия обученной модели
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        typer.echo("[Ошибка] Модель не найдена. Сначала запустите 'bank-cli train'", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"[INFO] Загрузка модели из {MODEL_PATH}...")
    try:
        model = SegmentationModel()
        model.load(str(MODEL_PATH), str(SCALER_PATH))
    except Exception as e:
        typer.echo(f"[Ошибка] Не удалось загрузить модель: {e}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"[INFO] Чтение данных из {input_file}...")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        
        # Преобразуем JSON в DataFrame (одна строка)
        df_input = pd.DataFrame([input_data])
        
        # Предсказание
        results = model.predict(df_input)
        
        # Вывод результата в красивом JSON
        typer.echo(json.dumps(results[0], indent=2, ensure_ascii=False))
        
    except json.JSONDecodeError:
        typer.echo("[Ошибка] Неверный формат JSON файла.", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"[Ошибка] Ошибка при предсказании: {e}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()