# project/Dockerfile

# 1. Берем за основу официальный легкий образ Python 3.11
FROM python:3.11-slim

# 2. Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# 3. Копируем файлы зависимостей сначала (это ускоряет повторную сборку, если код изменился, а зависимости нет)
COPY pyproject.toml ./

# 4. Устанавливаем зависимости проекта
# --no-cache-dir уменьшает размер образа
# .[all] может не работать с setuptools напрямую в docker без сборки, поэтому ставим как есть
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# 5. Копируем весь остальной код проекта
# Мы копируем src, configs и data (если данные нужны для обучения внутри, но лучше монтировать)
COPY src/ ./src/
COPY configs/ ./configs/

# 6. Создаем папку для артефактов (она будет пустой при первой сборке)
RUN mkdir -p artifacts

# 7. Открываем порт 8000 (на котором работает FastAPI)
EXPOSE 8000

# 8. Команда запуска сервиса
# Мы запускаем uvicorn, указывая путь к api.py
CMD ["uvicorn", "src.bank_segmentation.api:app", "--host", "0.0.0.0", "--port", "8000"]