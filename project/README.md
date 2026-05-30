# Итоговый проект по курсу «Инженерия Искусственного Интеллекта»

В этой папке находится итоговый мини-проект по курсу.  
Проек демонстрирует применение методов и инструментов инженерии ИИ: работу с данными, модели, пайплайны, сервис, эксперименты и (по возможности) воспроизводимость.

---

## 1. Паспорт проекта

- **Название проекта:** `Сервис сегментации клиентов банка и оценки кредитного риска`
- **Автор:** `Кузнецова Мария Сергеевна`
- **Группа:** `ИКБО-40-23`
- **Контакт:** `@marykuuu`

- **Краткое описание:**  
  > Проект представляет собой end-to-end сервис для автоматической сегментации клиентов банка на основе их транзакционной активности. Система принимает финансовые данные клиента (остатки, покупки, платежи), определяет его принадлежность к одному из 5 выявленных кластеров (например, "Ответственные плательщики", "Любители наличных") и выдает бизнес-рекомендации: оценку кредитного риска (низкий/средний/высокий) и стратегию взаимодействия (одобрить кредит, увеличить лимит, отказать). Решение упаковано в REST API с базовой наблюдаемостью и контейнеризировано для легкого развертывания.

---

## 2. Структура проекта

Проект организован в следующей структуре:

- `README.md` - этот файл.
- `pyproject.toml` - зависимости и настройки сборки (setuptools + uv).
- `requirements.txt` – зависимости проекта (библиотеки Python, необходимые для запуска).
- `report.md` – отчёт по проекту (постановка задачи, данные, эксперименты, результаты).
- `self-checklist.md` – чеклист самопроверки проекта перед сдачей.
- `.gitignore` - исключения для Git.
- `.dockerignore` - исключения для Docker.
- `Dockerfile` - инструкция для сборки образа.
- `docker-compose.yml` - оркестрация сервиса.
- `docker-compose.train.yml` - оркестрация задачи обучения.
- `notebooks/` – экспериментальные ноутбуки:
  - `exp_01_eda.ipynb` - EDA, выбор и трансформация признаков.
  - `exp_02_models.ipynb` - выбор лучшей модели.
- `src/` – основной код проекта:
  - `data.py` - модуль загрузки и подготовки данных.
  - `features.py` - модуль отбора и трансформации признаков.
  - `models.py` - лучшая модель и логика предсказания.
  - `train.py` - обучение модели.
  - `cli.py` - Typer CLI интерфейс.
  - `api.py` - FastAPI сервис (REST API).
- `data/` – демонстрационные/учебные данные:
  - `credit_cards_dataset.csv` - датасет транзакций (обезличенный).
  - `test_client.json` - данные 1 клиента для тестирования предсказания.
- `configs/` – конфигурационные файлы:
  - `training.yaml` - гиперпараметры модели и списки признаков.
  - `service.yaml` - настройки сервера и путей.
  - `.env.example` — список переменных окружения без реальных секретов.
- `tests/` – тесты (юнит-тесты, простые проверки):
  - `test_data.py` - тесты очистки данных.
  - `test_features.py` - тесты преобразования признаков.
  - `test_models.py` - тесты модели сегментации.
  - `test_api.py` - тесты API эндпоинтов.
- `artifacts/`
  - `best_model.pkl` - финальная обученная модель K-Means.
  - `best_model_config.yaml` - конфигурации лучшей модели.
  - `scaler.pkl` - объект StandardScaler для нормализации данных.
  - `experiment_summary.csv` - сводная таблица результатов сравнения моделей по метрикам.
  - `figures/` - визуализации из этапа EDA и анализа кластеров.

---

## 3. Требования и установка

### 3.1. Требования

- Python `>= 3.11`.
- Менеджер пакетов uv (рекомендуется) или pip.
- Docker Desktop (для запуска в контейнере).

### 3.2. Установка окружения

Необходимо создать файл `.env` в корневой директории на основе `configs/.env.example`

Рекомендуемый способ через `uv`:

```bash
# Перейти в папку проекта
cd project

# Установить зависимости
uv sync
```
Если используете `pip`:

```bash
# Перейти в папку проекта
cd project

# Создать виртуальное окружение
python -m venv .venv

# Активировать окружение:
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# Установить зависимости
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Как запустить проект

Перед использованием сервиса необходимо обучить модель и сохранить артефакты (best_model.pkl, scaler.pkl).

### 4.1. Запуск обучения модели

Локально через CLI

```bash
# Если используете uv
uv run bank-cli train

# Если активировали venv (pip)
bank-cli train
```

Через Docker Compose (автоматически монтирует папки, чтобы модель сохранилась на компьютере)

```bash
docker-compose -f docker-compose.train.yml up --build
```

Через Docker

```bash
# Сборка образа
docker build -t bank-api .

# Запуск обучения с монтированием томов (Windows PowerShell):
docker run --rm `
    -v "${PWD}/data:/app/data" `
    -v "${PWD}/artifacts:/app/artifacts" `
    bank-api `
    python -m src.bank_segmentation.cli train
```
(Для Linux/macOS замените ` на \ и ${PWD} на $(pwd))

### 4.2. Предсказание через CLI

Если у вас есть JSON-файл с данными клиента (например, `data/test_client.json`):

```bash
# Через uv:
uv run bank-cli predict data/test_client.json

# Через активированный venv:
bank-cli predict data/test_client.json
```

### 4.3. Запуск сервиса (REST API)

Сервис предоставляет HTTP API для интеграции с внешними системами.

Локальный запуск (FastAPI)

```bash
# Через uv:
uv run uvicorn src.bank_segmentation.api:app --reload

# Через активированный venv:
uvicorn src.bank_segmentation.api:app --reload
```

Через Docker Compose

```bash
docker-compose up --build
```

Через Docker

```bash
# Запуск API с монтированием артефактов (Windows PowerShell):
docker run -p 8000:8000 `
    -v "${PWD}/artifacts:/app/artifacts" `
    bank-api
```

(Предварительно убедитесь, что модель обучена, см. пункт 4.1)

- Порт: Сервис поднимается на порту 8000.
- Хост: По умолчанию доступен по адресу http://127.0.0.1:8000 (или http://localhost:8000).

### Эндпоинты сервиса

#### 1. `GET /`

Root

**Запрос:**

```http
GET /
```

**Ожидаемый ответ `200 OK` (JSON):**

```json
{
  "message": "Welcome to Bank Segmentation API",
  "docs": "/docs",
  "health": "/health"
}
```

Пример проверки через `curl`:

```bash
curl -X 'GET' \
  'http://0.0.0.0:8000/' \
  -H 'accept: application/json'
```

#### 2. `GET /health`

Простейший health-check.

**Запрос:**

```http
GET /health
```

**Ожидаемый ответ `200 OK` (JSON):**

```json
{
  "status": "ok",
  "service": "bank-segmentation",
  "version": "1.0.0",
  "model_loaded": true
}
```

Пример проверки через `curl`:

```bash
curl -X 'GET' \
  'http://0.0.0.0:8000/health' \
  -H 'accept: application/json'
```

---

#### 3. Swagger UI: `GET /docs`

Интерфейс документации и тестирования API:

```text
http://127.0.0.1:8000/docs
```

Через `/docs` можно:

- вызывать `GET /`;
- вызывать `GET /health`;
- вызывать `GET /metrics` - метрики Prometheus (счетчики запросов, время ответа);
- вызывать `POST /predict` - основной эндпоинт для получения сегмента и рекомендации.

---

### 3. `GET /metrics` – метрики Prometheus

**Запрос:**

```http
GET /metrics
```

**Пример ответа `200 OK`:**

```text
# HELP python_gc_objects_collected_total Objects collected during gc
# TYPE python_gc_objects_collected_total counter
python_gc_objects_collected_total{generation="0"} 888.0
python_gc_objects_collected_total{generation="1"} 284.0
python_gc_objects_collected_total{generation="2"} 23.0
```

**Пример вызова через `curl`:**

```bash
curl -X 'GET' \
  'http://0.0.0.0:8000/metrics' \
  -H 'accept: application/json'
```

---

### 4. `POST /predict` - основной эндпоинт для получения сегмента и рекомендации

- Вход: JSON с финансовыми показателями клиента (баланс, покупки, лимиты и т.д.).
- Выход: JSON с cluster_id, typology, risk_level, recommendation.

**Пример запроса:**

```http
POST /predict
Content-Type: application/json
```

**Тело:**

```json
{
  "BALANCE": 0,
  "PURCHASES": 0,
  "ONEOFF_PURCHASES": 0,
  "INSTALLMENTS_PURCHASES": 0,
  "CASH_ADVANCE": 0,
  "PAYMENTS": 0,
  "MINIMUM_PAYMENTS": 0,
  "CASH_ADVANCE_TRX": 0,
  "CREDIT_LIMIT": 0,
  "PURCHASES_TRX": 0,
  "PURCHASES_FREQUENCY": 1,
  "ONEOFF_PURCHASES_FREQUENCY": 1,
  "PURCHASES_INSTALLMENTS_FREQUENCY": 1,
  "CASH_ADVANCE_FREQUENCY": 1,
  "PRC_FULL_PAYMENT": 1,
  "TENURE": 0
}
```

**Пример ответа 200 OK:**

```json
{
  "client_index": 0,
  "cluster_id": 0,
  "typology": "Новые клиенты",
  "risk_level": "Средний",
  "recommendation": "Стимулировать активность через кэшбэк и бонусы",
  "description": "Клиенты с низким лимитом и короткой историей. Активно снимают наличные, редко гасят долг полностью. Потенциал роста высок, но требуется контроль."
}
```

Через Swagger:

- в `/docs` открыть `POST /predict`,
- нажать `Try it out`,
- в поле запроса появится пример JSON. Вы можете использовать данные из файла data/test_client.json или заполнить поля вручную.,
- нажать `Execute`;
- в разделе Response body вы увидите результат предсказания (кластер, типология, риск).

**Пример вызова через `curl` (Linux/macOS/WSL):**

```bash
curl -X 'POST' \
'http://0.0.0.0:8000/predict' \
-H 'accept: application/json' \
-H 'Content-Type: application/json' \
-d '{
"BALANCE": 0,
"PURCHASES": 0,
"ONEOFF_PURCHASES": 0,
"INSTALLMENTS_PURCHASES": 0,
"CASH_ADVANCE": 0,
"PAYMENTS": 0,
"MINIMUM_PAYMENTS": 0,
"CASH_ADVANCE_TRX": 0,
"CREDIT_LIMIT": 0,
"PURCHASES_TRX": 0,
"PURCHASES_FREQUENCY": 1,
"ONEOFF_PURCHASES_FREQUENCY": 1,
"PURCHASES_INSTALLMENTS_FREQUENCY": 1,
"CASH_ADVANCE_FREQUENCY": 1,
"PRC_FULL_PAYMENT": 1,
"TENURE": 0
}'
```

Ответ будет содержать:

- `clien_index` - индекс клиента;
- `cluster_id` - id определенного кластера;
- `typology` - Краское название кластера;
- `risk_level` - уровень риска ("Низкий", "Средний", "Высокий");
- `recommendation` - рекомендация;
- `description` - краткая характеристика кластера.

---

## 5. Данные

Кратко опишите используемые данные:

- Источник - открытый датасет "Credit Card Dataset for Clustering"
- Расположение: `data/credit_cards_dataset.csv.`

Датасет небольшой, поэтому полностью загружен в папку `data/`.

При необходимости его можно найти на Kaggle по ссылке `https://www.kaggle.com/datasets/arjunbhasin2013/ccdata`

---

## 6. Тесты

Проект покрыт модульными тестами для проверки пайплайна данных, модели и API.

Пример запуска:

```bash
# Через uv:
uv run pytest -q

# Через активированный venv:
pytest -q
```

---

## 7. Демонстрация на защите

На защите я:

1. Кратко покажу структуру проекта (`notebooks/`, `src/`, `data/`).
2. Запущу сервис через `docker-compose up --build`, покажу пару запросов через Swagger UI.
3. Покажу ноутбук с основными экспериментами и сравнение моделей по метрике качества.

---

## 8. Ограничения и дальнейшая работа

В текущей версии интерпретация кластеров задана экспертно в коде на основе анализа центров кластеров, в сервисе реализована только один метод предсказания и нет авторизации.

В дальнейшем можно добавить:

- несколько алгоритмов с выбором через конфиг;
- добавление модели для автоматической интерпетации кластеров;
- хранение истории запросов пользователей;
- базовую авторизацию.

---

## 9. Оценка проекта

Итоговая оценка за проект выставляется по пятибалльной шкале (2-5).

- **5** – сильный, хорошо проработанный проект:
  - аккуратно реализован сервис и пайплайн;
  - проведены осмысленные эксперименты и обоснован выбор финальной модели;
  - есть базовая наблюдаемость и работа с конфигами/секретами;
  - документация позволяет быстро понять и воспроизвести решение;
  - по чеклисту выполнено **10** пунктов.

---
