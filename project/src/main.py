# В вашем ноутбуке, после загрузки данных и импорта функций из src
import pandas as pd

# 1. Загрузка и предобработка данных
# Предположим, df - это ваш исходный датафрейм
from src.data import load_data, clean_data
from src.features import get_preprocessed_and_scaled_data

# Очищаем данные (предполагается, что у вас есть функция clean_data, если нет, нужно добавить простую очистку от NaN)
# df.dropna(inplace=True) # Простой пример очистки
df = load_data()
df = clean_data(df)
# Получаем масштабированные данные и скалер
X_scaled, scaler = get_preprocessed_and_scaled_data(df, fit_scaler=True)

# Сохраняем скалер для будущего инференса
from src.features import save_scaler
save_scaler(scaler, "artifacts/scaler.pkl")

# 2. Обучение и сравнение моделей
from src.models import train_and_compare_models, select_best_model, generate_cluster_profiles

# Обучаем модели
results = train_and_compare_models(X_scaled, n_clusters=4)

# Выбираем лучшую модель
best_model_name, best_model_wrapper = select_best_model(results)

# Сохраняем лучшую модель
best_model_wrapper.save_model(f"artifacts/{best_model_name}_model.pkl")

# 3. Генерация профилей кластеров
# Нам нужны исходные признаки (не масштабированные) для интерпретации
selected_features = [
    'PURCHASES', 'CASH_ADVANCE', 'INSTALLMENTS_PURCHASES',
    'PURCHASES_FREQUENCY', 'CASH_ADVANCE_FREQUENCY',
    'PRC_FULL_PAYMENT', 'CREDIT_LIMIT', 'TENURE'
]

profiles = generate_cluster_profiles(df[selected_features], best_model_wrapper.labels_, selected_features)
print("Cluster Profiles:")
print(profiles)

# 4. Инференс для нового клиента
from src.models import predict_client_segment, load_scaler

# Пример нового клиента (одна строка)
new_client_data = pd.DataFrame([{
    'PURCHASES': 1000.0,
    'CASH_ADVANCE': 100.0,
    'INSTALLMENTS_PURCHASES': 200.0,
    'PURCHASES_FREQUENCY': 0.9,
    'CASH_ADVANCE_FREQUENCY': 0.1,
    'PRC_FULL_PAYMENT': 0.9,
    'CREDIT_LIMIT': 5000.0,
    'TENURE': 12
}])

# Загружаем скалер и модель
scaler = load_scaler("artifacts/scaler.pkl")
best_model_wrapper.load_model(f"artifacts/{best_model_name}_model.pkl")

# Предсказываем сегмент
result = predict_client_segment(new_client_data, best_model_wrapper, scaler, selected_features, profiles)
print("Prediction for new client:")
print(result)