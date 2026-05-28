# src/models.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

class ClusterModel:
    """
    Базовый класс-обертка для моделей кластеризации.
    """
    def __init__(self, name: str, model):
        self.name = name
        self.model = model
        self.labels_ = None
        self.metrics_ = {}

    def fit_predict(self, data: pd.DataFrame):
        """
        Обучает модель и предсказывает кластеры.
        """
        self.labels_ = self.model.fit_predict(data)
        return self.labels_

    def calculate_metrics(self, data: pd.DataFrame):
        """
        Вычисляет метрики качества кластеризации.
        """
        if self.labels_ is None:
            raise ValueError("Model not fitted yet. Call fit_predict first.")
        
        # Silhouette Score требует как минимум 2 кластера и не работает с шумом (-1) в DBSCAN идеально,
        # но sklearn обрабатывает шум, исключая его из расчета, если labels содержит -1.
        # Если все метки -1 (шум), silhouette_score выдаст ошибку.
        unique_labels = set(self.labels_)
        if len(unique_labels) < 2 or (len(unique_labels) == 1 and -1 in unique_labels):
             self.metrics_ = {
                'Silhouette': np.nan,
                'Calinski-Harabasz': np.nan,
                'Davies-Bouldin': np.nan
            }
        else:
            try:
                sil = silhouette_score(data, self.labels_)
                ch = calinski_harabasz_score(data, self.labels_)
                db = davies_bouldin_score(data, self.labels_)
                self.metrics_ = {
                    'Silhouette': sil,
                    'Calinski-Harabasz': ch,
                    'Davies-Bouldin': db
                }
            except ValueError:
                 # Handle case where DBSCAN might label everything as noise or single cluster
                self.metrics_ = {
                    'Silhouette': np.nan,
                    'Calinski-Harabasz': np.nan,
                    'Davies-Bouldin': np.nan
                }
        
        return self.metrics_

    def save_model(self, path: str = "artifacts/model.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"Model {self.name} saved to {path}")

    def load_model(self, path: str = "artifacts/model.pkl"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
        self.model = joblib.load(path)

    def predict_new_data(self, data: pd.DataFrame):
        """
        Предсказывает кластеры для новых данных.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model first.")
        return self.model.predict(data)


def train_and_compare_models(X_scaled: pd.DataFrame, n_clusters: int = 4) -> dict:
    """
    Обучает несколько моделей кластеризации и сравнивает их по метрикам.
    
    Args:
        X_scaled: Масштабированные данные (pd.DataFrame).
        n_clusters: Количество кластеров для KMeans и AgglomerativeClustering.
        
    Returns:
        dict: Словарь с объектами моделей и их метриками.
    """
    models = {
        'KMeans': ClusterModel('KMeans', KMeans(n_clusters=n_clusters, random_state=42, n_init=10)),
        'DBSCAN': ClusterModel('DBSCAN', DBSCAN(eps=0.5, min_samples=5)), # eps и min_samples можно подбирать
        'Agglomerative': ClusterModel('Agglomerative', AgglomerativeClustering(n_clusters=n_clusters, linkage='ward'))
    }
    
    results = {}
    
    for name, model_wrapper in models.items():
        print(f"Training {name}...")
        labels = model_wrapper.fit_predict(X_scaled)
        metrics = model_wrapper.calculate_metrics(X_scaled)
        
        results[name] = {
            'model_wrapper': model_wrapper,
            'labels': labels,
            'metrics': metrics
        }
        
        print(f"Metrics for {name}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
        print("-" * 20)
        
    return results


def select_best_model(results: dict) -> tuple[str, ClusterModel]:
    """
    Выбирает лучшую модель на основе Silhouette Score.
    
    Args:
        results: Словарь с результатами обучения моделей.
        
    Returns:
        tuple[str, ClusterModel]: Имя лучшей модели и её объект-обертка.
    """
    best_score = -1
    best_model_name = None
    best_model_wrapper = None
    
    for name, data in results.items():
        score = data['metrics'].get('Silhouette', -1)
        if not np.isnan(score) and score > best_score:
            best_score = score
            best_model_name = name
            best_model_wrapper = data['model_wrapper']
            
    if best_model_name is None:
        raise ValueError("No valid model found with valid Silhouette Score.")
        
    print(f"Best model: {best_model_name} with Silhouette Score: {best_score:.4f}")
    return best_model_name, best_model_wrapper


def generate_cluster_profiles(original_df: pd.DataFrame, labels: np.ndarray, feature_columns: list) -> pd.DataFrame:
    """
    Генерирует профили кластеров на основе средних значений признаков.
    
    Args:
        original_df: Исходный датафрейм (до масштабирования, но после отбора признаков).
        labels: Метка кластера для каждого клиента.
        feature_columns: Список признаков, используемых для кластеризации.
        
    Returns:
        pd.DataFrame: Датафрейм с профилями кластеров.
    """
    df_with_clusters = original_df[feature_columns].copy()
    df_with_clusters['Cluster'] = labels
    
    cluster_profiles = df_with_clusters.groupby('Cluster').mean()
    
    # Добавим интерпретацию риска и рекомендаций
    risk_assessment = []
    recommendations = []
    
    for cluster_id, row in cluster_profiles.iterrows():
        # Эвристика для оценки риска и рекомендаций
        # Высокий Cash Advance и низкий PRC_FULL_PAYMENT -> Высокий риск
        # Высокий Purchases и высокий PRC_FULL_PAYMENT -> Низкий риск, лояльный клиент
        
        cash_advance_ratio = row['CASH_ADVANCE'] / (row['CREDIT_LIMIT'] + 1e-6) # избегаем деления на ноль
        full_payment_ratio = row['PRC_FULL_PAYMENT']
        purchases_freq = row['PURCHASES_FREQUENCY']
        
        if cash_advance_ratio > 0.5 and full_payment_ratio < 0.5:
            risk = "High"
            recommendation = "Review credit limit, monitor for default"
        elif purchases_freq > 0.8 and full_payment_ratio > 0.8:
            risk = "Low"
            recommendation = "Offer loyalty programs, increase credit limit"
        elif cash_advance_ratio > 0.3:
            risk = "Medium"
            recommendation = "Monitor spending habits"
        else:
            risk = "Low"
            recommendation = "Standard offer"
            
        risk_assessment.append(risk)
        recommendations.append(recommendation)
        
    cluster_profiles['Risk_Assessment'] = risk_assessment
    cluster_profiles['Recommendation'] = recommendations
    
    return cluster_profiles


def predict_client_segment(client_data: pd.DataFrame, model: ClusterModel, scaler: StandardScaler, feature_columns: list, profiles: pd.DataFrame) -> dict:
    """
    Предсказывает сегмент нового клиента и выдает рекомендацию.
    
    Args:
        client_data: DataFrame с данными одного клиента (строка).
        model: Объект обученной модели.
        scaler: Объект скалера.
        feature_columns: Список признаков, используемых для кластеризации.
        profiles: Датафрейм с профилями кластеров.
        
    Returns:
        dict: Словарь с номером кластера, риском и рекомендацией.
    """
    # Отбор признаков
    client_features = client_data[feature_columns]
    
    # Логарифмирование (так же, как при обучении)
    LOG_COLUMNS = [
        'BALANCE', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES',
        'CASH_ADVANCE', 'PAYMENTS', 'MINIMUM_PAYMENTS', 'CASH_ADVANCE_TRX',
        'CREDIT_LIMIT', 'PURCHASES_TRX'
    ]
    for col in LOG_COLUMNS:
        if col in client_features.columns:
            client_features[col] = np.log1p(client_features[col])
            
    # Масштабирование
    client_scaled = scaler.transform(client_features)
    
    # Предсказание кластера
    cluster_label = model.predict_new_data(client_scaled)[0]
    
    # Получение профиля кластера
    if cluster_label in profiles.index:
        profile = profiles.loc[cluster_label]
        risk = profile['Risk_Assessment']
        recommendation = profile['Recommendation']
    else:
        risk = "Unknown"
        recommendation = "Manual review"
        
    return {
        'Cluster': int(cluster_label),
        'Risk_Assessment': risk,
        'Recommendation': recommendation
    }