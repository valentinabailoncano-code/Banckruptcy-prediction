"""
Este módulo implementa la detección de drift de datos y concepto para
monitorear la degradación del rendimiento del modelo en producción.
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from pathlib import Path
import joblib
import json

# Estadísticas y tests
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency, wasserstein_distance
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Suprimir warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Configurar logging
logger = logging.getLogger(__name__)

@dataclass
class DriftConfig:
    """Configuración para detección de drift"""
    
    # Umbrales de drift
    drift_threshold_warning: float = 0.05  # p-value para warning
    drift_threshold_critical: float = 0.01  # p-value para crítico
    
    # Ventanas de comparación
    reference_window_days: int = 90  # Ventana de referencia
    monitoring_window_days: int = 30  # Ventana de monitoreo
    min_samples_per_window: int = 100  # Mínimo de muestras por ventana
    
    # Métodos de detección
    enable_statistical_tests: bool = True
    enable_distribution_drift: bool = True
    enable_performance_drift: bool = True
    enable_prediction_drift: bool = True
    
    # Configuración de tests estadísticos
    categorical_test: str = "chi2"  # "chi2", "cramers_v"
    numerical_test: str = "ks"  # "ks", "wasserstein", "psi"
    
    # Configuración de performance drift
    performance_metrics: List[str] = field(default_factory=lambda: [
        'roc_auc', 'pr_auc', 'brier_score'
    ])
    performance_degradation_threshold: float = 0.05  # 5% degradación
    
    # Configuración de prediction drift
    prediction_shift_threshold: float = 0.1  # 10% cambio en distribución
    
    # Configuración de alertas
    alert_on_warning: bool = True
    alert_on_critical: bool = True
    
    # Configuración de PCA para drift multivariado
    pca_components: int = 10
    pca_variance_threshold: float = 0.95

@dataclass
class DriftResult:
    """Resultado de detección de drift"""
    
    # Identificación
    feature_name: str = ""
    drift_type: str = ""  # "data", "concept", "prediction"
    detection_method: str = ""
    
    # Resultados del test
    test_statistic: float = 0.0
    p_value: float = 1.0
    drift_score: float = 0.0  # Score normalizado 0-1
    
    # Clasificación
    drift_detected: bool = False
    severity: str = "NONE"  # "NONE", "WARNING", "CRITICAL"
    
    # Metadatos
    reference_samples: int = 0
    monitoring_samples: int = 0
    detection_date: datetime = field(default_factory=datetime.now)
    
    # Información adicional
    reference_stats: Optional[Dict[str, float]] = None
    monitoring_stats: Optional[Dict[str, float]] = None
    recommendations: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte el resultado a diccionario"""
        return {
            'feature_name': self.feature_name,
            'drift_type': self.drift_type,
            'detection_method': self.detection_method,
            'test_statistic': self.test_statistic,
            'p_value': self.p_value,
            'drift_score': self.drift_score,
            'drift_detected': self.drift_detected,
            'severity': self.severity,
            'reference_samples': self.reference_samples,
            'monitoring_samples': self.monitoring_samples,
            'detection_date': self.detection_date.isoformat(),
            'reference_stats': self.reference_stats,
            'monitoring_stats': self.monitoring_stats,
            'recommendations': self.recommendations
        }

class StatisticalTests:
    """Colección de tests estadísticos para detección de drift"""
    
    @staticmethod
    def kolmogorov_smirnov_test(
        reference: np.ndarray, 
        monitoring: np.ndarray
    ) -> Tuple[float, float]:
        """
        Test de Kolmogorov-Smirnov para variables numéricas
        
        Args:
            reference: Datos de referencia
            monitoring: Datos de monitoreo
            
        Returns:
            Tuple con (estadística, p-value)
        """
        try:
            # Remover NaN
            ref_clean = reference[~np.isnan(reference)]
            mon_clean = monitoring[~np.isnan(monitoring)]
            
            if len(ref_clean) == 0 or len(mon_clean) == 0:
                return 0.0, 1.0
            
            statistic, p_value = ks_2samp(ref_clean, mon_clean)
            return float(statistic), float(p_value)
            
        except Exception as e:
            logger.error(f"Error en KS test: {str(e)}")
            return 0.0, 1.0
    
    @staticmethod
    def wasserstein_distance_test(
        reference: np.ndarray, 
        monitoring: np.ndarray
    ) -> Tuple[float, float]:
        """
        Test de distancia de Wasserstein (Earth Mover's Distance)
        
        Args:
            reference: Datos de referencia
            monitoring: Datos de monitoreo
            
        Returns:
            Tuple con (distancia, p-value_simulado)
        """
        try:
            ref_clean = reference[~np.isnan(reference)]
            mon_clean = monitoring[~np.isnan(monitoring)]
            
            if len(ref_clean) == 0 or len(mon_clean) == 0:
                return 0.0, 1.0
            
            distance = wasserstein_distance(ref_clean, mon_clean)
            
            # Simular p-value mediante bootstrap
            n_bootstrap = 1000
            bootstrap_distances = []
            
            combined = np.concatenate([ref_clean, mon_clean])
            n_ref = len(ref_clean)
            
            for _ in range(n_bootstrap):
                np.random.shuffle(combined)
                boot_ref = combined[:n_ref]
                boot_mon = combined[n_ref:]
                boot_distance = wasserstein_distance(boot_ref, boot_mon)
                bootstrap_distances.append(boot_distance)
            
            p_value = np.mean(np.array(bootstrap_distances) >= distance)
            
            return float(distance), float(p_value)
            
        except Exception as e:
            logger.error(f"Error en Wasserstein test: {str(e)}")
            return 0.0, 1.0
    
    @staticmethod
    def population_stability_index(
        reference: np.ndarray, 
        monitoring: np.ndarray,
        bins: int = 10
    ) -> float:
        """
        Calcula el Population Stability Index (PSI)
        
        Args:
            reference: Datos de referencia
            monitoring: Datos de monitoreo
            bins: Número de bins para discretización
            
        Returns:
            Valor PSI
        """
        try:
            ref_clean = reference[~np.isnan(reference)]
            mon_clean = monitoring[~np.isnan(monitoring)]
            
            if len(ref_clean) == 0 or len(mon_clean) == 0:
                return 0.0
            
            # Crear bins basados en datos de referencia
            _, bin_edges = np.histogram(ref_clean, bins=bins)
            
            # Calcular distribuciones
            ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
            mon_counts, _ = np.histogram(mon_clean, bins=bin_edges)
            
            # Normalizar a proporciones
            ref_props = ref_counts / len(ref_clean)
            mon_props = mon_counts / len(mon_clean)
            
            # Evitar divisiones por cero
            ref_props = np.where(ref_props == 0, 0.0001, ref_props)
            mon_props = np.where(mon_props == 0, 0.0001, mon_props)
            
            # Calcular PSI
            psi = np.sum((mon_props - ref_props) * np.log(mon_props / ref_props))
            
            return float(psi)
            
        except Exception as e:
            logger.error(f"Error en PSI: {str(e)}")
            return 0.0
    
    @staticmethod
    def chi_square_test(
        reference: np.ndarray, 
        monitoring: np.ndarray
    ) -> Tuple[float, float]:
        """
        Test de Chi-cuadrado para variables categóricas
        
        Args:
            reference: Datos de referencia
            monitoring: Datos de monitoreo
            
        Returns:
            Tuple con (estadística, p-value)
        """
        try:
            # Crear tabla de contingencia
            ref_counts = pd.Series(reference).value_counts()
            mon_counts = pd.Series(monitoring).value_counts()
            
            # Alinear categorías
            all_categories = set(ref_counts.index) | set(mon_counts.index)
            
            ref_aligned = [ref_counts.get(cat, 0) for cat in all_categories]
            mon_aligned = [mon_counts.get(cat, 0) for cat in all_categories]
            
            # Crear tabla de contingencia
            contingency_table = np.array([ref_aligned, mon_aligned])
            
            # Test de chi-cuadrado
            chi2, p_value, _, _ = chi2_contingency(contingency_table)
            
            return float(chi2), float(p_value)
            
        except Exception as e:
            logger.error(f"Error en Chi-square test: {str(e)}")
            return 0.0, 1.0
    
    @staticmethod
    def cramers_v(reference: np.ndarray, monitoring: np.ndarray) -> float:
        """
        Calcula el coeficiente V de Cramér para variables categóricas
        
        Args:
            reference: Datos de referencia
            monitoring: Datos de monitoreo
            
        Returns:
            Valor V de Cramér (0-1)
        """
        try:
            chi2, _, _, _ = StatisticalTests.chi_square_test(reference, monitoring)
            n = len(reference) + len(monitoring)
            
            # Número de categorías
            categories = set(reference) | set(monitoring)
            k = len(categories)
            
            if k <= 1:
                return 0.0
            
            cramers_v = np.sqrt(chi2 / (n * (k - 1)))
            return float(cramers_v)
            
        except Exception as e:
            logger.error(f"Error en Cramér's V: {str(e)}")
            return 0.0

class DataDriftDetector:
    """Detector de drift de datos"""
    
    def __init__(self, config: DriftConfig):
        self.config = config
        self.statistical_tests = StatisticalTests()
    
    def detect_univariate_drift(
        self, 
        reference_data: pd.Series, 
        monitoring_data: pd.Series,
        feature_name: str
    ) -> DriftResult:
        """
        Detecta drift en una característica individual
        
        Args:
            reference_data: Datos de referencia
            monitoring_data: Datos de monitoreo
            feature_name: Nombre de la característica
            
        Returns:
            Resultado de detección de drift
        """
        
        result = DriftResult(
            feature_name=feature_name,
            drift_type="data",
            reference_samples=len(reference_data),
            monitoring_samples=len(monitoring_data)
        )
        
        # Verificar tamaño mínimo de muestras
        if (len(reference_data) < self.config.min_samples_per_window or 
            len(monitoring_data) < self.config.min_samples_per_window):
            result.recommendations = ["Insuficientes muestras para detección confiable"]
            return result
        
        # Determinar tipo de variable
        is_categorical = (
            reference_data.dtype == 'object' or 
            reference_data.dtype.name == 'category' or
            len(reference_data.unique()) < 20
        )
        
        if is_categorical:
            result = self._detect_categorical_drift(
                reference_data, monitoring_data, result
            )
        else:
            result = self._detect_numerical_drift(
                reference_data, monitoring_data, result
            )
        
        # Clasificar severidad
        result.severity = self._classify_severity(result.p_value)
        result.drift_detected = result.severity != "NONE"
        
        return result
    
    def _detect_categorical_drift(
        self, 
        reference: pd.Series, 
        monitoring: pd.Series,
        result: DriftResult
    ) -> DriftResult:
        """Detecta drift en variables categóricas"""
        
        if self.config.categorical_test == "chi2":
            chi2, p_value = self.statistical_tests.chi_square_test(
                reference.values, monitoring.values
            )
            result.test_statistic = chi2
            result.p_value = p_value
            result.detection_method = "Chi-square test"
            
        elif self.config.categorical_test == "cramers_v":
            cramers_v = self.statistical_tests.cramers_v(
                reference.values, monitoring.values
            )
            result.test_statistic = cramers_v
            result.drift_score = cramers_v
            result.detection_method = "Cramér's V"
            # Para Cramér's V, usar umbral directo en lugar de p-value
            result.p_value = 1 - cramers_v
        
        # Estadísticas descriptivas
        ref_counts = reference.value_counts(normalize=True)
        mon_counts = monitoring.value_counts(normalize=True)
        
        result.reference_stats = ref_counts.to_dict()
        result.monitoring_stats = mon_counts.to_dict()
        
        return result
    
    def _detect_numerical_drift(
        self, 
        reference: pd.Series, 
        monitoring: pd.Series,
        result: DriftResult
    ) -> DriftResult:
        """Detecta drift en variables numéricas"""
        
        ref_values = reference.dropna().values
        mon_values = monitoring.dropna().values
        
        if self.config.numerical_test == "ks":
            statistic, p_value = self.statistical_tests.kolmogorov_smirnov_test(
                ref_values, mon_values
            )
            result.test_statistic = statistic
            result.p_value = p_value
            result.detection_method = "Kolmogorov-Smirnov test"
            
        elif self.config.numerical_test == "wasserstein":
            distance, p_value = self.statistical_tests.wasserstein_distance_test(
                ref_values, mon_values
            )
            result.test_statistic = distance
            result.p_value = p_value
            result.detection_method = "Wasserstein distance"
            
        elif self.config.numerical_test == "psi":
            psi = self.statistical_tests.population_stability_index(
                ref_values, mon_values
            )
            result.test_statistic = psi
            result.drift_score = psi
            result.detection_method = "Population Stability Index"
            # Para PSI, convertir a p-value aproximado
            result.p_value = max(0.001, 1 / (1 + psi))
        
        # Estadísticas descriptivas
        result.reference_stats = {
            'mean': float(np.mean(ref_values)),
            'std': float(np.std(ref_values)),
            'median': float(np.median(ref_values)),
            'min': float(np.min(ref_values)),
            'max': float(np.max(ref_values))
        }
        
        result.monitoring_stats = {
            'mean': float(np.mean(mon_values)),
            'std': float(np.std(mon_values)),
            'median': float(np.median(mon_values)),
            'min': float(np.min(mon_values)),
            'max': float(np.max(mon_values))
        }
        
        return result
    
    def detect_multivariate_drift(
        self, 
        reference_data: pd.DataFrame, 
        monitoring_data: pd.DataFrame
    ) -> DriftResult:
        """
        Detecta drift multivariado usando PCA
        
        Args:
            reference_data: Datos de referencia
            monitoring_data: Datos de monitoreo
            
        Returns:
            Resultado de detección de drift multivariado
        """
        
        result = DriftResult(
            feature_name="multivariate",
            drift_type="data",
            detection_method="PCA + KS test",
            reference_samples=len(reference_data),
            monitoring_samples=len(monitoring_data)
        )
        
        try:
            # Seleccionar características numéricas
            numeric_cols = reference_data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                result.recommendations = ["No hay características numéricas para análisis multivariado"]
                return result
            
            ref_numeric = reference_data[numeric_cols].fillna(0)
            mon_numeric = monitoring_data[numeric_cols].fillna(0)
            
            # Estandarizar datos
            scaler = StandardScaler()
            ref_scaled = scaler.fit_transform(ref_numeric)
            mon_scaled = scaler.transform(mon_numeric)
            
            # Aplicar PCA
            pca = PCA(n_components=min(self.config.pca_components, len(numeric_cols)))
            ref_pca = pca.fit_transform(ref_scaled)
            mon_pca = pca.transform(mon_scaled)
            
            # Test KS en primera componente principal
            statistic, p_value = self.statistical_tests.kolmogorov_smirnov_test(
                ref_pca[:, 0], mon_pca[:, 0]
            )
            
            result.test_statistic = statistic
            result.p_value = p_value
            result.drift_score = statistic
            
            # Información adicional
            result.reference_stats = {
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist()
            }
            
        except Exception as e:
            logger.error(f"Error en detección multivariada: {str(e)}")
            result.recommendations = [f"Error en análisis: {str(e)}"]
        
        result.severity = self._classify_severity(result.p_value)
        result.drift_detected = result.severity != "NONE"
        
        return result
    
    def _classify_severity(self, p_value: float) -> str:
        """Clasifica la severidad del drift basado en p-value"""
        
        if p_value <= self.config.drift_threshold_critical:
            return "CRITICAL"
        elif p_value <= self.config.drift_threshold_warning:
            return "WARNING"
        else:
            return "NONE"

class ConceptDriftDetector:
    """Detector de drift de concepto (cambios en la relación X->y)"""
    
    def __init__(self, config: DriftConfig):
        self.config = config
    
    def detect_performance_drift(
        self,
        model: Any,
        reference_data: pd.DataFrame,
        monitoring_data: pd.DataFrame,
        target_column: str,
        feature_columns: List[str]
    ) -> List[DriftResult]:
        """
        Detecta drift de concepto mediante degradación del rendimiento
        
        Args:
            model: Modelo entrenado
            reference_data: Datos de referencia con target
            monitoring_data: Datos de monitoreo con target
            target_column: Nombre de la columna objetivo
            feature_columns: Lista de características
            
        Returns:
            Lista de resultados de drift por métrica
        """
        
        results = []
        
        try:
            # Preparar datos
            X_ref = reference_data[feature_columns]
            y_ref = reference_data[target_column]
            X_mon = monitoring_data[feature_columns]
            y_mon = monitoring_data[target_column]
            
            # Predicciones
            y_ref_pred = model.predict_proba(X_ref)[:, 1]
            y_mon_pred = model.predict_proba(X_mon)[:, 1]
            
            # Calcular métricas para cada conjunto
            for metric_name in self.config.performance_metrics:
                result = DriftResult(
                    feature_name=metric_name,
                    drift_type="concept",
                    detection_method="Performance comparison"
                )
                
                ref_metric = self._calculate_metric(metric_name, y_ref, y_ref_pred)
                mon_metric = self._calculate_metric(metric_name, y_mon, y_mon_pred)
                
                # Calcular degradación
                if metric_name == 'brier_score':
                    # Para Brier score, menor es mejor
                    degradation = (mon_metric - ref_metric) / ref_metric
                else:
                    # Para ROC-AUC y PR-AUC, mayor es mejor
                    degradation = (ref_metric - mon_metric) / ref_metric
                
                result.test_statistic = degradation
                result.drift_score = abs(degradation)
                
                # Clasificar drift
                if abs(degradation) >= self.config.performance_degradation_threshold:
                    result.drift_detected = True
                    result.severity = "CRITICAL" if abs(degradation) >= 0.1 else "WARNING"
                
                result.reference_stats = {metric_name: ref_metric}
                result.monitoring_stats = {metric_name: mon_metric}
                
                results.append(result)
                
        except Exception as e:
            logger.error(f"Error en detección de concept drift: {str(e)}")
            error_result = DriftResult(
                feature_name="performance_drift",
                drift_type="concept",
                detection_method="Performance comparison"
            )
            error_result.recommendations = [f"Error en análisis: {str(e)}"]
            results.append(error_result)
        
        return results
    
    def _calculate_metric(self, metric_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcula una métrica específica"""
        
        try:
            if metric_name == 'roc_auc':
                return roc_auc_score(y_true, y_pred)
            elif metric_name == 'pr_auc':
                return average_precision_score(y_true, y_pred)
            elif metric_name == 'brier_score':
                from sklearn.metrics import brier_score_loss
                return brier_score_loss(y_true, y_pred)
            else:
                return 0.0
        except Exception:
            return 0.0

class PredictionDriftDetector:
    """Detector de drift en las predicciones del modelo"""
    
    def __init__(self, config: DriftConfig):
        self.config = config
        self.statistical_tests = StatisticalTests()
    
    def detect_prediction_drift(
        self,
        reference_predictions: np.ndarray,
        monitoring_predictions: np.ndarray
    ) -> DriftResult:
        """
        Detecta drift en la distribución de predicciones
        
        Args:
            reference_predictions: Predicciones de referencia
            monitoring_predictions: Predicciones de monitoreo
            
        Returns:
            Resultado de detección de drift
        """
        
        result = DriftResult(
            feature_name="predictions",
            drift_type="prediction",
            detection_method="KS test on predictions",
            reference_samples=len(reference_predictions),
            monitoring_samples=len(monitoring_predictions)
        )
        
        try:
            # Test KS en predicciones
            statistic, p_value = self.statistical_tests.kolmogorov_smirnov_test(
                reference_predictions, monitoring_predictions
            )
            
            result.test_statistic = statistic
            result.p_value = p_value
            result.drift_score = statistic
            
            # Estadísticas de predicciones
            result.reference_stats = {
                'mean_prediction': float(np.mean(reference_predictions)),
                'std_prediction': float(np.std(reference_predictions)),
                'median_prediction': float(np.median(reference_predictions))
            }
            
            result.monitoring_stats = {
                'mean_prediction': float(np.mean(monitoring_predictions)),
                'std_prediction': float(np.std(monitoring_predictions)),
                'median_prediction': float(np.median(monitoring_predictions))
            }
            
            # Clasificar severidad
            if p_value <= self.config.drift_threshold_critical:
                result.severity = "CRITICAL"
            elif p_value <= self.config.drift_threshold_warning:
                result.severity = "WARNING"
            else:
                result.severity = "NONE"
            
            result.drift_detected = result.severity != "NONE"
            
        except Exception as e:
            logger.error(f"Error en detección de prediction drift: {str(e)}")
            result.recommendations = [f"Error en análisis: {str(e)}"]
        
        return result

class DriftMonitor:
    """Monitor principal de drift que coordina todos los detectores"""
    
    def __init__(self, config: Optional[DriftConfig] = None):
        self.config = config or DriftConfig()
        self.data_detector = DataDriftDetector(self.config)
        self.concept_detector = ConceptDriftDetector(self.config)
        self.prediction_detector = PredictionDriftDetector(self.config)
        
        # Historial de detecciones
        self.drift_history = []
    
    def monitor_drift(
        self,
        reference_data: pd.DataFrame,
        monitoring_data: pd.DataFrame,
        model: Optional[Any] = None,
        target_column: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
        reference_predictions: Optional[np.ndarray] = None,
        monitoring_predictions: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Ejecuta monitoreo completo de drift
        
        Args:
            reference_data: Datos de referencia
            monitoring_data: Datos de monitoreo
            model: Modelo entrenado (opcional)
            target_column: Columna objetivo (opcional)
            feature_columns: Lista de características (opcional)
            reference_predictions: Predicciones de referencia (opcional)
            monitoring_predictions: Predicciones de monitoreo (opcional)
            
        Returns:
            Diccionario con resultados completos de drift
        """
        
        logger.info("Iniciando monitoreo de drift")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'data_drift': [],
            'concept_drift': [],
            'prediction_drift': None,
            'multivariate_drift': None,
            'summary': {
                'total_features_analyzed': 0,
                'features_with_drift': 0,
                'critical_drifts': 0,
                'warning_drifts': 0
            }
        }
        
        # 1. Detección de drift de datos (univariado)
        if self.config.enable_distribution_drift:
            results['data_drift'] = self._analyze_data_drift(
                reference_data, monitoring_data, feature_columns
            )
        
        # 2. Detección de drift multivariado
        if self.config.enable_distribution_drift:
            results['multivariate_drift'] = self.data_detector.detect_multivariate_drift(
                reference_data, monitoring_data
            ).to_dict()
        
        # 3. Detección de drift de concepto
        if (self.config.enable_performance_drift and model is not None and 
            target_column is not None and target_column in reference_data.columns and
            target_column in monitoring_data.columns):
            
            concept_results = self.concept_detector.detect_performance_drift(
                model, reference_data, monitoring_data, target_column, 
                feature_columns or [col for col in reference_data.columns if col != target_column]
            )
            results['concept_drift'] = [r.to_dict() for r in concept_results]
        
        # 4. Detección de drift de predicciones
        if (self.config.enable_prediction_drift and reference_predictions is not None and
            monitoring_predictions is not None):
            
            pred_drift = self.prediction_detector.detect_prediction_drift(
                reference_predictions, monitoring_predictions
            )
            results['prediction_drift'] = pred_drift.to_dict()
        
        # 5. Generar resumen
        results['summary'] = self._generate_summary(results)
        
        # 6. Guardar en historial
        self.drift_history.append(results)
        
        logger.info(f"Monitoreo completado. Características analizadas: {results['summary']['total_features_analyzed']}")
        logger.info(f"Drift detectado en: {results['summary']['features_with_drift']} características")
        
        return results
    
    def _analyze_data_drift(
        self,
        reference_data: pd.DataFrame,
        monitoring_data: pd.DataFrame,
        feature_columns: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Analiza drift de datos para todas las características"""
        
        if feature_columns is None:
            feature_columns = [col for col in reference_data.columns 
                             if col in monitoring_data.columns]
        
        drift_results = []
        
        for feature in feature_columns:
            if feature in reference_data.columns and feature in monitoring_data.columns:
                try:
                    result = self.data_detector.detect_univariate_drift(
                        reference_data[feature],
                        monitoring_data[feature],
                        feature
                    )
                    drift_results.append(result.to_dict())
                    
                except Exception as e:
                    logger.error(f"Error analizando drift en {feature}: {str(e)}")
        
        return drift_results
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Genera resumen de resultados de drift"""
        
        summary = {
            'total_features_analyzed': len(results['data_drift']),
            'features_with_drift': 0,
            'critical_drifts': 0,
            'warning_drifts': 0,
            'drift_by_type': {
                'data': 0,
                'concept': 0,
                'prediction': 0
            }
        }
        
        # Contar drifts de datos
        for drift in results['data_drift']:
            if drift['drift_detected']:
                summary['features_with_drift'] += 1
                summary['drift_by_type']['data'] += 1
                
                if drift['severity'] == 'CRITICAL':
                    summary['critical_drifts'] += 1
                elif drift['severity'] == 'WARNING':
                    summary['warning_drifts'] += 1
        
        # Contar drifts de concepto
        for drift in results.get('concept_drift', []):
            if drift['drift_detected']:
                summary['drift_by_type']['concept'] += 1
                
                if drift['severity'] == 'CRITICAL':
                    summary['critical_drifts'] += 1
                elif drift['severity'] == 'WARNING':
                    summary['warning_drifts'] += 1
        
        # Contar drift de predicciones
        pred_drift = results.get('prediction_drift')
        if pred_drift and pred_drift['drift_detected']:
            summary['drift_by_type']['prediction'] += 1
            
            if pred_drift['severity'] == 'CRITICAL':
                summary['critical_drifts'] += 1
            elif pred_drift['severity'] == 'WARNING':
                summary['warning_drifts'] += 1
        
        return summary
    
    def get_drift_report(self, last_n_days: int = 30) -> Dict[str, Any]:
        """
        Genera reporte de drift para los últimos N días
        
        Args:
            last_n_days: Número de días a incluir en el reporte
            
        Returns:
            Reporte de drift
        """
        
        cutoff_date = datetime.now() - timedelta(days=last_n_days)
        
        recent_history = [
            entry for entry in self.drift_history
            if datetime.fromisoformat(entry['timestamp']) >= cutoff_date
        ]
        
        if not recent_history:
            return {'message': 'No hay datos de drift en el período especificado'}
        
        # Agregar estadísticas
        total_analyses = len(recent_history)
        total_drifts = sum(entry['summary']['features_with_drift'] for entry in recent_history)
        critical_drifts = sum(entry['summary']['critical_drifts'] for entry in recent_history)
        
        # Características más problemáticas
        feature_drift_counts = {}
        for entry in recent_history:
            for drift in entry['data_drift']:
                if drift['drift_detected']:
                    feature = drift['feature_name']
                    feature_drift_counts[feature] = feature_drift_counts.get(feature, 0) + 1
        
        most_problematic = sorted(
            feature_drift_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        report = {
            'period_days': last_n_days,
            'total_analyses': total_analyses,
            'total_drifts_detected': total_drifts,
            'critical_drifts': critical_drifts,
            'most_problematic_features': most_problematic,
            'recent_history': recent_history[-5:]  # Últimas 5 entradas
        }
        
        return report
    
    def save_drift_history(self, filepath: Union[str, Path]):
        """Guarda el historial de drift"""
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.drift_history, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Historial de drift guardado en {filepath}")
    
    def load_drift_history(self, filepath: Union[str, Path]):
        """Carga el historial de drift"""
        
        with open(filepath, 'r', encoding='utf-8') as f:
            self.drift_history = json.load(f)
        
        logger.info(f"Historial de drift cargado desde {filepath}")

# Función de utilidad para monitoreo rápido
def monitor_model_drift(
    reference_data: pd.DataFrame,
    monitoring_data: pd.DataFrame,
    model: Optional[Any] = None,
    target_column: Optional[str] = None,
    config: Optional[DriftConfig] = None
) -> Dict[str, Any]:
    """
    Función de utilidad para monitoreo rápido de drift
    
    Args:
        reference_data: Datos de referencia
        monitoring_data: Datos de monitoreo
        model: Modelo entrenado (opcional)
        target_column: Columna objetivo (opcional)
        config: Configuración de drift (opcional)
        
    Returns:
        Resultados de monitoreo de drift
    """
    
    monitor = DriftMonitor(config)
    
    return monitor.monitor_drift(
        reference_data=reference_data,
        monitoring_data=monitoring_data,
        model=model,
        target_column=target_column
    )

if __name__ == "__main__":
    # Ejemplo de uso
    import pandas as pd
    from sklearn.datasets import make_classification
    
    # Crear datos sintéticos
    X, y = make_classification(
        n_samples=2000, n_features=10, n_informative=8,
        n_redundant=2, random_state=42
    )
    
    # Crear DataFrames
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    df['date'] = pd.date_range('2023-01-01', periods=len(df), freq='D')
    
    # Dividir en referencia y monitoreo
    split_point = len(df) // 2
    reference_data = df.iloc[:split_point].copy()
    monitoring_data = df.iloc[split_point:].copy()
    
    # Introducir drift artificial en algunas características
    monitoring_data['feature_0'] = monitoring_data['feature_0'] + 2  # Shift en media
    monitoring_data['feature_1'] = monitoring_data['feature_1'] * 1.5  # Cambio en varianza
    
    # Ejecutar monitoreo
    results = monitor_model_drift(
        reference_data=reference_data,
        monitoring_data=monitoring_data,
        target_column='target'
    )
    
    print("Monitoreo de drift completado:")
    print(f"Características analizadas: {results['summary']['total_features_analyzed']}")
    print(f"Drift detectado en: {results['summary']['features_with_drift']} características")
    print(f"Drifts críticos: {results['summary']['critical_drifts']}")
    print(f"Drifts de advertencia: {results['summary']['warning_drifts']}")
    
    # Mostrar características con drift
    for drift in results['data_drift']:
        if drift['drift_detected']:
            print(f"- {drift['feature_name']}: {drift['severity']} (p-value: {drift['p_value']:.4f})")

