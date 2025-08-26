"""
Este módulo implementa el servicio de predicción con explicabilidad,
blending de scores ML y Altman, y análisis de contribuciones de características.
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from datetime import datetime, date
from pathlib import Path
import joblib
import json

# Machine Learning y explicabilidad
from sklearn.base import BaseEstimator
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    import lime
    import lime.lime_tabular
    HAS_LIME = True
except ImportError:
    HAS_LIME = False

# Suprimir warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Configurar logging
logger = logging.getLogger(__name__)

@dataclass
class PredictionConfig:
    """Configuración para el servicio de predicción"""
    
    # Umbrales de clasificación
    threshold_medium: float = 0.15
    threshold_high: float = 0.30
    
    # Configuración de blending
    ml_weight: float = 0.7  # Peso del modelo ML en el blended score
    altman_weight: float = 0.3  # Peso del Altman Z-Score
    
    # Explicabilidad
    enable_explanations: bool = True
    explanation_method: str = "shap"  # "shap", "lime", "permutation"
    top_features_count: int = 10
    
    # Cache de predicciones
    enable_cache: bool = True
    cache_ttl_seconds: int = 3600  # 1 hora
    
    # Validación de inputs
    validate_inputs: bool = True
    min_feature_completeness: float = 0.8  # 80% de características requeridas
    
    # Configuración de batch processing
    batch_size: int = 1000
    parallel_processing: bool = True

@dataclass
class PredictionResult:
    """Resultado de una predicción individual"""
    
    # Identificación
    empresa_id: Optional[int] = None
    fecha_prediccion: datetime = field(default_factory=datetime.now)
    
    # Resultados principales
    probabilidad_ml: float = 0.0
    altman_z_score: float = 0.0
    blended_score: float = 0.0
    
    # Clasificaciones
    banda_riesgo_ml: str = "LOW"  # LOW, MEDIUM, HIGH
    banda_riesgo_altman: str = "SAFE"  # SAFE, GREY, DISTRESS
    banda_riesgo_blended: str = "LOW"
    
    # Explicabilidad
    feature_contributions: Optional[Dict[str, float]] = None
    top_positive_features: Optional[List[Tuple[str, float]]] = None
    top_negative_features: Optional[List[Tuple[str, float]]] = None
    
    # Metadatos
    modelo_version: Optional[str] = None
    tiempo_procesamiento_ms: Optional[float] = None
    confianza_prediccion: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte el resultado a diccionario"""
        return {
            'empresa_id': self.empresa_id,
            'fecha_prediccion': self.fecha_prediccion.isoformat(),
            'probabilidad_ml': self.probabilidad_ml,
            'altman_z_score': self.altman_z_score,
            'blended_score': self.blended_score,
            'banda_riesgo_ml': self.banda_riesgo_ml,
            'banda_riesgo_altman': self.banda_riesgo_altman,
            'banda_riesgo_blended': self.banda_riesgo_blended,
            'feature_contributions': self.feature_contributions,
            'top_positive_features': self.top_positive_features,
            'top_negative_features': self.top_negative_features,
            'modelo_version': self.modelo_version,
            'tiempo_procesamiento_ms': self.tiempo_procesamiento_ms,
            'confianza_prediccion': self.confianza_prediccion
        }

class AltmanCalculator:
    """Calculadora de Altman Z-Score y variantes"""
    
    @staticmethod
    def calculate_altman_original(
        wc_ta: float, re_ta: float, ebit_ta: float, 
        me_tl: float, s_ta: float
    ) -> float:
        """
        Calcula el Altman Z-Score original para empresas públicas
        Z = 1.2*WC/TA + 1.4*RE/TA + 3.3*EBIT/TA + 0.6*ME/TL + 1.0*S/TA
        """
        try:
            z_score = (
                1.2 * (wc_ta or 0) +
                1.4 * (re_ta or 0) +
                3.3 * (ebit_ta or 0) +
                0.6 * (me_tl or 0) +
                1.0 * (s_ta or 0)
            )
            return float(z_score)
        except Exception:
            return np.nan
    
    @staticmethod
    def calculate_altman_modified(
        wc_ta: float, re_ta: float, ebit_ta: float, 
        bv_tl: float, s_ta: float
    ) -> float:
        """
        Calcula el Altman Z-Score modificado para empresas privadas
        Z = 0.717*WC/TA + 0.847*RE/TA + 3.107*EBIT/TA + 0.420*BV/TL + 0.998*S/TA
        """
        try:
            z_score = (
                0.717 * (wc_ta or 0) +
                0.847 * (re_ta or 0) +
                3.107 * (ebit_ta or 0) +
                0.420 * (bv_tl or 0) +
                0.998 * (s_ta or 0)
            )
            return float(z_score)
        except Exception:
            return np.nan
    
    @staticmethod
    def calculate_altman_emerging_markets(
        wc_ta: float, re_ta: float, ebit_ta: float, 
        bv_tl: float, s_ta: float
    ) -> float:
        """
        Calcula el Altman Z-Score para mercados emergentes
        Z = 6.56*WC/TA + 3.26*RE/TA + 6.72*EBIT/TA + 1.05*BV/TL
        """
        try:
            z_score = (
                6.56 * (wc_ta or 0) +
                3.26 * (re_ta or 0) +
                6.72 * (ebit_ta or 0) +
                1.05 * (bv_tl or 0)
            )
            return float(z_score)
        except Exception:
            return np.nan
    
    @staticmethod
    def classify_altman_risk(z_score: float) -> str:
        """
        Clasifica el riesgo según el Altman Z-Score
        
        Args:
            z_score: Valor del Altman Z-Score
            
        Returns:
            Clasificación: "SAFE", "GREY", "DISTRESS"
        """
        if np.isnan(z_score):
            return "UNKNOWN"
        
        if z_score >= 2.99:
            return "SAFE"
        elif z_score >= 1.81:
            return "GREY"
        else:
            return "DISTRESS"
    
    @staticmethod
    def altman_to_probability(z_score: float) -> float:
        """
        Convierte Altman Z-Score a probabilidad de quiebra aproximada
        
        Args:
            z_score: Valor del Altman Z-Score
            
        Returns:
            Probabilidad de quiebra (0-1)
        """
        if np.isnan(z_score):
            return 0.5  # Incertidumbre máxima
        
        # Mapeo empírico basado en estudios históricos
        if z_score >= 2.99:
            return 0.05  # Zona segura: ~5% probabilidad
        elif z_score >= 2.7:
            return 0.10
        elif z_score >= 2.4:
            return 0.15
        elif z_score >= 2.1:
            return 0.25
        elif z_score >= 1.81:
            return 0.40  # Zona gris: 25-50% probabilidad
        elif z_score >= 1.5:
            return 0.60
        elif z_score >= 1.2:
            return 0.75
        elif z_score >= 0.9:
            return 0.85
        else:
            return 0.95  # Zona de distress: >90% probabilidad

class ExplainabilityEngine:
    """Motor de explicabilidad para predicciones"""
    
    def __init__(self, config: PredictionConfig):
        self.config = config
        self.shap_explainer = None
        self.lime_explainer = None
        
    def initialize_explainers(
        self, 
        model: BaseEstimator, 
        X_background: pd.DataFrame
    ):
        """
        Inicializa los explicadores con datos de fondo
        
        Args:
            model: Modelo entrenado
            X_background: Datos de fondo para explicabilidad
        """
        
        if self.config.explanation_method == "shap" and HAS_SHAP:
            try:
                # Usar TreeExplainer para modelos basados en árboles
                if hasattr(model, 'predict_proba'):
                    self.shap_explainer = shap.TreeExplainer(model)
                else:
                    # Fallback a KernelExplainer
                    background_sample = shap.sample(X_background, min(100, len(X_background)))
                    self.shap_explainer = shap.KernelExplainer(
                        model.predict_proba, background_sample
                    )
                logger.info("SHAP explainer inicializado")
            except Exception as e:
                logger.warning(f"Error inicializando SHAP: {str(e)}")
        
        elif self.config.explanation_method == "lime" and HAS_LIME:
            try:
                self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                    X_background.values,
                    feature_names=X_background.columns.tolist(),
                    class_names=['No Quiebra', 'Quiebra'],
                    mode='classification'
                )
                logger.info("LIME explainer inicializado")
            except Exception as e:
                logger.warning(f"Error inicializando LIME: {str(e)}")
    
    def explain_prediction(
        self, 
        model: BaseEstimator, 
        X: pd.DataFrame,
        instance_idx: int = 0
    ) -> Dict[str, float]:
        """
        Explica una predicción específica
        
        Args:
            model: Modelo entrenado
            X: Datos de entrada
            instance_idx: Índice de la instancia a explicar
            
        Returns:
            Diccionario con contribuciones de características
        """
        
        if not self.config.enable_explanations:
            return {}
        
        try:
            if self.config.explanation_method == "shap" and self.shap_explainer:
                return self._explain_with_shap(X, instance_idx)
            elif self.config.explanation_method == "lime" and self.lime_explainer:
                return self._explain_with_lime(model, X, instance_idx)
            else:
                return self._explain_with_permutation(model, X, instance_idx)
        
        except Exception as e:
            logger.error(f"Error en explicabilidad: {str(e)}")
            return {}
    
    def _explain_with_shap(self, X: pd.DataFrame, instance_idx: int) -> Dict[str, float]:
        """Explica usando SHAP values"""
        
        shap_values = self.shap_explainer.shap_values(X.iloc[[instance_idx]])
        
        # Para clasificación binaria, tomar la clase positiva
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Clase positiva (quiebra)
        
        # Crear diccionario de contribuciones
        contributions = dict(zip(X.columns, shap_values[0]))
        
        return contributions
    
    def _explain_with_lime(
        self, 
        model: BaseEstimator, 
        X: pd.DataFrame, 
        instance_idx: int
    ) -> Dict[str, float]:
        """Explica usando LIME"""
        
        instance = X.iloc[instance_idx].values
        
        explanation = self.lime_explainer.explain_instance(
            instance,
            model.predict_proba,
            num_features=self.config.top_features_count
        )
        
        # Extraer contribuciones
        contributions = {}
        for feature_idx, contribution in explanation.as_list():
            feature_name = X.columns[feature_idx]
            contributions[feature_name] = contribution
        
        return contributions
    
    def _explain_with_permutation(
        self, 
        model: BaseEstimator, 
        X: pd.DataFrame, 
        instance_idx: int
    ) -> Dict[str, float]:
        """Explica usando importancia por permutación local"""
        
        # Predicción base
        base_pred = model.predict_proba(X.iloc[[instance_idx]])[:, 1][0]
        
        contributions = {}
        instance = X.iloc[[instance_idx]].copy()
        
        # Para cada característica, permutar y medir impacto
        for col in X.columns:
            # Crear copia con característica permutada
            permuted_instance = instance.copy()
            permuted_instance[col] = X[col].mean()  # Usar media como valor neutral
            
            # Predicción con característica permutada
            permuted_pred = model.predict_proba(permuted_instance)[:, 1][0]
            
            # Contribución = diferencia en predicción
            contributions[col] = base_pred - permuted_pred
        
        return contributions

class InputValidator:
    """Validador de inputs para predicciones"""
    
    def __init__(self, config: PredictionConfig):
        self.config = config
        self.required_features = []
        self.feature_ranges = {}
        
    def set_feature_requirements(
        self, 
        required_features: List[str],
        feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None
    ):
        """
        Establece los requisitos de características
        
        Args:
            required_features: Lista de características requeridas
            feature_ranges: Rangos válidos para cada característica
        """
        self.required_features = required_features
        self.feature_ranges = feature_ranges or {}
    
    def validate_input(self, X: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Valida los datos de entrada
        
        Args:
            X: DataFrame con datos de entrada
            
        Returns:
            Tuple con (es_válido, lista_de_errores)
        """
        
        if not self.config.validate_inputs:
            return True, []
        
        errors = []
        
        # Verificar características requeridas
        missing_features = set(self.required_features) - set(X.columns)
        if missing_features:
            errors.append(f"Características faltantes: {list(missing_features)}")
        
        # Verificar completitud de datos
        available_features = [f for f in self.required_features if f in X.columns]
        if available_features:
            completeness = 1 - X[available_features].isnull().sum().sum() / (len(X) * len(available_features))
            if completeness < self.config.min_feature_completeness:
                errors.append(f"Completitud de datos ({completeness:.2%}) menor al mínimo ({self.config.min_feature_completeness:.2%})")
        
        # Verificar rangos de características
        for feature, (min_val, max_val) in self.feature_ranges.items():
            if feature in X.columns:
                out_of_range = X[feature].notna() & ((X[feature] < min_val) | (X[feature] > max_val))
                if out_of_range.any():
                    errors.append(f"Valores fuera de rango en {feature}: {out_of_range.sum()} casos")
        
        # Verificar tipos de datos
        for feature in available_features:
            if not pd.api.types.is_numeric_dtype(X[feature]):
                errors.append(f"Característica {feature} no es numérica")
        
        is_valid = len(errors) == 0
        return is_valid, errors

class PredictionService:
    """Servicio principal de predicción"""
    
    def __init__(self, config: Optional[PredictionConfig] = None):
        self.config = config or PredictionConfig()
        self.model = None
        self.feature_names = []
        self.altman_calculator = AltmanCalculator()
        self.explainer = ExplainabilityEngine(self.config)
        self.validator = InputValidator(self.config)
        
        # Cache de predicciones
        self.prediction_cache = {}
        
        # Metadatos del modelo
        self.model_version = None
        self.model_metadata = {}
    
    def load_model(self, model_path: Union[str, Path]):
        """
        Carga un modelo entrenado
        
        Args:
            model_path: Ruta al modelo guardado
        """
        
        try:
            model_data = joblib.load(model_path)
            
            self.model = model_data['calibrated_model']
            self.feature_names = model_data['feature_names']
            
            if 'training_history' in model_data:
                self.model_metadata = model_data['training_history']
                self.model_version = self.model_metadata.get('training_date', 'unknown')
            
            # Configurar validador
            self.validator.set_feature_requirements(self.feature_names)
            
            logger.info(f"Modelo cargado desde {model_path}")
            logger.info(f"Características del modelo: {len(self.feature_names)}")
            
        except Exception as e:
            logger.error(f"Error cargando modelo: {str(e)}")
            raise
    
    def initialize_explainability(self, background_data: pd.DataFrame):
        """
        Inicializa el motor de explicabilidad
        
        Args:
            background_data: Datos de fondo para explicabilidad
        """
        
        if self.model is None:
            raise ValueError("Debe cargar un modelo antes de inicializar explicabilidad")
        
        # Asegurar que los datos tengan las características correctas
        background_features = background_data[self.feature_names]
        
        self.explainer.initialize_explainers(self.model, background_features)
    
    def predict_single(
        self, 
        data: Union[pd.DataFrame, Dict[str, Any]],
        empresa_id: Optional[int] = None,
        include_explanations: bool = True
    ) -> PredictionResult:
        """
        Realiza predicción para una empresa individual
        
        Args:
            data: Datos financieros de la empresa
            empresa_id: ID de la empresa (opcional)
            include_explanations: Si incluir explicaciones
            
        Returns:
            Resultado de la predicción
        """
        
        start_time = datetime.now()
        
        if self.model is None:
            raise ValueError("Debe cargar un modelo antes de hacer predicciones")
        
        # Convertir a DataFrame si es necesario
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()
        
        # Validar entrada
        is_valid, errors = self.validator.validate_input(df)
        if not is_valid:
            logger.warning(f"Errores de validación: {errors}")
        
        # Preparar características
        X = self._prepare_features(df)
        
        # Predicción ML
        ml_probability = float(self.model.predict_proba(X)[:, 1][0])
        
        # Calcular Altman Z-Score
        altman_z = self._calculate_altman_score(df.iloc[0])
        altman_probability = self.altman_calculator.altman_to_probability(altman_z)
        
        # Blended score
        blended_score = (
            self.config.ml_weight * ml_probability +
            self.config.altman_weight * altman_probability
        )
        
        # Clasificaciones
        ml_band = self._classify_risk(ml_probability)
        altman_band = self.altman_calculator.classify_altman_risk(altman_z)
        blended_band = self._classify_risk(blended_score)
        
        # Explicabilidad
        feature_contributions = {}
        top_positive = []
        top_negative = []
        
        if include_explanations and self.config.enable_explanations:
            feature_contributions = self.explainer.explain_prediction(self.model, X, 0)
            
            # Ordenar contribuciones
            sorted_contributions = sorted(
                feature_contributions.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            )
            
            # Top características positivas y negativas
            top_positive = [(k, v) for k, v in sorted_contributions if v > 0][:self.config.top_features_count]
            top_negative = [(k, v) for k, v in sorted_contributions if v < 0][:self.config.top_features_count]
        
        # Calcular tiempo de procesamiento
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Crear resultado
        result = PredictionResult(
            empresa_id=empresa_id,
            fecha_prediccion=datetime.now(),
            probabilidad_ml=ml_probability,
            altman_z_score=altman_z,
            blended_score=blended_score,
            banda_riesgo_ml=ml_band,
            banda_riesgo_altman=altman_band,
            banda_riesgo_blended=blended_band,
            feature_contributions=feature_contributions,
            top_positive_features=top_positive,
            top_negative_features=top_negative,
            modelo_version=self.model_version,
            tiempo_procesamiento_ms=processing_time,
            confianza_prediccion=self._calculate_confidence(ml_probability)
        )
        
        return result
    
    def predict_batch(
        self, 
        data: pd.DataFrame,
        include_explanations: bool = False
    ) -> List[PredictionResult]:
        """
        Realiza predicciones en lote
        
        Args:
            data: DataFrame con datos de múltiples empresas
            include_explanations: Si incluir explicaciones (costoso para lotes grandes)
            
        Returns:
            Lista de resultados de predicción
        """
        
        if self.model is None:
            raise ValueError("Debe cargar un modelo antes de hacer predicciones")
        
        results = []
        
        # Procesar en batches
        for i in range(0, len(data), self.config.batch_size):
            batch = data.iloc[i:i + self.config.batch_size]
            
            for idx, row in batch.iterrows():
                empresa_id = row.get('empresa_id', None)
                row_data = pd.DataFrame([row])
                
                try:
                    result = self.predict_single(
                        row_data, 
                        empresa_id=empresa_id,
                        include_explanations=include_explanations
                    )
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error en predicción para empresa {empresa_id}: {str(e)}")
                    # Crear resultado con error
                    error_result = PredictionResult(
                        empresa_id=empresa_id,
                        probabilidad_ml=np.nan,
                        altman_z_score=np.nan,
                        blended_score=np.nan
                    )
                    results.append(error_result)
        
        logger.info(f"Predicciones en lote completadas: {len(results)} resultados")
        return results
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepara las características para predicción"""
        
        # Asegurar que todas las características requeridas estén presentes
        X = pd.DataFrame(index=df.index)
        
        for feature in self.feature_names:
            if feature in df.columns:
                X[feature] = df[feature]
            else:
                X[feature] = 0.0  # Valor por defecto
                logger.warning(f"Característica faltante {feature}, usando valor por defecto")
        
        # Manejar valores faltantes
        X = X.fillna(X.median())
        
        return X
    
    def _calculate_altman_score(self, row: pd.Series) -> float:
        """Calcula el Altman Z-Score para una fila de datos"""
        
        # Intentar con diferentes variantes según disponibilidad de datos
        if all(col in row.index for col in ['wc_ta', 're_ta', 'ebit_ta', 'me_tl', 's_ta']):
            # Altman original (empresas públicas)
            return self.altman_calculator.calculate_altman_original(
                row.get('wc_ta', 0), row.get('re_ta', 0), row.get('ebit_ta', 0),
                row.get('me_tl', 0), row.get('s_ta', 0)
            )
        elif all(col in row.index for col in ['wc_ta', 're_ta', 'ebit_ta', 'bv_tl', 's_ta']):
            # Altman modificado (empresas privadas)
            return self.altman_calculator.calculate_altman_modified(
                row.get('wc_ta', 0), row.get('re_ta', 0), row.get('ebit_ta', 0),
                row.get('bv_tl', 0), row.get('s_ta', 0)
            )
        else:
            logger.warning("Datos insuficientes para calcular Altman Z-Score")
            return np.nan
    
    def _classify_risk(self, probability: float) -> str:
        """Clasifica el riesgo según la probabilidad"""
        
        if np.isnan(probability):
            return "UNKNOWN"
        
        if probability >= self.config.threshold_high:
            return "HIGH"
        elif probability >= self.config.threshold_medium:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _calculate_confidence(self, probability: float) -> float:
        """
        Calcula la confianza de la predicción
        
        La confianza es mayor cuando la probabilidad está cerca de 0 o 1,
        y menor cuando está cerca de 0.5 (incertidumbre máxima)
        """
        
        if np.isnan(probability):
            return 0.0
        
        # Confianza basada en distancia a 0.5
        confidence = 1 - 2 * abs(probability - 0.5)
        return max(0.0, confidence)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retorna información del modelo cargado"""
        
        return {
            'model_loaded': self.model is not None,
            'model_version': self.model_version,
            'feature_count': len(self.feature_names),
            'feature_names': self.feature_names,
            'config': self.config.__dict__,
            'metadata': self.model_metadata
        }
    
    def update_thresholds(
        self, 
        threshold_medium: Optional[float] = None,
        threshold_high: Optional[float] = None
    ):
        """
        Actualiza los umbrales de clasificación
        
        Args:
            threshold_medium: Nuevo umbral medio
            threshold_high: Nuevo umbral alto
        """
        
        if threshold_medium is not None:
            self.config.threshold_medium = threshold_medium
        
        if threshold_high is not None:
            self.config.threshold_high = threshold_high
        
        logger.info(f"Umbrales actualizados: Medio={self.config.threshold_medium}, Alto={self.config.threshold_high}")

# Función de utilidad para predicción rápida
def predict_bankruptcy_risk(
    data: Union[pd.DataFrame, Dict[str, Any]],
    model_path: Union[str, Path],
    config: Optional[PredictionConfig] = None,
    include_explanations: bool = True
) -> Union[PredictionResult, List[PredictionResult]]:
    """
    Función de utilidad para predicción rápida de riesgo de quiebra
    
    Args:
        data: Datos financieros (DataFrame o dict)
        model_path: Ruta al modelo entrenado
        config: Configuración de predicción
        include_explanations: Si incluir explicaciones
        
    Returns:
        Resultado(s) de predicción
    """
    
    service = PredictionService(config)
    service.load_model(model_path)
    
    if isinstance(data, pd.DataFrame) and len(data) > 1:
        return service.predict_batch(data, include_explanations)
    else:
        return service.predict_single(data, include_explanations=include_explanations)

if __name__ == "__main__":
    # Ejemplo de uso
    import pandas as pd
    
    # Datos de ejemplo
    sample_data = {
        'wc_ta': 0.10,
        're_ta': 0.15,
        'ebit_ta': 0.08,
        'me_tl': 1.5,
        's_ta': 1.2,
        'current_ratio': 1.8,
        'debt_assets': 0.45,
        'roa': 0.06
    }
    
    # Crear servicio de predicción
    config = PredictionConfig(
        threshold_medium=0.15,
        threshold_high=0.30,
        enable_explanations=True
    )
    
    service = PredictionService(config)
    
    # Nota: En uso real, cargarías un modelo entrenado
    # service.load_model('path/to/trained/model.pkl')
    
    print("Servicio de predicción inicializado")
    print(f"Configuración: {config.__dict__}")
    
    # Ejemplo de cálculo de Altman Z-Score
    altman_z = service.altman_calculator.calculate_altman_original(
        sample_data['wc_ta'], sample_data['re_ta'], sample_data['ebit_ta'],
        sample_data['me_tl'], sample_data['s_ta']
    )
    
    print(f"Altman Z-Score: {altman_z:.2f}")
    print(f"Clasificación Altman: {service.altman_calculator.classify_altman_risk(altman_z)}")
    print(f"Probabilidad Altman: {service.altman_calculator.altman_to_probability(altman_z):.2%}")

