"""
Este módulo implementa un sistema de registro de modelos para gestionar
versiones, metadatos, rendimiento y ciclo de vida de modelos ML.
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
import pandas as pd
import numpy as np
from datetime import datetime, date
from pathlib import Path
import joblib
import json
import hashlib
import shutil
import sqlite3
from enum import Enum

# Suprimir warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Configurar logging
logger = logging.getLogger(__name__)

class ModelStatus(Enum):
    """Estados del modelo en el registro"""
    TRAINING = "training"
    VALIDATION = "validation"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"

@dataclass
class ModelMetadata:
    """Metadatos completos del modelo"""
    
    # Identificación
    model_id: str = ""
    model_name: str = ""
    version: str = "1.0.0"
    description: str = ""
    
    # Información del modelo
    algorithm: str = ""
    framework: str = ""
    model_type: str = "classification"
    
    # Datos de entrenamiento
    training_data_hash: str = ""
    training_samples: int = 0
    feature_count: int = 0
    feature_names: List[str] = field(default_factory=list)
    
    # Rendimiento
    metrics: Dict[str, float] = field(default_factory=dict)
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    test_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Configuración
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    preprocessing_config: Dict[str, Any] = field(default_factory=dict)
    
    # Ciclo de vida
    status: ModelStatus = ModelStatus.TRAINING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"
    
    # Archivos
    model_path: str = ""
    artifacts_path: str = ""
    
    # Validación y aprobación
    validation_results: Dict[str, Any] = field(default_factory=dict)
    approval_status: str = "pending"  # pending, approved, rejected
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    
    # Deployment
    deployment_config: Dict[str, Any] = field(default_factory=dict)
    endpoint_url: Optional[str] = None
    
    # Monitoreo
    performance_history: List[Dict[str, Any]] = field(default_factory=list)
    drift_alerts: List[Dict[str, Any]] = field(default_factory=list)
    
    # Tags y categorización
    tags: List[str] = field(default_factory=list)
    business_use_case: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte metadatos a diccionario"""
        data = asdict(self)
        # Convertir datetime a string
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        if self.approved_at:
            data['approved_at'] = self.approved_at.isoformat()
        # Convertir enum a string
        data['status'] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Crea metadatos desde diccionario"""
        # Convertir strings a datetime
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data and isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        if 'approved_at' in data and data['approved_at'] and isinstance(data['approved_at'], str):
            data['approved_at'] = datetime.fromisoformat(data['approved_at'])
        
        # Convertir string a enum
        if 'status' in data and isinstance(data['status'], str):
            data['status'] = ModelStatus(data['status'])
        
        return cls(**data)

@dataclass
class ModelValidationResult:
    """Resultado de validación de modelo"""
    
    model_id: str = ""
    validation_date: datetime = field(default_factory=datetime.now)
    validator: str = ""
    
    # Resultados de tests
    performance_tests: Dict[str, bool] = field(default_factory=dict)
    data_quality_tests: Dict[str, bool] = field(default_factory=dict)
    bias_tests: Dict[str, bool] = field(default_factory=dict)
    robustness_tests: Dict[str, bool] = field(default_factory=dict)
    
    # Resumen
    overall_passed: bool = False
    score: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    
    # Detalles
    detailed_results: Dict[str, Any] = field(default_factory=dict)

class ModelDatabase:
    """Base de datos SQLite para registro de modelos"""
    
    def __init__(self, db_path: Union[str, Path]):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()
    
    def _initialize_database(self):
        """Inicializa la base de datos con las tablas necesarias"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Tabla principal de modelos
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    description TEXT,
                    algorithm TEXT,
                    framework TEXT,
                    model_type TEXT,
                    training_data_hash TEXT,
                    training_samples INTEGER,
                    feature_count INTEGER,
                    feature_names TEXT,
                    metrics TEXT,
                    validation_metrics TEXT,
                    test_metrics TEXT,
                    hyperparameters TEXT,
                    preprocessing_config TEXT,
                    status TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    created_by TEXT,
                    model_path TEXT,
                    artifacts_path TEXT,
                    validation_results TEXT,
                    approval_status TEXT,
                    approved_by TEXT,
                    approved_at TEXT,
                    deployment_config TEXT,
                    endpoint_url TEXT,
                    performance_history TEXT,
                    drift_alerts TEXT,
                    tags TEXT,
                    business_use_case TEXT
                )
            """)
            
            # Tabla de validaciones
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS validations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT,
                    validation_date TEXT,
                    validator TEXT,
                    performance_tests TEXT,
                    data_quality_tests TEXT,
                    bias_tests TEXT,
                    robustness_tests TEXT,
                    overall_passed BOOLEAN,
                    score REAL,
                    recommendations TEXT,
                    detailed_results TEXT,
                    FOREIGN KEY (model_id) REFERENCES models (model_id)
                )
            """)
            
            # Tabla de métricas históricas
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT,
                    timestamp TEXT,
                    metric_name TEXT,
                    metric_value REAL,
                    dataset_type TEXT,
                    FOREIGN KEY (model_id) REFERENCES models (model_id)
                )
            """)
            
            # Índices para búsquedas eficientes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_name ON models (model_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON models (status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON models (created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_performance_model ON performance_history (model_id)")
            
            conn.commit()
    
    def insert_model(self, metadata: ModelMetadata) -> bool:
        """Inserta un nuevo modelo en la base de datos"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Convertir listas y diccionarios a JSON
                data = metadata.to_dict()
                for key in ['feature_names', 'metrics', 'validation_metrics', 'test_metrics',
                           'hyperparameters', 'preprocessing_config', 'validation_results',
                           'deployment_config', 'performance_history', 'drift_alerts', 'tags']:
                    if key in data:
                        data[key] = json.dumps(data[key])
                
                # Insertar
                columns = ', '.join(data.keys())
                placeholders = ', '.join(['?' for _ in data])
                query = f"INSERT INTO models ({columns}) VALUES ({placeholders})"
                
                cursor.execute(query, list(data.values()))
                conn.commit()
                
                logger.info(f"Modelo {metadata.model_id} insertado en la base de datos")
                return True
                
        except Exception as e:
            logger.error(f"Error insertando modelo: {str(e)}")
            return False
    
    def update_model(self, metadata: ModelMetadata) -> bool:
        """Actualiza un modelo existente"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Actualizar timestamp
                metadata.updated_at = datetime.now()
                
                # Convertir a diccionario
                data = metadata.to_dict()
                for key in ['feature_names', 'metrics', 'validation_metrics', 'test_metrics',
                           'hyperparameters', 'preprocessing_config', 'validation_results',
                           'deployment_config', 'performance_history', 'drift_alerts', 'tags']:
                    if key in data:
                        data[key] = json.dumps(data[key])
                
                # Construir query de actualización
                set_clause = ', '.join([f"{key} = ?" for key in data.keys() if key != 'model_id'])
                query = f"UPDATE models SET {set_clause} WHERE model_id = ?"
                
                values = [data[key] for key in data.keys() if key != 'model_id']
                values.append(metadata.model_id)
                
                cursor.execute(query, values)
                conn.commit()
                
                logger.info(f"Modelo {metadata.model_id} actualizado")
                return True
                
        except Exception as e:
            logger.error(f"Error actualizando modelo: {str(e)}")
            return False
    
    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Obtiene un modelo por ID"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM models WHERE model_id = ?", (model_id,))
                row = cursor.fetchone()
                
                if row:
                    # Obtener nombres de columnas
                    columns = [description[0] for description in cursor.description]
                    data = dict(zip(columns, row))
                    
                    # Convertir JSON strings de vuelta a objetos
                    for key in ['feature_names', 'metrics', 'validation_metrics', 'test_metrics',
                               'hyperparameters', 'preprocessing_config', 'validation_results',
                               'deployment_config', 'performance_history', 'drift_alerts', 'tags']:
                        if key in data and data[key]:
                            data[key] = json.loads(data[key])
                    
                    return ModelMetadata.from_dict(data)
                
                return None
                
        except Exception as e:
            logger.error(f"Error obteniendo modelo {model_id}: {str(e)}")
            return None
    
    def list_models(
        self, 
        status: Optional[ModelStatus] = None,
        model_name: Optional[str] = None,
        limit: int = 100
    ) -> List[ModelMetadata]:
        """Lista modelos con filtros opcionales"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = "SELECT * FROM models"
                conditions = []
                params = []
                
                if status:
                    conditions.append("status = ?")
                    params.append(status.value)
                
                if model_name:
                    conditions.append("model_name LIKE ?")
                    params.append(f"%{model_name}%")
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
                
                query += " ORDER BY created_at DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                # Convertir a objetos ModelMetadata
                columns = [description[0] for description in cursor.description]
                models = []
                
                for row in rows:
                    data = dict(zip(columns, row))
                    
                    # Convertir JSON strings
                    for key in ['feature_names', 'metrics', 'validation_metrics', 'test_metrics',
                               'hyperparameters', 'preprocessing_config', 'validation_results',
                               'deployment_config', 'performance_history', 'drift_alerts', 'tags']:
                        if key in data and data[key]:
                            data[key] = json.loads(data[key])
                    
                    models.append(ModelMetadata.from_dict(data))
                
                return models
                
        except Exception as e:
            logger.error(f"Error listando modelos: {str(e)}")
            return []
    
    def delete_model(self, model_id: str) -> bool:
        """Elimina un modelo de la base de datos"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Eliminar registros relacionados
                cursor.execute("DELETE FROM validations WHERE model_id = ?", (model_id,))
                cursor.execute("DELETE FROM performance_history WHERE model_id = ?", (model_id,))
                cursor.execute("DELETE FROM models WHERE model_id = ?", (model_id,))
                
                conn.commit()
                
                logger.info(f"Modelo {model_id} eliminado")
                return True
                
        except Exception as e:
            logger.error(f"Error eliminando modelo {model_id}: {str(e)}")
            return False

class ModelValidator:
    """Validador de modelos para control de calidad"""
    
    def __init__(self):
        self.validation_tests = {
            'performance': self._validate_performance,
            'data_quality': self._validate_data_quality,
            'bias': self._validate_bias,
            'robustness': self._validate_robustness
        }
    
    def validate_model(
        self,
        model: Any,
        test_data: pd.DataFrame,
        target_column: str,
        feature_columns: List[str],
        metadata: ModelMetadata
    ) -> ModelValidationResult:
        """
        Ejecuta validación completa del modelo
        
        Args:
            model: Modelo a validar
            test_data: Datos de test
            target_column: Columna objetivo
            feature_columns: Columnas de características
            metadata: Metadatos del modelo
            
        Returns:
            Resultado de validación
        """
        
        result = ModelValidationResult(
            model_id=metadata.model_id,
            validator="automated_validator"
        )
        
        X_test = test_data[feature_columns]
        y_test = test_data[target_column]
        
        # Ejecutar todos los tests
        for test_name, test_func in self.validation_tests.items():
            try:
                test_results = test_func(model, X_test, y_test, metadata)
                setattr(result, f"{test_name}_tests", test_results)
            except Exception as e:
                logger.error(f"Error en test {test_name}: {str(e)}")
                setattr(result, f"{test_name}_tests", {"error": str(e)})
        
        # Calcular score general
        result.overall_passed, result.score = self._calculate_overall_score(result)
        
        # Generar recomendaciones
        result.recommendations = self._generate_recommendations(result)
        
        return result
    
    def _validate_performance(
        self, 
        model: Any, 
        X_test: pd.DataFrame, 
        y_test: pd.Series,
        metadata: ModelMetadata
    ) -> Dict[str, bool]:
        """Valida el rendimiento del modelo"""
        
        tests = {}
        
        try:
            # Predicciones
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Test de ROC-AUC mínimo
            from sklearn.metrics import roc_auc_score
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            tests['roc_auc_minimum'] = roc_auc >= 0.7  # Umbral mínimo
            
            # Test de estabilidad de predicciones
            pred_std = np.std(y_pred_proba)
            tests['prediction_stability'] = 0.1 <= pred_std <= 0.4  # Rango razonable
            
            # Test de calibración
            from sklearn.calibration import calibration_curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_test, y_pred_proba, n_bins=10
            )
            calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
            tests['calibration_quality'] = calibration_error <= 0.1
            
        except Exception as e:
            logger.error(f"Error en validación de rendimiento: {str(e)}")
            tests['error'] = str(e)
        
        return tests
    
    def _validate_data_quality(
        self, 
        model: Any, 
        X_test: pd.DataFrame, 
        y_test: pd.Series,
        metadata: ModelMetadata
    ) -> Dict[str, bool]:
        """Valida la calidad de los datos"""
        
        tests = {}
        
        try:
            # Test de valores faltantes
            missing_ratio = X_test.isnull().sum().sum() / (len(X_test) * len(X_test.columns))
            tests['missing_values_acceptable'] = missing_ratio <= 0.1
            
            # Test de características constantes
            constant_features = X_test.nunique() == 1
            tests['no_constant_features'] = not constant_features.any()
            
            # Test de distribución de target
            target_balance = min(y_test.value_counts()) / len(y_test)
            tests['target_balance_acceptable'] = target_balance >= 0.05  # Al menos 5% de clase minoritaria
            
            # Test de outliers extremos
            numeric_cols = X_test.select_dtypes(include=[np.number]).columns
            outlier_ratios = []
            for col in numeric_cols:
                Q1 = X_test[col].quantile(0.25)
                Q3 = X_test[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((X_test[col] < Q1 - 3 * IQR) | (X_test[col] > Q3 + 3 * IQR)).sum()
                outlier_ratios.append(outliers / len(X_test))
            
            avg_outlier_ratio = np.mean(outlier_ratios) if outlier_ratios else 0
            tests['outliers_acceptable'] = avg_outlier_ratio <= 0.05
            
        except Exception as e:
            logger.error(f"Error en validación de calidad de datos: {str(e)}")
            tests['error'] = str(e)
        
        return tests
    
    def _validate_bias(
        self, 
        model: Any, 
        X_test: pd.DataFrame, 
        y_test: pd.Series,
        metadata: ModelMetadata
    ) -> Dict[str, bool]:
        """Valida sesgos del modelo"""
        
        tests = {}
        
        try:
            # Test de equidad demográfica (si hay columnas relevantes)
            demographic_cols = [col for col in X_test.columns if any(
                keyword in col.lower() for keyword in ['sector', 'region', 'size', 'tamaño']
            )]
            
            if demographic_cols:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                for col in demographic_cols[:3]:  # Limitar a 3 columnas
                    if X_test[col].nunique() > 1:
                        groups = X_test[col].unique()
                        group_rates = []
                        
                        for group in groups:
                            mask = X_test[col] == group
                            if mask.sum() > 10:  # Mínimo 10 muestras por grupo
                                group_rate = y_pred_proba[mask].mean()
                                group_rates.append(group_rate)
                        
                        if len(group_rates) > 1:
                            rate_ratio = max(group_rates) / min(group_rates)
                            tests[f'fairness_{col}'] = rate_ratio <= 2.0  # Ratio máximo 2:1
            
            # Test de consistencia de predicciones
            if len(X_test) > 100:
                # Dividir en dos mitades y comparar distribuciones
                mid = len(X_test) // 2
                pred1 = model.predict_proba(X_test.iloc[:mid])[:, 1]
                pred2 = model.predict_proba(X_test.iloc[mid:])[:, 1]
                
                from scipy.stats import ks_2samp
                _, p_value = ks_2samp(pred1, pred2)
                tests['prediction_consistency'] = p_value >= 0.05  # No diferencia significativa
            
        except Exception as e:
            logger.error(f"Error en validación de sesgo: {str(e)}")
            tests['error'] = str(e)
        
        return tests
    
    def _validate_robustness(
        self, 
        model: Any, 
        X_test: pd.DataFrame, 
        y_test: pd.Series,
        metadata: ModelMetadata
    ) -> Dict[str, bool]:
        """Valida la robustez del modelo"""
        
        tests = {}
        
        try:
            # Test de estabilidad con ruido
            if len(X_test) > 50:
                original_pred = model.predict_proba(X_test.iloc[:50])[:, 1]
                
                # Añadir ruido pequeño
                X_noisy = X_test.iloc[:50].copy()
                numeric_cols = X_noisy.select_dtypes(include=[np.number]).columns
                
                for col in numeric_cols:
                    noise = np.random.normal(0, X_noisy[col].std() * 0.01, len(X_noisy))
                    X_noisy[col] = X_noisy[col] + noise
                
                noisy_pred = model.predict_proba(X_noisy)[:, 1]
                
                # Calcular correlación
                correlation = np.corrcoef(original_pred, noisy_pred)[0, 1]
                tests['noise_robustness'] = correlation >= 0.95
            
            # Test de características faltantes
            if len(X_test) > 20:
                sample = X_test.iloc[:20].copy()
                original_pred = model.predict_proba(sample)[:, 1]
                
                # Simular valores faltantes en una característica
                test_col = sample.select_dtypes(include=[np.number]).columns[0]
                sample_missing = sample.copy()
                sample_missing[test_col] = sample_missing[test_col].median()  # Imputar con mediana
                
                missing_pred = model.predict_proba(sample_missing)[:, 1]
                
                # Verificar que las predicciones no cambien drásticamente
                max_change = np.max(np.abs(original_pred - missing_pred))
                tests['missing_value_robustness'] = max_change <= 0.1
            
        except Exception as e:
            logger.error(f"Error en validación de robustez: {str(e)}")
            tests['error'] = str(e)
        
        return tests
    
    def _calculate_overall_score(self, result: ModelValidationResult) -> Tuple[bool, float]:
        """Calcula el score general de validación"""
        
        all_tests = {}
        all_tests.update(result.performance_tests)
        all_tests.update(result.data_quality_tests)
        all_tests.update(result.bias_tests)
        all_tests.update(result.robustness_tests)
        
        # Filtrar errores
        valid_tests = {k: v for k, v in all_tests.items() if k != 'error' and isinstance(v, bool)}
        
        if not valid_tests:
            return False, 0.0
        
        # Calcular score
        passed_tests = sum(valid_tests.values())
        total_tests = len(valid_tests)
        score = passed_tests / total_tests
        
        # Criterio de aprobación: al menos 80% de tests pasados
        overall_passed = score >= 0.8
        
        return overall_passed, score
    
    def _generate_recommendations(self, result: ModelValidationResult) -> List[str]:
        """Genera recomendaciones basadas en los resultados"""
        
        recommendations = []
        
        # Revisar tests fallidos
        all_tests = {
            'performance': result.performance_tests,
            'data_quality': result.data_quality_tests,
            'bias': result.bias_tests,
            'robustness': result.robustness_tests
        }
        
        for category, tests in all_tests.items():
            for test_name, passed in tests.items():
                if test_name != 'error' and not passed:
                    if 'roc_auc' in test_name:
                        recommendations.append("Mejorar el rendimiento del modelo (ROC-AUC < 0.7)")
                    elif 'missing' in test_name:
                        recommendations.append("Reducir valores faltantes en los datos")
                    elif 'constant' in test_name:
                        recommendations.append("Eliminar características constantes")
                    elif 'balance' in test_name:
                        recommendations.append("Mejorar el balance de clases en el dataset")
                    elif 'fairness' in test_name:
                        recommendations.append(f"Revisar equidad en {test_name}")
                    elif 'robustness' in test_name:
                        recommendations.append("Mejorar la robustez del modelo")
        
        if not recommendations:
            recommendations.append("Modelo aprobado - cumple todos los criterios de calidad")
        
        return recommendations

class ModelRegistry:
    """Registro principal de modelos"""
    
    def __init__(self, registry_path: Union[str, Path]):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        # Inicializar base de datos
        self.db = ModelDatabase(self.registry_path / "models.db")
        
        # Inicializar validador
        self.validator = ModelValidator()
        
        # Directorios para artefactos
        self.models_dir = self.registry_path / "models"
        self.artifacts_dir = self.registry_path / "artifacts"
        self.models_dir.mkdir(exist_ok=True)
        self.artifacts_dir.mkdir(exist_ok=True)
    
    def register_model(
        self,
        model: Any,
        model_name: str,
        version: str,
        description: str,
        training_data: pd.DataFrame,
        test_data: pd.DataFrame,
        target_column: str,
        feature_columns: List[str],
        hyperparameters: Dict[str, Any],
        preprocessing_config: Dict[str, Any],
        metrics: Dict[str, float],
        created_by: str = "system",
        tags: List[str] = None,
        business_use_case: str = ""
    ) -> str:
        """
        Registra un nuevo modelo en el sistema
        
        Args:
            model: Modelo entrenado
            model_name: Nombre del modelo
            version: Versión del modelo
            description: Descripción del modelo
            training_data: Datos de entrenamiento
            test_data: Datos de test
            target_column: Columna objetivo
            feature_columns: Columnas de características
            hyperparameters: Hiperparámetros del modelo
            preprocessing_config: Configuración de preprocesamiento
            metrics: Métricas de rendimiento
            created_by: Usuario que registra el modelo
            tags: Tags del modelo
            business_use_case: Caso de uso de negocio
            
        Returns:
            ID del modelo registrado
        """
        
        # Generar ID único
        model_id = self._generate_model_id(model_name, version)
        
        # Calcular hash de datos de entrenamiento
        training_hash = self._calculate_data_hash(training_data)
        
        # Crear metadatos
        metadata = ModelMetadata(
            model_id=model_id,
            model_name=model_name,
            version=version,
            description=description,
            algorithm="XGBoost",  # Asumiendo XGBoost por defecto
            framework="scikit-learn",
            training_data_hash=training_hash,
            training_samples=len(training_data),
            feature_count=len(feature_columns),
            feature_names=feature_columns,
            metrics=metrics,
            hyperparameters=hyperparameters,
            preprocessing_config=preprocessing_config,
            created_by=created_by,
            tags=tags or [],
            business_use_case=business_use_case
        )
        
        # Guardar modelo y artefactos
        model_path = self._save_model_artifacts(model_id, model, metadata)
        metadata.model_path = str(model_path)
        metadata.artifacts_path = str(self.artifacts_dir / model_id)
        
        # Validar modelo
        validation_result = self.validator.validate_model(
            model, test_data, target_column, feature_columns, metadata
        )
        
        metadata.validation_results = validation_result.__dict__
        
        # Determinar estado inicial
        if validation_result.overall_passed:
            metadata.status = ModelStatus.VALIDATION
            metadata.approval_status = "approved"
        else:
            metadata.status = ModelStatus.TRAINING
            metadata.approval_status = "rejected"
        
        # Guardar en base de datos
        success = self.db.insert_model(metadata)
        
        if success:
            logger.info(f"Modelo {model_id} registrado exitosamente")
            return model_id
        else:
            raise Exception(f"Error registrando modelo {model_id}")
    
    def get_model(self, model_id: str) -> Optional[Tuple[Any, ModelMetadata]]:
        """
        Obtiene un modelo y sus metadatos
        
        Args:
            model_id: ID del modelo
            
        Returns:
            Tuple con (modelo, metadatos) o None si no existe
        """
        
        metadata = self.db.get_model(model_id)
        if not metadata:
            return None
        
        try:
            # Cargar modelo
            model = joblib.load(metadata.model_path)
            return model, metadata
        except Exception as e:
            logger.error(f"Error cargando modelo {model_id}: {str(e)}")
            return None
    
    def list_models(
        self,
        status: Optional[ModelStatus] = None,
        model_name: Optional[str] = None,
        limit: int = 100
    ) -> List[ModelMetadata]:
        """Lista modelos con filtros"""
        return self.db.list_models(status, model_name, limit)
    
    def promote_model(self, model_id: str, target_status: ModelStatus) -> bool:
        """
        Promueve un modelo a un estado superior
        
        Args:
            model_id: ID del modelo
            target_status: Estado objetivo
            
        Returns:
            True si la promoción fue exitosa
        """
        
        metadata = self.db.get_model(model_id)
        if not metadata:
            logger.error(f"Modelo {model_id} no encontrado")
            return False
        
        # Validar transición de estado
        valid_transitions = {
            ModelStatus.TRAINING: [ModelStatus.VALIDATION],
            ModelStatus.VALIDATION: [ModelStatus.STAGING],
            ModelStatus.STAGING: [ModelStatus.PRODUCTION],
            ModelStatus.PRODUCTION: [ModelStatus.ARCHIVED, ModelStatus.DEPRECATED]
        }
        
        if target_status not in valid_transitions.get(metadata.status, []):
            logger.error(f"Transición inválida de {metadata.status} a {target_status}")
            return False
        
        # Actualizar estado
        metadata.status = target_status
        metadata.updated_at = datetime.now()
        
        return self.db.update_model(metadata)
    
    def archive_model(self, model_id: str) -> bool:
        """Archiva un modelo"""
        return self.promote_model(model_id, ModelStatus.ARCHIVED)
    
    def delete_model(self, model_id: str, remove_files: bool = True) -> bool:
        """
        Elimina un modelo del registro
        
        Args:
            model_id: ID del modelo
            remove_files: Si eliminar también los archivos
            
        Returns:
            True si la eliminación fue exitosa
        """
        
        if remove_files:
            # Obtener rutas de archivos
            metadata = self.db.get_model(model_id)
            if metadata:
                # Eliminar archivos del modelo
                if metadata.model_path and Path(metadata.model_path).exists():
                    Path(metadata.model_path).unlink()
                
                # Eliminar directorio de artefactos
                artifacts_path = Path(metadata.artifacts_path)
                if artifacts_path.exists():
                    shutil.rmtree(artifacts_path)
        
        return self.db.delete_model(model_id)
    
    def get_production_model(self, model_name: str) -> Optional[Tuple[Any, ModelMetadata]]:
        """
        Obtiene el modelo en producción para un nombre dado
        
        Args:
            model_name: Nombre del modelo
            
        Returns:
            Tuple con (modelo, metadatos) del modelo en producción
        """
        
        models = self.db.list_models(
            status=ModelStatus.PRODUCTION,
            model_name=model_name,
            limit=1
        )
        
        if models:
            return self.get_model(models[0].model_id)
        
        return None
    
    def compare_models(self, model_ids: List[str]) -> pd.DataFrame:
        """
        Compara múltiples modelos
        
        Args:
            model_ids: Lista de IDs de modelos a comparar
            
        Returns:
            DataFrame con comparación de modelos
        """
        
        comparison_data = []
        
        for model_id in model_ids:
            metadata = self.db.get_model(model_id)
            if metadata:
                row = {
                    'model_id': model_id,
                    'model_name': metadata.model_name,
                    'version': metadata.version,
                    'status': metadata.status.value,
                    'created_at': metadata.created_at,
                    'training_samples': metadata.training_samples,
                    'feature_count': metadata.feature_count
                }
                
                # Añadir métricas
                for metric, value in metadata.metrics.items():
                    row[f'metric_{metric}'] = value
                
                comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def _generate_model_id(self, model_name: str, version: str) -> str:
        """Genera un ID único para el modelo"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{model_name}_{version}_{timestamp}"
    
    def _calculate_data_hash(self, data: pd.DataFrame) -> str:
        """Calcula hash de los datos de entrenamiento"""
        # Usar una muestra de los datos para el hash
        sample = data.head(1000) if len(data) > 1000 else data
        data_string = sample.to_string()
        return hashlib.md5(data_string.encode()).hexdigest()
    
    def _save_model_artifacts(
        self, 
        model_id: str, 
        model: Any, 
        metadata: ModelMetadata
    ) -> Path:
        """Guarda el modelo y artefactos asociados"""
        
        # Crear directorio para el modelo
        model_dir = self.models_dir / model_id
        model_dir.mkdir(exist_ok=True)
        
        # Guardar modelo
        model_path = model_dir / "model.pkl"
        joblib.dump(model, model_path)
        
        # Guardar metadatos
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False, default=str)
        
        # Crear directorio de artefactos
        artifacts_dir = self.artifacts_dir / model_id
        artifacts_dir.mkdir(exist_ok=True)
        
        return model_path
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del registro"""
        
        all_models = self.db.list_models(limit=10000)
        
        stats = {
            'total_models': len(all_models),
            'by_status': {},
            'by_model_name': {},
            'recent_registrations': 0,
            'avg_performance': {}
        }
        
        # Estadísticas por estado
        for status in ModelStatus:
            count = len([m for m in all_models if m.status == status])
            stats['by_status'][status.value] = count
        
        # Estadísticas por nombre de modelo
        model_names = {}
        for model in all_models:
            model_names[model.model_name] = model_names.get(model.model_name, 0) + 1
        stats['by_model_name'] = model_names
        
        # Registros recientes (últimos 7 días)
        cutoff = datetime.now() - timedelta(days=7)
        recent = [m for m in all_models if m.created_at >= cutoff]
        stats['recent_registrations'] = len(recent)
        
        # Rendimiento promedio
        all_metrics = {}
        for model in all_models:
            for metric, value in model.metrics.items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)
        
        for metric, values in all_metrics.items():
            stats['avg_performance'][metric] = np.mean(values)
        
        return stats

# Función de utilidad para registro rápido
def register_bankruptcy_model(
    model: Any,
    model_name: str,
    version: str,
    training_data: pd.DataFrame,
    test_data: pd.DataFrame,
    target_column: str = 'y',
    registry_path: Union[str, Path] = "./model_registry",
    **kwargs
) -> str:
    """
    Función de utilidad para registro rápido de modelos
    
    Args:
        model: Modelo entrenado
        model_name: Nombre del modelo
        version: Versión del modelo
        training_data: Datos de entrenamiento
        test_data: Datos de test
        target_column: Columna objetivo
        registry_path: Ruta del registro
        **kwargs: Argumentos adicionales
        
    Returns:
        ID del modelo registrado
    """
    
    registry = ModelRegistry(registry_path)
    
    # Extraer características
    feature_columns = [col for col in training_data.columns if col != target_column]
    
    # Calcular métricas básicas
    from sklearn.metrics import roc_auc_score, average_precision_score
    
    X_test = test_data[feature_columns]
    y_test = test_data[target_column]
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'pr_auc': average_precision_score(y_test, y_pred_proba)
    }
    
    return registry.register_model(
        model=model,
        model_name=model_name,
        version=version,
        description=kwargs.get('description', ''),
        training_data=training_data,
        test_data=test_data,
        target_column=target_column,
        feature_columns=feature_columns,
        hyperparameters=kwargs.get('hyperparameters', {}),
        preprocessing_config=kwargs.get('preprocessing_config', {}),
        metrics=metrics,
        created_by=kwargs.get('created_by', 'system'),
        tags=kwargs.get('tags', []),
        business_use_case=kwargs.get('business_use_case', '')
    )

if __name__ == "__main__":
    # Ejemplo de uso
    import pandas as pd
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    # Crear datos sintéticos
    X, y = make_classification(
        n_samples=1000, n_features=10, n_informative=8,
        n_redundant=2, random_state=42
    )
    
    # Crear DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Dividir datos
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Entrenar modelo simple
    X_train = train_df[feature_names]
    y_train = train_df['target']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Registrar modelo
    model_id = register_bankruptcy_model(
        model=model,
        model_name="bankruptcy_predictor",
        version="1.0.0",
        training_data=train_df,
        test_data=test_df,
        target_column='target',
        description="Modelo de ejemplo para predicción de quiebras",
        tags=['example', 'random_forest'],
        business_use_case="Evaluación de riesgo crediticio"
    )
    
    print(f"Modelo registrado con ID: {model_id}")
    
    # Obtener estadísticas del registro
    registry = ModelRegistry("./model_registry")
    stats = registry.get_registry_stats()
    print(f"Estadísticas del registro: {stats}")

