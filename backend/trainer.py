"""
Este módulo implementa el servicio de entrenamiento de modelos de Machine Learning,
incluyendo XGBoost con calibración, validación temporal, optimización de hiperparámetros
y gestión de modelos.
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

# Machine Learning
import xgboost as xgb
from sklearn.model_selection import (
    TimeSeriesSplit, StratifiedKFold, cross_val_score,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    classification_report, confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_class_weight

# Balanceo de clases
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTETomek, SMOTEENN

# Optimización de hiperparámetros
try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

# Suprimir warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Configurar logging
logger = logging.getLogger(__name__)

@dataclass
class ModelTrainingConfig:
    """Configuración para el entrenamiento de modelos"""
    
    # Configuración general
    random_state: int = 42
    n_jobs: int = -1
    verbose: bool = True
    
    # División de datos
    test_size: float = 0.2
    validation_size: float = 0.2
    temporal_split: bool = True  # Usar división temporal vs aleatoria
    
    # Balanceo de clases
    balance_classes: bool = True
    sampling_strategy: str = "smote"  # "smote", "adasyn", "borderline", "random_over", "random_under"
    smote_k_neighbors: int = 5
    
    # Configuración XGBoost
    xgb_params: Dict[str, Any] = field(default_factory=lambda: {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'gamma': 0,
        'reg_alpha': 0,
        'reg_lambda': 1,
        'scale_pos_weight': 1
    })
    
    # Validación cruzada
    cv_folds: int = 5
    cv_scoring: str = 'roc_auc'
    
    # Calibración
    calibration_method: str = 'isotonic'  # 'isotonic', 'sigmoid'
    calibration_cv: int = 3
    
    # Optimización de hiperparámetros
    hyperopt_trials: int = 100
    hyperopt_timeout: int = 3600  # segundos
    
    # Early stopping
    early_stopping_rounds: int = 50
    
    # Métricas de evaluación
    evaluation_metrics: List[str] = field(default_factory=lambda: [
        'roc_auc', 'pr_auc', 'brier_score', 'ks_statistic'
    ])

@dataclass
class ModelMetrics:
    """Métricas de rendimiento del modelo"""
    roc_auc: float = 0.0
    pr_auc: float = 0.0
    brier_score: float = 0.0
    ks_statistic: float = 0.0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    # Métricas por conjunto
    train_metrics: Optional[Dict[str, float]] = None
    val_metrics: Optional[Dict[str, float]] = None
    test_metrics: Optional[Dict[str, float]] = None
    
    # Información adicional
    confusion_matrix: Optional[np.ndarray] = None
    feature_importance: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte las métricas a diccionario"""
        return {
            'roc_auc': self.roc_auc,
            'pr_auc': self.pr_auc,
            'brier_score': self.brier_score,
            'ks_statistic': self.ks_statistic,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'test_metrics': self.test_metrics
        }

class MetricsCalculator:
    """Calculadora de métricas de rendimiento"""
    
    @staticmethod
    def calculate_ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Calcula la estadística Kolmogorov-Smirnov"""
        try:
            # Separar probabilidades por clase
            pos_probs = y_prob[y_true == 1]
            neg_probs = y_prob[y_true == 0]
            
            if len(pos_probs) == 0 or len(neg_probs) == 0:
                return 0.0
            
            # Calcular CDFs
            all_probs = np.concatenate([pos_probs, neg_probs])
            pos_cdf = np.searchsorted(np.sort(pos_probs), all_probs, side='right') / len(pos_probs)
            neg_cdf = np.searchsorted(np.sort(neg_probs), all_probs, side='right') / len(neg_probs)
            
            # KS es la máxima diferencia entre CDFs
            ks_stat = np.max(np.abs(pos_cdf - neg_cdf))
            return float(ks_stat)
            
        except Exception as e:
            logger.error(f"Error calculando KS statistic: {str(e)}")
            return 0.0
    
    @staticmethod
    def calculate_all_metrics(
        y_true: np.ndarray, 
        y_prob: np.ndarray, 
        y_pred: Optional[np.ndarray] = None,
        threshold: float = 0.5
    ) -> ModelMetrics:
        """Calcula todas las métricas de rendimiento"""
        
        if y_pred is None:
            y_pred = (y_prob >= threshold).astype(int)
        
        try:
            # Métricas básicas
            roc_auc = roc_auc_score(y_true, y_prob)
            pr_auc = average_precision_score(y_true, y_prob)
            brier = brier_score_loss(y_true, y_prob)
            ks_stat = MetricsCalculator.calculate_ks_statistic(y_true, y_prob)
            
            # Métricas de clasificación
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # Matriz de confusión
            cm = confusion_matrix(y_true, y_pred)
            
            return ModelMetrics(
                roc_auc=roc_auc,
                pr_auc=pr_auc,
                brier_score=brier,
                ks_statistic=ks_stat,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                confusion_matrix=cm
            )
            
        except Exception as e:
            logger.error(f"Error calculando métricas: {str(e)}")
            return ModelMetrics()

class DataSplitter:
    """Divisor de datos con soporte temporal"""
    
    def __init__(self, config: ModelTrainingConfig):
        self.config = config
    
    def split_data(
        self, 
        df: pd.DataFrame, 
        target_column: str,
        date_column: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Divide los datos en train, validation y test
        
        Args:
            df: DataFrame con los datos
            target_column: Nombre de la columna objetivo
            date_column: Nombre de la columna de fecha (para división temporal)
            
        Returns:
            Tuple con (train_df, val_df, test_df)
        """
        
        if self.config.temporal_split and date_column and date_column in df.columns:
            return self._temporal_split(df, target_column, date_column)
        else:
            return self._random_split(df, target_column)
    
    def _temporal_split(
        self, 
        df: pd.DataFrame, 
        target_column: str, 
        date_column: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """División temporal de datos"""
        
        # Ordenar por fecha
        df_sorted = df.sort_values(date_column)
        
        # Calcular puntos de corte
        n_total = len(df_sorted)
        n_test = int(n_total * self.config.test_size)
        n_val = int(n_total * self.config.validation_size)
        n_train = n_total - n_test - n_val
        
        # Dividir
        train_df = df_sorted.iloc[:n_train].copy()
        val_df = df_sorted.iloc[n_train:n_train + n_val].copy()
        test_df = df_sorted.iloc[n_train + n_val:].copy()
        
        logger.info(f"División temporal: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        return train_df, val_df, test_df
    
    def _random_split(
        self, 
        df: pd.DataFrame, 
        target_column: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """División aleatoria estratificada"""
        
        from sklearn.model_selection import train_test_split
        
        # Primera división: train+val vs test
        train_val_df, test_df = train_test_split(
            df,
            test_size=self.config.test_size,
            stratify=df[target_column],
            random_state=self.config.random_state
        )
        
        # Segunda división: train vs val
        val_size_adjusted = self.config.validation_size / (1 - self.config.test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size_adjusted,
            stratify=train_val_df[target_column],
            random_state=self.config.random_state
        )
        
        logger.info(f"División aleatoria: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        return train_df, val_df, test_df

class ClassBalancer:
    """Balanceador de clases"""
    
    def __init__(self, config: ModelTrainingConfig):
        self.config = config
        self.sampler = None
    
    def balance_classes(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Balancea las clases usando la estrategia configurada
        
        Args:
            X: Características
            y: Variable objetivo
            
        Returns:
            Tuple con (X_balanced, y_balanced)
        """
        
        if not self.config.balance_classes:
            return X, y
        
        # Crear sampler según estrategia
        if self.config.sampling_strategy == "smote":
            self.sampler = SMOTE(
                k_neighbors=self.config.smote_k_neighbors,
                random_state=self.config.random_state
            )
        elif self.config.sampling_strategy == "adasyn":
            self.sampler = ADASYN(random_state=self.config.random_state)
        elif self.config.sampling_strategy == "borderline":
            self.sampler = BorderlineSMOTE(
                k_neighbors=self.config.smote_k_neighbors,
                random_state=self.config.random_state
            )
        elif self.config.sampling_strategy == "smote_tomek":
            self.sampler = SMOTETomek(random_state=self.config.random_state)
        elif self.config.sampling_strategy == "smote_enn":
            self.sampler = SMOTEENN(random_state=self.config.random_state)
        elif self.config.sampling_strategy == "random_under":
            self.sampler = RandomUnderSampler(random_state=self.config.random_state)
        else:
            logger.warning(f"Estrategia de sampling desconocida: {self.config.sampling_strategy}")
            return X, y
        
        try:
            # Aplicar balanceo
            X_resampled, y_resampled = self.sampler.fit_resample(X, y)
            
            # Convertir de vuelta a DataFrame/Series
            X_balanced = pd.DataFrame(X_resampled, columns=X.columns)
            y_balanced = pd.Series(y_resampled, name=y.name)
            
            logger.info(f"Balanceo aplicado: {len(X)} -> {len(X_balanced)} muestras")
            logger.info(f"Distribución original: {y.value_counts().to_dict()}")
            logger.info(f"Distribución balanceada: {y_balanced.value_counts().to_dict()}")
            
            return X_balanced, y_balanced
            
        except Exception as e:
            logger.error(f"Error en balanceo de clases: {str(e)}")
            return X, y

class HyperparameterOptimizer:
    """Optimizador de hiperparámetros"""
    
    def __init__(self, config: ModelTrainingConfig):
        self.config = config
        self.study = None
    
    def optimize_xgboost(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: pd.DataFrame, 
        y_val: pd.Series
    ) -> Dict[str, Any]:
        """
        Optimiza hiperparámetros de XGBoost usando Optuna
        
        Args:
            X_train: Características de entrenamiento
            y_train: Target de entrenamiento
            X_val: Características de validación
            y_val: Target de validación
            
        Returns:
            Diccionario con los mejores hiperparámetros
        """
        
        if not HAS_OPTUNA:
            logger.warning("Optuna no disponible. Usando hiperparámetros por defecto.")
            return self.config.xgb_params
        
        def objective(trial):
            # Definir espacio de búsqueda
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': self.config.random_state,
                'n_jobs': self.config.n_jobs
            }
            
            # Entrenar modelo
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            # Evaluar
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_pred_proba)
            
            return score
        
        try:
            # Crear estudio
            self.study = optuna.create_study(direction='maximize')
            
            # Optimizar
            self.study.optimize(
                objective, 
                n_trials=self.config.hyperopt_trials,
                timeout=self.config.hyperopt_timeout
            )
            
            # Obtener mejores parámetros
            best_params = self.study.best_params
            best_params.update({
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'random_state': self.config.random_state,
                'n_jobs': self.config.n_jobs
            })
            
            logger.info(f"Optimización completada. Mejor score: {self.study.best_value:.4f}")
            logger.info(f"Mejores parámetros: {best_params}")
            
            return best_params
            
        except Exception as e:
            logger.error(f"Error en optimización de hiperparámetros: {str(e)}")
            return self.config.xgb_params
    
    def get_optimization_history(self) -> Optional[pd.DataFrame]:
        """Retorna el historial de optimización"""
        if self.study is None:
            return None
        
        return self.study.trials_dataframe()

class ModelTrainer:
    """Entrenador principal de modelos"""
    
    def __init__(self, config: Optional[ModelTrainingConfig] = None):
        self.config = config or ModelTrainingConfig()
        self.splitter = DataSplitter(self.config)
        self.balancer = ClassBalancer(self.config)
        self.optimizer = HyperparameterOptimizer(self.config)
        self.metrics_calculator = MetricsCalculator()
        
        # Modelos entrenados
        self.model = None
        self.calibrated_model = None
        self.feature_names = []
        self.training_history = {}
        
    def train_model(
        self,
        df: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        date_column: Optional[str] = None,
        optimize_hyperparams: bool = True
    ) -> Dict[str, Any]:
        """
        Entrena un modelo completo de predicción de quiebras
        
        Args:
            df: DataFrame con los datos
            target_column: Nombre de la columna objetivo
            feature_columns: Lista de columnas de características (opcional)
            date_column: Nombre de la columna de fecha (opcional)
            optimize_hyperparams: Si optimizar hiperparámetros
            
        Returns:
            Diccionario con resultados del entrenamiento
        """
        
        logger.info("Iniciando entrenamiento de modelo")
        
        # Preparar datos
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != target_column]
        
        self.feature_names = feature_columns
        
        # Dividir datos
        train_df, val_df, test_df = self.splitter.split_data(df, target_column, date_column)
        
        # Extraer características y target
        X_train = train_df[feature_columns]
        y_train = train_df[target_column]
        X_val = val_df[feature_columns]
        y_val = val_df[target_column]
        X_test = test_df[feature_columns]
        y_test = test_df[target_column]
        
        # Balancear clases en entrenamiento
        X_train_balanced, y_train_balanced = self.balancer.balance_classes(X_train, y_train)
        
        # Optimizar hiperparámetros
        if optimize_hyperparams:
            best_params = self.optimizer.optimize_xgboost(
                X_train_balanced, y_train_balanced, X_val, y_val
            )
        else:
            best_params = self.config.xgb_params
        
        # Entrenar modelo final
        self.model = xgb.XGBClassifier(**best_params)
        
        # Entrenar con early stopping
        eval_set = [(X_val, y_val)] if len(X_val) > 0 else None
        
        if eval_set:
            self.model.fit(
                X_train_balanced, y_train_balanced,
                eval_set=eval_set,
                verbose=self.config.verbose
            )
        else:
            self.model.fit(X_train_balanced, y_train_balanced)
        
        # Calibrar modelo
        self.calibrated_model = CalibratedClassifierCV(
            self.model,
            method=self.config.calibration_method,
            cv=self.config.calibration_cv
        )
        self.calibrated_model.fit(X_train_balanced, y_train_balanced)
        
        # Evaluar modelo
        results = self._evaluate_model(
            X_train, y_train, X_val, y_val, X_test, y_test
        )
        
        # Calcular importancia de características
        feature_importance = self._calculate_feature_importance(X_test, y_test)
        results['feature_importance'] = feature_importance
        
        # Guardar historial
        self.training_history = {
            'config': self.config.__dict__,
            'best_params': best_params,
            'feature_names': self.feature_names,
            'training_date': datetime.now().isoformat(),
            'data_splits': {
                'train_size': len(X_train),
                'val_size': len(X_val),
                'test_size': len(X_test)
            }
        }
        
        logger.info("Entrenamiento completado")
        return results
    
    def _evaluate_model(
        self,
        X_train: pd.DataFrame, y_train: pd.Series,
        X_val: pd.DataFrame, y_val: pd.Series,
        X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, Any]:
        """Evalúa el modelo en todos los conjuntos de datos"""
        
        results = {}
        
        # Predicciones
        train_proba = self.calibrated_model.predict_proba(X_train)[:, 1]
        val_proba = self.calibrated_model.predict_proba(X_val)[:, 1]
        test_proba = self.calibrated_model.predict_proba(X_test)[:, 1]
        
        # Calcular métricas para cada conjunto
        train_metrics = self.metrics_calculator.calculate_all_metrics(y_train, train_proba)
        val_metrics = self.metrics_calculator.calculate_all_metrics(y_val, val_proba)
        test_metrics = self.metrics_calculator.calculate_all_metrics(y_test, test_proba)
        
        results['metrics'] = {
            'train': train_metrics.to_dict(),
            'validation': val_metrics.to_dict(),
            'test': test_metrics.to_dict()
        }
        
        # Métricas principales (test)
        results['roc_auc'] = test_metrics.roc_auc
        results['pr_auc'] = test_metrics.pr_auc
        results['brier_score'] = test_metrics.brier_score
        results['ks_statistic'] = test_metrics.ks_statistic
        
        # Información adicional
        results['confusion_matrix'] = test_metrics.confusion_matrix.tolist()
        
        logger.info(f"Métricas de test - ROC-AUC: {test_metrics.roc_auc:.4f}, "
                   f"PR-AUC: {test_metrics.pr_auc:.4f}, "
                   f"Brier: {test_metrics.brier_score:.4f}, "
                   f"KS: {test_metrics.ks_statistic:.4f}")
        
        return results
    
    def _calculate_feature_importance(
        self, 
        X_test: pd.DataFrame, 
        y_test: pd.Series
    ) -> Dict[str, float]:
        """Calcula la importancia de características"""
        
        try:
            # Importancia del modelo base
            base_importance = dict(zip(
                self.feature_names,
                self.model.feature_importances_
            ))
            
            # Importancia por permutación
            perm_importance = permutation_importance(
                self.calibrated_model, X_test, y_test,
                n_repeats=5, random_state=self.config.random_state,
                scoring='roc_auc'
            )
            
            perm_importance_dict = dict(zip(
                self.feature_names,
                perm_importance.importances_mean
            ))
            
            # Combinar importancias
            combined_importance = {}
            for feature in self.feature_names:
                combined_importance[feature] = {
                    'base_importance': base_importance.get(feature, 0.0),
                    'permutation_importance': perm_importance_dict.get(feature, 0.0)
                }
            
            return combined_importance
            
        except Exception as e:
            logger.error(f"Error calculando importancia de características: {str(e)}")
            return {}
    
    def predict(
        self, 
        X: pd.DataFrame, 
        return_probabilities: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Realiza predicciones con el modelo entrenado
        
        Args:
            X: Características para predicción
            return_probabilities: Si retornar probabilidades además de clases
            
        Returns:
            Predicciones (y probabilidades si se solicita)
        """
        
        if self.calibrated_model is None:
            raise ValueError("El modelo debe ser entrenado antes de hacer predicciones")
        
        # Asegurar que las características coincidan
        X_features = X[self.feature_names]
        
        # Predicciones
        y_pred = self.calibrated_model.predict(X_features)
        
        if return_probabilities:
            y_proba = self.calibrated_model.predict_proba(X_features)[:, 1]
            return y_pred, y_proba
        else:
            return y_pred
    
    def cross_validate(
        self,
        df: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        cv_folds: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Realiza validación cruzada del modelo
        
        Args:
            df: DataFrame con los datos
            target_column: Nombre de la columna objetivo
            feature_columns: Lista de columnas de características
            cv_folds: Número de folds (opcional)
            
        Returns:
            Resultados de validación cruzada
        """
        
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != target_column]
        
        cv_folds = cv_folds or self.config.cv_folds
        
        X = df[feature_columns]
        y = df[target_column]
        
        # Crear modelo base
        model = xgb.XGBClassifier(**self.config.xgb_params)
        
        # Validación cruzada temporal si hay fecha
        if 'fecha_corte' in df.columns:
            cv = TimeSeriesSplit(n_splits=cv_folds)
        else:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.config.random_state)
        
        # Realizar validación cruzada
        cv_scores = cross_val_score(
            model, X, y, 
            cv=cv, 
            scoring=self.config.cv_scoring,
            n_jobs=self.config.n_jobs
        )
        
        results = {
            'cv_scores': cv_scores.tolist(),
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'scoring_metric': self.config.cv_scoring,
            'cv_folds': cv_folds
        }
        
        logger.info(f"Validación cruzada - {self.config.cv_scoring}: "
                   f"{cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return results
    
    def save_model(self, filepath: Union[str, Path], include_history: bool = True):
        """
        Guarda el modelo entrenado
        
        Args:
            filepath: Ruta donde guardar el modelo
            include_history: Si incluir el historial de entrenamiento
        """
        
        if self.calibrated_model is None:
            raise ValueError("No hay modelo entrenado para guardar")
        
        model_data = {
            'calibrated_model': self.calibrated_model,
            'base_model': self.model,
            'feature_names': self.feature_names,
            'config': self.config
        }
        
        if include_history:
            model_data['training_history'] = self.training_history
        
        joblib.dump(model_data, filepath)
        logger.info(f"Modelo guardado en {filepath}")
    
    @classmethod
    def load_model(cls, filepath: Union[str, Path]) -> 'ModelTrainer':
        """
        Carga un modelo entrenado
        
        Args:
            filepath: Ruta del modelo guardado
            
        Returns:
            Instancia de ModelTrainer con el modelo cargado
        """
        
        model_data = joblib.load(filepath)
        
        instance = cls(model_data['config'])
        instance.calibrated_model = model_data['calibrated_model']
        instance.model = model_data['base_model']
        instance.feature_names = model_data['feature_names']
        
        if 'training_history' in model_data:
            instance.training_history = model_data['training_history']
        
        logger.info(f"Modelo cargado desde {filepath}")
        return instance
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Retorna un resumen del modelo entrenado"""
        
        if self.calibrated_model is None:
            return {'status': 'No model trained'}
        
        return {
            'status': 'Trained',
            'model_type': 'XGBoost + Calibration',
            'feature_count': len(self.feature_names),
            'feature_names': self.feature_names,
            'config': self.config.__dict__,
            'training_history': self.training_history
        }

# Función de utilidad para entrenamiento rápido
def train_bankruptcy_model(
    df: pd.DataFrame,
    target_column: str = 'y',
    feature_columns: Optional[List[str]] = None,
    config: Optional[ModelTrainingConfig] = None,
    optimize_hyperparams: bool = True
) -> Tuple[ModelTrainer, Dict[str, Any]]:
    """
    Función de utilidad para entrenar un modelo de predicción de quiebras
    
    Args:
        df: DataFrame con los datos
        target_column: Nombre de la columna objetivo
        feature_columns: Lista de columnas de características
        config: Configuración de entrenamiento
        optimize_hyperparams: Si optimizar hiperparámetros
        
    Returns:
        Tuple con (trainer, resultados)
    """
    
    trainer = ModelTrainer(config)
    results = trainer.train_model(
        df, target_column, feature_columns, 
        optimize_hyperparams=optimize_hyperparams
    )
    
    return trainer, results

if __name__ == "__main__":
    # Ejemplo de uso
    import pandas as pd
    from sklearn.datasets import make_classification
    
    # Crear datos sintéticos
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15,
        n_redundant=5, n_clusters_per_class=1, 
        weights=[0.8, 0.2], random_state=42
    )
    
    # Crear DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['y'] = y
    df['fecha_corte'] = pd.date_range('2020-01-01', periods=len(df), freq='D')
    
    # Entrenar modelo
    trainer, results = train_bankruptcy_model(df, target_column='y')
    
    print("Entrenamiento completado:")
    print(f"ROC-AUC: {results['roc_auc']:.4f}")
    print(f"PR-AUC: {results['pr_auc']:.4f}")
    print(f"Brier Score: {results['brier_score']:.4f}")
    print(f"KS Statistic: {results['ks_statistic']:.4f}")
    
    # Ejemplo de predicción
    sample_data = df.head(5)[feature_names]
    predictions, probabilities = trainer.predict(sample_data)
    print(f"\nPredicciones de ejemplo: {predictions}")
    print(f"Probabilidades: {probabilities}")

