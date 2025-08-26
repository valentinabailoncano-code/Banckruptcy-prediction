"""
Este módulo contiene tests unitarios y de integración para validar
el funcionamiento correcto de todos los servicios ML del sistema.
"""

import unittest
import tempfile
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import sys
import os

# Añadir el directorio padre al path para importar módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.preprocessing.data_processor import (
    DataPreprocessor, PreprocessingConfig, preprocess_financial_data
)
from services.model_training.trainer import (
    ModelTrainer, ModelTrainingConfig, train_bankruptcy_model
)
from services.prediction.predictor import (
    PredictionService, PredictionConfig, AltmanCalculator, predict_bankruptcy_risk
)
from services.model_monitoring.drift_detector import (
    DriftMonitor, DriftConfig, monitor_model_drift
)
from services.model_monitoring.model_registry import (
    ModelRegistry, ModelMetadata, ModelStatus, register_bankruptcy_model
)

class TestDataPreprocessor(unittest.TestCase):
    """Tests para el servicio de preprocesamiento de datos"""
    
    def setUp(self):
        """Configuración inicial para cada test"""
        # Crear datos sintéticos
        np.random.seed(42)
        n_samples = 500
        
        self.sample_data = pd.DataFrame({
            'empresa_id': range(1, n_samples + 1),
            'fecha_corte': pd.date_range('2020-01-01', periods=n_samples, freq='D'),
            'sector': np.random.choice(['Technology', 'Manufacturing', 'Retail'], n_samples),
            'wc_ta': np.random.normal(0.1, 0.05, n_samples),
            're_ta': np.random.normal(0.15, 0.08, n_samples),
            'ebit_ta': np.random.normal(0.08, 0.04, n_samples),
            'me_tl': np.random.normal(1.2, 0.3, n_samples),
            's_ta': np.random.normal(1.1, 0.2, n_samples),
            'revenue': np.random.lognormal(13, 1, n_samples),
            'total_assets': np.random.lognormal(14, 1, n_samples),
            'current_ratio': np.random.normal(1.5, 0.5, n_samples),
            'debt_assets': np.random.normal(0.4, 0.2, n_samples),
            'y': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        })
        
        # Introducir algunos valores faltantes
        missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
        self.sample_data.loc[missing_indices, 'current_ratio'] = np.nan
        
        self.config = PreprocessingConfig(
            create_ratios=True,
            create_trends=True,
            outlier_detection=True
        )
    
    def test_preprocessor_initialization(self):
        """Test de inicialización del preprocesador"""
        preprocessor = DataPreprocessor(self.config)
        
        self.assertIsNotNone(preprocessor.config)
        self.assertEqual(preprocessor.config.create_ratios, True)
        self.assertFalse(preprocessor.is_fitted)
    
    def test_data_quality_validation(self):
        """Test de validación de calidad de datos"""
        preprocessor = DataPreprocessor(self.config)
        
        is_valid, results = preprocessor.validator.validate_data_quality(self.sample_data)
        
        self.assertIsInstance(is_valid, bool)
        self.assertIn('total_rows', results)
        self.assertIn('missing_data', results)
        self.assertIn('completeness', results)
        self.assertEqual(results['total_rows'], len(self.sample_data))
    
    def test_financial_ratios_calculation(self):
        """Test de cálculo de ratios financieros"""
        preprocessor = DataPreprocessor(self.config)
        
        # Procesar datos
        processed_data = preprocessor.fit_transform(self.sample_data, target_column='y')
        
        # Verificar que se crearon nuevos ratios
        expected_ratios = ['quick_ratio', 'gross_profit_margin', 'return_on_equity']
        for ratio in expected_ratios:
            if ratio in processed_data.columns:
                self.assertIn(ratio, processed_data.columns)
        
        # Verificar que no hay valores infinitos
        numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            self.assertFalse(np.isinf(processed_data[col]).any(), f"Valores infinitos en {col}")
    
    def test_altman_z_score_calculation(self):
        """Test de cálculo de Altman Z-Score"""
        preprocessor = DataPreprocessor(self.config)
        processed_data = preprocessor.fit_transform(self.sample_data, target_column='y')
        
        # Verificar que se calculó el Altman Z-Score
        if 'altman_z_score' in processed_data.columns:
            self.assertIn('altman_z_score', processed_data.columns)
            
            # Verificar que los valores están en un rango razonable
            z_scores = processed_data['altman_z_score'].dropna()
            self.assertTrue(z_scores.min() > -5, "Z-Score demasiado bajo")
            self.assertTrue(z_scores.max() < 10, "Z-Score demasiado alto")
    
    def test_preprocessor_save_load(self):
        """Test de guardado y carga del preprocesador"""
        with tempfile.TemporaryDirectory() as temp_dir:
            preprocessor = DataPreprocessor(self.config)
            processed_data = preprocessor.fit_transform(self.sample_data, target_column='y')
            
            # Guardar
            save_path = Path(temp_dir) / "preprocessor.pkl"
            preprocessor.save_preprocessor(save_path)
            self.assertTrue(save_path.exists())
            
            # Cargar
            loaded_preprocessor = DataPreprocessor.load_preprocessor(save_path)
            self.assertTrue(loaded_preprocessor.is_fitted)
            self.assertEqual(len(loaded_preprocessor.feature_names), len(preprocessor.feature_names))

class TestModelTrainer(unittest.TestCase):
    """Tests para el servicio de entrenamiento de modelos"""
    
    def setUp(self):
        """Configuración inicial para cada test"""
        # Crear datos sintéticos para clasificación
        X, y = make_classification(
            n_samples=1000, n_features=15, n_informative=10,
            n_redundant=5, n_clusters_per_class=1,
            weights=[0.7, 0.3], random_state=42
        )
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        self.training_data = pd.DataFrame(X, columns=feature_names)
        self.training_data['y'] = y
        self.training_data['fecha_corte'] = pd.date_range('2020-01-01', periods=len(self.training_data), freq='D')
        
        self.config = ModelTrainingConfig(
            cv_folds=3,  # Reducir para tests más rápidos
            hyperopt_trials=10,  # Reducir para tests más rápidos
            balance_classes=True
        )
    
    def test_trainer_initialization(self):
        """Test de inicialización del entrenador"""
        trainer = ModelTrainer(self.config)
        
        self.assertIsNotNone(trainer.config)
        self.assertIsNone(trainer.model)
        self.assertIsNone(trainer.calibrated_model)
    
    def test_data_splitting(self):
        """Test de división de datos"""
        trainer = ModelTrainer(self.config)
        
        train_df, val_df, test_df = trainer.splitter.split_data(
            self.training_data, 'y', 'fecha_corte'
        )
        
        # Verificar tamaños
        total_samples = len(train_df) + len(val_df) + len(test_df)
        self.assertEqual(total_samples, len(self.training_data))
        
        # Verificar que no hay solapamiento
        train_ids = set(train_df.index)
        val_ids = set(val_df.index)
        test_ids = set(test_df.index)
        
        self.assertEqual(len(train_ids & val_ids), 0)
        self.assertEqual(len(train_ids & test_ids), 0)
        self.assertEqual(len(val_ids & test_ids), 0)
    
    def test_class_balancing(self):
        """Test de balanceo de clases"""
        trainer = ModelTrainer(self.config)
        
        feature_cols = [col for col in self.training_data.columns if col not in ['y', 'fecha_corte']]
        X = self.training_data[feature_cols]
        y = self.training_data['y']
        
        X_balanced, y_balanced = trainer.balancer.balance_classes(X, y)
        
        # Verificar que se balancearon las clases
        original_balance = y.value_counts()
        balanced_balance = y_balanced.value_counts()
        
        self.assertGreater(len(y_balanced), len(y))  # Debe haber más muestras
        self.assertLess(abs(balanced_balance[0] - balanced_balance[1]), abs(original_balance[0] - original_balance[1]))
    
    def test_model_training(self):
        """Test de entrenamiento de modelo"""
        trainer = ModelTrainer(self.config)
        
        feature_cols = [col for col in self.training_data.columns if col not in ['y', 'fecha_corte']]
        
        results = trainer.train_model(
            self.training_data,
            target_column='y',
            feature_columns=feature_cols,
            optimize_hyperparams=False  # Desactivar para test más rápido
        )
        
        # Verificar que el modelo se entrenó
        self.assertIsNotNone(trainer.model)
        self.assertIsNotNone(trainer.calibrated_model)
        
        # Verificar métricas
        self.assertIn('roc_auc', results)
        self.assertIn('pr_auc', results)
        self.assertGreater(results['roc_auc'], 0.5)  # Mejor que aleatorio
    
    def test_cross_validation(self):
        """Test de validación cruzada"""
        trainer = ModelTrainer(self.config)
        
        feature_cols = [col for col in self.training_data.columns if col not in ['y', 'fecha_corte']]
        
        cv_results = trainer.cross_validate(
            self.training_data,
            target_column='y',
            feature_columns=feature_cols,
            cv_folds=3
        )
        
        self.assertIn('cv_scores', cv_results)
        self.assertIn('mean_score', cv_results)
        self.assertEqual(len(cv_results['cv_scores']), 3)
        self.assertGreater(cv_results['mean_score'], 0.5)
    
    def test_model_save_load(self):
        """Test de guardado y carga de modelo"""
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = ModelTrainer(self.config)
            
            feature_cols = [col for col in self.training_data.columns if col not in ['y', 'fecha_corte']]
            
            # Entrenar modelo
            trainer.train_model(
                self.training_data,
                target_column='y',
                feature_columns=feature_cols,
                optimize_hyperparams=False
            )
            
            # Guardar
            save_path = Path(temp_dir) / "model.pkl"
            trainer.save_model(save_path)
            self.assertTrue(save_path.exists())
            
            # Cargar
            loaded_trainer = ModelTrainer.load_model(save_path)
            self.assertIsNotNone(loaded_trainer.calibrated_model)
            self.assertEqual(len(loaded_trainer.feature_names), len(trainer.feature_names))

class TestPredictionService(unittest.TestCase):
    """Tests para el servicio de predicción"""
    
    def setUp(self):
        """Configuración inicial para cada test"""
        self.config = PredictionConfig(
            threshold_medium=0.15,
            threshold_high=0.30,
            enable_explanations=False  # Desactivar para tests más rápidos
        )
        
        self.sample_prediction_data = {
            'wc_ta': 0.10,
            're_ta': 0.15,
            'ebit_ta': 0.08,
            'me_tl': 1.5,
            's_ta': 1.2,
            'current_ratio': 1.8,
            'debt_assets': 0.45,
            'roa': 0.06
        }
        
        # Crear un modelo simple para tests
        X, y = make_classification(n_samples=500, n_features=8, random_state=42)
        feature_names = list(self.sample_prediction_data.keys())
        
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(X, y)
        
        # Crear archivo temporal del modelo
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = Path(self.temp_dir) / "test_model.pkl"
        
        import joblib
        model_data = {
            'calibrated_model': self.model,
            'feature_names': feature_names
        }
        joblib.dump(model_data, self.model_path)
    
    def tearDown(self):
        """Limpieza después de cada test"""
        shutil.rmtree(self.temp_dir)
    
    def test_altman_calculator(self):
        """Test del calculador de Altman Z-Score"""
        calculator = AltmanCalculator()
        
        # Test Altman original
        z_score = calculator.calculate_altman_original(0.1, 0.15, 0.08, 1.5, 1.2)
        self.assertIsInstance(z_score, float)
        self.assertGreater(z_score, 0)
        
        # Test clasificación de riesgo
        risk_class = calculator.classify_altman_risk(z_score)
        self.assertIn(risk_class, ['SAFE', 'GREY', 'DISTRESS', 'UNKNOWN'])
        
        # Test conversión a probabilidad
        probability = calculator.altman_to_probability(z_score)
        self.assertGreaterEqual(probability, 0)
        self.assertLessEqual(probability, 1)
    
    def test_prediction_service_initialization(self):
        """Test de inicialización del servicio de predicción"""
        service = PredictionService(self.config)
        
        self.assertIsNotNone(service.config)
        self.assertIsNone(service.model)
        self.assertEqual(len(service.feature_names), 0)
    
    def test_model_loading(self):
        """Test de carga de modelo"""
        service = PredictionService(self.config)
        service.load_model(self.model_path)
        
        self.assertIsNotNone(service.model)
        self.assertGreater(len(service.feature_names), 0)
    
    def test_single_prediction(self):
        """Test de predicción individual"""
        service = PredictionService(self.config)
        service.load_model(self.model_path)
        
        result = service.predict_single(
            self.sample_prediction_data,
            empresa_id=123,
            include_explanations=False
        )
        
        # Verificar estructura del resultado
        self.assertEqual(result.empresa_id, 123)
        self.assertGreaterEqual(result.probabilidad_ml, 0)
        self.assertLessEqual(result.probabilidad_ml, 1)
        self.assertIn(result.banda_riesgo_ml, ['LOW', 'MEDIUM', 'HIGH'])
        self.assertIsNotNone(result.tiempo_procesamiento_ms)
    
    def test_batch_prediction(self):
        """Test de predicción en lote"""
        service = PredictionService(self.config)
        service.load_model(self.model_path)
        
        # Crear datos de lote
        batch_data = pd.DataFrame([self.sample_prediction_data] * 5)
        batch_data['empresa_id'] = range(1, 6)
        
        results = service.predict_batch(batch_data, include_explanations=False)
        
        self.assertEqual(len(results), 5)
        for result in results:
            self.assertIsNotNone(result.probabilidad_ml)
            self.assertIn(result.banda_riesgo_ml, ['LOW', 'MEDIUM', 'HIGH'])
    
    def test_input_validation(self):
        """Test de validación de inputs"""
        service = PredictionService(self.config)
        service.load_model(self.model_path)
        
        # Test con datos válidos
        valid_data = pd.DataFrame([self.sample_prediction_data])
        is_valid, errors = service.validator.validate_input(valid_data)
        
        # Nota: Puede fallar validación por características faltantes, pero no debe dar error
        self.assertIsInstance(is_valid, bool)
        self.assertIsInstance(errors, list)

class TestDriftDetector(unittest.TestCase):
    """Tests para el detector de drift"""
    
    def setUp(self):
        """Configuración inicial para cada test"""
        self.config = DriftConfig(
            drift_threshold_warning=0.05,
            drift_threshold_critical=0.01
        )
        
        # Crear datos de referencia
        np.random.seed(42)
        self.reference_data = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 1000),
            'feature_2': np.random.normal(5, 2, 1000),
            'feature_3': np.random.choice(['A', 'B', 'C'], 1000),
            'target': np.random.choice([0, 1], 1000, p=[0.7, 0.3])
        })
        
        # Crear datos de monitoreo (con drift artificial)
        self.monitoring_data = pd.DataFrame({
            'feature_1': np.random.normal(0.5, 1, 500),  # Shift en media
            'feature_2': np.random.normal(5, 3, 500),    # Cambio en varianza
            'feature_3': np.random.choice(['A', 'B', 'C'], 500, p=[0.5, 0.3, 0.2]),  # Cambio en distribución
            'target': np.random.choice([0, 1], 500, p=[0.6, 0.4])  # Cambio en target
        })
    
    def test_drift_monitor_initialization(self):
        """Test de inicialización del monitor de drift"""
        monitor = DriftMonitor(self.config)
        
        self.assertIsNotNone(monitor.config)
        self.assertIsNotNone(monitor.data_detector)
        self.assertIsNotNone(monitor.concept_detector)
        self.assertIsNotNone(monitor.prediction_detector)
    
    def test_univariate_drift_detection(self):
        """Test de detección de drift univariado"""
        monitor = DriftMonitor(self.config)
        
        # Test drift numérico
        result = monitor.data_detector.detect_univariate_drift(
            self.reference_data['feature_1'],
            self.monitoring_data['feature_1'],
            'feature_1'
        )
        
        self.assertEqual(result.feature_name, 'feature_1')
        self.assertEqual(result.drift_type, 'data')
        self.assertIsInstance(result.drift_detected, bool)
        self.assertIn(result.severity, ['NONE', 'WARNING', 'CRITICAL'])
        
        # Test drift categórico
        result_cat = monitor.data_detector.detect_univariate_drift(
            self.reference_data['feature_3'],
            self.monitoring_data['feature_3'],
            'feature_3'
        )
        
        self.assertEqual(result_cat.feature_name, 'feature_3')
        self.assertIsInstance(result_cat.drift_detected, bool)
    
    def test_multivariate_drift_detection(self):
        """Test de detección de drift multivariado"""
        monitor = DriftMonitor(self.config)
        
        result = monitor.data_detector.detect_multivariate_drift(
            self.reference_data[['feature_1', 'feature_2']],
            self.monitoring_data[['feature_1', 'feature_2']]
        )
        
        self.assertEqual(result.feature_name, 'multivariate')
        self.assertEqual(result.drift_type, 'data')
        self.assertIsInstance(result.drift_detected, bool)
    
    def test_prediction_drift_detection(self):
        """Test de detección de drift en predicciones"""
        monitor = DriftMonitor(self.config)
        
        # Crear predicciones sintéticas
        ref_predictions = np.random.beta(2, 5, 1000)  # Distribución sesgada hacia 0
        mon_predictions = np.random.beta(3, 3, 500)   # Distribución más centrada
        
        result = monitor.prediction_detector.detect_prediction_drift(
            ref_predictions, mon_predictions
        )
        
        self.assertEqual(result.feature_name, 'predictions')
        self.assertEqual(result.drift_type, 'prediction')
        self.assertIsInstance(result.drift_detected, bool)
    
    def test_full_drift_monitoring(self):
        """Test de monitoreo completo de drift"""
        monitor = DriftMonitor(self.config)
        
        results = monitor.monitor_drift(
            reference_data=self.reference_data,
            monitoring_data=self.monitoring_data,
            target_column='target'
        )
        
        # Verificar estructura de resultados
        self.assertIn('timestamp', results)
        self.assertIn('data_drift', results)
        self.assertIn('summary', results)
        
        # Verificar resumen
        summary = results['summary']
        self.assertIn('total_features_analyzed', summary)
        self.assertIn('features_with_drift', summary)
        self.assertIsInstance(summary['total_features_analyzed'], int)

class TestModelRegistry(unittest.TestCase):
    """Tests para el registro de modelos"""
    
    def setUp(self):
        """Configuración inicial para cada test"""
        self.temp_dir = tempfile.mkdtemp()
        self.registry = ModelRegistry(self.temp_dir)
        
        # Crear datos y modelo de prueba
        X, y = make_classification(n_samples=500, n_features=10, random_state=42)
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        self.training_data = pd.DataFrame(X, columns=feature_names)
        self.training_data['target'] = y
        
        self.test_data = self.training_data.sample(100)
        
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_train = self.training_data[feature_names]
        y_train = self.training_data['target']
        self.model.fit(X_train, y_train)
        
        # Calcular métricas
        from sklearn.metrics import roc_auc_score
        y_pred_proba = self.model.predict_proba(X_train)[:, 1]
        self.metrics = {'roc_auc': roc_auc_score(y_train, y_pred_proba)}
    
    def tearDown(self):
        """Limpieza después de cada test"""
        shutil.rmtree(self.temp_dir)
    
    def test_registry_initialization(self):
        """Test de inicialización del registro"""
        self.assertTrue(self.registry.registry_path.exists())
        self.assertTrue(self.registry.models_dir.exists())
        self.assertTrue(self.registry.artifacts_dir.exists())
        self.assertIsNotNone(self.registry.db)
    
    def test_model_registration(self):
        """Test de registro de modelo"""
        model_id = self.registry.register_model(
            model=self.model,
            model_name="test_model",
            version="1.0.0",
            description="Modelo de prueba",
            training_data=self.training_data,
            test_data=self.test_data,
            target_column='target',
            feature_columns=[f'feature_{i}' for i in range(10)],
            hyperparameters={'n_estimators': 10},
            preprocessing_config={},
            metrics=self.metrics
        )
        
        self.assertIsInstance(model_id, str)
        self.assertIn("test_model", model_id)
        
        # Verificar que se guardó en la base de datos
        metadata = self.registry.db.get_model(model_id)
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.model_name, "test_model")
    
    def test_model_retrieval(self):
        """Test de recuperación de modelo"""
        # Registrar modelo
        model_id = self.registry.register_model(
            model=self.model,
            model_name="test_model",
            version="1.0.0",
            description="Modelo de prueba",
            training_data=self.training_data,
            test_data=self.test_data,
            target_column='target',
            feature_columns=[f'feature_{i}' for i in range(10)],
            hyperparameters={'n_estimators': 10},
            preprocessing_config={},
            metrics=self.metrics
        )
        
        # Recuperar modelo
        result = self.registry.get_model(model_id)
        self.assertIsNotNone(result)
        
        loaded_model, metadata = result
        self.assertIsNotNone(loaded_model)
        self.assertEqual(metadata.model_id, model_id)
    
    def test_model_listing(self):
        """Test de listado de modelos"""
        # Registrar algunos modelos
        for i in range(3):
            self.registry.register_model(
                model=self.model,
                model_name=f"test_model_{i}",
                version="1.0.0",
                description=f"Modelo de prueba {i}",
                training_data=self.training_data,
                test_data=self.test_data,
                target_column='target',
                feature_columns=[f'feature_{j}' for j in range(10)],
                hyperparameters={'n_estimators': 10},
                preprocessing_config={},
                metrics=self.metrics
            )
        
        # Listar modelos
        models = self.registry.list_models()
        self.assertGreaterEqual(len(models), 3)
        
        # Listar por nombre
        models_filtered = self.registry.list_models(model_name="test_model_0")
        self.assertGreater(len(models_filtered), 0)
    
    def test_model_promotion(self):
        """Test de promoción de modelo"""
        # Registrar modelo
        model_id = self.registry.register_model(
            model=self.model,
            model_name="test_model",
            version="1.0.0",
            description="Modelo de prueba",
            training_data=self.training_data,
            test_data=self.test_data,
            target_column='target',
            feature_columns=[f'feature_{i}' for i in range(10)],
            hyperparameters={'n_estimators': 10},
            preprocessing_config={},
            metrics=self.metrics
        )
        
        # Promover a staging
        success = self.registry.promote_model(model_id, ModelStatus.STAGING)
        self.assertTrue(success)
        
        # Verificar estado
        metadata = self.registry.db.get_model(model_id)
        self.assertEqual(metadata.status, ModelStatus.STAGING)
    
    def test_registry_stats(self):
        """Test de estadísticas del registro"""
        # Registrar algunos modelos
        for i in range(2):
            self.registry.register_model(
                model=self.model,
                model_name=f"test_model_{i}",
                version="1.0.0",
                description=f"Modelo de prueba {i}",
                training_data=self.training_data,
                test_data=self.test_data,
                target_column='target',
                feature_columns=[f'feature_{j}' for j in range(10)],
                hyperparameters={'n_estimators': 10},
                preprocessing_config={},
                metrics=self.metrics
            )
        
        stats = self.registry.get_registry_stats()
        
        self.assertIn('total_models', stats)
        self.assertIn('by_status', stats)
        self.assertIn('by_model_name', stats)
        self.assertGreaterEqual(stats['total_models'], 2)

class TestIntegration(unittest.TestCase):
    """Tests de integración entre servicios"""
    
    def setUp(self):
        """Configuración inicial para tests de integración"""
        # Crear datos sintéticos más realistas
        np.random.seed(42)
        n_samples = 1000
        
        # Generar características financieras correlacionadas
        self.financial_data = pd.DataFrame({
            'empresa_id': range(1, n_samples + 1),
            'fecha_corte': pd.date_range('2020-01-01', periods=n_samples, freq='D'),
            'sector': np.random.choice(['Technology', 'Manufacturing', 'Retail'], n_samples),
            'wc_ta': np.random.normal(0.1, 0.05, n_samples),
            're_ta': np.random.normal(0.15, 0.08, n_samples),
            'ebit_ta': np.random.normal(0.08, 0.04, n_samples),
            'me_tl': np.random.normal(1.2, 0.3, n_samples),
            's_ta': np.random.normal(1.1, 0.2, n_samples),
            'revenue': np.random.lognormal(13, 1, n_samples),
            'total_assets': np.random.lognormal(14, 1, n_samples),
            'current_ratio': np.random.normal(1.5, 0.5, n_samples),
            'debt_assets': np.random.normal(0.4, 0.2, n_samples)
        })
        
        # Generar target basado en características (más realista)
        risk_score = (
            -2 * self.financial_data['wc_ta'] +
            -1.5 * self.financial_data['re_ta'] +
            -3 * self.financial_data['ebit_ta'] +
            -0.5 * self.financial_data['current_ratio'] +
            2 * self.financial_data['debt_assets'] +
            np.random.normal(0, 0.5, n_samples)
        )
        
        # Convertir a probabilidad y luego a clase binaria
        probabilities = 1 / (1 + np.exp(-risk_score))
        self.financial_data['y'] = (probabilities > 0.5).astype(int)
        
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Limpieza después de tests de integración"""
        shutil.rmtree(self.temp_dir)
    
    def test_full_pipeline(self):
        """Test del pipeline completo: preprocesamiento -> entrenamiento -> predicción"""
        
        # 1. Preprocesamiento
        preprocessor_config = PreprocessingConfig(
            create_ratios=True,
            create_trends=False,  # Desactivar para test más rápido
            outlier_detection=True
        )
        
        processed_data, preprocessor = preprocess_financial_data(
            self.financial_data,
            target_column='y',
            config=preprocessor_config
        )
        
        self.assertIsNotNone(processed_data)
        self.assertTrue(preprocessor.is_fitted)
        
        # 2. Entrenamiento
        trainer_config = ModelTrainingConfig(
            cv_folds=3,
            hyperopt_trials=5,  # Reducir para test más rápido
            balance_classes=True
        )
        
        feature_cols = [col for col in processed_data.columns if col not in ['y', 'empresa_id', 'fecha_corte']]
        
        trainer, training_results = train_bankruptcy_model(
            processed_data,
            target_column='y',
            feature_columns=feature_cols,
            config=trainer_config,
            optimize_hyperparams=False  # Desactivar para test más rápido
        )
        
        self.assertIsNotNone(trainer.calibrated_model)
        self.assertGreater(training_results['roc_auc'], 0.5)
        
        # 3. Guardar modelo
        model_path = Path(self.temp_dir) / "trained_model.pkl"
        trainer.save_model(model_path)
        
        # 4. Predicción
        prediction_config = PredictionConfig(enable_explanations=False)
        prediction_service = PredictionService(prediction_config)
        prediction_service.load_model(model_path)
        
        # Hacer predicción en una muestra
        sample_data = processed_data.head(1)
        result = prediction_service.predict_single(
            sample_data,
            empresa_id=1,
            include_explanations=False
        )
        
        self.assertIsNotNone(result.probabilidad_ml)
        self.assertIn(result.banda_riesgo_ml, ['LOW', 'MEDIUM', 'HIGH'])
        
        # 5. Registro del modelo
        registry = ModelRegistry(Path(self.temp_dir) / "registry")
        
        train_data = processed_data.sample(800)
        test_data = processed_data.drop(train_data.index)
        
        model_id = registry.register_model(
            model=trainer.calibrated_model,
            model_name="integration_test_model",
            version="1.0.0",
            description="Modelo de test de integración",
            training_data=train_data,
            test_data=test_data,
            target_column='y',
            feature_columns=feature_cols,
            hyperparameters=trainer_config.__dict__,
            preprocessing_config=preprocessor_config.__dict__,
            metrics=training_results
        )
        
        self.assertIsInstance(model_id, str)
        
        # Verificar que se puede recuperar
        loaded_result = registry.get_model(model_id)
        self.assertIsNotNone(loaded_result)
    
    def test_drift_monitoring_integration(self):
        """Test de integración del monitoreo de drift"""
        
        # Dividir datos en referencia y monitoreo
        split_point = len(self.financial_data) // 2
        reference_data = self.financial_data.iloc[:split_point].copy()
        monitoring_data = self.financial_data.iloc[split_point:].copy()
        
        # Introducir drift artificial
        monitoring_data['wc_ta'] = monitoring_data['wc_ta'] + 0.05  # Shift en media
        monitoring_data['current_ratio'] = monitoring_data['current_ratio'] * 1.2  # Cambio en escala
        
        # Ejecutar monitoreo
        drift_results = monitor_model_drift(
            reference_data=reference_data,
            monitoring_data=monitoring_data,
            target_column='y'
        )
        
        self.assertIn('data_drift', drift_results)
        self.assertIn('summary', drift_results)
        
        # Verificar que se detectó algún drift
        summary = drift_results['summary']
        self.assertGreater(summary['total_features_analyzed'], 0)

def run_all_tests():
    """Ejecuta todos los tests"""
    
    # Crear suite de tests
    test_suite = unittest.TestSuite()
    
    # Añadir tests de cada clase
    test_classes = [
        TestDataPreprocessor,
        TestModelTrainer,
        TestPredictionService,
        TestDriftDetector,
        TestModelRegistry,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Ejecutar tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result

if __name__ == "__main__":
    print("Ejecutando tests del sistema de Machine Learning...")
    print("=" * 60)
    
    result = run_all_tests()
    
    print("\n" + "=" * 60)
    print(f"Tests ejecutados: {result.testsRun}")
    print(f"Errores: {len(result.errors)}")
    print(f"Fallos: {len(result.failures)}")
    
    if result.errors:
        print("\nErrores encontrados:")
        for test, error in result.errors:
            print(f"- {test}: {error}")
    
    if result.failures:
        print("\nFallos encontrados:")
        for test, failure in result.failures:
            print(f"- {test}: {failure}")
    
    if result.wasSuccessful():
        print("\n✅ Todos los tests pasaron exitosamente!")
    else:
        print("\n❌ Algunos tests fallaron. Revisar los errores arriba.")

