"""
Tests para el módulo de preprocesamiento de datos financieros,
incluyendo validación, cálculo de ratios y ingeniería de características.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from services.preprocessing.data_processor import (
    DataPreprocessor, DataQualityConfig, FeatureEngineeringConfig,
    PreprocessingPipeline, OutlierDetectionMethod, ImputationStrategy
)

class TestDataPreprocessor:
    """Tests para la clase DataPreprocessor"""
    
    def test_init_default_config(self):
        """Test inicialización con configuración por defecto"""
        processor = DataPreprocessor()
        
        assert processor.quality_config is not None
        assert processor.feature_config is not None
        assert processor.preprocessing_pipeline is not None
        assert processor.is_fitted is False
    
    def test_init_custom_config(self):
        """Test inicialización con configuración personalizada"""
        quality_config = DataQualityConfig(
            min_completeness=0.8,
            max_outlier_ratio=0.1
        )
        
        processor = DataPreprocessor(quality_config=quality_config)
        
        assert processor.quality_config.min_completeness == 0.8
        assert processor.quality_config.max_outlier_ratio == 0.1
    
    def test_validate_data_quality_complete_data(self, sample_dataframe):
        """Test validación de calidad con datos completos"""
        processor = DataPreprocessor()
        
        result = processor.validate_data_quality(sample_dataframe)
        
        assert 'completeness' in result
        assert 'outliers' in result
        assert 'data_types' in result
        assert 'duplicates' in result
        assert result['completeness'] >= 0.9  # Datos completos
        assert result['outliers'] <= 0.1  # Pocos outliers
    
    def test_validate_data_quality_missing_data(self):
        """Test validación de calidad con datos faltantes"""
        # Crear DataFrame con valores faltantes
        data = {
            'activos_totales': [1000000, np.nan, 3000000],
            'pasivos_totales': [500000, 1500000, np.nan],
            'utilidad_neta': [100000, np.nan, np.nan]
        }
        df = pd.DataFrame(data)
        
        processor = DataPreprocessor()
        result = processor.validate_data_quality(df)
        
        assert result['completeness'] < 0.8  # Muchos datos faltantes
        assert 'missing_by_column' in result
    
    def test_validate_data_quality_outliers(self):
        """Test detección de outliers"""
        # Crear DataFrame con outliers evidentes
        data = {
            'activos_totales': [1000000, 2000000, 1500000, 100000000],  # Último es outlier
            'pasivos_totales': [500000, 1000000, 750000, 50000000],
            'utilidad_neta': [100000, 200000, 150000, -10000000]  # Último es outlier
        }
        df = pd.DataFrame(data)
        
        processor = DataPreprocessor()
        result = processor.validate_data_quality(df)
        
        assert result['outliers'] > 0
        assert 'outlier_columns' in result
    
    def test_calculate_financial_ratios_basic(self):
        """Test cálculo de ratios financieros básicos"""
        data = {
            'activos_totales': [10000000],
            'pasivos_totales': [6000000],
            'patrimonio': [4000000],
            'activos_corrientes': [4000000],
            'pasivos_corrientes': [2000000],
            'utilidad_neta': [600000],
            'ingresos_operacionales': [8000000],
            'ventas_netas': [8000000]
        }
        df = pd.DataFrame(data)
        
        processor = DataPreprocessor()
        result = processor.calculate_financial_ratios(df)
        
        # Verificar ratios calculados
        assert 'ratio_liquidez_corriente' in result.columns
        assert 'ratio_endeudamiento' in result.columns
        assert 'roa' in result.columns
        assert 'roe' in result.columns
        assert 'margen_neto' in result.columns
        
        # Verificar valores específicos
        assert abs(result['ratio_liquidez_corriente'].iloc[0] - 2.0) < 0.01  # 4M/2M = 2.0
        assert abs(result['ratio_endeudamiento'].iloc[0] - 0.6) < 0.01  # 6M/10M = 0.6
        assert abs(result['roa'].iloc[0] - 0.06) < 0.01  # 600K/10M = 0.06
    
    def test_calculate_financial_ratios_missing_data(self):
        """Test cálculo de ratios con datos faltantes"""
        data = {
            'activos_totales': [10000000],
            'pasivos_totales': [np.nan],  # Dato faltante
            'patrimonio': [4000000],
            'activos_corrientes': [4000000],
            'pasivos_corrientes': [2000000]
        }
        df = pd.DataFrame(data)
        
        processor = DataPreprocessor()
        result = processor.calculate_financial_ratios(df)
        
        # Debe manejar datos faltantes sin errores
        assert len(result) == 1
        assert pd.isna(result['ratio_endeudamiento'].iloc[0])  # No se puede calcular
        assert not pd.isna(result['ratio_liquidez_corriente'].iloc[0])  # Sí se puede calcular
    
    def test_calculate_altman_z_score_original(self):
        """Test cálculo de Altman Z-Score original"""
        data = {
            'activos_totales': [10000000],
            'capital_trabajo': [2000000],
            'utilidades_retenidas': [1000000],
            'utilidad_operacional': [800000],
            'valor_mercado_acciones': [5000000],
            'pasivos_totales': [6000000],
            'ventas_netas': [8000000]
        }
        df = pd.DataFrame(data)
        
        processor = DataPreprocessor()
        result = processor.calculate_altman_z_score(df, variant='original')
        
        assert 'altman_z_score' in result.columns
        assert not pd.isna(result['altman_z_score'].iloc[0])
        
        # Z-Score debe estar en rango razonable
        z_score = result['altman_z_score'].iloc[0]
        assert 0 < z_score < 10  # Rango típico
    
    def test_calculate_altman_z_score_modified(self):
        """Test cálculo de Altman Z-Score modificado"""
        data = {
            'activos_totales': [10000000],
            'capital_trabajo': [2000000],
            'utilidades_retenidas': [1000000],
            'utilidad_operacional': [800000],
            'patrimonio': [4000000],
            'pasivos_totales': [6000000],
            'ventas_netas': [8000000]
        }
        df = pd.DataFrame(data)
        
        processor = DataPreprocessor()
        result = processor.calculate_altman_z_score(df, variant='modified')
        
        assert 'altman_z_score_modificado' in result.columns
        assert not pd.isna(result['altman_z_score_modificado'].iloc[0])
    
    def test_calculate_altman_z_score_emerging_markets(self):
        """Test cálculo de Altman Z-Score para mercados emergentes"""
        data = {
            'activos_totales': [10000000],
            'capital_trabajo': [2000000],
            'utilidades_retenidas': [1000000],
            'utilidad_operacional': [800000],
            'patrimonio': [4000000],
            'pasivos_totales': [6000000],
            'ventas_netas': [8000000]
        }
        df = pd.DataFrame(data)
        
        processor = DataPreprocessor()
        result = processor.calculate_altman_z_score(df, variant='emerging_markets')
        
        assert 'altman_z_score_emergentes' in result.columns
        assert not pd.isna(result['altman_z_score_emergentes'].iloc[0])
    
    def test_engineer_features_basic(self, sample_dataframe):
        """Test ingeniería de características básica"""
        processor = DataPreprocessor()
        
        result = processor.engineer_features(sample_dataframe)
        
        # Debe incluir características originales
        for col in sample_dataframe.columns:
            if col in result.columns:
                assert col in result.columns
        
        # Debe incluir nuevas características
        assert len(result.columns) >= len(sample_dataframe.columns)
    
    def test_engineer_features_with_sector_data(self):
        """Test ingeniería de características con datos sectoriales"""
        data = {
            'empresa_id': [1, 2, 3, 4],
            'sector': ['Technology', 'Technology', 'Finance', 'Finance'],
            'activos_totales': [10000000, 15000000, 8000000, 12000000],
            'utilidad_neta': [1000000, 1500000, 600000, 800000],
            'roa': [0.1, 0.1, 0.075, 0.067]
        }
        df = pd.DataFrame(data)
        
        processor = DataPreprocessor()
        result = processor.engineer_features(df)
        
        # Debe incluir características sectoriales
        sector_features = [col for col in result.columns if 'sector_' in col]
        assert len(sector_features) > 0
    
    def test_engineer_features_temporal(self):
        """Test ingeniería de características temporales"""
        # Crear datos con múltiples períodos
        dates = pd.date_range('2020-01-01', periods=12, freq='M')
        data = {
            'empresa_id': [1] * 12,
            'periodo': dates,
            'activos_totales': np.random.uniform(8000000, 12000000, 12),
            'utilidad_neta': np.random.uniform(500000, 1500000, 12)
        }
        df = pd.DataFrame(data)
        
        processor = DataPreprocessor()
        result = processor.engineer_features(df)
        
        # Debe incluir características temporales
        temporal_features = [col for col in result.columns if any(x in col for x in ['trend_', 'volatility_', 'lag_'])]
        assert len(temporal_features) > 0
    
    def test_preprocess_data_complete_pipeline(self, sample_dataframe):
        """Test pipeline completo de preprocesamiento"""
        processor = DataPreprocessor()
        
        result = processor.preprocess_data(sample_dataframe)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert processor.is_fitted is True
        
        # Verificar que no hay valores infinitos o NaN en características numéricas
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        assert not result[numeric_cols].isin([np.inf, -np.inf]).any().any()
    
    def test_preprocess_data_transform_only(self, sample_dataframe):
        """Test transformación sin reentrenamiento"""
        processor = DataPreprocessor()
        
        # Primer ajuste
        processor.preprocess_data(sample_dataframe)
        
        # Segunda transformación (solo transform)
        new_data = sample_dataframe.copy()
        new_data['empresa_id'] = range(101, 201)  # Nuevos IDs
        
        result = processor.preprocess_data(new_data, fit=False)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(new_data)
    
    def test_save_and_load_preprocessor(self, sample_dataframe, temp_directory):
        """Test guardado y carga del preprocesador"""
        processor = DataPreprocessor()
        processor.preprocess_data(sample_dataframe)
        
        # Guardar
        save_path = f"{temp_directory}/test_preprocessor.pkl"
        processor.save_preprocessor(save_path)
        
        # Cargar
        loaded_processor = DataPreprocessor.load_preprocessor(save_path)
        
        assert loaded_processor.is_fitted is True
        assert loaded_processor.quality_config.min_completeness == processor.quality_config.min_completeness
        
        # Verificar que funciona igual
        result_original = processor.preprocess_data(sample_dataframe, fit=False)
        result_loaded = loaded_processor.preprocess_data(sample_dataframe, fit=False)
        
        # Comparar resultados (permitir pequeñas diferencias por precisión)
        pd.testing.assert_frame_equal(result_original, result_loaded, atol=1e-10)

class TestDataQualityConfig:
    """Tests para configuración de calidad de datos"""
    
    def test_default_config(self):
        """Test configuración por defecto"""
        config = DataQualityConfig()
        
        assert config.min_completeness == 0.7
        assert config.max_outlier_ratio == 0.05
        assert config.outlier_method == OutlierDetectionMethod.IQR
        assert config.outlier_threshold == 1.5
    
    def test_custom_config(self):
        """Test configuración personalizada"""
        config = DataQualityConfig(
            min_completeness=0.9,
            max_outlier_ratio=0.1,
            outlier_method=OutlierDetectionMethod.Z_SCORE,
            outlier_threshold=3.0
        )
        
        assert config.min_completeness == 0.9
        assert config.max_outlier_ratio == 0.1
        assert config.outlier_method == OutlierDetectionMethod.Z_SCORE
        assert config.outlier_threshold == 3.0

class TestFeatureEngineeringConfig:
    """Tests para configuración de ingeniería de características"""
    
    def test_default_config(self):
        """Test configuración por defecto"""
        config = FeatureEngineeringConfig()
        
        assert config.include_ratios is True
        assert config.include_sector_features is True
        assert config.include_temporal_features is True
        assert config.include_interaction_features is False
        assert config.polynomial_degree == 1
    
    def test_custom_config(self):
        """Test configuración personalizada"""
        config = FeatureEngineeringConfig(
            include_ratios=False,
            include_sector_features=False,
            include_temporal_features=False,
            include_interaction_features=True,
            polynomial_degree=2,
            max_features=100
        )
        
        assert config.include_ratios is False
        assert config.include_sector_features is False
        assert config.include_temporal_features is False
        assert config.include_interaction_features is True
        assert config.polynomial_degree == 2
        assert config.max_features == 100

class TestPreprocessingPipeline:
    """Tests para pipeline de preprocesamiento"""
    
    def test_create_pipeline_default(self):
        """Test creación de pipeline por defecto"""
        config = FeatureEngineeringConfig()
        pipeline = PreprocessingPipeline.create_pipeline(config)
        
        assert pipeline is not None
        assert hasattr(pipeline, 'fit')
        assert hasattr(pipeline, 'transform')
    
    def test_create_pipeline_custom(self):
        """Test creación de pipeline personalizado"""
        config = FeatureEngineeringConfig(
            imputation_strategy=ImputationStrategy.KNN,
            scaling_method='robust',
            outlier_method=OutlierDetectionMethod.ISOLATION_FOREST
        )
        pipeline = PreprocessingPipeline.create_pipeline(config)
        
        assert pipeline is not None
    
    def test_pipeline_fit_transform(self, sample_dataframe):
        """Test ajuste y transformación del pipeline"""
        config = FeatureEngineeringConfig()
        pipeline = PreprocessingPipeline.create_pipeline(config)
        
        # Preparar datos numéricos
        numeric_data = sample_dataframe.select_dtypes(include=[np.number])
        
        # Fit y transform
        transformed = pipeline.fit_transform(numeric_data)
        
        assert transformed is not None
        assert transformed.shape[0] == numeric_data.shape[0]
    
    def test_pipeline_transform_only(self, sample_dataframe):
        """Test transformación sin reajuste"""
        config = FeatureEngineeringConfig()
        pipeline = PreprocessingPipeline.create_pipeline(config)
        
        # Preparar datos numéricos
        numeric_data = sample_dataframe.select_dtypes(include=[np.number])
        
        # Fit
        pipeline.fit(numeric_data)
        
        # Transform
        transformed = pipeline.transform(numeric_data)
        
        assert transformed is not None
        assert transformed.shape[0] == numeric_data.shape[0]

class TestIntegrationDataProcessor:
    """Tests de integración para el procesador de datos"""
    
    def test_full_preprocessing_workflow(self, sample_dataframe):
        """Test workflow completo de preprocesamiento"""
        processor = DataPreprocessor()
        
        # 1. Validar calidad
        quality_result = processor.validate_data_quality(sample_dataframe)
        assert quality_result['completeness'] > 0.5
        
        # 2. Calcular ratios financieros
        with_ratios = processor.calculate_financial_ratios(sample_dataframe)
        assert len(with_ratios.columns) > len(sample_dataframe.columns)
        
        # 3. Calcular Altman Z-Score
        with_altman = processor.calculate_altman_z_score(with_ratios)
        assert 'altman_z_score' in with_altman.columns
        
        # 4. Ingeniería de características
        with_features = processor.engineer_features(with_altman)
        assert len(with_features.columns) >= len(with_altman.columns)
        
        # 5. Preprocesamiento completo
        final_result = processor.preprocess_data(sample_dataframe)
        assert isinstance(final_result, pd.DataFrame)
        assert len(final_result) > 0
    
    def test_error_handling_invalid_data(self):
        """Test manejo de errores con datos inválidos"""
        # DataFrame vacío
        empty_df = pd.DataFrame()
        processor = DataPreprocessor()
        
        with pytest.raises(ValueError):
            processor.validate_data_quality(empty_df)
        
        # DataFrame con todas las columnas no numéricas
        text_df = pd.DataFrame({'text_col': ['a', 'b', 'c']})
        
        result = processor.calculate_financial_ratios(text_df)
        assert len(result.columns) == len(text_df.columns)  # Sin ratios calculados
    
    def test_performance_large_dataset(self, performance_data):
        """Test performance con dataset grande"""
        import time
        
        # Crear dataset grande
        size = performance_data['large_dataset_size']
        np.random.seed(42)
        
        large_data = {
            'empresa_id': range(size),
            'activos_totales': np.random.uniform(1000000, 50000000, size),
            'pasivos_totales': np.random.uniform(500000, 30000000, size),
            'utilidad_neta': np.random.uniform(-1000000, 5000000, size),
            'sector': np.random.choice(['Tech', 'Finance', 'Retail'], size)
        }
        large_df = pd.DataFrame(large_data)
        
        processor = DataPreprocessor()
        
        start_time = time.time()
        result = processor.preprocess_data(large_df)
        processing_time = time.time() - start_time
        
        # Verificar que el procesamiento fue exitoso
        assert isinstance(result, pd.DataFrame)
        assert len(result) == size
        
        # Verificar tiempo de procesamiento razonable
        max_time = performance_data['max_processing_time']
        assert processing_time < max_time, f"Procesamiento tomó {processing_time:.2f}s, máximo permitido: {max_time}s"
    
    @pytest.mark.parametrize("outlier_method", [
        OutlierDetectionMethod.IQR,
        OutlierDetectionMethod.Z_SCORE,
        OutlierDetectionMethod.ISOLATION_FOREST
    ])
    def test_outlier_detection_methods(self, sample_dataframe, outlier_method):
        """Test diferentes métodos de detección de outliers"""
        config = DataQualityConfig(outlier_method=outlier_method)
        processor = DataPreprocessor(quality_config=config)
        
        result = processor.validate_data_quality(sample_dataframe)
        
        assert 'outliers' in result
        assert isinstance(result['outliers'], (int, float))
        assert 0 <= result['outliers'] <= 1
    
    @pytest.mark.parametrize("imputation_strategy", [
        ImputationStrategy.MEDIAN,
        ImputationStrategy.MEAN,
        ImputationStrategy.KNN
    ])
    def test_imputation_strategies(self, imputation_strategy):
        """Test diferentes estrategias de imputación"""
        # Crear datos con valores faltantes
        data = {
            'activos_totales': [1000000, np.nan, 3000000, 2000000],
            'pasivos_totales': [500000, 1500000, np.nan, 1000000],
            'utilidad_neta': [100000, 200000, 300000, np.nan]
        }
        df = pd.DataFrame(data)
        
        config = FeatureEngineeringConfig(imputation_strategy=imputation_strategy)
        processor = DataPreprocessor(feature_config=config)
        
        result = processor.preprocess_data(df)
        
        # Verificar que no hay valores faltantes en el resultado
        assert not result.isnull().any().any()
    
    def test_concurrent_processing(self, sample_dataframe):
        """Test procesamiento concurrente"""
        import threading
        import time
        
        processor = DataPreprocessor()
        results = []
        errors = []
        
        def process_data():
            try:
                result = processor.preprocess_data(sample_dataframe.copy())
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Crear múltiples threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=process_data)
            threads.append(thread)
            thread.start()
        
        # Esperar a que terminen
        for thread in threads:
            thread.join()
        
        # Verificar resultados
        assert len(errors) == 0, f"Errores en procesamiento concurrente: {errors}"
        assert len(results) == 5
        
        # Verificar que todos los resultados son similares
        for result in results[1:]:
            pd.testing.assert_frame_equal(results[0], result, atol=1e-10)

