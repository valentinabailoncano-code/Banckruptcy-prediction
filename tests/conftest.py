"""
Este módulo contiene la configuración global para todos los tests
del sistema, incluyendo fixtures, configuraciones y utilidades.
"""

import pytest
import asyncio
import tempfile
import shutil
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock
import sqlite3

# Agregar el directorio raíz al path para imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import DatabaseManager
from services.models import Base, Empresa, DatoFinanciero, Prediccion, Alerta
from services.preprocessing.data_processor import DataPreprocessor
from services.model_training.trainer import ModelTrainer
from services.prediction.predictor import PredictionService
from services.alerts.alert_engine import AlertEngine
from services.notifications.notification_service import NotificationService
from powerbi.connectors.powerbi_connector import PowerBIConnector

# Configuración global de pytest
pytest_plugins = ['pytest_asyncio']

@pytest.fixture(scope="session")
def event_loop():
    """Crea un event loop para toda la sesión de testing"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def test_config():
    """Configuración de testing"""
    return {
        'database': {
            'url': 'sqlite:///:memory:',
            'echo': False
        },
        'testing': True,
        'debug': False,
        'ml_models': {
            'cache_dir': '/tmp/test_models',
            'max_features': 50
        },
        'notifications': {
            'enabled': False
        }
    }

@pytest.fixture(scope="session")
def temp_directory():
    """Directorio temporal para archivos de testing"""
    temp_dir = tempfile.mkdtemp(prefix='bankruptcy_test_')
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture(scope="function")
def test_db():
    """Base de datos en memoria para testing"""
    # Crear base de datos temporal
    db_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    db_file.close()
    
    db_url = f'sqlite:///{db_file.name}'
    
    # Configurar base de datos
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    engine = create_engine(db_url, echo=False)
    Base.metadata.create_all(engine)
    
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()
    
    yield session
    
    # Cleanup
    session.close()
    engine.dispose()
    os.unlink(db_file.name)

@pytest.fixture
def sample_empresa_data():
    """Datos de ejemplo para empresa"""
    return {
        'rut': '12345678-9',
        'razon_social': 'Empresa Test S.A.',
        'nombre_fantasia': 'Test Corp',
        'sector': 'Technology',
        'subsector': 'Software',
        'pais': 'Chile',
        'region': 'Metropolitana',
        'ciudad': 'Santiago',
        'tamaño_empresa': 'MEDIANA',
        'activa': True,
        'fecha_constitucion': datetime(2020, 1, 1),
        'numero_empleados': 150,
        'ingresos_anuales': 5000000.0,
        'es_publica': False
    }

@pytest.fixture
def sample_datos_financieros():
    """Datos financieros de ejemplo"""
    return {
        'periodo': datetime(2024, 12, 31),
        'tipo_periodo': 'anual',
        'activos_totales': 10000000.0,
        'pasivos_totales': 6000000.0,
        'patrimonio': 4000000.0,
        'ingresos_operacionales': 8000000.0,
        'costos_ventas': 5000000.0,
        'gastos_operacionales': 2000000.0,
        'utilidad_operacional': 1000000.0,
        'gastos_financieros': 200000.0,
        'utilidad_antes_impuestos': 800000.0,
        'impuestos': 200000.0,
        'utilidad_neta': 600000.0,
        'ebitda': 1200000.0,
        'flujo_caja_operacional': 800000.0,
        'activos_corrientes': 4000000.0,
        'pasivos_corrientes': 2000000.0,
        'inventarios': 1000000.0,
        'cuentas_por_cobrar': 1500000.0,
        'efectivo': 500000.0,
        'deuda_total': 3000000.0,
        'deuda_corto_plazo': 1000000.0,
        'deuda_largo_plazo': 2000000.0,
        'capital_trabajo': 2000000.0,
        'ventas_netas': 8000000.0,
        'valor_mercado_acciones': 5000000.0
    }

@pytest.fixture
def sample_prediccion_data():
    """Datos de predicción de ejemplo"""
    return {
        'fecha_prediccion': datetime.now(),
        'probabilidad_ml': 0.25,
        'probabilidad_altman': 0.30,
        'probabilidad_combinada': 0.275,
        'banda_riesgo': 'MEDIUM',
        'confianza_prediccion': 0.85,
        'modelo_version': 'v1.0.0',
        'tiempo_procesamiento': 150.5,
        'explicabilidad_shap': '{"feature_1": 0.1, "feature_2": -0.05}',
        'top_features_positivas': 'ratio_liquidez,roa,margen_operacional',
        'top_features_negativas': 'ratio_endeudamiento,gastos_financieros',
        'observaciones': 'Predicción dentro de rangos normales'
    }

@pytest.fixture
def sample_alerta_data():
    """Datos de alerta de ejemplo"""
    return {
        'tipo': 'RIESGO_CRITICO',
        'severidad': 'HIGH',
        'estado': 'ACTIVA',
        'mensaje': 'Probabilidad de quiebra superior al 80%',
        'fecha_creacion': datetime.now(),
        'prioridad': 1,
        'metadatos': '{"threshold": 0.8, "actual_value": 0.85}'
    }

@pytest.fixture
def sample_dataframe():
    """DataFrame de ejemplo para testing"""
    np.random.seed(42)
    
    data = {
        'empresa_id': range(1, 101),
        'rut': [f'{i:08d}-{i%10}' for i in range(1, 101)],
        'razon_social': [f'Empresa {i} S.A.' for i in range(1, 101)],
        'sector': np.random.choice(['Technology', 'Finance', 'Retail', 'Manufacturing'], 100),
        'activos_totales': np.random.uniform(1000000, 50000000, 100),
        'pasivos_totales': np.random.uniform(500000, 30000000, 100),
        'ingresos_operacionales': np.random.uniform(800000, 40000000, 100),
        'utilidad_neta': np.random.uniform(-1000000, 5000000, 100),
        'ratio_liquidez_corriente': np.random.uniform(0.5, 3.0, 100),
        'ratio_endeudamiento': np.random.uniform(0.2, 0.8, 100),
        'roa': np.random.uniform(-0.1, 0.2, 100),
        'roe': np.random.uniform(-0.2, 0.3, 100)
    }
    
    return pd.DataFrame(data)

@pytest.fixture
def mock_model():
    """Mock de modelo ML para testing"""
    model = Mock()
    model.predict_proba.return_value = np.array([[0.7, 0.3], [0.8, 0.2], [0.6, 0.4]])
    model.predict.return_value = np.array([0, 0, 1])
    model.feature_importances_ = np.random.random(10)
    return model

@pytest.fixture
def mock_preprocessor():
    """Mock de preprocesador para testing"""
    preprocessor = Mock(spec=DataPreprocessor)
    preprocessor.validate_data_quality.return_value = {'completeness': 0.95, 'outliers': 0.02}
    preprocessor.calculate_financial_ratios.return_value = pd.DataFrame()
    preprocessor.engineer_features.return_value = pd.DataFrame()
    return preprocessor

@pytest.fixture
def mock_notification_service():
    """Mock de servicio de notificaciones"""
    service = Mock(spec=NotificationService)
    service.send_notification.return_value = {'success': True, 'message_id': 'test_123'}
    service.send_custom_notification.return_value = {'success': True, 'message_id': 'custom_123'}
    return service

@pytest.fixture
def mock_powerbi_connector():
    """Mock de conector Power BI"""
    connector = Mock(spec=PowerBIConnector)
    connector.get_empresas_dataset.return_value = pd.DataFrame()
    connector.get_predicciones_dataset.return_value = pd.DataFrame()
    connector.get_datos_financieros_dataset.return_value = pd.DataFrame()
    return connector

@pytest.fixture
def api_client():
    """Cliente de API para testing"""
    from bankruptcy_api.src.main import create_app
    
    app = create_app(testing=True)
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False
    
    with app.test_client() as client:
        with app.app_context():
            yield client

@pytest.fixture
def auth_headers():
    """Headers de autenticación para testing de API"""
    # Mock JWT token para testing
    return {
        'Authorization': 'Bearer test_jwt_token',
        'Content-Type': 'application/json'
    }

@pytest.fixture
def sample_file_upload():
    """Archivo de ejemplo para testing de uploads"""
    import io
    
    # Crear CSV de ejemplo
    csv_content = """rut,razon_social,sector,activos_totales,pasivos_totales
12345678-9,Empresa Test S.A.,Technology,10000000,6000000
87654321-0,Otra Empresa Ltda.,Finance,5000000,3000000
"""
    
    return io.BytesIO(csv_content.encode('utf-8'))

@pytest.fixture
def performance_data():
    """Datos para testing de performance"""
    return {
        'large_dataset_size': 10000,
        'max_processing_time': 30.0,  # segundos
        'max_memory_usage': 500,  # MB
        'concurrent_requests': 10
    }

# Utilidades de testing
class TestUtils:
    """Utilidades para testing"""
    
    @staticmethod
    def create_test_empresa(session, **kwargs):
        """Crea empresa de testing en base de datos"""
        default_data = {
            'rut': '12345678-9',
            'razon_social': 'Test Company',
            'sector': 'Technology',
            'activa': True,
            'fecha_constitucion': datetime(2020, 1, 1)
        }
        default_data.update(kwargs)
        
        empresa = Empresa(**default_data)
        session.add(empresa)
        session.commit()
        session.refresh(empresa)
        return empresa
    
    @staticmethod
    def create_test_datos_financieros(session, empresa_id, **kwargs):
        """Crea datos financieros de testing"""
        default_data = {
            'empresa_id': empresa_id,
            'periodo': datetime(2024, 12, 31),
            'tipo_periodo': 'anual',
            'activos_totales': 10000000.0,
            'pasivos_totales': 6000000.0,
            'patrimonio': 4000000.0,
            'ingresos_operacionales': 8000000.0,
            'utilidad_neta': 600000.0
        }
        default_data.update(kwargs)
        
        datos = DatoFinanciero(**default_data)
        session.add(datos)
        session.commit()
        session.refresh(datos)
        return datos
    
    @staticmethod
    def create_test_prediccion(session, empresa_id, **kwargs):
        """Crea predicción de testing"""
        default_data = {
            'empresa_id': empresa_id,
            'fecha_prediccion': datetime.now(),
            'probabilidad_ml': 0.25,
            'probabilidad_altman': 0.30,
            'probabilidad_combinada': 0.275,
            'banda_riesgo': 'MEDIUM',
            'confianza_prediccion': 0.85,
            'modelo_version': 'v1.0.0'
        }
        default_data.update(kwargs)
        
        prediccion = Prediccion(**default_data)
        session.add(prediccion)
        session.commit()
        session.refresh(prediccion)
        return prediccion
    
    @staticmethod
    def create_test_alerta(session, empresa_id, **kwargs):
        """Crea alerta de testing"""
        default_data = {
            'empresa_id': empresa_id,
            'tipo': 'RIESGO_CRITICO',
            'severidad': 'HIGH',
            'estado': 'ACTIVA',
            'mensaje': 'Test alert message',
            'fecha_creacion': datetime.now(),
            'prioridad': 1
        }
        default_data.update(kwargs)
        
        alerta = Alerta(**default_data)
        session.add(alerta)
        session.commit()
        session.refresh(alerta)
        return alerta
    
    @staticmethod
    def assert_dataframe_equal(df1, df2, check_dtype=False):
        """Compara DataFrames con tolerancia para floats"""
        pd.testing.assert_frame_equal(df1, df2, check_dtype=check_dtype, atol=1e-6)
    
    @staticmethod
    def assert_prediction_valid(prediction_result):
        """Valida estructura de resultado de predicción"""
        required_fields = [
            'probabilidad_ml', 'probabilidad_altman', 'probabilidad_combinada',
            'banda_riesgo', 'confianza_prediccion'
        ]
        
        for field in required_fields:
            assert field in prediction_result, f"Campo requerido faltante: {field}"
        
        # Validar rangos
        assert 0 <= prediction_result['probabilidad_ml'] <= 1
        assert 0 <= prediction_result['probabilidad_altman'] <= 1
        assert 0 <= prediction_result['probabilidad_combinada'] <= 1
        assert 0 <= prediction_result['confianza_prediccion'] <= 1
        assert prediction_result['banda_riesgo'] in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']

@pytest.fixture
def test_utils():
    """Fixture para utilidades de testing"""
    return TestUtils

# Configuración de logging para testing
import logging
logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

# Markers personalizados
def pytest_configure(config):
    """Configuración personalizada de pytest"""
    config.addinivalue_line(
        "markers", "unit: marca tests unitarios"
    )
    config.addinivalue_line(
        "markers", "integration: marca tests de integración"
    )
    config.addinivalue_line(
        "markers", "e2e: marca tests end-to-end"
    )
    config.addinivalue_line(
        "markers", "performance: marca tests de performance"
    )
    config.addinivalue_line(
        "markers", "slow: marca tests que toman mucho tiempo"
    )

# Hooks de pytest
def pytest_collection_modifyitems(config, items):
    """Modifica items de testing según markers"""
    for item in items:
        # Agregar marker 'unit' por defecto si no tiene otros markers
        if not any(marker.name in ['integration', 'e2e', 'performance'] for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)

def pytest_runtest_setup(item):
    """Setup antes de cada test"""
    # Configurar variables de entorno para testing
    os.environ['TESTING'] = 'true'
    os.environ['DATABASE_URL'] = 'sqlite:///:memory:'

def pytest_runtest_teardown(item):
    """Cleanup después de cada test"""
    # Limpiar variables de entorno
    if 'TESTING' in os.environ:
        del os.environ['TESTING']

