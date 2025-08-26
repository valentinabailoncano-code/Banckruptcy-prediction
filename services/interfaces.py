"""
Este módulo define las interfaces abstractas y contratos que deben
implementar todos los servicios del sistema. Esto asegura consistencia,
facilita el testing y permite intercambiar implementaciones.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, date
import pandas as pd
import numpy as np
from enum import Enum

# =====================================================
# TIPOS DE DATOS Y ENUMS
# =====================================================

class RiskBand(Enum):
    """Bandas de riesgo para clasificación"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

class AltmanBand(Enum):
    """Bandas de Altman Z-Score"""
    SAFE = "SAFE"
    GREY = "GREY"
    DISTRESS = "DISTRESS"

class AlertSeverity(Enum):
    """Niveles de severidad de alertas"""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

class AlertType(Enum):
    """Tipos de alertas"""
    RIESGO_ALTO = "RIESGO_ALTO"
    DETERIORO_RAPIDO = "DETERIORO_RAPIDO"
    UMBRAL_SUPERADO = "UMBRAL_SUPERADO"
    MODELO_DRIFT = "MODELO_DRIFT"

class ModelStatus(Enum):
    """Estados del modelo"""
    TRAINING = "TRAINING"
    VALIDATING = "VALIDATING"
    READY = "READY"
    DEPLOYED = "DEPLOYED"
    DEPRECATED = "DEPRECATED"

# =====================================================
# DATACLASSES PARA TRANSFERENCIA DE DATOS
# =====================================================

@dataclass
class CompanyData:
    """Datos básicos de una empresa"""
    id: Optional[int] = None
    uuid: Optional[str] = None
    nombre: str = ""
    codigo_empresa: Optional[str] = None
    sector: str = ""
    subsector: Optional[str] = None
    pais: str = ""
    region: Optional[str] = None
    tamaño_empresa: Optional[str] = None
    fecha_fundacion: Optional[date] = None
    activa: bool = True
    notas: Optional[str] = None

@dataclass
class FinancialData:
    """Datos financieros de una empresa"""
    empresa_id: int
    fecha_corte: date
    periodo: str  # 'Q1', 'Q2', 'Q3', 'Q4', 'ANUAL'
    año: int
    
    # Ratios principales
    wc_ta: Optional[float] = None  # Working Capital / Total Assets
    re_ta: Optional[float] = None  # Retained Earnings / Total Assets
    ebit_ta: Optional[float] = None  # EBIT / Total Assets
    me_tl: Optional[float] = None  # Market Equity / Total Liabilities
    s_ta: Optional[float] = None  # Sales / Total Assets
    
    # Ratios adicionales
    ocf_ta: Optional[float] = None  # Operating Cash Flow / Total Assets
    debt_assets: Optional[float] = None  # Total Liabilities / Total Assets
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    debt_equity: Optional[float] = None
    roa: Optional[float] = None  # Return on Assets
    roe: Optional[float] = None  # Return on Equity
    
    # Métricas de crecimiento
    revenue_growth_yoy: Optional[float] = None
    ebitda_growth_yoy: Optional[float] = None
    volatilidad_ingresos: Optional[float] = None
    
    # Metadatos
    fuente_datos: Optional[str] = None
    validado: bool = False

@dataclass
class MacroeconomicData:
    """Datos macroeconómicos"""
    fecha: date
    pais: str
    
    # Variables principales
    gdp_yoy: Optional[float] = None
    gdp_qoq: Optional[float] = None
    unemp_rate: Optional[float] = None
    inflation_rate: Optional[float] = None
    pmi: Optional[float] = None
    
    # Tipos de interés
    y10y: Optional[float] = None
    y3m: Optional[float] = None
    y2y: Optional[float] = None
    credit_spread: Optional[float] = None
    term_spread: Optional[float] = None
    
    # Indicadores de mercado
    stock_index_return: Optional[float] = None
    volatility_index: Optional[float] = None
    exchange_rate_usd: Optional[float] = None
    
    # Indicadores de confianza
    consumer_confidence: Optional[float] = None
    business_confidence: Optional[float] = None
    
    fuente_datos: Optional[str] = None

@dataclass
class PredictionRequest:
    """Solicitud de predicción"""
    empresa_id: int
    fecha_datos: date
    financial_data: FinancialData
    macro_data: Optional[MacroeconomicData] = None
    modelo_id: Optional[int] = None
    include_explanations: bool = False

@dataclass
class PredictionResult:
    """Resultado de una predicción"""
    empresa_id: int
    modelo_id: int
    fecha_prediccion: datetime
    fecha_datos: date
    
    # Resultados principales
    probabilidad_quiebra: float
    altman_z_score: float
    banda_riesgo_ml: RiskBand
    banda_riesgo_altman: AltmanBand
    blended_score: float
    
    # Umbrales utilizados
    umbral_medio: float
    umbral_alto: float
    peso_ml_blended: float
    
    # Explicabilidad
    top_features_positivas: Optional[Dict[str, float]] = None
    top_features_negativas: Optional[Dict[str, float]] = None
    
    # Metadatos
    version_api: Optional[str] = None
    tiempo_procesamiento_ms: Optional[int] = None

@dataclass
class ModelMetrics:
    """Métricas de rendimiento del modelo"""
    roc_auc_train: Optional[float] = None
    roc_auc_test: Optional[float] = None
    pr_auc_train: Optional[float] = None
    pr_auc_test: Optional[float] = None
    brier_score_train: Optional[float] = None
    brier_score_test: Optional[float] = None
    ks_statistic_train: Optional[float] = None
    ks_statistic_test: Optional[float] = None

@dataclass
class ModelInfo:
    """Información del modelo"""
    id: Optional[int] = None
    uuid: Optional[str] = None
    nombre: str = ""
    version: str = ""
    algoritmo: str = ""
    descripcion: Optional[str] = None
    
    # Metadatos de entrenamiento
    fecha_entrenamiento: Optional[datetime] = None
    dataset_entrenamiento: Optional[str] = None
    tamaño_dataset_train: Optional[int] = None
    tamaño_dataset_test: Optional[int] = None
    
    # Hiperparámetros y métricas
    hiperparametros: Optional[Dict[str, Any]] = None
    metrics: Optional[ModelMetrics] = None
    feature_importances: Optional[Dict[str, float]] = None
    
    # Estado
    status: ModelStatus = ModelStatus.TRAINING
    activo: bool = False
    en_produccion: bool = False
    
    # Rutas
    ruta_modelo: Optional[str] = None
    ruta_preprocessor: Optional[str] = None
    ruta_scaler: Optional[str] = None

@dataclass
class AlertData:
    """Datos de una alerta"""
    id: Optional[int] = None
    empresa_id: int
    prediccion_id: Optional[int] = None
    tipo_alerta: AlertType
    severidad: AlertSeverity
    
    # Contenido
    titulo: str
    mensaje: str
    probabilidad_actual: Optional[float] = None
    probabilidad_anterior: Optional[float] = None
    cambio_probabilidad: Optional[float] = None
    
    # Estado
    estado: str = "PENDIENTE"
    fecha_creacion: Optional[datetime] = None
    fecha_envio: Optional[datetime] = None
    
    # Destinatarios
    destinatarios: Optional[List[str]] = None
    canales_envio: Optional[List[str]] = None

# =====================================================
# INTERFACES DE SERVICIOS
# =====================================================

class IDataIngestionService(ABC):
    """Interfaz para el servicio de ingesta de datos"""
    
    @abstractmethod
    async def ingest_financial_data(
        self, 
        source: str, 
        data: Union[pd.DataFrame, List[FinancialData]]
    ) -> bool:
        """Ingesta datos financieros desde una fuente"""
        pass
    
    @abstractmethod
    async def ingest_macro_data(
        self, 
        source: str, 
        data: Union[pd.DataFrame, List[MacroeconomicData]]
    ) -> bool:
        """Ingesta datos macroeconómicos"""
        pass
    
    @abstractmethod
    async def validate_data(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Valida la calidad de los datos"""
        pass
    
    @abstractmethod
    async def get_data_sources(self) -> List[str]:
        """Retorna las fuentes de datos disponibles"""
        pass

class IDataPreprocessingService(ABC):
    """Interfaz para el servicio de preprocesamiento de datos"""
    
    @abstractmethod
    async def preprocess_financial_data(
        self, 
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """Preprocesa datos financieros"""
        pass
    
    @abstractmethod
    async def calculate_ratios(
        self, 
        financial_data: FinancialData
    ) -> FinancialData:
        """Calcula ratios financieros derivados"""
        pass
    
    @abstractmethod
    async def calculate_altman_z_score(
        self, 
        financial_data: FinancialData
    ) -> float:
        """Calcula el Altman Z-Score"""
        pass
    
    @abstractmethod
    async def engineer_features(
        self, 
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """Realiza ingeniería de características"""
        pass
    
    @abstractmethod
    async def handle_missing_values(
        self, 
        data: pd.DataFrame, 
        strategy: str = "median"
    ) -> pd.DataFrame:
        """Maneja valores faltantes"""
        pass

class IModelTrainingService(ABC):
    """Interfaz para el servicio de entrenamiento de modelos"""
    
    @abstractmethod
    async def train_model(
        self, 
        training_data: pd.DataFrame, 
        target_column: str,
        model_config: Dict[str, Any]
    ) -> ModelInfo:
        """Entrena un nuevo modelo"""
        pass
    
    @abstractmethod
    async def validate_model(
        self, 
        model_info: ModelInfo, 
        validation_data: pd.DataFrame
    ) -> ModelMetrics:
        """Valida el rendimiento del modelo"""
        pass
    
    @abstractmethod
    async def optimize_hyperparameters(
        self, 
        training_data: pd.DataFrame, 
        target_column: str,
        param_space: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimiza hiperparámetros del modelo"""
        pass
    
    @abstractmethod
    async def save_model(self, model_info: ModelInfo) -> bool:
        """Guarda el modelo entrenado"""
        pass
    
    @abstractmethod
    async def load_model(self, model_id: int) -> ModelInfo:
        """Carga un modelo existente"""
        pass
    
    @abstractmethod
    async def get_feature_importance(self, model_id: int) -> Dict[str, float]:
        """Obtiene la importancia de las características"""
        pass

class IPredictionService(ABC):
    """Interfaz para el servicio de predicción"""
    
    @abstractmethod
    async def predict_single(
        self, 
        request: PredictionRequest
    ) -> PredictionResult:
        """Realiza predicción para una empresa"""
        pass
    
    @abstractmethod
    async def predict_batch(
        self, 
        requests: List[PredictionRequest]
    ) -> List[PredictionResult]:
        """Realiza predicciones en lote"""
        pass
    
    @abstractmethod
    async def predict_portfolio(
        self, 
        portfolio_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Realiza predicciones para una cartera completa"""
        pass
    
    @abstractmethod
    async def explain_prediction(
        self, 
        prediction_result: PredictionResult
    ) -> Dict[str, Any]:
        """Explica una predicción específica"""
        pass
    
    @abstractmethod
    async def get_model_info(self, model_id: Optional[int] = None) -> ModelInfo:
        """Obtiene información del modelo activo o especificado"""
        pass

class IAlertingService(ABC):
    """Interfaz para el servicio de alertas"""
    
    @abstractmethod
    async def evaluate_alerts(self) -> List[AlertData]:
        """Evalúa y genera alertas basadas en las predicciones"""
        pass
    
    @abstractmethod
    async def send_alert(self, alert: AlertData) -> bool:
        """Envía una alerta específica"""
        pass
    
    @abstractmethod
    async def send_batch_alerts(self, alerts: List[AlertData]) -> int:
        """Envía múltiples alertas y retorna el número enviado exitosamente"""
        pass
    
    @abstractmethod
    async def configure_alert_rules(self, rules: Dict[str, Any]) -> bool:
        """Configura las reglas de generación de alertas"""
        pass
    
    @abstractmethod
    async def get_alert_history(
        self, 
        empresa_id: Optional[int] = None,
        days: int = 30
    ) -> List[AlertData]:
        """Obtiene el historial de alertas"""
        pass

class IDatabaseService(ABC):
    """Interfaz para el servicio de base de datos"""
    
    @abstractmethod
    async def save_company(self, company: CompanyData) -> int:
        """Guarda información de una empresa"""
        pass
    
    @abstractmethod
    async def save_financial_data(self, financial_data: FinancialData) -> bool:
        """Guarda datos financieros"""
        pass
    
    @abstractmethod
    async def save_prediction(self, prediction: PredictionResult) -> int:
        """Guarda resultado de predicción"""
        pass
    
    @abstractmethod
    async def get_company(self, company_id: int) -> Optional[CompanyData]:
        """Obtiene información de una empresa"""
        pass
    
    @abstractmethod
    async def get_latest_financial_data(
        self, 
        company_id: int
    ) -> Optional[FinancialData]:
        """Obtiene los datos financieros más recientes de una empresa"""
        pass
    
    @abstractmethod
    async def get_predictions_history(
        self, 
        company_id: int, 
        days: int = 365
    ) -> List[PredictionResult]:
        """Obtiene historial de predicciones"""
        pass
    
    @abstractmethod
    async def execute_query(
        self, 
        query: str, 
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Ejecuta una consulta SQL personalizada"""
        pass

class IPowerBIService(ABC):
    """Interfaz para integración con Power BI"""
    
    @abstractmethod
    async def refresh_dataset(self, dataset_id: str) -> bool:
        """Actualiza un dataset en Power BI"""
        pass
    
    @abstractmethod
    async def push_data_to_dataset(
        self, 
        dataset_id: str, 
        table_name: str, 
        data: pd.DataFrame
    ) -> bool:
        """Envía datos a una tabla de Power BI"""
        pass
    
    @abstractmethod
    async def get_dataset_info(self, dataset_id: str) -> Dict[str, Any]:
        """Obtiene información de un dataset"""
        pass
    
    @abstractmethod
    async def create_report_url(
        self, 
        report_id: str, 
        filters: Optional[Dict[str, Any]] = None
    ) -> str:
        """Genera URL de reporte con filtros"""
        pass

# =====================================================
# FACTORY PATTERN PARA SERVICIOS
# =====================================================

class ServiceFactory:
    """Factory para crear instancias de servicios"""
    
    _services: Dict[str, Any] = {}
    
    @classmethod
    def register_service(cls, service_name: str, service_class: type):
        """Registra una implementación de servicio"""
        cls._services[service_name] = service_class
    
    @classmethod
    def create_service(cls, service_name: str, **kwargs) -> Any:
        """Crea una instancia del servicio especificado"""
        if service_name not in cls._services:
            raise ValueError(f"Servicio '{service_name}' no registrado")
        
        service_class = cls._services[service_name]
        return service_class(**kwargs)
    
    @classmethod
    def get_available_services(cls) -> List[str]:
        """Retorna la lista de servicios disponibles"""
        return list(cls._services.keys())

# =====================================================
# DECORADORES PARA SERVICIOS
# =====================================================

def service_method(func):
    """Decorador para métodos de servicio que añade logging y manejo de errores"""
    import functools
    import logging
    
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        service_name = args[0].__class__.__name__ if args else "Unknown"
        method_name = func.__name__
        
        logger.info(f"{service_name}.{method_name} - Iniciando")
        
        try:
            result = await func(*args, **kwargs)
            logger.info(f"{service_name}.{method_name} - Completado exitosamente")
            return result
        except Exception as e:
            logger.error(f"{service_name}.{method_name} - Error: {str(e)}")
            raise
    
    return wrapper

def validate_input(validation_func):
    """Decorador para validar inputs de métodos de servicio"""
    def decorator(func):
        import functools
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Validar inputs usando la función de validación proporcionada
            validation_result = validation_func(*args, **kwargs)
            if not validation_result:
                raise ValueError("Validación de input falló")
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator

# =====================================================
# UTILIDADES PARA SERVICIOS
# =====================================================

class ServiceResponse:
    """Clase para estandarizar respuestas de servicios"""
    
    def __init__(
        self, 
        success: bool, 
        data: Any = None, 
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.success = success
        self.data = data
        self.error = error
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte la respuesta a diccionario"""
        return {
            'success': self.success,
            'data': self.data,
            'error': self.error,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }

# Exportar todas las interfaces y clases principales
__all__ = [
    # Enums
    'RiskBand', 'AltmanBand', 'AlertSeverity', 'AlertType', 'ModelStatus',
    
    # Dataclasses
    'CompanyData', 'FinancialData', 'MacroeconomicData', 
    'PredictionRequest', 'PredictionResult', 'ModelMetrics', 'ModelInfo', 'AlertData',
    
    # Interfaces
    'IDataIngestionService', 'IDataPreprocessingService', 'IModelTrainingService',
    'IPredictionService', 'IAlertingService', 'IDatabaseService', 'IPowerBIService',
    
    # Utilidades
    'ServiceFactory', 'ServiceResponse', 'service_method', 'validate_input'
]

