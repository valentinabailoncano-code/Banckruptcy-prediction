"""
Este módulo centraliza toda la configuración del sistema, incluyendo
conexiones a base de datos, parámetros de modelos, configuración de APIs
y variables de entorno.
"""

import os
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import json
from dotenv import load_dotenv

# Cargar variables de entorno desde archivo .env
load_dotenv()

# Directorio base del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass
class DatabaseConfig:
    """Configuración de la base de datos PostgreSQL"""
    host: str = os.getenv('DB_HOST', 'localhost')
    port: int = int(os.getenv('DB_PORT', '5432'))
    database: str = os.getenv('DB_NAME', 'bankruptcy_prediction')
    username: str = os.getenv('DB_USER', 'postgres')
    password: str = os.getenv('DB_PASSWORD', 'postgres')
    pool_size: int = int(os.getenv('DB_POOL_SIZE', '10'))
    max_overflow: int = int(os.getenv('DB_MAX_OVERFLOW', '20'))
    pool_timeout: int = int(os.getenv('DB_POOL_TIMEOUT', '30'))

    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

    @property
    def async_connection_string(self) -> str:
        return f"postgresql+asyncpg://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class ModelConfig:
    """Configuración del modelo de Machine Learning"""
    umbral_medio: float = float(os.getenv('MODEL_THRESHOLD_MEDIUM', '0.15'))
    umbral_alto: float = float(os.getenv('MODEL_THRESHOLD_HIGH', '0.30'))
    peso_ml_blended: float = float(os.getenv('MODEL_BLEND_WEIGHT', '0.7'))

    models_dir: Path = BASE_DIR / 'data' / 'models'
    model_file: str = os.getenv('MODEL_FILE', 'model_calibrated.pkl')
    features_file: str = os.getenv('FEATURES_FILE', 'feature_list.pkl')
    scaler_file: str = os.getenv('SCALER_FILE', 'scaler.pkl')

    test_size: float = float(os.getenv('MODEL_TEST_SIZE', '0.2'))
    validation_size: float = float(os.getenv('MODEL_VALIDATION_SIZE', '0.2'))
    random_state: int = int(os.getenv('MODEL_RANDOM_STATE', '42'))

    xgb_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': int(os.getenv('XGB_N_ESTIMATORS', '100')),
        'max_depth': int(os.getenv('XGB_MAX_DEPTH', '6')),
        'learning_rate': float(os.getenv('XGB_LEARNING_RATE', '0.1')),
        'subsample': float(os.getenv('XGB_SUBSAMPLE', '0.8')),
        'colsample_bytree': float(os.getenv('XGB_COLSAMPLE_BYTREE', '0.8')),
        'random_state': int(os.getenv('MODEL_RANDOM_STATE', '42')),
        'n_jobs': int(os.getenv('XGB_N_JOBS', '-1'))
    })

    smote_sampling_strategy: float = float(os.getenv('SMOTE_SAMPLING_STRATEGY', '0.5'))
    smote_k_neighbors: int = int(os.getenv('SMOTE_K_NEIGHBORS', '5'))


@dataclass
class APIConfig:
    """Configuración de la API de predicción"""
    host: str = os.getenv('API_HOST', '0.0.0.0')
    port: int = int(os.getenv('API_PORT', '8000'))
    debug: bool = os.getenv('API_DEBUG', 'False').lower() == 'true'
    workers: int = int(os.getenv('API_WORKERS', '4'))

    # Configuración de CORS (arreglado con default_factory)
    cors_origins: List[str] = field(default_factory=lambda: os.getenv('CORS_ORIGINS', '*').split(','))

    rate_limit_requests: int = int(os.getenv('RATE_LIMIT_REQUESTS', '100'))
    rate_limit_window: int = int(os.getenv('RATE_LIMIT_WINDOW', '60'))
    request_timeout: int = int(os.getenv('REQUEST_TIMEOUT', '30'))

    enable_auth: bool = os.getenv('ENABLE_AUTH', 'False').lower() == 'true'
    jwt_secret: str = os.getenv('JWT_SECRET', 'your-secret-key-change-in-production')
    jwt_expiration_hours: int = int(os.getenv('JWT_EXPIRATION_HOURS', '24'))


@dataclass
class StreamlitConfig:
    """Configuración de la aplicación Streamlit"""
    host: str = os.getenv('STREAMLIT_HOST', '0.0.0.0')
    port: int = int(os.getenv('STREAMLIT_PORT', '8501'))
    cache_ttl: int = int(os.getenv('STREAMLIT_CACHE_TTL', '3600'))
    page_title: str = os.getenv('STREAMLIT_PAGE_TITLE', 'Sistema de Predicción de Quiebras')
    page_icon: str = os.getenv('STREAMLIT_PAGE_ICON', '📉')
    layout: str = os.getenv('STREAMLIT_LAYOUT', 'wide')
    max_portfolio_size: int = int(os.getenv('MAX_PORTFOLIO_SIZE', '10000'))
    max_file_size_mb: int = int(os.getenv('MAX_FILE_SIZE_MB', '50'))


@dataclass
class AlertingConfig:
    """Configuración del sistema de alertas"""
    smtp_host: str = os.getenv('SMTP_HOST', 'smtp.gmail.com')
    smtp_port: int = int(os.getenv('SMTP_PORT', '587'))
    smtp_username: str = os.getenv('SMTP_USERNAME', '')
    smtp_password: str = os.getenv('SMTP_PASSWORD', '')
    smtp_use_tls: bool = os.getenv('SMTP_USE_TLS', 'True').lower() == 'true'

    slack_webhook_url: str = os.getenv('SLACK_WEBHOOK_URL', '')
    slack_channel: str = os.getenv('SLACK_CHANNEL', '#alerts')
    teams_webhook_url: str = os.getenv('TEAMS_WEBHOOK_URL', '')

    evaluation_frequency_hours: int = int(os.getenv('ALERT_FREQUENCY_HOURS', '24'))
    max_alerts_per_batch: int = int(os.getenv('MAX_ALERTS_PER_BATCH', '50'))
    alert_threshold_high: float = float(os.getenv('ALERT_THRESHOLD_HIGH', '0.30'))
    alert_threshold_deterioration: float = float(os.getenv('ALERT_THRESHOLD_DETERIORATION', '0.10'))


@dataclass
class PowerBIConfig:
    tenant_id: str = os.getenv('POWERBI_TENANT_ID', '')
    client_id: str = os.getenv('POWERBI_CLIENT_ID', '')
    client_secret: str = os.getenv('POWERBI_CLIENT_SECRET', '')
    workspace_id: str = os.getenv('POWERBI_WORKSPACE_ID', '')
    dataset_id: str = os.getenv('POWERBI_DATASET_ID', '')
    auto_refresh_enabled: bool = os.getenv('POWERBI_AUTO_REFRESH', 'False').lower() == 'true'
    refresh_frequency_hours: int = int(os.getenv('POWERBI_REFRESH_FREQUENCY', '6'))


@dataclass
class LoggingConfig:
    level: str = os.getenv('LOG_LEVEL', 'INFO')
    format: str = os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logs_dir: Path = BASE_DIR / 'logs'
    app_log_file: str = 'app.log'
    api_log_file: str = 'api.log'
    model_log_file: str = 'model.log'
    alerts_log_file: str = 'alerts.log'
    max_file_size_mb: int = int(os.getenv('LOG_MAX_FILE_SIZE_MB', '10'))
    backup_count: int = int(os.getenv('LOG_BACKUP_COUNT', '5'))
    log_to_database: bool = os.getenv('LOG_TO_DATABASE', 'False').lower() == 'true'


@dataclass
class DataConfig:
    data_dir: Path = BASE_DIR / 'data'
    raw_data_dir: Path = data_dir / 'raw'
    processed_data_dir: Path = data_dir / 'processed'
    features_file: str = 'features.parquet'
    fred_api_key: str = os.getenv('FRED_API_KEY', '')
    alpha_vantage_api_key: str = os.getenv('ALPHA_VANTAGE_API_KEY', '')
    quandl_api_key: str = os.getenv('QUANDL_API_KEY', '')
    retention_days_raw: int = int(os.getenv('DATA_RETENTION_RAW_DAYS', '365'))
    retention_days_processed: int = int(os.getenv('DATA_RETENTION_PROCESSED_DAYS', '1095'))
    enable_data_validation: bool = os.getenv('ENABLE_DATA_VALIDATION', 'True').lower() == 'true'
    max_missing_ratio: float = float(os.getenv('MAX_MISSING_RATIO', '0.3'))


@dataclass
class SecurityConfig:
    encryption_key: str = os.getenv('ENCRYPTION_KEY', 'your-encryption-key-change-in-production')
    session_timeout_minutes: int = int(os.getenv('SESSION_TIMEOUT_MINUTES', '60'))
    enable_audit_log: bool = os.getenv('ENABLE_AUDIT_LOG', 'True').lower() == 'true'

    # mutable default corregido
    allowed_ips: List[str] = field(default_factory=lambda: os.getenv('ALLOWED_IPS', '').split(',') if os.getenv('ALLOWED_IPS') else [])

    enable_security_headers: bool = os.getenv('ENABLE_SECURITY_HEADERS', 'True').lower() == 'true'

class SystemConfig:
    """Configuración principal del sistema que agrupa todas las configuraciones"""
    
    def __init__(self):
        self.database = DatabaseConfig()
        self.model = ModelConfig()
        self.api = APIConfig()
        self.streamlit = StreamlitConfig()
        self.alerting = AlertingConfig()
        self.powerbi = PowerBIConfig()
        self.logging = LoggingConfig()
        self.data = DataConfig()
        self.security = SecurityConfig()
        
        # Configuración del entorno
        self.environment = os.getenv('ENVIRONMENT', 'development')
        self.debug = os.getenv('DEBUG', 'False').lower() == 'true'
        self.version = os.getenv('APP_VERSION', '1.0.0')
        
        # Crear directorios necesarios
        self._create_directories()
        
        # Configurar logging
        self._setup_logging()
    
    def _create_directories(self):
        """Crea los directorios necesarios si no existen"""
        directories = [
            self.logging.logs_dir,
            self.data.data_dir,
            self.data.raw_data_dir,
            self.data.processed_data_dir,
            self.model.models_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Configura el sistema de logging"""
        logging.basicConfig(
            level=getattr(logging, self.logging.level.upper()),
            format=self.logging.format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(
                    self.logging.logs_dir / self.logging.app_log_file,
                    encoding='utf-8'
                )
            ]
        )
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Retorna la configuración como diccionario para serialización"""
        return {
            'environment': self.environment,
            'debug': self.debug,
            'version': self.version,
            'database': {
                'host': self.database.host,
                'port': self.database.port,
                'database': self.database.database,
                'pool_size': self.database.pool_size
            },
            'model': {
                'umbral_medio': self.model.umbral_medio,
                'umbral_alto': self.model.umbral_alto,
                'peso_ml_blended': self.model.peso_ml_blended
            },
            'api': {
                'host': self.api.host,
                'port': self.api.port,
                'debug': self.api.debug
            }
        }
    
    def validate_config(self) -> bool:
        """Valida que la configuración sea correcta"""
        errors = []
        
        # Validar umbrales del modelo
        if self.model.umbral_medio >= self.model.umbral_alto:
            errors.append("El umbral medio debe ser menor que el umbral alto")
        
        if not (0 <= self.model.peso_ml_blended <= 1):
            errors.append("El peso del ML en blended debe estar entre 0 y 1")
        
        # Validar configuración de base de datos
        if not all([self.database.host, self.database.database, 
                   self.database.username, self.database.password]):
            errors.append("Faltan parámetros de configuración de base de datos")
        
        # Validar configuración de alertas si está habilitada
        if self.alerting.smtp_username and not self.alerting.smtp_password:
            errors.append("Se requiere contraseña SMTP si se especifica usuario")
        
        if errors:
            for error in errors:
                logging.error(f"Error de configuración: {error}")
            return False
        
        return True
    
    def save_config_to_file(self, filepath: Optional[Path] = None):
        """Guarda la configuración actual en un archivo JSON"""
        if filepath is None:
            filepath = BASE_DIR / 'config' / 'current_config.json'
        
        config_dict = self.get_config_dict()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Configuración guardada en {filepath}")

# Instancia global de configuración
config = SystemConfig()

# Validar configuración al importar
if not config.validate_config():
    logging.warning("La configuración tiene errores. Revise los logs para más detalles.")

# Funciones de utilidad para acceso rápido a configuraciones comunes
def get_db_connection_string() -> str:
    """Retorna la cadena de conexión a la base de datos"""
    return config.database.connection_string

def get_model_thresholds() -> tuple:
    """Retorna los umbrales del modelo (medio, alto)"""
    return config.model.umbral_medio, config.model.umbral_alto

def get_api_config() -> tuple:
    """Retorna la configuración de la API (host, port)"""
    return config.api.host, config.api.port

def is_development() -> bool:
    """Retorna True si estamos en entorno de desarrollo"""
    return config.environment == 'development'

def is_production() -> bool:
    """Retorna True si estamos en entorno de producción"""
    return config.environment == 'production'

# Exportar configuración para uso en otros módulos
__all__ = [
    'config',
    'SystemConfig',
    'DatabaseConfig',
    'ModelConfig',
    'APIConfig',
    'StreamlitConfig',
    'AlertingConfig',
    'PowerBIConfig',
    'LoggingConfig',
    'DataConfig',
    'SecurityConfig',
    'get_db_connection_string',
    'get_model_thresholds',
    'get_api_config',
    'is_development',
    'is_production'
]

