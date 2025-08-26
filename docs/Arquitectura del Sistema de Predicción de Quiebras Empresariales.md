# Arquitectura del Sistema de Predicción de Quiebras Empresariales

## Resumen Ejecutivo

El Sistema de Predicción de Quiebras Empresariales es una solución integral de Machine Learning diseñada para anticipar la probabilidad de quiebra de empresas utilizando datos financieros, sectoriales y macroeconómicos. El sistema combina modelos supervisados avanzados con el tradicional Altman Z-Score, proporcionando predicciones explicables y alertas proactivas a través de múltiples canales de comunicación.

## Arquitectura General

### Visión de Alto Nivel

El sistema está construido siguiendo una arquitectura de microservicios modular que separa claramente las responsabilidades entre diferentes componentes. Esta arquitectura permite escalabilidad horizontal, mantenimiento independiente de módulos y flexibilidad para futuras extensiones.

```
┌─────────────────────────────────────────────────────────────────┐
│                    CAPA DE PRESENTACIÓN                        │
├─────────────────────────────────────────────────────────────────┤
│  Streamlit Dashboard  │  Power BI Dashboards  │  API REST      │
│  - 7 páginas         │  - Executive Overview │  - 50+ endpoints│
│  - Visualizaciones   │  - Risk Analysis      │  - JWT Auth     │
│  - Interactividad    │  - Sector Analysis    │  - Rate Limiting│
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                    CAPA DE SERVICIOS                           │
├─────────────────────────────────────────────────────────────────┤
│  ML Services         │  ETL Services        │  Alert Services   │
│  - Preprocessing     │  - Data Ingestion    │  - Alert Engine   │
│  - Model Training    │  - Validation        │  - Notifications  │
│  - Prediction        │  - Transformation    │  - Escalation     │
│  - Drift Detection   │  - Quality Control   │  - Multi-channel  │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                    CAPA DE DATOS                               │
├─────────────────────────────────────────────────────────────────┤
│  PostgreSQL/MySQL    │  Model Registry      │  File Storage     │
│  - Empresas          │  - Model Versions    │  - Reports        │
│  - Datos Financieros │  - Metrics           │  - Exports        │
│  - Predicciones      │  - Validation        │  - Templates      │
│  - Alertas           │  - Lifecycle         │  - Logs           │
└─────────────────────────────────────────────────────────────────┘
```

### Principios Arquitectónicos

**1. Separación de Responsabilidades**
- Cada módulo tiene una responsabilidad específica y bien definida
- Interfaces claras entre componentes
- Bajo acoplamiento y alta cohesión

**2. Escalabilidad**
- Arquitectura horizontal con microservicios independientes
- Pool de conexiones configurables
- Cache inteligente en múltiples niveles
- Procesamiento asíncrono para operaciones pesadas

**3. Robustez y Confiabilidad**
- Manejo comprehensivo de errores
- Logging detallado en todos los niveles
- Health checks automáticos
- Fallbacks y recuperación automática

**4. Seguridad**
- Autenticación JWT con roles granulares
- Validación de inputs en todos los endpoints
- Sanitización de datos SQL
- Auditoría completa de operaciones

**5. Mantenibilidad**
- Código modular y bien documentado
- Tests comprehensivos (unitarios, integración, E2E)
- Configuración centralizada
- Patrones de diseño consistentes

## Componentes Principales

### 1. Capa de Presentación

#### 1.1 Dashboard Streamlit

**Propósito**: Interfaz web interactiva para usuarios finales

**Características Técnicas**:
- Framework: Streamlit 1.28+
- Autenticación: JWT con session state
- Visualizaciones: Plotly, Matplotlib
- Responsive design con CSS personalizado

**Páginas Implementadas**:
1. **Dashboard Principal** - Overview ejecutivo con KPIs
2. **Gestión de Empresas** - CRUD completo con búsqueda avanzada
3. **Predicciones ML** - Ejecución y análisis de modelos
4. **Análisis Sectorial** - Comparativas por industria
5. **Alertas y Monitoreo** - Sistema de notificaciones
6. **Carga de Datos** - ETL con validación
7. **Configuración** - Administración del sistema

**Arquitectura de Componentes**:
```python
streamlit_app/
├── main.py                 # Aplicación principal
├── config/
│   └── settings.py         # Configuración centralizada
├── components/
│   ├── auth.py            # Autenticación JWT
│   └── sidebar.py         # Navegación
├── pages/
│   ├── dashboard.py       # Dashboard principal
│   ├── empresas.py        # Gestión empresas
│   ├── predicciones.py    # ML predictions
│   ├── analisis_sectorial.py
│   ├── alertas.py
│   ├── carga_datos.py
│   └── configuracion.py
└── utils/
    └── api_client.py      # Cliente API REST
```

#### 1.2 Power BI Integration

**Propósito**: Dashboards ejecutivos para Business Intelligence

**Características Técnicas**:
- Conectores de datos optimizados
- Refresh automático configurable
- Templates predefinidos
- Exportación de configuración

**Dashboards Disponibles**:
1. **Executive Overview** - Métricas principales
2. **Risk Analysis** - Análisis detallado de riesgo
3. **Financial Analysis** - Ratios y tendencias financieras
4. **Operational Monitoring** - Métricas del sistema

#### 1.3 API REST

**Propósito**: Interfaz programática para integración externa

**Características Técnicas**:
- Framework: Flask 2.3+
- Autenticación: JWT con roles
- Documentación: Swagger/OpenAPI
- Rate limiting y throttling

**Grupos de Endpoints**:
- `/api/empresas` - CRUD de empresas (8 endpoints)
- `/api/predicciones` - ML predictions (6 endpoints)
- `/api/etl` - Procesos ETL (7 endpoints)
- `/api/auth` - Autenticación (8 endpoints)
- `/api/dashboard` - Métricas agregadas (6 endpoints)
- `/api/powerbi` - Conectores BI (5 endpoints)

### 2. Capa de Servicios

#### 2.1 Servicios de Machine Learning

**2.1.1 Preprocesamiento de Datos**

**Archivo**: `services/preprocessing/data_processor.py`

**Responsabilidades**:
- Validación de calidad de datos
- Cálculo de ratios financieros
- Altman Z-Score (3 variantes)
- Ingeniería de características
- Pipeline de preprocesamiento

**Configuración**:
```python
@dataclass
class DataQualityConfig:
    min_completeness: float = 0.7
    max_outlier_ratio: float = 0.05
    outlier_method: OutlierDetectionMethod = OutlierDetectionMethod.IQR
    outlier_threshold: float = 1.5

@dataclass  
class FeatureEngineeringConfig:
    include_ratios: bool = True
    include_sector_features: bool = True
    include_temporal_features: bool = True
    include_interaction_features: bool = False
    polynomial_degree: int = 1
    max_features: int = 200
```

**Ratios Financieros Calculados**:
- **Liquidez**: Corriente, Ácida, Efectivo
- **Actividad**: Rotación Activos, Inventarios, CxC
- **Rentabilidad**: ROA, ROE, Margen Neto, Margen Operacional
- **Apalancamiento**: Endeudamiento, Cobertura Intereses

**2.1.2 Entrenamiento de Modelos**

**Archivo**: `services/model_training/trainer.py`

**Responsabilidades**:
- Entrenamiento de modelos XGBoost
- Optimización de hiperparámetros con Optuna
- Validación cruzada temporal
- Calibración de probabilidades
- Balanceo de clases

**Configuración de Entrenamiento**:
```python
@dataclass
class TrainingConfig:
    model_type: str = 'xgboost'
    validation_strategy: str = 'temporal'
    test_size: float = 0.2
    cv_folds: int = 5
    class_balancing: bool = True
    balancing_method: str = 'smote'
    calibration: bool = True
    calibration_method: str = 'isotonic'
```

**Métricas de Evaluación**:
- ROC-AUC Score
- Precision-Recall AUC
- Brier Score (calibración)
- Kolmogorov-Smirnov Statistic
- Matriz de confusión
- Importancia de características

**2.1.3 Servicio de Predicción**

**Archivo**: `services/prediction/predictor.py`

**Responsabilidades**:
- Predicciones individuales y en lote
- Explicabilidad con SHAP/LIME
- Blending ML + Altman Z-Score
- Clasificación en bandas de riesgo
- Validación de inputs

**Bandas de Riesgo**:
```python
class RiskBand(Enum):
    LOW = "LOW"           # Probabilidad < 0.3
    MEDIUM = "MEDIUM"     # Probabilidad 0.3-0.6
    HIGH = "HIGH"         # Probabilidad 0.6-0.8
    CRITICAL = "CRITICAL" # Probabilidad > 0.8
```

**Blending de Scores**:
```python
def calculate_combined_score(ml_prob: float, altman_prob: float, 
                           ml_weight: float = 0.7) -> float:
    """Combina probabilidades ML y Altman con pesos configurables"""
    return ml_weight * ml_prob + (1 - ml_weight) * altman_prob
```

**2.1.4 Detección de Drift**

**Archivo**: `services/model_monitoring/drift_detector.py`

**Responsabilidades**:
- Detección de drift de datos
- Detección de drift de concepto
- Detección de drift de predicciones
- Alertas automáticas
- Reportes de drift

**Tests Estadísticos**:
- Kolmogorov-Smirnov Test
- Wasserstein Distance
- Chi-cuadrado Test
- Population Stability Index (PSI)
- Cramér's V

**2.1.5 Registro de Modelos**

**Archivo**: `services/model_monitoring/model_registry.py`

**Responsabilidades**:
- Versionado de modelos
- Metadatos y métricas
- Validación automatizada
- Gestión del ciclo de vida
- Comparación de modelos

**Estados del Modelo**:
```python
class ModelStatus(Enum):
    TRAINING = "training"
    VALIDATION = "validation"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
```

#### 2.2 Servicios ETL

**2.2.1 Ingesta de Datos**

**Archivo**: `services/etl/data_ingestion.py`

**Responsabilidades**:
- Carga de archivos CSV/Excel
- Validación de datos
- Transformación y normalización
- Detección de duplicados
- Logging de procesos

**Formatos Soportados**:
- CSV (UTF-8, Latin-1)
- Excel (.xlsx, .xls)
- JSON estructurado
- ZIP con múltiples archivos

**Validaciones Implementadas**:
- RUT chileno con dígito verificador
- Ecuación contable (Activos = Pasivos + Patrimonio)
- Rangos de ratios financieros
- Consistencia temporal
- Completitud de datos

#### 2.3 Servicios de Alertas

**2.3.1 Motor de Alertas**

**Archivo**: `services/alerts/alert_engine.py`

**Responsabilidades**:
- Evaluación de reglas de alerta
- Monitoreo continuo
- Escalamiento automático
- Cooldown inteligente
- Métricas de alertas

**Tipos de Alertas**:
1. **Riesgo Crítico** - Probabilidad ≥ 80%
2. **Deterioro Financiero** - Altman < 1.8 + Liquidez < 1.0
3. **Tendencia Negativa** - Deterioro en 3+ períodos
4. **Anomalía de Datos** - Score calidad < 70%
5. **Drift del Modelo** - Degradación significativa

**2.3.2 Servicio de Notificaciones**

**Archivo**: `services/notifications/notification_service.py`

**Responsabilidades**:
- Notificaciones multi-canal
- Templates configurables
- Escalamiento por severidad
- Tracking de entrega
- Fallbacks automáticos

**Canales Soportados**:
- Email SMTP con HTML
- Slack con webhooks
- Microsoft Teams con MessageCards
- Webhooks personalizados

#### 2.4 Servicios de Reporting

**2.4.1 Reportes Automáticos**

**Archivo**: `services/reporting/automated_reports.py`

**Responsabilidades**:
- Generación programada
- Múltiples formatos
- Distribución automática
- Templates personalizables
- Métricas de generación

**Tipos de Reportes**:
1. **Ejecutivo Semanal** - KPIs y tendencias
2. **Análisis Mensual** - Riesgo detallado
3. **Sectorial Trimestral** - Comparativas industria
4. **Operacional Diario** - Métricas sistema
5. **Alertas Diario** - Resumen notificaciones

### 3. Capa de Datos

#### 3.1 Base de Datos Principal

**Motor**: PostgreSQL 13+ / MySQL 8+ / SQLite 3.36+

**Esquema de Tablas**:

**3.1.1 Tabla Empresas**
```sql
CREATE TABLE empresas (
    id SERIAL PRIMARY KEY,
    rut VARCHAR(12) UNIQUE NOT NULL,
    razon_social VARCHAR(255) NOT NULL,
    nombre_fantasia VARCHAR(255),
    sector VARCHAR(100) NOT NULL,
    subsector VARCHAR(100),
    pais VARCHAR(50) DEFAULT 'Chile',
    region VARCHAR(100),
    ciudad VARCHAR(100),
    tamaño_empresa empresa_size NOT NULL,
    activa BOOLEAN DEFAULT TRUE,
    fecha_constitucion DATE,
    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    numero_empleados INTEGER,
    ingresos_anuales DECIMAL(15,2),
    es_publica BOOLEAN DEFAULT FALSE,
    ticker_bolsa VARCHAR(10),
    industria_codigo VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**3.1.2 Tabla Datos Financieros**
```sql
CREATE TABLE datos_financieros (
    id SERIAL PRIMARY KEY,
    empresa_id INTEGER REFERENCES empresas(id),
    periodo DATE NOT NULL,
    tipo_periodo periodo_type NOT NULL,
    activos_totales DECIMAL(15,2),
    pasivos_totales DECIMAL(15,2),
    patrimonio DECIMAL(15,2),
    ingresos_operacionales DECIMAL(15,2),
    utilidad_neta DECIMAL(15,2),
    ebitda DECIMAL(15,2),
    -- Ratios calculados
    ratio_liquidez_corriente DECIMAL(8,4),
    ratio_endeudamiento DECIMAL(8,4),
    roa DECIMAL(8,4),
    roe DECIMAL(8,4),
    -- Altman Z-Score
    altman_z_score DECIMAL(8,4),
    altman_z_score_modificado DECIMAL(8,4),
    altman_z_score_emergentes DECIMAL(8,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(empresa_id, periodo, tipo_periodo)
);
```

**3.1.3 Tabla Predicciones**
```sql
CREATE TABLE predicciones (
    id SERIAL PRIMARY KEY,
    empresa_id INTEGER REFERENCES empresas(id),
    fecha_prediccion TIMESTAMP NOT NULL,
    probabilidad_ml DECIMAL(8,6),
    probabilidad_altman DECIMAL(8,6),
    probabilidad_combinada DECIMAL(8,6),
    banda_riesgo risk_band NOT NULL,
    confianza_prediccion DECIMAL(8,6),
    modelo_version VARCHAR(50),
    tiempo_procesamiento DECIMAL(8,2),
    explicabilidad_shap TEXT,
    top_features_positivas TEXT,
    top_features_negativas TEXT,
    observaciones TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Índices Optimizados**:
```sql
-- Índices para consultas frecuentes
CREATE INDEX idx_empresas_rut ON empresas(rut);
CREATE INDEX idx_empresas_sector ON empresas(sector);
CREATE INDEX idx_empresas_activa ON empresas(activa);

CREATE INDEX idx_datos_financieros_empresa_periodo 
ON datos_financieros(empresa_id, periodo DESC);

CREATE INDEX idx_predicciones_empresa_fecha 
ON predicciones(empresa_id, fecha_prediccion DESC);

CREATE INDEX idx_predicciones_banda_riesgo 
ON predicciones(banda_riesgo, fecha_prediccion DESC);

CREATE INDEX idx_alertas_estado_severidad 
ON alertas(estado, severidad, fecha_creacion DESC);
```

#### 3.2 Model Registry

**Propósito**: Almacenamiento de metadatos de modelos ML

**Base de Datos**: SQLite para metadatos + filesystem para binarios

**Esquema**:
```sql
CREATE TABLE model_registry (
    id INTEGER PRIMARY KEY,
    model_name TEXT NOT NULL,
    version TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    status TEXT NOT NULL,
    metrics TEXT, -- JSON
    hyperparameters TEXT, -- JSON
    training_data_hash TEXT,
    model_path TEXT,
    created_at TIMESTAMP,
    validated_at TIMESTAMP,
    promoted_at TIMESTAMP,
    UNIQUE(model_name, version)
);
```

#### 3.3 Cache Layer

**Tecnología**: Redis 6+ (opcional) / In-memory Python

**Estrategias de Cache**:
- **API Responses**: TTL 5-60 minutos
- **Dashboard Data**: TTL 15 minutos
- **Model Predictions**: TTL 1 hora
- **Static Data**: TTL 24 horas

**Configuración**:
```python
CACHE_CONFIG = {
    'default_ttl': 900,  # 15 minutos
    'api_responses_ttl': 300,  # 5 minutos
    'dashboard_ttl': 900,  # 15 minutos
    'predictions_ttl': 3600,  # 1 hora
    'static_data_ttl': 86400,  # 24 horas
}
```

## Flujos de Datos

### 1. Flujo de Ingesta de Datos

```mermaid
graph TD
    A[Usuario carga archivo] --> B[Validación formato]
    B --> C[Validación datos]
    C --> D[Transformación]
    D --> E[Detección duplicados]
    E --> F[Inserción BD]
    F --> G[Log proceso]
    G --> H[Notificación resultado]
    
    C --> I[Error validación]
    I --> J[Log error]
    J --> K[Notificación error]
```

### 2. Flujo de Predicción ML

```mermaid
graph TD
    A[Request predicción] --> B[Validar inputs]
    B --> C[Cargar modelo]
    C --> D[Preprocesar datos]
    D --> E[Calcular Altman]
    E --> F[Predicción ML]
    F --> G[Blending scores]
    G --> H[Explicabilidad]
    H --> I[Guardar resultado]
    I --> J[Return response]
    
    B --> K[Error validación]
    K --> L[Return error]
```

### 3. Flujo de Alertas

```mermaid
graph TD
    A[Scheduler ejecuta] --> B[Evaluar reglas]
    B --> C[Detectar alertas]
    C --> D[Verificar cooldown]
    D --> E[Crear alerta]
    E --> F[Determinar canales]
    F --> G[Enviar notificaciones]
    G --> H[Log resultado]
    H --> I[Actualizar estado]
    
    D --> J[En cooldown]
    J --> K[Skip alerta]
```

## Configuración y Deployment

### Variables de Entorno

```bash
# Base de datos
DATABASE_URL=postgresql://user:pass@localhost:5432/bankruptcy_db
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30

# API
API_SECRET_KEY=your-secret-key-here
JWT_SECRET_KEY=your-jwt-secret-here
JWT_EXPIRATION_HOURS=24

# Machine Learning
ML_MODEL_PATH=/app/models
ML_CACHE_DIR=/app/cache
ML_MAX_FEATURES=200

# Notificaciones
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password

SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
TEAMS_WEBHOOK_URL=https://outlook.office.com/webhook/...

# Power BI
POWERBI_CLIENT_ID=your-powerbi-client-id
POWERBI_CLIENT_SECRET=your-powerbi-secret

# Logging
LOG_LEVEL=INFO
LOG_FILE=/app/logs/bankruptcy_system.log

# Cache
REDIS_URL=redis://localhost:6379/0
CACHE_DEFAULT_TTL=900
```

### Estructura de Deployment

```
bankruptcy_prediction_system/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
├── config/
│   ├── nginx.conf
│   ├── supervisor.conf
│   └── logging.conf
├── scripts/
│   ├── deploy.sh
│   ├── backup.sh
│   ├── migrate.sh
│   └── health_check.sh
├── database/
│   ├── migrations/
│   └── seeds/
└── monitoring/
    ├── prometheus.yml
    └── grafana/
```

### Docker Configuration

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY . .

# Crear usuario no-root
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Exponer puertos
EXPOSE 8000 8501

# Comando por defecto
CMD ["python", "bankruptcy_api/src/main.py"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  db:
    image: postgres:13
    environment:
      POSTGRES_DB: bankruptcy_db
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./database/schema.sql:/docker-entrypoint-initdb.d/schema.sql
    ports:
      - "5432:5432"

  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"

  api:
    build: .
    environment:
      DATABASE_URL: postgresql://postgres:password@db:5432/bankruptcy_db
      REDIS_URL: redis://redis:6379/0
    ports:
      - "8000:8000"
    depends_on:
      - db
      - redis
    volumes:
      - ./logs:/app/logs
      - ./models:/app/models

  streamlit:
    build: .
    command: streamlit run streamlit_app/main.py --server.port=8501 --server.address=0.0.0.0
    environment:
      API_BASE_URL: http://api:8000
    ports:
      - "8501:8501"
    depends_on:
      - api

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./config/nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - api
      - streamlit

volumes:
  postgres_data:
```

## Monitoreo y Observabilidad

### Logging

**Configuración**:
```python
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
        'json': {
            'format': '{"timestamp": "%(asctime)s", "logger": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}'
        }
    },
    'handlers': {
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': '/app/logs/bankruptcy_system.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'formatter': 'detailed'
        },
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'detailed'
        }
    },
    'loggers': {
        'bankruptcy_system': {
            'handlers': ['file', 'console'],
            'level': 'INFO',
            'propagate': False
        }
    }
}
```

### Health Checks

**Endpoints de Salud**:
- `/health` - Estado general del sistema
- `/health/db` - Estado de base de datos
- `/health/cache` - Estado de cache
- `/health/ml` - Estado de modelos ML

**Métricas Monitoreadas**:
- Tiempo de respuesta de APIs
- Uso de memoria y CPU
- Conexiones de base de datos
- Tasa de errores
- Throughput de predicciones
- Latencia de notificaciones

### Alertas de Sistema

**Umbrales de Alerta**:
```python
SYSTEM_ALERTS = {
    'api_response_time': 5.0,  # segundos
    'memory_usage': 0.85,      # 85%
    'cpu_usage': 0.80,         # 80%
    'disk_usage': 0.90,        # 90%
    'error_rate': 0.05,        # 5%
    'db_connections': 0.90     # 90% del pool
}
```

## Seguridad

### Autenticación y Autorización

**JWT Configuration**:
```python
JWT_CONFIG = {
    'algorithm': 'HS256',
    'expiration_hours': 24,
    'refresh_expiration_days': 30,
    'issuer': 'bankruptcy-prediction-system',
    'audience': 'bankruptcy-users'
}
```

**Roles y Permisos**:
```python
ROLES = {
    'usuario': {
        'empresas': ['read'],
        'datos_financieros': ['read'],
        'predicciones': ['read'],
        'etl': ['read'],
        'usuarios': []
    },
    'analista': {
        'empresas': ['read', 'write'],
        'datos_financieros': ['read', 'write'],
        'predicciones': ['read', 'write', 'execute'],
        'etl': ['read', 'execute'],
        'usuarios': ['read']
    },
    'admin': {
        'empresas': ['read', 'write', 'delete'],
        'datos_financieros': ['read', 'write', 'delete'],
        'predicciones': ['read', 'write', 'execute'],
        'etl': ['read', 'write', 'execute'],
        'usuarios': ['read', 'write', 'delete'],
        'sistema': ['read', 'write']
    }
}
```

### Validación de Datos

**Sanitización SQL**:
```python
def sanitize_sql_input(value: str) -> str:
    """Sanitiza inputs para prevenir SQL injection"""
    dangerous_chars = [';', '--', '/*', '*/', 'xp_', 'sp_']
    for char in dangerous_chars:
        value = value.replace(char, '')
    return value.strip()
```

**Validación de Archivos**:
```python
ALLOWED_EXTENSIONS = {'.csv', '.xlsx', '.xls'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
MAX_ROWS_PER_FILE = 100000
```

### Encriptación

**Configuración**:
```python
ENCRYPTION_CONFIG = {
    'algorithm': 'AES-256-GCM',
    'key_derivation': 'PBKDF2',
    'iterations': 100000,
    'salt_length': 32
}
```

## Performance y Optimización

### Optimizaciones de Base de Datos

**Índices Estratégicos**:
- Índices compuestos para consultas frecuentes
- Índices parciales para filtros comunes
- Índices de texto completo para búsquedas

**Particionamiento**:
```sql
-- Particionamiento por fecha para predicciones
CREATE TABLE predicciones_2024 PARTITION OF predicciones
FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');
```

**Connection Pooling**:
```python
DATABASE_CONFIG = {
    'pool_size': 20,
    'max_overflow': 30,
    'pool_timeout': 30,
    'pool_recycle': 3600,
    'pool_pre_ping': True
}
```

### Cache Strategies

**Multi-Level Caching**:
1. **Application Level** - In-memory Python dictionaries
2. **Redis Level** - Distributed cache
3. **Database Level** - Query result cache
4. **CDN Level** - Static assets

**Cache Invalidation**:
```python
def invalidate_cache_on_update(entity_type: str, entity_id: int):
    """Invalida cache cuando se actualiza una entidad"""
    cache_keys = [
        f"{entity_type}:{entity_id}",
        f"{entity_type}:list:*",
        f"dashboard:*",
        f"stats:*"
    ]
    for key in cache_keys:
        cache.delete_pattern(key)
```

### Optimizaciones de ML

**Model Caching**:
```python
class ModelCache:
    def __init__(self, max_models: int = 5):
        self.cache = {}
        self.max_models = max_models
        self.access_times = {}
    
    def get_model(self, model_version: str):
        if model_version in self.cache:
            self.access_times[model_version] = time.time()
            return self.cache[model_version]
        
        # Cargar modelo y gestionar cache
        model = self._load_model(model_version)
        self._manage_cache_size()
        self.cache[model_version] = model
        self.access_times[model_version] = time.time()
        return model
```

**Batch Processing**:
```python
def predict_batch(self, data: pd.DataFrame, batch_size: int = 1000) -> List[Dict]:
    """Procesa predicciones en lotes para optimizar performance"""
    results = []
    for i in range(0, len(data), batch_size):
        batch = data.iloc[i:i+batch_size]
        batch_results = self._predict_batch_internal(batch)
        results.extend(batch_results)
    return results
```

## Extensibilidad y Futuras Mejoras

### Arquitectura Pluggable

**Interface para Nuevos Modelos**:
```python
class ModelInterface(ABC):
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        pass
```

**Plugin System**:
```python
class PluginManager:
    def __init__(self):
        self.plugins = {}
    
    def register_plugin(self, name: str, plugin_class: Type):
        self.plugins[name] = plugin_class
    
    def get_plugin(self, name: str):
        return self.plugins.get(name)
```

### Roadmap de Mejoras

**Corto Plazo (3-6 meses)**:
- Integración con Apache Kafka para streaming
- Modelos de Deep Learning (LSTM, Transformers)
- API GraphQL para consultas flexibles
- Integración con Kubernetes

**Mediano Plazo (6-12 meses)**:
- AutoML para optimización automática
- Explicabilidad avanzada con LIME++
- Integración con Apache Airflow
- Multi-tenancy para múltiples clientes

**Largo Plazo (12+ meses)**:
- Modelos de NLP para análisis de noticias
- Computer Vision para análisis de documentos
- Blockchain para auditoría inmutable
- Edge computing para predicciones locales

## Conclusión

El Sistema de Predicción de Quiebras Empresariales representa una solución integral y robusta que combina las mejores prácticas de Machine Learning, ingeniería de software y arquitectura de sistemas. Su diseño modular y escalable permite adaptarse a las necesidades cambiantes del negocio mientras mantiene altos estándares de calidad, seguridad y performance.

La arquitectura implementada facilita el mantenimiento, la extensibilidad y la operación en entornos de producción empresarial, proporcionando una base sólida para la toma de decisiones basada en datos en el ámbito de la gestión de riesgos financieros.

