# Arquitectura Técnica del Sistema de Predicción de Quiebras Empresariales

## Tabla de Contenidos

1. [Visión General](#visión-general)
2. [Arquitectura de Alto Nivel](#arquitectura-de-alto-nivel)
3. [Componentes del Sistema](#componentes-del-sistema)
4. [Flujos de Datos](#flujos-de-datos)
5. [Tecnologías y Herramientas](#tecnologías-y-herramientas)
6. [Patrones de Diseño](#patrones-de-diseño)
7. [Seguridad](#seguridad)
8. [Escalabilidad y Rendimiento](#escalabilidad-y-rendimiento)
9. [Monitoreo y Observabilidad](#monitoreo-y-observabilidad)
10. [Despliegue](#despliegue)

## Visión General

El Sistema de Predicción de Quiebras Empresariales es una solución integral de Machine Learning diseñada para evaluar y predecir la probabilidad de quiebra de empresas utilizando datos financieros, sectoriales y macroeconómicos. El sistema está construido siguiendo principios de arquitectura de microservicios, con componentes modulares y bien definidos que permiten escalabilidad, mantenibilidad y extensibilidad.

### Objetivos Arquitectónicos

- **Modularidad**: Componentes independientes y reutilizables
- **Escalabilidad**: Capacidad de manejar grandes volúmenes de datos y usuarios
- **Confiabilidad**: Alta disponibilidad y tolerancia a fallos
- **Mantenibilidad**: Código limpio, bien documentado y fácil de modificar
- **Seguridad**: Protección de datos sensibles y acceso controlado
- **Observabilidad**: Monitoreo completo y trazabilidad de operaciones

## Arquitectura de Alto Nivel

```mermaid
graph TB
    subgraph "Frontend Layer"
        UI[Streamlit App]
        PBI[Power BI Dashboards]
        API_CLIENT[API Clients]
    end
    
    subgraph "API Gateway"
        GATEWAY[API Gateway / Load Balancer]
    end
    
    subgraph "Application Layer"
        PRED_API[Prediction API]
        ALERT_SVC[Alerting Service]
        TRAIN_SVC[Training Service]
        PREPROC_SVC[Preprocessing Service]
        INGEST_SVC[Data Ingestion Service]
    end
    
    subgraph "Data Layer"
        POSTGRES[(PostgreSQL)]
        REDIS[(Redis Cache)]
        S3[(Object Storage)]
    end
    
    subgraph "External Services"
        SMTP[SMTP Server]
        SLACK[Slack API]
        TEAMS[Teams API]
        POWERBI_API[Power BI API]
        DATA_APIS[Financial Data APIs]
    end
    
    subgraph "Infrastructure"
        MONITORING[Monitoring Stack]
        LOGGING[Centralized Logging]
        SCHEDULER[Task Scheduler]
    end
    
    UI --> GATEWAY
    PBI --> GATEWAY
    API_CLIENT --> GATEWAY
    
    GATEWAY --> PRED_API
    GATEWAY --> ALERT_SVC
    GATEWAY --> TRAIN_SVC
    
    PRED_API --> POSTGRES
    PRED_API --> REDIS
    ALERT_SVC --> POSTGRES
    ALERT_SVC --> SMTP
    ALERT_SVC --> SLACK
    ALERT_SVC --> TEAMS
    
    TRAIN_SVC --> POSTGRES
    TRAIN_SVC --> S3
    PREPROC_SVC --> POSTGRES
    INGEST_SVC --> POSTGRES
    INGEST_SVC --> DATA_APIS
    
    PBI --> POWERBI_API
    
    SCHEDULER --> TRAIN_SVC
    SCHEDULER --> INGEST_SVC
    SCHEDULER --> ALERT_SVC
    
    MONITORING --> PRED_API
    MONITORING --> ALERT_SVC
    MONITORING --> TRAIN_SVC
    
    LOGGING --> PRED_API
    LOGGING --> ALERT_SVC
    LOGGING --> TRAIN_SVC
```

## Componentes del Sistema

### 1. Capa de Presentación (Frontend Layer)

#### 1.1 Aplicación Streamlit
- **Propósito**: Interfaz web interactiva para usuarios finales
- **Funcionalidades**:
  - Scoring individual de empresas
  - Análisis de carteras
  - Stress testing y análisis de sensibilidad
  - Visualización de insights del modelo
  - Gestión de configuraciones
- **Tecnologías**: Streamlit, Plotly, Pandas
- **Comunicación**: HTTP REST con Prediction API

#### 1.2 Dashboards Power BI
- **Propósito**: Reportes ejecutivos y análisis avanzados
- **Funcionalidades**:
  - KPIs ejecutivos
  - Análisis de tendencias
  - Reportes por sector/región
  - Alertas visuales
- **Tecnologías**: Power BI, Power Query
- **Comunicación**: Conexión directa a PostgreSQL y Power BI API

#### 1.3 Clientes API
- **Propósito**: Integración con sistemas externos
- **Funcionalidades**:
  - Predicciones programáticas
  - Integración con sistemas de riesgo existentes
  - Automatización de procesos
- **Tecnologías**: HTTP REST, JSON
- **Comunicación**: HTTP REST con API Gateway

### 2. Capa de API Gateway

#### 2.1 API Gateway / Load Balancer
- **Propósito**: Punto de entrada único para todas las solicitudes
- **Funcionalidades**:
  - Enrutamiento de solicitudes
  - Balanceo de carga
  - Rate limiting
  - Autenticación y autorización
  - Logging de solicitudes
- **Tecnologías**: Nginx, Kong, o AWS API Gateway
- **Patrones**: Gateway Pattern, Circuit Breaker

### 3. Capa de Aplicación (Application Layer)

#### 3.1 Prediction API
- **Propósito**: Servicio principal de predicciones
- **Funcionalidades**:
  - Predicciones individuales y en lote
  - Cálculo de Altman Z-Score
  - Blending de scores ML y Altman
  - Explicabilidad de predicciones
- **Tecnologías**: FastAPI, XGBoost, Scikit-learn
- **Patrones**: Repository Pattern, Dependency Injection

```python
# Ejemplo de endpoint
@app.post("/predict/single")
async def predict_single(request: PredictionRequest) -> PredictionResult:
    """Realiza predicción para una empresa individual"""
    pass

@app.post("/predict/batch")
async def predict_batch(requests: List[PredictionRequest]) -> List[PredictionResult]:
    """Realiza predicciones en lote"""
    pass
```

#### 3.2 Alerting Service
- **Propósito**: Gestión y envío de alertas automáticas
- **Funcionalidades**:
  - Evaluación de reglas de alerta
  - Envío multicanal (email, Slack, Teams)
  - Gestión de estado de alertas
  - Configuración de umbrales
- **Tecnologías**: FastAPI, Celery, SMTP, Slack SDK
- **Patrones**: Observer Pattern, Strategy Pattern

#### 3.3 Training Service
- **Propósito**: Entrenamiento y gestión de modelos ML
- **Funcionalidades**:
  - Entrenamiento de modelos XGBoost
  - Validación cruzada temporal
  - Optimización de hiperparámetros
  - Versionado de modelos
  - Evaluación de rendimiento
- **Tecnologías**: Scikit-learn, XGBoost, Optuna, MLflow
- **Patrones**: Factory Pattern, Template Method

#### 3.4 Preprocessing Service
- **Propósito**: Preprocesamiento y ingeniería de características
- **Funcionalidades**:
  - Limpieza de datos
  - Cálculo de ratios financieros
  - Ingeniería de características
  - Normalización y escalado
  - Manejo de valores faltantes
- **Tecnologías**: Pandas, NumPy, Scikit-learn
- **Patrones**: Pipeline Pattern, Chain of Responsibility

#### 3.5 Data Ingestion Service
- **Propósito**: Ingesta de datos desde fuentes externas
- **Funcionalidades**:
  - Extracción de APIs financieras
  - Validación de calidad de datos
  - Transformación y normalización
  - Carga a base de datos
- **Tecnologías**: Pandas, Requests, Apache Airflow
- **Patrones**: ETL Pattern, Adapter Pattern

### 4. Capa de Datos (Data Layer)

#### 4.1 PostgreSQL
- **Propósito**: Base de datos principal del sistema
- **Esquemas**:
  - `empresas`: Información básica de empresas
  - `datos_financieros`: Ratios y métricas financieras
  - `datos_macroeconomicos`: Variables macroeconómicas
  - `predicciones`: Resultados de predicciones
  - `modelos`: Metadatos de modelos ML
  - `alertas`: Sistema de alertas
- **Optimizaciones**:
  - Índices en columnas frecuentemente consultadas
  - Particionado por fecha en tablas grandes
  - Vistas materializadas para consultas complejas

#### 4.2 Redis Cache
- **Propósito**: Cache en memoria para mejorar rendimiento
- **Uso**:
  - Cache de predicciones recientes
  - Cache de datos financieros frecuentemente accedidos
  - Sesiones de usuario
  - Rate limiting
- **Configuración**: TTL configurables, eviction policies

#### 4.3 Object Storage (S3/MinIO)
- **Propósito**: Almacenamiento de artefactos y archivos grandes
- **Contenido**:
  - Modelos ML serializados
  - Datasets de entrenamiento
  - Backups de base de datos
  - Logs archivados
- **Organización**: Estructura jerárquica por tipo y fecha

### 5. Servicios Externos

#### 5.1 APIs de Datos Financieros
- **Proveedores**: Alpha Vantage, Quandl, FRED, Yahoo Finance
- **Datos**: Precios de acciones, datos macroeconómicos, ratios sectoriales
- **Integración**: HTTP REST, autenticación por API key

#### 5.2 Servicios de Notificación
- **Email**: SMTP para alertas por correo
- **Slack**: Webhook para notificaciones en canales
- **Teams**: Webhook para notificaciones corporativas

#### 5.3 Power BI API
- **Funcionalidades**: Refresh de datasets, push de datos
- **Autenticación**: Azure AD OAuth 2.0

## Flujos de Datos

### 1. Flujo de Predicción Individual

```mermaid
sequenceDiagram
    participant User as Usuario
    participant UI as Streamlit UI
    participant API as Prediction API
    participant Cache as Redis
    participant DB as PostgreSQL
    participant Model as ML Model
    
    User->>UI: Ingresa datos financieros
    UI->>API: POST /predict/single
    API->>Cache: Buscar predicción en cache
    alt Cache hit
        Cache-->>API: Retorna predicción
    else Cache miss
        API->>DB: Obtener datos históricos
        API->>Model: Ejecutar predicción
        Model-->>API: Resultado predicción
        API->>DB: Guardar predicción
        API->>Cache: Guardar en cache
    end
    API-->>UI: Retorna resultado
    UI-->>User: Muestra predicción
```

### 2. Flujo de Entrenamiento de Modelo

```mermaid
sequenceDiagram
    participant Scheduler as Scheduler
    participant Training as Training Service
    participant DB as PostgreSQL
    participant Storage as Object Storage
    participant Registry as Model Registry
    
    Scheduler->>Training: Trigger entrenamiento
    Training->>DB: Extraer datos de entrenamiento
    Training->>Training: Preprocesar datos
    Training->>Training: Entrenar modelo XGBoost
    Training->>Training: Validar modelo
    Training->>Storage: Guardar artefactos
    Training->>Registry: Registrar modelo
    Training->>DB: Actualizar metadatos
```

### 3. Flujo de Alertas

```mermaid
sequenceDiagram
    participant Scheduler as Scheduler
    participant Alerting as Alerting Service
    participant DB as PostgreSQL
    participant SMTP as Email Service
    participant Slack as Slack API
    
    Scheduler->>Alerting: Evaluar alertas
    Alerting->>DB: Consultar predicciones recientes
    Alerting->>Alerting: Aplicar reglas de alerta
    alt Alerta generada
        Alerting->>DB: Guardar alerta
        Alerting->>SMTP: Enviar email
        Alerting->>Slack: Enviar notificación
    end
```

## Tecnologías y Herramientas

### Backend
- **Framework**: FastAPI (Python 3.11+)
- **Base de Datos**: PostgreSQL 15+
- **Cache**: Redis 7+
- **ML Framework**: Scikit-learn, XGBoost
- **Serialización**: Joblib, Pickle
- **Validación**: Pydantic
- **ORM**: SQLAlchemy 2.0
- **Migraciones**: Alembic

### Frontend
- **UI Framework**: Streamlit
- **Visualización**: Plotly, Matplotlib, Seaborn
- **Reportes**: Power BI

### Infraestructura
- **Contenedores**: Docker, Docker Compose
- **Orquestación**: Kubernetes (opcional)
- **Proxy/Load Balancer**: Nginx
- **Monitoreo**: Prometheus, Grafana
- **Logging**: ELK Stack (Elasticsearch, Logstash, Kibana)
- **Task Queue**: Celery con Redis

### DevOps
- **CI/CD**: GitHub Actions, GitLab CI
- **Testing**: Pytest, Coverage.py
- **Linting**: Black, Flake8, MyPy
- **Documentation**: Sphinx, MkDocs

## Patrones de Diseño

### 1. Repository Pattern
Abstrae el acceso a datos y permite intercambiar implementaciones.

```python
class IPredictionRepository(ABC):
    @abstractmethod
    async def save_prediction(self, prediction: PredictionResult) -> int:
        pass
    
    @abstractmethod
    async def get_predictions_by_company(self, company_id: int) -> List[PredictionResult]:
        pass

class PostgreSQLPredictionRepository(IPredictionRepository):
    async def save_prediction(self, prediction: PredictionResult) -> int:
        # Implementación específica para PostgreSQL
        pass
```

### 2. Dependency Injection
Facilita testing y permite intercambiar dependencias.

```python
class PredictionService:
    def __init__(
        self,
        model_service: IModelService,
        prediction_repo: IPredictionRepository,
        cache_service: ICacheService
    ):
        self.model_service = model_service
        self.prediction_repo = prediction_repo
        self.cache_service = cache_service
```

### 3. Factory Pattern
Para crear instancias de modelos y servicios.

```python
class ModelFactory:
    @staticmethod
    def create_model(model_type: str, config: Dict[str, Any]):
        if model_type == "xgboost":
            return XGBoostModel(config)
        elif model_type == "random_forest":
            return RandomForestModel(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
```

### 4. Observer Pattern
Para el sistema de alertas y notificaciones.

```python
class AlertObserver(ABC):
    @abstractmethod
    async def notify(self, alert: AlertData):
        pass

class EmailAlertObserver(AlertObserver):
    async def notify(self, alert: AlertData):
        # Enviar email
        pass

class SlackAlertObserver(AlertObserver):
    async def notify(self, alert: AlertData):
        # Enviar a Slack
        pass
```

### 5. Strategy Pattern
Para diferentes algoritmos de predicción y preprocesamiento.

```python
class PreprocessingStrategy(ABC):
    @abstractmethod
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

class StandardPreprocessingStrategy(PreprocessingStrategy):
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        # Implementación estándar
        pass

class AdvancedPreprocessingStrategy(PreprocessingStrategy):
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        # Implementación avanzada
        pass
```

## Seguridad

### 1. Autenticación y Autorización
- **JWT Tokens**: Para autenticación de APIs
- **Role-Based Access Control (RBAC)**: Diferentes niveles de acceso
- **API Keys**: Para clientes externos

### 2. Protección de Datos
- **Encriptación en Tránsito**: HTTPS/TLS 1.3
- **Encriptación en Reposo**: Encriptación de base de datos
- **Hashing de Contraseñas**: bcrypt
- **Sanitización de Inputs**: Validación con Pydantic

### 3. Seguridad de Red
- **Firewall**: Restricción de puertos y protocolos
- **VPN**: Acceso seguro para administradores
- **Rate Limiting**: Prevención de ataques DDoS
- **CORS**: Configuración restrictiva

### 4. Auditoría y Compliance
- **Audit Logs**: Registro de todas las operaciones críticas
- **Data Retention**: Políticas de retención de datos
- **GDPR Compliance**: Manejo de datos personales
- **SOX Compliance**: Para datos financieros

## Escalabilidad y Rendimiento

### 1. Escalabilidad Horizontal
- **Microservicios**: Escalado independiente de componentes
- **Load Balancing**: Distribución de carga entre instancias
- **Database Sharding**: Particionado de datos por empresa/región
- **Caching**: Redis para reducir carga en base de datos

### 2. Optimización de Base de Datos
- **Índices**: En columnas frecuentemente consultadas
- **Particionado**: Por fecha en tablas de predicciones
- **Connection Pooling**: Reutilización de conexiones
- **Query Optimization**: Análisis y optimización de consultas

### 3. Optimización de ML
- **Model Caching**: Cache de modelos en memoria
- **Batch Predictions**: Procesamiento en lotes
- **Feature Caching**: Cache de características calculadas
- **Model Quantization**: Reducción de tamaño de modelos

### 4. CDN y Caching
- **Static Assets**: CDN para archivos estáticos
- **API Response Caching**: Cache de respuestas frecuentes
- **Browser Caching**: Headers de cache apropiados

## Monitoreo y Observabilidad

### 1. Métricas de Aplicación
- **Latencia**: Tiempo de respuesta de APIs
- **Throughput**: Requests por segundo
- **Error Rate**: Porcentaje de errores
- **Availability**: Uptime del sistema

### 2. Métricas de ML
- **Model Drift**: Cambios en distribución de datos
- **Prediction Accuracy**: Precisión de predicciones
- **Feature Importance**: Cambios en importancia de características
- **Model Performance**: ROC-AUC, Precision, Recall

### 3. Métricas de Infraestructura
- **CPU/Memory Usage**: Utilización de recursos
- **Disk I/O**: Operaciones de disco
- **Network**: Tráfico de red
- **Database Performance**: Query performance, connections

### 4. Alertas y Notificaciones
- **Threshold Alerts**: Alertas basadas en umbrales
- **Anomaly Detection**: Detección de anomalías
- **Health Checks**: Verificación de salud de servicios
- **SLA Monitoring**: Monitoreo de SLAs

### 5. Logging
- **Structured Logging**: Logs en formato JSON
- **Centralized Logging**: ELK Stack
- **Log Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Correlation IDs**: Trazabilidad de requests

## Despliegue

### 1. Containerización
```dockerfile
# Ejemplo Dockerfile para Prediction API
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. Docker Compose
```yaml
version: '3.8'

services:
  prediction-api:
    build: ./services/prediction_api
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/db
    depends_on:
      - postgres
      - redis

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: bankruptcy_prediction
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
```

### 3. Kubernetes (Opcional)
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prediction-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: prediction-api
  template:
    metadata:
      labels:
        app: prediction-api
    spec:
      containers:
      - name: prediction-api
        image: bankruptcy-prediction/api:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
```

### 4. CI/CD Pipeline
```yaml
# GitHub Actions ejemplo
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    - name: Run tests
      run: pytest --cov=./ --cov-report=xml
    - name: Upload coverage
      uses: codecov/codecov-action@v3

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - name: Deploy to production
      run: |
        # Deployment steps
        echo "Deploying to production"
```

## Consideraciones de Producción

### 1. Alta Disponibilidad
- **Multi-AZ Deployment**: Despliegue en múltiples zonas
- **Database Replication**: Réplicas de lectura
- **Failover Mechanisms**: Mecanismos de failover automático
- **Health Checks**: Verificaciones de salud continuas

### 2. Backup y Recovery
- **Database Backups**: Backups automáticos diarios
- **Point-in-Time Recovery**: Recuperación a punto específico
- **Model Versioning**: Versionado de modelos ML
- **Disaster Recovery**: Plan de recuperación ante desastres

### 3. Performance Tuning
- **Database Tuning**: Optimización de PostgreSQL
- **Connection Pooling**: PgBouncer para conexiones
- **Caching Strategy**: Estrategia de cache multinivel
- **Load Testing**: Pruebas de carga regulares

### 4. Security Hardening
- **Security Scanning**: Escaneo de vulnerabilidades
- **Dependency Updates**: Actualizaciones de seguridad
- **Access Controls**: Controles de acceso estrictos
- **Penetration Testing**: Pruebas de penetración

Esta arquitectura proporciona una base sólida para un sistema de predicción de quiebras empresariales escalable, mantenible y confiable, siguiendo las mejores prácticas de la industria y patrones de diseño probados.

