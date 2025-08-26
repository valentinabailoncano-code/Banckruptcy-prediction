# Plan de Mejoras y Arquitectura Propuesta para el Sistema de Predicción de Quiebras

El código actual (`app.py`) es funcional para una demostración con Streamlit, pero para un sistema de nivel empresarial, se requieren mejoras significativas en modularidad, escalabilidad, robustez y profesionalismo. A continuación, se detalla un plan de mejoras y la arquitectura propuesta.

## 1. Análisis del Código Existente y Áreas de Mejora

### 1.1. Estructura y Modularidad

*   **Problema:** El archivo `app.py` es monolítico, conteniendo lógica de interfaz de usuario (Streamlit), carga de datos, preprocesamiento, lógica de modelo, cálculo de métricas y generación de datos sintéticos. Esto dificulta la lectura, el mantenimiento, las pruebas unitarias y la escalabilidad.
*   **Mejora:** Modularizar el código en componentes lógicos separados (ej. `data_processing.py`, `model.py`, `metrics.py`, `database.py`, `app_ui.py`).

### 1.2. Manejo de Errores y Robustez

*   **Problema:** El manejo de errores es básico (`try-except` genéricos que simplemente devuelven `None` o `st.info`). Esto puede ocultar problemas subyacentes y dificultar la depuración en un entorno de producción.
*   **Mejora:** Implementar un manejo de errores más granular y específico, con logging adecuado para registrar eventos y excepciones. Considerar validación de datos de entrada.

### 1.3. Rendimiento y Escalabilidad

*   **Problema:** Las funciones de carga de datos y cálculo de métricas están cacheada con `@st.cache_resource` o `@st.cache_data`, lo cual es bueno para Streamlit, pero la forma en que se manejan los datos (ej. `pd.read_parquet` dentro de una función de caché) podría optimizarse para grandes volúmenes de datos.
*   **Mejora:** Optimizar las operaciones de datos, posiblemente utilizando herramientas como Dask o Spark para conjuntos de datos muy grandes. Asegurar que las operaciones de base de datos sean eficientes.

### 1.4. Configuración y Entorno

*   **Problema:** Rutas de archivos (`ART`, `BASE`) y configuraciones (umbrales `thr_defaults`) están hardcodeadas o cargadas de archivos JSON simples. Esto no es ideal para diferentes entornos (desarrollo, staging, producción).
*   **Mejora:** Utilizar variables de entorno o un sistema de gestión de configuración (ej. `python-dotenv`, `ConfigParser`, `Pydantic Settings`) para manejar configuraciones de manera más flexible y segura.

### 1.5. Datos y Modelos

*   **Problema:** La generación de cartera sintética es útil para demostración, pero un sistema real necesita una ingesta de datos robusta y un pipeline de entrenamiento de modelos bien definido.
*   **Mejora:** Separar claramente el pipeline de datos (ingesta, limpieza, preprocesamiento, ingeniería de características) del pipeline de modelado (entrenamiento, evaluación, serialización). Utilizar un enfoque MLOps para la gestión de modelos (versionado, registro).

### 1.6. Interfaz de Usuario (Streamlit)

*   **Problema:** La lógica de UI está mezclada con la lógica de negocio. El uso de `st.session_state` es correcto, pero la estructura de la UI podría ser más modular si se usaran componentes o funciones dedicadas para cada sección.
*   **Mejora:** Refactorizar la UI para que sea más declarativa y menos acoplada a la lógica de negocio. Considerar el uso de callbacks y estados de sesión de Streamlit de manera más estructurada.

### 1.7. Comentarios y Documentación

*   **Problema:** Los comentarios existentes son útiles, pero un código 


profesional requiere comentarios más detallados, docstrings para funciones y clases, y una documentación externa completa.
*   **Mejora:** Añadir docstrings a todas las funciones y clases, y crear una documentación técnica externa que cubra la arquitectura, el despliegue, el uso y el mantenimiento del sistema.

## 2. Arquitectura Propuesta

Se propone una arquitectura de microservicios o al menos una arquitectura modular con componentes bien definidos, lo que permitirá un desarrollo, despliegue y escalado más eficientes.

```mermaid
graph TD
    subgraph Data Layer
        A[Fuentes de Datos Externas] --> B(Ingesta de Datos); 
        B --> C{Base de Datos SQL}; 
        C --> D[Data Lake/Warehouse];
    end

    subgraph Backend Services
        E[Servicio de Preprocesamiento] --> F[Servicio de Entrenamiento de Modelos];
        F --> G[Registro de Modelos (MLflow/DVC)];
        G --> H[Servicio de Predicción (API REST)];
        C --> E;
        E --> H;
    end

    subgraph Frontend & Reporting
        I[Aplicación Streamlit] --> H;
        J[Power BI] --> C;
        J --> H;
        K[Sistema de Alertas] --> H;
        K --> C;
    end

    subgraph MLOps & Monitoring
        L[Orquestador (Airflow/Prefect)] --> E;
        L --> F;
        L --> H;
        M[Monitoreo y Logging] --> E;
        M --> F;
        M --> H;
        M --> I;
    end

    A -- Datos Financieros/Sectoriales/Macroeconómicos --> B;
    B -- Datos Crudos --> C;
    C -- Datos Procesados --> E;
    E -- Features --> F;
    F -- Modelo Entrenado --> G;
    G -- Modelo Versionado --> H;
    H -- Predicciones --> I;
    H -- Predicciones --> J;
    H -- Alertas --> K;
    C -- Datos para Reportes --> J;
    C -- Datos para Alertas --> K;
```

### Componentes Clave de la Arquitectura:

*   **Capa de Datos:**
    *   **Fuentes de Datos Externas:** APIs financieras, bases de datos de mercado, proveedores de datos macroeconómicos.
    *   **Ingesta de Datos:** Scripts o servicios dedicados a la extracción, transformación y carga (ETL) de datos desde las fuentes externas a la base de datos SQL.
    *   **Base de Datos SQL (PostgreSQL/MySQL):** Almacenará los datos financieros, sectoriales y macroeconómicos de las empresas, así como los resultados de las predicciones y los metadatos del modelo. Será la fuente principal de datos para los servicios de backend y los dashboards.
    *   **Data Lake/Warehouse (Opcional, para grandes volúmenes):** Para almacenar datos históricos y crudos, permitiendo análisis más profundos y reprocesamiento.

*   **Servicios de Backend:**
    *   **Servicio de Preprocesamiento:** Encargado de la limpieza, normalización, imputación de datos y la ingeniería de características (cálculo de ratios, Altman Z-Score, etc.). Podría ser un servicio REST o una función serverless.
    *   **Servicio de Entrenamiento de Modelos:** Responsable de entrenar el modelo de Machine Learning (XGBoost calibrado), realizar la validación cruzada y la optimización de hiperparámetros. Se ejecutará bajo demanda o programado.
    *   **Registro de Modelos (MLflow/DVC):** Para versionar, registrar y gestionar los modelos entrenados, sus métricas y sus artefactos. Esto asegura la reproducibilidad y la trazabilidad.
    *   **Servicio de Predicción (API REST):** Una API ligera y de alto rendimiento que expone un endpoint para recibir datos de entrada y devolver predicciones de probabilidad de quiebra. Será consumida por la aplicación Streamlit y Power BI.

*   **Frontend y Reporting:**
    *   **Aplicación Streamlit:** La interfaz de usuario interactiva para el scoring individual, la gestión de carteras, el stress testing y la visualización de insights. Consumirá el Servicio de Predicción.
    *   **Power BI:** Se conectará directamente a la Base de Datos SQL para generar dashboards analíticos avanzados, reportes personalizados y visualizaciones de tendencias históricas y agregadas. También podría consumir el Servicio de Predicción para análisis en tiempo real.
    *   **Sistema de Alertas:** Un servicio que monitorea las predicciones de quiebra y envía notificaciones (email, Slack, etc.) cuando una empresa supera un umbral de riesgo predefinido. Se conectará a la Base de Datos SQL y/o al Servicio de Predicción.

*   **MLOps y Monitoreo:**
    *   **Orquestador (Airflow/Prefect):** Para automatizar y programar los pipelines de ingesta de datos, preprocesamiento, entrenamiento de modelos y despliegue. Asegura que los modelos estén siempre actualizados.
    *   **Monitoreo y Logging:** Herramientas para supervisar el rendimiento del modelo (deriva de datos, deriva de concepto), la salud de los servicios y el uso de recursos. Se implementará un sistema de logging centralizado.

## 3. Requisitos para la Base de Datos SQL

La base de datos SQL será el corazón del sistema, almacenando toda la información relevante para la predicción de quiebras. Se proponen las siguientes tablas:

*   **`empresas`:**
    *   `id` (PK, INT): Identificador único de la empresa.
    *   `nombre` (VARCHAR): Nombre de la empresa.
    *   `sector` (VARCHAR): Sector económico al que pertenece la empresa.
    *   `pais` (VARCHAR): País de origen de la empresa.
    *   `fecha_registro` (DATE): Fecha de registro de la empresa en el sistema.
    *   `activa` (BOOLEAN): Indica si la empresa está activa o no.

*   **`datos_financieros`:**
    *   `id` (PK, INT): Identificador único del registro financiero.
    *   `empresa_id` (FK, INT): Referencia a `empresas.id`.
    *   `fecha_corte` (DATE): Fecha de cierre del período financiero.
    *   `wc_ta` (FLOAT): Working Capital / Total Assets.
    *   `re_ta` (FLOAT): Retained Earnings / Total Assets.
    *   `ebit_ta` (FLOAT): EBIT / Total Assets.
    *   `me_tl` (FLOAT): Market Equity / Total Liabilities.
    *   `s_ta` (FLOAT): Sales / Total Assets.
    *   `ocf_ta` (FLOAT): Operating Cash Flow / Total Assets.
    *   `debt_assets` (FLOAT): Total Liabilities / Total Assets.
    *   ... (otras ratios financieras relevantes)

*   **`datos_macroeconomicos`:**
    *   `id` (PK, INT): Identificador único del registro macroeconómico.
    *   `fecha` (DATE): Fecha del dato macroeconómico.
    *   `pais` (VARCHAR): País al que aplica el dato macroeconómico.
    *   `gdp_yoy` (FLOAT): Crecimiento del PIB interanual.
    *   `unemp_rate` (FLOAT): Tasa de desempleo.
    *   `pmi` (FLOAT): Índice de Gerentes de Compras (PMI).
    *   `y10y` (FLOAT): Rendimiento del bono a 10 años.
    *   `y3m` (FLOAT): Rendimiento del bono a 3 meses.
    *   `credit_spread` (FLOAT): Spread de crédito.
    *   ... (otras variables macroeconómicas)

*   **`predicciones`:**
    *   `id` (PK, INT): Identificador único de la predicción.
    *   `empresa_id` (FK, INT): Referencia a `empresas.id`.
    *   `fecha_prediccion` (DATETIME): Fecha y hora en que se realizó la predicción.
    *   `modelo_id` (FK, INT): Referencia al modelo utilizado (si se implementa un registro de modelos).
    *   `probabilidad_quiebra` (FLOAT): Probabilidad de quiebra a 12 meses.
    *   `altman_z_score` (FLOAT): Altman Z-Score calculado.
    *   `banda_riesgo_ml` (VARCHAR): Banda de riesgo según el modelo ML (Low, Medium, High).
    *   `banda_riesgo_altman` (VARCHAR): Banda de riesgo según Altman (Safe, Grey, Distress).
    *   `blended_score` (FLOAT): Puntuación combinada ML y Altman.
    *   `umbral_medio` (FLOAT): Umbral medio utilizado para la predicción.
    *   `umbral_alto` (FLOAT): Umbral alto utilizado para la predicción.

*   **`modelos` (Opcional, si se usa MLflow/DVC para metadatos):**
    *   `id` (PK, INT): Identificador único del modelo.
    *   `nombre` (VARCHAR): Nombre del modelo (ej. XGBoost Calibrado v1.0).
    *   `version` (VARCHAR): Versión del modelo.
    *   `fecha_entrenamiento` (DATETIME): Fecha de entrenamiento.
    *   `ruta_artefacto` (VARCHAR): Ruta al archivo del modelo serializado.
    *   `metricas_test` (JSONB): Métricas de rendimiento en el conjunto de test.
    *   `feature_importances` (JSONB): Importancia de las características.

## 4. Planificación de la Integración con Power BI y Sistema de Alertas

### 4.1. Integración con Power BI

*   **Conexión Directa a SQL:** Power BI se conectará directamente a la base de datos `predicciones` y `empresas` para obtener los datos necesarios para los dashboards.
*   **Dashboards Clave:**
    *   **Resumen Ejecutivo:** KPIs principales (número de empresas en riesgo alto/medio, probabilidad promedio, etc.).
    *   **Análisis de Cartera:** Distribución de riesgo por sector, país, tamaño de empresa. Empresas con mayor riesgo.
    *   **Tendencias:** Evolución del riesgo a lo largo del tiempo para empresas individuales o agregados.
    *   **Análisis de Sensibilidad:** Visualización de cómo los cambios en las variables macro o financieras afectan el riesgo.
*   **Actualización de Datos:** Configurar la actualización programada de los datos en Power BI para mantener los dashboards al día.

### 4.2. Sistema de Alertas

*   **Servicio de Monitoreo:** Un servicio Python independiente que se ejecuta periódicamente (ej. diariamente o semanalmente).
*   **Lógica de Alerta:** Consulta la tabla `predicciones` para identificar empresas cuya `probabilidad_quiebra` o `blended_score` supere un umbral predefinido (configurable).
*   **Canales de Notificación:** Envío de alertas a través de:
    *   **Correo Electrónico:** Utilizando un servicio SMTP.
    *   **Slack/Teams:** Integración con APIs de mensajería.
    *   **Base de Datos:** Registrar las alertas en una tabla `alertas` para auditoría y seguimiento.
*   **Contenido de la Alerta:** Incluir información clave como el nombre de la empresa, la probabilidad de quiebra, la banda de riesgo, y un enlace al dashboard de Power BI o a la aplicación Streamlit para más detalles.

## 5. Estructura de la Documentación Final

La documentación será exhaustiva y estará dirigida a diferentes audiencias (desarrolladores, analistas de negocio, usuarios finales).

*   **Documentación Técnica (para desarrolladores):**
    *   **Arquitectura del Sistema:** Diagramas, descripción de componentes y flujos de datos.
    *   **Diseño de Base de Datos:** Esquema de la base de datos, relaciones, índices.
    *   **API de Predicción:** Endpoints, parámetros, respuestas, ejemplos de uso.
    *   **Pipeline de Datos y ML:** Descripción de los pasos de ETL, preprocesamiento, entrenamiento y despliegue de modelos.
    *   **Configuración y Despliegue:** Instrucciones detalladas para configurar el entorno y desplegar cada componente.
    *   **Manejo de Errores y Logging:** Estrategias de logging, monitoreo y resolución de problemas.
    *   **Pruebas:** Estrategias de pruebas unitarias, de integración y de sistema.

*   **Documentación de Usuario (para analistas/usuarios finales):**
    *   **Guía de Usuario de Streamlit:** Cómo interactuar con la aplicación, interpretar los resultados, realizar stress testing.
    *   **Guía de Dashboards Power BI:** Cómo navegar por los dashboards, filtrar información, entender las visualizaciones.
    *   **Interpretación de Resultados:** Explicación de la probabilidad de quiebra, Altman Z-Score, bandas de riesgo y la importancia de las variables.
    *   **Preguntas Frecuentes (FAQ):** Respuestas a preguntas comunes sobre el modelo y el sistema.

*   **Model Card (para gobernanza y transparencia):**
    *   **Propósito y Contexto:** Para qué se usa el modelo, quiénes son los usuarios previstos.
    *   **Datos de Entrenamiento:** Descripción del dataset, fuentes, preprocesamiento.
    *   **Detalles del Modelo:** Algoritmo, hiperparámetros, métricas de rendimiento (ROC-AUC, PR-AUC, Brier, KS).
    *   **Evaluación:** Resultados en conjuntos de validación y test, análisis de sesgos.
    *   **Limitaciones y Riesgos:** Escenarios donde el modelo podría no ser preciso, riesgos de uso indebido.
    *   **Uso Previsto y Fuera de Uso:** Cómo debe y no debe usarse el modelo.

Este plan proporciona una hoja de ruta para transformar el prototipo actual en un sistema de predicción de quiebras robusto, escalable y de nivel empresarial.

