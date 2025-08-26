-- =====================================================
-- SISTEMA DE PREDICCIÓN DE QUIEBRAS EMPRESARIALES
-- Esquema de Base de Datos PostgreSQL
-- Autor: Manus AI
-- Fecha: 2025-01-26
-- =====================================================

-- Configuración inicial
SET timezone = 'UTC';

-- Extensiones necesarias
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- =====================================================
-- TABLA: empresas
-- Información básica de las empresas en el sistema
-- =====================================================
CREATE TABLE empresas (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    nombre VARCHAR(255) NOT NULL,
    codigo_empresa VARCHAR(50) UNIQUE, -- Código interno de la empresa
    sector VARCHAR(100) NOT NULL,
    subsector VARCHAR(100),
    pais VARCHAR(3) NOT NULL, -- Código ISO 3166-1 alpha-3
    region VARCHAR(100),
    tamaño_empresa VARCHAR(20) CHECK (tamaño_empresa IN ('MICRO', 'PEQUEÑA', 'MEDIANA', 'GRANDE')),
    fecha_fundacion DATE,
    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    fecha_actualizacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    activa BOOLEAN DEFAULT TRUE,
    notas TEXT,
    
    -- Índices para búsquedas frecuentes
    CONSTRAINT empresas_nombre_check CHECK (LENGTH(nombre) >= 2),
    CONSTRAINT empresas_pais_check CHECK (LENGTH(pais) = 3)
);

-- Índices para optimización de consultas
CREATE INDEX idx_empresas_sector ON empresas(sector);
CREATE INDEX idx_empresas_pais ON empresas(pais);
CREATE INDEX idx_empresas_activa ON empresas(activa);
CREATE INDEX idx_empresas_nombre_gin ON empresas USING gin(nombre gin_trgm_ops);

-- =====================================================
-- TABLA: datos_financieros
-- Ratios financieros y métricas de las empresas
-- =====================================================
CREATE TABLE datos_financieros (
    id SERIAL PRIMARY KEY,
    empresa_id INTEGER NOT NULL REFERENCES empresas(id) ON DELETE CASCADE,
    fecha_corte DATE NOT NULL,
    periodo VARCHAR(10) NOT NULL, -- 'Q1', 'Q2', 'Q3', 'Q4', 'ANUAL'
    año INTEGER NOT NULL,
    
    -- Ratios principales para Altman Z-Score
    wc_ta DECIMAL(10,6), -- Working Capital / Total Assets
    re_ta DECIMAL(10,6), -- Retained Earnings / Total Assets
    ebit_ta DECIMAL(10,6), -- EBIT / Total Assets
    me_tl DECIMAL(10,6), -- Market Equity / Total Liabilities
    s_ta DECIMAL(10,6), -- Sales / Total Assets
    
    -- Ratios adicionales
    ocf_ta DECIMAL(10,6), -- Operating Cash Flow / Total Assets
    debt_assets DECIMAL(10,6), -- Total Liabilities / Total Assets
    current_ratio DECIMAL(10,6), -- Current Assets / Current Liabilities
    quick_ratio DECIMAL(10,6), -- (Current Assets - Inventory) / Current Liabilities
    debt_equity DECIMAL(10,6), -- Total Debt / Total Equity
    roa DECIMAL(10,6), -- Return on Assets
    roe DECIMAL(10,6), -- Return on Equity
    gross_margin DECIMAL(10,6), -- Gross Profit / Revenue
    operating_margin DECIMAL(10,6), -- Operating Income / Revenue
    net_margin DECIMAL(10,6), -- Net Income / Revenue
    asset_turnover DECIMAL(10,6), -- Revenue / Total Assets
    inventory_turnover DECIMAL(10,6), -- COGS / Average Inventory
    receivables_turnover DECIMAL(10,6), -- Revenue / Average Accounts Receivable
    
    -- Métricas de volatilidad y tendencia
    revenue_growth_yoy DECIMAL(10,6), -- Crecimiento de ingresos año a año
    ebitda_growth_yoy DECIMAL(10,6), -- Crecimiento de EBITDA año a año
    volatilidad_ingresos DECIMAL(10,6), -- Volatilidad de ingresos (rolling 12m)
    
    -- Metadatos
    fuente_datos VARCHAR(100), -- Fuente de los datos financieros
    fecha_carga TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    validado BOOLEAN DEFAULT FALSE,
    
    -- Restricciones
    CONSTRAINT datos_financieros_unique_empresa_fecha UNIQUE (empresa_id, fecha_corte, periodo),
    CONSTRAINT datos_financieros_año_check CHECK (año >= 1900 AND año <= EXTRACT(YEAR FROM CURRENT_DATE) + 1),
    CONSTRAINT datos_financieros_periodo_check CHECK (periodo IN ('Q1', 'Q2', 'Q3', 'Q4', 'ANUAL'))
);

-- Índices para optimización
CREATE INDEX idx_datos_financieros_empresa_fecha ON datos_financieros(empresa_id, fecha_corte DESC);
CREATE INDEX idx_datos_financieros_año ON datos_financieros(año);
CREATE INDEX idx_datos_financieros_validado ON datos_financieros(validado);

-- =====================================================
-- TABLA: datos_macroeconomicos
-- Variables macroeconómicas por país y fecha
-- =====================================================
CREATE TABLE datos_macroeconomicos (
    id SERIAL PRIMARY KEY,
    fecha DATE NOT NULL,
    pais VARCHAR(3) NOT NULL, -- Código ISO 3166-1 alpha-3
    
    -- Variables macroeconómicas principales
    gdp_yoy DECIMAL(8,4), -- Crecimiento del PIB interanual (%)
    gdp_qoq DECIMAL(8,4), -- Crecimiento del PIB trimestral (%)
    unemp_rate DECIMAL(6,3), -- Tasa de desempleo (%)
    inflation_rate DECIMAL(6,3), -- Tasa de inflación (%)
    pmi DECIMAL(6,2), -- Índice de Gerentes de Compras
    pmi_manufacturing DECIMAL(6,2), -- PMI Manufacturero
    pmi_services DECIMAL(6,2), -- PMI Servicios
    
    -- Tipos de interés y spreads
    y10y DECIMAL(8,4), -- Rendimiento del bono a 10 años (%)
    y3m DECIMAL(8,4), -- Rendimiento del bono a 3 meses (%)
    y2y DECIMAL(8,4), -- Rendimiento del bono a 2 años (%)
    credit_spread DECIMAL(8,4), -- Spread de crédito corporativo (%)
    term_spread DECIMAL(8,4), -- Spread de plazo (10Y - 3M)
    
    -- Indicadores de mercado
    stock_index_return DECIMAL(8,4), -- Retorno del índice bursátil principal
    volatility_index DECIMAL(8,4), -- Índice de volatilidad (VIX equivalente)
    exchange_rate_usd DECIMAL(12,6), -- Tipo de cambio vs USD
    
    -- Indicadores de confianza
    consumer_confidence DECIMAL(8,2), -- Índice de confianza del consumidor
    business_confidence DECIMAL(8,2), -- Índice de confianza empresarial
    
    -- Metadatos
    fuente_datos VARCHAR(100),
    fecha_carga TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Restricciones
    CONSTRAINT datos_macro_unique_fecha_pais UNIQUE (fecha, pais),
    CONSTRAINT datos_macro_pais_check CHECK (LENGTH(pais) = 3)
);

-- Índices para optimización
CREATE INDEX idx_datos_macro_fecha_pais ON datos_macroeconomicos(fecha DESC, pais);
CREATE INDEX idx_datos_macro_pais ON datos_macroeconomicos(pais);

-- =====================================================
-- TABLA: modelos
-- Registro de modelos de Machine Learning
-- =====================================================
CREATE TABLE modelos (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    nombre VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    algoritmo VARCHAR(100) NOT NULL, -- 'XGBoost', 'RandomForest', etc.
    descripcion TEXT,
    
    -- Metadatos del entrenamiento
    fecha_entrenamiento TIMESTAMP NOT NULL,
    fecha_validacion TIMESTAMP,
    dataset_entrenamiento VARCHAR(255), -- Referencia al dataset usado
    tamaño_dataset_train INTEGER,
    tamaño_dataset_test INTEGER,
    
    -- Hiperparámetros (JSON)
    hiperparametros JSONB,
    
    -- Métricas de rendimiento
    roc_auc_train DECIMAL(6,4),
    roc_auc_test DECIMAL(6,4),
    pr_auc_train DECIMAL(6,4),
    pr_auc_test DECIMAL(6,4),
    brier_score_train DECIMAL(8,6),
    brier_score_test DECIMAL(8,6),
    ks_statistic_train DECIMAL(6,4),
    ks_statistic_test DECIMAL(6,4),
    
    -- Umbrales optimizados
    umbral_medio DECIMAL(6,4) DEFAULT 0.15,
    umbral_alto DECIMAL(6,4) DEFAULT 0.30,
    
    -- Importancia de características (JSON)
    feature_importances JSONB,
    
    -- Rutas y artefactos
    ruta_modelo VARCHAR(500), -- Ruta al archivo del modelo serializado
    ruta_preprocessor VARCHAR(500), -- Ruta al preprocessor
    ruta_scaler VARCHAR(500), -- Ruta al scaler
    
    -- Estado del modelo
    activo BOOLEAN DEFAULT FALSE,
    en_produccion BOOLEAN DEFAULT FALSE,
    fecha_despliegue TIMESTAMP,
    
    -- Metadatos
    creado_por VARCHAR(100),
    notas TEXT,
    fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Restricciones
    CONSTRAINT modelos_nombre_version_unique UNIQUE (nombre, version),
    CONSTRAINT modelos_roc_auc_check CHECK (roc_auc_test >= 0 AND roc_auc_test <= 1)
);

-- Índices
CREATE INDEX idx_modelos_activo ON modelos(activo);
CREATE INDEX idx_modelos_fecha_entrenamiento ON modelos(fecha_entrenamiento DESC);

-- =====================================================
-- TABLA: predicciones
-- Resultados de las predicciones de quiebra
-- =====================================================
CREATE TABLE predicciones (
    id SERIAL PRIMARY KEY,
    empresa_id INTEGER NOT NULL REFERENCES empresas(id) ON DELETE CASCADE,
    modelo_id INTEGER NOT NULL REFERENCES modelos(id),
    fecha_prediccion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    fecha_datos DATE NOT NULL, -- Fecha de los datos financieros utilizados
    
    -- Resultados de la predicción
    probabilidad_quiebra DECIMAL(8,6) NOT NULL CHECK (probabilidad_quiebra >= 0 AND probabilidad_quiebra <= 1),
    altman_z_score DECIMAL(8,4),
    banda_riesgo_ml VARCHAR(20) CHECK (banda_riesgo_ml IN ('LOW', 'MEDIUM', 'HIGH')),
    banda_riesgo_altman VARCHAR(20) CHECK (banda_riesgo_altman IN ('SAFE', 'GREY', 'DISTRESS')),
    blended_score DECIMAL(8,6) CHECK (blended_score >= 0 AND blended_score <= 1),
    
    -- Umbrales utilizados
    umbral_medio DECIMAL(6,4) NOT NULL,
    umbral_alto DECIMAL(6,4) NOT NULL,
    peso_ml_blended DECIMAL(4,3) DEFAULT 0.7, -- Peso del ML en el blended score
    
    -- Contribuciones de características (top 5)
    top_features_positivas JSONB, -- Features que aumentan el riesgo
    top_features_negativas JSONB, -- Features que disminuyen el riesgo
    
    -- Metadatos
    version_api VARCHAR(20),
    tiempo_procesamiento_ms INTEGER,
    
    -- Restricciones
    CONSTRAINT predicciones_umbrales_check CHECK (umbral_medio < umbral_alto)
);

-- Índices para optimización
CREATE INDEX idx_predicciones_empresa_fecha ON predicciones(empresa_id, fecha_prediccion DESC);
CREATE INDEX idx_predicciones_fecha_prediccion ON predicciones(fecha_prediccion DESC);
CREATE INDEX idx_predicciones_banda_riesgo ON predicciones(banda_riesgo_ml);
CREATE INDEX idx_predicciones_probabilidad ON predicciones(probabilidad_quiebra DESC);

-- =====================================================
-- TABLA: alertas
-- Sistema de alertas y notificaciones
-- =====================================================
CREATE TABLE alertas (
    id SERIAL PRIMARY KEY,
    empresa_id INTEGER NOT NULL REFERENCES empresas(id) ON DELETE CASCADE,
    prediccion_id INTEGER REFERENCES predicciones(id),
    tipo_alerta VARCHAR(50) NOT NULL, -- 'RIESGO_ALTO', 'DETERIORO_RAPIDO', 'UMBRAL_SUPERADO'
    severidad VARCHAR(20) NOT NULL CHECK (severidad IN ('INFO', 'WARNING', 'CRITICAL')),
    
    -- Contenido de la alerta
    titulo VARCHAR(255) NOT NULL,
    mensaje TEXT NOT NULL,
    probabilidad_actual DECIMAL(8,6),
    probabilidad_anterior DECIMAL(8,6),
    cambio_probabilidad DECIMAL(8,6), -- Cambio en la probabilidad
    
    -- Estado de la alerta
    estado VARCHAR(20) DEFAULT 'PENDIENTE' CHECK (estado IN ('PENDIENTE', 'ENVIADA', 'LEIDA', 'ARCHIVADA')),
    fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    fecha_envio TIMESTAMP,
    fecha_lectura TIMESTAMP,
    
    -- Destinatarios y canales
    destinatarios JSONB, -- Lista de emails/usuarios
    canales_envio JSONB, -- ['EMAIL', 'SLACK', 'TEAMS']
    
    -- Metadatos
    regla_alerta VARCHAR(100), -- Regla que disparó la alerta
    notas TEXT
);

-- Índices
CREATE INDEX idx_alertas_empresa_fecha ON alertas(empresa_id, fecha_creacion DESC);
CREATE INDEX idx_alertas_estado ON alertas(estado);
CREATE INDEX idx_alertas_severidad ON alertas(severidad);
CREATE INDEX idx_alertas_tipo ON alertas(tipo_alerta);

-- =====================================================
-- TABLA: configuracion_sistema
-- Configuraciones globales del sistema
-- =====================================================
CREATE TABLE configuracion_sistema (
    id SERIAL PRIMARY KEY,
    clave VARCHAR(100) UNIQUE NOT NULL,
    valor TEXT NOT NULL,
    tipo_dato VARCHAR(20) DEFAULT 'STRING' CHECK (tipo_dato IN ('STRING', 'INTEGER', 'FLOAT', 'BOOLEAN', 'JSON')),
    descripcion TEXT,
    categoria VARCHAR(50), -- 'UMBRALES', 'ALERTAS', 'MODELO', 'SISTEMA'
    fecha_actualizacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    actualizado_por VARCHAR(100)
);

-- Insertar configuraciones por defecto
INSERT INTO configuracion_sistema (clave, valor, tipo_dato, descripcion, categoria) VALUES
('umbral_riesgo_medio', '0.15', 'FLOAT', 'Umbral para clasificar riesgo medio', 'UMBRALES'),
('umbral_riesgo_alto', '0.30', 'FLOAT', 'Umbral para clasificar riesgo alto', 'UMBRALES'),
('peso_ml_blended', '0.7', 'FLOAT', 'Peso del modelo ML en el score blended', 'MODELO'),
('frecuencia_alertas_horas', '24', 'INTEGER', 'Frecuencia de evaluación de alertas en horas', 'ALERTAS'),
('max_empresas_alerta', '50', 'INTEGER', 'Máximo número de empresas en una alerta', 'ALERTAS'),
('retention_predicciones_dias', '1095', 'INTEGER', 'Días de retención de predicciones (3 años)', 'SISTEMA'),
('retention_logs_dias', '90', 'INTEGER', 'Días de retención de logs', 'SISTEMA');

-- =====================================================
-- TABLA: auditoria
-- Log de auditoría para cambios importantes
-- =====================================================
CREATE TABLE auditoria (
    id SERIAL PRIMARY KEY,
    tabla_afectada VARCHAR(50) NOT NULL,
    registro_id INTEGER,
    operacion VARCHAR(10) NOT NULL CHECK (operacion IN ('INSERT', 'UPDATE', 'DELETE')),
    usuario VARCHAR(100),
    fecha_operacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    datos_anteriores JSONB,
    datos_nuevos JSONB,
    ip_address INET,
    user_agent TEXT
);

-- Índices
CREATE INDEX idx_auditoria_tabla_fecha ON auditoria(tabla_afectada, fecha_operacion DESC);
CREATE INDEX idx_auditoria_usuario ON auditoria(usuario);

-- =====================================================
-- VISTAS ÚTILES
-- =====================================================

-- Vista para el último estado financiero de cada empresa
CREATE VIEW v_ultimo_estado_financiero AS
SELECT DISTINCT ON (df.empresa_id)
    df.empresa_id,
    e.nombre,
    e.sector,
    e.pais,
    df.fecha_corte,
    df.wc_ta,
    df.re_ta,
    df.ebit_ta,
    df.me_tl,
    df.s_ta,
    df.ocf_ta,
    df.debt_assets,
    df.current_ratio,
    df.roa,
    df.roe
FROM datos_financieros df
JOIN empresas e ON df.empresa_id = e.id
WHERE e.activa = TRUE
ORDER BY df.empresa_id, df.fecha_corte DESC;

-- Vista para las últimas predicciones de cada empresa
CREATE VIEW v_ultimas_predicciones AS
SELECT DISTINCT ON (p.empresa_id)
    p.empresa_id,
    e.nombre,
    e.sector,
    e.pais,
    p.fecha_prediccion,
    p.probabilidad_quiebra,
    p.banda_riesgo_ml,
    p.altman_z_score,
    p.banda_riesgo_altman,
    p.blended_score,
    m.nombre as modelo_nombre,
    m.version as modelo_version
FROM predicciones p
JOIN empresas e ON p.empresa_id = e.id
JOIN modelos m ON p.modelo_id = m.id
WHERE e.activa = TRUE
ORDER BY p.empresa_id, p.fecha_prediccion DESC;

-- Vista para empresas en riesgo alto
CREATE VIEW v_empresas_riesgo_alto AS
SELECT 
    up.*,
    uf.current_ratio,
    uf.debt_assets,
    uf.roa
FROM v_ultimas_predicciones up
JOIN v_ultimo_estado_financiero uf ON up.empresa_id = uf.empresa_id
WHERE up.banda_riesgo_ml = 'HIGH'
ORDER BY up.probabilidad_quiebra DESC;

-- =====================================================
-- FUNCIONES Y TRIGGERS
-- =====================================================

-- Función para actualizar timestamp de actualización
CREATE OR REPLACE FUNCTION actualizar_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.fecha_actualizacion = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger para empresas
CREATE TRIGGER trigger_empresas_timestamp
    BEFORE UPDATE ON empresas
    FOR EACH ROW
    EXECUTE FUNCTION actualizar_timestamp();

-- Función para calcular Altman Z-Score
CREATE OR REPLACE FUNCTION calcular_altman_z(
    p_wc_ta DECIMAL,
    p_re_ta DECIMAL,
    p_ebit_ta DECIMAL,
    p_me_tl DECIMAL,
    p_s_ta DECIMAL
) RETURNS DECIMAL AS $$
BEGIN
    RETURN (1.2 * COALESCE(p_wc_ta, 0)) + 
           (1.4 * COALESCE(p_re_ta, 0)) + 
           (3.3 * COALESCE(p_ebit_ta, 0)) + 
           (0.6 * COALESCE(p_me_tl, 0)) + 
           (1.0 * COALESCE(p_s_ta, 0));
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- COMENTARIOS EN TABLAS Y COLUMNAS
-- =====================================================

COMMENT ON TABLE empresas IS 'Información básica de las empresas en el sistema de predicción de quiebras';
COMMENT ON COLUMN empresas.uuid IS 'Identificador único universal para integraciones externas';
COMMENT ON COLUMN empresas.codigo_empresa IS 'Código interno de la empresa (ej. ticker, código contable)';

COMMENT ON TABLE datos_financieros IS 'Ratios financieros y métricas de rendimiento de las empresas';
COMMENT ON COLUMN datos_financieros.wc_ta IS 'Working Capital / Total Assets - Liquidez';
COMMENT ON COLUMN datos_financieros.me_tl IS 'Market Equity / Total Liabilities - Solvencia de mercado';

COMMENT ON TABLE predicciones IS 'Resultados de las predicciones de probabilidad de quiebra';
COMMENT ON COLUMN predicciones.blended_score IS 'Puntuación combinada entre modelo ML y Altman Z-Score';

COMMENT ON TABLE alertas IS 'Sistema de alertas automáticas para empresas en riesgo';

-- =====================================================
-- PERMISOS Y ROLES (Opcional - para entorno de producción)
-- =====================================================

-- Crear roles para diferentes tipos de usuarios
-- CREATE ROLE bankruptcy_readonly;
-- CREATE ROLE bankruptcy_analyst;
-- CREATE ROLE bankruptcy_admin;

-- Otorgar permisos de solo lectura
-- GRANT SELECT ON ALL TABLES IN SCHEMA public TO bankruptcy_readonly;

-- Otorgar permisos de análisis (lectura + inserción de predicciones)
-- GRANT SELECT, INSERT ON predicciones TO bankruptcy_analyst;
-- GRANT SELECT ON ALL TABLES IN SCHEMA public TO bankruptcy_analyst;

-- Otorgar permisos completos a administradores
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO bankruptcy_admin;

-- =====================================================
-- FIN DEL ESQUEMA
-- =====================================================

