"""
Este módulo define los modelos de datos usando SQLAlchemy ORM para
gestionar empresas, datos financieros, predicciones y metadatos del sistema.
"""

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, Text, 
    ForeignKey, Index, UniqueConstraint, CheckConstraint,
    JSON, DECIMAL, Date, Enum as SQLEnum
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref
from sqlalchemy.sql import func
from datetime import datetime
from enum import Enum
import uuid

Base = declarative_base()

# Enums para campos categóricos
class EmpresaStatus(Enum):
    ACTIVA = "activa"
    INACTIVA = "inactiva"
    SUSPENDIDA = "suspendida"
    LIQUIDACION = "liquidacion"

class TamañoEmpresa(Enum):
    MICRO = "micro"
    PEQUEÑA = "pequeña"
    MEDIANA = "mediana"
    GRANDE = "grande"

class BandaRiesgo(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class TipoDato(Enum):
    TRIMESTRAL = "trimestral"
    ANUAL = "anual"
    MENSUAL = "mensual"

class EstadoPrediccion(Enum):
    PENDIENTE = "pendiente"
    PROCESADA = "procesada"
    ERROR = "error"

class Empresa(Base):
    """Modelo para información básica de empresas"""
    
    __tablename__ = 'empresas'
    
    # Identificación
    id = Column(Integer, primary_key=True, autoincrement=True)
    uuid = Column(String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    rut = Column(String(20), unique=True, nullable=False)
    razon_social = Column(String(255), nullable=False)
    nombre_fantasia = Column(String(255))
    
    # Clasificación
    sector = Column(String(100), nullable=False)
    subsector = Column(String(100))
    actividad_economica = Column(String(255))
    codigo_ciiu = Column(String(10))
    tamaño = Column(SQLEnum(TamañoEmpresa), nullable=False)
    
    # Ubicación
    pais = Column(String(50), nullable=False, default='Chile')
    region = Column(String(100))
    ciudad = Column(String(100))
    direccion = Column(Text)
    
    # Información corporativa
    fecha_constitucion = Column(Date)
    fecha_inicio_actividades = Column(Date)
    capital_inicial = Column(DECIMAL(15, 2))
    numero_empleados = Column(Integer)
    
    # Estado y metadatos
    status = Column(SQLEnum(EmpresaStatus), nullable=False, default=EmpresaStatus.ACTIVA)
    es_publica = Column(Boolean, default=False)
    ticker_bolsa = Column(String(10))
    
    # Timestamps
    fecha_registro = Column(DateTime, nullable=False, default=func.now())
    fecha_actualizacion = Column(DateTime, nullable=False, default=func.now(), onupdate=func.now())
    
    # Relaciones
    datos_financieros = relationship("DatoFinanciero", back_populates="empresa", cascade="all, delete-orphan")
    predicciones = relationship("Prediccion", back_populates="empresa", cascade="all, delete-orphan")
    alertas = relationship("Alerta", back_populates="empresa", cascade="all, delete-orphan")
    
    # Índices
    __table_args__ = (
        Index('idx_empresa_rut', 'rut'),
        Index('idx_empresa_sector', 'sector'),
        Index('idx_empresa_tamaño', 'tamaño'),
        Index('idx_empresa_status', 'status'),
        Index('idx_empresa_fecha_registro', 'fecha_registro'),
    )
    
    def __repr__(self):
        return f"<Empresa(id={self.id}, rut='{self.rut}', razon_social='{self.razon_social}')>"

class DatoFinanciero(Base):
    """Modelo para datos financieros de empresas"""
    
    __tablename__ = 'datos_financieros'
    
    # Identificación
    id = Column(Integer, primary_key=True, autoincrement=True)
    empresa_id = Column(Integer, ForeignKey('empresas.id'), nullable=False)
    
    # Período
    fecha_corte = Column(Date, nullable=False)
    periodo = Column(String(10), nullable=False)  # 2023-Q1, 2023-12, etc.
    tipo_dato = Column(SQLEnum(TipoDato), nullable=False)
    año = Column(Integer, nullable=False)
    trimestre = Column(Integer)  # 1-4 para datos trimestrales
    
    # Estados Financieros - Balance General
    activos_corrientes = Column(DECIMAL(15, 2))
    efectivo_equivalentes = Column(DECIMAL(15, 2))
    cuentas_por_cobrar = Column(DECIMAL(15, 2))
    inventarios = Column(DECIMAL(15, 2))
    otros_activos_corrientes = Column(DECIMAL(15, 2))
    
    activos_no_corrientes = Column(DECIMAL(15, 2))
    propiedades_planta_equipo = Column(DECIMAL(15, 2))
    activos_intangibles = Column(DECIMAL(15, 2))
    inversiones_largo_plazo = Column(DECIMAL(15, 2))
    otros_activos_no_corrientes = Column(DECIMAL(15, 2))
    
    total_activos = Column(DECIMAL(15, 2), nullable=False)
    
    pasivos_corrientes = Column(DECIMAL(15, 2))
    cuentas_por_pagar = Column(DECIMAL(15, 2))
    deuda_corto_plazo = Column(DECIMAL(15, 2))
    otros_pasivos_corrientes = Column(DECIMAL(15, 2))
    
    pasivos_no_corrientes = Column(DECIMAL(15, 2))
    deuda_largo_plazo = Column(DECIMAL(15, 2))
    otros_pasivos_no_corrientes = Column(DECIMAL(15, 2))
    
    total_pasivos = Column(DECIMAL(15, 2), nullable=False)
    patrimonio = Column(DECIMAL(15, 2), nullable=False)
    
    # Estados Financieros - Estado de Resultados
    ingresos_operacionales = Column(DECIMAL(15, 2))
    costo_ventas = Column(DECIMAL(15, 2))
    utilidad_bruta = Column(DECIMAL(15, 2))
    gastos_operacionales = Column(DECIMAL(15, 2))
    gastos_administracion = Column(DECIMAL(15, 2))
    gastos_ventas = Column(DECIMAL(15, 2))
    
    utilidad_operacional = Column(DECIMAL(15, 2))
    ingresos_no_operacionales = Column(DECIMAL(15, 2))
    gastos_financieros = Column(DECIMAL(15, 2))
    utilidad_antes_impuestos = Column(DECIMAL(15, 2))
    impuestos = Column(DECIMAL(15, 2))
    utilidad_neta = Column(DECIMAL(15, 2))
    
    # Estados Financieros - Flujo de Efectivo
    flujo_efectivo_operacional = Column(DECIMAL(15, 2))
    flujo_efectivo_inversion = Column(DECIMAL(15, 2))
    flujo_efectivo_financiamiento = Column(DECIMAL(15, 2))
    flujo_efectivo_neto = Column(DECIMAL(15, 2))
    
    # Ratios Financieros Calculados
    liquidez_corriente = Column(Float)
    liquidez_acida = Column(Float)
    razon_efectivo = Column(Float)
    
    rotacion_inventarios = Column(Float)
    rotacion_cuentas_cobrar = Column(Float)
    rotacion_cuentas_pagar = Column(Float)
    rotacion_activos = Column(Float)
    
    margen_bruto = Column(Float)
    margen_operacional = Column(Float)
    margen_neto = Column(Float)
    roa = Column(Float)  # Return on Assets
    roe = Column(Float)  # Return on Equity
    roic = Column(Float)  # Return on Invested Capital
    
    endeudamiento_total = Column(Float)
    endeudamiento_financiero = Column(Float)
    cobertura_intereses = Column(Float)
    
    # Ratios Altman Z-Score
    wc_ta = Column(Float)  # Working Capital / Total Assets
    re_ta = Column(Float)  # Retained Earnings / Total Assets
    ebit_ta = Column(Float)  # EBIT / Total Assets
    me_tl = Column(Float)  # Market Equity / Total Liabilities
    s_ta = Column(Float)  # Sales / Total Assets
    altman_z_score = Column(Float)
    
    # Información de mercado (para empresas públicas)
    precio_accion = Column(DECIMAL(10, 4))
    acciones_circulacion = Column(Integer)
    capitalizacion_mercado = Column(DECIMAL(15, 2))
    
    # Metadatos
    fuente_datos = Column(String(100))  # SVS, CMF, manual, etc.
    fecha_carga = Column(DateTime, nullable=False, default=func.now())
    usuario_carga = Column(String(100))
    validado = Column(Boolean, default=False)
    observaciones = Column(Text)
    
    # Relaciones
    empresa = relationship("Empresa", back_populates="datos_financieros")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('empresa_id', 'fecha_corte', 'tipo_dato', name='uq_empresa_periodo'),
        CheckConstraint('total_activos > 0', name='chk_activos_positivos'),
        CheckConstraint('año >= 1900 AND año <= 2100', name='chk_año_valido'),
        CheckConstraint('trimestre IS NULL OR (trimestre >= 1 AND trimestre <= 4)', name='chk_trimestre_valido'),
        Index('idx_dato_empresa_fecha', 'empresa_id', 'fecha_corte'),
        Index('idx_dato_periodo', 'año', 'trimestre'),
        Index('idx_dato_altman', 'altman_z_score'),
        Index('idx_dato_validado', 'validado'),
    )
    
    def __repr__(self):
        return f"<DatoFinanciero(id={self.id}, empresa_id={self.empresa_id}, fecha_corte='{self.fecha_corte}')>"

class Prediccion(Base):
    """Modelo para predicciones de quiebra"""
    
    __tablename__ = 'predicciones'
    
    # Identificación
    id = Column(Integer, primary_key=True, autoincrement=True)
    uuid = Column(String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    empresa_id = Column(Integer, ForeignKey('empresas.id'), nullable=False)
    dato_financiero_id = Column(Integer, ForeignKey('datos_financieros.id'))
    
    # Información del modelo
    modelo_id = Column(String(100), nullable=False)
    modelo_version = Column(String(50), nullable=False)
    fecha_prediccion = Column(DateTime, nullable=False, default=func.now())
    
    # Resultados de predicción
    probabilidad_ml = Column(Float, nullable=False)
    altman_z_score = Column(Float)
    blended_score = Column(Float, nullable=False)
    
    # Clasificaciones
    banda_riesgo_ml = Column(SQLEnum(BandaRiesgo), nullable=False)
    banda_riesgo_altman = Column(String(20))  # SAFE, GREY, DISTRESS
    banda_riesgo_blended = Column(SQLEnum(BandaRiesgo), nullable=False)
    
    # Explicabilidad
    feature_contributions = Column(JSON)  # Contribuciones de características
    top_positive_features = Column(JSON)  # Top características positivas
    top_negative_features = Column(JSON)  # Top características negativas
    
    # Métricas de calidad
    confianza_prediccion = Column(Float)
    tiempo_procesamiento_ms = Column(Float)
    
    # Estado y metadatos
    estado = Column(SQLEnum(EstadoPrediccion), nullable=False, default=EstadoPrediccion.PENDIENTE)
    observaciones = Column(Text)
    usuario_solicita = Column(String(100))
    
    # Relaciones
    empresa = relationship("Empresa", back_populates="predicciones")
    dato_financiero = relationship("DatoFinanciero")
    
    # Índices
    __table_args__ = (
        Index('idx_prediccion_empresa', 'empresa_id'),
        Index('idx_prediccion_fecha', 'fecha_prediccion'),
        Index('idx_prediccion_banda', 'banda_riesgo_blended'),
        Index('idx_prediccion_modelo', 'modelo_id', 'modelo_version'),
        Index('idx_prediccion_estado', 'estado'),
    )
    
    def __repr__(self):
        return f"<Prediccion(id={self.id}, empresa_id={self.empresa_id}, blended_score={self.blended_score})>"

class Alerta(Base):
    """Modelo para alertas y notificaciones"""
    
    __tablename__ = 'alertas'
    
    # Identificación
    id = Column(Integer, primary_key=True, autoincrement=True)
    empresa_id = Column(Integer, ForeignKey('empresas.id'), nullable=False)
    prediccion_id = Column(Integer, ForeignKey('predicciones.id'))
    
    # Tipo y severidad
    tipo_alerta = Column(String(50), nullable=False)  # riesgo_alto, drift_detectado, etc.
    severidad = Column(String(20), nullable=False)  # LOW, MEDIUM, HIGH, CRITICAL
    titulo = Column(String(255), nullable=False)
    mensaje = Column(Text, nullable=False)
    
    # Estado
    activa = Column(Boolean, default=True)
    fecha_creacion = Column(DateTime, nullable=False, default=func.now())
    fecha_resolucion = Column(DateTime)
    resuelto_por = Column(String(100))
    
    # Metadatos
    datos_adicionales = Column(JSON)  # Información adicional específica del tipo de alerta
    
    # Relaciones
    empresa = relationship("Empresa", back_populates="alertas")
    prediccion = relationship("Prediccion")
    
    # Índices
    __table_args__ = (
        Index('idx_alerta_empresa', 'empresa_id'),
        Index('idx_alerta_tipo', 'tipo_alerta'),
        Index('idx_alerta_severidad', 'severidad'),
        Index('idx_alerta_activa', 'activa'),
        Index('idx_alerta_fecha', 'fecha_creacion'),
    )
    
    def __repr__(self):
        return f"<Alerta(id={self.id}, empresa_id={self.empresa_id}, tipo='{self.tipo_alerta}')>"

class DatoMacroeconomico(Base):
    """Modelo para datos macroeconómicos"""
    
    __tablename__ = 'datos_macroeconomicos'
    
    # Identificación
    id = Column(Integer, primary_key=True, autoincrement=True)
    pais = Column(String(50), nullable=False)
    fecha = Column(Date, nullable=False)
    
    # Indicadores económicos
    pib_yoy = Column(Float)  # PIB año a año
    inflacion_yoy = Column(Float)  # Inflación año a año
    tasa_desempleo = Column(Float)
    tasa_interes_referencia = Column(Float)
    
    # Indicadores financieros
    tipo_cambio_usd = Column(Float)
    indice_bolsa = Column(Float)
    yield_10y = Column(Float)  # Rendimiento bono 10 años
    yield_3m = Column(Float)  # Rendimiento bono 3 meses
    
    # Indicadores de confianza
    pmi_manufacturero = Column(Float)
    confianza_consumidor = Column(Float)
    confianza_empresarial = Column(Float)
    
    # Metadatos
    fuente = Column(String(100))
    fecha_actualizacion = Column(DateTime, nullable=False, default=func.now())
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('pais', 'fecha', name='uq_pais_fecha_macro'),
        Index('idx_macro_pais_fecha', 'pais', 'fecha'),
        Index('idx_macro_fecha', 'fecha'),
    )
    
    def __repr__(self):
        return f"<DatoMacroeconomico(id={self.id}, pais='{self.pais}', fecha='{self.fecha}')>"

class LogETL(Base):
    """Modelo para logging de procesos ETL"""
    
    __tablename__ = 'logs_etl'
    
    # Identificación
    id = Column(Integer, primary_key=True, autoincrement=True)
    proceso = Column(String(100), nullable=False)
    subproceso = Column(String(100))
    
    # Ejecución
    fecha_inicio = Column(DateTime, nullable=False)
    fecha_fin = Column(DateTime)
    estado = Column(String(20), nullable=False)  # INICIADO, COMPLETADO, ERROR
    
    # Resultados
    registros_procesados = Column(Integer, default=0)
    registros_exitosos = Column(Integer, default=0)
    registros_error = Column(Integer, default=0)
    
    # Detalles
    mensaje = Column(Text)
    error_detalle = Column(Text)
    archivo_origen = Column(String(255))
    parametros = Column(JSON)
    
    # Metadatos
    servidor = Column(String(100))
    usuario = Column(String(100))
    
    # Índices
    __table_args__ = (
        Index('idx_log_proceso', 'proceso'),
        Index('idx_log_fecha', 'fecha_inicio'),
        Index('idx_log_estado', 'estado'),
    )
    
    def __repr__(self):
        return f"<LogETL(id={self.id}, proceso='{self.proceso}', estado='{self.estado}')>"

class Usuario(Base):
    """Modelo para usuarios del sistema"""
    
    __tablename__ = 'usuarios'
    
    # Identificación
    id = Column(Integer, primary_key=True, autoincrement=True)
    uuid = Column(String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    
    # Autenticación
    password_hash = Column(String(255), nullable=False)
    salt = Column(String(50), nullable=False)
    
    # Información personal
    nombre = Column(String(100), nullable=False)
    apellido = Column(String(100), nullable=False)
    telefono = Column(String(20))
    
    # Permisos y roles
    rol = Column(String(50), nullable=False, default='usuario')  # admin, analista, usuario
    permisos = Column(JSON)  # Permisos específicos
    
    # Estado
    activo = Column(Boolean, default=True)
    email_verificado = Column(Boolean, default=False)
    ultimo_login = Column(DateTime)
    intentos_login_fallidos = Column(Integer, default=0)
    
    # Timestamps
    fecha_registro = Column(DateTime, nullable=False, default=func.now())
    fecha_actualizacion = Column(DateTime, nullable=False, default=func.now(), onupdate=func.now())
    
    # Índices
    __table_args__ = (
        Index('idx_usuario_username', 'username'),
        Index('idx_usuario_email', 'email'),
        Index('idx_usuario_rol', 'rol'),
        Index('idx_usuario_activo', 'activo'),
    )
    
    def __repr__(self):
        return f"<Usuario(id={self.id}, username='{self.username}', rol='{self.rol}')>"

class ConfiguracionSistema(Base):
    """Modelo para configuración del sistema"""
    
    __tablename__ = 'configuracion_sistema'
    
    # Identificación
    id = Column(Integer, primary_key=True, autoincrement=True)
    clave = Column(String(100), unique=True, nullable=False)
    valor = Column(Text, nullable=False)
    
    # Metadatos
    descripcion = Column(Text)
    tipo_dato = Column(String(20), nullable=False)  # string, integer, float, boolean, json
    categoria = Column(String(50), nullable=False)  # ml, database, api, etc.
    
    # Control
    modificable = Column(Boolean, default=True)
    fecha_creacion = Column(DateTime, nullable=False, default=func.now())
    fecha_actualizacion = Column(DateTime, nullable=False, default=func.now(), onupdate=func.now())
    usuario_actualizacion = Column(String(100))
    
    # Índices
    __table_args__ = (
        Index('idx_config_categoria', 'categoria'),
        Index('idx_config_modificable', 'modificable'),
    )
    
    def __repr__(self):
        return f"<ConfiguracionSistema(clave='{self.clave}', categoria='{self.categoria}')>"

# Funciones de utilidad para crear tablas
def create_all_tables(engine):
    """Crea todas las tablas en la base de datos"""
    Base.metadata.create_all(engine)

def drop_all_tables(engine):
    """Elimina todas las tablas de la base de datos"""
    Base.metadata.drop_all(engine)

# Configuración de índices adicionales para optimización
def create_additional_indexes(engine):
    """Crea índices adicionales para optimización de consultas"""
    
    with engine.connect() as conn:
        # Índices compuestos para consultas frecuentes
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_empresa_sector_tamaño 
            ON empresas (sector, tamaño)
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_dato_empresa_año_trimestre 
            ON datos_financieros (empresa_id, año, trimestre)
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_prediccion_empresa_fecha 
            ON predicciones (empresa_id, fecha_prediccion DESC)
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_alerta_empresa_activa 
            ON alertas (empresa_id, activa, fecha_creacion DESC)
        """)
        
        conn.commit()

if __name__ == "__main__":
    # Ejemplo de uso
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    # Crear engine para SQLite (para testing)
    engine = create_engine('sqlite:///bankruptcy_prediction.db', echo=True)
    
    # Crear todas las tablas
    create_all_tables(engine)
    
    # Crear sesión
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Ejemplo de inserción
    empresa_ejemplo = Empresa(
        rut='12345678-9',
        razon_social='Empresa Ejemplo S.A.',
        sector='Technology',
        tamaño=TamañoEmpresa.MEDIANA,
        pais='Chile'
    )
    
    session.add(empresa_ejemplo)
    session.commit()
    
    print(f"Empresa creada: {empresa_ejemplo}")
    
    session.close()

