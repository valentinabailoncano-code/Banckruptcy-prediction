"""
Este módulo maneja las conexiones a la base de datos, configuración
y gestión de sesiones con SQLAlchemy.
"""

import logging
import os
from typing import Optional, Dict, Any
from contextlib import contextmanager
from sqlalchemy import create_engine, event, pool
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool
import time

from services.models import Base, create_all_tables, create_additional_indexes

# Configurar logging
logger = logging.getLogger(__name__)

class DatabaseConfig:
    """Configuración de base de datos"""
    
    def __init__(self):
        # Configuración desde variables de entorno
        self.database_url = os.getenv('DATABASE_URL', 'sqlite:///bankruptcy_prediction.db')
        self.echo = os.getenv('DB_ECHO', 'false').lower() == 'true'
        self.pool_size = int(os.getenv('DB_POOL_SIZE', '10'))
        self.max_overflow = int(os.getenv('DB_MAX_OVERFLOW', '20'))
        self.pool_timeout = int(os.getenv('DB_POOL_TIMEOUT', '30'))
        self.pool_recycle = int(os.getenv('DB_POOL_RECYCLE', '3600'))
        
        # Configuración específica por tipo de base de datos
        self.db_type = self._detect_db_type()
        
    def _detect_db_type(self) -> str:
        """Detecta el tipo de base de datos desde la URL"""
        if self.database_url.startswith('postgresql'):
            return 'postgresql'
        elif self.database_url.startswith('mysql'):
            return 'mysql'
        elif self.database_url.startswith('sqlite'):
            return 'sqlite'
        else:
            return 'unknown'
    
    def get_engine_kwargs(self) -> Dict[str, Any]:
        """Obtiene argumentos específicos para el engine según el tipo de DB"""
        
        base_kwargs = {
            'echo': self.echo,
            'future': True
        }
        
        if self.db_type == 'sqlite':
            # Configuración para SQLite
            base_kwargs.update({
                'connect_args': {
                    'check_same_thread': False,
                    'timeout': 20
                }
            })
        else:
            # Configuración para bases de datos con pool de conexiones
            base_kwargs.update({
                'poolclass': QueuePool,
                'pool_size': self.pool_size,
                'max_overflow': self.max_overflow,
                'pool_timeout': self.pool_timeout,
                'pool_recycle': self.pool_recycle,
                'pool_pre_ping': True
            })
            
            if self.db_type == 'postgresql':
                base_kwargs['connect_args'] = {
                    'connect_timeout': 10,
                    'application_name': 'bankruptcy_prediction_system'
                }
            elif self.db_type == 'mysql':
                base_kwargs['connect_args'] = {
                    'connect_timeout': 10,
                    'charset': 'utf8mb4'
                }
        
        return base_kwargs

class DatabaseManager:
    """Gestor principal de base de datos"""
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self.engine: Optional[Engine] = None
        self.SessionLocal: Optional[sessionmaker] = None
        self._initialized = False
        
    def initialize(self):
        """Inicializa la conexión a la base de datos"""
        if self._initialized:
            return
        
        try:
            # Crear engine
            engine_kwargs = self.config.get_engine_kwargs()
            self.engine = create_engine(self.config.database_url, **engine_kwargs)
            
            # Configurar eventos
            self._setup_events()
            
            # Crear sessionmaker
            self.SessionLocal = sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False
            )
            
            # Verificar conexión
            self._test_connection()
            
            self._initialized = True
            logger.info(f"Base de datos inicializada: {self.config.db_type}")
            
        except Exception as e:
            logger.error(f"Error inicializando base de datos: {str(e)}")
            raise
    
    def _setup_events(self):
        """Configura eventos de SQLAlchemy"""
        
        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            """Configura pragmas para SQLite"""
            if self.config.db_type == 'sqlite':
                cursor = dbapi_connection.cursor()
                # Habilitar foreign keys
                cursor.execute("PRAGMA foreign_keys=ON")
                # Configurar WAL mode para mejor concurrencia
                cursor.execute("PRAGMA journal_mode=WAL")
                # Configurar timeout
                cursor.execute("PRAGMA busy_timeout=30000")
                cursor.close()
        
        @event.listens_for(self.engine, "before_cursor_execute")
        def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            """Log de queries lentas"""
            context._query_start_time = time.time()
        
        @event.listens_for(self.engine, "after_cursor_execute")
        def receive_after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            """Log de queries lentas"""
            total = time.time() - context._query_start_time
            if total > 1.0:  # Log queries que toman más de 1 segundo
                logger.warning(f"Query lenta ({total:.2f}s): {statement[:100]}...")
    
    def _test_connection(self):
        """Prueba la conexión a la base de datos"""
        try:
            with self.engine.connect() as conn:
                conn.execute("SELECT 1")
            logger.info("Conexión a base de datos verificada")
        except Exception as e:
            logger.error(f"Error probando conexión: {str(e)}")
            raise
    
    def create_tables(self, drop_existing: bool = False):
        """Crea las tablas en la base de datos"""
        if not self._initialized:
            self.initialize()
        
        try:
            if drop_existing:
                logger.warning("Eliminando tablas existentes")
                Base.metadata.drop_all(self.engine)
            
            logger.info("Creando tablas")
            create_all_tables(self.engine)
            
            # Crear índices adicionales
            create_additional_indexes(self.engine)
            
            logger.info("Tablas creadas exitosamente")
            
        except Exception as e:
            logger.error(f"Error creando tablas: {str(e)}")
            raise
    
    @contextmanager
    def get_session(self):
        """Context manager para obtener una sesión de base de datos"""
        if not self._initialized:
            self.initialize()
        
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error en sesión de base de datos: {str(e)}")
            raise
        finally:
            session.close()
    
    def get_session_direct(self) -> Session:
        """Obtiene una sesión directa (debe cerrarse manualmente)"""
        if not self._initialized:
            self.initialize()
        
        return self.SessionLocal()
    
    def health_check(self) -> Dict[str, Any]:
        """Verifica el estado de la base de datos"""
        if not self._initialized:
            return {
                'status': 'error',
                'message': 'Base de datos no inicializada'
            }
        
        try:
            with self.get_session() as session:
                # Test básico de conectividad
                session.execute("SELECT 1")
                
                # Obtener estadísticas básicas
                from .models import Empresa, DatoFinanciero, Prediccion
                
                stats = {
                    'empresas': session.query(Empresa).count(),
                    'datos_financieros': session.query(DatoFinanciero).count(),
                    'predicciones': session.query(Prediccion).count()
                }
                
                return {
                    'status': 'healthy',
                    'database_type': self.config.db_type,
                    'statistics': stats,
                    'pool_info': self._get_pool_info()
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _get_pool_info(self) -> Dict[str, Any]:
        """Obtiene información del pool de conexiones"""
        if self.config.db_type == 'sqlite':
            return {'type': 'sqlite', 'pool': 'not_applicable'}
        
        try:
            pool = self.engine.pool
            return {
                'size': pool.size(),
                'checked_in': pool.checkedin(),
                'checked_out': pool.checkedout(),
                'overflow': pool.overflow(),
                'invalid': pool.invalid()
            }
        except Exception:
            return {'error': 'Unable to get pool info'}
    
    def close(self):
        """Cierra las conexiones a la base de datos"""
        if self.engine:
            self.engine.dispose()
            logger.info("Conexiones de base de datos cerradas")

# Instancia global del gestor de base de datos
db_manager = DatabaseManager()

# Funciones de conveniencia
def get_db_session():
    """Función de conveniencia para obtener una sesión"""
    return db_manager.get_session()

def get_db_session_direct():
    """Función de conveniencia para obtener una sesión directa"""
    return db_manager.get_session_direct()

def initialize_database(config: Optional[DatabaseConfig] = None, create_tables: bool = True):
    """Inicializa la base de datos con configuración opcional"""
    global db_manager
    
    if config:
        db_manager = DatabaseManager(config)
    
    db_manager.initialize()
    
    if create_tables:
        db_manager.create_tables()

def close_database():
    """Cierra las conexiones de base de datos"""
    db_manager.close()

# Dependency para FastAPI
def get_db():
    """Dependency para FastAPI que proporciona una sesión de base de datos"""
    session = db_manager.get_session_direct()
    try:
        yield session
    finally:
        session.close()

if __name__ == "__main__":
    # Ejemplo de uso
    import logging
    
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    # Inicializar base de datos
    config = DatabaseConfig()
    print(f"Configuración de DB: {config.database_url}")
    print(f"Tipo de DB: {config.db_type}")
    
    # Inicializar
    initialize_database(config, create_tables=True)
    
    # Health check
    health = db_manager.health_check()
    print(f"Estado de la DB: {health}")
    
    # Ejemplo de uso de sesión
    from .models import Empresa, TamañoEmpresa
    
    with get_db_session() as session:
        # Crear empresa de ejemplo
        empresa = Empresa(
            rut='12345678-9',
            razon_social='Empresa de Prueba S.A.',
            sector='Technology',
            tamaño=TamañoEmpresa.MEDIANA,
            pais='Chile'
        )
        
        session.add(empresa)
        session.commit()
        
        print(f"Empresa creada: {empresa}")
        
        # Consultar empresas
        empresas = session.query(Empresa).all()
        print(f"Total empresas: {len(empresas)}")
    
    # Cerrar conexiones
    close_database()

