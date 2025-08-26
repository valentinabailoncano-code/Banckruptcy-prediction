"""
Este módulo configura la aplicación Flask principal con todas las rutas
y servicios para el sistema de predicción de quiebras.
"""

import os
import sys
import logging
from datetime import datetime

# Configurar path para importar módulos del sistema
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from flask import Flask, send_from_directory, jsonify, request
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from werkzeug.exceptions import HTTPException

# Importar configuración y base de datos
from config.config import Config
from database.connection import initialize_database, close_database, db_manager

# Importar blueprints de rutas
from src.routes.empresas import empresas_bp
from src.routes.datos_financieros import datos_financieros_bp
from src.routes.predicciones import predicciones_bp
from src.routes.alertas import alertas_bp
from src.routes.etl import etl_bp
from src.routes.auth import auth_bp
from src.routes.dashboard import dashboard_bp

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_app(config_class=Config):
    """Factory para crear la aplicación Flask"""
    
    app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), 'static'))
    
    # Configuración
    app.config.from_object(config_class)
    
    # CORS para permitir requests desde frontend
    CORS(app, origins="*", supports_credentials=True)
    
    # JWT para autenticación
    jwt = JWTManager(app)
    
    # Inicializar base de datos
    try:
        initialize_database(create_tables=True)
        logger.info("Base de datos inicializada correctamente")
    except Exception as e:
        logger.error(f"Error inicializando base de datos: {str(e)}")
        raise
    
    # Registrar blueprints
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(empresas_bp, url_prefix='/api/empresas')
    app.register_blueprint(datos_financieros_bp, url_prefix='/api/datos-financieros')
    app.register_blueprint(predicciones_bp, url_prefix='/api/predicciones')
    app.register_blueprint(alertas_bp, url_prefix='/api/alertas')
    app.register_blueprint(etl_bp, url_prefix='/api/etl')
    app.register_blueprint(dashboard_bp, url_prefix='/api/dashboard')
    
    # Manejadores de errores
    @app.errorhandler(HTTPException)
    def handle_http_exception(e):
        """Maneja excepciones HTTP"""
        return jsonify({
            'error': e.name,
            'message': e.description,
            'status_code': e.code
        }), e.code
    
    @app.errorhandler(Exception)
    def handle_general_exception(e):
        """Maneja excepciones generales"""
        logger.error(f"Error no manejado: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Internal Server Error',
            'message': 'Ha ocurrido un error interno del servidor',
            'status_code': 500
        }), 500
    
    # Middleware para logging de requests
    @app.before_request
    def log_request_info():
        """Log información de requests"""
        logger.info(f"{request.method} {request.url} - {request.remote_addr}")
    
    @app.after_request
    def log_response_info(response):
        """Log información de responses"""
        logger.info(f"Response: {response.status_code}")
        return response
    
    # Rutas de salud y información
    @app.route('/api/health')
    def health_check():
        """Endpoint de health check"""
        try:
            db_health = db_manager.health_check()
            
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0',
                'database': db_health,
                'services': {
                    'api': 'running',
                    'ml': 'available',
                    'etl': 'available'
                }
            })
        except Exception as e:
            return jsonify({
                'status': 'unhealthy',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }), 503
    
    @app.route('/api/info')
    def api_info():
        """Información de la API"""
        return jsonify({
            'name': 'Sistema de Predicción de Quiebras Empresariales',
            'version': '1.0.0',
            'description': 'API REST para predicción de quiebras usando Machine Learning',
            'author': 'Manus AI',
            'endpoints': {
                'auth': '/api/auth',
                'empresas': '/api/empresas',
                'datos_financieros': '/api/datos-financieros',
                'predicciones': '/api/predicciones',
                'alertas': '/api/alertas',
                'etl': '/api/etl',
                'dashboard': '/api/dashboard'
            },
            'documentation': '/api/docs'
        })
    
    # Servir archivos estáticos y SPA
    @app.route('/', defaults={'path': ''})
    @app.route('/<path:path>')
    def serve_spa(path):
        """Sirve la aplicación SPA"""
        static_folder_path = app.static_folder
        
        if static_folder_path is None:
            return jsonify({'error': 'Static folder not configured'}), 404
        
        # Si el archivo existe, servirlo
        if path != "" and os.path.exists(os.path.join(static_folder_path, path)):
            return send_from_directory(static_folder_path, path)
        
        # Si no existe, servir index.html para SPA routing
        index_path = os.path.join(static_folder_path, 'index.html')
        if os.path.exists(index_path):
            return send_from_directory(static_folder_path, 'index.html')
        else:
            return jsonify({
                'message': 'Bienvenido al Sistema de Predicción de Quiebras',
                'api_info': '/api/info',
                'health': '/api/health'
            })
    
    # Cleanup al cerrar la aplicación
    @app.teardown_appcontext
    def cleanup(error):
        """Cleanup de recursos"""
        if error:
            logger.error(f"Error en contexto de aplicación: {str(error)}")
    
    return app

# Crear aplicación
app = create_app()

if __name__ == '__main__':
    try:
        logger.info("Iniciando servidor de desarrollo...")
        logger.info("API disponible en: http://0.0.0.0:5000")
        logger.info("Health check: http://0.0.0.0:5000/api/health")
        logger.info("Información API: http://0.0.0.0:5000/api/info")
        
        app.run(host='0.0.0.0', port=5000, debug=True)
        
    except KeyboardInterrupt:
        logger.info("Servidor detenido por usuario")
    except Exception as e:
        logger.error(f"Error iniciando servidor: {str(e)}")
    finally:
        # Cerrar conexiones de base de datos
        close_database()
        logger.info("Recursos liberados")

