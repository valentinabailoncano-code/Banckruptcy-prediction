# backend/main.py
import os, logging
from datetime import datetime
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from werkzeug.exceptions import HTTPException

# Usa tu SystemConfig
from config.config import SystemConfig
from database.connection import initialize_database, db_manager

from backend.auth import auth_bp
from backend.empresas import empresas_bp
from backend.etl import etl_bp
from backend.predicciones import predicciones_bp
from backend.alert_engine import alertas_bp
from backend.dashboard import dashboard_bp

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("bankruptcy_api")

def create_app():
    cfg = SystemConfig()  # ← tu config unificada

    app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), "static"))

    # Config clave para Flask/JWT
    app.config["JWT_SECRET_KEY"] = cfg.api.jwt_secret
    app.config["DEBUG"] = cfg.api.debug

    # CORS (lista desde cfg.api.cors_origins)
    CORS(app, resources={r"/api/*": {"origins": cfg.api.cors_origins}}, supports_credentials=True)

    # JWT
    JWTManager(app)

    # DB
    initialize_database(create_tables=True)

    # Blueprints
    app.register_blueprint(auth_bp, url_prefix="/api/auth")
    app.register_blueprint(empresas_bp, url_prefix="/api/empresas")
    app.register_blueprint(etl_bp, url_prefix="/api/etl")
    app.register_blueprint(predicciones_bp, url_prefix="/api/predicciones")
    app.register_blueprint(alertas_bp, url_prefix="/api/alertas")
    app.register_blueprint(dashboard_bp, url_prefix="/api/dashboard")

    @app.errorhandler(HTTPException)
    def handle_http(e):
        return jsonify({"error": e.name, "message": e.description, "status_code": e.code}), e.code

    @app.errorhandler(Exception)
    def handle_ex(e):
        logger.exception("Unhandled error")
        return jsonify({"error": "Internal Server Error", "status_code": 500}), 500

    @app.before_request
    def log_req():
        logger.info("%s %s - %s", request.method, request.path, request.remote_addr)

    @app.after_request
    def log_resp(r):
        logger.info("Response: %s", r.status_code)
        return r

    @app.get("/api/health")
    def health():
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "database": db_manager.health_check()
        })

    @app.get("/api/info")
    def info():
        return jsonify({
            "name": "Sistema de Predicción de Quiebras",
            "version": "1.0.0",
            "endpoints": {
                "auth": "/api/auth",
                "empresas": "/api/empresas",
                "predicciones": "/api/predicciones",
                "alertas": "/api/alertas",
                "etl": "/api/etl",
                "dashboard": "/api/dashboard",
            }
        })

    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def spa(path: str):
        static_dir = app.static_folder
        if static_dir and path and os.path.exists(os.path.join(static_dir, path)):
            return send_from_directory(static_dir, path)
        return jsonify({"message": "API en ejecución", "health": "/api/health"})

    return app

app = create_app()

if __name__ == "__main__":
    logger.info("API: http://0.0.0.0:5000 | /api/health  /api/info")
    app.run(host="0.0.0.0", port=5000, debug=True)
