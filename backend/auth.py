"""
Este módulo define las rutas REST para autenticación de usuarios,
gestión de tokens JWT y control de acceso.
"""

import logging
import hashlib
import secrets
import re
from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify
from flask_jwt_extended import (
    jwt_required, create_access_token, create_refresh_token,
    get_jwt_identity, get_jwt, verify_jwt_in_request
)
from sqlalchemy import func
from typing import Dict, Any, Optional
import json

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from database.connection import get_db_session
from database.models import Usuario

# Configurar logging
logger = logging.getLogger(__name__)

# Crear blueprint
auth_bp = Blueprint('auth', __name__)

# Lista de tokens revocados (en producción usar Redis)
revoked_tokens = set()

class AuthService:
    """Servicio de autenticación"""
    
    @staticmethod
    def hash_password(password: str, salt: str = None) -> tuple[str, str]:
        """Genera hash de contraseña con salt"""
        if salt is None:
            salt = secrets.token_hex(16)
        
        # Usar PBKDF2 con SHA-256
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000  # 100,000 iteraciones
        )
        
        return password_hash.hex(), salt
    
    @staticmethod
    def verify_password(password: str, password_hash: str, salt: str) -> bool:
        """Verifica contraseña contra hash"""
        computed_hash, _ = AuthService.hash_password(password, salt)
        return computed_hash == password_hash
    
    @staticmethod
    def validate_password_strength(password: str) -> tuple[bool, list[str]]:
        """Valida fortaleza de contraseña"""
        errors = []
        
        if len(password) < 8:
            errors.append("La contraseña debe tener al menos 8 caracteres")
        
        if not re.search(r'[A-Z]', password):
            errors.append("La contraseña debe contener al menos una mayúscula")
        
        if not re.search(r'[a-z]', password):
            errors.append("La contraseña debe contener al menos una minúscula")
        
        if not re.search(r'\d', password):
            errors.append("La contraseña debe contener al menos un número")
        
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            errors.append("La contraseña debe contener al menos un carácter especial")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Valida formato de email"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None

@auth_bp.route('/register', methods=['POST'])
def register():
    """Registra un nuevo usuario"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No se proporcionaron datos'}), 400
        
        # Validaciones obligatorias
        required_fields = ['username', 'email', 'password', 'nombre', 'apellido']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'Campo obligatorio: {field}'}), 400
        
        username = data['username'].strip().lower()
        email = data['email'].strip().lower()
        password = data['password']
        nombre = data['nombre'].strip()
        apellido = data['apellido'].strip()
        telefono = data.get('telefono', '').strip()
        rol = data.get('rol', 'usuario').lower()
        
        # Validaciones de formato
        if len(username) < 3:
            return jsonify({'error': 'El username debe tener al menos 3 caracteres'}), 400
        
        if not AuthService.validate_email(email):
            return jsonify({'error': 'Formato de email inválido'}), 400
        
        # Validar fortaleza de contraseña
        is_strong, password_errors = AuthService.validate_password_strength(password)
        if not is_strong:
            return jsonify({'error': 'Contraseña débil', 'details': password_errors}), 400
        
        # Validar rol
        valid_roles = ['usuario', 'analista', 'admin']
        if rol not in valid_roles:
            return jsonify({'error': f'Rol inválido. Válidos: {", ".join(valid_roles)}'}), 400
        
        with get_db_session() as session:
            # Verificar que no exista usuario con mismo username o email
            existing_user = session.query(Usuario).filter(
                (Usuario.username == username) | (Usuario.email == email)
            ).first()
            
            if existing_user:
                if existing_user.username == username:
                    return jsonify({'error': 'El username ya está en uso'}), 409
                else:
                    return jsonify({'error': 'El email ya está registrado'}), 409
            
            # Generar hash de contraseña
            password_hash, salt = AuthService.hash_password(password)
            
            # Crear usuario
            usuario = Usuario(
                username=username,
                email=email,
                password_hash=password_hash,
                salt=salt,
                nombre=nombre,
                apellido=apellido,
                telefono=telefono if telefono else None,
                rol=rol,
                permisos={}  # Permisos por defecto según rol
            )
            
            # Asignar permisos por defecto según rol
            if rol == 'admin':
                usuario.permisos = {
                    'empresas': ['read', 'write', 'delete'],
                    'datos_financieros': ['read', 'write', 'delete'],
                    'predicciones': ['read', 'write', 'execute'],
                    'etl': ['read', 'write', 'execute'],
                    'usuarios': ['read', 'write', 'delete'],
                    'sistema': ['read', 'write']
                }
            elif rol == 'analista':
                usuario.permisos = {
                    'empresas': ['read', 'write'],
                    'datos_financieros': ['read', 'write'],
                    'predicciones': ['read', 'write', 'execute'],
                    'etl': ['read', 'execute'],
                    'usuarios': ['read'],
                    'sistema': ['read']
                }
            else:  # usuario
                usuario.permisos = {
                    'empresas': ['read'],
                    'datos_financieros': ['read'],
                    'predicciones': ['read'],
                    'etl': ['read'],
                    'usuarios': [],
                    'sistema': []
                }
            
            session.add(usuario)
            session.commit()
            
            logger.info(f"Usuario registrado: {username} ({email}) - Rol: {rol}")
            
            return jsonify({
                'message': 'Usuario registrado exitosamente',
                'usuario': {
                    'id': usuario.id,
                    'username': usuario.username,
                    'email': usuario.email,
                    'nombre': usuario.nombre,
                    'apellido': usuario.apellido,
                    'rol': usuario.rol
                }
            }), 201
            
    except Exception as e:
        logger.error(f"Error registrando usuario: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500

@auth_bp.route('/login', methods=['POST'])
def login():
    """Autentica usuario y genera tokens"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No se proporcionaron datos'}), 400
        
        username_or_email = data.get('username', '').strip().lower()
        password = data.get('password', '')
        
        if not username_or_email or not password:
            return jsonify({'error': 'Username/email y contraseña son obligatorios'}), 400
        
        with get_db_session() as session:
            # Buscar usuario por username o email
            usuario = session.query(Usuario).filter(
                (Usuario.username == username_or_email) | (Usuario.email == username_or_email)
            ).first()
            
            if not usuario:
                return jsonify({'error': 'Credenciales inválidas'}), 401
            
            # Verificar si el usuario está activo
            if not usuario.activo:
                return jsonify({'error': 'Cuenta desactivada'}), 401
            
            # Verificar contraseña
            if not AuthService.verify_password(password, usuario.password_hash, usuario.salt):
                # Incrementar intentos fallidos
                usuario.intentos_login_fallidos += 1
                session.commit()
                
                # Bloquear cuenta después de 5 intentos
                if usuario.intentos_login_fallidos >= 5:
                    usuario.activo = False
                    session.commit()
                    logger.warning(f"Cuenta bloqueada por intentos fallidos: {usuario.username}")
                    return jsonify({'error': 'Cuenta bloqueada por múltiples intentos fallidos'}), 401
                
                return jsonify({'error': 'Credenciales inválidas'}), 401
            
            # Login exitoso - resetear intentos fallidos
            usuario.intentos_login_fallidos = 0
            usuario.ultimo_login = datetime.now()
            session.commit()
            
            # Crear tokens JWT
            access_token = create_access_token(
                identity=usuario.username,
                expires_delta=timedelta(hours=1),
                additional_claims={
                    'user_id': usuario.id,
                    'rol': usuario.rol,
                    'permisos': usuario.permisos
                }
            )
            
            refresh_token = create_refresh_token(
                identity=usuario.username,
                expires_delta=timedelta(days=30)
            )
            
            logger.info(f"Login exitoso: {usuario.username}")
            
            return jsonify({
                'message': 'Login exitoso',
                'access_token': access_token,
                'refresh_token': refresh_token,
                'usuario': {
                    'id': usuario.id,
                    'username': usuario.username,
                    'email': usuario.email,
                    'nombre': usuario.nombre,
                    'apellido': usuario.apellido,
                    'rol': usuario.rol,
                    'permisos': usuario.permisos,
                    'ultimo_login': usuario.ultimo_login.isoformat() if usuario.ultimo_login else None
                }
            })
            
    except Exception as e:
        logger.error(f"Error en login: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500

@auth_bp.route('/refresh', methods=['POST'])
@jwt_required(refresh=True)
def refresh():
    """Renueva token de acceso usando refresh token"""
    try:
        current_user = get_jwt_identity()
        
        with get_db_session() as session:
            usuario = session.query(Usuario).filter_by(username=current_user).first()
            
            if not usuario or not usuario.activo:
                return jsonify({'error': 'Usuario no válido'}), 401
            
            # Crear nuevo access token
            access_token = create_access_token(
                identity=usuario.username,
                expires_delta=timedelta(hours=1),
                additional_claims={
                    'user_id': usuario.id,
                    'rol': usuario.rol,
                    'permisos': usuario.permisos
                }
            )
            
            return jsonify({
                'access_token': access_token
            })
            
    except Exception as e:
        logger.error(f"Error renovando token: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500

@auth_bp.route('/logout', methods=['POST'])
@jwt_required()
def logout():
    """Cierra sesión y revoca token"""
    try:
        jti = get_jwt()['jti']  # JWT ID
        revoked_tokens.add(jti)
        
        current_user = get_jwt_identity()
        logger.info(f"Logout: {current_user}")
        
        return jsonify({'message': 'Logout exitoso'})
        
    except Exception as e:
        logger.error(f"Error en logout: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500

@auth_bp.route('/profile', methods=['GET'])
@jwt_required()
def get_profile():
    """Obtiene perfil del usuario actual"""
    try:
        current_user = get_jwt_identity()
        
        with get_db_session() as session:
            usuario = session.query(Usuario).filter_by(username=current_user).first()
            
            if not usuario:
                return jsonify({'error': 'Usuario no encontrado'}), 404
            
            return jsonify({
                'usuario': {
                    'id': usuario.id,
                    'username': usuario.username,
                    'email': usuario.email,
                    'nombre': usuario.nombre,
                    'apellido': usuario.apellido,
                    'telefono': usuario.telefono,
                    'rol': usuario.rol,
                    'permisos': usuario.permisos,
                    'email_verificado': usuario.email_verificado,
                    'ultimo_login': usuario.ultimo_login.isoformat() if usuario.ultimo_login else None,
                    'fecha_registro': usuario.fecha_registro.isoformat() if usuario.fecha_registro else None
                }
            })
            
    except Exception as e:
        logger.error(f"Error obteniendo perfil: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500

@auth_bp.route('/profile', methods=['PUT'])
@jwt_required()
def update_profile():
    """Actualiza perfil del usuario actual"""
    try:
        current_user = get_jwt_identity()
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No se proporcionaron datos'}), 400
        
        with get_db_session() as session:
            usuario = session.query(Usuario).filter_by(username=current_user).first()
            
            if not usuario:
                return jsonify({'error': 'Usuario no encontrado'}), 404
            
            # Campos actualizables
            updatable_fields = ['nombre', 'apellido', 'telefono']
            
            for field in updatable_fields:
                if field in data:
                    setattr(usuario, field, data[field])
            
            # Actualizar email si se proporciona
            if 'email' in data:
                new_email = data['email'].strip().lower()
                
                if not AuthService.validate_email(new_email):
                    return jsonify({'error': 'Formato de email inválido'}), 400
                
                # Verificar que no esté en uso
                existing = session.query(Usuario).filter(
                    Usuario.email == new_email,
                    Usuario.id != usuario.id
                ).first()
                
                if existing:
                    return jsonify({'error': 'El email ya está en uso'}), 409
                
                usuario.email = new_email
                usuario.email_verificado = False  # Requerir verificación
            
            session.commit()
            
            logger.info(f"Perfil actualizado: {usuario.username}")
            
            return jsonify({
                'message': 'Perfil actualizado exitosamente',
                'usuario': {
                    'id': usuario.id,
                    'username': usuario.username,
                    'email': usuario.email,
                    'nombre': usuario.nombre,
                    'apellido': usuario.apellido,
                    'telefono': usuario.telefono
                }
            })
            
    except Exception as e:
        logger.error(f"Error actualizando perfil: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500

@auth_bp.route('/change-password', methods=['POST'])
@jwt_required()
def change_password():
    """Cambia contraseña del usuario actual"""
    try:
        current_user = get_jwt_identity()
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No se proporcionaron datos'}), 400
        
        current_password = data.get('current_password', '')
        new_password = data.get('new_password', '')
        
        if not current_password or not new_password:
            return jsonify({'error': 'Contraseña actual y nueva son obligatorias'}), 400
        
        # Validar fortaleza de nueva contraseña
        is_strong, password_errors = AuthService.validate_password_strength(new_password)
        if not is_strong:
            return jsonify({'error': 'Nueva contraseña débil', 'details': password_errors}), 400
        
        with get_db_session() as session:
            usuario = session.query(Usuario).filter_by(username=current_user).first()
            
            if not usuario:
                return jsonify({'error': 'Usuario no encontrado'}), 404
            
            # Verificar contraseña actual
            if not AuthService.verify_password(current_password, usuario.password_hash, usuario.salt):
                return jsonify({'error': 'Contraseña actual incorrecta'}), 401
            
            # Generar hash de nueva contraseña
            new_password_hash, new_salt = AuthService.hash_password(new_password)
            
            usuario.password_hash = new_password_hash
            usuario.salt = new_salt
            session.commit()
            
            logger.info(f"Contraseña cambiada: {usuario.username}")
            
            return jsonify({'message': 'Contraseña cambiada exitosamente'})
            
    except Exception as e:
        logger.error(f"Error cambiando contraseña: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500

@auth_bp.route('/users', methods=['GET'])
@jwt_required()
def get_users():
    """Obtiene lista de usuarios (solo admin)"""
    try:
        # Verificar permisos
        claims = get_jwt()
        if claims.get('rol') != 'admin':
            return jsonify({'error': 'Acceso denegado'}), 403
        
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 20, type=int), 100)
        
        with get_db_session() as session:
            # Query con paginación
            query = session.query(Usuario).order_by(Usuario.fecha_registro.desc())
            total = query.count()
            usuarios = query.offset((page - 1) * per_page).limit(per_page).all()
            
            usuarios_data = []
            for usuario in usuarios:
                usuarios_data.append({
                    'id': usuario.id,
                    'username': usuario.username,
                    'email': usuario.email,
                    'nombre': usuario.nombre,
                    'apellido': usuario.apellido,
                    'rol': usuario.rol,
                    'activo': usuario.activo,
                    'email_verificado': usuario.email_verificado,
                    'ultimo_login': usuario.ultimo_login.isoformat() if usuario.ultimo_login else None,
                    'fecha_registro': usuario.fecha_registro.isoformat() if usuario.fecha_registro else None
                })
            
            return jsonify({
                'usuarios': usuarios_data,
                'pagination': {
                    'page': page,
                    'per_page': per_page,
                    'total': total,
                    'pages': (total + per_page - 1) // per_page
                }
            })
            
    except Exception as e:
        logger.error(f"Error obteniendo usuarios: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500

# Función para verificar si token está revocado
def check_if_token_revoked(jwt_header, jwt_payload):
    """Verifica si el token JWT está revocado"""
    jti = jwt_payload['jti']
    return jti in revoked_tokens

# Función para verificar permisos
def require_permission(resource: str, action: str):
    """Decorator para verificar permisos específicos"""
    def decorator(f):
        def wrapper(*args, **kwargs):
            try:
                verify_jwt_in_request()
                claims = get_jwt()
                permisos = claims.get('permisos', {})
                
                if resource not in permisos or action not in permisos[resource]:
                    return jsonify({'error': 'Permisos insuficientes'}), 403
                
                return f(*args, **kwargs)
            except Exception as e:
                return jsonify({'error': 'Error verificando permisos'}), 500
        
        wrapper.__name__ = f.__name__
        return wrapper
    return decorator

