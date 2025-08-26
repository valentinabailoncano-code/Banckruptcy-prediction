"""
Este módulo define las rutas REST para la gestión de empresas,
incluyendo CRUD completo y consultas especializadas.
"""

import logging
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from sqlalchemy import func, and_, or_
from sqlalchemy.orm import joinedload
from datetime import datetime, date
from typing import Dict, Any, List, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from database.connection import get_db_session
from services.models import (
    Empresa, DatoFinanciero, Prediccion, Alerta,
    TamañoEmpresa, EmpresaStatus, BandaRiesgo
)

# Configurar logging
logger = logging.getLogger(__name__)

# Crear blueprint
empresas_bp = Blueprint('empresas', __name__)

@empresas_bp.route('/', methods=['GET'])
def get_empresas():
    """Obtiene lista de empresas con filtros y paginación"""
    try:
        # Parámetros de consulta
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 20, type=int), 100)
        
        # Filtros
        sector = request.args.get('sector')
        tamaño = request.args.get('tamaño')
        status = request.args.get('status')
        pais = request.args.get('pais')
        search = request.args.get('search')  # Búsqueda por nombre o RUT
        
        # Ordenamiento
        sort_by = request.args.get('sort_by', 'fecha_registro')
        sort_order = request.args.get('sort_order', 'desc')
        
        with get_db_session() as session:
            # Construir query base
            query = session.query(Empresa)
            
            # Aplicar filtros
            if sector:
                query = query.filter(Empresa.sector == sector)
            
            if tamaño:
                query = query.filter(Empresa.tamaño == tamaño)
            
            if status:
                query = query.filter(Empresa.status == status)
            
            if pais:
                query = query.filter(Empresa.pais == pais)
            
            if search:
                search_term = f"%{search}%"
                query = query.filter(
                    or_(
                        Empresa.razon_social.ilike(search_term),
                        Empresa.nombre_fantasia.ilike(search_term),
                        Empresa.rut.ilike(search_term)
                    )
                )
            
            # Aplicar ordenamiento
            if hasattr(Empresa, sort_by):
                order_column = getattr(Empresa, sort_by)
                if sort_order.lower() == 'desc':
                    query = query.order_by(order_column.desc())
                else:
                    query = query.order_by(order_column.asc())
            
            # Paginación
            total = query.count()
            empresas = query.offset((page - 1) * per_page).limit(per_page).all()
            
            # Serializar resultados
            empresas_data = []
            for empresa in empresas:
                empresa_dict = {
                    'id': empresa.id,
                    'uuid': empresa.uuid,
                    'rut': empresa.rut,
                    'razon_social': empresa.razon_social,
                    'nombre_fantasia': empresa.nombre_fantasia,
                    'sector': empresa.sector,
                    'subsector': empresa.subsector,
                    'tamaño': empresa.tamaño.value if empresa.tamaño else None,
                    'pais': empresa.pais,
                    'region': empresa.region,
                    'ciudad': empresa.ciudad,
                    'status': empresa.status.value if empresa.status else None,
                    'es_publica': empresa.es_publica,
                    'fecha_constitucion': empresa.fecha_constitucion.isoformat() if empresa.fecha_constitucion else None,
                    'fecha_registro': empresa.fecha_registro.isoformat() if empresa.fecha_registro else None,
                    'numero_empleados': empresa.numero_empleados
                }
                empresas_data.append(empresa_dict)
            
            return jsonify({
                'empresas': empresas_data,
                'pagination': {
                    'page': page,
                    'per_page': per_page,
                    'total': total,
                    'pages': (total + per_page - 1) // per_page
                },
                'filters_applied': {
                    'sector': sector,
                    'tamaño': tamaño,
                    'status': status,
                    'pais': pais,
                    'search': search
                }
            })
            
    except Exception as e:
        logger.error(f"Error obteniendo empresas: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500

@empresas_bp.route('/<int:empresa_id>', methods=['GET'])
def get_empresa(empresa_id: int):
    """Obtiene una empresa específica con información detallada"""
    try:
        with get_db_session() as session:
            empresa = session.query(Empresa).options(
                joinedload(Empresa.datos_financieros),
                joinedload(Empresa.predicciones),
                joinedload(Empresa.alertas)
            ).filter_by(id=empresa_id).first()
            
            if not empresa:
                return jsonify({'error': 'Empresa no encontrada'}), 404
            
            # Obtener estadísticas adicionales
            stats = session.query(
                func.count(DatoFinanciero.id).label('total_datos_financieros'),
                func.count(Prediccion.id).label('total_predicciones'),
                func.count(Alerta.id).label('total_alertas'),
                func.count(Alerta.id).filter(Alerta.activa == True).label('alertas_activas')
            ).filter(
                DatoFinanciero.empresa_id == empresa_id,
                Prediccion.empresa_id == empresa_id,
                Alerta.empresa_id == empresa_id
            ).first()
            
            # Última predicción
            ultima_prediccion = session.query(Prediccion).filter_by(
                empresa_id=empresa_id
            ).order_by(Prediccion.fecha_prediccion.desc()).first()
            
            # Último dato financiero
            ultimo_dato = session.query(DatoFinanciero).filter_by(
                empresa_id=empresa_id
            ).order_by(DatoFinanciero.fecha_corte.desc()).first()
            
            empresa_data = {
                'id': empresa.id,
                'uuid': empresa.uuid,
                'rut': empresa.rut,
                'razon_social': empresa.razon_social,
                'nombre_fantasia': empresa.nombre_fantasia,
                'sector': empresa.sector,
                'subsector': empresa.subsector,
                'actividad_economica': empresa.actividad_economica,
                'codigo_ciiu': empresa.codigo_ciiu,
                'tamaño': empresa.tamaño.value if empresa.tamaño else None,
                'pais': empresa.pais,
                'region': empresa.region,
                'ciudad': empresa.ciudad,
                'direccion': empresa.direccion,
                'fecha_constitucion': empresa.fecha_constitucion.isoformat() if empresa.fecha_constitucion else None,
                'fecha_inicio_actividades': empresa.fecha_inicio_actividades.isoformat() if empresa.fecha_inicio_actividades else None,
                'capital_inicial': float(empresa.capital_inicial) if empresa.capital_inicial else None,
                'numero_empleados': empresa.numero_empleados,
                'status': empresa.status.value if empresa.status else None,
                'es_publica': empresa.es_publica,
                'ticker_bolsa': empresa.ticker_bolsa,
                'fecha_registro': empresa.fecha_registro.isoformat() if empresa.fecha_registro else None,
                'fecha_actualizacion': empresa.fecha_actualizacion.isoformat() if empresa.fecha_actualizacion else None,
                'estadisticas': {
                    'total_datos_financieros': stats.total_datos_financieros or 0,
                    'total_predicciones': stats.total_predicciones or 0,
                    'total_alertas': stats.total_alertas or 0,
                    'alertas_activas': stats.alertas_activas or 0
                },
                'ultima_prediccion': {
                    'id': ultima_prediccion.id,
                    'fecha': ultima_prediccion.fecha_prediccion.isoformat(),
                    'probabilidad_ml': ultima_prediccion.probabilidad_ml,
                    'banda_riesgo': ultima_prediccion.banda_riesgo_blended.value,
                    'altman_z_score': ultima_prediccion.altman_z_score
                } if ultima_prediccion else None,
                'ultimo_dato_financiero': {
                    'id': ultimo_dato.id,
                    'fecha_corte': ultimo_dato.fecha_corte.isoformat(),
                    'total_activos': float(ultimo_dato.total_activos) if ultimo_dato.total_activos else None,
                    'altman_z_score': ultimo_dato.altman_z_score
                } if ultimo_dato else None
            }
            
            return jsonify(empresa_data)
            
    except Exception as e:
        logger.error(f"Error obteniendo empresa {empresa_id}: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500

@empresas_bp.route('/', methods=['POST'])
@jwt_required()
def create_empresa():
    """Crea una nueva empresa"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No se proporcionaron datos'}), 400
        
        # Validaciones obligatorias
        required_fields = ['rut', 'razon_social', 'sector']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'Campo obligatorio: {field}'}), 400
        
        with get_db_session() as session:
            # Verificar que no exista empresa con el mismo RUT
            existing = session.query(Empresa).filter_by(rut=data['rut']).first()
            if existing:
                return jsonify({'error': 'Ya existe una empresa con este RUT'}), 409
            
            # Crear nueva empresa
            empresa_data = {
                'rut': data['rut'],
                'razon_social': data['razon_social'],
                'sector': data['sector']
            }
            
            # Campos opcionales
            optional_fields = [
                'nombre_fantasia', 'subsector', 'actividad_economica', 'codigo_ciiu',
                'pais', 'region', 'ciudad', 'direccion', 'numero_empleados',
                'es_publica', 'ticker_bolsa'
            ]
            
            for field in optional_fields:
                if field in data:
                    empresa_data[field] = data[field]
            
            # Convertir tamaño si se proporciona
            if 'tamaño' in data:
                try:
                    empresa_data['tamaño'] = TamañoEmpresa(data['tamaño'])
                except ValueError:
                    return jsonify({'error': f'Tamaño de empresa inválido: {data["tamaño"]}'}), 400
            
            # Convertir fechas
            date_fields = ['fecha_constitucion', 'fecha_inicio_actividades']
            for field in date_fields:
                if field in data and data[field]:
                    try:
                        empresa_data[field] = datetime.fromisoformat(data[field]).date()
                    except ValueError:
                        return jsonify({'error': f'Fecha inválida en {field}'}), 400
            
            empresa = Empresa(**empresa_data)
            session.add(empresa)
            session.commit()
            
            logger.info(f"Empresa creada: {empresa.rut} - {empresa.razon_social}")
            
            return jsonify({
                'message': 'Empresa creada exitosamente',
                'empresa': {
                    'id': empresa.id,
                    'uuid': empresa.uuid,
                    'rut': empresa.rut,
                    'razon_social': empresa.razon_social
                }
            }), 201
            
    except Exception as e:
        logger.error(f"Error creando empresa: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500

@empresas_bp.route('/<int:empresa_id>', methods=['PUT'])
@jwt_required()
def update_empresa(empresa_id: int):
    """Actualiza una empresa existente"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No se proporcionaron datos'}), 400
        
        with get_db_session() as session:
            empresa = session.query(Empresa).filter_by(id=empresa_id).first()
            
            if not empresa:
                return jsonify({'error': 'Empresa no encontrada'}), 404
            
            # Verificar RUT único si se está cambiando
            if 'rut' in data and data['rut'] != empresa.rut:
                existing = session.query(Empresa).filter_by(rut=data['rut']).first()
                if existing:
                    return jsonify({'error': 'Ya existe una empresa con este RUT'}), 409
            
            # Actualizar campos
            updatable_fields = [
                'rut', 'razon_social', 'nombre_fantasia', 'sector', 'subsector',
                'actividad_economica', 'codigo_ciiu', 'pais', 'region', 'ciudad',
                'direccion', 'numero_empleados', 'es_publica', 'ticker_bolsa'
            ]
            
            for field in updatable_fields:
                if field in data:
                    setattr(empresa, field, data[field])
            
            # Actualizar tamaño si se proporciona
            if 'tamaño' in data:
                try:
                    empresa.tamaño = TamañoEmpresa(data['tamaño'])
                except ValueError:
                    return jsonify({'error': f'Tamaño de empresa inválido: {data["tamaño"]}'}), 400
            
            # Actualizar status si se proporciona
            if 'status' in data:
                try:
                    empresa.status = EmpresaStatus(data['status'])
                except ValueError:
                    return jsonify({'error': f'Status de empresa inválido: {data["status"]}'}), 400
            
            # Actualizar fechas
            date_fields = ['fecha_constitucion', 'fecha_inicio_actividades']
            for field in date_fields:
                if field in data and data[field]:
                    try:
                        setattr(empresa, field, datetime.fromisoformat(data[field]).date())
                    except ValueError:
                        return jsonify({'error': f'Fecha inválida en {field}'}), 400
            
            session.commit()
            
            logger.info(f"Empresa actualizada: {empresa.rut} - {empresa.razon_social}")
            
            return jsonify({
                'message': 'Empresa actualizada exitosamente',
                'empresa': {
                    'id': empresa.id,
                    'rut': empresa.rut,
                    'razon_social': empresa.razon_social
                }
            })
            
    except Exception as e:
        logger.error(f"Error actualizando empresa {empresa_id}: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500

@empresas_bp.route('/<int:empresa_id>', methods=['DELETE'])
@jwt_required()
def delete_empresa(empresa_id: int):
    """Elimina una empresa (soft delete)"""
    try:
        with get_db_session() as session:
            empresa = session.query(Empresa).filter_by(id=empresa_id).first()
            
            if not empresa:
                return jsonify({'error': 'Empresa no encontrada'}), 404
            
            # Soft delete - cambiar status a inactiva
            empresa.status = EmpresaStatus.INACTIVA
            session.commit()
            
            logger.info(f"Empresa eliminada (soft delete): {empresa.rut} - {empresa.razon_social}")
            
            return jsonify({
                'message': 'Empresa eliminada exitosamente'
            })
            
    except Exception as e:
        logger.error(f"Error eliminando empresa {empresa_id}: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500

@empresas_bp.route('/search', methods=['GET'])
def search_empresas():
    """Búsqueda avanzada de empresas"""
    try:
        query_text = request.args.get('q', '').strip()
        
        if not query_text:
            return jsonify({'error': 'Parámetro de búsqueda requerido'}), 400
        
        limit = min(request.args.get('limit', 10, type=int), 50)
        
        with get_db_session() as session:
            search_term = f"%{query_text}%"
            
            empresas = session.query(Empresa).filter(
                and_(
                    Empresa.status == EmpresaStatus.ACTIVA,
                    or_(
                        Empresa.razon_social.ilike(search_term),
                        Empresa.nombre_fantasia.ilike(search_term),
                        Empresa.rut.ilike(search_term)
                    )
                )
            ).limit(limit).all()
            
            results = []
            for empresa in empresas:
                results.append({
                    'id': empresa.id,
                    'rut': empresa.rut,
                    'razon_social': empresa.razon_social,
                    'nombre_fantasia': empresa.nombre_fantasia,
                    'sector': empresa.sector,
                    'tamaño': empresa.tamaño.value if empresa.tamaño else None
                })
            
            return jsonify({
                'query': query_text,
                'results': results,
                'total_found': len(results)
            })
            
    except Exception as e:
        logger.error(f"Error en búsqueda de empresas: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500

@empresas_bp.route('/stats', methods=['GET'])
def get_empresas_stats():
    """Obtiene estadísticas generales de empresas"""
    try:
        with get_db_session() as session:
            # Estadísticas básicas
            total_empresas = session.query(Empresa).count()
            empresas_activas = session.query(Empresa).filter_by(status=EmpresaStatus.ACTIVA).count()
            
            # Por sector
            por_sector = session.query(
                Empresa.sector,
                func.count(Empresa.id).label('count')
            ).group_by(Empresa.sector).all()
            
            # Por tamaño
            por_tamaño = session.query(
                Empresa.tamaño,
                func.count(Empresa.id).label('count')
            ).group_by(Empresa.tamaño).all()
            
            # Por país
            por_pais = session.query(
                Empresa.pais,
                func.count(Empresa.id).label('count')
            ).group_by(Empresa.pais).all()
            
            # Empresas con mayor riesgo
            empresas_alto_riesgo = session.query(
                func.count(Prediccion.id).label('count')
            ).join(Empresa).filter(
                Prediccion.banda_riesgo_blended == BandaRiesgo.HIGH,
                Empresa.status == EmpresaStatus.ACTIVA
            ).scalar()
            
            return jsonify({
                'resumen': {
                    'total_empresas': total_empresas,
                    'empresas_activas': empresas_activas,
                    'empresas_alto_riesgo': empresas_alto_riesgo or 0
                },
                'distribucion': {
                    'por_sector': [{'sector': s, 'count': c} for s, c in por_sector],
                    'por_tamaño': [{'tamaño': t.value if t else 'No especificado', 'count': c} for t, c in por_tamaño],
                    'por_pais': [{'pais': p, 'count': c} for p, c in por_pais]
                }
            })
            
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas de empresas: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500

@empresas_bp.route('/<int:empresa_id>/riesgo-historico', methods=['GET'])
def get_riesgo_historico(empresa_id: int):
    """Obtiene el historial de riesgo de una empresa"""
    try:
        with get_db_session() as session:
            empresa = session.query(Empresa).filter_by(id=empresa_id).first()
            
            if not empresa:
                return jsonify({'error': 'Empresa no encontrada'}), 404
            
            # Obtener predicciones históricas
            predicciones = session.query(Prediccion).filter_by(
                empresa_id=empresa_id
            ).order_by(Prediccion.fecha_prediccion.asc()).all()
            
            historial = []
            for pred in predicciones:
                historial.append({
                    'fecha': pred.fecha_prediccion.isoformat(),
                    'probabilidad_ml': pred.probabilidad_ml,
                    'altman_z_score': pred.altman_z_score,
                    'banda_riesgo': pred.banda_riesgo_blended.value,
                    'modelo_version': pred.modelo_version
                })
            
            return jsonify({
                'empresa': {
                    'id': empresa.id,
                    'rut': empresa.rut,
                    'razon_social': empresa.razon_social
                },
                'historial_riesgo': historial,
                'total_predicciones': len(historial)
            })
            
    except Exception as e:
        logger.error(f"Error obteniendo historial de riesgo para empresa {empresa_id}: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500

