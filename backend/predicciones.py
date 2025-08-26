"""
Este módulo define las rutas REST para la gestión de predicciones
de quiebra, incluyendo ejecución de modelos ML y consultas.
"""

import logging
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from sqlalchemy import func, and_, or_, desc
from sqlalchemy.orm import joinedload
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional
import asyncio
import json

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from database.connection import get_db_session
from database.models import (
    Empresa, DatoFinanciero, Prediccion, Alerta,
    BandaRiesgo, EstadoPrediccion
)
from services.prediction.predictor import (
    PredictionService, PredictionConfig, predict_bankruptcy_risk
)
from services.model_monitoring.model_registry import ModelRegistry

# Configurar logging
logger = logging.getLogger(__name__)

# Crear blueprint
predicciones_bp = Blueprint('predicciones', __name__)

# Configurar servicios
prediction_config = PredictionConfig(
    threshold_medium=0.15,
    threshold_high=0.30,
    enable_explanations=True
)
prediction_service = PredictionService(prediction_config)

@predicciones_bp.route('/', methods=['GET'])
def get_predicciones():
    """Obtiene lista de predicciones con filtros y paginación"""
    try:
        # Parámetros de consulta
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 20, type=int), 100)
        
        # Filtros
        empresa_id = request.args.get('empresa_id', type=int)
        banda_riesgo = request.args.get('banda_riesgo')
        fecha_desde = request.args.get('fecha_desde')
        fecha_hasta = request.args.get('fecha_hasta')
        modelo_id = request.args.get('modelo_id')
        
        # Ordenamiento
        sort_by = request.args.get('sort_by', 'fecha_prediccion')
        sort_order = request.args.get('sort_order', 'desc')
        
        with get_db_session() as session:
            # Construir query base
            query = session.query(Prediccion).options(
                joinedload(Prediccion.empresa)
            )
            
            # Aplicar filtros
            if empresa_id:
                query = query.filter(Prediccion.empresa_id == empresa_id)
            
            if banda_riesgo:
                try:
                    banda = BandaRiesgo(banda_riesgo)
                    query = query.filter(Prediccion.banda_riesgo_blended == banda)
                except ValueError:
                    return jsonify({'error': f'Banda de riesgo inválida: {banda_riesgo}'}), 400
            
            if fecha_desde:
                try:
                    fecha_desde_dt = datetime.fromisoformat(fecha_desde)
                    query = query.filter(Prediccion.fecha_prediccion >= fecha_desde_dt)
                except ValueError:
                    return jsonify({'error': 'Formato de fecha_desde inválido'}), 400
            
            if fecha_hasta:
                try:
                    fecha_hasta_dt = datetime.fromisoformat(fecha_hasta)
                    query = query.filter(Prediccion.fecha_prediccion <= fecha_hasta_dt)
                except ValueError:
                    return jsonify({'error': 'Formato de fecha_hasta inválido'}), 400
            
            if modelo_id:
                query = query.filter(Prediccion.modelo_id == modelo_id)
            
            # Aplicar ordenamiento
            if hasattr(Prediccion, sort_by):
                order_column = getattr(Prediccion, sort_by)
                if sort_order.lower() == 'desc':
                    query = query.order_by(order_column.desc())
                else:
                    query = query.order_by(order_column.asc())
            
            # Paginación
            total = query.count()
            predicciones = query.offset((page - 1) * per_page).limit(per_page).all()
            
            # Serializar resultados
            predicciones_data = []
            for pred in predicciones:
                pred_dict = {
                    'id': pred.id,
                    'uuid': pred.uuid,
                    'empresa': {
                        'id': pred.empresa.id,
                        'rut': pred.empresa.rut,
                        'razon_social': pred.empresa.razon_social,
                        'sector': pred.empresa.sector
                    },
                    'fecha_prediccion': pred.fecha_prediccion.isoformat(),
                    'modelo_id': pred.modelo_id,
                    'modelo_version': pred.modelo_version,
                    'probabilidad_ml': pred.probabilidad_ml,
                    'altman_z_score': pred.altman_z_score,
                    'blended_score': pred.blended_score,
                    'banda_riesgo_ml': pred.banda_riesgo_ml.value,
                    'banda_riesgo_altman': pred.banda_riesgo_altman,
                    'banda_riesgo_blended': pred.banda_riesgo_blended.value,
                    'confianza_prediccion': pred.confianza_prediccion,
                    'tiempo_procesamiento_ms': pred.tiempo_procesamiento_ms,
                    'estado': pred.estado.value
                }
                predicciones_data.append(pred_dict)
            
            return jsonify({
                'predicciones': predicciones_data,
                'pagination': {
                    'page': page,
                    'per_page': per_page,
                    'total': total,
                    'pages': (total + per_page - 1) // per_page
                },
                'filters_applied': {
                    'empresa_id': empresa_id,
                    'banda_riesgo': banda_riesgo,
                    'fecha_desde': fecha_desde,
                    'fecha_hasta': fecha_hasta,
                    'modelo_id': modelo_id
                }
            })
            
    except Exception as e:
        logger.error(f"Error obteniendo predicciones: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500

@predicciones_bp.route('/<int:prediccion_id>', methods=['GET'])
def get_prediccion(prediccion_id: int):
    """Obtiene una predicción específica con detalles completos"""
    try:
        with get_db_session() as session:
            prediccion = session.query(Prediccion).options(
                joinedload(Prediccion.empresa),
                joinedload(Prediccion.dato_financiero)
            ).filter_by(id=prediccion_id).first()
            
            if not prediccion:
                return jsonify({'error': 'Predicción no encontrada'}), 404
            
            pred_data = {
                'id': prediccion.id,
                'uuid': prediccion.uuid,
                'empresa': {
                    'id': prediccion.empresa.id,
                    'rut': prediccion.empresa.rut,
                    'razon_social': prediccion.empresa.razon_social,
                    'sector': prediccion.empresa.sector,
                    'tamaño': prediccion.empresa.tamaño.value if prediccion.empresa.tamaño else None
                },
                'dato_financiero': {
                    'id': prediccion.dato_financiero.id,
                    'fecha_corte': prediccion.dato_financiero.fecha_corte.isoformat(),
                    'total_activos': float(prediccion.dato_financiero.total_activos) if prediccion.dato_financiero.total_activos else None
                } if prediccion.dato_financiero else None,
                'fecha_prediccion': prediccion.fecha_prediccion.isoformat(),
                'modelo_id': prediccion.modelo_id,
                'modelo_version': prediccion.modelo_version,
                'resultados': {
                    'probabilidad_ml': prediccion.probabilidad_ml,
                    'altman_z_score': prediccion.altman_z_score,
                    'blended_score': prediccion.blended_score,
                    'banda_riesgo_ml': prediccion.banda_riesgo_ml.value,
                    'banda_riesgo_altman': prediccion.banda_riesgo_altman,
                    'banda_riesgo_blended': prediccion.banda_riesgo_blended.value
                },
                'explicabilidad': {
                    'feature_contributions': prediccion.feature_contributions,
                    'top_positive_features': prediccion.top_positive_features,
                    'top_negative_features': prediccion.top_negative_features
                },
                'calidad': {
                    'confianza_prediccion': prediccion.confianza_prediccion,
                    'tiempo_procesamiento_ms': prediccion.tiempo_procesamiento_ms
                },
                'metadatos': {
                    'estado': prediccion.estado.value,
                    'observaciones': prediccion.observaciones,
                    'usuario_solicita': prediccion.usuario_solicita
                }
            }
            
            return jsonify(pred_data)
            
    except Exception as e:
        logger.error(f"Error obteniendo predicción {prediccion_id}: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500

@predicciones_bp.route('/ejecutar', methods=['POST'])
@jwt_required()
def ejecutar_prediccion():
    """Ejecuta una nueva predicción para una empresa"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No se proporcionaron datos'}), 400
        
        empresa_id = data.get('empresa_id')
        dato_financiero_id = data.get('dato_financiero_id')
        include_explanations = data.get('include_explanations', True)
        
        if not empresa_id:
            return jsonify({'error': 'empresa_id es obligatorio'}), 400
        
        usuario = get_jwt_identity()
        
        with get_db_session() as session:
            # Verificar que existe la empresa
            empresa = session.query(Empresa).filter_by(id=empresa_id).first()
            if not empresa:
                return jsonify({'error': 'Empresa no encontrada'}), 404
            
            # Obtener dato financiero
            if dato_financiero_id:
                dato_financiero = session.query(DatoFinanciero).filter_by(
                    id=dato_financiero_id,
                    empresa_id=empresa_id
                ).first()
                
                if not dato_financiero:
                    return jsonify({'error': 'Dato financiero no encontrado'}), 404
            else:
                # Usar el dato financiero más reciente
                dato_financiero = session.query(DatoFinanciero).filter_by(
                    empresa_id=empresa_id
                ).order_by(DatoFinanciero.fecha_corte.desc()).first()
                
                if not dato_financiero:
                    return jsonify({'error': 'No se encontraron datos financieros para la empresa'}), 404
            
            # Preparar datos para predicción
            financial_data = {
                'wc_ta': dato_financiero.wc_ta,
                're_ta': dato_financiero.re_ta,
                'ebit_ta': dato_financiero.ebit_ta,
                'me_tl': dato_financiero.me_tl,
                's_ta': dato_financiero.s_ta,
                'current_ratio': dato_financiero.liquidez_corriente,
                'debt_assets': dato_financiero.endeudamiento_total,
                'roa': dato_financiero.roa,
                'roe': dato_financiero.roe,
                'total_assets': float(dato_financiero.total_activos) if dato_financiero.total_activos else None
            }
            
            # Filtrar valores None
            financial_data = {k: v for k, v in financial_data.items() if v is not None}
            
            if len(financial_data) < 5:  # Mínimo de características requeridas
                return jsonify({'error': 'Datos financieros insuficientes para realizar predicción'}), 400
            
            # Crear registro de predicción
            prediccion = Prediccion(
                empresa_id=empresa_id,
                dato_financiero_id=dato_financiero.id,
                modelo_id='default_model',
                modelo_version='1.0.0',
                estado=EstadoPrediccion.PENDIENTE,
                usuario_solicita=usuario
            )
            
            session.add(prediccion)
            session.commit()
            
            try:
                # Ejecutar predicción usando el servicio
                resultado = predict_bankruptcy_risk(
                    financial_data,
                    empresa_id=empresa_id,
                    include_explanations=include_explanations
                )
                
                # Actualizar registro con resultados
                prediccion.probabilidad_ml = resultado.probabilidad_ml
                prediccion.altman_z_score = resultado.altman_z_score
                prediccion.blended_score = resultado.blended_score
                prediccion.banda_riesgo_ml = BandaRiesgo(resultado.banda_riesgo_ml)
                prediccion.banda_riesgo_altman = resultado.banda_riesgo_altman
                prediccion.banda_riesgo_blended = BandaRiesgo(resultado.banda_riesgo_blended)
                prediccion.confianza_prediccion = resultado.confianza_prediccion
                prediccion.tiempo_procesamiento_ms = resultado.tiempo_procesamiento_ms
                prediccion.estado = EstadoPrediccion.PROCESADA
                
                # Guardar explicabilidad si está disponible
                if hasattr(resultado, 'feature_contributions') and resultado.feature_contributions:
                    prediccion.feature_contributions = resultado.feature_contributions
                    prediccion.top_positive_features = resultado.top_positive_features
                    prediccion.top_negative_features = resultado.top_negative_features
                
                session.commit()
                
                # Crear alerta si es riesgo alto
                if resultado.banda_riesgo_blended in ['HIGH', 'CRITICAL']:
                    alerta = Alerta(
                        empresa_id=empresa_id,
                        prediccion_id=prediccion.id,
                        tipo_alerta='riesgo_alto',
                        severidad='HIGH' if resultado.banda_riesgo_blended == 'HIGH' else 'CRITICAL',
                        titulo=f'Riesgo de quiebra {resultado.banda_riesgo_blended.lower()} detectado',
                        mensaje=f'La empresa {empresa.razon_social} presenta un riesgo {resultado.banda_riesgo_blended.lower()} de quiebra con probabilidad ML de {resultado.probabilidad_ml:.2%}',
                        datos_adicionales={
                            'probabilidad_ml': resultado.probabilidad_ml,
                            'altman_z_score': resultado.altman_z_score,
                            'blended_score': resultado.blended_score
                        }
                    )
                    session.add(alerta)
                    session.commit()
                
                logger.info(f"Predicción ejecutada exitosamente para empresa {empresa_id}")
                
                return jsonify({
                    'message': 'Predicción ejecutada exitosamente',
                    'prediccion': {
                        'id': prediccion.id,
                        'uuid': prediccion.uuid,
                        'empresa': {
                            'id': empresa.id,
                            'rut': empresa.rut,
                            'razon_social': empresa.razon_social
                        },
                        'resultados': {
                            'probabilidad_ml': resultado.probabilidad_ml,
                            'altman_z_score': resultado.altman_z_score,
                            'blended_score': resultado.blended_score,
                            'banda_riesgo_blended': resultado.banda_riesgo_blended,
                            'confianza_prediccion': resultado.confianza_prediccion
                        }
                    }
                }), 201
                
            except Exception as e:
                # Marcar predicción como error
                prediccion.estado = EstadoPrediccion.ERROR
                prediccion.observaciones = str(e)
                session.commit()
                
                logger.error(f"Error ejecutando predicción: {str(e)}")
                return jsonify({'error': f'Error ejecutando predicción: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"Error en endpoint de predicción: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500

@predicciones_bp.route('/lote', methods=['POST'])
@jwt_required()
def ejecutar_predicciones_lote():
    """Ejecuta predicciones en lote para múltiples empresas"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No se proporcionaron datos'}), 400
        
        empresa_ids = data.get('empresa_ids', [])
        include_explanations = data.get('include_explanations', False)  # Desactivar por defecto para lotes
        
        if not empresa_ids:
            return jsonify({'error': 'Lista de empresa_ids es obligatoria'}), 400
        
        if len(empresa_ids) > 100:  # Límite de seguridad
            return jsonify({'error': 'Máximo 100 empresas por lote'}), 400
        
        usuario = get_jwt_identity()
        resultados = []
        errores = []
        
        with get_db_session() as session:
            for empresa_id in empresa_ids:
                try:
                    # Verificar empresa
                    empresa = session.query(Empresa).filter_by(id=empresa_id).first()
                    if not empresa:
                        errores.append(f'Empresa {empresa_id} no encontrada')
                        continue
                    
                    # Obtener último dato financiero
                    dato_financiero = session.query(DatoFinanciero).filter_by(
                        empresa_id=empresa_id
                    ).order_by(DatoFinanciero.fecha_corte.desc()).first()
                    
                    if not dato_financiero:
                        errores.append(f'No hay datos financieros para empresa {empresa_id}')
                        continue
                    
                    # Preparar datos
                    financial_data = {
                        'wc_ta': dato_financiero.wc_ta,
                        're_ta': dato_financiero.re_ta,
                        'ebit_ta': dato_financiero.ebit_ta,
                        'me_tl': dato_financiero.me_tl,
                        's_ta': dato_financiero.s_ta,
                        'current_ratio': dato_financiero.liquidez_corriente,
                        'debt_assets': dato_financiero.endeudamiento_total,
                        'roa': dato_financiero.roa
                    }
                    
                    financial_data = {k: v for k, v in financial_data.items() if v is not None}
                    
                    if len(financial_data) < 5:
                        errores.append(f'Datos insuficientes para empresa {empresa_id}')
                        continue
                    
                    # Crear registro
                    prediccion = Prediccion(
                        empresa_id=empresa_id,
                        dato_financiero_id=dato_financiero.id,
                        modelo_id='default_model',
                        modelo_version='1.0.0',
                        estado=EstadoPrediccion.PENDIENTE,
                        usuario_solicita=usuario
                    )
                    
                    session.add(prediccion)
                    session.flush()  # Para obtener el ID
                    
                    # Ejecutar predicción
                    resultado = predict_bankruptcy_risk(
                        financial_data,
                        empresa_id=empresa_id,
                        include_explanations=include_explanations
                    )
                    
                    # Actualizar registro
                    prediccion.probabilidad_ml = resultado.probabilidad_ml
                    prediccion.altman_z_score = resultado.altman_z_score
                    prediccion.blended_score = resultado.blended_score
                    prediccion.banda_riesgo_ml = BandaRiesgo(resultado.banda_riesgo_ml)
                    prediccion.banda_riesgo_altman = resultado.banda_riesgo_altman
                    prediccion.banda_riesgo_blended = BandaRiesgo(resultado.banda_riesgo_blended)
                    prediccion.confianza_prediccion = resultado.confianza_prediccion
                    prediccion.tiempo_procesamiento_ms = resultado.tiempo_procesamiento_ms
                    prediccion.estado = EstadoPrediccion.PROCESADA
                    
                    resultados.append({
                        'empresa_id': empresa_id,
                        'prediccion_id': prediccion.id,
                        'rut': empresa.rut,
                        'razon_social': empresa.razon_social,
                        'probabilidad_ml': resultado.probabilidad_ml,
                        'banda_riesgo': resultado.banda_riesgo_blended,
                        'altman_z_score': resultado.altman_z_score
                    })
                    
                except Exception as e:
                    errores.append(f'Error en empresa {empresa_id}: {str(e)}')
                    continue
            
            session.commit()
        
        return jsonify({
            'message': f'Procesadas {len(resultados)} predicciones exitosamente',
            'resultados': resultados,
            'errores': errores,
            'resumen': {
                'total_solicitadas': len(empresa_ids),
                'exitosas': len(resultados),
                'con_errores': len(errores)
            }
        })
        
    except Exception as e:
        logger.error(f"Error en predicciones en lote: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500

@predicciones_bp.route('/stats', methods=['GET'])
def get_predicciones_stats():
    """Obtiene estadísticas de predicciones"""
    try:
        with get_db_session() as session:
            # Estadísticas básicas
            total_predicciones = session.query(Prediccion).count()
            
            # Por banda de riesgo
            por_banda = session.query(
                Prediccion.banda_riesgo_blended,
                func.count(Prediccion.id).label('count')
            ).group_by(Prediccion.banda_riesgo_blended).all()
            
            # Por modelo
            por_modelo = session.query(
                Prediccion.modelo_id,
                func.count(Prediccion.id).label('count')
            ).group_by(Prediccion.modelo_id).all()
            
            # Predicciones recientes (últimos 30 días)
            fecha_limite = datetime.now() - timedelta(days=30)
            predicciones_recientes = session.query(Prediccion).filter(
                Prediccion.fecha_prediccion >= fecha_limite
            ).count()
            
            # Empresas con riesgo alto
            empresas_riesgo_alto = session.query(
                func.count(func.distinct(Prediccion.empresa_id)).label('count')
            ).filter(
                Prediccion.banda_riesgo_blended.in_([BandaRiesgo.HIGH, BandaRiesgo.CRITICAL])
            ).scalar()
            
            # Tiempo promedio de procesamiento
            tiempo_promedio = session.query(
                func.avg(Prediccion.tiempo_procesamiento_ms).label('avg_time')
            ).scalar()
            
            return jsonify({
                'resumen': {
                    'total_predicciones': total_predicciones,
                    'predicciones_recientes': predicciones_recientes,
                    'empresas_riesgo_alto': empresas_riesgo_alto or 0,
                    'tiempo_promedio_ms': float(tiempo_promedio) if tiempo_promedio else 0
                },
                'distribucion': {
                    'por_banda_riesgo': [
                        {'banda': b.value if b else 'No especificado', 'count': c} 
                        for b, c in por_banda
                    ],
                    'por_modelo': [
                        {'modelo': m, 'count': c} 
                        for m, c in por_modelo
                    ]
                }
            })
            
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas de predicciones: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500

@predicciones_bp.route('/modelos-disponibles', methods=['GET'])
def get_modelos_disponibles():
    """Obtiene lista de modelos disponibles para predicción"""
    try:
        # Aquí se integraría con el ModelRegistry
        # Por ahora retornamos información básica
        
        modelos = [
            {
                'id': 'default_model',
                'nombre': 'Modelo XGBoost Calibrado',
                'version': '1.0.0',
                'descripcion': 'Modelo principal con XGBoost y calibración isotónica',
                'estado': 'production',
                'metricas': {
                    'roc_auc': 0.85,
                    'precision': 0.78,
                    'recall': 0.82
                },
                'fecha_entrenamiento': '2025-01-26T00:00:00'
            }
        ]
        
        return jsonify({
            'modelos': modelos,
            'modelo_por_defecto': 'default_model'
        })
        
    except Exception as e:
        logger.error(f"Error obteniendo modelos disponibles: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500

