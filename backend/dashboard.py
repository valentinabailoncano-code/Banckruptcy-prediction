"""
Este módulo define las rutas REST para el dashboard principal,
incluyendo agregaciones, métricas y reportes del sistema.
"""

import logging
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity, get_jwt
from sqlalchemy import func, and_, or_, desc, extract, case
from sqlalchemy.orm import joinedload
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional
import pandas as pd
from collections import defaultdict

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from database.connection import get_db_session
from services.models import (
    Empresa, DatoFinanciero, Prediccion, Alerta, LogETL,
    TamañoEmpresa, EmpresaStatus, BandaRiesgo, EstadoPrediccion
)

# Configurar logging
logger = logging.getLogger(__name__)

# Crear blueprint
dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/overview', methods=['GET'])
@jwt_required()
def get_overview():
    """Obtiene resumen general del sistema"""
    try:
        with get_db_session() as session:
            # Estadísticas básicas
            total_empresas = session.query(Empresa).count()
            empresas_activas = session.query(Empresa).filter_by(status=EmpresaStatus.ACTIVA).count()
            
            # Predicciones
            total_predicciones = session.query(Prediccion).count()
            predicciones_mes = session.query(Prediccion).filter(
                Prediccion.fecha_prediccion >= datetime.now() - timedelta(days=30)
            ).count()
            
            # Empresas por banda de riesgo (últimas predicciones)
            subquery = session.query(
                Prediccion.empresa_id,
                func.max(Prediccion.fecha_prediccion).label('max_fecha')
            ).group_by(Prediccion.empresa_id).subquery()
            
            ultimas_predicciones = session.query(Prediccion).join(
                subquery,
                and_(
                    Prediccion.empresa_id == subquery.c.empresa_id,
                    Prediccion.fecha_prediccion == subquery.c.max_fecha
                )
            ).all()
            
            riesgo_counts = defaultdict(int)
            for pred in ultimas_predicciones:
                riesgo_counts[pred.banda_riesgo_blended.value] += 1
            
            # Alertas activas
            alertas_activas = session.query(Alerta).filter_by(activa=True).count()
            alertas_criticas = session.query(Alerta).filter(
                and_(Alerta.activa == True, Alerta.severidad == 'CRITICAL')
            ).count()
            
            # Datos financieros
            total_datos_financieros = session.query(DatoFinanciero).count()
            datos_ultimo_trimestre = session.query(DatoFinanciero).filter(
                DatoFinanciero.fecha_corte >= datetime.now() - timedelta(days=90)
            ).count()
            
            # Procesos ETL
            procesos_etl_mes = session.query(LogETL).filter(
                LogETL.fecha_inicio >= datetime.now() - timedelta(days=30)
            ).count()
            
            procesos_etl_exitosos = session.query(LogETL).filter(
                and_(
                    LogETL.fecha_inicio >= datetime.now() - timedelta(days=30),
                    LogETL.estado == 'COMPLETADO'
                )
            ).count()
            
            return jsonify({
                'empresas': {
                    'total': total_empresas,
                    'activas': empresas_activas,
                    'inactivas': total_empresas - empresas_activas
                },
                'predicciones': {
                    'total': total_predicciones,
                    'ultimo_mes': predicciones_mes,
                    'por_riesgo': dict(riesgo_counts)
                },
                'alertas': {
                    'activas': alertas_activas,
                    'criticas': alertas_criticas
                },
                'datos_financieros': {
                    'total': total_datos_financieros,
                    'ultimo_trimestre': datos_ultimo_trimestre
                },
                'etl': {
                    'procesos_mes': procesos_etl_mes,
                    'exitosos_mes': procesos_etl_exitosos,
                    'tasa_exito': (procesos_etl_exitosos / procesos_etl_mes * 100) if procesos_etl_mes > 0 else 0
                },
                'timestamp': datetime.now().isoformat()
            })
            
    except Exception as e:
        logger.error(f"Error obteniendo overview: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500

@dashboard_bp.route('/riesgo-sectorial', methods=['GET'])
@jwt_required()
def get_riesgo_sectorial():
    """Obtiene análisis de riesgo por sector"""
    try:
        with get_db_session() as session:
            # Obtener últimas predicciones por empresa
            subquery = session.query(
                Prediccion.empresa_id,
                func.max(Prediccion.fecha_prediccion).label('max_fecha')
            ).group_by(Prediccion.empresa_id).subquery()
            
            # Join con empresas para obtener sector
            query = session.query(
                Empresa.sector,
                Prediccion.banda_riesgo_blended,
                func.count(Prediccion.id).label('count'),
                func.avg(Prediccion.probabilidad_ml).label('avg_probabilidad'),
                func.avg(Prediccion.altman_z_score).label('avg_altman')
            ).join(
                Prediccion, Empresa.id == Prediccion.empresa_id
            ).join(
                subquery,
                and_(
                    Prediccion.empresa_id == subquery.c.empresa_id,
                    Prediccion.fecha_prediccion == subquery.c.max_fecha
                )
            ).filter(
                Empresa.status == EmpresaStatus.ACTIVA
            ).group_by(
                Empresa.sector, Prediccion.banda_riesgo_blended
            ).all()
            
            # Organizar datos por sector
            sectores = defaultdict(lambda: {
                'total_empresas': 0,
                'riesgo_bajo': 0,
                'riesgo_medio': 0,
                'riesgo_alto': 0,
                'riesgo_critico': 0,
                'probabilidad_promedio': 0,
                'altman_promedio': 0
            })
            
            for sector, banda_riesgo, count, avg_prob, avg_altman in query:
                sectores[sector]['total_empresas'] += count
                
                if banda_riesgo == BandaRiesgo.LOW:
                    sectores[sector]['riesgo_bajo'] = count
                elif banda_riesgo == BandaRiesgo.MEDIUM:
                    sectores[sector]['riesgo_medio'] = count
                elif banda_riesgo == BandaRiesgo.HIGH:
                    sectores[sector]['riesgo_alto'] = count
                elif banda_riesgo == BandaRiesgo.CRITICAL:
                    sectores[sector]['riesgo_critico'] = count
                
                # Promedios ponderados
                if avg_prob:
                    sectores[sector]['probabilidad_promedio'] += avg_prob * count
                if avg_altman:
                    sectores[sector]['altman_promedio'] += avg_altman * count
            
            # Calcular promedios finales
            for sector_data in sectores.values():
                if sector_data['total_empresas'] > 0:
                    sector_data['probabilidad_promedio'] /= sector_data['total_empresas']
                    sector_data['altman_promedio'] /= sector_data['total_empresas']
                    
                    # Calcular porcentajes
                    total = sector_data['total_empresas']
                    sector_data['porcentaje_riesgo_alto'] = (
                        (sector_data['riesgo_alto'] + sector_data['riesgo_critico']) / total * 100
                    )
            
            # Convertir a lista ordenada por riesgo
            sectores_list = []
            for sector, data in sectores.items():
                sectores_list.append({
                    'sector': sector,
                    **data
                })
            
            # Ordenar por porcentaje de riesgo alto
            sectores_list.sort(key=lambda x: x['porcentaje_riesgo_alto'], reverse=True)
            
            return jsonify({
                'sectores': sectores_list,
                'timestamp': datetime.now().isoformat()
            })
            
    except Exception as e:
        logger.error(f"Error obteniendo riesgo sectorial: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500

@dashboard_bp.route('/tendencias-temporales', methods=['GET'])
@jwt_required()
def get_tendencias_temporales():
    """Obtiene tendencias temporales de predicciones"""
    try:
        # Parámetros
        periodo = request.args.get('periodo', 'mes')  # mes, trimestre, año
        limite = request.args.get('limite', 12, type=int)  # últimos N períodos
        
        with get_db_session() as session:
            # Determinar agrupación temporal
            if periodo == 'mes':
                date_trunc = func.date_trunc('month', Prediccion.fecha_prediccion)
                fecha_limite = datetime.now() - timedelta(days=30 * limite)
            elif periodo == 'trimestre':
                date_trunc = func.date_trunc('quarter', Prediccion.fecha_prediccion)
                fecha_limite = datetime.now() - timedelta(days=90 * limite)
            else:  # año
                date_trunc = func.date_trunc('year', Prediccion.fecha_prediccion)
                fecha_limite = datetime.now() - timedelta(days=365 * limite)
            
            # Query para tendencias
            query = session.query(
                date_trunc.label('periodo'),
                Prediccion.banda_riesgo_blended,
                func.count(Prediccion.id).label('count'),
                func.avg(Prediccion.probabilidad_ml).label('avg_probabilidad')
            ).filter(
                Prediccion.fecha_prediccion >= fecha_limite
            ).group_by(
                date_trunc, Prediccion.banda_riesgo_blended
            ).order_by(date_trunc).all()
            
            # Organizar datos por período
            periodos = defaultdict(lambda: {
                'total_predicciones': 0,
                'riesgo_bajo': 0,
                'riesgo_medio': 0,
                'riesgo_alto': 0,
                'riesgo_critico': 0,
                'probabilidad_promedio': 0
            })
            
            for periodo_fecha, banda_riesgo, count, avg_prob in query:
                periodo_str = periodo_fecha.strftime('%Y-%m' if periodo == 'mes' else '%Y-%m' if periodo == 'trimestre' else '%Y')
                
                periodos[periodo_str]['total_predicciones'] += count
                
                if banda_riesgo == BandaRiesgo.LOW:
                    periodos[periodo_str]['riesgo_bajo'] = count
                elif banda_riesgo == BandaRiesgo.MEDIUM:
                    periodos[periodo_str]['riesgo_medio'] = count
                elif banda_riesgo == BandaRiesgo.HIGH:
                    periodos[periodo_str]['riesgo_alto'] = count
                elif banda_riesgo == BandaRiesgo.CRITICAL:
                    periodos[periodo_str]['riesgo_critico'] = count
                
                if avg_prob:
                    periodos[periodo_str]['probabilidad_promedio'] += avg_prob * count
            
            # Calcular promedios y porcentajes
            for periodo_data in periodos.values():
                if periodo_data['total_predicciones'] > 0:
                    periodo_data['probabilidad_promedio'] /= periodo_data['total_predicciones']
                    
                    total = periodo_data['total_predicciones']
                    periodo_data['porcentaje_riesgo_alto'] = (
                        (periodo_data['riesgo_alto'] + periodo_data['riesgo_critico']) / total * 100
                    )
            
            # Convertir a lista ordenada
            tendencias = []
            for periodo_str in sorted(periodos.keys()):
                tendencias.append({
                    'periodo': periodo_str,
                    **periodos[periodo_str]
                })
            
            return jsonify({
                'tendencias': tendencias,
                'parametros': {
                    'periodo': periodo,
                    'limite': limite
                },
                'timestamp': datetime.now().isoformat()
            })
            
    except Exception as e:
        logger.error(f"Error obteniendo tendencias temporales: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500

@dashboard_bp.route('/alertas-resumen', methods=['GET'])
@jwt_required()
def get_alertas_resumen():
    """Obtiene resumen de alertas del sistema"""
    try:
        with get_db_session() as session:
            # Alertas por severidad
            por_severidad = session.query(
                Alerta.severidad,
                func.count(Alerta.id).label('count')
            ).filter(
                Alerta.activa == True
            ).group_by(Alerta.severidad).all()
            
            # Alertas por tipo
            por_tipo = session.query(
                Alerta.tipo_alerta,
                func.count(Alerta.id).label('count')
            ).filter(
                Alerta.activa == True
            ).group_by(Alerta.tipo_alerta).all()
            
            # Alertas recientes (últimos 7 días)
            fecha_limite = datetime.now() - timedelta(days=7)
            alertas_recientes = session.query(Alerta).filter(
                and_(
                    Alerta.fecha_creacion >= fecha_limite,
                    Alerta.activa == True
                )
            ).order_by(desc(Alerta.fecha_creacion)).limit(10).all()
            
            # Top empresas con más alertas
            top_empresas = session.query(
                Empresa.razon_social,
                Empresa.rut,
                Empresa.sector,
                func.count(Alerta.id).label('total_alertas')
            ).join(
                Alerta, Empresa.id == Alerta.empresa_id
            ).filter(
                Alerta.activa == True
            ).group_by(
                Empresa.id, Empresa.razon_social, Empresa.rut, Empresa.sector
            ).order_by(
                desc(func.count(Alerta.id))
            ).limit(10).all()
            
            # Serializar alertas recientes
            alertas_recientes_data = []
            for alerta in alertas_recientes:
                alertas_recientes_data.append({
                    'id': alerta.id,
                    'tipo': alerta.tipo_alerta,
                    'severidad': alerta.severidad,
                    'titulo': alerta.titulo,
                    'mensaje': alerta.mensaje,
                    'fecha_creacion': alerta.fecha_creacion.isoformat(),
                    'empresa': {
                        'id': alerta.empresa.id,
                        'rut': alerta.empresa.rut,
                        'razon_social': alerta.empresa.razon_social
                    } if alerta.empresa else None
                })
            
            return jsonify({
                'resumen': {
                    'total_activas': sum(count for _, count in por_severidad),
                    'por_severidad': [{'severidad': sev, 'count': count} for sev, count in por_severidad],
                    'por_tipo': [{'tipo': tipo, 'count': count} for tipo, count in por_tipo]
                },
                'alertas_recientes': alertas_recientes_data,
                'top_empresas_alertas': [
                    {
                        'razon_social': razon_social,
                        'rut': rut,
                        'sector': sector,
                        'total_alertas': total_alertas
                    }
                    for razon_social, rut, sector, total_alertas in top_empresas
                ],
                'timestamp': datetime.now().isoformat()
            })
            
    except Exception as e:
        logger.error(f"Error obteniendo resumen de alertas: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500

@dashboard_bp.route('/metricas-modelo', methods=['GET'])
@jwt_required()
def get_metricas_modelo():
    """Obtiene métricas de rendimiento de los modelos"""
    try:
        with get_db_session() as session:
            # Métricas por modelo
            metricas_modelo = session.query(
                Prediccion.modelo_id,
                Prediccion.modelo_version,
                func.count(Prediccion.id).label('total_predicciones'),
                func.avg(Prediccion.probabilidad_ml).label('avg_probabilidad'),
                func.avg(Prediccion.confianza_prediccion).label('avg_confianza'),
                func.avg(Prediccion.tiempo_procesamiento_ms).label('avg_tiempo')
            ).filter(
                Prediccion.estado == EstadoPrediccion.PROCESADA
            ).group_by(
                Prediccion.modelo_id, Prediccion.modelo_version
            ).all()
            
            # Distribución de confianza
            distribucion_confianza = session.query(
                case(
                    (Prediccion.confianza_prediccion >= 0.8, 'Alta'),
                    (Prediccion.confianza_prediccion >= 0.6, 'Media'),
                    else_='Baja'
                ).label('nivel_confianza'),
                func.count(Prediccion.id).label('count')
            ).filter(
                Prediccion.confianza_prediccion.isnot(None)
            ).group_by('nivel_confianza').all()
            
            # Tendencia de tiempo de procesamiento (últimos 30 días)
            fecha_limite = datetime.now() - timedelta(days=30)
            tendencia_tiempo = session.query(
                func.date(Prediccion.fecha_prediccion).label('fecha'),
                func.avg(Prediccion.tiempo_procesamiento_ms).label('avg_tiempo')
            ).filter(
                and_(
                    Prediccion.fecha_prediccion >= fecha_limite,
                    Prediccion.tiempo_procesamiento_ms.isnot(None)
                )
            ).group_by(
                func.date(Prediccion.fecha_prediccion)
            ).order_by('fecha').all()
            
            # Comparación Altman vs ML
            comparacion_scores = session.query(
                Prediccion.banda_riesgo_ml,
                Prediccion.banda_riesgo_altman,
                func.count(Prediccion.id).label('count')
            ).filter(
                and_(
                    Prediccion.banda_riesgo_altman.isnot(None),
                    Prediccion.banda_riesgo_ml.isnot(None)
                )
            ).group_by(
                Prediccion.banda_riesgo_ml, Prediccion.banda_riesgo_altman
            ).all()
            
            return jsonify({
                'metricas_por_modelo': [
                    {
                        'modelo_id': modelo_id,
                        'version': version,
                        'total_predicciones': total,
                        'probabilidad_promedio': float(avg_prob) if avg_prob else 0,
                        'confianza_promedio': float(avg_conf) if avg_conf else 0,
                        'tiempo_promedio_ms': float(avg_tiempo) if avg_tiempo else 0
                    }
                    for modelo_id, version, total, avg_prob, avg_conf, avg_tiempo in metricas_modelo
                ],
                'distribucion_confianza': [
                    {'nivel': nivel, 'count': count}
                    for nivel, count in distribucion_confianza
                ],
                'tendencia_tiempo_procesamiento': [
                    {
                        'fecha': fecha.isoformat(),
                        'tiempo_promedio_ms': float(avg_tiempo) if avg_tiempo else 0
                    }
                    for fecha, avg_tiempo in tendencia_tiempo
                ],
                'comparacion_altman_ml': [
                    {
                        'banda_ml': banda_ml.value,
                        'banda_altman': banda_altman,
                        'count': count
                    }
                    for banda_ml, banda_altman, count in comparacion_scores
                ],
                'timestamp': datetime.now().isoformat()
            })
            
    except Exception as e:
        logger.error(f"Error obteniendo métricas de modelo: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500

@dashboard_bp.route('/top-empresas-riesgo', methods=['GET'])
@jwt_required()
def get_top_empresas_riesgo():
    """Obtiene empresas con mayor riesgo"""
    try:
        limite = request.args.get('limite', 20, type=int)
        limite = min(limite, 100)  # Máximo 100
        
        with get_db_session() as session:
            # Obtener últimas predicciones por empresa
            subquery = session.query(
                Prediccion.empresa_id,
                func.max(Prediccion.fecha_prediccion).label('max_fecha')
            ).group_by(Prediccion.empresa_id).subquery()
            
            # Top empresas por probabilidad ML
            top_empresas = session.query(
                Empresa.id,
                Empresa.rut,
                Empresa.razon_social,
                Empresa.sector,
                Empresa.tamaño,
                Prediccion.probabilidad_ml,
                Prediccion.altman_z_score,
                Prediccion.banda_riesgo_blended,
                Prediccion.fecha_prediccion
            ).join(
                Prediccion, Empresa.id == Prediccion.empresa_id
            ).join(
                subquery,
                and_(
                    Prediccion.empresa_id == subquery.c.empresa_id,
                    Prediccion.fecha_prediccion == subquery.c.max_fecha
                )
            ).filter(
                Empresa.status == EmpresaStatus.ACTIVA
            ).order_by(
                desc(Prediccion.probabilidad_ml)
            ).limit(limite).all()
            
            empresas_data = []
            for empresa in top_empresas:
                empresas_data.append({
                    'id': empresa.id,
                    'rut': empresa.rut,
                    'razon_social': empresa.razon_social,
                    'sector': empresa.sector,
                    'tamaño': empresa.tamaño.value if empresa.tamaño else None,
                    'probabilidad_ml': empresa.probabilidad_ml,
                    'altman_z_score': empresa.altman_z_score,
                    'banda_riesgo': empresa.banda_riesgo_blended.value,
                    'fecha_ultima_prediccion': empresa.fecha_prediccion.isoformat()
                })
            
            return jsonify({
                'empresas': empresas_data,
                'total_encontradas': len(empresas_data),
                'limite_aplicado': limite,
                'timestamp': datetime.now().isoformat()
            })
            
    except Exception as e:
        logger.error(f"Error obteniendo top empresas de riesgo: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500

@dashboard_bp.route('/estadisticas-sistema', methods=['GET'])
@jwt_required()
def get_estadisticas_sistema():
    """Obtiene estadísticas generales del sistema"""
    try:
        with get_db_session() as session:
            # Estadísticas de base de datos
            stats_db = {
                'empresas': session.query(Empresa).count(),
                'datos_financieros': session.query(DatoFinanciero).count(),
                'predicciones': session.query(Prediccion).count(),
                'alertas': session.query(Alerta).count(),
                'logs_etl': session.query(LogETL).count()
            }
            
            # Crecimiento mensual (últimos 6 meses)
            fecha_limite = datetime.now() - timedelta(days=180)
            
            crecimiento_empresas = session.query(
                extract('year', Empresa.fecha_registro).label('año'),
                extract('month', Empresa.fecha_registro).label('mes'),
                func.count(Empresa.id).label('count')
            ).filter(
                Empresa.fecha_registro >= fecha_limite
            ).group_by('año', 'mes').order_by('año', 'mes').all()
            
            crecimiento_predicciones = session.query(
                extract('year', Prediccion.fecha_prediccion).label('año'),
                extract('month', Prediccion.fecha_prediccion).label('mes'),
                func.count(Prediccion.id).label('count')
            ).filter(
                Prediccion.fecha_prediccion >= fecha_limite
            ).group_by('año', 'mes').order_by('año', 'mes').all()
            
            # Uso por usuario (si hay información de usuario en logs)
            actividad_usuarios = session.query(
                LogETL.usuario,
                func.count(LogETL.id).label('procesos'),
                func.sum(LogETL.registros_procesados).label('registros_totales')
            ).filter(
                and_(
                    LogETL.usuario.isnot(None),
                    LogETL.fecha_inicio >= datetime.now() - timedelta(days=30)
                )
            ).group_by(LogETL.usuario).order_by(desc('procesos')).limit(10).all()
            
            return jsonify({
                'estadisticas_db': stats_db,
                'crecimiento': {
                    'empresas': [
                        {
                            'periodo': f"{int(año)}-{int(mes):02d}",
                            'count': count
                        }
                        for año, mes, count in crecimiento_empresas
                    ],
                    'predicciones': [
                        {
                            'periodo': f"{int(año)}-{int(mes):02d}",
                            'count': count
                        }
                        for año, mes, count in crecimiento_predicciones
                    ]
                },
                'actividad_usuarios': [
                    {
                        'usuario': usuario,
                        'procesos': procesos,
                        'registros_procesados': registros_totales or 0
                    }
                    for usuario, procesos, registros_totales in actividad_usuarios
                ],
                'timestamp': datetime.now().isoformat()
            })
            
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas del sistema: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500

