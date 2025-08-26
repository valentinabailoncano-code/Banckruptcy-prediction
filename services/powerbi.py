"""
Este módulo implementa endpoints especializados para la integración
con Microsoft Power BI, optimizados para consultas de BI y reportes.
"""

from flask import Blueprint, request, jsonify, send_file
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import pandas as pd
import io
import json
import logging

from database.connection import DatabaseManager
from database.models import Empresa, DatoFinanciero, Prediccion, Alerta
from powerbi.connectors.powerbi_connector import PowerBIConnector, PowerBIConfig
from services.auth.auth_service import require_auth, require_permission

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear blueprint
powerbi_bp = Blueprint('powerbi', __name__, url_prefix='/api/powerbi')

# Inicializar conector
powerbi_connector = PowerBIConnector()
db_manager = DatabaseManager()

@powerbi_bp.route('/health', methods=['GET'])
def health_check():
    """
    Verifica el estado de la API de Power BI
    
    Returns:
        JSON con estado del servicio
    """
    try:
        return jsonify({
            'status': 'healthy',
            'service': 'PowerBI API',
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0'
        })
    except Exception as e:
        logger.error(f"Error en health check: {str(e)}")
        return jsonify({'error': str(e)}), 500

@powerbi_bp.route('/datasets/empresas', methods=['GET'])
@require_auth
@require_permission('empresas', 'read')
def get_empresas_dataset():
    """
    Obtiene dataset de empresas optimizado para Power BI
    
    Query Parameters:
        - sector: Filtro por sector
        - pais: Filtro por país
        - activa: Filtro por estado (true/false)
        - limit: Límite de registros (default: 10000)
        - format: Formato de salida (json/csv/parquet)
        
    Returns:
        Dataset de empresas en formato solicitado
    """
    try:
        # Obtener parámetros
        sector_filter = request.args.getlist('sector')
        pais_filter = request.args.get('pais')
        activa_filter = request.args.get('activa')
        limit = int(request.args.get('limit', 10000))
        format_type = request.args.get('format', 'json')
        
        # Validar límite
        if limit > 50000:
            return jsonify({'error': 'Límite máximo excedido (50,000 registros)'}), 400
        
        # Obtener dataset
        df = powerbi_connector.get_empresas_dataset(
            sector_filter=sector_filter if sector_filter else None,
            pais_filter=pais_filter,
            activa_filter=activa_filter == 'true' if activa_filter else None,
            limit=limit
        )
        
        # Retornar en formato solicitado
        return _format_response(df, format_type, 'empresas_dataset')
        
    except Exception as e:
        logger.error(f"Error obteniendo dataset de empresas: {str(e)}")
        return jsonify({'error': str(e)}), 500

@powerbi_bp.route('/datasets/datos-financieros', methods=['GET'])
@require_auth
@require_permission('datos_financieros', 'read')
def get_datos_financieros_dataset():
    """
    Obtiene dataset de datos financieros optimizado para Power BI
    
    Query Parameters:
        - fecha_desde: Fecha inicio (YYYY-MM-DD)
        - fecha_hasta: Fecha fin (YYYY-MM-DD)
        - tipo_periodo: Filtro por tipo (anual/trimestral/mensual)
        - sector: Filtro por sector
        - limit: Límite de registros (default: 10000)
        - format: Formato de salida (json/csv/parquet)
        
    Returns:
        Dataset de datos financieros en formato solicitado
    """
    try:
        # Obtener parámetros
        fecha_desde = request.args.get('fecha_desde')
        fecha_hasta = request.args.get('fecha_hasta')
        tipo_periodo = request.args.get('tipo_periodo')
        sector_filter = request.args.getlist('sector')
        limit = int(request.args.get('limit', 10000))
        format_type = request.args.get('format', 'json')
        
        # Validar fechas
        if fecha_desde:
            fecha_desde = datetime.strptime(fecha_desde, '%Y-%m-%d')
        if fecha_hasta:
            fecha_hasta = datetime.strptime(fecha_hasta, '%Y-%m-%d')
        
        # Validar límite
        if limit > 50000:
            return jsonify({'error': 'Límite máximo excedido (50,000 registros)'}), 400
        
        # Obtener dataset
        df = powerbi_connector.get_datos_financieros_dataset(
            fecha_desde=fecha_desde,
            fecha_hasta=fecha_hasta,
            tipo_periodo=tipo_periodo,
            sector_filter=sector_filter if sector_filter else None,
            limit=limit
        )
        
        # Retornar en formato solicitado
        return _format_response(df, format_type, 'datos_financieros_dataset')
        
    except ValueError as e:
        return jsonify({'error': f'Formato de fecha inválido: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Error obteniendo dataset financiero: {str(e)}")
        return jsonify({'error': str(e)}), 500

@powerbi_bp.route('/datasets/predicciones', methods=['GET'])
@require_auth
@require_permission('predicciones', 'read')
def get_predicciones_dataset():
    """
    Obtiene dataset de predicciones optimizado para Power BI
    
    Query Parameters:
        - fecha_desde: Fecha inicio (YYYY-MM-DD)
        - fecha_hasta: Fecha fin (YYYY-MM-DD)
        - banda_riesgo: Filtro por banda (LOW/MEDIUM/HIGH/CRITICAL)
        - modelo_version: Filtro por versión del modelo
        - sector: Filtro por sector
        - limit: Límite de registros (default: 10000)
        - format: Formato de salida (json/csv/parquet)
        
    Returns:
        Dataset de predicciones en formato solicitado
    """
    try:
        # Obtener parámetros
        fecha_desde = request.args.get('fecha_desde')
        fecha_hasta = request.args.get('fecha_hasta')
        banda_riesgo_filter = request.args.getlist('banda_riesgo')
        modelo_version = request.args.get('modelo_version')
        sector_filter = request.args.getlist('sector')
        limit = int(request.args.get('limit', 10000))
        format_type = request.args.get('format', 'json')
        
        # Validar fechas
        if fecha_desde:
            fecha_desde = datetime.strptime(fecha_desde, '%Y-%m-%d')
        if fecha_hasta:
            fecha_hasta = datetime.strptime(fecha_hasta, '%Y-%m-%d')
        
        # Validar límite
        if limit > 50000:
            return jsonify({'error': 'Límite máximo excedido (50,000 registros)'}), 400
        
        # Obtener dataset
        df = powerbi_connector.get_predicciones_dataset(
            fecha_desde=fecha_desde,
            fecha_hasta=fecha_hasta,
            banda_riesgo_filter=banda_riesgo_filter if banda_riesgo_filter else None,
            modelo_version=modelo_version,
            sector_filter=sector_filter if sector_filter else None,
            limit=limit
        )
        
        # Retornar en formato solicitado
        return _format_response(df, format_type, 'predicciones_dataset')
        
    except ValueError as e:
        return jsonify({'error': f'Formato de fecha inválido: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Error obteniendo dataset de predicciones: {str(e)}")
        return jsonify({'error': str(e)}), 500

@powerbi_bp.route('/datasets/alertas', methods=['GET'])
@require_auth
@require_permission('alertas', 'read')
def get_alertas_dataset():
    """
    Obtiene dataset de alertas optimizado para Power BI
    
    Query Parameters:
        - fecha_desde: Fecha inicio (YYYY-MM-DD)
        - fecha_hasta: Fecha fin (YYYY-MM-DD)
        - severidad: Filtro por severidad (LOW/MEDIUM/HIGH/CRITICAL)
        - estado: Filtro por estado (ACTIVA/EN_PROCESO/RESUELTA/DESCARTADA)
        - tipo: Filtro por tipo de alerta
        - limit: Límite de registros (default: 10000)
        - format: Formato de salida (json/csv/parquet)
        
    Returns:
        Dataset de alertas en formato solicitado
    """
    try:
        # Obtener parámetros
        fecha_desde = request.args.get('fecha_desde')
        fecha_hasta = request.args.get('fecha_hasta')
        severidad_filter = request.args.getlist('severidad')
        estado_filter = request.args.getlist('estado')
        tipo_filter = request.args.getlist('tipo')
        limit = int(request.args.get('limit', 10000))
        format_type = request.args.get('format', 'json')
        
        # Validar fechas
        if fecha_desde:
            fecha_desde = datetime.strptime(fecha_desde, '%Y-%m-%d')
        if fecha_hasta:
            fecha_hasta = datetime.strptime(fecha_hasta, '%Y-%m-%d')
        
        # Validar límite
        if limit > 50000:
            return jsonify({'error': 'Límite máximo excedido (50,000 registros)'}), 400
        
        # Obtener dataset
        df = powerbi_connector.get_alertas_dataset(
            fecha_desde=fecha_desde,
            fecha_hasta=fecha_hasta,
            severidad_filter=severidad_filter if severidad_filter else None,
            estado_filter=estado_filter if estado_filter else None,
            tipo_filter=tipo_filter if tipo_filter else None,
            limit=limit
        )
        
        # Retornar en formato solicitado
        return _format_response(df, format_type, 'alertas_dataset')
        
    except ValueError as e:
        return jsonify({'error': f'Formato de fecha inválido: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Error obteniendo dataset de alertas: {str(e)}")
        return jsonify({'error': str(e)}), 500

@powerbi_bp.route('/datasets/metricas-agregadas', methods=['GET'])
@require_auth
@require_permission('dashboard', 'read')
def get_metricas_agregadas_dataset():
    """
    Obtiene dataset de métricas agregadas optimizado para Power BI
    
    Query Parameters:
        - nivel_agregacion: Nivel (diario/semanal/mensual/trimestral)
        - fecha_desde: Fecha inicio (YYYY-MM-DD)
        - fecha_hasta: Fecha fin (YYYY-MM-DD)
        - incluir_sectores: Incluir desglose sectorial (true/false)
        - limit: Límite de registros (default: 1000)
        - format: Formato de salida (json/csv/parquet)
        
    Returns:
        Dataset de métricas agregadas en formato solicitado
    """
    try:
        # Obtener parámetros
        nivel_agregacion = request.args.get('nivel_agregacion', 'mensual')
        fecha_desde = request.args.get('fecha_desde')
        fecha_hasta = request.args.get('fecha_hasta')
        incluir_sectores = request.args.get('incluir_sectores', 'false') == 'true'
        limit = int(request.args.get('limit', 1000))
        format_type = request.args.get('format', 'json')
        
        # Validar fechas
        if fecha_desde:
            fecha_desde = datetime.strptime(fecha_desde, '%Y-%m-%d')
        if fecha_hasta:
            fecha_hasta = datetime.strptime(fecha_hasta, '%Y-%m-%d')
        
        # Validar límite
        if limit > 10000:
            return jsonify({'error': 'Límite máximo excedido (10,000 registros)'}), 400
        
        # Obtener dataset
        df = powerbi_connector.get_metricas_agregadas_dataset(
            nivel_agregacion=nivel_agregacion,
            fecha_desde=fecha_desde,
            fecha_hasta=fecha_hasta,
            incluir_sectores=incluir_sectores,
            limit=limit
        )
        
        # Retornar en formato solicitado
        return _format_response(df, format_type, 'metricas_agregadas_dataset')
        
    except ValueError as e:
        return jsonify({'error': f'Formato de fecha inválido: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Error obteniendo métricas agregadas: {str(e)}")
        return jsonify({'error': str(e)}), 500

@powerbi_bp.route('/datasets/schema/<dataset_name>', methods=['GET'])
@require_auth
def get_dataset_schema(dataset_name: str):
    """
    Obtiene el esquema de un dataset específico
    
    Args:
        dataset_name: Nombre del dataset (empresas/datos-financieros/predicciones/alertas/metricas-agregadas)
        
    Returns:
        Esquema del dataset con tipos de datos y descripciones
    """
    try:
        schemas = {
            'empresas': {
                'columns': [
                    {'name': 'id', 'type': 'int64', 'description': 'ID único de la empresa'},
                    {'name': 'rut', 'type': 'string', 'description': 'RUT de la empresa'},
                    {'name': 'razon_social', 'type': 'string', 'description': 'Razón social'},
                    {'name': 'nombre_fantasia', 'type': 'string', 'description': 'Nombre de fantasía'},
                    {'name': 'sector', 'type': 'string', 'description': 'Sector económico'},
                    {'name': 'subsector', 'type': 'string', 'description': 'Subsector económico'},
                    {'name': 'pais', 'type': 'string', 'description': 'País de origen'},
                    {'name': 'region', 'type': 'string', 'description': 'Región'},
                    {'name': 'ciudad', 'type': 'string', 'description': 'Ciudad'},
                    {'name': 'tamaño_empresa', 'type': 'string', 'description': 'Tamaño (MICRO/PEQUEÑA/MEDIANA/GRANDE)'},
                    {'name': 'activa', 'type': 'boolean', 'description': 'Estado activo'},
                    {'name': 'fecha_constitucion', 'type': 'datetime', 'description': 'Fecha de constitución'},
                    {'name': 'numero_empleados', 'type': 'int64', 'description': 'Número de empleados'},
                    {'name': 'ingresos_anuales', 'type': 'float64', 'description': 'Ingresos anuales'},
                    {'name': 'es_publica', 'type': 'boolean', 'description': 'Empresa pública'},
                    {'name': 'fecha_creacion', 'type': 'datetime', 'description': 'Fecha de registro en sistema'}
                ],
                'primary_key': 'id',
                'relationships': [
                    {'table': 'datos_financieros', 'column': 'empresa_id'},
                    {'table': 'predicciones', 'column': 'empresa_id'},
                    {'table': 'alertas', 'column': 'empresa_id'}
                ]
            },
            'datos-financieros': {
                'columns': [
                    {'name': 'id', 'type': 'int64', 'description': 'ID único del registro'},
                    {'name': 'empresa_id', 'type': 'int64', 'description': 'ID de la empresa'},
                    {'name': 'rut', 'type': 'string', 'description': 'RUT de la empresa'},
                    {'name': 'razon_social', 'type': 'string', 'description': 'Razón social'},
                    {'name': 'sector', 'type': 'string', 'description': 'Sector económico'},
                    {'name': 'periodo', 'type': 'datetime', 'description': 'Período del reporte'},
                    {'name': 'tipo_periodo', 'type': 'string', 'description': 'Tipo (anual/trimestral/mensual)'},
                    {'name': 'activos_totales', 'type': 'float64', 'description': 'Activos totales'},
                    {'name': 'pasivos_totales', 'type': 'float64', 'description': 'Pasivos totales'},
                    {'name': 'patrimonio', 'type': 'float64', 'description': 'Patrimonio'},
                    {'name': 'ingresos_operacionales', 'type': 'float64', 'description': 'Ingresos operacionales'},
                    {'name': 'utilidad_neta', 'type': 'float64', 'description': 'Utilidad neta'},
                    {'name': 'ebitda', 'type': 'float64', 'description': 'EBITDA'},
                    {'name': 'flujo_caja_operacional', 'type': 'float64', 'description': 'Flujo de caja operacional'},
                    {'name': 'altman_z_score', 'type': 'float64', 'description': 'Altman Z-Score'},
                    {'name': 'altman_z_score_modificado', 'type': 'float64', 'description': 'Altman Z-Score modificado'},
                    {'name': 'ratio_liquidez_corriente', 'type': 'float64', 'description': 'Ratio de liquidez corriente'},
                    {'name': 'ratio_endeudamiento', 'type': 'float64', 'description': 'Ratio de endeudamiento'},
                    {'name': 'roa', 'type': 'float64', 'description': 'Return on Assets'},
                    {'name': 'roe', 'type': 'float64', 'description': 'Return on Equity'},
                    {'name': 'margen_operacional', 'type': 'float64', 'description': 'Margen operacional'},
                    {'name': 'rotacion_activos', 'type': 'float64', 'description': 'Rotación de activos'}
                ],
                'primary_key': 'id',
                'foreign_keys': [
                    {'column': 'empresa_id', 'references': 'empresas.id'}
                ]
            },
            'predicciones': {
                'columns': [
                    {'name': 'id', 'type': 'int64', 'description': 'ID único de la predicción'},
                    {'name': 'empresa_id', 'type': 'int64', 'description': 'ID de la empresa'},
                    {'name': 'rut', 'type': 'string', 'description': 'RUT de la empresa'},
                    {'name': 'razon_social', 'type': 'string', 'description': 'Razón social'},
                    {'name': 'sector', 'type': 'string', 'description': 'Sector económico'},
                    {'name': 'fecha_prediccion', 'type': 'datetime', 'description': 'Fecha de la predicción'},
                    {'name': 'probabilidad_ml', 'type': 'float64', 'description': 'Probabilidad ML'},
                    {'name': 'probabilidad_altman', 'type': 'float64', 'description': 'Probabilidad Altman'},
                    {'name': 'probabilidad_combinada', 'type': 'float64', 'description': 'Probabilidad combinada'},
                    {'name': 'banda_riesgo', 'type': 'string', 'description': 'Banda de riesgo (LOW/MEDIUM/HIGH/CRITICAL)'},
                    {'name': 'confianza_prediccion', 'type': 'float64', 'description': 'Confianza de la predicción'},
                    {'name': 'modelo_version', 'type': 'string', 'description': 'Versión del modelo'},
                    {'name': 'tiempo_procesamiento', 'type': 'float64', 'description': 'Tiempo de procesamiento (ms)'},
                    {'name': 'top_feature_1', 'type': 'string', 'description': 'Feature más importante'},
                    {'name': 'top_feature_2', 'type': 'string', 'description': 'Segunda feature más importante'},
                    {'name': 'top_feature_3', 'type': 'string', 'description': 'Tercera feature más importante'},
                    {'name': 'shap_value_1', 'type': 'float64', 'description': 'Valor SHAP del top feature 1'},
                    {'name': 'shap_value_2', 'type': 'float64', 'description': 'Valor SHAP del top feature 2'},
                    {'name': 'shap_value_3', 'type': 'float64', 'description': 'Valor SHAP del top feature 3'}
                ],
                'primary_key': 'id',
                'foreign_keys': [
                    {'column': 'empresa_id', 'references': 'empresas.id'}
                ]
            },
            'alertas': {
                'columns': [
                    {'name': 'id', 'type': 'int64', 'description': 'ID único de la alerta'},
                    {'name': 'empresa_id', 'type': 'int64', 'description': 'ID de la empresa'},
                    {'name': 'rut', 'type': 'string', 'description': 'RUT de la empresa'},
                    {'name': 'razon_social', 'type': 'string', 'description': 'Razón social'},
                    {'name': 'sector', 'type': 'string', 'description': 'Sector económico'},
                    {'name': 'tipo', 'type': 'string', 'description': 'Tipo de alerta'},
                    {'name': 'severidad', 'type': 'string', 'description': 'Severidad (LOW/MEDIUM/HIGH/CRITICAL)'},
                    {'name': 'estado', 'type': 'string', 'description': 'Estado (ACTIVA/EN_PROCESO/RESUELTA/DESCARTADA)'},
                    {'name': 'mensaje', 'type': 'string', 'description': 'Mensaje de la alerta'},
                    {'name': 'fecha_creacion', 'type': 'datetime', 'description': 'Fecha de creación'},
                    {'name': 'fecha_resolucion', 'type': 'datetime', 'description': 'Fecha de resolución'},
                    {'name': 'tiempo_resolucion_horas', 'type': 'float64', 'description': 'Tiempo de resolución en horas'},
                    {'name': 'usuario_asignado', 'type': 'string', 'description': 'Usuario asignado'},
                    {'name': 'prioridad', 'type': 'int64', 'description': 'Prioridad numérica'},
                    {'name': 'metadatos', 'type': 'string', 'description': 'Metadatos adicionales (JSON)'}
                ],
                'primary_key': 'id',
                'foreign_keys': [
                    {'column': 'empresa_id', 'references': 'empresas.id'}
                ]
            },
            'metricas-agregadas': {
                'columns': [
                    {'name': 'periodo', 'type': 'datetime', 'description': 'Período de agregación'},
                    {'name': 'nivel_agregacion', 'type': 'string', 'description': 'Nivel (diario/semanal/mensual/trimestral)'},
                    {'name': 'sector', 'type': 'string', 'description': 'Sector económico (opcional)'},
                    {'name': 'total_empresas', 'type': 'int64', 'description': 'Total de empresas'},
                    {'name': 'empresas_activas', 'type': 'int64', 'description': 'Empresas activas'},
                    {'name': 'total_predicciones', 'type': 'int64', 'description': 'Total de predicciones'},
                    {'name': 'predicciones_low', 'type': 'int64', 'description': 'Predicciones LOW'},
                    {'name': 'predicciones_medium', 'type': 'int64', 'description': 'Predicciones MEDIUM'},
                    {'name': 'predicciones_high', 'type': 'int64', 'description': 'Predicciones HIGH'},
                    {'name': 'predicciones_critical', 'type': 'int64', 'description': 'Predicciones CRITICAL'},
                    {'name': 'probabilidad_promedio', 'type': 'float64', 'description': 'Probabilidad promedio'},
                    {'name': 'total_alertas', 'type': 'int64', 'description': 'Total de alertas'},
                    {'name': 'alertas_criticas', 'type': 'int64', 'description': 'Alertas críticas'},
                    {'name': 'tiempo_procesamiento_promedio', 'type': 'float64', 'description': 'Tiempo promedio de procesamiento'},
                    {'name': 'tasa_exito_etl', 'type': 'float64', 'description': 'Tasa de éxito ETL'}
                ],
                'primary_key': ['periodo', 'nivel_agregacion', 'sector']
            }
        }
        
        if dataset_name not in schemas:
            return jsonify({'error': f'Dataset no encontrado: {dataset_name}'}), 404
        
        return jsonify({
            'dataset': dataset_name,
            'schema': schemas[dataset_name],
            'last_updated': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error obteniendo esquema: {str(e)}")
        return jsonify({'error': str(e)}), 500

@powerbi_bp.route('/datasets/refresh', methods=['POST'])
@require_auth
@require_permission('sistema', 'write')
def refresh_datasets():
    """
    Fuerza la actualización de todos los datasets
    
    Returns:
        Estado de la actualización
    """
    try:
        # Limpiar cache del conector
        powerbi_connector.clear_cache()
        
        # Obtener estadísticas de actualización
        refresh_stats = {
            'timestamp': datetime.now().isoformat(),
            'cache_cleared': True,
            'datasets_available': [
                'empresas',
                'datos-financieros', 
                'predicciones',
                'alertas',
                'metricas-agregadas'
            ]
        }
        
        logger.info("Datasets actualizados exitosamente")
        return jsonify({
            'success': True,
            'message': 'Datasets actualizados exitosamente',
            'stats': refresh_stats
        })
        
    except Exception as e:
        logger.error(f"Error actualizando datasets: {str(e)}")
        return jsonify({'error': str(e)}), 500

@powerbi_bp.route('/export/dashboard-config', methods=['GET'])
@require_auth
@require_permission('dashboard', 'read')
def export_dashboard_config():
    """
    Exporta configuración de dashboard para Power BI
    
    Returns:
        Archivo JSON con configuración de dashboard
    """
    try:
        from powerbi.dashboards.executive_dashboard import ExecutiveDashboardGenerator
        
        # Crear generador de dashboard
        generator = ExecutiveDashboardGenerator()
        
        # Generar configuración de dashboard ejecutivo
        config_file = generator.create_executive_overview()
        
        return send_file(
            config_file,
            as_attachment=True,
            download_name='executive_dashboard_config.json',
            mimetype='application/json'
        )
        
    except Exception as e:
        logger.error(f"Error exportando configuración: {str(e)}")
        return jsonify({'error': str(e)}), 500

@powerbi_bp.route('/export/data-model', methods=['GET'])
@require_auth
@require_permission('dashboard', 'read')
def export_data_model():
    """
    Exporta modelo de datos para Power BI
    
    Returns:
        Archivo JSON con modelo de datos completo
    """
    try:
        # Crear modelo de datos
        data_model = {
            'version': '1.0.0',
            'created': datetime.now().isoformat(),
            'tables': [
                {
                    'name': 'Empresas',
                    'source': '/api/powerbi/datasets/empresas',
                    'refresh_schedule': 'daily',
                    'key_column': 'id'
                },
                {
                    'name': 'DatosFinancieros',
                    'source': '/api/powerbi/datasets/datos-financieros',
                    'refresh_schedule': 'daily',
                    'key_column': 'id'
                },
                {
                    'name': 'Predicciones',
                    'source': '/api/powerbi/datasets/predicciones',
                    'refresh_schedule': 'hourly',
                    'key_column': 'id'
                },
                {
                    'name': 'Alertas',
                    'source': '/api/powerbi/datasets/alertas',
                    'refresh_schedule': 'hourly',
                    'key_column': 'id'
                },
                {
                    'name': 'MetricasAgregadas',
                    'source': '/api/powerbi/datasets/metricas-agregadas',
                    'refresh_schedule': 'daily',
                    'key_column': 'periodo'
                }
            ],
            'relationships': [
                {
                    'from_table': 'DatosFinancieros',
                    'from_column': 'empresa_id',
                    'to_table': 'Empresas',
                    'to_column': 'id',
                    'cardinality': 'many_to_one'
                },
                {
                    'from_table': 'Predicciones',
                    'from_column': 'empresa_id',
                    'to_table': 'Empresas',
                    'to_column': 'id',
                    'cardinality': 'many_to_one'
                },
                {
                    'from_table': 'Alertas',
                    'from_column': 'empresa_id',
                    'to_table': 'Empresas',
                    'to_column': 'id',
                    'cardinality': 'many_to_one'
                }
            ],
            'measures': [
                {
                    'name': 'Total Empresas',
                    'expression': 'COUNT(Empresas[id])',
                    'format': '#,##0'
                },
                {
                    'name': 'Empresas en Riesgo Alto',
                    'expression': 'CALCULATE(COUNT(Predicciones[id]), Predicciones[banda_riesgo] IN {"HIGH", "CRITICAL"})',
                    'format': '#,##0'
                },
                {
                    'name': 'Probabilidad Promedio',
                    'expression': 'AVERAGE(Predicciones[probabilidad_combinada])',
                    'format': '0.0%'
                },
                {
                    'name': 'Alertas Activas',
                    'expression': 'CALCULATE(COUNT(Alertas[id]), Alertas[estado] = "ACTIVA")',
                    'format': '#,##0'
                }
            ]
        }
        
        # Crear archivo temporal
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data_model, f, indent=2, ensure_ascii=False)
            temp_file = f.name
        
        return send_file(
            temp_file,
            as_attachment=True,
            download_name='data_model.json',
            mimetype='application/json'
        )
        
    except Exception as e:
        logger.error(f"Error exportando modelo de datos: {str(e)}")
        return jsonify({'error': str(e)}), 500

def _format_response(df: pd.DataFrame, format_type: str, filename_base: str):
    """
    Formatea la respuesta según el tipo solicitado
    
    Args:
        df: DataFrame con los datos
        format_type: Tipo de formato (json/csv/parquet)
        filename_base: Nombre base del archivo
        
    Returns:
        Respuesta formateada
    """
    if format_type == 'json':
        # Convertir a JSON con manejo de tipos especiales
        return jsonify({
            'data': df.to_dict('records'),
            'count': len(df),
            'columns': list(df.columns),
            'timestamp': datetime.now().isoformat()
        })
    
    elif format_type == 'csv':
        # Crear CSV en memoria
        output = io.StringIO()
        df.to_csv(output, index=False, encoding='utf-8')
        output.seek(0)
        
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'{filename_base}_{datetime.now().strftime("%Y%m%d")}.csv'
        )
    
    elif format_type == 'parquet':
        # Crear Parquet en memoria
        output = io.BytesIO()
        df.to_parquet(output, index=False, engine='pyarrow')
        output.seek(0)
        
        return send_file(
            output,
            mimetype='application/octet-stream',
            as_attachment=True,
            download_name=f'{filename_base}_{datetime.now().strftime("%Y%m%d")}.parquet'
        )
    
    else:
        raise ValueError(f'Formato no soportado: {format_type}')

# Registrar manejadores de errores
@powerbi_bp.errorhandler(400)
def bad_request(error):
    return jsonify({'error': 'Solicitud inválida', 'details': str(error)}), 400

@powerbi_bp.errorhandler(401)
def unauthorized(error):
    return jsonify({'error': 'No autorizado'}), 401

@powerbi_bp.errorhandler(403)
def forbidden(error):
    return jsonify({'error': 'Acceso denegado'}), 403

@powerbi_bp.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Recurso no encontrado'}), 404

@powerbi_bp.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Error interno del servidor'}), 500

