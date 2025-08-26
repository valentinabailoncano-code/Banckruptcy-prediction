"""
Este módulo define las rutas REST para la gestión de procesos ETL,
incluyendo ingesta de datos y monitoreo de procesos.
"""

import logging
import os
import tempfile
from flask import Blueprint, request, jsonify, send_file
from flask_jwt_extended import jwt_required, get_jwt_identity
from werkzeug.utils import secure_filename
from sqlalchemy import func, desc
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from database.connection import get_db_session
from database.models import LogETL, Empresa, DatoFinanciero
from services.etl.data_ingestion import (
    DataIngestionService, IngestionConfig,
    ingest_empresas_from_file, ingest_financial_data_from_file
)

# Configurar logging
logger = logging.getLogger(__name__)

# Crear blueprint
etl_bp = Blueprint('etl', __name__)

# Configuración de archivos
ALLOWED_EXTENSIONS = {'.csv', '.xlsx', '.xls', '.json', '.zip'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

def allowed_file(filename):
    """Verifica si el archivo tiene una extensión permitida"""
    return any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS)

@etl_bp.route('/upload-empresas', methods=['POST'])
@jwt_required()
def upload_empresas():
    """Carga archivo de empresas"""
    try:
        # Verificar que se envió un archivo
        if 'file' not in request.files:
            return jsonify({'error': 'No se proporcionó archivo'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No se seleccionó archivo'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': f'Tipo de archivo no permitido. Permitidos: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
        
        # Obtener configuración desde form data
        config_data = {
            'validate_data': request.form.get('validate_data', 'true').lower() == 'true',
            'skip_duplicates': request.form.get('skip_duplicates', 'true').lower() == 'true',
            'update_existing': request.form.get('update_existing', 'false').lower() == 'true',
            'continue_on_error': request.form.get('continue_on_error', 'true').lower() == 'true',
            'max_error_rate': float(request.form.get('max_error_rate', '0.1'))
        }
        
        config = IngestionConfig(**config_data)
        
        # Guardar archivo temporalmente
        filename = secure_filename(file.filename)
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, filename)
        
        try:
            file.save(temp_path)
            
            # Verificar tamaño
            if os.path.getsize(temp_path) > MAX_FILE_SIZE:
                return jsonify({'error': f'Archivo excede el tamaño máximo de {MAX_FILE_SIZE // (1024*1024)}MB'}), 400
            
            # Procesar archivo
            usuario = get_jwt_identity()
            logger.info(f"Iniciando carga de empresas por usuario {usuario}: {filename}")
            
            results = ingest_empresas_from_file(temp_path, config)
            
            # Verificar tasa de error
            if results['error_rate'] > config.max_error_rate:
                return jsonify({
                    'warning': f'Tasa de error ({results["error_rate"]:.2%}) excede el límite ({config.max_error_rate:.2%})',
                    'results': results
                }), 206  # Partial Content
            
            logger.info(f"Carga de empresas completada: {results['successful_records']} exitosos, {results['error_records']} errores")
            
            return jsonify({
                'message': 'Archivo de empresas procesado exitosamente',
                'results': results
            })
            
        finally:
            # Limpiar archivo temporal
            if os.path.exists(temp_path):
                os.remove(temp_path)
            os.rmdir(temp_dir)
            
    except Exception as e:
        logger.error(f"Error procesando archivo de empresas: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500

@etl_bp.route('/upload-datos-financieros', methods=['POST'])
@jwt_required()
def upload_datos_financieros():
    """Carga archivo de datos financieros"""
    try:
        # Verificar que se envió un archivo
        if 'file' not in request.files:
            return jsonify({'error': 'No se proporcionó archivo'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No se seleccionó archivo'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': f'Tipo de archivo no permitido. Permitidos: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
        
        # Obtener configuración desde form data
        config_data = {
            'validate_data': request.form.get('validate_data', 'true').lower() == 'true',
            'skip_duplicates': request.form.get('skip_duplicates', 'true').lower() == 'true',
            'update_existing': request.form.get('update_existing', 'false').lower() == 'true',
            'auto_calculate_ratios': request.form.get('auto_calculate_ratios', 'true').lower() == 'true',
            'continue_on_error': request.form.get('continue_on_error', 'true').lower() == 'true',
            'max_error_rate': float(request.form.get('max_error_rate', '0.1'))
        }
        
        config = IngestionConfig(**config_data)
        
        # Guardar archivo temporalmente
        filename = secure_filename(file.filename)
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, filename)
        
        try:
            file.save(temp_path)
            
            # Verificar tamaño
            if os.path.getsize(temp_path) > MAX_FILE_SIZE:
                return jsonify({'error': f'Archivo excede el tamaño máximo de {MAX_FILE_SIZE // (1024*1024)}MB'}), 400
            
            # Procesar archivo
            usuario = get_jwt_identity()
            logger.info(f"Iniciando carga de datos financieros por usuario {usuario}: {filename}")
            
            results = ingest_financial_data_from_file(temp_path, config)
            
            # Verificar tasa de error
            if results['error_rate'] > config.max_error_rate:
                return jsonify({
                    'warning': f'Tasa de error ({results["error_rate"]:.2%}) excede el límite ({config.max_error_rate:.2%})',
                    'results': results
                }), 206  # Partial Content
            
            logger.info(f"Carga de datos financieros completada: {results['successful_records']} exitosos, {results['error_records']} errores")
            
            return jsonify({
                'message': 'Archivo de datos financieros procesado exitosamente',
                'results': results
            })
            
        finally:
            # Limpiar archivo temporal
            if os.path.exists(temp_path):
                os.remove(temp_path)
            os.rmdir(temp_dir)
            
    except Exception as e:
        logger.error(f"Error procesando archivo de datos financieros: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500

@etl_bp.route('/logs', methods=['GET'])
@jwt_required()
def get_etl_logs():
    """Obtiene logs de procesos ETL"""
    try:
        # Parámetros de consulta
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 20, type=int), 100)
        
        # Filtros
        proceso = request.args.get('proceso')
        estado = request.args.get('estado')
        fecha_desde = request.args.get('fecha_desde')
        fecha_hasta = request.args.get('fecha_hasta')
        
        with get_db_session() as session:
            # Construir query
            query = session.query(LogETL)
            
            # Aplicar filtros
            if proceso:
                query = query.filter(LogETL.proceso == proceso)
            
            if estado:
                query = query.filter(LogETL.estado == estado)
            
            if fecha_desde:
                try:
                    fecha_desde_dt = datetime.fromisoformat(fecha_desde)
                    query = query.filter(LogETL.fecha_inicio >= fecha_desde_dt)
                except ValueError:
                    return jsonify({'error': 'Formato de fecha_desde inválido'}), 400
            
            if fecha_hasta:
                try:
                    fecha_hasta_dt = datetime.fromisoformat(fecha_hasta)
                    query = query.filter(LogETL.fecha_inicio <= fecha_hasta_dt)
                except ValueError:
                    return jsonify({'error': 'Formato de fecha_hasta inválido'}), 400
            
            # Ordenar por fecha más reciente
            query = query.order_by(desc(LogETL.fecha_inicio))
            
            # Paginación
            total = query.count()
            logs = query.offset((page - 1) * per_page).limit(per_page).all()
            
            # Serializar resultados
            logs_data = []
            for log in logs:
                log_dict = {
                    'id': log.id,
                    'proceso': log.proceso,
                    'subproceso': log.subproceso,
                    'fecha_inicio': log.fecha_inicio.isoformat(),
                    'fecha_fin': log.fecha_fin.isoformat() if log.fecha_fin else None,
                    'estado': log.estado,
                    'registros_procesados': log.registros_procesados,
                    'registros_exitosos': log.registros_exitosos,
                    'registros_error': log.registros_error,
                    'mensaje': log.mensaje,
                    'error_detalle': log.error_detalle,
                    'archivo_origen': log.archivo_origen,
                    'parametros': log.parametros,
                    'servidor': log.servidor,
                    'usuario': log.usuario,
                    'duracion_segundos': (
                        (log.fecha_fin - log.fecha_inicio).total_seconds() 
                        if log.fecha_fin else None
                    )
                }
                logs_data.append(log_dict)
            
            return jsonify({
                'logs': logs_data,
                'pagination': {
                    'page': page,
                    'per_page': per_page,
                    'total': total,
                    'pages': (total + per_page - 1) // per_page
                },
                'filters_applied': {
                    'proceso': proceso,
                    'estado': estado,
                    'fecha_desde': fecha_desde,
                    'fecha_hasta': fecha_hasta
                }
            })
            
    except Exception as e:
        logger.error(f"Error obteniendo logs ETL: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500

@etl_bp.route('/logs/<int:log_id>', methods=['GET'])
@jwt_required()
def get_etl_log(log_id: int):
    """Obtiene un log específico de ETL"""
    try:
        with get_db_session() as session:
            log = session.query(LogETL).filter_by(id=log_id).first()
            
            if not log:
                return jsonify({'error': 'Log no encontrado'}), 404
            
            log_data = {
                'id': log.id,
                'proceso': log.proceso,
                'subproceso': log.subproceso,
                'fecha_inicio': log.fecha_inicio.isoformat(),
                'fecha_fin': log.fecha_fin.isoformat() if log.fecha_fin else None,
                'estado': log.estado,
                'registros_procesados': log.registros_procesados,
                'registros_exitosos': log.registros_exitosos,
                'registros_error': log.registros_error,
                'mensaje': log.mensaje,
                'error_detalle': log.error_detalle,
                'archivo_origen': log.archivo_origen,
                'parametros': log.parametros,
                'servidor': log.servidor,
                'usuario': log.usuario,
                'duracion_segundos': (
                    (log.fecha_fin - log.fecha_inicio).total_seconds() 
                    if log.fecha_fin else None
                )
            }
            
            return jsonify(log_data)
            
    except Exception as e:
        logger.error(f"Error obteniendo log ETL {log_id}: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500

@etl_bp.route('/stats', methods=['GET'])
@jwt_required()
def get_etl_stats():
    """Obtiene estadísticas de procesos ETL"""
    try:
        with get_db_session() as session:
            # Estadísticas básicas
            total_procesos = session.query(LogETL).count()
            
            # Por estado
            por_estado = session.query(
                LogETL.estado,
                func.count(LogETL.id).label('count')
            ).group_by(LogETL.estado).all()
            
            # Por proceso
            por_proceso = session.query(
                LogETL.proceso,
                func.count(LogETL.id).label('count')
            ).group_by(LogETL.proceso).all()
            
            # Procesos recientes (últimos 7 días)
            fecha_limite = datetime.now() - timedelta(days=7)
            procesos_recientes = session.query(LogETL).filter(
                LogETL.fecha_inicio >= fecha_limite
            ).count()
            
            # Procesos con errores recientes
            errores_recientes = session.query(LogETL).filter(
                LogETL.fecha_inicio >= fecha_limite,
                LogETL.estado == 'ERROR'
            ).count()
            
            # Tiempo promedio de procesamiento
            tiempo_promedio = session.query(
                func.avg(
                    func.extract('epoch', LogETL.fecha_fin - LogETL.fecha_inicio)
                ).label('avg_duration')
            ).filter(
                LogETL.fecha_fin.isnot(None)
            ).scalar()
            
            # Registros procesados totales
            total_registros = session.query(
                func.sum(LogETL.registros_procesados).label('total')
            ).scalar()
            
            return jsonify({
                'resumen': {
                    'total_procesos': total_procesos,
                    'procesos_recientes': procesos_recientes,
                    'errores_recientes': errores_recientes,
                    'tiempo_promedio_segundos': float(tiempo_promedio) if tiempo_promedio else 0,
                    'total_registros_procesados': total_registros or 0
                },
                'distribucion': {
                    'por_estado': [
                        {'estado': estado, 'count': count} 
                        for estado, count in por_estado
                    ],
                    'por_proceso': [
                        {'proceso': proceso, 'count': count} 
                        for proceso, count in por_proceso
                    ]
                }
            })
            
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas ETL: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500

@etl_bp.route('/validate-file', methods=['POST'])
@jwt_required()
def validate_file():
    """Valida un archivo antes de procesarlo"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No se proporcionó archivo'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No se seleccionó archivo'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': f'Tipo de archivo no permitido'}), 400
        
        # Guardar archivo temporalmente
        filename = secure_filename(file.filename)
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, filename)
        
        try:
            file.save(temp_path)
            
            # Verificar tamaño
            file_size = os.path.getsize(temp_path)
            if file_size > MAX_FILE_SIZE:
                return jsonify({
                    'valid': False,
                    'error': f'Archivo excede el tamaño máximo de {MAX_FILE_SIZE // (1024*1024)}MB'
                }), 400
            
            # Intentar leer el archivo para validar formato
            try:
                if filename.lower().endswith('.csv'):
                    df = pd.read_csv(temp_path, nrows=5)  # Solo leer primeras 5 filas
                elif filename.lower().endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(temp_path, nrows=5)
                elif filename.lower().endswith('.json'):
                    df = pd.read_json(temp_path, lines=True, nrows=5)
                else:
                    return jsonify({'valid': False, 'error': 'Formato no soportado'}), 400
                
                # Información del archivo
                info = {
                    'valid': True,
                    'filename': filename,
                    'size_bytes': file_size,
                    'size_mb': round(file_size / (1024 * 1024), 2),
                    'columns': list(df.columns),
                    'sample_data': df.head(3).to_dict('records'),
                    'estimated_rows': 'unknown'  # Sería costoso contar todas las filas
                }
                
                return jsonify(info)
                
            except Exception as e:
                return jsonify({
                    'valid': False,
                    'error': f'Error leyendo archivo: {str(e)}'
                }), 400
            
        finally:
            # Limpiar archivo temporal
            if os.path.exists(temp_path):
                os.remove(temp_path)
            os.rmdir(temp_dir)
            
    except Exception as e:
        logger.error(f"Error validando archivo: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500

@etl_bp.route('/template/empresas', methods=['GET'])
def download_empresas_template():
    """Descarga plantilla para carga de empresas"""
    try:
        # Crear DataFrame con columnas de ejemplo
        template_data = {
            'rut': ['12345678-9', '98765432-1'],
            'razon_social': ['Empresa Ejemplo S.A.', 'Otra Empresa Ltda.'],
            'nombre_fantasia': ['Ejemplo', 'Otra'],
            'sector': ['Technology', 'Manufacturing'],
            'subsector': ['Software', 'Textiles'],
            'tamaño': ['mediana', 'pequeña'],
            'pais': ['Chile', 'Chile'],
            'region': ['Metropolitana', 'Valparaíso'],
            'ciudad': ['Santiago', 'Valparaíso'],
            'fecha_constitucion': ['2020-01-15', '2018-06-30'],
            'numero_empleados': [50, 25],
            'es_publica': [False, False]
        }
        
        df = pd.DataFrame(template_data)
        
        # Crear archivo temporal
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
        df.to_excel(temp_file.name, index=False)
        temp_file.close()
        
        return send_file(
            temp_file.name,
            as_attachment=True,
            download_name='plantilla_empresas.xlsx',
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
    except Exception as e:
        logger.error(f"Error generando plantilla de empresas: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500

@etl_bp.route('/template/datos-financieros', methods=['GET'])
def download_datos_financieros_template():
    """Descarga plantilla para carga de datos financieros"""
    try:
        # Crear DataFrame con columnas de ejemplo
        template_data = {
            'rut': ['12345678-9', '12345678-9'],
            'fecha_corte': ['2023-12-31', '2023-09-30'],
            'total_activos': [1000000, 950000],
            'total_pasivos': [600000, 580000],
            'patrimonio': [400000, 370000],
            'activos_corrientes': [300000, 280000],
            'pasivos_corrientes': [200000, 190000],
            'ingresos_operacionales': [800000, 600000],
            'utilidad_neta': [50000, 40000],
            'flujo_efectivo_operacional': [60000, 45000],
            'tipo_dato': ['anual', 'trimestral']
        }
        
        df = pd.DataFrame(template_data)
        
        # Crear archivo temporal
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
        df.to_excel(temp_file.name, index=False)
        temp_file.close()
        
        return send_file(
            temp_file.name,
            as_attachment=True,
            download_name='plantilla_datos_financieros.xlsx',
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
    except Exception as e:
        logger.error(f"Error generando plantilla de datos financieros: {str(e)}")
        return jsonify({'error': 'Error interno del servidor'}), 500

