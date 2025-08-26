"""
Este módulo proporciona conectores optimizados para integrar
los datos del sistema con Microsoft Power BI.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
import json
import logging
from dataclasses import dataclass
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from database.connection import DatabaseManager
from database.models import (
    Empresa, DatoFinanciero, Prediccion, Alerta,
    DatoMacroeconomico, LogETL
)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PowerBIConfig:
    """Configuración para conectores de Power BI"""
    refresh_interval_minutes: int = 15
    max_rows_per_query: int = 100000
    enable_incremental_refresh: bool = True
    cache_duration_minutes: int = 30
    compression_enabled: bool = True
    
class PowerBIConnector:
    """
    Conector principal para integración con Power BI
    
    Proporciona métodos optimizados para extraer datos del sistema
    en formatos compatibles con Power BI, incluyendo:
    - Datos empresariales consolidados
    - Métricas financieras agregadas
    - Resultados de predicciones ML
    - Alertas y notificaciones
    - Datos macroeconómicos
    """
    
    def __init__(self, config: PowerBIConfig = None):
        """
        Inicializa el conector de Power BI
        
        Args:
            config: Configuración del conector
        """
        self.config = config or PowerBIConfig()
        self.db_manager = DatabaseManager()
        self._cache = {}
        self._cache_timestamps = {}
        
        logger.info("PowerBI Connector inicializado")
    
    def _is_cache_valid(self, key: str) -> bool:
        """Verifica si el cache es válido para una clave"""
        if key not in self._cache_timestamps:
            return False
        
        cache_time = self._cache_timestamps[key]
        expiry_time = cache_time + timedelta(minutes=self.config.cache_duration_minutes)
        
        return datetime.now() < expiry_time
    
    def _get_cached_data(self, key: str) -> Optional[pd.DataFrame]:
        """Obtiene datos del cache si están disponibles y válidos"""
        if self._is_cache_valid(key):
            logger.info(f"Retornando datos cacheados para: {key}")
            return self._cache[key]
        return None
    
    def _cache_data(self, key: str, data: pd.DataFrame) -> None:
        """Almacena datos en cache"""
        self._cache[key] = data.copy()
        self._cache_timestamps[key] = datetime.now()
        logger.info(f"Datos cacheados para: {key}")
    
    def get_empresas_dataset(self, 
                           include_inactive: bool = False,
                           sector_filter: List[str] = None,
                           pais_filter: List[str] = None) -> pd.DataFrame:
        """
        Obtiene dataset de empresas para Power BI
        
        Args:
            include_inactive: Incluir empresas inactivas
            sector_filter: Filtrar por sectores específicos
            pais_filter: Filtrar por países específicos
            
        Returns:
            DataFrame con datos de empresas optimizado para Power BI
        """
        cache_key = f"empresas_{include_inactive}_{sector_filter}_{pais_filter}"
        
        # Verificar cache
        cached_data = self._get_cached_data(cache_key)
        if cached_data is not None:
            return cached_data
        
        try:
            with self.db_manager.get_session() as session:
                # Query base
                query = session.query(Empresa)
                
                # Aplicar filtros
                if not include_inactive:
                    query = query.filter(Empresa.activa == True)
                
                if sector_filter:
                    query = query.filter(Empresa.sector.in_(sector_filter))
                
                if pais_filter:
                    query = query.filter(Empresa.pais.in_(pais_filter))
                
                # Limitar resultados
                query = query.limit(self.config.max_rows_per_query)
                
                empresas = query.all()
                
                # Convertir a DataFrame
                data = []
                for empresa in empresas:
                    data.append({
                        'empresa_id': empresa.id,
                        'rut': empresa.rut,
                        'razon_social': empresa.razon_social,
                        'nombre_fantasia': empresa.nombre_fantasia,
                        'sector': empresa.sector,
                        'subsector': empresa.subsector,
                        'pais': empresa.pais,
                        'region': empresa.region,
                        'ciudad': empresa.ciudad,
                        'tamaño': empresa.tamaño,
                        'numero_empleados': empresa.numero_empleados,
                        'es_publica': empresa.es_publica,
                        'ticker_bolsa': empresa.ticker_bolsa,
                        'activa': empresa.activa,
                        'fecha_constitucion': empresa.fecha_constitucion,
                        'fecha_registro': empresa.fecha_registro,
                        'fecha_actualizacion': empresa.fecha_actualizacion
                    })
                
                df = pd.DataFrame(data)
                
                # Optimizaciones para Power BI
                df = self._optimize_dataframe_for_powerbi(df)
                
                # Cachear resultados
                self._cache_data(cache_key, df)
                
                logger.info(f"Dataset de empresas generado: {len(df)} registros")
                return df
                
        except Exception as e:
            logger.error(f"Error generando dataset de empresas: {str(e)}")
            raise
    
    def get_datos_financieros_dataset(self,
                                    fecha_desde: datetime = None,
                                    fecha_hasta: datetime = None,
                                    tipo_periodo: str = None,
                                    empresa_ids: List[int] = None) -> pd.DataFrame:
        """
        Obtiene dataset de datos financieros para Power BI
        
        Args:
            fecha_desde: Fecha inicio del período
            fecha_hasta: Fecha fin del período
            tipo_periodo: Tipo de período (anual, trimestral, mensual)
            empresa_ids: IDs específicos de empresas
            
        Returns:
            DataFrame con datos financieros optimizado para Power BI
        """
        # Configurar fechas por defecto
        if fecha_hasta is None:
            fecha_hasta = datetime.now()
        if fecha_desde is None:
            fecha_desde = fecha_hasta - timedelta(days=365*2)  # 2 años por defecto
        
        cache_key = f"financieros_{fecha_desde}_{fecha_hasta}_{tipo_periodo}_{empresa_ids}"
        
        # Verificar cache
        cached_data = self._get_cached_data(cache_key)
        if cached_data is not None:
            return cached_data
        
        try:
            with self.db_manager.get_session() as session:
                # Query con JOIN para incluir datos de empresa
                query = session.query(
                    DatoFinanciero,
                    Empresa.rut,
                    Empresa.razon_social,
                    Empresa.sector,
                    Empresa.tamaño
                ).join(Empresa, DatoFinanciero.empresa_id == Empresa.id)
                
                # Aplicar filtros
                query = query.filter(DatoFinanciero.periodo >= fecha_desde)
                query = query.filter(DatoFinanciero.periodo <= fecha_hasta)
                
                if tipo_periodo:
                    query = query.filter(DatoFinanciero.tipo_periodo == tipo_periodo)
                
                if empresa_ids:
                    query = query.filter(DatoFinanciero.empresa_id.in_(empresa_ids))
                
                # Ordenar por fecha
                query = query.order_by(DatoFinanciero.periodo.desc())
                
                # Limitar resultados
                query = query.limit(self.config.max_rows_per_query)
                
                resultados = query.all()
                
                # Convertir a DataFrame
                data = []
                for dato_financiero, rut, razon_social, sector, tamaño in resultados:
                    data.append({
                        'dato_financiero_id': dato_financiero.id,
                        'empresa_id': dato_financiero.empresa_id,
                        'rut': rut,
                        'razon_social': razon_social,
                        'sector': sector,
                        'tamaño': tamaño,
                        'periodo': dato_financiero.periodo,
                        'tipo_periodo': dato_financiero.tipo_periodo,
                        'año': dato_financiero.periodo.year,
                        'trimestre': (dato_financiero.periodo.month - 1) // 3 + 1,
                        'mes': dato_financiero.periodo.month,
                        
                        # Estados financieros
                        'ingresos_operacionales': dato_financiero.ingresos_operacionales,
                        'activos_totales': dato_financiero.activos_totales,
                        'activos_corrientes': dato_financiero.activos_corrientes,
                        'pasivos_totales': dato_financiero.pasivos_totales,
                        'pasivos_corrientes': dato_financiero.pasivos_corrientes,
                        'patrimonio': dato_financiero.patrimonio,
                        'inventarios': dato_financiero.inventarios,
                        'cuentas_por_cobrar': dato_financiero.cuentas_por_cobrar,
                        'efectivo': dato_financiero.efectivo,
                        'deuda_financiera': dato_financiero.deuda_financiera,
                        'utilidad_neta': dato_financiero.utilidad_neta,
                        'ebitda': dato_financiero.ebitda,
                        'ventas': dato_financiero.ventas,
                        
                        # Ratios financieros
                        'ratio_liquidez_corriente': dato_financiero.ratio_liquidez_corriente,
                        'ratio_liquidez_acida': dato_financiero.ratio_liquidez_acida,
                        'ratio_endeudamiento': dato_financiero.ratio_endeudamiento,
                        'ratio_cobertura_intereses': dato_financiero.ratio_cobertura_intereses,
                        'ratio_rentabilidad_activos': dato_financiero.ratio_rentabilidad_activos,
                        'ratio_rentabilidad_patrimonio': dato_financiero.ratio_rentabilidad_patrimonio,
                        'ratio_margen_operacional': dato_financiero.ratio_margen_operacional,
                        'ratio_rotacion_activos': dato_financiero.ratio_rotacion_activos,
                        
                        # Altman Z-Score
                        'altman_z_score': dato_financiero.altman_z_score,
                        'altman_z_score_modificado': dato_financiero.altman_z_score_modificado,
                        'altman_z_score_emergentes': dato_financiero.altman_z_score_emergentes,
                        
                        # Metadatos
                        'fecha_carga': dato_financiero.fecha_carga,
                        'fecha_actualizacion': dato_financiero.fecha_actualizacion
                    })
                
                df = pd.DataFrame(data)
                
                # Optimizaciones para Power BI
                df = self._optimize_dataframe_for_powerbi(df)
                
                # Cachear resultados
                self._cache_data(cache_key, df)
                
                logger.info(f"Dataset de datos financieros generado: {len(df)} registros")
                return df
                
        except Exception as e:
            logger.error(f"Error generando dataset de datos financieros: {str(e)}")
            raise
    
    def get_predicciones_dataset(self,
                               fecha_desde: datetime = None,
                               fecha_hasta: datetime = None,
                               banda_riesgo: List[str] = None,
                               modelo_version: str = None) -> pd.DataFrame:
        """
        Obtiene dataset de predicciones para Power BI
        
        Args:
            fecha_desde: Fecha inicio del período
            fecha_hasta: Fecha fin del período
            banda_riesgo: Filtrar por bandas de riesgo específicas
            modelo_version: Versión específica del modelo
            
        Returns:
            DataFrame con predicciones optimizado para Power BI
        """
        # Configurar fechas por defecto
        if fecha_hasta is None:
            fecha_hasta = datetime.now()
        if fecha_desde is None:
            fecha_desde = fecha_hasta - timedelta(days=90)  # 3 meses por defecto
        
        cache_key = f"predicciones_{fecha_desde}_{fecha_hasta}_{banda_riesgo}_{modelo_version}"
        
        # Verificar cache
        cached_data = self._get_cached_data(cache_key)
        if cached_data is not None:
            return cached_data
        
        try:
            with self.db_manager.get_session() as session:
                # Query con JOINs para incluir datos de empresa y financieros
                query = session.query(
                    Prediccion,
                    Empresa.rut,
                    Empresa.razon_social,
                    Empresa.sector,
                    Empresa.tamaño,
                    DatoFinanciero.periodo.label('periodo_financiero'),
                    DatoFinanciero.altman_z_score
                ).join(Empresa, Prediccion.empresa_id == Empresa.id)\
                 .join(DatoFinanciero, Prediccion.dato_financiero_id == DatoFinanciero.id)
                
                # Aplicar filtros
                query = query.filter(Prediccion.fecha_prediccion >= fecha_desde)
                query = query.filter(Prediccion.fecha_prediccion <= fecha_hasta)
                
                if banda_riesgo:
                    query = query.filter(Prediccion.banda_riesgo.in_(banda_riesgo))
                
                if modelo_version:
                    query = query.filter(Prediccion.modelo_version == modelo_version)
                
                # Ordenar por fecha
                query = query.order_by(Prediccion.fecha_prediccion.desc())
                
                # Limitar resultados
                query = query.limit(self.config.max_rows_per_query)
                
                resultados = query.all()
                
                # Convertir a DataFrame
                data = []
                for prediccion, rut, razon_social, sector, tamaño, periodo_financiero, altman_z_score in resultados:
                    data.append({
                        'prediccion_id': prediccion.id,
                        'empresa_id': prediccion.empresa_id,
                        'rut': rut,
                        'razon_social': razon_social,
                        'sector': sector,
                        'tamaño': tamaño,
                        
                        # Información de la predicción
                        'fecha_prediccion': prediccion.fecha_prediccion,
                        'año_prediccion': prediccion.fecha_prediccion.year,
                        'mes_prediccion': prediccion.fecha_prediccion.month,
                        'trimestre_prediccion': (prediccion.fecha_prediccion.month - 1) // 3 + 1,
                        
                        # Resultados ML
                        'probabilidad_ml': prediccion.probabilidad_ml,
                        'probabilidad_altman': prediccion.probabilidad_altman,
                        'probabilidad_combinada': prediccion.probabilidad_combinada,
                        'banda_riesgo': prediccion.banda_riesgo,
                        'confianza_prediccion': prediccion.confianza_prediccion,
                        
                        # Información del modelo
                        'modelo_version': prediccion.modelo_version,
                        'modelo_algoritmo': prediccion.modelo_algoritmo,
                        'tiempo_procesamiento': prediccion.tiempo_procesamiento,
                        
                        # Datos financieros relacionados
                        'periodo_financiero': periodo_financiero,
                        'altman_z_score': altman_z_score,
                        
                        # Explicabilidad (top 3 features)
                        'top_feature_1': self._extract_top_feature(prediccion.explicabilidad_shap, 0),
                        'top_feature_2': self._extract_top_feature(prediccion.explicabilidad_shap, 1),
                        'top_feature_3': self._extract_top_feature(prediccion.explicabilidad_shap, 2),
                        
                        # Metadatos
                        'usuario_solicitante': prediccion.usuario_solicitante,
                        'observaciones': prediccion.observaciones
                    })
                
                df = pd.DataFrame(data)
                
                # Optimizaciones para Power BI
                df = self._optimize_dataframe_for_powerbi(df)
                
                # Cachear resultados
                self._cache_data(cache_key, df)
                
                logger.info(f"Dataset de predicciones generado: {len(df)} registros")
                return df
                
        except Exception as e:
            logger.error(f"Error generando dataset de predicciones: {str(e)}")
            raise
    
    def get_alertas_dataset(self,
                          fecha_desde: datetime = None,
                          fecha_hasta: datetime = None,
                          severidad: List[str] = None,
                          estado: List[str] = None) -> pd.DataFrame:
        """
        Obtiene dataset de alertas para Power BI
        
        Args:
            fecha_desde: Fecha inicio del período
            fecha_hasta: Fecha fin del período
            severidad: Filtrar por severidades específicas
            estado: Filtrar por estados específicos
            
        Returns:
            DataFrame con alertas optimizado para Power BI
        """
        # Configurar fechas por defecto
        if fecha_hasta is None:
            fecha_hasta = datetime.now()
        if fecha_desde is None:
            fecha_desde = fecha_hasta - timedelta(days=30)  # 30 días por defecto
        
        cache_key = f"alertas_{fecha_desde}_{fecha_hasta}_{severidad}_{estado}"
        
        # Verificar cache
        cached_data = self._get_cached_data(cache_key)
        if cached_data is not None:
            return cached_data
        
        try:
            with self.db_manager.get_session() as session:
                # Query con JOIN para incluir datos de empresa
                query = session.query(
                    Alerta,
                    Empresa.rut,
                    Empresa.razon_social,
                    Empresa.sector
                ).join(Empresa, Alerta.empresa_id == Empresa.id)
                
                # Aplicar filtros
                query = query.filter(Alerta.fecha_creacion >= fecha_desde)
                query = query.filter(Alerta.fecha_creacion <= fecha_hasta)
                
                if severidad:
                    query = query.filter(Alerta.severidad.in_(severidad))
                
                if estado:
                    query = query.filter(Alerta.estado.in_(estado))
                
                # Ordenar por fecha
                query = query.order_by(Alerta.fecha_creacion.desc())
                
                # Limitar resultados
                query = query.limit(self.config.max_rows_per_query)
                
                resultados = query.all()
                
                # Convertir a DataFrame
                data = []
                for alerta, rut, razon_social, sector in resultados:
                    data.append({
                        'alerta_id': alerta.id,
                        'empresa_id': alerta.empresa_id,
                        'rut': rut,
                        'razon_social': razon_social,
                        'sector': sector,
                        
                        # Información de la alerta
                        'tipo': alerta.tipo,
                        'severidad': alerta.severidad,
                        'estado': alerta.estado,
                        'titulo': alerta.titulo,
                        'mensaje': alerta.mensaje,
                        
                        # Fechas
                        'fecha_creacion': alerta.fecha_creacion,
                        'fecha_actualizacion': alerta.fecha_actualizacion,
                        'fecha_resolucion': alerta.fecha_resolucion,
                        
                        # Dimensiones temporales
                        'año_creacion': alerta.fecha_creacion.year,
                        'mes_creacion': alerta.fecha_creacion.month,
                        'dia_semana': alerta.fecha_creacion.weekday(),
                        'hora_creacion': alerta.fecha_creacion.hour,
                        
                        # Métricas calculadas
                        'tiempo_resolucion_horas': self._calculate_resolution_time(alerta),
                        'es_critica': alerta.severidad in ['CRITICAL', 'HIGH'],
                        'esta_resuelta': alerta.estado == 'resuelta',
                        
                        # IDs relacionados
                        'prediccion_id': alerta.prediccion_id,
                        'dato_financiero_id': alerta.dato_financiero_id
                    })
                
                df = pd.DataFrame(data)
                
                # Optimizaciones para Power BI
                df = self._optimize_dataframe_for_powerbi(df)
                
                # Cachear resultados
                self._cache_data(cache_key, df)
                
                logger.info(f"Dataset de alertas generado: {len(df)} registros")
                return df
                
        except Exception as e:
            logger.error(f"Error generando dataset de alertas: {str(e)}")
            raise
    
    def get_metricas_agregadas_dataset(self,
                                     nivel_agregacion: str = 'mensual',
                                     fecha_desde: datetime = None,
                                     fecha_hasta: datetime = None) -> pd.DataFrame:
        """
        Obtiene dataset de métricas agregadas para Power BI
        
        Args:
            nivel_agregacion: Nivel de agregación (diario, semanal, mensual, trimestral)
            fecha_desde: Fecha inicio del período
            fecha_hasta: Fecha fin del período
            
        Returns:
            DataFrame con métricas agregadas optimizado para Power BI
        """
        # Configurar fechas por defecto
        if fecha_hasta is None:
            fecha_hasta = datetime.now()
        if fecha_desde is None:
            fecha_desde = fecha_hasta - timedelta(days=365)  # 1 año por defecto
        
        cache_key = f"metricas_agregadas_{nivel_agregacion}_{fecha_desde}_{fecha_hasta}"
        
        # Verificar cache
        cached_data = self._get_cached_data(cache_key)
        if cached_data is not None:
            return cached_data
        
        try:
            # Generar métricas agregadas usando SQL directo para mejor performance
            with self.db_manager.get_session() as session:
                
                # Configurar formato de fecha según nivel de agregación
                date_format_map = {
                    'diario': '%Y-%m-%d',
                    'semanal': '%Y-%u',
                    'mensual': '%Y-%m',
                    'trimestral': '%Y-Q%q'
                }
                
                date_format = date_format_map.get(nivel_agregacion, '%Y-%m')
                
                # Query SQL para métricas agregadas
                sql_query = text(f"""
                SELECT 
                    DATE_FORMAT(p.fecha_prediccion, '{date_format}') as periodo,
                    e.sector,
                    e.tamaño,
                    COUNT(*) as total_predicciones,
                    AVG(p.probabilidad_ml) as probabilidad_ml_promedio,
                    AVG(p.probabilidad_combinada) as probabilidad_combinada_promedio,
                    SUM(CASE WHEN p.banda_riesgo = 'CRITICAL' THEN 1 ELSE 0 END) as predicciones_criticas,
                    SUM(CASE WHEN p.banda_riesgo = 'HIGH' THEN 1 ELSE 0 END) as predicciones_altas,
                    SUM(CASE WHEN p.banda_riesgo = 'MEDIUM' THEN 1 ELSE 0 END) as predicciones_medias,
                    SUM(CASE WHEN p.banda_riesgo = 'LOW' THEN 1 ELSE 0 END) as predicciones_bajas,
                    AVG(df.altman_z_score) as altman_promedio,
                    COUNT(DISTINCT p.empresa_id) as empresas_unicas
                FROM predicciones p
                JOIN empresas e ON p.empresa_id = e.id
                JOIN datos_financieros df ON p.dato_financiero_id = df.id
                WHERE p.fecha_prediccion >= :fecha_desde 
                    AND p.fecha_prediccion <= :fecha_hasta
                GROUP BY periodo, e.sector, e.tamaño
                ORDER BY periodo DESC, e.sector, e.tamaño
                """)
                
                result = session.execute(sql_query, {
                    'fecha_desde': fecha_desde,
                    'fecha_hasta': fecha_hasta
                })
                
                # Convertir a DataFrame
                columns = [
                    'periodo', 'sector', 'tamaño', 'total_predicciones',
                    'probabilidad_ml_promedio', 'probabilidad_combinada_promedio',
                    'predicciones_criticas', 'predicciones_altas', 
                    'predicciones_medias', 'predicciones_bajas',
                    'altman_promedio', 'empresas_unicas'
                ]
                
                data = [dict(zip(columns, row)) for row in result.fetchall()]
                df = pd.DataFrame(data)
                
                # Calcular métricas adicionales
                if not df.empty:
                    df['porcentaje_riesgo_alto'] = (
                        (df['predicciones_criticas'] + df['predicciones_altas']) / 
                        df['total_predicciones'] * 100
                    )
                    
                    df['tasa_crecimiento_predicciones'] = df.groupby(['sector', 'tamaño'])['total_predicciones'].pct_change()
                
                # Optimizaciones para Power BI
                df = self._optimize_dataframe_for_powerbi(df)
                
                # Cachear resultados
                self._cache_data(cache_key, df)
                
                logger.info(f"Dataset de métricas agregadas generado: {len(df)} registros")
                return df
                
        except Exception as e:
            logger.error(f"Error generando dataset de métricas agregadas: {str(e)}")
            raise
    
    def _optimize_dataframe_for_powerbi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimiza un DataFrame para su uso en Power BI
        
        Args:
            df: DataFrame a optimizar
            
        Returns:
            DataFrame optimizado
        """
        if df.empty:
            return df
        
        # Convertir tipos de datos para mejor performance en Power BI
        for col in df.columns:
            if df[col].dtype == 'object':
                # Intentar convertir strings a categorías si hay pocos valores únicos
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.5:  # Menos del 50% de valores únicos
                    df[col] = df[col].astype('category')
            
            elif df[col].dtype == 'float64':
                # Reducir precisión de floats si es posible
                if df[col].max() < 2**31 and df[col].min() > -2**31:
                    df[col] = pd.to_numeric(df[col], downcast='float')
            
            elif df[col].dtype == 'int64':
                # Reducir tamaño de enteros si es posible
                df[col] = pd.to_numeric(df[col], downcast='integer')
        
        # Ordenar por columnas más importantes para mejor compresión
        if 'fecha_prediccion' in df.columns:
            df = df.sort_values('fecha_prediccion', ascending=False)
        elif 'fecha_creacion' in df.columns:
            df = df.sort_values('fecha_creacion', ascending=False)
        elif 'periodo' in df.columns:
            df = df.sort_values('periodo', ascending=False)
        
        return df
    
    def _extract_top_feature(self, explicabilidad_json: str, index: int) -> Optional[str]:
        """Extrae el top feature de la explicabilidad SHAP"""
        try:
            if not explicabilidad_json:
                return None
            
            explicabilidad = json.loads(explicabilidad_json)
            features = explicabilidad.get('top_features', [])
            
            if index < len(features):
                return features[index].get('feature', None)
            
            return None
        except:
            return None
    
    def _calculate_resolution_time(self, alerta: Alerta) -> Optional[float]:
        """Calcula el tiempo de resolución de una alerta en horas"""
        if alerta.fecha_resolucion and alerta.fecha_creacion:
            delta = alerta.fecha_resolucion - alerta.fecha_creacion
            return delta.total_seconds() / 3600  # Convertir a horas
        return None
    
    def clear_cache(self) -> None:
        """Limpia el cache del conector"""
        self._cache.clear()
        self._cache_timestamps.clear()
        logger.info("Cache del conector limpiado")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del cache"""
        return {
            'cached_datasets': len(self._cache),
            'cache_keys': list(self._cache.keys()),
            'total_memory_mb': sum(df.memory_usage(deep=True).sum() for df in self._cache.values()) / 1024 / 1024
        }

# Funciones de utilidad para Power BI
def create_powerbi_connector(config: PowerBIConfig = None) -> PowerBIConnector:
    """
    Factory function para crear un conector de Power BI
    
    Args:
        config: Configuración del conector
        
    Returns:
        Instancia del conector configurada
    """
    return PowerBIConnector(config)

def export_to_powerbi_format(df: pd.DataFrame, 
                            output_path: str,
                            format_type: str = 'parquet') -> str:
    """
    Exporta un DataFrame en formato optimizado para Power BI
    
    Args:
        df: DataFrame a exportar
        output_path: Ruta de salida
        format_type: Formato de exportación (parquet, csv, excel)
        
    Returns:
        Ruta del archivo generado
    """
    try:
        if format_type == 'parquet':
            df.to_parquet(output_path, compression='snappy', index=False)
        elif format_type == 'csv':
            df.to_csv(output_path, index=False, encoding='utf-8')
        elif format_type == 'excel':
            df.to_excel(output_path, index=False, engine='openpyxl')
        else:
            raise ValueError(f"Formato no soportado: {format_type}")
        
        logger.info(f"Dataset exportado a: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error exportando dataset: {str(e)}")
        raise

