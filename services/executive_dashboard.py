"""
Este módulo genera automáticamente dashboards ejecutivos
optimizados para Power BI con métricas clave del negocio.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DashboardConfig:
    """Configuración para dashboards de Power BI"""
    dashboard_name: str
    refresh_schedule: str = "Daily"  # Daily, Weekly, Monthly
    auto_refresh_enabled: bool = True
    data_source_timeout: int = 30  # minutos
    cache_enabled: bool = True
    export_formats: List[str] = None
    
    def __post_init__(self):
        if self.export_formats is None:
            self.export_formats = ["PDF", "Excel", "PowerPoint"]

@dataclass
class VisualConfig:
    """Configuración para visualizaciones individuales"""
    visual_type: str
    title: str
    data_source: str
    x_axis: str = None
    y_axis: str = None
    legend: str = None
    filters: Dict[str, Any] = None
    color_scheme: str = "Corporate"
    show_data_labels: bool = True
    
    def __post_init__(self):
        if self.filters is None:
            self.filters = {}

class PowerBIDashboardGenerator:
    """
    Generador de dashboards ejecutivos para Power BI
    
    Crea automáticamente dashboards optimizados con:
    - Métricas KPI principales
    - Visualizaciones interactivas
    - Filtros dinámicos
    - Configuración de refresh automático
    """
    
    def __init__(self, output_dir: str = "powerbi/dashboards/generated"):
        """
        Inicializa el generador de dashboards
        
        Args:
            output_dir: Directorio de salida para archivos generados
        """
        self.output_dir = output_dir
        self.ensure_output_directory()
        
        # Esquemas de colores corporativos
        self.color_schemes = {
            "Corporate": {
                "primary": "#1f77b4",
                "secondary": "#ff7f0e", 
                "success": "#2ca02c",
                "warning": "#ffbb78",
                "danger": "#d62728",
                "info": "#17becf"
            },
            "Risk": {
                "low": "#2ca02c",
                "medium": "#ffbb78", 
                "high": "#ff7f0e",
                "critical": "#d62728"
            },
            "Financial": {
                "positive": "#2ca02c",
                "neutral": "#1f77b4",
                "negative": "#d62728",
                "warning": "#ff7f0e"
            }
        }
        
        logger.info(f"Dashboard Generator inicializado - Output: {self.output_dir}")
    
    def ensure_output_directory(self):
        """Asegura que el directorio de salida existe"""
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_executive_overview_dashboard(self, config: DashboardConfig = None) -> str:
        """
        Genera dashboard ejecutivo de overview general
        
        Args:
            config: Configuración del dashboard
            
        Returns:
            Ruta del archivo de configuración generado
        """
        if config is None:
            config = DashboardConfig(
                dashboard_name="Executive Overview - Bankruptcy Prediction",
                refresh_schedule="Daily"
            )
        
        # Definir visualizaciones del dashboard
        visuals = [
            # KPIs principales
            VisualConfig(
                visual_type="Card",
                title="Total Empresas Activas",
                data_source="empresas_dataset",
                filters={"activa": True}
            ),
            VisualConfig(
                visual_type="Card", 
                title="Predicciones Este Mes",
                data_source="predicciones_dataset",
                filters={"fecha_prediccion": "current_month"}
            ),
            VisualConfig(
                visual_type="Card",
                title="Alertas Críticas Activas", 
                data_source="alertas_dataset",
                filters={"severidad": ["CRITICAL"], "estado": "activa"}
            ),
            VisualConfig(
                visual_type="Card",
                title="% Empresas Riesgo Alto",
                data_source="metricas_agregadas_dataset"
            ),
            
            # Gráficos principales
            VisualConfig(
                visual_type="Donut",
                title="Distribución por Banda de Riesgo",
                data_source="predicciones_dataset",
                legend="banda_riesgo",
                color_scheme="Risk"
            ),
            VisualConfig(
                visual_type="Column",
                title="Predicciones por Sector",
                data_source="predicciones_dataset", 
                x_axis="sector",
                y_axis="count",
                color_scheme="Corporate"
            ),
            VisualConfig(
                visual_type="Line",
                title="Evolución Temporal del Riesgo",
                data_source="metricas_agregadas_dataset",
                x_axis="periodo",
                y_axis="porcentaje_riesgo_alto",
                color_scheme="Financial"
            ),
            VisualConfig(
                visual_type="Treemap",
                title="Empresas por Sector y Tamaño",
                data_source="empresas_dataset",
                legend="sector",
                filters={"activa": True}
            ),
            VisualConfig(
                visual_type="Table",
                title="Top 10 Empresas Mayor Riesgo",
                data_source="predicciones_dataset",
                filters={"top_n": 10, "order_by": "probabilidad_combinada_desc"}
            ),
            VisualConfig(
                visual_type="Gauge",
                title="Score Promedio del Sistema",
                data_source="metricas_agregadas_dataset"
            )
        ]
        
        # Generar configuración del dashboard
        dashboard_config = self._create_dashboard_config(config, visuals)
        
        # Guardar configuración
        output_file = os.path.join(self.output_dir, f"{config.dashboard_name.replace(' ', '_')}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dashboard_config, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Dashboard ejecutivo generado: {output_file}")
        return output_file
    
    def generate_risk_analysis_dashboard(self, config: DashboardConfig = None) -> str:
        """
        Genera dashboard especializado en análisis de riesgo
        
        Args:
            config: Configuración del dashboard
            
        Returns:
            Ruta del archivo de configuración generado
        """
        if config is None:
            config = DashboardConfig(
                dashboard_name="Risk Analysis Dashboard",
                refresh_schedule="Daily"
            )
        
        visuals = [
            # Métricas de riesgo
            VisualConfig(
                visual_type="Card",
                title="Probabilidad ML Promedio",
                data_source="predicciones_dataset"
            ),
            VisualConfig(
                visual_type="Card",
                title="Altman Z-Score Promedio",
                data_source="datos_financieros_dataset"
            ),
            VisualConfig(
                visual_type="Card",
                title="Empresas en Zona Gris",
                data_source="predicciones_dataset",
                filters={"banda_riesgo": ["MEDIUM"]}
            ),
            
            # Análisis de distribución
            VisualConfig(
                visual_type="Histogram",
                title="Distribución de Probabilidades ML",
                data_source="predicciones_dataset",
                x_axis="probabilidad_ml",
                color_scheme="Risk"
            ),
            VisualConfig(
                visual_type="Scatter",
                title="ML vs Altman Z-Score",
                data_source="predicciones_dataset",
                x_axis="probabilidad_ml",
                y_axis="altman_z_score",
                legend="banda_riesgo"
            ),
            VisualConfig(
                visual_type="Heatmap",
                title="Riesgo por Sector y Tamaño",
                data_source="metricas_agregadas_dataset",
                x_axis="sector",
                y_axis="tamaño",
                color_scheme="Risk"
            ),
            
            # Análisis temporal
            VisualConfig(
                visual_type="Line",
                title="Tendencia de Riesgo por Sector",
                data_source="metricas_agregadas_dataset",
                x_axis="periodo",
                y_axis="probabilidad_ml_promedio",
                legend="sector"
            ),
            VisualConfig(
                visual_type="Area",
                title="Evolución de Bandas de Riesgo",
                data_source="metricas_agregadas_dataset",
                x_axis="periodo",
                y_axis="count",
                legend="banda_riesgo",
                color_scheme="Risk"
            ),
            
            # Tablas detalladas
            VisualConfig(
                visual_type="Matrix",
                title="Matriz de Riesgo Sectorial",
                data_source="metricas_agregadas_dataset"
            ),
            VisualConfig(
                visual_type="Table",
                title="Alertas de Riesgo Recientes",
                data_source="alertas_dataset",
                filters={"fecha_creacion": "last_7_days", "severidad": ["HIGH", "CRITICAL"]}
            )
        ]
        
        dashboard_config = self._create_dashboard_config(config, visuals)
        
        output_file = os.path.join(self.output_dir, f"{config.dashboard_name.replace(' ', '_')}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dashboard_config, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Dashboard de análisis de riesgo generado: {output_file}")
        return output_file
    
    def generate_financial_dashboard(self, config: DashboardConfig = None) -> str:
        """
        Genera dashboard de análisis financiero
        
        Args:
            config: Configuración del dashboard
            
        Returns:
            Ruta del archivo de configuración generado
        """
        if config is None:
            config = DashboardConfig(
                dashboard_name="Financial Analysis Dashboard",
                refresh_schedule="Weekly"
            )
        
        visuals = [
            # KPIs financieros
            VisualConfig(
                visual_type="Card",
                title="Activos Totales Promedio",
                data_source="datos_financieros_dataset"
            ),
            VisualConfig(
                visual_type="Card",
                title="Ratio Endeudamiento Promedio",
                data_source="datos_financieros_dataset"
            ),
            VisualConfig(
                visual_type="Card",
                title="ROA Promedio",
                data_source="datos_financieros_dataset"
            ),
            
            # Análisis de ratios
            VisualConfig(
                visual_type="Column",
                title="Ratios Financieros por Sector",
                data_source="datos_financieros_dataset",
                x_axis="sector",
                y_axis="ratio_liquidez_corriente",
                color_scheme="Financial"
            ),
            VisualConfig(
                visual_type="Box",
                title="Distribución de Altman Z-Score",
                data_source="datos_financieros_dataset",
                x_axis="sector",
                y_axis="altman_z_score"
            ),
            VisualConfig(
                visual_type="Waterfall",
                title="Evolución de Ratios Clave",
                data_source="datos_financieros_dataset",
                x_axis="periodo",
                y_axis="ratio_rentabilidad_activos"
            ),
            
            # Análisis temporal financiero
            VisualConfig(
                visual_type="Line",
                title="Evolución de Ingresos por Sector",
                data_source="datos_financieros_dataset",
                x_axis="periodo",
                y_axis="ingresos_operacionales",
                legend="sector"
            ),
            VisualConfig(
                visual_type="Area",
                title="Composición de Activos",
                data_source="datos_financieros_dataset",
                x_axis="periodo",
                y_axis="activos_totales",
                legend="tipo_activo"
            ),
            
            # Correlaciones
            VisualConfig(
                visual_type="Scatter",
                title="ROA vs Ratio de Endeudamiento",
                data_source="datos_financieros_dataset",
                x_axis="ratio_rentabilidad_activos",
                y_axis="ratio_endeudamiento",
                legend="sector"
            ),
            VisualConfig(
                visual_type="Table",
                title="Top Empresas por Rentabilidad",
                data_source="datos_financieros_dataset",
                filters={"top_n": 15, "order_by": "ratio_rentabilidad_activos_desc"}
            )
        ]
        
        dashboard_config = self._create_dashboard_config(config, visuals)
        
        output_file = os.path.join(self.output_dir, f"{config.dashboard_name.replace(' ', '_')}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dashboard_config, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Dashboard financiero generado: {output_file}")
        return output_file
    
    def generate_operational_dashboard(self, config: DashboardConfig = None) -> str:
        """
        Genera dashboard operacional para monitoreo del sistema
        
        Args:
            config: Configuración del dashboard
            
        Returns:
            Ruta del archivo de configuración generado
        """
        if config is None:
            config = DashboardConfig(
                dashboard_name="Operational Monitoring Dashboard",
                refresh_schedule="Daily"
            )
        
        visuals = [
            # KPIs operacionales
            VisualConfig(
                visual_type="Card",
                title="Predicciones Hoy",
                data_source="predicciones_dataset",
                filters={"fecha_prediccion": "today"}
            ),
            VisualConfig(
                visual_type="Card",
                title="Tiempo Promedio Procesamiento",
                data_source="predicciones_dataset"
            ),
            VisualConfig(
                visual_type="Card",
                title="Tasa de Éxito ETL",
                data_source="etl_logs_dataset"
            ),
            VisualConfig(
                visual_type="Card",
                title="Alertas Pendientes",
                data_source="alertas_dataset",
                filters={"estado": "activa"}
            ),
            
            # Monitoreo de procesos
            VisualConfig(
                visual_type="Line",
                title="Volumen de Predicciones Diarias",
                data_source="predicciones_dataset",
                x_axis="fecha_prediccion",
                y_axis="count"
            ),
            VisualConfig(
                visual_type="Column",
                title="Procesos ETL por Estado",
                data_source="etl_logs_dataset",
                x_axis="estado",
                y_axis="count",
                color_scheme="Corporate"
            ),
            VisualConfig(
                visual_type="Area",
                title="Distribución de Alertas por Hora",
                data_source="alertas_dataset",
                x_axis="hora_creacion",
                y_axis="count",
                legend="severidad"
            ),
            
            # Performance del sistema
            VisualConfig(
                visual_type="Gauge",
                title="Confianza Promedio del Modelo",
                data_source="predicciones_dataset"
            ),
            VisualConfig(
                visual_type="Line",
                title="Tiempo de Respuesta API",
                data_source="system_metrics_dataset",
                x_axis="timestamp",
                y_axis="response_time_ms"
            ),
            VisualConfig(
                visual_type="Table",
                title="Errores Recientes del Sistema",
                data_source="system_logs_dataset",
                filters={"level": "ERROR", "timestamp": "last_24_hours"}
            )
        ]
        
        dashboard_config = self._create_dashboard_config(config, visuals)
        
        output_file = os.path.join(self.output_dir, f"{config.dashboard_name.replace(' ', '_')}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dashboard_config, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Dashboard operacional generado: {output_file}")
        return output_file
    
    def _create_dashboard_config(self, config: DashboardConfig, visuals: List[VisualConfig]) -> Dict[str, Any]:
        """
        Crea la configuración completa del dashboard
        
        Args:
            config: Configuración del dashboard
            visuals: Lista de visualizaciones
            
        Returns:
            Diccionario con configuración completa
        """
        return {
            "dashboard_metadata": {
                "name": config.dashboard_name,
                "created_date": datetime.now().isoformat(),
                "version": "1.0",
                "description": f"Dashboard generado automáticamente - {config.dashboard_name}"
            },
            "configuration": asdict(config),
            "data_sources": self._get_data_sources_config(),
            "visuals": [asdict(visual) for visual in visuals],
            "filters": self._get_global_filters(),
            "layout": self._get_layout_config(len(visuals)),
            "themes": self._get_theme_config(),
            "refresh_settings": {
                "schedule": config.refresh_schedule,
                "auto_refresh": config.auto_refresh_enabled,
                "timeout_minutes": config.data_source_timeout
            },
            "export_settings": {
                "enabled_formats": config.export_formats,
                "default_format": "PDF"
            }
        }
    
    def _get_data_sources_config(self) -> Dict[str, Any]:
        """Obtiene configuración de fuentes de datos"""
        return {
            "empresas_dataset": {
                "type": "SQL",
                "connection_string": "powerbi_connector.get_empresas_dataset",
                "refresh_method": "incremental",
                "cache_enabled": True
            },
            "datos_financieros_dataset": {
                "type": "SQL", 
                "connection_string": "powerbi_connector.get_datos_financieros_dataset",
                "refresh_method": "incremental",
                "cache_enabled": True
            },
            "predicciones_dataset": {
                "type": "SQL",
                "connection_string": "powerbi_connector.get_predicciones_dataset", 
                "refresh_method": "incremental",
                "cache_enabled": True
            },
            "alertas_dataset": {
                "type": "SQL",
                "connection_string": "powerbi_connector.get_alertas_dataset",
                "refresh_method": "full",
                "cache_enabled": False
            },
            "metricas_agregadas_dataset": {
                "type": "SQL",
                "connection_string": "powerbi_connector.get_metricas_agregadas_dataset",
                "refresh_method": "incremental", 
                "cache_enabled": True
            }
        }
    
    def _get_global_filters(self) -> List[Dict[str, Any]]:
        """Obtiene filtros globales del dashboard"""
        return [
            {
                "name": "Período",
                "type": "date_range",
                "default_value": "last_12_months",
                "applies_to": ["predicciones_dataset", "datos_financieros_dataset", "alertas_dataset"]
            },
            {
                "name": "Sector",
                "type": "multi_select",
                "source": "empresas_dataset.sector",
                "applies_to": ["empresas_dataset", "predicciones_dataset", "datos_financieros_dataset"]
            },
            {
                "name": "Tamaño Empresa",
                "type": "multi_select", 
                "source": "empresas_dataset.tamaño",
                "applies_to": ["empresas_dataset", "predicciones_dataset", "datos_financieros_dataset"]
            },
            {
                "name": "País",
                "type": "multi_select",
                "source": "empresas_dataset.pais",
                "applies_to": ["empresas_dataset", "predicciones_dataset", "datos_financieros_dataset"]
            }
        ]
    
    def _get_layout_config(self, num_visuals: int) -> Dict[str, Any]:
        """Obtiene configuración de layout del dashboard"""
        # Calcular layout automático basado en número de visualizaciones
        if num_visuals <= 4:
            layout = "2x2"
        elif num_visuals <= 6:
            layout = "3x2"
        elif num_visuals <= 9:
            layout = "3x3"
        else:
            layout = "4x3"
        
        return {
            "type": "grid",
            "grid_layout": layout,
            "responsive": True,
            "mobile_optimized": True,
            "padding": "medium",
            "background_color": "#ffffff"
        }
    
    def _get_theme_config(self) -> Dict[str, Any]:
        """Obtiene configuración de tema visual"""
        return {
            "name": "Corporate",
            "colors": self.color_schemes["Corporate"],
            "fonts": {
                "title": "Segoe UI Semibold",
                "subtitle": "Segoe UI",
                "body": "Segoe UI Light"
            },
            "chart_styles": {
                "border_width": 1,
                "border_color": "#e0e0e0",
                "grid_lines": True,
                "data_labels": True
            }
        }
    
    def generate_all_dashboards(self) -> List[str]:
        """
        Genera todos los dashboards predefinidos
        
        Returns:
            Lista de rutas de archivos generados
        """
        generated_files = []
        
        try:
            # Dashboard ejecutivo
            executive_file = self.generate_executive_overview_dashboard()
            generated_files.append(executive_file)
            
            # Dashboard de análisis de riesgo
            risk_file = self.generate_risk_analysis_dashboard()
            generated_files.append(risk_file)
            
            # Dashboard financiero
            financial_file = self.generate_financial_dashboard()
            generated_files.append(financial_file)
            
            # Dashboard operacional
            operational_file = self.generate_operational_dashboard()
            generated_files.append(operational_file)
            
            logger.info(f"Todos los dashboards generados exitosamente: {len(generated_files)} archivos")
            
        except Exception as e:
            logger.error(f"Error generando dashboards: {str(e)}")
            raise
        
        return generated_files
    
    def create_custom_dashboard(self, 
                              name: str,
                              visuals: List[VisualConfig],
                              config: DashboardConfig = None) -> str:
        """
        Crea un dashboard personalizado
        
        Args:
            name: Nombre del dashboard
            visuals: Lista de visualizaciones personalizadas
            config: Configuración opcional
            
        Returns:
            Ruta del archivo generado
        """
        if config is None:
            config = DashboardConfig(dashboard_name=name)
        else:
            config.dashboard_name = name
        
        dashboard_config = self._create_dashboard_config(config, visuals)
        
        output_file = os.path.join(self.output_dir, f"{name.replace(' ', '_')}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dashboard_config, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Dashboard personalizado generado: {output_file}")
        return output_file

# Funciones de utilidad
def create_dashboard_generator(output_dir: str = None) -> PowerBIDashboardGenerator:
    """
    Factory function para crear un generador de dashboards
    
    Args:
        output_dir: Directorio de salida opcional
        
    Returns:
        Instancia del generador configurada
    """
    return PowerBIDashboardGenerator(output_dir or "powerbi/dashboards/generated")

def generate_standard_dashboards(output_dir: str = None) -> List[str]:
    """
    Genera todos los dashboards estándar
    
    Args:
        output_dir: Directorio de salida opcional
        
    Returns:
        Lista de archivos generados
    """
    generator = create_dashboard_generator(output_dir)
    return generator.generate_all_dashboards()

