"""
Este módulo implementa un sistema completo de generación automática
de reportes ejecutivos, análisis sectoriales y reportes operacionales.
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template
import pdfkit
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import smtplib
import schedule
import time

from powerbi.connectors.powerbi_connector import PowerBIConnector
from database.connection import DatabaseManager
from services.models import Empresa, Prediccion, Alerta

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReportType(Enum):
    """Tipos de reportes disponibles"""
    EXECUTIVE_SUMMARY = "executive_summary"
    RISK_ANALYSIS = "risk_analysis"
    SECTOR_ANALYSIS = "sector_analysis"
    OPERATIONAL_REPORT = "operational_report"
    FINANCIAL_DASHBOARD = "financial_dashboard"
    ALERT_SUMMARY = "alert_summary"

class ReportFrequency(Enum):
    """Frecuencias de generación de reportes"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"

class ReportFormat(Enum):
    """Formatos de reporte disponibles"""
    PDF = "pdf"
    HTML = "html"
    EXCEL = "excel"
    JSON = "json"

@dataclass
class ReportConfig:
    """Configuración de reporte"""
    report_id: str
    name: str
    description: str
    report_type: ReportType
    frequency: ReportFrequency
    format: ReportFormat
    enabled: bool = True
    recipients: List[str] = None
    parameters: Dict[str, Any] = None
    template_path: str = None
    output_directory: str = "reports/generated"
    
    def __post_init__(self):
        if self.recipients is None:
            self.recipients = []
        if self.parameters is None:
            self.parameters = {}

@dataclass
class ReportData:
    """Datos para generación de reporte"""
    title: str
    subtitle: str
    generation_date: datetime
    period_start: datetime
    period_end: datetime
    data: Dict[str, Any]
    charts: List[str] = None
    
    def __post_init__(self):
        if self.charts is None:
            self.charts = []

class AutomatedReportGenerator:
    """
    Generador de reportes automáticos
    
    Funcionalidades:
    - Generación programada de reportes
    - Múltiples formatos de salida
    - Templates personalizables
    - Distribución automática por email
    - Análisis de datos integrado
    """
    
    def __init__(self, output_dir: str = "reports/generated"):
        """
        Inicializa el generador de reportes
        
        Args:
            output_dir: Directorio de salida para reportes
        """
        self.output_dir = output_dir
        self.powerbi_connector = PowerBIConnector()
        self.db_manager = DatabaseManager()
        
        # Configurar directorios
        self.ensure_directories()
        
        # Templates de reportes
        self.templates_dir = "services/reporting/templates"
        self.charts_dir = os.path.join(output_dir, "charts")
        
        # Configurar matplotlib para generación de gráficos
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Reportes configurados
        self.report_configs = self._initialize_default_reports()
        
        logger.info(f"Generador de reportes inicializado - Output: {self.output_dir}")
    
    def ensure_directories(self):
        """Asegura que los directorios necesarios existen"""
        directories = [
            self.output_dir,
            os.path.join(self.output_dir, "charts"),
            "services/reporting/templates"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def _initialize_default_reports(self) -> List[ReportConfig]:
        """Inicializa configuraciones de reportes predefinidos"""
        return [
            # Reporte ejecutivo semanal
            ReportConfig(
                report_id="weekly_executive",
                name="Reporte Ejecutivo Semanal",
                description="Resumen ejecutivo de métricas clave y alertas",
                report_type=ReportType.EXECUTIVE_SUMMARY,
                frequency=ReportFrequency.WEEKLY,
                format=ReportFormat.PDF,
                recipients=["ceo@empresa.com", "cfo@empresa.com"],
                parameters={"include_charts": True, "detail_level": "summary"}
            ),
            
            # Análisis de riesgo mensual
            ReportConfig(
                report_id="monthly_risk_analysis",
                name="Análisis de Riesgo Mensual",
                description="Análisis detallado de riesgos por sector y empresa",
                report_type=ReportType.RISK_ANALYSIS,
                frequency=ReportFrequency.MONTHLY,
                format=ReportFormat.PDF,
                recipients=["risk@empresa.com", "analytics@empresa.com"],
                parameters={"include_predictions": True, "top_n_companies": 20}
            ),
            
            # Reporte sectorial trimestral
            ReportConfig(
                report_id="quarterly_sector",
                name="Análisis Sectorial Trimestral",
                description="Análisis comparativo por sectores económicos",
                report_type=ReportType.SECTOR_ANALYSIS,
                frequency=ReportFrequency.QUARTERLY,
                format=ReportFormat.PDF,
                recipients=["strategy@empresa.com", "research@empresa.com"],
                parameters={"include_benchmarks": True, "trend_analysis": True}
            ),
            
            # Reporte operacional diario
            ReportConfig(
                report_id="daily_operations",
                name="Reporte Operacional Diario",
                description="Métricas operacionales y estado del sistema",
                report_type=ReportType.OPERATIONAL_REPORT,
                frequency=ReportFrequency.DAILY,
                format=ReportFormat.HTML,
                recipients=["ops@empresa.com", "tech@empresa.com"],
                parameters={"include_system_metrics": True, "alert_summary": True}
            ),
            
            # Resumen de alertas diario
            ReportConfig(
                report_id="daily_alerts",
                name="Resumen de Alertas Diario",
                description="Resumen de alertas generadas en las últimas 24 horas",
                report_type=ReportType.ALERT_SUMMARY,
                frequency=ReportFrequency.DAILY,
                format=ReportFormat.HTML,
                recipients=["alerts@empresa.com", "monitoring@empresa.com"],
                parameters={"severity_filter": ["HIGH", "CRITICAL"]}
            )
        ]
    
    async def generate_report(self, config: ReportConfig, custom_period: tuple = None) -> str:
        """
        Genera un reporte según la configuración especificada
        
        Args:
            config: Configuración del reporte
            custom_period: Período personalizado (start_date, end_date)
            
        Returns:
            Ruta del archivo generado
        """
        try:
            logger.info(f"Generando reporte: {config.name}")
            
            # Determinar período del reporte
            period_start, period_end = self._calculate_report_period(config.frequency, custom_period)
            
            # Recopilar datos
            report_data = await self._collect_report_data(config, period_start, period_end)
            
            # Generar gráficos si es necesario
            if config.parameters.get("include_charts", False):
                await self._generate_charts(config, report_data)
            
            # Generar reporte según formato
            output_file = await self._generate_report_file(config, report_data)
            
            logger.info(f"Reporte generado exitosamente: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error generando reporte {config.name}: {str(e)}")
            raise
    
    def _calculate_report_period(self, frequency: ReportFrequency, custom_period: tuple = None) -> tuple:
        """Calcula el período del reporte según la frecuencia"""
        if custom_period:
            return custom_period
        
        end_date = datetime.now()
        
        if frequency == ReportFrequency.DAILY:
            start_date = end_date - timedelta(days=1)
        elif frequency == ReportFrequency.WEEKLY:
            start_date = end_date - timedelta(weeks=1)
        elif frequency == ReportFrequency.MONTHLY:
            start_date = end_date - timedelta(days=30)
        elif frequency == ReportFrequency.QUARTERLY:
            start_date = end_date - timedelta(days=90)
        else:
            start_date = end_date - timedelta(days=7)  # Default a semanal
        
        return start_date, end_date
    
    async def _collect_report_data(self, config: ReportConfig, start_date: datetime, end_date: datetime) -> ReportData:
        """Recopila datos necesarios para el reporte"""
        
        if config.report_type == ReportType.EXECUTIVE_SUMMARY:
            return await self._collect_executive_data(config, start_date, end_date)
        elif config.report_type == ReportType.RISK_ANALYSIS:
            return await self._collect_risk_analysis_data(config, start_date, end_date)
        elif config.report_type == ReportType.SECTOR_ANALYSIS:
            return await self._collect_sector_analysis_data(config, start_date, end_date)
        elif config.report_type == ReportType.OPERATIONAL_REPORT:
            return await self._collect_operational_data(config, start_date, end_date)
        elif config.report_type == ReportType.ALERT_SUMMARY:
            return await self._collect_alert_summary_data(config, start_date, end_date)
        else:
            raise ValueError(f"Tipo de reporte no soportado: {config.report_type}")
    
    async def _collect_executive_data(self, config: ReportConfig, start_date: datetime, end_date: datetime) -> ReportData:
        """Recopila datos para reporte ejecutivo"""
        
        # Obtener datasets principales
        empresas_df = self.powerbi_connector.get_empresas_dataset()
        predicciones_df = self.powerbi_connector.get_predicciones_dataset(start_date, end_date)
        alertas_df = self.powerbi_connector.get_alertas_dataset(start_date, end_date)
        
        # Calcular métricas ejecutivas
        total_empresas = len(empresas_df)
        empresas_activas = len(empresas_df[empresas_df['activa'] == True])
        total_predicciones = len(predicciones_df)
        alertas_criticas = len(alertas_df[alertas_df['severidad'] == 'CRITICAL'])
        
        # Distribución de riesgo
        if not predicciones_df.empty:
            riesgo_distribution = predicciones_df['banda_riesgo'].value_counts().to_dict()
            probabilidad_promedio = predicciones_df['probabilidad_combinada'].mean()
        else:
            riesgo_distribution = {}
            probabilidad_promedio = 0
        
        # Top empresas de riesgo
        top_risk_companies = []
        if not predicciones_df.empty:
            top_risk = predicciones_df.nlargest(10, 'probabilidad_combinada')
            top_risk_companies = top_risk[['rut', 'razon_social', 'sector', 'probabilidad_combinada', 'banda_riesgo']].to_dict('records')
        
        data = {
            'kpis': {
                'total_empresas': total_empresas,
                'empresas_activas': empresas_activas,
                'total_predicciones': total_predicciones,
                'alertas_criticas': alertas_criticas,
                'probabilidad_promedio': probabilidad_promedio
            },
            'riesgo_distribution': riesgo_distribution,
            'top_risk_companies': top_risk_companies,
            'sector_summary': self._calculate_sector_summary(predicciones_df),
            'trend_analysis': self._calculate_trend_analysis(predicciones_df)
        }
        
        return ReportData(
            title="Reporte Ejecutivo",
            subtitle=f"Período: {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}",
            generation_date=datetime.now(),
            period_start=start_date,
            period_end=end_date,
            data=data
        )
    
    async def _collect_risk_analysis_data(self, config: ReportConfig, start_date: datetime, end_date: datetime) -> ReportData:
        """Recopila datos para análisis de riesgo"""
        
        predicciones_df = self.powerbi_connector.get_predicciones_dataset(start_date, end_date)
        datos_financieros_df = self.powerbi_connector.get_datos_financieros_dataset(start_date, end_date)
        
        # Análisis de distribución de riesgo
        risk_distribution = self._analyze_risk_distribution(predicciones_df)
        
        # Correlaciones financieras
        financial_correlations = self._calculate_financial_correlations(datos_financieros_df)
        
        # Análisis por sector
        sector_risk_analysis = self._analyze_sector_risk(predicciones_df)
        
        # Top empresas por riesgo
        top_n = config.parameters.get('top_n_companies', 20)
        top_risk_companies = predicciones_df.nlargest(top_n, 'probabilidad_combinada') if not predicciones_df.empty else pd.DataFrame()
        
        data = {
            'risk_distribution': risk_distribution,
            'financial_correlations': financial_correlations,
            'sector_risk_analysis': sector_risk_analysis,
            'top_risk_companies': top_risk_companies.to_dict('records') if not top_risk_companies.empty else [],
            'model_performance': self._analyze_model_performance(predicciones_df),
            'risk_trends': self._analyze_risk_trends(predicciones_df)
        }
        
        return ReportData(
            title="Análisis de Riesgo",
            subtitle=f"Análisis detallado - {start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')}",
            generation_date=datetime.now(),
            period_start=start_date,
            period_end=end_date,
            data=data
        )
    
    async def _collect_sector_analysis_data(self, config: ReportConfig, start_date: datetime, end_date: datetime) -> ReportData:
        """Recopila datos para análisis sectorial"""
        
        empresas_df = self.powerbi_connector.get_empresas_dataset()
        predicciones_df = self.powerbi_connector.get_predicciones_dataset(start_date, end_date)
        datos_financieros_df = self.powerbi_connector.get_datos_financieros_dataset(start_date, end_date)
        
        # Análisis por sector
        sector_metrics = self._calculate_sector_metrics(empresas_df, predicciones_df, datos_financieros_df)
        
        # Benchmarking sectorial
        sector_benchmarks = self._calculate_sector_benchmarks(datos_financieros_df)
        
        # Tendencias sectoriales
        sector_trends = self._analyze_sector_trends(predicciones_df)
        
        data = {
            'sector_metrics': sector_metrics,
            'sector_benchmarks': sector_benchmarks,
            'sector_trends': sector_trends,
            'sector_rankings': self._rank_sectors_by_risk(predicciones_df),
            'sector_recommendations': self._generate_sector_recommendations(sector_metrics)
        }
        
        return ReportData(
            title="Análisis Sectorial",
            subtitle=f"Comparativo por industrias - {start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')}",
            generation_date=datetime.now(),
            period_start=start_date,
            period_end=end_date,
            data=data
        )
    
    async def _collect_operational_data(self, config: ReportConfig, start_date: datetime, end_date: datetime) -> ReportData:
        """Recopila datos operacionales"""
        
        predicciones_df = self.powerbi_connector.get_predicciones_dataset(start_date, end_date)
        alertas_df = self.powerbi_connector.get_alertas_dataset(start_date, end_date)
        
        # Métricas operacionales
        operational_metrics = {
            'total_predictions': len(predicciones_df),
            'avg_processing_time': predicciones_df['tiempo_procesamiento'].mean() if not predicciones_df.empty else 0,
            'total_alerts': len(alertas_df),
            'critical_alerts': len(alertas_df[alertas_df['severidad'] == 'CRITICAL']),
            'system_uptime': 99.9,  # Placeholder - integrar con monitoreo real
            'data_quality_score': 0.95  # Placeholder - integrar con métricas reales
        }
        
        # Distribución temporal
        temporal_distribution = self._analyze_temporal_distribution(predicciones_df, alertas_df)
        
        # Performance del sistema
        system_performance = self._analyze_system_performance(predicciones_df)
        
        data = {
            'operational_metrics': operational_metrics,
            'temporal_distribution': temporal_distribution,
            'system_performance': system_performance,
            'alert_breakdown': alertas_df['severidad'].value_counts().to_dict() if not alertas_df.empty else {},
            'model_usage': predicciones_df['modelo_version'].value_counts().to_dict() if not predicciones_df.empty else {}
        }
        
        return ReportData(
            title="Reporte Operacional",
            subtitle=f"Métricas del sistema - {start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')}",
            generation_date=datetime.now(),
            period_start=start_date,
            period_end=end_date,
            data=data
        )
    
    async def _collect_alert_summary_data(self, config: ReportConfig, start_date: datetime, end_date: datetime) -> ReportData:
        """Recopila datos de resumen de alertas"""
        
        severity_filter = config.parameters.get('severity_filter', [])
        alertas_df = self.powerbi_connector.get_alertas_dataset(start_date, end_date, severity_filter)
        
        # Resumen de alertas
        alert_summary = {
            'total_alerts': len(alertas_df),
            'by_severity': alertas_df['severidad'].value_counts().to_dict() if not alertas_df.empty else {},
            'by_type': alertas_df['tipo'].value_counts().to_dict() if not alertas_df.empty else {},
            'by_status': alertas_df['estado'].value_counts().to_dict() if not alertas_df.empty else {},
            'resolution_time_avg': alertas_df['tiempo_resolucion_horas'].mean() if not alertas_df.empty else 0
        }
        
        # Alertas recientes
        recent_alerts = alertas_df.head(20).to_dict('records') if not alertas_df.empty else []
        
        # Empresas con más alertas
        companies_with_alerts = []
        if not alertas_df.empty:
            company_alert_counts = alertas_df.groupby(['empresa_id', 'rut', 'razon_social']).size().reset_index(name='alert_count')
            companies_with_alerts = company_alert_counts.nlargest(10, 'alert_count').to_dict('records')
        
        data = {
            'alert_summary': alert_summary,
            'recent_alerts': recent_alerts,
            'companies_with_alerts': companies_with_alerts,
            'hourly_distribution': self._analyze_hourly_alert_distribution(alertas_df),
            'sector_alert_analysis': self._analyze_sector_alerts(alertas_df)
        }
        
        return ReportData(
            title="Resumen de Alertas",
            subtitle=f"Alertas del período - {start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')}",
            generation_date=datetime.now(),
            period_start=start_date,
            period_end=end_date,
            data=data
        )
    
    async def _generate_charts(self, config: ReportConfig, report_data: ReportData):
        """Genera gráficos para el reporte"""
        
        chart_files = []
        
        try:
            if config.report_type == ReportType.EXECUTIVE_SUMMARY:
                chart_files.extend(await self._generate_executive_charts(report_data))
            elif config.report_type == ReportType.RISK_ANALYSIS:
                chart_files.extend(await self._generate_risk_charts(report_data))
            elif config.report_type == ReportType.SECTOR_ANALYSIS:
                chart_files.extend(await self._generate_sector_charts(report_data))
            
            report_data.charts = chart_files
            
        except Exception as e:
            logger.error(f"Error generando gráficos: {str(e)}")
    
    async def _generate_executive_charts(self, report_data: ReportData) -> List[str]:
        """Genera gráficos para reporte ejecutivo"""
        
        chart_files = []
        
        # Gráfico de distribución de riesgo
        if report_data.data.get('riesgo_distribution'):
            fig, ax = plt.subplots(figsize=(10, 6))
            
            distribution = report_data.data['riesgo_distribution']
            colors = {'LOW': '#2ca02c', 'MEDIUM': '#ffbb78', 'HIGH': '#ff7f0e', 'CRITICAL': '#d62728'}
            
            bars = ax.bar(distribution.keys(), distribution.values(), 
                         color=[colors.get(k, '#1f77b4') for k in distribution.keys()])
            
            ax.set_title('Distribución de Empresas por Banda de Riesgo', fontsize=14, fontweight='bold')
            ax.set_xlabel('Banda de Riesgo')
            ax.set_ylabel('Número de Empresas')
            
            # Agregar valores en las barras
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            chart_file = os.path.join(self.charts_dir, f"risk_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            chart_files.append(chart_file)
        
        return chart_files
    
    async def _generate_risk_charts(self, report_data: ReportData) -> List[str]:
        """Genera gráficos para análisis de riesgo"""
        
        chart_files = []
        
        # Implementar gráficos específicos de análisis de riesgo
        # Por ejemplo: correlaciones, distribuciones, tendencias
        
        return chart_files
    
    async def _generate_sector_charts(self, report_data: ReportData) -> List[str]:
        """Genera gráficos para análisis sectorial"""
        
        chart_files = []
        
        # Implementar gráficos específicos de análisis sectorial
        # Por ejemplo: comparativas, benchmarks, rankings
        
        return chart_files
    
    async def _generate_report_file(self, config: ReportConfig, report_data: ReportData) -> str:
        """Genera el archivo de reporte según el formato especificado"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f"{config.report_id}_{timestamp}"
        
        if config.format == ReportFormat.PDF:
            return await self._generate_pdf_report(config, report_data, base_filename)
        elif config.format == ReportFormat.HTML:
            return await self._generate_html_report(config, report_data, base_filename)
        elif config.format == ReportFormat.EXCEL:
            return await self._generate_excel_report(config, report_data, base_filename)
        elif config.format == ReportFormat.JSON:
            return await self._generate_json_report(config, report_data, base_filename)
        else:
            raise ValueError(f"Formato no soportado: {config.format}")
    
    async def _generate_pdf_report(self, config: ReportConfig, report_data: ReportData, filename: str) -> str:
        """Genera reporte en formato PDF"""
        
        # Generar HTML primero
        html_content = await self._generate_html_content(config, report_data)
        
        # Convertir a PDF
        output_file = os.path.join(self.output_dir, f"{filename}.pdf")
        
        options = {
            'page-size': 'A4',
            'margin-top': '0.75in',
            'margin-right': '0.75in',
            'margin-bottom': '0.75in',
            'margin-left': '0.75in',
            'encoding': "UTF-8",
            'no-outline': None,
            'enable-local-file-access': None
        }
        
        try:
            pdfkit.from_string(html_content, output_file, options=options)
        except Exception as e:
            logger.error(f"Error generando PDF: {str(e)}")
            # Fallback: guardar como HTML
            output_file = os.path.join(self.output_dir, f"{filename}.html")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
        
        return output_file
    
    async def _generate_html_report(self, config: ReportConfig, report_data: ReportData, filename: str) -> str:
        """Genera reporte en formato HTML"""
        
        html_content = await self._generate_html_content(config, report_data)
        
        output_file = os.path.join(self.output_dir, f"{filename}.html")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_file
    
    async def _generate_html_content(self, config: ReportConfig, report_data: ReportData) -> str:
        """Genera contenido HTML para el reporte"""
        
        # Template HTML básico
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>{{ title }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { text-align: center; margin-bottom: 30px; }
                .kpi-section { display: flex; justify-content: space-around; margin: 20px 0; }
                .kpi-card { text-align: center; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
                .kpi-value { font-size: 24px; font-weight: bold; color: #1f77b4; }
                .kpi-label { font-size: 12px; color: #666; }
                .section { margin: 30px 0; }
                .section h2 { color: #333; border-bottom: 2px solid #1f77b4; padding-bottom: 5px; }
                table { width: 100%; border-collapse: collapse; margin: 15px 0; }
                th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #f2f2f2; }
                .chart { text-align: center; margin: 20px 0; }
                .footer { margin-top: 50px; text-align: center; font-size: 12px; color: #666; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{{ title }}</h1>
                <h3>{{ subtitle }}</h3>
                <p>Generado el: {{ generation_date.strftime('%Y-%m-%d %H:%M:%S') }}</p>
            </div>
            
            {% if data.kpis %}
            <div class="section">
                <h2>Métricas Principales</h2>
                <div class="kpi-section">
                    {% for key, value in data.kpis.items() %}
                    <div class="kpi-card">
                        <div class="kpi-value">{{ "{:,.0f}".format(value) if value is number else value }}</div>
                        <div class="kpi-label">{{ key.replace('_', ' ').title() }}</div>
                    </div>
                    {% endfor %}
                </div>
            </div>
            {% endif %}
            
            {% if data.top_risk_companies %}
            <div class="section">
                <h2>Empresas de Mayor Riesgo</h2>
                <table>
                    <thead>
                        <tr>
                            <th>RUT</th>
                            <th>Razón Social</th>
                            <th>Sector</th>
                            <th>Probabilidad</th>
                            <th>Banda de Riesgo</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for company in data.top_risk_companies[:10] %}
                        <tr>
                            <td>{{ company.rut }}</td>
                            <td>{{ company.razon_social }}</td>
                            <td>{{ company.sector }}</td>
                            <td>{{ "{:.1%}".format(company.probabilidad_combinada) }}</td>
                            <td>{{ company.banda_riesgo }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
            {% endif %}
            
            {% for chart in charts %}
            <div class="chart">
                <img src="{{ chart }}" alt="Gráfico" style="max-width: 100%; height: auto;">
            </div>
            {% endfor %}
            
            <div class="footer">
                <p>Sistema de Predicción de Quiebras Empresariales - Reporte Automático</p>
            </div>
        </body>
        </html>
        """
        
        template = Template(html_template)
        return template.render(
            title=report_data.title,
            subtitle=report_data.subtitle,
            generation_date=report_data.generation_date,
            data=report_data.data,
            charts=report_data.charts
        )
    
    async def _generate_excel_report(self, config: ReportConfig, report_data: ReportData, filename: str) -> str:
        """Genera reporte en formato Excel"""
        
        output_file = os.path.join(self.output_dir, f"{filename}.xlsx")
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Hoja de resumen
            summary_data = {
                'Métrica': [],
                'Valor': []
            }
            
            if 'kpis' in report_data.data:
                for key, value in report_data.data['kpis'].items():
                    summary_data['Métrica'].append(key.replace('_', ' ').title())
                    summary_data['Valor'].append(value)
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Resumen', index=False)
            
            # Hoja de empresas de riesgo
            if 'top_risk_companies' in report_data.data and report_data.data['top_risk_companies']:
                risk_df = pd.DataFrame(report_data.data['top_risk_companies'])
                risk_df.to_excel(writer, sheet_name='Empresas de Riesgo', index=False)
        
        return output_file
    
    async def _generate_json_report(self, config: ReportConfig, report_data: ReportData, filename: str) -> str:
        """Genera reporte en formato JSON"""
        
        output_file = os.path.join(self.output_dir, f"{filename}.json")
        
        report_dict = asdict(report_data)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False, default=str)
        
        return output_file
    
    # Métodos de análisis de datos (implementaciones simplificadas)
    def _calculate_sector_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calcula resumen por sector"""
        if df.empty:
            return {}
        
        return df.groupby('sector').agg({
            'probabilidad_combinada': ['mean', 'count'],
            'banda_riesgo': lambda x: (x == 'HIGH').sum() + (x == 'CRITICAL').sum()
        }).to_dict()
    
    def _calculate_trend_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calcula análisis de tendencias"""
        if df.empty:
            return {}
        
        # Análisis temporal simple
        df['fecha'] = pd.to_datetime(df['fecha_prediccion'])
        daily_avg = df.groupby(df['fecha'].dt.date)['probabilidad_combinada'].mean()
        
        return {
            'daily_averages': daily_avg.to_dict(),
            'trend_direction': 'up' if daily_avg.iloc[-1] > daily_avg.iloc[0] else 'down'
        }
    
    def _analyze_risk_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analiza distribución de riesgo"""
        if df.empty:
            return {}
        
        return {
            'by_band': df['banda_riesgo'].value_counts().to_dict(),
            'probability_stats': df['probabilidad_combinada'].describe().to_dict()
        }
    
    def _calculate_financial_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calcula correlaciones financieras"""
        if df.empty:
            return {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlations = df[numeric_cols].corr()
        
        return correlations.to_dict()
    
    def _analyze_sector_risk(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analiza riesgo por sector"""
        if df.empty:
            return {}
        
        return df.groupby('sector').agg({
            'probabilidad_combinada': ['mean', 'std', 'count'],
            'banda_riesgo': lambda x: (x.isin(['HIGH', 'CRITICAL'])).sum()
        }).to_dict()
    
    def _analyze_model_performance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analiza rendimiento del modelo"""
        if df.empty:
            return {}
        
        return {
            'avg_confidence': df['confianza_prediccion'].mean(),
            'avg_processing_time': df['tiempo_procesamiento'].mean(),
            'model_versions': df['modelo_version'].value_counts().to_dict()
        }
    
    def _analyze_risk_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analiza tendencias de riesgo"""
        if df.empty:
            return {}
        
        df['fecha'] = pd.to_datetime(df['fecha_prediccion'])
        monthly_trends = df.groupby(df['fecha'].dt.to_period('M'))['probabilidad_combinada'].mean()
        
        return {
            'monthly_averages': monthly_trends.to_dict(),
            'overall_trend': 'increasing' if monthly_trends.iloc[-1] > monthly_trends.iloc[0] else 'decreasing'
        }
    
    def _calculate_sector_metrics(self, empresas_df: pd.DataFrame, predicciones_df: pd.DataFrame, financieros_df: pd.DataFrame) -> Dict[str, Any]:
        """Calcula métricas por sector"""
        # Implementación simplificada
        return {}
    
    def _calculate_sector_benchmarks(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calcula benchmarks sectoriales"""
        # Implementación simplificada
        return {}
    
    def _analyze_sector_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analiza tendencias sectoriales"""
        # Implementación simplificada
        return {}
    
    def _rank_sectors_by_risk(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Rankea sectores por riesgo"""
        # Implementación simplificada
        return []
    
    def _generate_sector_recommendations(self, sector_metrics: Dict[str, Any]) -> List[str]:
        """Genera recomendaciones sectoriales"""
        # Implementación simplificada
        return []
    
    def _analyze_temporal_distribution(self, predicciones_df: pd.DataFrame, alertas_df: pd.DataFrame) -> Dict[str, Any]:
        """Analiza distribución temporal"""
        # Implementación simplificada
        return {}
    
    def _analyze_system_performance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analiza rendimiento del sistema"""
        # Implementación simplificada
        return {}
    
    def _analyze_hourly_alert_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analiza distribución horaria de alertas"""
        if df.empty:
            return {}
        
        df['hora'] = pd.to_datetime(df['fecha_creacion']).dt.hour
        return df['hora'].value_counts().sort_index().to_dict()
    
    def _analyze_sector_alerts(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analiza alertas por sector"""
        if df.empty:
            return {}
        
        return df['sector'].value_counts().to_dict()
    
    async def send_report_by_email(self, report_file: str, config: ReportConfig, smtp_config: Dict[str, str]):
        """Envía reporte por email"""
        
        try:
            msg = MIMEMultipart()
            msg['From'] = smtp_config['from_email']
            msg['To'] = ", ".join(config.recipients)
            msg['Subject'] = f"{config.name} - {datetime.now().strftime('%Y-%m-%d')}"
            
            # Cuerpo del mensaje
            body = f"""
            Estimado/a,
            
            Se adjunta el {config.name} correspondiente al período solicitado.
            
            Reporte: {config.name}
            Descripción: {config.description}
            Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            Saludos cordiales,
            Sistema de Predicción de Quiebras Empresariales
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Adjuntar archivo
            with open(report_file, "rb") as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
            
            encoders.encode_base64(part)
            part.add_header(
                'Content-Disposition',
                f'attachment; filename= {os.path.basename(report_file)}'
            )
            
            msg.attach(part)
            
            # Enviar email
            server = smtplib.SMTP(smtp_config['smtp_server'], smtp_config['smtp_port'])
            server.starttls()
            server.login(smtp_config['username'], smtp_config['password'])
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Reporte enviado por email: {report_file}")
            
        except Exception as e:
            logger.error(f"Error enviando reporte por email: {str(e)}")
    
    def schedule_reports(self):
        """Programa la generación automática de reportes"""
        
        for config in self.report_configs:
            if not config.enabled:
                continue
            
            if config.frequency == ReportFrequency.DAILY:
                schedule.every().day.at("08:00").do(self._run_scheduled_report, config)
            elif config.frequency == ReportFrequency.WEEKLY:
                schedule.every().monday.at("08:00").do(self._run_scheduled_report, config)
            elif config.frequency == ReportFrequency.MONTHLY:
                schedule.every().month.do(self._run_scheduled_report, config)
        
        logger.info(f"Programados {len([c for c in self.report_configs if c.enabled])} reportes automáticos")
    
    def _run_scheduled_report(self, config: ReportConfig):
        """Ejecuta un reporte programado"""
        try:
            asyncio.run(self.generate_report(config))
        except Exception as e:
            logger.error(f"Error en reporte programado {config.name}: {str(e)}")
    
    def run_scheduler(self):
        """Ejecuta el programador de reportes"""
        logger.info("Iniciando programador de reportes")
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # Verificar cada minuto

# Funciones de utilidad
def create_report_generator(output_dir: str = None) -> AutomatedReportGenerator:
    """
    Factory function para crear un generador de reportes
    
    Args:
        output_dir: Directorio de salida opcional
        
    Returns:
        Instancia del generador configurada
    """
    return AutomatedReportGenerator(output_dir or "reports/generated")

async def generate_all_reports(generator: AutomatedReportGenerator) -> List[str]:
    """
    Genera todos los reportes configurados
    
    Args:
        generator: Generador de reportes
        
    Returns:
        Lista de archivos generados
    """
    generated_files = []
    
    for config in generator.report_configs:
        if config.enabled:
            try:
                file_path = await generator.generate_report(config)
                generated_files.append(file_path)
            except Exception as e:
                logger.error(f"Error generando reporte {config.name}: {str(e)}")
    
    return generated_files

