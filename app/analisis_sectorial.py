"""
Esta página proporciona análisis comparativo entre sectores económicos,
tendencias temporales y benchmarking de riesgo por industria.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from utils.api_client import api_client, handle_api_errors
from config.settings import RISK_COLORS, RISK_ICONS
from components.sidebar import show_page_header
from components.auth import has_permission, show_permission_error

@handle_api_errors
def get_sectorial_data():
    """Obtiene datos para análisis sectorial"""
    
    # Obtener datos de múltiples endpoints
    riesgo_sectorial = api_client.get_riesgo_sectorial()
    tendencias = api_client.get_tendencias_temporales(periodo='trimestre', limite=8)
    overview = api_client.get_dashboard_overview()
    
    return {
        'riesgo_sectorial': riesgo_sectorial.data if riesgo_sectorial.success else None,
        'tendencias': tendencias.data if tendencias.success else None,
        'overview': overview.data if overview.success else None
    }

def show_sectorial_overview(data: Dict[str, Any]):
    """Muestra overview del análisis sectorial"""
    
    riesgo_data = data.get('riesgo_sectorial', {})
    sectores = riesgo_data.get('sectores', [])
    
    if not sectores:
        st.error("No hay datos sectoriales disponibles")
        return
    
    st.markdown("### 📊 Resumen Sectorial")
    
    # Métricas generales
    total_sectores = len(sectores)
    total_empresas = sum(s.get('total_empresas', 0) for s in sectores)
    sector_mas_riesgo = max(sectores, key=lambda x: x.get('porcentaje_riesgo_alto', 0))
    sector_menos_riesgo = min(sectores, key=lambda x: x.get('porcentaje_riesgo_alto', 0))
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Sectores Analizados",
            total_sectores,
            delta=None
        )
    
    with col2:
        st.metric(
            "Total Empresas",
            f"{total_empresas:,}",
            delta=None
        )
    
    with col3:
        st.metric(
            "Mayor Riesgo",
            sector_mas_riesgo.get('sector', 'N/A'),
            delta=f"{sector_mas_riesgo.get('porcentaje_riesgo_alto', 0):.1f}%",
            delta_color="inverse"
        )
    
    with col4:
        st.metric(
            "Menor Riesgo",
            sector_menos_riesgo.get('sector', 'N/A'),
            delta=f"{sector_menos_riesgo.get('porcentaje_riesgo_alto', 0):.1f}%",
            delta_color="normal"
        )

def show_risk_by_sector_chart(sectores: List[Dict[str, Any]]):
    """Muestra gráfico de riesgo por sector"""
    
    if not sectores:
        st.warning("No hay datos de sectores disponibles")
        return
    
    st.markdown("### 🏭 Distribución de Riesgo por Sector")
    
    # Preparar datos
    df_sectores = pd.DataFrame(sectores)
    
    # Ordenar por porcentaje de riesgo alto
    df_sectores = df_sectores.sort_values('porcentaje_riesgo_alto', ascending=True)
    
    # Crear gráfico de barras horizontales
    fig = go.Figure()
    
    # Añadir barras para cada nivel de riesgo
    risk_levels = ['riesgo_bajo', 'riesgo_medio', 'riesgo_alto', 'riesgo_critico']
    risk_colors = ['#2ca02c', '#ffbb78', '#ff7f0e', '#d62728']
    risk_names = ['Bajo', 'Medio', 'Alto', 'Crítico']
    
    for i, (level, color, name) in enumerate(zip(risk_levels, risk_colors, risk_names)):
        if level in df_sectores.columns:
            fig.add_trace(go.Bar(
                name=f'Riesgo {name}',
                y=df_sectores['sector'],
                x=df_sectores[level],
                orientation='h',
                marker_color=color,
                hovertemplate=f'<b>%{{y}}</b><br>Riesgo {name}: %{{x}}<br><extra></extra>'
            ))
    
    fig.update_layout(
        title="Distribución de Empresas por Nivel de Riesgo y Sector",
        xaxis_title="Número de Empresas",
        yaxis_title="Sector",
        barmode='stack',
        height=max(400, len(sectores) * 40),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_sector_comparison_table(sectores: List[Dict[str, Any]]):
    """Muestra tabla comparativa de sectores"""
    
    if not sectores:
        return
    
    st.markdown("### 📋 Comparativa Detallada por Sector")
    
    # Preparar datos para la tabla
    tabla_sectores = []
    for sector in sectores:
        tabla_sectores.append({
            'Sector': sector['sector'],
            'Total Empresas': sector['total_empresas'],
            '% Riesgo Alto': f"{sector.get('porcentaje_riesgo_alto', 0):.1f}%",
            'Prob. ML Promedio': f"{sector.get('probabilidad_promedio', 0):.3f}",
            'Altman Promedio': f"{sector.get('altman_promedio', 0):.2f}",
            'Empresas Riesgo Alto': sector.get('riesgo_alto', 0),
            'Ranking Riesgo': 0  # Se calculará después
        })
    
    df_tabla = pd.DataFrame(tabla_sectores)
    
    # Calcular ranking de riesgo
    df_tabla['Ranking Riesgo'] = df_tabla['% Riesgo Alto'].str.rstrip('%').astype(float).rank(ascending=False, method='min').astype(int)
    
    # Ordenar por ranking
    df_tabla = df_tabla.sort_values('Ranking Riesgo')
    
    # Mostrar tabla con configuración especial
    st.dataframe(
        df_tabla,
        use_container_width=True,
        hide_index=True,
        column_config={
            "% Riesgo Alto": st.column_config.ProgressColumn(
                "% Riesgo Alto",
                help="Porcentaje de empresas en riesgo alto",
                min_value=0,
                max_value=100,
                format="%.1f%%"
            ),
            "Prob. ML Promedio": st.column_config.ProgressColumn(
                "Probabilidad ML Promedio",
                help="Probabilidad promedio de quiebra según ML",
                min_value=0,
                max_value=1,
                format="%.3f"
            ),
            "Ranking Riesgo": st.column_config.NumberColumn(
                "Ranking",
                help="Posición en ranking de riesgo (1 = mayor riesgo)",
                format="%d"
            )
        }
    )

def show_sector_heatmap(sectores: List[Dict[str, Any]]):
    """Muestra heatmap de métricas por sector"""
    
    if not sectores:
        return
    
    st.markdown("### 🔥 Mapa de Calor - Métricas por Sector")
    
    # Preparar datos para heatmap
    df_sectores = pd.DataFrame(sectores)
    
    # Seleccionar métricas numéricas
    metricas = ['total_empresas', 'porcentaje_riesgo_alto', 'probabilidad_promedio', 'altman_promedio']
    metricas_nombres = ['Total Empresas', '% Riesgo Alto', 'Prob. ML Promedio', 'Altman Promedio']
    
    # Crear matriz para heatmap
    matriz_datos = []
    for metrica in metricas:
        if metrica in df_sectores.columns:
            matriz_datos.append(df_sectores[metrica].values)
    
    if not matriz_datos:
        st.warning("No hay datos suficientes para el heatmap")
        return
    
    # Normalizar datos para mejor visualización
    matriz_normalizada = []
    for fila in matriz_datos:
        if fila.max() > 0:
            fila_norm = (fila - fila.min()) / (fila.max() - fila.min())
        else:
            fila_norm = fila
        matriz_normalizada.append(fila_norm)
    
    # Crear heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matriz_normalizada,
        x=df_sectores['sector'].values,
        y=metricas_nombres,
        colorscale='RdYlBu_r',
        hoverongaps=False,
        hovertemplate='<b>%{y}</b><br>Sector: %{x}<br>Valor normalizado: %{z:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Mapa de Calor - Métricas Normalizadas por Sector",
        xaxis_title="Sector",
        yaxis_title="Métrica",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_temporal_trends_by_sector(tendencias_data: Dict[str, Any]):
    """Muestra tendencias temporales por sector"""
    
    if not tendencias_data:
        st.warning("No hay datos de tendencias disponibles")
        return
    
    st.markdown("### 📈 Evolución Temporal por Sector")
    
    # Simular datos de tendencias por sector (en una implementación real vendría de la API)
    # Por ahora mostraremos tendencias generales
    tendencias = tendencias_data.get('tendencias', [])
    
    if not tendencias:
        st.info("No hay datos de tendencias temporales disponibles")
        return
    
    # Preparar datos
    df_tendencias = pd.DataFrame(tendencias)
    df_tendencias['periodo'] = pd.to_datetime(df_tendencias['periodo'])
    
    # Crear gráfico de líneas
    fig = go.Figure()
    
    # Línea de porcentaje de riesgo alto
    fig.add_trace(go.Scatter(
        x=df_tendencias['periodo'],
        y=df_tendencias['porcentaje_riesgo_alto'],
        mode='lines+markers',
        name='% Riesgo Alto General',
        line=dict(color='#d62728', width=3),
        marker=dict(size=8)
    ))
    
    # Línea de total de predicciones (escala secundaria)
    fig.add_trace(go.Scatter(
        x=df_tendencias['periodo'],
        y=df_tendencias['total_predicciones'],
        mode='lines+markers',
        name='Total Predicciones',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title="Evolución Temporal del Riesgo Empresarial",
        xaxis_title="Período",
        yaxis_title="Porcentaje de Riesgo Alto (%)",
        yaxis2=dict(
            title="Total Predicciones",
            overlaying='y',
            side='right'
        ),
        height=400,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_sector_benchmarking():
    """Muestra benchmarking sectorial"""
    
    st.markdown("### 🎯 Benchmarking Sectorial")
    
    # Selector de sector para benchmarking
    col1, col2 = st.columns([2, 1])
    
    with col1:
        sector_seleccionado = st.selectbox(
            "Seleccionar Sector para Benchmarking",
            ["Technology", "Manufacturing", "Services", "Finance", "Healthcare", "Retail"],
            index=0
        )
    
    with col2:
        if st.button("🔄 Actualizar Análisis"):
            st.rerun()
    
    # Cargar datos específicos del sector
    with st.spinner(f"Analizando sector {sector_seleccionado}..."):
        # En una implementación real, esto vendría de una API específica
        show_sector_specific_analysis(sector_seleccionado)

def show_sector_specific_analysis(sector: str):
    """Muestra análisis específico de un sector"""
    
    # Simular datos específicos del sector
    # En implementación real vendría de API
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"#### 📊 Métricas de {sector}")
        
        # Métricas simuladas
        st.metric("Empresas en el Sector", "1,234", delta="12 vs mes anterior")
        st.metric("% Riesgo Alto", "15.3%", delta="-2.1% vs promedio general", delta_color="normal")
        st.metric("Prob. ML Promedio", "0.087", delta="-0.023 vs promedio general", delta_color="normal")
        st.metric("Altman Promedio", "2.45", delta="+0.34 vs promedio general", delta_color="normal")
    
    with col2:
        st.markdown(f"#### 🏆 Ranking vs Otros Sectores")
        
        # Ranking simulado
        ranking_data = {
            'Métrica': ['Menor Riesgo', 'Mayor Estabilidad', 'Mejor Altman', 'Más Empresas'],
            'Posición': ['3/6', '2/6', '4/6', '1/6'],
            'Percentil': ['50%', '83%', '33%', '100%']
        }
        
        df_ranking = pd.DataFrame(ranking_data)
        st.dataframe(df_ranking, use_container_width=True, hide_index=True)
    
    # Distribución de empresas en el sector
    st.markdown(f"#### 📈 Distribución de Riesgo en {sector}")
    
    # Datos simulados de distribución
    distribucion_simulada = {
        'Banda de Riesgo': ['Bajo', 'Medio', 'Alto', 'Crítico'],
        'Cantidad': [450, 320, 180, 50],
        'Porcentaje': [45.0, 32.0, 18.0, 5.0]
    }
    
    df_dist = pd.DataFrame(distribucion_simulada)
    
    # Gráfico de barras
    fig = px.bar(
        df_dist,
        x='Banda de Riesgo',
        y='Cantidad',
        title=f"Distribución de Empresas por Riesgo - {sector}",
        color='Banda de Riesgo',
        color_discrete_map={
            'Bajo': '#2ca02c',
            'Medio': '#ffbb78',
            'Alto': '#ff7f0e',
            'Crítico': '#d62728'
        }
    )
    
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def show_sector_recommendations(sector: str):
    """Muestra recomendaciones específicas del sector"""
    
    st.markdown(f"#### 💡 Recomendaciones para {sector}")
    
    # Recomendaciones simuladas basadas en el sector
    recomendaciones = {
        'Technology': [
            "🔍 Monitorear empresas con alta dependencia de financiamiento externo",
            "📊 Evaluar impacto de cambios en tasas de interés",
            "🚀 Considerar volatilidad del mercado tech en evaluaciones",
            "💰 Revisar métricas de burn rate y runway financiero"
        ],
        'Manufacturing': [
            "🏭 Monitorear costos de materias primas y energía",
            "📦 Evaluar cadenas de suministro y dependencias",
            "🌍 Considerar impacto de regulaciones ambientales",
            "⚙️ Revisar niveles de automatización y eficiencia"
        ],
        'Services': [
            "👥 Evaluar dependencia de recursos humanos especializados",
            "📍 Considerar impacto de ubicación geográfica",
            "💼 Monitorear contratos a largo plazo y recurrencia",
            "🔄 Revisar capacidad de adaptación a cambios del mercado"
        ],
        'Finance': [
            "🏦 Monitorear exposición a riesgo crediticio",
            "📈 Evaluar impacto de cambios regulatorios",
            "💱 Considerar volatilidad de mercados financieros",
            "🔒 Revisar adecuación de capital y liquidez"
        ],
        'Healthcare': [
            "🏥 Evaluar dependencia de reembolsos gubernamentales",
            "💊 Monitorear cambios en regulaciones sanitarias",
            "👨‍⚕️ Considerar escasez de personal especializado",
            "🔬 Revisar inversiones en tecnología médica"
        ],
        'Retail': [
            "🛒 Monitorear cambios en patrones de consumo",
            "🌐 Evaluar competencia del comercio electrónico",
            "📦 Considerar costos de inventario y logística",
            "🏪 Revisar ubicación y costos de locales físicos"
        ]
    }
    
    sector_recs = recomendaciones.get(sector, [
        "📊 Realizar análisis específico del sector",
        "🔍 Monitorear indicadores clave de la industria",
        "📈 Evaluar tendencias del mercado",
        "💡 Considerar factores específicos del negocio"
    ])
    
    for rec in sector_recs:
        st.markdown(f"- {rec}")

def show_export_options():
    """Muestra opciones de exportación de análisis"""
    
    st.markdown("### 📥 Exportar Análisis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Exportar a Excel", use_container_width=True):
            st.info("Funcionalidad de exportación en desarrollo")
    
    with col2:
        if st.button("📄 Generar Reporte PDF", use_container_width=True):
            st.info("Funcionalidad de reporte en desarrollo")
    
    with col3:
        if st.button("📧 Enviar por Email", use_container_width=True):
            st.info("Funcionalidad de email en desarrollo")

def show_page():
    """Función principal de la página de análisis sectorial"""
    
    # Header de la página
    show_page_header(
        "Análisis Sectorial",
        "Comparativas y tendencias por sector económico",
        "📈"
    )
    
    # Verificar permisos
    if not has_permission('predicciones', 'read'):
        show_permission_error('predicciones', 'read')
        return
    
    # Cargar datos
    with st.spinner("Cargando datos sectoriales..."):
        data = get_sectorial_data()
    
    if not data or not data.get('riesgo_sectorial'):
        st.error("No se pudieron cargar los datos sectoriales")
        return
    
    # Overview sectorial
    show_sectorial_overview(data)
    
    st.markdown("---")
    
    # Tabs principales
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Comparativa", "📈 Tendencias", "🎯 Benchmarking", "📥 Exportar"])
    
    with tab1:
        sectores = data['riesgo_sectorial'].get('sectores', [])
        
        # Gráfico de riesgo por sector
        show_risk_by_sector_chart(sectores)
        
        st.markdown("---")
        
        # Tabla comparativa
        show_sector_comparison_table(sectores)
        
        st.markdown("---")
        
        # Heatmap
        show_sector_heatmap(sectores)
    
    with tab2:
        # Tendencias temporales
        show_temporal_trends_by_sector(data.get('tendencias'))
        
        st.markdown("---")
        
        # Análisis de estacionalidad
        st.markdown("### 📅 Análisis de Estacionalidad")
        st.info("Funcionalidad de análisis de estacionalidad en desarrollo")
        
        # Placeholder para análisis futuro
        st.markdown("""
        **Próximas funcionalidades:**
        - Análisis de estacionalidad por sector
        - Correlaciones con indicadores macroeconómicos
        - Predicción de tendencias futuras
        - Alertas de cambios significativos
        """)
    
    with tab3:
        # Benchmarking sectorial
        show_sector_benchmarking()
        
        # Selector de sector
        sector_seleccionado = st.session_state.get('sector_benchmark', 'Technology')
        
        st.markdown("---")
        
        # Recomendaciones
        show_sector_recommendations(sector_seleccionado)
    
    with tab4:
        # Opciones de exportación
        show_export_options()
        
        st.markdown("---")
        
        # Configuración de reportes
        st.markdown("### ⚙️ Configuración de Reportes")
        
        col1, col2 = st.columns(2)
        
        with col1:
            incluir_graficos = st.checkbox("Incluir gráficos", value=True)
            incluir_tablas = st.checkbox("Incluir tablas detalladas", value=True)
        
        with col2:
            formato_fecha = st.selectbox("Formato de fecha", ["DD/MM/YYYY", "MM/DD/YYYY", "YYYY-MM-DD"])
            idioma_reporte = st.selectbox("Idioma", ["Español", "Inglés"])
        
        st.markdown("---")
        
        # Programar reportes
        st.markdown("### 📅 Programar Reportes Automáticos")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            frecuencia = st.selectbox("Frecuencia", ["Semanal", "Mensual", "Trimestral"])
        
        with col2:
            dia_envio = st.selectbox("Día de envío", ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"])
        
        with col3:
            email_destino = st.text_input("Email destino", placeholder="usuario@empresa.com")
        
        if st.button("📧 Programar Reporte Automático"):
            st.success("Reporte programado exitosamente (funcionalidad en desarrollo)")

if __name__ == "__main__":
    show_page()

