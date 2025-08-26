"""
Esta página maneja el sistema de alertas, notificaciones y monitoreo
proactivo de riesgos empresariales en tiempo real.
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

def show_alerts_overview():
    """Muestra overview general de alertas"""
    
    response = api_client.get_alertas_resumen()
    
    if not response.success:
        st.error(f"Error cargando resumen de alertas: {response.error}")
        return
    
    data = response.data
    resumen = data.get('resumen', {})
    
    st.markdown("### 🚨 Estado General de Alertas")
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_activas = resumen.get('total_activas', 0)
        st.metric(
            "Alertas Activas",
            f"{total_activas:,}",
            delta=None
        )
    
    with col2:
        criticas = resumen.get('criticas', 0)
        st.metric(
            "Críticas",
            f"{criticas:,}",
            delta=f"{(criticas/total_activas*100):.1f}%" if total_activas > 0 else "0%",
            delta_color="inverse"
        )
    
    with col3:
        nuevas_hoy = resumen.get('nuevas_hoy', 0)
        st.metric(
            "Nuevas Hoy",
            f"{nuevas_hoy:,}",
            delta=None
        )
    
    with col4:
        resueltas_semana = resumen.get('resueltas_semana', 0)
        st.metric(
            "Resueltas (7 días)",
            f"{resueltas_semana:,}",
            delta=None,
            delta_color="normal"
        )
    
    # Distribución por severidad
    por_severidad = resumen.get('por_severidad', [])
    if por_severidad:
        show_severity_distribution(por_severidad)

def show_severity_distribution(por_severidad: List[Dict[str, Any]]):
    """Muestra distribución de alertas por severidad"""
    
    df_severidad = pd.DataFrame(por_severidad)
    
    # Mapear colores por severidad
    color_map = {
        'CRITICAL': '#d62728',
        'HIGH': '#ff7f0e',
        'MEDIUM': '#ffbb78',
        'LOW': '#2ca02c',
        'INFO': '#1f77b4'
    }
    
    colors = [color_map.get(sev['severidad'], '#666') for sev in por_severidad]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de dona
        fig = go.Figure(data=[go.Pie(
            labels=df_severidad['severidad'],
            values=df_severidad['count'],
            hole=0.4,
            marker_colors=colors,
            textinfo='label+percent+value',
            hovertemplate='<b>%{label}</b><br>Cantidad: %{value}<br>Porcentaje: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title="Distribución por Severidad",
            height=300,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Tabla de severidad
        st.markdown("**📊 Detalle por Severidad:**")
        
        for sev in por_severidad:
            severidad = sev['severidad']
            count = sev['count']
            color = color_map.get(severidad, '#666')
            
            # Emoji según severidad
            emoji_map = {
                'CRITICAL': '🔴',
                'HIGH': '🟠',
                'MEDIUM': '🟡',
                'LOW': '🟢',
                'INFO': 'ℹ️'
            }
            
            emoji = emoji_map.get(severidad, '📊')
            
            st.markdown(
                f"""
                <div style="padding: 0.5rem; margin: 0.25rem 0; background: {color}20; border-left: 4px solid {color}; border-radius: 5px;">
                    <strong>{emoji} {severidad}</strong>: {count:,} alertas
                </div>
                """,
                unsafe_allow_html=True
            )

def show_alerts_list():
    """Muestra lista de alertas con filtros"""
    
    st.markdown("### 📋 Lista de Alertas")
    
    # Filtros
    with st.expander("🔧 Filtros"):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            severidad_filter = st.selectbox(
                "Severidad",
                ["Todas", "CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"],
                index=0
            )
        
        with col2:
            estado_filter = st.selectbox(
                "Estado",
                ["Todas", "activa", "resuelta", "descartada"],
                index=0
            )
        
        with col3:
            tipo_filter = st.selectbox(
                "Tipo",
                ["Todos", "riesgo_alto", "deterioro_financiero", "anomalia_datos", "modelo_drift"],
                index=0
            )
        
        with col4:
            fecha_filter = st.selectbox(
                "Período",
                ["Todas", "Hoy", "Última semana", "Último mes"],
                index=0
            )
    
    # Paginación
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        page = st.number_input("Página", min_value=1, value=1)
    
    with col2:
        per_page = st.selectbox("Por página", [10, 20, 50], index=1)
    
    with col3:
        sort_by = st.selectbox("Ordenar por", ["fecha_creacion", "severidad", "empresa"])
    
    # Construir filtros para API
    filters = {}
    if severidad_filter != "Todas":
        filters['severidad'] = severidad_filter
    if estado_filter != "Todas":
        filters['estado'] = estado_filter
    if tipo_filter != "Todos":
        filters['tipo'] = tipo_filter
    
    # Mapear filtros de fecha
    if fecha_filter == "Hoy":
        filters['fecha_desde'] = datetime.now().date().isoformat()
    elif fecha_filter == "Última semana":
        filters['fecha_desde'] = (datetime.now() - timedelta(days=7)).date().isoformat()
    elif fecha_filter == "Último mes":
        filters['fecha_desde'] = (datetime.now() - timedelta(days=30)).date().isoformat()
    
    # Cargar alertas
    with st.spinner("Cargando alertas..."):
        response = api_client.get_alertas(
            page=page,
            per_page=per_page,
            sort_by=sort_by,
            **filters
        )
    
    if not response.success:
        st.error(f"Error cargando alertas: {response.error}")
        return
    
    data = response.data
    alertas = data.get('alertas', [])
    pagination = data.get('pagination', {})
    
    if not alertas:
        st.info("No se encontraron alertas con los filtros aplicados")
        return
    
    # Información de paginación
    st.info(f"Mostrando {len(alertas)} de {pagination.get('total', 0)} alertas")
    
    # Lista de alertas
    for alerta in alertas:
        show_alert_card(alerta)

def show_alert_card(alerta: Dict[str, Any]):
    """Muestra tarjeta individual de alerta"""
    
    severidad = alerta.get('severidad', 'INFO')
    estado = alerta.get('estado', 'activa')
    
    # Colores según severidad
    color_map = {
        'CRITICAL': '#ffebee',
        'HIGH': '#fff3e0',
        'MEDIUM': '#fffde7',
        'LOW': '#e8f5e8',
        'INFO': '#e3f2fd'
    }
    
    border_color_map = {
        'CRITICAL': '#d32f2f',
        'HIGH': '#f57c00',
        'MEDIUM': '#fbc02d',
        'LOW': '#388e3c',
        'INFO': '#1976d2'
    }
    
    bg_color = color_map.get(severidad, '#f5f5f5')
    border_color = border_color_map.get(severidad, '#666')
    
    # Emoji según severidad
    emoji_map = {
        'CRITICAL': '🔴',
        'HIGH': '🟠',
        'MEDIUM': '🟡',
        'LOW': '🟢',
        'INFO': 'ℹ️'
    }
    
    emoji = emoji_map.get(severidad, '📊')
    
    with st.container():
        st.markdown(
            f"""
            <div style="
                background: {bg_color}; 
                border-left: 5px solid {border_color}; 
                padding: 1rem; 
                margin: 0.5rem 0; 
                border-radius: 5px;
            ">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h4 style="margin: 0; color: {border_color};">
                        {emoji} {alerta.get('titulo', 'Sin título')}
                    </h4>
                    <span style="
                        background: {border_color}; 
                        color: white; 
                        padding: 0.25rem 0.5rem; 
                        border-radius: 15px; 
                        font-size: 0.8rem;
                    ">
                        {severidad}
                    </span>
                </div>
                <p style="margin: 0.5rem 0; color: #333;">
                    {alerta.get('mensaje', 'Sin mensaje')}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Información adicional en columnas
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            empresa = alerta.get('empresa', {})
            st.caption(f"**Empresa:** {empresa.get('razon_social', 'N/A')}")
        
        with col2:
            st.caption(f"**Tipo:** {alerta.get('tipo', 'N/A')}")
        
        with col3:
            fecha = alerta.get('fecha_creacion', '')[:10] if alerta.get('fecha_creacion') else 'N/A'
            st.caption(f"**Fecha:** {fecha}")
        
        with col4:
            # Botones de acción
            col_a, col_b = st.columns(2)
            
            with col_a:
                if st.button("👁️ Ver", key=f"view_alert_{alerta['id']}"):
                    st.session_state.selected_alert_id = alerta['id']
                    st.session_state.show_alert_detail = True
                    st.rerun()
            
            with col_b:
                if estado == 'activa' and has_permission('alertas', 'write'):
                    if st.button("✅ Marcar Leída", key=f"read_alert_{alerta['id']}"):
                        mark_alert_as_read(alerta['id'])
        
        st.markdown("---")

def mark_alert_as_read(alert_id: int):
    """Marca una alerta como leída"""
    
    with st.spinner("Marcando alerta como leída..."):
        response = api_client.marcar_alerta_leida(alert_id)
    
    if response.success:
        st.success("✅ Alerta marcada como leída")
        st.rerun()
    else:
        st.error(f"Error marcando alerta: {response.error}")

def show_alert_detail(alert_id: int):
    """Muestra detalles completos de una alerta"""
    
    with st.spinner("Cargando detalles de la alerta..."):
        response = api_client.get_alerta(alert_id)
    
    if not response.success:
        st.error(f"Error cargando alerta: {response.error}")
        return
    
    alerta = response.data
    
    # Header
    col1, col2 = st.columns([3, 1])
    
    with col1:
        severidad = alerta.get('severidad', 'INFO')
        emoji_map = {
            'CRITICAL': '🔴',
            'HIGH': '🟠',
            'MEDIUM': '🟡',
            'LOW': '🟢',
            'INFO': 'ℹ️'
        }
        emoji = emoji_map.get(severidad, '📊')
        
        st.markdown(f"# {emoji} {alerta.get('titulo', 'Sin título')}")
        st.markdown(f"**Severidad:** {severidad} | **Estado:** {alerta.get('estado', 'N/A')}")
    
    with col2:
        if st.button("← Volver"):
            st.session_state.show_alert_detail = False
            st.rerun()
    
    # Tabs con información detallada
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Detalles", "🏢 Empresa", "📊 Datos", "⚙️ Acciones"])
    
    with tab1:
        show_alert_details(alerta)
    
    with tab2:
        show_alert_company_info(alerta)
    
    with tab3:
        show_alert_data(alerta)
    
    with tab4:
        show_alert_actions(alerta)

def show_alert_details(alerta: Dict[str, Any]):
    """Muestra detalles de la alerta"""
    
    st.markdown("### 📋 Información de la Alerta")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Información Básica:**")
        st.write(f"**ID:** {alerta.get('id', 'N/A')}")
        st.write(f"**Tipo:** {alerta.get('tipo', 'N/A')}")
        st.write(f"**Severidad:** {alerta.get('severidad', 'N/A')}")
        st.write(f"**Estado:** {alerta.get('estado', 'N/A')}")
    
    with col2:
        st.markdown("**Fechas:**")
        st.write(f"**Creación:** {alerta.get('fecha_creacion', 'N/A')}")
        st.write(f"**Última actualización:** {alerta.get('fecha_actualizacion', 'N/A')}")
        if alerta.get('fecha_resolucion'):
            st.write(f"**Resolución:** {alerta['fecha_resolucion']}")
    
    # Mensaje completo
    st.markdown("### 💬 Mensaje")
    st.markdown(alerta.get('mensaje', 'Sin mensaje disponible'))
    
    # Detalles adicionales
    detalles = alerta.get('detalles', {})
    if detalles:
        st.markdown("### 🔍 Detalles Técnicos")
        
        for key, value in detalles.items():
            st.write(f"**{key.replace('_', ' ').title()}:** {value}")

def show_alert_company_info(alerta: Dict[str, Any]):
    """Muestra información de la empresa relacionada"""
    
    empresa = alerta.get('empresa', {})
    
    if not empresa:
        st.info("No hay información de empresa asociada")
        return
    
    st.markdown("### 🏢 Empresa Afectada")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Información Corporativa:**")
        st.write(f"**RUT:** {empresa.get('rut', 'N/A')}")
        st.write(f"**Razón Social:** {empresa.get('razon_social', 'N/A')}")
        st.write(f"**Sector:** {empresa.get('sector', 'N/A')}")
        st.write(f"**Tamaño:** {empresa.get('tamaño', 'N/A')}")
    
    with col2:
        st.markdown("**Ubicación:**")
        st.write(f"**País:** {empresa.get('pais', 'N/A')}")
        st.write(f"**Región:** {empresa.get('region', 'N/A')}")
        st.write(f"**Ciudad:** {empresa.get('ciudad', 'N/A')}")
    
    # Botón para ver empresa completa
    if st.button("🔍 Ver Empresa Completa"):
        st.session_state.selected_empresa_id = empresa.get('id')
        st.session_state.show_empresa_detail = True
        st.session_state.show_alert_detail = False
        st.rerun()

def show_alert_data(alerta: Dict[str, Any]):
    """Muestra datos relacionados con la alerta"""
    
    st.markdown("### 📊 Datos de la Alerta")
    
    # Datos de predicción si están disponibles
    prediccion = alerta.get('prediccion', {})
    if prediccion:
        st.markdown("#### 🔮 Predicción Relacionada")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Probabilidad ML",
                f"{prediccion.get('probabilidad_ml', 0):.3f}"
            )
        
        with col2:
            st.metric(
                "Altman Z-Score",
                f"{prediccion.get('altman_z_score', 0):.2f}"
            )
        
        with col3:
            banda = prediccion.get('banda_riesgo', 'LOW')
            icon = RISK_ICONS.get(banda, '📊')
            st.metric(
                "Banda Riesgo",
                f"{icon} {banda.title()}"
            )
    
    # Datos financieros si están disponibles
    datos_financieros = alerta.get('datos_financieros', {})
    if datos_financieros:
        st.markdown("#### 💰 Datos Financieros")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Período:** {datos_financieros.get('periodo', 'N/A')}")
            st.write(f"**Tipo:** {datos_financieros.get('tipo_periodo', 'N/A')}")
        
        with col2:
            st.write(f"**Ingresos:** ${datos_financieros.get('ingresos_operacionales', 0):,.0f}")
            st.write(f"**Activos:** ${datos_financieros.get('activos_totales', 0):,.0f}")
    
    # Metadatos adicionales
    metadatos = alerta.get('metadatos', {})
    if metadatos:
        st.markdown("#### 🔧 Metadatos")
        
        for key, value in metadatos.items():
            st.write(f"**{key.replace('_', ' ').title()}:** {value}")

def show_alert_actions(alerta: Dict[str, Any]):
    """Muestra acciones disponibles para la alerta"""
    
    st.markdown("### ⚙️ Acciones Disponibles")
    
    estado = alerta.get('estado', 'activa')
    
    if estado == 'activa':
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("✅ Marcar como Resuelta", use_container_width=True):
                if has_permission('alertas', 'write'):
                    # Aquí iría la lógica para marcar como resuelta
                    st.success("Funcionalidad en desarrollo")
                else:
                    show_permission_error('alertas', 'write')
        
        with col2:
            if st.button("❌ Descartar Alerta", use_container_width=True):
                if has_permission('alertas', 'write'):
                    # Aquí iría la lógica para descartar
                    st.success("Funcionalidad en desarrollo")
                else:
                    show_permission_error('alertas', 'write')
        
        with col3:
            if st.button("🔄 Actualizar Estado", use_container_width=True):
                if has_permission('alertas', 'write'):
                    # Aquí iría la lógica para actualizar
                    st.success("Funcionalidad en desarrollo")
                else:
                    show_permission_error('alertas', 'write')
    
    # Comentarios y notas
    st.markdown("#### 💬 Comentarios y Notas")
    
    comentarios = alerta.get('comentarios', [])
    if comentarios:
        for comentario in comentarios:
            st.markdown(
                f"""
                <div style="background: #f0f2f6; padding: 0.5rem; margin: 0.5rem 0; border-radius: 5px;">
                    <strong>{comentario.get('usuario', 'Usuario')}</strong> - {comentario.get('fecha', 'Fecha')}
                    <br>
                    {comentario.get('texto', 'Sin comentario')}
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.info("No hay comentarios para esta alerta")
    
    # Añadir nuevo comentario
    if has_permission('alertas', 'write'):
        with st.form("add_comment"):
            nuevo_comentario = st.text_area("Añadir Comentario", placeholder="Escribe tu comentario aquí...")
            
            if st.form_submit_button("💬 Añadir Comentario"):
                if nuevo_comentario:
                    st.success("Comentario añadido (funcionalidad en desarrollo)")
                else:
                    st.warning("El comentario no puede estar vacío")

def show_monitoring_dashboard():
    """Muestra dashboard de monitoreo en tiempo real"""
    
    st.markdown("### 📊 Dashboard de Monitoreo")
    
    # Métricas en tiempo real
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Alertas Últimas 24h", "47", delta="12 vs ayer", delta_color="inverse")
    
    with col2:
        st.metric("Tiempo Respuesta Promedio", "2.3h", delta="-0.5h vs ayer", delta_color="normal")
    
    with col3:
        st.metric("Tasa Resolución", "85%", delta="5% vs semana anterior", delta_color="normal")
    
    with col4:
        st.metric("Alertas Críticas Pendientes", "3", delta="-2 vs ayer", delta_color="normal")
    
    # Gráfico de tendencias
    st.markdown("#### 📈 Tendencias de Alertas")
    
    # Datos simulados para el gráfico
    fechas = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
    alertas_diarias = np.random.poisson(15, len(fechas))
    criticas_diarias = np.random.poisson(2, len(fechas))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=fechas,
        y=alertas_diarias,
        mode='lines+markers',
        name='Total Alertas',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=fechas,
        y=criticas_diarias,
        mode='lines+markers',
        name='Alertas Críticas',
        line=dict(color='#d62728', width=2)
    ))
    
    fig.update_layout(
        title="Evolución de Alertas (Últimos 30 días)",
        xaxis_title="Fecha",
        yaxis_title="Número de Alertas",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_alert_configuration():
    """Muestra configuración de alertas"""
    
    st.markdown("### ⚙️ Configuración de Alertas")
    
    if not has_permission('sistema', 'write'):
        show_permission_error('sistema', 'write')
        return
    
    # Configuración de umbrales
    st.markdown("#### 🎯 Umbrales de Alerta")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Umbrales de Probabilidad ML:**")
        
        umbral_critico = st.slider(
            "Crítico (>)",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            step=0.01,
            format="%.2f"
        )
        
        umbral_alto = st.slider(
            "Alto (>)",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.01,
            format="%.2f"
        )
        
        umbral_medio = st.slider(
            "Medio (>)",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.01,
            format="%.2f"
        )
    
    with col2:
        st.markdown("**Umbrales de Altman Z-Score:**")
        
        altman_critico = st.slider(
            "Crítico (<)",
            min_value=0.0,
            max_value=5.0,
            value=1.8,
            step=0.1,
            format="%.1f"
        )
        
        altman_alto = st.slider(
            "Alto (<)",
            min_value=0.0,
            max_value=5.0,
            value=2.99,
            step=0.1,
            format="%.1f"
        )
    
    # Configuración de notificaciones
    st.markdown("#### 📧 Configuración de Notificaciones")
    
    col1, col2 = st.columns(2)
    
    with col1:
        email_criticas = st.checkbox("Email para alertas críticas", value=True)
        email_altas = st.checkbox("Email para alertas altas", value=False)
        
        if email_criticas or email_altas:
            email_destino = st.text_input("Email destino", placeholder="admin@empresa.com")
    
    with col2:
        slack_integration = st.checkbox("Integración con Slack", value=False)
        teams_integration = st.checkbox("Integración con Teams", value=False)
        
        if slack_integration:
            slack_webhook = st.text_input("Slack Webhook URL", placeholder="https://hooks.slack.com/...")
    
    # Botón para guardar configuración
    if st.button("💾 Guardar Configuración", use_container_width=True):
        st.success("Configuración guardada exitosamente (funcionalidad en desarrollo)")

def show_page():
    """Función principal de la página de alertas"""
    
    # Header de la página
    show_page_header(
        "Alertas y Monitoreo",
        "Sistema de alertas y monitoreo proactivo de riesgos",
        "🚨"
    )
    
    # Verificar permisos
    if not has_permission('predicciones', 'read'):
        show_permission_error('predicciones', 'read')
        return
    
    # Manejo de estados de sesión
    if st.session_state.get('show_alert_detail'):
        alert_id = st.session_state.get('selected_alert_id')
        if alert_id:
            show_alert_detail(alert_id)
            return
    
    # Overview de alertas
    show_alerts_overview()
    
    st.markdown("---")
    
    # Tabs principales
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Alertas", "📊 Monitoreo", "⚙️ Configuración", "📈 Reportes"])
    
    with tab1:
        show_alerts_list()
    
    with tab2:
        show_monitoring_dashboard()
    
    with tab3:
        show_alert_configuration()
    
    with tab4:
        st.markdown("### 📈 Reportes de Alertas")
        st.info("Funcionalidad de reportes en desarrollo")
        
        # Placeholder para reportes
        st.markdown("""
        **Próximas funcionalidades:**
        - Reportes automáticos de alertas
        - Análisis de tendencias de alertas
        - Métricas de tiempo de respuesta
        - Dashboards ejecutivos
        """)

if __name__ == "__main__":
    show_page()

