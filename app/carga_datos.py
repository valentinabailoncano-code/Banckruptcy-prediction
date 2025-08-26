"""
Esta página maneja la carga masiva de datos empresariales y financieros,
incluyendo validación, procesamiento y monitoreo de procesos ETL.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import io

from utils.api_client import api_client, handle_api_errors
from config.settings import RISK_COLORS, RISK_ICONS
from components.sidebar import show_page_header
from components.auth import has_permission, show_permission_error

def show_etl_overview():
    """Muestra overview de procesos ETL"""
    
    response = api_client.get_etl_stats()
    
    if not response.success:
        st.error(f"Error cargando estadísticas ETL: {response.error}")
        return
    
    data = response.data
    resumen = data.get('resumen', {})
    
    st.markdown("### 📊 Estado General de ETL")
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_procesos = resumen.get('total_procesos', 0)
        st.metric(
            "Total Procesos",
            f"{total_procesos:,}",
            delta=None
        )
    
    with col2:
        exitosos = resumen.get('exitosos', 0)
        tasa_exito = (exitosos / total_procesos * 100) if total_procesos > 0 else 0
        st.metric(
            "Tasa de Éxito",
            f"{tasa_exito:.1f}%",
            delta=f"{exitosos}/{total_procesos}",
            delta_color="normal"
        )
    
    with col3:
        procesos_hoy = resumen.get('procesos_hoy', 0)
        st.metric(
            "Procesos Hoy",
            f"{procesos_hoy:,}",
            delta=None
        )
    
    with col4:
        registros_procesados = resumen.get('registros_procesados', 0)
        st.metric(
            "Registros Procesados",
            f"{registros_procesados:,}",
            delta=None
        )
    
    # Gráfico de tendencias
    show_etl_trends(data.get('tendencias', []))

def show_etl_trends(tendencias: List[Dict[str, Any]]):
    """Muestra tendencias de procesos ETL"""
    
    if not tendencias:
        st.info("No hay datos de tendencias ETL disponibles")
        return
    
    df_tendencias = pd.DataFrame(tendencias)
    df_tendencias['fecha'] = pd.to_datetime(df_tendencias['fecha'])
    
    # Crear gráfico de líneas
    fig = go.Figure()
    
    # Línea de procesos exitosos
    fig.add_trace(go.Scatter(
        x=df_tendencias['fecha'],
        y=df_tendencias['exitosos'],
        mode='lines+markers',
        name='Exitosos',
        line=dict(color='#2ca02c', width=3),
        marker=dict(size=8)
    ))
    
    # Línea de procesos con errores
    fig.add_trace(go.Scatter(
        x=df_tendencias['fecha'],
        y=df_tendencias['con_errores'],
        mode='lines+markers',
        name='Con Errores',
        line=dict(color='#d62728', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title="Tendencias de Procesos ETL (Últimos 30 días)",
        xaxis_title="Fecha",
        yaxis_title="Número de Procesos",
        height=400,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_file_upload_section():
    """Muestra sección de carga de archivos"""
    
    if not has_permission('etl', 'execute'):
        show_permission_error('etl', 'execute')
        return
    
    st.markdown("### 📥 Carga de Archivos")
    
    # Tabs para diferentes tipos de carga
    tab1, tab2 = st.tabs(["🏢 Empresas", "💰 Datos Financieros"])
    
    with tab1:
        show_empresas_upload()
    
    with tab2:
        show_datos_financieros_upload()

def show_empresas_upload():
    """Muestra interfaz de carga de empresas"""
    
    st.markdown("#### 🏢 Carga de Empresas")
    
    # Información sobre el formato
    with st.expander("ℹ️ Formato de Archivo"):
        st.markdown("""
        **Formatos soportados:** CSV, Excel (.xlsx, .xls)
        
        **Columnas requeridas:**
        - `rut`: RUT de la empresa (formato: 12345678-9)
        - `razon_social`: Razón social de la empresa
        - `sector`: Sector económico
        
        **Columnas opcionales:**
        - `nombre_fantasia`: Nombre comercial
        - `subsector`: Subsector específico
        - `pais`: País (por defecto: Chile)
        - `region`: Región o estado
        - `ciudad`: Ciudad
        - `direccion`: Dirección completa
        - `tamaño`: Tamaño de empresa (micro, pequeña, mediana, grande)
        - `numero_empleados`: Número de empleados
        - `es_publica`: Si es empresa pública (true/false)
        - `ticker_bolsa`: Ticker bursátil si aplica
        """)
    
    # Descarga de plantilla
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("📄 Descargar Plantilla", use_container_width=True):
            download_empresas_template()
    
    with col2:
        if st.button("📋 Ver Ejemplo", use_container_width=True):
            show_empresas_example()
    
    # Carga de archivo
    uploaded_file = st.file_uploader(
        "Seleccionar archivo de empresas",
        type=['csv', 'xlsx', 'xls'],
        help="Máximo 100MB, hasta 1000 empresas por archivo"
    )
    
    if uploaded_file is not None:
        # Mostrar información del archivo
        st.info(f"Archivo: {uploaded_file.name} ({uploaded_file.size:,} bytes)")
        
        # Configuración de carga
        with st.expander("⚙️ Configuración de Carga"):
            col1, col2 = st.columns(2)
            
            with col1:
                skip_duplicates = st.checkbox("Omitir duplicados", value=True)
                validate_rut = st.checkbox("Validar RUT", value=True)
            
            with col2:
                update_existing = st.checkbox("Actualizar existentes", value=False)
                max_errors = st.slider("Máximo errores permitidos", 0, 100, 10)
        
        # Validación previa
        if st.button("🔍 Validar Archivo", use_container_width=True):
            validate_empresas_file(uploaded_file)
        
        # Procesamiento
        if st.button("🚀 Procesar Archivo", use_container_width=True):
            process_empresas_file(
                uploaded_file,
                skip_duplicates=skip_duplicates,
                validate_rut=validate_rut,
                update_existing=update_existing,
                max_errors=max_errors
            )

def show_datos_financieros_upload():
    """Muestra interfaz de carga de datos financieros"""
    
    st.markdown("#### 💰 Carga de Datos Financieros")
    
    # Información sobre el formato
    with st.expander("ℹ️ Formato de Archivo"):
        st.markdown("""
        **Formatos soportados:** CSV, Excel (.xlsx, .xls)
        
        **Columnas requeridas:**
        - `rut`: RUT de la empresa
        - `periodo`: Período del reporte (YYYY-MM-DD)
        - `tipo_periodo`: Tipo (anual, trimestral, mensual)
        - `ingresos_operacionales`: Ingresos operacionales
        - `activos_totales`: Activos totales
        - `pasivos_totales`: Pasivos totales
        - `patrimonio`: Patrimonio neto
        
        **Columnas opcionales:**
        - `activos_corrientes`: Activos corrientes
        - `pasivos_corrientes`: Pasivos corrientes
        - `inventarios`: Inventarios
        - `cuentas_por_cobrar`: Cuentas por cobrar
        - `efectivo`: Efectivo y equivalentes
        - `deuda_financiera`: Deuda financiera total
        - `utilidad_neta`: Utilidad neta del período
        - `ebitda`: EBITDA
        - `ventas`: Ventas totales
        """)
    
    # Descarga de plantilla
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("📄 Descargar Plantilla Financiera", use_container_width=True):
            download_financieros_template()
    
    with col2:
        if st.button("📋 Ver Ejemplo Financiero", use_container_width=True):
            show_financieros_example()
    
    # Carga de archivo
    uploaded_file = st.file_uploader(
        "Seleccionar archivo de datos financieros",
        type=['csv', 'xlsx', 'xls'],
        help="Máximo 100MB, hasta 5000 registros por archivo",
        key="financieros_upload"
    )
    
    if uploaded_file is not None:
        # Mostrar información del archivo
        st.info(f"Archivo: {uploaded_file.name} ({uploaded_file.size:,} bytes)")
        
        # Configuración de carga
        with st.expander("⚙️ Configuración de Carga"):
            col1, col2 = st.columns(2)
            
            with col1:
                validate_equations = st.checkbox("Validar ecuaciones contables", value=True)
                calculate_ratios = st.checkbox("Calcular ratios automáticamente", value=True)
            
            with col2:
                update_existing = st.checkbox("Actualizar existentes", value=False)
                max_errors = st.slider("Máximo errores permitidos", 0, 100, 20, key="fin_max_errors")
        
        # Validación previa
        if st.button("🔍 Validar Archivo Financiero", use_container_width=True):
            validate_financieros_file(uploaded_file)
        
        # Procesamiento
        if st.button("🚀 Procesar Archivo Financiero", use_container_width=True):
            process_financieros_file(
                uploaded_file,
                validate_equations=validate_equations,
                calculate_ratios=calculate_ratios,
                update_existing=update_existing,
                max_errors=max_errors
            )

def download_empresas_template():
    """Genera y descarga plantilla de empresas"""
    
    # Crear DataFrame de ejemplo
    template_data = {
        'rut': ['12345678-9', '98765432-1'],
        'razon_social': ['Empresa Ejemplo S.A.', 'Comercial Demo Ltda.'],
        'sector': ['Technology', 'Retail'],
        'nombre_fantasia': ['Ejemplo', 'Demo'],
        'subsector': ['Software', 'Comercio'],
        'pais': ['Chile', 'Chile'],
        'region': ['Metropolitana', 'Valparaíso'],
        'ciudad': ['Santiago', 'Viña del Mar'],
        'tamaño': ['mediana', 'pequeña'],
        'numero_empleados': [150, 25],
        'es_publica': [False, False]
    }
    
    df_template = pd.DataFrame(template_data)
    
    # Convertir a CSV
    csv_buffer = io.StringIO()
    df_template.to_csv(csv_buffer, index=False, encoding='utf-8')
    csv_data = csv_buffer.getvalue()
    
    st.download_button(
        label="📥 Descargar plantilla_empresas.csv",
        data=csv_data,
        file_name="plantilla_empresas.csv",
        mime="text/csv"
    )

def download_financieros_template():
    """Genera y descarga plantilla de datos financieros"""
    
    # Crear DataFrame de ejemplo
    template_data = {
        'rut': ['12345678-9', '12345678-9', '98765432-1'],
        'periodo': ['2023-12-31', '2023-09-30', '2023-12-31'],
        'tipo_periodo': ['anual', 'trimestral', 'anual'],
        'ingresos_operacionales': [1000000, 250000, 500000],
        'activos_totales': [2000000, 1900000, 800000],
        'pasivos_totales': [1200000, 1100000, 400000],
        'patrimonio': [800000, 800000, 400000],
        'activos_corrientes': [600000, 550000, 300000],
        'pasivos_corrientes': [400000, 350000, 200000],
        'efectivo': [100000, 80000, 50000],
        'utilidad_neta': [80000, 20000, 40000]
    }
    
    df_template = pd.DataFrame(template_data)
    
    # Convertir a CSV
    csv_buffer = io.StringIO()
    df_template.to_csv(csv_buffer, index=False, encoding='utf-8')
    csv_data = csv_buffer.getvalue()
    
    st.download_button(
        label="📥 Descargar plantilla_financieros.csv",
        data=csv_data,
        file_name="plantilla_financieros.csv",
        mime="text/csv"
    )

def show_empresas_example():
    """Muestra ejemplo de datos de empresas"""
    
    st.markdown("#### 📋 Ejemplo de Datos de Empresas")
    
    example_data = {
        'rut': ['12345678-9', '98765432-1', '11111111-1'],
        'razon_social': ['Tecnología Avanzada S.A.', 'Comercial Sur Ltda.', 'Industrias Norte S.A.'],
        'sector': ['Technology', 'Retail', 'Manufacturing'],
        'nombre_fantasia': ['TechAdvanced', 'ComercialSur', 'IndustriasNorte'],
        'pais': ['Chile', 'Chile', 'Chile'],
        'region': ['Metropolitana', 'Biobío', 'Antofagasta'],
        'tamaño': ['grande', 'mediana', 'grande']
    }
    
    df_example = pd.DataFrame(example_data)
    st.dataframe(df_example, use_container_width=True)

def show_financieros_example():
    """Muestra ejemplo de datos financieros"""
    
    st.markdown("#### 📋 Ejemplo de Datos Financieros")
    
    example_data = {
        'rut': ['12345678-9', '12345678-9', '98765432-1'],
        'periodo': ['2023-12-31', '2023-09-30', '2023-12-31'],
        'tipo_periodo': ['anual', 'trimestral', 'anual'],
        'ingresos_operacionales': [1500000, 375000, 800000],
        'activos_totales': [3000000, 2900000, 1200000],
        'pasivos_totales': [1800000, 1700000, 600000],
        'patrimonio': [1200000, 1200000, 600000]
    }
    
    df_example = pd.DataFrame(example_data)
    st.dataframe(df_example, use_container_width=True)

def validate_empresas_file(uploaded_file):
    """Valida archivo de empresas"""
    
    with st.spinner("Validando archivo de empresas..."):
        # Leer archivo
        file_bytes = uploaded_file.read()
        
        # Enviar a API para validación
        response = api_client.validate_file(file_bytes, uploaded_file.name)
    
    if response.success:
        validation_result = response.data
        
        st.success("✅ Archivo validado exitosamente")
        
        # Mostrar resultados de validación
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Registros Válidos", validation_result.get('valid_records', 0))
        
        with col2:
            st.metric("Registros con Errores", validation_result.get('invalid_records', 0))
        
        with col3:
            st.metric("Duplicados Detectados", validation_result.get('duplicates', 0))
        
        # Mostrar errores si los hay
        errors = validation_result.get('errors', [])
        if errors:
            with st.expander(f"⚠️ Errores Encontrados ({len(errors)})"):
                for error in errors[:10]:  # Mostrar solo los primeros 10
                    st.error(f"Fila {error.get('row', 'N/A')}: {error.get('message', 'Error desconocido')}")
                
                if len(errors) > 10:
                    st.info(f"... y {len(errors) - 10} errores más")
    else:
        st.error(f"❌ Error validando archivo: {response.error}")

def validate_financieros_file(uploaded_file):
    """Valida archivo de datos financieros"""
    
    with st.spinner("Validando archivo de datos financieros..."):
        # Leer archivo
        file_bytes = uploaded_file.read()
        
        # Enviar a API para validación
        response = api_client.validate_file(file_bytes, uploaded_file.name)
    
    if response.success:
        validation_result = response.data
        
        st.success("✅ Archivo validado exitosamente")
        
        # Mostrar resultados de validación
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Registros Válidos", validation_result.get('valid_records', 0))
        
        with col2:
            st.metric("Registros con Errores", validation_result.get('invalid_records', 0))
        
        with col3:
            st.metric("Ecuaciones Incorrectas", validation_result.get('equation_errors', 0))
        
        with col4:
            st.metric("Empresas No Encontradas", validation_result.get('missing_companies', 0))
        
        # Mostrar errores si los hay
        errors = validation_result.get('errors', [])
        if errors:
            with st.expander(f"⚠️ Errores Encontrados ({len(errors)})"):
                for error in errors[:10]:
                    st.error(f"Fila {error.get('row', 'N/A')}: {error.get('message', 'Error desconocido')}")
                
                if len(errors) > 10:
                    st.info(f"... y {len(errors) - 10} errores más")
    else:
        st.error(f"❌ Error validando archivo: {response.error}")

def process_empresas_file(uploaded_file, **config):
    """Procesa archivo de empresas"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Iniciando procesamiento de empresas...")
    progress_bar.progress(0.1)
    
    with st.spinner("Procesando archivo de empresas..."):
        # Leer archivo
        file_bytes = uploaded_file.read()
        
        # Configuración de procesamiento
        processing_config = {
            'skip_duplicates': config.get('skip_duplicates', True),
            'validate_rut': config.get('validate_rut', True),
            'update_existing': config.get('update_existing', False),
            'max_errors': config.get('max_errors', 10)
        }
        
        progress_bar.progress(0.3)
        status_text.text("Enviando archivo al servidor...")
        
        # Enviar a API para procesamiento
        response = api_client.upload_empresas(file_bytes, uploaded_file.name, processing_config)
        
        progress_bar.progress(1.0)
    
    if response.success:
        result = response.data
        
        st.success("✅ Archivo procesado exitosamente")
        
        # Mostrar resultados
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Empresas Creadas", result.get('created', 0))
        
        with col2:
            st.metric("Empresas Actualizadas", result.get('updated', 0))
        
        with col3:
            st.metric("Errores", result.get('errors', 0))
        
        with col4:
            st.metric("Duplicados Omitidos", result.get('skipped', 0))
        
        # Mostrar log del proceso
        log_id = result.get('log_id')
        if log_id:
            st.info(f"ID del proceso: {log_id}")
            
            if st.button("📋 Ver Log Completo"):
                show_process_log(log_id)
    else:
        st.error(f"❌ Error procesando archivo: {response.error}")
    
    status_text.empty()

def process_financieros_file(uploaded_file, **config):
    """Procesa archivo de datos financieros"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Iniciando procesamiento de datos financieros...")
    progress_bar.progress(0.1)
    
    with st.spinner("Procesando archivo de datos financieros..."):
        # Leer archivo
        file_bytes = uploaded_file.read()
        
        # Configuración de procesamiento
        processing_config = {
            'validate_equations': config.get('validate_equations', True),
            'calculate_ratios': config.get('calculate_ratios', True),
            'update_existing': config.get('update_existing', False),
            'max_errors': config.get('max_errors', 20)
        }
        
        progress_bar.progress(0.3)
        status_text.text("Enviando archivo al servidor...")
        
        # Enviar a API para procesamiento
        response = api_client.upload_datos_financieros(file_bytes, uploaded_file.name, processing_config)
        
        progress_bar.progress(1.0)
    
    if response.success:
        result = response.data
        
        st.success("✅ Archivo procesado exitosamente")
        
        # Mostrar resultados
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Registros Creados", result.get('created', 0))
        
        with col2:
            st.metric("Registros Actualizados", result.get('updated', 0))
        
        with col3:
            st.metric("Errores", result.get('errors', 0))
        
        with col4:
            st.metric("Ratios Calculados", result.get('ratios_calculated', 0))
        
        # Mostrar log del proceso
        log_id = result.get('log_id')
        if log_id:
            st.info(f"ID del proceso: {log_id}")
            
            if st.button("📋 Ver Log Completo", key="log_financieros"):
                show_process_log(log_id)
    else:
        st.error(f"❌ Error procesando archivo: {response.error}")
    
    status_text.empty()

def show_etl_logs():
    """Muestra logs de procesos ETL"""
    
    st.markdown("### 📋 Historial de Procesos ETL")
    
    # Filtros
    with st.expander("🔧 Filtros"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            estado_filter = st.selectbox(
                "Estado",
                ["Todos", "completado", "error", "en_proceso"],
                index=0
            )
        
        with col2:
            tipo_filter = st.selectbox(
                "Tipo",
                ["Todos", "empresas", "datos_financieros"],
                index=0
            )
        
        with col3:
            fecha_filter = st.selectbox(
                "Período",
                ["Todos", "Hoy", "Última semana", "Último mes"],
                index=0
            )
    
    # Paginación
    col1, col2 = st.columns([1, 1])
    
    with col1:
        page = st.number_input("Página", min_value=1, value=1, key="etl_page")
    
    with col2:
        per_page = st.selectbox("Por página", [10, 20, 50], index=1, key="etl_per_page")
    
    # Construir filtros
    filters = {}
    if estado_filter != "Todos":
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
    
    # Cargar logs
    with st.spinner("Cargando logs ETL..."):
        response = api_client.get_etl_logs(
            page=page,
            per_page=per_page,
            **filters
        )
    
    if not response.success:
        st.error(f"Error cargando logs: {response.error}")
        return
    
    data = response.data
    logs = data.get('logs', [])
    pagination = data.get('pagination', {})
    
    if not logs:
        st.info("No se encontraron logs con los filtros aplicados")
        return
    
    # Información de paginación
    st.info(f"Mostrando {len(logs)} de {pagination.get('total', 0)} procesos")
    
    # Lista de logs
    for log in logs:
        show_etl_log_card(log)

def show_etl_log_card(log: Dict[str, Any]):
    """Muestra tarjeta de log ETL"""
    
    estado = log.get('estado', 'desconocido')
    tipo = log.get('tipo', 'desconocido')
    
    # Colores según estado
    color_map = {
        'completado': '#e8f5e8',
        'error': '#ffebee',
        'en_proceso': '#fff3e0',
        'cancelado': '#f5f5f5'
    }
    
    border_color_map = {
        'completado': '#388e3c',
        'error': '#d32f2f',
        'en_proceso': '#f57c00',
        'cancelado': '#666'
    }
    
    bg_color = color_map.get(estado, '#f5f5f5')
    border_color = border_color_map.get(estado, '#666')
    
    # Emoji según estado
    emoji_map = {
        'completado': '✅',
        'error': '❌',
        'en_proceso': '⏳',
        'cancelado': '⏹️'
    }
    
    emoji = emoji_map.get(estado, '📊')
    
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
                        {emoji} Proceso {tipo.title()} - {log.get('id', 'N/A')}
                    </h4>
                    <span style="
                        background: {border_color}; 
                        color: white; 
                        padding: 0.25rem 0.5rem; 
                        border-radius: 15px; 
                        font-size: 0.8rem;
                    ">
                        {estado.upper()}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Información del proceso
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.caption(f"**Archivo:** {log.get('nombre_archivo', 'N/A')}")
        
        with col2:
            st.caption(f"**Registros:** {log.get('registros_procesados', 0)}")
        
        with col3:
            fecha = log.get('fecha_inicio', '')[:16] if log.get('fecha_inicio') else 'N/A'
            st.caption(f"**Inicio:** {fecha}")
        
        with col4:
            if st.button("👁️ Ver Detalles", key=f"view_log_{log['id']}"):
                show_process_log(log['id'])
        
        st.markdown("---")

def show_process_log(log_id: int):
    """Muestra detalles completos de un log de proceso"""
    
    with st.spinner("Cargando detalles del proceso..."):
        response = api_client._make_request('GET', f'/etl/logs/{log_id}')
    
    if not response.success:
        st.error(f"Error cargando log: {response.error}")
        return
    
    log = response.data
    
    with st.expander(f"📋 Detalles del Proceso {log_id}", expanded=True):
        # Información básica
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Información Básica:**")
            st.write(f"ID: {log.get('id', 'N/A')}")
            st.write(f"Tipo: {log.get('tipo', 'N/A')}")
            st.write(f"Estado: {log.get('estado', 'N/A')}")
            st.write(f"Usuario: {log.get('usuario', 'N/A')}")
        
        with col2:
            st.markdown("**Archivo:**")
            st.write(f"Nombre: {log.get('nombre_archivo', 'N/A')}")
            st.write(f"Tamaño: {log.get('tamaño_archivo', 0):,} bytes")
            st.write(f"Registros: {log.get('registros_procesados', 0)}")
        
        with col3:
            st.markdown("**Tiempos:**")
            st.write(f"Inicio: {log.get('fecha_inicio', 'N/A')}")
            st.write(f"Fin: {log.get('fecha_fin', 'N/A')}")
            duracion = log.get('duracion_segundos', 0)
            st.write(f"Duración: {duracion:.1f}s")
        
        # Resultados
        resultados = log.get('resultados', {})
        if resultados:
            st.markdown("**Resultados:**")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Creados", resultados.get('created', 0))
            
            with col2:
                st.metric("Actualizados", resultados.get('updated', 0))
            
            with col3:
                st.metric("Errores", resultados.get('errors', 0))
            
            with col4:
                st.metric("Omitidos", resultados.get('skipped', 0))
        
        # Errores
        errores = log.get('errores', [])
        if errores:
            st.markdown("**Errores Encontrados:**")
            
            for error in errores[:10]:
                st.error(f"Fila {error.get('row', 'N/A')}: {error.get('message', 'Error desconocido')}")
            
            if len(errores) > 10:
                st.info(f"... y {len(errores) - 10} errores más")

def show_page():
    """Función principal de la página de carga de datos"""
    
    # Header de la página
    show_page_header(
        "Carga de Datos (ETL)",
        "Procesamiento masivo de datos empresariales y financieros",
        "📥"
    )
    
    # Verificar permisos
    if not has_permission('etl', 'read'):
        show_permission_error('etl', 'read')
        return
    
    # Overview de ETL
    show_etl_overview()
    
    st.markdown("---")
    
    # Tabs principales
    tab1, tab2, tab3 = st.tabs(["📥 Carga de Archivos", "📋 Historial", "📊 Estadísticas"])
    
    with tab1:
        show_file_upload_section()
    
    with tab2:
        show_etl_logs()
    
    with tab3:
        st.markdown("### 📊 Estadísticas Detalladas")
        st.info("Funcionalidad de estadísticas detalladas en desarrollo")
        
        # Placeholder para estadísticas futuras
        st.markdown("""
        **Próximas funcionalidades:**
        - Análisis de calidad de datos
        - Métricas de rendimiento ETL
        - Alertas de procesos fallidos
        - Optimización de cargas
        """)

if __name__ == "__main__":
    show_page()

