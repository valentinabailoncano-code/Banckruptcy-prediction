"""
Este módulo maneja la navegación lateral de la aplicación Streamlit,
incluyendo menú de páginas y información del usuario.
"""

import streamlit as st
from streamlit_option_menu import option_menu
from datetime import datetime

from config.settings import PAGES_CONFIG, settings
from components.auth import show_logout_button, get_user_permissions
from utils.api_client import api_client

def create_sidebar() -> str:
    """Crea el sidebar con navegación y retorna la página seleccionada"""
    
    with st.sidebar:
        # Logo y título
        st.markdown(
            f"""
            <div style="text-align: center; margin-bottom: 1rem;">
                <h2>📊 {settings.app.app_name}</h2>
                <p style="color: #666; font-size: 0.9rem;">v{settings.app.app_version}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Estado de la API
        show_api_status()
        
        # Menú principal de navegación
        selected_page = create_navigation_menu()
        
        # Información del usuario y logout
        show_logout_button()
        
        # Información adicional
        show_sidebar_info()
        
        return selected_page

def show_api_status():
    """Muestra el estado de conexión con la API"""
    
    # Verificar estado de la API
    if st.button("🔄 Verificar API", help="Verificar conexión con el backend"):
        with st.spinner("Verificando..."):
            response = api_client.health_check()
            
            if response.success:
                st.success("✅ API Conectada")
                
                # Mostrar información adicional si está disponible
                if response.data:
                    data = response.data
                    st.markdown(
                        f"""
                        **Estado:** {data.get('status', 'unknown')}  
                        **Versión:** {data.get('version', 'unknown')}  
                        **Base de datos:** {'✅' if data.get('database', {}).get('connected') else '❌'}
                        """
                    )
            else:
                st.error("❌ API Desconectada")
                st.caption(f"Error: {response.error}")

def create_navigation_menu() -> str:
    """Crea el menú de navegación principal"""
    
    # Obtener permisos del usuario
    permisos = get_user_permissions()
    
    # Filtrar páginas según permisos
    available_pages = filter_pages_by_permissions(permisos)
    
    # Crear menú con option_menu
    selected = option_menu(
        menu_title="🧭 Navegación",
        options=list(available_pages.keys()),
        icons=[page["icon"] for page in available_pages.values()],
        menu_icon="list",
        default_index=0,
        orientation="vertical",
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": settings.theme.primary_color, "font-size": "18px"},
            "nav-link": {
                "font-size": "14px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#eee"
            },
            "nav-link-selected": {"background-color": settings.theme.primary_color},
        }
    )
    
    # Mostrar descripción de la página seleccionada
    if selected in available_pages:
        st.caption(available_pages[selected]["description"])
    
    return selected

def filter_pages_by_permissions(permisos: dict) -> dict:
    """Filtra las páginas disponibles según los permisos del usuario"""
    
    # Mapeo de páginas a permisos requeridos
    page_permissions = {
        "🏠 Dashboard": None,  # Siempre disponible
        "🏢 Empresas": ("empresas", "read"),
        "📊 Predicciones": ("predicciones", "read"),
        "📈 Análisis Sectorial": ("predicciones", "read"),
        "🚨 Alertas": ("predicciones", "read"),
        "📥 Carga de Datos": ("etl", "read"),
        "⚙️ Configuración": ("sistema", "read")
    }
    
    available_pages = {}
    
    for page_name, page_config in PAGES_CONFIG.items():
        # Verificar permisos
        if page_name in page_permissions:
            required_permission = page_permissions[page_name]
            
            if required_permission is None:
                # Página siempre disponible
                available_pages[page_name] = page_config
            else:
                resource, action = required_permission
                if resource in permisos and action in permisos[resource]:
                    available_pages[page_name] = page_config
        else:
            # Si no está en el mapeo, incluir por defecto
            available_pages[page_name] = page_config
    
    return available_pages

def show_sidebar_info():
    """Muestra información adicional en el sidebar"""
    
    st.markdown("---")
    
    # Estadísticas rápidas
    with st.expander("📊 Estadísticas Rápidas"):
        show_quick_stats()
    
    # Alertas recientes
    with st.expander("🚨 Alertas Recientes"):
        show_recent_alerts()
    
    # Información del sistema
    with st.expander("ℹ️ Información"):
        st.markdown(
            f"""
            **Aplicación:** {settings.app.app_name}  
            **Versión:** {settings.app.app_version}  
            **Desarrollado por:** Manus AI  
            **Fecha:** {datetime.now().strftime('%Y-%m-%d')}
            
            **Soporte técnico:**  
            📧 support@manus.ai  
            📚 [Documentación](https://docs.manus.ai)
            """
        )

@st.cache_data(ttl=60)  # Cache por 1 minuto
def show_quick_stats():
    """Muestra estadísticas rápidas del sistema"""
    
    try:
        response = api_client.get_dashboard_overview()
        
        if response and response.success:
            data = response.data
            
            # Empresas
            empresas = data.get('empresas', {})
            st.metric(
                "Empresas Activas",
                empresas.get('activas', 0),
                delta=None
            )
            
            # Predicciones
            predicciones = data.get('predicciones', {})
            st.metric(
                "Predicciones (Mes)",
                predicciones.get('ultimo_mes', 0),
                delta=None
            )
            
            # Alertas
            alertas = data.get('alertas', {})
            st.metric(
                "Alertas Activas",
                alertas.get('activas', 0),
                delta=f"+{alertas.get('criticas', 0)} críticas" if alertas.get('criticas', 0) > 0 else None,
                delta_color="inverse"
            )
        else:
            st.caption("No se pudieron cargar las estadísticas")
            
    except Exception as e:
        st.caption(f"Error cargando estadísticas: {str(e)}")

@st.cache_data(ttl=120)  # Cache por 2 minutos
def show_recent_alerts():
    """Muestra alertas recientes"""
    
    try:
        response = api_client.get_alertas_resumen()
        
        if response and response.success:
            data = response.data
            alertas_recientes = data.get('alertas_recientes', [])
            
            if alertas_recientes:
                for alerta in alertas_recientes[:3]:  # Solo las 3 más recientes
                    severidad = alerta.get('severidad', 'INFO')
                    
                    # Emoji según severidad
                    emoji_map = {
                        'CRITICAL': '🔴',
                        'HIGH': '🟠',
                        'MEDIUM': '🟡',
                        'LOW': '🟢',
                        'INFO': 'ℹ️'
                    }
                    
                    emoji = emoji_map.get(severidad, 'ℹ️')
                    
                    st.markdown(
                        f"""
                        {emoji} **{alerta.get('titulo', 'Sin título')}**  
                        {alerta.get('mensaje', '')[:50]}...
                        """
                    )
                    
                    st.caption(f"Empresa: {alerta.get('empresa', {}).get('razon_social', 'N/A')}")
                    st.markdown("---")
            else:
                st.caption("No hay alertas recientes")
        else:
            st.caption("No se pudieron cargar las alertas")
            
    except Exception as e:
        st.caption(f"Error cargando alertas: {str(e)}")

def show_navigation_breadcrumb(current_page: str):
    """Muestra breadcrumb de navegación en la página principal"""
    
    if current_page in PAGES_CONFIG:
        page_config = PAGES_CONFIG[current_page]
        
        st.markdown(
            f"""
            <div style="margin-bottom: 1rem; padding: 0.5rem; background: #f0f2f6; border-radius: 5px;">
                <span style="color: #666;">📍 Navegación:</span>
                <strong>{page_config['icon']} {page_config['title']}</strong>
                <br>
                <small>{page_config['description']}</small>
            </div>
            """,
            unsafe_allow_html=True
        )

def show_page_header(title: str, description: str = None, icon: str = "📊"):
    """Muestra header estándar para las páginas"""
    
    st.markdown(
        f"""
        <div class="fade-in" style="margin-bottom: 2rem;">
            <h1>{icon} {title}</h1>
            {f'<p style="color: #666; font-size: 1.1rem;">{description}</p>' if description else ''}
        </div>
        """,
        unsafe_allow_html=True
    )

