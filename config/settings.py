"""
Este módulo contiene toda la configuración necesaria para la aplicación
Streamlit, incluyendo conexiones a APIs, configuración de páginas y temas.
"""

import os
import streamlit as st
from typing import Dict, Any
from dataclasses import dataclass
from pathlib import Path

@dataclass
class APIConfig:
    """Configuración de conexión a la API"""
    base_url: str
    timeout: int = 30
    verify_ssl: bool = True

@dataclass
class AppConfig:
    """Configuración general de la aplicación"""
    app_name: str = "Sistema de Predicción de Quiebras"
    app_version: str = "1.0.0"
    page_title: str = "Predicción de Quiebras"
    page_icon: str = "📊"
    layout: str = "wide"
    initial_sidebar_state: str = "expanded"
    
@dataclass
class ThemeConfig:
    """Configuración del tema visual"""
    primary_color: str = "#FF6B6B"
    background_color: str = "#FFFFFF"
    secondary_background_color: str = "#F0F2F6"
    text_color: str = "#262730"
    font: str = "sans serif"

class Settings:
    """Clase principal de configuración"""
    
    def __init__(self):
        self.app = AppConfig()
        self.theme = ThemeConfig()
        self.api = self._get_api_config()
        
        # Configurar página de Streamlit
        self._configure_page()
        
        # Configurar CSS personalizado
        self._configure_custom_css()
    
    def _get_api_config(self) -> APIConfig:
        """Obtiene configuración de API desde variables de entorno"""
        api_base_url = os.getenv('API_BASE_URL', 'http://localhost:5000/api')
        
        return APIConfig(
            base_url=api_base_url,
            timeout=int(os.getenv('API_TIMEOUT', '30')),
            verify_ssl=os.getenv('API_VERIFY_SSL', 'true').lower() == 'true'
        )
    
    def _configure_page(self):
        """Configura la página principal de Streamlit"""
        st.set_page_config(
            page_title=self.app.page_title,
            page_icon=self.app.page_icon,
            layout=self.app.layout,
            initial_sidebar_state=self.app.initial_sidebar_state,
            menu_items={
                'Get Help': 'https://github.com/manus-ai/bankruptcy-prediction',
                'Report a bug': 'https://github.com/manus-ai/bankruptcy-prediction/issues',
                'About': f"""
                # {self.app.app_name}
                
                **Versión:** {self.app.app_version}
                
                Sistema avanzado de predicción de quiebras empresariales usando Machine Learning.
                
                **Características principales:**
                - Modelos ML con XGBoost y calibración
                - Análisis de Altman Z-Score
                - Dashboard interactivo
                - Alertas automáticas
                - Análisis sectorial
                
                Desarrollado por Manus AI.
                """
            }
        )
    
    def _configure_custom_css(self):
        """Configura CSS personalizado para la aplicación"""
        custom_css = f"""
        <style>
        /* Configuración general */
        .main .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
        }}
        
        /* Header personalizado */
        .custom-header {{
            background: linear-gradient(90deg, {self.theme.primary_color}, #FF8E8E);
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            color: white;
            text-align: center;
        }}
        
        /* Métricas personalizadas */
        .metric-container {{
            background: {self.theme.secondary_background_color};
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid {self.theme.primary_color};
            margin: 0.5rem 0;
        }}
        
        .metric-value {{
            font-size: 2rem;
            font-weight: bold;
            color: {self.theme.primary_color};
        }}
        
        .metric-label {{
            font-size: 0.9rem;
            color: {self.theme.text_color};
            opacity: 0.8;
        }}
        
        /* Alertas personalizadas */
        .alert-high {{
            background: #FFE6E6;
            border-left: 4px solid #FF4444;
            padding: 1rem;
            border-radius: 5px;
            margin: 0.5rem 0;
        }}
        
        .alert-medium {{
            background: #FFF4E6;
            border-left: 4px solid #FF8800;
            padding: 1rem;
            border-radius: 5px;
            margin: 0.5rem 0;
        }}
        
        .alert-low {{
            background: #E6F7E6;
            border-left: 4px solid #44AA44;
            padding: 1rem;
            border-radius: 5px;
            margin: 0.5rem 0;
        }}
        
        /* Tablas personalizadas */
        .dataframe {{
            border: none !important;
        }}
        
        .dataframe th {{
            background: {self.theme.primary_color} !important;
            color: white !important;
            font-weight: bold !important;
        }}
        
        .dataframe td {{
            border-bottom: 1px solid #E0E0E0 !important;
        }}
        
        /* Sidebar personalizado */
        .css-1d391kg {{
            background: {self.theme.secondary_background_color};
        }}
        
        /* Botones personalizados */
        .stButton > button {{
            background: {self.theme.primary_color};
            color: white;
            border: none;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            font-weight: bold;
            transition: all 0.3s;
        }}
        
        .stButton > button:hover {{
            background: #FF5252;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}
        
        /* Selectbox personalizado */
        .stSelectbox > div > div {{
            border: 2px solid {self.theme.primary_color};
            border-radius: 5px;
        }}
        
        /* Progress bar personalizado */
        .stProgress > div > div {{
            background: {self.theme.primary_color};
        }}
        
        /* Ocultar elementos de Streamlit */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        
        /* Animaciones */
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .fade-in {{
            animation: fadeIn 0.5s ease-in;
        }}
        
        /* Responsive design */
        @media (max-width: 768px) {{
            .main .block-container {{
                padding-left: 1rem;
                padding-right: 1rem;
            }}
            
            .metric-value {{
                font-size: 1.5rem;
            }}
        }}
        </style>
        """
        
        st.markdown(custom_css, unsafe_allow_html=True)

# Configuración de páginas disponibles
PAGES_CONFIG = {
    "🏠 Dashboard": {
        "icon": "🏠",
        "title": "Dashboard Principal",
        "description": "Resumen general del sistema y métricas principales"
    },
    "🏢 Empresas": {
        "icon": "🏢", 
        "title": "Gestión de Empresas",
        "description": "CRUD de empresas y información corporativa"
    },
    "📊 Predicciones": {
        "icon": "📊",
        "title": "Predicciones ML",
        "description": "Ejecución y análisis de predicciones de quiebra"
    },
    "📈 Análisis Sectorial": {
        "icon": "📈",
        "title": "Análisis por Sector",
        "description": "Análisis de riesgo por sectores económicos"
    },
    "🚨 Alertas": {
        "icon": "🚨",
        "title": "Sistema de Alertas",
        "description": "Monitoreo de alertas y notificaciones"
    },
    "📥 Carga de Datos": {
        "icon": "📥",
        "title": "ETL y Carga",
        "description": "Carga masiva de datos financieros"
    },
    "⚙️ Configuración": {
        "icon": "⚙️",
        "title": "Administración",
        "description": "Configuración del sistema y usuarios"
    }
}

# Configuración de colores para bandas de riesgo
RISK_COLORS = {
    "LOW": "#4CAF50",      # Verde
    "MEDIUM": "#FF9800",   # Naranja
    "HIGH": "#F44336",     # Rojo
    "CRITICAL": "#9C27B0"  # Púrpura
}

# Configuración de iconos
RISK_ICONS = {
    "LOW": "✅",
    "MEDIUM": "⚠️", 
    "HIGH": "🚨",
    "CRITICAL": "💀"
}

# Configuración de cache
CACHE_CONFIG = {
    "ttl": 300,  # 5 minutos
    "max_entries": 1000,
    "allow_output_mutation": True
}

# Instancia global de configuración
settings = Settings()

