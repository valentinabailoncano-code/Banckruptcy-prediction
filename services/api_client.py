"""
Este módulo proporciona una interfaz para conectar la aplicación Streamlit
con la API REST del backend, incluyendo autenticación y manejo de errores.
"""

import requests
import streamlit as st
from typing import Dict, Any, Optional, List
import json
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

from config.settings import settings

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class APIResponse:
    """Respuesta de la API"""
    success: bool
    data: Any = None
    error: str = None
    status_code: int = None

class APIClient:
    """Cliente para interactuar con la API REST"""
    
    def __init__(self):
        self.base_url = settings.api.base_url
        self.timeout = settings.api.timeout
        self.verify_ssl = settings.api.verify_ssl
        self.session = requests.Session()
        
        # Configurar headers por defecto
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Obtiene headers de autenticación"""
        headers = {}
        
        if 'access_token' in st.session_state:
            headers['Authorization'] = f"Bearer {st.session_state.access_token}"
        
        return headers
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> APIResponse:
        """Realiza una petición HTTP a la API"""
        url = f"{self.base_url}{endpoint}"
        
        # Añadir headers de autenticación
        headers = kwargs.get('headers', {})
        headers.update(self._get_auth_headers())
        kwargs['headers'] = headers
        
        # Configurar timeout y SSL
        kwargs.setdefault('timeout', self.timeout)
        kwargs.setdefault('verify', self.verify_ssl)
        
        try:
            response = self.session.request(method, url, **kwargs)
            
            # Intentar parsear JSON
            try:
                data = response.json()
            except json.JSONDecodeError:
                data = response.text
            
            if response.status_code < 400:
                return APIResponse(
                    success=True,
                    data=data,
                    status_code=response.status_code
                )
            else:
                error_msg = data.get('error', f'HTTP {response.status_code}') if isinstance(data, dict) else str(data)
                return APIResponse(
                    success=False,
                    error=error_msg,
                    status_code=response.status_code
                )
                
        except requests.exceptions.Timeout:
            return APIResponse(
                success=False,
                error="Timeout: La API no respondió en el tiempo esperado"
            )
        except requests.exceptions.ConnectionError:
            return APIResponse(
                success=False,
                error="Error de conexión: No se pudo conectar con la API"
            )
        except Exception as e:
            return APIResponse(
                success=False,
                error=f"Error inesperado: {str(e)}"
            )
    
    # Métodos de autenticación
    def login(self, username: str, password: str) -> APIResponse:
        """Inicia sesión en la API"""
        data = {
            'username': username,
            'password': password
        }
        
        response = self._make_request('POST', '/auth/login', json=data)
        
        if response.success and response.data:
            # Guardar tokens en session state
            st.session_state.access_token = response.data.get('access_token')
            st.session_state.refresh_token = response.data.get('refresh_token')
            st.session_state.user_info = response.data.get('usuario')
            st.session_state.authenticated = True
            
            logger.info(f"Login exitoso para usuario: {username}")
        
        return response
    
    def logout(self) -> APIResponse:
        """Cierra sesión"""
        response = self._make_request('POST', '/auth/logout')
        
        # Limpiar session state
        for key in ['access_token', 'refresh_token', 'user_info', 'authenticated']:
            if key in st.session_state:
                del st.session_state[key]
        
        return response
    
    def refresh_token(self) -> APIResponse:
        """Renueva el token de acceso"""
        if 'refresh_token' not in st.session_state:
            return APIResponse(success=False, error="No hay refresh token disponible")
        
        headers = {'Authorization': f"Bearer {st.session_state.refresh_token}"}
        response = self._make_request('POST', '/auth/refresh', headers=headers)
        
        if response.success and response.data:
            st.session_state.access_token = response.data.get('access_token')
        
        return response
    
    def get_profile(self) -> APIResponse:
        """Obtiene perfil del usuario actual"""
        return self._make_request('GET', '/auth/profile')
    
    # Métodos para empresas
    def get_empresas(self, page: int = 1, per_page: int = 20, **filters) -> APIResponse:
        """Obtiene lista de empresas"""
        params = {'page': page, 'per_page': per_page}
        params.update(filters)
        
        return self._make_request('GET', '/empresas/', params=params)
    
    def get_empresa(self, empresa_id: int) -> APIResponse:
        """Obtiene una empresa específica"""
        return self._make_request('GET', f'/empresas/{empresa_id}')
    
    def create_empresa(self, empresa_data: Dict[str, Any]) -> APIResponse:
        """Crea una nueva empresa"""
        return self._make_request('POST', '/empresas/', json=empresa_data)
    
    def update_empresa(self, empresa_id: int, empresa_data: Dict[str, Any]) -> APIResponse:
        """Actualiza una empresa"""
        return self._make_request('PUT', f'/empresas/{empresa_id}', json=empresa_data)
    
    def delete_empresa(self, empresa_id: int) -> APIResponse:
        """Elimina una empresa"""
        return self._make_request('DELETE', f'/empresas/{empresa_id}')
    
    def search_empresas(self, query: str, limit: int = 10) -> APIResponse:
        """Busca empresas"""
        params = {'q': query, 'limit': limit}
        return self._make_request('GET', '/empresas/search', params=params)
    
    def get_empresas_stats(self) -> APIResponse:
        """Obtiene estadísticas de empresas"""
        return self._make_request('GET', '/empresas/stats')
    
    # Métodos para predicciones
    def get_predicciones(self, page: int = 1, per_page: int = 20, **filters) -> APIResponse:
        """Obtiene lista de predicciones"""
        params = {'page': page, 'per_page': per_page}
        params.update(filters)
        
        return self._make_request('GET', '/predicciones/', params=params)
    
    def get_prediccion(self, prediccion_id: int) -> APIResponse:
        """Obtiene una predicción específica"""
        return self._make_request('GET', f'/predicciones/{prediccion_id}')
    
    def ejecutar_prediccion(self, empresa_id: int, dato_financiero_id: Optional[int] = None, 
                          include_explanations: bool = True) -> APIResponse:
        """Ejecuta una predicción"""
        data = {
            'empresa_id': empresa_id,
            'include_explanations': include_explanations
        }
        
        if dato_financiero_id:
            data['dato_financiero_id'] = dato_financiero_id
        
        return self._make_request('POST', '/predicciones/ejecutar', json=data)
    
    def ejecutar_predicciones_lote(self, empresa_ids: List[int], 
                                 include_explanations: bool = False) -> APIResponse:
        """Ejecuta predicciones en lote"""
        data = {
            'empresa_ids': empresa_ids,
            'include_explanations': include_explanations
        }
        
        return self._make_request('POST', '/predicciones/lote', json=data)
    
    def get_predicciones_stats(self) -> APIResponse:
        """Obtiene estadísticas de predicciones"""
        return self._make_request('GET', '/predicciones/stats')
    
    # Métodos para dashboard
    def get_dashboard_overview(self) -> APIResponse:
        """Obtiene resumen del dashboard"""
        return self._make_request('GET', '/dashboard/overview')
    
    def get_riesgo_sectorial(self) -> APIResponse:
        """Obtiene análisis de riesgo sectorial"""
        return self._make_request('GET', '/dashboard/riesgo-sectorial')
    
    def get_tendencias_temporales(self, periodo: str = 'mes', limite: int = 12) -> APIResponse:
        """Obtiene tendencias temporales"""
        params = {'periodo': periodo, 'limite': limite}
        return self._make_request('GET', '/dashboard/tendencias-temporales', params=params)
    
    def get_alertas_resumen(self) -> APIResponse:
        """Obtiene resumen de alertas"""
        return self._make_request('GET', '/dashboard/alertas-resumen')
    
    def get_metricas_modelo(self) -> APIResponse:
        """Obtiene métricas de modelos"""
        return self._make_request('GET', '/dashboard/metricas-modelo')
    
    def get_top_empresas_riesgo(self, limite: int = 20) -> APIResponse:
        """Obtiene empresas con mayor riesgo"""
        params = {'limite': limite}
        return self._make_request('GET', '/dashboard/top-empresas-riesgo', params=params)
    
    def get_estadisticas_sistema(self) -> APIResponse:
        """Obtiene estadísticas del sistema"""
        return self._make_request('GET', '/dashboard/estadisticas-sistema')
    
    # Métodos para ETL
    def upload_empresas(self, file_data: bytes, filename: str, config: Dict[str, Any]) -> APIResponse:
        """Sube archivo de empresas"""
        files = {'file': (filename, file_data)}
        data = config
        
        # Para multipart/form-data, no usar json
        return self._make_request('POST', '/etl/upload-empresas', files=files, data=data)
    
    def upload_datos_financieros(self, file_data: bytes, filename: str, config: Dict[str, Any]) -> APIResponse:
        """Sube archivo de datos financieros"""
        files = {'file': (filename, file_data)}
        data = config
        
        return self._make_request('POST', '/etl/upload-datos-financieros', files=files, data=data)
    
    def get_etl_logs(self, page: int = 1, per_page: int = 20, **filters) -> APIResponse:
        """Obtiene logs de ETL"""
        params = {'page': page, 'per_page': per_page}
        params.update(filters)
        
        return self._make_request('GET', '/etl/logs', params=params)
    
    def get_etl_stats(self) -> APIResponse:
        """Obtiene estadísticas de ETL"""
        return self._make_request('GET', '/etl/stats')
    
    def validate_file(self, file_data: bytes, filename: str) -> APIResponse:
        """Valida un archivo antes de procesarlo"""
        files = {'file': (filename, file_data)}
        return self._make_request('POST', '/etl/validate-file', files=files)
    
    # Métodos para alertas
    def get_alertas(self, page: int = 1, per_page: int = 20, **filters) -> APIResponse:
        """Obtiene lista de alertas"""
        params = {'page': page, 'per_page': per_page}
        params.update(filters)
        
        return self._make_request('GET', '/alertas/', params=params)
    
    def get_alerta(self, alerta_id: int) -> APIResponse:
        """Obtiene una alerta específica"""
        return self._make_request('GET', f'/alertas/{alerta_id}')
    
    def marcar_alerta_leida(self, alerta_id: int) -> APIResponse:
        """Marca una alerta como leída"""
        return self._make_request('POST', f'/alertas/{alerta_id}/marcar-leida')
    
    def get_alertas_stats(self) -> APIResponse:
        """Obtiene estadísticas de alertas"""
        return self._make_request('GET', '/alertas/stats')
    
    # Método para health check
    def health_check(self) -> APIResponse:
        """Verifica el estado de la API"""
        return self._make_request('GET', '/health')

# Instancia global del cliente API
api_client = APIClient()

# Decorador para manejar errores de API
def handle_api_errors(func):
    """Decorador para manejar errores de API en Streamlit"""
    def wrapper(*args, **kwargs):
        try:
            response = func(*args, **kwargs)
            
            if not response.success:
                if response.status_code == 401:
                    st.error("Sesión expirada. Por favor, inicia sesión nuevamente.")
                    # Limpiar session state
                    for key in ['access_token', 'refresh_token', 'user_info', 'authenticated']:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()
                else:
                    st.error(f"Error: {response.error}")
                
                return None
            
            return response.data
            
        except Exception as e:
            st.error(f"Error inesperado: {str(e)}")
            logger.error(f"Error en {func.__name__}: {str(e)}")
            return None
    
    return wrapper

# Función para verificar autenticación
def check_authentication():
    """Verifica si el usuario está autenticado"""
    if not st.session_state.get('authenticated', False):
        return False
    
    if 'access_token' not in st.session_state:
        return False
    
    # Verificar si el token está próximo a expirar (opcional)
    # Aquí podrías implementar lógica para renovar automáticamente
    
    return True

# Función para requerir autenticación
def require_authentication():
    """Requiere autenticación para acceder a una página"""
    if not check_authentication():
        st.warning("Debes iniciar sesión para acceder a esta página.")
        st.stop()

# Cache para datos frecuentemente accedidos
@st.cache_data(ttl=300)  # Cache por 5 minutos
def cached_api_call(method: str, endpoint: str, **kwargs):
    """Realiza llamadas a la API con cache"""
    client = APIClient()
    response = client._make_request(method, endpoint, **kwargs)
    
    if response.success:
        return response.data
    else:
        st.error(f"Error en API: {response.error}")
        return None

