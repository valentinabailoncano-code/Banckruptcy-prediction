"""
Esta página maneja la configuración del sistema, gestión de usuarios,
parámetros del modelo y herramientas administrativas.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from utils.api_client import api_client, handle_api_errors
from config.settings import RISK_COLORS, RISK_ICONS
from components.sidebar import show_page_header
from components.auth import has_permission, show_permission_error, require_role

def show_system_overview():
    """Muestra overview del sistema"""
    
    st.markdown("### ⚙️ Estado del Sistema")
    
    # Verificar estado de la API
    response = api_client.health_check()
    
    if response.success:
        health_data = response.data
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Estado API",
                "🟢 Activa" if health_data.get('status') == 'healthy' else "🔴 Inactiva",
                delta=None
            )
        
        with col2:
            db_status = health_data.get('database', {}).get('connected', False)
            st.metric(
                "Base de Datos",
                "🟢 Conectada" if db_status else "🔴 Desconectada",
                delta=None
            )
        
        with col3:
            version = health_data.get('version', 'Desconocida')
            st.metric(
                "Versión API",
                version,
                delta=None
            )
        
        with col4:
            uptime = health_data.get('uptime', 'Desconocido')
            st.metric(
                "Tiempo Activo",
                uptime,
                delta=None
            )
        
        # Información adicional del sistema
        if health_data.get('details'):
            with st.expander("🔍 Detalles del Sistema"):
                details = health_data['details']
                
                for key, value in details.items():
                    st.write(f"**{key.replace('_', ' ').title()}:** {value}")
    else:
        st.error(f"❌ Error conectando con la API: {response.error}")

def show_user_management():
    """Muestra gestión de usuarios"""
    
    require_role(['admin'])
    
    st.markdown("### 👥 Gestión de Usuarios")
    
    # Cargar lista de usuarios
    response = api_client._make_request('GET', '/auth/users')
    
    if not response.success:
        st.error(f"Error cargando usuarios: {response.error}")
        return
    
    usuarios = response.data.get('usuarios', [])
    
    # Estadísticas de usuarios
    col1, col2, col3, col4 = st.columns(4)
    
    total_usuarios = len(usuarios)
    usuarios_activos = len([u for u in usuarios if u.get('activo', True)])
    admins = len([u for u in usuarios if u.get('rol') == 'admin'])
    analistas = len([u for u in usuarios if u.get('rol') == 'analista'])
    
    with col1:
        st.metric("Total Usuarios", total_usuarios)
    
    with col2:
        st.metric("Usuarios Activos", usuarios_activos)
    
    with col3:
        st.metric("Administradores", admins)
    
    with col4:
        st.metric("Analistas", analistas)
    
    # Botón para crear nuevo usuario
    if st.button("➕ Crear Nuevo Usuario", use_container_width=True):
        st.session_state.show_user_form = True
        st.session_state.edit_user_id = None
        st.rerun()
    
    # Lista de usuarios
    st.markdown("#### 📋 Lista de Usuarios")
    
    for usuario in usuarios:
        show_user_card(usuario)

def show_user_card(usuario: Dict[str, Any]):
    """Muestra tarjeta de usuario"""
    
    activo = usuario.get('activo', True)
    rol = usuario.get('rol', 'usuario')
    
    # Color según rol
    color_map = {
        'admin': '#d32f2f',
        'analista': '#f57c00',
        'usuario': '#1976d2'
    }
    
    color = color_map.get(rol, '#666')
    bg_color = f"{color}20"
    
    # Emoji según rol
    emoji_map = {
        'admin': '👑',
        'analista': '📊',
        'usuario': '👤'
    }
    
    emoji = emoji_map.get(rol, '👤')
    
    with st.container():
        st.markdown(
            f"""
            <div style="
                background: {bg_color}; 
                border-left: 5px solid {color}; 
                padding: 1rem; 
                margin: 0.5rem 0; 
                border-radius: 5px;
                opacity: {'1' if activo else '0.6'};
            ">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h4 style="margin: 0; color: {color};">
                        {emoji} {usuario.get('nombre', '')} {usuario.get('apellido', '')}
                    </h4>
                    <span style="
                        background: {color}; 
                        color: white; 
                        padding: 0.25rem 0.5rem; 
                        border-radius: 15px; 
                        font-size: 0.8rem;
                    ">
                        {rol.upper()}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Información del usuario
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.caption(f"**Username:** {usuario.get('username', 'N/A')}")
        
        with col2:
            st.caption(f"**Email:** {usuario.get('email', 'N/A')}")
        
        with col3:
            estado = "🟢 Activo" if activo else "🔴 Inactivo"
            st.caption(f"**Estado:** {estado}")
        
        with col4:
            # Botones de acción
            col_a, col_b = st.columns(2)
            
            with col_a:
                if st.button("✏️ Editar", key=f"edit_user_{usuario['id']}"):
                    st.session_state.show_user_form = True
                    st.session_state.edit_user_id = usuario['id']
                    st.rerun()
            
            with col_b:
                action_text = "🔒 Desactivar" if activo else "🔓 Activar"
                if st.button(action_text, key=f"toggle_user_{usuario['id']}"):
                    toggle_user_status(usuario['id'], not activo)
        
        st.markdown("---")

def toggle_user_status(user_id: int, new_status: bool):
    """Activa/desactiva un usuario"""
    
    action = "activar" if new_status else "desactivar"
    
    with st.spinner(f"{'Activando' if new_status else 'Desactivando'} usuario..."):
        # En implementación real, esto sería una llamada a la API
        st.success(f"Usuario {'activado' if new_status else 'desactivado'} exitosamente (funcionalidad en desarrollo)")
        st.rerun()

def show_user_form():
    """Muestra formulario de usuario"""
    
    user_id = st.session_state.get('edit_user_id')
    is_edit = user_id is not None
    title = "✏️ Editar Usuario" if is_edit else "➕ Nuevo Usuario"
    
    st.markdown(f"### {title}")
    
    # Cargar datos si es edición
    user_data = {}
    if is_edit:
        # En implementación real, cargar datos del usuario
        user_data = {
            'nombre': 'Usuario',
            'apellido': 'Ejemplo',
            'username': 'usuario_ejemplo',
            'email': 'usuario@ejemplo.com',
            'rol': 'usuario',
            'activo': True
        }
    
    # Formulario
    with st.form("user_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            nombre = st.text_input(
                "Nombre *",
                value=user_data.get('nombre', ''),
                placeholder="Nombre del usuario"
            )
            
            username = st.text_input(
                "Username *",
                value=user_data.get('username', ''),
                placeholder="nombre_usuario"
            )
            
            if not is_edit:
                password = st.text_input(
                    "Contraseña *",
                    type="password",
                    placeholder="Mínimo 8 caracteres"
                )
        
        with col2:
            apellido = st.text_input(
                "Apellido *",
                value=user_data.get('apellido', ''),
                placeholder="Apellido del usuario"
            )
            
            email = st.text_input(
                "Email *",
                value=user_data.get('email', ''),
                placeholder="usuario@empresa.com"
            )
            
            if not is_edit:
                confirm_password = st.text_input(
                    "Confirmar Contraseña *",
                    type="password",
                    placeholder="Repetir contraseña"
                )
        
        # Configuración adicional
        col1, col2, col3 = st.columns(3)
        
        with col1:
            rol = st.selectbox(
                "Rol *",
                ["usuario", "analista", "admin"],
                index=["usuario", "analista", "admin"].index(user_data.get('rol', 'usuario'))
            )
        
        with col2:
            if is_edit:
                activo = st.checkbox(
                    "Usuario Activo",
                    value=user_data.get('activo', True)
                )
        
        with col3:
            telefono = st.text_input(
                "Teléfono",
                value=user_data.get('telefono', ''),
                placeholder="+56 9 1234 5678"
            )
        
        # Botones
        col1, col2 = st.columns(2)
        
        with col1:
            submit_button = st.form_submit_button(
                "💾 Guardar" if is_edit else "➕ Crear Usuario",
                use_container_width=True
            )
        
        with col2:
            if st.form_submit_button("❌ Cancelar", use_container_width=True):
                st.session_state.show_user_form = False
                st.rerun()
    
    # Procesar formulario
    if submit_button:
        # Validaciones
        errors = []
        
        if not all([nombre, apellido, username, email]):
            errors.append("Todos los campos marcados con * son obligatorios")
        
        if not is_edit:
            if not password or not confirm_password:
                errors.append("La contraseña es obligatoria")
            elif password != confirm_password:
                errors.append("Las contraseñas no coinciden")
            elif len(password) < 8:
                errors.append("La contraseña debe tener al menos 8 caracteres")
        
        if errors:
            for error in errors:
                st.error(f"❌ {error}")
        else:
            # Preparar datos
            form_data = {
                'nombre': nombre,
                'apellido': apellido,
                'username': username,
                'email': email,
                'rol': rol,
                'telefono': telefono if telefono else None
            }
            
            if not is_edit:
                form_data['password'] = password
            else:
                form_data['activo'] = activo
            
            # Enviar datos (simulado)
            with st.spinner("Guardando usuario..."):
                st.success(f"Usuario {'actualizado' if is_edit else 'creado'} exitosamente (funcionalidad en desarrollo)")
                st.session_state.show_user_form = False
                st.rerun()

def show_model_configuration():
    """Muestra configuración de modelos"""
    
    require_role(['admin', 'analista'])
    
    st.markdown("### 🤖 Configuración de Modelos")
    
    # Configuración de umbrales
    st.markdown("#### 🎯 Umbrales de Predicción")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Umbrales de Probabilidad ML:**")
        
        umbral_critico = st.slider(
            "Crítico (≥)",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            step=0.01,
            format="%.2f",
            help="Probabilidad mínima para clasificar como riesgo crítico"
        )
        
        umbral_alto = st.slider(
            "Alto (≥)",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.01,
            format="%.2f",
            help="Probabilidad mínima para clasificar como riesgo alto"
        )
        
        umbral_medio = st.slider(
            "Medio (≥)",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.01,
            format="%.2f",
            help="Probabilidad mínima para clasificar como riesgo medio"
        )
    
    with col2:
        st.markdown("**Umbrales de Altman Z-Score:**")
        
        altman_critico = st.slider(
            "Crítico (<)",
            min_value=0.0,
            max_value=5.0,
            value=1.8,
            step=0.1,
            format="%.1f",
            help="Z-Score máximo para clasificar como riesgo crítico"
        )
        
        altman_alto = st.slider(
            "Alto (<)",
            min_value=0.0,
            max_value=5.0,
            value=2.99,
            step=0.1,
            format="%.1f",
            help="Z-Score máximo para clasificar como riesgo alto"
        )
        
        altman_seguro = st.slider(
            "Seguro (>)",
            min_value=0.0,
            max_value=5.0,
            value=2.99,
            step=0.1,
            format="%.1f",
            help="Z-Score mínimo para clasificar como seguro"
        )
    
    # Configuración de blending
    st.markdown("#### ⚖️ Configuración de Blending")
    
    col1, col2 = st.columns(2)
    
    with col1:
        peso_ml = st.slider(
            "Peso Modelo ML",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            format="%.2f",
            help="Peso del modelo ML en el score combinado"
        )
    
    with col2:
        peso_altman = st.slider(
            "Peso Altman Z-Score",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            format="%.2f",
            help="Peso del Altman Z-Score en el score combinado"
        )
    
    # Validar que los pesos sumen 1
    if abs(peso_ml + peso_altman - 1.0) > 0.01:
        st.warning("⚠️ Los pesos deben sumar 1.0")
    
    # Configuración de reentrenamiento
    st.markdown("#### 🔄 Reentrenamiento Automático")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        auto_retrain = st.checkbox(
            "Reentrenamiento Automático",
            value=True,
            help="Reentrenar modelo automáticamente"
        )
    
    with col2:
        retrain_frequency = st.selectbox(
            "Frecuencia",
            ["Semanal", "Mensual", "Trimestral"],
            index=1,
            disabled=not auto_retrain
        )
    
    with col3:
        min_new_data = st.number_input(
            "Mínimo Datos Nuevos",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100,
            disabled=not auto_retrain,
            help="Mínimo de nuevos datos para reentrenar"
        )
    
    # Botón para guardar configuración
    if st.button("💾 Guardar Configuración de Modelos", use_container_width=True):
        config_data = {
            'umbrales_ml': {
                'critico': umbral_critico,
                'alto': umbral_alto,
                'medio': umbral_medio
            },
            'umbrales_altman': {
                'critico': altman_critico,
                'alto': altman_alto,
                'seguro': altman_seguro
            },
            'blending': {
                'peso_ml': peso_ml,
                'peso_altman': peso_altman
            },
            'reentrenamiento': {
                'automatico': auto_retrain,
                'frecuencia': retrain_frequency,
                'min_datos': min_new_data
            }
        }
        
        st.success("✅ Configuración guardada exitosamente (funcionalidad en desarrollo)")

def show_system_configuration():
    """Muestra configuración general del sistema"""
    
    require_role(['admin'])
    
    st.markdown("### ⚙️ Configuración del Sistema")
    
    # Configuración de la aplicación
    st.markdown("#### 📱 Configuración de la Aplicación")
    
    col1, col2 = st.columns(2)
    
    with col1:
        app_name = st.text_input(
            "Nombre de la Aplicación",
            value="Sistema de Predicción de Quiebras",
            help="Nombre que aparece en la interfaz"
        )
        
        max_file_size = st.number_input(
            "Tamaño Máximo de Archivo (MB)",
            min_value=1,
            max_value=500,
            value=100,
            help="Tamaño máximo para cargas ETL"
        )
        
        session_timeout = st.number_input(
            "Timeout de Sesión (minutos)",
            min_value=15,
            max_value=480,
            value=60,
            help="Tiempo antes de cerrar sesión automáticamente"
        )
    
    with col2:
        app_version = st.text_input(
            "Versión de la Aplicación",
            value="1.0.0",
            help="Versión actual del sistema"
        )
        
        max_concurrent_users = st.number_input(
            "Usuarios Concurrentes Máximos",
            min_value=1,
            max_value=1000,
            value=50,
            help="Número máximo de usuarios simultáneos"
        )
        
        backup_frequency = st.selectbox(
            "Frecuencia de Backup",
            ["Diario", "Semanal", "Mensual"],
            index=0,
            help="Frecuencia de respaldo automático"
        )
    
    # Configuración de notificaciones
    st.markdown("#### 📧 Configuración de Notificaciones")
    
    col1, col2 = st.columns(2)
    
    with col1:
        email_enabled = st.checkbox(
            "Notificaciones por Email",
            value=True,
            help="Habilitar envío de emails"
        )
        
        if email_enabled:
            smtp_server = st.text_input(
                "Servidor SMTP",
                value="smtp.gmail.com",
                help="Servidor de correo saliente"
            )
            
            smtp_port = st.number_input(
                "Puerto SMTP",
                min_value=1,
                max_value=65535,
                value=587,
                help="Puerto del servidor SMTP"
            )
    
    with col2:
        slack_enabled = st.checkbox(
            "Notificaciones Slack",
            value=False,
            help="Habilitar notificaciones a Slack"
        )
        
        if slack_enabled:
            slack_webhook = st.text_input(
                "Slack Webhook URL",
                placeholder="https://hooks.slack.com/services/...",
                help="URL del webhook de Slack"
            )
        
        teams_enabled = st.checkbox(
            "Notificaciones Teams",
            value=False,
            help="Habilitar notificaciones a Microsoft Teams"
        )
    
    # Configuración de seguridad
    st.markdown("#### 🔒 Configuración de Seguridad")
    
    col1, col2 = st.columns(2)
    
    with col1:
        password_min_length = st.number_input(
            "Longitud Mínima de Contraseña",
            min_value=6,
            max_value=20,
            value=8,
            help="Caracteres mínimos para contraseñas"
        )
        
        max_login_attempts = st.number_input(
            "Intentos Máximos de Login",
            min_value=3,
            max_value=10,
            value=5,
            help="Intentos antes de bloquear cuenta"
        )
    
    with col2:
        require_strong_passwords = st.checkbox(
            "Contraseñas Fuertes",
            value=True,
            help="Requerir mayúsculas, números y símbolos"
        )
        
        two_factor_auth = st.checkbox(
            "Autenticación de Dos Factores",
            value=False,
            help="Habilitar 2FA para administradores"
        )
    
    # Botón para guardar configuración
    if st.button("💾 Guardar Configuración del Sistema", use_container_width=True):
        system_config = {
            'aplicacion': {
                'nombre': app_name,
                'version': app_version,
                'max_file_size_mb': max_file_size,
                'session_timeout_min': session_timeout,
                'max_concurrent_users': max_concurrent_users,
                'backup_frequency': backup_frequency
            },
            'notificaciones': {
                'email_enabled': email_enabled,
                'smtp_server': smtp_server if email_enabled else None,
                'smtp_port': smtp_port if email_enabled else None,
                'slack_enabled': slack_enabled,
                'slack_webhook': slack_webhook if slack_enabled else None,
                'teams_enabled': teams_enabled
            },
            'seguridad': {
                'password_min_length': password_min_length,
                'max_login_attempts': max_login_attempts,
                'require_strong_passwords': require_strong_passwords,
                'two_factor_auth': two_factor_auth
            }
        }
        
        st.success("✅ Configuración del sistema guardada exitosamente (funcionalidad en desarrollo)")

def show_maintenance_tools():
    """Muestra herramientas de mantenimiento"""
    
    require_role(['admin'])
    
    st.markdown("### 🔧 Herramientas de Mantenimiento")
    
    # Limpieza de datos
    st.markdown("#### 🧹 Limpieza de Datos")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Limpiar Logs Antiguos", use_container_width=True):
            with st.spinner("Limpiando logs antiguos..."):
                # Simulación de limpieza
                st.success("✅ Logs antiguos eliminados (funcionalidad en desarrollo)")
    
    with col2:
        if st.button("📊 Recalcular Estadísticas", use_container_width=True):
            with st.spinner("Recalculando estadísticas..."):
                # Simulación de recálculo
                st.success("✅ Estadísticas recalculadas (funcionalidad en desarrollo)")
    
    with col3:
        if st.button("🔄 Optimizar Base de Datos", use_container_width=True):
            with st.spinner("Optimizando base de datos..."):
                # Simulación de optimización
                st.success("✅ Base de datos optimizada (funcionalidad en desarrollo)")
    
    # Backup y restauración
    st.markdown("#### 💾 Backup y Restauración")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Crear Backup:**")
        
        backup_type = st.selectbox(
            "Tipo de Backup",
            ["Completo", "Solo Datos", "Solo Configuración"],
            index=0
        )
        
        if st.button("💾 Crear Backup", use_container_width=True):
            with st.spinner(f"Creando backup {backup_type.lower()}..."):
                # Simulación de backup
                backup_file = f"backup_{backup_type.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sql"
                st.success(f"✅ Backup creado: {backup_file} (funcionalidad en desarrollo)")
    
    with col2:
        st.markdown("**Restaurar Backup:**")
        
        uploaded_backup = st.file_uploader(
            "Seleccionar archivo de backup",
            type=['sql', 'zip'],
            help="Archivo de backup para restaurar"
        )
        
        if uploaded_backup and st.button("🔄 Restaurar Backup", use_container_width=True):
            st.warning("⚠️ Esta operación sobrescribirá los datos actuales")
            
            if st.button("✅ Confirmar Restauración"):
                with st.spinner("Restaurando backup..."):
                    # Simulación de restauración
                    st.success("✅ Backup restaurado exitosamente (funcionalidad en desarrollo)")
    
    # Monitoreo del sistema
    st.markdown("#### 📊 Monitoreo del Sistema")
    
    # Métricas de rendimiento simuladas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        cpu_usage = np.random.uniform(20, 80)
        st.metric(
            "Uso de CPU",
            f"{cpu_usage:.1f}%",
            delta=f"{np.random.uniform(-5, 5):.1f}% vs ayer"
        )
    
    with col2:
        memory_usage = np.random.uniform(40, 90)
        st.metric(
            "Uso de Memoria",
            f"{memory_usage:.1f}%",
            delta=f"{np.random.uniform(-10, 10):.1f}% vs ayer"
        )
    
    with col3:
        disk_usage = np.random.uniform(30, 70)
        st.metric(
            "Uso de Disco",
            f"{disk_usage:.1f}%",
            delta=f"{np.random.uniform(-2, 2):.1f}% vs ayer"
        )
    
    with col4:
        active_connections = np.random.randint(5, 50)
        st.metric(
            "Conexiones Activas",
            active_connections,
            delta=f"{np.random.randint(-5, 5)} vs ayer"
        )

def show_page():
    """Función principal de la página de configuración"""
    
    # Header de la página
    show_page_header(
        "Configuración y Administración",
        "Gestión del sistema, usuarios y configuración avanzada",
        "⚙️"
    )
    
    # Verificar permisos básicos
    if not has_permission('sistema', 'read'):
        show_permission_error('sistema', 'read')
        return
    
    # Manejo de estados de sesión
    if st.session_state.get('show_user_form'):
        show_user_form()
        return
    
    # Overview del sistema
    show_system_overview()
    
    st.markdown("---")
    
    # Tabs principales
    tab1, tab2, tab3, tab4 = st.tabs(["👥 Usuarios", "🤖 Modelos", "⚙️ Sistema", "🔧 Mantenimiento"])
    
    with tab1:
        show_user_management()
    
    with tab2:
        show_model_configuration()
    
    with tab3:
        show_system_configuration()
    
    with tab4:
        show_maintenance_tools()

if __name__ == "__main__":
    show_page()

