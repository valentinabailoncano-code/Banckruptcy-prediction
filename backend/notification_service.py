"""
Este módulo implementa un sistema completo de notificaciones
que integra email, Slack, Teams y otros canales de comunicación.
"""

import asyncio
import json
import logging
import smtplib
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import requests
import aiohttp
from jinja2 import Template

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NotificationChannel(Enum):
    """Canales de notificación disponibles"""
    EMAIL = "email"
    SLACK = "slack"
    TEAMS = "teams"
    WEBHOOK = "webhook"
    SMS = "sms"

class NotificationPriority(Enum):
    """Prioridades de notificación"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

class NotificationType(Enum):
    """Tipos de notificación"""
    ALERT = "alert"
    REPORT = "report"
    SYSTEM = "system"
    REMINDER = "reminder"
    WELCOME = "welcome"

@dataclass
class NotificationTemplate:
    """Template de notificación"""
    id: str
    name: str
    channel: NotificationChannel
    notification_type: NotificationType
    subject_template: str
    body_template: str
    html_template: str = None
    variables: List[str] = None
    
    def __post_init__(self):
        if self.variables is None:
            self.variables = []

@dataclass
class NotificationRecipient:
    """Destinatario de notificación"""
    id: str
    name: str
    email: str = None
    slack_user_id: str = None
    phone: str = None
    preferences: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.preferences is None:
            self.preferences = {}

@dataclass
class NotificationMessage:
    """Mensaje de notificación"""
    id: str
    channel: NotificationChannel
    notification_type: NotificationType
    priority: NotificationPriority
    recipients: List[NotificationRecipient]
    subject: str
    body: str
    html_body: str = None
    attachments: List[str] = None
    metadata: Dict[str, Any] = None
    scheduled_time: datetime = None
    
    def __post_init__(self):
        if self.attachments is None:
            self.attachments = []
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ChannelConfig:
    """Configuración de canal de notificación"""
    channel: NotificationChannel
    enabled: bool = True
    config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.config is None:
            self.config = {}

class NotificationService:
    """
    Servicio principal de notificaciones multi-canal
    
    Funcionalidades:
    - Envío de notificaciones por múltiples canales
    - Templates personalizables
    - Gestión de destinatarios
    - Programación de envíos
    - Tracking de entrega
    - Fallbacks automáticos
    """
    
    def __init__(self, channel_configs: List[ChannelConfig] = None):
        """
        Inicializa el servicio de notificaciones
        
        Args:
            channel_configs: Configuraciones de canales
        """
        self.channel_configs = {
            config.channel: config for config in (channel_configs or [])
        }
        
        # Templates predefinidos
        self.templates = self._initialize_default_templates()
        
        # Destinatarios registrados
        self.recipients = {}
        
        # Cola de mensajes pendientes
        self.message_queue = []
        
        # Estadísticas de envío
        self.delivery_stats = {
            'sent': 0,
            'failed': 0,
            'by_channel': {},
            'by_type': {}
        }
        
        logger.info("Servicio de notificaciones inicializado")
    
    def _initialize_default_templates(self) -> Dict[str, NotificationTemplate]:
        """Inicializa templates predefinidos"""
        templates = {}
        
        # Template de alerta crítica por email
        templates['critical_alert_email'] = NotificationTemplate(
            id='critical_alert_email',
            name='Alerta Crítica - Email',
            channel=NotificationChannel.EMAIL,
            notification_type=NotificationType.ALERT,
            subject_template='🚨 ALERTA CRÍTICA: {{ empresa_nombre }}',
            body_template='''
Estimado/a {{ recipient_name }},

Se ha detectado una alerta crítica en el sistema de predicción de quiebras:

EMPRESA: {{ empresa_nombre }} (RUT: {{ empresa_rut }})
SECTOR: {{ empresa_sector }}
PROBABILIDAD DE QUIEBRA: {{ probabilidad }}%
BANDA DE RIESGO: {{ banda_riesgo }}

DETALLES:
{{ mensaje_detalle }}

Esta alerta requiere atención inmediata. Por favor, revise el dashboard del sistema para más información.

Fecha y hora: {{ fecha_alerta }}

Saludos,
Sistema de Predicción de Quiebras Empresariales
            ''',
            html_template='''
<html>
<body style="font-family: Arial, sans-serif; margin: 20px;">
    <div style="background-color: #d62728; color: white; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
        <h2>🚨 ALERTA CRÍTICA</h2>
    </div>
    
    <p>Estimado/a <strong>{{ recipient_name }}</strong>,</p>
    
    <p>Se ha detectado una alerta crítica en el sistema de predicción de quiebras:</p>
    
    <div style="background-color: #f8f9fa; padding: 15px; border-left: 4px solid #d62728; margin: 20px 0;">
        <table style="width: 100%; border-collapse: collapse;">
            <tr><td style="font-weight: bold; padding: 5px 0;">EMPRESA:</td><td>{{ empresa_nombre }} (RUT: {{ empresa_rut }})</td></tr>
            <tr><td style="font-weight: bold; padding: 5px 0;">SECTOR:</td><td>{{ empresa_sector }}</td></tr>
            <tr><td style="font-weight: bold; padding: 5px 0;">PROBABILIDAD:</td><td style="color: #d62728; font-weight: bold;">{{ probabilidad }}%</td></tr>
            <tr><td style="font-weight: bold; padding: 5px 0;">BANDA DE RIESGO:</td><td><span style="background-color: #d62728; color: white; padding: 2px 8px; border-radius: 3px;">{{ banda_riesgo }}</span></td></tr>
        </table>
    </div>
    
    <div style="background-color: #fff3cd; padding: 15px; border-radius: 5px; margin: 20px 0;">
        <h4>Detalles:</h4>
        <p>{{ mensaje_detalle }}</p>
    </div>
    
    <p><strong>Esta alerta requiere atención inmediata.</strong> Por favor, revise el dashboard del sistema para más información.</p>
    
    <hr style="margin: 30px 0;">
    <p style="font-size: 12px; color: #666;">
        Fecha y hora: {{ fecha_alerta }}<br>
        Sistema de Predicción de Quiebras Empresariales
    </p>
</body>
</html>
            ''',
            variables=['recipient_name', 'empresa_nombre', 'empresa_rut', 'empresa_sector', 'probabilidad', 'banda_riesgo', 'mensaje_detalle', 'fecha_alerta']
        )
        
        # Template de alerta para Slack
        templates['critical_alert_slack'] = NotificationTemplate(
            id='critical_alert_slack',
            name='Alerta Crítica - Slack',
            channel=NotificationChannel.SLACK,
            notification_type=NotificationType.ALERT,
            subject_template='Alerta Crítica: {{ empresa_nombre }}',
            body_template='''
🚨 *ALERTA CRÍTICA DETECTADA*

*Empresa:* {{ empresa_nombre }} ({{ empresa_rut }})
*Sector:* {{ empresa_sector }}
*Probabilidad de Quiebra:* {{ probabilidad }}%
*Banda de Riesgo:* {{ banda_riesgo }}

*Detalles:*
{{ mensaje_detalle }}

⚠️ Esta alerta requiere atención inmediata.
            ''',
            variables=['empresa_nombre', 'empresa_rut', 'empresa_sector', 'probabilidad', 'banda_riesgo', 'mensaje_detalle']
        )
        
        # Template de reporte semanal
        templates['weekly_report_email'] = NotificationTemplate(
            id='weekly_report_email',
            name='Reporte Semanal - Email',
            channel=NotificationChannel.EMAIL,
            notification_type=NotificationType.REPORT,
            subject_template='📊 Reporte Semanal - Sistema de Predicción de Quiebras',
            body_template='''
Estimado/a {{ recipient_name }},

Se adjunta el reporte semanal del sistema de predicción de quiebras correspondiente al período {{ periodo_inicio }} - {{ periodo_fin }}.

RESUMEN EJECUTIVO:
- Total de empresas analizadas: {{ total_empresas }}
- Predicciones realizadas: {{ total_predicciones }}
- Alertas generadas: {{ total_alertas }}
- Empresas en riesgo alto/crítico: {{ empresas_riesgo_alto }}

PRINCIPALES HALLAZGOS:
{{ principales_hallazgos }}

El reporte completo se encuentra adjunto a este correo.

Saludos cordiales,
Sistema de Predicción de Quiebras Empresariales
            ''',
            variables=['recipient_name', 'periodo_inicio', 'periodo_fin', 'total_empresas', 'total_predicciones', 'total_alertas', 'empresas_riesgo_alto', 'principales_hallazgos']
        )
        
        # Template de bienvenida
        templates['welcome_email'] = NotificationTemplate(
            id='welcome_email',
            name='Bienvenida - Email',
            channel=NotificationChannel.EMAIL,
            notification_type=NotificationType.WELCOME,
            subject_template='Bienvenido/a al Sistema de Predicción de Quiebras',
            body_template='''
Estimado/a {{ recipient_name }},

¡Bienvenido/a al Sistema de Predicción de Quiebras Empresariales!

Su cuenta ha sido creada exitosamente con los siguientes detalles:
- Usuario: {{ username }}
- Rol: {{ user_role }}
- Fecha de activación: {{ activation_date }}

PRIMEROS PASOS:
1. Acceda al sistema en: {{ system_url }}
2. Complete su perfil de usuario
3. Configure sus preferencias de notificación
4. Explore el dashboard principal

RECURSOS DISPONIBLES:
- Manual de usuario: {{ manual_url }}
- Videos tutoriales: {{ tutorials_url }}
- Soporte técnico: {{ support_email }}

Si tiene alguna pregunta, no dude en contactarnos.

Saludos cordiales,
Equipo del Sistema de Predicción de Quiebras
            ''',
            variables=['recipient_name', 'username', 'user_role', 'activation_date', 'system_url', 'manual_url', 'tutorials_url', 'support_email']
        )
        
        return templates
    
    def add_channel_config(self, config: ChannelConfig):
        """Agrega configuración de canal"""
        self.channel_configs[config.channel] = config
        logger.info(f"Configuración agregada para canal: {config.channel.value}")
    
    def add_recipient(self, recipient: NotificationRecipient):
        """Agrega destinatario al registro"""
        self.recipients[recipient.id] = recipient
        logger.info(f"Destinatario agregado: {recipient.name}")
    
    def add_template(self, template: NotificationTemplate):
        """Agrega template personalizado"""
        self.templates[template.id] = template
        logger.info(f"Template agregado: {template.name}")
    
    async def send_notification(self, 
                              template_id: str,
                              recipients: List[Union[str, NotificationRecipient]],
                              variables: Dict[str, Any],
                              priority: NotificationPriority = NotificationPriority.NORMAL,
                              attachments: List[str] = None) -> Dict[str, Any]:
        """
        Envía notificación usando un template
        
        Args:
            template_id: ID del template a usar
            recipients: Lista de destinatarios
            variables: Variables para el template
            priority: Prioridad de la notificación
            attachments: Archivos adjuntos
            
        Returns:
            Resultado del envío
        """
        try:
            # Obtener template
            if template_id not in self.templates:
                raise ValueError(f"Template no encontrado: {template_id}")
            
            template = self.templates[template_id]
            
            # Resolver destinatarios
            resolved_recipients = self._resolve_recipients(recipients)
            
            # Renderizar template
            subject = self._render_template(template.subject_template, variables)
            body = self._render_template(template.body_template, variables)
            html_body = None
            
            if template.html_template:
                html_body = self._render_template(template.html_template, variables)
            
            # Crear mensaje
            message = NotificationMessage(
                id=f"msg_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{template_id}",
                channel=template.channel,
                notification_type=template.notification_type,
                priority=priority,
                recipients=resolved_recipients,
                subject=subject,
                body=body,
                html_body=html_body,
                attachments=attachments or [],
                metadata={'template_id': template_id, 'variables': variables}
            )
            
            # Enviar mensaje
            result = await self._send_message(message)
            
            # Actualizar estadísticas
            self._update_stats(message, result['success'])
            
            return result
            
        except Exception as e:
            logger.error(f"Error enviando notificación: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def send_custom_notification(self,
                                     channel: NotificationChannel,
                                     recipients: List[Union[str, NotificationRecipient]],
                                     subject: str,
                                     body: str,
                                     html_body: str = None,
                                     priority: NotificationPriority = NotificationPriority.NORMAL,
                                     attachments: List[str] = None) -> Dict[str, Any]:
        """
        Envía notificación personalizada sin template
        
        Args:
            channel: Canal de envío
            recipients: Lista de destinatarios
            subject: Asunto del mensaje
            body: Cuerpo del mensaje
            html_body: Cuerpo HTML opcional
            priority: Prioridad de la notificación
            attachments: Archivos adjuntos
            
        Returns:
            Resultado del envío
        """
        try:
            # Resolver destinatarios
            resolved_recipients = self._resolve_recipients(recipients)
            
            # Crear mensaje
            message = NotificationMessage(
                id=f"custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                channel=channel,
                notification_type=NotificationType.SYSTEM,
                priority=priority,
                recipients=resolved_recipients,
                subject=subject,
                body=body,
                html_body=html_body,
                attachments=attachments or [],
                metadata={'custom': True}
            )
            
            # Enviar mensaje
            result = await self._send_message(message)
            
            # Actualizar estadísticas
            self._update_stats(message, result['success'])
            
            return result
            
        except Exception as e:
            logger.error(f"Error enviando notificación personalizada: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _resolve_recipients(self, recipients: List[Union[str, NotificationRecipient]]) -> List[NotificationRecipient]:
        """Resuelve lista de destinatarios"""
        resolved = []
        
        for recipient in recipients:
            if isinstance(recipient, str):
                # Buscar en registro por ID
                if recipient in self.recipients:
                    resolved.append(self.recipients[recipient])
                else:
                    # Asumir que es email
                    resolved.append(NotificationRecipient(
                        id=recipient,
                        name=recipient,
                        email=recipient
                    ))
            elif isinstance(recipient, NotificationRecipient):
                resolved.append(recipient)
        
        return resolved
    
    def _render_template(self, template_str: str, variables: Dict[str, Any]) -> str:
        """Renderiza template con variables"""
        template = Template(template_str)
        return template.render(**variables)
    
    async def _send_message(self, message: NotificationMessage) -> Dict[str, Any]:
        """Envía mensaje por el canal especificado"""
        
        # Verificar si el canal está habilitado
        if message.channel not in self.channel_configs:
            return {'success': False, 'error': f'Canal no configurado: {message.channel.value}'}
        
        config = self.channel_configs[message.channel]
        if not config.enabled:
            return {'success': False, 'error': f'Canal deshabilitado: {message.channel.value}'}
        
        # Enviar según el canal
        if message.channel == NotificationChannel.EMAIL:
            return await self._send_email(message, config)
        elif message.channel == NotificationChannel.SLACK:
            return await self._send_slack(message, config)
        elif message.channel == NotificationChannel.TEAMS:
            return await self._send_teams(message, config)
        elif message.channel == NotificationChannel.WEBHOOK:
            return await self._send_webhook(message, config)
        else:
            return {'success': False, 'error': f'Canal no soportado: {message.channel.value}'}
    
    async def _send_email(self, message: NotificationMessage, config: ChannelConfig) -> Dict[str, Any]:
        """Envía notificación por email"""
        
        try:
            smtp_config = config.config
            
            # Crear mensaje MIME
            msg = MIMEMultipart('alternative')
            msg['From'] = smtp_config.get('from_email', 'noreply@sistema.com')
            msg['Subject'] = message.subject
            
            # Agregar destinatarios
            email_recipients = [r.email for r in message.recipients if r.email]
            if not email_recipients:
                return {'success': False, 'error': 'No hay destinatarios con email válido'}
            
            msg['To'] = ", ".join(email_recipients)
            
            # Agregar cuerpo del mensaje
            msg.attach(MIMEText(message.body, 'plain', 'utf-8'))
            
            if message.html_body:
                msg.attach(MIMEText(message.html_body, 'html', 'utf-8'))
            
            # Agregar archivos adjuntos
            for attachment_path in message.attachments:
                try:
                    with open(attachment_path, "rb") as attachment:
                        part = MIMEBase('application', 'octet-stream')
                        part.set_payload(attachment.read())
                    
                    encoders.encode_base64(part)
                    part.add_header(
                        'Content-Disposition',
                        f'attachment; filename= {attachment_path.split("/")[-1]}'
                    )
                    msg.attach(part)
                except Exception as e:
                    logger.warning(f"Error adjuntando archivo {attachment_path}: {str(e)}")
            
            # Enviar email
            server = smtplib.SMTP(smtp_config['smtp_server'], smtp_config.get('smtp_port', 587))
            
            if smtp_config.get('use_tls', True):
                server.starttls()
            
            if smtp_config.get('username') and smtp_config.get('password'):
                server.login(smtp_config['username'], smtp_config['password'])
            
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email enviado exitosamente a {len(email_recipients)} destinatarios")
            return {'success': True, 'recipients_count': len(email_recipients)}
            
        except Exception as e:
            logger.error(f"Error enviando email: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def _send_slack(self, message: NotificationMessage, config: ChannelConfig) -> Dict[str, Any]:
        """Envía notificación a Slack"""
        
        try:
            slack_config = config.config
            webhook_url = slack_config.get('webhook_url')
            
            if not webhook_url:
                return {'success': False, 'error': 'Webhook URL no configurado para Slack'}
            
            # Determinar color según prioridad
            color_map = {
                NotificationPriority.LOW: "#36a64f",      # Verde
                NotificationPriority.NORMAL: "#1f77b4",   # Azul
                NotificationPriority.HIGH: "#ff7f0e",     # Naranja
                NotificationPriority.URGENT: "#d62728"    # Rojo
            }
            
            # Crear payload para Slack
            payload = {
                "channel": slack_config.get('channel', '#general'),
                "username": slack_config.get('username', 'Sistema de Predicción'),
                "icon_emoji": slack_config.get('icon_emoji', ':warning:'),
                "attachments": [
                    {
                        "color": color_map.get(message.priority, "#1f77b4"),
                        "title": message.subject,
                        "text": message.body,
                        "footer": "Sistema de Predicción de Quiebras",
                        "ts": int(datetime.now().timestamp())
                    }
                ]
            }
            
            # Enviar a Slack
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    if response.status == 200:
                        logger.info("Notificación Slack enviada exitosamente")
                        return {'success': True, 'channel': slack_config.get('channel')}
                    else:
                        error_text = await response.text()
                        return {'success': False, 'error': f'Error HTTP {response.status}: {error_text}'}
            
        except Exception as e:
            logger.error(f"Error enviando notificación Slack: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def _send_teams(self, message: NotificationMessage, config: ChannelConfig) -> Dict[str, Any]:
        """Envía notificación a Microsoft Teams"""
        
        try:
            teams_config = config.config
            webhook_url = teams_config.get('webhook_url')
            
            if not webhook_url:
                return {'success': False, 'error': 'Webhook URL no configurado para Teams'}
            
            # Determinar color según prioridad
            color_map = {
                NotificationPriority.LOW: "00FF00",      # Verde
                NotificationPriority.NORMAL: "0078D4",   # Azul
                NotificationPriority.HIGH: "FF8C00",     # Naranja
                NotificationPriority.URGENT: "FF0000"    # Rojo
            }
            
            # Crear payload para Teams
            payload = {
                "@type": "MessageCard",
                "@context": "http://schema.org/extensions",
                "themeColor": color_map.get(message.priority, "0078D4"),
                "summary": message.subject,
                "sections": [
                    {
                        "activityTitle": message.subject,
                        "activitySubtitle": f"Prioridad: {message.priority.value.upper()}",
                        "text": message.body,
                        "markdown": True
                    }
                ],
                "potentialAction": [
                    {
                        "@type": "OpenUri",
                        "name": "Ver Dashboard",
                        "targets": [
                            {
                                "os": "default",
                                "uri": teams_config.get('dashboard_url', 'https://dashboard.sistema.com')
                            }
                        ]
                    }
                ]
            }
            
            # Enviar a Teams
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    if response.status == 200:
                        logger.info("Notificación Teams enviada exitosamente")
                        return {'success': True}
                    else:
                        error_text = await response.text()
                        return {'success': False, 'error': f'Error HTTP {response.status}: {error_text}'}
            
        except Exception as e:
            logger.error(f"Error enviando notificación Teams: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def _send_webhook(self, message: NotificationMessage, config: ChannelConfig) -> Dict[str, Any]:
        """Envía notificación via webhook personalizado"""
        
        try:
            webhook_config = config.config
            webhook_url = webhook_config.get('url')
            
            if not webhook_url:
                return {'success': False, 'error': 'URL no configurado para webhook'}
            
            # Crear payload
            payload = {
                'id': message.id,
                'channel': message.channel.value,
                'type': message.notification_type.value,
                'priority': message.priority.value,
                'subject': message.subject,
                'body': message.body,
                'html_body': message.html_body,
                'recipients': [asdict(r) for r in message.recipients],
                'metadata': message.metadata,
                'timestamp': datetime.now().isoformat()
            }
            
            # Headers personalizados
            headers = webhook_config.get('headers', {})
            headers['Content-Type'] = 'application/json'
            
            # Enviar webhook
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload, headers=headers) as response:
                    if response.status in [200, 201, 202]:
                        logger.info("Webhook enviado exitosamente")
                        return {'success': True, 'status_code': response.status}
                    else:
                        error_text = await response.text()
                        return {'success': False, 'error': f'Error HTTP {response.status}: {error_text}'}
            
        except Exception as e:
            logger.error(f"Error enviando webhook: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _update_stats(self, message: NotificationMessage, success: bool):
        """Actualiza estadísticas de envío"""
        if success:
            self.delivery_stats['sent'] += 1
        else:
            self.delivery_stats['failed'] += 1
        
        # Por canal
        channel_key = message.channel.value
        if channel_key not in self.delivery_stats['by_channel']:
            self.delivery_stats['by_channel'][channel_key] = {'sent': 0, 'failed': 0}
        
        if success:
            self.delivery_stats['by_channel'][channel_key]['sent'] += 1
        else:
            self.delivery_stats['by_channel'][channel_key]['failed'] += 1
        
        # Por tipo
        type_key = message.notification_type.value
        if type_key not in self.delivery_stats['by_type']:
            self.delivery_stats['by_type'][type_key] = {'sent': 0, 'failed': 0}
        
        if success:
            self.delivery_stats['by_type'][type_key]['sent'] += 1
        else:
            self.delivery_stats['by_type'][type_key]['failed'] += 1
    
    def get_delivery_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de entrega"""
        return self.delivery_stats.copy()
    
    def get_templates(self) -> List[Dict[str, Any]]:
        """Obtiene lista de templates disponibles"""
        return [
            {
                'id': template.id,
                'name': template.name,
                'channel': template.channel.value,
                'type': template.notification_type.value,
                'variables': template.variables
            }
            for template in self.templates.values()
        ]
    
    def get_recipients(self) -> List[Dict[str, Any]]:
        """Obtiene lista de destinatarios registrados"""
        return [asdict(recipient) for recipient in self.recipients.values()]

# Funciones de utilidad
def create_notification_service(channel_configs: List[ChannelConfig] = None) -> NotificationService:
    """
    Factory function para crear servicio de notificaciones
    
    Args:
        channel_configs: Configuraciones de canales
        
    Returns:
        Instancia del servicio configurada
    """
    return NotificationService(channel_configs)

def create_email_config(smtp_server: str,
                       smtp_port: int,
                       username: str,
                       password: str,
                       from_email: str,
                       use_tls: bool = True) -> ChannelConfig:
    """
    Crea configuración para canal de email
    
    Args:
        smtp_server: Servidor SMTP
        smtp_port: Puerto SMTP
        username: Usuario SMTP
        password: Contraseña SMTP
        from_email: Email remitente
        use_tls: Usar TLS
        
    Returns:
        Configuración de canal
    """
    return ChannelConfig(
        channel=NotificationChannel.EMAIL,
        enabled=True,
        config={
            'smtp_server': smtp_server,
            'smtp_port': smtp_port,
            'username': username,
            'password': password,
            'from_email': from_email,
            'use_tls': use_tls
        }
    )

def create_slack_config(webhook_url: str,
                       channel: str = '#general',
                       username: str = 'Sistema de Predicción',
                       icon_emoji: str = ':warning:') -> ChannelConfig:
    """
    Crea configuración para canal de Slack
    
    Args:
        webhook_url: URL del webhook de Slack
        channel: Canal de destino
        username: Nombre de usuario del bot
        icon_emoji: Emoji del bot
        
    Returns:
        Configuración de canal
    """
    return ChannelConfig(
        channel=NotificationChannel.SLACK,
        enabled=True,
        config={
            'webhook_url': webhook_url,
            'channel': channel,
            'username': username,
            'icon_emoji': icon_emoji
        }
    )

def create_teams_config(webhook_url: str,
                       dashboard_url: str = None) -> ChannelConfig:
    """
    Crea configuración para canal de Teams
    
    Args:
        webhook_url: URL del webhook de Teams
        dashboard_url: URL del dashboard para botón de acción
        
    Returns:
        Configuración de canal
    """
    return ChannelConfig(
        channel=NotificationChannel.TEAMS,
        enabled=True,
        config={
            'webhook_url': webhook_url,
            'dashboard_url': dashboard_url or 'https://dashboard.sistema.com'
        }
    )

