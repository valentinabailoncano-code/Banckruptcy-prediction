"""
Este módulo implementa un sistema completo de alertas automatizadas
para detección proactiva de riesgos y notificaciones en tiempo real.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from sqlalchemy.orm import sessionmaker

from database.connection import DatabaseManager
from database.models import Empresa, Prediccion, Alerta, DatoFinanciero
from services.prediction.predictor import PredictionService

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Niveles de severidad de alertas"""
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class AlertType(Enum):
    """Tipos de alertas del sistema"""
    RIESGO_ALTO = "riesgo_alto"
    DETERIORO_FINANCIERO = "deterioro_financiero"
    ANOMALIA_DATOS = "anomalia_datos"
    MODELO_DRIFT = "modelo_drift"
    SISTEMA_ERROR = "sistema_error"
    THRESHOLD_BREACH = "threshold_breach"
    TREND_ALERT = "trend_alert"

class AlertStatus(Enum):
    """Estados de las alertas"""
    ACTIVA = "activa"
    RESUELTA = "resuelta"
    DESCARTADA = "descartada"
    EN_PROCESO = "en_proceso"

@dataclass
class AlertRule:
    """Regla de alerta configurable"""
    id: str
    name: str
    description: str
    alert_type: AlertType
    severity: AlertSeverity
    condition: str  # Expresión SQL o Python
    threshold_value: float = None
    comparison_operator: str = None  # >, <, >=, <=, ==, !=
    time_window_minutes: int = 60
    cooldown_minutes: int = 30
    enabled: bool = True
    escalation_rules: List[Dict[str, Any]] = None
    notification_channels: List[str] = None
    
    def __post_init__(self):
        if self.escalation_rules is None:
            self.escalation_rules = []
        if self.notification_channels is None:
            self.notification_channels = ["email"]

@dataclass
class AlertEvent:
    """Evento de alerta generado"""
    rule_id: str
    empresa_id: int
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    data: Dict[str, Any]
    timestamp: datetime
    prediccion_id: int = None
    dato_financiero_id: int = None

@dataclass
class NotificationConfig:
    """Configuración de notificaciones"""
    email_enabled: bool = True
    email_smtp_server: str = "smtp.gmail.com"
    email_smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    email_from: str = ""
    email_recipients: List[str] = None
    
    slack_enabled: bool = False
    slack_webhook_url: str = ""
    slack_channel: str = "#alerts"
    
    teams_enabled: bool = False
    teams_webhook_url: str = ""
    
    def __post_init__(self):
        if self.email_recipients is None:
            self.email_recipients = []

class AlertEngine:
    """
    Motor principal de alertas automatizadas
    
    Funcionalidades:
    - Evaluación continua de reglas de alerta
    - Detección proactiva de riesgos
    - Escalamiento automático de alertas
    - Notificaciones multi-canal
    - Gestión de cooldowns y duplicados
    """
    
    def __init__(self, notification_config: NotificationConfig = None):
        """
        Inicializa el motor de alertas
        
        Args:
            notification_config: Configuración de notificaciones
        """
        self.db_manager = DatabaseManager()
        self.prediction_service = PredictionService()
        self.notification_config = notification_config or NotificationConfig()
        
        # Reglas de alerta predefinidas
        self.alert_rules = self._initialize_default_rules()
        
        # Cache para cooldowns
        self._cooldown_cache = {}
        
        # Estado del motor
        self.is_running = False
        self.evaluation_interval = 300  # 5 minutos por defecto
        
        logger.info("Alert Engine inicializado")
    
    def _initialize_default_rules(self) -> List[AlertRule]:
        """Inicializa reglas de alerta predefinidas"""
        return [
            # Alerta de riesgo crítico
            AlertRule(
                id="critical_risk_alert",
                name="Riesgo Crítico Detectado",
                description="Empresa con probabilidad de quiebra crítica",
                alert_type=AlertType.RIESGO_ALTO,
                severity=AlertSeverity.CRITICAL,
                condition="probabilidad_combinada >= 0.8",
                threshold_value=0.8,
                comparison_operator=">=",
                time_window_minutes=60,
                cooldown_minutes=120,
                notification_channels=["email", "slack"]
            ),
            
            # Alerta de deterioro financiero
            AlertRule(
                id="financial_deterioration",
                name="Deterioro Financiero Significativo",
                description="Deterioro significativo en ratios financieros",
                alert_type=AlertType.DETERIORO_FINANCIERO,
                severity=AlertSeverity.HIGH,
                condition="altman_z_score < 1.8 AND ratio_liquidez_corriente < 1.0",
                threshold_value=1.8,
                comparison_operator="<",
                time_window_minutes=1440,  # 24 horas
                cooldown_minutes=720,  # 12 horas
                notification_channels=["email"]
            ),
            
            # Alerta de tendencia negativa
            AlertRule(
                id="negative_trend_alert",
                name="Tendencia Negativa Sostenida",
                description="Tendencia negativa en múltiples períodos",
                alert_type=AlertType.TREND_ALERT,
                severity=AlertSeverity.MEDIUM,
                condition="trend_slope < -0.1 AND trend_periods >= 3",
                time_window_minutes=2880,  # 48 horas
                cooldown_minutes=1440,  # 24 horas
                notification_channels=["email"]
            ),
            
            # Alerta de anomalía en datos
            AlertRule(
                id="data_anomaly_alert",
                name="Anomalía en Datos Financieros",
                description="Valores anómalos detectados en datos financieros",
                alert_type=AlertType.ANOMALIA_DATOS,
                severity=AlertSeverity.MEDIUM,
                condition="data_quality_score < 0.7",
                threshold_value=0.7,
                comparison_operator="<",
                time_window_minutes=60,
                cooldown_minutes=180,
                notification_channels=["email"]
            ),
            
            # Alerta de drift del modelo
            AlertRule(
                id="model_drift_alert",
                name="Drift del Modelo Detectado",
                description="Degradación en el rendimiento del modelo",
                alert_type=AlertType.MODELO_DRIFT,
                severity=AlertSeverity.HIGH,
                condition="model_performance_drop > 0.1",
                threshold_value=0.1,
                comparison_operator=">",
                time_window_minutes=1440,  # 24 horas
                cooldown_minutes=720,  # 12 horas
                notification_channels=["email", "slack"]
            )
        ]
    
    async def start_monitoring(self):
        """Inicia el monitoreo continuo de alertas"""
        if self.is_running:
            logger.warning("Alert Engine ya está ejecutándose")
            return
        
        self.is_running = True
        logger.info("Iniciando monitoreo de alertas")
        
        try:
            while self.is_running:
                await self._evaluate_all_rules()
                await asyncio.sleep(self.evaluation_interval)
        except Exception as e:
            logger.error(f"Error en monitoreo de alertas: {str(e)}")
            self.is_running = False
            raise
    
    def stop_monitoring(self):
        """Detiene el monitoreo de alertas"""
        self.is_running = False
        logger.info("Monitoreo de alertas detenido")
    
    async def _evaluate_all_rules(self):
        """Evalúa todas las reglas de alerta activas"""
        logger.debug("Evaluando reglas de alerta")
        
        for rule in self.alert_rules:
            if not rule.enabled:
                continue
            
            try:
                await self._evaluate_rule(rule)
            except Exception as e:
                logger.error(f"Error evaluando regla {rule.id}: {str(e)}")
    
    async def _evaluate_rule(self, rule: AlertRule):
        """
        Evalúa una regla de alerta específica
        
        Args:
            rule: Regla a evaluar
        """
        # Verificar cooldown
        if self._is_in_cooldown(rule.id):
            return
        
        # Obtener datos para evaluación
        candidates = await self._get_evaluation_candidates(rule)
        
        for candidate in candidates:
            if await self._evaluate_condition(rule, candidate):
                # Crear evento de alerta
                alert_event = AlertEvent(
                    rule_id=rule.id,
                    empresa_id=candidate['empresa_id'],
                    alert_type=rule.alert_type,
                    severity=rule.severity,
                    title=rule.name,
                    message=self._generate_alert_message(rule, candidate),
                    data=candidate,
                    timestamp=datetime.now(),
                    prediccion_id=candidate.get('prediccion_id'),
                    dato_financiero_id=candidate.get('dato_financiero_id')
                )
                
                # Procesar alerta
                await self._process_alert_event(alert_event, rule)
                
                # Activar cooldown
                self._activate_cooldown(rule.id, rule.cooldown_minutes)
    
    async def _get_evaluation_candidates(self, rule: AlertRule) -> List[Dict[str, Any]]:
        """
        Obtiene candidatos para evaluación de una regla
        
        Args:
            rule: Regla de alerta
            
        Returns:
            Lista de candidatos con datos relevantes
        """
        candidates = []
        
        try:
            with self.db_manager.get_session() as session:
                # Determinar ventana de tiempo
                time_threshold = datetime.now() - timedelta(minutes=rule.time_window_minutes)
                
                if rule.alert_type in [AlertType.RIESGO_ALTO, AlertType.TREND_ALERT]:
                    # Consultar predicciones recientes
                    query = session.query(
                        Prediccion,
                        Empresa.rut,
                        Empresa.razon_social,
                        Empresa.sector,
                        DatoFinanciero.altman_z_score,
                        DatoFinanciero.ratio_liquidez_corriente
                    ).join(Empresa, Prediccion.empresa_id == Empresa.id)\
                     .join(DatoFinanciero, Prediccion.dato_financiero_id == DatoFinanciero.id)\
                     .filter(Prediccion.fecha_prediccion >= time_threshold)
                    
                    for pred, rut, razon_social, sector, altman, liquidez in query.all():
                        candidates.append({
                            'empresa_id': pred.empresa_id,
                            'prediccion_id': pred.id,
                            'dato_financiero_id': pred.dato_financiero_id,
                            'rut': rut,
                            'razon_social': razon_social,
                            'sector': sector,
                            'probabilidad_ml': pred.probabilidad_ml,
                            'probabilidad_combinada': pred.probabilidad_combinada,
                            'banda_riesgo': pred.banda_riesgo,
                            'altman_z_score': altman,
                            'ratio_liquidez_corriente': liquidez,
                            'fecha_prediccion': pred.fecha_prediccion
                        })
                
                elif rule.alert_type == AlertType.DETERIORO_FINANCIERO:
                    # Consultar datos financieros recientes
                    query = session.query(
                        DatoFinanciero,
                        Empresa.rut,
                        Empresa.razon_social,
                        Empresa.sector
                    ).join(Empresa, DatoFinanciero.empresa_id == Empresa.id)\
                     .filter(DatoFinanciero.fecha_carga >= time_threshold)
                    
                    for dato, rut, razon_social, sector in query.all():
                        candidates.append({
                            'empresa_id': dato.empresa_id,
                            'dato_financiero_id': dato.id,
                            'rut': rut,
                            'razon_social': razon_social,
                            'sector': sector,
                            'altman_z_score': dato.altman_z_score,
                            'ratio_liquidez_corriente': dato.ratio_liquidez_corriente,
                            'ratio_endeudamiento': dato.ratio_endeudamiento,
                            'fecha_carga': dato.fecha_carga
                        })
                
                # Agregar cálculos adicionales según el tipo de regla
                if rule.alert_type == AlertType.TREND_ALERT:
                    candidates = await self._add_trend_analysis(candidates, session)
                elif rule.alert_type == AlertType.ANOMALIA_DATOS:
                    candidates = await self._add_data_quality_analysis(candidates)
                
        except Exception as e:
            logger.error(f"Error obteniendo candidatos para regla {rule.id}: {str(e)}")
        
        return candidates
    
    async def _evaluate_condition(self, rule: AlertRule, candidate: Dict[str, Any]) -> bool:
        """
        Evalúa si un candidato cumple la condición de la regla
        
        Args:
            rule: Regla de alerta
            candidate: Datos del candidato
            
        Returns:
            True si cumple la condición
        """
        try:
            # Crear contexto para evaluación
            context = candidate.copy()
            
            # Evaluar condición
            if rule.condition:
                # Reemplazar variables en la condición
                condition = rule.condition
                for key, value in context.items():
                    if isinstance(value, (int, float)):
                        condition = condition.replace(key, str(value))
                    elif value is None:
                        condition = condition.replace(key, "None")
                
                # Evaluar expresión (en producción usar un evaluador más seguro)
                try:
                    result = eval(condition, {"__builtins__": {}}, {})
                    return bool(result)
                except:
                    logger.warning(f"Error evaluando condición: {condition}")
                    return False
            
            # Evaluación simple por threshold
            if rule.threshold_value is not None and rule.comparison_operator:
                value_key = self._get_value_key_for_rule(rule)
                if value_key in candidate:
                    value = candidate[value_key]
                    if value is not None:
                        return self._compare_values(value, rule.comparison_operator, rule.threshold_value)
            
            return False
            
        except Exception as e:
            logger.error(f"Error evaluando condición para regla {rule.id}: {str(e)}")
            return False
    
    def _get_value_key_for_rule(self, rule: AlertRule) -> str:
        """Obtiene la clave del valor a evaluar según el tipo de regla"""
        mapping = {
            AlertType.RIESGO_ALTO: "probabilidad_combinada",
            AlertType.DETERIORO_FINANCIERO: "altman_z_score",
            AlertType.ANOMALIA_DATOS: "data_quality_score",
            AlertType.MODELO_DRIFT: "model_performance_drop",
            AlertType.TREND_ALERT: "trend_slope"
        }
        return mapping.get(rule.alert_type, "value")
    
    def _compare_values(self, value: float, operator: str, threshold: float) -> bool:
        """Compara valores según el operador especificado"""
        if operator == ">":
            return value > threshold
        elif operator == ">=":
            return value >= threshold
        elif operator == "<":
            return value < threshold
        elif operator == "<=":
            return value <= threshold
        elif operator == "==":
            return value == threshold
        elif operator == "!=":
            return value != threshold
        return False
    
    async def _add_trend_analysis(self, candidates: List[Dict[str, Any]], session) -> List[Dict[str, Any]]:
        """Agrega análisis de tendencias a los candidatos"""
        for candidate in candidates:
            try:
                # Obtener predicciones históricas de la empresa
                empresa_id = candidate['empresa_id']
                historical_predictions = session.query(Prediccion)\
                    .filter(Prediccion.empresa_id == empresa_id)\
                    .order_by(Prediccion.fecha_prediccion.desc())\
                    .limit(5).all()
                
                if len(historical_predictions) >= 3:
                    # Calcular tendencia simple
                    probabilities = [p.probabilidad_combinada for p in historical_predictions]
                    trend_slope = self._calculate_trend_slope(probabilities)
                    
                    candidate['trend_slope'] = trend_slope
                    candidate['trend_periods'] = len(historical_predictions)
                else:
                    candidate['trend_slope'] = 0
                    candidate['trend_periods'] = 0
                    
            except Exception as e:
                logger.error(f"Error calculando tendencia para empresa {candidate['empresa_id']}: {str(e)}")
                candidate['trend_slope'] = 0
                candidate['trend_periods'] = 0
        
        return candidates
    
    def _calculate_trend_slope(self, values: List[float]) -> float:
        """Calcula la pendiente de tendencia simple"""
        if len(values) < 2:
            return 0
        
        n = len(values)
        x_values = list(range(n))
        
        # Regresión lineal simple
        x_mean = sum(x_values) / n
        y_mean = sum(values) / n
        
        numerator = sum((x_values[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0
        
        return numerator / denominator
    
    async def _add_data_quality_analysis(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Agrega análisis de calidad de datos a los candidatos"""
        for candidate in candidates:
            try:
                # Calcular score de calidad de datos simple
                quality_score = 1.0
                
                # Verificar valores nulos o anómalos
                financial_fields = ['altman_z_score', 'ratio_liquidez_corriente', 'ratio_endeudamiento']
                null_count = sum(1 for field in financial_fields if candidate.get(field) is None)
                
                if null_count > 0:
                    quality_score -= (null_count / len(financial_fields)) * 0.5
                
                # Verificar valores extremos
                if candidate.get('altman_z_score'):
                    if candidate['altman_z_score'] < -10 or candidate['altman_z_score'] > 50:
                        quality_score -= 0.3
                
                candidate['data_quality_score'] = max(0, quality_score)
                
            except Exception as e:
                logger.error(f"Error calculando calidad de datos: {str(e)}")
                candidate['data_quality_score'] = 0.5
        
        return candidates
    
    def _generate_alert_message(self, rule: AlertRule, candidate: Dict[str, Any]) -> str:
        """Genera mensaje personalizado para la alerta"""
        empresa_info = f"{candidate.get('razon_social', 'N/A')} (RUT: {candidate.get('rut', 'N/A')})"
        
        if rule.alert_type == AlertType.RIESGO_ALTO:
            prob = candidate.get('probabilidad_combinada', 0)
            return f"La empresa {empresa_info} presenta un riesgo crítico de quiebra con probabilidad de {prob:.1%}."
        
        elif rule.alert_type == AlertType.DETERIORO_FINANCIERO:
            altman = candidate.get('altman_z_score', 0)
            liquidez = candidate.get('ratio_liquidez_corriente', 0)
            return f"Deterioro financiero detectado en {empresa_info}. Altman Z-Score: {altman:.2f}, Liquidez: {liquidez:.2f}."
        
        elif rule.alert_type == AlertType.TREND_ALERT:
            slope = candidate.get('trend_slope', 0)
            periods = candidate.get('trend_periods', 0)
            return f"Tendencia negativa sostenida en {empresa_info} durante {periods} períodos (pendiente: {slope:.3f})."
        
        elif rule.alert_type == AlertType.ANOMALIA_DATOS:
            quality = candidate.get('data_quality_score', 0)
            return f"Anomalía en datos financieros de {empresa_info}. Score de calidad: {quality:.2f}."
        
        else:
            return f"Alerta {rule.name} activada para {empresa_info}."
    
    async def _process_alert_event(self, event: AlertEvent, rule: AlertRule):
        """
        Procesa un evento de alerta
        
        Args:
            event: Evento de alerta
            rule: Regla que generó el evento
        """
        try:
            # Guardar alerta en base de datos
            alert_id = await self._save_alert_to_database(event)
            
            # Enviar notificaciones
            await self._send_notifications(event, rule)
            
            # Procesar escalamiento si aplica
            await self._process_escalation(event, rule)
            
            logger.info(f"Alerta procesada: {event.title} para empresa {event.empresa_id}")
            
        except Exception as e:
            logger.error(f"Error procesando alerta: {str(e)}")
    
    async def _save_alert_to_database(self, event: AlertEvent) -> int:
        """Guarda la alerta en la base de datos"""
        try:
            with self.db_manager.get_session() as session:
                alerta = Alerta(
                    empresa_id=event.empresa_id,
                    prediccion_id=event.prediccion_id,
                    dato_financiero_id=event.dato_financiero_id,
                    tipo=event.alert_type.value,
                    severidad=event.severity.value,
                    estado=AlertStatus.ACTIVA.value,
                    titulo=event.title,
                    mensaje=event.message,
                    detalles=json.dumps(event.data, default=str),
                    fecha_creacion=event.timestamp
                )
                
                session.add(alerta)
                session.commit()
                
                return alerta.id
                
        except Exception as e:
            logger.error(f"Error guardando alerta en BD: {str(e)}")
            raise
    
    async def _send_notifications(self, event: AlertEvent, rule: AlertRule):
        """Envía notificaciones según los canales configurados"""
        for channel in rule.notification_channels:
            try:
                if channel == "email" and self.notification_config.email_enabled:
                    await self._send_email_notification(event)
                elif channel == "slack" and self.notification_config.slack_enabled:
                    await self._send_slack_notification(event)
                elif channel == "teams" and self.notification_config.teams_enabled:
                    await self._send_teams_notification(event)
            except Exception as e:
                logger.error(f"Error enviando notificación por {channel}: {str(e)}")
    
    async def _send_email_notification(self, event: AlertEvent):
        """Envía notificación por email"""
        if not self.notification_config.email_recipients:
            return
        
        try:
            # Crear mensaje
            msg = MIMEMultipart()
            msg['From'] = self.notification_config.email_from
            msg['To'] = ", ".join(self.notification_config.email_recipients)
            msg['Subject'] = f"[{event.severity.value}] {event.title}"
            
            # Cuerpo del mensaje
            body = f"""
            Alerta del Sistema de Predicción de Quiebras
            
            Severidad: {event.severity.value}
            Tipo: {event.alert_type.value}
            Empresa: {event.data.get('razon_social', 'N/A')} (RUT: {event.data.get('rut', 'N/A')})
            Fecha: {event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
            
            Mensaje:
            {event.message}
            
            Datos adicionales:
            {json.dumps(event.data, indent=2, default=str)}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Enviar email
            server = smtplib.SMTP(self.notification_config.email_smtp_server, self.notification_config.email_smtp_port)
            server.starttls()
            server.login(self.notification_config.email_username, self.notification_config.email_password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email enviado para alerta: {event.title}")
            
        except Exception as e:
            logger.error(f"Error enviando email: {str(e)}")
    
    async def _send_slack_notification(self, event: AlertEvent):
        """Envía notificación a Slack"""
        try:
            # Mapear colores por severidad
            color_map = {
                AlertSeverity.CRITICAL: "#d62728",
                AlertSeverity.HIGH: "#ff7f0e",
                AlertSeverity.MEDIUM: "#ffbb78",
                AlertSeverity.LOW: "#2ca02c",
                AlertSeverity.INFO: "#1f77b4"
            }
            
            payload = {
                "channel": self.notification_config.slack_channel,
                "username": "Sistema de Predicción de Quiebras",
                "icon_emoji": ":warning:",
                "attachments": [
                    {
                        "color": color_map.get(event.severity, "#666666"),
                        "title": event.title,
                        "text": event.message,
                        "fields": [
                            {
                                "title": "Severidad",
                                "value": event.severity.value,
                                "short": True
                            },
                            {
                                "title": "Empresa",
                                "value": f"{event.data.get('razon_social', 'N/A')} ({event.data.get('rut', 'N/A')})",
                                "short": True
                            },
                            {
                                "title": "Fecha",
                                "value": event.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                                "short": True
                            }
                        ],
                        "footer": "Sistema de Predicción de Quiebras",
                        "ts": int(event.timestamp.timestamp())
                    }
                ]
            }
            
            response = requests.post(self.notification_config.slack_webhook_url, json=payload)
            response.raise_for_status()
            
            logger.info(f"Notificación Slack enviada para alerta: {event.title}")
            
        except Exception as e:
            logger.error(f"Error enviando notificación Slack: {str(e)}")
    
    async def _send_teams_notification(self, event: AlertEvent):
        """Envía notificación a Microsoft Teams"""
        try:
            payload = {
                "@type": "MessageCard",
                "@context": "http://schema.org/extensions",
                "themeColor": "FF0000" if event.severity == AlertSeverity.CRITICAL else "FFA500",
                "summary": event.title,
                "sections": [
                    {
                        "activityTitle": event.title,
                        "activitySubtitle": f"Severidad: {event.severity.value}",
                        "activityImage": "https://example.com/alert-icon.png",
                        "facts": [
                            {
                                "name": "Empresa",
                                "value": f"{event.data.get('razon_social', 'N/A')} ({event.data.get('rut', 'N/A')})"
                            },
                            {
                                "name": "Tipo",
                                "value": event.alert_type.value
                            },
                            {
                                "name": "Fecha",
                                "value": event.timestamp.strftime('%Y-%m-%d %H:%M:%S')
                            }
                        ],
                        "markdown": True,
                        "text": event.message
                    }
                ]
            }
            
            response = requests.post(self.notification_config.teams_webhook_url, json=payload)
            response.raise_for_status()
            
            logger.info(f"Notificación Teams enviada para alerta: {event.title}")
            
        except Exception as e:
            logger.error(f"Error enviando notificación Teams: {str(e)}")
    
    async def _process_escalation(self, event: AlertEvent, rule: AlertRule):
        """Procesa reglas de escalamiento de alertas"""
        # Implementar lógica de escalamiento según configuración
        # Por ejemplo, escalar a nivel superior después de cierto tiempo
        pass
    
    def _is_in_cooldown(self, rule_id: str) -> bool:
        """Verifica si una regla está en período de cooldown"""
        if rule_id not in self._cooldown_cache:
            return False
        
        cooldown_until = self._cooldown_cache[rule_id]
        return datetime.now() < cooldown_until
    
    def _activate_cooldown(self, rule_id: str, cooldown_minutes: int):
        """Activa el período de cooldown para una regla"""
        cooldown_until = datetime.now() + timedelta(minutes=cooldown_minutes)
        self._cooldown_cache[rule_id] = cooldown_until
    
    def add_custom_rule(self, rule: AlertRule):
        """Agrega una regla personalizada"""
        self.alert_rules.append(rule)
        logger.info(f"Regla personalizada agregada: {rule.name}")
    
    def remove_rule(self, rule_id: str):
        """Remueve una regla por ID"""
        self.alert_rules = [r for r in self.alert_rules if r.id != rule_id]
        logger.info(f"Regla removida: {rule_id}")
    
    def get_rule_status(self) -> List[Dict[str, Any]]:
        """Obtiene el estado de todas las reglas"""
        return [
            {
                "id": rule.id,
                "name": rule.name,
                "enabled": rule.enabled,
                "in_cooldown": self._is_in_cooldown(rule.id),
                "cooldown_until": self._cooldown_cache.get(rule.id)
            }
            for rule in self.alert_rules
        ]

# Funciones de utilidad
def create_alert_engine(notification_config: NotificationConfig = None) -> AlertEngine:
    """
    Factory function para crear un motor de alertas
    
    Args:
        notification_config: Configuración de notificaciones
        
    Returns:
        Instancia del motor configurada
    """
    return AlertEngine(notification_config)

async def run_alert_monitoring(engine: AlertEngine):
    """
    Ejecuta el monitoreo de alertas
    
    Args:
        engine: Motor de alertas configurado
    """
    try:
        await engine.start_monitoring()
    except KeyboardInterrupt:
        logger.info("Monitoreo interrumpido por usuario")
        engine.stop_monitoring()
    except Exception as e:
        logger.error(f"Error en monitoreo: {str(e)}")
        engine.stop_monitoring()
        raise

