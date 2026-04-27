"""
Notification Service

Sends notifications to pathologists when high-priority cases arrive.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
import asyncio

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

logger = logging.getLogger(__name__)


class NotificationChannel(Enum):
    """Notification delivery channels."""
    EMAIL = "email"
    WEBHOOK = "webhook"
    SMS = "sms"
    PUSH = "push"


class NotificationPriority(Enum):
    """Notification priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class NotificationService:
    """
    Notification service for pathologist alerts.
    
    Responsibilities:
    - Send email notifications
    - Send webhook notifications
    - Track notification delivery
    - Handle notification preferences
    """
    
    def __init__(
        self,
        smtp_host: Optional[str] = None,
        smtp_port: int = 587,
        smtp_username: Optional[str] = None,
        smtp_password: Optional[str] = None,
        from_email: str = "noreply@histocore.ai"
    ):
        """
        Initialize notification service.
        
        Args:
            smtp_host: SMTP server host
            smtp_port: SMTP server port
            smtp_username: SMTP username
            smtp_password: SMTP password
            from_email: From email address
        """
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_username = smtp_username
        self.smtp_password = smtp_password
        self.from_email = from_email
        
        # Webhook registrations
        self.webhooks: Dict[str, List[str]] = {}
        
        # Notification history
        self.notification_history: List[Dict[str, Any]] = []
        
        self.logger = logging.getLogger(__name__)
    
    async def notify_new_annotation_task(
        self,
        expert_id: str,
        task_id: str,
        slide_id: str,
        priority: float,
        uncertainty_score: float,
        channels: List[NotificationChannel] = None
    ):
        """
        Notify expert about new annotation task.
        
        Args:
            expert_id: Expert identifier
            task_id: Task identifier
            slide_id: Slide identifier
            priority: Task priority (0-1)
            uncertainty_score: Uncertainty score (0-1)
            channels: Notification channels to use
        """
        if channels is None:
            channels = [NotificationChannel.EMAIL, NotificationChannel.WEBHOOK]
        
        notification_priority = self._determine_notification_priority(priority, uncertainty_score)
        
        message = {
            'type': 'new_annotation_task',
            'expert_id': expert_id,
            'task_id': task_id,
            'slide_id': slide_id,
            'priority': priority,
            'uncertainty_score': uncertainty_score,
            'notification_priority': notification_priority.value,
            'timestamp': datetime.now().isoformat()
        }
        
        # Send through requested channels
        for channel in channels:
            try:
                if channel == NotificationChannel.EMAIL:
                    await self._send_email_notification(expert_id, message)
                elif channel == NotificationChannel.WEBHOOK:
                    await self._send_webhook_notification(expert_id, message)
            
            except Exception as e:
                self.logger.error(f"Failed to send {channel.value} notification: {e}")
        
        # Record notification
        self._record_notification(message, channels)
    
    async def notify_task_assignment(
        self,
        expert_id: str,
        task_id: str,
        slide_id: str,
        deadline: Optional[datetime] = None
    ):
        """
        Notify expert about task assignment.
        
        Args:
            expert_id: Expert identifier
            task_id: Task identifier
            slide_id: Slide identifier
            deadline: Optional deadline
        """
        message = {
            'type': 'task_assignment',
            'expert_id': expert_id,
            'task_id': task_id,
            'slide_id': slide_id,
            'deadline': deadline.isoformat() if deadline else None,
            'timestamp': datetime.now().isoformat()
        }
        
        await self._send_email_notification(expert_id, message)
        await self._send_webhook_notification(expert_id, message)
        
        self._record_notification(message, [NotificationChannel.EMAIL, NotificationChannel.WEBHOOK])
    
    async def notify_urgent_case(
        self,
        expert_ids: List[str],
        task_id: str,
        slide_id: str,
        reason: str
    ):
        """
        Notify multiple experts about urgent case.
        
        Args:
            expert_ids: List of expert identifiers
            task_id: Task identifier
            slide_id: Slide identifier
            reason: Reason for urgency
        """
        message = {
            'type': 'urgent_case',
            'task_id': task_id,
            'slide_id': slide_id,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        }
        
        # Send to all experts
        for expert_id in expert_ids:
            message['expert_id'] = expert_id
            
            await self._send_email_notification(expert_id, message)
            await self._send_webhook_notification(expert_id, message)
        
        self._record_notification(message, [NotificationChannel.EMAIL, NotificationChannel.WEBHOOK])
    
    async def _send_email_notification(
        self,
        expert_id: str,
        message: Dict[str, Any]
    ):
        """Send email notification."""
        if not self.smtp_host:
            self.logger.debug("SMTP not configured, skipping email notification")
            return
        
        try:
            # Get expert email (in production, would look up from database)
            expert_email = f"{expert_id}@hospital.org"
            
            # Create email
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = expert_email
            msg['Subject'] = self._create_email_subject(message)
            
            body = self._create_email_body(message)
            msg.attach(MIMEText(body, 'html'))
            
            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.smtp_username and self.smtp_password:
                    server.starttls()
                    server.login(self.smtp_username, self.smtp_password)
                
                server.send_message(msg)
            
            self.logger.info(f"Sent email notification to {expert_id}")
        
        except Exception as e:
            self.logger.error(f"Failed to send email: {e}")
            raise
    
    async def _send_webhook_notification(
        self,
        expert_id: str,
        message: Dict[str, Any]
    ):
        """Send webhook notification."""
        if not AIOHTTP_AVAILABLE:
            self.logger.debug("aiohttp not available, skipping webhook notification")
            return
        
        webhooks = self.webhooks.get(expert_id, [])
        
        if not webhooks:
            self.logger.debug(f"No webhooks registered for {expert_id}")
            return
        
        try:
            async with aiohttp.ClientSession() as session:
                for webhook_url in webhooks:
                    try:
                        async with session.post(
                            webhook_url,
                            json=message,
                            timeout=aiohttp.ClientTimeout(total=10)
                        ) as response:
                            if response.status == 200:
                                self.logger.info(f"Sent webhook notification to {webhook_url}")
                            else:
                                self.logger.warning(
                                    f"Webhook returned status {response.status}: {webhook_url}"
                                )
                    
                    except Exception as e:
                        self.logger.error(f"Failed to send webhook to {webhook_url}: {e}")
        
        except Exception as e:
            self.logger.error(f"Failed to send webhook notifications: {e}")
    
    def register_webhook(
        self,
        expert_id: str,
        webhook_url: str
    ):
        """
        Register webhook for expert.
        
        Args:
            expert_id: Expert identifier
            webhook_url: Webhook URL
        """
        if expert_id not in self.webhooks:
            self.webhooks[expert_id] = []
        
        if webhook_url not in self.webhooks[expert_id]:
            self.webhooks[expert_id].append(webhook_url)
            self.logger.info(f"Registered webhook for {expert_id}: {webhook_url}")
    
    def unregister_webhook(
        self,
        expert_id: str,
        webhook_url: str
    ):
        """
        Unregister webhook for expert.
        
        Args:
            expert_id: Expert identifier
            webhook_url: Webhook URL
        """
        if expert_id in self.webhooks:
            if webhook_url in self.webhooks[expert_id]:
                self.webhooks[expert_id].remove(webhook_url)
                self.logger.info(f"Unregistered webhook for {expert_id}: {webhook_url}")
    
    def _determine_notification_priority(
        self,
        task_priority: float,
        uncertainty_score: float
    ) -> NotificationPriority:
        """Determine notification priority based on task metrics."""
        # High uncertainty + high priority = urgent
        if task_priority > 0.8 and uncertainty_score > 0.9:
            return NotificationPriority.URGENT
        elif task_priority > 0.6 or uncertainty_score > 0.85:
            return NotificationPriority.HIGH
        elif task_priority > 0.4:
            return NotificationPriority.NORMAL
        else:
            return NotificationPriority.LOW
    
    def _create_email_subject(self, message: Dict[str, Any]) -> str:
        """Create email subject line."""
        msg_type = message.get('type', 'notification')
        
        if msg_type == 'new_annotation_task':
            priority = message.get('notification_priority', 'normal')
            return f"[{priority.upper()}] New Annotation Task - {message.get('slide_id')}"
        elif msg_type == 'task_assignment':
            return f"Task Assigned - {message.get('slide_id')}"
        elif msg_type == 'urgent_case':
            return f"[URGENT] Case Requires Immediate Attention - {message.get('slide_id')}"
        else:
            return "HistoCore Notification"
    
    def _create_email_body(self, message: Dict[str, Any]) -> str:
        """Create email body HTML."""
        msg_type = message.get('type', 'notification')
        
        if msg_type == 'new_annotation_task':
            return f"""
            <html>
            <body>
                <h2>New Annotation Task</h2>
                <p>A new high-priority case requires your expert review.</p>
                <ul>
                    <li><strong>Task ID:</strong> {message.get('task_id')}</li>
                    <li><strong>Slide ID:</strong> {message.get('slide_id')}</li>
                    <li><strong>Priority:</strong> {message.get('priority', 0):.2f}</li>
                    <li><strong>Uncertainty Score:</strong> {message.get('uncertainty_score', 0):.2f}</li>
                </ul>
                <p>Please review this case at your earliest convenience.</p>
                <p><a href="http://annotation-interface/task/{message.get('task_id')}">View Task</a></p>
            </body>
            </html>
            """
        elif msg_type == 'urgent_case':
            return f"""
            <html>
            <body>
                <h2 style="color: red;">URGENT: Case Requires Immediate Attention</h2>
                <p><strong>Reason:</strong> {message.get('reason')}</p>
                <ul>
                    <li><strong>Task ID:</strong> {message.get('task_id')}</li>
                    <li><strong>Slide ID:</strong> {message.get('slide_id')}</li>
                </ul>
                <p><a href="http://annotation-interface/task/{message.get('task_id')}">View Task Immediately</a></p>
            </body>
            </html>
            """
        else:
            return f"""
            <html>
            <body>
                <h2>HistoCore Notification</h2>
                <pre>{json.dumps(message, indent=2)}</pre>
            </body>
            </html>
            """
    
    def _record_notification(
        self,
        message: Dict[str, Any],
        channels: List[NotificationChannel]
    ):
        """Record notification in history."""
        record = {
            'message': message,
            'channels': [c.value for c in channels],
            'sent_at': datetime.now().isoformat()
        }
        
        self.notification_history.append(record)
        
        # Keep only recent history (last 1000)
        if len(self.notification_history) > 1000:
            self.notification_history = self.notification_history[-1000:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get notification statistics."""
        return {
            'total_notifications_sent': len(self.notification_history),
            'registered_webhooks': sum(len(urls) for urls in self.webhooks.values()),
            'experts_with_webhooks': len(self.webhooks),
            'smtp_configured': self.smtp_host is not None
        }
