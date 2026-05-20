"""Production database management."""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

from .config import get_config

logger = logging.getLogger(__name__)
config = get_config()

# Database setup
engine = create_engine(
    config.database.url,
    pool_size=config.database.pool_size,
    max_overflow=config.database.max_overflow,
    pool_timeout=config.database.pool_timeout,
    pool_recycle=config.database.pool_recycle,
    echo=config.debug,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Client(Base):
    """Client registration and status."""

    __tablename__ = "clients"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    organization = Column(String(255), nullable=True)

    # Status
    status = Column(
        String(50), nullable=False, default="registered"
    )  # registered, active, inactive, suspended
    last_seen = Column(DateTime, nullable=True)

    # Capabilities
    max_batch_size = Column(Integer, nullable=True)
    supported_algorithms = Column(JSON, nullable=True)

    # Privacy
    privacy_budget_epsilon = Column(Float, nullable=False, default=0.0)
    privacy_budget_delta = Column(Float, nullable=False, default=0.0)
    max_epsilon = Column(Float, nullable=False, default=10.0)

    # Metadata
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class TrainingRound(Base):
    """Training round metadata."""

    __tablename__ = "training_rounds"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    round_id = Column(Integer, nullable=False, index=True)

    # Configuration
    algorithm = Column(String(100), nullable=False)
    min_clients = Column(Integer, nullable=False)
    max_clients = Column(Integer, nullable=False)

    # Status
    status = Column(
        String(50), nullable=False, default="pending"
    )  # pending, active, completed, failed
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Results
    participants = Column(JSON, nullable=True)  # List of client IDs
    aggregated_metrics = Column(JSON, nullable=True)
    model_version = Column(Integer, nullable=True)

    # Metadata
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class ClientUpdate(Base):
    """Client update submissions."""

    __tablename__ = "client_updates"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    round_id = Column(Integer, nullable=False, index=True)
    client_id = Column(String(255), nullable=False, index=True)

    # Update metadata
    dataset_size = Column(Integer, nullable=False)
    training_time_seconds = Column(Float, nullable=False)

    # Privacy
    privacy_epsilon = Column(Float, nullable=False, default=0.0)
    privacy_delta = Column(Float, nullable=False, default=0.0)

    # Model update (stored as reference to file system)
    model_update_path = Column(String(500), nullable=False)
    model_update_hash = Column(String(64), nullable=False)  # SHA-256

    # Validation
    is_valid = Column(Boolean, nullable=False, default=True)
    validation_errors = Column(JSON, nullable=True)

    # Byzantine detection
    is_byzantine = Column(Boolean, nullable=False, default=False)
    byzantine_score = Column(Float, nullable=True)

    # Metadata
    submitted_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)


class ModelCheckpoint(Base):
    """Model checkpoint storage."""

    __tablename__ = "model_checkpoints"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    version = Column(Integer, nullable=False, unique=True, index=True)
    round_id = Column(Integer, nullable=False)

    # Storage
    checkpoint_path = Column(String(500), nullable=False)
    checkpoint_hash = Column(String(64), nullable=False)  # SHA-256
    size_bytes = Column(Integer, nullable=False)

    # Metrics
    validation_metrics = Column(JSON, nullable=True)
    training_metrics = Column(JSON, nullable=True)

    # Metadata
    contributors = Column(JSON, nullable=True)  # List of client IDs
    algorithm_config = Column(JSON, nullable=True)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class AuditLog(Base):
    """HIPAA-compliant audit logging."""

    __tablename__ = "audit_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Event details
    event_type = Column(String(100), nullable=False, index=True)
    event_category = Column(String(50), nullable=False, index=True)  # access, modification, system

    # Actor
    user_id = Column(String(255), nullable=True, index=True)
    client_id = Column(String(255), nullable=True, index=True)
    ip_address = Column(String(45), nullable=True)  # IPv6 compatible
    user_agent = Column(Text, nullable=True)

    # Resource
    resource_type = Column(String(100), nullable=True)
    resource_id = Column(String(255), nullable=True)

    # Details
    description = Column(Text, nullable=False)
    additional_data = Column(JSON, nullable=True)

    # Outcome
    success = Column(Boolean, nullable=False)
    error_message = Column(Text, nullable=True)

    # Timestamp
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)


class SystemMetrics(Base):
    """System performance metrics."""

    __tablename__ = "system_metrics"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Metrics
    metric_name = Column(String(100), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(50), nullable=True)

    # Context
    component = Column(String(100), nullable=False, index=True)  # coordinator, client, database
    instance_id = Column(String(255), nullable=True)

    # Additional data
    labels = Column(JSON, nullable=True)

    # Timestamp
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)


# Database operations
class DatabaseManager:
    """Production database manager."""

    def __init__(self):
        self.engine = engine
        self.SessionLocal = SessionLocal

    def get_session(self) -> Session:
        """Get database session."""
        return self.SessionLocal()

    def create_tables(self):
        """Create all tables."""
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created")

    def drop_tables(self):
        """Drop all tables (use with caution)."""
        Base.metadata.drop_all(bind=self.engine)
        logger.warning("Database tables dropped")

    # Client operations
    def register_client(self, client_id: str, name: str, organization: str = None) -> Client:
        """Register a new client."""
        with self.get_session() as session:
            client = Client(
                client_id=client_id, name=name, organization=organization, status="registered"
            )
            session.add(client)
            session.commit()
            session.refresh(client)
            return client

    def get_client(self, client_id: str) -> Optional[Client]:
        """Get client by ID."""
        with self.get_session() as session:
            return session.query(Client).filter(Client.client_id == client_id).first()

    def update_client_status(self, client_id: str, status: str):
        """Update client status."""
        with self.get_session() as session:
            client = session.query(Client).filter(Client.client_id == client_id).first()
            if client:
                client.status = status
                client.last_seen = datetime.utcnow()
                session.commit()

    def get_active_clients(self) -> List[Client]:
        """Get all active clients."""
        with self.get_session() as session:
            return session.query(Client).filter(Client.status == "active").all()

    # Training round operations
    def create_training_round(
        self, round_id: int, algorithm: str, min_clients: int, max_clients: int
    ) -> TrainingRound:
        """Create a new training round."""
        with self.get_session() as session:
            round_obj = TrainingRound(
                round_id=round_id,
                algorithm=algorithm,
                min_clients=min_clients,
                max_clients=max_clients,
                status="pending",
            )
            session.add(round_obj)
            session.commit()
            session.refresh(round_obj)
            return round_obj

    def update_training_round(self, round_id: int, **kwargs):
        """Update training round."""
        with self.get_session() as session:
            round_obj = (
                session.query(TrainingRound).filter(TrainingRound.round_id == round_id).first()
            )
            if round_obj:
                for key, value in kwargs.items():
                    setattr(round_obj, key, value)
                session.commit()

    # Audit logging
    def log_audit_event(
        self,
        event_type: str,
        event_category: str,
        description: str,
        user_id: str = None,
        client_id: str = None,
        success: bool = True,
        **kwargs,
    ):
        """Log audit event for HIPAA compliance."""
        with self.get_session() as session:
            audit_log = AuditLog(
                event_type=event_type,
                event_category=event_category,
                description=description,
                user_id=user_id,
                client_id=client_id,
                success=success,
                **kwargs,
            )
            session.add(audit_log)
            session.commit()

    # Metrics
    def record_metric(
        self,
        metric_name: str,
        metric_value: float,
        component: str,
        metric_unit: str = None,
        labels: Dict[str, Any] = None,
    ):
        """Record system metric."""
        with self.get_session() as session:
            metric = SystemMetrics(
                metric_name=metric_name,
                metric_value=metric_value,
                metric_unit=metric_unit,
                component=component,
                labels=labels,
            )
            session.add(metric)
            session.commit()


# Global database manager instance
db_manager = DatabaseManager()


def get_db_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    return db_manager


def init_database():
    """Initialize the production database."""
    try:
        db_manager.create_tables()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise
