#!/usr/bin/env python3
"""
Database Models

SQLAlchemy models for the Medical AI platform database schema.
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Boolean, Text, JSON,
    ForeignKey, Index, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

Base = declarative_base()


class TimestampMixin:
    """Mixin for created_at and updated_at timestamps."""
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class User(Base, TimestampMixin):
    """User model for authentication and authorization."""
    __tablename__ = 'users'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False, default='pathologist')
    is_active = Column(Boolean, default=True, nullable=False)
    last_login = Column(DateTime(timezone=True))
    
    # Relationships
    analyses = relationship("Analysis", back_populates="user")
    cases = relationship("Case", back_populates="assigned_user")
    audit_logs = relationship("AuditLog", back_populates="user")
    
    def __repr__(self):
        return f"<User(username='{self.username}', role='{self.role}')>"


class Case(Base, TimestampMixin):
    """Case model for patient cases and studies."""
    __tablename__ = 'cases'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(String(100), nullable=False, index=True)
    study_id = Column(String(100), nullable=False, index=True)
    case_type = Column(String(50), nullable=False, default='breast_cancer_screening')
    priority = Column(String(20), nullable=False, default='normal')
    status = Column(String(20), nullable=False, default='pending')
    notes = Column(Text)
    
    # Foreign keys
    assigned_user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    
    # Relationships
    assigned_user = relationship("User", back_populates="cases")
    analyses = relationship("Analysis", back_populates="case")
    dicom_studies = relationship("DicomStudy", back_populates="case")
    
    # Indexes
    __table_args__ = (
        Index('idx_case_patient_study', 'patient_id', 'study_id'),
        Index('idx_case_status_priority', 'status', 'priority'),
    )
    
    def __repr__(self):
        return f"<Case(patient_id='{self.patient_id}', status='{self.status}')>"


class Analysis(Base, TimestampMixin):
    """Analysis model for AI model inference results."""
    __tablename__ = 'analyses'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(255), nullable=False)
    content_type = Column(String(100), nullable=False)
    file_size = Column(Integer)
    file_path = Column(String(500))  # Path to stored file
    
    # Analysis status and results
    status = Column(String(20), nullable=False, default='queued')
    model_version = Column(String(50))
    processing_time_ms = Column(Integer)
    
    # Foreign keys
    case_id = Column(UUID(as_uuid=True), ForeignKey('cases.id'))
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    
    # Relationships
    case = relationship("Case", back_populates="analyses")
    user = relationship("User", back_populates="analyses")
    results = relationship("ModelResult", back_populates="analysis", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_analysis_status', 'status'),
        Index('idx_analysis_case', 'case_id'),
        Index('idx_analysis_created', 'created_at'),
    )
    
    def __repr__(self):
        return f"<Analysis(filename='{self.filename}', status='{self.status}')>"


class ModelResult(Base, TimestampMixin):
    """Model inference results for each analysis."""
    __tablename__ = 'model_results'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Prediction results
    prediction_class = Column(String(50), nullable=False)
    confidence_score = Column(Float, nullable=False)
    probability_scores = Column(JSON)  # All class probabilities
    
    # Model metadata
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=False)
    
    # Additional results
    attention_maps = Column(JSON)  # Attention visualization data
    feature_importance = Column(JSON)  # Feature importance scores
    uncertainty_score = Column(Float)  # Model uncertainty
    
    # Foreign keys
    analysis_id = Column(UUID(as_uuid=True), ForeignKey('analyses.id'), nullable=False)
    
    # Relationships
    analysis = relationship("Analysis", back_populates="results")
    
    # Indexes
    __table_args__ = (
        Index('idx_result_analysis', 'analysis_id'),
        Index('idx_result_prediction', 'prediction_class', 'confidence_score'),
    )
    
    def __repr__(self):
        return f"<ModelResult(prediction='{self.prediction_class}', confidence={self.confidence_score:.3f})>"


class DicomStudy(Base, TimestampMixin):
    """DICOM study metadata and file information."""
    __tablename__ = 'dicom_studies'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # DICOM identifiers
    study_instance_uid = Column(String(100), unique=True, nullable=False, index=True)
    series_instance_uid = Column(String(100), nullable=False, index=True)
    sop_instance_uid = Column(String(100), nullable=False, index=True)
    
    # Patient information
    patient_id = Column(String(100), nullable=False, index=True)
    patient_name = Column(String(255))
    patient_birth_date = Column(String(10))  # YYYYMMDD format
    patient_sex = Column(String(1))
    
    # Study information
    study_date = Column(String(8))  # YYYYMMDD format
    study_time = Column(String(6))  # HHMMSS format
    study_description = Column(String(255))
    modality = Column(String(10), nullable=False)
    
    # File information
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer)
    
    # Processing status
    status = Column(String(20), nullable=False, default='received')
    
    # Foreign keys
    case_id = Column(UUID(as_uuid=True), ForeignKey('cases.id'))
    
    # Relationships
    case = relationship("Case", back_populates="dicom_studies")
    
    # Indexes
    __table_args__ = (
        Index('idx_dicom_study_uid', 'study_instance_uid'),
        Index('idx_dicom_patient', 'patient_id'),
        Index('idx_dicom_study_date', 'study_date'),
        UniqueConstraint('study_instance_uid', 'series_instance_uid', 'sop_instance_uid'),
    )
    
    def __repr__(self):
        return f"<DicomStudy(study_uid='{self.study_instance_uid[:20]}...', patient='{self.patient_id}')>"


class AuditLog(Base, TimestampMixin):
    """Audit log for tracking user actions and system events."""
    __tablename__ = 'audit_logs'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Event information
    event_type = Column(String(50), nullable=False, index=True)
    event_description = Column(Text, nullable=False)
    resource_type = Column(String(50))  # e.g., 'analysis', 'case', 'user'
    resource_id = Column(String(100))   # ID of the affected resource
    
    # Request information
    ip_address = Column(String(45))  # IPv6 compatible
    user_agent = Column(String(500))
    request_method = Column(String(10))
    request_path = Column(String(500))
    
    # Result information
    status_code = Column(Integer)
    error_message = Column(Text)
    
    # Additional metadata
    metadata = Column(JSON)
    
    # Foreign keys
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    
    # Relationships
    user = relationship("User", back_populates="audit_logs")
    
    # Indexes
    __table_args__ = (
        Index('idx_audit_event_type', 'event_type'),
        Index('idx_audit_user', 'user_id'),
        Index('idx_audit_created', 'created_at'),
        Index('idx_audit_resource', 'resource_type', 'resource_id'),
    )
    
    def __repr__(self):
        return f"<AuditLog(event='{self.event_type}', user_id='{self.user_id}')>"


class SystemConfig(Base, TimestampMixin):
    """System configuration settings."""
    __tablename__ = 'system_config'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key = Column(String(100), unique=True, nullable=False, index=True)
    value = Column(Text, nullable=False)
    description = Column(Text)
    config_type = Column(String(20), nullable=False, default='string')  # string, int, float, bool, json
    is_sensitive = Column(Boolean, default=False, nullable=False)  # For passwords, API keys, etc.
    
    def __repr__(self):
        return f"<SystemConfig(key='{self.key}', type='{self.config_type}')>"