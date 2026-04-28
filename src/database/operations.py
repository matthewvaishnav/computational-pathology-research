#!/usr/bin/env python3
"""
Database Operations

High-level database operations for the Medical AI platform.
"""

import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func

from .models import Analysis, Case, User, DicomStudy, ModelResult, AuditLog

logger = logging.getLogger(__name__)


class BaseOperations:
    """Base class for database operations."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def commit(self):
        """Commit current transaction."""
        self.session.commit()
    
    def rollback(self):
        """Rollback current transaction."""
        self.session.rollback()


class UserOperations(BaseOperations):
    """User-related database operations."""
    
    def create_user(self, username: str, email: str, password_hash: str, 
                   role: str = 'pathologist') -> User:
        """Create a new user."""
        user = User(
            username=username,
            email=email,
            password_hash=password_hash,
            role=role
        )
        self.session.add(user)
        self.session.flush()  # Get the ID without committing
        
        logger.info(f"Created user: {username} (role: {role})")
        return user
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        return self.session.query(User).filter(User.username == username).first()
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        return self.session.query(User).filter(User.email == email).first()
    
    def get_user_by_id(self, user_id: UUID) -> Optional[User]:
        """Get user by ID."""
        return self.session.query(User).filter(User.id == user_id).first()
    
    def update_last_login(self, user_id: UUID) -> bool:
        """Update user's last login timestamp."""
        user = self.get_user_by_id(user_id)
        if user:
            user.last_login = datetime.utcnow()
            return True
        return False
    
    def list_users(self, limit: int = 50, offset: int = 0) -> List[User]:
        """List users with pagination."""
        return (self.session.query(User)
                .order_by(User.created_at.desc())
                .limit(limit)
                .offset(offset)
                .all())


class CaseOperations(BaseOperations):
    """Case-related database operations."""
    
    def create_case(self, patient_id: str, study_id: str, case_type: str = 'breast_cancer_screening',
                   priority: str = 'normal', assigned_user_id: Optional[UUID] = None) -> Case:
        """Create a new case."""
        case = Case(
            patient_id=patient_id,
            study_id=study_id,
            case_type=case_type,
            priority=priority,
            assigned_user_id=assigned_user_id
        )
        self.session.add(case)
        self.session.flush()
        
        logger.info(f"Created case: {patient_id}/{study_id} (type: {case_type})")
        return case
    
    def get_case_by_id(self, case_id: UUID) -> Optional[Case]:
        """Get case by ID."""
        return self.session.query(Case).filter(Case.id == case_id).first()
    
    def get_case_by_patient_study(self, patient_id: str, study_id: str) -> Optional[Case]:
        """Get case by patient ID and study ID."""
        return (self.session.query(Case)
                .filter(and_(Case.patient_id == patient_id, Case.study_id == study_id))
                .first())
    
    def update_case_status(self, case_id: UUID, status: str, notes: Optional[str] = None) -> bool:
        """Update case status and notes."""
        case = self.get_case_by_id(case_id)
        if case:
            case.status = status
            if notes:
                case.notes = notes
            logger.info(f"Updated case {case_id} status to: {status}")
            return True
        return False
    
    def list_cases(self, status: Optional[str] = None, priority: Optional[str] = None,
                  assigned_user_id: Optional[UUID] = None, limit: int = 50, 
                  offset: int = 0) -> List[Case]:
        """List cases with filtering and pagination."""
        query = self.session.query(Case)
        
        if status:
            query = query.filter(Case.status == status)
        if priority:
            query = query.filter(Case.priority == priority)
        if assigned_user_id:
            query = query.filter(Case.assigned_user_id == assigned_user_id)
        
        return (query.order_by(Case.created_at.desc())
                .limit(limit)
                .offset(offset)
                .all())
    
    def get_case_statistics(self) -> Dict[str, Any]:
        """Get case statistics."""
        total_cases = self.session.query(func.count(Case.id)).scalar()
        
        status_counts = (self.session.query(Case.status, func.count(Case.id))
                        .group_by(Case.status)
                        .all())
        
        priority_counts = (self.session.query(Case.priority, func.count(Case.id))
                          .group_by(Case.priority)
                          .all())
        
        return {
            'total_cases': total_cases,
            'status_distribution': dict(status_counts),
            'priority_distribution': dict(priority_counts)
        }


class AnalysisOperations(BaseOperations):
    """Analysis-related database operations."""
    
    def create_analysis(self, filename: str, content_type: str, file_size: int,
                       file_path: str, case_id: Optional[UUID] = None,
                       user_id: Optional[UUID] = None) -> Analysis:
        """Create a new analysis."""
        analysis = Analysis(
            filename=filename,
            content_type=content_type,
            file_size=file_size,
            file_path=file_path,
            case_id=case_id,
            user_id=user_id
        )
        self.session.add(analysis)
        self.session.flush()
        
        logger.info(f"Created analysis: {filename} (size: {file_size} bytes)")
        return analysis
    
    def get_analysis_by_id(self, analysis_id: UUID) -> Optional[Analysis]:
        """Get analysis by ID."""
        return self.session.query(Analysis).filter(Analysis.id == analysis_id).first()
    
    def update_analysis_status(self, analysis_id: UUID, status: str, 
                              processing_time_ms: Optional[int] = None,
                              model_version: Optional[str] = None) -> bool:
        """Update analysis status and metadata."""
        analysis = self.get_analysis_by_id(analysis_id)
        if analysis:
            analysis.status = status
            if processing_time_ms is not None:
                analysis.processing_time_ms = processing_time_ms
            if model_version:
                analysis.model_version = model_version
            logger.info(f"Updated analysis {analysis_id} status to: {status}")
            return True
        return False
    
    def add_model_result(self, analysis_id: UUID, prediction_class: str, 
                        confidence_score: float, model_name: str, model_version: str,
                        probability_scores: Optional[Dict] = None,
                        attention_maps: Optional[Dict] = None,
                        uncertainty_score: Optional[float] = None) -> ModelResult:
        """Add model result to analysis."""
        result = ModelResult(
            analysis_id=analysis_id,
            prediction_class=prediction_class,
            confidence_score=confidence_score,
            model_name=model_name,
            model_version=model_version,
            probability_scores=probability_scores,
            attention_maps=attention_maps,
            uncertainty_score=uncertainty_score
        )
        self.session.add(result)
        self.session.flush()
        
        logger.info(f"Added model result: {prediction_class} ({confidence_score:.3f})")
        return result
    
    def list_analyses(self, status: Optional[str] = None, case_id: Optional[UUID] = None,
                     user_id: Optional[UUID] = None, limit: int = 50, 
                     offset: int = 0) -> List[Analysis]:
        """List analyses with filtering and pagination."""
        query = self.session.query(Analysis)
        
        if status:
            query = query.filter(Analysis.status == status)
        if case_id:
            query = query.filter(Analysis.case_id == case_id)
        if user_id:
            query = query.filter(Analysis.user_id == user_id)
        
        return (query.order_by(Analysis.created_at.desc())
                .limit(limit)
                .offset(offset)
                .all())
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get analysis statistics."""
        total_analyses = self.session.query(func.count(Analysis.id)).scalar()
        
        status_counts = (self.session.query(Analysis.status, func.count(Analysis.id))
                        .group_by(Analysis.status)
                        .all())
        
        # Average processing time for completed analyses
        avg_processing_time = (self.session.query(func.avg(Analysis.processing_time_ms))
                              .filter(Analysis.status == 'completed')
                              .scalar())
        
        return {
            'total_analyses': total_analyses,
            'status_distribution': dict(status_counts),
            'average_processing_time_ms': float(avg_processing_time) if avg_processing_time else None
        }


class DicomOperations(BaseOperations):
    """DICOM-related database operations."""
    
    def create_dicom_study(self, study_instance_uid: str, series_instance_uid: str,
                          sop_instance_uid: str, patient_id: str, filename: str,
                          file_path: str, file_size: int, modality: str = 'SM',
                          case_id: Optional[UUID] = None, **metadata) -> DicomStudy:
        """Create a new DICOM study record."""
        study = DicomStudy(
            study_instance_uid=study_instance_uid,
            series_instance_uid=series_instance_uid,
            sop_instance_uid=sop_instance_uid,
            patient_id=patient_id,
            filename=filename,
            file_path=file_path,
            file_size=file_size,
            modality=modality,
            case_id=case_id,
            **metadata
        )
        self.session.add(study)
        self.session.flush()
        
        logger.info(f"Created DICOM study: {study_instance_uid} (patient: {patient_id})")
        return study
    
    def get_study_by_uid(self, study_instance_uid: str) -> Optional[DicomStudy]:
        """Get DICOM study by Study Instance UID."""
        return (self.session.query(DicomStudy)
                .filter(DicomStudy.study_instance_uid == study_instance_uid)
                .first())
    
    def get_studies_by_patient(self, patient_id: str) -> List[DicomStudy]:
        """Get all DICOM studies for a patient."""
        return (self.session.query(DicomStudy)
                .filter(DicomStudy.patient_id == patient_id)
                .order_by(DicomStudy.study_date.desc())
                .all())
    
    def update_study_status(self, study_id: UUID, status: str) -> bool:
        """Update DICOM study status."""
        study = self.session.query(DicomStudy).filter(DicomStudy.id == study_id).first()
        if study:
            study.status = status
            logger.info(f"Updated DICOM study {study_id} status to: {status}")
            return True
        return False


class AuditOperations(BaseOperations):
    """Audit log operations."""
    
    def log_event(self, event_type: str, event_description: str, 
                 user_id: Optional[UUID] = None, resource_type: Optional[str] = None,
                 resource_id: Optional[str] = None, ip_address: Optional[str] = None,
                 user_agent: Optional[str] = None, request_method: Optional[str] = None,
                 request_path: Optional[str] = None, status_code: Optional[int] = None,
                 error_message: Optional[str] = None, metadata: Optional[Dict] = None) -> AuditLog:
        """Log an audit event."""
        audit_log = AuditLog(
            event_type=event_type,
            event_description=event_description,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            ip_address=ip_address,
            user_agent=user_agent,
            request_method=request_method,
            request_path=request_path,
            status_code=status_code,
            error_message=error_message,
            metadata=metadata
        )
        self.session.add(audit_log)
        self.session.flush()
        
        return audit_log
    
    def get_audit_logs(self, user_id: Optional[UUID] = None, event_type: Optional[str] = None,
                      resource_type: Optional[str] = None, limit: int = 100,
                      offset: int = 0) -> List[AuditLog]:
        """Get audit logs with filtering."""
        query = self.session.query(AuditLog)
        
        if user_id:
            query = query.filter(AuditLog.user_id == user_id)
        if event_type:
            query = query.filter(AuditLog.event_type == event_type)
        if resource_type:
            query = query.filter(AuditLog.resource_type == resource_type)
        
        return (query.order_by(AuditLog.created_at.desc())
                .limit(limit)
                .offset(offset)
                .all())