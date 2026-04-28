#!/usr/bin/env python3
"""
Production Database Setup Script

Sets up PostgreSQL database for the Medical AI platform.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.connection import DatabaseManager, initialize_database
from src.database.operations import UserOperations
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_admin_user(db_manager: DatabaseManager):
    """Create default admin user."""
    with db_manager.get_session() as session:
        user_ops = UserOperations(session)
        
        # Check if admin user already exists
        admin_user = user_ops.get_user_by_username("admin")
        if admin_user:
            logger.info("Admin user already exists")
            return
        
        # Create admin user
        password_hash = hashlib.sha256("admin123".encode()).hexdigest()  # Change in production!
        admin_user = user_ops.create_user(
            username="admin",
            email="admin@medical-ai.com",
            password_hash=password_hash,
            role="admin"
        )
        
        logger.info(f"Created admin user: {admin_user.username}")


def setup_sample_data(db_manager: DatabaseManager):
    """Set up sample data for testing."""
    with db_manager.get_session() as session:
        from src.database.operations import CaseOperations
        
        case_ops = CaseOperations(session)
        
        # Create sample cases
        sample_cases = [
            {
                "patient_id": "PAT001",
                "study_id": "STU001",
                "case_type": "breast_cancer_screening",
                "priority": "normal"
            },
            {
                "patient_id": "PAT002", 
                "study_id": "STU002",
                "case_type": "breast_cancer_screening",
                "priority": "high"
            }
        ]
        
        for case_data in sample_cases:
            existing_case = case_ops.get_case_by_patient_study(
                case_data["patient_id"], 
                case_data["study_id"]
            )
            
            if not existing_case:
                case = case_ops.create_case(**case_data)
                logger.info(f"Created sample case: {case.patient_id}/{case.study_id}")


def main():
    """Main setup function."""
    logger.info("Setting up production database...")
    
    try:
        # Initialize database
        db_manager = initialize_database()
        logger.info("Database tables created successfully")
        
        # Create admin user
        create_admin_user(db_manager)
        
        # Set up sample data (optional)
        if os.getenv("SETUP_SAMPLE_DATA", "false").lower() == "true":
            setup_sample_data(db_manager)
            logger.info("Sample data created")
        
        # Test database connectivity
        health = db_manager.health_check()
        if health["status"] == "healthy":
            logger.info("Database setup completed successfully!")
            logger.info(f"Connection pool: {health}")
        else:
            logger.error(f"Database health check failed: {health}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()