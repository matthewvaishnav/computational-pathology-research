#!/usr/bin/env python3
"""
Database Connection Management

Handles PostgreSQL connections, session management, and connection pooling
for the Medical AI platform.
"""

import os
import logging
from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

from .models import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and sessions."""
    
    def __init__(self, database_url: Optional[str] = None):
        """Initialize database manager.
        
        Args:
            database_url: PostgreSQL connection URL. If None, reads from environment.
        """
        self.database_url = database_url or self._get_database_url()
        self.engine = None
        self.SessionLocal = None
        self._initialize_engine()
    
    def _get_database_url(self) -> str:
        """Get database URL from environment variables."""
        # Try different environment variable names
        url = (
            os.getenv('DATABASE_URL') or
            os.getenv('POSTGRES_URL') or
            os.getenv('DB_URL')
        )
        
        if url:
            return url
        
        # Build URL from components
        host = os.getenv('DB_HOST', 'localhost')
        port = os.getenv('DB_PORT', '5432')
        database = os.getenv('DB_NAME', 'medical_ai')
        username = os.getenv('DB_USER', 'postgres')
        password = os.getenv('DB_PASSWORD', 'postgres')
        
        return f"postgresql://{username}:{password}@{host}:{port}/{database}"
    
    def _initialize_engine(self):
        """Initialize SQLAlchemy engine with connection pooling."""
        self.engine = create_engine(
            self.database_url,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=os.getenv('DB_ECHO', 'false').lower() == 'true'
        )
        
        # Add connection event listeners
        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            """Set connection parameters for PostgreSQL."""
            if 'postgresql' in self.database_url:
                with dbapi_connection.cursor() as cursor:
                    # Set timezone
                    cursor.execute("SET timezone TO 'UTC'")
        
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        logger.info(f"Database engine initialized: {self.database_url.split('@')[-1]}")
    
    def create_tables(self):
        """Create all database tables."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise
    
    def drop_tables(self):
        """Drop all database tables (use with caution)."""
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.info("Database tables dropped successfully")
        except Exception as e:
            logger.error(f"Failed to drop database tables: {e}")
            raise
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session with automatic cleanup.
        
        Yields:
            SQLAlchemy session
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def health_check(self) -> dict:
        """Check database connectivity and return health status."""
        try:
            with self.get_session() as session:
                # Simple query to test connection
                result = session.execute("SELECT 1 as health_check")
                row = result.fetchone()
                
                if row and row[0] == 1:
                    # Get connection pool stats
                    pool = self.engine.pool
                    return {
                        "status": "healthy",
                        "connection_count": pool.size(),
                        "checked_out": pool.checkedout(),
                        "overflow": pool.overflow(),
                        "checked_in": pool.checkedin()
                    }
                else:
                    return {"status": "unhealthy", "error": "Query failed"}
                    
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}
    
    def close(self):
        """Close database connections."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connections closed")


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """Get global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def get_db_session() -> Generator[Session, None, None]:
    """Dependency for FastAPI to get database session."""
    db_manager = get_database_manager()
    with db_manager.get_session() as session:
        yield session


def initialize_database():
    """Initialize database and create tables."""
    db_manager = get_database_manager()
    db_manager.create_tables()
    return db_manager