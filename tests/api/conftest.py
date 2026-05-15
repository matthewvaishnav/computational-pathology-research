"""
Test configuration for API tests.

Provides fixtures and setup for testing the API routers.
"""

import os
import tempfile
from typing import Generator
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Set test environment variables
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["JWT_SECRET_KEY"] = "test-secret-key-for-testing-only"
os.environ["ENVIRONMENT"] = "test"

from src.api.main import app
from src.database.connection import get_db_session
from src.database.models import Base


def get_test_db():
    """Create test database session."""
    # Create in-memory SQLite database for testing
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})

    # Create all tables
    Base.metadata.create_all(bind=engine)

    # Create session
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


# Mock password hashing functions for testing
def mock_hash_password(password: str) -> str:
    """Mock password hashing for testing."""
    return f"hashed_{password}"


def mock_verify_password(plain_password: str, hashed_password: str) -> bool:
    """Mock password verification for testing."""
    return hashed_password == f"hashed_{plain_password}"


@pytest.fixture(scope="function")
def test_client() -> Generator[TestClient, None, None]:
    """Create test client with mocked database and security functions."""
    # Override database dependency
    app.dependency_overrides[get_db_session] = get_test_db

    # Mock password hashing functions in the auth router
    with (
        patch("src.api.routers.auth.hash_password", side_effect=mock_hash_password),
        patch("src.api.routers.auth.verify_password", side_effect=mock_verify_password),
    ):

        with TestClient(app) as client:
            yield client

    # Clean up
    app.dependency_overrides.clear()


@pytest.fixture(scope="function")
def test_user_data():
    """Test user data for registration."""
    return {"username": "test_user", "email": "test@example.com", "password": "TestPassword123!"}


@pytest.fixture(scope="function")
def admin_user_data():
    """Admin user data for registration."""
    return {"username": "admin_user", "email": "admin@example.com", "password": "AdminPassword123!"}
