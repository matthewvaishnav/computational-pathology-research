"""
Unit tests for the authentication router.

Tests user registration, login, OAuth flows, and security features.
"""

import json
from unittest.mock import MagicMock, Mock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.api.main import app

# Import the router and dependencies
from src.api.routers.auth import UserLogin, UserRegistration, router, users_db

# Create test client
client = TestClient(app)


class TestAuthRouter:
    """Test suite for authentication router."""

    def setup_method(self):
        """Set up test environment before each test."""
        # Clear the in-memory users database
        users_db.clear()

    def teardown_method(self):
        """Clean up after each test."""
        # Clear the in-memory users database
        users_db.clear()

    def test_auth_router_exists(self):
        """Test that auth router module exists and is importable."""
        from src.api.routers import auth

        assert hasattr(auth, "router")
        assert auth.router.prefix == "/api/v1/auth"
        assert "authentication" in auth.router.tags

    def test_auth_router_configuration(self):
        """Test that auth router has correct configuration."""
        from src.api.routers import auth

        assert auth.router.prefix == "/api/v1/auth"
        assert auth.router.tags == ["authentication"]

        # Check that router has expected routes
        route_paths = [route.path for route in auth.router.routes]
        expected_paths = ["/register", "/login", "/me", "/oauth/login", "/oauth/callback"]

        for expected_path in expected_paths:
            assert expected_path in route_paths

    @patch("src.api.routers.auth.validate_email")
    @patch("src.api.routers.auth.validate_password")
    @patch("src.api.routers.auth.hash_password")
    @patch("src.api.routers.auth.log_security_event")
    def test_user_registration_success(
        self, mock_log, mock_hash, mock_validate_password, mock_validate_email
    ):
        """Test successful user registration."""
        # Mock dependencies
        mock_validate_email.return_value = None  # No exception means valid
        mock_validate_password.return_value = None  # No exception means valid
        mock_hash.return_value = "hashed_password_123"

        # Test data
        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "SecurePass123!",
        }

        # Make request
        response = client.post("/api/v1/auth/register", json=user_data)

        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "User registered successfully"
        assert "user_id" in data

        # Verify user was added to database
        assert "testuser" in users_db
        user = users_db["testuser"]
        assert user["username"] == "testuser"
        assert user["email"] == "test@example.com"
        assert user["role"] == "pathologist"  # Default role
        assert user["password_hash"] == "hashed_password_123"

        # Verify security logging
        mock_log.assert_called()

    @patch("src.api.routers.auth.validate_email")
    def test_user_registration_invalid_email(self, mock_validate_email):
        """Test user registration with invalid email."""
        # Mock email validation to raise exception
        mock_validate_email.side_effect = ValueError("Invalid email format")

        user_data = {"username": "testuser", "email": "invalid-email", "password": "SecurePass123!"}

        response = client.post("/api/v1/auth/register", json=user_data)

        assert response.status_code == 500  # Validation error becomes 500
        assert "testuser" not in users_db

    @patch("src.api.routers.auth.validate_password")
    def test_user_registration_weak_password(self, mock_validate_password):
        """Test user registration with weak password."""
        # Mock password validation to raise exception
        mock_validate_password.side_effect = ValueError("Password too weak")

        user_data = {"username": "testuser", "email": "test@example.com", "password": "weak"}

        response = client.post("/api/v1/auth/register", json=user_data)

        assert response.status_code == 500  # Validation error becomes 500
        assert "testuser" not in users_db

    @patch("src.api.routers.auth.validate_email")
    @patch("src.api.routers.auth.validate_password")
    @patch("src.api.routers.auth.hash_password")
    @patch("src.api.routers.auth.log_security_event")
    def test_user_registration_duplicate_username(
        self, mock_log, mock_hash, mock_validate_password, mock_validate_email
    ):
        """Test user registration with duplicate username."""
        # Mock dependencies
        mock_validate_email.return_value = None
        mock_validate_password.return_value = None
        mock_hash.return_value = "hashed_password_123"

        # Add existing user
        users_db["testuser"] = {
            "user_id": "existing-id",
            "username": "testuser",
            "email": "existing@example.com",
            "password_hash": "existing_hash",
            "role": "pathologist",
        }

        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "SecurePass123!",
        }

        response = client.post("/api/v1/auth/register", json=user_data)

        assert response.status_code == 409
        data = response.json()
        assert data["detail"] == "User already exists"

    @patch("src.api.routers.auth.check_account_lockout")
    @patch("src.api.routers.auth.verify_password")
    @patch("src.api.routers.auth.clear_failed_login")
    @patch("src.api.routers.auth.create_access_token")
    @patch("src.api.routers.auth.log_security_event")
    @patch("time.time")
    @patch("time.sleep")
    def test_user_login_success(
        self, mock_sleep, mock_time, mock_log, mock_token, mock_clear, mock_verify, mock_lockout
    ):
        """Test successful user login."""
        # Mock dependencies
        mock_lockout.return_value = None  # No lockout
        mock_verify.return_value = True  # Password valid
        mock_token.return_value = "jwt_token_123"
        mock_time.side_effect = [0, 0.6]  # Simulate timing

        # Add test user
        users_db["testuser"] = {
            "user_id": "user-123",
            "username": "testuser",
            "email": "test@example.com",
            "password_hash": "hashed_password",
            "role": "pathologist",
        }

        login_data = {"username": "testuser", "password": "correct_password"}

        response = client.post("/api/v1/auth/login", json=login_data)

        assert response.status_code == 200
        data = response.json()
        assert data["access_token"] == "jwt_token_123"
        assert data["token_type"] == "bearer"

        # Verify security functions called
        mock_lockout.assert_called_once_with("testuser")
        mock_verify.assert_called_once()
        mock_clear.assert_called_once_with("testuser")
        mock_token.assert_called_once()

    @patch("src.api.routers.auth.check_account_lockout")
    @patch("src.api.routers.auth.verify_password")
    @patch("src.api.routers.auth.hash_password")
    @patch("src.api.routers.auth.record_failed_login")
    @patch("src.api.routers.auth.log_security_event")
    @patch("time.time")
    @patch("time.sleep")
    def test_user_login_invalid_credentials(
        self, mock_sleep, mock_time, mock_log, mock_record, mock_hash, mock_verify, mock_lockout
    ):
        """Test user login with invalid credentials."""
        # Mock dependencies
        mock_lockout.return_value = None  # No lockout
        mock_verify.return_value = False  # Password invalid
        mock_hash.return_value = "dummy_hash"
        mock_time.side_effect = [0, 0.3]  # Simulate timing

        # Add test user
        users_db["testuser"] = {
            "user_id": "user-123",
            "username": "testuser",
            "email": "test@example.com",
            "password_hash": "hashed_password",
            "role": "pathologist",
        }

        login_data = {"username": "testuser", "password": "wrong_password"}

        response = client.post("/api/v1/auth/login", json=login_data)

        assert response.status_code == 401
        data = response.json()
        assert data["detail"] == "Invalid credentials"

        # Verify security functions called
        mock_record.assert_called_once_with("testuser")
        mock_sleep.assert_called()  # Timing attack protection

    @patch("src.api.routers.auth.check_account_lockout")
    @patch("src.api.routers.auth.verify_password")
    @patch("src.api.routers.auth.hash_password")
    @patch("src.api.routers.auth.record_failed_login")
    @patch("src.api.routers.auth.log_security_event")
    @patch("time.time")
    @patch("time.sleep")
    def test_user_login_nonexistent_user(
        self, mock_sleep, mock_time, mock_log, mock_record, mock_hash, mock_verify, mock_lockout
    ):
        """Test user login with nonexistent username."""
        # Mock dependencies
        mock_lockout.return_value = None  # No lockout
        mock_verify.return_value = False  # Password invalid (dummy check)
        mock_hash.return_value = "dummy_hash"
        mock_time.side_effect = [0, 0.3]  # Simulate timing

        login_data = {"username": "nonexistent", "password": "any_password"}

        response = client.post("/api/v1/auth/login", json=login_data)

        assert response.status_code == 401
        data = response.json()
        assert data["detail"] == "Invalid credentials"

        # Verify security functions called
        mock_record.assert_called_once_with("nonexistent")
        mock_verify.assert_called_once()  # Dummy password check for timing

    @patch("src.api.routers.auth.check_account_lockout")
    def test_user_login_account_locked(self, mock_lockout):
        """Test user login with locked account."""
        # Mock account lockout to raise exception
        mock_lockout.side_effect = HTTPException(status_code=429, detail="Account locked")

        login_data = {"username": "testuser", "password": "any_password"}

        response = client.post("/api/v1/auth/login", json=login_data)

        assert response.status_code == 429
        data = response.json()
        assert data["detail"] == "Account locked"

    @patch("src.api.routers.auth.get_current_user")
    def test_get_current_user_info_success(self, mock_get_user):
        """Test getting current user info when authenticated."""
        # Mock current user
        mock_user = Mock()
        mock_user.id = 123
        mock_user.username = "testuser"
        mock_user.email = "test@example.com"
        mock_user.role = "pathologist"
        mock_get_user.return_value = mock_user

        response = client.get("/api/v1/auth/me")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "123"
        assert data["username"] == "testuser"
        assert data["email"] == "test@example.com"
        assert data["role"] == "pathologist"

    @patch("src.api.routers.auth.get_current_user")
    def test_get_current_user_info_not_authenticated(self, mock_get_user):
        """Test getting current user info when not authenticated."""
        # Mock no current user
        mock_get_user.return_value = None

        response = client.get("/api/v1/auth/me")

        assert response.status_code == 401
        data = response.json()
        assert data["detail"] == "Not authenticated"

    @patch("src.api.routers.auth.create_oauth_client")
    @patch("src.api.routers.auth.log_security_event")
    def test_oauth_login_success(self, mock_log, mock_create_client):
        """Test successful OAuth login initiation."""
        # Mock OAuth client
        mock_client = Mock()
        mock_client.get_authorization_url.return_value = ("https://auth.example.com", "state123")
        mock_create_client.return_value = mock_client

        response = client.get("/api/v1/auth/oauth/login?provider=azure")

        assert response.status_code == 200
        data = response.json()
        assert data["authorization_url"] == "https://auth.example.com"
        assert data["state"] == "state123"
        assert data["provider"] == "azure"

        mock_create_client.assert_called_once_with(provider="azure")

    @patch("src.api.routers.auth.create_oauth_client")
    @patch("src.api.routers.auth.log_security_event")
    def test_oauth_login_failure(self, mock_log, mock_create_client):
        """Test OAuth login initiation failure."""
        # Mock OAuth client creation failure
        mock_create_client.side_effect = Exception("OAuth setup failed")

        response = client.get("/api/v1/auth/oauth/login?provider=azure")

        assert response.status_code == 500
        data = response.json()
        assert data["detail"] == "Failed to initiate OAuth login"

    @patch("src.api.routers.auth.create_oauth_client")
    @patch("src.api.routers.auth.oauth_callback_handler")
    @patch("src.api.routers.auth.get_db_session")
    @patch("src.api.routers.auth.UserOperations")
    @patch("src.api.routers.auth.create_access_token")
    @patch("src.api.routers.auth.log_security_event")
    def test_oauth_callback_success_existing_user(
        self, mock_log, mock_token, mock_user_ops_class, mock_db, mock_callback, mock_create_client
    ):
        """Test successful OAuth callback for existing user."""
        # Mock dependencies
        mock_client = Mock()
        mock_create_client.return_value = mock_client

        mock_callback.return_value = {
            "userinfo": {"email": "test@example.com", "preferred_username": "testuser"},
            "access_token": "oauth_token_123",
        }

        mock_db_session = Mock()
        mock_db.return_value = iter([mock_db_session])

        mock_user_ops = Mock()
        mock_user_ops_class.return_value = mock_user_ops

        # Mock existing user
        mock_user = Mock()
        mock_user.id = 123
        mock_user.username = "testuser"
        mock_user.email = "test@example.com"
        mock_user.role = "pathologist"
        mock_user_ops.get_user_by_email.return_value = mock_user

        mock_token.return_value = "jwt_token_123"

        response = client.get("/api/v1/auth/oauth/callback?provider=azure")

        assert response.status_code == 200
        data = response.json()
        assert data["access_token"] == "jwt_token_123"
        assert data["token_type"] == "bearer"
        assert data["user"]["email"] == "test@example.com"
        assert data["oauth_provider"] == "azure"

    @patch("src.api.routers.auth.create_oauth_client")
    @patch("src.api.routers.auth.oauth_callback_handler")
    @patch("src.api.routers.auth.get_db_session")
    @patch("src.api.routers.auth.UserOperations")
    @patch("src.api.routers.auth.create_access_token")
    @patch("src.api.routers.auth.log_security_event")
    def test_oauth_callback_success_new_user(
        self, mock_log, mock_token, mock_user_ops_class, mock_db, mock_callback, mock_create_client
    ):
        """Test successful OAuth callback for new user."""
        # Mock dependencies
        mock_client = Mock()
        mock_create_client.return_value = mock_client

        mock_callback.return_value = {
            "userinfo": {"email": "newuser@example.com", "preferred_username": "newuser"},
            "access_token": "oauth_token_123",
        }

        mock_db_session = Mock()
        mock_db.return_value = iter([mock_db_session])

        mock_user_ops = Mock()
        mock_user_ops_class.return_value = mock_user_ops

        # Mock no existing user, then new user creation
        mock_user_ops.get_user_by_email.return_value = None

        mock_new_user = Mock()
        mock_new_user.id = 456
        mock_new_user.username = "newuser"
        mock_new_user.email = "newuser@example.com"
        mock_new_user.role = "pathologist"
        mock_user_ops.create_user.return_value = mock_new_user

        mock_token.return_value = "jwt_token_456"

        response = client.get("/api/v1/auth/oauth/callback?provider=azure")

        assert response.status_code == 200
        data = response.json()
        assert data["access_token"] == "jwt_token_456"
        assert data["user"]["email"] == "newuser@example.com"

        # Verify user creation was called
        mock_user_ops.create_user.assert_called_once()

    @patch("src.api.routers.auth.create_oauth_client")
    @patch("src.api.routers.auth.oauth_callback_handler")
    def test_oauth_callback_no_email(self, mock_callback, mock_create_client):
        """Test OAuth callback when no email is provided."""
        # Mock dependencies
        mock_client = Mock()
        mock_create_client.return_value = mock_client

        mock_callback.return_value = {
            "userinfo": {
                "preferred_username": "testuser"
                # No email provided
            },
            "access_token": "oauth_token_123",
        }

        response = client.get("/api/v1/auth/oauth/callback?provider=azure")

        assert response.status_code == 400
        data = response.json()
        assert data["detail"] == "Email not provided by OAuth provider"

    def test_pydantic_models_validation(self):
        """Test Pydantic model validation."""
        # Test UserRegistration model
        valid_registration = UserRegistration(
            username="testuser", email="test@example.com", password="SecurePass123!"
        )
        assert valid_registration.username == "testuser"
        assert valid_registration.email == "test@example.com"
        assert valid_registration.password == "SecurePass123!"

        # Test UserLogin model
        valid_login = UserLogin(username="testuser", password="SecurePass123!")
        assert valid_login.username == "testuser"
        assert valid_login.password == "SecurePass123!"

    def test_router_endpoints_count(self):
        """Test that auth router has the expected number of endpoints."""
        from src.api.routers import auth

        # Count actual route handlers (exclude OPTIONS)
        route_methods = []
        for route in auth.router.routes:
            if hasattr(route, "methods"):
                route_methods.extend(route.methods)

        # Should have POST register, POST login, GET me, GET oauth/login, GET oauth/callback
        # Plus OPTIONS for each
        assert len([m for m in route_methods if m != "OPTIONS"]) >= 5

    def test_router_security_features(self):
        """Test that router implements expected security features."""
        from src.api.routers import auth

        # Check imports of security functions
        assert hasattr(auth, "check_account_lockout")
        assert hasattr(auth, "hash_password")
        assert hasattr(auth, "verify_password")
        assert hasattr(auth, "create_access_token")
        assert hasattr(auth, "log_security_event")
        assert hasattr(auth, "limiter")

    def test_default_role_assignment(self):
        """Test that new users get default pathologist role."""
        # This is tested in the registration success test, but worth emphasizing
        # that role cannot be overridden by user input
        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "SecurePass123!",
            "role": "admin",  # This should be ignored
        }

        # The Pydantic model should reject extra fields
        with pytest.raises(Exception):  # ValidationError from Pydantic
            UserRegistration(**user_data)
