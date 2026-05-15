"""
Tests for API dependencies module.
"""

import pytest
from unittest.mock import Mock, patch
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from src.api.dependencies import get_inference_engine, get_current_user


class TestGetInferenceEngine:
    """Test the get_inference_engine dependency."""

    def test_get_inference_engine_singleton(self):
        """Test that get_inference_engine returns the same instance."""
        engine1 = get_inference_engine()
        engine2 = get_inference_engine()

        assert engine1 is engine2
        assert engine1 is not None

    def test_get_inference_engine_type(self):
        """Test that get_inference_engine returns InferenceEngine instance."""
        from src.inference import InferenceEngine

        engine = get_inference_engine()
        assert isinstance(engine, InferenceEngine)


class TestGetCurrentUser:
    """Test the get_current_user dependency."""

    @patch("src.api.dependencies.decode_access_token")
    @patch("src.api.dependencies.UserOperations")
    @patch("src.api.dependencies.log_security_event")
    def test_get_current_user_success(self, mock_log, mock_user_ops_class, mock_decode):
        """Test successful user authentication."""
        # Setup mocks
        mock_decode.return_value = {"sub": "123e4567-e89b-12d3-a456-426614174000"}
        mock_user_ops = Mock()
        mock_user_ops_class.return_value = mock_user_ops
        mock_user = Mock()
        mock_user_ops.get_user_by_id.return_value = mock_user

        # Create mock credentials
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="valid_token")

        # Create mock db session
        mock_db = Mock()

        # Call function
        result = get_current_user(credentials, mock_db)

        # Assertions
        assert result == mock_user
        mock_decode.assert_called_once_with("valid_token")
        mock_user_ops_class.assert_called_once_with(mock_db)

    @patch("src.api.dependencies.log_security_event")
    def test_get_current_user_no_credentials(self, mock_log):
        """Test authentication failure with no credentials."""
        mock_db = Mock()

        with pytest.raises(HTTPException) as exc_info:
            get_current_user(None, mock_db)

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Not authenticated"
        mock_log.assert_called_once()

    @patch("src.api.dependencies.decode_access_token")
    @patch("src.api.dependencies.log_security_event")
    def test_get_current_user_invalid_token(self, mock_log, mock_decode):
        """Test authentication failure with invalid token."""
        # Setup mocks
        mock_decode.return_value = {}  # No 'sub' field

        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="invalid_token")
        mock_db = Mock()

        with pytest.raises(HTTPException) as exc_info:
            get_current_user(credentials, mock_db)

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Invalid token"

    @patch("src.api.dependencies.decode_access_token")
    @patch("src.api.dependencies.UserOperations")
    @patch("src.api.dependencies.log_security_event")
    def test_get_current_user_user_not_found(self, mock_log, mock_user_ops_class, mock_decode):
        """Test authentication failure when user not found."""
        # Setup mocks
        mock_decode.return_value = {"sub": "123e4567-e89b-12d3-a456-426614174000"}
        mock_user_ops = Mock()
        mock_user_ops_class.return_value = mock_user_ops
        mock_user_ops.get_user_by_id.return_value = None  # User not found

        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="valid_token")
        mock_db = Mock()

        with pytest.raises(HTTPException) as exc_info:
            get_current_user(credentials, mock_db)

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "User not found"

    @patch("src.api.dependencies.decode_access_token")
    @patch("src.api.dependencies.log_security_event")
    def test_get_current_user_decode_exception(self, mock_log, mock_decode):
        """Test authentication failure when token decode raises exception."""
        # Setup mocks
        mock_decode.side_effect = Exception("Token decode error")

        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="malformed_token")
        mock_db = Mock()

        with pytest.raises(HTTPException) as exc_info:
            get_current_user(credentials, mock_db)

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Authentication failed"
