"""
Unit tests for API error handlers module.
"""

import pytest
from unittest.mock import Mock
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.errors import (
    not_found_handler,
    internal_error_handler,
    validation_error_handler,
    http_exception_handler,
    get_error_name,
    create_error_response,
)


class TestNotFoundHandler:
    """Test 404 not found error handler."""

    @pytest.mark.asyncio
    async def test_not_found_handler(self):
        """Test that 404 handler returns correct response."""
        # Mock request
        request = Mock(spec=Request)
        request.method = "GET"
        request.url = Mock()
        request.url.path = "/nonexistent"
        request.url.__str__ = Mock(return_value="http://localhost/nonexistent")

        # Mock exception
        exc = StarletteHTTPException(status_code=404, detail="Not found")

        # Call handler
        response = await not_found_handler(request, exc)

        # Verify response
        assert response.status_code == 404
        content = response.body.decode()
        assert "Not Found" in content
        assert "/nonexistent" in content
        assert "GET" in content


class TestInternalErrorHandler:
    """Test 500 internal server error handler."""

    @pytest.mark.asyncio
    async def test_internal_error_handler(self):
        """Test that 500 handler returns correct response."""
        # Mock request
        request = Mock(spec=Request)
        request.method = "POST"
        request.url = Mock()
        request.url.path = "/api/test"
        request.url.__str__ = Mock(return_value="http://localhost/api/test")

        # Mock exception
        exc = Exception("Something went wrong")

        # Call handler
        response = await internal_error_handler(request, exc)

        # Verify response
        assert response.status_code == 500
        content = response.body.decode()
        assert "Internal Server Error" in content
        assert "/api/test" in content
        assert "POST" in content


class TestValidationErrorHandler:
    """Test request validation error handler."""

    @pytest.mark.asyncio
    async def test_validation_error_handler(self):
        """Test that validation error handler returns correct response."""
        # Mock request
        request = Mock(spec=Request)
        request.method = "POST"
        request.url.path = "/api/register"
        request.url = Mock()
        request.url.__str__ = Mock(return_value="http://localhost/api/register")

        # Mock validation error
        errors = [
            {"loc": ("body", "email"), "msg": "field required", "type": "value_error.missing"},
            {
                "loc": ("body", "password"),
                "msg": "ensure this value has at least 8 characters",
                "type": "value_error.any_str.min_length",
            },
        ]
        exc = RequestValidationError(errors)

        # Call handler
        response = await validation_error_handler(request, exc)

        # Verify response
        assert response.status_code == 422
        content = response.body.decode()
        assert "Validation Error" in content
        assert "body -> email" in content
        assert "body -> password" in content
        assert "field required" in content


class TestHttpExceptionHandler:
    """Test general HTTP exception handler."""

    @pytest.mark.asyncio
    async def test_http_exception_handler_401(self):
        """Test HTTP exception handler for 401 Unauthorized."""
        # Mock request
        request = Mock(spec=Request)
        request.method = "GET"
        request.url = Mock()
        request.url.path = "/api/protected"
        request.url.__str__ = Mock(return_value="http://localhost/api/protected")

        # Mock HTTP exception
        exc = StarletteHTTPException(status_code=401, detail="Not authenticated")

        # Call handler
        response = await http_exception_handler(request, exc)

        # Verify response
        assert response.status_code == 401
        content = response.body.decode()
        assert "Unauthorized" in content
        assert "Not authenticated" in content
        assert "/api/protected" in content

    @pytest.mark.asyncio
    async def test_http_exception_handler_403(self):
        """Test HTTP exception handler for 403 Forbidden."""
        # Mock request
        request = Mock(spec=Request)
        request.method = "DELETE"
        request.url = Mock()
        request.url.path = "/api/admin/users"
        request.url.__str__ = Mock(return_value="http://localhost/api/admin/users")

        # Mock HTTP exception
        exc = StarletteHTTPException(status_code=403, detail="Admin access required")

        # Call handler
        response = await http_exception_handler(request, exc)

        # Verify response
        assert response.status_code == 403
        content = response.body.decode()
        assert "Forbidden" in content
        assert "Admin access required" in content
        assert "/api/admin/users" in content


class TestGetErrorName:
    """Test error name utility function."""

    def test_common_error_codes(self):
        """Test that common HTTP status codes return correct names."""
        assert get_error_name(400) == "Bad Request"
        assert get_error_name(401) == "Unauthorized"
        assert get_error_name(403) == "Forbidden"
        assert get_error_name(404) == "Not Found"
        assert get_error_name(422) == "Unprocessable Entity"
        assert get_error_name(429) == "Too Many Requests"
        assert get_error_name(500) == "Internal Server Error"

    def test_unknown_error_code(self):
        """Test that unknown status codes return generic format."""
        assert get_error_name(418) == "HTTP 418"
        assert get_error_name(999) == "HTTP 999"


class TestCreateErrorResponse:
    """Test error response creation utility."""

    def test_basic_error_response(self):
        """Test creating basic error response."""
        response = create_error_response(400, "Invalid input")

        assert response["error"] == "Bad Request"
        assert response["message"] == "Invalid input"
        assert "detail" not in response

    def test_error_response_with_detail(self):
        """Test creating error response with detail."""
        response = create_error_response(422, "Validation failed", detail="Email format is invalid")

        assert response["error"] == "Unprocessable Entity"
        assert response["message"] == "Validation failed"
        assert response["detail"] == "Email format is invalid"

    def test_error_response_with_extra_data(self):
        """Test creating error response with extra data."""
        extra_data = {"field": "email", "code": "INVALID_FORMAT"}
        response = create_error_response(400, "Invalid email", extra_data=extra_data)

        assert response["error"] == "Bad Request"
        assert response["message"] == "Invalid email"
        assert response["field"] == "email"
        assert response["code"] == "INVALID_FORMAT"

    def test_error_response_with_all_parameters(self):
        """Test creating error response with all parameters."""
        extra_data = {"timestamp": "2023-01-01T00:00:00Z"}
        response = create_error_response(
            500, "Server error", detail="Database connection failed", extra_data=extra_data
        )

        assert response["error"] == "Internal Server Error"
        assert response["message"] == "Server error"
        assert response["detail"] == "Database connection failed"
        assert response["timestamp"] == "2023-01-01T00:00:00Z"
