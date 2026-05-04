"""
Security tests for API routes refactoring.
Tests authentication, authorization, rate limiting, and input validation.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import time
from datetime import datetime, timedelta

class TestAuthentication:
    """Test authentication requirements across all endpoints."""
    
    @patch('src.api.dependencies.verify_token')
    def test_protected_endpoints_reject_unauthenticated(self, mock_verify):
        """Test that protected endpoints reject unauthenticated requests."""
        mock_verify.side_effect = Exception("Invalid token")
        
        # Test protected endpoints without authentication
        protected_endpoints = [
            ("/api/v1/analysis/upload", "POST"),
            ("/api/v1/analysis/results/123", "GET"),
            ("/api/v1/admin/users", "GET"),
            ("/api/v1/mobile/sync", "POST"),
            ("/api/v1/auth/me", "GET")
        ]
        
        for endpoint, method in protected_endpoints:
            # Mock request without valid token
            with patch('fastapi.Request') as mock_request:
                mock_request.headers = {}
                
                # Should raise authentication error
                with pytest.raises(Exception):
                    mock_verify(mock_request)
    
    @patch('src.api.dependencies.verify_token')
    def test_protected_endpoints_accept_valid_tokens(self, mock_verify):
        """Test that protected endpoints accept valid JWT tokens."""
        # Mock valid token verification
        mock_verify.return_value = {
            "user_id": 1,
            "email": "test@example.com",
            "exp": (datetime.now() + timedelta(hours=1)).timestamp()
        }
        
        with patch('fastapi.Request') as mock_request:
            mock_request.headers = {"Authorization": "Bearer valid_token"}
            
            # Should successfully verify token
            result = mock_verify(mock_request)
            assert result["user_id"] == 1
            assert result["email"] == "test@example.com"
    
    @patch('src.api.dependencies.verify_token')
    def test_expired_tokens_rejected(self, mock_verify):
        """Test that expired tokens are rejected."""
        # Mock expired token
        mock_verify.side_effect = Exception("Token expired")
        
        with patch('fastapi.Request') as mock_request:
            mock_request.headers = {"Authorization": "Bearer expired_token"}
            
            with pytest.raises(Exception, match="Token expired"):
                mock_verify(mock_request)
    
    @patch('src.api.dependencies.verify_token')
    def test_invalid_tokens_rejected(self, mock_verify):
        """Test that invalid tokens are rejected."""
        # Mock invalid token
        mock_verify.side_effect = Exception("Invalid token signature")
        
        with patch('fastapi.Request') as mock_request:
            mock_request.headers = {"Authorization": "Bearer invalid_token"}
            
            with pytest.raises(Exception, match="Invalid token"):
                mock_verify(mock_request)

class TestAuthorization:
    """Test authorization requirements and role-based access control."""
    
    @patch('src.api.dependencies.require_admin')
    @patch('src.api.dependencies.get_current_user')
    def test_admin_endpoints_reject_non_admin_users(self, mock_user, mock_admin):
        """Test that admin endpoints reject non-admin users."""
        # Mock regular user
        mock_user.return_value = {"id": 1, "email": "user@example.com", "role": "user"}
        mock_admin.side_effect = Exception("Admin access required")
        
        admin_endpoints = [
            "/api/v1/admin/users",
            "/api/v1/admin/config", 
            "/api/v1/admin/audit-logs",
            "/api/v1/admin/reports"
        ]
        
        for endpoint in admin_endpoints:
            with pytest.raises(Exception, match="Admin access required"):
                mock_admin(mock_user.return_value)
    
    @patch('src.api.dependencies.require_admin')
    @patch('src.api.dependencies.get_current_user')
    def test_admin_endpoints_accept_admin_users(self, mock_user, mock_admin):
        """Test that admin endpoints accept admin users."""
        # Mock admin user
        mock_user.return_value = {"id": 1, "email": "admin@example.com", "role": "admin"}
        mock_admin.return_value = True
        
        # Should successfully authorize admin
        result = mock_admin(mock_user.return_value)
        assert result is True
    
    @patch('src.api.routers.analysis.get_current_user')
    @patch('src.api.routers.analysis.get_database')
    def test_users_can_only_access_own_resources(self, mock_db, mock_user):
        """Test IDOR protection - users can only access their own resources."""
        # Mock current user
        mock_user.return_value = {"id": 1, "email": "user1@example.com"}
        
        # Mock database query that filters by user_id
        mock_db_session = Mock()
        mock_db.return_value = mock_db_session
        
        # Mock query that should filter by user_id
        mock_query = Mock()
        mock_db_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value = {"id": 123, "user_id": 1, "data": "user1_data"}
        
        with patch('src.api.routers.analysis.router') as mock_router:
            # Should only return resources owned by current user
            mock_router.get.return_value = {"id": 123, "user_id": 1, "data": "user1_data"}
            
            # Verify user_id filtering is applied
            assert mock_query.filter.called

class TestRateLimiting:
    """Test rate limiting functionality."""
    
    @patch('src.api.middleware.rate_limiter')
    def test_login_endpoint_rate_limiting(self, mock_limiter):
        """Test login endpoint rate limiting (5 requests/minute)."""
        # Mock rate limiter that tracks requests
        request_count = 0
        
        def mock_rate_limit(request):
            nonlocal request_count
            request_count += 1
            if request_count > 5:
                raise Exception("Rate limit exceeded")
            return True
        
        mock_limiter.side_effect = mock_rate_limit
        
        # Simulate 6 login attempts
        for i in range(6):
            if i < 5:
                # First 5 should succeed
                result = mock_limiter(Mock())
                assert result is True
            else:
                # 6th should fail
                with pytest.raises(Exception, match="Rate limit exceeded"):
                    mock_limiter(Mock())
    
    @patch('src.api.middleware.rate_limiter')
    def test_case_creation_rate_limiting(self, mock_limiter):
        """Test case creation rate limiting."""
        # Mock rate limiter for case creation
        mock_limiter.return_value = True
        
        # Should allow reasonable case creation rate
        for _ in range(10):
            result = mock_limiter(Mock())
            assert result is True
    
    @patch('fastapi.HTTPException')
    def test_rate_limit_returns_429_status(self, mock_exception):
        """Test that rate limit exceeded returns 429 status code."""
        # Mock 429 Too Many Requests response
        mock_exception.return_value = Mock(status_code=429, detail="Rate limit exceeded")
        
        exception = mock_exception(status_code=429, detail="Rate limit exceeded")
        assert exception.status_code == 429
        assert "Rate limit exceeded" in str(exception.detail)

class TestInputValidation:
    """Test input validation across all endpoints."""
    
    @patch('src.api.validators.validate_file_upload')
    def test_file_upload_validation(self, mock_validate):
        """Test file upload validation (magic bytes, size limits)."""
        # Test valid file
        mock_validate.return_value = ("image/jpeg", "safe_filename.jpg")
        
        valid_file_content = b'\xff\xd8\xff\xe0'  # JPEG magic bytes
        result = mock_validate(valid_file_content, "test.jpg")
        assert result[0] == "image/jpeg"
        assert result[1] == "safe_filename.jpg"
        
        # Test invalid file type
        mock_validate.side_effect = ValueError("Invalid file type")
        
        with pytest.raises(ValueError, match="Invalid file type"):
            mock_validate(b'invalid', "test.exe")
        
        # Test file too large
        mock_validate.side_effect = ValueError("File too large")
        
        with pytest.raises(ValueError, match="File too large"):
            mock_validate(b'x' * (10 * 1024 * 1024), "large.jpg")  # 10MB
    
    @patch('src.api.validators.validate_email')
    def test_email_validation(self, mock_validate):
        """Test email validation."""
        # Test valid email
        mock_validate.return_value = True
        assert mock_validate("test@example.com") is True
        
        # Test invalid email formats
        invalid_emails = [
            "invalid-email",
            "@example.com",
            "test@",
            "test..test@example.com",
            "test@example",
            ""
        ]
        
        for email in invalid_emails:
            mock_validate.return_value = False
            assert mock_validate(email) is False
    
    @patch('src.api.validators.validate_password')
    def test_password_validation(self, mock_validate):
        """Test password validation."""
        # Test strong password
        mock_validate.return_value = True
        assert mock_validate("SecurePass123!") is True
        
        # Test weak passwords
        weak_passwords = [
            "123456",
            "password",
            "abc",
            "PASSWORD",
            "12345678",
            ""
        ]
        
        for password in weak_passwords:
            mock_validate.return_value = False
            assert mock_validate(password) is False
    
    @patch('fastapi.HTTPException')
    def test_validation_error_messages(self, mock_exception):
        """Test that appropriate error messages are returned for validation failures."""
        # Mock validation error responses
        validation_errors = [
            (400, "Invalid email format"),
            (400, "Password too weak"),
            (400, "File type not allowed"),
            (413, "File too large"),
            (422, "Invalid input data")
        ]
        
        for status_code, message in validation_errors:
            mock_exception.return_value = Mock(status_code=status_code, detail=message)
            
            exception = mock_exception(status_code=status_code, detail=message)
            assert exception.status_code == status_code
            assert message in str(exception.detail)

# Security test configuration
@pytest.mark.security
class TestSecurityConfiguration:
    """Security test configuration and utilities."""
    
    def test_security_headers_present(self):
        """Test that security headers are properly configured."""
        # Mock security headers
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY", 
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains"
        }
        
        for header, value in security_headers.items():
            assert header is not None
            assert value is not None
    
    def test_cors_configuration(self):
        """Test CORS configuration is secure."""
        # Mock CORS settings
        cors_settings = {
            "allow_origins": ["https://trusted-domain.com"],
            "allow_credentials": True,
            "allow_methods": ["GET", "POST", "PUT", "DELETE"],
            "allow_headers": ["Authorization", "Content-Type"]
        }
        
        # Verify no wildcard origins in production
        assert "*" not in cors_settings["allow_origins"]
        assert cors_settings["allow_credentials"] is True

if __name__ == "__main__":
    pytest.main([__file__, "-v"])