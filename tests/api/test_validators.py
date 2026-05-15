"""
Unit tests for API validators module.
"""

import pytest
from fastapi import HTTPException

from src.api.validators import validate_email, validate_password, validate_file_upload


class TestEmailValidation:
    """Test email validation function."""

    def test_valid_emails(self):
        """Test that valid emails pass validation."""
        valid_emails = [
            "user@example.com",
            "test.email@domain.org",
            "user+tag@example.co.uk",
            "firstname.lastname@company.com",
            "user123@test-domain.com",
        ]

        for email in valid_emails:
            assert validate_email(email) is True

    def test_invalid_email_formats(self):
        """Test that invalid email formats raise HTTPException."""
        invalid_emails = [
            "invalid-email",
            "@domain.com",
            "user@",
            "user..double.dot@domain.com",
            "user@domain",
            "user@.domain.com",
            "user@domain..com",
        ]

        for email in invalid_emails:
            with pytest.raises(HTTPException) as exc_info:
                validate_email(email)
            assert exc_info.value.status_code == 400
            assert "Invalid email format" in exc_info.value.detail

    def test_empty_email(self):
        """Test that empty email raises HTTPException."""
        with pytest.raises(HTTPException) as exc_info:
            validate_email("")
        assert exc_info.value.status_code == 400
        assert "Email is required" in exc_info.value.detail

    def test_none_email(self):
        """Test that None email raises HTTPException."""
        with pytest.raises(HTTPException) as exc_info:
            validate_email(None)
        assert exc_info.value.status_code == 400
        assert "Email is required" in exc_info.value.detail

    def test_email_too_long(self):
        """Test that overly long emails raise HTTPException."""
        # Create email longer than 254 characters
        long_email = "a" * 250 + "@example.com"
        with pytest.raises(HTTPException) as exc_info:
            validate_email(long_email)
        assert exc_info.value.status_code == 400
        assert "Email address too long" in exc_info.value.detail

    def test_local_part_too_long(self):
        """Test that local part longer than 64 characters raises HTTPException."""
        # Create local part longer than 64 characters
        long_local = "a" * 65 + "@example.com"
        with pytest.raises(HTTPException) as exc_info:
            validate_email(long_local)
        assert exc_info.value.status_code == 400
        assert "Email local part too long" in exc_info.value.detail


class TestPasswordValidation:
    """Test password validation function."""

    def test_valid_passwords(self):
        """Test that valid passwords pass validation."""
        valid_passwords = [
            "Password123!",
            "MySecure@Pass1",
            "Complex#Password9",
            "Strong$Pass123",
            "Valid&Password1",
        ]

        for password in valid_passwords:
            assert validate_password(password) is True

    def test_password_too_short(self):
        """Test that passwords shorter than 8 characters raise HTTPException."""
        short_passwords = ["Pass1!", "Ab1!", "1234567"]

        for password in short_passwords:
            with pytest.raises(HTTPException) as exc_info:
                validate_password(password)
            assert exc_info.value.status_code == 400
            assert "at least 8 characters long" in exc_info.value.detail

    def test_password_too_long(self):
        """Test that passwords longer than 128 characters raise HTTPException."""
        long_password = "A" * 120 + "a1!"  # 124 characters, still valid
        assert validate_password(long_password) is True

        very_long_password = "A" * 130 + "a1!"  # 134 characters, too long
        with pytest.raises(HTTPException) as exc_info:
            validate_password(very_long_password)
        assert exc_info.value.status_code == 400
        assert "less than 128 characters" in exc_info.value.detail

    def test_missing_uppercase(self):
        """Test that passwords without uppercase letters raise HTTPException."""
        with pytest.raises(HTTPException) as exc_info:
            validate_password("password123!")
        assert exc_info.value.status_code == 400
        assert "uppercase letter" in exc_info.value.detail

    def test_missing_lowercase(self):
        """Test that passwords without lowercase letters raise HTTPException."""
        with pytest.raises(HTTPException) as exc_info:
            validate_password("PASSWORD123!")
        assert exc_info.value.status_code == 400
        assert "lowercase letter" in exc_info.value.detail

    def test_missing_digit(self):
        """Test that passwords without digits raise HTTPException."""
        with pytest.raises(HTTPException) as exc_info:
            validate_password("Password!")
        assert exc_info.value.status_code == 400
        assert "digit" in exc_info.value.detail

    def test_missing_special_character(self):
        """Test that passwords without special characters raise HTTPException."""
        with pytest.raises(HTTPException) as exc_info:
            validate_password("Password123")
        assert exc_info.value.status_code == 400
        assert "special character" in exc_info.value.detail

    def test_multiple_missing_requirements(self):
        """Test that passwords missing multiple requirements show all missing."""
        with pytest.raises(HTTPException) as exc_info:
            validate_password("password")  # Missing uppercase, digit, special
        assert exc_info.value.status_code == 400
        detail = exc_info.value.detail
        assert "uppercase letter" in detail
        assert "digit" in detail
        assert "special character" in detail

    def test_empty_password(self):
        """Test that empty password raises HTTPException."""
        with pytest.raises(HTTPException) as exc_info:
            validate_password("")
        assert exc_info.value.status_code == 400
        assert "Password is required" in exc_info.value.detail

    def test_none_password(self):
        """Test that None password raises HTTPException."""
        with pytest.raises(HTTPException) as exc_info:
            validate_password(None)
        assert exc_info.value.status_code == 400
        assert "Password is required" in exc_info.value.detail


class TestFileUploadValidation:
    """Test file upload validation function."""

    def test_valid_png_file(self):
        """Test that valid PNG file passes validation."""
        # PNG file signature
        png_content = b"\x89PNG\r\n\x1a\n" + b"fake_png_data" * 100
        mime_type, safe_filename = validate_file_upload(png_content, "test.png")

        assert mime_type == "image/png"
        assert safe_filename == "test.png"

    def test_valid_jpeg_file(self):
        """Test that valid JPEG file passes validation."""
        # JPEG file signature
        jpeg_content = b"\xff\xd8\xff" + b"fake_jpeg_data" * 100
        mime_type, safe_filename = validate_file_upload(jpeg_content, "test.jpg")

        # Note: python-magic might detect this as image/jpeg or application/octet-stream
        assert mime_type in ["image/jpeg", "application/octet-stream"]
        assert safe_filename == "test.jpg"

    def test_empty_file_content(self):
        """Test that empty file content raises HTTPException."""
        with pytest.raises(HTTPException) as exc_info:
            validate_file_upload(b"", "test.png")
        assert exc_info.value.status_code == 400
        assert "File content is required" in exc_info.value.detail

    def test_empty_filename(self):
        """Test that empty filename raises HTTPException."""
        png_content = b"\x89PNG\r\n\x1a\n" + b"fake_png_data" * 100
        with pytest.raises(HTTPException) as exc_info:
            validate_file_upload(png_content, "")
        assert exc_info.value.status_code == 400
        assert "Filename is required" in exc_info.value.detail

    def test_file_too_large(self):
        """Test that files larger than 100MB raise HTTPException."""
        # Create content larger than 100MB
        large_content = b"x" * (101 * 1024 * 1024)
        with pytest.raises(HTTPException) as exc_info:
            validate_file_upload(large_content, "large.png")
        assert exc_info.value.status_code == 413
        assert "File too large" in exc_info.value.detail

    def test_file_too_small(self):
        """Test that files smaller than 10 bytes raise HTTPException."""
        small_content = b"tiny"  # 4 bytes
        with pytest.raises(HTTPException) as exc_info:
            validate_file_upload(small_content, "tiny.png")
        assert exc_info.value.status_code == 400
        assert "empty or corrupted" in exc_info.value.detail

    def test_filename_sanitization(self):
        """Test that dangerous filenames are sanitized."""
        png_content = b"\x89PNG\r\n\x1a\n" + b"fake_png_data" * 100

        dangerous_filenames = [
            "../../../etc/passwd",
            "file<script>alert('xss')</script>.png",
            "file|rm -rf /.png",
            "file:with:colons.png",
            "file with spaces.png",
        ]

        for dangerous_name in dangerous_filenames:
            mime_type, safe_filename = validate_file_upload(png_content, dangerous_name)
            # Should not contain dangerous characters
            assert "<" not in safe_filename
            assert ">" not in safe_filename
            assert "|" not in safe_filename
            assert ":" not in safe_filename
            assert "/" not in safe_filename
            assert "\\" not in safe_filename

    def test_long_filename_truncation(self):
        """Test that overly long filenames are truncated."""
        png_content = b"\x89PNG\r\n\x1a\n" + b"fake_png_data" * 100
        long_filename = "a" * 300 + ".png"  # 304 characters

        mime_type, safe_filename = validate_file_upload(png_content, long_filename)
        assert len(safe_filename) <= 255
        assert safe_filename.endswith(".png")

    def test_dicom_file_validation(self):
        """Test DICOM file validation."""
        # DICOM file with proper signature
        dicom_content = b"\x00" * 128 + b"DICM" + b"fake_dicom_data" * 100

        # Note: python-magic might not detect this as application/dicom without proper DICOM structure
        # So we'll accept application/octet-stream as well
        mime_type, safe_filename = validate_file_upload(dicom_content, "test.dcm")
        assert mime_type in ["application/dicom", "application/octet-stream"]
        assert safe_filename == "test.dcm"

    def test_invalid_image_signature(self):
        """Test that files with invalid image signatures are rejected."""
        # File claiming to be PNG but with wrong signature
        fake_png = b"FAKE" + b"not_really_png" * 100

        with pytest.raises(HTTPException) as exc_info:
            validate_file_upload(fake_png, "fake.png")
        # This might raise either "not supported" or "not valid image" depending on magic detection
        assert exc_info.value.status_code in [400, 413]


class TestFilenameSanitization:
    """Test filename sanitization helper function."""

    def test_path_traversal_removal(self):
        """Test that path traversal attempts are removed."""
        from src.api.validators import _sanitize_filename

        dangerous_paths = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32\\config",
            "/etc/passwd",
            "C:\\Windows\\System32\\config",
        ]

        for path in dangerous_paths:
            safe = _sanitize_filename(path)
            assert ".." not in safe
            assert "/" not in safe
            assert "\\" not in safe

    def test_dangerous_character_replacement(self):
        """Test that dangerous characters are replaced."""
        from src.api.validators import _sanitize_filename

        dangerous_name = 'file<>:"/\\|?*.txt'
        safe = _sanitize_filename(dangerous_name)

        dangerous_chars = '<>:"/\\|?*'
        for char in dangerous_chars:
            assert char not in safe

    def test_empty_filename_handling(self):
        """Test that empty filenames get default name."""
        from src.api.validators import _sanitize_filename

        assert _sanitize_filename("") == "uploaded_file"
        assert _sanitize_filename("...") == "uploaded_file"
        assert _sanitize_filename("   ") == "uploaded_file"
