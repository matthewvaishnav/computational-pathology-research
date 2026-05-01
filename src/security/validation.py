"""Input validation for API endpoints."""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Input validation error."""

    pass


class InputValidator:
    """Validate user inputs for security.

    Prevents:
    - Path traversal attacks
    - SQL injection
    - Command injection
    - XSS attacks
    - Invalid file types
    """

    # Allowed file extensions
    ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".svs", ".ndpi"}
    ALLOWED_MODEL_EXTENSIONS = {".pth", ".pt", ".ckpt", ".h5"}

    # Dangerous patterns
    PATH_TRAVERSAL_PATTERN = re.compile(r"\.\.|~|/etc|/root|/sys|/proc")
    SQL_INJECTION_PATTERN = re.compile(
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)|(--)|(;)",
        re.IGNORECASE,
    )
    COMMAND_INJECTION_PATTERN = re.compile(r"[;&|`$(){}[\]<>]")

    @staticmethod
    def validate_path(
        path: Union[str, Path],
        allowed_dirs: Optional[List[Path]] = None,
        must_exist: bool = False,
    ) -> Path:
        """Validate file path for security.

        Args:
            path: Path to validate
            allowed_dirs: List of allowed base directories
            must_exist: Whether path must exist

        Returns:
            Validated Path object

        Raises:
            ValidationError: If path invalid or unsafe
        """
        try:
            path = Path(path).resolve()
        except Exception as e:
            raise ValidationError(f"Invalid path: {e}")

        # Check for path traversal
        path_str = str(path)
        if InputValidator.PATH_TRAVERSAL_PATTERN.search(path_str):
            raise ValidationError(f"Path traversal detected: {path}")

        # Check allowed directories
        if allowed_dirs:
            allowed = False
            for allowed_dir in allowed_dirs:
                try:
                    path.relative_to(allowed_dir.resolve())
                    allowed = True
                    break
                except ValueError:
                    continue

            if not allowed:
                raise ValidationError(f"Path outside allowed directories: {path}")

        # Check existence
        if must_exist and not path.exists():
            raise ValidationError(f"Path does not exist: {path}")

        return path

    @staticmethod
    def validate_file_extension(
        filename: str,
        allowed_extensions: Optional[set] = None,
    ) -> str:
        """Validate file extension.

        Args:
            filename: Filename to validate
            allowed_extensions: Set of allowed extensions (e.g., {'.jpg', '.png'})

        Returns:
            Validated filename

        Raises:
            ValidationError: If extension not allowed
        """
        path = Path(filename)
        ext = path.suffix.lower()

        if allowed_extensions and ext not in allowed_extensions:
            raise ValidationError(
                f"File extension not allowed: {ext}. "
                f"Allowed: {', '.join(sorted(allowed_extensions))}"
            )

        return filename

    @staticmethod
    def validate_image_file(filename: str) -> str:
        """Validate image file.

        Args:
            filename: Image filename

        Returns:
            Validated filename

        Raises:
            ValidationError: If not valid image file
        """
        return InputValidator.validate_file_extension(
            filename, InputValidator.ALLOWED_IMAGE_EXTENSIONS
        )

    @staticmethod
    def validate_model_file(filename: str) -> str:
        """Validate model file.

        Args:
            filename: Model filename

        Returns:
            Validated filename

        Raises:
            ValidationError: If not valid model file
        """
        return InputValidator.validate_file_extension(
            filename, InputValidator.ALLOWED_MODEL_EXTENSIONS
        )

    @staticmethod
    def validate_string(
        value: str,
        min_length: int = 0,
        max_length: int = 1000,
        pattern: Optional[re.Pattern] = None,
        check_sql_injection: bool = True,
        check_command_injection: bool = True,
    ) -> str:
        """Validate string input.

        Args:
            value: String to validate
            min_length: Minimum length
            max_length: Maximum length
            pattern: Optional regex pattern to match
            check_sql_injection: Check for SQL injection
            check_command_injection: Check for command injection

        Returns:
            Validated string

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(value, str):
            raise ValidationError(f"Expected string, got {type(value)}")

        # Length checks
        if len(value) < min_length:
            raise ValidationError(f"String too short: {len(value)} < {min_length}")

        if len(value) > max_length:
            raise ValidationError(f"String too long: {len(value)} > {max_length}")

        # Pattern check
        if pattern and not pattern.match(value):
            raise ValidationError(f"String does not match pattern: {value}")

        # SQL injection check
        if check_sql_injection and InputValidator.SQL_INJECTION_PATTERN.search(value):
            raise ValidationError(f"Potential SQL injection detected: {value}")

        # Command injection check
        if check_command_injection and InputValidator.COMMAND_INJECTION_PATTERN.search(value):
            raise ValidationError(f"Potential command injection detected: {value}")

        return value

    @staticmethod
    def validate_integer(
        value: Any,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None,
    ) -> int:
        """Validate integer input.

        Args:
            value: Value to validate
            min_value: Minimum allowed value
            max_value: Maximum allowed value

        Returns:
            Validated integer

        Raises:
            ValidationError: If validation fails
        """
        try:
            value = int(value)
        except (ValueError, TypeError) as e:
            raise ValidationError(f"Invalid integer: {e}")

        if min_value is not None and value < min_value:
            raise ValidationError(f"Value too small: {value} < {min_value}")

        if max_value is not None and value > max_value:
            raise ValidationError(f"Value too large: {value} > {max_value}")

        return value

    @staticmethod
    def validate_float(
        value: Any,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> float:
        """Validate float input.

        Args:
            value: Value to validate
            min_value: Minimum allowed value
            max_value: Maximum allowed value

        Returns:
            Validated float

        Raises:
            ValidationError: If validation fails
        """
        try:
            value = float(value)
        except (ValueError, TypeError) as e:
            raise ValidationError(f"Invalid float: {e}")

        if min_value is not None and value < min_value:
            raise ValidationError(f"Value too small: {value} < {min_value}")

        if max_value is not None and value > max_value:
            raise ValidationError(f"Value too large: {value} > {max_value}")

        return value

    @staticmethod
    def validate_batch_size(batch_size: Any) -> int:
        """Validate batch size.

        Args:
            batch_size: Batch size to validate

        Returns:
            Validated batch size

        Raises:
            ValidationError: If invalid
        """
        return InputValidator.validate_integer(batch_size, min_value=1, max_value=512)

    @staticmethod
    def validate_confidence_threshold(threshold: Any) -> float:
        """Validate confidence threshold.

        Args:
            threshold: Threshold to validate

        Returns:
            Validated threshold

        Raises:
            ValidationError: If invalid
        """
        return InputValidator.validate_float(threshold, min_value=0.0, max_value=1.0)

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe storage.

        Args:
            filename: Filename to sanitize

        Returns:
            Sanitized filename
        """
        # Remove path components
        filename = Path(filename).name

        # Remove dangerous characters
        filename = re.sub(r'[<>:"/\\|?*]', "_", filename)

        # Limit length
        if len(filename) > 255:
            name, ext = filename.rsplit(".", 1) if "." in filename else (filename, "")
            filename = name[: 255 - len(ext) - 1] + "." + ext if ext else name[:255]

        return filename


def validate_inference_request(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate inference request data.

    Args:
        data: Request data

    Returns:
        Validated data

    Raises:
        ValidationError: If validation fails
    """
    validated = {}

    # Validate image path
    if "image_path" in data:
        validated["image_path"] = InputValidator.validate_path(
            data["image_path"], must_exist=True
        )
        InputValidator.validate_image_file(str(validated["image_path"]))

    # Validate model path
    if "model_path" in data:
        validated["model_path"] = InputValidator.validate_path(
            data["model_path"], must_exist=True
        )
        InputValidator.validate_model_file(str(validated["model_path"]))

    # Validate batch size
    if "batch_size" in data:
        validated["batch_size"] = InputValidator.validate_batch_size(data["batch_size"])

    # Validate confidence threshold
    if "confidence_threshold" in data:
        validated["confidence_threshold"] = InputValidator.validate_confidence_threshold(
            data["confidence_threshold"]
        )

    # Validate slide ID
    if "slide_id" in data:
        validated["slide_id"] = InputValidator.validate_string(
            data["slide_id"],
            min_length=1,
            max_length=100,
            check_sql_injection=True,
            check_command_injection=True,
        )

    return validated
