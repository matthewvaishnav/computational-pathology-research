"""
Input validation utilities for deployment modules.

Prevents injection attacks by validating user-controlled identifiers.
"""

import re


class ValidationError(ValueError):
    """Raised when input validation fails."""



def validate_site_id(site_id: str, max_length: int = 50) -> str:
    """
    Validate hospital site identifier.

    Args:
        site_id: Site identifier to validate
        max_length: Maximum allowed length

    Returns:
        Validated site_id

    Raises:
        ValidationError: If validation fails
    """
    if not site_id:
        raise ValidationError("site_id cannot be empty")

    if len(site_id) > max_length:
        raise ValidationError(f"site_id exceeds maximum length of {max_length}")

    if not re.match(r"^[a-zA-Z0-9_-]+$", site_id):
        raise ValidationError(
            f"site_id contains invalid characters: {site_id}. "
            "Only alphanumeric, underscore, and hyphen allowed"
        )

    return site_id


def validate_patient_id(patient_id: str, max_length: int = 100) -> str:
    """
    Validate patient identifier.

    Args:
        patient_id: Patient identifier to validate
        max_length: Maximum allowed length

    Returns:
        Validated patient_id

    Raises:
        ValidationError: If validation fails
    """
    if not patient_id:
        raise ValidationError("patient_id cannot be empty")

    if len(patient_id) > max_length:
        raise ValidationError(f"patient_id exceeds maximum length of {max_length}")

    if not re.match(r"^[a-zA-Z0-9_-]+$", patient_id):
        raise ValidationError(
            f"patient_id contains invalid characters: {patient_id}. "
            "Only alphanumeric, underscore, and hyphen allowed"
        )

    return patient_id


def validate_user_id(user_id: str, max_length: int = 100) -> str:
    """
    Validate user identifier.

    Args:
        user_id: User identifier to validate
        max_length: Maximum allowed length

    Returns:
        Validated user_id

    Raises:
        ValidationError: If validation fails
    """
    if not user_id:
        raise ValidationError("user_id cannot be empty")

    if len(user_id) > max_length:
        raise ValidationError(f"user_id exceeds maximum length of {max_length}")

    if not re.match(r"^[a-zA-Z0-9_@.-]+$", user_id):
        raise ValidationError(
            f"user_id contains invalid characters: {user_id}. "
            "Only alphanumeric, underscore, @, dot, and hyphen allowed"
        )

    return user_id


def validate_case_id(case_id: str, max_length: int = 100) -> str:
    """
    Validate case identifier.

    Args:
        case_id: Case identifier to validate
        max_length: Maximum allowed length

    Returns:
        Validated case_id

    Raises:
        ValidationError: If validation fails
    """
    if not case_id:
        raise ValidationError("case_id cannot be empty")

    if len(case_id) > max_length:
        raise ValidationError(f"case_id exceeds maximum length of {max_length}")

    if not re.match(r"^[a-zA-Z0-9_-]+$", case_id):
        raise ValidationError(
            f"case_id contains invalid characters: {case_id}. "
            "Only alphanumeric, underscore, and hyphen allowed"
        )

    return case_id
