"""
Password Strength Validation

Provides comprehensive password strength checking beyond basic requirements.
"""

import re
from typing import List, Tuple


class PasswordStrengthChecker:
    """Check password strength and provide feedback."""

    # Common weak passwords to reject
    COMMON_PASSWORDS = {
        "password",
        "123456",
        "12345678",
        "qwerty",
        "abc123",
        "monkey",
        "1234567",
        "letmein",
        "trustno1",
        "dragon",
        "baseball",
        "111111",
        "iloveyou",
        "master",
        "sunshine",
        "ashley",
        "bailey",
        "passw0rd",
        "shadow",
        "123123",
        "654321",
        "superman",
        "qazwsx",
        "michael",
        "football",
    }

    @staticmethod
    def check_strength(password: str) -> Tuple[int, List[str]]:
        """Check password strength and return score with feedback.

        Args:
            password: Password to check

        Returns:
            Tuple of (score 0-100, list of feedback messages)
        """
        score = 0
        feedback = []

        # Length scoring
        length = len(password)
        if length < 8:
            feedback.append("Password must be at least 8 characters")
            return 0, feedback
        elif length >= 8:
            score += 20
        if length >= 12:
            score += 10
        if length >= 16:
            score += 10

        # Character variety
        has_lower = bool(re.search(r"[a-z]", password))
        has_upper = bool(re.search(r"[A-Z]", password))
        has_digit = bool(re.search(r"\d", password))
        has_special = bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', password))

        char_types = sum([has_lower, has_upper, has_digit, has_special])

        if char_types == 1:
            feedback.append("Use a mix of character types")
            score += 10
        elif char_types == 2:
            score += 20
        elif char_types == 3:
            score += 30
        elif char_types == 4:
            score += 40

        # Check for common patterns
        if password.lower() in PasswordStrengthChecker.COMMON_PASSWORDS:
            feedback.append("This is a commonly used password")
            score = min(score, 20)

        # Check for sequential characters
        if re.search(r"(012|123|234|345|456|567|678|789|890|abc|bcd|cde)", password.lower()):
            feedback.append("Avoid sequential characters")
            score -= 10

        # Check for repeated characters
        if re.search(r"(.)\1{2,}", password):
            feedback.append("Avoid repeated characters")
            score -= 10

        # Check for keyboard patterns
        keyboard_patterns = ["qwerty", "asdfgh", "zxcvbn", "1qaz2wsx"]
        if any(pattern in password.lower() for pattern in keyboard_patterns):
            feedback.append("Avoid keyboard patterns")
            score -= 15

        # Ensure score is in valid range
        score = max(0, min(100, score))

        # Generate strength feedback
        if score < 40:
            feedback.insert(0, "Weak password")
        elif score < 60:
            feedback.insert(0, "Fair password")
        elif score < 80:
            feedback.insert(0, "Good password")
        else:
            feedback.insert(0, "Strong password")

        return score, feedback

    @staticmethod
    def meets_minimum_requirements(password: str) -> bool:
        """Check if password meets minimum security requirements.

        Args:
            password: Password to check

        Returns:
            True if meets requirements
        """
        if len(password) < 8:
            return False

        has_lower = bool(re.search(r"[a-z]", password))
        has_upper = bool(re.search(r"[A-Z]", password))
        has_digit = bool(re.search(r"\d", password))
        has_special = bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', password))

        # Require at least 3 of 4 character types
        char_types = sum([has_lower, has_upper, has_digit, has_special])

        return char_types >= 3
