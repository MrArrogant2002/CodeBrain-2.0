"""
Input validation module.
All validation functions return (is_valid: bool, error_message: str).
"""

import re
from typing import Tuple

USERNAME_RE = re.compile(r"^[a-zA-Z0-9_]{3,32}$")
PASSWORD_MIN_LENGTH = 8
ALLOWED_ROLES = {"user", "admin", "moderator"}


def validate_username(username: str) -> Tuple[bool, str]:
    """
    Username must be 3-32 characters, alphanumeric + underscore only.
    """
    if not isinstance(username, str):
        return False, "Username must be a string"
    if not USERNAME_RE.match(username):
        return False, "Username must be 3-32 chars, letters/digits/underscore only"
    return True, ""


def validate_password(password: str) -> Tuple[bool, str]:
    """
    Password must be >= 8 chars and contain at least one digit and one letter.
    """
    if not isinstance(password, str):
        return False, "Password must be a string"
    if len(password) < PASSWORD_MIN_LENGTH:
        return False, f"Password must be at least {PASSWORD_MIN_LENGTH} characters"
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one digit"
    if not any(c.isalpha() for c in password):
        return False, "Password must contain at least one letter"
    return True, ""


def validate_role(role: str) -> Tuple[bool, str]:
    """Role must be one of the predefined allowed values."""
    if role not in ALLOWED_ROLES:
        return False, f"Role must be one of {sorted(ALLOWED_ROLES)}"
    return True, ""


def validate_user_input(
    username: str, password: str, role: str = "user"
) -> Tuple[bool, str]:
    """
    Validate all fields for a user registration request.
    Returns on first failure.
    """
    ok, msg = validate_username(username)
    if not ok:
        return False, msg
    ok, msg = validate_password(password)
    if not ok:
        return False, msg
    ok, msg = validate_role(role)
    if not ok:
        return False, msg
    return True, ""


def sanitize_string(value: str, max_length: int = 256) -> str:
    """
    Strip leading/trailing whitespace and truncate to max_length.
    Does NOT HTML-encode — callers must escape for their context.
    """
    return value.strip()[:max_length]
