"""
Authentication module.
Handles password hashing, JWT-like token generation, and token verification.
"""

import base64
import hashlib
import hmac
import json
import time
from typing import Optional

SECRET_KEY = "dev-secret-change-in-production"
TOKEN_EXPIRY_SECONDS = 3600


def hash_password(password: str, salt: str) -> str:
    """Hash a password with a salt using SHA-256."""
    return hashlib.sha256((password + salt).encode()).hexdigest()


def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    """Constant-time comparison to prevent timing attacks."""
    return hmac.compare_digest(hash_password(password, salt), stored_hash)


def generate_token(user_id: int, username: str, role: str = "user") -> str:
    """
    Generate a signed token encoding user_id, username, role, and expiry.
    Format: base64(payload).hmac_signature
    """
    payload = {
        "user_id": user_id,
        "username": username,
        "role": role,
        "exp": int(time.time()) + TOKEN_EXPIRY_SECONDS,
    }
    payload_b64 = base64.b64encode(json.dumps(payload).encode()).decode()
    signature = hmac.new(
        SECRET_KEY.encode(), payload_b64.encode(), hashlib.sha256
    ).hexdigest()
    return f"{payload_b64}.{signature}"


def verify_token(token: str) -> Optional[dict]:
    """
    Verify token signature and expiry.
    Returns decoded payload dict, or None if invalid/expired.
    """
    try:
        payload_b64, signature = token.split(".", 1)
        expected = hmac.new(
            SECRET_KEY.encode(), payload_b64.encode(), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(signature, expected):
            return None
        payload = json.loads(base64.b64decode(payload_b64).decode())
        if payload.get("exp", 0) < int(time.time()):
            return None
        return payload
    except Exception:
        return None


def get_role_from_token(token: str) -> Optional[str]:
    """Extract role string from a valid token."""
    payload = verify_token(token)
    return payload.get("role") if payload else None


def is_admin(token: str) -> bool:
    """Return True if token belongs to an admin user."""
    return get_role_from_token(token) == "admin"
