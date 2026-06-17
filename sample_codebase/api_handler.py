"""
API handler module.
REST-like request handlers that orchestrate auth, database, and validation.
Each handler accepts a plain dict (the "request body") and returns a dict response.
"""

import logging
import os
from typing import Any, Dict

from sample_codebase import auth, database, data_validator

logger = logging.getLogger(__name__)


def _make_response(success: bool, data: Any = None, error: str = "") -> Dict[str, Any]:
    return {"success": success, "data": data, "error": error}


def register_user(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    POST /register
    Expected body keys: username, password, role (optional)
    """
    username = data_validator.sanitize_string(body.get("username", ""))
    password = body.get("password", "")
    role = body.get("role", "user")

    ok, msg = data_validator.validate_user_input(username, password, role)
    if not ok:
        return _make_response(False, error=msg)

    salt = os.urandom(16).hex()
    password_hash = auth.hash_password(password, salt)

    try:
        user_id = database.insert_user(username, password_hash, salt, role)
    except ValueError as exc:
        return _make_response(False, error=str(exc))

    database.log_action(user_id, "register")
    logger.info("Registered user %s (id=%d)", username, user_id)
    return _make_response(
        True, data={"user_id": user_id, "username": username, "role": role}
    )


def login(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    POST /login
    Expected body keys: username, password
    Returns a signed token on success.
    """
    username = data_validator.sanitize_string(body.get("username", ""))
    password = body.get("password", "")

    ok, msg = data_validator.validate_username(username)
    if not ok:
        return _make_response(False, error=msg)

    user = database.get_user_by_username(username)
    if user is None:
        return _make_response(False, error="Invalid credentials")

    if not auth.verify_password(password, user["password_hash"], user["salt"]):
        return _make_response(False, error="Invalid credentials")

    token = auth.generate_token(user["id"], user["username"], user["role"])
    database.log_action(user["id"], "login")
    logger.info("User %s logged in", username)
    return _make_response(True, data={"token": token})


def get_profile(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    GET /profile
    Expected body keys: token
    """
    token = body.get("token", "")
    payload = auth.verify_token(token)
    if payload is None:
        return _make_response(False, error="Invalid or expired token")

    user = database.get_user_by_id(payload["user_id"])
    if user is None:
        return _make_response(False, error="User not found")

    return _make_response(
        True,
        data={
            "user_id": user["id"],
            "username": user["username"],
            "role": user["role"],
        },
    )


def admin_list_users(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    GET /admin/users  (admin-only)
    Expected body keys: token
    """
    token = body.get("token", "")
    if not auth.is_admin(token):
        return _make_response(False, error="Admin access required")

    users = database.list_all_users()
    return _make_response(True, data={"users": users, "count": len(users)})
