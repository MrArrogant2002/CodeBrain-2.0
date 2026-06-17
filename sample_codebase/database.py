"""
Database module.
SQLite-backed storage for users and audit logs.
"""

import logging
import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
DB_PATH = "codebrain_demo.db"


def get_connection() -> sqlite3.Connection:
    """Open a SQLite connection with row_factory set for dict-like access."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def db_session():
    """Context manager: commits on success, rolls back on exception."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception as exc:
        conn.rollback()
        logger.error("Transaction failed: %s", exc)
        raise
    finally:
        conn.close()


def initialize_schema() -> None:
    """Create tables if they do not exist."""
    with db_session() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt          TEXT NOT NULL,
                role          TEXT DEFAULT 'user',
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id   INTEGER,
                action    TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
            """)


def insert_user(
    username: str, password_hash: str, salt: str, role: str = "user"
) -> int:
    """
    Insert a new user row.
    Returns the new user's id.
    Raises ValueError if username is already taken.
    """
    with db_session() as conn:
        try:
            cursor = conn.execute(
                "INSERT INTO users (username, password_hash, salt, role) VALUES (?, ?, ?, ?)",
                (username, password_hash, salt, role),
            )
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            raise ValueError(f"Username '{username}' already exists")


def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    """Fetch one user row by username, or None."""
    with db_session() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """Fetch one user row by id, or None."""
    with db_session() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None


def list_all_users() -> List[Dict[str, Any]]:
    """Return all users (id, username, role, created_at only — no hashes)."""
    with db_session() as conn:
        rows = conn.execute(
            "SELECT id, username, role, created_at FROM users"
        ).fetchall()
        return [dict(r) for r in rows]


def log_action(user_id: int, action: str) -> None:
    """Append an entry to audit_log."""
    with db_session() as conn:
        conn.execute(
            "INSERT INTO audit_log (user_id, action) VALUES (?, ?)",
            (user_id, action),
        )
