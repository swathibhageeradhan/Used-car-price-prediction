"""SQLite persistence layer for AutoPrice AI user accounts.

Uses only the Python standard library (`sqlite3`) plus Werkzeug's password
hashing helpers, which ship with Flask. No extra packages required.

The database file is created automatically on first run, along with the
table schema and a default admin account, so the app works out of the box.
"""
import os
import sqlite3

from flask import current_app, g
from werkzeug.security import check_password_hash, generate_password_hash

# Seeded on first run so there is always a working login.
DEFAULT_ADMIN_EMAIL = "admin@example.com"
DEFAULT_ADMIN_PASSWORD = "password123"

MIN_PASSWORD_LENGTH = 8


def get_db():
    """Return the SQLite connection for the current request."""
    if "db" not in g:
        os.makedirs(os.path.dirname(current_app.config["DATABASE"]), exist_ok=True)
        g.db = sqlite3.connect(current_app.config["DATABASE"])
        g.db.row_factory = sqlite3.Row      # lets us read rows by column name
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


def close_db(exception=None):
    """Close the request-scoped connection. Registered via teardown_appcontext."""
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    """Create the schema (if missing) and seed the default admin user."""
    db = get_db()
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            email         TEXT    NOT NULL UNIQUE,
            password_hash TEXT    NOT NULL,
            full_name     TEXT    NOT NULL,
            created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    db.commit()
    seed_admin(db)


def seed_admin(db):
    """Insert the default admin account the first time the DB is created."""
    existing = db.execute(
        "SELECT id FROM users WHERE email = ?", (DEFAULT_ADMIN_EMAIL,)
    ).fetchone()
    if existing is None:
        db.execute(
            "INSERT INTO users (email, password_hash, full_name) VALUES (?, ?, ?)",
            (
                DEFAULT_ADMIN_EMAIL,
                generate_password_hash(DEFAULT_ADMIN_PASSWORD),
                "System Admin",
            ),
        )
        db.commit()


# --------------------------------------------------------------------------
# Queries
# --------------------------------------------------------------------------
def get_user_by_email(email):
    """Fetch a single user row by email, or None."""
    return get_db().execute(
        "SELECT * FROM users WHERE email = ?", (email,)
    ).fetchone()


def email_exists(email):
    """True when an account already uses this email address."""
    return get_user_by_email(email) is not None


def create_user(email, password, full_name):
    """Insert a new user with a hashed password. Returns the new row id."""
    db = get_db()
    cursor = db.execute(
        "INSERT INTO users (email, password_hash, full_name) VALUES (?, ?, ?)",
        (email, generate_password_hash(password), full_name),
    )
    db.commit()
    return cursor.lastrowid


def count_users():
    """Total number of registered accounts (shown on the dashboard)."""
    return get_db().execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"]


def authenticate(email, password):
    """Return the user row when the credentials are valid, otherwise None."""
    user = get_user_by_email(email)
    if user is None:
        return None
    if not check_password_hash(user["password_hash"], password):
        return None
    return user
