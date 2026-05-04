"""
user_filter.py — Utilities for per-user data isolation.

Every query for documents, schemas, jobs, and batches is filtered
by the current user's ID so users only see their own data.

Admin users can see all data (for management purposes).
"""
from __future__ import annotations
from typing import Optional
from sqlalchemy.orm import Query
from app.models.user import User


def filter_by_user(query: Query, user: Optional[User], model) -> Query:
    """
    Filter a SQLAlchemy query to only return records owned by the user.
    Admins see all records. Non-admins only see their own records.
    Strict isolation — records with user_id=None are NOT shown to regular users.
    """
    if user is None:
        # No auth — return nothing
        return query.filter(False)
    if user.is_admin:
        # Admins see everything
        return query
    # Regular users see ONLY their own records
    return query.filter(model.user_id == user.id)


def owned_by(user: Optional[User], record) -> bool:
    """Check if a user owns a record (or is admin)."""
    if user is None:
        return False
    if user.is_admin:
        return True
    record_user_id = getattr(record, "user_id", None)
    return record_user_id == user.id
