"""
auth.py — Authentication endpoints.

POST /api/v1/auth/register   — Create a new account
POST /api/v1/auth/login      — Login and get JWT tokens
POST /api/v1/auth/refresh    — Refresh access token
GET  /api/v1/auth/me         — Get current user profile
PUT  /api/v1/auth/me         — Update profile (name, password)
POST /api/v1/auth/logout     — Logout (client-side token removal)

Admin only:
GET  /api/v1/auth/users      — List all users
PUT  /api/v1/auth/users/{id}/toggle — Enable/disable user
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.auth import (
    hash_password, verify_password,
    create_access_token, create_refresh_token, decode_token,
    get_current_user, require_admin,
)
from app.models.user import User

router = APIRouter(prefix="/auth", tags=["Auth"])


# ── Request / Response models ─────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: str = Field(..., description="Email address")
    password: str = Field(..., min_length=8, description="Password (min 8 characters)")
    full_name: Optional[str] = Field(default=None)


class LoginRequest(BaseModel):
    email: str
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class UpdateProfileRequest(BaseModel):
    full_name: Optional[str] = None
    current_password: Optional[str] = None
    new_password: Optional[str] = Field(default=None, min_length=8)


class UserResponse(BaseModel):
    id: str
    email: str
    full_name: Optional[str]
    is_active: bool
    is_admin: bool
    created_at: Optional[str]
    last_login: Optional[str]


def _user_to_response(user: User) -> dict:
    return {
        "id": user.id,
        "email": user.email,
        "full_name": user.full_name,
        "is_active": user.is_active,
        "is_admin": user.is_admin,
        "created_at": user.created_at.isoformat() if user.created_at else None,
        "last_login": user.last_login.isoformat() if user.last_login else None,
    }


# ── Register ──────────────────────────────────────────────────────────────────

@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(req: RegisterRequest, db: Session = Depends(get_db)):
    """
    Create a new user account.
    The first registered user is automatically made admin.
    """
    # Check if email already exists
    existing = db.query(User).filter(User.email == req.email.lower().strip()).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    # Validate password strength
    if len(req.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters.")

    # First user becomes admin
    is_first_user = db.query(User).count() == 0

    user = User(
        email=req.email.lower().strip(),
        hashed_password=hash_password(req.password),
        full_name=req.full_name,
        is_active=True,
        is_admin=is_first_user,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    access_token = create_access_token(user.id, user.email)
    refresh_token = create_refresh_token(user.id)

    return {
        "message": "Account created successfully",
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "user": _user_to_response(user),
        "is_admin": user.is_admin,
    }


# ── Login ─────────────────────────────────────────────────────────────────────

@router.post("/login")
async def login(req: LoginRequest, db: Session = Depends(get_db)):
    """Login with email and password. Returns JWT access + refresh tokens."""
    user = db.query(User).filter(User.email == req.email.lower().strip()).first()

    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your account has been disabled. Contact an administrator.",
        )

    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()

    access_token = create_access_token(user.id, user.email)
    refresh_token = create_refresh_token(user.id)

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "user": _user_to_response(user),
    }


# ── Refresh token ─────────────────────────────────────────────────────────────

@router.post("/refresh")
async def refresh_token(req: RefreshRequest, db: Session = Depends(get_db)):
    """Exchange a refresh token for a new access token."""
    payload = decode_token(req.refresh_token)

    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    user_id = payload.get("sub")
    user = db.query(User).filter(User.id == user_id).first()

    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or disabled")

    new_access_token = create_access_token(user.id, user.email)

    return {
        "access_token": new_access_token,
        "token_type": "bearer",
    }


# ── Get current user ──────────────────────────────────────────────────────────

@router.get("/me")
async def get_me(current_user: User = Depends(get_current_user)):
    """Get the currently authenticated user's profile."""
    return _user_to_response(current_user)


# ── Update profile ────────────────────────────────────────────────────────────

@router.put("/me")
async def update_profile(
    req: UpdateProfileRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update name or password."""
    if req.full_name is not None:
        current_user.full_name = req.full_name

    if req.new_password:
        if not req.current_password:
            raise HTTPException(400, "Current password required to set a new password.")
        if not verify_password(req.current_password, current_user.hashed_password):
            raise HTTPException(400, "Current password is incorrect.")
        current_user.hashed_password = hash_password(req.new_password)

    db.commit()
    db.refresh(current_user)

    return {
        "message": "Profile updated successfully",
        "user": _user_to_response(current_user),
    }


# ── Logout ────────────────────────────────────────────────────────────────────

@router.post("/logout")
async def logout(current_user: User = Depends(get_current_user)):
    """
    Logout endpoint. JWT tokens are stateless so actual invalidation
    happens on the client by deleting the stored token.
    """
    return {"message": "Logged out successfully"}


# ── Admin: list users ─────────────────────────────────────────────────────────

@router.get("/users")
async def list_users(
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Admin only: list all registered users."""
    users = db.query(User).order_by(User.created_at.desc()).all()
    return {
        "users": [_user_to_response(u) for u in users],
        "total": len(users),
    }


@router.put("/users/{user_id}/toggle")
async def toggle_user(
    user_id: str,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Admin only: enable or disable a user account."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(404, "User not found")
    if user.id == admin.id:
        raise HTTPException(400, "Cannot disable your own account")

    user.is_active = not user.is_active
    db.commit()

    return {
        "message": f"User {'enabled' if user.is_active else 'disabled'} successfully",
        "user": _user_to_response(user),
    }


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: str,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Admin only: delete a user account."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(404, "User not found")
    if user.id == admin.id:
        raise HTTPException(400, "Cannot delete your own account")

    db.delete(user)
    db.commit()
    return {"deleted": user_id}


# ── Admin: reset any user's password ─────────────────────────────────────────

class ResetPasswordRequest(BaseModel):
    email: str
    new_password: str = Field(..., min_length=8)


@router.post("/reset-password")
async def reset_password(
    req: ResetPasswordRequest,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Admin only: reset any user's password."""
    user = db.query(User).filter(User.email == req.email.lower().strip()).first()
    if not user:
        raise HTTPException(404, "User not found")
    user.hashed_password = hash_password(req.new_password)
    db.commit()
    return {"message": f"Password reset for {user.email}"}
