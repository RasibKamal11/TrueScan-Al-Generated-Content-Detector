"""
TrueScan — Auth Router
========================
POST /auth/register   → create account
POST /auth/login      → get JWT (also accepts OAuth2 form for /docs compatibility)
GET  /auth/me         → current user info
POST /auth/logout     → (client-side: discard token; endpoint for logging)
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from loguru import logger

import auth as auth_module

router = APIRouter(prefix="/auth", tags=["auth"])


# ── Request / Response models ─────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str
    email: str       # EmailStr requires email-validator; use plain str for zero extra deps
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    user:         dict


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/register", status_code=201)
def register(req: RegisterRequest):
    """Create a new user account."""
    if len(req.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    user = auth_module.create_user(req.username, req.email, req.password)
    token = auth_module.create_access_token(user["id"], user["username"], user["role"])
    logger.info(f"New user registered: {req.username}")
    return {"access_token": token, "token_type": "bearer", "user": user}


@router.post("/login")
def login_json(req: LoginRequest):
    """Login with JSON body — for frontend API calls."""
    user = auth_module.authenticate_user(req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = auth_module.create_access_token(user["id"], user["username"], user["role"])
    logger.info(f"User logged in: {req.username}")
    return {"access_token": token, "token_type": "bearer", "user": user}


@router.post("/token")
def login_form(form: OAuth2PasswordRequestForm = Depends()):
    """OAuth2 password flow — makes /auth/login work in FastAPI /docs."""
    user = auth_module.authenticate_user(form.username, form.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = auth_module.create_access_token(user["id"], user["username"], user["role"])
    return {"access_token": token, "token_type": "bearer"}


@router.get("/me")
def me(current_user: dict = Depends(auth_module.get_current_user)):
    """Return current user profile (requires Bearer token)."""
    return current_user


@router.post("/logout")
def logout(current_user: dict = Depends(auth_module.get_current_user)):
    """
    Server-side logout stub.
    JWT tokens are stateless; client must discard the token.
    This endpoint exists for audit-logging purposes.
    """
    logger.info(f"User logged out: {current_user.get('username')}")
    return {"message": "Logged out successfully"}
