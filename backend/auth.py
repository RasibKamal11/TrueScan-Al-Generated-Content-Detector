"""
TrueScan — JWT Authentication Helpers
=======================================
Zero-external-service auth using:
  - bcrypt for password hashing  (via passlib[bcrypt])
  - python-jose for JWT signing  (HS256)
  - SQLite for user storage

No Supabase / Auth0 / NextAuth.js required.
"""

from __future__ import annotations

import os
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from loguru import logger

# ── Optional deps — graceful fallback if not installed ────────────────────────
try:
    from passlib.context import CryptContext
    from jose import JWTError, jwt
    _AUTH_AVAILABLE = True
except ImportError:
    _AUTH_AVAILABLE = False
    logger.warning(
        "Auth deps missing. Run: pip install passlib[bcrypt] python-jose[cryptography]"
    )

# ── Config ────────────────────────────────────────────────────────────────────

SECRET_KEY      = os.environ.get("SECRET_KEY", "change-me-in-production-" + uuid.uuid4().hex)
ALGORITHM       = "HS256"
EXPIRE_MINUTES  = int(os.environ.get("JWT_EXPIRE_MINUTES", "1440"))  # 24h default
_DB_PATH        = os.path.join(os.path.dirname(__file__), "scans.db")

if _AUTH_AVAILABLE:
    _pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
    oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)
else:
    oauth2_scheme = None  # type: ignore


# ── User DB ───────────────────────────────────────────────────────────────────

from database import SessionLocal, User


# ── Password helpers ──────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    if not _AUTH_AVAILABLE:
        raise RuntimeError("passlib not installed")
    return _pwd_ctx.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    if not _AUTH_AVAILABLE:
        return False
    return _pwd_ctx.verify(plain, hashed)


# ── User CRUD ─────────────────────────────────────────────────────────────────

def create_user(username: str, email: str, password: str) -> dict:
    db = SessionLocal()
    try:
        existing = db.query(User).filter((User.username == username) | (User.email == email)).first()
        if existing:
            raise HTTPException(status_code=409, detail="Username or email already registered")
        user_id = str(uuid.uuid4())
        new_user = User(
            id=user_id,
            username=username,
            email=email,
            hashed_pw=hash_password(password),
            role="user",
            created_at=datetime.now(timezone.utc).isoformat()
        )
        db.add(new_user)
        db.commit()
        return {"id": user_id, "username": username, "email": email, "role": "user"}
    finally:
        db.close()


def authenticate_user(username: str, password: str) -> Optional[dict]:
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == username).first()
        if not user:
            return None
        if not verify_password(password, user.hashed_pw):
            return None
        return {"id": user.id, "username": user.username, "email": user.email, "role": user.role}
    finally:
        db.close()


def get_user_by_id(user_id: str) -> Optional[dict]:
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return None
        return {"id": user.id, "username": user.username, "email": user.email, "role": user.role, "created_at": user.created_at}
    finally:
        db.close()


# ── JWT ───────────────────────────────────────────────────────────────────────

def create_access_token(user_id: str, username: str, role: str = "user") -> str:
    if not _AUTH_AVAILABLE:
        raise RuntimeError("python-jose not installed")
    expire = datetime.now(timezone.utc) + timedelta(minutes=EXPIRE_MINUTES)
    payload = {
        "sub":      user_id,
        "username": username,
        "role":     role,
        "exp":      expire,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    if not _AUTH_AVAILABLE:
        raise HTTPException(status_code=503, detail="Auth not available — install passlib & python-jose")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── FastAPI dependency ────────────────────────────────────────────────────────

async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """FastAPI dependency. Raises 401 if token missing/invalid."""
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = decode_token(token)
    user = get_user_by_id(payload["sub"])
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


async def get_current_user_optional(token: str = Depends(oauth2_scheme)) -> Optional[dict]:
    """Same as get_current_user but returns None instead of raising."""
    if not token:
        return None
    try:
        payload = decode_token(token)
        return get_user_by_id(payload["sub"])
    except Exception:
        return None
