"""
TrueScan — Unified Database Engine
===================================
A progressive production-ready database layer that dynamically switches
between PostgreSQL (if DATABASE_URL is set) and local SQLite scans.db.

All CRUD operations in auth and jobs are unified under this engine.
"""
from __future__ import annotations

import os
from loguru import logger
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.environ.get("DATABASE_URL", "")

if DATABASE_URL:
    try:
        # PostgreSQL Production Engine
        engine = create_engine(
            DATABASE_URL,
            pool_size=10,
            max_overflow=5,
            pool_recycle=1800,
            connect_args={}
        )
        logger.success("Production PostgreSQL connection pool established")
    except Exception as e:
        logger.error(f"PostgreSQL connection failed: {e}. Falling back to SQLite.")
        DATABASE_URL = ""

if not DATABASE_URL:
    # Local SQLite Fallback
    _db_dir = os.path.dirname(os.path.abspath(__file__))
    _db_path = os.path.join(_db_dir, "scans.db")
    engine = create_engine(
        f"sqlite:///{_db_path}",
        connect_args={"check_same_thread": False}
    )
    logger.info(f"Local SQLite database engine active: {_db_path}")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db_session():
    """Context manager or dependency for db sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

from sqlalchemy import Column, String, Text

class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_pw = Column(String, nullable=False)
    role = Column(String, default="user")
    created_at = Column(String, nullable=False)

class JobRecord(Base):
    __tablename__ = "jobs"
    id = Column(String, primary_key=True, index=True)
    type = Column(String, nullable=False)
    status = Column(String, default="PENDING", index=True)
    payload = Column(Text, nullable=True)
    result = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(String, nullable=False)
    started_at = Column(String, nullable=True)
    finished_at = Column(String, nullable=True)

# Create all tables at startup if they do not exist
try:
    Base.metadata.create_all(bind=engine)
    logger.success("Database tables verified/created successfully")
except Exception as e:
    logger.error(f"Failed to initialize database tables: {e}")
