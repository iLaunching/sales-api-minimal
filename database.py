"""
Database connection for Sales API
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from models import Base
import os
import logging

logger = logging.getLogger(__name__)

# Database URL from environment (Railway internal connection)
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/sales_db"
)

# Convert postgresql:// to postgresql+asyncpg:// if needed
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

logger.info(f"Database URL configured: {DATABASE_URL.split('@')[0]}@...")

# Create async engine with connection pool settings
try:
    engine = create_async_engine(
        DATABASE_URL,
        echo=False,  # Set to True for SQL query logging
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,  # Verify connections before using
    )

    # Create async session maker
    AsyncSessionLocal = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    logger.info("Database engine created successfully")
except Exception as e:
    logger.error(f"Failed to create database engine: {e}")
    engine = None
    AsyncSessionLocal = None


async def init_db():
    """Initialize database tables"""
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        logger.warning("Continuing without database - health check will still work")
        # Don't raise - allow app to start even if DB fails


async def get_db():
    """Dependency for getting database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def close_db():
    """Close database connections"""
    await engine.dispose()
    logger.info("Database connections closed")
