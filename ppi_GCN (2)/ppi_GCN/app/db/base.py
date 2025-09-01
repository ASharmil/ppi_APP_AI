"""
Database base models
"""
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

# Import all models to ensure they're registered
from app.db.models.user import User
from app.db.models.institution import Institution
from app.db.models.protein import Protein
from app.db.models.prediction import Prediction
from app.db.models.drug import Drug
from app.db.models.job import Job
from app.db.models.blockchain_event import BlockchainEvent

# app/db/session.py
"""
Database session management
"""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.core.config import settings

# Create async engine
engine = create_async_engine(
    settings.DATABASE_URL_ASYNC,
    echo=settings.DEBUG,
    pool_pre_ping=True,
    pool_recycle=300
)

# Create session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

async def get_db() -> AsyncSession:
    """Dependency to get database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()