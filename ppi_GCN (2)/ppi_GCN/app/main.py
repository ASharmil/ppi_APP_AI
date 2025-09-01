from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.core.config import settings
from app.core.logging_conf import setup_logging
from app.db.session import engine
from app.db.models import user, institution, protein, prediction, drug, job, blockchain_event
from app.api.v1 import routes_auth, routes_ppi, routes_docking, routes_drugs, routes_institutions, routes_models, routes_admin, routes_hooks
import asyncio

setup_logging()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting PPI Prediction Backend...")
    yield
    # Shutdown
    print("Shutting down...")

app = FastAPI(
    title="PPI Prediction & Drug Discovery API",
    description="Production-ready backend for AI-powered protein-protein interaction prediction and drug discovery",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(routes_auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(routes_ppi.router, prefix="/api/v1/ppi", tags=["PPI Prediction"])
app.include_router(routes_docking.router, prefix="/api/v1/docking", tags=["Docking"])
app.include_router(routes_drugs.router, prefix="/api/v1/drugs", tags=["Drug Discovery"])
app.include_router(routes_institutions.router, prefix="/api/v1/institutions", tags=["Institutions"])
app.include_router(routes_models.router, prefix="/api/v1/models", tags=["Model Training"])
app.include_router(routes_admin.router, prefix="/api/v1/admin", tags=["Administration"])
app.include_router(routes_hooks.router, prefix="/api/v1/hooks", tags=["Hardware Hooks"])

@app.get("/healthz")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

@app.get("/readyz")
async def readiness_check():
    # Check database connectivity, Redis, etc.
    try:
        from app.db.session import SessionLocal
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        return {"status": "ready"}
    except Exception as e:
        return {"status": "not ready", "error": str(e)}
