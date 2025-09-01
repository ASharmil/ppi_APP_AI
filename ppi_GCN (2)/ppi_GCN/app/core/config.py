from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    PROJECT_NAME: str = "PPI Prediction Backend"
    VERSION: str = "1.0.0"
    
    # Database
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/ppi_db"
    TEST_DATABASE_URL: str = "postgresql://user:password@localhost:5432/ppi_test_db"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # JWT
    SECRET_KEY: str = "your-super-secret-jwt-key-here"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # API Keys
    DRUGBANK_API_KEY: str = ""
    CHEMBL_API_KEY: str = ""
    PUBCHEM_API_KEY: str = ""
    
    # Blockchain
    ETHEREUM_RPC_URL: str = "http://localhost:8545"
    CONTRACT_ADDRESS: str = ""
    PRIVATE_KEY: str = ""
    
    # Storage
    STORAGE_PATH: str = "./storage"
    MODELS_PATH: str = "./models"
    
    # Hardware
    ARDUINO_PORT: str = "/dev/ttyUSB0"
    ARDUINO_BAUD_RATE: int = 9600
    
    # Security
    ALLOWED_HOSTS: List[str] = ["*"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()

# Ensure directories exist
os.makedirs(settings.STORAGE_PATH, exist_ok=True)
os.makedirs(settings.MODELS_PATH, exist_ok=True)