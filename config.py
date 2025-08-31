import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).parent

class Settings(BaseSettings):
    # API Configuration
    api_token: str = os.getenv("API_TOKEN", "hackathon-secret-token")
    port: int = int(os.getenv("PORT", "8002"))
    host: str = os.getenv("HOST", "0.0.0.0")
    log_level: str = os.getenv("LOG_LEVEL", "info")
    
    # ngrok Configuration
    ngrok_token: str = os.getenv("NGROK_TOKEN", "")
    use_ngrok: bool = os.getenv("USE_NGROK", "false").lower() == "true"
    ngrok_path: str = os.getenv("NGROK_PATH", str(BASE_DIR / "ngrok.exe"))
    
    # Model Configuration
    artifacts_dir: str = os.getenv("ARTIFACTS_DIR", "./artifacts_backup")
    model_version: str = os.getenv("MODEL_VERSION", "mnv2-local-v1")
    
    @property
    def model_path(self) -> str:
        return os.path.join(self.artifacts_dir, "model_mnv2.pt")
    
    @property
    def class_path(self) -> str:
        return os.path.join(self.artifacts_dir, "class_index.json")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Create global settings instance
settings = Settings()

# Validate required files
def validate_setup():
    """Validate that all required files and configurations are present"""
    errors = []
    
    if not os.path.exists(settings.model_path):
        errors.append(f"❌ Model file not found: {settings.model_path}")
    
    if not os.path.exists(settings.class_path):
        errors.append(f"❌ Class file not found: {settings.class_path}")
    
    if settings.use_ngrok:
        if not settings.ngrok_token:
            errors.append("❌ NGROK_TOKEN is required when USE_NGROK=true")
        if not os.path.exists(settings.ngrok_path):
            errors.append(f"❌ ngrok.exe not found: {settings.ngrok_path}")
            errors.append("💡 Download from: https://ngrok.com/download")
            errors.append("💡 Place ngrok.exe in project root directory")
            # Don't fail completely, just disable ngrok
            settings.use_ngrok = False
    
    if errors:
        for error in errors:
            print(error)
        return False
    
    return True