from pathlib import Path
from pydantic_settings import BaseSettings,SettingsConfigDict

class Settings(BaseSettings):
    
    CHUNK_SIZE:int= 500
    CHUNK_OVERLAP:int= 200
    TOP_K:int= 5
    
    CHROMA_PERSISTENT_DIRECTORY:Path = "data/chromadb_storage"
    RAW_DOCS_DIR:Path = "data/raw_documents"
    
    
    EMBEDDING_MODEL_NAME:str = "sentence-transformers/all-MiniLM-L6-v2"
    GROQ_API_KEY:str = ""
    GROQ_MODEL_NAME:str = ""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
settings = Settings()