from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: str

    chroma_host: str = "chromadb"
    chroma_port: int = 8000
    collection_name: str = "audit_documents"

    data_dir: str = "/app/data"
    db_path: str = "/app/db/audit.db"

    model: str = "claude-sonnet-4-5"
    judge_model: str = "claude-haiku-4-5-20251001"
    embedding_model: str = "all-MiniLM-L6-v2"

    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5

    router_mode: str = "hybrid"   # regex | embedding | llm | hybrid | compare


settings = Settings()
