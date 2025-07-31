"""
Configuration module for UCKG GraphRAG with Post-Traversal Ranking.
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    """Application settings for GraphRAG with post-traversal ranking."""
    
    # Neo4j Configuration
    neo4j_uri: str = Field(default="bolt://localhost:7687", env="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", env="NEO4J_USER") 
    neo4j_password: str = Field(env="NEO4J_PASSWORD")
    
    # Ollama Configuration
    ollama_url: str = Field(default="http://localhost:11434", env="OLLAMA_URL")
    embedding_model: str = Field(default="nomic-embed-text:latest", env="EMBEDDING_MODEL")
    
    # Vector Index Configuration  
    vector_index_name: str = Field(default="uckg_universal_embeddings", env="VECTOR_INDEX_NAME")
    vector_dimension: int = Field(default=768, env="VECTOR_DIMENSION")
    top_k_results: int = Field(default=3, env="TOP_K_RESULTS")  # Reduced for focused traversal
    
    # Ranking Configuration (Embedding-optimized mode)
    embedding_similarity_threshold: float = Field(default=0.60, env="EMBEDDING_THRESHOLD")  # Lowered to include direct connections
    llm_ranking_threshold: float = Field(default=7.0, env="LLM_THRESHOLD")  # Unused but kept for compatibility
    max_final_results: int = Field(default=10, env="MAX_RESULTS")  # Top 10 results for better coverage
    use_llm_ranking: bool = Field(default=False, env="USE_LLM_RANKING")  # Permanently disabled for speed
    use_embedding_ranking: bool = Field(default=True, env="USE_EMBEDDING_RANKING")  # Primary ranking method
    
    # Application Configuration
    app_host: str = Field(default="0.0.0.0", env="APP_HOST")
    app_port: int = Field(default=8000, env="APP_PORT")
    debug: bool = Field(default=True, env="DEBUG")
    
    # LLM Configuration for response generation
    llm_model: str = Field(default="llama3.2:3b", env="LLM_MODEL")
    llm_temperature: float = Field(default=0.1, env="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=1000, env="LLM_MAX_TOKENS")
    
    # Dual-model optimization: smaller model for ranking tasks
    ranking_model: str = Field(default="llama3.2:1b", env="RANKING_MODEL")
    ranking_max_tokens: int = Field(default=200, env="RANKING_MAX_TOKENS")  # Reduced for ranking
    
    # CORS Configuration
    cors_origins: list = ["http://localhost:3000", "http://localhost:3001"]
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra environment variables like OPENAI_API_KEY

# Global settings instance
settings = Settings()

def get_neo4j_config() -> dict:
    """Get Neo4j connection configuration."""
    return {
        "uri": settings.neo4j_uri,
        "auth": (settings.neo4j_user, settings.neo4j_password)
    }

def get_ollama_config() -> dict:
    """Get Ollama configuration for embeddings and LLM."""
    return {
        "base_url": settings.ollama_url,
        "embedding_model": settings.embedding_model,
        "llm_model": settings.llm_model,
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_tokens,
        "ranking_model": settings.ranking_model,
        "ranking_max_tokens": settings.ranking_max_tokens
    }

def get_vector_config() -> dict:
    """Get vector index configuration."""
    return {
        "index_name": settings.vector_index_name,
        "dimensions": settings.vector_dimension,
        "top_k": settings.top_k_results
    }

def get_ranking_config() -> dict:
    """Get ranking configuration for post-traversal ranking."""
    return {
        "embedding_threshold": settings.embedding_similarity_threshold,
        "llm_threshold": settings.llm_ranking_threshold,
        "max_results": settings.max_final_results,
        "use_llm_ranking": settings.use_llm_ranking,
        "use_embedding_ranking": settings.use_embedding_ranking
    } 