"""
Ollama embeddings implementation for enhanced GraphRAG.
"""
import logging
import requests
from typing import List, Union
import numpy as np

logger = logging.getLogger(__name__)

class OllamaNomicEmbeddings:
    """Ollama-based embeddings using nomic-embed-text model."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "nomic-embed-text"):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.api_url = f"{self.base_url}/api/embed"
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self) -> None:
        """Test connection to Ollama server."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m.get('name', '') for m in models]
                if self.model in model_names:
                    logger.info(f"✅ Connected to Ollama embeddings with model {self.model}")
                else:
                    logger.warning(f"⚠️ Embedding model {self.model} not found in available models: {model_names}")
            else:
                logger.warning(f"⚠️ Ollama server responded with status {response.status_code}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Ollama for embeddings: {e}")
            raise ConnectionError(f"Cannot connect to Ollama at {self.base_url}")
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        return self.embed_documents([text])[0]
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        try:
            payload = {
                "model": self.model,
                "input": texts
            }
            
            response = requests.post(self.api_url, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            embeddings = result.get('embeddings', [])
            
            if not embeddings:
                raise ValueError("No embeddings returned")
            
            if len(embeddings) != len(texts):
                raise ValueError(f"Expected {len(texts)} embeddings, got {len(embeddings)}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Embedding failed: {e}")
            raise
    
    def embed_single(self, text: str) -> List[float]:
        """Embed a single text (alias for embed_query)."""
        return self.embed_query(text)