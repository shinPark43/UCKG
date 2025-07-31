"""
Ollama LLM implementation for Neo4j GraphRAG using local models.
Compatible with Neo4j GraphRAG LLM interface.
"""
import logging
import requests
import json
import asyncio
from typing import List, Optional, Dict, Any
from neo4j_graphrag.llm.base import LLMInterface

logger = logging.getLogger(__name__)

class OllamaLLM(LLMInterface):
    """
    Ollama-based LLM implementation for local response generation.
    Compatible with Neo4j GraphRAG LLM interface.
    """
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2:3b", **kwargs):
        """
        Initialize Ollama LLM client.
        
        Args:
            base_url: Ollama server URL
            model: LLM model name
            **kwargs: Additional model parameters
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.api_url = f"{self.base_url}/api/generate"
        
        # Model parameters
        self.temperature = kwargs.get('temperature', 0.1)
        self.max_tokens = kwargs.get('max_tokens', 1000)
        self.top_p = kwargs.get('top_p', 0.9)
        
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
                    logger.info(f"✅ Connected to Ollama with model {self.model}")
                else:
                    logger.warning(f"⚠️ Model {self.model} not found in available models: {model_names}")
            else:
                logger.warning(f"⚠️ Ollama server responded with status {response.status_code}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Ollama: {e}")
            raise ConnectionError(f"Cannot connect to Ollama at {self.base_url}")
    
    def invoke(self, input_text: str) -> str:
        """
        Generate response from input text.
        
        Args:
            input_text: The prompt text
            
        Returns:
            Generated response text
        """
        try:
            payload = {
                "model": self.model,
                "prompt": input_text,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                    "top_p": self.top_p
                }
            }
            
            response = requests.post(self.api_url, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            generated_text = result.get('response', '').strip()
            
            if not generated_text:
                logger.warning("Empty response from Ollama")
                return "I apologize, but I couldn't generate a proper response. Please try again."
                
            return generated_text
            
        except requests.exceptions.Timeout:
            logger.error("Ollama request timed out")
            return "I apologize, but the response took too long to generate. Please try again."
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Ollama API request failed: {e}")
            return "I apologize, but I encountered an error generating the response. Please try again."
        except Exception as e:
            logger.error(f"❌ LLM generation failed: {e}")
            return "I apologize, but I encountered an unexpected error. Please try again."

    async def ainvoke(self, input_text: str) -> str:
        """
        Async version of invoke for compatibility with LLMInterface.
        
        Args:
            input_text: The prompt text
            
        Returns:
            Generated response text
        """
        # Run the synchronous invoke method in a thread pool to make it async
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.invoke, input_text)

class OllamaLLMOptimized(OllamaLLM):
    """
    Optimized Ollama LLM with cybersecurity-focused prompting and dual-model support.
    """
    
    def __init__(self, *args, **kwargs):
        # Extract ranking model configuration
        self.ranking_model = kwargs.pop('ranking_model', None)
        self.ranking_max_tokens = kwargs.pop('ranking_max_tokens', 200)
        super().__init__(*args, **kwargs)
        
        # Test ranking model connection if provided
        if self.ranking_model:
            self._test_ranking_model_connection()
    
    def _test_ranking_model_connection(self) -> None:
        """Test connection to the ranking model."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m.get('name', '') for m in models]
                if self.ranking_model in model_names:
                    logger.info(f"✅ Connected to ranking model {self.ranking_model}")
                else:
                    logger.warning(f"⚠️ Ranking model {self.ranking_model} not found. Available: {model_names}")
                    logger.info(f"Will use main model {self.model} for ranking")
                    self.ranking_model = None
            else:
                logger.warning(f"⚠️ Could not verify ranking model. Will use main model {self.model}")
                self.ranking_model = None
        except Exception as e:
            logger.warning(f"⚠️ Could not test ranking model connection: {e}. Will use main model {self.model}")
            self.ranking_model = None
        
    def generate_cybersecurity_response(self, question: str, context: str) -> str:
        """
        Generate a cybersecurity-focused response using the retrieved context.
        
        Args:
            question: User's question
            context: Retrieved CAPEC context
            
        Returns:
            Generated response
        """
        prompt = self._build_cybersecurity_prompt(question, context)
        return self.invoke(prompt)
    
    def rank_entities_for_relevance(self, question: str, entities_text: str) -> str:
        """Rank cybersecurity entities for relevance to the query using smaller model for speed."""
        ranking_prompt = self._build_ranking_prompt(question, entities_text)
        
        # Use smaller model for ranking if available
        if self.ranking_model:
            return self._invoke_with_model(ranking_prompt, self.ranking_model, self.ranking_max_tokens)
        else:
            # Fallback to main model with reduced tokens
            return self._invoke_with_model(ranking_prompt, self.model, self.ranking_max_tokens)
    
    def _invoke_with_model(self, prompt: str, model: str, max_tokens: int) -> str:
        """Invoke Ollama with specific model and token limit."""
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Lower temperature for ranking consistency
                    "num_predict": max_tokens,
                    "top_p": 0.9
                }
            }
            
            response = requests.post(self.api_url, json=payload, timeout=30)  # Shorter timeout for ranking
            response.raise_for_status()
            
            result = response.json()
            generated_text = result.get('response', '').strip()
            
            if not generated_text:
                logger.warning(f"Empty response from Ollama model {model}")
                return "Unable to generate ranking scores"
                
            return generated_text
            
        except requests.exceptions.Timeout:
            logger.error(f"Ollama ranking request timed out for model {model}")
            return "Ranking timeout error"
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Ollama ranking API request failed for model {model}: {e}")
            return "Ranking API error"
        except Exception as e:
            logger.error(f"❌ Ranking generation failed for model {model}: {e}")
            return "Ranking generation error"
    
    def _build_ranking_prompt(self, question: str, entities_text: str) -> str:
        """Build prompt for entity relevance ranking."""
        prompt = f"""You are an expert cybersecurity analyst. Your task is to evaluate the relevance of cybersecurity entities to a user's query.

**User's Question:**
{question}

**Entities to Evaluate:**
{entities_text}

**Your Task:**
Evaluate each entity for relevance to the user's question. Rate each entity from 1-10 based on:
- Direct relevance to the query topic
- Contextual importance for cybersecurity understanding
- Relationship strength to the query concept
- Practical value in answering the question

**Rating Scale:**
- 9-10: Extremely relevant, directly answers the question
- 7-8: Highly relevant, provides important context
- 5-6: Moderately relevant, useful background information
- 3-4: Slightly relevant, tangential connection
- 1-2: Not relevant, no clear connection

**Format your response exactly as:**
EntityID: Score

For example:
CAPEC-123: 8
CWE-79: 9
CVE-2023-1234: 6

Only return the scores, no explanations.
"""
        return prompt
    
    def _build_cybersecurity_prompt(self, question: str, context: str) -> str:
        """Build a streamlined prompt for fast cybersecurity responses."""
        
        prompt = f"""You are a helpful expert assistant. Answer the user's question using only the information provided in the context below.

**Context:**
{context}

**Question:**
{question}

**Instructions:**
1. Provide a concise, well-structured answer using clear sections
2. Highlight entity names in **bold** when referencing them (e.g., **CAPEC-66**, **CWE-79**, **T1068**)
3. Use bullet points or numbered lists for clarity
4. Keep explanations simple and focused

**Response Format:**
## Overview
Brief explanation of the topic in 4 sentences or less

## Key Points
- Point 1 with most relevant **entity-name**
- Point 2 with second most relevant **entity-name**

## Technical Details
Additional technical information if needed

**Example:**
## Overview
Buffer overflow attacks exploit memory management weaknesses to overwrite memory.

## Key Points
- **CAPEC-100** describes the main attack pattern
- **CWE-120** identifies the underlying weakness

"""

        return prompt 