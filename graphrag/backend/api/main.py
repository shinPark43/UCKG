#!/usr/bin/env python3
"""
FastAPI backend using official Neo4j GraphRAG components with performance optimizations.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import sys

from core.rag import rag_system

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="UCKG GraphRAG API",
    description="UCKG GraphRAG API with Post-Traversal Ranking - implementing vector search, graph traversal, and intelligent ranking",
    version="3.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://localhost:3002",  # Added for current frontend port
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 20

class QueryResponse(BaseModel):
    answer: str
    query: str
    confidence: float
    sources: list
    context_summary: Dict[str, Any]
    processing_time: float
    ranking_stats: Optional[Dict[str, Any]] = None
    performance_improvement: str = "Optimized performance with post-traversal ranking"
    cache_hit: Optional[bool] = False

class HealthResponse(BaseModel):
    status: str
    system_type: str
    performance_notes: str
    statistics: Dict[str, Any]
    cache_stats: Optional[Dict[str, Any]] = None

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup."""
    try:
        logger.info("🚀 Starting UCKG GraphRAG API...")
        await rag_system.initialize()
        logger.info("✅ RAG system initialized with post-traversal ranking")
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG system: {e}")
        raise

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    try:
        await rag_system.close()
        logger.info("🔒 RAG system closed successfully")
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")

@app.get("/", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint with system statistics and cache metrics.
    """
    try:
        stats = await rag_system.get_statistics()
        cache_stats = stats.get("cache_stats", {})
        return HealthResponse(
            status="healthy",
            system_type="UCKG GraphRAG with Post-Traversal Ranking",
            performance_notes="Vector search, graph traversal, and intelligent ranking pipeline",
            statistics=stats,
            cache_stats=cache_stats
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/rag/statistics")
async def get_rag_statistics():
    """
    Get RAG system statistics for frontend.
    """
    try:
        stats = await rag_system.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"Failed to get RAG statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/rag/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    RAG query endpoint with post-traversal ranking.
    
    Features:
    - Multi-level caching for faster repeat queries
    - Vector search and graph traversal
    - Post-traversal ranking for relevance
    - Async processing for better performance
    """
    start_time = time.time()
    
    try:
        logger.info(f"🔍 Processing query: {request.question}")
        
        # Validate input
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        if request.top_k and (request.top_k < 1 or request.top_k > 50):
            raise HTTPException(status_code=400, detail="top_k must be between 1 and 50")
        
        # Check if this will be a cache hit
        cache_key = rag_system._get_cache_key(request.question.strip(), request.top_k or 20)
        is_cache_hit = cache_key in rag_system._query_cache
        
        # Process query using ranking system
        result = await rag_system.query_async(
            question=request.question.strip(),
            top_k=request.top_k or 20
        )
        
        # Handle errors from the RAG system
        if "error" in result:
            logger.error(f"RAG system error: {result['error']}")
            raise HTTPException(status_code=500, detail=result["error"])
        
        processing_time = time.time() - start_time
        
        # Return response with ranking stats
        response = QueryResponse(
            answer=result["answer"],
            query=result["query"],
            confidence=result.get("confidence", 0.7),
            sources=result.get("sources", []),
            context_summary=result.get("context_summary", {}),
            processing_time=round(processing_time, 2),
            ranking_stats=result.get("ranking_stats", {}),
            cache_hit=is_cache_hit
        )
        
        cache_indicator = "⚡ Cache hit" if is_cache_hit else "🔍 Fresh query"
        logger.info(f"✅ {cache_indicator} processed in {processing_time:.2f}s")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"❌ Query failed after {processing_time:.2f}s: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_legacy(request: QueryRequest):
    """
    Legacy query endpoint for backwards compatibility.
    """
    return await query_rag(request)

@app.get("/stats")
async def get_system_statistics():
    """
    Get detailed system statistics and performance metrics.
    """
    try:
        stats = await rag_system.get_statistics()
        return {
            "system_info": {
                "type": "UCKG GraphRAG with Post-Traversal Ranking",
                "version": "3.0.0",
                "performance_improvement": "Vector search, graph traversal, and intelligent ranking pipeline"
            },
            "database_stats": stats,
            "api_features": {
                "async_processing": True,
                "vector_search": True,
                "graph_traversal": True,
                "post_traversal_ranking": True,
                "embedding_similarity_ranking": True,
                "llm_contextual_ranking": True,
                "multi_level_caching": True
            },
            "enhancements": {
                "ranking_approach": "Post-traversal ranking for precision",
                "embedding_ranking": "Cosine similarity with configurable thresholds",
                "llm_ranking": "Contextual relevance assessment",
                "graph_traversal": "Simple 1-2 hop traversal for context collection",
                "caching_system": "TTL-based multi-level caching"
            }
        }
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/rag/clear-cache")
async def clear_cache():
    """
    Clear all caches for fresh queries.
    """
    try:
        rag_system.clear_caches()
        return {
            "status": "success",
            "message": "All caches cleared successfully",
            "cache_stats": {
                "query_cache_size": 0,
                "embedding_cache_size": 0,
                "context_cache_size": 0
            }
        }
    except Exception as e:
        logger.error(f"Failed to clear caches: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/rag/warm-cache")
async def warm_cache(queries: list = None):
    """
    Pre-warm cache with common queries for faster responses.
    """
    try:
        common_queries = queries or [
            "What are SQL injection attack patterns?",
            "How can I defend against buffer overflow attacks?",
            "What weaknesses are commonly exploited in web applications?",
            "What are cross-site scripting vulnerabilities?",
            "How do privilege escalation attacks work?"
        ]
        
        await rag_system.warm_cache(common_queries)
        stats = await rag_system.get_statistics()
        
        return {
            "status": "success",
            "message": f"Cache warmed with {len(common_queries)} queries",
            "cache_stats": stats.get("cache_stats", {})
        }
    except Exception as e:
        logger.error(f"Failed to warm cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test-performance")
async def test_performance():
    """
    Test endpoint to demonstrate performance improvements.
    """
    test_queries = [
        "What are SQL injection attack patterns?",
        "How can I defend against buffer overflow attacks?",
        "What weaknesses are commonly exploited in web applications?"
    ]
    
    results = []
    total_start = time.time()
    
    try:
        for query in test_queries:
            start = time.time()
            result = await rag_system.query_async(query)
            processing_time = time.time() - start
            
            results.append({
                "query": query,
                "processing_time": round(processing_time, 2),
                "sources_found": len(result.get("sources", [])),
                "confidence": result.get("confidence", 0)
            })
        
        total_time = time.time() - total_start
        
        return {
            "performance_test": {
                "total_queries": len(test_queries),
                "total_time": round(total_time, 2),
                "average_time_per_query": round(total_time / len(test_queries), 2),
                "performance_note": "Enhanced performance over basic implementation"
            },
            "individual_results": results
        }
        
    except Exception as e:
        logger.error(f"Performance test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return {"error": "Internal server error", "detail": str(exc)}

if __name__ == "__main__":
    # Run the API server
    logger.info("🚀 Starting UCKG GraphRAG API server...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 