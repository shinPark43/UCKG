#!/usr/bin/env python3
"""
UCKG GraphRAG with Post-Traversal Ranking.
Implements vector search, graph traversal, and intelligent ranking pipeline.
"""

import asyncio
import logging
import hashlib
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from functools import lru_cache
from cachetools import TTLCache
from sklearn.metrics.pairwise import cosine_similarity

from neo4j import GraphDatabase
from .config import settings, get_neo4j_config, get_ollama_config, get_ranking_config
from .ollama_llm import OllamaLLMOptimized
from .embeddings import OllamaNomicEmbeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UCKGRAG:
    """
    UCKG RAG system with post-traversal ranking.
    
    Features:
    - Vector search for initial node discovery
    - Graph traversal for context collection
    - Post-traversal ranking for relevance filtering
    - Dual ranking: embedding similarity + LLM assessment
    - Multi-level caching and async processing
    """
    
    def __init__(self):
        self.driver = None
        self.embedder = None
        self.llm = None
        self.vector_index_name = None
        
        # Multi-level caching system
        self._query_cache = TTLCache(maxsize=100, ttl=300)  # 5 min query cache
        self._embedding_cache = TTLCache(maxsize=500, ttl=1800)  # 30 min embedding cache
        self._context_cache = TTLCache(maxsize=200, ttl=600)  # 10 min context cache
        
        # Load ranking configuration from settings
        self.ranking_config = get_ranking_config()
        
    async def initialize(self) -> None:
        """Initialize the RAG system with local Ollama embeddings."""
        # Database connection using existing config
        neo4j_config = get_neo4j_config()
        self.driver = GraphDatabase.driver(neo4j_config["uri"], auth=neo4j_config["auth"])
        
        # Set vector index name from config
        self.vector_index_name = settings.vector_index_name
        
        # Initialize local Ollama embeddings
        ollama_config = get_ollama_config()
        self.embedder = OllamaNomicEmbeddings(
            base_url=ollama_config["base_url"],
            model=ollama_config["embedding_model"]
        )
        
        # Simplified vector search query - traversal happens separately
        vector_search_query = """
        CALL db.index.vector.queryNodes($index_name, $top_k, $query_embedding)
        YIELD node, score
        RETURN 
            elementId(node) as node_id,
            CASE 
                WHEN 'UcoexCAPEC' IN labels(node) THEN 'UcoexCAPEC'
                WHEN 'UcoCWE' IN labels(node) THEN 'UcoCWE'
                WHEN 'UcoCVE' IN labels(node) THEN 'UcoCVE'
                WHEN 'UcoexMITREATTACK' IN labels(node) THEN 'UcoexMITREATTACK'
                WHEN 'UcoexCPE' IN labels(node) THEN 'UcoexCPE'
                WHEN 'UcoVulnerability' IN labels(node) THEN 'UcoVulnerability'
                ELSE labels(node)[0]
            END as node_type,
            node.embedding as embedding,
            properties(node) as properties,
            score
        ORDER BY score DESC
        """
        
        # Store query for later use
        self.vector_search_query = vector_search_query
        
        # Initialize local Ollama LLM for response generation with dual-model support
        self.llm = OllamaLLMOptimized(
            base_url=ollama_config["base_url"],
            model=ollama_config["llm_model"],
            temperature=ollama_config["temperature"],
            max_tokens=ollama_config["max_tokens"],
            ranking_model=ollama_config["ranking_model"],
            ranking_max_tokens=ollama_config["ranking_max_tokens"]
        )
        
        logger.info("✅ UCKG RAG system initialized with post-traversal ranking")
    
    def _get_cache_key(self, question: str, top_k: int) -> str:
        """Generate cache key for queries."""
        return hashlib.md5(f"{question}:{top_k}".encode()).hexdigest()
    
    async def _vector_search(self, question: str, top_k: int) -> List[Dict[str, Any]]:
        """Step 1: Vector search for initial relevant nodes."""
        try:
            # Get query embedding
            query_embedding = self.embedder.embed_query(question)
            
            # Execute vector search
            with self.driver.session() as session:
                result = session.run(self.vector_search_query, {
                    'index_name': self.vector_index_name,
                    'top_k': top_k,
                    'query_embedding': query_embedding
                })
                
                nodes = []
                for record in result:
                    nodes.append({
                        'node_id': record['node_id'],
                        'node_type': record['node_type'],
                        'embedding': record['embedding'],
                        'properties': record['properties'],
                        'vector_score': record['score']
                    })
                
                return nodes
                
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    async def _graph_traversal(self, initial_nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Step 2: Simple graph traversal for context collection."""
        all_results = []
        
        try:
            with self.driver.session() as session:
                for node in initial_nodes:
                    # Distance-based multi-hop traversal with decay weighting
                    traversal_query = """
                    MATCH (start) WHERE elementId(start) = $node_id
                    OPTIONAL MATCH path = (start)-[r*1..2]-(related)
                    WHERE related <> start 
                    AND related.embedding IS NOT NULL
                    AND size(coalesce(
                        related.searchContent,
                        related.ucoexDescription, 
                        related.ucodescription,
                        related.ucosummary,
                        related.ucoexDESCRIPTION,
                        ""
                    )) > 20
                    
                    WITH start, related, r, length(path) as distance,
                         1.0 / length(path) as distance_weight
                    
                    RETURN 
                        elementId(start) as source_id,
                        elementId(related) as related_id,
                        CASE 
                            WHEN 'UcoexCAPEC' IN labels(related) THEN 'UcoexCAPEC'
                            WHEN 'UcoCWE' IN labels(related) THEN 'UcoCWE'
                            WHEN 'UcoCVE' IN labels(related) THEN 'UcoCVE'
                            WHEN 'UcoexMITREATTACK' IN labels(related) THEN 'UcoexMITREATTACK'
                            WHEN 'UcoexCPE' IN labels(related) THEN 'UcoexCPE'
                            WHEN 'UcoVulnerability' IN labels(related) THEN 'UcoVulnerability'
                            ELSE labels(related)[0]
                        END as related_type,
                        related.embedding as related_embedding,
                        properties(related) as related_properties,
                        distance,
                        distance_weight,
                        [rel in r | type(rel)] as relationship_path
                    ORDER BY distance_weight DESC
                    LIMIT 10
                    """
                    
                    result = session.run(traversal_query, {'node_id': node['node_id']})
                    
                    # Add source node
                    source_result = {
                        'node_id': node['node_id'],
                        'node_type': node['node_type'],
                        'embedding': node['embedding'],
                        'properties': node['properties'],
                        'vector_score': node['vector_score'],
                        'distance': 0,
                        'relationship_path': [],
                        'source_id': node['node_id']
                    }
                    all_results.append(source_result)
                    
                    # Add related nodes
                    for record in result:
                        if record['related_id']:  # Only if related node exists
                            related_result = {
                                'node_id': record['related_id'],
                                'node_type': record['related_type'],
                                'embedding': record['related_embedding'],
                                'properties': record['related_properties'],
                                'vector_score': 0.0,  # No direct vector score
                                'distance': record['distance'],
                                'distance_weight': record['distance_weight'],
                                'relationship_path': record['relationship_path'],
                                'source_id': record['source_id']
                            }
                            all_results.append(related_result)
            
            # Remove duplicates based on node_id
            seen_ids = set()
            unique_results = []
            for result in all_results:
                if result['node_id'] not in seen_ids:
                    seen_ids.add(result['node_id'])
                    unique_results.append(result)
            
            logger.info(f"Graph traversal: {len(initial_nodes)} initial → {len(unique_results)} total nodes")
            return unique_results
            
        except Exception as e:
            logger.error(f"Graph traversal failed: {e}")
            return initial_nodes  # Fallback to initial nodes
    
    async def _post_traversal_ranking(self, question: str, traversal_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Step 3: Embedding-based post-traversal ranking for optimal performance."""
        if not traversal_results:
            return []
        
        # Initialize embedding similarity scores
        for result in traversal_results:
            result.setdefault('embedding_similarity', 0.0)
        
        # Direct embedding-based ranking to final count (simplified single-stage)
        max_results = self.ranking_config['max_results']
        final_results = await self._embedding_based_ranking(question, traversal_results, max_results)

        return final_results
    
    async def _embedding_based_ranking(self, question: str, results: List[Dict[str, Any]], max_results: int = 10) -> List[Dict[str, Any]]:
        """Optimized embedding similarity ranking with advanced scoring."""
        try:
            # Cache query embedding
            cache_key = f"query_emb_{hashlib.md5(question.encode()).hexdigest()}"
            if cache_key in self._embedding_cache:
                query_embedding = self._embedding_cache[cache_key]
            else:
                query_embedding = self.embedder.embed_query(question)
                self._embedding_cache[cache_key] = query_embedding
            
            query_embedding = np.array(query_embedding).reshape(1, -1)
            
            # Enhanced similarity calculation with multiple factors
            for result in results:
                if result.get('embedding'):
                    node_embedding = np.array(result['embedding']).reshape(1, -1)
                    similarity = cosine_similarity(query_embedding, node_embedding)[0][0]
                    result['embedding_similarity'] = float(similarity)
                    
                    # Calculate composite score with multiple factors
                    vector_score = result.get('vector_score', 0.0)
                    distance_weight = result.get('distance_weight', 1.0)  # Use the new distance weight
                    
                    # Composite score: weighted combination with distance-based weighting
                    # Boost direct connections (distance=1) significantly
                    distance_boost = 0.2 if result.get('distance') == 1 else 0.0
                    
                    result['composite_score'] = (
                        vector_score * 0.4 +           # Original vector search relevance
                        similarity * 0.5 +             # Embedding similarity (primary)
                        distance_weight * 0.1 +        # Distance-based proximity weighting  
                        distance_boost                 # Extra boost for direct connections
                    )
                else:
                    result['embedding_similarity'] = 0.0
                    result['composite_score'] = result.get('vector_score', 0.0) * 0.5
            
            # Smart threshold filtering with adaptive threshold
            threshold = self.ranking_config['embedding_threshold']
            
            # Keep top results even if below threshold (but not too many)
            sorted_by_composite = sorted(results, key=lambda x: x['composite_score'], reverse=True)
            
            # Filter strategy: keep high-confidence results + direct connections + source nodes
            filtered_results = []
            for result in sorted_by_composite:
                if (result['embedding_similarity'] >= threshold or 
                    result['distance'] == 0 or  # Always keep source nodes
                    result['distance'] == 1 or  # Always keep direct connections (1-hop)
                    len(filtered_results) < 3):  # Ensure minimum 3 results
                    filtered_results.append(result)
                    if len(filtered_results) >= max_results:  # Cap at max_results
                        break
            
            # Final sort by composite score
            filtered_results.sort(key=lambda x: x['composite_score'], reverse=True)
            
            logger.info(f"Embedding ranking: {len(results)} → {len(filtered_results)} nodes (threshold: {threshold})")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Embedding ranking failed: {e}")
            return results
    
    
    async def query_async(self, question: str, top_k: int = 20) -> Dict[str, Any]:
        """
        Async query processing with post-traversal ranking.
        """
        try:
            # Check cache first
            cache_key = self._get_cache_key(question, top_k)
            if cache_key in self._query_cache:
                logger.info(f"Cache hit for query: {question[:50]}...")
                return self._query_cache[cache_key]
            
            logger.info(f"Processing query: {question}")
            
            # Step 1: Vector search for top 5 most relevant nodes (focused approach)
            initial_nodes = await self._vector_search(question, 5)  # Always use 5 for consistency
            if not initial_nodes:
                return self._no_results_response(question)
            
            # Step 2: Graph traversal for context collection
            traversal_results = await self._graph_traversal(initial_nodes)
            
            # Step 3: Post-traversal ranking (the key enhancement)
            ranked_results = await self._post_traversal_ranking(question, traversal_results)
            
            # Generate final response
            if ranked_results:
                context = self._format_context(ranked_results)
                answer = self.llm.generate_cybersecurity_response(question, context)
            else:
                answer = "I couldn't find sufficiently relevant cybersecurity information for your query. Please try rephrasing your question."
            
            # Format response
            formatted_response = {
                "answer": answer,
                "query": question,
                "confidence": self._calculate_confidence(ranked_results),
                "sources": self._extract_sources(ranked_results),
                "context_summary": self._generate_context_summary(ranked_results),
                "ranking_stats": {
                    "initial_nodes": len(initial_nodes),
                    "post_traversal": len(traversal_results),
                    "post_ranking": len(ranked_results)
                }
            }
            
            # Cache the result
            self._query_cache[cache_key] = formatted_response
            
            logger.info(f"Query processed: {len(initial_nodes)} initial → {len(traversal_results)} traversed → {len(ranked_results)} ranked")
            return formatted_response
            
        except Exception as e:
            logger.error(f"Enhanced query failed: {e}")
            return {
                "error": str(e),
                "query": question,
                "answer": "I encountered an error while processing your question. Please try again."
            }
    
    def query(self, question: str, top_k: int = 20) -> Dict[str, Any]:
        """
        Sync query wrapper for ranking system.
        """
        # Check cache first
        cache_key = self._get_cache_key(question, top_k)
        if cache_key in self._query_cache:
            logger.info(f"Sync cache hit for query: {question[:50]}...")
            return self._query_cache[cache_key]
        
        return asyncio.run(self.query_async(question, top_k))
    
    
    def _get_entity_id(self, result: Dict[str, Any]) -> str:
        """Extract clean entity ID with node type prefix."""
        props = result.get('properties', {})
        node_type = result.get('node_type', '')
        
        if node_type == 'UcoexCAPEC':
            capec_id = props.get('ucoexCAPEC_id', 'unknown')
            return f"UcoexCAPEC-{capec_id}"
        elif node_type == 'UcoCWE':
            cwe_id = props.get('ucocweID', 'unknown').replace('CWE-', '')
            return f"UcoCWE-{cwe_id}"
        elif node_type == 'UcoCVE':
            cve_label = props.get('label', 'unknown')
            return f"UcoCVE-{cve_label}"
        elif node_type == 'UcoexMITREATTACK':
            uri = props.get('uri', '')
            attack_id = uri.split('#')[-1] if '#' in uri else 'unknown'
            return f"UcoexMITREATTACK-{attack_id}"
        elif node_type == 'UcoexCPE':
            uri = props.get('uri', '')
            # Extract meaningful part from CPE URI
            if 'cpe:' in uri:
                cpe_part = uri.split('cpe:')[-1][:20] + '...' if len(uri.split('cpe:')[-1]) > 20 else uri.split('cpe:')[-1]
                return f"UcoexCPE-{cpe_part}"
            return f"UcoexCPE-{props.get('label', 'unknown')}"
        elif node_type == 'UcoVulnerability':
            return f"UcoVulnerability-{props.get('label', 'unknown')}"
        else:
            # Handle any other node types
            identifier = props.get('label', props.get('uri', 'unknown'))
            if len(identifier) > 30:
                identifier = identifier[:30] + '...'
            return f"{node_type}-{identifier}"
    
    def _smart_truncate_searchcontent(self, content: str, max_chars: int = 400) -> str:
        """
        Smart truncation of searchContent to ~400 characters while preserving key information.
        Prioritizes critical cybersecurity information and maintains readability.
        """
        if not content or len(content) <= max_chars:
            return content
        
        # Split content into sentences
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        if not sentences:
            return content[:max_chars] + "..."
        
        # Priority keywords for cybersecurity content
        priority_keywords = [
            'attack', 'vulnerability', 'exploit', 'weakness', 'threat', 'malicious',
            'security', 'breach', 'injection', 'overflow', 'bypass', 'escalation',
            'unauthorized', 'compromise', 'malware', 'phishing', 'dos', 'denial',
            'authentication', 'authorization', 'encryption', 'credential'
        ]
        
        # Score sentences by relevance
        scored_sentences = []
        for sentence in sentences:
            score = 0
            sentence_lower = sentence.lower()
            
            # Boost score for priority keywords
            for keyword in priority_keywords:
                if keyword in sentence_lower:
                    score += 2
            
            # Boost score for specific patterns
            if any(pattern in sentence_lower for pattern in ['cve-', 'cwe-', 'capec-']):
                score += 3
            if 'impact' in sentence_lower or 'consequence' in sentence_lower:
                score += 2
            if len(sentence) > 20:  # Prefer substantial sentences
                score += 1
            
            scored_sentences.append((score, sentence))
        
        # Sort by score (highest first)
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        
        # Build truncated content
        result = []
        current_length = 0
        
        for score, sentence in scored_sentences:
            sentence_with_period = sentence if sentence.endswith('.') else sentence + '.'
            if current_length + len(sentence_with_period) + 1 <= max_chars:
                result.append(sentence_with_period)
                current_length += len(sentence_with_period) + 1
            elif current_length < max_chars * 0.8:  # If we have room, try to fit partial sentence
                remaining_chars = max_chars - current_length - 4  # Reserve space for "..."
                if remaining_chars > 20:  # Only if meaningful portion can fit
                    partial = sentence[:remaining_chars].rsplit(' ', 1)[0]  # Cut at word boundary
                    result.append(partial + "...")
                break
            else:
                break
        
        return ' '.join(result) if result else content[:max_chars] + "..."

    def _get_entity_description(self, result: Dict[str, Any]) -> str:
        """Extract entity description using pre-generated searchContent with smart truncation."""
        props = result.get('properties', {})
        
        # Use pre-generated searchContent if available (much faster and more comprehensive)
        search_content = props.get('searchContent', '')
        if search_content and len(search_content.strip()) > 50:  # Ensure it's substantial
            return self._smart_truncate_searchcontent(search_content)
        
        # Fallback to manual construction only if searchContent is missing
        return self._get_entity_description_fallback(result)

    def _get_entity_description_fallback(self, result: Dict[str, Any]) -> str:
        """Fallback method for nodes without searchContent."""
        props = result.get('properties', {})
        node_type = result.get('node_type', '')
        
        # Get name and description based on node type
        name = ""
        description = ""
        
        if node_type == 'UcoexCAPEC':
            name = props.get('ucoexCAPEC_name', '')
            description = props.get('ucoexDescription', '')
        elif node_type == 'UcoCWE':
            name = props.get('ucocweName', '')
            description = props.get('ucodescription', '') or props.get('ucocweSummary', '')
        elif node_type == 'UcoCVE':
            name = props.get('label', '')
            description = props.get('ucosummary', '')
        elif node_type == 'UcoexMITREATTACK':
            name = props.get('ucoexNAME', '')
            description = props.get('ucoexDESCRIPTION', '')
        else:
            name = props.get('label', props.get('uri', ''))
            description = props.get('ucosummary', props.get('description', ''))
        
        return f"{name} - {description}"
    
    
    def _no_results_response(self, question: str) -> Dict[str, Any]:
        """Generate response when no results found."""
        return {
            "answer": "I couldn't find any relevant cybersecurity information for your query. Please try rephrasing your question or asking about specific attack patterns, weaknesses, or vulnerabilities.",
            "query": question,
            "confidence": 0.0,
            "sources": [],
            "context_summary": {},
            "ranking_stats": {"initial_nodes": 0, "post_traversal": 0, "post_ranking": 0}
        }
    
    def _format_context(self, results: List[Dict[str, Any]]) -> str:
        """Format ranked results into optimized context using smart truncation."""
        context_parts = []
        
        # Limit to top 10 most relevant results for better coverage
        max_results = min(10, len(results))
        top_results = results[:max_results]
        
        for result in top_results:
            entity_id = self._get_entity_id(result)
            props = result.get('properties', {})
            
            # Use smart truncated searchContent for optimal performance
            if 'searchContent' in props and len(props['searchContent'].strip()) > 50:
                # Apply smart truncation to searchContent (~400 chars)
                truncated_content = self._smart_truncate_searchcontent(props['searchContent'])
                context_parts.append(f"**{entity_id}**\n{truncated_content}")
            else:
                # Fallback to manual description (also truncated)
                entity_desc = self._get_entity_description_fallback(result)
                truncated_desc = self._smart_truncate_searchcontent(entity_desc)
                context_parts.append(f"**{entity_id}**\n{truncated_desc}")
            
            # Add concise relationship info if available
            if result.get('relationship_path') and len(result['relationship_path']) <= 3:
                path = " → ".join(result['relationship_path'][:3])  # Limit path length
                context_parts.append(f"*Via: {path}*")
            
            context_parts.append("")  # Separator for readability
        
        return "\n".join(context_parts)
    
    def _calculate_confidence(self, results: List[Dict[str, Any]]) -> float:
        """Calculate confidence from embedding-based ranked results."""
        if not results:
            return 0.0
        
        # Average of embedding-based scores
        total_score = 0.0
        count = 0
        
        for result in results[:5]:  # Top 5 results
            # Use composite score if available, otherwise fallback to individual scores
            if 'composite_score' in result:
                score = result['composite_score']
            else:
                score = (
                    result.get('vector_score', 0.0) * 0.5 +
                    result.get('embedding_similarity', 0.0) * 0.5
                )
            total_score += score
            count += 1
        
        return min(total_score / count if count > 0 else 0.0, 1.0)
    
    def _extract_sources(self, results: List[Dict[str, Any]]) -> List[str]:
        """Extract source citations from embedding-based ranked results."""
        sources = []
        for result in results:
            entity_id = self._get_entity_id(result)
            # Use composite score if available, otherwise calculate from embedding scores
            if 'composite_score' in result:
                confidence = result['composite_score']
            else:
                confidence = (
                    result.get('vector_score', 0.0) * 0.5 +
                    result.get('embedding_similarity', 0.0) * 0.5
                )
            sources.append(f"{entity_id} (confidence: {confidence:.3f})")
        
        return sources
    
    def _generate_context_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary of embedding-based context."""
        if not results:
            return {}
        
        # Count by node type
        type_counts = {}
        total_confidence = 0.0
        
        for result in results:
            node_type = result.get('node_type', 'Unknown')
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
            
            # Use composite score if available, otherwise calculate from embedding scores
            if 'composite_score' in result:
                confidence = result['composite_score']
            else:
                confidence = (
                    result.get('vector_score', 0.0) * 0.5 +
                    result.get('embedding_similarity', 0.0) * 0.5
                )
            total_confidence += confidence
        
        return {
            "total_entities": len(results),
            "entity_types": type_counts,
            "avg_confidence": total_confidence / len(results) if results else 0.0,
            "ranking_method": "embedding_only"
        }
    
    def _fast_calculate_confidence_direct(self, retriever_result) -> float:
        """Calculate confidence from direct retriever result."""
        try:
            if hasattr(retriever_result, 'items') and retriever_result.items:
                scores = [item.metadata.get('score', 0) for item in retriever_result.items if item.metadata]
                if scores:
                    avg_score = sum(scores) / len(scores)
                    return min(avg_score * 1.2, 1.0)
            return 0.7
        except Exception:
            return 0.7
    
    def _fast_extract_sources_direct(self, retriever_result) -> List[str]:
        """Extract sources from direct retriever result."""
        sources = []
        try:
            if hasattr(retriever_result, 'items'):
                for item in retriever_result.items:
                    if item.metadata:
                        node_type = item.metadata.get('node_type', 'UNKNOWN')
                        score = item.metadata.get('score', 0)
                        
                        if node_type == "CAPEC" and item.metadata.get('capec_id'):
                            sources.append(f"CAPEC-{item.metadata.get('capec_id')} ({score:.3f})")
                        elif node_type == "CWE" and item.metadata.get('cwe_id'):
                            sources.append(f"CWE-{item.metadata.get('cwe_id')} ({score:.3f})")
                        elif node_type == "MITRE_ATTACK" and item.metadata.get('attack_id'):
                            sources.append(f"MITRE ATT&CK {item.metadata.get('attack_id')} ({score:.3f})")
                        elif node_type == "CVE" and item.metadata.get('cve_id'):
                            sources.append(f"CVE {item.metadata.get('cve_id')} ({score:.3f})")
                        elif node_type == "CPE" and item.metadata.get('cpe_id'):
                            sources.append(f"CPE {item.metadata.get('cpe_id')} ({score:.3f})")
                        else:
                            sources.append(f"{node_type} ({score:.3f})")
            return sources
        except Exception:
            return []
    
    def _fast_context_summary_direct(self, retriever_result) -> Dict[str, Any]:
        """Generate context summary from direct retriever result."""
        try:
            if hasattr(retriever_result, 'items') and retriever_result.items:
                items = retriever_result.items
                
                # Count different node types
                node_type_counts = {}
                total_weaknesses = 0
                total_attacks = 0
                
                for item in items:
                    if item.metadata:
                        node_type = item.metadata.get('node_type', 'UNKNOWN')
                        node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1
                        total_weaknesses += item.metadata.get('weakness_count', 0)
                        total_attacks += item.metadata.get('attack_count', 0)
                
                summary = {
                    "total_retrieved": len(items),
                    "node_type_distribution": node_type_counts,
                    "total_related_weaknesses": total_weaknesses,
                    "total_related_attacks": total_attacks,
                    "avg_relevance": round(sum(item.metadata.get('score', 0) for item in items if item.metadata) / len(items), 3) if items else 0
                }
                
                return summary
            return {}
        except Exception:
            return {}
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        try:
            with self.driver.session() as session:
                # Universal statistics query for all node types
                stats_query = """
                MATCH (n)
                WHERE n:UcoexCAPEC OR n:UcoCWE OR n:UcoexMITREATTACK OR n:UcoCVE OR n:UcoexCPE OR n:UcoVulnerability
                RETURN 
                    {
                        capec: {
                            total: size([x IN collect(n) WHERE x:UcoexCAPEC]),
                            with_embeddings: size([x IN collect(n) WHERE x:UcoexCAPEC AND x.embedding IS NOT NULL])
                        },
                        cwe: {
                            total: size([x IN collect(n) WHERE x:UcoCWE]),
                            with_embeddings: size([x IN collect(n) WHERE x:UcoCWE AND x.embedding IS NOT NULL])
                        },
                        mitre_attack: {
                            total: size([x IN collect(n) WHERE x:UcoexMITREATTACK]),
                            with_embeddings: size([x IN collect(n) WHERE x:UcoexMITREATTACK AND x.embedding IS NOT NULL])
                        },
                        cve: {
                            total: size([x IN collect(n) WHERE x:UcoCVE]),
                            with_embeddings: size([x IN collect(n) WHERE x:UcoCVE AND x.embedding IS NOT NULL])
                        },
                        cpe: {
                            total: size([x IN collect(n) WHERE x:UcoexCPE]),
                            with_embeddings: size([x IN collect(n) WHERE x:UcoexCPE AND x.embedding IS NOT NULL])
                        },
                        vulnerability: {
                            total: size([x IN collect(n) WHERE x:UcoVulnerability]),
                            with_embeddings: size([x IN collect(n) WHERE x:UcoVulnerability AND x.embedding IS NOT NULL])
                        }
                    } as node_stats
                """
                result = session.run(stats_query)
                record = result.single()
                node_stats = record["node_stats"]
                
                # Calculate totals
                total_nodes = sum(stats["total"] for stats in node_stats.values())
                total_with_embeddings = sum(stats["with_embeddings"] for stats in node_stats.values())
                overall_coverage = (total_with_embeddings / total_nodes * 100) if total_nodes > 0 else 0
                
                return {
                    "node_statistics": node_stats,
                    "total_nodes": total_nodes,
                    "total_with_embeddings": total_with_embeddings,
                    "overall_coverage": round(overall_coverage, 2),
                    "vector_index_name": self.vector_index_name,
                    "system_type": "Universal Neo4j GraphRAG",
                    "optimizations": "Universal query optimization and caching enabled",
                    "cache_stats": {
                        "query_cache_size": len(self._query_cache),
                        "embedding_cache_size": len(self._embedding_cache),
                        "context_cache_size": len(self._context_cache)
                    }
                }
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
    
    def clear_caches(self) -> None:
        """Clear all caches for fresh queries."""
        self._query_cache.clear()
        self._embedding_cache.clear()  
        self._context_cache.clear()
        logger.info("All caches cleared")
    
    async def warm_cache(self, common_queries: List[str]) -> None:
        """Pre-warm cache with common queries for faster responses."""
        logger.info(f"Warming cache with {len(common_queries)} common queries...")
        for query in common_queries:
            try:
                await self.query_async(query)
            except Exception as e:
                logger.warning(f"Failed to warm cache for query '{query}': {e}")
        logger.info("Cache warming completed")
    
    async def close(self) -> None:
        """Close all connections."""
        if self.driver:
            self.driver.close()
        logger.info("RAG connections closed")

# Global instance for easy import
rag_system = UCKGRAG()

async def main():
    """Test the UCKG RAG system."""
    print("Testing UCKG RAG System with Post-Traversal Ranking")
    print("=" * 60)
    
    try:
        # Initialize the system
        await rag_system.initialize()
        
        # Get statistics
        stats = await rag_system.get_statistics()
        print(f"System Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        print()
        
        # Test enhanced ranking with sample queries
        test_queries = [
            "What are SQL injection attack patterns?",
            "How can I defend against buffer overflow attacks?",
            "What weaknesses are commonly exploited in web applications?"
        ]
        
        # Test each query with ranking details
        for query in test_queries:
            print(f"Query: {query}")
            print("-" * 50)
            
            result = await rag_system.query_async(query)
            
            print(f"Answer: {result['answer'][:300]}...")
            print(f"Confidence: {result.get('confidence', 0):.3f}")
            
            # Show ranking statistics
            ranking_stats = result.get('ranking_stats', {})
            print(f"Ranking Pipeline: {ranking_stats.get('initial_nodes', 0)} initial → {ranking_stats.get('post_traversal', 0)} traversed → {ranking_stats.get('post_ranking', 0)} ranked")
            
            # Show top sources
            sources = result.get('sources', [])
            print(f"Top Sources: {', '.join(sources[:3])}")
            
            # Show context summary
            context_summary = result.get('context_summary', {})
            if context_summary.get('entity_types'):
                print(f"Entity Types: {context_summary['entity_types']}")
            
            print("\n" + "=" * 70 + "\n")
            
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await rag_system.close()

if __name__ == "__main__":
    asyncio.run(main()) 