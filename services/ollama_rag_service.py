#!/usr/bin/env python3
"""
Ollama-based RAG Service for DanzarVLM
Uses Ollama's nomic-embed-text model directly instead of sentence_transformers
"""

import requests
import json
import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
import time
import hashlib

class OllamaRAGService:
    """RAG service using Ollama embeddings directly"""
    
    def __init__(self, global_settings: Dict[str, Any]):
        self.logger = logging.getLogger(f"{__name__}.OllamaRAGService")
        self.global_settings = global_settings
        
        # LLM configuration - default to LM Studio port 1234
        llm_endpoint = global_settings.get("LLM_SERVER", {}).get("endpoint", "http://localhost:1234")
        
        # Extract base URL from various endpoint formats
        if "/v1/chat/completions" in llm_endpoint:
            self.ollama_base_url = llm_endpoint.replace("/v1/chat/completions", "")
        elif "/api/chat" in llm_endpoint:
            self.ollama_base_url = llm_endpoint.replace("/api/chat", "")
        elif "/v1" in llm_endpoint:
            self.ollama_base_url = llm_endpoint.replace("/v1", "")
        else:
            self.ollama_base_url = llm_endpoint
            
        # Ensure we have the correct base URL - default to LM Studio
        if not any(port in self.ollama_base_url for port in [":1234", ":11434"]):
            self.ollama_base_url = "http://localhost:1234"
            
        self.logger.info(f"[OllamaRAG] Using LLM base URL: {self.ollama_base_url}")
        self.embedding_model = "nomic-embed-text"
        
        # Qdrant configuration
        qdrant_config = global_settings.get("QDRANT_SERVER", {})
        self.qdrant_host = qdrant_config.get("host", "localhost")
        self.qdrant_port = qdrant_config.get("port", 6333)
        
        # Initialize Qdrant client
        try:
            self.qdrant_client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
            self.logger.info(f"[OllamaRAG] Connected to Qdrant at {self.qdrant_host}:{self.qdrant_port}")
        except Exception as e:
            self.logger.error(f"[OllamaRAG] Failed to connect to Qdrant: {e}")
            self.qdrant_client = None
            
        # Cache for embeddings
        self.embedding_cache = {}
        
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding from Ollama using nomic-embed-text model"""
        if not text or not text.strip():
            return None
            
        # Check cache first
        if text in self.embedding_cache:
            return self.embedding_cache[text]
            
        try:
            # Call Ollama embeddings API
            url = f"{self.ollama_base_url}/api/embeddings"
            payload = {
                "model": self.embedding_model,
                "prompt": text
            }
            
            self.logger.debug(f"[OllamaRAG] Making embedding request to: {url}")
            self.logger.debug(f"[OllamaRAG] Payload: {payload}")
            
            response = requests.post(
                url,
                json=payload,
                timeout=30
            )
            
            self.logger.debug(f"[OllamaRAG] Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                embedding = result.get("embedding")
                if embedding:
                    # Cache the embedding
                    self.embedding_cache[text] = embedding
                    self.logger.info(f"[OllamaRAG] Successfully generated embedding of length {len(embedding)}")
                    return embedding
                else:
                    self.logger.error(f"[OllamaRAG] No embedding in response: {result}")
            else:
                self.logger.error(f"[OllamaRAG] Embedding request failed: {response.status_code}")
                self.logger.error(f"[OllamaRAG] Response text: {response.text}")
                
        except Exception as e:
            self.logger.error(f"[OllamaRAG] Embedding generation failed: {e}")
            
        return None
    
    def query(self, collection: str, query_text: str, n_results: int = 5) -> List[str]:
        """Query the RAG service for relevant documents"""
        self.logger.info(f"[OllamaRAG:DEBUG] === Starting query for collection '{collection}' ===")
        self.logger.info(f"[OllamaRAG:DEBUG] Query text: '{query_text}'")
        self.logger.info(f"[OllamaRAG:DEBUG] Requested results: {n_results}")
        
        if not self.qdrant_client:
            self.logger.error("[OllamaRAG:DEBUG] Qdrant client not initialized")
            return []
        
        try:
            # Check if collection exists
            collections_response = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections_response.collections]
            self.logger.info(f"[OllamaRAG:DEBUG] Available collections: {collection_names}")
            
            if collection not in collection_names:
                self.logger.error(f"[OllamaRAG:DEBUG] Collection '{collection}' not found in {collection_names}")
                return []
            
            # Skip collection info due to Qdrant client version compatibility issues
            self.logger.info(f"[OllamaRAG:DEBUG] Collection '{collection}' exists, proceeding with query")
            
            # Generate embedding for query
            self.logger.info(f"[OllamaRAG:DEBUG] Generating embedding for query...")
            query_embedding = self.get_embedding(query_text)
            if not query_embedding:
                self.logger.error("[OllamaRAG:DEBUG] Failed to generate query embedding")
                return []
            
            self.logger.info(f"[OllamaRAG:DEBUG] Query embedding dimension: {len(query_embedding)}")
            
            # Search for similar documents
            self.logger.info(f"[OllamaRAG:DEBUG] Searching collection for similar documents...")
            search_result = self.qdrant_client.search(
                collection_name=collection,
                query_vector=query_embedding,
                limit=n_results
            )
            
            self.logger.info(f"[OllamaRAG:DEBUG] Search returned {len(search_result)} results")
            
            if not search_result:
                self.logger.warning(f"[OllamaRAG:DEBUG] No search results found for query '{query_text}' in collection '{collection}'")
                return []
            
            # Extract and format results
            results = []
            for i, hit in enumerate(search_result):
                score = hit.score
                payload = hit.payload or {}  # Handle None payload
                text = payload.get('text', 'No text available')
                
                self.logger.info(f"[OllamaRAG:DEBUG] Result {i+1}: Score={score:.4f}, Text length={len(text)}")
                self.logger.info(f"[OllamaRAG:DEBUG] Result {i+1} preview: {text[:150]}...")
                
                # Include score in results for debugging
                formatted_result = f"[Score: {score:.3f}] {text}"
                results.append(formatted_result)
            
            self.logger.info(f"[OllamaRAG:DEBUG] Returning {len(results)} formatted results")
            return results
            
        except Exception as e:
            self.logger.error(f"[OllamaRAG:DEBUG] Query failed: {e}")
            import traceback
            self.logger.error(f"[OllamaRAG:DEBUG] Full traceback: {traceback.format_exc()}")
            return []
    
    def get_collections(self) -> List[str]:
        """Get list of available collections"""
        if not self.qdrant_client:
            return []
            
        try:
            collections = self.qdrant_client.get_collections()
            return [c.name for c in collections.collections]
        except Exception as e:
            self.logger.error(f"[OllamaRAG] Failed to get collections: {e}")
            return []
    
    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists"""
        if not self.qdrant_client:
            return False
        try:
            self.qdrant_client.get_collection(collection_name)
            return True
        except Exception:
            return False
    
    def get_collection_info(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a collection"""
        if not self.qdrant_client:
            return None
        if not self.collection_exists(collection_name):
            return None
            
        try:
            info = self.qdrant_client.get_collection(collection_name)
            count = self.qdrant_client.count(collection_name=collection_name)
            
            return {
                'name': collection_name,
                'vectors_count': count.count,
                'config': info.config.dict() if hasattr(info, 'config') else {}
            }
        except Exception as e:
            self.logger.error(f"[OllamaRAG] Failed to get collection info: {e}")
            return None
    
    def ingest_text(self, collection: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Ingest text into a collection (compatibility method for MemoryService)"""
        if not self.qdrant_client:
            self.logger.warning("[OllamaRAG] Qdrant client not available for text ingestion")
            return False
            
        try:
            # Auto-create collection if it doesn't exist
            if not self.collection_exists(collection):
                self.logger.info(f"[OllamaRAG] Collection '{collection}' does not exist, creating it...")
                success = self._create_collection(collection)
                if not success:
                    self.logger.error(f"[OllamaRAG] Failed to create collection '{collection}'")
                    return False
            
            # Generate embedding for the text
            embedding = self.get_embedding(text)
            if not embedding:
                self.logger.error("[OllamaRAG] Failed to generate embedding for ingestion")
                return False
            
            # Create a point ID (use timestamp + hash for uniqueness)
            point_id = int(hashlib.md5(f"{time.time()}_{text[:100]}".encode()).hexdigest()[:8], 16)
            
            # Prepare payload
            payload = {"text": text}
            if metadata:
                payload.update(metadata)
            
            # Insert into Qdrant
            self.qdrant_client.upsert(
                collection_name=collection,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=payload
                    )
                ]
            )
            
            self.logger.info(f"[OllamaRAG] Successfully ingested text into collection '{collection}'")
            return True
            
        except Exception as e:
            self.logger.error(f"[OllamaRAG] Failed to ingest text into collection '{collection}': {e}")
            return False
    
    def _create_collection(self, collection_name: str) -> bool:
        """Create a new collection in Qdrant"""
        try:
            # Double-check if collection exists to avoid 409 conflicts
            if self.collection_exists(collection_name):
                self.logger.info(f"[OllamaRAG] Collection '{collection_name}' already exists, skipping creation")
                return True
            
            from qdrant_client.models import Distance, VectorParams, CreateCollection
            
            # Get embedding dimension from a test embedding
            test_embedding = self.get_embedding("test")
            if not test_embedding:
                self.logger.error("[OllamaRAG] Cannot determine embedding dimension")
                return False
            
            vector_size = len(test_embedding)
            self.logger.info(f"[OllamaRAG] Creating collection '{collection_name}' with vector size {vector_size}")
            
            # Create collection with appropriate vector configuration
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE  # Use cosine similarity for text embeddings
                )
            )
            
            self.logger.info(f"[OllamaRAG] Successfully created collection '{collection_name}'")
            return True
            
        except Exception as e:
            # Handle the specific case where collection already exists
            if "already exists" in str(e).lower() or "409" in str(e):
                self.logger.info(f"[OllamaRAG] Collection '{collection_name}' already exists")
                return True
            else:
                self.logger.error(f"[OllamaRAG] Failed to create collection '{collection_name}': {e}")
                return False 