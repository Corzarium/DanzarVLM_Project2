"""
Llama.cpp-based RAG Service for DanzarVLM
Uses llama.cpp server's embeddings API with Qdrant vector database
"""

import logging
import requests
from typing import Dict, List, Optional, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid


class LlamaCppRAGService:
    """RAG service using llama.cpp server embeddings with Qdrant"""
    
    def __init__(self, global_settings: Dict[str, Any]):
        self.logger = logging.getLogger(f"{__name__}.LlamaCppRAGService")
        self.global_settings = global_settings
        
        # LLM configuration - use llama.cpp server
        llm_config = global_settings.get("LLAMACPP_PRIMARY", {})
        self.llamacpp_host = llm_config.get("server_host", "localhost")
        self.llamacpp_port = llm_config.get("server_port", 8080)
        self.llamacpp_base_url = f"http://{self.llamacpp_host}:{self.llamacpp_port}"
            
        self.logger.info(f"[LlamaCppRAG] Using llama.cpp server: {self.llamacpp_base_url}")
        self.embedding_model = "qwen2.5-omni-7b"  # Model loaded in llama.cpp server
        
        # Qdrant configuration
        qdrant_config = global_settings.get("QDRANT_SERVER", {})
        self.qdrant_host = qdrant_config.get("host", "localhost")
        self.qdrant_port = qdrant_config.get("port", 6333)
        
        # Initialize Qdrant client
        try:
            self.qdrant_client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
            self.logger.info(f"[LlamaCppRAG] Connected to Qdrant at {self.qdrant_host}:{self.qdrant_port}")
        except Exception as e:
            self.logger.error(f"[LlamaCppRAG] Failed to connect to Qdrant: {e}")
            self.qdrant_client = None
            
        # Cache for embeddings
        self.embedding_cache = {}
        
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding from llama.cpp server"""
        if not text or not text.strip():
            return None
            
        # Check cache first
        if text in self.embedding_cache:
            return self.embedding_cache[text]
            
        try:
            # Call llama.cpp server embeddings API (if available)
            # Note: llama.cpp server may not have embeddings endpoint
            # This is a placeholder for when/if it becomes available
            url = f"{self.llamacpp_base_url}/v1/embeddings"
            payload = {
                "model": self.embedding_model,
                "input": text
            }
            
            self.logger.debug(f"[LlamaCppRAG] Making embedding request to: {url}")
            
            response = requests.post(
                url,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if "data" in result and len(result["data"]) > 0:
                    embedding = result["data"][0].get("embedding")
                    if embedding:
                        # Cache the embedding
                        self.embedding_cache[text] = embedding
                        self.logger.info(f"[LlamaCppRAG] Successfully generated embedding of length {len(embedding)}")
                        return embedding
                else:
                    self.logger.error(f"[LlamaCppRAG] No embedding in response: {result}")
            else:
                self.logger.error(f"[LlamaCppRAG] Embedding request failed: {response.status_code}")
                self.logger.error(f"[LlamaCppRAG] Response text: {response.text}")
                
        except Exception as e:
            self.logger.warning(f"[LlamaCppRAG] llama.cpp embeddings not available, using fallback: {e}")
            # Fallback to a simple text-based similarity (not ideal but functional)
            return self._generate_simple_embedding(text)
            
        return None
    
    def _generate_simple_embedding(self, text: str) -> List[float]:
        """Simple text-based embedding fallback (not ideal but functional)"""
        # This is a very basic implementation - in production you'd want
        # to use a proper embedding model like sentence-transformers
        import hashlib
        import struct
        
        # Create a deterministic hash-based embedding
        text_hash = hashlib.sha256(text.encode()).digest()
        embedding = []
        
        # Convert hash bytes to float values
        for i in range(0, len(text_hash), 4):
            if i + 4 <= len(text_hash):
                float_val = struct.unpack('f', text_hash[i:i+4])[0]
                embedding.append(float_val)
        
        # Pad or truncate to 768 dimensions (common embedding size)
        target_dim = 768
        if len(embedding) < target_dim:
            embedding.extend([0.0] * (target_dim - len(embedding)))
        else:
            embedding = embedding[:target_dim]
            
        # Normalize the embedding
        import math
        magnitude = math.sqrt(sum(x*x for x in embedding))
        if magnitude > 0:
            embedding = [x/magnitude for x in embedding]
            
        self.logger.warning(f"[LlamaCppRAG] Using fallback hash-based embedding (not ideal)")
        return embedding
    
    def query(self, collection: str, query_text: str, n_results: int = 5) -> List[str]:
        """Query the RAG service for relevant documents"""
        self.logger.info(f"[LlamaCppRAG:DEBUG] === Starting query for collection '{collection}' ===")
        self.logger.info(f"[LlamaCppRAG:DEBUG] Query text: '{query_text}'")
        self.logger.info(f"[LlamaCppRAG:DEBUG] Requested results: {n_results}")
        
        if not self.qdrant_client:
            self.logger.error("[LlamaCppRAG:DEBUG] Qdrant client not initialized")
            return []
        
        try:
            # Check if collection exists
            collections_response = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections_response.collections]
            self.logger.info(f"[LlamaCppRAG:DEBUG] Available collections: {collection_names}")
            
            if collection not in collection_names:
                self.logger.error(f"[LlamaCppRAG:DEBUG] Collection '{collection}' not found in {collection_names}")
                return []
            
            self.logger.info(f"[LlamaCppRAG:DEBUG] Collection '{collection}' exists, proceeding with query")
            
            # Generate embedding for query
            self.logger.info(f"[LlamaCppRAG:DEBUG] Generating embedding for query...")
            query_embedding = self.get_embedding(query_text)
            if not query_embedding:
                self.logger.error("[LlamaCppRAG:DEBUG] Failed to generate query embedding")
                return []
            
            self.logger.info(f"[LlamaCppRAG:DEBUG] Query embedding dimension: {len(query_embedding)}")
            
            # Search for similar documents
            self.logger.info(f"[LlamaCppRAG:DEBUG] Searching collection for similar documents...")
            search_result = self.qdrant_client.search(
                collection_name=collection,
                query_vector=query_embedding,
                limit=n_results
            )
            
            self.logger.info(f"[LlamaCppRAG:DEBUG] Search returned {len(search_result)} results")
            
            if not search_result:
                self.logger.warning(f"[LlamaCppRAG:DEBUG] No search results found for query '{query_text}' in collection '{collection}'")
                return []
            
            # Extract and format results
            results = []
            for i, hit in enumerate(search_result):
                score = hit.score
                payload = hit.payload or {}
                text = payload.get('text', 'No text available')
                
                self.logger.info(f"[LlamaCppRAG:DEBUG] Result {i+1}: Score={score:.4f}, Text length={len(text)}")
                self.logger.info(f"[LlamaCppRAG:DEBUG] Result {i+1} preview: {text[:150]}...")
                
                formatted_result = f"[Score: {score:.3f}] {text}"
                results.append(formatted_result)
            
            self.logger.info(f"[LlamaCppRAG:DEBUG] Returning {len(results)} formatted results")
            return results
            
        except Exception as e:
            self.logger.error(f"[LlamaCppRAG:DEBUG] Query failed: {e}")
            import traceback
            self.logger.error(f"[LlamaCppRAG:DEBUG] Full traceback: {traceback.format_exc()}")
            return []
    
    def get_collections(self) -> List[str]:
        """Get list of available collections"""
        if not self.qdrant_client:
            return []
            
        try:
            collections = self.qdrant_client.get_collections()
            return [c.name for c in collections.collections]
        except Exception as e:
            self.logger.error(f"[LlamaCppRAG] Failed to get collections: {e}")
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
            self.logger.error(f"[LlamaCppRAG] Failed to get collection info: {e}")
            return None
    
    def ingest_text(self, collection: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Ingest text into a collection"""
        if not self.qdrant_client:
            self.logger.warning("[LlamaCppRAG] Qdrant client not available for text ingestion")
            return False
        
        try:
            # Create collection if it doesn't exist
            if not self.collection_exists(collection):
                self.logger.info(f"[LlamaCppRAG] Collection '{collection}' does not exist, creating it...")
                if not self._create_collection(collection):
                    self.logger.error(f"[LlamaCppRAG] Failed to create collection '{collection}'")
                    return False
            
            # Generate embedding
            embedding = self.get_embedding(text)
            if not embedding:
                self.logger.error("[LlamaCppRAG] Failed to generate embedding for ingestion")
                return False
            
            # Prepare point for insertion
            point_id = str(uuid.uuid4())
            payload = {"text": text}
            if metadata:
                payload.update(metadata)
            
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload
            )
            
            # Insert into Qdrant
            self.qdrant_client.upsert(
                collection_name=collection,
                points=[point]
            )
            
            self.logger.info(f"[LlamaCppRAG] Successfully ingested text into collection '{collection}'")
            return True
            
        except Exception as e:
            self.logger.error(f"[LlamaCppRAG] Failed to ingest text into collection '{collection}': {e}")
            return False
    
    def _create_collection(self, collection_name: str) -> bool:
        """Create a new collection in Qdrant"""
        try:
            if self.collection_exists(collection_name):
                self.logger.info(f"[LlamaCppRAG] Collection '{collection_name}' already exists, skipping creation")
                return True
            
            # Generate a sample embedding to determine vector size
            sample_embedding = self.get_embedding("sample text for vector size")
            if not sample_embedding:
                self.logger.error("[LlamaCppRAG] Cannot determine embedding dimension")
                return False
            
            vector_size = len(sample_embedding)
            self.logger.info(f"[LlamaCppRAG] Creating collection '{collection_name}' with vector size {vector_size}")
            
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            
            self.logger.info(f"[LlamaCppRAG] Successfully created collection '{collection_name}'")
            return True
            
        except Exception as e:
            self.logger.error(f"[LlamaCppRAG] Failed to create collection '{collection_name}': {e}")
            return False 