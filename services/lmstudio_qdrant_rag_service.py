#!/usr/bin/env python3
"""
LM Studio + Qdrant RAG Service
Works with:
- LM Studio for LLM inference
- Qdrant for vector storage
- sentence-transformers for embeddings (local)
- Chatterbox for TTS
- Whisper for STT
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
from .qdrant_service import QdrantService

class LMStudioQdrantRAGService:
    def __init__(self, global_settings: Dict[str, Any]):
        self.logger = logging.getLogger("DanzarVLM.LMStudioQdrantRAGService")
        self.global_settings = global_settings
        
        # Initialize Qdrant connection
        qdrant_config = global_settings.get("QDRANT_SERVER", {})
        self.qdrant_client = QdrantService(
            host=qdrant_config.get("host", "localhost"),
            port=qdrant_config.get("port", 6333),
            api_key=qdrant_config.get("api_key", ""),
            prefer_grpc=qdrant_config.get("prefer_grpc", False),
            https=qdrant_config.get("https", False)
        )
        
        self.default_collection = qdrant_config.get("default_collection", "Everquest")
        
        # Initialize embedding model (local sentence-transformers)
        try:
            from sentence_transformers import SentenceTransformer
            # Use all-mpnet-base-v2 which produces 768-dimensional vectors
            # This matches the existing Qdrant collection dimensions
            self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
            self.logger.info("✅ Local embedding model (sentence-transformers) loaded: all-mpnet-base-v2 (768 dims)")
        except ImportError:
            self.logger.error("❌ sentence-transformers not available. Install with: pip install sentence-transformers")
            self.embedding_model = None
        except Exception as e:
            self.logger.error(f"❌ Failed to load embedding model: {e}")
            self.embedding_model = None
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding using local sentence-transformers model"""
        if not self.embedding_model:
            self.logger.error("❌ Embedding model not available")
            return None
        
        try:
            embedding = self.embedding_model.encode(text)
            return embedding.tolist()
        except Exception as e:
            self.logger.error(f"❌ Failed to generate embedding: {e}")
            return None
    
    def search_all_collections(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search across all available collections to find the best matches."""
        try:
            self.logger.info(f"🔍 Searching across all collections for: '{query}'")
            
            # Get all collections using the underlying Qdrant client
            collections = self.qdrant_client.client.get_collections()
            all_results = []
            
            for collection in collections.collections:
                collection_name = collection.name
                try:
                    self.logger.info(f"🔍 Searching collection: {collection_name}")
                    
                    # Generate embedding for this query
                    if not self.embedding_model:
                        self.logger.error("❌ Embedding model not available")
                        continue
                        
                    query_embedding = self.embedding_model.encode(query)
                    # Convert to list using the QdrantService helper method
                    query_embedding = self.qdrant_client._ensure_vector_is_list(query_embedding)
                    
                    # Try to search this collection using QdrantService.query method
                    results = self.qdrant_client.query(
                        collection_name=collection_name,
                        query_vector=query_embedding,
                        limit=limit
                    )
                    
                    # Add collection name to results
                    for result in results:
                        result["collection"] = collection_name
                        all_results.append(result)
                        
                    self.logger.info(f"✅ Found {len(results)} results in {collection_name}")
                    
                except Exception as e:
                    if "Vector dimension error" in str(e):
                        self.logger.info(f"⚠️ Dimension mismatch in {collection_name}, skipping")
                    else:
                        self.logger.warning(f"⚠️ Error searching {collection_name}: {e}")
                    continue
            
            # Sort all results by score (highest first)
            all_results.sort(key=lambda x: x["score"], reverse=True)
            
            # Return top results
            top_results = all_results[:limit]
            self.logger.info(f"🎯 Found {len(top_results)} total results across all collections")
            
            return top_results
            
        except Exception as e:
            self.logger.error(f"❌ Error in multi-collection search: {e}")
            return []

    def search(self, query: str, collection_name: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents in Qdrant with fallback to all collections."""
        try:
            # Use default collection if none specified
            target_collection = collection_name or self.default_collection
            
            self.logger.info(f"🔍 Searching collection '{target_collection}' for: '{query}'")
            
            # Generate embedding
            if not self.embedding_model:
                self.logger.error("❌ Embedding model not available")
                return []
                
            query_embedding = self.embedding_model.encode(query)
            # Convert to list using the QdrantService helper method
            query_embedding = self.qdrant_client._ensure_vector_is_list(query_embedding)
            
            # Try to search the specific collection first
            try:
                results = self.qdrant_client.query(
                    collection_name=target_collection,
                    query_vector=query_embedding,
                    limit=limit
                )
                
                # Add collection name to results
                for result in results:
                    result["collection"] = target_collection
                
                self.logger.info(f"✅ Found {len(results)} results in {target_collection}")
                return results
                
            except Exception as e:
                if "Vector dimension error" in str(e):
                    self.logger.warning(f"⚠️ Dimension mismatch in {target_collection}, searching all collections...")
                    return self.search_all_collections(query, limit)
                else:
                    raise e
                    
        except Exception as e:
            self.logger.error(f"❌ Search failed: {e}")
            # Final fallback: try searching all collections
            self.logger.info("🔄 Falling back to multi-collection search...")
            return self.search_all_collections(query, limit)
    
    def add_documents(self, texts: List[str], collection_name: Optional[str] = None, metadatas: Optional[List[Dict]] = None) -> bool:
        """Add documents to Qdrant collection"""
        if not collection_name:
            collection_name = self.default_collection
        
        if not self.embedding_model:
            self.logger.error("❌ Cannot add documents without embedding model")
            return False
        
        try:
            # Generate embeddings for all texts
            embeddings = []
            for text in texts:
                embedding = self.generate_embedding(text)
                if embedding:
                    embeddings.append(embedding)
                else:
                    self.logger.warning(f"⚠️ Failed to generate embedding for text: {text[:100]}...")
                    return False
            
            # Add to Qdrant
            success = self.qdrant_client.add_texts(
                collection_name=collection_name,
                texts=texts,
                vectors=embeddings,
                metadatas=metadatas
            )
            
            if success:
                self.logger.info(f"✅ Added {len(texts)} documents to collection '{collection_name}'")
            else:
                self.logger.error(f"❌ Failed to add documents to collection '{collection_name}'")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Failed to add documents: {e}")
            return False
    
    def ingest_text(self, text: str, metadata: Optional[Dict] = None, collection_name: Optional[str] = None, collection: Optional[str] = None) -> bool:
        """Ingest a single text document (compatibility method)"""
        try:
            # Support both collection_name and collection parameters for compatibility
            target_collection = collection_name or collection
            return self.add_documents(
                texts=[text],
                collection_name=target_collection,
                metadatas=[metadata] if metadata else None
            )
        except Exception as e:
            self.logger.error(f"❌ Failed to ingest text: {e}")
            return False
    
    def query_knowledge(self, query: str, limit: int = 5, collection_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query knowledge base (compatibility method)"""
        return self.search(query, collection_name, limit)
    
    def get_relevant_context(self, query: str, limit: int = 3) -> List[str]:
        """Get relevant context as text strings (compatibility method)"""
        results = self.search(query, limit=limit)
        return [result.get("text", "") for result in results if result.get("text")]
    
    def query(self, query: str = None, collection_name: Optional[str] = None, limit: int = 5, collection: Optional[str] = None, query_text: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """Query method for AgenticRAG compatibility - handles multiple parameter formats"""
        # Handle different parameter names that AgenticRAG might use
        actual_query = query or query_text
        if not actual_query:
            self.logger.error("❌ No query provided")
            return []
        
        # Support both collection_name and collection parameters for compatibility
        target_collection = collection_name or collection
        
        # Log the actual parameters received for debugging
        self.logger.info(f"🔍 Query called with: query='{actual_query}', collection='{target_collection}', limit={limit}")
        
        return self.search(actual_query, target_collection, limit) 