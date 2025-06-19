"""
Memory service for managing both short-term and long-term memory.
"""
import logging
import asyncio
from typing import List, Dict, Any, Optional, Deque, cast
from collections import deque
from datetime import datetime
import json
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class MemoryService:
    """Memory service for managing conversation memory."""
    
    def __init__(self, settings: dict):
        """Initialize memory service with settings."""
        self.settings = settings
        
        # Initialize short-term memory
        self.stm_size = settings["SHORT_TERM_SIZE"]
        self.short_term_memory: Deque[Dict[str, Any]] = deque(maxlen=self.stm_size)
        
        # Initialize RAG client
        self.rag_host = settings["RAG_HOST"]
        self.rag_port = settings["RAG_PORT"]
        self.rag_collection = settings["RAG_COLLECTION"]
        
        try:
            self.qdrant_client = QdrantClient(
                host=self.rag_host,
                port=self.rag_port
            )
            logger.info(f"Connected to Qdrant at {self.rag_host}:{self.rag_port}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
        
        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer(settings["EMBEDDING_MODEL"])
            logger.info(f"Loaded embedding model: {settings['EMBEDDING_MODEL']}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
        
        # Ensure collection exists
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Ensure the RAG collection exists with proper configuration."""
        try:
            collections = self.qdrant_client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.rag_collection not in collection_names:
                # Get embedding dimension
                embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
                if embedding_dim is None:
                    raise ValueError("Failed to get embedding dimension")
                
                self.qdrant_client.create_collection(
                    collection_name=self.rag_collection,
                    vectors_config=models.VectorParams(
                        size=embedding_dim,
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created new collection: {self.rag_collection}")
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
            raise
    
    def add_to_stm(self, user_message: str, bot_response: str):
        """
        Add a message pair to short-term memory.
        
        Args:
            user_message: User's message
            bot_response: Bot's response
        """
        self.short_term_memory.append({
            "timestamp": datetime.utcnow().isoformat(),
            "user_message": user_message,
            "bot_response": bot_response
        })
    
    def get_stm_context(self) -> str:
        """
        Get formatted context from short-term memory.
        
        Returns:
            str: Formatted conversation history
        """
        context = []
        for entry in self.short_term_memory:
            context.append(f"User: {entry['user_message']}")
            context.append(f"Assistant: {entry['bot_response']}")
        return "\n".join(context)
    
    async def store_conversation(self, summary: str, topic: str):
        """
        Store conversation summary in long-term memory.
        
        Args:
            summary: Conversation summary
            topic: Conversation topic
        """
        try:
            # Generate embedding
            embedding = self.embedding_model.encode(summary)
            if isinstance(embedding, np.ndarray):
                embedding_list = embedding.tolist()
            else:
                embedding_list = list(embedding)
            
            # Store in Qdrant
            self.qdrant_client.upsert(
                collection_name=self.rag_collection,
                points=[
                    models.PointStruct(
                        id=hash(f"{summary}{datetime.utcnow().isoformat()}"),
                        vector=embedding_list,
                        payload={
                            "summary": summary,
                            "topic": topic,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    )
                ]
            )
            logger.info(f"Stored conversation summary in {self.rag_collection}")
            
        except Exception as e:
            logger.error(f"Error storing conversation: {e}")
            raise
    
    async def search_memory(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Search long-term memory for relevant conversations.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List[Dict[str, Any]]: List of relevant conversation summaries
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query)
            if isinstance(query_embedding, np.ndarray):
                query_embedding_list = query_embedding.tolist()
            else:
                query_embedding_list = list(query_embedding)
            
            # Search in Qdrant
            search_results = self.qdrant_client.search(
                collection_name=self.rag_collection,
                query_vector=query_embedding_list,
                limit=limit
            )
            
            # Format results
            results = []
            for hit in search_results:
                if hit.payload is not None:
                    results.append({
                        "summary": hit.payload.get("summary", ""),
                        "topic": hit.payload.get("topic", ""),
                        "timestamp": hit.payload.get("timestamp", ""),
                        "score": hit.score
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching memory: {e}")
            return []
    
    def clear_stm(self):
        """Clear short-term memory."""
        self.short_term_memory.clear()
        logger.info("Cleared short-term memory") 