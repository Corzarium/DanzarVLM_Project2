import logging
from typing import List, Dict, Any, Union, Optional
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    PointStruct, 
    Distance, 
    VectorParams,
    Filter,
    CreateCollection
)

class QdrantService:
    def __init__(self, host: str = "localhost", port: int = 6333, api_key: Optional[str] = None, prefer_grpc: bool = False, https: bool = False):
        self.logger = logging.getLogger("DanzarVLM.QdrantService")
        self.client = QdrantClient(host=host, port=port, api_key=api_key, prefer_grpc=prefer_grpc, https=https)
        self.logger.info(f"[QdrantService] Connected to Qdrant at {host}:{port} (GRPC: {prefer_grpc}, HTTPS: {https})")

    def create_collection_if_not_exists(self, collection_name: str, vector_size: int = 384):
        """Create a collection if it doesn't exist"""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            existing_names = [c.name for c in collections.collections]
            
            if collection_name not in existing_names:
                self.logger.info(f"[QdrantService] Creating collection: {collection_name}")
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
                self.logger.info(f"[QdrantService] Collection {collection_name} created successfully")
                return True
            else:
                self.logger.debug(f"[QdrantService] Collection {collection_name} already exists")
                return True
                
        except Exception as e:
            self.logger.error(f"[QdrantService] Failed to create collection {collection_name}: {e}")
            return False

    def query(self, collection_name: str, query_vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        try:
            # Convert query vector to list if it's a numpy array
            processed_query_vector = self._ensure_vector_is_list(query_vector)
            
            # Auto-create collection if it doesn't exist
            if len(processed_query_vector) > 0:
                self.create_collection_if_not_exists(collection_name, len(processed_query_vector))
            
            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=processed_query_vector,
                limit=limit
            )
            
            return [
                {
                    "text": point.payload.get("text", "") if point.payload else "",
                    "metadata": point.payload.get("metadata", {}) if point.payload else {},
                    "score": point.score
                }
                for point in search_result
            ]
        except Exception as e:
            self.logger.error(f"[QdrantService] Search failed: {e}")
            return []

    def add_texts(self, collection_name: str, texts: List[str], vectors: List[Union[List[float], Any]], metadatas: Optional[List[Dict]] = None):
        try:
            # Auto-create collection if it doesn't exist
            if vectors and len(vectors) > 0:
                # Safely get vector size from first vector
                first_vector = self._ensure_vector_is_list(vectors[0])
                if len(first_vector) > 0:
                    vector_size = len(first_vector)
                else:
                    vector_size = 384
                self.create_collection_if_not_exists(collection_name, vector_size)
            
            points = []
            for i, (text, vector) in enumerate(zip(texts, vectors)):
                # Robust numpy array to list conversion
                processed_vector = self._ensure_vector_is_list(vector)
                
                # Skip empty vectors
                if len(processed_vector) == 0:
                    self.logger.warning(f"[QdrantService] Skipping point {i} due to empty vector")
                    continue
                
                point = PointStruct(
                    id=i,
                    vector=processed_vector,
                    payload={
                        "text": text,
                        "metadata": metadatas[i] if metadatas and i < len(metadatas) else {}
                    }
                )
                points.append(point)
                
            if points:
                self.client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                self.logger.debug(f"[QdrantService] Successfully upserted {len(points)} points to {collection_name}")
                return True
            else:
                self.logger.warning(f"[QdrantService] No valid points to upsert")
                return False
        except Exception as e:
            self.logger.error(f"[QdrantService] Upsert failed: {e}")
            return False

    def _ensure_vector_is_list(self, vector: Any) -> List[float]:
        """Safely convert any vector format to a Python list of floats"""
        try:
            # Handle None or empty cases
            if vector is None:
                self.logger.warning("[QdrantService] Received None vector")
                return []
                
            # If it's already a list, ensure all elements are floats
            if isinstance(vector, list):
                return [float(x) for x in vector]
            
            # Handle NumPy arrays (most common case for sentence transformers)
            if isinstance(vector, np.ndarray):
                # Convert numpy array to list
                result = vector.tolist()
                # Ensure all elements are floats
                if isinstance(result, list):
                    return [float(x) for x in result]
                else:
                    # Handle single element arrays
                    return [float(result)]
            
            # Check for objects with tolist method (other array-like objects)
            if hasattr(vector, 'tolist') and callable(getattr(vector, 'tolist', None)):
                result = vector.tolist()
                if isinstance(result, list):
                    return [float(x) for x in result]
                else:
                    return [float(result)]
            
            # Check for tensor-like objects (PyTorch, TensorFlow)
            if hasattr(vector, 'numpy') and callable(getattr(vector, 'numpy', None)):
                numpy_array = vector.numpy()
                result = numpy_array.tolist()
                if isinstance(result, list):
                    return [float(x) for x in result]
                else:
                    return [float(result)]
            
            # Check if it's iterable and can be converted to list
            if hasattr(vector, '__iter__') and not isinstance(vector, (str, bytes)):
                return [float(x) for x in vector]
            
            # Single value case
            if isinstance(vector, (int, float)):
                return [float(vector)]
            
            # Fallback - try direct conversion
            return [float(x) for x in list(vector)]
            
        except Exception as e:
            self.logger.error(f"[QdrantService] Failed to convert vector to list: {e}, type: {type(vector)}")
            # Return empty list for failed conversions
            return []
