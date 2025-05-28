# services/rag_service.py
import requests
import time 
import json # For logging complex payloads if needed
from typing import List, Optional, Dict, Any

# This should match the default collection name used in your rag_server_mm.py
# if no collection_name is provided in the payload to its /add or /query routes.
# However, rag_server_mm.py currently uses a hardcoded "multimodal_rag" for its
# global 'collection' variable if client doesn't specify.
# For consistency, let's assume the client (this RAGService) might want to know
# about a default it can expect or pass.
DEFAULT_RAG_SERVER_COLLECTION_NAME = "multimodal_rag_default" # Or "multimodal_rag" if that's the true fixed default on server

class RAGService:
    def __init__(self, app_context): # app_context: AppContext
        self.app_context = app_context
        self.logger = self.app_context.logger
        self.rag_server_url = self.app_context.global_settings.get("RAG_SERVER_URL")
        
        # This is the default collection your RAG server (rag_server_mm.py) might operate on
        # if no collection_name is specified in the payload to its endpoints.
        # Your current rag_server_mm.py uses a global `collection` object initialized to "multimodal_rag".
        # So, this client-side default might be more for logical grouping here,
        # or if you later make rag_server_mm.py handle dynamic collection names.
        self.server_side_default_collection = self.app_context.global_settings.get(
            "RAG_SERVER_DEFAULT_COLLECTION", "multimodal_rag" # Align with rag_server_mm.py's actual default
        )
        
        if not self.rag_server_url:
            self.logger.warning("[RAGService] RAG_SERVER_URL not configured. RAG functionality will be disabled.")
        else:
            self.logger.info(f"[RAGService] Initialized. RAG Server URL: {self.rag_server_url}, Server Default Collection: '{self.server_side_default_collection}'")

    def _call_rag_server(self, endpoint: str, payload: Dict, method: str = "POST") -> Optional[Dict[str, Any]]:
        if not self.rag_server_url:
            self.logger.warning(f"[RAGService] Cannot call RAG server, URL not configured. Endpoint: {endpoint}")
            return None

        # Add v2 prefix to endpoint
        url = f"{self.rag_server_url.rstrip('/')}/{endpoint.lstrip('/')}"
        rag_timeout = self.app_context.global_settings.get("RAG_REQUEST_TIMEOUT_S", 20) 

        # Create a loggable version of the payload, redacting long image data
        log_payload = {}
        for k, v in payload.items():
            if k in ['image', 'images'] and isinstance(v, str) and v.startswith('data:image'):
                log_payload[k] = f"[base64_image_data_uri_len:{len(v)}]"
            elif k in ['image', 'images'] and isinstance(v, list):
                log_payload[k] = f"[{len(v)}_base64_images]"
            else:
                log_payload[k] = v
        
        self.logger.debug(f"[RAGService] Calling RAG Server. URL: {url}, Method: {method}, Payload: {json.dumps(log_payload, indent=2, default=str)}")

        response = None  # Initialize response to None
        try:
            if method.upper() == "POST":
                self.logger.info(f"!!!! EXACT PAYLOAD TO RAG SERVER ({url}): {json.dumps(payload, indent=2)} !!!!") # Log the actual payload
                response = requests.post(url, json=payload, timeout=rag_timeout)
            else:
                self.logger.error(f"[RAGService] Unsupported HTTP method: {method} for RAG server call.")
                return None
            
            self.logger.debug(f"[RAGService] RAG Server response status: {response.status_code} from {url}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            self.logger.error(f"[RAGService] RAG server call timed out ({rag_timeout}s) to {url}")
            return None
        except requests.exceptions.HTTPError as e:
            self.logger.error(f"[RAGService] RAG server HTTP error: {e.response.status_code} for URL {url}. Response: {e.response.text[:500]}")
            return None
        except requests.exceptions.RequestException as e:
            self.logger.error(f"[RAGService] RAG server request failed for URL {url}: {e}")
            return None
        except json.JSONDecodeError as e_json:
            response_text_preview = response.text[:500] if response and hasattr(response, 'text') else "N/A"
            self.logger.error(f"[RAGService] Failed to parse JSON response from RAG server {url}: {e_json}. Response text: {response_text_preview}", exc_info=True)
            return None
        except Exception as e_gen: # Catch-all for other unexpected errors
            self.logger.error(f"[RAGService] Unexpected error calling RAG server {url}: {e_gen}", exc_info=True)
            return None


    def ingest_text(self, collection_name: str, text_content: str, metadata: Optional[Dict[str, Any]] = None, doc_id: Optional[str] = None):
        if not self.rag_server_url: return
        if not text_content: self.logger.debug("[RAGService] Empty text content for ingestion. Skipping."); return

        payload = {"text": text_content}
        # Your rag_server_mm.py /add endpoint currently doesn't use collection_name, metadata, or id for text directly
        # but it uses a global collection "multimodal_rag".
        # If you adapt rag_server_mm.py to accept these, you can pass them:
        # if collection_name != self.server_side_default_collection:
        #     payload["collection_name"] = collection_name
        # if metadata: payload["metadata"] = metadata # rag_server_mm.py /add uses this
        # if doc_id: payload["id"] = doc_id # rag_server_mm.py /add uses this

        # For the current rag_server_mm.py, metadata and doc_id are used internally when adding to Chroma.
        # We can pass them if the /add route is updated to accept them in the JSON body.
        if metadata: payload["metadata"] = metadata
        if doc_id: payload["id"] = doc_id # If you want to suggest an ID to the server

        self.logger.info(f"[RAGService] Ingesting text to RAG server: '{text_content[:60].replace(chr(10), ' ')}...' (Target Collection on Server: '{collection_name}')")
        
        response_data = self._call_rag_server(endpoint="/add", payload=payload, method="POST")
        
        if response_data and response_data.get("status") == "added":
            self.logger.debug(f"[RAGService] Text successfully ingested by RAG server. Server assigned ID: {response_data.get('id')}")
        elif response_data:
            self.logger.warning(f"[RAGService] RAG server ingestion for text returned: {response_data}")
        else:
            self.logger.error("[RAGService] Failed to ingest text to RAG server (no valid response or error).")


    def ingest_image_b64(self, collection_name: str, image_b64_data_uri: str, caption: str = "", metadata: Optional[Dict[str, Any]] = None, doc_id: Optional[str] = None):
        if not self.rag_server_url: return
        if not image_b64_data_uri: self.logger.debug("[RAGService] Empty image data URI for ingestion. Skipping."); return

        payload = {
            "image": image_b64_data_uri,
            "caption": caption
        }
        # if collection_name != self.server_side_default_collection:
        #     payload["collection_name"] = collection_name
        if metadata: payload["metadata"] = metadata # Pass metadata as a dictionary
        if doc_id: payload["id"] = doc_id

        self.logger.info(f"[RAGService] Ingesting image to RAG server (caption: '{caption[:50]}...'). (Target Collection on Server: '{collection_name}')")
        response_data = self._call_rag_server(endpoint="/add", payload=payload, method="POST")

        if response_data and response_data.get("status") == "added":
            self.logger.debug(f"[RAGService] Image successfully ingested by RAG server. Server assigned ID: {response_data.get('id')}, Type: {response_data.get('type')}")
        elif response_data:
            self.logger.warning(f"[RAGService] RAG server ingestion for image returned: {response_data}")
        else:
            self.logger.error("[RAGService] Failed to ingest image to RAG server (no valid response or error).")

    def query_rag(self, collection_name: str, query_text: str, top_k: Optional[int] = 3, 
                  query_images_b64: Optional[List[str]] = None, 
                  metadata_filter: Optional[Dict] = None) -> Optional[List[str]]:
        if not self.rag_server_url: return None 
        if not query_text: self.logger.debug("[RAGService] Empty query text provided to query_rag."); return []

        # Determine effective top_k
        profile_top_k = getattr(self.app_context.active_profile, 'rag_top_k', 3) # Default if not in profile
        final_top_k = top_k if top_k is not None else profile_top_k

        payload = {
            "query_text": query_text, # rag_server_mm.py /query expects 'query_text'
            "n_results": final_top_k  # rag_server_mm.py /query expects 'n_results'
        }
        
        # If your rag_server_mm.py /query is updated to take collection_name:
        if collection_name != self.server_side_default_collection:
             payload["collection_name"] = collection_name
        
        # If your rag_server_mm.py /query supports multimodal queries via "images" key (it doesn't currently)
        # if query_images_b64:
        #     payload["images"] = query_images_b64 
        
        # If your rag_server_mm.py /query supports metadata filters (it doesn't currently)
        # if metadata_filter:
        #     payload["filter"] = metadata_filter # Chroma uses "where" for filters

        self.logger.info(f"[RAGService] Querying RAG server: Collection '{collection_name}', TopK {final_top_k}, Query '{query_text[:50].replace(chr(10), ' ')}...'")
        
        response_data = self._call_rag_server(endpoint="/query", payload=payload, method="POST")

        if response_data and "retrieved_sources" in response_data:
            sources_list = response_data.get("retrieved_sources", []) # This list contains dicts
            retrieved_texts: List[str] = []
            if sources_list:
                for source_item in sources_list:
                    if isinstance(source_item, dict) and "text_content" in source_item:
                        # The 'text_content' from rag_server_mm.py should be the raw document
                        retrieved_texts.append(source_item["text_content"].strip())
                    elif isinstance(source_item, str): # Fallback if server somehow returns list of strings
                        retrieved_texts.append(source_item.strip())
            
            self.logger.info(f"[RAGService] RAG server query for '{query_text[:30]}...' returned {len(retrieved_texts)} source documents.")
            return retrieved_texts # Returns list of strings (document contents)
        elif response_data: # Received a response, but not in the expected format
            self.logger.warning(f"[RAGService] RAG server /query response from for query '{query_text[:30]}...' in unexpected format: {json.dumps(response_data, indent=2, default=str)}")
            return [] # Return empty list to signify no usable data
        else: # No response or _call_rag_server returned None (due to timeout, HTTP error, etc.)
            self.logger.error(f"[RAGService] Failed to get valid response from RAG server /query endpoint for query '{query_text[:30]}...'.")
            return None # Indicates an error occurred during the call