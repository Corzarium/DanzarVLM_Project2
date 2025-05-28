#!/usr/bin/env python3
# clone_eq_collection.py

import requests
import json
import logging
import time
from typing import List, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ─── CONFIG ─────────────────────────────────────────────────────────────
RAG_SERVER_URL = "http://localhost:5000"
SRC = "multimodal_rag_default"
DST = "danzarvlm_everquest_history"
BATCH_SIZE = 1000  # Number of documents to fetch per query
# ─────────────────────────────────────────────────────────────────────────

def query_rag_server(query_text: str, collection_name: str, n_results: int = 100) -> Dict[str, Any]:
    """Query the RAG server for documents."""
    try:
        response = requests.post(
            f"{RAG_SERVER_URL}/query",
            json={
                "query_text": query_text,
                "n_results": n_results,
                "collection_name": collection_name
            }
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error querying RAG server: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response content: {e.response.text}")
        raise

def add_to_rag_server(text: str, collection_name: str, metadata: Dict = None) -> Dict[str, Any]:
    """Add a document to the RAG server."""
    try:
        payload = {
            "text": text,
            "collection_name": collection_name
        }
        if metadata:
            payload["metadata"] = metadata

        response = requests.post(
            f"{RAG_SERVER_URL}/add",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error adding to RAG server: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response content: {e.response.text}")
        raise

def get_all_documents(collection_name: str) -> List[Dict[str, Any]]:
    """Get all documents from a collection using pagination."""
    all_documents = []
    offset = 0
    total_processed = 0
    
    while True:
        try:
            # Use a query that should match all documents
            results = query_rag_server(
                query_text="EverQuest",  # Broad query to match most documents
                collection_name=collection_name,
                n_results=BATCH_SIZE
            )
            
            if not results or "retrieved_sources" not in results:
                logger.error("No results found in source collection")
                break
                
            sources = results["retrieved_sources"]
            if not sources:
                logger.info("No more documents to fetch")
                break
                
            all_documents.extend(sources)
            total_processed += len(sources)
            logger.info(f"Fetched batch of {len(sources)} documents. Total processed: {total_processed}")
            
            # If we got fewer documents than requested, we've reached the end
            if len(sources) < BATCH_SIZE:
                break
                
            # Small delay between batches
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Error fetching batch: {e}")
            break
            
    return all_documents

def main():
    try:
        # 1) Get all documents from source collection
        logger.info(f"Starting to fetch all documents from '{SRC}'")
        all_documents = get_all_documents(SRC)
        
        if not all_documents:
            logger.error("No documents found to copy")
            return
            
        logger.info(f"Retrieved total of {len(all_documents)} documents from source collection")

        # 2) Add each document to the destination collection
        successful_copies = 0
        for i, source in enumerate(all_documents, 1):
            try:
                # Extract the text content and metadata
                text_content = source.get("text_content", "")
                metadata = source.get("metadata", {})
                
                if not text_content:
                    logger.warning(f"Skipping document {i} - empty text content")
                    continue
                
                # Add to destination collection
                add_to_rag_server(
                    text=text_content,
                    collection_name=DST,
                    metadata=metadata
                )
                
                successful_copies += 1
                if i % 10 == 0:  # Log progress every 10 documents
                    logger.info(f"Processed {i}/{len(all_documents)} documents ({successful_copies} successful copies)")
                
                # Small delay to prevent overwhelming the server
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error processing document {i}: {e}")
                continue

        logger.info(f"Successfully copied {successful_copies} out of {len(all_documents)} documents to '{DST}'")

    except Exception as e:
        logger.error(f"Error during collection cloning: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
