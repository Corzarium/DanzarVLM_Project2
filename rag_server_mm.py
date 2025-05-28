#!/usr/bin/env python3
# rag_server_mm.py

import os
import io
import base64
import time
import threading
import logging # For more structured logging

from typing import Optional
from flask import Flask, request, jsonify
from PIL import Image
import torch
import requests # Still needed if you plan to have an endpoint that calls LLM

from sentence_transformers import SentenceTransformer
# For BLIP, ensure you have transformers and Pillow installed
# For Salesforce/blip-image-captioning-base, you need BlipProcessor and BlipForConditionalGeneration for captioning
# For Salesforce/blip-image-captioning-large, same
# For Salesforce/blip-itm-base-coco, you need BlipProcessor and BlipForImageTextRetrieval
# For embeddings (like in your original), BlipModel is fine.
from transformers import BlipProcessor, BlipModel # For image embeddings

import chromadb
# from chromadb.config import Settings # Settings might not be needed for basic client

# ─── CONFIGURATION ─────────────────────────────────────────────────────────
PERSIST_DIR       = os.environ.get("CHROMA_PERSIST_DIR", "chroma_rag_db")
TEXT_MODEL_NAME   = os.environ.get("TEXT_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
IMAGE_MODEL_NAME  = os.environ.get("IMAGE_EMBED_MODEL", "Salesforce/blip-image-captioning-base") # Actually using BlipModel for embeddings
SERVER_PORT       = int(os.environ.get("RAG_SERVER_PORT", 5000))
DEFAULT_COLLECTION_NAME = "multimodal_rag_default" # Default collection if not specified

# Optional: Llama.cpp server (multimodal) endpoint IF you add a route that uses it
LLM_API_URL = os.environ.get("LLM_API_URL", "http://127.0.0.1:8080/completion") # Example for llama.cpp main endpoint
LLM_API_MODEL_ALIAS = os.environ.get("LLM_API_MODEL_ALIAS", "") # e.g., llava
# Note: The /v1/completions or /v1/chat/completions are OpenAI-compatible. 
# Plain /completion is often the basic llama.cpp one. Adjust LLM_API_URL accordingly.


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.makedirs(PERSIST_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# ─── EMBEDDERS ─────────────────────────────────────────────────────────────
text_encoder = None
img_processor = None
img_model = None

try:
    logger.info(f"Loading text model: {TEXT_MODEL_NAME}...")
    text_encoder  = SentenceTransformer(TEXT_MODEL_NAME, device=device) # Specify device
    logger.info(f"Loading image model: {IMAGE_MODEL_NAME}...")
    img_processor = BlipProcessor.from_pretrained(IMAGE_MODEL_NAME)
    img_model     = BlipModel.from_pretrained(IMAGE_MODEL_NAME).to(device) # BlipModel for embeddings
    logger.info("Embedding models loaded successfully.")
except Exception as e:
    logger.error(f"FATAL: Could not load embedding models: {e}", exc_info=True)
    # Optionally, exit if models are critical for server startup
    # exit(1)


def embed_text(text: str) -> Optional[list[float]]:
    if not text_encoder:
        logger.error("Text encoder not loaded. Cannot embed text.")
        return None
    try:
        return text_encoder.encode(text).tolist()
    except Exception as e:
        logger.error(f"Error embedding text '{text[:30]}...': {e}", exc_info=True)
        return None

def embed_image_b64(data_uri: str, caption: str="") -> Optional[list[float]]:
    if not img_model or not img_processor:
        logger.error("Image model/processor not loaded. Cannot embed image.")
        return None
    try:
        header, b64_str = data_uri.split(",", 1) if "," in data_uri else ("", data_uri)
        img_bytes = base64.b64decode(b64_str)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # For BlipModel, text input is optional but can influence the embedding
        inputs = img_processor(images=image, text=caption if caption else None, return_tensors="pt").to(device)
        with torch.no_grad():
            # Get pooled output for a general image embedding
            image_features = img_model.get_image_features(**inputs) 
            # Or, if you want text-conditioned image embedding (visual-semantic embedding):
            # outputs = img_model(**inputs)
            # image_features = outputs.image_embeds # This might require text input
            # If using image_embeds, and text is not None, it's a joint embedding
            # If text is None, image_embeds would be based on vision_outputs.pooler_output
            
            # Fallback logic as you had, check attribute names for your specific BLIP model variant
            # For base BlipModel, get_image_features often gives a [batch_size, hidden_size] tensor,
            # and we might want the pooled output (often the [CLS] token equivalent).
            # If image_features is already [1, embedding_dim], then that's it.
            # If it's [1, sequence_length, embedding_dim], take image_features[:, 0, :]
            
            if image_features.ndim == 3: # [batch, seq_len, hidden_dim]
                emb = image_features[:, 0, :] # Take the first token's embedding (like CLS)
            else: # Assumed [batch, hidden_dim]
                emb = image_features

        return emb.squeeze().cpu().tolist()
    except Exception as e:
        logger.error(f"Error embedding image (caption: '{caption[:30]}...'): {e}", exc_info=True)
        # Fallback to caption embedding if image embedding fails
        if caption:
            logger.warning("Falling back to embedding caption for image.")
            return embed_text(caption)
        return None


# ─── CHROMA DB SETUP ────────────────────────────────────────────────────────
chroma_client = None
try:
    chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
    # You can list collections to see if it's working:
    logger.info(f"ChromaDB client initialized. Existing collections: {[col.name for col in chroma_client.list_collections()]}")
except Exception as e:
    logger.error(f"FATAL: Could not initialize ChromaDB client: {e}", exc_info=True)
    # Optionally, exit if DB is critical
    # exit(1)

add_lock = threading.Lock() # To protect collection.add if it's not thread-safe for concurrent adds to same collection

# ─── FLASK APP ─────────────────────────────────────────────────────────────
app = Flask(__name__)

def get_collection_for_request(collection_name_req: Optional[str]) -> Optional[chromadb.api.models.Collection.Collection]:
    if not chroma_client:
        logger.error("ChromaDB client not available.")
        return None
    
    name_to_use = collection_name_req if collection_name_req else DEFAULT_COLLECTION_NAME
    try:
        # Assuming default embedding function of ChromaDB or SentenceTransformer is fine if not specified
        # Or pass your self.embedding_function from this server if you want to enforce it per collection.
        # For this server, embeddings are pre-computed, so Chroma's EF is for potential internal use or if you don't pass embeddings.
        collection = chroma_client.get_or_create_collection(name=name_to_use)
        logger.debug(f"Using ChromaDB collection: '{name_to_use}'")
        return collection
    except Exception as e:
        logger.error(f"Error accessing/creating collection '{name_to_use}': {e}", exc_info=True)
        return None


@app.route("/add", methods=["POST"])
def add_document():
    data = request.get_json(force=True) or {}
    text_content = data.get("text")
    image_data_uri = data.get("image")
    caption = data.get("caption", "")
    collection_name_req = data.get("collection_name") # Optional: client can suggest collection

    doc_to_store = None
    meta = {}
    vec = None
    doc_id_provided = data.get("id") # Optional ID from client

    if text_content:
        if not isinstance(text_content, str) or not text_content.strip():
            return jsonify({"error": "Text content must be a non-empty string"}), 400
        logger.info(f"Processing /add request for TEXT: '{text_content[:50]}...'")
        vec = embed_text(text_content)
        doc_to_store = text_content
        meta = {"type": "text", "original_content": text_content[:200]} # Store a snippet
        meta.update(data.get("metadata", {})) # Merge any client-provided metadata
    elif image_data_uri:
        if not isinstance(image_data_uri, str) or not image_data_uri.strip():
            return jsonify({"error": "Image data URI must be a non-empty string"}), 400
        logger.info(f"Processing /add request for IMAGE. Caption: '{caption[:50]}...'")
        vec = embed_image_b64(image_data_uri, caption)
        # Store a reference or placeholder for the image document
        doc_to_store = f"<image_ref:{int(time.time())}>" # Or a hash, or store actual URI if small
        meta = {"type": "image", "caption": caption, "image_data_uri_preview": image_data_uri[:100]+"..."}
        meta.update(data.get("metadata", {}))
    else:
        return jsonify({"error": "Request must contain 'text' or 'image' (with base64 data_uri)"}), 400

    if vec is None:
        return jsonify({"error": "Failed to generate embedding for content"}), 500

    collection_to_use = get_collection_for_request(collection_name_req)
    if not collection_to_use:
        return jsonify({"error": "Failed to access or create RAG collection"}), 500
    logger.info(f"Attempting to add document to collection: '{collection_to_use.name}'")

    entry_id = doc_id_provided or str(int(time.time() * 1_000_000)) # Use provided ID or generate one

    try:
        with add_lock: # Protect concurrent writes if underlying add isn't fully thread-safe
            collection_to_use.add(
                ids=[entry_id],
                embeddings=[vec],
                documents=[doc_to_store], 
                metadatas=[meta]
            )
        logger.info(f"Added to collection '{collection_to_use.name}', ID: {entry_id}, Type: {meta.get('type')}")
        return jsonify({"status": "added", "id": entry_id, "type": meta.get("type")}), 200
    except Exception as e:
        logger.error(f"Error adding to ChromaDB collection '{collection_to_use.name}': {e}", exc_info=True)
        return jsonify({"error": f"ChromaDB add operation failed: {str(e)}"}), 500


@app.route("/query", methods=["POST"])
def query_chroma_only(): # Function name can be anything, route is what matters
    """
    Query the RAG store (ChromaDB only) based on text query.
    Body: { 
        "query_text": "...", 
        "n_results": 3 (optional, defaults to 5 on server), 
        "collection_name": "optional_collection_name" (optional, defaults to DEFAULT_COLLECTION_NAME on server)
    }
    Returns the retrieved documents/sources.
    """
    logger.info(f"RAG Server /query endpoint HIT by {request.remote_addr}")

    if not request.is_json:
        logger.error("RAG Server /query: Request content type was not application/json.")
        return jsonify({"error": "Request must be JSON"}), 415

    try:
        data = request.get_json()
        if data is None:
            logger.error("RAG Server /query: Received empty JSON payload or failed to parse.")
            return jsonify({"error": "Empty or invalid JSON payload"}), 400
    except Exception as e: # Catches Werkzeug's BadRequest from get_json if malformed
        logger.error(f"RAG Server /query: Error parsing JSON payload: {e}", exc_info=True)
        raw_data_preview = request.get_data(as_text=True)[:200] # Get a preview of raw data
        logger.debug(f"Raw data preview on JSON parse error: {raw_data_preview}")
        return jsonify({"error": f"Malformed JSON payload: {str(e)}"}), 400

    logger.info(f"RAG Server /query received PARSED JSON data: {data}")

    q_text = data.get("query_text") 
    if q_text is None:
        q_text = data.get("query", "").strip() # Fallback
    else:
        q_text = str(q_text).strip()

    if not q_text:
        logger.error(f"RAG Server /query: 'query_text' or 'query' key not found or empty in parsed data: {data}")
        return jsonify({"error": "No query_text or query provided in JSON payload"}), 400

    n_results_req = data.get("n_results", 5)
    collection_name_req = data.get("collection_name") 

    try:
        n_results = int(n_results_req)
        if n_results <= 0: 
            logger.warning(f"Invalid n_results '{n_results_req}' for query '{q_text[:30]}...', defaulting to 5.")
            n_results = 5
    except (ValueError, TypeError):
        logger.warning(f"Invalid n_results type '{n_results_req}' for query '{q_text[:30]}...', defaulting to 5.")
        n_results = 5

    logger.info(f"Processing /query: Query='{q_text[:60].replace(chr(10), ' ')}...', N_Results={n_results}, Target_Collection='{collection_name_req or DEFAULT_COLLECTION_NAME}'")

    if not text_encoder: # Global text_encoder instance
        logger.error("Text encoder not loaded in RAG server. Cannot process query.")
        return jsonify({"error": "RAG server text embedding model not available"}), 503 

    query_embedding = embed_text(q_text) # Uses global text_encoder
    if query_embedding is None:
        logger.error(f"Failed to generate embedding for query: '{q_text[:60]}...'")
        return jsonify({"error": "Failed to generate query embedding"}), 500

    collection_to_use = get_collection_for_request(collection_name_req)
    if not collection_to_use:
        logger.error(f"Failed to access RAG collection '{collection_name_req or DEFAULT_COLLECTION_NAME}' for query.")
        return jsonify({"error": "Failed to access RAG collection for query"}), 500
    
    logger.debug(f"Querying actual ChromaDB collection: '{collection_to_use.name}' with {n_results} results.")

    try:
        chroma_results = collection_to_use.query(
            query_embeddings=[query_embedding], 
            n_results=n_results,
            include=["documents", "metadatas", "distances"] # Corrected: 'ids' is not part of 'include' for what to return in this list
        )
    except Exception as e:
        logger.error(f"Error querying ChromaDB collection '{collection_to_use.name}': {e}", exc_info=True)
        # Log specific ChromaDB errors if possible
        if "Expected include item to be one of" in str(e): # Example of specific error check
            logger.error("This ChromaDB error often means an invalid field was in the 'include' list for query().")
        return jsonify({"error": f"ChromaDB query operation failed: {str(e)}"}), 500

    response_sources = []
    if chroma_results and chroma_results.get("ids") and chroma_results["ids"][0] and len(chroma_results["ids"][0]) > 0:
        num_retrieved = len(chroma_results["ids"][0])
        logger.debug(f"ChromaDB query to '{collection_to_use.name}' returned {num_retrieved} items.")
        
        # Safely access parts of chroma_results
        docs_list = (chroma_results.get("documents", [[]])[0] if chroma_results.get("documents") and chroma_results["documents"] else []) or ([None] * num_retrieved)
        meta_list = (chroma_results.get("metadatas", [[]])[0] if chroma_results.get("metadatas") and chroma_results["metadatas"] else []) or ([{}] * num_retrieved)
        dist_list = (chroma_results.get("distances", [[]])[0] if chroma_results.get("distances") and chroma_results["distances"] else []) or ([None] * num_retrieved)
        ids_list = chroma_results["ids"][0] # IDs list should correspond to number of results

        for i in range(num_retrieved):
            source_entry = {
                "id": ids_list[i] if i < len(ids_list) else None, # Should always exist if items returned
                "text_content": docs_list[i] if i < len(docs_list) else None, 
                "metadata": meta_list[i] if i < len(meta_list) else {},
                "distance": dist_list[i] if i < len(dist_list) else None
            }
            response_sources.append(source_entry)
    else:
        logger.info(f"No results found in ChromaDB collection '{collection_to_use.name}' for the query: '{q_text[:60]}...'")

    logger.info(f"Returning {len(response_sources)} sources for query '{q_text[:60]}...'")
    return jsonify({
        "query_text_received": q_text,
        "retrieved_sources": response_sources 
    }), 200



# Optional: An endpoint that does RAG + LLM call (if you want this server to do it)
# @app.route("/answer_with_rag", methods=["POST"])
# def answer_with_rag():
#     data = request.get_json(force=True) or {}
#     q_text = data.get("query", "").strip()
#     query_images_b64 = data.get("images") # List of base64 data URIs
#     # ... (perform query_chroma_documents logic to get sources) ...
#     # retrieved_sources = ...
#     # ... (build context_str from retrieved_sources) ...
#     # ... (build full_prompt for LLM) ...
#     # try:
#     #     llm_answer = call_llm(full_prompt, images=query_images_b64) # Uses the internal LLM_API_URL
#     #     return jsonify({"query": q_text, "sources": retrieved_sources, "answer": llm_answer})
#     # except Exception as e:
#     #     return jsonify({"error": f"LLM call failed during answer_with_rag: {str(e)}"}), 500
#     return jsonify({"error": "answer_with_rag endpoint not fully implemented"}), 501

if __name__ == "__main__":
    if not text_encoder or not img_model or not img_processor or not chroma_client:
        logger.critical("One or more critical components (embedding models, ChromaDB client) failed to initialize. Server cannot start.")
    else:
        logger.info(f"Starting RAG server (ChromaDB backend) on 0.0.0.0:{SERVER_PORT}")
        logger.info(f"ChromaDB persistence directory: '{PERSIST_DIR}'")
        # Set debug=False for production or when running with Gunicorn/Waitress
        # For development, debug=True can be useful but might cause models to load twice.
        app.run(host="0.0.0.0", port=SERVER_PORT, debug=False)