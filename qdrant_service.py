"""
Qdrant Client - Vector database for semantic search
"""

import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import os
import logging
from typing import List, Dict, Optional
import hashlib
import json

logger = logging.getLogger(__name__)

# Qdrant URL from environment
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Initialize client - use HTTP client instead of native Qdrant client for Railway
http_client = None

if QDRANT_URL:
    try:
        logger.info(f"Initializing Qdrant HTTP client with URL: {QDRANT_URL}")
        logger.info(f"Qdrant API Key present: {bool(QDRANT_API_KEY)}")
        
        # Use httpx for direct HTTP communication - more reliable on Railway
        headers = {}
        if QDRANT_API_KEY:
            headers["api-key"] = QDRANT_API_KEY
        
        http_client = httpx.AsyncClient(
            base_url=QDRANT_URL,
            headers=headers,
            timeout=30.0
        )
        
        logger.info("Qdrant HTTP client initialized successfully")
            
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant client: {type(e).__name__}: {e}")
        http_client = None
else:
    logger.info("QDRANT_URL not set - semantic search unavailable")


COLLECTION_NAME = "sales_knowledge"


async def ensure_collection():
    """Ensure Qdrant collection exists using HTTP API"""
    if not http_client:
        logger.warning("Qdrant HTTP client not initialized - skipping collection setup")
        return False
    
    try:
        # Check existing collections
        response = await http_client.get("/collections")
        response.raise_for_status()
        data = response.json()
        
        collections = data.get("result", {}).get("collections", [])
        exists = any(c.get("name") == COLLECTION_NAME for c in collections)
        
        if not exists:
            # Create collection
            payload = {
                "vectors": {
                    "size": 1536,
                    "distance": "Cosine"
                }
            }
            response = await http_client.put(f"/collections/{COLLECTION_NAME}", json=payload)
            response.raise_for_status()
            logger.info(f"Created Qdrant collection: {COLLECTION_NAME}")
        else:
            logger.info(f"Qdrant collection already exists: {COLLECTION_NAME}")
        
        return True
    except Exception as e:
        logger.warning(f"Qdrant collection setup skipped: {type(e).__name__}: {e}")
        return False


async def search_knowledge(query_vector: List[float], limit: int = 5) -> List[Dict]:
    """
    Search sales knowledge base with vector using HTTP API
    
    Args:
        query_vector: Embedding vector (1536 dimensions)
        limit: Max results
    
    Returns:
        List of matching documents
    """
    if not http_client:
        logger.warning("Qdrant not available for search")
        return []
    
    try:
        payload = {
            "vector": query_vector,
            "limit": limit,
            "with_payload": True
        }
        response = await http_client.post(f"/collections/{COLLECTION_NAME}/points/search", json=payload)
        response.raise_for_status()
        data = response.json()
        
        documents = []
        for result in data.get("result", []):
            documents.append({
                "id": result.get("id"),
                "score": result.get("score"),
                "content": result.get("payload", {}).get("content", ""),
                "metadata": result.get("payload", {}).get("metadata", {})
            })
        
        logger.info(f"Qdrant search returned {len(documents)} results")
        return documents
        
    except Exception as e:
        logger.error(f"Qdrant search failed: {e}")
        return []


async def store_knowledge(content: str, metadata: Dict, embedding: List[float]) -> bool:
    """
    Store document in Qdrant using HTTP API
    
    Args:
        content: Document text
        metadata: Document metadata (type, category, etc)
        embedding: Vector embedding (1536 dimensions)
    
    Returns:
        Success status
    """
    if not http_client:
        return False
    
    try:
        # Generate ID from content hash
        doc_id = hashlib.md5(content.encode()).hexdigest()
        
        # Ensure collection exists
        await ensure_collection()
        
        # Store point
        payload = {
            "points": [
                {
                    "id": doc_id,
                    "vector": embedding,
                    "payload": {
                        "content": content,
                        "metadata": metadata
                    }
                }
            ]
        }
        response = await http_client.put(f"/collections/{COLLECTION_NAME}/points", json=payload)
        response.raise_for_status()
        
        logger.info(f"Stored document in Qdrant: {doc_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to store in Qdrant: {e}")
        return False


def get_qdrant_stats() -> Optional[Dict]:
    """Get Qdrant collection stats using HTTP API"""
    if not http_client:
        return None
    
    try:
        import asyncio
        # Create a sync wrapper for async call
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If event loop is running, return placeholder
            return {"status": "checking"}
        else:
            response = loop.run_until_complete(http_client.get(f"/collections/{COLLECTION_NAME}"))
            response.raise_for_status()
            data = response.json()
            
            result = data.get("result", {})
            return {
                "collection": COLLECTION_NAME,
                "vectors_count": result.get("vectors_count", 0),
                "points_count": result.get("points_count", 0),
                "status": result.get("status", "unknown")
            }
    except Exception as e:
        logger.error(f"Failed to get Qdrant stats: {e}")
        return None
