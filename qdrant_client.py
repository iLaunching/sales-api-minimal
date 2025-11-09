"""
Qdrant Client - Vector database for semantic search
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import os
import logging
from typing import List, Dict, Optional
import hashlib

logger = logging.getLogger(__name__)

# Qdrant URL from environment
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Initialize client
qdrant_client = None

if QDRANT_URL:
    try:
        qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=30
        )
        logger.info(f"Qdrant client initialized: {QDRANT_URL}")
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant: {e}")
        qdrant_client = None
else:
    logger.warning("QDRANT_URL not set - semantic search unavailable")


COLLECTION_NAME = "sales_knowledge"


async def ensure_collection():
    """Ensure Qdrant collection exists"""
    if not qdrant_client:
        return False
    
    try:
        collections = qdrant_client.get_collections().collections
        exists = any(c.name == COLLECTION_NAME for c in collections)
        
        if not exists:
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )
            logger.info(f"Created Qdrant collection: {COLLECTION_NAME}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to ensure collection: {e}")
        return False


async def search_knowledge(query_vector: List[float], limit: int = 5) -> List[Dict]:
    """
    Search sales knowledge base with vector
    
    Args:
        query_vector: Embedding vector (1536 dimensions)
        limit: Max results
    
    Returns:
        List of matching documents
    """
    if not qdrant_client:
        logger.warning("Qdrant not available for search")
        return []
    
    try:
        results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=limit
        )
        
        documents = []
        for result in results:
            documents.append({
                "id": result.id,
                "score": result.score,
                "content": result.payload.get("content", ""),
                "metadata": result.payload.get("metadata", {})
            })
        
        logger.info(f"Qdrant search returned {len(documents)} results")
        return documents
        
    except Exception as e:
        logger.error(f"Qdrant search failed: {e}")
        return []


async def store_knowledge(content: str, metadata: Dict, embedding: List[float]) -> bool:
    """
    Store document in Qdrant
    
    Args:
        content: Document text
        metadata: Document metadata (type, category, etc)
        embedding: Vector embedding (1536 dimensions)
    
    Returns:
        Success status
    """
    if not qdrant_client:
        return False
    
    try:
        # Generate ID from content hash
        doc_id = hashlib.md5(content.encode()).hexdigest()
        
        # Ensure collection exists
        await ensure_collection()
        
        # Store point
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=doc_id,
                    vector=embedding,
                    payload={
                        "content": content,
                        "metadata": metadata
                    }
                )
            ]
        )
        
        logger.info(f"Stored document in Qdrant: {doc_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to store in Qdrant: {e}")
        return False


def get_qdrant_stats() -> Optional[Dict]:
    """Get Qdrant collection stats"""
    if not qdrant_client:
        return None
    
    try:
        info = qdrant_client.get_collection(COLLECTION_NAME)
        return {
            "collection": COLLECTION_NAME,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status
        }
    except Exception as e:
        logger.error(f"Failed to get Qdrant stats: {e}")
        return None
