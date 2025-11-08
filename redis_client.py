"""
Redis connection for Sales API - Session caching
"""

import redis
import os
import json
import logging

logger = logging.getLogger(__name__)

# Redis connection from environment
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Parse Redis URL for public connection
redis_client = None

try:
    redis_client = redis.from_url(
        REDIS_URL,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5
    )
    # Test connection
    redis_client.ping()
    logger.info(f"Redis connected successfully")
except Exception as e:
    logger.warning(f"Redis connection failed: {e}. Running without cache.")
    redis_client = None


def get_redis():
    """Get Redis client"""
    return redis_client


def cache_conversation(session_id: str, data: dict, ttl: int = 3600):
    """Cache conversation data in Redis"""
    if not redis_client:
        return False
    
    try:
        key = f"conversation:{session_id}"
        redis_client.setex(key, ttl, json.dumps(data))
        logger.info(f"Cached conversation {session_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to cache conversation: {e}")
        return False


def get_cached_conversation(session_id: str):
    """Get cached conversation from Redis"""
    if not redis_client:
        return None
    
    try:
        key = f"conversation:{session_id}"
        data = redis_client.get(key)
        if data:
            logger.info(f"Cache hit for {session_id}")
            return json.loads(data)
        logger.info(f"Cache miss for {session_id}")
        return None
    except Exception as e:
        logger.error(f"Failed to get cached conversation: {e}")
        return None


def invalidate_cache(session_id: str):
    """Invalidate cached conversation"""
    if not redis_client:
        return False
    
    try:
        key = f"conversation:{session_id}"
        redis_client.delete(key)
        logger.info(f"Invalidated cache for {session_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to invalidate cache: {e}")
        return False


def close_redis():
    """Close Redis connection"""
    if redis_client:
        redis_client.close()
        logger.info("Redis connection closed")
