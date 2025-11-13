"""
Minimal Sales API - Full Stack with Vector Search
"""

from fastapi import FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime
import logging
import asyncio

from database import init_db, close_db, get_db
from models import Conversation
from redis_client import close_redis, cache_conversation, get_cached_conversation, invalidate_cache
from llm_client import get_sales_response
from mcp_client import handle_objection, get_pitch_template, calculate_value
from qdrant_service import ensure_collection, get_qdrant_stats
from content_processor import smart_chunk_content, analyze_content_complexity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    logger.info("Starting Sales API...")
    
    # Initialize database
    try:
        await init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.warning(f"Database init failed: {e}. Running without DB.")
    
    # Qdrant initialization - don't block startup if it fails
    # Run in background to avoid blocking healthcheck
    import asyncio
    asyncio.create_task(initialize_qdrant())
    
    yield
    logger.info("Shutting down...")
    await close_db()
    close_redis()


async def initialize_qdrant():
    """Initialize Qdrant collection in background"""
    try:
        await ensure_collection()
        logger.info("Qdrant collection ready")
    except Exception as e:
        logger.warning(f"Qdrant init failed: {e}. Running without vector search.")


app = FastAPI(
    title="Sales API - Complete System",
    version="1.5.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "service": "Sales API",
        "version": "1.5.0",
        "status": "running",
        "features": ["PostgreSQL", "Redis", "AI", "MCP Tools", "Qdrant", "Conversations", "Caching", "Vector Search"]
    }


@app.get("/health")
async def health():
    """Health check with system status"""
    try:
        qdrant_stats = get_qdrant_stats()
    except Exception as e:
        logger.warning(f"Failed to get Qdrant stats: {e}")
        qdrant_stats = None
    
    return {
        "status": "healthy",
        "qdrant": qdrant_stats if qdrant_stats else "not_configured"
    }


@app.post("/api/sales/conversations")
async def create_conversation(data: dict, db: AsyncSession = Depends(get_db)):
    """Create a new sales conversation"""
    try:
        conversation = Conversation(
            session_id=data.get("session_id", f"session-{datetime.now().timestamp()}"),
            email=data.get("email"),
            name=data.get("name"),
            company=data.get("company"),
            messages=[]
        )
        db.add(conversation)
        await db.commit()
        await db.refresh(conversation)
        
        return {
            "id": conversation.id,
            "session_id": conversation.session_id,
            "email": conversation.email,
            "status": "created"
        }
    except Exception as e:
        logger.error(f"Error creating conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sales/conversations/{session_id}")
async def get_conversation(session_id: str, db: AsyncSession = Depends(get_db)):
    """Get conversation by session ID - with Redis caching"""
    try:
        # Try cache first
        cached = get_cached_conversation(session_id)
        if cached:
            logger.info(f"Returning cached conversation for {session_id}")
            return cached
        
        # Cache miss - get from database
        result = await db.execute(
            select(Conversation).where(Conversation.session_id == session_id)
        )
        conversation = result.scalar_one_or_none()
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        response = {
            "id": conversation.id,
            "session_id": conversation.session_id,
            "email": conversation.email,
            "name": conversation.name,
            "company": conversation.company,
            "messages": conversation.messages,
            "current_stage": conversation.current_stage,
            "qualification_score": conversation.qualification_score,
            "created_at": conversation.created_at.isoformat() if conversation.created_at else None
        }
        
        # Cache for 30 minutes
        cache_conversation(session_id, response, ttl=1800)
        
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/sales/message")
async def send_message(data: dict, db: AsyncSession = Depends(get_db)):
    """Send message and save to database"""
    try:
        session_id = data.get("session_id", "default")
        message_text = data.get("message", "")
        
        # Get or create conversation
        result = await db.execute(
            select(Conversation).where(Conversation.session_id == session_id)
        )
        conversation = result.scalar_one_or_none()
        
        if not conversation:
            conversation = Conversation(
                session_id=session_id,
                email=data.get("email"),
                messages=[]
            )
            db.add(conversation)
        
        # Add message to conversation
        messages = conversation.messages or []
        messages.append({
            "role": "user",
            "content": message_text,
            "timestamp": datetime.now().isoformat()
        })
        
        # Get AI response from LLM Gateway
        response_text = await get_sales_response(
            conversation_history=messages,
            user_message=message_text,
            context={
                "email": conversation.email,
                "name": conversation.name,
                "company": conversation.company,
                "stage": conversation.current_stage
            }
        )
        
        messages.append({
            "role": "assistant",
            "content": response_text,
            "timestamp": datetime.now().isoformat()
        })
        
        conversation.messages = messages
        conversation.updated_at = datetime.now()
        
        await db.commit()
        await db.refresh(conversation)
        
        # Invalidate cache since conversation updated
        invalidate_cache(session_id)
        
        return {
            "message": response_text,
            "session_id": session_id,
            "conversation_id": conversation.id,
            "status": "ok"
        }
    except Exception as e:
        logger.error(f"Error sending message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============= MCP TOOL ENDPOINTS =============

@app.post("/api/mcp/objection")
async def handle_sales_objection(data: dict):
    """Handle sales objection using MCP"""
    try:
        objection_type = data.get("objection_type")
        context = data.get("context", {})
        
        result = await handle_objection(objection_type, context)
        
        if result:
            return {"status": "ok", "data": result}
        else:
            return {"status": "error", "message": "MCP tool unavailable"}
    except Exception as e:
        logger.error(f"Error handling objection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/mcp/pitch")
async def get_sales_pitch(data: dict):
    """Get pitch template using MCP"""
    try:
        industry = data.get("industry")
        pain_points = data.get("pain_points", [])
        company_size = data.get("company_size")
        
        result = await get_pitch_template(industry, pain_points, company_size)
        
        if result:
            return {"status": "ok", "data": result}
        else:
            return {"status": "error", "message": "MCP tool unavailable"}
    except Exception as e:
        logger.error(f"Error getting pitch: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/mcp/value")
async def calculate_roi(data: dict):
    """Calculate value/ROI using MCP"""
    try:
        company_size = data.get("company_size")
        industry = data.get("industry")
        current_process = data.get("current_process")
        
        result = await calculate_value(company_size, industry, current_process)
        
        if result:
            return {"status": "ok", "data": result}
        else:
            return {"status": "error", "message": "MCP tool unavailable"}
    except Exception as e:
        logger.error(f"Error calculating value: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# STREAMING ENDPOINT - Tiptap Editor Content Streaming
# ============================================================================

@app.websocket("/ws/stream/{session_id}")
async def stream_content_websocket(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for streaming content to Tiptap editor.
    
    Flow:
    1. Client connects and sends content request
    2. Server processes content (sanitize, chunk, transform)
    3. Server streams chunks at optimal rate
    4. Client receives chunks and displays in editor
    
    Message Types:
    - Client -> Server: {"type": "stream_request", "content": "...", "content_type": "text|html|markdown", "speed": "normal", "chunk_by": "word"}
    - Server -> Client: {"type": "connected", "session_id": "..."}
    - Server -> Client: {"type": "stream_start", "total_chunks": 100}
    - Server -> Client: {"type": "chunk", "data": "...", "index": 0}
    - Server -> Client: {"type": "stream_complete", "total_chunks": 100}
    """
    await websocket.accept()
    logger.info(f"Streaming WebSocket connected: {session_id}")
    
    try:
        # Send connection confirmation
        await websocket.send_json({
            "type": "connected",
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Wait for streaming requests
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "stream_request":
                # Extract request parameters
                content = data.get("content", "")
                content_type = data.get("content_type", "text")
                speed = data.get("speed", "normal")
                chunk_by = data.get("chunk_by", "word")
                
                logger.info(f"Stream request: session={session_id}, length={len(content)}, type={content_type}, speed={speed}")
                
                # Process and stream content
                await process_and_stream_content(
                    websocket=websocket,
                    content=content,
                    content_type=content_type,
                    speed=speed,
                    chunk_by=chunk_by
                )
                
            elif data.get("type") == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                })
                
    except WebSocketDisconnect:
        logger.info(f"Streaming WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"Streaming WebSocket error: {session_id} - {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        except:
            pass


async def process_and_stream_content(
    websocket: WebSocket,
    content: str,
    content_type: str,
    speed: str,
    chunk_by: str
):
    """
    Process content and stream chunks to client.
    
    Phase 2.2: Enhanced with smart content processing
    - HTML sanitization
    - Markdown to HTML conversion
    - Intelligent chunking
    - Adaptive speed calculation
    """
    try:
        # Analyze content for optimal strategy
        analysis = analyze_content_complexity(content)
        logger.info(f"Content analysis: {analysis}")
        
        # Use adaptive speed if requested
        if speed == "adaptive":
            speed = analysis["recommended_speed"]
            logger.info(f"Adaptive speed selected: {speed}")
        
        # Speed preset delays (seconds)
        speed_delays = {
            "slow": 0.3,
            "normal": 0.1,
            "fast": 0.05,
            "superfast": 0.03
        }
        delay = speed_delays.get(speed, 0.1)
        
        # Smart chunking with content processing
        chunks, metadata = smart_chunk_content(
            content=content,
            content_type=content_type,
            chunk_by=chunk_by
        )
        
        logger.info(f"Processed content: {metadata['chunk_count']} chunks, complexity: {analysis['complexity']}")
        
        # Send stream start event with metadata
        await websocket.send_json({
            "type": "stream_start",
            "total_chunks": len(chunks),
            "content_type": content_type,
            "metadata": {
                "complexity": analysis["complexity"],
                "word_count": analysis["word_count"],
                "has_html": analysis["has_html"],
                "speed_used": speed
            },
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Stream chunks
        for i, chunk in enumerate(chunks):
            await websocket.send_json({
                "type": "chunk",
                "data": chunk,
                "index": i,
                "total": len(chunks),
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Delay between chunks
            await asyncio.sleep(delay)
        
        # Send stream complete event
        await websocket.send_json({
            "type": "stream_complete",
            "total_chunks": len(chunks),
            "metadata": metadata,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        logger.info(f"Stream completed: {len(chunks)} chunks")
        
    except Exception as e:
        logger.error(f"Stream processing error: {e}")
        await websocket.send_json({
            "type": "error",
            "message": f"Streaming error: {str(e)}"
        })


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
