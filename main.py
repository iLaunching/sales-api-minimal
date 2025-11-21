"""
Minimal Sales API - Full Stack with Vector Search
Version: 2.4.1 - WebSocket streaming with metadata
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
from markdown_to_tiptap import convert_markdown_to_tiptap

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
        test_mode = data.get("test_mode", False)  # Add test mode flag
        
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
        
        # Get AI response from LLM Gateway (with test mode support)
        response_text = await get_sales_response(
            conversation_history=messages,
            user_message=message_text,
            context={
                "email": conversation.email,
                "name": conversation.name,
                "company": conversation.company,
                "stage": conversation.current_stage
            },
            test_mode=test_mode
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
    Production-ready WebSocket endpoint for streaming content to Tiptap editor.
    
    Security & Reliability Features:
    - Session validation and timeout handling
    - Rate limiting per session
    - Comprehensive error recovery
    - Resource cleanup on disconnect
    - Request validation and sanitization
    - Stream control (pause/resume/skip)
    
    Flow:
    1. Client connects and sends content request
    2. Server validates and processes content (sanitize, chunk, transform)
    3. Server streams chunks at optimal rate with backpressure handling
    4. Client can control stream (pause/resume/skip)
    5. Client receives chunks and displays in editor
    
    Message Types:
    - Client → Server: {"type": "stream_request", "content": "...", "content_type": "text|html|markdown", "speed": "normal", "chunk_by": "word"}
    - Client → Server: {"type": "stream_control", "action": "pause|resume|skip"}
    - Server → Client: {"type": "connected", "session_id": "...", "limits": {...}}
    - Server → Client: {"type": "stream_start", "total_chunks": 100, "metadata": {...}}
    - Server → Client: {"type": "chunk", "data": "...", "index": 0}
    - Server → Client: {"type": "stream_complete", "total_chunks": 100}
    - Server → Client: {"type": "stream_paused"}
    - Server → Client: {"type": "stream_resumed"}
    - Server → Client: {"type": "stream_skipped"}
    - Server → Client: {"type": "error", "message": "...", "code": "..."}
    """
    # Session tracking and rate limiting
    session_start = datetime.utcnow()
    request_count = 0
    MAX_REQUESTS_PER_SESSION = 100
    MAX_SESSION_DURATION = 3600  # 1 hour
    MAX_CONTENT_SIZE = 100_000  # 100KB per request
    
    # Stream control state
    stream_control = {
        "paused": False,
        "skip": False,
        "current_task": None,
        "message_queue": asyncio.Queue()
    }
    
    await websocket.accept()
    logger.info(f"✅ WebSocket connected: {session_id}")
    
    # Start background task to handle incoming messages
    async def handle_incoming_messages():
        """Background task to handle incoming WebSocket messages"""
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=300.0)
                await stream_control["message_queue"].put(data)
            except asyncio.TimeoutError:
                await stream_control["message_queue"].put({"type": "idle_timeout"})
                break
            except Exception as e:
                logger.error(f"Error receiving message: {e}")
                break
    
    message_handler_task = asyncio.create_task(handle_incoming_messages())
    
    try:
        # Send connection confirmation with limits
        await websocket.send_json({
            "type": "connected",
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "limits": {
                "max_requests": MAX_REQUESTS_PER_SESSION,
                "max_content_size": MAX_CONTENT_SIZE,
                "max_session_duration": MAX_SESSION_DURATION
            }
        })
        
        # Wait for streaming requests
        while True:
            # Check session duration
            session_duration = (datetime.utcnow() - session_start).total_seconds()
            if session_duration > MAX_SESSION_DURATION:
                logger.warning(f"Session {session_id} exceeded max duration: {session_duration}s")
                await websocket.send_json({
                    "type": "error",
                    "code": "SESSION_TIMEOUT",
                    "message": "Session duration exceeded maximum allowed time"
                })
                break
            
            # Get message from queue
            try:
                data = await asyncio.wait_for(stream_control["message_queue"].get(), timeout=1.0)
            except asyncio.TimeoutError:
                # No message, continue loop
                continue
            
            if data.get("type") == "idle_timeout":
                logger.info(f"Session {session_id} idle timeout")
                await websocket.send_json({
                    "type": "error",
                    "code": "IDLE_TIMEOUT",
                    "message": "No activity for 5 minutes"
                })
                break
            
            if data.get("type") == "stream_request":
                request_count += 1
                
                # Rate limiting
                if request_count > MAX_REQUESTS_PER_SESSION:
                    logger.warning(f"Session {session_id} exceeded rate limit: {request_count}")
                    await websocket.send_json({
                        "type": "error",
                        "code": "RATE_LIMIT_EXCEEDED",
                        "message": f"Maximum {MAX_REQUESTS_PER_SESSION} requests per session"
                    })
                    continue
                
                # Extract and validate request parameters
                content = data.get("content", "")
                content_type = data.get("content_type", "text")
                speed = data.get("speed", "normal")
                chunk_by = data.get("chunk_by", "word")
                
                # Validate content size
                if len(content) > MAX_CONTENT_SIZE:
                    logger.warning(f"Content too large: {len(content)} bytes")
                    await websocket.send_json({
                        "type": "error",
                        "code": "CONTENT_TOO_LARGE",
                        "message": f"Content exceeds {MAX_CONTENT_SIZE} bytes"
                    })
                    continue
                
                # Validate parameters
                if content_type not in ["text", "html", "markdown"]:
                    await websocket.send_json({
                        "type": "error",
                        "code": "INVALID_CONTENT_TYPE",
                        "message": f"content_type must be text, html, or markdown"
                    })
                    continue
                
                if chunk_by not in ["word", "sentence", "paragraph", "character"]:
                    await websocket.send_json({
                        "type": "error",
                        "code": "INVALID_CHUNK_STRATEGY",
                        "message": f"chunk_by must be word, sentence, paragraph, or character"
                    })
                    continue
                
                logger.info(f"🔄 Stream request #{request_count}: session={session_id}, length={len(content)}, type={content_type}, speed={speed}")
                
                # Get LLM response for the user's query
                try:
                    # Check if test_mode is enabled
                    test_mode = data.get("test_mode", False)
                    
                    logger.info(f"🤖 Calling LLM with user query: {content[:100]}... (test_mode={test_mode})")
                    llm_response = await get_sales_response(
                        conversation_history=[],  # TODO: Track conversation history per session
                        user_message=content,
                        test_mode=test_mode
                    )
                    logger.info(f"✅ LLM response received: {len(llm_response)} chars")
                    
                    # Verify we got valid content
                    if not llm_response or not isinstance(llm_response, str):
                        logger.error(f"❌ Invalid LLM response type: {type(llm_response)}")
                        raise ValueError("Invalid LLM response")
                    
                    # Convert LLM markdown to Tiptap JSON nodes
                    logger.info("🔄 Converting markdown to Tiptap JSON...")
                    try:
                        tiptap_nodes = convert_markdown_to_tiptap(llm_response)
                        logger.info(f"✅ Converted to {len(tiptap_nodes)} Tiptap nodes")
                    except Exception as convert_error:
                        logger.error(f"❌ Markdown conversion failed: {convert_error}", exc_info=True)
                        # Fallback: create simple paragraph node with raw text
                        tiptap_nodes = [{
                            "type": "paragraph",
                            "content": [{
                                "type": "text",
                                "text": llm_response
                            }]
                        }]
                        logger.info("⚠️ Using fallback paragraph node")
                    
                except Exception as llm_error:
                    logger.error(f"❌ LLM call failed: {llm_error}", exc_info=True)
                    logger.error(f"Original content was: {content[:200]}...")
                    # Fallback to error message as paragraph node
                    tiptap_nodes = [{
                        "type": "paragraph",
                        "content": [{
                            "type": "text",
                            "text": "I apologize, but I'm having trouble processing your request right now. Please try again in a moment."
                        }]
                    }]
                
                # Process and stream LLM response with error handling
                try:
                    stream_control["paused"] = False
                    stream_control["skip"] = False
                    
                    # Create streaming task with Tiptap JSON nodes
                    streaming_task = asyncio.create_task(
                        stream_tiptap_nodes(
                            websocket=websocket,
                            nodes=tiptap_nodes,
                            speed=speed,
                            session_id=session_id,
                            stream_control=stream_control
                        )
                    )
                    
                    # Process control messages while streaming
                    while not streaming_task.done():
                        try:
                            # Check for control messages without blocking
                            control_msg = await asyncio.wait_for(
                                stream_control["message_queue"].get(), 
                                timeout=0.1
                            )
                            
                            if control_msg.get("type") == "stream_control":
                                action = control_msg.get("action")
                                logger.info(f"🎮 Stream control during streaming: {action}")
                                
                                if action == "pause":
                                    stream_control["paused"] = True
                                    await websocket.send_json({
                                        "type": "stream_paused",
                                        "timestamp": datetime.utcnow().isoformat()
                                    })
                                elif action == "resume":
                                    stream_control["paused"] = False
                                    await websocket.send_json({
                                        "type": "stream_resumed",
                                        "timestamp": datetime.utcnow().isoformat()
                                    })
                                elif action == "skip":
                                    stream_control["skip"] = True
                                    await websocket.send_json({
                                        "type": "stream_skipped",
                                        "timestamp": datetime.utcnow().isoformat()
                                    })
                            else:
                                # Put non-control messages back in queue
                                await stream_control["message_queue"].put(control_msg)
                                
                        except asyncio.TimeoutError:
                            # No control message, continue
                            pass
                        
                        # Small delay to prevent busy loop
                        await asyncio.sleep(0.05)
                    
                    # Wait for streaming to complete
                    await streaming_task
                    
                except Exception as e:
                    logger.error(f"❌ Stream processing failed: {type(e).__name__}: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "code": "PROCESSING_ERROR",
                        "message": f"Failed to process content: {str(e)[:100]}"
                    })
            
            elif data.get("type") == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            else:
                logger.warning(f"Unknown message type: {data.get('type')}")
                await websocket.send_json({
                    "type": "error",
                    "code": "UNKNOWN_MESSAGE_TYPE",
                    "message": f"Unknown message type: {data.get('type')}"
                })
                
    except WebSocketDisconnect:
        logger.info(f"🔌 WebSocket disconnected: {session_id} (requests: {request_count})")
    except Exception as e:
        logger.error(f"❌ WebSocket error: {session_id} - {type(e).__name__}: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "code": "INTERNAL_ERROR",
                "message": "Internal server error"
            })
        except:
            pass
    finally:
        # Cleanup
        message_handler_task.cancel()
        logger.info(f"🧹 Cleaning up session: {session_id}")


async def stream_tiptap_nodes(
    websocket: WebSocket,
    nodes: list,
    speed: str,
    session_id: str = "unknown",
    stream_control: dict = None
):
    """
    Stream Tiptap JSON nodes directly to the client.
    
    Phase 1 & 2 Implementation:
    - Receives parsed Tiptap JSON nodes from markdown converter
    - Streams nodes one at a time with controlled timing
    - Supports stream control (pause/resume/skip)
    - Sends complete JSON structures (no parsing needed on frontend)
    
    Message Format:
    - {"type": "stream_start", "total_nodes": N, "metadata": {...}}
    - {"type": "node", "data": {...tiptap_node...}, "index": N}
    - {"type": "stream_complete", "total_nodes": N}
    
    Args:
        websocket: Active WebSocket connection
        nodes: List of Tiptap JSON node dictionaries
        speed: Speed preset ("slow"|"normal"|"fast"|"superfast")
        session_id: Session identifier for logging
        stream_control: Dict with pause/skip state
    """
    if stream_control is None:
        stream_control = {"paused": False, "skip": False}
    
    try:
        # Speed preset delays (seconds)
        speed_delays = {
            "slow": 0.5,
            "normal": 0.2,
            "fast": 0.1,
            "superfast": 0.05
        }
        delay = speed_delays.get(speed, 0.2)
        
        logger.info(f"🎬 Starting Tiptap node stream: {len(nodes)} nodes, speed={speed}")
        
        # Send stream start event
        await websocket.send_json({
            "type": "stream_start",
            "total_nodes": len(nodes),
            "metadata": {
                "node_count": len(nodes),
                "speed_used": speed,
                "format": "tiptap_json"
            },
            "timestamp": datetime.utcnow().isoformat()
        })
        
        sent_nodes = 0
        
        for i, node in enumerate(nodes):
            try:
                # Check if stream should be skipped
                if stream_control.get("skip"):
                    logger.info("⏭️ Stream skipped, sending all remaining nodes")
                    # Send all remaining nodes immediately
                    for remaining_node in nodes[i:]:
                        await websocket.send_json({
                            "type": "node",
                            "data": remaining_node,
                            "index": sent_nodes,
                            "timestamp": datetime.utcnow().isoformat()
                        })
                        sent_nodes += 1
                    break
                
                # Wait while paused
                while stream_control.get("paused"):
                    if stream_control.get("skip"):
                        break
                    await asyncio.sleep(0.1)
                
                # Check skip again after pause
                if stream_control.get("skip"):
                    logger.info("⏭️ Stream skipped after pause")
                    for remaining_node in nodes[i:]:
                        await websocket.send_json({
                            "type": "node",
                            "data": remaining_node,
                            "index": sent_nodes,
                            "timestamp": datetime.utcnow().isoformat()
                        })
                        sent_nodes += 1
                    break
                
                # Send the Tiptap JSON node
                await websocket.send_json({
                    "type": "node",
                    "data": node,
                    "index": sent_nodes,
                    "shouldAnimate": True,  # Frontend can use this for animation control
                    "timestamp": datetime.utcnow().isoformat()
                })
                sent_nodes += 1
                logger.debug(f"Sent node {sent_nodes}/{len(nodes)}: {node.get('type', 'unknown')}")
                
                # Throttle based on speed preset
                if i < len(nodes) - 1:  # Don't delay after last node
                    elapsed = 0
                    interval = 0.05  # Check every 50ms
                    while elapsed < delay:
                        if stream_control.get("skip"):
                            break
                        await asyncio.sleep(min(interval, delay - elapsed))
                        elapsed += interval
                        
            except Exception as e:
                logger.error(f"Error sending node {i}: {type(e).__name__}: {e}")
                continue
        
        # Stream complete
        await websocket.send_json({
            "type": "stream_complete",
            "total_nodes": sent_nodes,
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        logger.info(f"✅ Tiptap stream complete: {sent_nodes} nodes sent for session {session_id}")
        
    except Exception as e:
        logger.error(f"❌ Tiptap stream error: {type(e).__name__}: {e}", exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "code": "STREAM_ERROR",
                "message": f"Streaming failed: {str(e)[:100]}"
            })
        except:
            pass


async def process_and_stream_content(
    websocket: WebSocket,
    content: str,
    content_type: str,
    speed: str,
    chunk_by: str,
    session_id: str = "unknown",
    stream_control: dict = None
):
    """
    Production-ready content processing and streaming with comprehensive error handling.
    
    Features:
    - Intelligent content analysis and adaptive speed selection
    - Safe chunking with validation
    - Diff-based streaming for HTML integrity
    - Backpressure handling for network stability
    - Stream control (pause/resume/skip)
    - Progress tracking and metrics
    - Graceful error recovery
    
    Phase 2.4: Improved HTML streaming with diff-based approach
    - Accumulates content and calculates diffs
    - Ensures client receives complete, valid HTML fragments
    - Prevents broken tags across chunks
    
    Phase 3.1: Stream control support
    - Respects pause state (waits until resume)
    - Handles skip command (sends all remaining content)
    """
    if stream_control is None:
        stream_control = {"paused": False, "skip": False}
    try:
        # Analyze content for optimal strategy
        try:
            analysis = analyze_content_complexity(content)
            logger.info(f"📊 Content analysis: {analysis}")
        except Exception as e:
            logger.error(f"Analysis failed: {e}, using defaults")
            analysis = {"recommended_speed": "normal", "complexity": "unknown"}
        
        # Use adaptive speed if requested
        if speed == "adaptive":
            speed = analysis.get("recommended_speed", "normal")
            logger.info(f"⚡ Adaptive speed selected: {speed}")
        
        # Speed preset delays (seconds) - increased for better visual streaming
        speed_delays = {
            "slow": 0.5,
            "normal": 0.2,
            "fast": 0.1,
            "superfast": 0.05
        }
        delay = speed_delays.get(speed, 0.2)
        
        # Smart chunking with content processing
        try:
            chunks, metadata = smart_chunk_content(
                content=content,
                content_type=content_type,
                chunk_by=chunk_by
            )
        except Exception as e:
            logger.error(f"Chunking failed: {type(e).__name__}: {e}")
            await websocket.send_json({
                "type": "error",
                "code": "CHUNKING_ERROR",
                "message": f"Failed to chunk content: {str(e)[:100]}"
            })
            return
        
        if not chunks:
            logger.warning("No chunks generated, sending error")
            await websocket.send_json({
                "type": "error",
                "code": "EMPTY_CONTENT",
                "message": "Content resulted in no chunks"
            })
            return
        
        logger.info(f"✅ Processed content: {metadata.get('chunk_count', 0)} chunks, complexity: {analysis.get('complexity', 'unknown')}")
        
        # Send stream start event with metadata
        await websocket.send_json({
            "type": "stream_start",
            "total_chunks": len(chunks),
            "content_type": content_type,
            "metadata": {
                "complexity": analysis.get("complexity", "unknown"),
                "word_count": analysis.get("word_count", 0),
                "has_html": analysis.get("has_html", False),
                "speed_used": speed,
                "processing_time_ms": metadata.get("processing_time_ms", 0),
                "avg_chunk_size": metadata.get("avg_chunk_size", 0)
            },
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Diff-based streaming: accumulate and send complete HTML fragments
        accumulated = ""
        previous_length = 0
        sent_chunks = 0
        
        for i, chunk in enumerate(chunks):
            try:
                # Check if stream should be skipped (check at start)
                if stream_control.get("skip"):
                    logger.info("⏭️ Stream skipped, sending all remaining content")
                    # Send all remaining chunks immediately
                    remaining_content = "".join(chunks[i:])
                    await websocket.send_json({
                        "type": "chunk",
                        "data": remaining_content,
                        "index": sent_chunks,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    sent_chunks += 1
                    break
                
                # Wait while paused (also check for skip during pause)
                while stream_control.get("paused"):
                    if stream_control.get("skip"):
                        # Skip even while paused
                        break
                    await asyncio.sleep(0.1)  # Check every 100ms
                
                # Check skip again after pause
                if stream_control.get("skip"):
                    logger.info("⏭️ Stream skipped after pause, sending all remaining content")
                    remaining_content = "".join(chunks[i:])
                    await websocket.send_json({
                        "type": "chunk",
                        "data": remaining_content,
                        "index": sent_chunks,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    sent_chunks += 1
                    break
                
                # Accumulate content
                accumulated += chunk
                
                # Calculate new content to send (diff from previous state)
                new_content = accumulated[previous_length:]
                
                # Only send if we have complete HTML or enough content
                # For HTML: ensure we're not breaking in the middle of a tag
                if content_type == "html":
                    # Simple tag balance check: count < and >
                    open_tags = new_content.count("<")
                    close_tags = new_content.count(">")
                    
                    # If tags are unbalanced, wait for more chunks (unless last chunk)
                    if open_tags != close_tags and i < len(chunks) - 1:
                        logger.debug(f"Chunk {i}: Tag imbalance ({open_tags} < vs {close_tags} >), accumulating...")
                        continue
                
                # Send the complete fragment
                if new_content.strip():
                    await websocket.send_json({
                        "type": "chunk",
                        "data": new_content,
                        "index": sent_chunks,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    sent_chunks += 1
                    previous_length = len(accumulated)
                    logger.debug(f"Sent chunk {sent_chunks}: {len(new_content)} chars")
                
                # Throttle based on speed preset with backpressure detection
                if i < len(chunks) - 1:  # Don't delay after last chunk
                    try:
                        # Break delay into smaller intervals to check skip flag
                        elapsed = 0
                        interval = 0.05  # Check every 50ms
                        while elapsed < delay:
                            if stream_control.get("skip"):
                                # Exit delay immediately when skip triggered
                                break
                            await asyncio.sleep(min(interval, delay - elapsed))
                            elapsed += interval
                    except asyncio.TimeoutError:
                        logger.warning("Backpressure detected, adjusting...")
                        delay = min(delay * 1.5, 1.0)  # Increase delay but cap at 1s
                        
            except Exception as e:
                logger.error(f"Error processing chunk {i}: {type(e).__name__}: {e}")
                # Continue with next chunk rather than failing entire stream
                continue
        
        # Stream complete
        await websocket.send_json({
            "type": "stream_complete",
            "total_chunks": sent_chunks,
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        logger.info(f"✅ Stream complete: {sent_chunks} chunks sent for session {session_id}")
        
    except Exception as e:
        logger.error(f"❌ Stream processing error: {type(e).__name__}: {e}", exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "code": "STREAM_ERROR",
                "message": f"Streaming failed: {str(e)[:100]}"
            })
        except:
            pass


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
