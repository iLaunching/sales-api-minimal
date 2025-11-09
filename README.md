# Sales API - Complete System 🚀

Full-stack FastAPI server with PostgreSQL, Redis, AI, MCP tools, and vector search.

## Features

✅ **PostgreSQL** - Conversation storage
✅ **Redis** - Session caching (30min TTL)
✅ **AI (LLM Gateway)** - GPT-4o-mini smart responses
✅ **MCP Sales Tools** - Objections, pitches, ROI calculation
✅ **Qdrant** - Vector search for sales knowledge
✅ **Full conversation history**
✅ **Cache invalidation**
✅ **Sales-focused AI agent**

## Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:pass@host:port/db
REDIS_URL=redis://host:port

# AI & Tools
LLM_GATEWAY_URL=https://your-llm-gateway.railway.app
MCP_SERVER_URL=https://your-mcp-server.railway.app

# Vector Search
QDRANT_URL=https://your-qdrant.railway.app
QDRANT_API_KEY=your-api-key  # Optional

# LLM Config (Optional)
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=500
```

## API Endpoints

**System:**
- `GET /health` - Health check with Qdrant stats
- `GET /` - Service info

**Conversations:**
- `POST /api/sales/conversations` - Create conversation
- `GET /api/sales/conversations/{session_id}` - Get conversation (cached)
- `POST /api/sales/message` - Send message with AI

**MCP Tools:**
- `POST /api/mcp/objection` - Handle objection
- `POST /api/mcp/pitch` - Get pitch template
- `POST /api/mcp/value` - Calculate ROI

## Architecture

- **Version:** 1.5.0
- **Stack:** FastAPI + PostgreSQL + Redis + Qdrant
- **AI:** LLM Gateway (multi-model)
- **Tools:** MCP Sales Server
- **Deployment:** Railway

## Status

✅ All systems operational
