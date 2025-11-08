# Sales API - Minimal + PostgreSQL + Redis

Simple FastAPI server with PostgreSQL for conversations and Redis for caching.

## What This Has

✅ FastAPI server
✅ PostgreSQL database
✅ Redis caching
✅ Conversation storage
✅ Message history
✅ Session caching (30min TTL)
✅ Health check at `/health`
✅ CRUD endpoints for conversations

## Environment Variables

```bash
DATABASE_URL=postgresql://user:pass@host:port/db
REDIS_URL=redis://host:port
```

## Endpoints

- `GET /health` - Health check
- `GET /` - Service info
- `POST /api/sales/conversations` - Create conversation
- `GET /api/sales/conversations/{session_id}` - Get conversation (cached)
- `POST /api/sales/message` - Send message (invalidates cache)

## Next Steps

- ✅ PostgreSQL (DONE)
- ✅ Redis (DONE)
- ⏳ AI responses with OpenAI
- ⏳ MCP tools integration
- ⏳ Qdrant vector search
