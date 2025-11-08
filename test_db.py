import os
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine

async def test():
    # Simulate Railway's DATABASE_URL
    url = "postgresql://postgres:password@postgres-sales.railway.internal:5432/railway"
    
    # Convert to asyncpg
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    
    print(f"Testing connection to: {url.split('@')[0]}@...")
    
    try:
        engine = create_async_engine(url, echo=True)
        async with engine.begin() as conn:
            result = await conn.execute("SELECT 1")
            print("✅ Connection successful!")
    except Exception as e:
        print(f"❌ Connection failed: {e}")

asyncio.run(test())
