"""
LLM Gateway Client - Calls the LLM Gateway for AI responses
"""

import httpx
import os
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# LLM Gateway URL from environment
LLM_GATEWAY_URL = os.getenv("LLM_GATEWAY_URL")

if not LLM_GATEWAY_URL:
    logger.error("LLM_GATEWAY_URL environment variable not set!")
    raise ValueError("LLM_GATEWAY_URL is required")

# Model configuration from environment
DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
DEFAULT_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "500"))

# Sales-specific system prompt
SALES_SYSTEM_PROMPT = """You are an expert B2B sales assistant specializing in the iLaunching platform. 

Your role is to:
- Qualify leads by understanding their business needs, pain points, and goals
- Educate prospects about iLaunching's AI-powered platform features
- Handle objections professionally and provide value-focused responses
- Guide conversations toward scheduling demos for high-quality leads
- Ask discovery questions to understand: company size, industry, current tools, budget authority

CRITICAL OUTPUT FORMATTING RULES:
- **NEVER use HTML tags** - NO <p>, <strong>, <em>, <ul>, <li>, <div>, etc.
- **ONLY use Markdown syntax** - plain text with markdown formatting
- Use blank lines to separate paragraphs (NOT <p> tags)
- Use ## and ### for headings when appropriate  
- Use - or * for bullet lists of benefits or features
- Use **bold** for emphasis on key points (NOT <strong> tags)
- Use *italic* for subtle emphasis or questions (NOT <em> tags)
- Include emojis where appropriate to add personality
- Output PLAIN TEXT with Markdown formatting ONLY

Platform capabilities you can discuss:
- AI-powered competitor analysis and market research
- Automated sales intelligence and lead qualification
- Real-time data enrichment and verification
- Integrated CRM and workflow automation
- Custom AI agents for specific sales tasks

Conversation stages: greeting → discovery → qualification → education → objection_handling → closing

Example response formats (PLAIN TEXT MARKDOWN ONLY):
- Greeting: "Hi there! Welcome to iLaunching! 👋\n\nI'm excited to learn about your business. **What industry are you in**, and what's the biggest challenge you're facing right now?"
- Discovery: "That sounds challenging! I can definitely see how that would impact your growth.\n\n### Quick question:\n\nHave you tried any solutions for this before? I'd love to understand what worked (or didn't work) for you."
- Qualification: "Based on what you've shared, this sounds exactly like what we help businesses solve.\n\n### 🎯 Here's what I'm thinking:\n\n- **Market validation** - We can help you understand your competition in weeks, not months\n- **Strategic insights** - AI-powered analysis that adapts to your specific market\n\n*What's your timeline for getting this resolved?*"

REMEMBER: OUTPUT RAW MARKDOWN TEXT ONLY. NO HTML TAGS WHATSOEVER.

Keep responses concise, professional, and focused on understanding their needs before pitching features.
"""


async def get_ai_response(
    messages: List[Dict[str, str]],
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: int = 500
) -> Optional[str]:
    """
    Get AI response from LLM Gateway
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        model: Model name (gpt-4o-mini, claude-3-5-sonnet-20241022, etc)
        temperature: Sampling temperature
        max_tokens: Max response tokens
    
    Returns:
        AI response text or None if failed
    """
    try:
        # Prepend system prompt
        full_messages = [
            {"role": "system", "content": SALES_SYSTEM_PROMPT}
        ] + messages
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{LLM_GATEWAY_URL}/generate",
                json={
                    "model": model,
                    "messages": full_messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                ai_message = data.get("content", "")
                logger.info(f"LLM response received: {len(ai_message)} chars")
                return ai_message
            else:
                logger.error(f"LLM Gateway error: {response.status_code} - {response.text[:200]}")
                return None
                
    except httpx.TimeoutException:
        logger.error(f"LLM Gateway timeout after 60s - URL: {LLM_GATEWAY_URL}")
        return None
    except httpx.ConnectError as e:
        logger.error(f"LLM Gateway connection error - URL: {LLM_GATEWAY_URL} - Error: {e}")
        return None
    except Exception as e:
        logger.error(f"LLM Gateway request failed - URL: {LLM_GATEWAY_URL} - Error: {type(e).__name__}: {e}")
        return None


async def get_sales_response(
    conversation_history: List[Dict[str, str]],
    user_message: str,
    context: Optional[Dict] = None
) -> str:
    """
    Get contextual sales response based on conversation history
    
    Args:
        conversation_history: Previous messages in conversation
        user_message: Latest user message
        context: Optional context (company, email, stage, etc)
    
    Returns:
        AI-generated sales response
    """
    # Build message list from history
    messages = []
    for msg in conversation_history:
        if msg.get("role") in ["user", "assistant"]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
    
    # Add current user message
    messages.append({
        "role": "user",
        "content": user_message
    })
    
    # Get AI response with configured defaults
    response = await get_ai_response(
        messages=messages,
        model=DEFAULT_MODEL,
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=DEFAULT_MAX_TOKENS
    )
    
    # Fallback if LLM fails
    if not response:
        logger.warning("LLM failed, using fallback response")
        return "Thank you for your message. I'm having trouble connecting right now. Could you please tell me more about what you're looking for, and I'll get back to you shortly?"
    
    return response
