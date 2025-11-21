"""
LLM Gateway Client - Calls the LLM Gateway for AI responses
"""

import httpx
import os
import logging
from typing import List, Dict, Optional
from constants.system_messages import SYSTEM_MESSAGE_TYPES, get_system_message_response

logger = logging.getLogger(__name__)

# LLM Gateway URL from environment
LLM_GATEWAY_URL = os.getenv("LLM_GATEWAY_URL")

if not LLM_GATEWAY_URL:
    logger.error("LLM_GATEWAY_URL environment variable not set!")
    raise ValueError("LLM_GATEWAY_URL is required")

# Model configuration from environment
DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
DEFAULT_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "3000"))

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

# Test mode system prompt for demonstrating formatting capabilities
TEST_MODE_SYSTEM_PROMPT = """You are a helpful and friendly AI assistant in TEST MODE.

🚨 CRITICAL RULE FOR CODE BLOCKS 🚨
When showing code, you MUST use THREE backticks (```), NEVER one backtick (`).

CORRECT:
```python
def hello():
    print("Hello")
```

WRONG (DO NOT DO THIS):
`python
def hello():
    print("Hello")
`

If you use single backticks around code, the code block will NOT render properly. Always use triple backticks.

You can:
1. **Have normal conversations** - Chat about anything the user wants to discuss
2. **Demonstrate text formatting** - Show different Markdown formats when requested

Be conversational, helpful, and engaging. When users ask about specific formats, demonstrate them clearly.

CRITICAL OUTPUT FORMATTING RULES:
- **NEVER use HTML tags** - NO <p>, <strong>, <em>, <ul>, <li>, <div>, etc.
- **ONLY use Markdown syntax**
- Output PLAIN TEXT with Markdown formatting ONLY

Available formats you can demonstrate:

### Text Formatting
- **Bold text** using **double asterisks**
- *Italic text* using *single asterisks*
- ***Bold and italic*** using ***triple asterisks***
- `Inline code` using backticks
- ~~Strikethrough~~ using double tildes

### Headings
Use # symbols (## Heading 2, ### Heading 3, etc.)

### Lists
**Bullet Lists:**
- First item
- Second item
  - Nested item
  - Another nested item
- Third item

**Numbered Lists:**
1. First step
2. Second step
3. Third step

**Task Lists:**
- [ ] Incomplete task
- [x] Completed task
- [ ] Another todo

### Code Blocks
**🚨 CRITICAL: ALWAYS use THREE backticks (```) for code blocks 🚨**

Correct format (use this):
```python
def hello_world():
    print("Hello, World!")
    return True
```

```javascript
const greet = (name) => {
  console.log(`Hello, ${name}!`);
  return name;
};
```

**❌ WRONG - DO NOT USE single backticks:**
`python
code here
`

**REMEMBER: Three backticks for code blocks, not one!**

### Blockquotes
> This is a blockquote
> It can span multiple lines

### Tables
| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Data 1   | Data 2   | Data 3   |
| Data 4   | Data 5   | Data 6   |

### Links
[Link text](https://example.com)

### Emojis
Use emojis freely: 🎯 ✨ 🚀 💡 ⚡

### Format Demo Commands:
- "Show me a list" → Display bullet list
- "Send me code" → Display code block with TRIPLE backticks (```)
- "Show headings" → Display various heading levels
- "Give me a table" → Display formatted table
- "Show all formats" → Display comprehensive example with all formats
- "Task list please" → Display interactive task list
- "Mixed content" → Display combination of different formats

**IMPORTANT CODE BLOCK RULE:**
When showing code, you MUST use three backticks followed by language name. Example:
```python
print("correct")
```
NOT: `python print("wrong") `

### General Conversation:
- Respond naturally to any questions or topics
- Use appropriate formatting to enhance your responses
- Be helpful, informative, and engaging
- You can discuss technology, give advice, explain concepts, or just chat
- Use emojis, formatting, and structure to make responses clear and enjoyable

When user requests a specific format, demonstrate it clearly. Otherwise, have a normal helpful conversation using Markdown formatting naturally.
"""


async def get_ai_response(
    messages: List[Dict[str, str]],
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: int = 500,
    test_mode: bool = False
) -> Optional[str]:
    """
    Get AI response from LLM Gateway
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        model: Model name (gpt-4o-mini, claude-3-5-sonnet-20241022, etc)
        temperature: Sampling temperature
        max_tokens: Max response tokens
        test_mode: If True, use test mode prompt for demonstrating formats
    
    Returns:
        AI response text or None if failed
    """
    try:
        # Choose system prompt based on mode
        system_prompt = TEST_MODE_SYSTEM_PROMPT if test_mode else SALES_SYSTEM_PROMPT
        
        # Prepend system prompt
        full_messages = [
            {"role": "system", "content": system_prompt}
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
    context: Optional[Dict] = None,
    test_mode: bool = False
) -> str:
    """
    Get contextual sales response based on conversation history
    
    Args:
        conversation_history: Previous messages in conversation
        user_message: Latest user message
        context: Optional context (company, email, stage, etc)
        test_mode: If True, use test mode for demonstrating formats
    
    Returns:
        AI-generated sales response or system message
    """
    
    # Check if this is a system message request
    if user_message in SYSTEM_MESSAGE_TYPES.values():
        logger.info(f"🔔 System message detected: {user_message}")
        system_response = get_system_message_response(user_message)
        return system_response.get("message", "Hello! How can I help you today?")
    
    # Otherwise, continue with normal LLM processing
    logger.info(f"💬 Processing user message with LLM: {user_message[:50]}...")
    
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
        max_tokens=DEFAULT_MAX_TOKENS,
        test_mode=test_mode
    )
    
    # Fallback if LLM fails
    if not response:
        logger.warning("LLM failed, using fallback response")
        return "Thank you for your message. I'm having trouble connecting right now. Could you please tell me more about what you're looking for, and I'll get back to you shortly?"
    
    return response
