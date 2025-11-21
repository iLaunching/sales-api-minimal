"""
System Messages
Pre-written messages that are returned without LLM involvement
"""
import random
from typing import Dict, Any

# System message types
SYSTEM_MESSAGE_TYPES = {
    'SALES_WELCOME': '__SYSTEM_SALES_WELCOME__',
    'STAGE_TRANSITION': '__SYSTEM_STAGE_TRANSITION__',
    'FEATURE_INTRO': '__SYSTEM_FEATURE_INTRO__',
    'PROGRESS_UPDATE': '__SYSTEM_PROGRESS_UPDATE__',
}

# Welcome messages (match frontend)
SALES_WELCOME_MESSAGES = [
    """# Welcome! 🚀

I'm excited to help you **launch your business idea**. 

Let's start by understanding what you're building:

- What **problem** are you trying to solve?
- Who is your **target customer**?
- What makes your solution **unique**?

**Don't worry if you're not sure yet** - we'll figure it out together!

---

*Type your response below and let's get started...*""",

    """# Hey there! 👋

Great to have you here. I'm here to help turn your idea into reality.

Let's dive right in:

### First, tell me about your idea
- What inspired you to start this?
- What specific problem does it solve?
- Who needs this solution the most?

**No pressure** - just share what's on your mind, and we'll build from there!

---

*I'm listening...*""",

    """# Let's Build Something Amazing! ✨

I'm your AI sales consultant, and I'm here to guide you through launching your business.

### To get started, I need to understand:

1. **Your Vision** - What are you trying to create?
2. **The Problem** - What pain point does it address?
3. **Your Customers** - Who will benefit most?

**Think of this as a conversation** - there are no wrong answers!

---

*Share your thoughts below...*""",

    """# Ready to Launch? 🎯

Welcome! I'm here to help you validate, refine, and launch your business idea.

### Let's start with the basics:

- **What's your idea?** (In your own words)
- **Why now?** What makes this the right time?
- **Who's it for?** Your ideal customer

**Remember**: Every great business started with a simple conversation like this one.

---

*Let's begin...*""",

    """# Hi! Let's Talk Business 💡

I'm thrilled to work with you on bringing your idea to life.

### Here's what I need to know first:

> **Your Idea**: What are you planning to build or offer?
> 
> **The Gap**: What problem or need does it address?
> 
> **Your Advantage**: What makes you different from competitors?

**Feeling stuck?** No worries - just tell me what's on your mind, and we'll explore it together!

---

*Start typing below...*"""
]


def get_random_welcome_message() -> str:
    """Get a random welcome message"""
    return random.choice(SALES_WELCOME_MESSAGES)


def get_system_message_response(message_type: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Get system message response based on type
    Returns response in the same format as LLM responses
    """
    metadata = metadata or {}
    
    if message_type == SYSTEM_MESSAGE_TYPES['SALES_WELCOME']:
        return {
            "message": get_random_welcome_message(),
            "stage": "discovery",
            "extracted_data": {},
            "ui_commands": [
                {
                    "tool": "update_conversation_stage",
                    "arguments": {
                        "stage": "discovery",
                        "progress_percentage": 0
                    }
                }
            ]
        }
    
    # Add more system message types here as needed
    
    # Default fallback
    return {
        "message": "Hello! How can I help you today?",
        "stage": "discovery",
        "extracted_data": {},
        "ui_commands": []
    }
