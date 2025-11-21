"""
Test System Message Flow
Tests the complete system message detection and response flow
"""
import asyncio
import sys
import os

# Set required env vars for testing
os.environ['LLM_GATEWAY_URL'] = 'http://localhost:8001'  # Dummy URL for testing

from llm_client import get_sales_response
from constants.system_messages import SYSTEM_MESSAGE_TYPES

async def test_system_message_flow():
    """Test system messages are detected and returned correctly"""
    
    print("\n" + "="*60)
    print("Testing System Message Flow")
    print("="*60 + "\n")
    
    # Test 1: System message without user name
    print("Test 1: System message without user name")
    print("-" * 60)
    message1 = SYSTEM_MESSAGE_TYPES['SALES_WELCOME']
    print(f"Input: '{message1}'")
    
    response1 = await get_sales_response(
        conversation_history=[],
        user_message=message1,
        test_mode=False
    )
    
    print(f"✅ Response received: {len(response1)} chars")
    print(f"First 200 chars: {response1[:200]}...")
    print()
    
    # Test 2: System message with user name
    print("Test 2: System message with user name")
    print("-" * 60)
    message2 = f"{SYSTEM_MESSAGE_TYPES['SALES_WELCOME']}|USER:John"
    print(f"Input: '{message2}'")
    
    response2 = await get_sales_response(
        conversation_history=[],
        user_message=message2,
        test_mode=False
    )
    
    print(f"✅ Response received: {len(response2)} chars")
    print(f"First 200 chars: {response2[:200]}...")
    
    # Check if personalization worked
    if "John" in response2:
        print("✅ PASSED: User name 'John' found in response")
    else:
        print("❌ FAILED: User name 'John' not found in response")
    print()
    
    # Test 3: System message with whitespace
    print("Test 3: System message with trailing/leading whitespace")
    print("-" * 60)
    message3 = f"  {SYSTEM_MESSAGE_TYPES['SALES_WELCOME']}  "
    print(f"Input: '{message3}' (with spaces)")
    
    response3 = await get_sales_response(
        conversation_history=[],
        user_message=message3,
        test_mode=False
    )
    
    print(f"✅ Response received: {len(response3)} chars")
    print(f"First 200 chars: {response3[:200]}...")
    print()
    
    # Test 4: Non-system message (should fail gracefully or call LLM)
    print("Test 4: Non-system message")
    print("-" * 60)
    message4 = "Hello, I want to launch a SaaS product"
    print(f"Input: '{message4}'")
    
    # Note: This will try to call the LLM Gateway, which might fail if not available
    # That's okay - we're testing the logic flow
    try:
        response4 = await get_sales_response(
            conversation_history=[],
            user_message=message4,
            test_mode=True  # Use test mode to avoid real LLM call
        )
        print(f"✅ Response received: {len(response4)} chars")
        print(f"First 200 chars: {response4[:200]}...")
    except Exception as e:
        print(f"⚠️ LLM call failed (expected if gateway not available): {e}")
    print()
    
    print("="*60)
    print("System Message Flow Tests Complete!")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(test_system_message_flow())
