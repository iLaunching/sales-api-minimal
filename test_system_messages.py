"""
Test System Messages
Verify that system messages are returned without LLM involvement
"""
from constants.system_messages import (
    SYSTEM_MESSAGE_TYPES,
    get_random_welcome_message,
    get_system_message_response
)


def test_get_random_welcome_message():
    """Test that random welcome message is returned"""
    message = get_random_welcome_message()
    assert isinstance(message, str)
    assert len(message) > 0
    # Check that it contains common welcome elements
    assert any(word in message for word in ["Welcome", "Hey", "Ready", "Hi", "Let's"])


def test_get_system_message_response_welcome():
    """Test welcome message response format"""
    response = get_system_message_response(SYSTEM_MESSAGE_TYPES['SALES_WELCOME'])
    
    assert 'message' in response
    assert 'stage' in response
    assert 'extracted_data' in response
    assert 'ui_commands' in response
    
    assert response['stage'] == 'discovery'
    assert isinstance(response['ui_commands'], list)
    assert len(response['ui_commands']) > 0
    assert response['ui_commands'][0]['tool'] == 'update_conversation_stage'


def test_system_message_types_constant():
    """Test that system message types are defined"""
    assert SYSTEM_MESSAGE_TYPES['SALES_WELCOME'] == '__SYSTEM_SALES_WELCOME__'
    assert SYSTEM_MESSAGE_TYPES['STAGE_TRANSITION'] == '__SYSTEM_STAGE_TRANSITION__'
    assert SYSTEM_MESSAGE_TYPES['FEATURE_INTRO'] == '__SYSTEM_FEATURE_INTRO__'
    assert SYSTEM_MESSAGE_TYPES['PROGRESS_UPDATE'] == '__SYSTEM_PROGRESS_UPDATE__'


def test_all_welcome_messages_have_content():
    """Test that all welcome messages are properly formatted"""
    from constants.system_messages import SALES_WELCOME_MESSAGES
    
    assert len(SALES_WELCOME_MESSAGES) >= 3, "Should have at least 3 welcome messages"
    
    for i, message in enumerate(SALES_WELCOME_MESSAGES):
        assert isinstance(message, str), f"Message {i} should be a string"
        assert len(message) > 50, f"Message {i} should be substantial"
        assert "#" in message, f"Message {i} should have markdown headers"
        assert "**" in message or "*" in message, f"Message {i} should have markdown emphasis"


def test_get_system_message_response_fallback():
    """Test fallback for unknown system message types"""
    response = get_system_message_response("__UNKNOWN_MESSAGE__")
    
    assert 'message' in response
    assert response['message'] == "Hello! How can I help you today?"
    assert response['stage'] == 'discovery'
    assert isinstance(response['ui_commands'], list)


if __name__ == "__main__":
    # Run tests
    print("Running system message tests...")
    test_get_random_welcome_message()
    print("✅ test_get_random_welcome_message passed")
    
    test_get_system_message_response_welcome()
    print("✅ test_get_system_message_response_welcome passed")
    
    test_system_message_types_constant()
    print("✅ test_system_message_types_constant passed")
    
    test_all_welcome_messages_have_content()
    print("✅ test_all_welcome_messages_have_content passed")
    
    test_get_system_message_response_fallback()
    print("✅ test_get_system_message_response_fallback passed")
    
    print("\n🎉 All tests passed!")
