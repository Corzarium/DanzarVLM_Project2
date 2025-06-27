#!/usr/bin/env python3
"""
Simple test for greeting detection logic
"""

def test_greeting_detection():
    """Test the greeting detection logic directly"""
    
    # Test cases
    test_cases = [
        ("Hello Danza", "greeting"),
        ("Hi there", "greeting"), 
        ("Hey buddy", "greeting"),
        ("Good morning", "greeting"),
        ("Good afternoon", "greeting"),
        ("Good evening", "greeting"),
        ("Greetings", "greeting"),
        ("What's the weather like?", "non-greeting"),
        ("Tell me about EverQuest", "non-greeting"),
        ("How do I play this game?", "non-greeting"),
        ("Search for something", "non-greeting"),
        ("Hello, how are you?", "greeting"),  # Contains "hello" at start
        ("Hi, what's up?", "greeting"),       # Contains "hi" at start
        ("How are you?", "non-greeting"),     # Contains "how" but not "hi"
        ("What is this?", "non-greeting"),    # Should not be detected as greeting
    ]
    
    print("Testing improved greeting detection logic...")
    print("=" * 50)
    
    # The improved greeting detection logic from the LLM service
    simple_greetings = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"]
    
    passed = 0
    total = len(test_cases)
    
    for text, expected_type in test_cases:
        print(f"\nTesting: '{text}' (expected: {expected_type})")
        
        # Apply the improved logic as in the LLM service
        text_lower = text.lower().strip()
        words = text_lower.split()
        
        # More precise greeting detection - check for exact word matches or phrases
        is_greeting = False
        
        # Check for single-word greetings
        if len(words) == 1 and words[0] in simple_greetings:
            is_greeting = True
        # Check for greetings at the start of sentences
        elif len(words) > 1 and words[0] in ["hello", "hi", "hey", "greetings"]:
            is_greeting = True
        # Check for "good morning/afternoon/evening" phrases
        elif len(words) >= 2 and f"{words[0]} {words[1]}" in ["good morning", "good afternoon", "good evening"]:
            is_greeting = True
        
        if is_greeting:
            print(f"✅ Detected as greeting - would respond directly")
        else:
            print(f"❌ Not detected as greeting - would search RAG/web")
        
        # Check if our detection matches expectation
        if (is_greeting and expected_type == "greeting") or (not is_greeting and expected_type == "non-greeting"):
            print(f"✅ Test PASSED")
            passed += 1
        else:
            print(f"❌ Test FAILED")
    
    print(f"\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Greeting detection is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the logic.")

if __name__ == "__main__":
    test_greeting_detection() 