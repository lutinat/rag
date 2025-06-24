#!/usr/bin/env python3
"""
Test script for the conversation history API functionality.
This demonstrates how to send conversation history to the RAG API.
"""

import requests
import json
import time

API_BASE_URL = "http://localhost:5000/api"

def test_conversation_api():
    """Test the chat-based conversation history functionality step by step."""
    
    print("🧪 Testing Chat-Based Conversation History API")
    print("=" * 60)
    
    # Generate unique chat IDs for testing
    chat_id_1 = f"test_chat_{int(time.time())}"
    chat_id_2 = f"test_chat_{int(time.time()) + 1}"
    
    # Test 1: Send a question to first chat without conversation history
    print(f"\n1️⃣ Testing question in chat {chat_id_1} without conversation history...")
    response1 = requests.post(f"{API_BASE_URL}/question", json={
        "question": "What is PSF?",
        "chat_id": chat_id_1
    })
    
    if response1.status_code == 200:
        result1 = response1.json()
        print(f"✅ Response: {result1.get('answer', 'No answer')[:100]}...")
        print(f"📊 Chat ID: {result1.get('chat_id')}")
        print(f"📊 Conversation history used: {result1.get('conversation_history_used', 0)} turns")
    else:
        print(f"❌ Error: {response1.status_code} - {response1.text}")
        return
    
    # Test 2: Send a follow-up question to the same chat with conversation history
    print(f"\n2️⃣ Testing follow-up question in same chat {chat_id_1}...")
    conversation_history = [
        {
            "user": "What is PSF?",
            "assistant": result1.get('answer', 'PSF explanation from previous response')
        }
    ]
    
    response2 = requests.post(f"{API_BASE_URL}/question", json={
        "question": "How does it affect image quality?",
        "chat_id": chat_id_1,
        "conversation_history": conversation_history
    })
    
    if response2.status_code == 200:
        result2 = response2.json()
        print(f"✅ Response: {result2.get('answer', 'No answer')[:100]}...")
        print(f"📊 Conversation history used: {result2.get('conversation_history_used', 0)} turns")
    else:
        print(f"❌ Error: {response2.status_code} - {response2.text}")
    
    # Test 3: Start a new chat session with the same user
    print(f"\n3️⃣ Testing new chat session {chat_id_2} for same user...")
    response3 = requests.post(f"{API_BASE_URL}/question", json={
        "question": "What is GSD?",
        "chat_id": chat_id_2
    })
    
    if response3.status_code == 200:
        result3 = response3.json()
        print(f"✅ Response: {result3.get('answer', 'No answer')[:100]}...")
        print(f"📊 Chat ID: {result3.get('chat_id')}")
        print(f"📊 New chat has {result3.get('conversation_history_used', 0)} turns (should be 0)")
    else:
        print(f"❌ Error: {response3.status_code} - {response3.text}")
    
    # Test 4: Get chat history from first chat (wait a moment for async conversation storage)
    print(f"\n4️⃣ Testing chat history retrieval for {chat_id_1}...")
    time.sleep(0.5)  # Brief wait for async conversation history to be stored
    response4 = requests.get(f"{API_BASE_URL}/chat/history", params={
        "chat_id": chat_id_1,
        "max_turns": 10
    })
    
    if response4.status_code == 200:
        history = response4.json()
        print(f"✅ Retrieved {history.get('total_turns', 0)} conversation turns")
        print(f"📊 Chat info: {history.get('chat_info')}")
        for i, turn in enumerate(history.get('history', []), 1):
            print(f"   Turn {i}:")
            print(f"     User: {turn.get('user', '')[:50]}...")
            print(f"     Assistant: {turn.get('assistant', '')[:50]}...")
    else:
        print(f"❌ Error: {response4.status_code} - {response4.text}")
    
    # Test 5: Test validation (missing chat_id)
    print("\n5️⃣ Testing validation (missing chat_id)...")
    response5 = requests.post(f"{API_BASE_URL}/question", json={
        "question": "Test question"
        # Missing chat_id
    })
    
    if response5.status_code == 400:
        error = response5.json()
        print(f"✅ Validation works: {error.get('error', 'Unknown error')}")
    else:
        print(f"❌ Validation failed: Expected 400, got {response5.status_code}")
    
    # Test 7: Clear first chat history
    print(f"\n7️⃣ Testing chat history clearing for {chat_id_1}...")
    response7 = requests.post(f"{API_BASE_URL}/chat/clear", json={
        "chat_id": chat_id_1
    })
    
    if response7.status_code == 200:
        result7 = response7.json()
        print(f"✅ {result7.get('message', 'History cleared')}")
    else:
        print(f"❌ Error: {response7.status_code} - {response7.text}")
    
    # Test 8: Verify first chat was cleared but second chat remains
    print(f"\n8️⃣ Verifying chat {chat_id_1} was cleared...")
    response8 = requests.get(f"{API_BASE_URL}/chat/history", params={
        "chat_id": chat_id_1
    })
    
    if response8.status_code == 200:
        history = response8.json()
        turns = history.get('total_turns', 0)
        if turns == 0:
            print("✅ First chat successfully cleared")
        else:
            print(f"❌ First chat not cleared: still has {turns} turns")
    else:
        print(f"❌ Error: {response8.status_code} - {response8.text}")
    
    # Test 9: Verify second chat still exists
    print(f"\n9️⃣ Verifying chat {chat_id_2} still exists...")
    response9 = requests.get(f"{API_BASE_URL}/chat/history", params={
        "chat_id": chat_id_2
    })
    
    if response9.status_code == 200:
        history = response9.json()
        turns = history.get('total_turns', 0)
        if turns > 0:
            print(f"✅ Second chat still exists with {turns} turns")
        else:
            print("❌ Second chat was unexpectedly cleared")
    else:
        print(f"❌ Error: {response9.status_code} - {response9.text}")
    
    # Cleanup: Clear second chat
    print(f"\n🧹 Cleaning up chat {chat_id_2}...")
    requests.post(f"{API_BASE_URL}/chat/clear", json={"chat_id": chat_id_2})

def test_api_availability():
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ API is running and healthy")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to API: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Conversation History API Test")
    print("Make sure the API server is running on localhost:5000")
    print()
    
    if test_api_availability():
        test_conversation_api()
    else:
        print("\n💡 To start the API server, run:")
        print("   python api.py") 