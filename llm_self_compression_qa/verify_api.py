#!/usr/bin/env python3
"""
Verification script to test OpenAI API setup and detect available API version.

Run this FIRST before running any experiments to verify:
1. API key is configured correctly
2. Which API version is available (Responses or Chat Completions)
3. Which models are accessible
4. Basic generation works

Usage:
    python verify_api.py
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("=" * 70)
print("OpenAI API Verification Script")
print("=" * 70)
print()

# Check for API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ ERROR: OPENAI_API_KEY not found in environment")
    print()
    print("Please set your API key:")
    print("1. Copy .env.example to .env")
    print("2. Add your API key: OPENAI_API_KEY=sk-...")
    print()
    sys.exit(1)

print(f"✅ API key found: {api_key[:15]}...{api_key[-4:]}")
print()

# Try importing OpenAI
try:
    from openai import OpenAI
    print("✅ OpenAI library imported successfully")
except ImportError as e:
    print(f"❌ ERROR: Failed to import OpenAI library: {e}")
    print()
    print("Please install dependencies:")
    print("  pip install -e .")
    print()
    sys.exit(1)

# Initialize client
try:
    client = OpenAI(api_key=api_key)
    print("✅ OpenAI client initialized")
except Exception as e:
    print(f"❌ ERROR: Failed to initialize client: {e}")
    sys.exit(1)

print()
print("-" * 70)
print("Testing API Detection")
print("-" * 70)
print()

# Test 1: Check if Responses API exists
has_responses_api = hasattr(client, "responses")
print(f"Responses API available: {'✅ YES' if has_responses_api else '❌ NO'}")

# Test 2: Try Responses API if available
if has_responses_api:
    print()
    print("Attempting Responses API call with gpt-5.1-chat-latest...")
    try:
        response = client.responses.create(
            model="gpt-5.1-chat-latest",
            input="Say 'Hello from GPT-5.1!' if you can read this.",
            temperature=0.0,
            seed=42,
            reasoning_effort="none",
        )

        # Try to extract text
        if hasattr(response, "output_text"):
            text = response.output_text
            print(f"✅ Responses API works!")
            print(f"   Response: {text}")
            print(f"   API type: Responses API (March 2025+)")
        elif hasattr(response, "choices"):
            text = response.choices[0].message.content
            print(f"✅ Responses API works (with Chat Completions format)!")
            print(f"   Response: {text}")
        else:
            print(f"⚠️  Unexpected response format: {response}")

    except Exception as e:
        print(f"❌ Responses API failed: {e}")
        print(f"   This is normal if you don't have access to GPT-5.1 yet")
        has_responses_api = False

# Test 3: Try Chat Completions API as fallback
print()
print("Attempting Chat Completions API call with gpt-4o...")
try:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Say 'Hello from GPT-4o!' if you can read this."}],
        temperature=0.0,
        seed=42,
    )

    text = response.choices[0].message.content
    print(f"✅ Chat Completions API works!")
    print(f"   Response: {text}")
    print(f"   API type: Chat Completions API (stable)")

except Exception as e:
    print(f"❌ Chat Completions API failed: {e}")
    print()
    print("ERROR: No working API found. Please check:")
    print("  1. API key is valid")
    print("  2. Account has API access")
    print("  3. Network connection")
    sys.exit(1)

# Test 4: Check tiktoken
print()
print("-" * 70)
print("Testing Token Counting")
print("-" * 70)
print()

try:
    import tiktoken
    print("✅ tiktoken library imported")

    # Test encoding
    try:
        encoding = tiktoken.encoding_for_model("gpt-4o")
        print("✅ Got encoding for gpt-4o")
    except KeyError:
        encoding = tiktoken.get_encoding("o200k_base")
        print("✅ Using o200k_base encoding (GPT-4o/5 fallback)")

    # Count tokens in test text
    test_text = "This is a test sentence for token counting."
    tokens = encoding.encode(test_text)
    print(f"✅ Token counting works: '{test_text}' = {len(tokens)} tokens")

except ImportError as e:
    print(f"❌ tiktoken import failed: {e}")
    print("   Token counting may not work correctly")

# Summary
print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()

if has_responses_api:
    print("✅ Recommended configuration:")
    print("   - Use Responses API")
    print("   - Model: gpt-5.1-chat-latest or gpt-5.1-thinking")
    print("   - The code will use this automatically")
else:
    print("✅ Recommended configuration:")
    print("   - Use Chat Completions API")
    print("   - Model: gpt-4o or gpt-3.5-turbo")
    print("   - The code will automatically fall back to this")

print()
print("You can now proceed with experiments:")
print("  python scripts/pilot.py")
print()
