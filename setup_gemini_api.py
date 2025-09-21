#!/usr/bin/env python3
"""
Setup script for Gemini API key configuration
"""

import os
import sys

def setup_gemini_api_key():
    """Interactive setup for Gemini API key"""
    print("🔑 Gemini API Key Setup")
    print("=" * 50)
    print()
    print("To use JARVIS with vision capabilities, you need a Gemini API key.")
    print("Get your free API key from: https://makersuite.google.com/app/apikey")
    print()
    
    # Check if already set
    existing_key = os.getenv('GEMINI_API_KEY')
    if existing_key:
        print(f"✅ GEMINI_API_KEY is already set: {existing_key[:10]}...")
        choice = input("Do you want to update it? (y/n): ").lower().strip()
        if choice != 'y':
            print("Keeping existing API key.")
            return True
    
    # Get new API key
    print("Enter your Gemini API key:")
    api_key = input("API Key: ").strip()
    
    if not api_key:
        print("❌ No API key provided. Exiting.")
        return False
    
    # Validate key format (basic check)
    if not api_key.startswith('AIza'):
        print("⚠️  Warning: Gemini API keys usually start with 'AIza'. Please verify your key.")
        confirm = input("Continue anyway? (y/n): ").lower().strip()
        if confirm != 'y':
            return False
    
    # Set environment variable for current session
    os.environ['GEMINI_API_KEY'] = api_key
    
    # Add to shell profile
    shell_profile = os.path.expanduser('~/.zshrc')  # Default to zsh since you're using it
    if not os.path.exists(shell_profile):
        shell_profile = os.path.expanduser('~/.bash_profile')
    
    print(f"\n📝 Adding API key to {shell_profile}")
    
    # Check if already in profile
    try:
        with open(shell_profile, 'r') as f:
            content = f.read()
            if 'GEMINI_API_KEY' in content:
                print("✅ API key already exists in shell profile")
                return True
    except FileNotFoundError:
        print(f"⚠️  Shell profile not found: {shell_profile}")
        return True
    
    # Add to profile
    try:
        with open(shell_profile, 'a') as f:
            f.write(f'\n# Gemini API Key for JARVIS\n')
            f.write(f'export GEMINI_API_KEY="{api_key}"\n')
        print("✅ API key added to shell profile")
        print("🔄 Please run 'source ~/.zshrc' or restart your terminal to apply changes")
    except Exception as e:
        print(f"❌ Error writing to shell profile: {e}")
        print("You can manually add this line to your shell profile:")
        print(f'export GEMINI_API_KEY="{api_key}"')
        return True
    
    return True

def test_api_key():
    """Test the API key by importing and initializing the vision API"""
    try:
        from gemini_vision_api import GeminiVisionAPI
        print("\n🧪 Testing API key...")
        vision_api = GeminiVisionAPI()
        health = vision_api.health_check()
        
        if health['status'] == 'healthy':
            print("✅ API key is working correctly!")
            print(f"Test response: {health.get('test_response', 'N/A')}")
            return True
        else:
            print(f"❌ API key test failed: {health.get('message', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing API key: {e}")
        return False

if __name__ == "__main__":
    print("🤖 JARVIS Gemini API Setup")
    print("=" * 50)
    
    if setup_gemini_api_key():
        if test_api_key():
            print("\n🎉 Setup complete! JARVIS is ready to use.")
            print("You can now run: python3 jarvis_backend.py")
        else:
            print("\n❌ Setup failed. Please check your API key and try again.")
            sys.exit(1)
    else:
        print("\n❌ Setup cancelled.")
        sys.exit(1)
