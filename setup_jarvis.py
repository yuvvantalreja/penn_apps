#!/usr/bin/env python3
"""
Setup script for Jarvis Voice Assistant
Helps configure the system and test components
"""

import os
import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True


def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def setup_environment():
    """Setup environment configuration"""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists() and env_example.exists():
        print("🔧 Setting up environment configuration...")
        
        # Copy example to .env
        with open(env_example, 'r') as f:
            content = f.read()
        
        # Get API key from user
        print("\n🔑 Gemini API Key Setup:")
        print("1. Go to https://makersuite.google.com/app/apikey")
        print("2. Create a new API key")
        print("3. Copy the key and paste it below")
        
        api_key = input("\nEnter your Gemini API key (or press Enter to skip): ").strip()
        
        if api_key:
            content = content.replace("your_gemini_api_key_here", api_key)
            print("✅ API key configured")
        else:
            print("⚠️ API key skipped - you can add it later to .env file")
        
        with open(env_file, 'w') as f:
            f.write(content)
        
        print("✅ Environment file created")
    else:
        print("✅ Environment already configured")


def test_components():
    """Test system components"""
    print("\n🧪 Testing system components...")
    
    # Test imports
    try:
        import cv2
        print("✅ OpenCV available")
    except ImportError:
        print("❌ OpenCV not available")
        return False
    
    try:
        import mediapipe
        print("✅ MediaPipe available")
    except ImportError:
        print("❌ MediaPipe not available")
        return False
    
    try:
        import speech_recognition
        print("✅ SpeechRecognition available")
    except ImportError:
        print("❌ SpeechRecognition not available")
        return False
    
    try:
        import pyttsx3
        print("✅ pyttsx3 (TTS) available")
    except ImportError:
        print("❌ pyttsx3 (TTS) not available")
        return False
    
    # Test camera
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ Camera available")
            cap.release()
        else:
            print("⚠️ Camera not available - check permissions")
    except Exception as e:
        print(f"⚠️ Camera test failed: {e}")
    
    # Test microphone
    try:
        import speech_recognition as sr
        r = sr.Recognizer()
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, duration=0.5)
        print("✅ Microphone available")
    except Exception as e:
        print(f"⚠️ Microphone test failed: {e}")
    
    # Test TTS
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        if voices:
            print(f"✅ TTS available with {len(voices)} voices")
        else:
            print("⚠️ TTS available but no voices found")
        engine.stop()
    except Exception as e:
        print(f"⚠️ TTS test failed: {e}")
    
    return True


def main():
    """Main setup function"""
    print("🤖 Jarvis Voice Assistant Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Setup environment
    setup_environment()
    
    # Test components
    if not test_components():
        print("\n⚠️ Some components failed testing, but you can still try running the system")
    
    print("\n🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Make sure your camera and microphone permissions are enabled")
    print("2. Add your Gemini API key to the .env file if you haven't already")
    print("3. Run: python main.py")
    print("4. Press 'J' to activate Jarvis")
    print("5. Say 'what is this' while pointing at objects")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
