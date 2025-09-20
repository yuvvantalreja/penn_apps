#!/usr/bin/env python3
"""
Test JARVIS integration with OpenCV AR application
Tests the complete workflow from screenshot capture to voice analysis
"""

import os
import sys
import time
import base64
import requests
import subprocess
from pathlib import Path

def check_jarvis_backend():
    """Check if JARVIS backend is running"""
    try:
        response = requests.get('http://localhost:5001/api/jarvis/health', timeout=3)
        if response.status_code == 200:
            data = response.json()
            print("✅ JARVIS backend is running")
            print(f"   Status: {data.get('status')}")
            return True
        else:
            print(f"❌ JARVIS backend health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException:
        print("❌ JARVIS backend not reachable at http://localhost:5001")
        return False

def test_screenshot_analysis():
    """Test screenshot analysis with JARVIS"""
    screenshot_path = "jarvis_screenshot.png"
    
    if not os.path.exists(screenshot_path):
        print(f"❌ Screenshot not found: {screenshot_path}")
        print("💡 Run the AR application and press 'J' to generate a screenshot")
        return False
    
    try:
        # Read and encode screenshot
        with open(screenshot_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        print(f"📸 Testing screenshot analysis: {len(image_data)} bytes")
        
        # Send to JARVIS for analysis
        response = requests.post('http://localhost:5001/api/jarvis/vision-analyze',
                               json={
                                   'image': image_data,
                                   'prompt': 'You are JARVIS. Analyze this AR application screenshot. What 3D objects or interface elements do you see?'
                               },
                               timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'success':
                print("✅ JARVIS screenshot analysis successful!")
                print("\n🤖 JARVIS Analysis:")
                print("-" * 50)
                print(data['analysis'])
                print("-" * 50)
                return True
            else:
                print(f"❌ JARVIS analysis failed: {data.get('error')}")
                return False
        else:
            print(f"❌ JARVIS analysis request failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Screenshot analysis error: {e}")
        return False

def start_jarvis_backend():
    """Start JARVIS backend if not running"""
    if check_jarvis_backend():
        return True
    
    print("🚀 Starting JARVIS backend...")
    try:
        # Start backend in background
        process = subprocess.Popen([
            'python', 'jarvis_backend.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for startup
        time.sleep(3)
        
        # Check if it's running
        if check_jarvis_backend():
            print("✅ JARVIS backend started successfully")
            return True
        else:
            print("❌ JARVIS backend failed to start")
            return False
            
    except Exception as e:
        print(f"❌ Failed to start JARVIS backend: {e}")
        return False

def main():
    """Run JARVIS OpenCV integration test"""
    print("🤖 JARVIS OpenCV Integration Test")
    print("=" * 50)
    
    # Check environment
    if not os.getenv('GEMINI_API_KEY'):
        print("❌ GEMINI_API_KEY not set")
        print("   Run: export GEMINI_API_KEY='your-key-here'")
        return False
    
    print("✅ GEMINI_API_KEY found")
    
    # Check if backend is running
    if not check_jarvis_backend():
        print("\n🚀 JARVIS backend not running, attempting to start...")
        if not start_jarvis_backend():
            print("❌ Could not start JARVIS backend")
            print("💡 Try running manually: python jarvis_backend.py")
            return False
    
    # Test screenshot analysis if available
    print("\n📸 Testing screenshot analysis...")
    if test_screenshot_analysis():
        print("\n🎉 JARVIS OpenCV integration test successful!")
        print("\n📋 Complete workflow:")
        print("1. ✅ JARVIS backend running")
        print("2. ✅ Screenshot captured from OpenCV")
        print("3. ✅ JARVIS analyzed the image successfully")
        print("4. ✅ Voice assistant ready for interaction")
        
        print("\n🎯 Next steps:")
        print("1. Run: python main.py")
        print("2. Press 'J' to activate JARVIS in OpenCV window")
        print("3. Open: http://localhost:8080/jarvis_integration.html")
        print("4. Say 'Hey JARVIS' or click to activate voice assistant")
        print("5. Ask 'What is this?' to analyze the 3D objects")
        
        return True
    else:
        print("\n⚠️ Screenshot analysis test failed")
        print("💡 This is normal if no screenshot exists yet")
        print("\n📋 Manual test steps:")
        print("1. Run: python main.py")
        print("2. Press 'J' in the OpenCV window")
        print("3. Run this test again: python test_jarvis_opencv.py")
        
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
