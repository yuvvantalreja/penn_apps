#!/usr/bin/env python3
"""
Test script for JARVIS Voice Assistant integration
Tests the vision analysis and backend functionality
"""

import os
import sys
import base64
import json
import requests
from PIL import Image, ImageDraw
import io

def create_test_image():
    """Create a test image with 3D-like content"""
    # Create a simple test image that looks like a 3D model
    img = Image.new('RGB', (800, 600), color='black')
    draw = ImageDraw.Draw(img)
    
    # Draw a simple 3D cube representation
    # Front face
    draw.polygon([(200, 200), (400, 200), (400, 400), (200, 400)], outline='white', fill='gray')
    
    # Top face (perspective)
    draw.polygon([(200, 200), (250, 150), (450, 150), (400, 200)], outline='white', fill='lightgray')
    
    # Right face (perspective)
    draw.polygon([(400, 200), (450, 150), (450, 350), (400, 400)], outline='white', fill='darkgray')
    
    # Add some labels
    draw.text((300, 300), "3D CUBE", fill='white')
    draw.text((300, 320), "Test Model", fill='white')
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    
    return base64.b64encode(buffer.read()).decode('utf-8')

def test_backend_health():
    """Test if JARVIS backend is running"""
    try:
        print("🏥 Testing JARVIS backend health...")
        response = requests.get('http://localhost:5001/api/jarvis/health', timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Backend health check passed")
            print(f"   Status: {data.get('status')}")
            print(f"   JARVIS Backend: {data.get('jarvis_backend')}")
            return True
        else:
            print(f"❌ Backend health check failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Backend not reachable: {e}")
        return False

def test_vision_analysis():
    """Test vision analysis with a sample image"""
    try:
        print("🔍 Testing JARVIS vision analysis...")
        
        # Create test image
        test_image_b64 = create_test_image()
        print(f"📸 Created test image: {len(test_image_b64)} bytes")
        
        # Send to JARVIS for analysis
        response = requests.post('http://localhost:5001/api/jarvis/vision-analyze', 
                               json={
                                   'image': test_image_b64,
                                   'prompt': 'You are JARVIS. What do you see in this image? Describe it as JARVIS would.'
                               },
                               timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'success':
                print("✅ Vision analysis successful")
                print("🤖 JARVIS Analysis:")
                print(f"   {data['analysis']}")
                return True
            else:
                print(f"❌ Vision analysis failed: {data.get('error')}")
                return False
        else:
            print(f"❌ Vision analysis request failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Vision analysis request error: {e}")
        return False

def test_websocket_endpoint():
    """Test WebSocket URL generation"""
    try:
        print("🔌 Testing WebSocket endpoint...")
        response = requests.get('http://localhost:5001/api/gemini/websocket', timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'success':
                print("✅ WebSocket endpoint working")
                print(f"   URL available: {len(data['websocket_url'])} chars")
                return True
            else:
                print(f"❌ WebSocket endpoint error: {data.get('error')}")
                return False
        else:
            print(f"❌ WebSocket endpoint failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ WebSocket endpoint error: {e}")
        return False

def test_jarvis_status():
    """Test JARVIS status endpoint"""
    try:
        print("📊 Testing JARVIS status...")
        response = requests.get('http://localhost:5001/api/jarvis/status', timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ JARVIS status retrieved")
            print(f"   Service: {data.get('service')}")
            print(f"   Version: {data.get('version')}")
            print(f"   Vision API: {'Available' if data.get('vision_api_available') else 'Not Available'}")
            print(f"   Capabilities: {', '.join(data.get('capabilities', []))}")
            return True
        else:
            print(f"❌ JARVIS status failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ JARVIS status error: {e}")
        return False

def check_environment():
    """Check if environment is properly configured"""
    print("🔧 Checking environment configuration...")
    
    # Check for Gemini API key
    api_key = os.getenv('GEMINI_API_KEY')
    if api_key:
        print(f"✅ GEMINI_API_KEY found: {api_key[:10]}...")
    else:
        print("❌ GEMINI_API_KEY not found")
        print("   Set it with: export GEMINI_API_KEY='your-key-here'")
        return False
    
    # Check for required Python packages
    required_packages = ['flask', 'flask_cors', 'google.generativeai', 'PIL', 'requests']
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            elif package == 'flask_cors':
                import flask_cors
            elif package == 'google.generativeai':
                import google.generativeai
            else:
                __import__(package)
            print(f"✅ {package} available")
        except ImportError:
            print(f"❌ {package} not found")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("   Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def main():
    """Run all JARVIS tests"""
    print("🤖 JARVIS Voice Assistant Test Suite")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed. Please fix configuration and try again.")
        return False
    
    print("\n🚀 Starting JARVIS tests...")
    
    # Test results
    results = {
        'backend_health': test_backend_health(),
        'jarvis_status': test_jarvis_status(),
        'websocket_endpoint': test_websocket_endpoint(),
        'vision_analysis': test_vision_analysis()
    }
    
    # Summary
    print("\n📊 Test Results Summary:")
    print("=" * 30)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! JARVIS is ready for use.")
        print("\nNext steps:")
        print("1. Start the AR application: python main.py")
        print("2. Open jarvis_integration.html in your browser")
        print("3. Press 'J' or say 'Hey JARVIS' to activate")
        return True
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please check the issues above.")
        print("\nTroubleshooting:")
        print("- Make sure JARVIS backend is running: python jarvis_backend.py")
        print("- Check your GEMINI_API_KEY is valid")
        print("- Verify all dependencies are installed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
