#!/usr/bin/env python3
"""
JARVIS Backend Integration
Provides vision analysis endpoints for the JARVIS voice assistant
"""

import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from gemini_vision_api import GeminiVisionAPI

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Initialize Gemini Vision API
try:
    vision_api = GeminiVisionAPI()
    print("✅ JARVIS Backend initialized with Gemini Vision API")
except Exception as e:
    print(f"❌ Failed to initialize Gemini Vision API: {e}")
    vision_api = None

@app.route('/api/jarvis/health', methods=['GET'])
def jarvis_health():
    """Health check endpoint for JARVIS backend"""
    try:
        if not vision_api:
            return jsonify({
                "status": "error",
                "message": "Gemini Vision API not available"
            }), 500
        
        # Test the vision API
        health_result = vision_api.health_check()
        
        return jsonify({
            "status": "healthy",
            "jarvis_backend": "operational",
            "vision_api": health_result
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/jarvis/vision-analyze', methods=['POST'])
def jarvis_vision_analyze():
    """Analyze image for JARVIS voice assistant"""
    try:
        if not vision_api:
            return jsonify({
                "status": "error",
                "error": "Gemini Vision API not available"
            }), 500
        
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                "status": "error",
                "error": "No image data provided"
            }), 400
        
        image_data = data['image']
        prompt = data.get('prompt', None)
        
        print(f"🔍 JARVIS analyzing image: {len(image_data)} bytes")
        
        # Use JARVIS-specific analysis if no custom prompt
        if not prompt:
            result = vision_api.analyze_screenshot_for_jarvis(image_data)
        else:
            result = vision_api.analyze_image(image_data, prompt)
        
        print(f"✅ JARVIS analysis complete: {result['status']}")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ JARVIS vision analysis error: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/api/jarvis/test-vision', methods=['POST'])
def jarvis_test_vision():
    """Test vision analysis with a sample prompt"""
    try:
        if not vision_api:
            return jsonify({
                "status": "error",
                "error": "Gemini Vision API not available"
            }), 500
        
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                "status": "error",
                "error": "No image data provided for test"
            }), 400
        
        image_data = data['image']
        
        # Test with a simple JARVIS prompt
        test_prompt = """You are JARVIS. Analyze this image and tell me what you see. 
        Be specific about any objects, components, or technical elements visible. 
        Respond as JARVIS would - sophisticated and informative."""
        
        result = vision_api.analyze_image(image_data, test_prompt)
        
        return jsonify({
            "status": "success",
            "test_type": "jarvis_vision_test",
            "analysis": result
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/api/gemini/websocket', methods=['GET'])
def get_gemini_websocket():
    """Get secure Gemini WebSocket URL (proxy for JARVIS)"""
    try:
        # This would normally create a secure WebSocket URL
        # For now, return the direct Gemini Live WebSocket URL
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            return jsonify({
                "status": "error",
                "error": "Gemini API key not configured"
            }), 500
        
        # Gemini Live WebSocket URL
        websocket_url = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key={api_key}"
        
        return jsonify({
            "status": "success",
            "websocket_url": websocket_url
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/api/jarvis/status', methods=['GET'])
def jarvis_status():
    """Get JARVIS backend status"""
    return jsonify({
        "status": "operational",
        "service": "JARVIS Backend",
        "version": "1.0.0",
        "capabilities": [
            "voice_interaction",
            "vision_analysis", 
            "screenshot_analysis",
            "gemini_live_integration"
        ],
        "vision_api_available": vision_api is not None
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "status": "error",
        "error": "Endpoint not found"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "status": "error", 
        "error": "Internal server error"
    }), 500

if __name__ == "__main__":
    print("🤖 Starting JARVIS Backend Server...")
    print("🔍 Vision Analysis: Available" if vision_api else "❌ Vision Analysis: Not Available")
    print("🌐 CORS: Enabled for frontend integration")
    print("📡 Endpoints:")
    print("  - GET  /api/jarvis/health")
    print("  - POST /api/jarvis/vision-analyze")
    print("  - POST /api/jarvis/test-vision")
    print("  - GET  /api/jarvis/status")
    print("  - GET  /api/gemini/websocket")
    
    # Run the server
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True
    )
