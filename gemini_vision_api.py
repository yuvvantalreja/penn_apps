"""
Gemini Vision API Handler
Simple backend service to handle Gemini Vision API requests
"""

import base64
import json
import requests
from typing import Optional, Dict, Any
import os
from PIL import Image
import io


class GeminiVisionAPI:
    """Handler for Gemini Vision API requests"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
        
        if not self.api_key:
            print("⚠️ Warning: No Gemini API key found. Vision analysis will not work.")
    
    def analyze_image(self, image_base64: str, prompt: str, max_tokens: int = 150) -> Optional[str]:
        """
        Analyze an image using Gemini Vision API
        
        Args:
            image_base64: Base64 encoded image
            prompt: Text prompt for analysis
            max_tokens: Maximum tokens in response
            
        Returns:
            Analysis text or None if failed
        """
        if not self.api_key:
            return "I apologize sir, but I don't have access to my vision systems at the moment."
        
        try:
            # Prepare the request payload
            payload = {
                "contents": [{
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": image_base64
                            }
                        }
                    ]
                }],
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature": 0.4,
                    "topP": 0.8,
                    "topK": 40
                }
            }
            
            # Make the API request
            headers = {
                "Content-Type": "application/json",
            }
            
            url = f"{self.base_url}?key={self.api_key}"
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract the generated text
                if 'candidates' in result and len(result['candidates']) > 0:
                    candidate = result['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                        parts = candidate['content']['parts']
                        if len(parts) > 0 and 'text' in parts[0]:
                            return parts[0]['text'].strip()
                
                print("❌ Unexpected response format from Gemini Vision API")
                return None
                
            else:
                print(f"❌ Gemini Vision API error: {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Error calling Gemini Vision API: {e}")
            return None
    
    def analyze_image_with_pointing(self, image_base64: str, pointing_coords: tuple = None) -> Optional[str]:
        """
        Analyze image with specific focus on pointed object
        
        Args:
            image_base64: Base64 encoded image
            pointing_coords: (x, y) coordinates of pointing location
            
        Returns:
            Analysis text or None if failed
        """
        if pointing_coords:
            prompt = f"""You are Jarvis, Tony Stark's AI assistant. Analyze this image and identify the specific object or component that the user is pointing at (marked with a red circle at coordinates approximately {pointing_coords}).

Focus on:
1. What is the specific object/component being pointed at?
2. What is its likely function or purpose?
3. If it's part of a larger system (like a 3D model or mechanical assembly), explain its role

Respond in Jarvis's characteristic style - intelligent, concise, and slightly formal. Start your response with something like "Sir, you are pointing at..." or "That appears to be..."

Keep the response under 100 words and be specific about what you observe."""
        else:
            prompt = """You are Jarvis, Tony Stark's AI assistant. Analyze this image and describe what you see in the scene.

Focus on:
1. Main objects or components visible
2. Their likely purpose or function
3. Overall context of the scene

Respond in Jarvis's characteristic style - intelligent, concise, and slightly formal. Keep the response under 100 words."""
        
        return self.analyze_image(image_base64, prompt)
    
    def chat_with_context(self, message: str, max_tokens: int = 100) -> Optional[str]:
        """
        Simple chat functionality for general queries
        
        Args:
            message: User's message/query
            max_tokens: Maximum tokens in response
            
        Returns:
            Response text or None if failed
        """
        if not self.api_key:
            return "I apologize sir, but I don't have access to my systems at the moment."
        
        try:
            payload = {
                "contents": [{
                    "parts": [{"text": message}]
                }],
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature": 0.7,
                    "topP": 0.8,
                    "topK": 40
                }
            }
            
            headers = {"Content-Type": "application/json"}
            url = f"{self.base_url}?key={self.api_key}"
            
            response = requests.post(url, headers=headers, json=payload, timeout=15)
            
            if response.status_code == 200:
                result = response.json()
                
                if 'candidates' in result and len(result['candidates']) > 0:
                    candidate = result['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                        parts = candidate['content']['parts']
                        if len(parts) > 0 and 'text' in parts[0]:
                            return parts[0]['text'].strip()
                
                return None
                
            else:
                print(f"❌ Gemini Chat API error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error calling Gemini Chat API: {e}")
            return None


# Global instance for easy access
gemini_vision = GeminiVisionAPI()


def analyze_image_simple(image_base64: str, prompt: str) -> Dict[str, Any]:
    """
    Simple function to analyze image - compatible with existing backend expectations
    
    Returns:
        Dictionary with status and analysis/error
    """
    try:
        analysis = gemini_vision.analyze_image(image_base64, prompt)
        
        if analysis:
            return {
                "status": "success",
                "analysis": analysis
            }
        else:
            return {
                "status": "error",
                "error": "Failed to analyze image"
            }
            
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


def chat_simple(message: str) -> Dict[str, Any]:
    """
    Simple function to chat with Gemini - compatible with existing backend expectations
    
    Returns:
        Dictionary with status and response/error
    """
    try:
        response = gemini_vision.chat_with_context(message)
        
        if response:
            return {
                "status": "success",
                "response": response
            }
        else:
            return {
                "status": "error",
                "error": "Failed to get response"
            }
            
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


if __name__ == "__main__":
    # Test the API
    print("🧪 Testing Gemini Vision API...")
    
    # Create a simple test image
    test_image = Image.new('RGB', (100, 100), color='red')
    buffer = io.BytesIO()
    test_image.save(buffer, format='JPEG')
    test_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # Test analysis
    result = analyze_image_simple(test_base64, "What color is this image?")
    print("Test result:", result)
