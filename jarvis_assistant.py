"""
Jarvis Voice Assistant for AR Hand Control
Integrates with Gemini Live and Vision APIs for intelligent object identification
"""

import cv2
import numpy as np
import base64
import json
import asyncio
import threading
import time
from typing import Optional, Dict, Any, Tuple
import requests
import io
from PIL import Image
import speech_recognition as sr
import pyttsx3
from gemini_vision_api import gemini_vision


class JarvisAssistant:
    """
    Jarvis Voice Assistant that can:
    1. Respond to voice commands
    2. Take screenshots of AR scenes
    3. Identify objects using Gemini Vision
    4. Provide voice responses using TTS
    """
    
    def __init__(self, ar_controller=None):
        self.ar_controller = ar_controller
        self.is_active = False
        self.is_listening = False
        self.is_processing = False
        
        # Voice components
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.tts_engine = None
        
        # Gemini API configuration
        self.api_base_url = "https://pitchperfect2-api-373812504656.asia-southeast1.run.app"
        
        # Hand pointing detection
        self.pointing_threshold = 0.8
        self.last_pointing_position = None
        
        # Screenshot and analysis
        self.last_screenshot = None
        self.analysis_cache = {}
        
        self.initialize_components()
    
    def initialize_components(self):
        """Initialize voice recognition and TTS components"""
        try:
            # Initialize TTS engine
            self.tts_engine = pyttsx3.init()
            
            # Configure TTS voice (try to find a suitable voice)
            voices = self.tts_engine.getProperty('voices')
            for voice in voices:
                if 'male' in voice.name.lower() or 'david' in voice.name.lower():
                    self.tts_engine.setProperty('voice', voice.id)
                    break
            
            # Set speech rate and volume
            self.tts_engine.setProperty('rate', 180)  # Slightly faster for Jarvis
            self.tts_engine.setProperty('volume', 0.9)
            
            # Calibrate microphone for ambient noise
            print("🎤 Calibrating microphone for ambient noise...")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            print("✅ Jarvis voice components initialized")
            
        except Exception as e:
            print(f"❌ Failed to initialize Jarvis components: {e}")
    
    def activate_jarvis(self):
        """Activate Jarvis with greeting"""
        if self.is_active:
            return
            
        self.is_active = True
        print("🤖 Jarvis activated!")
        
        # Jarvis greeting
        self.speak("Hello sir, what can I do for you today?")
        
        # Start listening in a separate thread
        threading.Thread(target=self.listen_for_commands, daemon=True).start()
    
    def deactivate_jarvis(self):
        """Deactivate Jarvis"""
        self.is_active = False
        self.is_listening = False
        print("🤖 Jarvis deactivated")
    
    def speak(self, text: str):
        """Make Jarvis speak using TTS"""
        try:
            if self.tts_engine:
                print(f"🤖 Jarvis: {text}")
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
        except Exception as e:
            print(f"❌ TTS Error: {e}")
    
    def listen_for_commands(self):
        """Listen for voice commands continuously"""
        self.is_listening = True
        
        while self.is_active and self.is_listening:
            try:
                with self.microphone as source:
                    print("🎤 Jarvis listening...")
                    # Listen for audio with timeout
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                
                try:
                    # Recognize speech
                    command = self.recognizer.recognize_google(audio).lower()
                    print(f"👤 User said: {command}")
                    
                    # Process the command
                    self.process_voice_command(command)
                    
                except sr.UnknownValueError:
                    # No speech detected, continue listening
                    continue
                except sr.RequestError as e:
                    print(f"❌ Speech recognition error: {e}")
                    continue
                    
            except sr.WaitTimeoutError:
                # Timeout is normal, continue listening
                continue
            except Exception as e:
                print(f"❌ Listening error: {e}")
                time.sleep(1)
    
    def process_voice_command(self, command: str):
        """Process voice commands from user"""
        if self.is_processing:
            self.speak("Please wait, I'm still processing your previous request.")
            return
        
        self.is_processing = True
        
        try:
            # Check for object identification commands
            if any(phrase in command for phrase in ["what is this", "what's this", "identify this", "what am i pointing at"]):
                self.identify_pointed_object()
            
            elif any(phrase in command for phrase in ["what do you see", "describe the scene", "what's on screen"]):
                self.describe_scene()
            
            elif any(phrase in command for phrase in ["help", "what can you do", "commands"]):
                self.list_capabilities()
            
            elif any(phrase in command for phrase in ["goodbye", "bye", "deactivate", "stop"]):
                self.speak("Goodbye sir. Deactivating Jarvis.")
                self.deactivate_jarvis()
            
            else:
                # General query - send to Gemini for processing
                self.handle_general_query(command)
                
        except Exception as e:
            print(f"❌ Command processing error: {e}")
            self.speak("I apologize sir, I encountered an error processing your request.")
        finally:
            self.is_processing = False
    
    def identify_pointed_object(self):
        """Identify the object the user is pointing at"""
        try:
            self.speak("Let me analyze what you're pointing at, sir.")
            
            # Take screenshot of current scene
            screenshot = self.capture_ar_screenshot()
            if screenshot is None:
                self.speak("I'm unable to capture the current view, sir.")
                return
            
            # Detect pointing gesture and get coordinates
            pointing_coords = self.detect_pointing_gesture()
            
            # Analyze the image with Gemini Vision
            analysis = self.analyze_image_with_gemini(screenshot, pointing_coords)
            
            if analysis:
                self.speak(analysis)
            else:
                self.speak("I'm having trouble identifying that object, sir.")
                
        except Exception as e:
            print(f"❌ Object identification error: {e}")
            self.speak("I encountered an error while analyzing the object, sir.")
    
    def capture_ar_screenshot(self) -> Optional[np.ndarray]:
        """Capture screenshot of the current AR scene"""
        try:
            if not self.ar_controller or not hasattr(self.ar_controller, 'current_frame'):
                print("❌ No AR controller or current frame available")
                return None
            
            # Get the current frame from AR controller
            frame = self.ar_controller.current_frame.copy() if self.ar_controller.current_frame is not None else None
            
            if frame is None:
                print("❌ No current frame available from AR controller")
                return None
            
            print(f"📸 Captured AR screenshot: {frame.shape}")
            self.last_screenshot = frame
            return frame
            
        except Exception as e:
            print(f"❌ Screenshot capture error: {e}")
            return None
    
    def detect_pointing_gesture(self) -> Optional[Tuple[int, int]]:
        """Detect if user is pointing and get coordinates"""
        try:
            if not self.ar_controller or not hasattr(self.ar_controller, 'detector'):
                return None
            
            # Get current hand landmarks from the detector
            if hasattr(self.ar_controller.detector, 'current_hands_info'):
                hands_info = self.ar_controller.detector.current_hands_info
                
                for hand_info in hands_info:
                    # Check if hand is in pointing gesture
                    if self.is_pointing_gesture(hand_info):
                        # Get index finger tip position
                        index_tip = hand_info.get('index_tip')
                        if index_tip:
                            print(f"👉 Pointing detected at: {index_tip}")
                            self.last_pointing_position = index_tip
                            return index_tip
            
            # If no pointing detected, use center of screen
            if self.last_screenshot is not None:
                h, w = self.last_screenshot.shape[:2]
                center = (w // 2, h // 2)
                print(f"📍 No pointing detected, using screen center: {center}")
                return center
            
            return None
            
        except Exception as e:
            print(f"❌ Pointing detection error: {e}")
            return None
    
    def is_pointing_gesture(self, hand_info: Dict) -> bool:
        """Determine if hand is in pointing gesture"""
        try:
            # Simple pointing detection: index finger extended, others curled
            landmarks = hand_info.get('landmarks')
            if not landmarks:
                return False
            
            # Check if index finger is extended (tip higher than PIP joint)
            index_tip_y = landmarks[8][1] if len(landmarks) > 8 else 0
            index_pip_y = landmarks[6][1] if len(landmarks) > 6 else 0
            
            # Check if middle finger is curled (tip lower than PIP joint)
            middle_tip_y = landmarks[12][1] if len(landmarks) > 12 else 0
            middle_pip_y = landmarks[10][1] if len(landmarks) > 10 else 0
            
            # Simple heuristic: index extended, middle curled
            index_extended = index_tip_y < index_pip_y
            middle_curled = middle_tip_y > middle_pip_y
            
            return index_extended and middle_curled
            
        except Exception as e:
            print(f"❌ Pointing gesture detection error: {e}")
            return False
    
    def analyze_image_with_gemini(self, image: np.ndarray, pointing_coords: Optional[Tuple[int, int]] = None) -> Optional[str]:
        """Analyze image using Gemini Vision API"""
        try:
            # Convert image to base64
            image_base64 = self.image_to_base64(image, pointing_coords)
            if not image_base64:
                return None
            
            # Use local Gemini Vision API
            analysis = gemini_vision.analyze_image_with_pointing(image_base64, pointing_coords)
            
            if analysis:
                print(f"🔍 Gemini Vision analysis: {analysis}")
                return analysis
            else:
                print("❌ Failed to get analysis from Gemini Vision")
                return None
            
        except Exception as e:
            print(f"❌ Gemini Vision analysis error: {e}")
            return None
    
    def image_to_base64(self, image: np.ndarray, pointing_coords: Optional[Tuple[int, int]] = None) -> Optional[str]:
        """Convert OpenCV image to base64 with optional pointing indicator"""
        try:
            # Add pointing indicator if coordinates provided
            if pointing_coords:
                image = image.copy()
                x, y = pointing_coords
                # Draw a red circle at pointing location
                cv2.circle(image, (int(x), int(y)), 20, (0, 0, 255), 3)
                cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_rgb)
            
            # Convert to base64
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=85)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return image_base64
            
        except Exception as e:
            print(f"❌ Image to base64 conversion error: {e}")
            return None
    
    def build_vision_prompt(self, pointing_coords: Optional[Tuple[int, int]] = None) -> str:
        """Build prompt for Gemini Vision analysis"""
        if pointing_coords:
            return """You are Jarvis, Tony Stark's AI assistant. Analyze this image and identify the specific object or component that the user is pointing at (marked with a red circle). 

Focus on:
1. What is the specific object/component being pointed at?
2. What is its likely function or purpose?
3. If it's part of a larger system (like Iron Man's suit), explain its role

Respond in Jarvis's characteristic style - intelligent, concise, and slightly formal. Start your response with something like "Sir, you are pointing at..." or "That appears to be..."

Keep the response under 100 words."""
        else:
            return """You are Jarvis, Tony Stark's AI assistant. Analyze this image and describe what you see in the scene.

Focus on:
1. Main objects or components visible
2. Their likely purpose or function
3. Overall context of the scene

Respond in Jarvis's characteristic style - intelligent, concise, and slightly formal. Keep the response under 100 words."""
    
    def describe_scene(self):
        """Describe the entire scene"""
        try:
            self.speak("Let me analyze the current scene, sir.")
            
            screenshot = self.capture_ar_screenshot()
            if screenshot is None:
                self.speak("I'm unable to capture the current view, sir.")
                return
            
            analysis = self.analyze_image_with_gemini(screenshot, None)
            
            if analysis:
                self.speak(analysis)
            else:
                self.speak("I'm having trouble analyzing the current scene, sir.")
                
        except Exception as e:
            print(f"❌ Scene description error: {e}")
            self.speak("I encountered an error while analyzing the scene, sir.")
    
    def list_capabilities(self):
        """List Jarvis capabilities"""
        capabilities = """I can help you with several tasks, sir:
        - Say 'what is this' while pointing at an object to identify it
        - Say 'describe the scene' to get an overview of what's visible
        - Ask me general questions about what you're working on
        - Say 'goodbye' to deactivate me"""
        
        self.speak(capabilities)
    
    def handle_general_query(self, query: str):
        """Handle general queries using Gemini"""
        try:
            self.speak("Let me think about that, sir.")
            
            # Use local Gemini API for chat
            jarvis_query = f"You are Jarvis, Tony Stark's AI assistant. Answer this query in character: {query}"
            response = gemini_vision.chat_with_context(jarvis_query, max_tokens=100)
            
            if response:
                self.speak(response)
            else:
                self.speak("I'm not sure how to help with that, sir.")
            
        except Exception as e:
            print(f"❌ General query error: {e}")
            self.speak("I encountered an error processing your query, sir.")


# Integration function for AR Hand Control
def integrate_jarvis_with_ar_controller(ar_controller):
    """Create and integrate Jarvis with AR controller"""
    jarvis = JarvisAssistant(ar_controller)
    
    # Store reference in AR controller
    ar_controller.jarvis = jarvis
    
    return jarvis
