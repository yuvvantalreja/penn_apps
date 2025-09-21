import cv2
import mediapipe as mp
import numpy as np
import math
from typing import List, Tuple, Optional
import time
import os
from renderer_3d import Renderer3D
from virtual_object_3d import VirtualObject3D, CADAssembly

class VirtualObject:
    """Represents a virtual object that can be manipulated in AR space"""
    
    def __init__(self, x: float, y: float, size: float = 50, color: Tuple[int, int, int] = (0, 255, 255), shape: str = "circle"):
        self.x = x
        self.y = y
        self.size = size
        self.original_size = size
        self.color = color
        self.shape = shape
        self.is_grabbed = 0  # 0 = not grabbed, 1 = grabbed with 1 hand, 2 = grabbed with 2 hands
        self.grabbed_by_hand = []  # List of hand indices that are grabbing this object
        self.z_depth = 0.0 
        
    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Draw the virtual object with Apple Vision Pro-inspired glassmorphism"""
        center = (int(self.x), int(self.y))
        radius = int(self.size)
        
        if self.shape == "circle":
            # Create glassmorphism effect with multiple layers
            self._draw_glass_object(frame, center, radius)
            
            # Draw interaction states with elegant indicators
            if self.is_grabbed == 1:
                # Single hand grab - subtle pulsing ring
                self._draw_interaction_ring(frame, center, radius + 8, (200, 220, 255), 2, "single")
            elif self.is_grabbed == 2:
                # Two hand grab - scaling mode indicator
                self._draw_interaction_ring(frame, center, radius + 12, (255, 200, 120), 3, "dual")
                
        return frame
    
    def _draw_glass_object(self, frame: np.ndarray, center: tuple, radius: int):
        """Draw object with glassmorphism effect"""
        # Base glass layer with transparency
        overlay = frame.copy()
        
        # Outer glow
        cv2.circle(overlay, center, radius + 3, self.color, -1)
        cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)
        
        # Main glass body
        overlay = frame.copy()
        cv2.circle(overlay, center, radius, self.color, -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Inner highlight for depth
        highlight_color = tuple(min(255, c + 80) for c in self.color)
        cv2.circle(frame, (center[0] - radius//3, center[1] - radius//3), radius//3, highlight_color, -1)
        overlay = frame.copy()
        cv2.circle(overlay, (center[0] - radius//3, center[1] - radius//3), radius//3, (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        
        # Subtle border
        border_color = tuple(min(255, c + 40) for c in self.color)
        cv2.circle(frame, center, radius, border_color, 1)
    
    def _draw_interaction_ring(self, frame: np.ndarray, center: tuple, radius: int, color: tuple, thickness: int, mode: str):
        """Draw interaction ring with animation-like effects"""
        # Main interaction ring
        cv2.circle(frame, center, radius, color, thickness)
        
        if mode == "dual":
            # Add scaling indicators for two-hand interaction
            # Draw small directional indicators
            import math
            for angle in [0, 90, 180, 270]:
                rad = math.radians(angle)
                indicator_x = int(center[0] + (radius + 8) * math.cos(rad))
                indicator_y = int(center[1] + (radius + 8) * math.sin(rad))
                cv2.circle(frame, (indicator_x, indicator_y), 3, color, -1)
                
        # Add subtle glow effect
        overlay = frame.copy()
        cv2.circle(overlay, center, radius, color, thickness * 2)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    def is_point_inside(self, x: float, y: float) -> bool:
        """Check if a point is inside the object"""
        distance = math.sqrt((x - self.x) ** 2 + (y - self.y) ** 2)
        return distance <= self.size
    
    def move_to(self, x: float, y: float):
        self.x = x
        self.y = y
    
    def scale(self, scale_factor: float):
        self.size = max(10, min(200, self.original_size * scale_factor))


class HandGestureDetector:
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def detect_hands(self, frame: np.ndarray) -> List[dict]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        hands_info = []
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand_info = self._extract_hand_info(hand_landmarks, frame.shape)
                hand_info['landmarks'] = hand_landmarks
                hand_info['hand_idx'] = hand_idx
                hands_info.append(hand_info)
                
        return hands_info
    
    def _extract_hand_info(self, landmarks, frame_shape) -> dict:
        h, w = frame_shape[:2]
        
        thumb_tip = landmarks.landmark[4]
        index_tip = landmarks.landmark[8]
        middle_tip = landmarks.landmark[12]
        ring_tip = landmarks.landmark[16]
        pinky_tip = landmarks.landmark[20]
        
        wrist = landmarks.landmark[0]
        index_mcp = landmarks.landmark[5]
        
        # Convert to pixel coordinates
        thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))
        index_pos = (int(index_tip.x * w), int(index_tip.y * h))
        middle_pos = (int(middle_tip.x * w), int(middle_tip.y * h))
        wrist_pos = (int(wrist.x * w), int(wrist.y * h))
        
        # Calculate palm center
        palm_x = int((wrist.x + index_mcp.x) * w / 2)
        palm_y = int((wrist.y + index_mcp.y) * h / 2)
        palm_center = (palm_x, palm_y)
        
        # Detect gestures
        gestures = self._detect_gestures(landmarks)
        
        pinch_distance = math.sqrt((thumb_pos[0] - index_pos[0]) ** 2 + (thumb_pos[1] - index_pos[1]) ** 2)
        
        # Calculate pinch center for better tracking
        pinch_center = ((thumb_pos[0] + index_pos[0]) // 2, (thumb_pos[1] + index_pos[1]) // 2)
        
        # More reliable pinch detection with multiple criteria
        # Check if fingers are actually bent (not just close together)
        thumb_bent = thumb_tip.x < landmarks.landmark[3].x  # Thumb is bent inward
        index_bent = index_tip.y > landmarks.landmark[6].y  # Index finger is bent down
        
        # Much more lenient pinch detection - prioritize distance over finger position
        # Primary method: distance-based
        is_pinching = (pinch_distance < 60 and  # Increased distance threshold
                      pinch_distance > 10 and  # Not too close (avoid noise)
                      (thumb_bent or index_bent or pinch_distance < 35))  # Either finger bent OR very close
        
        # Fallback method: if distance is very close, always consider it pinching
        if pinch_distance < 25:
            is_pinching = True
        
        return {
            'thumb_pos': thumb_pos,
            'index_pos': index_pos,
            'middle_pos': middle_pos,
            'palm_center': palm_center,
            'pinch_center': pinch_center,
            'wrist_pos': wrist_pos,
            'gestures': gestures,
            'pinch_distance': pinch_distance,
            'is_pinching': is_pinching,
            'thumb_bent': thumb_bent,
            'index_bent': index_bent
        }
    
    def _detect_gestures(self, landmarks) -> List[str]:
        """Detect specific hand gestures"""
        gestures = []
        
        # Get finger tip and pip positions
        fingertips = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
        pip_joints = [3, 6, 10, 14, 18]
        
        # Count extended fingers
        extended_fingers = []
        
        # Thumb (special case - check x coordinate)
        if landmarks.landmark[4].x > landmarks.landmark[3].x:
            extended_fingers.append(True)
        else:
            extended_fingers.append(False)
            
        # Other fingers
        for i in range(1, 5):
            if landmarks.landmark[fingertips[i]].y < landmarks.landmark[pip_joints[i]].y:
                extended_fingers.append(True)
            else:
                extended_fingers.append(False)
        
        extended_count = sum(extended_fingers)
        
        # Classify gestures
        if extended_count == 0:
            gestures.append("fist")
        elif extended_count == 1 and extended_fingers[1]:  # Only index finger
            gestures.append("pointing")
        elif extended_count == 2 and extended_fingers[0] and extended_fingers[1]:  # Thumb + index
            gestures.append("pinch")
        elif extended_count == 5:
            gestures.append("open_hand")
        elif extended_count == 2 and extended_fingers[1] and extended_fingers[2]:  # Index + middle
            gestures.append("peace")
            
        return gestures
    
    def draw_landmarks(self, frame: np.ndarray, hands_info: List[dict]) -> np.ndarray:
        """Draw minimal, elegant hand landmarks"""
        for hand_info in hands_info:
            if 'landmarks' in hand_info:
                # Only draw key landmarks with Apple Vision Pro styling
                landmarks = hand_info['landmarks']
                h, w = frame.shape[:2]
                
                # Key landmark indices for minimal visualization
                key_landmarks = [0, 4, 8, 12, 16, 20]  # Wrist, thumb tip, index tip, middle tip, ring tip, pinky tip
                
                # Draw minimal landmarks
                for idx in key_landmarks:
                    if idx < len(landmarks.landmark):
                        landmark = landmarks.landmark[idx]
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        
                        # Subtle landmark indicators
                        if idx == 0:  # Wrist - slightly larger
                            cv2.circle(frame, (x, y), 4, (150, 150, 150), 1)
                            cv2.circle(frame, (x, y), 2, (200, 200, 200), -1)
                        else:  # Fingertips - small dots
                            cv2.circle(frame, (x, y), 2, (180, 180, 180), 1)
                            cv2.circle(frame, (x, y), 1, (220, 220, 220), -1)
                
                # Draw minimal hand outline connections (only essential ones)
                essential_connections = [
                    (0, 5), (5, 9), (9, 13), (13, 17),  # Palm outline
                    (0, 17)  # Close the palm
                ]
                
                for connection in essential_connections:
                    start_idx, end_idx = connection
                    if start_idx < len(landmarks.landmark) and end_idx < len(landmarks.landmark):
                        start_landmark = landmarks.landmark[start_idx]
                        end_landmark = landmarks.landmark[end_idx]
                        
                        start_x = int(start_landmark.x * w)
                        start_y = int(start_landmark.y * h)
                        end_x = int(end_landmark.x * w)
                        end_y = int(end_landmark.y * h)
                        
                        # Subtle connection lines
                        cv2.line(frame, (start_x, start_y), (end_x, end_y), (120, 120, 120), 1)
        
        return frame


class ARHandController:
    """Main AR application for hand-controlled object manipulation"""
    
    def __init__(self):
        # Hand detector
        self.detector = HandGestureDetector()
        
        # Virtual objects
        self.objects: List[VirtualObject] = []
        
        # Test mode for JARVIS without camera
        self.test_mode = False
        self.jarvis_activated = False
        
        # Interaction state
        self.grab_states = {}  # hand_idx -> {object, initial_pinch_distance, initial_size}
        self.grab_states_3d = {}  # hand_idx -> {object, initial_pinch_distance, initial_size, last_hand_pos}
        self.last_frame_time = time.time()
        
        # Pinch state tracking for hysteresis
        self.pinch_states = {}  # hand_idx -> {'was_pinching': bool, 'pinch_frames': int}
        
        # Selection system tracking (two-hand selection)
        self.selection_mode_objects = set()  # Set of objects currently in selection mode (grabbed by 2 hands)
        
        # Explicit rotation mode tracking (simplified)
        # No longer needed - using explicit rotation state in VirtualObject3D
        
        # Display mode
        self.show_3d_objects = True
        self.show_2d_objects = True
        
        # Create some initial objects
        self._create_initial_objects()
        
    def _create_initial_objects(self):
        """Create initial virtual objects"""
        # 2D objects
        self.objects = [
            VirtualObject(200, 200, 60, (0, 255, 255), "circle"),  # Yellow ball
            VirtualObject(400, 300, 80, (255, 100, 100), "cube"),   # Blue cube
            VirtualObject(600, 250, 50, (100, 255, 100), "circle"), # Green ball
        ]
        
        # Initialize 3D objects list - will be populated with individual components from assemblies
        self.objects_3d = []
        
        # Initialize assemblies list
        self.assemblies = []
        
        # Load all objects through the unified CAD Assembly pipeline
        # This handles both single objects and multi-component assemblies
        models_to_load = [
            # {
            #     "path": "online/Wooden Crate.obj",
            #     "name": "Wood Crate",
            #     "position": (2.0, 0.0, -4.0),
            #     "scale": 0.7,
            #     "color": (139, 69, 19)  # Brown color for wood
            # },
            # {
            #     "path": "online/valve.obj",
            #     "name": "Valve",
            #     "position": (-2.0, 1.0, -4.0),
            #     "scale": 0.5,
            #     "color": (192, 192, 192)  # Silver/gray color for metal valve
            # },
            {
                "path": "online/duck.obj",
                "name": "duck",
                "position": (0.0, 0.0, -3.0),
                "scale": 0.8,
                "color": (100, 200, 200)  # Blue color for assembly
            },
            # {
            #     "path": "online/Lowpoly_tree_sample.obj",
            #     "name": "Tree",
            #     "position": (0.0, -1.0, -5.0),
            #     "scale": 1.0,
            #     "color": (34, 139, 34)  # Forest green for tree
            # }
        ]
        
        # Load each model through the CADAssembly pipeline
        for model_info in models_to_load:
            model_path = os.path.join(os.path.dirname(__file__), model_info["path"])
            if os.path.exists(model_path):
                try:
                    # Create assembly (works for both single objects and multi-component assemblies)
                    assembly = CADAssembly(
                        model_path,
                        x=model_info["position"][0],
                        y=model_info["position"][1], 
                        z=model_info["position"][2],
                        scale=model_info["scale"],
                        color=model_info["color"],
                        name=model_info["name"]
                    )
                    
                    # Add assembly to list
                    self.assemblies.append(assembly)
                    
                    # Add all components to objects_3d for individual manipulation
                    for component in assembly.get_all_components():
                        component.set_render_mode("solid")
                        # Set different auto-rotation speeds for variety
                        component.auto_rotation_speed = 0.01 + len(self.objects_3d) * 0.005
                        self.objects_3d.append(component)
                    
                    print(f"✅ Loaded {assembly.get_assembly_info()}")
                    
                except Exception as e:
                    print(f"⚠️  Error loading {model_info['name']}: {e}")
            else:
                print(f"⚠️  {model_info['name']} not found at {model_path}")
    
    def start(self):
        """Start the AR application"""
        # Try different camera indices with better error handling
        camera_indices = [0, 1, 2]  # Try multiple camera indices
        
        for index in camera_indices:
            print(f"Trying camera index {index}...")
            self.cap = cv2.VideoCapture(index)
            
            # Set camera properties for better compatibility
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            if self.cap.isOpened():
                # Test if we can actually read from the camera
                ret, test_frame = self.cap.read()
                if ret and test_frame is not None:
                    print(f"✅ Successfully opened camera {index}")
                    print(f"   Resolution: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
                    print(f"   FPS: {int(self.cap.get(cv2.CAP_PROP_FPS))}")
                    break
                else:
                    print(f"❌ Camera {index} opened but cannot read frames")
                    self.cap.release()
            else:
                print(f"❌ Could not open camera {index}")
        else:
            print("❌ Error: Could not open any camera")
            print("\n🔧 Troubleshooting tips:")
            print("1. Check System Preferences → Privacy & Security → Camera")
            print("2. Allow Terminal (or your Python IDE) to access the camera")
            print("3. Make sure no other applications are using the camera")
            print("4. Try running with different backend: 'export OPENCV_VIDEOIO_PRIORITY_LIST=AVFOUNDATION'")
            print("5. Try running: 'python3.11 launch_app.py' for GUI launcher")
            
            # Continue without camera for JARVIS testing
            print("\n🤖 JARVIS Integration Test Mode:")
            print("- Camera disabled, but JARVIS voice assistant still works")
            print("- Press 'J' to test JARVIS activation")
            print("- Use synthetic 3D objects for JARVIS visual analysis")
            
            self.test_mode = True
            self.cap = None
            
        # Set camera resolution if camera is available
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Initialize 3D renderer
        self.renderer_3d = Renderer3D(1280, 720)
        
        self.is_running = True
        print("AR Hand Control started!")
        print("Features:")
        print("- Multi-component CAD assembly support")
        print("- Individual component manipulation")
        print("- Each component can be moved, rotated, and scaled independently")
        print("- Multi-axis 3D rotation (Yaw, Pitch, Roll)")
        print("Gestures:")
        print("- Pinch (thumb + index) near object: Grab object or component")
        print("- Move hand while pinching: Move object/component (improved 3D tracking!)")
        print("- Grab object with TWO hands and move apart/closer: Scale object/component")
        print("- Two hands then release one: Enter multi-axis rotation mode")
        print("- In rotation mode:")
        print("  • Move horizontally: Yaw rotation (Y-axis)")
        print("  • Move vertically: Pitch rotation (X-axis)")
        print("  • Move diagonally: Roll rotation (Z-axis)")
        print("Controls:")
        print("- Press 'q' to quit")
        print("- Press 'r' to reset objects")
        print("- Press 'c' to add new 2D object")
        print("- Press '1' to toggle 2D objects")
        print("- Press '2' to toggle 3D objects")
        print("- Press 'w' to toggle wireframe/solid rendering")
        print("- Press 't' to toggle auto-rotation")
        print("- Press 'x/y/z' to reset rotation on specific axis")
        print("- Press 'space' to cycle through 3D objects")
        print("- Press 'j' to activate JARVIS voice assistant")
        
        while self.is_running:
            self._process_frame()
            
        self._cleanup()
    
    def _create_test_frame(self):
        """Create a synthetic frame for testing JARVIS without camera"""
        # Create a black frame with 3D objects rendered
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Add some visual elements to make it interesting for JARVIS
        cv2.putText(frame, "AR Hand Control - JARVIS Test Mode", 
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        if self.jarvis_activated:
            cv2.putText(frame, "JARVIS ACTIVE - Voice Assistant Ready", 
                       (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, "Say 'What is this?' to analyze 3D objects", 
                       (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Add instructions
        cv2.putText(frame, "Press 'J' to toggle JARVIS", 
                   (50, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(frame, "Press '2' to show 3D objects", 
                   (50, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        return frame
    
    def save_screenshot_for_jarvis(self, frame):
        """Save current frame as screenshot for JARVIS analysis"""
        try:
            screenshot_path = "jarvis_screenshot.png"
            cv2.imwrite(screenshot_path, frame)
            print(f"📸 Screenshot saved for JARVIS analysis: {screenshot_path}")
            return screenshot_path
        except Exception as e:
            print(f"❌ Failed to save screenshot: {e}")
            return None
    
    def _process_frame(self):
        """Process a single frame"""
        if self.test_mode or self.cap is None:
            # Create synthetic frame for JARVIS testing
            frame = self._create_test_frame()
        else:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                self.is_running = False
                return
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect hands
        hands_info = self.detector.detect_hands(frame)
        
        # Process interactions
        self._process_interactions(hands_info)
        
        # Store hands_info for scaling calculations
        self.current_hands_info = hands_info
        
        # Draw 2D objects
        if self.show_2d_objects:
            for obj in self.objects:
                frame = obj.draw(frame)
        
        # Draw 3D objects
        if self.show_3d_objects and self.renderer_3d:
            for i, obj_3d in enumerate(self.objects_3d):
                # Only show pinchable radius for the first two 3D objects
                if i < 2:
                    obj_3d.selected = (i == 0)  # First object is selected (yellow), second is not (blue)
                else:
                    obj_3d.selected = None  # Hide radius for objects beyond the first two
                frame = obj_3d.draw(frame, self.renderer_3d)
                
                # Draw elegant targeting indicator for 3D objects when pinching near them
                for hand_info in hands_info:
                    if (hand_info['is_pinching'] and 
                        not obj_3d.is_grabbed and 
                        obj_3d.is_point_inside(hand_info['pinch_center'][0], hand_info['pinch_center'][1], self.renderer_3d)):
                        
                        screen_pos = obj_3d.get_screen_position(self.renderer_3d)
                        if screen_pos:
                            self._draw_elegant_targeting_indicator(frame, screen_pos, hand_info['pinch_center'])
            
        # Draw hand landmarks
        frame = self.detector.draw_landmarks(frame, hands_info)
        
        # Draw UI
        frame = self._draw_ui(frame, hands_info)
        
        # Draw rotation hand indicators
        frame = self._draw_rotation_hand_indicators(frame, hands_info)
        
        # Show frame
        cv2.imshow('AR Hand Control', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.is_running = False
        elif key == ord('r'):
            self._create_initial_objects()
            self.grab_states.clear()
            self.grab_states_3d.clear()
            print("Reset all objects and assemblies")
        elif key == ord('c'):
            self._add_random_object()
        elif key == ord('1'):
            self.show_2d_objects = not self.show_2d_objects
            print(f"2D objects: {'ON' if self.show_2d_objects else 'OFF'}")
        elif key == ord('2'):
            self.show_3d_objects = not self.show_3d_objects
            print(f"3D objects: {'ON' if self.show_3d_objects else 'OFF'}")
        elif key == ord('w'):
            # Toggle wireframe/solid for 3D objects
            for obj_3d in self.objects_3d:
                if obj_3d.render_mode == "solid":
                    obj_3d.set_render_mode("wireframe")
                else:
                    obj_3d.set_render_mode("solid")
            print(f"3D render mode: {self.objects_3d[0].render_mode if self.objects_3d else 'N/A'}")
        elif key == ord('t'):
            # Toggle auto-rotation for 3D objects
            for obj_3d in self.objects_3d:
                obj_3d.toggle_auto_rotation()
            auto_rotate_status = self.objects_3d[0].auto_rotate if self.objects_3d else False
            print(f"Auto-rotation: {'ON' if auto_rotate_status else 'OFF'}")
        elif key == ord('x'):
            # Reset X-axis rotation for all 3D objects
            for obj_3d in self.objects_3d:
                obj_3d.rotation_x = 0.0
            print("Reset X-axis rotation")
        elif key == ord('y'):
            # Reset Y-axis rotation for all 3D objects
            for obj_3d in self.objects_3d:
                obj_3d.rotation_y = 0.0
            print("Reset Y-axis rotation")
        elif key == ord('z'):
            # Reset Z-axis rotation for all 3D objects
            for obj_3d in self.objects_3d:
                obj_3d.rotation_z = 0.0
            print("Reset Z-axis rotation")
        elif key == ord('j'):
            # Activate JARVIS voice assistant
            print("🤖 Activating JARVIS voice assistant...")
            self.jarvis_activated = not self.jarvis_activated
            if self.jarvis_activated:
                print("✅ JARVIS activated - Voice assistant ready")
                print("🗣️  Say 'What is this?' to analyze the 3D objects")
                print("📸 JARVIS will take a screenshot and analyze what you're looking at")
                
                # Save screenshot for JARVIS analysis
                screenshot_path = self.save_screenshot_for_jarvis(frame)
                if screenshot_path:
                    print("🧠 Screenshot ready for JARVIS vision analysis")
                    print("💡 Open the web interface and activate JARVIS to analyze this image")
                
                cv2.putText(frame, "JARVIS ACTIVATED - Voice Assistant Ready", 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Screenshot saved for analysis", 
                           (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            else:
                print("🤖 JARVIS deactivated")
                cv2.putText(frame, "JARVIS DEACTIVATED", 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif key == ord(' '):  # Spacebar
            # Cycle through 3D objects (select next one)
            if self.objects_3d:
                # Find currently selected object or start with first
                current_selected = -1
                for i, obj_3d in enumerate(self.objects_3d):
                    if obj_3d.selected and not obj_3d.is_grabbed:  # Don't cycle away from grabbed objects
                        current_selected = i
                        obj_3d.selected = False
                        break
                
                # Select next object (skip grabbed objects)
                next_index = (current_selected + 1) % len(self.objects_3d)
                attempts = 0
                while self.objects_3d[next_index].is_grabbed and attempts < len(self.objects_3d):
                    next_index = (next_index + 1) % len(self.objects_3d)
                    attempts += 1
                
                self.objects_3d[next_index].selected = True
                print(f"Selected 3D object {next_index + 1}/{len(self.objects_3d)}")
        elif key == ord('s'):  # 'S' key for cycling 2D objects
            # Cycle through 2D objects (select next one)
            if self.objects:
                # Find currently selected object or start with first
                current_selected = -1
                for i, obj in enumerate(self.objects):
                    if obj.selected and not obj.is_grabbed:  # Don't cycle away from grabbed objects
                        current_selected = i
                        obj.selected = False
                        break
                
                # Select next object (skip grabbed objects)
                next_index = (current_selected + 1) % len(self.objects)
                attempts = 0
                while self.objects[next_index].is_grabbed and attempts < len(self.objects):
                    next_index = (next_index + 1) % len(self.objects)
                    attempts += 1
                
                self.objects[next_index].selected = True
                print(f"Selected 2D object {next_index + 1}/{len(self.objects)}")
    
    def _process_interactions(self, hands_info: List[dict]):
        """Process hand interactions with objects"""
        current_grabs = set()
        current_grabs_3d = set()
        
        for hand_info in hands_info:
            hand_idx = hand_info['hand_idx']
            is_pinching = hand_info['is_pinching']
            
            # Initialize pinch state if not exists
            if hand_idx not in self.pinch_states:
                self.pinch_states[hand_idx] = {'was_pinching': False, 'pinch_frames': 0}
            
            pinch_state = self.pinch_states[hand_idx]
            
            # Minimal hysteresis: require 1 frame to start, 1 frame to stop
            if is_pinching:
                pinch_state['pinch_frames'] += 1
                if pinch_state['pinch_frames'] >= 1:  # Start pinching after 1 frame
                    pinch_state['was_pinching'] = True
            else:
                pinch_state['pinch_frames'] = 0  # Immediately reset when not pinching
                pinch_state['was_pinching'] = False  # Immediately stop pinching
            
            # Use the stabilized pinch state
            stabilized_pinching = pinch_state['was_pinching']
            
            if stabilized_pinching:
                # Try 3D objects first
                if self.show_3d_objects and self._handle_pinch_interaction_3d(hand_info, hand_idx):
                    current_grabs_3d.add(hand_idx)
                # Then try 2D objects
                elif self.show_2d_objects:
                    self._handle_pinch_interaction(hand_info, hand_idx)
                    current_grabs.add(hand_idx)
            
            else:
                # Release any grabbed 2D object
                if hand_idx in self.grab_states:
                    obj = self.grab_states[hand_idx]['object']
                    # Remove this hand from the grabbed_by_hand list
                    if hand_idx in obj.grabbed_by_hand:
                        obj.grabbed_by_hand.remove(hand_idx)
                    # Update grab state based on remaining hands
                    obj.is_grabbed = len(obj.grabbed_by_hand)
                    # Deselect object when completely released
                    if obj.is_grabbed == 0:
                        obj.selected = False
                    
                    # Reset scaling state if transitioning from two-hand to one-hand or no hands
                    if obj.is_grabbed < 2:
                        # Clear two-hand scaling state from all hands grabbing this object
                        for other_hand_idx in obj.grabbed_by_hand:
                            if other_hand_idx in self.grab_states:
                                other_hand_state = self.grab_states[other_hand_idx]
                                if 'initial_two_hand_distance' in other_hand_state:
                                    del other_hand_state['initial_two_hand_distance']
                                if 'initial_size' in other_hand_state:
                                    del other_hand_state['initial_size']
                    
                    del self.grab_states[hand_idx]
                
                # Release any grabbed 3D object
                if hand_idx in self.grab_states_3d:
                    obj_3d = self.grab_states_3d[hand_idx]['object']
                    grab_state = self.grab_states_3d[hand_idx]
                    
                    # Check if this release would create a rotation mode opportunity
                    # (transitioning from 2-hand to 1-hand grab)
                    if obj_3d.is_grabbed == 2 and len(obj_3d.grabbed_by_hand) == 2:
                        # This is a 2-hand grab, check if we should enter rotation mode
                        other_hand_idx = None
                        for other_hand in obj_3d.grabbed_by_hand:
                            if other_hand != hand_idx:
                                other_hand_idx = other_hand
                                break
                        
                        if other_hand_idx is not None:
                            # Check if the other hand is still pinching
                            other_hand_pinching = any(hand['hand_idx'] == other_hand_idx and hand['is_pinching'] for hand in hands_info)
                            
                            if other_hand_pinching:
                                # Other hand is still pinching - enter rotation mode with released hand
                                obj_3d.is_in_rotation_mode = True
                                obj_3d.rotation_hand_idx = hand_idx
                                
                                # Get initial position of rotation hand
                                rotation_hand_info = next((hand for hand in hands_info if hand['hand_idx'] == hand_idx), None)
                                if rotation_hand_info:
                                    obj_3d.last_rotation_hand_pos = rotation_hand_info['palm_center']
                                
                                print(f"Object entered multi-axis rotation mode - Hand {hand_idx} controlling rotation")
                                
                                # Remove this hand from grabbed_by_hand but keep it in grab_states_3d for rotation tracking
                                if hand_idx in obj_3d.grabbed_by_hand:
                                    obj_3d.grabbed_by_hand.remove(hand_idx)
                                obj_3d.is_grabbed = len(obj_3d.grabbed_by_hand)
                                
                                # Clear scaling state
                                if 'initial_two_hand_distance' in grab_state:
                                    del grab_state['initial_two_hand_distance']
                                if 'initial_scale' in grab_state:
                                    del grab_state['initial_scale']
                                
                                # Don't delete from grab_states_3d - keep for rotation tracking
                                continue
                    
                    # Normal release logic (not transitioning to rotation mode)
                    # Remove this hand from the grabbed_by_hand list
                    if hand_idx in obj_3d.grabbed_by_hand:
                        obj_3d.grabbed_by_hand.remove(hand_idx)
                    # Update grab state based on remaining hands
                    obj_3d.is_grabbed = len(obj_3d.grabbed_by_hand)
                    
                    # Handle rotation mode exit if this hand was controlling rotation
                    if obj_3d.is_in_rotation_mode and obj_3d.rotation_hand_idx == hand_idx:
                        self._exit_rotation_mode(obj_3d)
                    
                    # Deselect object when completely released
                    if obj_3d.is_grabbed == 0:
                        obj_3d.selected = False
                        obj_3d.is_in_rotation_mode = False
                        obj_3d.rotation_hand_idx = None
                        obj_3d.last_rotation_hand_pos = None
                    
                    # Reset scaling state if transitioning from two-hand to one-hand or no hands
                    if obj_3d.is_grabbed < 2:
                        # Clear two-hand scaling state from all hands grabbing this object
                        for other_hand_idx in obj_3d.grabbed_by_hand:
                            if other_hand_idx in self.grab_states_3d:
                                other_hand_state = self.grab_states_3d[other_hand_idx]
                                if 'initial_two_hand_distance' in other_hand_state:
                                    del other_hand_state['initial_two_hand_distance']
                                if 'initial_scale' in other_hand_state:
                                    del other_hand_state['initial_scale']
                    
                    # Only delete if the hand is still in grab_states_3d (might have been removed by rotation mode logic)
                    if hand_idx in self.grab_states_3d:
                        del self.grab_states_3d[hand_idx]
        
        
        # Clean up grab states for hands that are no longer detected
        hands_to_remove = []
        for hand_idx in self.grab_states:
            if hand_idx not in current_grabs:
                obj = self.grab_states[hand_idx]['object']
                # Remove this hand from the grabbed_by_hand list
                if hand_idx in obj.grabbed_by_hand:
                    obj.grabbed_by_hand.remove(hand_idx)
                # Update grab state based on remaining hands
                obj.is_grabbed = len(obj.grabbed_by_hand)
                
                # Reset scaling state if transitioning from two-hand to one-hand or no hands
                if obj.is_grabbed < 2:
                    # Clear two-hand scaling state from all hands grabbing this object
                    for other_hand_idx in obj.grabbed_by_hand:
                        if other_hand_idx in self.grab_states:
                            other_hand_state = self.grab_states[other_hand_idx]
                            if 'initial_two_hand_distance' in other_hand_state:
                                del other_hand_state['initial_two_hand_distance']
                            if 'initial_size' in other_hand_state:
                                del other_hand_state['initial_size']
                
                hands_to_remove.append(hand_idx)
        
        for hand_idx in hands_to_remove:
            del self.grab_states[hand_idx]
            # Also clean up pinch state
            if hand_idx in self.pinch_states:
                del self.pinch_states[hand_idx]
        
        # Clean up 3D grab states
        hands_to_remove_3d = []
        for hand_idx in self.grab_states_3d:
            if hand_idx not in current_grabs_3d:
                obj_3d = self.grab_states_3d[hand_idx]['object']
                # Remove this hand from the grabbed_by_hand list
                if hand_idx in obj_3d.grabbed_by_hand:
                    obj_3d.grabbed_by_hand.remove(hand_idx)
                # Update grab state based on remaining hands
                obj_3d.is_grabbed = len(obj_3d.grabbed_by_hand)
                
                # Reset scaling state if transitioning from two-hand to one-hand or no hands
                if obj_3d.is_grabbed < 2:
                    # Clear two-hand scaling state from all hands grabbing this object
                    for other_hand_idx in obj_3d.grabbed_by_hand:
                        if other_hand_idx in self.grab_states_3d:
                            other_hand_state = self.grab_states_3d[other_hand_idx]
                            if 'initial_two_hand_distance' in other_hand_state:
                                del other_hand_state['initial_two_hand_distance']
                            if 'initial_scale' in other_hand_state:
                                del other_hand_state['initial_scale']
                
                hands_to_remove_3d.append(hand_idx)
        
        for hand_idx in hands_to_remove_3d:
            del self.grab_states_3d[hand_idx]
        
        # Handle selection mode interactions for 3D objects
        if self.show_3d_objects:
            self._handle_selection_mode_3d(hands_info)
        
        # Clean up rotation mode for objects that are no longer grabbed
        for obj_3d in self.objects_3d:
            if obj_3d.is_in_rotation_mode and obj_3d.is_grabbed == 0:
                self._exit_rotation_mode(obj_3d)
        
        # Clean up rotation mode for hands that are no longer detected
        for obj_3d in self.objects_3d:
            if obj_3d.is_in_rotation_mode and obj_3d.rotation_hand_idx is not None:
                # Check if rotation hand is still detected
                rotation_hand_detected = any(hand['hand_idx'] == obj_3d.rotation_hand_idx for hand in hands_info)
                if not rotation_hand_detected:
                    self._exit_rotation_mode(obj_3d)
        
        # Additional safety: release objects if no hands are detected
        if not hands_info:
            for obj in self.objects:
                if obj.is_grabbed > 0:
                    obj.is_grabbed = 0
                    obj.grabbed_by_hand = []
            for obj_3d in self.objects_3d:
                if obj_3d.is_grabbed > 0:
                    obj_3d.is_grabbed = 0
                    obj_3d.grabbed_by_hand = []
            self.grab_states.clear()
            self.grab_states_3d.clear()
            self.pinch_states.clear()
    
    def _handle_pinch_interaction(self, hand_info: dict, hand_idx: int):
        """Handle pinch gesture interaction with improved tracking"""
        pinch_center = hand_info['pinch_center']
        pinch_distance = hand_info['pinch_distance']
        
        if hand_idx not in self.grab_states:
            # Try to grab an object - use pinch center for better accuracy
            closest_obj = None
            closest_distance = float('inf')
            
            for obj in self.objects:
                if obj.is_grabbed < 2:  # Allow grabbing if not already grabbed by 2 hands
                    # Use actual object size for grab radius to match pinchable radius
                    grab_radius = obj.size
                    distance = math.sqrt((obj.x - pinch_center[0]) ** 2 + (obj.y - pinch_center[1]) ** 2)
                    if distance <= grab_radius and distance < closest_distance:
                        closest_distance = distance
                        closest_obj = obj
            
            if closest_obj:
                # Add this hand to the grabbed_by_hand list
                if hand_idx not in closest_obj.grabbed_by_hand:
                    closest_obj.grabbed_by_hand.append(hand_idx)
                # Update grab state based on number of hands
                closest_obj.is_grabbed = len(closest_obj.grabbed_by_hand)
                # Mark object as selected when grabbed
                closest_obj.selected = True
                self.grab_states[hand_idx] = {
                    'object': closest_obj,
                    'initial_pinch_distance': pinch_distance,
                    'initial_size': closest_obj.size,
                    'grab_offset_x': pinch_center[0] - closest_obj.x,
                    'grab_offset_y': pinch_center[1] - closest_obj.y
                }
        else:
            # Continue interaction with grabbed object
            grab_state = self.grab_states[hand_idx]
            obj = grab_state['object']
            
            # Move object with offset for natural feel
            target_x = pinch_center[0] - grab_state['grab_offset_x']
            target_y = pinch_center[1] - grab_state['grab_offset_y']
            obj.move_to(target_x, target_y)
            
            # Only scale if object is grabbed by two hands
            if obj.is_grabbed == 2:
                self._handle_two_hand_scaling_2d(obj)
    
    def _handle_two_hand_scaling_2d(self, obj):
        """Handle scaling when 2D object is grabbed by two hands"""
        # Get the two hands that are grabbing this object
        grabbing_hands = [hand_idx for hand_idx in obj.grabbed_by_hand if hand_idx in self.grab_states]
        
        if len(grabbing_hands) == 2:
            hand1_idx, hand2_idx = grabbing_hands[0], grabbing_hands[1]
            hand1_state = self.grab_states[hand1_idx]
            hand2_state = self.grab_states[hand2_idx]
            
            # Get current hand positions from the detector
            hand1_info = None
            hand2_info = None
            
            for hand_info in self.current_hands_info:
                if hand_info['hand_idx'] == hand1_idx:
                    hand1_info = hand_info
                elif hand_info['hand_idx'] == hand2_idx:
                    hand2_info = hand_info
            
            if hand1_info and hand2_info:
                # Calculate current distance between the two hands using pinch centers
                hand1_pos = hand1_info['pinch_center']
                hand2_pos = hand2_info['pinch_center']
                current_distance = math.sqrt((hand2_pos[0] - hand1_pos[0]) ** 2 + (hand2_pos[1] - hand1_pos[1]) ** 2)
                
                # Get initial distance when two-hand grab started
                if 'initial_two_hand_distance' not in hand1_state:
                    # Initialize the two-hand scaling
                    hand1_state['initial_two_hand_distance'] = current_distance
                    hand1_state['initial_size'] = obj.size
                    hand2_state['initial_two_hand_distance'] = current_distance
                    hand2_state['initial_size'] = obj.size
                
                initial_distance = hand1_state['initial_two_hand_distance']
                
                # Calculate scale factor based on distance change
                if initial_distance > 0:
                    scale_factor = current_distance / initial_distance
                    target_size = hand1_state['initial_size'] * scale_factor
                    
                    # Apply scaling with smoothing
                    obj.size = obj.size * 0.7 + target_size * 0.3  # Smooth scaling
                    obj.size = max(20, min(200, obj.size))  # Clamp size
    
    
    def _handle_selection_mode_3d(self, hands_info: List[dict]):
        """Handle selection mode interactions for 3D objects (two-hand selection)"""
        # Update selection mode objects based on current grab states
        self._update_selection_mode_objects()
        
        # Handle interactions for objects in selection mode
        for obj_3d in self.selection_mode_objects:
            self._handle_selection_mode_interactions(obj_3d, hands_info)
        
        # Handle rotation mode for objects that transitioned from 2-hand to 1-hand grabbing
        self._handle_rotation_mode_3d(hands_info)
    
    def _update_selection_mode_objects(self):
        """Update which objects are in selection mode (grabbed by exactly 2 hands)"""
        # Clear current selection mode
        self.selection_mode_objects.clear()
        
        # Add objects that are grabbed by exactly 2 hands
        for obj_3d in self.objects_3d:
            if obj_3d.is_grabbed == 2:
                self.selection_mode_objects.add(obj_3d)
                obj_3d.is_selected = True  # Mark as selected for visual feedback
            else:
                obj_3d.is_selected = False  # Clear selection state
    
    def _handle_selection_mode_interactions(self, obj_3d, hands_info: List[dict]):
        """Handle scaling for objects in selection mode (2 hands grabbing)"""
        # Get the two hands that are currently grabbing this object
        grabbing_hands = [hand_idx for hand_idx in obj_3d.grabbed_by_hand if hand_idx in self.grab_states_3d]
        
        if len(grabbing_hands) == 2:
            hand1_idx, hand2_idx = grabbing_hands[0], grabbing_hands[1]
            
            # Check if both hands are still pinching (scaling mode)
            hand1_pinching = any(hand['hand_idx'] == hand1_idx and hand['is_pinching'] for hand in hands_info)
            hand2_pinching = any(hand['hand_idx'] == hand2_idx and hand['is_pinching'] for hand in hands_info)
            
            if hand1_pinching and hand2_pinching:
                # Both hands pinching - scaling mode
                self._handle_selection_scaling(obj_3d, hand1_idx, hand2_idx, hands_info)
            elif hand1_pinching or hand2_pinching:
                # One hand released - transition to rotation mode
                self._transition_to_rotation_mode(obj_3d, hand1_idx, hand2_idx, hands_info)
    
    def _handle_selection_scaling(self, obj_3d, hand1_idx: int, hand2_idx: int, hands_info: List[dict]):
        """Handle scaling when both hands are pinching in selection mode"""
        # Get hand positions
        hand1_info = next((hand for hand in hands_info if hand['hand_idx'] == hand1_idx), None)
        hand2_info = next((hand for hand in hands_info if hand['hand_idx'] == hand2_idx), None)
        
        if hand1_info and hand2_info:
            # Calculate current distance between the two hands
            hand1_pos = hand1_info['pinch_center']
            hand2_pos = hand2_info['pinch_center']
            current_distance = math.sqrt((hand2_pos[0] - hand1_pos[0]) ** 2 + (hand2_pos[1] - hand1_pos[1]) ** 2)
            
            # Get initial distance when two-hand grab started
            hand1_state = self.grab_states_3d[hand1_idx]
            if 'initial_two_hand_distance' not in hand1_state:
                hand1_state['initial_two_hand_distance'] = current_distance
                hand1_state['initial_scale'] = obj_3d.scale
            
            initial_distance = hand1_state['initial_two_hand_distance']
            
            # Calculate scale factor based on distance change
            if initial_distance > 0:
                scale_factor = current_distance / initial_distance
                target_scale = hand1_state['initial_scale'] * scale_factor
                
                # Apply scaling with smoothing
                obj_3d.scale = obj_3d.scale * 0.7 + target_scale * 0.3  # Smooth scaling
                obj_3d.scale = max(0.1, min(5.0, obj_3d.scale))  # Clamp scale
    
    
    def _transition_to_rotation_mode(self, obj_3d, hand1_idx: int, hand2_idx: int, hands_info: List[dict]):
        """Transition object to rotation mode when one hand releases from 2-hand grab"""
        # Determine which hand is still pinching and which is released
        hand1_pinching = any(hand['hand_idx'] == hand1_idx and hand['is_pinching'] for hand in hands_info)
        hand2_pinching = any(hand['hand_idx'] == hand2_idx and hand['is_pinching'] for hand in hands_info)
        
        if hand1_pinching and not hand2_pinching:
            # Hand1 still pinching, Hand2 released - Hand2 controls rotation
            rotation_hand = hand2_idx
        elif hand2_pinching and not hand1_pinching:
            # Hand2 still pinching, Hand1 released - Hand1 controls rotation
            rotation_hand = hand1_idx
        else:
            # Both hands released or both still pinching - no rotation mode
            return
        
        # Set rotation mode
        obj_3d.is_in_rotation_mode = True
        obj_3d.rotation_hand_idx = rotation_hand
        
        # Get initial position of rotation hand
        rotation_hand_info = next((hand for hand in hands_info if hand['hand_idx'] == rotation_hand), None)
        if rotation_hand_info:
            obj_3d.last_rotation_hand_pos = rotation_hand_info['palm_center']
        
        print(f"Object entered multi-axis rotation mode - Hand {rotation_hand} controlling rotation")
    
    def _handle_rotation_mode_3d(self, hands_info: List[dict]):
        """Handle rotation mode for 3D objects with multi-axis support"""
        for obj_3d in self.objects_3d:
            if obj_3d.is_in_rotation_mode and obj_3d.rotation_hand_idx is not None:
                # Get current position of rotation hand
                rotation_hand_info = next((hand for hand in hands_info if hand['hand_idx'] == obj_3d.rotation_hand_idx), None)
                
                if rotation_hand_info and obj_3d.last_rotation_hand_pos is not None:
                    current_pos = rotation_hand_info['palm_center']
                    last_pos = obj_3d.last_rotation_hand_pos
                    
                    # Calculate movement in both axes
                    delta_x = current_pos[0] - last_pos[0]  # Horizontal movement
                    delta_y = current_pos[1] - last_pos[1]  # Vertical movement
                    
                    rotation_sensitivity = 0.01  # Base rotation speed
                    movement_threshold = 5  # Minimum movement threshold
                    
                    # Apply Y-axis rotation based on horizontal movement (yaw)
                    if abs(delta_x) > movement_threshold:
                        # Invert rotation direction: left movement = clockwise, right movement = counter-clockwise
                        obj_3d.rotation_y -= delta_x * rotation_sensitivity
                        
                        # Keep rotation in reasonable range
                        obj_3d.rotation_y = obj_3d.rotation_y % (2 * math.pi)
                    
                    # Apply X-axis rotation based on vertical movement (pitch)
                    if abs(delta_y) > movement_threshold:
                        # Natural direction: up movement = rotate up, down movement = rotate down
                        obj_3d.rotation_x -= delta_y * rotation_sensitivity
                        
                        # Keep rotation in reasonable range
                        obj_3d.rotation_x = obj_3d.rotation_x % (2 * math.pi)
                    
                    # Apply Z-axis rotation based on diagonal movement (roll)
                    # Calculate diagonal movement magnitude and direction
                    movement_magnitude = math.sqrt(delta_x**2 + delta_y**2)
                    if movement_magnitude > movement_threshold:
                        # Use the cross product concept - if moving diagonally, apply roll
                        # This creates roll when moving in diagonal directions
                        diagonal_threshold = movement_threshold * 1.5  # Require more diagonal movement
                        
                        if abs(delta_x) > movement_threshold and abs(delta_y) > movement_threshold:
                            # Diagonal movement detected - apply roll
                            # Roll direction based on diagonal quadrant
                            if (delta_x > 0 and delta_y > 0) or (delta_x < 0 and delta_y < 0):
                                # Top-right or bottom-left diagonal - positive roll
                                roll_factor = movement_magnitude * 0.005  # Smaller sensitivity for roll
                            else:
                                # Top-left or bottom-right diagonal - negative roll
                                roll_factor = -movement_magnitude * 0.005
                            
                            obj_3d.rotation_z += roll_factor
                            obj_3d.rotation_z = obj_3d.rotation_z % (2 * math.pi)
                    
                    # Update last position
                    obj_3d.last_rotation_hand_pos = current_pos
                else:
                    # Hand not detected or no last position - exit rotation mode
                    self._exit_rotation_mode(obj_3d)
    
    def _exit_rotation_mode(self, obj_3d):
        """Exit rotation mode for an object"""
        # Clean up the rotation hand's grab state if it exists
        if obj_3d.rotation_hand_idx is not None and obj_3d.rotation_hand_idx in self.grab_states_3d:
            del self.grab_states_3d[obj_3d.rotation_hand_idx]
        
        obj_3d.is_in_rotation_mode = False
        obj_3d.rotation_hand_idx = None
        obj_3d.last_rotation_hand_pos = None
        print(f"Object exited multi-axis rotation mode")
                
    
    def _handle_pinch_interaction_3d(self, hand_info: dict, hand_idx: int) -> bool:
        """Handle pinch gesture interaction with 3D objects with improved tracking"""
        pinch_center = hand_info['pinch_center']
        pinch_distance = hand_info['pinch_distance']
        thumb_pos = hand_info['thumb_pos']
        index_pos = hand_info['index_pos']
        
        if hand_idx not in self.grab_states_3d:
            # Try to grab a 3D object
            closest_obj_3d = None
            closest_distance = float('inf')
            
            for obj_3d in self.objects_3d:
                if obj_3d.is_grabbed < 2 and obj_3d.is_point_inside(pinch_center[0], pinch_center[1], self.renderer_3d):
                    # For 3D objects, we use a simpler distance check
                    distance = 0  # If point is inside, it's the closest
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_obj_3d = obj_3d
            
            if closest_obj_3d:
                # Add this hand to the grabbed_by_hand list
                if hand_idx not in closest_obj_3d.grabbed_by_hand:
                    closest_obj_3d.grabbed_by_hand.append(hand_idx)
                # Update grab state based on number of hands
                closest_obj_3d.is_grabbed = len(closest_obj_3d.grabbed_by_hand)
                # Mark object as selected when grabbed
                closest_obj_3d.selected = True
                
                # Get the current 3D object's screen position for calculating grab offset
                current_screen_pos = closest_obj_3d.get_screen_position(self.renderer_3d)
                
                self.grab_states_3d[hand_idx] = {
                    'object': closest_obj_3d,
                    'initial_pinch_distance': pinch_distance,
                    'initial_scale': closest_obj_3d.scale,
                    'initial_world_pos': (closest_obj_3d.x, closest_obj_3d.y, closest_obj_3d.z),
                    'grab_offset_x': pinch_center[0] - current_screen_pos[0] if current_screen_pos else 0,
                    'grab_offset_y': pinch_center[1] - current_screen_pos[1] if current_screen_pos else 0,
                    'last_pinch_center': pinch_center,
                    'last_thumb_pos': thumb_pos,
                    'last_index_pos': index_pos,
                    'movement_sensitivity': 0.015,  # Controls how much screen movement affects 3D position (increased for better responsiveness)
                    'initial_rotation': (closest_obj_3d.rotation_x, closest_obj_3d.rotation_y, closest_obj_3d.rotation_z),
                    'rotation_sensitivity': 0.01  # Controls rotation sensitivity
                }
                return True
        else:
            # Continue interaction with grabbed 3D object
            grab_state = self.grab_states_3d[hand_idx]
            obj_3d = grab_state['object']
            
            # Only move the object if the hand has actually moved
            last_pinch_center = grab_state['last_pinch_center']
            pinch_delta_x = pinch_center[0] - last_pinch_center[0]
            pinch_delta_y = pinch_center[1] - last_pinch_center[1]
            
            # Calculate movement magnitude to determine if hand actually moved
            movement_magnitude = math.sqrt(pinch_delta_x**2 + pinch_delta_y**2)
            movement_threshold = 0.5  # pixels - minimum movement to trigger object movement (reduced for better responsiveness)
            
            if movement_magnitude > movement_threshold:
                # Convert screen movement to world space movement
                sensitivity = grab_state['movement_sensitivity']
                world_delta_x = pinch_delta_x * sensitivity
                world_delta_y = -pinch_delta_y * sensitivity  # Invert Y for correct direction
                
                # Apply movement directly without smoothing for immediate response
                obj_3d.x += world_delta_x
                obj_3d.y += world_delta_y
            
            
            # Track finger positions for better interaction feedback
            grab_state['last_pinch_center'] = pinch_center
            grab_state['last_thumb_pos'] = thumb_pos
            grab_state['last_index_pos'] = index_pos
            
            return True
        
        return False
    
    
    def _add_random_object(self):
        """Add a new random object"""
        import random
        x = random.randint(100, 600)
        y = random.randint(100, 400)
        size = random.randint(40, 80)
        color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        shape = random.choice(["circle", "cube"])
        
        self.objects.append(VirtualObject(x, y, size, color, shape))
    
    def _draw_ui(self, frame: np.ndarray, hands_info: List[dict]) -> np.ndarray:
        """Draw Apple Vision Pro-inspired user interface elements"""
        h, w = frame.shape[:2]
        
        # Apple Vision Pro inspired color palette
        vision_blue = (255, 200, 120)      # Light blue accent (BGR)
        vision_silver = (200, 200, 200)    # Silver/white text
        vision_glass = (80, 80, 80)        # Glass overlay base
        vision_accent = (255, 150, 0)      # Orange/blue accent
        
        # Draw minimal header with glassmorphism effect
        self._draw_glass_panel(frame, (20, 15), (w - 40, 45), alpha=0.15)
        
        # Elegant title with modern typography feel
        title = "AR SPATIAL CONTROL"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        title_x = (w - title_size[0]) // 2
        cv2.putText(frame, title, (title_x, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, vision_silver, 2)
        
        # Subtle gesture hint
        hint = f"Pinch gestures • {len(hands_info)} hands detected"
        hint_size = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        hint_x = (w - hint_size[0]) // 2
        cv2.putText(frame, hint, (hint_x, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Draw floating status panel (bottom)
        self._draw_floating_status_panel(frame, hands_info)
        
        # Draw hand interaction indicators with elegance
        self._draw_elegant_hand_indicators(frame, hands_info)
        
        # Draw object count indicators in corners
        self._draw_corner_indicators(frame)
        
        return frame
    
    def _draw_glass_panel(self, frame: np.ndarray, top_left: tuple, size: tuple, alpha: float = 0.1):
        """Draw a glassmorphism-style panel"""
        x, y = top_left
        w, h = size
        
        # Create glass overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (40, 40, 40), -1)
        
        # Apply glass effect with subtle border
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)
    
    def _draw_floating_status_panel(self, frame: np.ndarray, hands_info: List[dict]):
        """Draw a floating status panel at the bottom"""
        h, w = frame.shape[:2]
        panel_w = 400
        panel_h = 60
        panel_x = (w - panel_w) // 2
        panel_y = h - panel_h - 20
        
        # Glass panel background
        self._draw_glass_panel(frame, (panel_x, panel_y), (panel_w, panel_h), alpha=0.2)
        
        # Status information with clean typography
        status_items = []
        
        # Object counts
        obj_2d_status = f"2D: {len(self.objects)}" + (" ●" if self.show_2d_objects else " ○")
        obj_3d_status = f"3D: {len(self.objects_3d)}" + (" ●" if self.show_3d_objects else " ○")
        
        # Render mode
        render_mode = self.objects_3d[0].render_mode if self.objects_3d else "N/A"
        
        status_text = f"{obj_2d_status}  |  {obj_3d_status}  |  Mode: {render_mode}"
        
        # Center the status text
        text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = panel_x + (panel_w - text_size[0]) // 2
        text_y = panel_y + 25
        
        cv2.putText(frame, status_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Quick controls hint
        controls_hint = "Q: Quit  •  R: Reset  •  C: Add  •  Space: Select"
        hint_size = cv2.getTextSize(controls_hint, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
        hint_x = panel_x + (panel_w - hint_size[0]) // 2
        hint_y = panel_y + 45
        
        cv2.putText(frame, controls_hint, (hint_x, hint_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    
    def _draw_elegant_hand_indicators(self, frame: np.ndarray, hands_info: List[dict]):
        """Draw elegant, minimal hand tracking indicators"""
        for hand_info in hands_info:
            hand_idx = hand_info['hand_idx']
            palm_pos = hand_info['palm_center']
            pinch_center = hand_info['pinch_center']
            is_pinching = hand_info['is_pinching']
            
            # Determine interaction state and colors
            stabilized_pinching = hand_idx in self.pinch_states and self.pinch_states[hand_idx]['was_pinching']
            
            if stabilized_pinching:
                if hand_idx in self.grab_states_3d:
                    # 3D object interaction - cyan theme
                    primary_color = (255, 200, 100)  # Light cyan
                    secondary_color = (200, 150, 50)  # Darker cyan
                    status_text = "3D"
                elif hand_idx in self.grab_states:
                    # 2D object interaction - blue theme  
                    primary_color = (255, 180, 120)  # Light blue
                    secondary_color = (200, 140, 80)  # Darker blue
                    status_text = "2D"
                else:
                    # Pinching but not grabbing
                    primary_color = (180, 180, 180)  # Neutral
                    secondary_color = (120, 120, 120)
                    status_text = "PINCH"
            else:
                # Hand detected but not pinching
                primary_color = (100, 100, 100)  # Subtle gray
                secondary_color = (60, 60, 60)
                status_text = "READY"
            
            # Draw elegant hand center indicator
            cv2.circle(frame, palm_pos, 12, secondary_color, 2)
            cv2.circle(frame, palm_pos, 6, primary_color, -1)
            
            # Draw pinch point with connection line
            if stabilized_pinching:
                # Active pinch visualization
                cv2.circle(frame, pinch_center, 8, primary_color, 2)
                cv2.circle(frame, pinch_center, 3, (255, 255, 255), -1)
                
                # Connection line between palm and pinch
                cv2.line(frame, palm_pos, pinch_center, primary_color, 1)
            else:
                # Subtle pinch point indicator
                cv2.circle(frame, pinch_center, 4, secondary_color, 1)
            
            # Floating status label with glass background
            label_w, label_h = 60, 25
            label_x = palm_pos[0] - label_w // 2
            label_y = palm_pos[1] - 35
            
            # Ensure label stays on screen
            label_x = max(5, min(frame.shape[1] - label_w - 5, label_x))
            label_y = max(25, min(frame.shape[0] - 5, label_y))
            
            # Glass background for label
            self._draw_glass_panel(frame, (label_x, label_y - label_h), (label_w, label_h), alpha=0.3)
            
            # Status text
            text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            text_x = label_x + (label_w - text_size[0]) // 2
            text_y = label_y - 8
            
            cv2.putText(frame, status_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, primary_color, 1)
    
    def _draw_corner_indicators(self, frame: np.ndarray):
        """Draw minimal corner indicators for system status"""
        h, w = frame.shape[:2]
        
        # Top-right: JARVIS status
        if hasattr(self, 'jarvis_activated') and self.jarvis_activated:
            cv2.circle(frame, (w - 30, 30), 8, (100, 255, 100), 2)  # Green for active
            cv2.circle(frame, (w - 30, 30), 4, (150, 255, 150), -1)
            cv2.putText(frame, "J", (w - 34, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 255, 200), 1)
        else:
            cv2.circle(frame, (w - 30, 30), 6, (100, 100, 100), 1)  # Subtle inactive
        
        # Top-left: Frame rate (if needed for debugging)
        if hasattr(self, 'last_frame_time'):
            current_time = time.time()
            fps = 1.0 / (current_time - self.last_frame_time) if current_time != self.last_frame_time else 0
            self.last_frame_time = current_time
            
            if fps > 20:  # Only show if performance is good
                fps_text = f"{int(fps)}"
                cv2.putText(frame, fps_text, (15, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    def _draw_elegant_targeting_indicator(self, frame: np.ndarray, target_pos: tuple, pinch_pos: tuple):
        """Draw Apple Vision Pro-inspired targeting indicator"""
        target_x, target_y = int(target_pos[0]), int(target_pos[1])
        pinch_x, pinch_y = pinch_pos
        
        # Apple Vision Pro inspired targeting colors
        primary_color = (255, 200, 120)  # Light blue (BGR)
        secondary_color = (200, 150, 80)  # Darker blue
        
        # Outer targeting ring with glassmorphism
        overlay = frame.copy()
        cv2.circle(overlay, (target_x, target_y), 50, secondary_color, -1)
        cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)
        
        # Main targeting rings
        cv2.circle(frame, (target_x, target_y), 45, primary_color, 2)
        cv2.circle(frame, (target_x, target_y), 30, primary_color, 1)
        
        # Center targeting dot
        cv2.circle(frame, (target_x, target_y), 4, (255, 255, 255), -1)
        cv2.circle(frame, (target_x, target_y), 2, primary_color, -1)
        
        # Elegant connection line with gradient effect
        # Draw multiple lines with decreasing opacity for gradient effect
        for i in range(3):
            alpha = 0.7 - i * 0.2
            line_color = tuple(int(c * alpha) for c in primary_color)
            thickness = max(1, 2 - i)  # Ensure thickness is always >= 1
            cv2.line(frame, (pinch_x, pinch_y), (target_x, target_y), line_color, thickness)
        
        # Add directional indicators
        import math
        distance = math.sqrt((target_x - pinch_x)**2 + (target_y - pinch_y)**2)
        if distance > 10:  # Avoid division by zero
            # Calculate direction vector
            dir_x = (target_x - pinch_x) / distance
            dir_y = (target_y - pinch_y) / distance
            
            # Draw small directional arrows along the line
            for t in [0.3, 0.7]:
                arrow_x = int(pinch_x + t * (target_x - pinch_x))
                arrow_y = int(pinch_y + t * (target_y - pinch_y))
                
                # Arrow head points
                arrow_size = 8
                arrow_x1 = int(arrow_x - arrow_size * dir_x + arrow_size * 0.5 * dir_y)
                arrow_y1 = int(arrow_y - arrow_size * dir_y - arrow_size * 0.5 * dir_x)
                arrow_x2 = int(arrow_x - arrow_size * dir_x - arrow_size * 0.5 * dir_y)
                arrow_y2 = int(arrow_y - arrow_size * dir_y + arrow_size * 0.5 * dir_x)
                
                cv2.line(frame, (arrow_x, arrow_y), (arrow_x1, arrow_y1), primary_color, 2)
                cv2.line(frame, (arrow_x, arrow_y), (arrow_x2, arrow_y2), primary_color, 2)
    
    def _draw_rotation_hand_indicators(self, frame: np.ndarray, hands_info: List[dict]) -> np.ndarray:
        """Draw elegant Apple Vision Pro-inspired rotation indicators"""
        for obj_3d in self.objects_3d:
            if obj_3d.is_in_rotation_mode and obj_3d.rotation_hand_idx is not None:
                # Find the rotation hand
                rotation_hand_info = next((hand for hand in hands_info if hand['hand_idx'] == obj_3d.rotation_hand_idx), None)
                
                if rotation_hand_info:
                    hand_center = rotation_hand_info['palm_center']
                    
                    # Apple Vision Pro inspired rotation colors
                    rotation_color = (255, 200, 100)  # Light cyan (BGR)
                    accent_color = (200, 150, 50)     # Darker variant
                    
                    # Draw elegant rotation indicator with glassmorphism
                    # Outer glow
                    overlay = frame.copy()
                    cv2.circle(overlay, hand_center, 35, accent_color, -1)
                    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
                    
                    # Main rotation ring
                    cv2.circle(frame, hand_center, 28, rotation_color, 3)
                    cv2.circle(frame, hand_center, 20, rotation_color, 1)
                    
                    # Central rotation symbol (circular arrows)
                    import math
                    # Draw circular rotation arrows
                    for angle_offset in [0, 180]:  # Two arrows opposite each other
                        start_angle = angle_offset
                        end_angle = angle_offset + 120
                        
                        # Calculate arc points
                        arc_radius = 15
                        num_points = 20
                        arc_points = []
                        
                        for i in range(num_points):
                            angle = math.radians(start_angle + (end_angle - start_angle) * i / (num_points - 1))
                            x = int(hand_center[0] + arc_radius * math.cos(angle))
                            y = int(hand_center[1] + arc_radius * math.sin(angle))
                            arc_points.append((x, y))
                        
                        # Draw arc
                        for i in range(len(arc_points) - 1):
                            cv2.line(frame, arc_points[i], arc_points[i + 1], rotation_color, 2)
                        
                        # Arrow head at the end
                        if len(arc_points) >= 2:
                            end_point = arc_points[-1]
                            prev_point = arc_points[-2]
                            
                            # Calculate arrow head direction
                            dx = end_point[0] - prev_point[0]
                            dy = end_point[1] - prev_point[1]
                            length = math.sqrt(dx*dx + dy*dy)
                            
                            if length > 0:
                                dx /= length
                                dy /= length
                                
                                # Arrow head points
                                arrow_size = 6
                                arrow_x1 = int(end_point[0] - arrow_size * dx + arrow_size * 0.5 * dy)
                                arrow_y1 = int(end_point[1] - arrow_size * dy - arrow_size * 0.5 * dx)
                                arrow_x2 = int(end_point[0] - arrow_size * dx - arrow_size * 0.5 * dy)
                                arrow_y2 = int(end_point[1] - arrow_size * dy + arrow_size * 0.5 * dx)
                                
                                cv2.line(frame, end_point, (arrow_x1, arrow_y1), rotation_color, 2)
                                cv2.line(frame, end_point, (arrow_x2, arrow_y2), rotation_color, 2)
                    
                    # Elegant label with glass background
                    label_text = "ROTATE"
                    label_w, label_h = 80, 25
                    label_x = hand_center[0] - label_w // 2
                    label_y = hand_center[1] - 55
                    
                    # Glass background for label
                    self._draw_glass_panel(frame, (label_x, label_y - label_h), (label_w, label_h), alpha=0.3)
                    
                    # Label text
                    text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    text_x = label_x + (label_w - text_size[0]) // 2
                    text_y = label_y - 8
                    
                    cv2.putText(frame, label_text, (text_x, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, rotation_color, 1)
        
        return frame
    
    def _cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


def main():
    """Main function to run the AR hand control application"""
    try:
        app = ARHandController()
        app.start()
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
