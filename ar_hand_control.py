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
        self.id = None  # Unique identifier for dock management 
        
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
                
                # Add handedness information if available
                if results.multi_handedness and hand_idx < len(results.multi_handedness):
                    handedness = results.multi_handedness[hand_idx]
                    hand_info['handedness'] = handedness.classification[0].label  # 'Left' or 'Right'
                    hand_info['handedness_score'] = handedness.classification[0].score
                else:
                    hand_info['handedness'] = 'Unknown'
                    hand_info['handedness_score'] = 0.0
                
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
        
        # Screenshot flag for JARVIS
        self.should_take_screenshot = False
        
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
        
        # Object visibility tracking for dock
        self.object_visibility = {}  # object_id -> bool
        self.object_isolation_mode = False  # True when only one object is shown
        self.isolated_object_id = None  # ID of the isolated object
        
        # Dock interaction state
        self.dock_interaction_state = {}  # hand_idx -> {'object_id': str, 'action': str, 'has_toggled': bool}
        self.dock_toggle_cooldown = {}  # object_id -> timestamp to prevent rapid toggling
        self.dock_pinch_states = {}  # hand_idx -> {'was_pinching_in_dock': bool, 'target_object_id': str}
        
        # Drag interaction state
        self.drag_states = {}  # hand_idx -> {'object_id': str, 'start_pos': tuple, 'current_pos': tuple, 'is_dragging': bool, 'dock_pos': tuple}
        self.drag_threshold = 30  # pixels to start dragging
        self.drag_out_threshold = 80  # pixels outside dock to activate model
        
        # Create some initial objects
        self._create_initial_objects()
        
    def _create_initial_objects(self):
        """Create initial virtual objects"""
        # 2D objects
        self.objects = [
            VirtualObject(200, 200, 60, (0, 255, 255), "circle"),  # Cyan ball
            VirtualObject(400, 300, 80, (255, 100, 100), "cube"),   # Red cube
            VirtualObject(600, 250, 50, (100, 255, 100), "circle"), # Green ball
        ]
        
        # Assign unique IDs to 2D objects - start all hidden
        for i, obj in enumerate(self.objects):
            obj.id = f"2d_{i}"
            self.object_visibility[obj.id] = False  # Start hidden
            # Move objects off-screen initially
            obj.x = -1000  # Way off screen
            obj.y = -1000
        
        # Initialize 3D objects list - will be populated with individual components from assemblies
        self.objects_3d = []
        
        # Initialize assemblies list
        self.assemblies = []
        
        # Load all objects through the unified CAD Assembly pipeline
        # This handles both single objects and multi-component assemblies
        models_to_load = [
            {
                "path": "online/Wooden Crate.obj",
                "name": "Wood Crate",
                "position": (2.0, 0.0, -4.0),
                "scale": 0.7,
                "color": (139, 69, 19)  # Brown color for wood
            },
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
                        # Disable auto-rotation completely - only rotate in explicit rotation mode
                        component.auto_rotate = False
                        component.auto_rotation_speed = 0.01 + len(self.objects_3d) * 0.005
                        # Assign unique ID for dock management - start hidden
                        component.id = f"3d_{len(self.objects_3d)}"
                        self.object_visibility[component.id] = False  # Start hidden
                        # Move 3D objects off-screen initially
                        component.x = -1000
                        component.y = -1000
                        component.z = -1000
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
        print("- Hand-specific single-axis 3D rotation")
        print("Gestures:")
        print("- Pinch (thumb + index) near object: Grab object or component")
        print("- Move hand while pinching: Move object/component (improved 3D tracking!)")
        print("- Grab object with TWO hands and move apart/closer: Scale object/component")
        print("- Two hands then release one: Enter hand-specific rotation mode")
        print("- In rotation mode:")
        print("  • RIGHT HAND: Controls X-axis rotation (pitch) - move up/down")
        print("  • LEFT HAND: Controls Y-axis rotation (yaw) - move left/right")
        print("  • Unknown hand: Falls back to multi-axis rotation")
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
    
    def _notify_jarvis_object_grabbed(self):
        """Notify JARVIS backend that an object has been grabbed"""
        try:
            import requests
            import threading
            
            def notify_background():
                try:
                    response = requests.post(
                        "http://localhost:5001/api/jarvis/object-grabbed",
                        json={"object_grabbed": True},
                        timeout=2
                    )
                    if response.status_code == 200:
                        print("✅ JARVIS notified of object grab")
                    else:
                        print(f"⚠️ JARVIS notification failed: {response.status_code}")
                except Exception as e:
                    print(f"⚠️ Failed to notify JARVIS: {e}")
            
            # Send notification in background to avoid blocking
            threading.Thread(target=notify_background, daemon=True).start()
            
        except ImportError:
            print("⚠️ Requests library not available for JARVIS notification")
        except Exception as e:
            print(f"⚠️ Failed to notify JARVIS: {e}")
    
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
        self._process_interactions(hands_info, frame)
        
        # Store hands_info for scaling calculations
        self.current_hands_info = hands_info
        
        # Draw 2D objects (respecting visibility and isolation mode)
        if self.show_2d_objects:
            for obj in self.objects:
                if self.object_visibility.get(obj.id, True):
                    frame = obj.draw(frame)
        
        # Draw 3D objects (respecting visibility and isolation mode)
        if self.show_3d_objects and self.renderer_3d:
            for i, obj_3d in enumerate(self.objects_3d):
                if self.object_visibility.get(obj_3d.id, True):
                    # Only show pinchable radius for the first two 3D objects
                    if i < 2:
                        obj_3d.selected = (i == 0)  # First object is selected (yellow), second is not (blue)
                    else:
                        obj_3d.selected = None  # Hide radius for objects beyond the first two
                    frame = obj_3d.draw(frame, self.renderer_3d)
                
                # Draw elegant targeting indicator for 3D objects when pinching near them
                if self.object_visibility.get(obj_3d.id, True):
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
        
        # Take screenshot for JARVIS if flag is set (after all objects are rendered)
        if self.should_take_screenshot and self.jarvis_activated:
            screenshot_path = self.save_screenshot_for_jarvis(frame)
            if screenshot_path:
                print("📸 Screenshot updated for JARVIS analysis - objects rendered")
            self.should_take_screenshot = False  # Reset flag
        
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
        elif key == ord('e'):
            # Exit isolation mode - show all objects
            if self.object_isolation_mode:
                self.object_isolation_mode = False
                self.isolated_object_id = None
                for obj_id in self.object_visibility:
                    self.object_visibility[obj_id] = True
                print("Exited isolation mode - all objects visible")
            else:
                print("Not in isolation mode")
        elif key == ord('j'):
            # Activate JARVIS voice assistant
            print("🤖 Activating JARVIS voice assistant...")
            self.jarvis_activated = not self.jarvis_activated
            if self.jarvis_activated:
                print("✅ JARVIS activated - Voice assistant ready")
                print("🗣️  Say 'What is this?' to analyze the 3D objects")
                print("📸 JARVIS will analyze the current screenshot when you ask")
                
                cv2.putText(frame, "JARVIS ACTIVATED - Voice Assistant Ready", 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Select objects to update screenshot", 
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
    
    def _process_interactions(self, hands_info: List[dict], frame: np.ndarray):
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
                if self.show_3d_objects and self._handle_pinch_interaction_3d(hand_info, hand_idx, frame):
                    current_grabs_3d.add(hand_idx)
                # Then try 2D objects
                elif self.show_2d_objects:
                    self._handle_pinch_interaction(hand_info, hand_idx, frame)
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
                                
                                # Get initial position of rotation hand and store handedness
                                rotation_hand_info = next((hand for hand in hands_info if hand['hand_idx'] == hand_idx), None)
                                if rotation_hand_info:
                                    obj_3d.last_rotation_hand_pos = rotation_hand_info['palm_center']
                                    obj_3d.rotation_hand_type = rotation_hand_info.get('handedness', 'Unknown')
                                
                                print(f"Object entered rotation mode - {obj_3d.rotation_hand_type} hand {hand_idx} controlling rotation")
                                
                                # Remove this hand from grabbed_by_hand but keep it in grab_states_3d for rotation tracking
                                if hand_idx in obj_3d.grabbed_by_hand:
                                    obj_3d.grabbed_by_hand.remove(hand_idx)
                                obj_3d.is_grabbed = len(obj_3d.grabbed_by_hand)
                                
                                # Clear scaling state
                                if 'initial_two_hand_distance' in grab_state:
                                    del grab_state['initial_two_hand_distance']
                                if 'initial_scale' in grab_state:
                                    del grab_state['initial_scale']
                                
                                # Clear locked rotation state
                                if 'locked_rotation_x' in grab_state:
                                    del grab_state['locked_rotation_x']
                                if 'locked_rotation_y' in grab_state:
                                    del grab_state['locked_rotation_y']
                                if 'locked_rotation_z' in grab_state:
                                    del grab_state['locked_rotation_z']
                                
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
                        obj_3d.rotation_hand_type = None
                    
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
                                # Clear locked rotation state
                                if 'locked_rotation_x' in other_hand_state:
                                    del other_hand_state['locked_rotation_x']
                                if 'locked_rotation_y' in other_hand_state:
                                    del other_hand_state['locked_rotation_y']
                                if 'locked_rotation_z' in other_hand_state:
                                    del other_hand_state['locked_rotation_z']
                    
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
                            # Clear locked rotation state
                            if 'locked_rotation_x' in other_hand_state:
                                del other_hand_state['locked_rotation_x']
                            if 'locked_rotation_y' in other_hand_state:
                                del other_hand_state['locked_rotation_y']
                            if 'locked_rotation_z' in other_hand_state:
                                del other_hand_state['locked_rotation_z']
                
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
    
    def _handle_pinch_interaction(self, hand_info: dict, hand_idx: int, frame: np.ndarray):
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
                was_grabbed = closest_obj.is_grabbed > 0
                if hand_idx not in closest_obj.grabbed_by_hand:
                    closest_obj.grabbed_by_hand.append(hand_idx)
                # Update grab state based on number of hands
                closest_obj.is_grabbed = len(closest_obj.grabbed_by_hand)
                # Mark object as selected when grabbed
                closest_obj.selected = True
                
                # Set flag to take screenshot for JARVIS analysis when object is first grabbed
                if not was_grabbed and self.jarvis_activated:
                    self.should_take_screenshot = True
                    print("📸 Screenshot will be updated for JARVIS analysis - 2D object grabbed")
                    # Notify JARVIS that an object was grabbed
                    self._notify_jarvis_object_grabbed()
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
        
        # Get initial position of rotation hand and store handedness
        rotation_hand_info = next((hand for hand in hands_info if hand['hand_idx'] == rotation_hand), None)
        if rotation_hand_info:
            obj_3d.last_rotation_hand_pos = rotation_hand_info['palm_center']
            obj_3d.rotation_hand_type = rotation_hand_info.get('handedness', 'Unknown')
        
        print(f"Object entered rotation mode - {obj_3d.rotation_hand_type} hand {rotation_hand} controlling rotation")
    
    def _handle_rotation_mode_3d(self, hands_info: List[dict]):
        """Handle rotation mode for 3D objects with hand-specific single-axis rotation"""
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
                    
                    rotation_sensitivity = 0.015  # Increased sensitivity for single-axis rotation
                    movement_threshold = 3  # Lower threshold for more responsive rotation
                    
                    # Get handedness for this rotation hand
                    hand_type = getattr(obj_3d, 'rotation_hand_type', 'Unknown')
                    
                    # Apply hand-specific single-axis rotation
                    if hand_type == 'Right':
                        # Right hand controls X-axis rotation (pitch)
                        if abs(delta_y) > movement_threshold:
                            # Natural direction: up movement = rotate up, down movement = rotate down
                            obj_3d.rotation_x -= delta_y * rotation_sensitivity
                            # Keep rotation in reasonable range
                            obj_3d.rotation_x = obj_3d.rotation_x % (2 * math.pi)
                    
                    elif hand_type == 'Left':
                        # Left hand controls Y-axis rotation (yaw)
                        if abs(delta_x) > movement_threshold:
                            # Natural direction: left movement = counter-clockwise, right movement = clockwise
                            obj_3d.rotation_y -= delta_x * rotation_sensitivity
                            # Keep rotation in reasonable range
                            obj_3d.rotation_y = obj_3d.rotation_y % (2 * math.pi)
                    
                    else:
                        # Unknown hand type - fallback to original multi-axis behavior
                        if abs(delta_x) > movement_threshold:
                            obj_3d.rotation_y -= delta_x * rotation_sensitivity
                            obj_3d.rotation_y = obj_3d.rotation_y % (2 * math.pi)
                        if abs(delta_y) > movement_threshold:
                            obj_3d.rotation_x -= delta_y * rotation_sensitivity
                            obj_3d.rotation_x = obj_3d.rotation_x % (2 * math.pi)
                    
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
        obj_3d.rotation_hand_type = None
        print(f"Object exited rotation mode")
                
    
    def _handle_pinch_interaction_3d(self, hand_info: dict, hand_idx: int, frame: np.ndarray) -> bool:
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
                was_grabbed = closest_obj_3d.is_grabbed > 0
                if hand_idx not in closest_obj_3d.grabbed_by_hand:
                    closest_obj_3d.grabbed_by_hand.append(hand_idx)
                # Update grab state based on number of hands
                closest_obj_3d.is_grabbed = len(closest_obj_3d.grabbed_by_hand)
                # Mark object as selected when grabbed
                closest_obj_3d.selected = True
                
                # Set flag to take screenshot for JARVIS analysis when object is first grabbed
                if not was_grabbed and self.jarvis_activated:
                    self.should_take_screenshot = True
                    print("📸 Screenshot will be updated for JARVIS analysis - 3D object grabbed")
                    # Notify JARVIS that an object was grabbed
                    self._notify_jarvis_object_grabbed()
                
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
                # Only allow position changes during normal movement - NO ROTATION
                # Ensure object is not in rotation mode before allowing position changes
                if not obj_3d.is_in_rotation_mode:
                    # Convert screen movement to world space movement
                    sensitivity = grab_state['movement_sensitivity']
                    world_delta_x = pinch_delta_x * sensitivity
                    world_delta_y = -pinch_delta_y * sensitivity  # Invert Y for correct direction
                    
                    # Apply movement directly without smoothing for immediate response
                    obj_3d.x += world_delta_x
                    obj_3d.y += world_delta_y
                    
                    # Explicitly lock rotation during normal movement
                    # Store current rotation to prevent any drift
                    if 'locked_rotation_x' not in grab_state:
                        grab_state['locked_rotation_x'] = obj_3d.rotation_x
                        grab_state['locked_rotation_y'] = obj_3d.rotation_y
                        grab_state['locked_rotation_z'] = obj_3d.rotation_z
                    
                    # Force rotation to stay locked
                    obj_3d.rotation_x = grab_state['locked_rotation_x']
                    obj_3d.rotation_y = grab_state['locked_rotation_y']
                    obj_3d.rotation_z = grab_state['locked_rotation_z']
            
            
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
        
        new_obj = VirtualObject(x, y, size, color, shape)
        new_obj.id = f"2d_{len(self.objects)}"
        self.object_visibility[new_obj.id] = False  # Start hidden
        # Move off-screen initially
        new_obj.x = -1000
        new_obj.y = -1000
        self.objects.append(new_obj)
    
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
        hint = f"Pinch dot: toggle on/off • Drag out: place • Two hands drag: isolate • E: show all • {len(hands_info)} hands detected"
        hint_size = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
        hint_x = (w - hint_size[0]) // 2
        cv2.putText(frame, hint, (hint_x, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        
        # Remove the floating status panel - dock only
        
        # Draw hand interaction indicators with elegance
        self._draw_elegant_hand_indicators(frame, hands_info)
        
        # Draw object count indicators in corners
        self._draw_corner_indicators(frame)
        
        # Draw Apple-inspired capsule dock
        self._draw_capsule_dock(frame, hands_info)
        
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
    
    def _draw_capsule_dock(self, frame: np.ndarray, hands_info: List[dict]):
        """Draw Apple-inspired capsule dock with all objects"""
        h, w = frame.shape[:2]
        
        # Collect all objects (2D and 3D)
        all_objects = []
        
        # Add 2D objects
        for obj in self.objects:
            all_objects.append({
                'id': obj.id,
                'type': '2D',
                'color': obj.color,
                'shape': obj.shape,
                'object': obj
            })
        
        # Add 3D objects
        for obj in self.objects_3d:
            all_objects.append({
                'id': obj.id,
                'type': '3D',
                'color': obj.color,
                'shape': 'model',
                'object': obj
            })
        
        if not all_objects:
            return
        
        # Dock dimensions - bigger and positioned higher
        dock_height = 100
        dock_padding = 25
        item_size = 65
        item_spacing = 20
        dock_width = len(all_objects) * (item_size + item_spacing) - item_spacing + dock_padding * 2
        
        # Center dock horizontally and position higher from bottom
        dock_x = (w - dock_width) // 2
        dock_y = h - dock_height - 60
        
        # Draw main capsule background with glassmorphism
        self._draw_capsule_background(frame, dock_x, dock_y, dock_width, dock_height)
        
        # Draw object thumbnails
        for i, obj_info in enumerate(all_objects):
            item_x = dock_x + dock_padding + i * (item_size + item_spacing)
            item_y = dock_y + (dock_height - item_size) // 2
            
            self._draw_object_thumbnail(frame, item_x, item_y, item_size, obj_info)
        
        # Handle dock interactions
        self._handle_dock_interactions(frame, hands_info, all_objects, dock_x, dock_y, dock_width, dock_height, item_size, item_spacing)
        
        # Clean up old dock states for hands that are no longer detected
        self._cleanup_dock_states(hands_info)
    
    def _draw_capsule_background(self, frame: np.ndarray, x: int, y: int, width: int, height: int):
        """Draw the capsule-shaped dock background"""
        # Create mask for rounded rectangle
        overlay = frame.copy()
        
        # Draw rounded rectangle using circles and rectangle
        radius = height // 2
        
        # Main rectangle
        cv2.rectangle(overlay, (x + radius, y), (x + width - radius, y + height), (40, 40, 40), -1)
        
        # Left semicircle
        cv2.circle(overlay, (x + radius, y + radius), radius, (40, 40, 40), -1)
        
        # Right semicircle  
        cv2.circle(overlay, (x + width - radius, y + radius), radius, (40, 40, 40), -1)
        
        # Apply glassmorphism effect
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Add subtle border
        cv2.rectangle(frame, (x + radius, y), (x + width - radius, y + height), (100, 100, 100), 1)
        cv2.circle(frame, (x + radius, y + radius), radius, (100, 100, 100), 1)
        cv2.circle(frame, (x + width - radius, y + radius), radius, (100, 100, 100), 1)
    
    def _draw_object_thumbnail(self, frame: np.ndarray, x: int, y: int, size: int, obj_info: dict):
        """Draw individual object thumbnail in dock"""
        center_x = x + size // 2
        center_y = y + size // 2
        
        # Determine colors based on ACTUAL visibility state from object_visibility dict
        actual_visibility = self.object_visibility.get(obj_info['id'], True)
        
        if actual_visibility:
            # Object is visible in scene - lights ON (full brightness)
            base_color = obj_info['color']
            border_color = (200, 200, 200)
        else:
            # Object is hidden in scene - lights OFF (dimmed)
            base_color = tuple(int(c * 0.3) for c in obj_info['color'])
            border_color = (80, 80, 80)
        
        # Draw thumbnail background
        thumbnail_size = size - 10
        thumbnail_radius = thumbnail_size // 2
        
        # Glassmorphism background
        overlay = frame.copy()
        cv2.circle(overlay, (center_x, center_y), thumbnail_radius, base_color, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Draw shape indicator
        if obj_info['shape'] == 'circle':
            cv2.circle(frame, (center_x, center_y), thumbnail_radius - 8, base_color, 2)
        elif obj_info['shape'] == 'cube':
            rect_size = thumbnail_radius - 8
            cv2.rectangle(frame, 
                         (center_x - rect_size, center_y - rect_size),
                         (center_x + rect_size, center_y + rect_size),
                         base_color, 2)
        else:  # 3D model
            # Draw 3D indicator (diamond shape)
            points = np.array([
                [center_x, center_y - thumbnail_radius + 8],
                [center_x + thumbnail_radius - 8, center_y],
                [center_x, center_y + thumbnail_radius - 8],
                [center_x - thumbnail_radius + 8, center_y]
            ], np.int32)
            cv2.polylines(frame, [points], True, base_color, 2)
        
        # Draw border
        cv2.circle(frame, (center_x, center_y), thumbnail_radius, border_color, 2)
        
        # Add type indicator with larger text for bigger dock
        type_text = obj_info['type']
        text_size = cv2.getTextSize(type_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        text_x = center_x - text_size[0] // 2
        text_y = center_y + size // 2 + 18
        
        cv2.putText(frame, type_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, border_color, 1)
    
    def _handle_dock_interactions(self, frame: np.ndarray, hands_info: List[dict], all_objects: list,
                                 dock_x: int, dock_y: int, dock_width: int, dock_height: int,
                                 item_size: int, item_spacing: int):
        """Handle pinch interactions with dock items"""
        dock_padding = 25  # Updated padding
        current_time = time.time()
        
        # Handle drag interactions (both single and two-hand)
        self._handle_dock_drag_interactions(hands_info, all_objects, dock_x, dock_y, dock_width, dock_height, item_size, item_spacing, frame)
    
    def _handle_dock_drag_interactions(self, hands_info: List[dict], all_objects: list,
                                      dock_x: int, dock_y: int, dock_width: int, dock_height: int,
                                      item_size: int, item_spacing: int, frame: np.ndarray):
        """Handle drag interactions - single hand shows model, two hands isolate"""
        dock_padding = 25
        current_time = time.time()
        
        # Process each hand
        for hand_info in hands_info:
            hand_idx = hand_info['hand_idx']
            is_pinching = hand_info['is_pinching']
            pinch_pos = hand_info['pinch_center']
            
            if is_pinching:
                # Check if starting a new drag from dock
                if hand_idx not in self.drag_states:
                    # Check if pinch started in dock area
                    if (dock_x <= pinch_pos[0] <= dock_x + dock_width and 
                        dock_y <= pinch_pos[1] <= dock_y + dock_height):
                        
                        # Find which dock item is being targeted
                        target_object_id = None
                        dock_item_pos = None
                        
                        for i, obj_info in enumerate(all_objects):
                            item_x = dock_x + dock_padding + i * (item_size + item_spacing)
                            item_center_x = item_x + item_size // 2
                            item_center_y = dock_y + dock_height // 2
                            
                            distance = math.sqrt((pinch_pos[0] - item_center_x)**2 + (pinch_pos[1] - item_center_y)**2)
                            
                            if distance <= item_size // 2:
                                target_object_id = obj_info['id']
                                dock_item_pos = (item_center_x, item_center_y)
                                break
                        
                        if target_object_id:
                            # Start drag state
                            self.drag_states[hand_idx] = {
                                'object_id': target_object_id,
                                'start_pos': pinch_pos,
                                'current_pos': pinch_pos,
                                'is_dragging': False,
                                'dock_pos': dock_item_pos
                            }
                
                # Update existing drag state
                if hand_idx in self.drag_states:
                    drag_state = self.drag_states[hand_idx]
                    drag_state['current_pos'] = pinch_pos
                    
                    # Calculate distance from start position
                    start_pos = drag_state['start_pos']
                    drag_distance = math.sqrt((pinch_pos[0] - start_pos[0])**2 + (pinch_pos[1] - start_pos[1])**2)
                    
                    # Check if dragging has started
                    if not drag_state['is_dragging'] and drag_distance > self.drag_threshold:
                        drag_state['is_dragging'] = True
                        print(f"Started dragging {drag_state['object_id']}")
                    
                    # If dragging, check distance from dock
                    if drag_state['is_dragging']:
                        dock_distance = math.sqrt((pinch_pos[0] - dock_x - dock_width/2)**2 + (pinch_pos[1] - dock_y - dock_height/2)**2)
                        
                        if dock_distance > self.drag_out_threshold:
                            # Far enough from dock - activate model
                            self._activate_dragged_model(drag_state['object_id'], hands_info)
                        
                        # Draw dragged dot
                        self._draw_dragged_dot(frame, drag_state, all_objects, item_size)
            
            else:
                # Released pinch - handle drop or toggle
                if hand_idx in self.drag_states:
                    drag_state = self.drag_states[hand_idx]
                    
                    if drag_state['is_dragging']:
                        # This was a drag operation - place object
                        hands_dragging_this = sum(1 for ds in self.drag_states.values() 
                                                 if ds['object_id'] == drag_state['object_id'] and ds['is_dragging'])
                        
                        if hands_dragging_this >= 2:
                            # Two hands - place object at center between the two drag points
                            self._place_object_at_isolation_center(drag_state['object_id'])
                        else:
                            # Single hand - place object where dot was dropped
                            self._place_object_at_drop_position(drag_state['object_id'], drag_state['current_pos'])
                        
                        print(f"Dropped {drag_state['object_id']} - placed in scene")
                    else:
                        # This was just a pinch (no drag) - toggle visibility
                        self._toggle_object_with_placement(drag_state['object_id'])
                    
                    # Remove drag state
                    del self.drag_states[hand_idx]
    
    def _activate_dragged_model(self, object_id: str, hands_info: List[dict]):
        """Prepare model for drag interaction - DO NOT show until dropped"""
        # Count how many hands are dragging this same object
        hands_dragging_this = 0
        for drag_state in self.drag_states.values():
            if drag_state['object_id'] == object_id and drag_state['is_dragging']:
                hands_dragging_this += 1
        
        # Don't actually activate/show anything during drag
        # Objects will only be shown when dropped via _place_object_at_drop_position
        # or _place_object_at_isolation_center methods
        print(f"Preparing {object_id} for placement ({'isolate' if hands_dragging_this >= 2 else 'show'} mode)")
    
    def _draw_dragged_dot(self, frame: np.ndarray, drag_state: dict, all_objects: list, item_size: int):
        """Draw the dragged dot following the hand"""
        current_pos = drag_state['current_pos']
        object_id = drag_state['object_id']
        
        # Find object info
        obj_info = None
        for obj in all_objects:
            if obj['id'] == object_id:
                obj_info = obj
                break
        
        if not obj_info:
            return
        
        # Draw dot at current position
        dot_radius = item_size // 4
        
        # Determine color based on drag distance and number of hands
        hands_on_this = sum(1 for ds in self.drag_states.values() 
                           if ds['object_id'] == object_id and ds['is_dragging'])
        
        if hands_on_this >= 2:
            # Two hands - isolation mode color
            color = (255, 200, 100)  # Orange
            border_color = (255, 150, 50)
            text = "ISOLATE"
        else:
            # Single hand - show model color
            color = obj_info['color']
            border_color = (255, 255, 255)
            text = "SHOW"
        
        # Draw main dot
        cv2.circle(frame, current_pos, dot_radius, color, -1)
        cv2.circle(frame, current_pos, dot_radius, border_color, 2)
        
        # Draw connection line back to dock
        dock_pos = drag_state['dock_pos']
        cv2.line(frame, current_pos, dock_pos, (150, 150, 150), 1)
        
        # Draw text indicator
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        text_pos = (current_pos[0] - text_size[0] // 2, current_pos[1] - dot_radius - 10)
        cv2.putText(frame, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, border_color, 1)
        
        # Don't show placement preview during drag - only show after unpinch
        # self._draw_placement_preview(frame, current_pos, obj_info, hands_on_this >= 2)
    
    def _draw_placement_preview(self, frame: np.ndarray, position: tuple, obj_info: dict, is_isolation: bool):
        """Draw a ghost preview of where the object will be placed"""
        x, y = position
        
        # Draw a subtle preview circle/shape
        if is_isolation:
            # For isolation mode, draw a special preview
            preview_color = (100, 150, 255)  # Light orange
            preview_radius = 25
        else:
            # For single object, use object's color but dimmed
            preview_color = tuple(int(c * 0.6) for c in obj_info['color'])
            preview_radius = 20
        
        # Draw ghost object preview
        if obj_info['shape'] == 'circle':
            cv2.circle(frame, position, preview_radius, preview_color, 2)
            cv2.circle(frame, position, preview_radius - 5, preview_color, 1)
        elif obj_info['shape'] == 'cube':
            cv2.rectangle(frame, 
                         (x - preview_radius, y - preview_radius),
                         (x + preview_radius, y + preview_radius),
                         preview_color, 2)
        else:  # 3D model
            # Draw diamond shape for 3D objects
            points = np.array([
                [x, y - preview_radius],
                [x + preview_radius, y],
                [x, y + preview_radius],
                [x - preview_radius, y]
            ], np.int32)
            cv2.polylines(frame, [points], True, preview_color, 2)
        
        # Add small placement indicator
        cv2.circle(frame, position, 3, (255, 255, 255), -1)
    
    def _place_object_at_drop_position(self, object_id: str, drop_position: tuple):
        """Place object at the dropped position in 3D space"""
        # Convert screen coordinates to world coordinates
        coords = self._screen_to_world_coordinates(drop_position)
        world_x_2d, world_y_2d, world_x_3d, world_y_3d, world_z_3d = coords
        
        # Find and move the object, then make it visible
        for obj in self.objects:
            if obj.id == object_id:
                obj.x, obj.y = world_x_2d, world_y_2d
                self.object_visibility[object_id] = True  # Make visible after placement
                print(f"Placed 2D object {object_id} at screen {drop_position} -> world ({world_x_2d:.2f}, {world_y_2d:.2f})")
                return
        
        for obj_3d in self.objects_3d:
            if obj_3d.id == object_id:
                obj_3d.x, obj_3d.y, obj_3d.z = world_x_3d, world_y_3d, world_z_3d
                self.object_visibility[object_id] = True  # Make visible after placement
                print(f"Placed 3D object {object_id} at screen {drop_position} -> world ({world_x_3d:.3f}, {world_y_3d:.3f}, {world_z_3d:.1f})")
                return
    
    def _place_object_at_isolation_center(self, object_id: str):
        """Place object at center between two isolation drag points"""
        # Find all drag states for this object
        drag_positions = []
        for drag_state in self.drag_states.values():
            if drag_state['object_id'] == object_id and drag_state['is_dragging']:
                drag_positions.append(drag_state['current_pos'])
        
        if len(drag_positions) >= 2:
            # Calculate center point between the drag positions
            center_x = sum(pos[0] for pos in drag_positions) // len(drag_positions)
            center_y = sum(pos[1] for pos in drag_positions) // len(drag_positions)
            center_position = (center_x, center_y)
            
            # Convert to world coordinates and place object
            coords = self._screen_to_world_coordinates(center_position)
            world_x_2d, world_y_2d, world_x_3d, world_y_3d, world_z_3d = coords
            
            # Find and move the object, then enter isolation mode properly
            for obj in self.objects:
                if obj.id == object_id:
                    obj.x, obj.y = world_x_2d, world_y_2d
                    self._enter_isolation_mode_proper(object_id)  # This handles visibility
                    print(f"Placed 2D object {object_id} at isolation center {center_position} -> world ({world_x_2d:.2f}, {world_y_2d:.2f})")
                    return
            
            for obj_3d in self.objects_3d:
                if obj_3d.id == object_id:
                    obj_3d.x, obj_3d.y, obj_3d.z = world_x_3d, world_y_3d, world_z_3d
                    self._enter_isolation_mode_proper(object_id)  # This handles visibility
                    print(f"Placed 3D object {object_id} at isolation center {center_position} -> world ({world_x_3d:.3f}, {world_y_3d:.3f}, {world_z_3d:.1f})")
                    return
    
    def _screen_to_world_coordinates(self, screen_pos: tuple) -> tuple:
        """Convert screen coordinates to 3D world coordinates"""
        screen_x, screen_y = screen_pos
        
        # Get frame dimensions
        if hasattr(self, 'renderer_3d'):
            frame_width = self.renderer_3d.width
            frame_height = self.renderer_3d.height
        else:
            frame_width = 1280  # Default width
            frame_height = 720   # Default height
        
        # Map screen coordinates to world coordinates
        # Center of screen maps to world (0, 0)
        # Scale factors adjusted for better object placement
        
        # For 2D objects - they use direct pixel coordinates, so use 1:1 mapping
        scale_factor_2d = 1.0  # 2D objects use screen coordinates directly
        
        # For 3D objects - use smaller scale for precise placement  
        scale_factor_3d = 0.005  # 3D objects move less, more precise
        
        # Calculate relative position from screen center
        rel_x = screen_x - frame_width / 2
        rel_y = screen_y - frame_height / 2
        
        # For 2D objects - use direct screen coordinates (no conversion needed)
        world_x_2d = screen_x
        world_y_2d = screen_y
        
        # For 3D objects - convert screen to world coordinates
        world_x_3d = rel_x * scale_factor_3d
        world_y_3d = -rel_y * scale_factor_3d  # Invert Y axis
        world_z_3d = -3.0  # Place 3D objects at reasonable depth
        
        # Return both 2D and 3D coordinates (caller will use appropriate ones)
        return (world_x_2d, world_y_2d, world_x_3d, world_y_3d, world_z_3d)
    
    def _handle_two_hand_dock_pinch(self, hands_info: List[dict], all_objects: list,
                                   dock_x: int, dock_y: int, dock_width: int, dock_height: int,
                                   item_size: int, item_spacing: int, frame: np.ndarray):
        """Handle two hands pinching the same dock item for isolation mode"""
        if len(hands_info) < 2:
            return
        
        dock_padding = 25
        current_time = time.time()
        
        # Check if exactly two hands are pinching
        pinching_hands = [hand for hand in hands_info if hand['is_pinching']]
        if len(pinching_hands) != 2:
            return
        
        # Check if both hands are pinching within dock area
        hands_in_dock = []
        for hand_info in pinching_hands:
            pinch_x, pinch_y = hand_info['pinch_center']
            if (dock_x <= pinch_x <= dock_x + dock_width and 
                dock_y <= pinch_y <= dock_y + dock_height):
                hands_in_dock.append(hand_info)
        
        if len(hands_in_dock) != 2:
            return
        
        # Find which dock item both hands are targeting
        target_object_id = None
        target_center = None
        
        for i, obj_info in enumerate(all_objects):
            item_x = dock_x + dock_padding + i * (item_size + item_spacing)
            item_center_x = item_x + item_size // 2
            item_center_y = dock_y + dock_height // 2
            
            hands_on_this_item = 0
            for hand_info in hands_in_dock:
                pinch_x, pinch_y = hand_info['pinch_center']
                distance = math.sqrt((pinch_x - item_center_x)**2 + (pinch_y - item_center_y)**2)
                if distance <= item_size // 2 + 10:  # Slightly larger detection area for two-hand
                    hands_on_this_item += 1
            
            # If both hands are on the same dock item
            if hands_on_this_item >= 2:
                target_object_id = obj_info['id']
                target_center = (item_center_x, item_center_y)
                break
        
        if target_object_id:
            # Check if this is a new two-hand gesture
            two_hand_key = f"two_hand_{target_object_id}"
            
            # Use a special state tracking for two-hand gestures
            if not hasattr(self, 'two_hand_gesture_active'):
                self.two_hand_gesture_active = {}
            
            if two_hand_key not in self.two_hand_gesture_active:
                # New two-hand gesture detected
                self.two_hand_gesture_active[two_hand_key] = current_time
                
                # Check cooldown to prevent rapid toggling
                if (target_object_id not in self.dock_toggle_cooldown or 
                    current_time - self.dock_toggle_cooldown[target_object_id] > 1.0):
                    
                    # Turn off all other objects, turn on this one
                    self._enter_isolation_mode_proper(target_object_id)
                    self.dock_toggle_cooldown[target_object_id] = current_time
                    print(f"Two-hand isolation: {target_object_id}")
            
            # Always draw isolation indicator when two hands are detected
            cv2.circle(frame, target_center, item_size // 2 + 8, (255, 200, 100), 4)
            cv2.circle(frame, target_center, item_size // 2 + 12, (255, 200, 100), 2)
            
            # Add text indicator
            cv2.putText(frame, "ISOLATE", (target_center[0] - 25, target_center[1] - item_size // 2 - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 100), 2)
        else:
            # No two-hand gesture detected, clear active state
            if hasattr(self, 'two_hand_gesture_active'):
                self.two_hand_gesture_active.clear()
    
    def _cleanup_dock_states(self, hands_info: List[dict]):
        """Clean up dock interaction states for hands that are no longer detected"""
        current_hand_indices = {hand['hand_idx'] for hand in hands_info}
        
        # Clean up drag states
        hands_to_remove = []
        for hand_idx in self.drag_states:
            if hand_idx not in current_hand_indices:
                hands_to_remove.append(hand_idx)
        
        for hand_idx in hands_to_remove:
            print(f"Cleaning up drag state for disappeared hand {hand_idx}")
            del self.drag_states[hand_idx]
        
        # Clean up dock pinch states
        hands_to_remove = []
        for hand_idx in self.dock_pinch_states:
            if hand_idx not in current_hand_indices:
                hands_to_remove.append(hand_idx)
        
        for hand_idx in hands_to_remove:
            del self.dock_pinch_states[hand_idx]
        
        # Clean up dock interaction states
        hands_to_remove = []
        for hand_idx in self.dock_interaction_state:
            if hand_idx not in current_hand_indices:
                hands_to_remove.append(hand_idx)
        
        for hand_idx in hands_to_remove:
            del self.dock_interaction_state[hand_idx]
    
    def _toggle_object_visibility(self, object_id: str):
        """Toggle object visibility in the scene"""
        if object_id in self.object_visibility:
            self.object_visibility[object_id] = not self.object_visibility[object_id]
            print(f"Toggled {object_id}: {'ON' if self.object_visibility[object_id] else 'OFF'}")
    
    def _toggle_object_with_placement(self, object_id: str):
        """Toggle object visibility - if turning on, place at center; if turning off, hide"""
        current_visibility = self.object_visibility.get(object_id, True)
        
        if current_visibility:
            # Currently visible - turn off
            self.object_visibility[object_id] = False
            print(f"Turned off {object_id}")
        else:
            # Currently hidden - turn on and place at center
            self.object_visibility[object_id] = True
            self._place_object_at_center(object_id)
            print(f"Turned on {object_id} and placed at center")
    
    def _place_object_at_center(self, object_id: str):
        """Place object at screen center"""
        # Get frame dimensions for center calculation
        if hasattr(self, 'renderer_3d'):
            center_x = self.renderer_3d.width // 2
            center_y = self.renderer_3d.height // 2
        else:
            center_x = 640  # Default center
            center_y = 360
        
        # Find and move the object to center
        for obj in self.objects:
            if obj.id == object_id:
                obj.x, obj.y = center_x, center_y
                print(f"Placed 2D object {object_id} at center ({center_x}, {center_y})")
                return
        
        for obj_3d in self.objects_3d:
            if obj_3d.id == object_id:
                # For 3D objects, place at world center
                obj_3d.x, obj_3d.y, obj_3d.z = 0.0, 0.0, -3.0
                print(f"Placed 3D object {object_id} at world center (0, 0, -3)")
                return
    
    def _enter_isolation_mode_proper(self, object_id: str):
        """Enter isolation mode - turn off all other objects, turn on selected object"""
        # Always enter isolation mode with this specific object
        self.object_isolation_mode = True
        self.isolated_object_id = object_id
        
        # Turn off all objects, then turn on only the selected one
        for obj_id in self.object_visibility:
            self.object_visibility[obj_id] = (obj_id == object_id)
        
        print(f"Isolation mode: Only {object_id} is now visible")
    
    def _enter_isolation_mode(self, object_id: str):
        """Enter isolation mode showing only the specified object (toggle version)"""
        if self.object_isolation_mode and self.isolated_object_id == object_id:
            # Exit isolation mode
            self.object_isolation_mode = False
            self.isolated_object_id = None
            # Restore all object visibility
            for obj_id in self.object_visibility:
                self.object_visibility[obj_id] = True
            print("Exited isolation mode - all objects visible")
        else:
            # Enter isolation mode
            self.object_isolation_mode = True
            self.isolated_object_id = object_id
            # Hide all objects except the isolated one
            for obj_id in self.object_visibility:
                self.object_visibility[obj_id] = (obj_id == object_id)
            print(f"Entered isolation mode for {object_id}")
    
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
                    
                    # Get hand type and create appropriate label
                    hand_type = getattr(obj_3d, 'rotation_hand_type', 'Unknown')
                    if hand_type == 'Right':
                        label_text = "RIGHT: X-AXIS"
                        axis_text = "(PITCH)"
                    elif hand_type == 'Left':
                        label_text = "LEFT: Y-AXIS"
                        axis_text = "(YAW)"
                    else:
                        label_text = "ROTATE"
                        axis_text = "(MULTI-AXIS)"
                    
                    # Elegant label with glass background
                    label_w, label_h = 120, 35
                    label_x = hand_center[0] - label_w // 2
                    label_y = hand_center[1] - 60
                    
                    # Glass background for label
                    self._draw_glass_panel(frame, (label_x, label_y - label_h), (label_w, label_h), alpha=0.3)
                    
                    # Main label text
                    text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
                    text_x = label_x + (label_w - text_size[0]) // 2
                    text_y = label_y - 18
                    
                    cv2.putText(frame, label_text, (text_x, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, rotation_color, 1)
                    
                    # Axis description text
                    axis_size = cv2.getTextSize(axis_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
                    axis_x = label_x + (label_w - axis_size[0]) // 2
                    axis_y = label_y - 5
                    
                    cv2.putText(frame, axis_text, (axis_x, axis_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
        
        return frame
    
    def _cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def _draw_capsule_dock(self, frame: np.ndarray, hands_info: List[dict]):
        """Draw Apple-inspired capsule dock with all objects"""
        h, w = frame.shape[:2]
        
        # Collect all objects (2D and 3D)
        all_objects = []
        
        # Add 2D objects
        for obj in self.objects:
            all_objects.append({
                'id': obj.id,
                'type': '2D',
                'color': obj.color,
                'shape': obj.shape,
                'object': obj
            })
        
        # Add 3D objects
        for obj in self.objects_3d:
            all_objects.append({
                'id': obj.id,
                'type': '3D',
                'color': obj.color,
                'shape': 'model',
                'object': obj
            })
        
        if not all_objects:
            return
        
        # Dock dimensions - much more spaced out
        dock_height = 120
        dock_padding = 40
        item_size = 80
        item_spacing = 50  # Much more space between dots
        dock_width = len(all_objects) * (item_size + item_spacing) - item_spacing + dock_padding * 2
        
        # Center dock horizontally and position higher from bottom
        dock_x = (w - dock_width) // 2
        dock_y = h - dock_height - 60
        
        # Draw main capsule background with glassmorphism
        self._draw_capsule_background(frame, dock_x, dock_y, dock_width, dock_height)
        
        # Draw object thumbnails
        for i, obj_info in enumerate(all_objects):
            item_x = dock_x + dock_padding + i * (item_size + item_spacing)
            item_y = dock_y + (dock_height - item_size) // 2
            
            self._draw_object_thumbnail(frame, item_x, item_y, item_size, obj_info)
        
        # Handle dock interactions
        self._handle_dock_interactions(frame, hands_info, all_objects, dock_x, dock_y, dock_width, dock_height, item_size, item_spacing)
        
        # Clean up old dock states for hands that are no longer detected
        self._cleanup_dock_states(hands_info)
    
    def _draw_capsule_background(self, frame: np.ndarray, x: int, y: int, width: int, height: int):
        """Draw the capsule-shaped dock background"""
        # Create mask for rounded rectangle
        overlay = frame.copy()
        
        # Draw rounded rectangle using circles and rectangle
        radius = height // 2
        
        # Main rectangle
        cv2.rectangle(overlay, (x + radius, y), (x + width - radius, y + height), (40, 40, 40), -1)
        
        # Left semicircle
        cv2.circle(overlay, (x + radius, y + radius), radius, (40, 40, 40), -1)
        
        # Right semicircle  
        cv2.circle(overlay, (x + width - radius, y + radius), radius, (40, 40, 40), -1)
        
        # Apply glassmorphism effect
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Add subtle border
        cv2.rectangle(frame, (x + radius, y), (x + width - radius, y + height), (100, 100, 100), 1)
        cv2.circle(frame, (x + radius, y + radius), radius, (100, 100, 100), 1)
        cv2.circle(frame, (x + width - radius, y + radius), radius, (100, 100, 100), 1)
    
    def _draw_object_thumbnail(self, frame: np.ndarray, x: int, y: int, size: int, obj_info: dict):
        """Draw individual object thumbnail in dock"""
        center_x = x + size // 2
        center_y = y + size // 2
        
        # Determine colors based on ACTUAL visibility state from object_visibility dict
        actual_visibility = self.object_visibility.get(obj_info['id'], True)
        
        if actual_visibility:
            # Object is visible in scene - lights ON (full brightness)
            base_color = obj_info['color']
            border_color = (200, 200, 200)
        else:
            # Object is hidden in scene - lights OFF (dimmed)
            base_color = tuple(int(c * 0.3) for c in obj_info['color'])
            border_color = (80, 80, 80)
        
        # Draw thumbnail background
        thumbnail_size = size - 10
        thumbnail_radius = thumbnail_size // 2
        
        # Glassmorphism background
        overlay = frame.copy()
        cv2.circle(overlay, (center_x, center_y), thumbnail_radius, base_color, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Draw shape indicator
        if obj_info['shape'] == 'circle':
            cv2.circle(frame, (center_x, center_y), thumbnail_radius - 8, base_color, 2)
        elif obj_info['shape'] == 'cube':
            rect_size = thumbnail_radius - 8
            cv2.rectangle(frame, 
                         (center_x - rect_size, center_y - rect_size),
                         (center_x + rect_size, center_y + rect_size),
                         base_color, 2)
        else:  # 3D model
            # Draw 3D indicator (diamond shape)
            points = np.array([
                [center_x, center_y - thumbnail_radius + 8],
                [center_x + thumbnail_radius - 8, center_y],
                [center_x, center_y + thumbnail_radius - 8],
                [center_x - thumbnail_radius + 8, center_y]
            ], np.int32)
            cv2.polylines(frame, [points], True, base_color, 2)
        
        # Draw border
        cv2.circle(frame, (center_x, center_y), thumbnail_radius, border_color, 2)
        
        # Add type indicator with larger text for bigger dock
        type_text = obj_info['type']
        text_size = cv2.getTextSize(type_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = center_x - text_size[0] // 2
        text_y = center_y + size // 2 + 25  # More space for larger dock
        
        cv2.putText(frame, type_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, border_color, 1)
    
    def _handle_dock_interactions(self, frame: np.ndarray, hands_info: List[dict], all_objects: list,
                                 dock_x: int, dock_y: int, dock_width: int, dock_height: int,
                                 item_size: int, item_spacing: int):
        """Handle pinch interactions with dock items"""
        dock_padding = 40  # Updated padding to match dock
        current_time = time.time()
        
        # Handle drag interactions (both single and two-hand)
        self._handle_dock_drag_interactions(hands_info, all_objects, dock_x, dock_y, dock_width, dock_height, item_size, item_spacing, frame)

    def _handle_dock_drag_interactions(self, hands_info: List[dict], all_objects: list,
                                      dock_x: int, dock_y: int, dock_width: int, dock_height: int,
                                      item_size: int, item_spacing: int, frame: np.ndarray):
        """Handle drag interactions - single hand shows model, two hands isolate"""
        dock_padding = 40  # Updated padding to match dock
        current_time = time.time()
        
        # Process each hand
        for hand_info in hands_info:
            hand_idx = hand_info['hand_idx']
            is_pinching = hand_info['is_pinching']
            pinch_pos = hand_info['pinch_center']
            
            if is_pinching:
                # Check if starting a new drag from dock
                if hand_idx not in self.drag_states:
                    # Check if pinch started in dock area
                    if (dock_x <= pinch_pos[0] <= dock_x + dock_width and 
                        dock_y <= pinch_pos[1] <= dock_y + dock_height):
                        
                        # Find which dock item is being targeted
                        target_object_id = None
                        dock_item_pos = None
                        
                        for i, obj_info in enumerate(all_objects):
                            item_x = dock_x + dock_padding + i * (item_size + item_spacing)
                            item_center_x = item_x + item_size // 2
                            item_center_y = dock_y + dock_height // 2
                            
                            distance = math.sqrt((pinch_pos[0] - item_center_x)**2 + (pinch_pos[1] - item_center_y)**2)
                            
                            if distance <= item_size // 2:
                                target_object_id = obj_info['id']
                                dock_item_pos = (item_center_x, item_center_y)
                                break
                        
                        if target_object_id:
                            # Start drag state
                            self.drag_states[hand_idx] = {
                                'object_id': target_object_id,
                                'start_pos': pinch_pos,
                                'current_pos': pinch_pos,
                                'is_dragging': False,
                                'dock_pos': dock_item_pos
                            }
                
                # Update existing drag state
                if hand_idx in self.drag_states:
                    drag_state = self.drag_states[hand_idx]
                    drag_state['current_pos'] = pinch_pos
                    
                    # Calculate distance from start position
                    start_pos = drag_state['start_pos']
                    drag_distance = math.sqrt((pinch_pos[0] - start_pos[0])**2 + (pinch_pos[1] - start_pos[1])**2)
                    
                    # Check if dragging has started
                    if not drag_state['is_dragging'] and drag_distance > self.drag_threshold:
                        drag_state['is_dragging'] = True
                        print(f"Started dragging {drag_state['object_id']}")
                    
                    # If dragging, check distance from dock
                    if drag_state['is_dragging']:
                        dock_distance = math.sqrt((pinch_pos[0] - dock_x - dock_width/2)**2 + (pinch_pos[1] - dock_y - dock_height/2)**2)
                        
                        if dock_distance > self.drag_out_threshold:
                            # Far enough from dock - activate model
                            self._activate_dragged_model(drag_state['object_id'], hands_info)
                        
                        # Draw dragged dot
                        self._draw_dragged_dot(frame, drag_state, all_objects, item_size)
            
            else:
                # Released pinch - handle drop or toggle
                if hand_idx in self.drag_states:
                    drag_state = self.drag_states[hand_idx]
                    
                    if drag_state['is_dragging']:
                        # This was a drag operation - place object
                        hands_dragging_this = sum(1 for ds in self.drag_states.values() 
                                                 if ds['object_id'] == drag_state['object_id'] and ds['is_dragging'])
                        
                        if hands_dragging_this >= 2:
                            # Two hands - place object at center between the two drag points
                            self._place_object_at_isolation_center(drag_state['object_id'])
                        else:
                            # Single hand - place object where dot was dropped
                            self._place_object_at_drop_position(drag_state['object_id'], drag_state['current_pos'])
                        
                        print(f"Dropped {drag_state['object_id']} - placed in scene")
                    else:
                        # This was just a pinch (no drag) - toggle visibility
                        self._toggle_object_with_placement(drag_state['object_id'])
                    
                    # Remove drag state
                    del self.drag_states[hand_idx]
    
    def _activate_dragged_model(self, object_id: str, hands_info: List[dict]):
        """Prepare model for drag interaction - DO NOT show until dropped"""
        # Count how many hands are dragging this same object
        hands_dragging_this = 0
        for drag_state in self.drag_states.values():
            if drag_state['object_id'] == object_id and drag_state['is_dragging']:
                hands_dragging_this += 1
        
        # Don't actually activate/show anything during drag
        # Objects will only be shown when dropped via _place_object_at_drop_position
        # or _place_object_at_isolation_center methods
        print(f"Preparing {object_id} for placement ({'isolate' if hands_dragging_this >= 2 else 'show'} mode)")
    
    def _draw_dragged_dot(self, frame: np.ndarray, drag_state: dict, all_objects: list, item_size: int):
        """Draw the dragged dot following the hand"""
        current_pos = drag_state['current_pos']
        object_id = drag_state['object_id']
        
        # Find object info
        obj_info = None
        for obj in all_objects:
            if obj['id'] == object_id:
                obj_info = obj
                break
        
        if not obj_info:
            return
        
        # Draw dot at current position
        dot_radius = item_size // 4
        
        # Determine color based on drag distance and number of hands
        hands_on_this = sum(1 for ds in self.drag_states.values() 
                           if ds['object_id'] == object_id and ds['is_dragging'])
        
        if hands_on_this >= 2:
            # Two hands - isolation mode color
            color = (255, 200, 100)  # Orange
            border_color = (255, 150, 50)
            text = "ISOLATE"
        else:
            # Single hand - show model color
            color = obj_info['color']
            border_color = (255, 255, 255)
            text = "SHOW"
        
        # Draw main dot
        cv2.circle(frame, current_pos, dot_radius, color, -1)
        cv2.circle(frame, current_pos, dot_radius, border_color, 2)
        
        # Draw connection line back to dock
        dock_pos = drag_state['dock_pos']
        cv2.line(frame, current_pos, dock_pos, (150, 150, 150), 1)
        
        # Draw text indicator
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        text_pos = (current_pos[0] - text_size[0] // 2, current_pos[1] - dot_radius - 10)
        cv2.putText(frame, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, border_color, 1)
    
    def _place_object_at_drop_position(self, object_id: str, drop_position: tuple):
        """Place object at the dropped position in 3D space"""
        # Convert screen coordinates to world coordinates
        coords = self._screen_to_world_coordinates(drop_position)
        world_x_2d, world_y_2d, world_x_3d, world_y_3d, world_z_3d = coords
        
        # Find and move the object, then make it visible
        for obj in self.objects:
            if obj.id == object_id:
                obj.x, obj.y = world_x_2d, world_y_2d
                self.object_visibility[object_id] = True  # Make visible after placement
                print(f"Placed 2D object {object_id} at screen {drop_position} -> world ({world_x_2d:.2f}, {world_y_2d:.2f})")
                return
        
        for obj_3d in self.objects_3d:
            if obj_3d.id == object_id:
                obj_3d.x, obj_3d.y, obj_3d.z = world_x_3d, world_y_3d, world_z_3d
                self.object_visibility[object_id] = True  # Make visible after placement
                print(f"Placed 3D object {object_id} at screen {drop_position} -> world ({world_x_3d:.3f}, {world_y_3d:.3f}, {world_z_3d:.1f})")
                return
    
    def _place_object_at_isolation_center(self, object_id: str):
        """Place object at center between two isolation drag points"""
        # Find all drag states for this object
        drag_positions = []
        for drag_state in self.drag_states.values():
            if drag_state['object_id'] == object_id and drag_state['is_dragging']:
                drag_positions.append(drag_state['current_pos'])
        
        if len(drag_positions) >= 2:
            # Calculate center point between the drag positions
            center_x = sum(pos[0] for pos in drag_positions) // len(drag_positions)
            center_y = sum(pos[1] for pos in drag_positions) // len(drag_positions)
            center_position = (center_x, center_y)
            
            # Convert to world coordinates and place object
            coords = self._screen_to_world_coordinates(center_position)
            world_x_2d, world_y_2d, world_x_3d, world_y_3d, world_z_3d = coords
            
            # Find and move the object, then enter isolation mode properly
            for obj in self.objects:
                if obj.id == object_id:
                    obj.x, obj.y = world_x_2d, world_y_2d
                    self._enter_isolation_mode_proper(object_id)  # This handles visibility
                    print(f"Placed 2D object {object_id} at isolation center {center_position} -> world ({world_x_2d:.2f}, {world_y_2d:.2f})")
                    return
            
            for obj_3d in self.objects_3d:
                if obj_3d.id == object_id:
                    obj_3d.x, obj_3d.y, obj_3d.z = world_x_3d, world_y_3d, world_z_3d
                    self._enter_isolation_mode_proper(object_id)  # This handles visibility
                    print(f"Placed 3D object {object_id} at isolation center {center_position} -> world ({world_x_3d:.3f}, {world_y_3d:.3f}, {world_z_3d:.1f})")
                    return
    
    def _screen_to_world_coordinates(self, screen_pos: tuple) -> tuple:
        """Convert screen coordinates to 3D world coordinates"""
        screen_x, screen_y = screen_pos
        
        # Get frame dimensions
        if hasattr(self, 'renderer_3d'):
            frame_width = self.renderer_3d.width
            frame_height = self.renderer_3d.height
        else:
            frame_width = 1280  # Default width
            frame_height = 720   # Default height
        
        # Map screen coordinates to world coordinates
        # Center of screen maps to world (0, 0)
        # Scale factors adjusted for better object placement
        
        # For 2D objects - they use direct pixel coordinates, so use 1:1 mapping
        scale_factor_2d = 1.0  # 2D objects use screen coordinates directly
        
        # For 3D objects - use smaller scale for precise placement  
        scale_factor_3d = 0.005  # 3D objects move less, more precise
        
        # Calculate relative position from screen center
        rel_x = screen_x - frame_width / 2
        rel_y = screen_y - frame_height / 2
        
        # For 2D objects - use direct screen coordinates (no conversion needed)
        world_x_2d = screen_x
        world_y_2d = screen_y
        
        # For 3D objects - convert screen to world coordinates
        world_x_3d = rel_x * scale_factor_3d
        world_y_3d = -rel_y * scale_factor_3d  # Invert Y axis
        world_z_3d = -3.0  # Place 3D objects at reasonable depth
        
        # Return both 2D and 3D coordinates (caller will use appropriate ones)
        return (world_x_2d, world_y_2d, world_x_3d, world_y_3d, world_z_3d)
    
    def _cleanup_dock_states(self, hands_info: List[dict]):
        """Clean up dock interaction states for hands that are no longer detected"""
        current_hand_indices = {hand['hand_idx'] for hand in hands_info}
        
        # Clean up drag states
        hands_to_remove = []
        for hand_idx in self.drag_states:
            if hand_idx not in current_hand_indices:
                hands_to_remove.append(hand_idx)
        
        for hand_idx in hands_to_remove:
            print(f"Cleaning up drag state for disappeared hand {hand_idx}")
            del self.drag_states[hand_idx]
        
        # Clean up dock pinch states
        hands_to_remove = []
        for hand_idx in self.dock_pinch_states:
            if hand_idx not in current_hand_indices:
                hands_to_remove.append(hand_idx)
        
        for hand_idx in hands_to_remove:
            del self.dock_pinch_states[hand_idx]
        
        # Clean up dock interaction states
        hands_to_remove = []
        for hand_idx in self.dock_interaction_state:
            if hand_idx not in current_hand_indices:
                hands_to_remove.append(hand_idx)
        
        for hand_idx in hands_to_remove:
            del self.dock_interaction_state[hand_idx]
    
    def _toggle_object_visibility(self, object_id: str):
        """Toggle object visibility in the scene"""
        if object_id in self.object_visibility:
            self.object_visibility[object_id] = not self.object_visibility[object_id]
            print(f"Toggled {object_id}: {'ON' if self.object_visibility[object_id] else 'OFF'}")
    
    def _toggle_object_with_placement(self, object_id: str):
        """Toggle object visibility - if turning on, place at center; if turning off, hide"""
        current_visibility = self.object_visibility.get(object_id, True)
        
        if current_visibility:
            # Currently visible - turn off
            self.object_visibility[object_id] = False
            print(f"Turned off {object_id}")
        else:
            # Currently hidden - turn on and place at center
            self.object_visibility[object_id] = True
            self._place_object_at_center(object_id)
            print(f"Turned on {object_id} and placed at center")
    
    def _place_object_at_center(self, object_id: str):
        """Place object at screen center"""
        # Get frame dimensions for center calculation
        if hasattr(self, 'renderer_3d'):
            center_x = self.renderer_3d.width // 2
            center_y = self.renderer_3d.height // 2
        else:
            center_x = 640  # Default center
            center_y = 360
        
        # Find and move the object to center
        for obj in self.objects:
            if obj.id == object_id:
                obj.x, obj.y = center_x, center_y
                print(f"Placed 2D object {object_id} at center ({center_x}, {center_y})")
                return
        
        for obj_3d in self.objects_3d:
            if obj_3d.id == object_id:
                # For 3D objects, place at world center
                obj_3d.x, obj_3d.y, obj_3d.z = 0.0, 0.0, -3.0
                print(f"Placed 3D object {object_id} at world center (0, 0, -3)")
                return
    
    def _enter_isolation_mode_proper(self, object_id: str):
        """Enter isolation mode - turn off all other objects, turn on selected object"""
        # Always enter isolation mode with this specific object
        self.object_isolation_mode = True
        self.isolated_object_id = object_id
        
        # Turn off all objects, then turn on only the selected one
        for obj_id in self.object_visibility:
            self.object_visibility[obj_id] = (obj_id == object_id)
        
        print(f"Isolation mode: Only {object_id} is now visible")


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