import cv2
import mediapipe as mp
import numpy as np
import math
from typing import List, Tuple, Optional
import time
import os
from renderer_3d import Renderer3D
from virtual_object_3d import VirtualObject3D

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
        self.selected = False  # Track selection state for highlighting 
        
    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Draw the virtual object on the frame"""
        center = (int(self.x), int(self.y))
        radius = int(self.size)
        
        # Draw pinchable radius highlighting - make radius match actual object size
        pinchable_radius = int(self.size)  # Use actual object size for pinchable radius
        if self.selected:
            # Yellow highlighting when selected
            cv2.circle(frame, center, pinchable_radius, (0, 255, 255), 2)  # Yellow outline
        else:
            # Blue highlighting when not selected
            cv2.circle(frame, center, pinchable_radius, (255, 0, 0), 2)  # Blue outline
        
        if self.shape == "circle":
            for i in range(radius, 0, -2):
                alpha = 0.8 * (i / radius)
                color = tuple(int(c * alpha) for c in self.color)
                cv2.circle(frame, center, i, color, -1)
                
            # Draw grab indicator based on grab state
            if self.is_grabbed == 1:
                cv2.circle(frame, center, radius + 5, (255, 255, 255), 3)  # White ring for 1 hand
            elif self.is_grabbed == 2:
                cv2.circle(frame, center, radius + 5, (255, 255, 0), 5)  # Yellow ring for 2 hands
                
        return frame
    
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
        """Draw hand landmarks on frame"""
        for hand_info in hands_info:
            if 'landmarks' in hand_info:
                self.mp_drawing.draw_landmarks(
                    frame, hand_info['landmarks'], self.mp_hands.HAND_CONNECTIONS)
        return frame


class ARHandController:
    """Main AR application for hand-controlled object manipulation"""
    
    def __init__(self):
        self.detector = HandGestureDetector()
        self.objects: List[VirtualObject] = []
        self.objects_3d: List[VirtualObject3D] = []
        self.cap = None
        self.is_running = False
        
        # 3D Renderer
        self.renderer_3d = None
        
        # Interaction state
        self.grab_states = {}  # hand_idx -> {object, initial_pinch_distance, initial_size}
        self.grab_states_3d = {}  # hand_idx -> {object, initial_pinch_distance, initial_size, last_hand_pos}
        self.last_frame_time = time.time()
        
        # Pinch state tracking for hysteresis
        self.pinch_states = {}  # hand_idx -> {'was_pinching': bool, 'pinch_frames': int}
        
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
        
        # 3D objects
        self.objects_3d = []
        
        # Load multiple CAD components for testing
        cad_models = [
            {
                "path": "online/Wooden Crate.obj",
                "name": "Complex CAD Assembly",
                "position": (0.0, 0.0, -4.0),
                "scale": 1.2,
                "color": (100, 150, 255)
            },
        ]
        
        for model_info in cad_models:
            model_path = os.path.join(os.path.dirname(__file__), model_info["path"])
            if os.path.exists(model_path):
                obj_3d = VirtualObject3D(
                    model_path, 
                    x=model_info["position"][0], 
                    y=model_info["position"][1], 
                    z=model_info["position"][2],
                    scale=model_info["scale"], 
                    color=model_info["color"]
                )
                obj_3d.set_render_mode("solid")
                # Set different auto-rotation speeds for variety
                obj_3d.auto_rotation_speed = 0.01 + len(self.objects_3d) * 0.005
                self.objects_3d.append(obj_3d)
                print(f"✅ Loaded {model_info['name']} from {model_info['path']}")
            else:
                print(f"⚠️  {model_info['name']} not found at {model_path}")
    
    def start(self):
        """Start the AR application"""
        # Try different camera indices
        camera_indices = [0]  # Try multiple camera sources
        self.cap = None
        
        for idx in camera_indices:
            print(f"Trying camera index {idx}...")
            test_cap = cv2.VideoCapture(0)
            if test_cap.isOpened():
                # Test if we can actually read a frame
                ret, frame = test_cap.read()
                if ret and frame is not None:
                    self.cap = test_cap
                    print(f"✅ Successfully opened camera {idx}")
                    break
                else:
                    test_cap.release()
            else:
                test_cap.release()
        
        if self.cap is None:
            print("❌ Error: Could not open any camera")
            print("\n🔧 Troubleshooting tips:")
            print("1. Check System Preferences → Privacy & Security → Camera")
            print("2. Allow Terminal (or your Python IDE) to access the camera")
            print("3. Make sure no other applications are using the camera")
            print("4. Try running: 'python3.11 launch_app.py' for GUI launcher")
            return
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Initialize 3D renderer
        self.renderer_3d = Renderer3D(1280, 720)
        
        self.is_running = True
        print("AR Hand Control started!")
        print("Gestures:")
        print("- Pinch (thumb + index) near object: Grab object")
        print("- Move hand while pinching: Move object (improved 3D tracking!)")
        print("- Grab object with TWO hands and move apart/closer: Scale object")
        print("- Two hands: Rotate 3D objects")
        print("- 3D objects now follow pinch point more accurately")
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
        print("- Press 's' to cycle through 2D objects")
        
        while self.is_running:
            self._process_frame()
            
        self._cleanup()
    
    def _process_frame(self):
        """Process a single frame"""
        ret, frame = self.cap.read()
        if not ret:
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
            for obj_3d in self.objects_3d:
                frame = obj_3d.draw(frame, self.renderer_3d)
                
                # Draw targeting indicator for 3D objects when pinching near them
                for hand_info in hands_info:
                    if (hand_info['is_pinching'] and 
                        not obj_3d.is_grabbed and 
                        obj_3d.is_point_inside(hand_info['pinch_center'][0], hand_info['pinch_center'][1], self.renderer_3d)):
                        
                        screen_pos = obj_3d.get_screen_position(self.renderer_3d)
                        if screen_pos:
                            # Draw targeting circle
                            cv2.circle(frame, (int(screen_pos[0]), int(screen_pos[1])), 60, (255, 255, 0), 3)
                            cv2.circle(frame, (int(screen_pos[0]), int(screen_pos[1])), 40, (255, 255, 0), 2)
                            
                            # Draw line from pinch center to object center
                            cv2.line(frame, hand_info['pinch_center'], (int(screen_pos[0]), int(screen_pos[1])), (255, 255, 0), 2)
            
        # Draw hand landmarks
        frame = self.detector.draw_landmarks(frame, hands_info)
        
        # Draw UI
        frame = self._draw_ui(frame, hands_info)
        
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
                    # Remove this hand from the grabbed_by_hand list
                    if hand_idx in obj_3d.grabbed_by_hand:
                        obj_3d.grabbed_by_hand.remove(hand_idx)
                    # Update grab state based on remaining hands
                    obj_3d.is_grabbed = len(obj_3d.grabbed_by_hand)
                    # Deselect object when completely released
                    if obj_3d.is_grabbed == 0:
                        obj_3d.selected = False
                    
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
                    
                    del self.grab_states_3d[hand_idx]
        
        # Handle two-hand rotation for 3D objects
        if len(hands_info) >= 2 and len(self.objects_3d) > 0:
            self._handle_two_hand_rotation(hands_info)
        
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
    
    def _handle_two_hand_scaling_3d(self, obj_3d):
        """Handle scaling when 3D object is grabbed by two hands"""
        # Get the two hands that are grabbing this object
        grabbing_hands = [hand_idx for hand_idx in obj_3d.grabbed_by_hand if hand_idx in self.grab_states_3d]
        
        if len(grabbing_hands) == 2:
            hand1_idx, hand2_idx = grabbing_hands[0], grabbing_hands[1]
            hand1_state = self.grab_states_3d[hand1_idx]
            hand2_state = self.grab_states_3d[hand2_idx]
            
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
                    hand1_state['initial_scale'] = obj_3d.scale
                    hand2_state['initial_two_hand_distance'] = current_distance
                    hand2_state['initial_scale'] = obj_3d.scale
                
                initial_distance = hand1_state['initial_two_hand_distance']
                
                # Calculate scale factor based on distance change
                if initial_distance > 0:
                    scale_factor = current_distance / initial_distance
                    target_scale = hand1_state['initial_scale'] * scale_factor
                    
                    # Apply scaling with smoothing
                    obj_3d.scale = obj_3d.scale * 0.7 + target_scale * 0.3  # Smooth scaling
                    obj_3d.scale = max(0.1, min(5.0, obj_3d.scale))  # Clamp scale
    
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
                    'movement_sensitivity': 0.015  # Controls how much screen movement affects 3D position (increased for better responsiveness)
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
            
            # Only scale if object is grabbed by two hands
            if obj_3d.is_grabbed == 2:
                self._handle_two_hand_scaling_3d(obj_3d)
            
            # Track finger positions for better interaction feedback
            grab_state['last_pinch_center'] = pinch_center
            grab_state['last_thumb_pos'] = thumb_pos
            grab_state['last_index_pos'] = index_pos
            
            return True
        
        return False
    
    def _handle_two_hand_rotation(self, hands_info: List[dict]):
        """Handle two-hand rotation for 3D objects"""
        if len(hands_info) < 2 or len(self.objects_3d) == 0:
            return
        
        hand1 = hands_info[0]
        hand2 = hands_info[1]
        
        # Only rotate if both hands are pinching
        if hand1['is_pinching'] and hand2['is_pinching']:
            # Calculate rotation based on hand positions
            pos1 = hand1['pinch_center']
            pos2 = hand2['pinch_center']
            
            # Calculate angle between hands
            dx = pos2[0] - pos1[0]
            dy = pos2[1] - pos1[1]
            angle = math.atan2(dy, dx)
            
            # Apply rotation to the first 3D object (could be extended to all)
            if self.objects_3d:
                obj_3d = self.objects_3d[0]
                # Disable auto-rotation when manually rotating
                obj_3d.auto_rotate = False
                
                # Apply small rotation increments
                rotation_speed = 0.02
                obj_3d.rotate(0, rotation_speed, 0)  # Rotate around Y-axis
    
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
        """Draw user interface elements"""
        # Draw instructions
        instructions = [
            "AR Hand Control - Pinch to grab, move, and scale objects",
            "Two hands required for scaling - move apart/closer to scale",
            "Q: Quit | R: Reset | C: Add Object",
            f"Objects: {len(self.objects)} | Hands: {len(hands_info)}"
        ]
        
        for i, instruction in enumerate(instructions):
            y_pos = 30 + i * 25
            cv2.putText(frame, instruction, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw hand status with pinch feedback
        for hand_info in hands_info:
            hand_idx = hand_info['hand_idx']
            palm_pos = hand_info['palm_center']
            pinch_center = hand_info['pinch_center']
            
            # Draw palm center
            cv2.circle(frame, palm_pos, 8, (0, 255, 0), -1)
            
            # Draw pinch center
            cv2.circle(frame, pinch_center, 5, (255, 0, 255), -1)
            
            # Show pinch status with more detail
            is_pinching = hand_info['is_pinching']
            stabilized_pinching = hand_idx in self.pinch_states and self.pinch_states[hand_idx]['was_pinching']
            
            if stabilized_pinching:
                status = "PINCHING ✓"
                color = (0, 255, 0)  # Green for successful pinch
                if hand_idx in self.grab_states:
                    obj = self.grab_states[hand_idx]['object']
                    grab_state_text = ["Not grabbed", "1 hand", "2 hands"][obj.is_grabbed]
                    scaling_text = " (SCALING)" if obj.is_grabbed == 2 else ""
                    status = f"2D GRABBED ✓ ({grab_state_text}{scaling_text}, Size: {int(obj.size)})"
                    color = (255, 255, 0)  # Yellow for grabbed 2D
                elif hand_idx in self.grab_states_3d:
                    obj_3d = self.grab_states_3d[hand_idx]['object']
                    grab_state_text = ["Not grabbed", "1 hand", "2 hands"][obj_3d.is_grabbed]
                    scaling_text = " (SCALING)" if obj_3d.is_grabbed == 2 else ""
                    status = f"3D GRABBED ✓ ({grab_state_text}{scaling_text}, Scale: {obj_3d.scale:.1f})"
                    color = (0, 255, 255)  # Cyan for grabbed 3D
            elif is_pinching:
                status = "PINCHING..."
                color = (0, 255, 255)  # Cyan for detecting
            else:
                status = "OPEN"
                color = (128, 128, 128)  # Gray for open
            
            cv2.putText(frame, f"Hand {hand_idx}: {status}", 
                       (palm_pos[0] - 60, palm_pos[1] - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Show pinch distance
            pinch_distance = hand_info['pinch_distance']
            cv2.putText(frame, f"Distance: {int(pinch_distance)}", 
                       (palm_pos[0] - 30, palm_pos[1] + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Show object counts and mode info
        info_y = frame.shape[0] - 80
        cv2.putText(frame, f"2D Objects: {len(self.objects)} {'(ON)' if self.show_2d_objects else '(OFF)'}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"3D Objects: {len(self.objects_3d)} {'(ON)' if self.show_3d_objects else '(OFF)'}", 
                   (10, info_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if self.objects_3d:
            render_mode = self.objects_3d[0].render_mode
            auto_rotate = self.objects_3d[0].auto_rotate
            cv2.putText(frame, f"3D Mode: {render_mode} | Auto-rotate: {'ON' if auto_rotate else 'OFF'}", 
                       (10, info_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
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
