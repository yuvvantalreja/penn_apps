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
        """Draw the virtual object on the frame"""
        center = (int(self.x), int(self.y))
        radius = int(self.size)
        
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
        self.assemblies: List[CADAssembly] = []  # List for all assemblies (both single objects and multi-component)
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
            for i, obj_3d in enumerate(self.objects_3d):
                # Only show pinchable radius for the first two 3D objects
                if i < 2:
                    obj_3d.selected = (i == 0)  # First object is selected (yellow), second is not (blue)
                else:
                    obj_3d.selected = None  # Hide radius for objects beyond the first two
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
        """Draw user interface elements"""
        # Draw instructions
        instructions = [
            "AR Hand Control - Pinch to grab, move, scale, and multi-axis rotate",
            "Two hands: scaling | Release one hand: multi-axis rotation",
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
                    rotation_text = f" (MULTI-AXIS ROTATION - Hand {obj_3d.rotation_hand_idx})" if obj_3d.is_in_rotation_mode else ""
                    status = f"3D GRABBED ✓ ({grab_state_text}{scaling_text}{rotation_text}, Scale: {obj_3d.scale:.1f})"
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
        info_y = frame.shape[0] - 100
        cv2.putText(frame, f"2D Objects: {len(self.objects)} {'(ON)' if self.show_2d_objects else '(OFF)'}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"3D Objects: {len(self.objects_3d)} {'(ON)' if self.show_3d_objects else '(OFF)'}", 
                   (10, info_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show assembly info
        total_components = sum(len(assembly.get_components()) for assembly in self.assemblies)
        cv2.putText(frame, f"Assemblies: {len(self.assemblies)} ({total_components} components)", 
                   (10, info_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show detailed assembly breakdown
        assembly_info_y = info_y + 60
        for i, assembly in enumerate(self.assemblies[:3]):  # Show up to 3 assemblies to avoid clutter
            assembly_text = f"  • {assembly.get_assembly_info()}"
            cv2.putText(frame, assembly_text, 
                       (10, assembly_info_y + i * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        if len(self.assemblies) > 3:
            cv2.putText(frame, f"  + {len(self.assemblies) - 3} more...", 
                       (10, assembly_info_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        if self.objects_3d:
            render_mode = self.objects_3d[0].render_mode
            auto_rotate = self.objects_3d[0].auto_rotate
            mode_info_y = assembly_info_y + max(45, len(self.assemblies[:3]) * 15) + 15
            cv2.putText(frame, f"3D Mode: {render_mode} | Auto-rotate: {'ON' if auto_rotate else 'OFF'}", 
                       (10, mode_info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def _draw_rotation_hand_indicators(self, frame: np.ndarray, hands_info: List[dict]) -> np.ndarray:
        """Draw visual indicators for hands controlling multi-axis rotation"""
        for obj_3d in self.objects_3d:
            if obj_3d.is_in_rotation_mode and obj_3d.rotation_hand_idx is not None:
                # Find the rotation hand
                rotation_hand_info = next((hand for hand in hands_info if hand['hand_idx'] == obj_3d.rotation_hand_idx), None)
                
                if rotation_hand_info:
                    # Draw a large circle around the rotation hand
                    hand_center = rotation_hand_info['palm_center']
                    cv2.circle(frame, hand_center, 35, (0, 255, 255), 3)  # Cyan circle (slightly larger)
                    cv2.circle(frame, hand_center, 30, (0, 255, 255), -1)  # Filled cyan circle (slightly larger)
                    
                    # Draw "MULTI-AXIS ROTATOR" text above the hand
                    text = "MULTI-AXIS ROTATOR"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    font_thickness = 2
                    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                    text_x = hand_center[0] - text_size[0] // 2
                    text_y = hand_center[1] - 50
                    
                    # Draw text background
                    cv2.rectangle(frame, 
                                (text_x - 5, text_y - text_size[1] - 5), 
                                (text_x + text_size[0] + 5, text_y + 5), 
                                (0, 0, 0), -1)
                    
                    # Draw text
                    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 255, 255), font_thickness)
                    
                    # Draw control indicators around the hand
                    arrow_length = 25
                    offset_distance = 60
                    
                    # Horizontal arrows for Y-axis rotation (yaw)
                    left_arrow_start = (hand_center[0] - offset_distance, hand_center[1])
                    left_arrow_end = (hand_center[0] - offset_distance - arrow_length, hand_center[1])
                    right_arrow_start = (hand_center[0] + offset_distance, hand_center[1])
                    right_arrow_end = (hand_center[0] + offset_distance + arrow_length, hand_center[1])
                    
                    cv2.arrowedLine(frame, left_arrow_start, left_arrow_end, (255, 100, 100), 3, tipLength=0.3)  # Red for yaw
                    cv2.arrowedLine(frame, right_arrow_start, right_arrow_end, (255, 100, 100), 3, tipLength=0.3)
                    cv2.putText(frame, "YAW", (hand_center[0] - 15, hand_center[1] - 35), font, 0.4, (255, 100, 100), 1)
                    
                    # Vertical arrows for X-axis rotation (pitch)
                    up_arrow_start = (hand_center[0], hand_center[1] - offset_distance)
                    up_arrow_end = (hand_center[0], hand_center[1] - offset_distance - arrow_length)
                    down_arrow_start = (hand_center[0], hand_center[1] + offset_distance)
                    down_arrow_end = (hand_center[0], hand_center[1] + offset_distance + arrow_length)
                    
                    cv2.arrowedLine(frame, up_arrow_start, up_arrow_end, (100, 255, 100), 3, tipLength=0.3)  # Green for pitch
                    cv2.arrowedLine(frame, down_arrow_start, down_arrow_end, (100, 255, 100), 3, tipLength=0.3)
                    cv2.putText(frame, "PITCH", (hand_center[0] + 20, hand_center[1] - 5), font, 0.4, (100, 255, 100), 1)
                    
                    # Diagonal arrows for Z-axis rotation (roll)
                    diagonal_offset = int(offset_distance * 0.707)  # 45 degrees
                    
                    # Top-left to bottom-right diagonal
                    tl_start = (hand_center[0] - diagonal_offset, hand_center[1] - diagonal_offset)
                    tl_end = (hand_center[0] - diagonal_offset - int(arrow_length * 0.707), hand_center[1] - diagonal_offset - int(arrow_length * 0.707))
                    
                    # Top-right to bottom-left diagonal  
                    tr_start = (hand_center[0] + diagonal_offset, hand_center[1] - diagonal_offset)
                    tr_end = (hand_center[0] + diagonal_offset + int(arrow_length * 0.707), hand_center[1] - diagonal_offset - int(arrow_length * 0.707))
                    
                    cv2.arrowedLine(frame, tl_start, tl_end, (100, 100, 255), 2, tipLength=0.4)  # Blue for roll
                    cv2.arrowedLine(frame, tr_start, tr_end, (100, 100, 255), 2, tipLength=0.4)
                    cv2.putText(frame, "ROLL", (hand_center[0] - 40, hand_center[1] + 45), font, 0.4, (100, 100, 255), 1)
                    
                    # Draw instruction text below the hand
                    instruction_lines = [
                        "Move horizontally: Yaw rotation",
                        "Move vertically: Pitch rotation", 
                        "Move diagonally: Roll rotation"
                    ]
                    
                    instruction_y_start = hand_center[1] + 70
                    for i, line in enumerate(instruction_lines):
                        instruction_y = instruction_y_start + i * 20
                        line_text_size = cv2.getTextSize(line, font, 0.4, 1)[0]
                        instruction_x = hand_center[0] - line_text_size[0] // 2
                        
                        # Draw instruction background
                        cv2.rectangle(frame,
                                    (instruction_x - 3, instruction_y - 12),
                                    (instruction_x + line_text_size[0] + 3, instruction_y + 3),
                                    (0, 0, 0), -1)
                        
                        # Draw instruction text
                        colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255)]  # Red, Green, Blue
                        cv2.putText(frame, line, (instruction_x, instruction_y), font, 0.4, colors[i], 1)
        
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
