import cv2
import mediapipe as mp
import numpy as np
import math
from typing import List, Tuple, Optional
import time
import os
from renderer_3d import Renderer3D
from cad_assembly import CADAssembly

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
        
        # Improved pinch detection with better accuracy
        # Primary method: distance-based with finger position consideration
        is_pinching = (pinch_distance < 50 and  # Reasonable distance threshold
                      pinch_distance > 8 and   # Not too close (avoid noise)
                      (thumb_bent or index_bent or pinch_distance < 30))  # Either finger bent OR very close
        
        # Fallback method: if distance is very close, always consider it pinching
        if pinch_distance < 20:
            is_pinching = True
        
        # Additional stability: require consistent detection for very edge cases
        if 30 < pinch_distance < 40:
            is_pinching = is_pinching and (thumb_bent and index_bent)
        
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
        self.cad_assemblies: List[CADAssembly] = []  # CAD assemblies
        self.cap = None
        self.is_running = False
        
        # 3D Renderer
        self.renderer_3d = None
        
        # Interaction state
        self.grab_states_components = {}  # hand_idx -> {component, initial_pinch_distance, initial_scale, last_hand_pos}
        self.last_frame_time = time.time()
        
        # Pinch state tracking for hysteresis
        self.pinch_states = {}  # hand_idx -> {'was_pinching': bool, 'pinch_frames': int}
        
        # Double pinch detection for selection
        self.pinch_events = {}  # hand_idx -> [{'time': timestamp, 'position': (x, y), 'component': component_or_none}]
        self.double_pinch_window = 0.8  # Time window for double pinch detection (800ms)
        self.double_pinch_distance_threshold = 80  # Maximum distance between pinches for same object
        
        # Selected component tracking
        self.selected_component = None  # Currently selected component for two-handed operations
        self.selection_highlight_time = 0  # Time when component was selected for visual feedback
        
        # Two-handed zoom state
        self.two_handed_zoom_active = False
        self.two_handed_zoom_initial_distance = 0
        self.two_handed_zoom_initial_scale = 1.0
        
        
        # Display mode
        self.show_cad_assemblies = True  # Show CAD assemblies
        
        # CAD assembly management
        self.selected_assembly_index = 0  # Currently selected assembly
        
        # Create some initial objects
        self._create_initial_objects()
        
    def _create_initial_objects(self):
        """Create initial virtual objects"""
        
        # Load CAD Assemblies (both multi-component and single models)
        cad_assemblies = [
            {
                "path": "complex_cad_assembly.obj",
                "name": "Complex CAD",
                "position": (-2.0, 0.0, -3.0),
                "scale": 1.0,
                "color": (100, 150, 255)  # Blue base
            },
            {
                "path": "online/Wooden Crate.obj",
                "name": "Wood Crate",
                "position": (0.0, 0.0, -1.0),
                "scale": 0.7,
                "color": (150, 255, 100)  # Green
            },
        ]
        
        # Load CAD assemblies
        self.cad_assemblies = []
        for assembly_info in cad_assemblies:
            assembly_path = os.path.join(os.path.dirname(__file__), assembly_info["path"])
            if os.path.exists(assembly_path):
                cad_assembly = CADAssembly(
                    name=assembly_info["name"],
                    obj_path=assembly_path,
                    x=assembly_info["position"][0],
                    y=assembly_info["position"][1], 
                    z=assembly_info["position"][2],
                    scale=assembly_info["scale"],
                    base_color=assembly_info["color"]
                )
                cad_assembly.set_render_mode("solid")
                self.cad_assemblies.append(cad_assembly)
                print(f"✅ Loaded CAD Assembly '{assembly_info['name']}' with {cad_assembly.get_component_count()} components")
            else:
                print(f"⚠️  CAD Assembly '{assembly_info['name']}' not found at {assembly_path}")
        
    
    def start(self):
        """Start the AR application"""
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
            return
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        

        self.renderer_3d = Renderer3D(1280, 720)
        
        self.is_running = True
        print("AR Hand Control with CAD Multi-Component Support started!")
        print("Gestures:")
        print("- Single Pinch: Grab and move individual CAD components")  
        print("- Double Pinch: Select a component for special operations")
        print("- Two-Hand Pinch (when component selected): Zoom selected component")
        print("- Move apart/closer with both hands: Scale selected component up/down")
        print("- CAD components can be manipulated independently!")
        print("")
        print("Controls:")
        print("General:")
        print("- Press 'q' to quit")
        print("- Press 'r' to reset all objects")
        print("- Press 'ESC' to deselect current component")
        print("")
        print("Display toggles:")
        print("- Press '1' to toggle CAD assemblies")
        print("")
        print("CAD Assembly controls:")
        print("- Press 'a' to cycle through CAD assemblies")
        print("- Press 'd' to cycle through components in selected assembly")
        print("")
        print("Rendering:")
        print("- Press 'w' to toggle wireframe/solid rendering")
        print("- Press 't' to toggle auto-rotation")
        print("- Press 'x/y/z' to reset rotation on specific axis")

        
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
        
        
        # Draw CAD assemblies
        if self.show_cad_assemblies and self.renderer_3d:
            for i, assembly in enumerate(self.cad_assemblies):
                # Highlight selected assembly
                assembly.highlighted = (i == self.selected_assembly_index)
                frame = assembly.draw(frame, self.renderer_3d)
                
                # Draw targeting indicators for CAD components when pinching near them
                for hand_info in hands_info:
                    if hand_info['is_pinching']:
                        closest_component = assembly.find_component_at_point(
                            hand_info['pinch_center'][0], 
                            hand_info['pinch_center'][1], 
                            self.renderer_3d
                        )
                        if closest_component and not closest_component.is_grabbed:
                            screen_pos = closest_component.get_screen_position(self.renderer_3d)
                            if screen_pos:
                                # Draw targeting circle around component
                                cv2.circle(frame, (int(screen_pos[0]), int(screen_pos[1])), 50, (0, 255, 255), 3)
                                cv2.circle(frame, (int(screen_pos[0]), int(screen_pos[1])), 30, (0, 255, 255), 2)
                                
                                # Draw line from pinch center to component center  
                                cv2.line(frame, hand_info['pinch_center'], (int(screen_pos[0]), int(screen_pos[1])), (0, 255, 255), 2)
                
                # Draw selection indicators for selected component
                if self.selected_component:
                    screen_pos = self.selected_component.get_screen_position(self.renderer_3d)
                    if screen_pos:
                        # Draw pulsating selection indicator
                        current_time = time.time()
                        pulse_factor = 0.5 + 0.5 * math.sin((current_time - self.selection_highlight_time) * 4)
                        radius = int(60 + 20 * pulse_factor)
                        
                        # Draw selection rings
                        cv2.circle(frame, (int(screen_pos[0]), int(screen_pos[1])), radius, (0, 255, 0), 4)
                        cv2.circle(frame, (int(screen_pos[0]), int(screen_pos[1])), radius - 15, (0, 255, 0), 2)
                        
                        # Draw selection text
                        cv2.putText(frame, "SELECTED", 
                                   (int(screen_pos[0]) - 40, int(screen_pos[1]) - radius - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        # If two-handed zoom is active, show zoom indicators
                        if self.two_handed_zoom_active:
                            # Draw zoom indicator lines between hands
                            pinching_hands = []
                            for hand_info in hands_info:
                                hand_idx = hand_info['hand_idx']
                                if (hand_idx in self.pinch_states and 
                                    self.pinch_states[hand_idx]['was_pinching']):
                                    pinching_hands.append(hand_info)
                            
                            if len(pinching_hands) >= 2:
                                hand1, hand2 = pinching_hands[0], pinching_hands[1]
                                cv2.line(frame, hand1['pinch_center'], hand2['pinch_center'], (0, 255, 0), 3)
                                
                                # Draw midpoint
                                mid_x = (hand1['pinch_center'][0] + hand2['pinch_center'][0]) // 2
                                mid_y = (hand1['pinch_center'][1] + hand2['pinch_center'][1]) // 2
                                cv2.circle(frame, (mid_x, mid_y), 8, (0, 255, 0), -1)
                                
                                # Show zoom text
                                cv2.putText(frame, "ZOOMING", 
                                           (mid_x - 40, mid_y - 20),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            
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
        elif key == 27:  # ESC key
            # Deselect current component
            if self.selected_component:
                self.selected_component.selected = None
                self.selected_component = None
                self.two_handed_zoom_active = False
                print("Deselected component")
        elif key == ord('r'):
            self._create_initial_objects()
            self.grab_states_components.clear()
            # Clear selection and gesture state
            if self.selected_component:
                self.selected_component.selected = None
            self.selected_component = None
            self.pinch_events.clear()
            self.two_handed_zoom_active = False
        elif key == ord('1'):
            self.show_cad_assemblies = not self.show_cad_assemblies
            print(f"CAD assemblies: {'ON' if self.show_cad_assemblies else 'OFF'}")
        elif key == ord('w'):
            # Toggle wireframe/solid for all 3D objects and CAD assemblies
            new_mode = "wireframe"
            
            # Check current mode from CAD assemblies
            if self.cad_assemblies and self.cad_assemblies[0].render_mode == "solid":
                new_mode = "wireframe"
            else:
                new_mode = "solid"
            
            
            # Apply to CAD assemblies
            for assembly in self.cad_assemblies:
                assembly.set_render_mode(new_mode)
            
            print(f"Render mode: {new_mode}")
        elif key == ord('t'):
            # Toggle auto-rotation for CAD assemblies
            
            for assembly in self.cad_assemblies:
                assembly.toggle_auto_rotation()
            
            # Get status from CAD assemblies
            auto_rotate_status = False
            if self.cad_assemblies:
                auto_rotate_status = self.cad_assemblies[0].auto_rotate
                
            print(f"Auto-rotation: {'ON' if auto_rotate_status else 'OFF'}")
        elif key == ord('x'):
            # Reset X-axis rotation for all CAD components
            for assembly in self.cad_assemblies:
                for component in assembly.get_all_components():
                    component.rotation_x = 0.0
            print("Reset X-axis rotation")
        elif key == ord('y'):
            # Reset Y-axis rotation for all CAD components
            for assembly in self.cad_assemblies:
                assembly.assembly_rotation_y = 0.0
                for component in assembly.get_all_components():
                    component.rotation_y = 0.0
            print("Reset Y-axis rotation")
        elif key == ord('z'):
            # Reset Z-axis rotation for all CAD components
            for assembly in self.cad_assemblies:
                for component in assembly.get_all_components():
                    component.rotation_z = 0.0
            print("Reset Z-axis rotation")
        elif key == ord('a'):  # 'A' key for cycling CAD assemblies
            # Cycle through CAD assemblies
            if self.cad_assemblies:
                self.selected_assembly_index = (self.selected_assembly_index + 1) % len(self.cad_assemblies)
                assembly = self.cad_assemblies[self.selected_assembly_index]
                print(f"Selected CAD assembly: {assembly.name} ({assembly.get_component_count()} components)")
        elif key == ord('d'):  # 'D' key for cycling components within selected assembly
            # Cycle through components in selected assembly
            if self.cad_assemblies and 0 <= self.selected_assembly_index < len(self.cad_assemblies):
                assembly = self.cad_assemblies[self.selected_assembly_index]
                assembly.cycle_selected_component()
    
    def _process_interactions(self, hands_info: List[dict]):
        """Process hand interactions with objects"""
        current_time = time.time()
        current_grabs = set()
        current_grabs_3d = set()
        
        # First, handle two-handed zoom (highest priority when active)
        if self._handle_two_handed_zoom(hands_info):
            # Two-handed zoom is active, skip other interactions
            return
        
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
            
            # Check for double pinch gesture (for selection)
            double_pinch_result = self._detect_double_pinch(hand_info, hand_idx, current_time)
            if double_pinch_result:
                self._select_component(double_pinch_result['component'], double_pinch_result['position'])
                continue  # Skip regular interaction handling for this frame
            
            if stabilized_pinching:
                # Try CAD components first (highest priority)
                if self.show_cad_assemblies and self._handle_pinch_interaction_components(hand_info, hand_idx):
                    pass  # Component interaction handled
            
            else:
                pass  # No pinching, continue with cleanup below
        
        # Clean up component grab states  
        hands_to_remove_components = []
        for hand_idx in self.grab_states_components:
            current_hand_indices = {hand_info['hand_idx'] for hand_info in hands_info}
            if hand_idx not in current_hand_indices:
                component = self.grab_states_components[hand_idx]['component']
                # Release component
                component.is_grabbed = 0
                component.grabbed_by_hand = []
                component.highlighted = False
                hands_to_remove_components.append(hand_idx)
        
        for hand_idx in hands_to_remove_components:
            del self.grab_states_components[hand_idx]
        
        
        # Additional safety: release objects if no hands are detected
        if not hands_info:
            # Release all CAD components
            for assembly in self.cad_assemblies:
                for component in assembly.get_all_components():
                    if component.is_grabbed > 0:
                        component.is_grabbed = 0
                        component.grabbed_by_hand = []
                        component.highlighted = False
            self.grab_states_components.clear()
            self.pinch_states.clear()
            self.pinch_events.clear()
            self.two_handed_zoom_active = False
    
    
    
    
    
    
                
    
    
    def _handle_pinch_interaction_components(self, hand_info: dict, hand_idx: int) -> bool:
        """Handle pinch gesture interaction with CAD components"""
        pinch_center = hand_info['pinch_center']
        pinch_distance = hand_info['pinch_distance']
        
        if hand_idx not in self.grab_states_components:
            # Try to grab a CAD component
            closest_component = None
            closest_distance = float('inf')
            
            # Search through all assemblies for components
            for assembly in self.cad_assemblies:
                if not assembly.visible:
                    continue
                    
                component = assembly.find_component_at_point(pinch_center[0], pinch_center[1], self.renderer_3d)
                if component and component.is_grabbed == 0:  # Only grab ungrabbed components
                    screen_pos = component.get_screen_position(self.renderer_3d)
                    if screen_pos:
                        distance = math.sqrt((pinch_center[0] - screen_pos[0])**2 + (pinch_center[1] - screen_pos[1])**2)
                        if distance < closest_distance:
                            closest_distance = distance
                            closest_component = component
            
            if closest_component:
                # Grab the component
                closest_component.is_grabbed = 1
                closest_component.grabbed_by_hand = [hand_idx]
                closest_component.highlighted = True
                
                # Get current screen position for offset calculation
                current_screen_pos = closest_component.get_screen_position(self.renderer_3d)
                
                self.grab_states_components[hand_idx] = {
                    'component': closest_component,
                    'initial_pinch_distance': pinch_distance,
                    'initial_scale': closest_component.scale,
                    'initial_world_pos': (closest_component.x, closest_component.y, closest_component.z),
                    'grab_offset_x': pinch_center[0] - current_screen_pos[0] if current_screen_pos else 0,
                    'grab_offset_y': pinch_center[1] - current_screen_pos[1] if current_screen_pos else 0,
                    'last_pinch_center': pinch_center,
                    'movement_sensitivity': 0.015
                }
                print(f"Grabbed component: {closest_component.component_name}")
                return True
        
        else:
            # Continue interaction with grabbed component
            grab_state = self.grab_states_components[hand_idx]
            component = grab_state['component']
            
            # Calculate movement
            last_pinch_center = grab_state['last_pinch_center']
            pinch_delta_x = pinch_center[0] - last_pinch_center[0]
            pinch_delta_y = pinch_center[1] - last_pinch_center[1]
            
            # Apply movement if hand actually moved
            movement_magnitude = math.sqrt(pinch_delta_x**2 + pinch_delta_y**2)
            if movement_magnitude > 0.5:  # Movement threshold
                # Convert screen movement to world space
                sensitivity = grab_state['movement_sensitivity']
                world_delta_x = pinch_delta_x * sensitivity
                world_delta_y = -pinch_delta_y * sensitivity  # Invert Y
                
                # Apply movement
                component.x += world_delta_x
                component.y += world_delta_y
            
            # Handle scaling if pinch distance changed significantly
            distance_change = abs(pinch_distance - grab_state['initial_pinch_distance'])
            if distance_change > 10:  # Scaling threshold
                scale_factor = pinch_distance / grab_state['initial_pinch_distance']
                target_scale = grab_state['initial_scale'] * scale_factor
                component.scale_object(scale_factor)
            
            # Update tracking
            grab_state['last_pinch_center'] = pinch_center
            
            return True
        
        return False
    
    def _detect_double_pinch(self, hand_info: dict, hand_idx: int, current_time: float) -> Optional[dict]:
        """Detect double pinch gesture for component selection"""
        pinch_center = hand_info['pinch_center']
        is_pinching = hand_info['is_pinching']
        
        # Initialize pinch events for this hand if not exists
        if hand_idx not in self.pinch_events:
            self.pinch_events[hand_idx] = []
        
        pinch_events = self.pinch_events[hand_idx]
        
        # Clean up old events outside the time window
        pinch_events[:] = [event for event in pinch_events if (current_time - event['time']) <= self.double_pinch_window]
        
        # If currently pinching, check if this could be a double pinch
        if is_pinching and hand_idx in self.pinch_states and self.pinch_states[hand_idx]['was_pinching']:
            # Check if we just started pinching (transition from not pinching to pinching)
            if self.pinch_states[hand_idx]['pinch_frames'] == 1:  # Just started pinching
                # Find component at pinch location
                target_component = None
                for assembly in self.cad_assemblies:
                    if assembly.visible:
                        component = assembly.find_component_at_point(pinch_center[0], pinch_center[1], self.renderer_3d)
                        if component:
                            target_component = component
                            break
                
                # Add this pinch event
                pinch_event = {
                    'time': current_time,
                    'position': pinch_center,
                    'component': target_component
                }
                pinch_events.append(pinch_event)
                
                # Check for double pinch
                if len(pinch_events) >= 2:
                    latest_event = pinch_events[-1]
                    previous_event = pinch_events[-2]
                    
                    # Check if both events target the same component and are close in space
                    if (latest_event['component'] and 
                        previous_event['component'] and 
                        latest_event['component'] == previous_event['component']):
                        
                        # Check distance between pinch positions
                        distance = math.sqrt(
                            (latest_event['position'][0] - previous_event['position'][0])**2 +
                            (latest_event['position'][1] - previous_event['position'][1])**2
                        )
                        
                        if distance <= self.double_pinch_distance_threshold:
                            # Double pinch detected!
                            return {
                                'component': latest_event['component'],
                                'position': latest_event['position']
                            }
        
        return None
    
    def _handle_two_handed_zoom(self, hands_info: List[dict]) -> bool:
        """Handle two-handed zoom gesture for selected component"""
        if not self.selected_component or len(hands_info) < 2:
            self.two_handed_zoom_active = False
            return False
        
        # Check if both hands are pinching (more lenient for two-handed zoom)
        pinching_hands = []
        for hand_info in hands_info:
            hand_idx = hand_info['hand_idx']
            # Allow either stabilized pinching OR direct pinching for two-handed zoom
            is_pinching_for_zoom = (hand_info['is_pinching'] or 
                                   (hand_idx in self.pinch_states and self.pinch_states[hand_idx]['was_pinching']))
            if (is_pinching_for_zoom and
                hand_idx not in self.grab_states_components):  # Not already grabbing something
                pinching_hands.append(hand_info)
        
        if len(pinching_hands) >= 2:
            # Calculate distance between the two pinch points
            hand1, hand2 = pinching_hands[0], pinching_hands[1]
            current_distance = math.sqrt(
                (hand1['pinch_center'][0] - hand2['pinch_center'][0])**2 +
                (hand1['pinch_center'][1] - hand2['pinch_center'][1])**2
            )
            
            if not self.two_handed_zoom_active:
                # Initialize two-handed zoom
                self.two_handed_zoom_active = True
                self.two_handed_zoom_initial_distance = current_distance
                self.two_handed_zoom_initial_scale = self.selected_component.scale
                print(f"Started two-handed zoom on {self.selected_component.component_name}")
            else:
                # Apply scaling based on distance change
                if self.two_handed_zoom_initial_distance > 0:
                    scale_factor = current_distance / self.two_handed_zoom_initial_distance
                    target_scale = self.two_handed_zoom_initial_scale * scale_factor
                    
                    # Apply reasonable scaling limits
                    target_scale = max(0.1, min(5.0, target_scale))
                    
                    # Calculate the actual scale factor to apply
                    actual_scale_factor = target_scale / self.selected_component.scale
                    self.selected_component.scale_object(actual_scale_factor)
            
            return True
        else:
            self.two_handed_zoom_active = False
            return False
    
    def _select_component(self, component, position: Tuple[int, int]):
        """Select a component for special operations"""
        # Deselect previous component
        if self.selected_component:
            self.selected_component.selected = None
        
        # Select new component
        self.selected_component = component
        self.selected_component.selected = True
        self.selection_highlight_time = time.time()
        print(f"Selected component: {component.component_name} (use two hands to zoom)")
    
    def _draw_ui(self, frame: np.ndarray, hands_info: List[dict]) -> np.ndarray:
        """Draw user interface elements"""
        # Draw instructions
        instructions = [
            "AR Hand Control - Pinch: grab/move | Double-pinch: select | Two hands: zoom selected",
            f"Selected: {self.selected_component.component_name if self.selected_component else 'None'} | Two-handed zoom: {'ON' if self.two_handed_zoom_active else 'OFF'}",
            "Q: Quit | R: Reset | ESC: Deselect",
            f"CAD Assemblies: {len(self.cad_assemblies)} | Components: {sum(assembly.get_component_count() for assembly in self.cad_assemblies)} | Hands: {len(hands_info)}"
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
                if hand_idx in self.grab_states_components:
                    component = self.grab_states_components[hand_idx]['component']
                    status = f"CAD COMPONENT ✓ ({component.component_name}, Scale: {component.scale:.1f})"
                    color = (255, 0, 255)  # Magenta for grabbed CAD component
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
        
        # CAD assembly info
        total_components = sum(assembly.get_component_count() for assembly in self.cad_assemblies)
        cv2.putText(frame, f"CAD Assemblies: {len(self.cad_assemblies)} ({total_components} components) {'(ON)' if self.show_cad_assemblies else 'OFF'}", 
                   (10, info_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Current CAD assembly selection info
        if self.cad_assemblies and 0 <= self.selected_assembly_index < len(self.cad_assemblies):
            selected_assembly = self.cad_assemblies[self.selected_assembly_index]
            selected_component = selected_assembly.get_selected_component()
            component_info = f" -> {selected_component.component_name}" if selected_component else ""
            cv2.putText(frame, f"Selected: {selected_assembly.name}{component_info}", 
                       (10, info_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Render mode and auto-rotation info
        render_mode = "N/A"
        auto_rotate = False
        if self.cad_assemblies:
            render_mode = self.cad_assemblies[0].render_mode
            auto_rotate = self.cad_assemblies[0].auto_rotate
        
        cv2.putText(frame, f"3D Mode: {render_mode} | Auto-rotate: {'ON' if auto_rotate else 'OFF'}", 
                   (10, info_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
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
