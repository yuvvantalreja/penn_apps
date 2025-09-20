import cv2
import mediapipe as mp
import numpy as np
import math
from typing import List, Tuple, Optional
import time

class VirtualObject:
    """Represents a virtual object that can be manipulated in AR space"""
    
    def __init__(self, x: float, y: float, size: float = 50, color: Tuple[int, int, int] = (0, 255, 255), shape: str = "circle"):
        self.x = x
        self.y = y
        self.size = size
        self.original_size = size
        self.color = color
        self.shape = shape
        self.is_grabbed = False
        self.grabbed_by_hand = None
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
                
            if self.is_grabbed:
                cv2.circle(frame, center, radius + 5, (255, 255, 255), 3)
                
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
        
        return {
            'thumb_pos': thumb_pos,
            'index_pos': index_pos,
            'middle_pos': middle_pos,
            'palm_center': palm_center,
            'wrist_pos': wrist_pos,
            'gestures': gestures,
            'pinch_distance': pinch_distance,
            'is_pinching': pinch_distance < 40
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
        self.cap = None
        self.is_running = False
        
        # Interaction state
        self.grab_states = {}  # hand_idx -> {object, initial_pinch_distance, initial_size}
        self.last_frame_time = time.time()
        
        # Create some initial objects
        self._create_initial_objects()
        
    def _create_initial_objects(self):
        """Create initial virtual objects"""
        self.objects = [
            VirtualObject(200, 200, 60, (0, 255, 255), "circle"),  # Yellow ball
            VirtualObject(400, 300, 80, (255, 100, 100), "cube"),   # Blue cube
            VirtualObject(600, 250, 50, (100, 255, 100), "circle"), # Green ball
        ]
    
    def start(self):
        """Start the AR application"""
        # Try different camera indices
        camera_indices = [0, 1, 2]  # Try multiple camera sources
        self.cap = None
        
        for idx in camera_indices:
            print(f"Trying camera index {idx}...")
            test_cap = cv2.VideoCapture(idx)
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
        
        self.is_running = True
        print("AR Hand Control started!")
        print("Gestures:")
        print("- Pinch (thumb + index) near object: Grab object")
        print("- Move hand while pinching: Move object")
        print("- Pinch and spread: Scale object")
        print("- Press 'q' to quit")
        print("- Press 'r' to reset objects")
        print("- Press 'c' to add new object")
        
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
        
        # Draw objects
        for obj in self.objects:
            frame = obj.draw(frame)
            
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
        elif key == ord('c'):
            self._add_random_object()
    
    def _process_interactions(self, hands_info: List[dict]):
        """Process hand interactions with objects"""
        current_grabs = set()
        
        for hand_info in hands_info:
            hand_idx = hand_info['hand_idx']
            
            if hand_info['is_pinching']:
                self._handle_pinch_interaction(hand_info, hand_idx)
                current_grabs.add(hand_idx)
            else:
                # Release any grabbed object
                if hand_idx in self.grab_states:
                    obj = self.grab_states[hand_idx]['object']
                    obj.is_grabbed = False
                    obj.grabbed_by_hand = None
                    del self.grab_states[hand_idx]
        
        # Clean up grab states for hands that are no longer detected
        hands_to_remove = []
        for hand_idx in self.grab_states:
            if hand_idx not in current_grabs:
                obj = self.grab_states[hand_idx]['object']
                obj.is_grabbed = False
                obj.grabbed_by_hand = None
                hands_to_remove.append(hand_idx)
        
        for hand_idx in hands_to_remove:
            del self.grab_states[hand_idx]
    
    def _handle_pinch_interaction(self, hand_info: dict, hand_idx: int):
        """Handle pinch gesture interaction"""
        index_pos = hand_info['index_pos']
        pinch_distance = hand_info['pinch_distance']
        
        if hand_idx not in self.grab_states:
            # Try to grab an object
            for obj in self.objects:
                if obj.is_point_inside(index_pos[0], index_pos[1]) and not obj.is_grabbed:
                    obj.is_grabbed = True
                    obj.grabbed_by_hand = hand_idx
                    self.grab_states[hand_idx] = {
                        'object': obj,
                        'initial_pinch_distance': pinch_distance,
                        'initial_size': obj.size
                    }
                    break
        else:
            # Continue interaction with grabbed object
            grab_state = self.grab_states[hand_idx]
            obj = grab_state['object']
            
            # Move object to hand position
            obj.move_to(index_pos[0], index_pos[1])
            
            # Scale object based on pinch distance change
            initial_distance = grab_state['initial_pinch_distance']
            scale_factor = pinch_distance / initial_distance if initial_distance > 0 else 1.0
            
            # Apply scaling with some smoothing
            target_size = grab_state['initial_size'] * scale_factor
            obj.size = obj.size * 0.8 + target_size * 0.2  # Smooth scaling
            obj.size = max(20, min(150, obj.size))  # Clamp size
    
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
            "Q: Quit | R: Reset | C: Add Object",
            f"Objects: {len(self.objects)} | Hands: {len(hands_info)}"
        ]
        
        for i, instruction in enumerate(instructions):
            y_pos = 30 + i * 25
            cv2.putText(frame, instruction, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw hand status
        for hand_info in hands_info:
            hand_idx = hand_info['hand_idx']
            palm_pos = hand_info['palm_center']
            
            # Draw palm center
            cv2.circle(frame, palm_pos, 8, (0, 255, 0), -1)
            
            # Show pinch status
            if hand_info['is_pinching']:
                status = "PINCHING"
                color = (0, 0, 255)
                if hand_idx in self.grab_states:
                    obj = self.grab_states[hand_idx]['object']
                    status = f"GRABBED (Size: {int(obj.size)})"
            else:
                status = "OPEN"
                color = (0, 255, 0)
            
            cv2.putText(frame, f"Hand {hand_idx}: {status}", 
                       (palm_pos[0] - 50, palm_pos[1] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
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
