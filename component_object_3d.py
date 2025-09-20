import numpy as np
import cv2
import math
from typing import Tuple, Optional
from multi_component_loader import ComponentData
from renderer_3d import Renderer3D, Transform3D


class ComponentObject3D:
    """3D virtual object representing a single CAD component that can be manipulated in AR space"""
    
    def __init__(self, component_data: ComponentData, assembly_name: str = "", 
                 x: float = 0, y: float = 0, z: float = 0, 
                 scale: float = 1.0, color: Tuple[int, int, int] = (100, 150, 255)):
        self.component_data = component_data
        self.component_name = component_data.name
        self.assembly_name = assembly_name
        
        # Position and transformation
        self.x = x
        self.y = y
        self.z = z
        self.scale = scale
        self.original_scale = scale
        self.color = color
        
        # Rotation angles (in radians)
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.rotation_z = 0.0
        
        # Interaction state
        self.is_grabbed = 0  # 0 = not grabbed, 1 = grabbed with 1 hand, 2 = grabbed with 2 hands
        self.grabbed_by_hand = []  # List of hand indices that are grabbing this object
        self.highlighted = False  # For object highlighting
        self.selected = False  # Track selection state for UI highlighting
        
        # Component-specific selection system
        self.is_selected = False  # True when component is in selection mode for rotation
        self.selection_hand_idx = None  # Hand index that selected this component
        self.last_selection_hand_pos = None  # Last position of the selecting hand
        
        # Auto-rotation disabled by default - components only rotate when manipulated
        self.auto_rotate = False
        self.auto_rotation_speed = 0.01
        
        # Pre-calculate geometry data
        self.vertices = component_data.get_vertices_array()
        self.faces = component_data.get_faces_array()
        self.face_normals = component_data.calculate_face_normals()
        self.bounding_box_size = 1.0
        
        # Rendering mode
        self.render_mode = "solid"  # "wireframe", "solid", "points"
        
        # Component visibility
        self.visible = True
        
        # Calculate bounding box for collision detection
        if len(self.vertices) > 0:
            min_bounds, max_bounds = component_data.get_bounding_box()
            self.bounding_box_size = np.max(max_bounds - min_bounds) * self.scale
        
        print(f"Created ComponentObject3D: '{self.component_name}' with {len(self.vertices)} vertices, {len(self.faces)} faces")
    
    def get_model_matrix(self) -> np.ndarray:
        """Get the transformation matrix for this component"""
        # Create transformation matrices
        translation = Transform3D.translation_matrix(self.x, self.y, self.z)
        rotation_x = Transform3D.rotation_matrix_x(self.rotation_x)
        rotation_y = Transform3D.rotation_matrix_y(self.rotation_y)
        rotation_z = Transform3D.rotation_matrix_z(self.rotation_z)
        scale_matrix = Transform3D.scale_matrix(self.scale, self.scale, self.scale)
        
        # Combine transformations: Scale -> Rotate -> Translate
        model_matrix = translation @ rotation_z @ rotation_y @ rotation_x @ scale_matrix
        
        return model_matrix
    
    def draw(self, frame: np.ndarray, renderer: Renderer3D) -> np.ndarray:
        """Draw the component on the frame"""
        if not self.visible or len(self.vertices) == 0 or len(self.faces) == 0:
            return frame
        
        # Update auto-rotation if enabled
        if self.auto_rotate and not self.is_grabbed:
            self.rotation_y += self.auto_rotation_speed
            if self.rotation_y > 2 * math.pi:
                self.rotation_y -= 2 * math.pi
        
        # Get transformation matrix
        model_matrix = self.get_model_matrix()
        
        # Choose color based on state
        if self.is_selected:
            color = (0, 255, 255)  # Cyan when selected for rotation
        elif self.is_grabbed:
            color = tuple(min(255, c + 100) for c in self.color)  # Much brighter when grabbed
        elif self.highlighted:
            color = tuple(min(255, c + 60) for c in self.color)  # Brighter when highlighted
        else:
            color = self.color
        
        # Choose rendering method based on mode
        if self.render_mode == "wireframe":
            frame = renderer.render_wireframe(frame, self.vertices, self.faces, model_matrix, color)
        elif self.render_mode == "solid":
            frame = renderer.render_solid(frame, self.vertices, self.faces, self.face_normals, model_matrix, color)
        elif self.render_mode == "points":
            frame = renderer.render_points(frame, self.vertices, model_matrix, color)
        
        # Draw bounding box if grabbed
        if self.is_grabbed:
            frame = self._draw_bounding_box(frame, renderer, model_matrix)
        
        # Draw component name when highlighted or grabbed
        if self.highlighted or self.is_grabbed:
            frame = self._draw_component_label(frame, renderer)
        
        # Draw pinchable radius highlighting
        frame = self._draw_pinchable_radius(frame, renderer)
        
        return frame
    
    def _draw_bounding_box(self, frame: np.ndarray, renderer: Renderer3D, model_matrix: np.ndarray) -> np.ndarray:
        """Draw a bounding box around the component when grabbed"""
        # Create a simple cube for bounding box
        size = self.bounding_box_size * 0.6
        box_vertices = np.array([
            [-size, -size, -size], [size, -size, -size], [size, size, -size], [-size, size, -size],  # Back face
            [-size, -size, size], [size, -size, size], [size, size, size], [-size, size, size]       # Front face
        ], dtype=np.float32)
        
        box_edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Back face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Front face
            [0, 4], [1, 5], [2, 6], [3, 7]   # Connecting edges
        ]
        
        # Project box vertices
        screen_coords = renderer.project_vertices(box_vertices, model_matrix)
        
        # Draw box edges
        for edge in box_edges:
            start_idx, end_idx = edge
            if start_idx < len(screen_coords) and end_idx < len(screen_coords):
                start_point = tuple(screen_coords[start_idx])
                end_point = tuple(screen_coords[end_idx])
                
                # Check bounds
                if (0 <= start_point[0] < renderer.width and 0 <= start_point[1] < renderer.height and
                    0 <= end_point[0] < renderer.width and 0 <= end_point[1] < renderer.height):
                    cv2.line(frame, start_point, end_point, (0, 255, 255), 2)  # Cyan bounding box
        
        return frame
    
    def _draw_component_label(self, frame: np.ndarray, renderer: Renderer3D) -> np.ndarray:
        """Draw component name label"""
        screen_pos = self.get_screen_position(renderer)
        if screen_pos:
            x, y = int(screen_pos[0]), int(screen_pos[1])
            
            # Create label text
            if self.assembly_name:
                label = f"{self.assembly_name}::{self.component_name}"
            else:
                label = self.component_name
            
            # Draw background rectangle for better readability
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            bg_x1, bg_y1 = x - 5, y - text_size[1] - 10
            bg_x2, bg_y2 = x + text_size[0] + 5, y + 5
            
            # Ensure background is within frame bounds
            bg_x1 = max(0, bg_x1)
            bg_y1 = max(0, bg_y1)
            bg_x2 = min(frame.shape[1], bg_x2)
            bg_y2 = min(frame.shape[0], bg_y2)
            
            # Draw semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Draw text
            cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def _draw_pinchable_radius(self, frame: np.ndarray, renderer: Renderer3D) -> np.ndarray:
        """Draw the pinchable radius highlighting around the component"""
        # Only draw if this component should show the radius
        if not hasattr(self, 'selected') or self.selected is None:
            return frame
        
        screen_pos = self.get_screen_position(renderer)
        if not screen_pos:
            return frame
        
        center = (int(screen_pos[0]), int(screen_pos[1]))
        
        # Calculate pinchable radius
        pinchable_radius = int(max(35, self.bounding_box_size * 50 * self.scale))
        
        # Choose color based on selection state
        if self.selected:
            # Yellow highlighting when selected
            color = (0, 255, 255)  # BGR format: Yellow
        else:
            # Blue highlighting when not selected
            color = (255, 0, 0)  # BGR format: Blue
        
        # Draw the pinchable radius circle
        cv2.circle(frame, center, pinchable_radius, color, 2)
        
        return frame
    
    def is_point_inside(self, x: float, y: float, renderer: Renderer3D) -> bool:
        """Check if a 2D point is inside the projected component"""
        if not self.visible or len(self.vertices) == 0:
            return False
        
        screen_pos = self.get_screen_position(renderer)
        if not screen_pos:
            return False
        
        # Use scaled bounding box for hit detection
        hit_radius = max(35, self.bounding_box_size * 50 * self.scale)
        distance = math.sqrt((x - screen_pos[0]) ** 2 + (y - screen_pos[1]) ** 2)
        
        return distance <= hit_radius
    
    def get_screen_position(self, renderer: Renderer3D) -> Optional[Tuple[float, float]]:
        """Get the screen position of the component's center"""
        if not self.visible or len(self.vertices) == 0:
            return None
        
        # Project component center to screen space
        center_3d = np.array([[self.x, self.y, self.z, 1.0]])
        
        # Use only view and projection matrices
        vp_matrix = renderer.projection_matrix @ renderer.view_matrix
        projected_center = center_3d @ vp_matrix.T
        
        if projected_center[0, 3] <= 0:  # Behind camera
            return None
        
        # Perspective divide
        projected_center[:, :3] /= projected_center[:, 3:4]
        
        # Convert to screen coordinates
        screen_x = (projected_center[0, 0] + 1) * renderer.width / 2
        screen_y = (1 - projected_center[0, 1]) * renderer.height / 2
        
        return (screen_x, screen_y)
    
    def move_to(self, x: float, y: float, z: Optional[float] = None):
        """Move the component to a new position"""
        # Convert 2D screen coordinates to 3D world coordinates
        world_x = (x - 640) / 200.0  # Adjust scaling as needed
        world_y = -(y - 360) / 200.0  # Flip Y and adjust scaling
        
        self.x = world_x
        self.y = world_y
        if z is not None:
            self.z = z
    
    def scale_object(self, scale_factor: float):
        """Scale the component"""
        self.scale = max(0.1, min(5.0, self.original_scale * scale_factor))
        # Update bounding box
        min_bounds, max_bounds = self.component_data.get_bounding_box()
        self.bounding_box_size = np.max(max_bounds - min_bounds) * self.scale
    
    def rotate(self, delta_x: float, delta_y: float, delta_z: float = 0.0):
        """Rotate the component"""
        self.rotation_x += delta_x
        self.rotation_y += delta_y
        self.rotation_z += delta_z
        
        # Keep angles in reasonable range
        self.rotation_x = self.rotation_x % (2 * math.pi)
        self.rotation_y = self.rotation_y % (2 * math.pi)
        self.rotation_z = self.rotation_z % (2 * math.pi)
    
    def set_render_mode(self, mode: str):
        """Set the rendering mode"""
        if mode in ["wireframe", "solid", "points"]:
            self.render_mode = mode
    
    def toggle_auto_rotation(self):
        """Toggle auto-rotation"""
        self.auto_rotate = not self.auto_rotate
    
    def reset_rotation(self):
        """Reset rotation to default"""
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.rotation_z = 0.0
    
    def set_visibility(self, visible: bool):
        """Set component visibility"""
        self.visible = visible
    
    def get_info_string(self) -> str:
        """Get a string with component information"""
        return f"{self.component_name} ({len(self.vertices)} verts, {len(self.faces)} faces)"
