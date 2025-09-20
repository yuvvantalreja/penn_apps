import numpy as np
import cv2
import math
from typing import Tuple, Optional, List, Dict
from obj_loader import OBJLoader
from renderer_3d import Renderer3D, Transform3D

class VirtualObject3D:
    """3D virtual object that can be manipulated in AR space"""
    
    def __init__(self, obj_path: str, x: float = 0, y: float = 0, z: float = 0, 
                 scale: float = 1.0, color: Tuple[int, int, int] = (100, 150, 255)):
        self.obj_path = obj_path
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
        self.highlighted = False  # For object selection
        self.selected = False  # Track selection state for highlighting
        
        # Selection system for rotation
        self.is_selected = False  # True when object is in selection mode for rotation
        self.selection_hand_idx = None  # Hand index that selected this object
        self.last_selection_hand_pos = None  # Last position of the selecting hand
        
        # Explicit rotation state tracking
        self.is_in_rotation_mode = False  # True when object is in rotation mode (1 hand grabbing)
        self.rotation_hand_idx = None  # Hand index that is controlling rotation (the open hand)
        self.last_rotation_hand_pos = None  # Last position of the rotation hand
        
        # Auto-rotation disabled by default - objects only rotate when pinched
        self.auto_rotate = False
        self.auto_rotation_speed = 0.02
        
        # Load 3D model
        self.loader = OBJLoader()
        self.vertices = np.array([])
        self.faces = np.array([])
        self.face_normals = np.array([])
        self.bounding_box_size = 1.0
        
        # Rendering mode
        self.render_mode = "solid"  # "wireframe", "solid", "points"
        
        self.load_model()
        
    def load_model(self) -> bool:
        """Load the 3D model from OBJ file"""
        if not self.loader.load_obj(self.obj_path):
            print(f"Failed to load 3D model: {self.obj_path}")
            return False
            
        self.vertices = self.loader.get_vertices()
        self.faces = self.loader.get_faces()
        
        # Calculate face normals if not provided
        if len(self.loader.get_normals()) > 0:
            self.face_normals = self.loader.get_normals()
        else:
            self.face_normals = self.loader.calculate_face_normals()
        
        # Normalize model to reasonable size
        self.loader.normalize_model(target_size=2.0)
        self.vertices = self.loader.get_vertices()
        
        # Calculate bounding box for collision detection
        if len(self.vertices) > 0:
            min_bounds, max_bounds = self.loader.get_bounding_box()
            self.bounding_box_size = np.max(max_bounds - min_bounds) * self.scale
        
        print(f"Loaded 3D model: {len(self.vertices)} vertices, {len(self.faces)} faces")
        return True
    
    def get_model_matrix(self) -> np.ndarray:
        """Get the transformation matrix for this object"""
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
        """Draw the 3D object on the frame"""
        if len(self.vertices) == 0 or len(self.faces) == 0:
            return frame
        
        # Update auto-rotation
        if self.auto_rotate and not self.is_grabbed:
            self.rotation_y += self.auto_rotation_speed
            if self.rotation_y > 2 * math.pi:
                self.rotation_y -= 2 * math.pi
        
        # Get transformation matrix
        model_matrix = self.get_model_matrix()
        
        # Choose color based on state
        if self.is_in_rotation_mode:
            color = (0, 255, 255)  # Cyan when in rotation mode
        elif self.is_selected:
            color = (255, 255, 0)  # Bright yellow when selected for rotation
        elif self.is_grabbed:
            color = tuple(min(255, c + 80) for c in self.color)  # Brighter when grabbed
        elif self.highlighted:
            color = tuple(min(255, c + 40) for c in self.color)  # Slightly brighter when highlighted
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
        
        # Draw pinchable radius highlighting
        frame = self._draw_pinchable_radius(frame, renderer)
        
        return frame
    
    def _draw_bounding_box(self, frame: np.ndarray, renderer: Renderer3D, model_matrix: np.ndarray) -> np.ndarray:
        """Draw a bounding box around the object when grabbed"""
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
                    cv2.line(frame, start_point, end_point, (255, 255, 0), 2)
        
        return frame
    
    def _draw_pinchable_radius(self, frame: np.ndarray, renderer: Renderer3D) -> np.ndarray:
        """Draw the pinchable radius highlighting around the object"""
        # Only draw if this object should show the radius (controlled by selected state)
        # selected=True means show yellow, selected=False means show blue, None means don't show
        if not hasattr(self, 'selected') or self.selected is None:
            return frame
            
        # Calculate screen position directly without model matrix transformations
        # This ensures the radius is always centered on the object's actual position
        center_3d = np.array([[self.x, self.y, self.z, 1.0]])
        
        # Use only view and projection matrices, not the model matrix
        vp_matrix = renderer.projection_matrix @ renderer.view_matrix
        projected_center = center_3d @ vp_matrix.T
        
        if projected_center[0, 3] <= 0:  # Behind camera
            return frame
        
        # Perspective divide
        projected_center[:, :3] /= projected_center[:, 3:4]
        
        # Convert to screen coordinates
        screen_x = (projected_center[0, 0] + 1) * renderer.width / 2
        screen_y = (1 - projected_center[0, 1]) * renderer.height / 2
        
        center = (int(screen_x), int(screen_y))
        
        # Calculate pinchable radius based on the same logic as is_point_inside
        pinchable_radius = int(max(45, self.bounding_box_size * 65 * self.scale))
        
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
        """Check if a 2D point is inside the projected 3D object"""
        if len(self.vertices) == 0:
            return False
        
        # Project object center to screen space directly without model matrix transformations
        center_3d = np.array([[self.x, self.y, self.z, 1.0]])
        
        # Use only view and projection matrices, not the model matrix
        vp_matrix = renderer.projection_matrix @ renderer.view_matrix
        projected_center = center_3d @ vp_matrix.T
        
        if projected_center[0, 3] <= 0:  # Behind camera
            return False
        
        # Perspective divide
        projected_center[:, :3] /= projected_center[:, 3:4]
        
        # Convert to screen coordinates
        screen_x = (projected_center[0, 0] + 1) * renderer.width / 2
        screen_y = (1 - projected_center[0, 1]) * renderer.height / 2
        
        # Use scaled bounding box for hit detection (moderately increased for easier grabbing)
        hit_radius = max(45, self.bounding_box_size * 65 * self.scale)  # Balanced grab area
        distance = math.sqrt((x - screen_x) ** 2 + (y - screen_y) ** 2)
        
        return distance <= hit_radius
    
    def get_screen_position(self, renderer: Renderer3D) -> Optional[Tuple[float, float]]:
        """Get the screen position of the object's center"""
        if len(self.vertices) == 0:
            return None
        
        # Project object center to screen space directly without model matrix transformations
        center_3d = np.array([[self.x, self.y, self.z, 1.0]])
        
        # Use only view and projection matrices, not the model matrix
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
        """Move the object to a new position"""
        # Convert 2D screen coordinates to 3D world coordinates
        # This is a simplified approach - in practice you'd want proper ray casting
        
        # For now, we'll map screen coordinates to world coordinates
        # assuming a fixed Z plane
        world_x = (x - 640) / 200.0  # Adjust scaling as needed
        world_y = -(y - 360) / 200.0  # Flip Y and adjust scaling
        
        self.x = world_x
        self.y = world_y
        if z is not None:
            self.z = z
    
    def scale_object(self, scale_factor: float):
        """Scale the object"""
        self.scale = max(0.1, min(5.0, self.original_scale * scale_factor))
        self.bounding_box_size = np.max(self.loader.get_bounding_box()[1] - self.loader.get_bounding_box()[0]) * self.scale
    
    def rotate(self, delta_x: float, delta_y: float, delta_z: float = 0.0):
        """Rotate the object"""
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


class CADAssembly:
    """Unified pipeline for loading both single objects and multi-component CAD assemblies"""
    
    def __init__(self, obj_path: str, x: float = 0, y: float = 0, z: float = 0, 
                 scale: float = 1.0, color: Tuple[int, int, int] = (100, 150, 255),
                 name: str = "Assembly"):
        self.obj_path = obj_path
        self.name = name
        self.base_x = x
        self.base_y = y
        self.base_z = z
        self.base_scale = scale
        self.base_color = color
        self.components: Dict[str, VirtualObject3D] = {}
        self.component_offsets: Dict[str, Tuple[float, float, float]] = {}
        self.is_single_object = False
        
        # Load the assembly
        self.load_assembly()
    
    def load_assembly(self) -> bool:
        """Load the assembly - works for both single objects and multi-component assemblies"""
        loader = OBJLoader()
        if not loader.load_obj(self.obj_path):
            print(f"Failed to load object/assembly: {self.obj_path}")
            return False
        
        # Check if it's a single object or multi-component assembly
        if not loader.has_multiple_components():
            self.is_single_object = True
            return self._load_as_single_object(loader)
        else:
            self.is_single_object = False
            return self._load_as_multi_component_assembly(loader)
    
    def _load_as_single_object(self, loader: OBJLoader) -> bool:
        """Load as a single-component assembly"""
        print(f"Loading '{self.name}' as single object")
        
        component_obj = VirtualObject3D.__new__(VirtualObject3D)
        # Initialize without calling __init__ to avoid loading the file again
        component_obj.obj_path = self.obj_path
        component_obj.x = self.base_x
        component_obj.y = self.base_y
        component_obj.z = self.base_z
        component_obj.scale = self.base_scale
        component_obj.original_scale = self.base_scale
        component_obj.color = self.base_color
        
        # Initialize interaction attributes
        component_obj.rotation_x = 0.0
        component_obj.rotation_y = 0.0
        component_obj.rotation_z = 0.0
        component_obj.is_grabbed = 0
        component_obj.grabbed_by_hand = []
        component_obj.highlighted = False
        component_obj.selected = False
        component_obj.is_selected = False
        component_obj.selection_hand_idx = None
        component_obj.last_selection_hand_pos = None
        component_obj.is_in_rotation_mode = False
        component_obj.rotation_hand_idx = None
        component_obj.last_rotation_hand_pos = None
        component_obj.auto_rotate = False
        component_obj.auto_rotation_speed = 0.02
        component_obj.render_mode = "solid"
        
        # Set the geometry directly
        component_obj.loader = loader
        component_obj.vertices = loader.get_vertices()
        component_obj.faces = loader.get_faces()
        
        # Calculate face normals if not provided
        if len(loader.get_normals()) > 0:
            component_obj.face_normals = loader.get_normals()
        else:
            component_obj.face_normals = loader.calculate_face_normals()
        
        # Normalize model to reasonable size
        loader.normalize_model(target_size=2.0)
        component_obj.vertices = loader.get_vertices()
        
        # Calculate bounding box for collision detection
        if len(component_obj.vertices) > 0:
            min_bounds, max_bounds = loader.get_bounding_box()
            component_obj.bounding_box_size = np.max(max_bounds - min_bounds) * component_obj.scale
        else:
            component_obj.bounding_box_size = 1.0
        
        # Store as single component with the assembly name
        self.components[self.name] = component_obj
        self.component_offsets[self.name] = (0, 0, 0)
        
        print(f"✅ Loaded '{self.name}' as single-component assembly")
        return True
    
    def _load_as_multi_component_assembly(self, loader: OBJLoader) -> bool:
        """Load as a multi-component assembly"""
        components = loader.get_all_components()
        print(f"Loading '{self.name}' as multi-component assembly with {len(components)} components: {list(components.keys())}")
        
        # Calculate the overall bounding box to center components relative to assembly origin
        all_vertices = []
        component_centers = {}
        
        for comp_name, (vertices, faces) in components.items():
            if len(vertices) > 0:
                center = np.mean(vertices, axis=0)
                component_centers[comp_name] = center
                all_vertices.extend(vertices)
        
        if all_vertices:
            assembly_center = np.mean(all_vertices, axis=0)
        else:
            assembly_center = np.array([0, 0, 0])
        
        # Create individual VirtualObject3D for each component
        for comp_name, (vertices, faces) in components.items():
            if len(vertices) == 0 or len(faces) == 0:
                continue
                
            # Calculate component offset from assembly center
            component_center = component_centers[comp_name]
            offset = component_center - assembly_center
            
            # Create component object
            component_obj = VirtualObject3D.__new__(VirtualObject3D)
            # Initialize without calling __init__ to avoid loading file
            component_obj.obj_path = f"{self.obj_path}#{comp_name}"
            component_obj.x = self.base_x + offset[0] * self.base_scale
            component_obj.y = self.base_y + offset[1] * self.base_scale
            component_obj.z = self.base_z + offset[2] * self.base_scale
            component_obj.scale = self.base_scale
            component_obj.original_scale = self.base_scale
            component_obj.color = self._get_component_color(comp_name)
            
            # Initialize interaction attributes
            component_obj.rotation_x = 0.0
            component_obj.rotation_y = 0.0
            component_obj.rotation_z = 0.0
            component_obj.is_grabbed = 0
            component_obj.grabbed_by_hand = []
            component_obj.highlighted = False
            component_obj.selected = False
            component_obj.is_selected = False
            component_obj.selection_hand_idx = None
            component_obj.last_selection_hand_pos = None
            component_obj.is_in_rotation_mode = False
            component_obj.rotation_hand_idx = None
            component_obj.last_rotation_hand_pos = None
            component_obj.auto_rotate = False
            component_obj.auto_rotation_speed = 0.02
            component_obj.render_mode = "solid"
            
            # Create a temporary loader for this component
            temp_loader = OBJLoader()
            temp_loader.vertices = vertices.tolist()
            temp_loader.faces = faces.tolist()
            
            component_obj.loader = temp_loader
            component_obj.vertices = vertices
            component_obj.faces = faces
            component_obj.face_normals = temp_loader.calculate_face_normals()
            
            # Calculate bounding box
            if len(vertices) > 0:
                min_bounds = np.min(vertices, axis=0)
                max_bounds = np.max(vertices, axis=0)
                component_obj.bounding_box_size = np.max(max_bounds - min_bounds) * component_obj.scale
            else:
                component_obj.bounding_box_size = 1.0
            
            self.components[comp_name] = component_obj
            self.component_offsets[comp_name] = tuple(offset)
        
        print(f"✅ Loaded '{self.name}' with {len(self.components)} components:")
        for comp_name in self.components.keys():
            print(f"   - {comp_name}")
        
        return True
    
    def _get_component_color(self, component_name: str) -> Tuple[int, int, int]:
        """Get a unique color for each component"""
        # Define colors for different components
        colors = {
            "housing": (100, 150, 255),      # Blue
            "shaft": (255, 100, 100),        # Red
            "gear_wheel": (100, 255, 100),   # Green
            "bracket": (255, 255, 100),      # Yellow
            "connector": (255, 100, 255),    # Magenta
            "gear": (100, 255, 100),         # Green (alternative name)
            "wheel": (255, 150, 100),        # Orange
            "support": (255, 255, 100),      # Yellow (alternative)
            "base": (150, 150, 150),         # Gray
            "arm": (255, 150, 50),           # Orange
            "joint": (200, 100, 200),        # Purple
        }
        
        # Return specific color if available, otherwise generate based on hash
        if component_name.lower() in colors:
            return colors[component_name.lower()]
        
        # Generate color based on component name hash for unknown components
        hash_val = hash(component_name) % 360
        # Convert HSV to RGB for varied colors
        import colorsys
        rgb = colorsys.hsv_to_rgb(hash_val / 360.0, 0.7, 0.9)
        return (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))  # BGR format
    
    def get_all_components(self) -> List[VirtualObject3D]:
        """Get all components as a list for adding to objects_3d"""
        return list(self.components.values())
    
    def get_components(self) -> Dict[str, VirtualObject3D]:
        """Get all components as a dictionary"""
        return self.components
    
    def get_component_names(self) -> List[str]:
        """Get list of component names"""
        return list(self.components.keys())
    
    def get_component(self, name: str) -> Optional[VirtualObject3D]:
        """Get a specific component by name"""
        return self.components.get(name)
    
    def is_any_component_grabbed(self) -> bool:
        """Check if any component is currently grabbed"""
        return any(comp.is_grabbed > 0 for comp in self.components.values())
    
    def set_render_mode(self, mode: str):
        """Set render mode for all components"""
        for component in self.components.values():
            component.set_render_mode(mode)
    
    def set_auto_rotation_speed(self, speed: float):
        """Set auto rotation speed for all components"""
        for i, component in enumerate(self.components.values()):
            component.auto_rotation_speed = speed + i * 0.005  # Slight variation
    
    def toggle_auto_rotation(self):
        """Toggle auto rotation for all components"""
        for component in self.components.values():
            component.toggle_auto_rotation()
    
    def reset_rotation(self):
        """Reset rotation for all components"""
        for component in self.components.values():
            component.reset_rotation()
    
    def get_assembly_info(self) -> str:
        """Get assembly information string"""
        if self.is_single_object:
            return f"{self.name} (Single Object)"
        else:
            return f"{self.name} ({len(self.components)} Components)"