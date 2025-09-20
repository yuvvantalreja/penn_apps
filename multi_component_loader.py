import numpy as np
from typing import List, Tuple, Optional, Dict
import os

class ComponentData:
    """Represents a single component within a CAD assembly"""
    
    def __init__(self, name: str):
        self.name = name
        self.vertices = []
        self.faces = []
        self.normals = []
        self.texture_coords = []
        self.vertex_offset = 0  # Offset to adjust face indices
        
    def add_vertex(self, vertex: List[float]):
        """Add a vertex to this component"""
        self.vertices.append(vertex)
    
    def add_face(self, face: List[int]):
        """Add a face to this component (indices adjusted for this component)"""
        # Adjust indices to be relative to this component's vertices
        adjusted_face = [idx - self.vertex_offset - 1 for idx in face]
        self.faces.append(adjusted_face)
    
    def get_vertices_array(self) -> np.ndarray:
        """Get vertices as numpy array"""
        return np.array(self.vertices, dtype=np.float32) if self.vertices else np.array([])
    
    def get_faces_array(self) -> np.ndarray:
        """Get faces as numpy array"""
        return np.array(self.faces, dtype=np.int32) if self.faces else np.array([])
    
    def calculate_face_normals(self) -> np.ndarray:
        """Calculate face normals for this component"""
        if len(self.vertices) == 0 or len(self.faces) == 0:
            return np.array([])
        
        vertices = self.get_vertices_array()
        faces = self.get_faces_array()
        face_normals = []
        
        for face in faces:
            if len(face) >= 3:
                # Check if all face indices are valid
                valid_face = True
                for idx in face[:3]:
                    if idx < 0 or idx >= len(vertices):
                        valid_face = False
                        break
                
                if not valid_face:
                    face_normals.append(np.array([0, 0, 1], dtype=np.float32))
                    continue
                
                v1 = vertices[face[0]]
                v2 = vertices[face[1]]
                v3 = vertices[face[2]]
                
                # Calculate normal using cross product
                edge1 = v2 - v1
                edge2 = v3 - v1
                normal = np.cross(edge1, edge2)
                
                # Normalize
                length = np.linalg.norm(normal)
                if length > 0:
                    normal = normal / length
                else:
                    normal = np.array([0, 0, 1])
                
                face_normals.append(normal)
        
        return np.array(face_normals, dtype=np.float32)
    
    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the bounding box of this component"""
        if len(self.vertices) == 0:
            return np.array([0, 0, 0]), np.array([0, 0, 0])
        
        vertices = self.get_vertices_array()
        min_bounds = np.min(vertices, axis=0)
        max_bounds = np.max(vertices, axis=0)
        
        return min_bounds, max_bounds
    
    def normalize_vertices(self, center: np.ndarray, scale_factor: float):
        """Normalize vertices relative to assembly center and scale"""
        if len(self.vertices) == 0:
            return
        
        vertices = np.array(self.vertices)
        vertices -= center
        vertices *= scale_factor
        self.vertices = vertices.tolist()
    
    def get_center(self) -> np.ndarray:
        """Get the center point of this component"""
        if len(self.vertices) == 0:
            return np.array([0, 0, 0])
        
        vertices = self.get_vertices_array()
        return np.mean(vertices, axis=0)


class MultiComponentOBJLoader:
    """Enhanced OBJ loader that can parse multiple components/groups"""
    
    def __init__(self):
        self.components = []  # List of ComponentData objects
        self.component_names = []  # List of component names in order
        self.current_component = None
        self.global_vertex_count = 0
        
    def load_obj(self, filepath: str) -> bool:
        """Load an OBJ file and parse its components"""
        if not os.path.exists(filepath):
            print(f"Error: OBJ file not found: {filepath}")
            return False
        
        self.components = []
        self.component_names = []
        self.current_component = None
        self.global_vertex_count = 0
        
        # Create default component in case file has no groups
        self._create_component("default_component")
        
        try:
            with open(filepath, 'r') as file:
                for line_num, line in enumerate(file):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split()
                    if not parts:
                        continue
                    
                    if parts[0] == 'g':  # Group definition
                        if len(parts) > 1:
                            group_name = parts[1]
                            self._create_component(group_name)
                        else:
                            self._create_component(f"group_{len(self.components)}")
                    
                    elif parts[0] == 'o':  # Object definition (alternative to groups)
                        if len(parts) > 1:
                            object_name = parts[1]
                            self._create_component(object_name)
                        else:
                            self._create_component(f"object_{len(self.components)}")
                    
                    elif parts[0] == 'v':  # Vertex
                        if len(parts) >= 4:
                            vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                            self.current_component.add_vertex(vertex)
                            self.global_vertex_count += 1
                    
                    elif parts[0] == 'vn':  # Vertex normal
                        if len(parts) >= 4:
                            normal = [float(parts[1]), float(parts[2]), float(parts[3])]
                            self.current_component.normals.append(normal)
                    
                    elif parts[0] == 'vt':  # Texture coordinate
                        if len(parts) >= 3:
                            tex_coord = [float(parts[1]), float(parts[2])]
                            self.current_component.texture_coords.append(tex_coord)
                    
                    elif parts[0] == 'f':  # Face
                        face_vertices = []
                        valid_face = True
                        for vertex_data in parts[1:]:
                            # Handle different face formats
                            vertex_indices = vertex_data.split('/')
                            try:
                                vertex_index = int(vertex_indices[0])  # OBJ indices start at 1
                                face_vertices.append(vertex_index)
                            except (ValueError, IndexError):
                                print(f"Warning: Invalid vertex index in face: {vertex_data}")
                                valid_face = False
                                break
                        
                        # Convert to triangles if necessary
                        if valid_face and len(face_vertices) >= 3:
                            for i in range(1, len(face_vertices) - 1):
                                triangle = [face_vertices[0], face_vertices[i], face_vertices[i + 1]]
                                self.current_component.add_face(triangle)
        
        except Exception as e:
            print(f"Error loading OBJ file: {e}")
            return False
        
        # Remove empty components and validate
        self._cleanup_components()
        self._validate_and_fix_faces()
        
        print(f"Loaded CAD Assembly with {len(self.components)} components:")
        for i, component in enumerate(self.components):
            print(f"  {i+1}. {component.name}: {len(component.vertices)} vertices, {len(component.faces)} faces")
        
        return len(self.components) > 0
    
    def _create_component(self, name: str):
        """Create a new component and set it as current"""
        # Set vertex offset for face index adjustment
        vertex_offset = self.global_vertex_count
        
        component = ComponentData(name)
        component.vertex_offset = vertex_offset
        
        self.components.append(component)
        self.component_names.append(name)
        self.current_component = component
    
    def _cleanup_components(self):
        """Remove empty components"""
        valid_components = []
        valid_names = []
        
        for i, component in enumerate(self.components):
            if len(component.vertices) > 0:
                valid_components.append(component)
                valid_names.append(self.component_names[i])
        
        self.components = valid_components
        self.component_names = valid_names
    
    def _validate_and_fix_faces(self):
        """Validate and fix face indices for each component"""
        for component in self.components:
            if not component.faces or not component.vertices:
                continue
            
            num_vertices = len(component.vertices)
            valid_faces = []
            invalid_count = 0
            
            for face in component.faces:
                valid_face = True
                for vertex_idx in face:
                    if vertex_idx < 0 or vertex_idx >= num_vertices:
                        valid_face = False
                        invalid_count += 1
                        break
                
                if valid_face:
                    valid_faces.append(face)
            
            if invalid_count > 0:
                print(f"Warning: Component '{component.name}' - removed {invalid_count} faces with invalid vertex references")
            
            component.faces = valid_faces
    
    def get_component_count(self) -> int:
        """Get the number of components"""
        return len(self.components)
    
    def get_component(self, index: int) -> Optional[ComponentData]:
        """Get a component by index"""
        if 0 <= index < len(self.components):
            return self.components[index]
        return None
    
    def get_component_by_name(self, name: str) -> Optional[ComponentData]:
        """Get a component by name"""
        for component in self.components:
            if component.name == name:
                return component
        return None
    
    def get_assembly_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the bounding box of the entire assembly"""
        if not self.components:
            return np.array([0, 0, 0]), np.array([0, 0, 0])
        
        all_vertices = []
        for component in self.components:
            if len(component.vertices) > 0:
                all_vertices.extend(component.vertices)
        
        if not all_vertices:
            return np.array([0, 0, 0]), np.array([0, 0, 0])
        
        vertices = np.array(all_vertices)
        min_bounds = np.min(vertices, axis=0)
        max_bounds = np.max(vertices, axis=0)
        
        return min_bounds, max_bounds
    
    def normalize_assembly(self, target_size: float = 2.0):
        """Normalize the entire assembly to fit within target size"""
        if not self.components:
            return
        
        # Get overall bounding box
        min_bounds, max_bounds = self.get_assembly_bounding_box()
        
        # Calculate center and scale
        center = (min_bounds + max_bounds) / 2
        max_extent = np.max(max_bounds - min_bounds)
        
        if max_extent > 0:
            scale_factor = target_size / max_extent
        else:
            scale_factor = 1.0
        
        # Apply normalization to all components
        for component in self.components:
            component.normalize_vertices(center, scale_factor)
        
        print(f"Normalized assembly: center={center}, scale={scale_factor}")
