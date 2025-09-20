import numpy as np
from typing import List, Tuple, Optional
import os

class OBJLoader:
    """Loader for OBJ 3D model files with multi-component support"""
    
    def __init__(self):
        self.vertices = []
        self.faces = []
        self.normals = []
        self.texture_coords = []
        self.groups = {}  # Dictionary to store groups/components
        self.current_group = "default"  # Current group being processed
        
    def load_obj(self, filepath: str) -> bool:
        """Load an OBJ file and parse its contents with group support"""
        if not os.path.exists(filepath):
            print(f"Error: OBJ file not found: {filepath}")
            return False
            
        self.vertices = []
        self.faces = []
        self.normals = []
        self.texture_coords = []
        self.groups = {}
        self.current_group = "default"
        self.groups[self.current_group] = {"faces": [], "vertex_indices": set()}
        
        try:
            with open(filepath, 'r') as file:
                for line_num, line in enumerate(file):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                        
                    parts = line.split()
                    if not parts:
                        continue
                        
                    if parts[0] == 'v':  # Vertex
                        if len(parts) >= 4:
                            vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                            self.vertices.append(vertex)
                            
                    elif parts[0] == 'vn':  # Vertex normal
                        if len(parts) >= 4:
                            normal = [float(parts[1]), float(parts[2]), float(parts[3])]
                            self.normals.append(normal)
                            
                    elif parts[0] == 'vt':  # Texture coordinate
                        if len(parts) >= 3:
                            tex_coord = [float(parts[1]), float(parts[2])]
                            self.texture_coords.append(tex_coord)
                            
                    elif parts[0] == 'g':  # Group
                        if len(parts) >= 2:
                            self.current_group = parts[1]
                            if self.current_group not in self.groups:
                                self.groups[self.current_group] = {"faces": [], "vertex_indices": set()}
                            
                    elif parts[0] == 'f':  # Face
                        face_vertices = []
                        valid_face = True
                        for vertex_data in parts[1:]:
                            # Handle different face formats: v, v/vt, v/vt/vn, v//vn
                            vertex_indices = vertex_data.split('/')
                            try:
                                vertex_index = int(vertex_indices[0]) - 1  # OBJ indices start at 1
                                face_vertices.append(vertex_index)
                            except (ValueError, IndexError):
                                print(f"Warning: Invalid vertex index in face: {vertex_data}")
                                valid_face = False
                                break
                        
                        # Convert to triangles if necessary
                        if valid_face and len(face_vertices) >= 3:
                            # For quads and higher polygons, triangulate
                            for i in range(1, len(face_vertices) - 1):
                                triangle = [face_vertices[0], face_vertices[i], face_vertices[i + 1]]
                                # Only add triangle if all vertices are valid (will be checked later)
                                self.faces.append(triangle)
                                # Track which group this face belongs to
                                face_index = len(self.faces) - 1
                                self.groups[self.current_group]["faces"].append(face_index)
                                # Track vertex indices used by this group
                                for vertex_idx in triangle:
                                    self.groups[self.current_group]["vertex_indices"].add(vertex_idx)
                                
        except Exception as e:
            print(f"Error loading OBJ file: {e}")
            return False
            
        # Validate and clean up faces
        self._validate_faces()
        
        # Clean up empty groups
        empty_groups = [name for name, data in self.groups.items() if not data["faces"]]
        for name in empty_groups:
            del self.groups[name]
        
        print(f"Loaded OBJ: {len(self.vertices)} vertices, {len(self.faces)} faces, {len(self.groups)} groups")
        if len(self.groups) > 1:
            print(f"Groups: {list(self.groups.keys())}")
        return True
        
    def _validate_faces(self) -> None:
        """Remove faces that reference invalid vertex indices"""
        if not self.faces or not self.vertices:
            return
            
        num_vertices = len(self.vertices)
        valid_faces = []
        invalid_count = 0
        
        for face in self.faces:
            valid_face = True
            for vertex_idx in face:
                if vertex_idx < 0 or vertex_idx >= num_vertices:
                    valid_face = False
                    invalid_count += 1
                    break
            
            if valid_face:
                valid_faces.append(face)
        
        if invalid_count > 0:
            print(f"Warning: Removed {invalid_count} faces with invalid vertex references")
            
        self.faces = valid_faces
        
    def get_vertices(self) -> np.ndarray:
        """Get vertices as numpy array"""
        return np.array(self.vertices, dtype=np.float32)
        
    def get_faces(self) -> np.ndarray:
        """Get faces as numpy array"""
        return np.array(self.faces, dtype=np.int32)
        
    def get_normals(self) -> np.ndarray:
        """Get normals as numpy array"""
        return np.array(self.normals, dtype=np.float32)
        
    def calculate_face_normals(self) -> np.ndarray:
        """Calculate face normals if not provided in OBJ file"""
        if len(self.vertices) == 0 or len(self.faces) == 0:
            return np.array([])
            
        vertices = self.get_vertices()
        faces = self.get_faces()
        face_normals = []
        num_vertices = len(vertices)
        
        for face in faces:
            if len(face) >= 3:
                # Check if all face indices are valid
                valid_face = True
                for idx in face[:3]:  # Only check first 3 vertices for triangle
                    if idx < 0 or idx >= num_vertices:
                        print(f"Warning: Face references invalid vertex index {idx} (max: {num_vertices-1})")
                        valid_face = False
                        break
                
                if not valid_face:
                    # Skip this face and use default normal
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
                    normal = np.array([0, 0, 1])  # Default normal
                    
                face_normals.append(normal)
                
        return np.array(face_normals, dtype=np.float32)
        
    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the bounding box of the model (min, max)"""
        if len(self.vertices) == 0:
            return np.array([0, 0, 0]), np.array([0, 0, 0])
            
        vertices = self.get_vertices()
        min_bounds = np.min(vertices, axis=0)
        max_bounds = np.max(vertices, axis=0)
        
        return min_bounds, max_bounds
        
    def normalize_model(self, target_size: float = 1.0) -> None:
        """Normalize the model to fit within a target size"""
        if len(self.vertices) == 0:
            return
            
        vertices = np.array(self.vertices)
        
        # Center the model
        center = np.mean(vertices, axis=0)
        vertices -= center
        
        # Scale to target size
        max_extent = np.max(np.abs(vertices))
        if max_extent > 0:
            scale_factor = target_size / max_extent
            vertices *= scale_factor
            
        self.vertices = vertices.tolist()
    
    def has_multiple_components(self) -> bool:
        """Check if the OBJ file has multiple components/groups"""
        return len(self.groups) > 1
    
    def get_group_names(self) -> List[str]:
        """Get list of group names"""
        return list(self.groups.keys())
    
    def get_component_data(self, group_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get vertices and faces for a specific component/group"""
        if group_name not in self.groups:
            print(f"Warning: Group '{group_name}' not found")
            return np.array([]), np.array([])
        
        group_data = self.groups[group_name]
        
        # Get unique vertex indices used by this group
        vertex_indices = sorted(list(group_data["vertex_indices"]))
        
        if not vertex_indices:
            return np.array([]), np.array([])
        
        # Extract vertices for this group
        group_vertices = []
        vertex_mapping = {}  # Old index -> new index mapping
        
        for new_idx, old_idx in enumerate(vertex_indices):
            if old_idx < len(self.vertices):
                group_vertices.append(self.vertices[old_idx])
                vertex_mapping[old_idx] = new_idx
        
        # Extract faces for this group and remap vertex indices
        group_faces = []
        for face_idx in group_data["faces"]:
            if face_idx < len(self.faces):
                old_face = self.faces[face_idx]
                # Remap vertex indices to the new local indices
                new_face = []
                for vertex_idx in old_face:
                    if vertex_idx in vertex_mapping:
                        new_face.append(vertex_mapping[vertex_idx])
                
                if len(new_face) == len(old_face):  # Only add if all vertices were mapped
                    group_faces.append(new_face)
        
        return np.array(group_vertices, dtype=np.float32), np.array(group_faces, dtype=np.int32)
    
    def get_all_components(self) -> dict:
        """Get all components as a dictionary of {name: (vertices, faces)}"""
        components = {}
        for group_name in self.groups.keys():
            vertices, faces = self.get_component_data(group_name)
            if len(vertices) > 0 and len(faces) > 0:
                components[group_name] = (vertices, faces)
        return components
