import numpy as np
import cv2
import os
from typing import List, Tuple, Optional, Dict
from multi_component_loader import MultiComponentOBJLoader
from component_object_3d import ComponentObject3D
from renderer_3d import Renderer3D


class CADAssembly:
    """Manages a complete CAD assembly with multiple manipulable components"""
    
    def __init__(self, name: str, obj_path: str, 
                 x: float = 0, y: float = 0, z: float = 0,
                 scale: float = 1.0, base_color: Tuple[int, int, int] = (100, 150, 255)):
        self.name = name
        self.obj_path = obj_path
        self.base_x = x
        self.base_y = y
        self.base_z = z
        self.base_scale = scale
        self.base_color = base_color
        
        # Components
        self.components: List[ComponentObject3D] = []
        self.component_colors = []  # Track colors for each component
        
        # Assembly state
        self.visible = True
        self.selected = False  # Assembly-level selection
        self.highlighted = False  # Assembly-level highlighting
        
        # Auto-rotation for entire assembly
        self.auto_rotate = False
        self.auto_rotation_speed = 0.01
        self.assembly_rotation_y = 0.0
        
        # Rendering mode for all components
        self.render_mode = "solid"
        
        # Component selection tracking
        self.selected_component_index = 0  # Index of currently highlighted component
        
        # Load the assembly
        self.loader = MultiComponentOBJLoader()
        self.load_assembly()
    
    def load_assembly(self) -> bool:
        """Load the CAD assembly from OBJ file"""
        if not self.loader.load_obj(self.obj_path):
            print(f"Failed to load CAD assembly: {self.obj_path}")
            return False
        
        # Normalize the entire assembly
        self.loader.normalize_assembly(target_size=2.0)
        
        # Create component colors - generate different colors for each component
        component_count = self.loader.get_component_count()
        self.component_colors = self._generate_component_colors(component_count)
        
        # Create ComponentObject3D instances for each component
        self.components = []
        for i in range(component_count):
            component_data = self.loader.get_component(i)
            if component_data:
                # Calculate individual component position relative to assembly base
                component_center = component_data.get_center()
                
                # Position each component relative to the assembly base position
                comp_x = self.base_x + component_center[0] * self.base_scale
                comp_y = self.base_y + component_center[1] * self.base_scale  
                comp_z = self.base_z + component_center[2] * self.base_scale
                
                component_obj = ComponentObject3D(
                    component_data=component_data,
                    assembly_name=self.name,
                    x=comp_x,
                    y=comp_y,
                    z=comp_z,
                    scale=self.base_scale,
                    color=self.component_colors[i]
                )
                
                # Set rendering mode
                component_obj.set_render_mode(self.render_mode)
                
                self.components.append(component_obj)
        
        print(f"✅ Loaded CAD Assembly '{self.name}' with {len(self.components)} components")
        return True
    
    def _generate_component_colors(self, count: int) -> List[Tuple[int, int, int]]:
        """Generate distinct colors for each component"""
        colors = []
        
        if count == 1:
            # Single component - use base color
            colors.append(self.base_color)
        else:
            # Multiple components - generate color variations
            base_hue = 0
            if self.base_color == (100, 150, 255):  # Blue base
                base_hue = 0
            elif self.base_color == (255, 150, 100):  # Orange base  
                base_hue = 120
            elif self.base_color == (150, 255, 100):  # Green base
                base_hue = 240
            
            for i in range(count):
                # Create color variations by adjusting hue and brightness
                hue_shift = (360 // count) * i
                brightness_factor = 0.7 + (0.3 * (i % 3) / 2)  # Vary brightness
                
                # Generate color based on base color with variations
                if i == 0:
                    # First component keeps base color
                    colors.append(self.base_color)
                else:
                    # Other components get variations
                    r = min(255, int(self.base_color[0] * brightness_factor))
                    g = min(255, int(self.base_color[1] * brightness_factor * (1 + 0.3 * (i % 2))))
                    b = min(255, int(self.base_color[2] * brightness_factor * (1 + 0.2 * (i % 3))))
                    colors.append((r, g, b))
        
        return colors
    
    def draw(self, frame: np.ndarray, renderer: Renderer3D) -> np.ndarray:
        """Draw all components of the assembly"""
        if not self.visible:
            return frame
        
        # Update assembly-level auto-rotation
        if self.auto_rotate:
            self.assembly_rotation_y += self.auto_rotation_speed
            if self.assembly_rotation_y > 2 * np.pi:
                self.assembly_rotation_y -= 2 * np.pi
            
            # Apply rotation to all components
            for component in self.components:
                if not component.is_grabbed:  # Don't auto-rotate grabbed components
                    component.rotation_y = self.assembly_rotation_y
        
        # Draw each component
        for i, component in enumerate(self.components):
            # Highlight the selected component
            if i == self.selected_component_index and self.highlighted:
                component.highlighted = True
                component.selected = True  # Show pinchable radius
            else:
                component.highlighted = False
                component.selected = None  # Hide pinchable radius
            
            frame = component.draw(frame, renderer)
        
        return frame
    
    def get_component_count(self) -> int:
        """Get the number of components in this assembly"""
        return len(self.components)
    
    def get_component(self, index: int) -> Optional[ComponentObject3D]:
        """Get a component by index"""
        if 0 <= index < len(self.components):
            return self.components[index]
        return None
    
    def get_component_by_name(self, name: str) -> Optional[ComponentObject3D]:
        """Get a component by name"""
        for component in self.components:
            if component.component_name == name:
                return component
        return None
    
    def get_all_components(self) -> List[ComponentObject3D]:
        """Get all components"""
        return self.components
    
    def find_component_at_point(self, x: float, y: float, renderer: Renderer3D) -> Optional[ComponentObject3D]:
        """Find which component (if any) contains the given point"""
        closest_component = None
        closest_distance = float('inf')
        
        for component in self.components:
            if component.visible and component.is_point_inside(x, y, renderer):
                # Calculate distance to component center for prioritization
                screen_pos = component.get_screen_position(renderer)
                if screen_pos:
                    distance = np.sqrt((x - screen_pos[0])**2 + (y - screen_pos[1])**2)
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_component = component
        
        return closest_component
    
    def cycle_selected_component(self):
        """Cycle to next component for highlighting"""
        if self.components:
            self.selected_component_index = (self.selected_component_index + 1) % len(self.components)
            selected_comp = self.components[self.selected_component_index]
            print(f"Selected component: {selected_comp.component_name}")
    
    def get_selected_component(self) -> Optional[ComponentObject3D]:
        """Get the currently selected/highlighted component"""
        if 0 <= self.selected_component_index < len(self.components):
            return self.components[self.selected_component_index]
        return None
    
    def set_render_mode(self, mode: str):
        """Set rendering mode for all components"""
        if mode in ["wireframe", "solid", "points"]:
            self.render_mode = mode
            for component in self.components:
                component.set_render_mode(mode)
    
    def toggle_auto_rotation(self):
        """Toggle auto-rotation for the entire assembly"""
        self.auto_rotate = not self.auto_rotate
        print(f"Assembly '{self.name}' auto-rotation: {'ON' if self.auto_rotate else 'OFF'}")
    
    def reset_positions(self):
        """Reset all components to their initial positions"""
        # Reload the assembly to reset positions
        self.load_assembly()
        print(f"Reset assembly '{self.name}' to initial configuration")
    
    def reset_rotations(self):
        """Reset all component rotations"""
        for component in self.components:
            component.reset_rotation()
        self.assembly_rotation_y = 0.0
        print(f"Reset rotations for assembly '{self.name}'")
    
    def set_visibility(self, visible: bool):
        """Set visibility for entire assembly"""
        self.visible = visible
        for component in self.components:
            component.set_visibility(visible)
    
    def set_component_visibility(self, component_index: int, visible: bool):
        """Set visibility for a specific component"""
        if 0 <= component_index < len(self.components):
            self.components[component_index].set_visibility(visible)
    
    def get_assembly_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the bounding box of the entire assembly"""
        if not self.components:
            return np.array([0, 0, 0]), np.array([0, 0, 0])
        
        all_positions = []
        for component in self.components:
            all_positions.append([component.x, component.y, component.z])
        
        positions = np.array(all_positions)
        min_bounds = np.min(positions, axis=0)
        max_bounds = np.max(positions, axis=0)
        
        return min_bounds, max_bounds
    
    def move_assembly(self, delta_x: float, delta_y: float, delta_z: float = 0.0):
        """Move entire assembly by the given deltas"""
        for component in self.components:
            component.x += delta_x
            component.y += delta_y
            component.z += delta_z
        
        # Update base position
        self.base_x += delta_x
        self.base_y += delta_y
        self.base_z += delta_z
    
    def scale_assembly(self, scale_factor: float):
        """Scale entire assembly uniformly"""
        # Calculate assembly center
        center_x = sum(comp.x for comp in self.components) / len(self.components)
        center_y = sum(comp.y for comp in self.components) / len(self.components)
        center_z = sum(comp.z for comp in self.components) / len(self.components)
        
        # Scale each component relative to center
        for component in self.components:
            # Scale position relative to center
            component.x = center_x + (component.x - center_x) * scale_factor
            component.y = center_y + (component.y - center_y) * scale_factor
            component.z = center_z + (component.z - center_z) * scale_factor
            
            # Scale the component itself
            component.scale_object(scale_factor)
    
    def get_info_string(self) -> str:
        """Get assembly information string"""
        return f"Assembly '{self.name}': {len(self.components)} components"
    
    def get_component_info_strings(self) -> List[str]:
        """Get information strings for all components"""
        return [comp.get_info_string() for comp in self.components]
