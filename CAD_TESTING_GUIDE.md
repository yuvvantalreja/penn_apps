# CAD Component Testing Guide

## Overview
The AR Hand Control application now supports multiple 3D CAD-like components that you can manipulate individually using hand gestures. This guide explains how to test the different components and interactions.

## Available CAD Models

### 1. Complex CAD Assembly (`complex_cad_assembly.obj`)
**Components:**
- **Housing**: Main rectangular block base
- **Shaft**: Vertical cylindrical shaft
- **Gear Wheel**: Circular gear with teeth at the top
- **Support Bracket**: Side mounting bracket
- **Connector**: Small rectangular connector piece

**Features:**
- Multiple distinct geometric components
- Different component types (blocks, cylinders, gears)
- Realistic CAD-like assembly structure

### 2. Simple CAD Assembly (`simple_cad_assembly.obj`)
**Components:**
- **Base Plate**: Flat rectangular foundation
- **Cylinder**: Vertical shaft/rod
- **Hexagonal Gear**: 6-sided gear component
- **Small Cube**: Connector/joint element

**Features:**
- Basic geometric shapes
- Clear component separation
- Good for testing basic interactions

## Hand Gesture Controls

### Basic Manipulation
- **Pinch to Grab**: Use thumb and index finger pinch gesture near any component
- **Move While Pinching**: Drag components around in 3D space
- **Pinch and Spread**: Change pinch distance to scale components
- **Release**: Open hand to release the component

### Advanced 3D Controls
- **Two-Hand Rotation**: Use both hands pinching simultaneously to rotate objects
- **Individual Component Selection**: Each component can be grabbed and manipulated separately

## Keyboard Controls

### Object Management
- **1**: Toggle 2D objects on/off
- **2**: Toggle 3D objects on/off
- **Space**: Cycle through 3D objects (highlights next object)

### Rendering Controls
- **W**: Toggle between wireframe and solid rendering
- **T**: Toggle auto-rotation on/off

### Rotation Controls
- **X**: Reset X-axis rotation for all objects
- **Y**: Reset Y-axis rotation for all objects  
- **Z**: Reset Z-axis rotation for all objects

### General Controls
- **R**: Reset all objects to initial positions
- **Q**: Quit application

## Testing Scenarios

### 1. Individual Component Manipulation
1. Start the application
2. Use pinch gesture to grab different components of the CAD assembly
3. Move each component independently
4. Test scaling by changing pinch distance
5. Verify that each component responds individually

### 2. Multi-Component Assembly
1. Grab one component (e.g., the main housing)
2. Position it in desired location
3. Grab another component (e.g., the shaft)
4. Try to "assemble" components by positioning them relative to each other
5. Test how multiple components interact

### 3. Rotation Testing
1. Use two-hand pinch gesture to rotate entire assemblies
2. Test X, Y, Z axis reset controls
3. Compare auto-rotation vs manual rotation
4. Test rotation of different component types

### 4. Rendering Mode Testing
1. Press 'W' to switch between wireframe and solid rendering
2. Observe how different components appear in each mode
3. Test interaction in both rendering modes
4. Verify that complex geometries are clearly visible

### 5. Multi-Object Scene
1. Load multiple CAD assemblies simultaneously
2. Use Space bar to cycle through objects
3. Test grabbing different assemblies
4. Verify that highlighted objects are clearly indicated

## Visual Feedback

### Object States
- **Normal**: Default color
- **Highlighted**: Slightly brighter (when selected with spacebar)
- **Grabbed**: Much brighter color + bounding box
- **Auto-rotating**: Smooth Y-axis rotation when not grabbed

### Component Identification
- Each CAD assembly has distinct colors
- Complex assembly: Blue tones
- Simple assembly: Orange tones
- Test object: Purple tones
- Bow model: Brown tones

## Performance Tips

1. **Wireframe Mode**: Use for better performance with complex models
2. **Auto-rotation Off**: Disable for more precise manual control
3. **Single Object Focus**: Use spacebar to focus on one object at a time
4. **Camera Positioning**: Ensure good lighting for hand detection

## Troubleshooting

### Component Not Responding
- Ensure pinch gesture is close enough to the component
- Check that the object is not already grabbed by another hand
- Verify the component is visible and not behind other objects

### Performance Issues
- Switch to wireframe mode (W key)
- Disable auto-rotation (T key)
- Reduce number of loaded objects

### Hand Detection Issues
- Improve lighting conditions
- Keep hands clearly visible to camera
- Ensure pinch gesture is distinct (thumb and index finger close)

## Development Notes

The CAD components are defined using standard OBJ format with:
- Vertex positions (v)
- Face definitions (f)
- Group names (g) for component identification

Each component can be extended with:
- Texture coordinates (vt)
- Vertex normals (vn)
- Material definitions (mtl files)

This testing framework provides a foundation for more complex CAD manipulation scenarios and can be extended for specific engineering applications.
