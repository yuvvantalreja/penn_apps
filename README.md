# AR Hand Control Application

A real-time augmented reality application that allows you to manipulate 3D CAD assemblies and components using hand gestures detected through your webcam.

## Features

- **Hand Tracking**: Uses MediaPipe for robust real-time hand detection
- **CAD Assembly Manipulation**: Load and manipulate multi-component CAD assemblies
- **Component-Level Control**: Grab and manipulate individual components within assemblies
- **Advanced 3D Controls**: Move, rotate, and scale components using intuitive hand gestures
- **Multiple Render Modes**: Wireframe, solid, and point cloud rendering for 3D objects
- **Real-time Interaction**: Smooth, responsive gesture recognition
- **Multi-hand Support**: Track up to 2 hands simultaneously
- **Assembly Management**: Toggle between different CAD assemblies and components

## Gestures

### Basic Gestures
- **Pinch (Thumb + Index Finger)**: Grab a CAD component when fingers are close to it
- **Move while Pinching**: Move the grabbed component in 3D space
- **Pinch and Spread**: Scale the component larger or smaller by changing pinch distance
- **Open Hand**: Release the currently grabbed component

### CAD-Specific Gestures
- **Component Selection**: Pinch near individual components to grab and manipulate them
- **Independent Movement**: Each component can be moved and scaled independently
- **Assembly Navigation**: Use keyboard controls to cycle through assemblies and components

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have a working webcam connected to your computer.

## Usage

Run the application:
```bash
python main.py
```

### Controls

#### Basic Controls
- **Q**: Quit the application
- **R**: Reset all objects to initial positions

#### 3D Controls
- **1**: Toggle CAD assemblies on/off
- **W**: Toggle between wireframe and solid rendering
- **T**: Toggle auto-rotation for assemblies
- **A**: Cycle through CAD assemblies
- **D**: Cycle through components in selected assembly

#### Advanced Controls
- **X/Y/Z**: Reset rotation on specific axis

### Tips

- Ensure good lighting for better hand detection
- Keep your hands visible to the camera
- The pinch gesture works best when thumb and index finger are close (< 40 pixels apart)
- Objects will highlight when grabbed
- You can manipulate multiple objects with both hands simultaneously

## Technical Details

- **Hand Detection**: MediaPipe Hands solution
- **Computer Vision**: OpenCV for camera input and rendering
- **3D Rendering**: Full 3D pipeline with perspective projection, lighting, and transformation matrices
- **Multi-Component Loading**: Advanced OBJ parser supporting multi-component CAD assemblies
- **3D Mathematics**: Matrix transformations for translation, rotation, and scaling
- **Lighting**: Diffuse lighting model with ambient and directional light
- **Gesture Recognition**: Real-time finger position analysis
- **Performance**: Optimized for smooth real-time interaction with complex CAD models

## CAD Assembly Support

The application supports complex CAD assemblies from OBJ files:

- **Supported Format**: Wavefront OBJ (.obj) files with groups (g) or objects (o)
- **Multi-Component**: Each group/object becomes an independently manipulable component
- **Features**: Vertices (v), faces (f), normals (vn), groups (g), objects (o)
- **Auto-Processing**: Assemblies are automatically normalized and centered
- **Component Colors**: Each component gets a unique color for easy identification

### Adding Your Own CAD Assemblies

1. Place your `.obj` file in the project directory
2. Modify `_create_initial_objects()` in `ar_hand_control.py`
3. Add a new entry to the `cad_assemblies` list

```python
{
    "path": "my_assembly.obj",
    "name": "My CAD Assembly",
    "position": (0.0, 0.0, -2.0),
    "scale": 1.0,
    "color": (255, 100, 100)  # Base color
}
```

## Customization

You can easily customize the application by:

### CAD Assemblies
- Loading different multi-component OBJ models
- Adjusting component colors and scaling
- Modifying initial positions and orientations
- Creating complex assembly hierarchies

### Rendering
- Adjusting render modes (wireframe, solid, points)
- Modifying lighting parameters in `Renderer3D`
- Customizing transformation matrices
- Tuning component highlighting and selection

### Gestures
- Implementing new gestures in the `HandGestureDetector._detect_gestures()` method
- Adjusting sensitivity values for pinch detection and component manipulation
- Adding new interaction modes for assembly navigation

## File Structure

```
penn_apps/
├── main.py                     # Entry point
├── ar_hand_control.py          # Main application logic
├── multi_component_loader.py   # Advanced OBJ loader for CAD assemblies
├── component_object_3d.py      # Individual CAD component class
├── cad_assembly.py             # CAD assembly management
├── renderer_3d.py              # 3D rendering engine
├── complex_cad_assembly.obj    # Multi-component CAD model
├── online/Wooden Crate.obj     # Simple example model
├── requirements.txt            # Dependencies
└── README.md                  # This file
```

## Troubleshooting

### Camera Issues
- **Camera not working**: Make sure no other applications are using the webcam
- **Poor hand detection**: Improve lighting conditions and ensure hands are clearly visible

### Performance Issues
- **Slow performance**: Close other applications and ensure good CPU resources are available
- **3D rendering lag**: Try wireframe mode (press 'W') for better performance
- **Large OBJ files**: Consider using simpler models or reducing polygon count

### CAD Assembly Issues
- **Assembly not loading**: Ensure the OBJ file path is correct and contains groups/objects
- **Components too small/large**: Adjust the scale parameter in the assembly configuration
- **Assembly not visible**: Check the Z position (should be negative, e.g., -3.0)
- **No components detected**: Ensure your OBJ file has group (g) or object (o) declarations

### Installation Issues
- **Module not found**: Run `pip3 install -r requirements.txt`
- **MediaPipe issues**: Try `pip3 install mediapipe --upgrade`
- **OpenCV issues**: Try `pip3 install opencv-python --upgrade`

Enjoy manipulating complex 3D CAD assemblies with your hands! 🎮✨🔧
