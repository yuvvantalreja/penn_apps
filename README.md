# AR Hand Control Application with Jarvis AI Assistant

A real-time augmented reality application that allows you to manipulate both 2D and 3D virtual objects using hand gestures detected through your webcam. Now featuring **Jarvis**, an AI-powered voice assistant that can identify objects and answer questions about your AR scene.

## Features

- **Hand Tracking**: Uses MediaPipe for robust real-time hand detection
- **2D Object Manipulation**: Grab, move, and scale virtual circles and cubes
- **3D Model Rendering**: Load and manipulate 3D OBJ models with full rotation and scaling
- **Dual Object Support**: Switch between 2D and 3D objects or display both simultaneously
- **Advanced 3D Controls**: Rotate 3D models using two-hand gestures
- **Multiple Render Modes**: Wireframe, solid, and point cloud rendering for 3D objects
- **Real-time Interaction**: Smooth, responsive gesture recognition
- **Multi-hand Support**: Track up to 2 hands simultaneously
- **🤖 Jarvis AI Assistant**: Voice-activated AI that can identify objects and answer questions
- **🔍 Computer Vision**: AI-powered object recognition using Gemini Vision API
- **🎤 Voice Control**: Natural language interaction with your AR environment

## Gestures

### Basic Gestures
- **Pinch (Thumb + Index Finger)**: Grab an object when fingers are close to it
- **Move while Pinching**: Move the grabbed object around the screen
- **Pinch and Spread**: Scale the object larger or smaller by changing pinch distance
- **Open Hand**: Release the currently grabbed object

### 3D-Specific Gestures
- **Two-Hand Pinch**: Rotate 3D objects when both hands are pinching
- **Single Hand on 3D Object**: Move and scale 3D models in 3D space
- **Auto-Rotation**: 3D objects rotate automatically when not being manipulated

### 🤖 Jarvis Voice Commands
- **"What is this?"** (while pointing): Identify the object you're pointing at
- **"Describe the scene"**: Get an overview of what's visible in the AR environment
- **"Help"**: List available voice commands
- **General questions**: Ask Jarvis anything about your work or the objects
- **"Goodbye"**: Deactivate Jarvis

## Installation

### Quick Setup
```bash
# Clone or download the project
# Navigate to the project directory
python setup_jarvis.py
```

### Manual Setup
1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your Gemini API key:
   - Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Copy `.env.example` to `.env` and add your key

3. Make sure you have a working webcam and microphone connected to your computer.

## Usage

Run the application:
```bash
python main.py
```

### Controls

#### Basic Controls
- **Q**: Quit the application
- **R**: Reset all objects to initial positions
- **C**: Add a new random 2D object to the scene
- **🤖 J**: Activate/Deactivate Jarvis voice assistant

#### 3D Controls
- **1**: Toggle 2D objects on/off
- **2**: Toggle 3D objects on/off
- **W**: Toggle between wireframe and solid rendering for 3D objects
- **T**: Toggle auto-rotation for 3D objects
- **X/Y/Z**: Reset rotation on specific axis
- **Space**: Cycle through 3D objects

### Tips

- Ensure good lighting for better hand detection
- Keep your hands visible to the camera
- The pinch gesture works best when thumb and index finger are close (< 40 pixels apart)
- Objects will highlight when grabbed
- You can manipulate multiple objects with both hands simultaneously

## Technical Details

- **Hand Detection**: MediaPipe Hands solution
- **Computer Vision**: OpenCV for camera input and rendering
- **2D Rendering**: Custom 2D rendering with gradient effects
- **3D Rendering**: Full 3D pipeline with perspective projection, lighting, and transformation matrices
- **3D Model Loading**: OBJ file parser supporting vertices, faces, and normals
- **3D Mathematics**: Matrix transformations for translation, rotation, and scaling
- **Lighting**: Diffuse lighting model with ambient and directional light
- **Gesture Recognition**: Real-time finger position analysis
- **Performance**: Optimized for smooth real-time interaction with both 2D and 3D objects

## 3D Model Support

The application supports OBJ 3D model files:

- **Supported Format**: Wavefront OBJ (.obj) files
- **Features**: Vertices (v), faces (f), normals (vn)
- **Auto-Processing**: Models are automatically normalized and centered
- **Example**: The included `bow.obj` file demonstrates a complex 3D model

### Adding Your Own 3D Models

1. Place your `.obj` file in the project directory
2. Modify `_create_initial_objects()` in `ar_hand_control.py`
3. Create a new `VirtualObject3D` instance with your model path

```python
my_model = VirtualObject3D("my_model.obj", x=0, y=0, z=-3, scale=1.0, color=(255, 100, 100))
self.objects_3d.append(my_model)
```

## Customization

You can easily customize the application by:

### 2D Objects
- Adding new object shapes in the `VirtualObject.draw()` method
- Modifying colors, sizes, and initial positions

### 3D Objects
- Loading different OBJ models
- Adjusting render modes (wireframe, solid, points)
- Modifying lighting parameters in `Renderer3D`
- Customizing transformation matrices

### Gestures
- Implementing new gestures in the `HandGestureDetector._detect_gestures()` method
- Adjusting sensitivity values for pinch detection and scaling
- Adding new interaction modes for 3D objects

## File Structure

```
penn_apps/
├── main.py                 # Entry point
├── ar_hand_control.py      # Main application logic
├── obj_loader.py           # 3D model loading
├── renderer_3d.py          # 3D rendering engine
├── virtual_object_3d.py    # 3D object class
├── test_3d_rendering.py    # Test script
├── bow.obj                 # Example 3D model
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Troubleshooting

### Camera Issues
- **Camera not working**: Make sure no other applications are using the webcam
- **Poor hand detection**: Improve lighting conditions and ensure hands are clearly visible

### Performance Issues
- **Slow performance**: Close other applications and ensure good CPU resources are available
- **3D rendering lag**: Try wireframe mode (press 'W') for better performance
- **Large OBJ files**: Consider using simpler models or reducing polygon count

### 3D Model Issues
- **Model not loading**: Ensure the OBJ file path is correct and the file is valid
- **Model too small/large**: Adjust the scale parameter when creating VirtualObject3D
- **Model not visible**: Check the Z position (should be negative, e.g., -3.0)

### Installation Issues
- **Module not found**: Run `pip3 install -r requirements.txt`
- **MediaPipe issues**: Try `pip3 install mediapipe --upgrade`
- **OpenCV issues**: Try `pip3 install opencv-python --upgrade`

Enjoy manipulating both 2D and 3D virtual objects with your hands! 🎮✨
