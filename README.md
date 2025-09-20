# AR Hand Control Application

A real-time augmented reality application that allows you to manipulate virtual objects using hand gestures detected through your webcam.

## Features

- **Hand Tracking**: Uses MediaPipe for robust real-time hand detection
- **Object Manipulation**: Grab, move, and scale virtual objects with natural hand gestures
- **Multiple Objects**: Support for circles and cubes with different colors and sizes
- **Real-time Interaction**: Smooth, responsive gesture recognition
- **Multi-hand Support**: Track up to 2 hands simultaneously

## Gestures

- **Pinch (Thumb + Index Finger)**: Grab an object when fingers are close to it
- **Move while Pinching**: Move the grabbed object around the screen
- **Pinch and Spread**: Scale the object larger or smaller by changing pinch distance
- **Open Hand**: Release the currently grabbed object

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

- **Q**: Quit the application
- **R**: Reset all objects to initial positions
- **C**: Add a new random object to the scene

### Tips

- Ensure good lighting for better hand detection
- Keep your hands visible to the camera
- The pinch gesture works best when thumb and index finger are close (< 40 pixels apart)
- Objects will highlight when grabbed
- You can manipulate multiple objects with both hands simultaneously

## Technical Details

- **Hand Detection**: MediaPipe Hands solution
- **Computer Vision**: OpenCV for camera input and rendering
- **Object Rendering**: Custom 2D/pseudo-3D rendering with gradient effects
- **Gesture Recognition**: Real-time finger position analysis
- **Performance**: Optimized for smooth real-time interaction

## Customization

You can easily customize the application by:

- Adding new object shapes in the `VirtualObject.draw()` method
- Implementing new gestures in the `HandGestureDetector._detect_gestures()` method
- Adjusting sensitivity values for pinch detection and scaling
- Modifying colors, sizes, and initial object positions

## Troubleshooting

- **Camera not working**: Make sure no other applications are using the webcam
- **Poor hand detection**: Improve lighting conditions and ensure hands are clearly visible
- **Slow performance**: Close other applications and ensure good CPU resources are available

Enjoy manipulating virtual objects with your hands!
