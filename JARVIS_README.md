# JARVIS Voice Assistant Integration

A sophisticated AI voice assistant built on Gemini Live 2.5 Flash Preview that can analyze screenshots and provide intelligent responses about 3D models and AR content.

## Features

🤖 **Voice Activation**: Say "Hey JARVIS" or press 'J' key to activate
👁️ **Visual Analysis**: Takes screenshots and analyzes what you're pointing at
🎯 **AR Integration**: Works seamlessly with the existing AR hand control system
🗣️ **Natural Speech**: Uses Gemini Live for natural voice conversations
🔍 **Smart Recognition**: Identifies 3D models, CAD components, and technical elements

## Quick Start

### 1. Prerequisites

```bash
# Install Python dependencies
pip install flask flask-cors google-generativeai pillow

# Set up Gemini API key
export GEMINI_API_KEY="your-gemini-api-key-here"
```

### 2. Start the Backend

```bash
# Start JARVIS backend server
python jarvis_backend.py
```

### 3. Run the AR Application

```bash
# Start the main AR hand control application
python main.py
```

### 4. Open the Web Interface

Open `jarvis_integration.html` in your browser or integrate the JavaScript files into your existing web application.

## Usage

### Voice Activation

1. **Wake Word**: Say "Hey JARVIS" to activate
2. **Manual**: Press 'J' key in the OpenCV window
3. **UI Button**: Click the JARVIS indicator in the web interface

### Voice Commands

Once JARVIS is active, you can say:

- **"What is this?"** - Analyzes the current screen/3D model
- **"What am I looking at?"** - Describes visible objects
- **"Analyze this"** - Provides detailed technical analysis
- **"What part is this?"** - Identifies specific components
- **"Describe this object"** - Gives comprehensive description

### Visual Analysis Flow

1. User activates JARVIS (voice or key press)
2. JARVIS greets: "Hello Sir, what can I do for you today?"
3. User asks about visual content: "What is this?"
4. JARVIS automatically:
   - Takes a screenshot of the current view
   - Sends it to Gemini Vision for analysis
   - Provides detailed explanation of what's visible
   - Focuses on 3D models, CAD components, technical details

## Integration with AR Hand Control

### Keyboard Controls

The existing AR application now includes:

```
- Press 'j' to activate JARVIS voice assistant
- Press 'q' to quit
- Press 'r' to reset objects  
- Press 'c' to add new 2D object
- Press '1' to toggle 2D objects
- Press '2' to toggle 3D objects
- Press 'w' to toggle wireframe/solid rendering
- Press 't' to toggle auto-rotation
- Press 'x/y/z' to reset rotation on specific axis
- Press 'space' to cycle through 3D objects
```

### Screenshot Capture

JARVIS can capture screenshots from:
1. **Canvas elements** (for 3D renders) - Primary method
2. **Screen Capture API** - Fallback for full screen
3. **html2canvas** - Last resort for web content

## File Structure

```
penn_apps/
├── jarvis_voice_assistant.js    # Main JARVIS JavaScript class
├── jarvis_ui.css               # JARVIS UI styling
├── jarvis_integration.html     # Example integration
├── jarvis_backend.py          # Flask backend server
├── gemini_vision_api.py       # Gemini Vision API wrapper
├── ar_hand_control.py         # Modified AR app (added J key)
└── js/
    ├── audio-recording-worklet.js
    └── audio-streamer.js
```

## API Endpoints

### JARVIS Backend (Port 5001)

- `GET /api/jarvis/health` - Health check
- `POST /api/jarvis/vision-analyze` - Analyze image
- `POST /api/jarvis/test-vision` - Test vision analysis
- `GET /api/jarvis/status` - Backend status
- `GET /api/gemini/websocket` - Get WebSocket URL

### Example Vision Analysis Request

```javascript
const response = await fetch('http://localhost:5001/api/jarvis/vision-analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        image: base64ImageData,
        prompt: "What do you see in this image, JARVIS?"
    })
});
```

## Configuration

### Environment Variables

```bash
# Required
GEMINI_API_KEY=your-gemini-api-key

# Optional
JARVIS_VOICE=Charon          # Voice for JARVIS (default: Charon)
JARVIS_DEBUG=true            # Enable debug logging
```

### Voice Configuration

JARVIS uses the "Charon" voice by default for a sophisticated British accent. Available voices:
- **Charon** (British, sophisticated) - Default for JARVIS
- **Fenrir** (Deep, authoritative)
- **Aoede** (Clear, professional)
- **Puck** (Friendly, approachable)

## Technical Details

### Audio Processing

- **Input**: 16kHz PCM audio via AudioWorklet
- **Output**: 24kHz audio via AudioStreamer
- **WebSocket**: Direct connection to Gemini Live API
- **Latency**: ~200-500ms for voice responses

### Vision Analysis

- **Model**: Gemini 1.5 Flash Vision
- **Input**: Base64 encoded PNG screenshots
- **Analysis**: Technical component identification
- **Response**: JARVIS-formatted explanations

### Security

- API keys handled securely via backend proxy
- No direct client-side API key exposure
- CORS enabled for frontend integration
- WebSocket connections through secure backend

## Troubleshooting

### Common Issues

1. **"JARVIS not responding"**
   - Check microphone permissions
   - Verify Gemini API key is set
   - Check backend server is running

2. **"Vision analysis failed"**
   - Ensure screenshot capture is working
   - Check network connectivity
   - Verify Gemini Vision API quota

3. **"Audio not working"**
   - Check browser audio permissions
   - Verify AudioStreamer is loaded
   - Test with different browsers

### Debug Mode

Enable debug logging:

```javascript
// In browser console
window.jarvisAssistant.features.debugMode = true;
```

### Testing

```bash
# Test vision API directly
python gemini_vision_api.py path/to/test/image.png

# Test backend health
curl http://localhost:5001/api/jarvis/health

# Test JARVIS status
curl http://localhost:5001/api/jarvis/status
```

## Example Use Cases

### 3D Model Analysis

1. Load a 3D CAD model in the AR application
2. Press 'J' to activate JARVIS
3. Say "What is this part?"
4. JARVIS analyzes the 3D model and explains:
   - Component type and function
   - Technical specifications
   - Engineering context

### AR Object Identification

1. Display AR objects using hand gestures
2. Point at a specific object
3. Activate JARVIS and ask "What am I pointing at?"
4. JARVIS identifies the object and provides details

### Technical Documentation

1. Show technical drawings or CAD assemblies
2. Ask JARVIS "Explain this assembly"
3. Get detailed breakdown of components and relationships

## Advanced Features

### Custom Prompts

You can customize JARVIS's analysis behavior:

```javascript
// Custom analysis prompt
const customPrompt = `You are JARVIS analyzing a mechanical assembly. 
Focus on identifying individual components and their functions.`;

await jarvisAssistant.analyzeWithCustomPrompt(screenshot, customPrompt);
```

### Integration with Existing Systems

JARVIS can be integrated with:
- CAD software interfaces
- Technical documentation systems
- Engineering workflow tools
- Educational platforms

## Performance Optimization

### Audio Optimization

- Use AudioWorklet for low-latency recording
- Implement audio buffering for smooth playback
- Optimize sample rates (16kHz input, 24kHz output)

### Vision Optimization

- Compress screenshots before sending
- Cache analysis results for similar images
- Implement request debouncing

### Network Optimization

- Use WebSocket for real-time communication
- Implement connection pooling
- Add retry logic for failed requests

## Contributing

To extend JARVIS functionality:

1. **Add new voice commands** in `jarvis_voice_assistant.js`
2. **Enhance vision analysis** in `gemini_vision_api.py`
3. **Improve UI** in `jarvis_ui.css`
4. **Add backend endpoints** in `jarvis_backend.py`

## License

This JARVIS integration is part of the AR Hand Control project and follows the same licensing terms.
