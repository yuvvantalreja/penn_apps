// Jarvis Voice Assistant - AI Vision Analysis with Voice Interface
// Built on Gemini Live 2.5 Flash Preview for voice + Gemini Vision for screenshot analysis

class JarvisVoiceAssistant {
    constructor() {
        this.websocket = null;
        this.isConnected = false;
        this.isListening = false;
        this.isAnalyzing = false;
        
        // Audio management
        this.audioContext = null;
        this.audioStreamer = null;
        this.mediaRecorder = null;
        this.audioStream = null;
        this.isRecording = false;
        
        // Audio worklet for real-time processing
        this.audioWorkletNode = null;
        this.mediaStreamSource = null;
        
        // State management
        this.currentSession = null;
        this.sessionStartTime = null;
        
        // Voice recognition for wake word detection
        this.speechRecognition = null;
        this.isWakeWordMode = true;
        
        console.log('🤖 Jarvis Voice Assistant initialized');
    }
    
    async initialize() {
        try {
            console.log('🚀 Initializing Jarvis Voice Assistant...');
            
            // Initialize audio context (24kHz for Gemini Live compatibility)
            await this.initializeAudioContext();
            
            // Initialize audio streamer for voice playback
            if (window.AudioStreamer) {
                this.audioStreamer = new window.AudioStreamer(this.audioContext, () => {
                    console.log('🔇 Jarvis audio playback complete');
                    this.updateSpeakingStatus(false);
                });
                console.log('✅ AudioStreamer initialized for Jarvis');
            }
            
            // Setup wake word detection
            await this.setupWakeWordDetection();
            
            console.log('✅ Jarvis Voice Assistant ready');
            return true;
            
        } catch (error) {
            console.error('❌ Failed to initialize Jarvis:', error);
            return false;
        }
    }
    
    async initializeAudioContext() {
        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: 24000 // Gemini Live uses 24kHz
            });
            console.log('🎵 Jarvis audio context initialized at 24kHz');
        } catch (error) {
            console.error('❌ Failed to initialize Jarvis audio context:', error);
            throw error;
        }
    }
    
    async setupWakeWordDetection() {
        try {
            if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
                console.warn('⚠️ Speech recognition not supported for wake word detection');
                return;
            }
            
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            this.speechRecognition = new SpeechRecognition();
            
            this.speechRecognition.continuous = true;
            this.speechRecognition.interimResults = true;
            this.speechRecognition.lang = 'en-US';
            
            this.speechRecognition.onresult = (event) => {
                for (let i = event.resultIndex; i < event.results.length; i++) {
                    const result = event.results[i];
                    const transcript = result[0].transcript.toLowerCase();
                    
                    if (result.isFinal && this.isWakeWordMode) {
                        console.log('🎤 Speech detected:', transcript);
                        
                        // Check for wake words
                        if (transcript.includes('jarvis') || transcript.includes('hey jarvis')) {
                            console.log('🤖 Wake word detected! Activating Jarvis...');
                            this.activateJarvis();
                        }
                    }
                }
            };
            
            this.speechRecognition.onerror = (event) => {
                console.error('❌ Speech recognition error:', event.error);
                // Auto-restart on error
                setTimeout(() => {
                    if (this.isWakeWordMode && this.speechRecognition) {
                        try {
                            this.speechRecognition.start();
                        } catch (e) {
                            console.warn('⚠️ Could not restart speech recognition');
                        }
                    }
                }, 1000);
            };
            
            this.speechRecognition.onend = () => {
                // Auto-restart for continuous wake word detection
                if (this.isWakeWordMode && !this.isListening) {
                    setTimeout(() => {
                        try {
                            this.speechRecognition.start();
                        } catch (e) {
                            console.warn('⚠️ Could not restart wake word detection');
                        }
                    }, 100);
                }
            };
            
            // Start wake word detection
            this.speechRecognition.start();
            console.log('👂 Wake word detection started (say "Hey Jarvis")');
            
        } catch (error) {
            console.error('❌ Failed to setup wake word detection:', error);
        }
    }
    
    async activateJarvis() {
        try {
            if (this.isListening) {
                console.log('🤖 Jarvis already active');
                return;
            }
            
            console.log('🤖 Activating Jarvis Voice Assistant...');
            
            // Stop wake word detection temporarily
            this.isWakeWordMode = false;
            if (this.speechRecognition) {
                this.speechRecognition.stop();
            }
            
            // Connect to Gemini Live
            await this.connectToGeminiLive();
            
            // Start conversation session
            await this.startJarvisSession();
            
            // Setup audio streaming
            await this.setupAudioInput();
            await this.startAudioStreaming();
            
            this.isListening = true;
            
            // Update UI
            this.updateUI('active');
            
            console.log('✅ Jarvis activated and listening');
            
        } catch (error) {
            console.error('❌ Failed to activate Jarvis:', error);
            this.deactivateJarvis();
        }
    }
    
    async connectToGeminiLive() {
        return new Promise(async (resolve, reject) => {
            try {
                console.log('🔌 Connecting Jarvis to Gemini Live...');
                
                // Get secure WebSocket URL from backend
                const apiBaseUrl = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' ? 
                    'http://localhost:5001' : 'https://pitchperfect2-api-373812504656.asia-southeast1.run.app';
                
                const response = await fetch(`${apiBaseUrl}/api/gemini/websocket`);
                if (!response.ok) {
                    throw new Error('Failed to get secure Gemini WebSocket URL');
                }
                
                const data = await response.json();
                if (data.status !== 'success') {
                    throw new Error(data.error || 'Backend returned error');
                }
                
                console.log('🔒 Got secure WebSocket URL for Jarvis');
                this.connectWebSocket(data.websocket_url, resolve, reject);
                
            } catch (error) {
                console.error('❌ Failed to connect Jarvis to Gemini Live:', error);
                reject(error);
            }
        });
    }
    
    connectWebSocket(websocketUrl, resolve, reject) {
        this.websocket = new WebSocket(websocketUrl);
        
        this.websocket.onopen = () => {
            console.log('✅ Jarvis WebSocket connected to Gemini Live');
            this.isConnected = true;
            resolve();
        };
        
        this.websocket.onmessage = async (event) => {
            await this.handleGeminiMessage(event.data);
        };
        
        this.websocket.onclose = (event) => {
            console.log('🔌 Jarvis WebSocket closed:', event.code, event.reason);
            this.isConnected = false;
            this.deactivateJarvis();
        };
        
        this.websocket.onerror = (error) => {
            console.error('❌ Jarvis WebSocket error:', error);
            this.isConnected = false;
            reject(error);
        };
    }
    
    async startJarvisSession() {
        try {
            const jarvisSystemPrompt = `You are JARVIS, Tony Stark's AI assistant from Iron Man. You are sophisticated, helpful, and speak with British elegance and wit.

PERSONALITY:
- Speak with refined British accent and vocabulary
- Be helpful but occasionally witty or sarcastic
- Address the user as "Sir" or "Mr. Stark" 
- Show intelligence and capability in your responses
- Keep responses concise but informative

CAPABILITIES:
- You can analyze images and identify objects, components, and details
- You have access to visual analysis when screenshots are provided
- You can explain technical details about what you see
- You understand 3D models, CAD designs, and engineering components

CONVERSATION FLOW:
1. When activated, greet the user: "Hello Sir, what can I do for you today?"
2. Listen for requests like "What is this?" or "Analyze this"
3. When asked about visual content, wait for image analysis results
4. Provide detailed, intelligent explanations of what you observe
5. Ask follow-up questions if clarification is needed

RESPONSE STYLE:
- Always maintain JARVIS's sophisticated tone
- Be direct and informative
- Use technical terminology when appropriate
- Show confidence in your analysis
- End conversations gracefully when dismissed

Remember: You are an advanced AI assistant capable of visual analysis and technical explanation.`;

            const setupMessage = {
                setup: {
                    model: "models/gemini-live-2.5-flash-preview",
                    generation_config: {
                        response_modalities: ["audio"],
                        speech_config: {
                            voice_config: {
                                prebuilt_voice_config: {
                                    voice_name: "Charon" // British-sounding voice for JARVIS
                                }
                            },
                            language_code: "en-US"
                        }
                    },
                    input_audio_transcription: {},
                    output_audio_transcription: {},
                    system_instruction: {
                        parts: [{ text: jarvisSystemPrompt }]
                    }
                }
            };
            
            console.log('📝 Sending JARVIS session setup to Gemini Live...');
            this.websocket.send(JSON.stringify(setupMessage));
            
            // Send initial greeting trigger
            setTimeout(() => {
                if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
                    const greetingTrigger = {
                        clientContent: {
                            turns: [{
                                role: "user",
                                parts: [{ text: "Hello JARVIS, I need your assistance." }]
                            }],
                            turnComplete: true
                        }
                    };
                    
                    this.websocket.send(JSON.stringify(greetingTrigger));
                    console.log('✅ JARVIS greeting triggered');
                }
            }, 100);
            
            this.currentSession = {
                startTime: Date.now(),
                type: 'jarvis_assistant'
            };
            
        } catch (error) {
            console.error('❌ Failed to start JARVIS session:', error);
            throw error;
        }
    }
    
    async setupAudioInput() {
        try {
            this.audioStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: 16000,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: false
                }
            });
            
            console.log('🎤 JARVIS audio input setup complete');
            
        } catch (error) {
            console.error('❌ Failed to setup JARVIS audio input:', error);
            throw error;
        }
    }
    
    async startAudioStreaming() {
        try {
            console.log('📡 Starting JARVIS audio streaming...');
            
            // Create 16kHz recording context
            const recordingContext = new AudioContext({ sampleRate: 16000 });
            
            // Load AudioWorklet module
            await recordingContext.audioWorklet.addModule('js/audio-recording-worklet.js');
            
            // Create audio source and worklet
            this.mediaStreamSource = recordingContext.createMediaStreamSource(this.audioStream);
            this.audioWorkletNode = new AudioWorkletNode(recordingContext, 'audio-recording-worklet');
            
            // Handle PCM16 data from worklet
            this.audioWorkletNode.port.onmessage = (event) => {
                if (event.data.event === 'chunk') {
                    this.handleAudioChunk(event.data.data.int16arrayBuffer);
                }
            };
            
            // Connect audio graph
            this.mediaStreamSource.connect(this.audioWorkletNode);
            
            this.isRecording = true;
            console.log('🎵 JARVIS audio streaming active');
            
        } catch (error) {
            console.error('❌ Failed to start JARVIS audio streaming:', error);
            throw error;
        }
    }
    
    handleAudioChunk(int16ArrayBuffer) {
        try {
            if (!this.websocket || this.websocket.readyState !== WebSocket.OPEN) {
                return;
            }
            
            // Convert ArrayBuffer to base64
            const bytes = new Uint8Array(int16ArrayBuffer);
            let binary = '';
            for (let i = 0; i < bytes.byteLength; i++) {
                binary += String.fromCharCode(bytes[i]);
            }
            const base64Audio = btoa(binary);
            
            // Send to Gemini Live
            const audioMessage = {
                realtime_input: {
                    media_chunks: [{
                        mime_type: 'audio/pcm',
                        data: base64Audio
                    }]
                }
            };
            
            this.websocket.send(JSON.stringify(audioMessage));
            
        } catch (error) {
            console.error('❌ Failed to handle JARVIS audio chunk:', error);
        }
    }
    
    async handleGeminiMessage(data) {
        try {
            // Handle Blob data
            if (data instanceof Blob) {
                const text = await data.text();
                try {
                    const json = JSON.parse(text);
                    await this.handleGeminiMessage(text);
                } catch (e) {
                    // Ignore non-JSON blobs
                }
                return;
            }
            
            // Ignore ArrayBuffer data
            if (data instanceof ArrayBuffer) {
                return;
            }
            
            // Handle JSON messages
            if (typeof data === 'string') {
                let message;
                try {
                    message = JSON.parse(data);
                } catch (parseError) {
                    return;
                }
                
                if (message.serverContent) {
                    await this.processServerContent(message.serverContent);
                } else if (message.setupComplete) {
                    console.log('✅ JARVIS session setup complete');
                } else if (message.error) {
                    console.error('❌ JARVIS Gemini error:', message.error);
                }
            }
            
        } catch (error) {
            console.error('❌ Failed to handle JARVIS Gemini message:', error);
        }
    }
    
    async processServerContent(content) {
        try {
            if (content.modelTurn) {
                const parts = content.modelTurn.parts;
                
                for (const part of parts) {
                    if (part.text) {
                        console.log('🤖 JARVIS text:', part.text);
                        
                        // Check if JARVIS is asking about visual content
                        if (this.isVisualAnalysisRequest(part.text)) {
                            console.log('👁️ JARVIS requesting visual analysis...');
                            await this.performVisualAnalysis();
                        }
                    }
                    
                    if (part.inlineData && part.inlineData.mimeType?.includes('audio/pcm')) {
                        this.streamAudioData(part.inlineData.data);
                    }
                }
            }
            
            if (content.turnComplete) {
                console.log('🔄 JARVIS turn complete');
                this.updateSpeakingStatus(false);
            }
            
        } catch (error) {
            console.error('❌ Failed to process JARVIS server content:', error);
        }
    }
    
    isVisualAnalysisRequest(text) {
        const visualKeywords = [
            'what is this',
            'what am i looking at',
            'analyze this',
            'what do you see',
            'identify this',
            'what is that',
            'describe this',
            'what part is this'
        ];
        
        const lowerText = text.toLowerCase();
        return visualKeywords.some(keyword => lowerText.includes(keyword));
    }
    
    async performVisualAnalysis() {
        try {
            if (this.isAnalyzing) {
                console.log('🔍 Visual analysis already in progress...');
                return;
            }
            
            this.isAnalyzing = true;
            this.updateUI('analyzing');
            
            console.log('📸 Taking screenshot for visual analysis...');
            
            // Capture screenshot of current canvas/screen
            const screenshot = await this.captureScreenshot();
            
            if (!screenshot) {
                throw new Error('Failed to capture screenshot');
            }
            
            console.log('🧠 Sending screenshot to Gemini Vision for analysis...');
            
            // Analyze with Gemini Vision
            const analysis = await this.analyzeWithGeminiVision(screenshot);
            
            if (analysis) {
                console.log('✅ Visual analysis complete:', analysis.substring(0, 100) + '...');
                
                // Send analysis results back to JARVIS
                await this.sendAnalysisToJarvis(analysis);
            }
            
        } catch (error) {
            console.error('❌ Failed to perform visual analysis:', error);
            
            // Send error message to JARVIS
            const errorMessage = "I apologize, Sir, but I'm having difficulty with the visual analysis system at the moment.";
            await this.sendAnalysisToJarvis(errorMessage);
            
        } finally {
            this.isAnalyzing = false;
            this.updateUI('active');
        }
    }
    
    async captureScreenshot() {
        try {
            // Try to capture from canvas first (for 3D renders)
            const canvas = document.querySelector('canvas');
            if (canvas) {
                console.log('📸 Capturing from canvas element');
                return canvas.toDataURL('image/png');
            }
            
            // Fallback to screen capture API if available
            if (navigator.mediaDevices && navigator.mediaDevices.getDisplayMedia) {
                console.log('📸 Using screen capture API');
                
                const stream = await navigator.mediaDevices.getDisplayMedia({
                    video: { mediaSource: 'screen' }
                });
                
                const video = document.createElement('video');
                video.srcObject = stream;
                video.play();
                
                return new Promise((resolve) => {
                    video.onloadedmetadata = () => {
                        const canvas = document.createElement('canvas');
                        canvas.width = video.videoWidth;
                        canvas.height = video.videoHeight;
                        
                        const ctx = canvas.getContext('2d');
                        ctx.drawImage(video, 0, 0);
                        
                        // Stop the stream
                        stream.getTracks().forEach(track => track.stop());
                        
                        resolve(canvas.toDataURL('image/png'));
                    };
                });
            }
            
            // Last resort: try html2canvas if available
            if (window.html2canvas) {
                console.log('📸 Using html2canvas fallback');
                const canvas = await html2canvas(document.body);
                return canvas.toDataURL('image/png');
            }
            
            throw new Error('No screenshot method available');
            
        } catch (error) {
            console.error('❌ Failed to capture screenshot:', error);
            return null;
        }
    }
    
    async analyzeWithGeminiVision(imageDataUrl) {
        try {
            // Convert data URL to base64
            const base64Image = imageDataUrl.split(',')[1];
            
            // Get API base URL
            const apiBaseUrl = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' ? 
                'http://localhost:5001' : 'https://pitchperfect2-api-373812504656.asia-southeast1.run.app';
            
            const analysisPrompt = `You are JARVIS, Tony Stark's AI assistant. Analyze this image and identify what the user is pointing at or what's prominently displayed. 

Focus on:
1. Main objects or components visible
2. Technical details if it's engineering/CAD content
3. Specific parts or features that stand out
4. Any 3D models, mechanical components, or technical elements

Respond as JARVIS would - sophisticated, detailed, and helpful. If you see 3D models or technical components, explain what they are and their likely function.

Be specific about what you observe and maintain JARVIS's characteristic wit and intelligence.`;
            
            const response = await fetch(`${apiBaseUrl}/api/gemini/vision-analyze`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    image: base64Image,
                    prompt: analysisPrompt
                })
            });
            
            if (!response.ok) {
                throw new Error(`Vision API error: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.status === 'success') {
                return data.analysis;
            } else {
                throw new Error(data.error || 'Vision analysis failed');
            }
            
        } catch (error) {
            console.error('❌ Failed to analyze with Gemini Vision:', error);
            return null;
        }
    }
    
    async sendAnalysisToJarvis(analysis) {
        try {
            if (!this.websocket || this.websocket.readyState !== WebSocket.OPEN) {
                console.warn('⚠️ WebSocket not ready for analysis results');
                return;
            }
            
            const analysisMessage = {
                clientContent: {
                    turns: [{
                        role: "user",
                        parts: [{
                            text: `Visual analysis results: ${analysis}`
                        }]
                    }],
                    turnComplete: true
                }
            };
            
            console.log('📡 Sending visual analysis to JARVIS...');
            this.websocket.send(JSON.stringify(analysisMessage));
            
        } catch (error) {
            console.error('❌ Failed to send analysis to JARVIS:', error);
        }
    }
    
    streamAudioData(base64Data) {
        try {
            if (!this.audioStreamer) {
                console.warn('⚠️ AudioStreamer not available for JARVIS');
                return;
            }
            
            // Convert base64 to Uint8Array
            const binaryString = atob(base64Data);
            const bytes = new Uint8Array(binaryString.length);
            for (let i = 0; i < binaryString.length; i++) {
                bytes[i] = binaryString.charCodeAt(i);
            }
            
            // Check for silent audio
            const nonZeroBytes = bytes.filter(b => b !== 0).length;
            if (nonZeroBytes < bytes.length * 0.1) {
                return;
            }
            
            console.log('🎵 Streaming JARVIS audio response');
            
            // Update speaking status
            this.updateSpeakingStatus(true);
            
            // Stream to AudioStreamer
            this.audioStreamer.addPCM16(bytes);
            this.audioStreamer.resume();
            
        } catch (error) {
            console.error('❌ Failed to stream JARVIS audio:', error);
        }
    }
    
    updateSpeakingStatus(isSpeaking) {
        // Update UI to show JARVIS speaking status
        const status = document.getElementById('jarvis-status');
        if (status) {
            status.textContent = isSpeaking ? 'JARVIS is speaking...' : 'JARVIS is listening...';
        }
        
        // Update visual indicator
        const indicator = document.getElementById('jarvis-indicator');
        if (indicator) {
            if (isSpeaking) {
                indicator.classList.add('speaking');
            } else {
                indicator.classList.remove('speaking');
            }
        }
    }
    
    updateUI(state) {
        const indicator = document.getElementById('jarvis-indicator');
        const status = document.getElementById('jarvis-status');
        
        if (!indicator || !status) return;
        
        switch (state) {
            case 'active':
                indicator.className = 'jarvis-indicator active';
                status.textContent = 'JARVIS is listening...';
                break;
            case 'analyzing':
                indicator.className = 'jarvis-indicator analyzing';
                status.textContent = 'JARVIS is analyzing...';
                break;
            case 'inactive':
                indicator.className = 'jarvis-indicator';
                status.textContent = 'Say "Hey JARVIS" to activate';
                break;
        }
    }
    
    async deactivateJarvis() {
        try {
            console.log('🤖 Deactivating JARVIS...');
            
            this.isListening = false;
            this.isAnalyzing = false;
            
            // Stop audio recording
            this.isRecording = false;
            if (this.audioWorkletNode) {
                this.audioWorkletNode.disconnect();
                this.audioWorkletNode = null;
            }
            if (this.mediaStreamSource) {
                this.mediaStreamSource.disconnect();
                this.mediaStreamSource = null;
            }
            
            // Stop audio stream
            if (this.audioStream) {
                this.audioStream.getTracks().forEach(track => track.stop());
                this.audioStream = null;
            }
            
            // Close WebSocket
            if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
                this.websocket.close();
            }
            this.isConnected = false;
            
            // Restart wake word detection
            this.isWakeWordMode = true;
            setTimeout(() => {
                this.setupWakeWordDetection();
            }, 1000);
            
            // Update UI
            this.updateUI('inactive');
            
            console.log('✅ JARVIS deactivated - wake word detection restarted');
            
        } catch (error) {
            console.error('❌ Error deactivating JARVIS:', error);
        }
    }
    
    // Manual activation method (for J key press)
    async manualActivate() {
        console.log('⌨️ Manual JARVIS activation triggered');
        await this.activateJarvis();
    }
    
    // Manual deactivation method
    async manualDeactivate() {
        console.log('⌨️ Manual JARVIS deactivation triggered');
        await this.deactivateJarvis();
    }
    
    // Get current status
    getStatus() {
        return {
            isConnected: this.isConnected,
            isListening: this.isListening,
            isAnalyzing: this.isAnalyzing,
            isWakeWordMode: this.isWakeWordMode
        };
    }
}

// Global instance
window.jarvisAssistant = null;

// Initialize JARVIS when page loads
document.addEventListener('DOMContentLoaded', async () => {
    try {
        window.jarvisAssistant = new JarvisVoiceAssistant();
        await window.jarvisAssistant.initialize();
        console.log('🤖 JARVIS Voice Assistant ready for activation');
    } catch (error) {
        console.error('❌ Failed to initialize JARVIS:', error);
    }
});

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = JarvisVoiceAssistant;
}
