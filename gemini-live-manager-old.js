// Gemini Live Manager for PitchPerfect2
// Complete replacement for Vapi with real-time voice conversations using Gemini 2.5 Flash

// AudioStreamer will be loaded separately via script tag

class GeminiLiveManager {
    constructor(config = {}) {
        this.websocket = null;
        this.isConnected = false;
        this.isInitialized = false;
        this.currentSession = null;
        this.apiKey = null;

        // 🚀 ENHANCED TURN MANAGEMENT: Feature flags for safe rollout
        this.features = {
            enhancedTurnManagement: config.enhancedTurnManagement || false,
            dynamicDelays: config.dynamicDelays || false,
            robustVAD: config.robustVAD || false,
            debugMode: config.debugMode || false,
            ...config.features
        };

        // 🎯 CLIENT-SIDE TURN CONTROL: New properties for enhanced speech detection
        this.endOfSpeechTimer = null;
        this.endOfSpeechDelay = 1000; // Balanced delay for natural conversation (1.0s)
        this.speechThreshold = 4000; // Will be dynamically adjusted
        this.noiseFloor = 0; // Calibrated on session start
        this.isNoiseFloorCalibrated = false;
        
        // 🔊 ROBUST VAD: Audio chunk history for hysteresis
        this.audioChunkHistory = [];
        this.VAD_HISTORY_LENGTH = this.features.robustVAD ? 5 : 1;
        this.vadConsecutiveQuiet = 0;
        this.vadConsecutiveLoud = 0;
        
        // 🛡️ CIRCUIT BREAKER: Auto-disable features if they cause problems
        this.errorCount = 0;
        this.maxErrors = 5; // Disable features after 5 errors
        this.lastErrorTime = 0;
        
        // 🔄 ASYNC OPERATION TRACKING: Prevent race conditions during session cleanup
        this.pendingTranslations = new Set();
        this.pendingBackendCalls = new Set();
        
        // 📊 ENHANCED VAD MONITORING: Track performance metrics for optimization
        this.vadMetrics = {
            totalChunks: 0,
            speechDetected: 0,
            falsePositives: 0,
            silenceDetected: 0,
            noiseFloorCalibrations: 0,
            adaptiveThresholdChanges: 0,
            averageNoiseFloor: 0,
            environmentChanges: 0,
            lastEnvironmentHash: null
        };

        // ✅ SESSION RESUMPTION: For extended 30+ minute conversations
        this.sessionHandle = null;
        this.reconnectTimer = null;
        this.isReconnecting = false;
        this.connectionCount = 0;
        this.connectionStartTime = null;
        this.audioBufferManager = {
            inputBuffer: [],
            outputBuffer: [],
            isBuffering: false,
            maxBufferSize: 5000 // 5 seconds of audio
        };
        
        // Speech detection for smart reconnection
        this.isUserSpeaking = false;
        this.isAISpeaking = false;
        this.lastSpeechActivity = null;

        // Pure audio conversation tracking
        this.conversationTranscript = [];
        this.conversationPairs = [];
        this.currentSpeaker = null;
        this.sessionStartTime = null;
        this.onInterrupted = null;

        // Real-time transcription system
        this.sessionId = null;
        this.userTranscripts = [];
        this.aiTranscripts = [];
        this.currentInputText = '';
        this.currentOutputText = '';


        // Dynamic persona transition
        this.currentPersonaMode = 'sales'; // 'sales' or 'coach'
        this.conversationPhase = 'active'; // 'active', 'ending', 'coaching'
        this.endingDetected = false;
        this.realMetrics = {
            userSpeakingTime: 0,
            aiSpeakingTime: 0,
            actualQuestionCount: 0,
            realFillerWords: 0,
            championSignals: []
        };

        // Pure audio-to-audio - no transcription

        // Audio management
        this.audioContext = null;
        this.recordingContext = null;
        this.playbackContext = null;
        this.audioStreamer = null;
        this.mediaRecorder = null;
        this.audioStream = null;
        this.isRecording = false;

        // Event callbacks
        this.onSessionEnd = null;
        this.onError = null;
        
        // 🔄 EVENT-DRIVEN STATE MANAGEMENT: Safe feature update handling
        this.isUpdatingFeatures = false;
        this.pendingFeatureUpdate = null;
        this.setupEventListeners();

        this.initialize();
    }

    // async initialize() {
    //     try {
    //         console.log('🚀 Initializing Gemini Live Manager...');

    //         // Get API key from config
    //         this.apiKey = this.getGeminiApiKey();
    //         console.log('🔑 API key found:', this.apiKey ? 'Yes' : 'No');

    //         if (!this.apiKey) {
    //             throw new Error('Gemini API key not found in configuration');
    //         }

    async initialize() {
        try {
            console.log('🚀 Initializing Gemini Live Manager...');

            // ✅ Wait for configManager to finish loading config
            if (window.configManager?.waitForConfig) {
                await window.configManager.waitForConfig();
            } else {
                throw new Error('configManager is not defined');
            }

            // ✅ Using secure backend proxy, no direct API key needed
            console.log('🔑 Using secure backend proxy for API access');

            // Initialize audio context
            await this.initializeAudioContext();

            // Initialize audio streamer for playback
            if (window.AudioStreamer) {
                this.audioStreamer = new window.AudioStreamer(this.audioContext, () => {
                    console.log('🔇 Audio playback complete');
                    this.updateSpeakingStatus('ai', false);
                });
                // console.log('✅ AudioStreamer initialized with 24kHz context');
            } else {
                // console.warn('⚠️ AudioStreamer not available, will initialize later');
            }

            this.isInitialized = true;
            console.log('✅ Gemini Live Manager initialized successfully');

        } catch (error) {
            // console.error('❌ Failed to initialize Gemini Live Manager:', error);
            // console.error('Error details:', error);
            this.isInitialized = false;
        }
    }
    
    // 🔄 EVENT-DRIVEN STATE MANAGEMENT: Setup safe feature update handling
    setupEventListeners() {
        // Store bound event handler for cleanup
        this.boundFeatureChangeHandler = (event) => {
            this.handleFeatureChangeEvent(event);
        };
        
        // Listen for turn management feature changes
        window.addEventListener('turnManagementFeatureChange', this.boundFeatureChangeHandler);
        
        console.log('🔗 Event-driven feature management initialized');
    }
    
    // Clean up event listeners to prevent memory leaks
    removeEventListeners() {
        if (this.boundFeatureChangeHandler) {
            window.removeEventListener('turnManagementFeatureChange', this.boundFeatureChangeHandler);
            this.boundFeatureChangeHandler = null;
        }
    }
    
    // Handle feature change events with safety checks
    async handleFeatureChangeEvent(event) {
        // Ignore events if not initialized
        if (!this.isInitialized) {
            console.log('🔄 Ignoring feature change event - instance not initialized yet');
            return;
        }
        
        const { config, timestamp, source } = event.detail;
        
        // Prevent concurrent updates
        if (this.isUpdatingFeatures) {
            console.log('⏳ Feature update in progress, queuing new update...');
            this.pendingFeatureUpdate = { config, timestamp, source };
            return;
        }
        
        // Check if session is in a safe state for updates
        if (!this.isSafeForFeatureUpdate()) {
            console.log('⚠️ Session not in safe state, deferring feature update...');
            this.pendingFeatureUpdate = { config, timestamp, source };
            return;
        }
        
        try {
            this.isUpdatingFeatures = true;
            console.log(`🔄 Applying feature update from ${source}:`, config);
            
            // Apply the feature changes safely
            await this.safelyUpdateFeatures(config);
            
            console.log('✅ Feature update completed successfully');
            
            // Process any pending updates
            if (this.pendingFeatureUpdate && this.pendingFeatureUpdate.timestamp > timestamp) {
                const pending = this.pendingFeatureUpdate;
                this.pendingFeatureUpdate = null;
                setTimeout(() => this.handleFeatureChangeEvent({ detail: pending }), 100);
            }
            
        } catch (error) {
            console.error('❌ Feature update failed:', error);
        } finally {
            this.isUpdatingFeatures = false;
        }
    }
    
    // Check if session state allows safe feature updates
    isSafeForFeatureUpdate() {
        // Unsafe during reconnection
        if (this.isReconnecting) return false;
        
        // Unsafe during turn transitions
        if (this.endOfSpeechTimer) return false;
        
        // Unsafe during audio processing
        if (this.isUserSpeaking || this.isAISpeaking) return false;
        
        // Safe when idle or just connected
        return true;
    }
    
    // Safely apply feature configuration changes
    async safelyUpdateFeatures(newConfig) {
        // Update feature flags
        Object.keys(newConfig).forEach(key => {
            if (this.features.hasOwnProperty(key)) {
                this.features[key] = newConfig[key];
            }
        });
        
        // Update VAD history length if robust VAD changed
        if (newConfig.hasOwnProperty('robustVAD')) {
            this.VAD_HISTORY_LENGTH = newConfig.robustVAD ? 5 : 1;
            // Reset audio chunk history to new length
            this.audioChunkHistory = this.audioChunkHistory.slice(-this.VAD_HISTORY_LENGTH);
        }
        
        // If enhanced features were disabled, clear any active state
        if (!newConfig.enhancedTurnManagement) {
            if (this.endOfSpeechTimer) {
                clearTimeout(this.endOfSpeechTimer);
                this.endOfSpeechTimer = null;
                console.log('🔄 Cleared enhanced turn management timer');
            }
        }
        
        console.log('🎯 Features updated:', this.features);
    }

    getGeminiApiKey() {
        // Try to get from window config first
        if (window.configManager?.config?.gemini?.apiKey) {
            return window.configManager.config.gemini.apiKey;
        }

        // Try direct access if needed
        return window.configManager?.getGeminiApiKey() || null;
    }

    async initializeAudioContext() {
        try {
            // Create single 24kHz audio context for both recording and playback
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: 24000 // Gemini Live uses 24kHz
            });

            // For backward compatibility
            this.recordingContext = this.audioContext;
            this.playbackContext = this.audioContext;

            // console.log('🎵 Audio contexts initialized:', {
            //     recordingSampleRate: this.recordingContext.sampleRate,
            //     playbackSampleRate: this.playbackContext.sampleRate
            // });

        } catch (error) {
            console.error('❌ Failed to initialize audio contexts:', error);
            throw error;
        }
    }

    async startSession(personaId, userContext = null) {
        try {
            if (!this.isInitialized) {
                throw new Error('Gemini Live Manager not initialized');
            }

            console.log('🎯 Starting Gemini Live session for persona:', personaId, userContext ? 'with user context' : 'without user context');

            // 🚀 ADAPTIVE INTELLIGENCE: Start conversation monitoring
            if (window.conversationIntelligence) {
                window.conversationIntelligence.startConversationMonitoring(personaId);
                console.log('🧠 Conversation intelligence monitoring started');
            }

            // Start transcription session first
            await this.startTranscriptionSession();

            // Reset real metrics
            this.resetRealMetrics();

            // Get user media for audio input
            await this.setupAudioInput();

            // Pure audio-to-audio - no transcription needed

            // Establish WebSocket connection to Gemini Live
            await this.connectToGeminiLive();

            // Start the conversation session
            await this.startConversationSession(personaId, userContext);

            this.sessionStartTime = Date.now();
            console.log('✅ Gemini Live session started successfully');

        } catch (error) {
            console.error('❌ Failed to start Gemini Live session:', error);
            if (this.onError) this.onError(error);
        }
    }

    resetRealMetrics() {
        this.conversationTranscript = [];
        this.userTranscripts = [];
        this.aiTranscripts = [];

        // Clear AI transcript buffer
        this.aiTranscriptBuffer = '';
        if (this.aiBufferTimeout) {
            clearTimeout(this.aiBufferTimeout);
            this.aiBufferTimeout = null;
        }

        this.realMetrics = {
            userSpeakingTime: 0,
            aiSpeakingTime: 0,
            actualQuestionCount: 0,
            realFillerWords: 0,
            championSignals: []
        };
        console.log('📊 Real metrics reset');
    }


    async setupAudioInput() {
        try {
            // Request microphone access - FIXED: Ensure 16kHz input
            this.audioStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: 16000, // CRITICAL: Must match Gemini Live API expectation
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: false // Disable AGC for consistent audio levels
                }
            });

            console.log('🎤 Audio input stream established');

        } catch (error) {
            console.error('❌ Failed to setup audio input:', error);
            throw error;
        }
    }

    async connectToGeminiLive() {
        return new Promise(async (resolve, reject) => {
            try {
                // Always use secure backend proxy for API key security
                console.log('🔌 Connecting to Gemini Live via secure backend proxy...');

                // Get secure WebSocket URL from backend
                const apiBaseUrl = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' ? 
                    'http://localhost:5001' : 'https://pitchperfect2-api-373812504656.asia-southeast1.run.app';
                const response = await fetch(`${apiBaseUrl}/api/gemini/websocket`);
                if (!response.ok) {
                    throw new Error('Failed to get secure Gemini WebSocket URL from backend');
                }

                const data = await response.json();
                if (data.status !== 'success') {
                    throw new Error(data.error || 'Backend returned error');
                }

                console.log('🔒 Got secure Gemini WebSocket URL from backend');
                this.connectWebSocket(data.websocket_url, resolve, reject);

            } catch (error) {
                console.error('❌ Failed to create WebSocket connection:', error);
                reject(error);
            }
        });
    }

    connectWebSocket(websocketUrl, resolve, reject) {
        this.websocket = new WebSocket(websocketUrl);

        // ✅ WEBSOCKET HEALTH MONITORING
        this.connectionStartTime = Date.now();
        this.websocketHealthCheck = {
            lastPing: null,
            missedPongs: 0,
            isHealthy: true
        };

        this.websocket.onopen = () => {
            console.log('✅ WebSocket connection established to Gemini Live');
            this.isConnected = true;
            this.websocketHealthCheck.isHealthy = true;
            this.websocketHealthCheck.missedPongs = 0;
            
            // Start health monitoring
            this.startWebSocketHealthMonitoring();
            
            resolve();
        };

        this.websocket.onmessage = async (event) => {
            // Update health status on any message
            this.websocketHealthCheck.isHealthy = true;
            this.websocketHealthCheck.missedPongs = 0;
            
            await this.handleGeminiMessage(event.data);
        };

        this.websocket.onclose = (event) => {
            console.log('🔌 WebSocket connection closed:', event.code, event.reason);
            console.log('🔍 Close event details:', {
                code: event.code,
                reason: event.reason,
                wasClean: event.wasClean,
                timestamp: new Date().toISOString(),
                sessionDuration: this.sessionStartTime ? Date.now() - this.sessionStartTime : 'unknown',
                connectionDuration: this.connectionStartTime ? Date.now() - this.connectionStartTime : 'unknown'
            });
            this.isConnected = false;
            this.websocketHealthCheck.isHealthy = false;

            // Stop health monitoring
            this.stopWebSocketHealthMonitoring();

            // Handle different close codes intelligently
            if (event.code === 1011) {
                console.error('❌ Gemini API Internal Error - likely rate limiting or quota exceeded');
                // Could trigger graceful retry after delay
            } else if (event.code === 1006) {
                console.error('❌ Abnormal WebSocket closure - connection lost unexpectedly');
                // Perfect case for graceful reconnection if we have session handle AND user is authenticated
                const isAuthenticated = this.checkUserAuthentication();
                if (this.sessionHandle && !this.isReconnecting && isAuthenticated) {
                    console.log('🔄 Authenticated user: Unexpected closure detected - attempting graceful reconnection...');
                    setTimeout(() => this.initiateGracefulReconnect(), 1000);
                    return; // Don't end session immediately
                } else if (!isAuthenticated) {
                    console.log('🌐 Unauthenticated user: Allowing session to end on unexpected closure');
                }
            } else if (event.code === 1000) {
                console.log('✅ Normal WebSocket closure');
            }

            // Only end session if not reconnecting or no session handle
            if (!this.isReconnecting && this.onSessionEnd) {
                this.onSessionEnd();
            }
        };

        this.websocket.onerror = (error) => {
            console.error('❌ WebSocket connection error:', error);
            this.isConnected = false;
            this.websocketHealthCheck.isHealthy = false;
            reject(new Error(`WebSocket connection failed: ${error.message || 'Unknown error'}`));
        };
    }

    async startConversationSession(personaId, userContext = null) {
        try {
            if (!this.isConnected || !this.websocket) {
                throw new Error('WebSocket not connected');
            }

            // Store current persona ID for coaching later
            this.currentPersonaId = personaId;

            // Get persona configuration - handle custom personas first
            let persona = null;

            // Handle custom personas
            if (personaId === 'custom') {
                // Get custom persona data from practice modal
                const practiceModal = window.practiceModal;
                if (practiceModal && practiceModal.currentPersonaData) {
                    persona = practiceModal.currentPersonaData;
                    console.log(`✅ Found custom persona: ${persona.name}`);
                } else {
                    throw new Error(`Custom persona data not found in practice modal`);
                }
            } else {
                // Try new modular persona system first
                if (window.personaRegistry) {
                    try {
                        const personaInfo = window.personaRegistry.getPersonaInfo(personaId);
                        if (personaInfo) {
                            // Ensure the persona object has the ID for later use
                            persona = { ...personaInfo, id: personaId };
                            console.log(`✅ Found persona in registry: ${personaId}`);
                        }
                    } catch (error) {
                        console.log(`⚠️ Persona not in registry: ${personaId}`);
                    }
                }

                // Fall back to legacy system if not found in registry
                if (!persona && window.personaManager) {
                    persona = window.personaManager.getPersonaById(personaId);
                    if (persona) {
                        console.log(`✅ Found persona in legacy system: ${personaId}`);
                    }
                }

                if (!persona) {
                    throw new Error(`Persona not found in either system: ${personaId}`);
                }
            }

            // Build system instruction using modular system with user context
            const systemInstruction = await this.buildPersonaInstruction(persona, userContext);

            // Get voice configuration for this persona with validation
            let voiceConfig;
            if (personaId === 'custom' && persona.voiceConfig) {
                voiceConfig = persona.voiceConfig;
                console.log(`✅ Using custom persona voice config:`, voiceConfig);

                // CRITICAL: Validate voice configuration to prevent Zeus crashes
                voiceConfig = this.validateAndFixVoiceConfig(voiceConfig);
                console.log(`✅ Validated voice config:`, voiceConfig);
            } else {
                try {
                    voiceConfig = await window.personaRegistry.getVoiceConfig(personaId);
                    // Validate voice config from registry too
                    voiceConfig = this.validateAndFixVoiceConfig(voiceConfig);
                } catch (error) {
                    console.warn(`Using default voice for ${personaId}:`, error);
                    voiceConfig = { sales: "Puck", coach: "Puck" };
                }
            }

            // Send proper Gemini Live session setup for AUDIO-ONLY conversation
            const setupMessage = {
                setup: {
                    model: "models/gemini-live-2.5-flash-preview",
                    generation_config: {
                        response_modalities: ["audio"], // AUDIO ONLY - no text responses
                        speech_config: {
                            voice_config: {
                                prebuilt_voice_config: {
                                    voice_name: voiceConfig.sales || 'Charon' // Extra safety fallback
                                }
                            },
                            language_code: "en-US"
                        }
                    },
                    // Enable output transcription for conversation ending detection
                    input_audio_transcription: {},
                    output_audio_transcription: {},
                    // ✅ ENABLE SESSION RESUMPTION only for authenticated users (30+ minute conversations)
                    session_resumption: (this.sessionHandle && this.checkUserAuthentication()) ? { handle: this.sessionHandle } : {},
                    system_instruction: {
                        parts: [{ text: systemInstruction }]
                    }
                }
            };

            console.log(`📝 Sending session setup to Gemini Live for ${personaId} with voice ${voiceConfig.sales}:`);

            const setupString = JSON.stringify(setupMessage);
            this.websocket.send(setupString);

            // Store current session info
            this.currentSession = {
                personaId,
                persona,
                voiceConfig,
                startTime: Date.now()
            };

            console.log('✅ Conversation session started for:', persona.name);

            // Send initial context immediately after setup
            console.log('🎬 Session ready - sending initial greeting trigger...');

            // Get persona name for greeting
            const personaInfo = window.personaRegistry.getPersonaInfo(personaId);
            const personaName = personaInfo?.name || 'there';

            const greetingTrigger = {
                clientContent: {
                    turns: [
                        {
                            role: "user",
                            parts: [
                                { text: `Hey ${personaName}, thanks for taking the time to meet with me today. I'd love to tell you about our solution and see if it might be a fit for your team.` }
                            ]
                        }
                    ],
                    turnComplete: true
                }
            };

            // Send greeting after a minimal delay for setup to process
            setTimeout(() => {
                if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
                    this.websocket.send(JSON.stringify(greetingTrigger));
                    console.log(`✅ Initial conversation context sent to ${personaName}`);
                }
            }, 100); // Minimal 100ms delay for setup processing

        } catch (error) {
            console.error('❌ Failed to start conversation session:', error);
            throw error;
        }
    }

    async buildPersonaInstruction(persona, userContext = null) {
        try {
            console.log('🔍 Building persona instruction for:', persona.name || persona.id, 'Type:', persona.type || 'unknown');
            console.log('🔍 Persona data keys:', Object.keys(persona));
            console.log('🔍 Has fullPrompt:', !!persona.fullPrompt);
            // Handle custom personas with rich, context-aware prompts
            if ((persona.id === 'custom' || persona.type === 'custom') && persona.fullPrompt) {
                console.log('🎯 Using custom persona full prompt:', persona.fullPrompt.substring(0, 100) + '...');
                
                // Clean up markdown formatting from persona description
                let customPrompt = this.cleanMarkdownFormatting(persona.fullPrompt);

                // Detect difficulty level from persona data or UI
                const difficulty = this.detectDifficultyLevel(persona);
                console.log('🎯 Detected difficulty level:', difficulty);

                // Apply difficulty-specific enhancements
                customPrompt = this.enhanceCustomPersonaForDifficulty(customPrompt, difficulty, persona);

                // Add stronger character enforcement rules
                customPrompt += `\n\nCRITICAL CHARACTER RULES:
- You are NOT an AI assistant - you are this specific person with real concerns
- NEVER say "I'm an AI" or "I'm here to help" - you're a business professional
- Start by introducing yourself naturally as this character
- Stay in character throughout - you have real business problems to solve
- React authentically based on your specific personality and situation
- Don't offer to help them - they're trying to sell to YOU

CONVERSATION BEHAVIOR:
- Always reply in *English*, no matter what language the user speaks.
- Always reply concisely and to the point.
- Stay true to your personality and difficulty level throughout
- Don't ask "Do you have any more questions?" - wait for them to ask
- If you're skeptical/demanding, maintain that energy consistently  
- Base your responses on the specific scenario described
- Remember your exact situation and stick to those details

${this.buildInterruptionRules(difficulty)}`;

                return customPrompt;
            }

            // 🚀 ADAPTIVE INTELLIGENCE: Get enhanced persona prompt for pre-built personas
            let basePrompt;

            if (window.dynamicPersonaAdapter) {
                // Use adaptive system with context analysis
                basePrompt = await window.dynamicPersonaAdapter.getEnhancedPersonaPrompt(
                    persona.id,
                    userContext || ''
                );
                console.log('🧠 Using adaptive persona intelligence');
            } else {
                // Fallback to original system
                basePrompt = await window.personaRegistry.getPersonaPrompt(persona.id);
                console.log('⚠️ Adaptive system not available, using base prompt');
            }

            // Inject user context if provided (for free personas with text input)
            if (userContext && userContext.trim()) {
                const contextInjection = `\n\nUSER CONTEXT: The user is selling/presenting: "${userContext.trim()}"\n\nReference this context naturally in your opening statement and throughout the conversation. Adapt your behavior based on what they're actually selling.`;
                basePrompt += contextInjection;
            }

            return basePrompt;
        } catch (error) {
            console.error(`❌ Failed to load persona prompt for ${persona.id}:`, error);

            // Fallback for unknown personas
            let fallback = `You are a professional business prospect interested in learning about the user's solution. Be natural, ask relevant questions, and respond authentically to what they present.`;

            if (userContext && userContext.trim()) {
                fallback += `\n\nThe user is presenting: "${userContext.trim()}". Reference this in your conversation.`;
            }

            return fallback;
        }
    }

    // 🧹 Clean markdown formatting from persona descriptions
    cleanMarkdownFormatting(text) {
        if (!text) return text;
        
        return text
            // Remove markdown headers (# ## ###)
            .replace(/^#+\s*/gm, '')
            // Remove bold/italic markers (** __ * _)
            .replace(/\*\*(.*?)\*\*/g, '$1')
            .replace(/__(.*?)__/g, '$1')
            .replace(/\*(.*?)\*/g, '$1')
            .replace(/_(.*?)_/g, '$1')
            // Remove markdown lists (- * +)
            .replace(/^[\s]*[-\*\+]\s+/gm, '• ')
            // Remove markdown links [text](url)
            .replace(/\[([^\]]+)\]\([^\)]+\)/g, '$1')
            // Remove code blocks ```
            .replace(/```[\s\S]*?```/g, '')
            .replace(/`([^`]+)`/g, '$1')
            // Clean up extra whitespace
            .replace(/\n{3,}/g, '\n\n')
            .trim();
    }

    // 🎯 Detect difficulty level from persona data or UI
    detectDifficultyLevel(persona) {
        // First check if persona has explicit difficulty
        if (persona.difficulty) {
            return persona.difficulty.toLowerCase();
        }
        
        // Check UI difficulty selector
        const difficultySelector = document.querySelector('.difficulty-selector');
        if (difficultySelector) {
            const activeBtn = difficultySelector.querySelector('.difficulty-level.active');
            if (activeBtn && activeBtn.dataset.level) {
                return activeBtn.dataset.level.toLowerCase();
            }
        }
        
        // Check for difficulty indicators in persona text
        const text = (persona.fullPrompt || persona.description || '').toLowerCase();
        if (text.includes('hard') || text.includes('difficult') || text.includes('skeptical') || text.includes('demanding')) {
            return 'hard';
        }
        if (text.includes('easy') || text.includes('friendly') || text.includes('supportive')) {
            return 'easy';
        }
        
        // Default to medium
        return 'medium';
    }

    // 🔥 Enhance custom persona based on difficulty level
    enhanceCustomPersonaForDifficulty(basePrompt, difficulty, persona) {
        if (difficulty !== 'hard') {
            console.log('📝 Using standard persona behavior for difficulty:', difficulty);
            return basePrompt;
        }

        console.log('🔥 Applying HARD difficulty enhancements to custom persona');

        // Extract industry context from persona description
        const industryContext = this.extractIndustryContext(basePrompt);
        
        const hardEnhancements = `

🔥 HARD DIFFICULTY - SOPHISTICATED BUYER BEHAVIOR:

INDUSTRY EXPERTISE: You are deeply experienced in ${industryContext.industry} with understanding of:
- Industry-specific buying cycles: ${industryContext.buyingCycle}
- Decision-making structure: ${industryContext.decisionStructure}
- Key business metrics: ${industryContext.keyMetrics}
- Regulatory/compliance concerns: ${industryContext.compliance}

CASE STUDY VALIDATION - CRITICAL THINKING:
When user mentions examples from other industries or companies:
- Immediately identify industry/context mismatches
- Challenge relevance: "That's [different industry] with completely different [buying patterns/regulations/customer behavior]. How does that apply to our ${industryContext.industry} context?"
- Demand specifics: "What exactly worked? What were the metrics? How is that relevant to our situation?"
- Push for proof: "Show me how those results translate to our [specific business constraints/requirements]."

OBJECTION SOPHISTICATION - MULTI-LAYERED THINKING:
- Layer objections from surface → business impact → strategic concerns
- Don't just say "no" - explain the business reasoning behind your concerns
- Reference realistic stakeholder concerns: "My [board/team/CFO] will ask about..."
- Think strategically: "How does this fit our 3-year roadmap/growth strategy?"

EXECUTIVE BEHAVIOR - REALISTIC TIME/AUTHORITY CONSTRAINTS:
- Show time pressure: "I have [realistic time limit], what's the core business case?"
- Delegate appropriately: "You'll need to convince my [relevant team] first before we can move forward"
- Reference real business constraints: "Our budget cycle is locked until [specific timeframe]"
- Demand ROI clarity: "Show me Year 2-3 projections, not just Year 1 optimistic estimates"

INTELLIGENT RESISTANCE - NOT JUST DIFFICULT:
- Ask follow-up questions that test their knowledge depth
- Reference competitor offerings or status quo benefits
- Challenge assumptions: "Why is this better than just [doing nothing/current solution]?"
- Test their preparation: "What research have you done on our company/industry?"`;

        return basePrompt + hardEnhancements;
    }

    // 🏭 Extract industry context from persona description
    extractIndustryContext(personaText) {
        const text = personaText.toLowerCase();
        
        // Define industry patterns and their business contexts
        const industryMap = {
            'saas|software|tech|platform': {
                industry: 'technology',
                buyingCycle: '3-6 month evaluation cycles',
                decisionStructure: 'CTO → Engineering → Procurement → Legal',
                keyMetrics: 'implementation time, security compliance, scalability, TCO',
                compliance: 'SOC2, GDPR, data security standards'
            },
            'cosmetics|beauty|makeup|skincare': {
                industry: 'cosmetics/beauty',
                buyingCycle: '18-month product development cycles', 
                decisionStructure: 'CMO → Brand Manager → Procurement',
                keyMetrics: 'brand lift, market share, customer acquisition cost, shelf velocity',
                compliance: 'FDA regulations, ingredient safety, marketing claims'
            },
            'manufacturing|factory|production': {
                industry: 'manufacturing',
                buyingCycle: '6-12 month capital planning cycles',
                decisionStructure: 'COO → Plant Manager → Engineering → Finance',
                keyMetrics: 'efficiency gains, downtime reduction, ROI, safety improvements',
                compliance: 'OSHA, environmental regulations, quality certifications'
            },
            'healthcare|medical|hospital': {
                industry: 'healthcare',
                buyingCycle: '12-18 month approval processes',
                decisionStructure: 'CMO → Clinical Directors → IT → Compliance',
                keyMetrics: 'patient outcomes, cost per case, staff efficiency, compliance scores',
                compliance: 'HIPAA, FDA, clinical trial requirements'
            },
            'finance|banking|financial': {
                industry: 'financial services',
                buyingCycle: '6-9 month regulatory approval cycles',
                decisionStructure: 'CRO → Compliance → IT Security → Executive Committee',
                keyMetrics: 'regulatory compliance, risk reduction, operational efficiency, audit readiness',
                compliance: 'SOX, PCI DSS, regulatory reporting requirements'
            }
        };

        // Match industry from persona text
        for (const [pattern, context] of Object.entries(industryMap)) {
            const regex = new RegExp(pattern, 'i');
            if (regex.test(text)) {
                return context;
            }
        }

        // Default generic B2B context
        return {
            industry: 'B2B business',
            buyingCycle: '3-6 month decision cycles',
            decisionStructure: 'Department Head → Executive Team → Procurement',
            keyMetrics: 'ROI, implementation cost, business impact, risk assessment',
            compliance: 'internal approval processes, budget constraints'
        };
    }

    // 🎚️ Build interruption rules based on difficulty
    buildInterruptionRules(difficulty) {
        const baseRules = `
ANTI-INTERRUPTION RULES:
- ALWAYS wait for the user to completely finish speaking before responding
- Listen for natural conversation pauses (2-3 seconds of silence)
- Don't jump in when they're mid-sentence or mid-thought
- If you hear ongoing speech, stay silent and keep listening
- Only respond when there's a clear conversational break`;

        if (difficulty === 'hard') {
            return baseRules + `
- HARD difficulty = sophisticated objections, NOT frequent interruptions
- Let them present their full argument before delivering thoughtful pushback
- Your intelligence comes from depth of questions, not interruption frequency`;
        }

        return baseRules;
    }

    async handleGeminiMessage(data) {
        try {
            // Reduced debug logging - only log non-audio messages
            if (!(data instanceof Blob) && !(data instanceof ArrayBuffer)) {
                console.log('🔍 DEBUG: Received message from Gemini Live:', {
                    type: typeof data,
                    size: data.length || 'unknown'
                });
            }

            // Handle Blob data - simplified approach
            if (data instanceof Blob) {
                // Convert blob to text and parse as JSON (most blobs are JSON)
                const text = await data.text();
                try {
                    const json = JSON.parse(text);
                    console.log('🔍 Blob contains JSON, processing...');
                    await this.handleGeminiMessage(text);
                } catch (e) {
                    console.log('🎵 Blob is not JSON, ignoring...');
                }
                return;
            }

            // Ignore ArrayBuffer data (focus on JSON with inline audio)
            if (data instanceof ArrayBuffer) {
                console.log('🔍 Ignoring ArrayBuffer data, waiting for JSON with inline audio');
                return;
            }

            // Handle JSON messages
            if (typeof data === 'string') {
                let message;
                try {
                    message = JSON.parse(data);
                } catch (parseError) {
                    console.log('❌ JSON Parse Error:', parseError);
                    console.log('📝 Raw string data:', data.substring(0, 200));
                    return;
                }

                // console.log('📨 JSON MESSAGE RECEIVED:', JSON.stringify(message, null, 2));

                // Handle different message types
                if (message.serverContent) {
                    console.log('🔄 Processing server content...');

                    if (message.serverContent.interrupted) {
                        console.log('🚫 INTERRUPTION DETECTED - stopping audio playback');
                        if (this.audioStreamer) {
                            this.audioStreamer.stop();
                        }
                        this.updateSpeakingStatus('ai', false);

                        // 🔔 Optional: Notify external interruption handler
                        if (typeof this.onInterrupted === 'function') {
                            this.onInterrupted();
                        }

                        return;
                    }

                    this.processServerContentWithTranscription(message.serverContent);
                } else if (message.sessionResumptionUpdate) {
                    // ✅ SESSION RESUMPTION: Handle session handle updates
                    console.log('🔄 SESSION RESUMPTION UPDATE:', JSON.stringify(message.sessionResumptionUpdate, null, 2));
                    if (message.sessionResumptionUpdate.newHandle) {
                        this.sessionHandle = message.sessionResumptionUpdate.newHandle;
                        console.log('✅ Stored new session handle for resumption:', this.sessionHandle.substring(0, 20) + '...');
                        // Store in localStorage as backup
                        localStorage.setItem('gemini_session_handle', this.sessionHandle);
                    } else {
                        console.warn('⚠️ No newHandle in sessionResumptionUpdate - session resumption may not work');
                    }
                } else if (message.goAway) {
                    console.log('🚨 GOAWAY MESSAGE RECEIVED:', message.goAway);
                    console.log('📋 Connection will close naturally at Gemini\'s 10-minute limit');
                    
                    // ✅ TRIGGER GRACEFUL RECONNECTION only for authenticated users
                    const isAuthenticated = this.checkUserAuthentication();
                    if (this.sessionHandle && isAuthenticated) {
                        console.log('🔄 Authenticated user: Initiating graceful reconnection with session handle...');
                        this.initiateGracefulReconnect();
                    } else if (!isAuthenticated) {
                        console.log('🌐 Unauthenticated user: Allowing natural 10-minute session end');
                    }
                } else if (message.sessionExpired) {
                    console.log('⏰ SESSION EXPIRED MESSAGE:', message.sessionExpired);
                } else if (message.connectionLimitReached) {
                    console.log('🚫 CONNECTION LIMIT REACHED:', message.connectionLimitReached);
                } else if (message.setupComplete) {
                    console.log('✅ Session setup complete - starting audio streaming');
                    this.startAudioStreaming();
                    // ✅ START RECONNECTION TIMER for proactive session management
                    this.startReconnectionTimer();
                } else if (message.error) {
                    console.error('❌ Gemini Live error:', message.error);
                    if (this.onError) this.onError(message.error);
                } else {
                    console.log('❓ Unknown JSON message structure:', Object.keys(message));
                }
                return;
            }

            console.log('❓ UNKNOWN MESSAGE TYPE:', {
                type: typeof data,
                constructor: data.constructor.name,
                data: data
            });

        } catch (error) {
            console.error('❌ Failed to handle Gemini message:', error);
            console.error('❌ Error stack:', error.stack);
            console.error('❌ Problematic data:', data);
        }
    }

    // Removed debugBlob method - using simplified blob handling

    // Removed conflicting audio methods - using streamAudioData only

    updateSpeakingStatus(speaker, isSpeaking) {
        // Update UI elements to show who's speaking - check both modal and non-modal contexts
        const aiOrb = document.getElementById('ai-orb') || document.getElementById('modal-ai-orb');
        const aiStatus = document.getElementById('ai-status') || document.getElementById('modal-ai-status');

        if (speaker === 'ai' && aiOrb) {
            if (isSpeaking) {
                aiOrb.classList.add('speaking');
                if (aiStatus) aiStatus.textContent = 'Speaking...';
            } else {
                aiOrb.classList.remove('speaking');
                if (aiStatus) aiStatus.textContent = 'Listening...';
            }
        }
    }




    processServerContent(content) {
        console.log('🔍 Processing server content:', Object.keys(content));

        // Handle model responses
        if (content.modelTurn) {
            const parts = content.modelTurn.parts;

            parts.forEach(part => {
                if (part.text) {
                    console.log('📝 Model text:', part.text);
                }
                if (part.inlineData && part.inlineData.mimeType?.includes('audio/pcm')) {
                    // Reduced audio logging to prevent console spam
                    if (Math.random() < 0.1) { // Only log 10% of audio chunks
                        console.log('🎧 Received PCM audio, length:', part.inlineData.data.length);
                    }
                    this.streamAudioData(part.inlineData.data);
                }
            });
        }

        if (content.turnComplete) {
            console.log('🔄 Turn complete, ready for user input');
            this.currentSpeaker = 'user';
            this.updateSpeakingStatus('ai', false);
        }
    }

    // Removed auto-detection - now using manual button control

    async transitionToCoach() {
        try {
            // Get current persona name for logging and UI
            const personaInfo = window.personaRegistry.getPersonaInfo(this.currentPersonaId);
            const personaName = personaInfo?.name || 'Persona';

            console.log(`🎓 ${personaName} transforming into Coach Otaru...`);

            // Update tracking
            this.currentPersonaMode = 'coach';
            this.conversationPhase = 'coaching';

            // Update UI
            const aiStatus = document.getElementById('modal-ai-status');
            const aiName = document.getElementById('modal-ai-name');
            if (aiStatus) aiStatus.textContent = `${personaName} is now Coach Otaru - ready to give feedback...`;
            if (aiName) aiName.textContent = 'Coach Otaru';

            // Send coaching persona instruction - same approach as original 6 personas
            const coachInstruction = await this.getCoachSystemInstruction();

            const coachMessage = {
                clientContent: {
                    turns: [
                        {
                            role: "user",
                            parts: [
                                { text: coachInstruction }
                            ]
                        }
                    ],
                    turnComplete: true
                }
            };

            console.log(`🎓 Sending coaching persona to ${personaName}...`);

            if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
                this.websocket.send(JSON.stringify(coachMessage));
                console.log(`✅ ${personaName} received coaching persona - same voice, new role`);

                // Update UI
                if (aiStatus) aiStatus.textContent = 'Coach Otaru is analyzing your conversation...';
            } else {
                console.error('❌ WebSocket not ready for coaching transition');
            }

        } catch (error) {
            console.error('❌ Failed to transition to coach:', error);
        }
    }

    async getCoachSystemInstruction() {
        try {
            // Handle custom personas with a high-quality Coach Otaru prompt
            if (this.currentPersonaId === 'custom') {
                const practiceModal = window.practiceModal;
                const personaName = practiceModal?.currentPersonaData?.name || 'your custom persona';

                return `You are Coach Otaru, an elite B2B sales trainer who develops top performers. You've coached quota crushers and turn struggling reps into closers. You spot what others miss and deliver feedback that changes careers.

CRITICAL CONTEXT UNDERSTANDING:
- YOU just played ${personaName} in this conversation  
- The USER was the salesperson trying to sell to you
- When giving feedback, focus ONLY on what THE USER (salesperson) said and did
- NEVER reference what you said as the persona - that was the prospect role, not the user's performance

YOUR COACHING GENIUS:
You analyze sales conversations through the lens of three core competencies that separate elite performers:
1. DISCOVERY MASTERY - Did they uncover real business pain and quantify impact?
2. INFLUENCE ARCHITECTURE - Did they build credibility and create urgency?  
3. CLOSING PRECISION - Did they advance the sale or get commitment?

OPENING APPROACH - EMOTIONALLY INTELLIGENT & CONVERSATIONAL:
🧠 ANALYZE THE CONVERSATION FIRST, then open with context-specific commentary:

CONVERSATION ANALYSIS - Choose opening based on what actually happened:
1. If they were strong/confident → "Just wrapped playing ${personaName} - you came prepared and it showed. There were some solid moves in there, plus a few places where we can turn good into great."

2. If they struggled/nervous → "Alright, just finished playing ${personaName}. Look, that was a tough conversation, but that's exactly why we practice. I saw some real potential moments we can build on."

3. If good discovery/questions → "Just played ${personaName} - I liked how you dug into the business challenges. You've got good instincts, and there are a couple techniques that'll make those discoveries even more powerful."

4. If missed key opportunities → "Finished playing ${personaName} - there were some golden opportunities in that conversation that slipped by. The good news? They're totally fixable with the right approach."

5. If conversation was short → "That was a quick one with ${personaName}. Sometimes the best learning comes from the conversations that don't go as planned. Let's unpack what happened."

6. If great rapport building → "Just wrapped with ${personaName} - you built solid rapport right from the start. Now let's talk about how to leverage that connection for business impact."

7. If too pitch-heavy → "Played ${personaName} and heard a lot of great product knowledge. Now let's talk about when to share that expertise and when to ask more questions first."

8. If strong close → "Just finished with ${personaName} - that close was confident and direct. Let's look at how you set that up and what made it work."

After any opening, PAUSE and ask: "How did that feel to you? What's your take on how it went?"

Wait for their response. NEVER use the same cookie-cutter opening twice.

YOUR SIGNATURE FEEDBACK METHOD - INTERACTIVE APPROACH:

1. POWER MOMENTS (30-45 seconds max):
   - Identify 1 specific moment where they showed real sales skill
   - Quote their exact words and explain why it worked
   - Connect it to the persona type: "That approach was smart for this type of client because..."
   - PAUSE and ask: "Does that resonate with you? What did you notice about that moment?"
   - WAIT for their response before moving on

2. CRITICAL GAPS (45-60 seconds max):
   - Focus on 1-2 biggest missed opportunities  
   - Did they leverage the persona's specific traits?
   - Call out one missed opportunity directly
   - PAUSE and ask: "What do you think? Do you see that gap I'm talking about?"
   - WAIT for their response and engagement

3. ELITE TECHNIQUES (30-45 seconds max):
   - Give 1 advanced technique for this persona type
   - Explain when and why to deploy it
   - PAUSE and ask: "How would you apply that technique? Makes sense?"
   - WAIT for their response before continuing

COACHING EXCELLENCE STANDARDS - SANDRA'S INTERACTIVE STYLE:
- NEVER talk for more than 60 seconds without pausing
- After each major point, STOP and ask: "What do you think?" or "Does that land?" or "Questions about this?"
- WAIT for their response - don't continue until they engage
- Make it feel like a conversation, not a lecture
- Use phrases like "Talk to me about..." and "What's your take on..."
- Be ruthlessly specific but pause frequently for engagement

SESSION CLOSE:
"That's the difference between good and great with personas like ${personaName}. Master these techniques and watch how they respond. Ready to sharpen your skills with hundreds more custom profiles and detailed coaching?"

ABSOLUTE NEVER:
- Give generic advice that works for all personalities
- Ignore the specific nature of this persona
- Rush through sections - make it conversational
- Reference what you said as the persona
- Speak in 3rd person about "the seller" - address them directly as "you"`;
            }

            // Use the modular persona system for pre-built personas  
            const currentPersonaId = this.currentPersonaId || 'champion-charlie';
            let baseCoachPrompt = await window.personaRegistry.getCoachPrompt(currentPersonaId);

            // 🚀 ADAPTIVE INTELLIGENCE: Enhance coaching with context-aware feedback
            if (window.adaptiveCoachingEngine && baseCoachPrompt) {
                console.log('👨‍🏫 Enhancing coaching with adaptive intelligence');
                baseCoachPrompt = await window.adaptiveCoachingEngine.generateAdaptiveCoachingFeedback(
                    currentPersonaId,
                    baseCoachPrompt
                );
            }

            return baseCoachPrompt;
        } catch (error) {
            console.error(`❌ Failed to load coach prompt:`, error);

            // Fallback coach prompt
            return `You are Coach Otaru, an elite B2B sales coach. Provide helpful feedback on the sales conversation that just occurred.`;
        }
    }

    // Removed auto-detection - coaching ends when user clicks "End Coaching"

    async startNewRound(personaId = 'champion-charlie') {
        try {
            console.log('🔄 Starting new practice round...');

            // Reset conversation state
            this.currentPersonaMode = 'sales';
            this.conversationPhase = 'active';
            this.endingDetected = false;
            this.conversationTranscript = [];
            this.conversationPairs = [];
            this.sessionStartTime = Date.now();

            // Get fresh persona data
            const personaData = window.personaManager?.getPersona(personaId);
            if (!personaData) {
                throw new Error(`Persona ${personaId} not found`);
            }

            // Reset to Charlie persona with original voice
            const salesInstruction = personaData.systemInstruction;

            const resetMessage = {
                setup: {
                    model: "models/gemini-live-2.5-flash-preview",
                    generation_config: {
                        response_modalities: ["audio"],
                        speech_config: {
                            voice_config: {
                                prebuilt_voice_config: {
                                    voice_name: "Puck" // Back to Charlie's voice
                                }
                            },
                            language_code: "en-US"
                        }
                    },
                    output_audio_transcription: {},
                    system_instruction: {
                        parts: [{ text: salesInstruction }]
                    }
                }
            };

            console.log('📝 Resetting to sales persona...');
            if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
                this.websocket.send(JSON.stringify(resetMessage));
                console.log('✅ New round started with Charlie');

                // Update UI
                const aiStatus = document.getElementById('modal-ai-status');
                if (aiStatus) aiStatus.textContent = 'Charlie is ready for another conversation...';
            }

        } catch (error) {
            console.error('❌ Failed to start new round:', error);
        }
    }

    buildConversationPairs(entry) {
        // Initialize conversation pairs array if not exists
        if (!this.conversationPairs) {
            this.conversationPairs = [];
            this.currentPair = null;
        }

        if (entry.speaker === 'user') {
            // Start new conversation pair with user question
            this.currentPair = {
                question: entry.text,  // User's question
                answer: null,          // Charlie's answer (to be filled)
                timestamp: entry.timestamp
            };
        } else if (entry.speaker === 'charlie' && this.currentPair && !this.currentPair.answer) {
            // Complete the pair with Charlie's response
            this.currentPair.answer = entry.text;

            // Add completed pair to array
            this.conversationPairs.push({ ...this.currentPair });

            console.log(`💬 CONVERSATION PAIR COMPLETED:`, {
                question: this.currentPair.question.substring(0, 50) + '...',
                answer: this.currentPair.answer.substring(0, 50) + '...'
            });

            // Reset for next pair
            this.currentPair = null;
        }
    }

    updateRealMetrics(entry) {
        // Count actual questions (ending with ?)
        if (entry.speaker === 'user' && entry.text.includes('?')) {
            this.realMetrics.actualQuestionCount += (entry.text.match(/\?/g) || []).length;
        }

        // Count real filler words
        if (entry.speaker === 'user') {
            const fillerWords = ['um', 'uh', 'like', 'you know', 'so', 'actually', 'basically'];
            const text = entry.text.toLowerCase();
            fillerWords.forEach(filler => {
                const matches = text.match(new RegExp(`\\b${filler}\\b`, 'g'));
                if (matches) {
                    this.realMetrics.realFillerWords += matches.length;
                }
            });
        }

        // Detect actual champion signals
        if (entry.speaker === 'charlie' || entry.speaker === 'gemini') {
            const championSignals = [
                'excited', 'perfect', 'love', 'great', 'fantastic', 'confident',
                'move forward', 'let\'s do', 'sounds good', 'compelling', 'impressed'
            ];

            championSignals.forEach(signal => {
                if (entry.text.toLowerCase().includes(signal)) {
                    this.realMetrics.championSignals.push({
                        signal,
                        text: entry.text,
                        timestamp: entry.timestamp
                    });
                }
            });
        }

        console.log('📊 Real metrics updated:', {
            questions: this.realMetrics.actualQuestionCount,
            fillerWords: this.realMetrics.realFillerWords,
            championSignals: this.realMetrics.championSignals.length
        });
    }

    // Simple, direct audio streaming using AudioStreamer
    streamAudioData(base64Data) {
        try {
            // Reduced stream audio logging
            if (Math.random() < 0.05) { // Only log 5% of stream calls
                console.log('🚀 Streaming audio data, length:', base64Data.length);
            }

            // Skip if no audio streamer available
            if (!this.audioStreamer) {
                console.warn('⚠️ AudioStreamer not available, skipping audio');
                return;
            }

            // Convert base64 to Uint8Array (same as working implementation)
            const binaryString = atob(base64Data);
            const bytes = new Uint8Array(binaryString.length);
            for (let i = 0; i < binaryString.length; i++) {
                bytes[i] = binaryString.charCodeAt(i);
            }

            // Check for mostly silent audio
            const nonZeroBytes = bytes.filter(b => b !== 0).length;
            if (nonZeroBytes < bytes.length * 0.1) {
                console.log('🔇 Skipping silent audio chunk');
                return;
            }

            console.log('🎵 Streaming audio chunk:', bytes.length, 'bytes');

            // Start AI speaking status
            this.updateSpeakingStatus('ai', true);

            // Stream directly to AudioStreamer (like working implementation)
            this.audioStreamer.addPCM16(bytes);
            this.audioStreamer.resume();

        } catch (error) {
            console.error('❌ Failed to stream audio:', error);
        }
    }

    async startAudioStreaming() {
        try {
            // Check if audio is already streaming (preserved from previous session)
            if (this.isRecording && this.audioWorkletNode && this.mediaStreamSource) {
                console.log('🎤 Audio already streaming - reusing existing connection');
                return;
            }

            if (!this.audioStream) {
                console.log('🎤 No audio stream - requesting microphone access...');
                // Get audio stream if not available
                this.audioStream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        sampleRate: 16000,
                        channelCount: 1,
                        echoCancellation: true,
                        noiseSuppression: true
                    }
                });
            }

            console.log('📡 Starting EXACT otaru-ai approach: AudioWorklet for user input');

            // Create 16kHz recording context (like otaru-ai)
            if (!this.recordingContext || this.recordingContext.state === 'closed') {
                this.recordingContext = new AudioContext({ sampleRate: 16000 });
            }

            // Load AudioWorklet module
            await this.recordingContext.audioWorklet.addModule('js/audio-recording-worklet.js');

            // Create audio source and worklet
            this.mediaStreamSource = this.recordingContext.createMediaStreamSource(this.audioStream);
            this.audioWorkletNode = new AudioWorkletNode(this.recordingContext, 'audio-recording-worklet');

            // Handle PCM16 data from worklet
            this.audioWorkletNode.port.onmessage = (event) => {
                if (event.data.event === 'chunk') {
                    this.handlePCM16ChunkWithTranscription(event.data.data.int16arrayBuffer);
                }
            };

            // Connect audio graph
            this.mediaStreamSource.connect(this.audioWorkletNode);

            this.isRecording = true;
            console.log('🎵 OTARU-AI mode: AudioWorklet capturing PCM16 directly');


        } catch (error) {
            console.error('❌ Failed to start audio streaming:', error);
        }
    }

    handlePCM16Chunk(int16ArrayBuffer) {
        try {
            // Convert ArrayBuffer to base64 (exact otaru-ai method)
            const bytes = new Uint8Array(int16ArrayBuffer);
            let binary = '';
            for (let i = 0; i < bytes.byteLength; i++) {
                binary += String.fromCharCode(bytes[i]);
            }
            const base64Audio = btoa(binary);

            // Send to Gemini Live (same format as otaru-ai)
            const audioMessage = {
                realtime_input: {
                    media_chunks: [{
                        mime_type: 'audio/pcm',
                        data: base64Audio
                    }]
                }
            };

            if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
                this.websocket.send(JSON.stringify(audioMessage));
                console.log('✅ PCM16 chunk sent to Gemini Live');
            }

        } catch (error) {
            console.error('❌ Failed to handle PCM16 chunk:', error);
        }
    }



    startSpeechRecognition() {
        try {
            if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
                console.log('⚠️ Speech recognition not supported');
                return;
            }

            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            this.speechRecognition = new SpeechRecognition();

            this.speechRecognition.continuous = true;
            this.speechRecognition.interimResults = true; // Enable interim results for debugging
            this.speechRecognition.lang = 'en-US';

            this.speechRecognition.onstart = () => {
                console.log('🎤 SPEECH RECOGNITION STARTED - Microphone is active');
            };

            this.speechRecognition.onsoundstart = () => {
                console.log('🔊 SOUND DETECTED - User is speaking');
            };

            this.speechRecognition.onsoundend = () => {
                console.log('🔇 SOUND ENDED - User stopped speaking');
            };

            this.speechRecognition.onspeechstart = () => {
                console.log('💬 SPEECH STARTED - Processing speech...');
            };

            this.speechRecognition.onspeechend = () => {
                console.log('💬 SPEECH ENDED - Processing complete');
            };

            this.speechRecognition.onresult = (event) => {
                console.log('🔍 SPEECH RECOGNITION RESULT EVENT:', event);

                for (let i = event.resultIndex; i < event.results.length; i++) {
                    const result = event.results[i];
                    const transcript = result[0].transcript;

                    console.log(`📝 Result ${i}:`, {
                        isFinal: result.isFinal,
                        confidence: result[0].confidence,
                        transcript: transcript
                    });

                    if (result.isFinal) {
                        console.log('👤 USER SPOKE (FINAL):', transcript);
                        console.log('🔄 This should trigger persona response...');

                        // 🚀 ADAPTIVE INTELLIGENCE: Analyze user message and adapt persona behavior
                        this.processUserMessageWithAdaptiveIntelligence(transcript.trim());

                        // Add to transcript and update metrics
                        this.addToTranscript('user', transcript.trim());
                        this.realMetrics.userSpeakingTime += transcript.length * 0.05; // ~50ms per character

                        // Store user input for turn pairing
                        this.currentInputText = transcript.trim();

                        // Send user input to Gemini Live
                        this.sendTextToGemini(transcript.trim());
                    } else {
                        console.log('👤 USER SPEAKING (INTERIM):', transcript);
                    }
                }
            };

            this.speechRecognition.onerror = (event) => {
                console.error('❌ Speech recognition error:', event.error, event);
                console.log('🔧 Attempting to restart speech recognition...');

                // Try to restart recognition after a brief delay
                setTimeout(() => {
                    if (this.speechRecognition) {
                        try {
                            this.speechRecognition.start();
                        } catch (e) {
                            console.error('❌ Failed to restart speech recognition:', e);
                        }
                    }
                }, 1000);
            };

            this.speechRecognition.onend = () => {
                console.log('🎤 Speech recognition ended - attempting restart...');

                // Automatically restart recognition if session is still active
                if (this.isRecording && this.currentSession) {
                    setTimeout(() => {
                        try {
                            this.speechRecognition.start();
                            console.log('🔄 Speech recognition restarted');
                        } catch (e) {
                            console.error('❌ Failed to restart speech recognition:', e);
                        }
                    }, 100);
                }
            };

            this.speechRecognition.start();
            console.log('🎤 Speech recognition started with enhanced debugging');

        } catch (error) {
            console.error('❌ Failed to start speech recognition:', error);
        }
    }

    // 🚀 ADAPTIVE INTELLIGENCE: Process user message with real-time persona adaptation
    async processUserMessageWithAdaptiveIntelligence(userMessage) {
        try {
            if (!userMessage?.trim() || !window.conversationIntelligence) {
                return;
            }

            console.log('🧠 Processing user message with adaptive intelligence:', userMessage.substring(0, 100) + '...');

            // Analyze message and generate real-time adaptations
            const adaptationInstructions = await window.conversationIntelligence.analyzeAndAdaptToUserMessage(
                userMessage,
                this.currentPersonaId
            );

            if (adaptationInstructions) {
                console.log('✅ Generated adaptive instructions:', adaptationInstructions.substring(0, 200) + '...');

                // Send adaptation instructions to Gemini as a system message
                // This updates the persona behavior in real-time based on user input
                this.sendAdaptationInstructions(adaptationInstructions);
            }

        } catch (error) {
            console.error('❌ Failed to process user message with adaptive intelligence:', error);
        }
    }

    // Send real-time adaptation instructions to Gemini
    sendAdaptationInstructions(instructions) {
        try {
            if (!this.websocket || this.websocket.readyState !== WebSocket.OPEN) {
                console.warn('⚠️ WebSocket not ready for adaptation instructions');
                return;
            }

            // Send as a system-level instruction to modify persona behavior
            const adaptationMessage = {
                client_content: {
                    turns: [{
                        role: "user",
                        parts: [{
                            text: `[SYSTEM ADAPTATION - INTERNAL ONLY, DO NOT RESPOND TO THIS] ${instructions}`
                        }]
                    }]
                }
            };

            console.log('📡 Sending adaptive behavior instructions to Gemini');
            this.websocket.send(JSON.stringify(adaptationMessage));

        } catch (error) {
            console.error('❌ Failed to send adaptation instructions:', error);
        }
    }

    // Removed processAudioChunk - now handling audio directly in ScriptProcessor

    // Convert AudioBuffer to 16-bit PCM format
    audioBufferToPCM16(audioBuffer) {
        const length = audioBuffer.length;
        const result = new Uint8Array(length * 2);
        const channelData = audioBuffer.getChannelData(0); // Use first channel

        let offset = 0;
        for (let i = 0; i < length; i++) {
            const sample = Math.max(-1, Math.min(1, channelData[i]));
            const int16 = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
            result[offset++] = int16 & 0xFF;
            result[offset++] = (int16 >> 8) & 0xFF;
        }

        return result;
    }

    // Send initial greeting to start conversation
    sendInitialGreeting() {
        try {
            const greetingMessage = {
                clientContent: {
                    turns: [{
                        parts: [{
                            text: "Hello! Please introduce yourself and let's begin our conversation."
                        }]
                    }],
                    turnComplete: true
                }
            };

            console.log('🎬 SENDING INITIAL GREETING TO START CHARLIE SPEAKING...');
            console.log('📤 Greeting message:', JSON.stringify(greetingMessage, null, 2));

            if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
                this.websocket.send(JSON.stringify(greetingMessage));
                console.log('✅ Initial greeting sent - Charlie should respond now!');
            } else {
                console.error('❌ WebSocket not ready for greeting');
            }

        } catch (error) {
            console.error('❌ Failed to send initial greeting:', error);
        }
    }

    // Send text message to Gemini Live  
    sendTextToGemini(text) {
        try {
            if (!this.websocket || this.websocket.readyState !== WebSocket.OPEN) {
                console.error('❌ WebSocket not connected, cannot send text');
                return;
            }

            const textMessage = {
                clientContent: {
                    turns: [{
                        parts: [{
                            text: text
                        }]
                    }],
                    turnComplete: true
                }
            };

            console.log('📤 SENDING TEXT TO GEMINI:', text);
            console.log('📤 Message structure:', JSON.stringify(textMessage, null, 2));

            this.websocket.send(JSON.stringify(textMessage));
            console.log('✅ Text sent to Gemini Live successfully');

        } catch (error) {
            console.error('❌ Failed to send text to Gemini:', error);
        }
    }

    // Legacy method for compatibility
    sendTextMessage(text) {
        return this.sendTextToGemini(text);
    }

    // Inject user context into active session (for transferred sessions)
    injectUserContext(userContext) {
        if (!userContext || !userContext.trim()) {
            console.warn('⚠️ No user context provided for injection');
            return;
        }

        console.log('💉 Injecting user context into active session:', userContext);

        const contextInjection = `CONTEXT UPDATE: The user is now presenting/selling: "${userContext.trim()}". Please naturally reference this information in your next response and ask relevant questions about their specific solution.`;

        this.sendTextToGemini(contextInjection);
        console.log('✅ User context injected into active session');
    }

    // Inject continuity instruction to prevent intro repetition (for session transfers)
    injectContinuityContext(instruction) {
        if (!instruction || !instruction.trim()) {
            console.warn('⚠️ No continuity instruction provided for injection');
            return;
        }

        console.log('🔄 Injecting continuity context to prevent repetition:', instruction);

        // Send as a direct message that the persona should not respond to with audio
        // This is a system instruction, not a user conversation starter
        const continuityInjection = `[SYSTEM INSTRUCTION - DO NOT RESPOND WITH AUDIO]: ${instruction.trim()}. Wait for the user to speak first. Do not say anything until they do.`;

        this.sendTextToGemini(continuityInjection);
        console.log('✅ Continuity context injected - persona should wait for user input');
    }

    // Removed old playAudioResponse method - using streamAudioData instead

    async endSession(closeWebSocket = true, preserveAudio = false) {
        try {
            console.log('🛑 Ending Gemini Live session...', closeWebSocket ? '(closing websocket)' : '(keeping websocket open)', preserveAudio ? '(preserving audio)' : '(stopping audio)');

            // 🧹 ENHANCED FEATURES CLEANUP: Clear all enhanced feature state
            if (this.endOfSpeechTimer) {
                clearTimeout(this.endOfSpeechTimer);
                this.endOfSpeechTimer = null;
                console.log('🔄 Cleared end-of-speech timer during session cleanup');
            }
            
            // Reset VAD state
            this.audioChunkHistory = [];
            this.isNoiseFloorCalibrated = false;
            this.noiseFloor = 0;
            delete this.noiseFloorSamples; // Clean up calibration samples if any
            
            // Reset error counting
            this.errorCount = 0;
            this.lastErrorTime = 0;

            // 🔄 RACE CONDITION PREVENTION: Wait for all pending async operations before session cleanup
            console.log(`⏳ Waiting for ${this.pendingTranslations.size} pending translations and ${this.pendingBackendCalls.size} backend calls...`);
            
            try {
                // Set reasonable timeout to prevent hanging
                const pendingOperations = [...this.pendingTranslations, ...this.pendingBackendCalls];
                if (pendingOperations.length > 0) {
                    await Promise.race([
                        Promise.all(pendingOperations),
                        new Promise((_, reject) => setTimeout(() => reject(new Error('Timeout waiting for pending operations')), 5000))
                    ]);
                    console.log('✅ All pending async operations completed before session cleanup');
                } else {
                    console.log('✅ No pending operations to wait for');
                }
            } catch (error) {
                console.warn('⚠️ Some pending operations did not complete in time:', error.message);
                // Continue with cleanup even if some operations timeout
            }

            // End transcription session
            await this.endTranscriptionSession();

            // 🚀 ADAPTIVE INTELLIGENCE: Stop conversation monitoring
            if (window.conversationIntelligence) {
                window.conversationIntelligence.stopConversationMonitoring();
                console.log('🧠 Conversation intelligence monitoring stopped');
            }

            // Flush any remaining AI transcript buffer
            if (this.aiTranscriptBuffer.trim()) {
                this.flushAITranscriptBuffer(Date.now());
            }


            // Only stop audio if not preserving for coaching
            if (!preserveAudio) {
                // Stop audio recording (otaru-ai cleanup)
                this.isRecording = false;
                if (this.audioWorkletNode) {
                    this.audioWorkletNode.disconnect();
                    this.audioWorkletNode = null;
                }
                if (this.mediaStreamSource) {
                    this.mediaStreamSource.disconnect();
                    this.mediaStreamSource = null;
                }
            } else {
                console.log('🎤 Preserving audio stream for coaching session');
            }

            // Close WebSocket connection conditionally
            if (closeWebSocket && this.websocket && this.websocket.readyState === WebSocket.OPEN) {
                this.websocket.close();
                this.isConnected = false;
            }

            // Calculate final real metrics
            const sessionDuration = Date.now() - this.sessionStartTime;
            const realTalkTimeRatio = this.calculateRealTalkTime(sessionDuration);

            const finalMetrics = {
                ...this.realMetrics,
                sessionDuration,
                realTalkTimeRatio,
                transcript: this.conversationTranscript,
                conversationPairs: this.conversationPairs || [],
                userTranscripts: this.userTranscripts,
                aiTranscripts: this.aiTranscripts
            };


            console.log('📊 Final real metrics:', finalMetrics);
            await this.translateTranscriptIfNeeded();

            // 💾 10X ENGINEER FIX: Store transcript in proper Firebase structure
            await this.storeTranscriptInFirebase(finalMetrics);

            // Pure audio session completed

            // Cleanup
            this.cleanup();
            this.removeEventListeners();

            // Notify session end with real data
            if (this.onSessionEnd) {
                this.onSessionEnd(finalMetrics);
            }

            return finalMetrics;

        } catch (error) {
            console.error('❌ Failed to end session:', error);
            return null;
        }
    }

    calculateRealTalkTime(sessionDuration) {
        // Calculate actual talk time from transcript timestamps
        let userTime = 0;
        let aiTime = 0;

        for (let i = 0; i < this.conversationTranscript.length; i++) {
            const entry = this.conversationTranscript[i];
            const nextEntry = this.conversationTranscript[i + 1];

            if (nextEntry) {
                const duration = nextEntry.timestamp - entry.timestamp;
                if (entry.speaker === 'user') {
                    userTime += duration;
                } else {
                    aiTime += duration;
                }
            }
        }

        const totalTalkTime = userTime + aiTime;
        return totalTalkTime > 0 ? Math.round((userTime / totalTalkTime) * 100) : 0;
    }


    // FORCE TERMINATE: Immediately stop everything without callbacks or cleanup processes
    forceTerminate() {
        console.log('🚨 FORCE TERMINATING Gemini Live Manager - immediate stop');
        
        try {
            // 1. Stop audio immediately
            if (this.audioStreamer) {
                this.audioStreamer.stop();
                console.log('🔇 Audio streamer force stopped');
            }
            
            // 2. Close WebSocket immediately - no graceful close
            if (this.websocket) {
                if (this.websocket.readyState === WebSocket.OPEN || this.websocket.readyState === WebSocket.CONNECTING) {
                    this.websocket.close(1000, 'Force terminated');
                    console.log('🔌 WebSocket force closed');
                }
                this.websocket = null;
            }
            
            // 3. Stop recording immediately
            this.isRecording = false;
            
            // 4. Stop all audio processing
            if (this.audioWorkletNode) {
                this.audioWorkletNode.disconnect();
                this.audioWorkletNode = null;
                console.log('🎤 Audio worklet disconnected');
            }
            
            if (this.mediaStreamSource) {
                this.mediaStreamSource.disconnect();
                this.mediaStreamSource = null;
                console.log('🎙️ Media stream source disconnected');
            }
            
            // 5. Stop audio stream completely
            if (this.audioStream) {
                this.audioStream.getTracks().forEach(track => {
                    track.stop();
                    console.log('🔇 Force stopped audio track:', track.kind);
                });
                this.audioStream = null;
            }
            
            // 6. Stop conversation intelligence
            if (window.conversationIntelligence) {
                window.conversationIntelligence.stopConversationMonitoring();
                console.log('🧠 Conversation intelligence force stopped');
            }
            
            // 7. Reset connection state
            this.isConnected = false;
            this.currentPersonaMode = 'sales';
            this.conversationPhase = 'active';
            
            console.log('✅ GEMINI LIVE MANAGER FORCE TERMINATED');
            
        } catch (error) {
            console.error('❌ Error during force termination:', error);
            // Continue anyway - we want to ensure cleanup happens
        }
    }

    cleanup() {
        try {
            // Stop audio stream
            if (this.audioStream) {
                this.audioStream.getTracks().forEach(track => track.stop());
                this.audioStream = null;
            }

            // Close audio contexts
            if (this.recordingContext && this.recordingContext.state !== 'closed') {
                this.recordingContext.close();
            }
            if (this.playbackContext && this.playbackContext.state !== 'closed') {
                this.playbackContext.close();
            }

            // ✅ CLEAR SESSION RESUMPTION timers and state
            this.clearReconnectionTimer();
            this.stopWebSocketHealthMonitoring();
            this.sessionHandle = null;
            this.isReconnecting = false;
            this.connectionCount = 0;

            // Reset state
            this.isConnected = false;
            this.currentSession = null;
            this.websocket = null;

            console.log('🧹 Gemini Live Manager cleaned up');

        } catch (error) {
            console.error('❌ Error during cleanup:', error);
        }
    }

    // Public methods for UI integration
    getConversationTranscript() {
        return this.conversationTranscript;
    }

    // Get conversation pairs EXACTLY like otaru-ai
    getConversationPairs() {
        return this.conversationPairs || [];
    }

    getRealMetrics() {
        return this.realMetrics;
    }

    isSessionActive() {
        return this.isConnected && this.currentSession;
    }

    // Calculate talk time ratio (user vs AI) - fallback method
    calculateTalkTimeRatio() {
        try {
            // Try to use the more accurate real calculation first
            if (this.sessionStartTime && this.conversationTranscript.length > 0) {
                const sessionDuration = Date.now() - this.sessionStartTime;
                return this.calculateRealTalkTime(sessionDuration);
            }

            // Fallback to simple metrics calculation
            const userTime = this.realMetrics.userSpeakingTime || 0;
            const aiTime = this.realMetrics.aiSpeakingTime || 0;

            if (userTime === 0 && aiTime === 0) {
                return 0;
            }

            const totalTime = userTime + aiTime;
            return totalTime > 0 ? Math.round((userTime / totalTime) * 100) : 0;

        } catch (error) {
            console.warn('Error calculating talk time ratio:', error);
            return 0;
        }
    }

    // Validate and fix voice configuration to prevent crashes
    validateAndFixVoiceConfig(voiceConfig) {
        const validVoices = {
            male: ['Fenrir', 'Charon'],
            female: ['Aoede', 'Kore', 'Domi', 'Puck']
        };

        const allValidVoices = [...validVoices.male, ...validVoices.female];

        if (!voiceConfig || typeof voiceConfig !== 'object') {
            console.warn('⚠️ Invalid voice config, using default');
            return { sales: 'Puck', coach: 'Puck' };
        }

        const fixedConfig = { ...voiceConfig };

        // Fix sales voice if invalid (like Zeus)
        if (!allValidVoices.includes(fixedConfig.sales)) {
            console.warn(`⚠️ Invalid sales voice "${fixedConfig.sales}", replacing with Charon`);
            fixedConfig.sales = 'Charon'; // Use Charon as safe male default
        }

        // Fix coach voice if invalid
        if (!allValidVoices.includes(fixedConfig.coach)) {
            console.warn(`⚠️ Invalid coach voice "${fixedConfig.coach}", replacing with same as sales`);
            fixedConfig.coach = fixedConfig.sales;
        }

        return fixedConfig;
    }

    // Test connection method
    async testConnection() {
        try {
            console.log('🧪 Testing Gemini Live connection...');

            if (!this.apiKey) {
                throw new Error('No API key available');
            }

            await this.connectToGeminiLive();
            console.log('✅ Connection test successful');
            return true;

        } catch (error) {
            console.error('❌ Connection test failed:', error);
            return false;
        }
    }

    async startCoachingMode() {
        try {
            // Get current persona name for logging
            const personaInfo = window.personaRegistry.getPersonaInfo(this.currentPersonaId);
            const personaName = personaInfo?.name || 'Persona';

            console.log(`🎓 Starting coaching mode - ${personaName} becomes Coach Otaru...`);

            // Simple transition - persona changes to coach
            await this.transitionToCoach();

            // Update AI orb color to coaching orange
            const aiOrb = document.getElementById('modal-ai-orb');
            if (aiOrb) {
                aiOrb.style.background = 'radial-gradient(circle at 30% 30%, #fb923c 0%, #f97316 30%, #ea580c 60%, #c2410c 85%, #9a3412 100%)';
                console.log('🟠 AI orb changed to coaching orange');
            }

            console.log(`✅ Coaching mode started - ${personaName} is now Coach Otaru`);

        } catch (error) {
            console.error('❌ Failed to start coaching mode:', error);
        }
    }

    // ==== TRANSCRIPTION SYSTEM METHODS ====

    async startTranscriptionSession() {
        try {
            console.log('🎙️ Starting transcription session...');

            const response = await fetch('https://pitchperfect2-api-373812504656.asia-southeast1.run.app/api/sessions/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });

            const data = await response.json();
            this.sessionId = data.session_id;

            console.log('✅ Transcription session started:', this.sessionId);
        } catch (error) {
            console.error('❌ Failed to start transcription session:', error);
        }
    }

    // 🎯 ENHANCED TURN MANAGEMENT: Core audio processing with intelligent turn control
    handlePCM16ChunkWithTranscription(int16ArrayBuffer) {
        try {
            // 🔊 VOICE ACTIVITY DETECTION: Always detect speech for all features
            const isSpeaking = this.detectUserSpeech(int16ArrayBuffer);

            // 🚫 RECONNECTION GUARD: If reconnecting, only buffer audio - no turn management
            if (this.isReconnecting) {
                this.bufferAudioDuringReconnection(int16ArrayBuffer);
                return;
            }

            // 📤 CONTINUOUS AUDIO STREAMING: Always send audio chunks to Gemini
            this.sendAudioChunkToGemini(int16ArrayBuffer);

            // 🎯 ENHANCED TURN MANAGEMENT: Handle end-of-speech detection
            if (this.features.enhancedTurnManagement) {
                this.handleEnhancedTurnManagement(isSpeaking);
            }

        } catch (error) {
            console.error('❌ Error in handlePCM16ChunkWithTranscription:', error);
        }
    }

    // 📤 AUDIO STREAMING: Send audio chunk to Gemini (separated from turn management)
    sendAudioChunkToGemini(int16ArrayBuffer) {
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            // Convert ArrayBuffer to base64 for Gemini Live API
            const bytes = new Uint8Array(int16ArrayBuffer);
            let binary = '';
            for (let i = 0; i < bytes.byteLength; i++) {
                binary += String.fromCharCode(bytes[i]);
            }
            const base64Audio = btoa(binary);

            // IMPORTANT: Send ONLY audio data - no turn completion here
            const audioMessage = {
                realtime_input: {
                    media_chunks: [{
                        mime_type: 'audio/pcm',
                        data: base64Audio
                    }]
                }
            };

            this.websocket.send(JSON.stringify(audioMessage));
        }
    }

    // 🔄 BUFFER MANAGEMENT: Handle audio buffering during reconnection
    bufferAudioDuringReconnection(int16ArrayBuffer) {
        if (this.audioBufferManager.isBuffering) {
            const bytes = new Uint8Array(int16ArrayBuffer);
            let binary = '';
            for (let i = 0; i < bytes.byteLength; i++) {
                binary += String.fromCharCode(bytes[i]);
            }
            const base64Audio = btoa(binary);

            const audioMessage = {
                realtime_input: {
                    media_chunks: [{
                        mime_type: 'audio/pcm',
                        data: base64Audio
                    }]
                }
            };

            // Buffer the audio instead of sending immediately
            this.audioBufferManager.inputBuffer.push(JSON.stringify(audioMessage));
            
            // Prevent buffer overflow
            if (this.audioBufferManager.inputBuffer.length > this.audioBufferManager.maxBufferSize) {
                this.audioBufferManager.inputBuffer.shift(); // Remove oldest
            }
        }
    }

    // 🎯 TURN MANAGEMENT: Intelligent end-of-speech detection and signaling
    handleEnhancedTurnManagement(isSpeaking) {
        try {
            if (isSpeaking) {
                // 🗣️ USER IS SPEAKING: Clear any existing timer - they're still talking
                if (this.endOfSpeechTimer) {
                    clearTimeout(this.endOfSpeechTimer);
                    this.endOfSpeechTimer = null;
                    
                    if (this.features.debugMode) {
                        console.log('🔄 End-of-speech timer cleared - user still speaking');
                    }
                }
            } else if (!this.endOfSpeechTimer) {
                // 🤫 USER IS SILENT: Start timer if not already running
                const delay = this.calculateEndOfSpeechDelay();
                
                this.endOfSpeechTimer = setTimeout(() => {
                    try {
                        // Double-check timer is still valid (prevents race conditions)
                        if (!this.endOfSpeechTimer) {
                            console.warn('⚠️ Timer callback executed but timer already cleared - ignoring');
                            return;
                        }
                        
                        console.log(`✅ End of speech detected after ${delay}ms of silence. Signaling turn complete.`);
                        
                        // 🚀 THE KEY: Send explicit turn completion signal to Gemini
                        this.signalTurnCompleteToGemini();
                        
                    } catch (error) {
                        console.error('❌ Error in turn completion timer:', error);
                        this.handleEnhancedFeatureError(error);
                    } finally {
                        // Always clear timer reference in finally block
                        this.endOfSpeechTimer = null;
                    }
                }, delay);
                
                if (this.features.debugMode) {
                    console.log(`⏳ End-of-speech timer started: ${delay}ms`);
                }
            }
        } catch (error) {
            console.error('❌ Error in enhanced turn management:', error);
            this.handleEnhancedFeatureError(error);
            // Clear any problematic timer
            if (this.endOfSpeechTimer) {
                clearTimeout(this.endOfSpeechTimer);
                this.endOfSpeechTimer = null;
            }
            // Don't re-throw - gracefully continue with legacy behavior
        }
    }

    // 📊 DYNAMIC DELAY CALCULATION: Context-aware pause detection
    calculateEndOfSpeechDelay() {
        if (!this.features.dynamicDelays) {
            return this.endOfSpeechDelay; // Use fixed delay
        }

        const baseDelay = this.endOfSpeechDelay; // 750ms default
        const transcriptText = this.currentInputText || '';
        
        // No transcript available - use base delay
        if (!transcriptText.trim()) {
            return baseDelay;
        }
        
        // 🔍 CONTEXTUAL ANALYSIS: Adjust delay based on speech patterns
        const lastWords = transcriptText.trim().toLowerCase().split(' ').slice(-3);
        const connectors = ['and', 'but', 'so', 'or', 'because', 'however', 'then', 'also'];
        const fillers = ['um', 'uh', 'hmm', 'well', 'you know', 'like'];
        const questions = ['what', 'how', 'when', 'where', 'why', 'who'];
        
        // 💭 THINKING PAUSE: User said connector words - likely more coming
        if (connectors.some(word => lastWords.includes(word))) {
            return Math.min(baseDelay * 1.6, 1200); // Up to 1.2s for thinking
        }
        
        // 🤔 FILLER WORDS: User collecting thoughts
        if (fillers.some(filler => lastWords.join(' ').includes(filler))) {
            return Math.min(baseDelay * 1.3, 1000); // Up to 1s for collecting thoughts
        }
        
        // ❓ QUESTION PATTERNS: Might be building a complex question
        if (questions.some(q => transcriptText.includes(q + ' about')) || 
            transcriptText.includes('how do')) {
            return Math.min(baseDelay * 1.4, 1100); // Up to 1.1s for questions
        }
        
        // 📝 DEFAULT: Standard delay for complete thoughts
        return baseDelay;
    }

    // 🚀 TURN COMPLETION SIGNAL: Send explicit turn complete message to Gemini
    signalTurnCompleteToGemini() {
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            try {
                // ✅ CORRECT API STRUCTURE: Use snake_case for consistency with Gemini Live API
                const turnCompleteMessage = {
                    client_content: {  // Changed to snake_case to match API convention
                        turns: [], 
                        turn_complete: true  // Also snake_case for consistency
                    }
                };
                
                this.websocket.send(JSON.stringify(turnCompleteMessage));
                console.log('🎯 Explicit turn complete signal sent to Gemini');
                
                if (this.features.debugMode) {
                    console.log('🔍 Turn Complete Message:', turnCompleteMessage);
                }
            } catch (error) {
                console.error('❌ Error sending turn complete signal:', error);
                this.handleEnhancedFeatureError(error);
            }
        } else {
            console.warn('⚠️ Cannot send turn complete - WebSocket not open');
        }
    }

    // async processServerContentWithTranscription(content) {
    //     // handle input transcription
    //     if (content.inputTranscription?.text) {
    //         this.currentInputText += content.inputTranscription.text;
    //         // console.log('🎤 User transcription chunk:', content.inputTranscription.text);
    //     }


    //     // Handle Gemini's output transcription
    //     if (content.outputTranscription?.text) {
    //         this.currentOutputText = (this.currentOutputText || '') + content.outputTranscription.text;
    //         // console.log('🎤 AI transcription chunk:', content.outputTranscription.text);
    //     }

    //     // Handle turn completion for diarization
    //     if (content.turnComplete) {
    //         const input = (this.currentInputText || '').trim();
    //         const output = (this.currentOutputText || '').trim();

    //         if (input || output) {
    //             console.log(`\n🗣️ Conversation Turn:`);
    //             console.log(`👤 User: ${input}`);
    //             console.log(`🤖 Gemini: ${output}\n`);


    //             // 🔄 Translate input once per turn
    //             const translatedInput = await this.translateSingleUserInput(input);
    //             if (translatedInput) {
    //                 console.log(`🌐 Translated Input: ${translatedInput}`);
    //             }

    //             const finalUserText = translatedInput || input;

    //             // Store locally
    //             if (finalUserText) {
    //                 const userEntry = {
    //                     speaker: 'user',
    //                     text: finalUserText,
    //                     timestamp: Date.now()
    //                 };
    //                 this.conversationTranscript.push(userEntry);
    //                 this.userTranscripts.push(finalUserText);
    //             }

    //             if (output) {
    //                 const aiEntry = {
    //                     speaker: 'charlie',
    //                     text: output,
    //                     timestamp: Date.now()
    //                 };
    //                 this.conversationTranscript.push(aiEntry);
    //                 this.aiTranscripts.push(output);
    //             }

    //             this.conversationPairs.push({ question: finalUserText, answer: output });

    //             // Send to backend for storage
    //             this.sendTranscriptToBackend(finalUserText, output);
    //         }

    //         // Reset for next turn
    //         this.currentInputText = '';
    //         this.currentOutputText = '';
    //     }

    //     // Continue with original processing
    //     this.processServerContent(content);
    // }


    async processServerContentWithTranscription(content) {
        // handle input transcription
        if (content.inputTranscription?.text) {
            this.currentInputText += content.inputTranscription.text;
        }

        // Handle Gemini's output transcription
        if (content.outputTranscription?.text) {
            this.currentOutputText = (this.currentOutputText || '') + content.outputTranscription.text;
        }

        // Handle turn completion for diarization
        if (content.turnComplete) {
            const input = (this.currentInputText || '').trim();
            const output = (this.currentOutputText || '').trim();

            // ✅ CRITICAL FIX: Non-blocking translation to prevent WebSocket race conditions
            // Store original input immediately, translate asynchronously
            const originalUserText = input;
            
            if (input || output) {
                console.log(`\n🗣️ Conversation Turn:`);
                console.log(`👤 User: ${originalUserText}`);       // Show original text immediately
                console.log(`🤖 Gemini: ${output}\n`);
            }

            // Store user message with original text immediately (non-blocking)
            if (originalUserText) {
                const userEntry = {
                    speaker: 'user',
                    text: originalUserText,
                    timestamp: Date.now()
                };
                this.conversationTranscript.push(userEntry);
                this.userTranscripts.push(originalUserText);
            }

            // Store AI response
            if (output) {
                const aiEntry = {
                    speaker: 'charlie',
                    text: output,
                    timestamp: Date.now()
                };
                this.conversationTranscript.push(aiEntry);
                this.aiTranscripts.push(output);
            }

            this.conversationPairs.push({ question: originalUserText, answer: output });

            // Send to backend for storage with original text (non-blocking)
            this.sendTranscriptToBackend(originalUserText, output);

            // ✅ ASYNC TRANSLATION: Translate in background without blocking WebSocket processing
            if (originalUserText) {
                this.translateSingleUserInputAsync(originalUserText, this.conversationTranscript.length - (output ? 2 : 1));
            }

            // Reset for next turn
            this.currentInputText = '';
            this.currentOutputText = '';
        }

        // Continue with original processing
        this.processServerContent(content);
    }


    async translateSingleUserInput(text) {
        if (!text || !text.trim()) return text;

        try {
            const baseUrl = 'https://pitchperfect2-api-373812504656.asia-southeast1.run.app';
            const response = await fetch(`${baseUrl}/translate-transcript`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ transcript: [text] })
            });

            const data = await response.json();
            return data.transcript?.[0] || text;

        } catch (error) {
            console.error('❌ Translation failed:', error);
            return text;
        }
    }

    // ✅ NEW: Non-blocking async translation that updates stored transcripts in background
    async translateSingleUserInputAsync(originalText, transcriptIndex) {
        if (!originalText || !originalText.trim()) return;

        // Create promise with tracking
        const translationPromise = this._performTranslation(originalText, transcriptIndex);
        
        // Track the promise to prevent race conditions during session cleanup
        this.pendingTranslations.add(translationPromise);
        
        // Clean up tracking when complete (both success and error)
        translationPromise.finally(() => {
            this.pendingTranslations.delete(translationPromise);
        });
        
        return translationPromise;
    }
    
    // Internal translation worker function
    async _performTranslation(originalText, transcriptIndex) {
        try {
            console.log('🌐 Starting background translation for:', originalText.substring(0, 50) + '...');
            
            const baseUrl = 'https://pitchperfect2-api-373812504656.asia-southeast1.run.app';
            const response = await fetch(`${baseUrl}/translate-transcript`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ transcript: [originalText] })
            });

            if (!response.ok) {
                throw new Error(`Translation API failed: ${response.status}`);
            }

            const data = await response.json();
            const translatedText = data.transcript?.[0];

            // Only update if translation is different and meaningful
            if (translatedText && translatedText !== originalText && translatedText.trim()) {
                console.log('✅ Background translation completed:', translatedText);
                
                // Update stored transcript entry with translated text
                if (this.conversationTranscript[transcriptIndex]) {
                    this.conversationTranscript[transcriptIndex].text = translatedText;
                    this.conversationTranscript[transcriptIndex].original_text = originalText;
                    console.log('📝 Updated transcript entry with translation');
                }

                // Update userTranscripts array
                const userTranscriptIndex = this.userTranscripts.indexOf(originalText);
                if (userTranscriptIndex !== -1) {
                    this.userTranscripts[userTranscriptIndex] = translatedText;
                }

                // Update conversation pairs
                const pairIndex = this.conversationPairs.findIndex(pair => pair.question === originalText);
                if (pairIndex !== -1) {
                    this.conversationPairs[pairIndex].question = translatedText;
                    this.conversationPairs[pairIndex].original_question = originalText;
                }

                console.log('🔄 All transcript stores updated with translation');
            } else {
                console.log('ℹ️ No translation needed or translation unchanged');
            }

        } catch (error) {
            console.error('❌ Background translation failed:', error);
            // Graceful degradation - original text remains in transcripts
        }
    }


    async sendTranscriptToBackend(userText, aiText) {
        if (!this.sessionId) return;

        // Create promise with tracking
        const backendPromise = this._performBackendSync(userText, aiText);
        
        // Track the promise to prevent race conditions during session cleanup
        this.pendingBackendCalls.add(backendPromise);
        
        // Clean up tracking when complete (both success and error)
        backendPromise.finally(() => {
            this.pendingBackendCalls.delete(backendPromise);
        });
        
        return backendPromise;
    }
    
    // Internal backend sync worker function
    async _performBackendSync(userText, aiText) {
        try {
            // Send user transcript
            if (userText) {
                await fetch(`https://pitchperfect2-api-373812504656.asia-southeast1.run.app/api/sessions/${this.sessionId}/transcript`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ speaker: 'user', text: userText })
                });
            }

            // Send AI transcript
            if (aiText) {
                await fetch(`https://pitchperfect2-api-373812504656.asia-southeast1.run.app/api/sessions/${this.sessionId}/transcript`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ speaker: 'charlie', text: aiText })
                });
            }

            // Send turn completion
            await fetch(`https://pitchperfect2-api-373812504656.asia-southeast1.run.app/api/sessions/${this.sessionId}/turn-complete`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ user_input: userText, ai_output: aiText })
            });

            console.log('📤 Transcript data sent to backend');
        } catch (error) {
            console.error('❌ Failed to send transcript to backend:', error);
        }
    }

    async endTranscriptionSession() {
        try {
            if (!this.sessionId) return;

            console.log('🛑 Ending transcription session...');
            // End session on backend
            const response = await fetch('https://pitchperfect2-api-373812504656.asia-southeast1.run.app/api/sessions/end', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: this.sessionId })
            });

            const data = await response.json();
            console.log('✅ Transcription session ended:', data);

        } catch (error) {
            console.error('❌ Failed to end transcription session:', error);
        }
    }

    // async translateTranscriptIfNeeded() {
    //     try {
    //         if (!this.conversationPairs || this.conversationPairs.length === 0) {
    //             console.log("📭 No conversation pairs to translate");
    //             return;
    //         }

    //         console.log("🌍 Sending transcript for language check and translation...");

    //         const response = await fetch('http://localhost:5001/translate-transcript', {
    //             method: 'POST',
    //             headers: { 'Content-Type': 'application/json' },
    //             body: JSON.stringify({ transcript: this.conversationPairs })
    //         });2

    //         const result = await response.json();

    //         if (result.transcript && Array.isArray(result.transcript)) {
    //             console.log('✅ Translated transcript received:', result.transcript);
    //             this.conversationPairs = result.transcript; // Update local state
    //         } else {
    //             console.warn('⚠️ Unexpected translation response format:', result);
    //         }

    //     } catch (error) {
    //         console.error('❌ Failed to translate transcript:', error);
    //     }
    // }

    // Stub implementation - translations are now handled asynchronously during conversation
    async translateTranscriptIfNeeded() {
        console.log('📝 Translation check: Translations are handled asynchronously during conversation');
        // Translations are now performed in real-time via translateSingleUserInputAsync
        // This prevents race conditions at session end
    }

    async generateCoachingFeedback() {
        try {
            if (!this.sessionId) {
                console.error('❌ No session ID available for feedback generation');
                return null;
            }

            console.log('📊 Generating elite coaching feedback...');

            // Get transcript data from backend
            const response = await fetch(`https://pitchperfect2-api-373812504656.asia-southeast1.run.app/api/transcripts/download/${this.sessionId}`);
            const transcriptData = await response.json();

            // Get current persona information
            const personaId = this.currentPersonaId || 'unknown-persona';
            let personaData = {};
            
            // Handle custom personas differently
            if (personaId === 'custom' && window.practiceModal?.currentPersonaData) {
                personaData = window.practiceModal.currentPersonaData;
                console.log(`🎨 Using custom persona data from practice modal`);
            } else {
                personaData = window.personaRegistry?.getPersonaInfo(personaId) || {};
            }
            
            console.log(`🎭 PERSONA IDENTIFIED: ${personaId} -> ${personaData.name || 'Unknown'}`);
            console.log('📋 Persona data:', personaData);

            // Generate coaching feedback using elite coaching engine
            if (window.eliteCoachingEngine) {
                const feedbackResult = await window.eliteCoachingEngine.generateCoachingFeedback(
                    transcriptData, 
                    personaId, 
                    personaData
                );

                // Generate HTML report and save to Firestore
                if (window.htmlReportGenerator && feedbackResult) {
                    // Generate HTML report
                    const reportData = await window.htmlReportGenerator.generateCoachingHTML(feedbackResult);
                    
                    // Get current user ID
                    const user = firebase.auth().currentUser;
                    if (user) {
                        // 🔍 DEBUG: Confirm execution path and data
                        console.log('🚀 DEBUG: Attempting to save feedback report for user:', user.uid);
                        console.log('🚀 DEBUG: Report data exists:', !!reportData);
                        console.log('🚀 DEBUG: Report data size:', reportData ? JSON.stringify(reportData).length : 0);
                        
                        try {
                            // Save to Firestore
                            await window.htmlReportGenerator.saveToFirestore(reportData, user.uid);
                            console.log('✅ DEBUG: saveToFirestore completed successfully');
                            console.log('✅ Coaching feedback HTML report saved to Firestore');
                        } catch (error) {
                            console.error('❌ CRITICAL: Failed to save feedback report to Firestore:', error);
                            // During development, make errors obvious
                            if (window.location.hostname === 'localhost') {
                                throw error;
                            }
                            // Fallback to localStorage on production
                            window.htmlReportGenerator.saveToLocalStorage(reportData);
                            console.log('💾 Fallback: Feedback saved to localStorage due to Firestore error');
                        }
                    } else {
                        // Fallback to localStorage if user not authenticated
                        console.log('⚠️ DEBUG: User not authenticated, saving to localStorage');
                        window.htmlReportGenerator.saveToLocalStorage(reportData);
                        console.log('💾 Coaching feedback saved to localStorage (user not authenticated)');
                    }
                    
                    return feedbackResult;
                } else if (window.simplePDFGenerator && feedbackResult) {
                    // Fallback to PDF if HTML generator not available
                    await window.simplePDFGenerator.savePDFToHistory(feedbackResult);
                    console.log('✅ Coaching feedback PDF saved to history (fallback)');
                    
                    return feedbackResult;
                } else {
                    console.error('❌ No report generator available');
                    return null;
                }
            } else {
                console.error('❌ Elite coaching engine not available');
                return null;
            }

        } catch (error) {
            console.error('❌ Failed to generate coaching feedback:', error);
            return null;
        }
    }

    async downloadTranscript() {
        try {
            if (!this.sessionId) {
                console.error('❌ No session ID available for transcript download');
                return;
            }

            const response = await fetch(`https://pitchperfect2-api-373812504656.asia-southeast1.run.app/api/transcripts/download/${this.sessionId}`);
            const transcriptData = await response.json();

            // Download as JSON file
            const blob = new Blob([JSON.stringify(transcriptData, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `transcript_${this.sessionId}_${new Date().toISOString().split('T')[0]}.json`;
            a.click();
            URL.revokeObjectURL(url);

            console.log('📥 Transcript downloaded:', transcriptData);
            return transcriptData;

        } catch (error) {
            console.error('❌ Failed to download transcript:', error);
        }
    }

    // ==== END TRANSCRIPTION SYSTEM METHODS ====

    async endCoachingMode() {
        try {
            console.log('🎓 Ending coaching mode - preparing for transcript download...');

            // End transcription session (this saves the transcript)
            await this.endTranscriptionSession();

            // Stop the WebSocket connection (coaching is complete)
            if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
                this.websocket.close();
                this.isConnected = false;
            }

            // Reset conversation state for potential new session
            this.currentPersonaMode = 'sales';
            this.conversationPhase = 'active';
            this.endingDetected = false;
            // ✅ CLEAR SESSION RESUMPTION data when coaching ends
            this.sessionHandle = null;
            this.connectionCount = 0;
            this.clearReconnectionTimer();

            console.log('✅ Coaching ended - transcript ready for download');

        } catch (error) {
            console.error('❌ Failed to end coaching mode:', error);
        }
    }

    // ==== SESSION RESUMPTION METHODS FOR 30+ MINUTE CONVERSATIONS ====
    
    startReconnectionTimer() {
        // Clear any existing timer
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            console.log('🔄 Cleared existing reconnection timer');
        }
        
        // ✅ PLAN-BASED TIMER: Check if user is authenticated (logged in)
        const isAuthenticated = this.checkUserAuthentication();
        
        if (isAuthenticated) {
            // 🔐 AUTHENTICATED USERS (Free Trial + Paid): 30-minute sessions with proactive reconnection
            const reconnectDelay = 29 * 60 * 1000; // 29 minutes - reconnect before 30-min limit
            
            this.reconnectTimer = setTimeout(() => {
                console.log('⏰ 29-minute proactive reconnection triggered (authenticated user)...');
                if (this.sessionHandle && !this.isReconnecting) {
                    this.initiateGracefulReconnect();
                }
            }, reconnectDelay);
            
            console.log('🔐 Authenticated user: Reconnection timer set for 29 minutes (30-min sessions)');
        } else {
            // 🌐 FREE/UNAUTHENTICATED USERS: Standard 10-minute limit, no reconnection
            console.log('🌐 Unauthenticated user: Using standard 10-minute session limit (no auto-reconnection)');
            // No timer set - let natural 10-minute Gemini limit end the session
        }
    }
    
    // Check if user is authenticated (logged in to the app)
    checkUserAuthentication() {
        try {
            // Method 1: Check Firebase Auth state
            if (window.firebase?.auth && window.firebase.auth().currentUser) {
                console.log('✅ User authenticated via Firebase Auth');
                return true;
            }
            
            // Method 2: Check localStorage for auth tokens
            const authToken = localStorage.getItem('authToken');
            const userToken = localStorage.getItem('userToken');
            if (authToken || userToken) {
                console.log('✅ User authenticated via stored tokens');
                return true;
            }
            
            // Method 3: Check if user is on authenticated pages
            const currentPath = window.location.pathname;
            const authenticatedPaths = ['/dashboard', '/app', '/personas', '/practice'];
            if (authenticatedPaths.some(path => currentPath.includes(path))) {
                console.log('✅ User on authenticated page path');
                return true;
            }
            
            // Method 4: Check for any sign of user session
            const sessionData = localStorage.getItem('userSession');
            if (sessionData) {
                console.log('✅ User session found in localStorage');
                return true;
            }
            
            console.log('ℹ️ User appears to be unauthenticated/free user');
            return false;
            
        } catch (error) {
            console.error('❌ Error checking authentication status:', error);
            // Default to unauthenticated on error
            return false;
        }
    }
    
    async initiateGracefulReconnect() {
        if (this.isReconnecting) {
            console.log('🔄 Reconnection already in progress, skipping...');
            return;
        }
        
        try {
            this.isReconnecting = true;
            console.log('🔄 Starting graceful reconnection to extend session...');
            
            // 🧹 CRITICAL STATE CLEANUP: Clear all timers and state before reconnecting
            if (this.endOfSpeechTimer) {
                clearTimeout(this.endOfSpeechTimer);
                this.endOfSpeechTimer = null;
                console.log('🔄 Cleared end-of-speech timer for reconnection');
            }
            
            // Reset VAD state for clean reconnection
            this.audioChunkHistory = [];
            this.isNoiseFloorCalibrated = false;
            this.noiseFloor = 0;
            
            // Start audio buffering to capture any ongoing conversation
            this.startAudioBuffering();
            
            // Store current session state INCLUDING transcription session
            const currentPersonaId = this.currentPersonaId;
            const currentSession = this.currentSession;
            const currentTranscriptionSessionId = this.sessionId; // Preserve transcription session
            
            // Close current WebSocket gracefully (disable callbacks first)
            if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
                // Temporarily disable onclose to prevent triggering session end
                this.websocket.onclose = null;
                this.websocket.close();
            }
            
            // Wait a moment for clean closure
            await new Promise(resolve => setTimeout(resolve, 100));
            
            // Reconnect with session handle
            if (this.sessionHandle) {
                console.log('🔗 Reconnecting with session handle:', this.sessionHandle.substring(0, 20) + '...');
            } else {
                console.log('🔗 Reconnecting without session handle (first connection or no handle received)');
            }
            await this.connectToGeminiLive();
            
            // Resume conversation with stored session data
            if (currentPersonaId && currentSession) {
                await this.resumeConversationSession(currentPersonaId, currentSession);
            }
            
            // ✅ CRITICAL: Restore transcription session after reconnection
            if (currentTranscriptionSessionId) {
                this.sessionId = currentTranscriptionSessionId;
                console.log('✅ Transcription session restored:', this.sessionId);
            } else if (!this.sessionId) {
                console.log('⚠️ No transcription session to restore - starting new one...');
                await this.startTranscriptionSession();
            }
            
            // Stop audio buffering and resume normal operation
            this.stopAudioBuffering();
            
            console.log('✅ Graceful reconnection completed successfully');
            
        } catch (error) {
            console.error('❌ Failed to reconnect gracefully:', error);
            // Fall back to normal session end if reconnection fails
            if (this.onSessionEnd) {
                this.onSessionEnd();
            }
        } finally {
            this.isReconnecting = false;
        }
    }
    
    async resumeConversationSession(personaId, sessionData) {
        try {
            console.log('🔄 Resuming conversation session for:', personaId);
            
            // ✅ VOICE PERSISTENCE FIX: Include voice configuration during resumption
            // Get the original voice config to maintain consistency
            let voiceConfig = sessionData?.voiceConfig;
            
            // If no stored voice config, retrieve it again
            if (!voiceConfig) {
                if (personaId === 'custom') {
                    const practiceModal = window.practiceModal;
                    if (practiceModal?.currentPersonaData?.voiceConfig) {
                        voiceConfig = practiceModal.currentPersonaData.voiceConfig;
                    }
                } else {
                    try {
                        voiceConfig = await window.personaRegistry.getVoiceConfig(personaId);
                    } catch (error) {
                        console.warn(`Failed to get voice config for ${personaId}:`, error);
                        voiceConfig = { sales: "Puck", coach: "Puck" }; // Fallback only if retrieval fails
                    }
                }
            }
            
            // Validate voice config to prevent crashes
            voiceConfig = this.validateAndFixVoiceConfig(voiceConfig);
            
            // Get current voice (sales mode during conversation, coach mode during coaching)
            const currentVoice = this.currentPersonaMode === 'coach' 
                ? voiceConfig.coach || voiceConfig.sales || 'Puck'
                : voiceConfig.sales || 'Puck';
                
            console.log(`🎤 VOICE PERSISTENCE: Resuming with ${currentVoice} (mode: ${this.currentPersonaMode})`);
            
            // Enhanced resume message with voice configuration
            const resumeMessage = {
                setup: {
                    model: "models/gemini-live-2.5-flash-preview",
                    generation_config: {
                        response_modalities: ["audio"],
                        speech_config: {
                            voice_config: {
                                prebuilt_voice_config: {
                                    voice_name: currentVoice
                                }
                            },
                            language_code: "en-US"
                        }
                    },
                    // Enable transcription for continued logging
                    input_audio_transcription: {},
                    output_audio_transcription: {},
                    session_resumption: {
                        handle: this.sessionHandle
                    }
                }
            };
            
            console.log('📤 ENHANCED RESUME MESSAGE (with voice persistence):', JSON.stringify(resumeMessage, null, 2));
            this.websocket.send(JSON.stringify(resumeMessage));
            
            // Update stored voice config in current session
            if (this.currentSession) {
                this.currentSession.voiceConfig = voiceConfig;
            }
            
            // Restart the reconnection timer
            this.startReconnectionTimer();
            this.connectionCount++;
            
            console.log(`✅ Session resumed with ${currentVoice} voice, connection count:`, this.connectionCount);
            
        } catch (error) {
            console.error('❌ Failed to resume conversation session:', error);
            throw error;
        }
    }
    
    startAudioBuffering() {
        console.log('🎵 Starting audio buffering for seamless transition');
        this.audioBufferManager.isBuffering = true;
        this.audioBufferManager.inputBuffer = [];
        this.audioBufferManager.outputBuffer = [];
    }
    
    stopAudioBuffering() {
        console.log('🎵 Stopping audio buffering, resuming normal operation');
        this.audioBufferManager.isBuffering = false;
        
        // Replay any buffered audio if needed
        if (this.audioBufferManager.inputBuffer.length > 0) {
            console.log('🔄 Replaying buffered input audio');
            for (const audioData of this.audioBufferManager.inputBuffer) {
                if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
                    this.websocket.send(audioData);
                }
            }
        }
        
        // Clear buffers
        this.audioBufferManager.inputBuffer = [];
        this.audioBufferManager.outputBuffer = [];
    }
    
    // Clear timers when session ends
    clearReconnectionTimer() {
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
            console.log('🔄 Reconnection timer cleared');
        }
    }
    
    // Simple speech detection based on audio amplitude
    // 🔊 ENHANCED VAD: Robust Voice Activity Detection with hysteresis and noise floor calibration
    detectUserSpeech(int16ArrayBuffer) {
        try {
            const startTime = performance.now();
            const samples = new Int16Array(int16ArrayBuffer);
            const sum = samples.reduce((acc, sample) => acc + Math.abs(sample), 0);
            const average = sum / samples.length;
            
            // 📊 VAD PERFORMANCE MONITORING: Track metrics for optimization
            this.vadMetrics.totalChunks++;
            
            // Detect environment changes (microphone, background noise, etc.)
            const environmentHash = this.computeEnvironmentHash(average, samples);
            if (this.vadMetrics.lastEnvironmentHash && 
                this.vadMetrics.lastEnvironmentHash !== environmentHash) {
                this.vadMetrics.environmentChanges++;
            }
            this.vadMetrics.lastEnvironmentHash = environmentHash;
            
            // 📊 NOISE FLOOR CALIBRATION: Auto-adjust threshold based on environment
            if (!this.isNoiseFloorCalibrated) {
                this.calibrateNoiseFloor(average);
            }
            
            // Use legacy detection if enhanced features are disabled
            if (!this.features.robustVAD) {
                const result = this.detectUserSpeechLegacy(average);
                this.updateVADMetricsLegacy(result, average);
                return result;
            }
            
            // 🎯 ROBUST VAD: Determine if this chunk indicates speech
            const isLoudEnough = average > this.speechThreshold;
            
            // 📈 HYSTERESIS: Build history for smoother state transitions
            this.audioChunkHistory.push(isLoudEnough);
            if (this.audioChunkHistory.length > this.VAD_HISTORY_LENGTH) {
                this.audioChunkHistory.shift();
            }
            
            // Count speaking vs silent chunks in recent history
            const speakingChunks = this.audioChunkHistory.filter(Boolean).length;
            const wasSpeaking = this.isUserSpeaking;
            
            // 🔄 HYSTERESIS LOGIC: Different thresholds for starting vs continuing speech
            if (this.isUserSpeaking) {
                // Currently speaking - need evidence of silence to stop
                this.isUserSpeaking = speakingChunks >= 1; // Easier to continue speaking
            } else {
                // Currently silent - need stronger evidence to start speaking
                this.isUserSpeaking = speakingChunks >= 2; // Harder to start speaking
            }
            
            // 📊 UPDATE PERFORMANCE METRICS: Track VAD accuracy and performance
            this.updateVADMetrics(this.isUserSpeaking, wasSpeaking, average, isLoudEnough);
            
            // 📝 LOG STATE CHANGES: Track transitions for debugging
            if (this.isUserSpeaking !== wasSpeaking) {
                const stateChange = this.isUserSpeaking ? 'Speaking' : 'Silent';
                console.log(`🎤 VAD State Change: ${stateChange} (avg: ${Math.round(average)}, threshold: ${this.speechThreshold}, chunks: ${speakingChunks}/${this.VAD_HISTORY_LENGTH})`);
                this.lastSpeechActivity = Date.now();
                
                // Debug logging when enabled
                if (this.features.debugMode) {
                    console.log('🔍 VAD Debug:', {
                        average: Math.round(average),
                        threshold: this.speechThreshold,
                        noiseFloor: this.noiseFloor,
                        speakingChunks,
                        history: this.audioChunkHistory,
                        transition: `${wasSpeaking ? 'Speaking' : 'Silent'} → ${stateChange}`,
                        processingTime: `${(performance.now() - startTime).toFixed(2)}ms`,
                        metrics: this.getVADMetricsSummary()
                    });
                }
            }
            
            return this.isUserSpeaking;
        } catch (error) {
            console.error('❌ Error in enhanced VAD:', error);
            return false;
        }
    }

    // 📊 NOISE FLOOR CALIBRATION: Establish baseline audio level
    calibrateNoiseFloor(currentAverage) {
        try {
            const CALIBRATION_SAMPLES = 20; // About 1 second of audio
            
            if (!this.noiseFloorSamples) {
                this.noiseFloorSamples = [];
            }
            
            this.noiseFloorSamples.push(currentAverage);
            
            // Prevent unbounded growth (safety limit)
            if (this.noiseFloorSamples.length > CALIBRATION_SAMPLES * 2) {
                console.warn('⚠️ Noise floor calibration taking too long, forcing completion');
                this.noiseFloorSamples = this.noiseFloorSamples.slice(-CALIBRATION_SAMPLES);
            }
            
            if (this.noiseFloorSamples.length >= CALIBRATION_SAMPLES) {
                // Calculate noise floor as the median of the samples
                const sorted = [...this.noiseFloorSamples].sort((a, b) => a - b);
                this.noiseFloor = sorted[Math.floor(sorted.length / 2)];
                
                // Adaptive threshold with bounds checking
                const adaptiveThreshold = this.noiseFloor + 3000;
                this.speechThreshold = Math.max(1000, Math.min(10000, adaptiveThreshold));
                
                this.isNoiseFloorCalibrated = true;
                
                // 📊 Track calibration events
                this.vadMetrics.noiseFloorCalibrations++;
                this.vadMetrics.averageNoiseFloor = this.noiseFloor;
                
                console.log(`🔧 VAD Calibrated - Noise Floor: ${Math.round(this.noiseFloor)}, Speech Threshold: ${this.speechThreshold}`);
                
                // Clean up calibration data properly
                delete this.noiseFloorSamples;
            }
        } catch (error) {
            console.error('❌ Error in noise floor calibration:', error);
            // Fallback to default threshold
            this.speechThreshold = 4000;
            this.isNoiseFloorCalibrated = true;
            delete this.noiseFloorSamples;
        }
    }

    // 🔙 LEGACY VAD: Fallback to original simple detection
    detectUserSpeechLegacy(average) {
        const wasSpeaking = this.isUserSpeaking;
        this.isUserSpeaking = average > this.speechThreshold;
        
        if (this.isUserSpeaking !== wasSpeaking) {
            console.log('🎤 Legacy VAD - User speech state changed:', this.isUserSpeaking ? 'Speaking' : 'Silent');
            this.lastSpeechActivity = Date.now();
        }
        
        return this.isUserSpeaking;
    }
    
    // 📊 VAD PERFORMANCE MONITORING: Helper functions for tracking metrics
    computeEnvironmentHash(average, samples) {
        try {
            // Simple hash based on audio characteristics
            if (!samples || samples.length === 0) return 0;
            
            const variance = this.computeVariance(samples);
            
            // Safe peak calculation for large arrays
            let peak = 0;
            for (let i = 0; i < Math.min(samples.length, 1000); i++) {
                peak = Math.max(peak, Math.abs(samples[i]));
            }
            
            const peakRatio = peak / (average || 1);
            return Math.round(average / 100) * 1000 + Math.round(variance / 100) * 10 + Math.round(peakRatio);
        } catch (error) {
            console.warn('⚠️ Environment hash computation failed:', error);
            return 0;
        }
    }
    
    computeVariance(samples) {
        try {
            if (!samples || samples.length === 0) return 0;
            const mean = samples.reduce((sum, val) => sum + Math.abs(val), 0) / samples.length;
            const variance = samples.reduce((sum, val) => sum + Math.pow(Math.abs(val) - mean, 2), 0) / samples.length;
            return Math.sqrt(variance);
        } catch (error) {
            console.warn('⚠️ Variance computation failed:', error);
            return 0;
        }
    }
    
    updateVADMetrics(isSpeaking, wasSpeaking, average, isLoudEnough) {
        // Track speech detection
        if (isSpeaking) {
            this.vadMetrics.speechDetected++;
        } else {
            this.vadMetrics.silenceDetected++;
        }
        
        // Track potential false positives (loud but not actual speech)
        if (isLoudEnough && !isSpeaking) {
            this.vadMetrics.falsePositives++;
        }
        
        // Update rolling average noise floor
        if (this.vadMetrics.totalChunks > 0) {
            const weight = 0.01; // Small weight for rolling average
            this.vadMetrics.averageNoiseFloor = 
                (this.vadMetrics.averageNoiseFloor * (1 - weight)) + (this.noiseFloor * weight);
        }
    }
    
    updateVADMetricsLegacy(isSpeaking, average) {
        // Track legacy VAD performance for comparison
        if (isSpeaking) {
            this.vadMetrics.speechDetected++;
        } else {
            this.vadMetrics.silenceDetected++;
        }
    }
    
    getVADMetricsSummary() {
        if (this.vadMetrics.totalChunks === 0) return null;
        
        return {
            totalChunks: this.vadMetrics.totalChunks,
            speechRatio: (this.vadMetrics.speechDetected / this.vadMetrics.totalChunks * 100).toFixed(1) + '%',
            falsePositiveRate: (this.vadMetrics.falsePositives / this.vadMetrics.totalChunks * 100).toFixed(1) + '%',
            environmentChanges: this.vadMetrics.environmentChanges,
            noiseFloorCalibrations: this.vadMetrics.noiseFloorCalibrations,
            currentNoiseFloor: Math.round(this.noiseFloor),
            currentThreshold: this.speechThreshold,
            averageNoiseFloor: Math.round(this.vadMetrics.averageNoiseFloor),
            vadMode: this.features.robustVAD ? 'Enhanced' : 'Legacy'
        };
    }
    
    // Add to TurnManagementConfig integration
    getConversationMetrics() {
        const sessionDuration = this.sessionStartTime ? Date.now() - this.sessionStartTime : 0;
        
        // Safe WebSocket health check
        let webSocketHealth = null;
        try {
            webSocketHealth = this.getWebSocketHealth();
        } catch (error) {
            console.warn('⚠️ WebSocket health check failed:', error.message);
            webSocketHealth = { error: 'Health check unavailable' };
        }
        
        return {
            sessionDuration: Math.round(sessionDuration / 1000) + 's',
            averageTurnDelay: this.endOfSpeechDelay + 'ms',
            vadMetrics: this.getVADMetricsSummary(),
            reconnections: this.connectionCount - 1,
            features: this.features,
            webSocketHealth
        };
    }

    // ==== WEBSOCKET HEALTH MONITORING METHODS ====
    
    startWebSocketHealthMonitoring() {
        console.log('❤️ Starting WebSocket health monitoring');
        
        // Clear any existing health monitoring
        this.stopWebSocketHealthMonitoring();
        
        // Ping every 30 seconds to check connection health
        this.healthMonitorInterval = setInterval(() => {
            this.checkWebSocketHealth();
        }, 30000); // 30 seconds
        
        console.log('✅ WebSocket health monitoring started');
    }
    
    stopWebSocketHealthMonitoring() {
        if (this.healthMonitorInterval) {
            clearInterval(this.healthMonitorInterval);
            this.healthMonitorInterval = null;
            console.log('🛑 WebSocket health monitoring stopped');
        }
    }
    
    checkWebSocketHealth() {
        try {
            if (!this.websocket || this.websocket.readyState !== WebSocket.OPEN) {
                console.warn('⚠️ WebSocket health check: Connection not open');
                this.websocketHealthCheck.isHealthy = false;
                return;
            }
            
            // Simple health check - just verify connection is still active
            // Gemini Live doesn't support ping/pong, so we rely on message activity
            const timeSinceLastActivity = Date.now() - (this.websocketHealthCheck.lastActivity || this.connectionStartTime);
            
            if (timeSinceLastActivity > 120000) { // 2 minutes of silence
                console.warn('⚠️ WebSocket health check: No activity for 2+ minutes');
                this.websocketHealthCheck.isHealthy = false;
                this.websocketHealthCheck.missedPongs++;
                
                // If connection seems stale and we have session handle, proactively reconnect
                if (this.websocketHealthCheck.missedPongs >= 3 && this.sessionHandle && !this.isReconnecting) {
                    console.log('🔄 Health check failed - initiating proactive reconnection...');
                    this.initiateGracefulReconnect();
                }
            } else {
                this.websocketHealthCheck.isHealthy = true;
                this.websocketHealthCheck.missedPongs = 0;
            }
            
            // Update last activity timestamp on health check
            this.websocketHealthCheck.lastActivity = Date.now();
            
        } catch (error) {
            console.error('❌ Error in WebSocket health check:', error);
            this.websocketHealthCheck.isHealthy = false;
        }
    }
    
    // Get current WebSocket health status
    getWebSocketHealth() {
        return {
            isConnected: this.isConnected,
            isHealthy: this.websocketHealthCheck?.isHealthy || false,
            connectionDuration: this.connectionStartTime ? Date.now() - this.connectionStartTime : 0,
            missedPongs: this.websocketHealthCheck?.missedPongs || 0,
            sessionHandle: !!this.sessionHandle,
            connectionCount: this.connectionCount,
            features: this.features,
            vadCalibrated: this.isNoiseFloorCalibrated,
            speechThreshold: this.speechThreshold,
            endOfSpeechTimer: !!this.endOfSpeechTimer
        };
    }

    // 🧪 API VALIDATION: Test turn completion message structure
    async validateTurnCompleteAPI() {
        if (!this.websocket || this.websocket.readyState !== WebSocket.OPEN) {
            console.warn('⚠️ Cannot validate API - WebSocket not connected');
            return false;
        }

        return new Promise((resolve) => {
            const originalReadyState = this.websocket.readyState;
            let validationTimeout;
            
            try {
                // Match the updated snake_case structure
                const testMessage = {
                    client_content: {
                        turns: [],
                        turn_complete: true
                    }
                };
                
                console.log('🧪 Testing turn_complete API structure...');
                
                // Monitor for errors before sending
                const originalOnError = this.websocket.onerror;
                let errorOccurred = false;
                
                this.websocket.onerror = (error) => {
                    errorOccurred = true;
                    console.error('❌ WebSocket error during validation:', error);
                    if (originalOnError) originalOnError(error);
                };
                
                this.websocket.send(JSON.stringify(testMessage));
                
                // Wait 2 seconds to check if connection is still alive
                validationTimeout = setTimeout(() => {
                    // Restore original error handler
                    this.websocket.onerror = originalOnError;
                    
                    const isStillConnected = this.websocket.readyState === WebSocket.OPEN;
                    if (isStillConnected && !errorOccurred) {
                        console.log('✅ turn_complete API structure validated - connection stable');
                        resolve(true);
                    } else {
                        console.error('❌ turn_complete API validation failed', {
                            connected: isStillConnected,
                            errorOccurred
                        });
                        resolve(false);
                    }
                }, 2000);
                
            } catch (error) {
                console.error('❌ turn_complete API validation error:', error);
                if (validationTimeout) clearTimeout(validationTimeout);
                resolve(false);
            }
        });
    }

    // ⚙️ FEATURE CONFIGURATION: Update feature flags at runtime
    updateFeatures(newFeatures) {
        const previousFeatures = { ...this.features };
        this.features = { ...this.features, ...newFeatures };
        
        console.log('🔧 Feature configuration updated:', {
            previous: previousFeatures,
            current: this.features,
            changes: newFeatures
        });

        // Update VAD history length if robustVAD changed
        if (newFeatures.hasOwnProperty('robustVAD')) {
            this.VAD_HISTORY_LENGTH = this.features.robustVAD ? 5 : 1;
            this.audioChunkHistory = []; // Reset history when changing modes
            console.log(`🔊 VAD mode changed: ${this.features.robustVAD ? 'Robust' : 'Legacy'} (history length: ${this.VAD_HISTORY_LENGTH})`);
        }

        // Recalibrate noise floor if switching to enhanced VAD
        if (newFeatures.robustVAD && !previousFeatures.robustVAD) {
            this.isNoiseFloorCalibrated = false;
            this.noiseFloor = 0;
            console.log('🔧 Resetting noise floor calibration for enhanced VAD');
        }
    }

    // 📊 METRICS COLLECTION: Get conversation quality metrics
    getConversationMetrics() {
        const now = Date.now();
        const sessionDuration = this.sessionStartTime ? now - this.sessionStartTime : 0;
        
        return {
            sessionDuration,
            averageTurnDelay: this.endOfSpeechDelay,
            vadMode: this.features.robustVAD ? 'robust' : 'legacy',
            enhancedTurnManagement: this.features.enhancedTurnManagement,
            dynamicDelays: this.features.dynamicDelays,
            noiseFloor: this.noiseFloor,
            speechThreshold: this.speechThreshold,
            conversationPairs: this.conversationPairs.length,
            reconnections: this.connectionCount - 1
        };
    }

    // 🎛️ TUNING INTERFACE: Adjust turn management parameters
    tuneTurnManagement(parameters) {
        if (parameters.endOfSpeechDelay) {
            this.endOfSpeechDelay = Math.max(300, Math.min(2000, parameters.endOfSpeechDelay));
            console.log(`🎯 End-of-speech delay updated: ${this.endOfSpeechDelay}ms`);
        }

        if (parameters.speechThreshold) {
            this.speechThreshold = Math.max(1000, Math.min(10000, parameters.speechThreshold));
            console.log(`🔊 Speech threshold updated: ${this.speechThreshold}`);
        }

        if (parameters.vadHistoryLength) {
            this.VAD_HISTORY_LENGTH = Math.max(1, Math.min(10, parameters.vadHistoryLength));
            this.audioChunkHistory = []; // Reset history
            console.log(`📈 VAD history length updated: ${this.VAD_HISTORY_LENGTH}`);
        }
    }

    // 🛡️ CIRCUIT BREAKER: Handle errors in enhanced features
    handleEnhancedFeatureError(error) {
        const now = Date.now();
        
        // Reset error count if it's been more than 5 minutes since last error
        if (now - this.lastErrorTime > 300000) {
            this.errorCount = 0;
        }
        
        this.errorCount++;
        this.lastErrorTime = now;
        
        console.warn(`⚠️ Enhanced feature error #${this.errorCount}:`, error.message);
        
        // Auto-disable features if too many errors
        if (this.errorCount >= this.maxErrors) {
            console.error(`🚨 CIRCUIT BREAKER ACTIVATED: Disabling enhanced features after ${this.errorCount} errors`);
            
            const originalFeatures = { ...this.features };
            this.features.enhancedTurnManagement = false;
            this.features.dynamicDelays = false;
            this.features.robustVAD = false;
            
            console.log('🔙 Falling back to legacy mode:', {
                original: originalFeatures,
                fallback: this.features
            });
            
            // Clean up any active enhanced features
            if (this.endOfSpeechTimer) {
                clearTimeout(this.endOfSpeechTimer);
                this.endOfSpeechTimer = null;
            }
            
            this.audioChunkHistory = [];
            this.VAD_HISTORY_LENGTH = 1;
        }
    }
    
    // 💾 10X ENGINEER FIX: Store transcript in proper Firebase structure at user level
    async storeTranscriptInFirebase(finalMetrics) {
        try {
            // Check if user is authenticated
            if (!firebase.auth || !firebase.auth().currentUser) {
                console.log('🔒 User not authenticated, skipping Firebase transcript storage');
                return;
            }
            
            const user = firebase.auth().currentUser;
            const sessionId = this.sessionId || this.generateSessionId();
            const timestamp = Date.now();
            
            // Create structured transcript data
            const transcriptData = {
                sessionId: sessionId,
                userId: user.uid,
                userEmail: user.email,
                
                // Session metadata
                sessionInfo: {
                    startTime: this.sessionStartTime,
                    endTime: timestamp,
                    duration: finalMetrics.sessionDuration,
                    personaId: this.currentPersonaId,
                    personaMode: this.currentPersonaMode
                },
                
                // Complete conversation data
                conversationData: {
                    transcript: finalMetrics.transcript || this.conversationTranscript,
                    conversationPairs: finalMetrics.conversationPairs || this.conversationPairs,
                    userTranscripts: finalMetrics.userTranscripts || this.userTranscripts,
                    aiTranscripts: finalMetrics.aiTranscripts || this.aiTranscripts
                },
                
                // Performance metrics
                metrics: {
                    sessionDuration: finalMetrics.sessionDuration,
                    userSpeakingTime: finalMetrics.userSpeakingTime || 0,
                    aiSpeakingTime: finalMetrics.aiSpeakingTime || 0,
                    turnCount: (finalMetrics.transcript || []).length,
                    realTalkTimeRatio: finalMetrics.realTalkTimeRatio
                },
                
                // Metadata
                createdAt: firebase.firestore.FieldValue.serverTimestamp(),
                version: '2.0',
                source: 'GeminiLiveManager'
            };
            
            // Store in users/{userId}/transcripts/{sessionId}
            await firebase.firestore()
                .collection('users')
                .doc(user.uid)
                .collection('transcripts')
                .doc(sessionId)
                .set(transcriptData);
            
            console.log('💾 ✅ Transcript stored in Firebase:', `users/${user.uid}/transcripts/${sessionId}`);
            
            console.log('📊 ✅ Transcript stored successfully - app_sessions unchanged');
            
        } catch (error) {
            console.error('❌ CRITICAL: Failed to store transcript in Firebase:', error);
            console.error('❌ CRITICAL: This likely means Firestore security rules are blocking the write operation');
            
            // During development, re-throw the error to make it obvious
            if (window.location.hostname === 'localhost') {
                throw error;
            }
        }
    }
    
    // Generate session ID if not already present
    generateSessionId() {
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

}

// Export for use in other modules
window.GeminiLiveManager = GeminiLiveManager;

// Debug: Confirm GeminiLiveManager is loaded
console.log('✅ GeminiLiveManager class defined and exported to window');
console.log('🔍 window.GeminiLiveManager available:', !!window.GeminiLiveManager);

// Debug: Test Gemini connection when manager is loaded
console.log('📦 Gemini Live Manager script loaded');