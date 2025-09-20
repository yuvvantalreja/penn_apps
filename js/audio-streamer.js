// Audio Streamer for JARVIS Voice Playback
// Handles real-time audio playback from Gemini Live

class AudioStreamer {
    constructor(audioContext, onPlaybackComplete = null) {
        this.audioContext = audioContext;
        this.onPlaybackComplete = onPlaybackComplete;
        
        // Audio buffer management
        this.audioQueue = [];
        this.isPlaying = false;
        this.currentSource = null;
        
        // PCM16 streaming
        this.sampleRate = 24000; // Gemini Live uses 24kHz
        this.channels = 1;
        
        console.log('🎵 AudioStreamer initialized for JARVIS voice playback');
    }
    
    addPCM16(pcm16Data) {
        try {
            // Convert PCM16 data to Float32Array for Web Audio API
            const samples = new Float32Array(pcm16Data.length / 2);
            const dataView = new DataView(pcm16Data.buffer);
            
            for (let i = 0; i < samples.length; i++) {
                const int16 = dataView.getInt16(i * 2, true); // little endian
                samples[i] = int16 / 32768.0; // Convert to float [-1, 1]
            }
            
            // Create audio buffer
            const audioBuffer = this.audioContext.createBuffer(
                this.channels,
                samples.length,
                this.sampleRate
            );
            
            audioBuffer.getChannelData(0).set(samples);
            
            // Add to queue
            this.audioQueue.push(audioBuffer);
            
            // Start playing if not already playing
            if (!this.isPlaying) {
                this.playNext();
            }
            
        } catch (error) {
            console.error('❌ Error adding PCM16 data to AudioStreamer:', error);
        }
    }
    
    playNext() {
        if (this.audioQueue.length === 0) {
            this.isPlaying = false;
            if (this.onPlaybackComplete) {
                this.onPlaybackComplete();
            }
            return;
        }
        
        this.isPlaying = true;
        const audioBuffer = this.audioQueue.shift();
        
        // Create and play audio source
        this.currentSource = this.audioContext.createBufferSource();
        this.currentSource.buffer = audioBuffer;
        this.currentSource.connect(this.audioContext.destination);
        
        this.currentSource.onended = () => {
            this.currentSource = null;
            // Play next buffer in queue
            setTimeout(() => this.playNext(), 10); // Small delay to prevent audio gaps
        };
        
        this.currentSource.start();
    }
    
    resume() {
        // Resume audio context if suspended (required by some browsers)
        if (this.audioContext.state === 'suspended') {
            this.audioContext.resume().then(() => {
                console.log('🎵 AudioContext resumed for JARVIS playback');
            });
        }
    }
    
    stop() {
        // Stop current playback and clear queue
        if (this.currentSource) {
            this.currentSource.stop();
            this.currentSource = null;
        }
        
        this.audioQueue = [];
        this.isPlaying = false;
        
        console.log('🔇 AudioStreamer stopped');
    }
    
    getQueueLength() {
        return this.audioQueue.length;
    }
    
    isCurrentlyPlaying() {
        return this.isPlaying;
    }
}

// Make AudioStreamer available globally
window.AudioStreamer = AudioStreamer;
