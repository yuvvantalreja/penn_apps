#!/usr/bin/env python3
"""
Simple HTTP server to serve JARVIS integration files
Avoids CORS issues with file:// protocol
"""

import os
import http.server
import socketserver
from pathlib import Path

class JarvisHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def serve_jarvis_files(port=8080):
    """Serve JARVIS integration files on specified port"""
    
    # Change to the current directory
    os.chdir(Path(__file__).parent)
    
    print(f"🤖 Starting JARVIS file server...")
    print(f"📁 Serving files from: {os.getcwd()}")
    print(f"🌐 Server running at: http://localhost:{port}")
    print(f"🔗 Open JARVIS: http://localhost:{port}/jarvis_integration.html")
    print(f"📋 Available files:")
    
    # List relevant files
    files_to_show = [
        'jarvis_integration.html',
        'jarvis_voice_assistant.js', 
        'jarvis_ui.css',
        'js/audio-recording-worklet.js',
        'js/audio-streamer.js'
    ]
    
    for file in files_to_show:
        if os.path.exists(file):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} (missing)")
    
    print(f"\n🎯 Instructions:")
    print(f"1. Make sure JARVIS backend is running: python jarvis_backend.py")
    print(f"2. Open: http://localhost:{port}/jarvis_integration.html")
    print(f"3. Press 'J' or say 'Hey JARVIS' to activate")
    print(f"4. Press Ctrl+C to stop the server")
    print(f"\n" + "="*50)
    
    try:
        with socketserver.TCPServer(("", port), JarvisHTTPRequestHandler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print(f"\n🛑 JARVIS file server stopped")
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"❌ Port {port} is already in use. Try a different port:")
            print(f"   python serve_jarvis.py --port 8081")
        else:
            print(f"❌ Server error: {e}")

if __name__ == "__main__":
    import sys
    
    port = 8080
    
    # Check for custom port
    if len(sys.argv) > 1:
        try:
            if sys.argv[1] == '--port' and len(sys.argv) > 2:
                port = int(sys.argv[2])
            else:
                port = int(sys.argv[1])
        except ValueError:
            print("❌ Invalid port number. Using default port 8080.")
    
    serve_jarvis_files(port)
