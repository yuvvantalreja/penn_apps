#!/bin/bash

# Setup script to configure GEMINI API key
echo "🔑 Setting up GEMINI API key..."

# Create or update .env file
echo "GEMINI_API_KEY=AIzaSyBFehBZS4C1LW7QM6pI0U8crkQi7JUbsgQ" > .env

echo "✅ API key updated in .env file"
echo "🔧 To use the key, run: source .env"
echo "🚀 Or start JARVIS with: ./start_jarvis.sh"

# Make the key available in current session
export GEMINI_API_KEY=AIzaSyBFehBZS4C1LW7QM6pI0U8crkQi7JUbsgQ
echo "✅ API key set for current session"
