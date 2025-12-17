#!/bin/bash
# Powerball Analyzer Runner

cd "$(dirname "$0")"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Install dependencies
pip install -q flask

echo ""
echo "=============================================="
echo "   POWERBALL MATHEMATICAL ANALYZER"
echo "=============================================="
echo ""

# Check for web or CLI mode
if [ "$1" == "cli" ]; then
    echo "Starting CLI mode..."
    python3 powerball_analyzer.py
else
    echo "Starting web server at http://localhost:5050"
    echo "Press Ctrl+C to stop"
    echo ""
    python3 app.py
fi
