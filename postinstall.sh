#!/bin/bash
echo "🔧 Installing system-level dependencies..."
sudo apt-get update
sudo apt-get install -y libgl1

echo "🐍 Installing Python dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt