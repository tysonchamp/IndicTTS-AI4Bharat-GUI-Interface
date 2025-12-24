#!/bin/bash

echo "Installing system dependencies..."
# Ensure git is installed for the pip install git+...
# sudo apt-get install git -y # Assuming user has permissions or git is installed. 
# Since I cannot use sudo reliably without knowing user perms, I'll rely on conda/existing env mostly, 
# but pip commands are safe.

echo "Installing Python libraries..."
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130
# Adjust cuda version if needed, or just let it find the best one. 
# For simplicity and broad compatibility often standard pip install torch suffices if no specific cuda requirement is hardcoded, 
# but user mentioned "SDPA, torch.compile" which often benefits from newer torch.

pip install streamlit transformers soundfile numpy scipy

echo "Installing Parler TTS from GitHub..."
pip install git+https://github.com/huggingface/parler-tts.git

echo "Installation complete!"
echo "Run the app with: streamlit run app.py"
