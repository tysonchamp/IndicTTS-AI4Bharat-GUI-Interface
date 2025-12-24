# IndicTTS - Interactive Indian Language Text-to-Speech

A user-friendly Steamlit web interface for generating high-quality speech in various Indian languages using the **Indic Parler TTS** model by AI4Bharat.

## Features

- **Multilingual Support**: Generate speech in over 15 Indian languages including Hindi, English, Bengali, Tamil, Telugu, Marathi, and more.
- **Natural Language Control**: Control speaker style, pitch, speed, and gender using descriptive text prompts (e.g., "A female speaker with a high pitch speaking fast").
- **Interactive GUI**: Easy-to-use web interface built with Streamlit.
- **Guidance & Presets**: Built-in guidance for available speakers per language and one-click style presets.
- **Optimized Performance**: Leverages `torch.compile` and CUDA (if available) for faster inference.

## Prerequisites

- Python 3.8+
- CUDA-enabled GPU (Recommended for faster generation)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/IndicTTS.git
    cd IndicTTS
    ```

2.  **Install dependencies:**

    ### Linux / macOS
    We provide a setup script to simplify installation:
    ```bash
    chmod +x setup.sh
    ./setup.sh
    ```

    ### Windows
    You can install the dependencies manually using PowerShell or Command Prompt:

    1.  **Install PyTorch**:
        Visit [pytorch.org](https://pytorch.org/get-started/locally/) to get the command for your specific CUDA version. For example (CUDA 12.4):
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
        ```
    2.  **Install Common Libraries**:
        ```bash
        pip install streamlit transformers soundfile numpy scipy
        ```
    3.  **Install Parler TTS**:
        *(Requires Git to be installed and available in your PATH)*
        ```bash
        pip install git+https://github.com/huggingface/parler-tts.git
        ```

    *Note: The Linux/macOS setup script attempts to install Pytorch Nightly with CUDA 13.0 support. Windows users should stick to the stable releases unless they specifically need nightly features.*

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

The interface will automatically open in your default browser at `http://localhost:8501`.

1.  **Select a Language**: Check the sidebar for recommended speakers for your desired language.
2.  **Enter Text**: Type the text you want to convert to speech.
3.  **Describe Voice**: Use the "Description" box to define the speaker's voice (e.g., "Amit - Slow, Deep voice") or select a **Preset** from the dropdown.
4.  **Generate**: Click "Generate Audio" and listen to or download the result.

## Supported Languages

Assamese, Bengali, Bodo, Chhattisgarhi, Dogri, English, Gujarati, Hindi, Kannada, Malayalam, Manipuri, Marathi, Nepali, Odia, Punjabi, Sanskrit, Tamil, Telugu.

## Models Used

This project uses the [Indic Parler TTS](https://huggingface.co/ai4bharat/indic-parler-tts) model developed by AI4Bharat.

## Credits

- **Model**: [AI4Bharat](https://ai4bharat.iitm.ac.in/)
- **Library**: [Parler TTS](https://github.com/huggingface/parler-tts)

## License

[MIT](LICENSE)
