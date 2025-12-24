
import streamlit as st
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import io
import numpy as np
import gc

# Page Config
st.set_page_config(page_title="IndicTTS Interface", layout="wide")

# Title and Description
st.title("IndicTTS Interactive Interface")
st.markdown("Generate high-quality speech in various Indian languages using the Parler TTS model.")

# --- Sidebar / Guidance ---
with st.sidebar:
    st.header("Guidance & Tips")
    
    st.subheader("Language & Speaker Availability")
    # Javascript-like object structure to Python dict for reference/display
    guidance_data = {
        "Assamese": {"Speakers": "Amit, Sita, Poonam, Rakesh", "Recommended": "Amit, Sita"},
        "Bengali": {"Speakers": "Arjun, Aditi, Tapan, Rashmi, Arnav, Riya", "Recommended": "Arjun, Aditi"},
        "Bodo": {"Speakers": "Bikram, Maya, Kalpana", "Recommended": "Bikram, Maya"},
        "Chhattisgarhi": {"Speakers": "Bhanu, Champa", "Recommended": "Bhanu, Champa"},
        "Dogri": {"Speakers": "Karan", "Recommended": "Karan"},
        "English": {"Speakers": "Thoma, Mary, Swapna, Dinesh, Meera, Jatin, Aakash, Sneha, Kabir, Tisha, Chingkhei, Thoiba, Priya, Tarun, Gauri, Nisha, Raghav, Kavya, Ravi, Vikas, Riya", "Recommended": "Thoma, Mary"},
        "Gujarati": {"Speakers": "Yash, Neha", "Recommended": "Yash, Neha"},
        "Hindi": {"Speakers": "Rohit, Divya, Aman, Rani", "Recommended": "Rohit, Divya"},
        "Kannada": {"Speakers": "Suresh, Anu, Chetan, Vidya", "Recommended": "Suresh, Anu"},
        "Malayalam": {"Speakers": "Anjali, Anju, Harish", "Recommended": "Anjali, Harish"},
        "Manipuri": {"Speakers": "Laishram, Ranjit", "Recommended": "Laishram, Ranjit"},
        "Marathi": {"Speakers": "Sanjay, Sunita, Nikhil, Radha, Varun, Isha", "Recommended": "Sanjay, Sunita"},
        "Nepali": {"Speakers": "Amrita", "Recommended": "Amrita"},
        "Odia": {"Speakers": "Manas, Debjani", "Recommended": "Manas, Debjani"},
        "Punjabi": {"Speakers": "Divjot, Gurpreet", "Recommended": "Divjot, Gurpreet"},
        "Sanskrit": {"Speakers": "Aryan", "Recommended": "Aryan"},
        "Tamil": {"Speakers": "Kavitha, Jaya", "Recommended": "Jaya"},
        "Telugu": {"Speakers": "Prakash, Lalitha, Kiran", "Recommended": "Prakash, Lalitha"},
    }
    
    # Display as a dataframe or simply iter
    # Creating a simple markdown table
    md_table = "| Language | Available Speakers | Recommended |\n|---|---|---|\n"
    for lang, details in guidance_data.items():
        md_table += f"| {lang} | {details['Speakers']} | {details['Recommended']} |\n"
    
    st.markdown(md_table)
    
    st.subheader("Optimization Tips")
    st.info("""
    - **Inference Speed**: Using `torch.compile`, batching, and streaming (where applicable).
    - **Quality**: Use "very clear audio" for best quality, "very noisy audio" for background noise.
    - **Prosody**: Use punctuation (commas) for breaks.
    - **Control**: Gender, rate, pitch, reverberation can be described in the prompt.
    """)

# --- Model Loading ---
@st.cache_resource
def load_model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Load model and tokenizers
    model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to(device)
    tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
    description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    
    # Optimization: torch.compile
    # Note: torch.compile can sometimes increase startup time significantly, 
    # but subsequent runs are faster.
    if torch.cuda.is_available():
        try:
             model = torch.compile(model)
        except Exception as e:
            st.warning(f"Could not apply torch.compile: {e}")
            
    return model, tokenizer, description_tokenizer, device

with st.spinner("Loading Model..."):
    model, tokenizer, description_tokenizer, device = load_model()

# --- Main UI ---

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Text")
    prompt_text = st.text_area("Enter text to synthesize:", value="अरे, तुम आज कैसे हो?", height=150)

with col2:
    st.subheader("Voice Description")
    
    # Presets
    presets = {
        "Custom": "",
        "Aditi - Expressive": "Aditi speaks with a slightly higher pitch in a close-sounding environment. Her voice is clear, with subtle emotional depth and a normal pace, all captured in high-quality recording.",
        "Sita - Rapid": "Sita speaks at a fast pace with a slightly low-pitched voice, captured clearly in a close-sounding environment with excellent recording quality.",
        "Tapan - Monotone": "Tapan speaks at a moderate pace with a slightly monotone tone. The recording is clear, with a close sound and only minimal ambient noise.",
        "Sunita - Happy": "Sunita speaks with a high pitch in a close environment. Her voice is clear, with slight dynamic changes, and the recording is of excellent quality.",
        "Karan - Positive": "Karan’s high-pitched, engaging voice is captured in a clear, close-sounding recording. His slightly slower delivery conveys a positive tone.",
        "Amrita - Flat": "Amrita speaks with a high pitch at a slow pace. Her voice is clear, with excellent recording quality and only moderate background noise.",
        "Aditi - Slow": "Aditi speaks slowly with a high pitch and expressive tone. The recording is clear, showcasing her energetic and emotive voice.",
        "Male - American Accent": "A young male speaker with a high-pitched American accent delivers speech at a slightly fast pace in a clear, close-sounding recording.",
        "Bikram - Urgent": "Bikram speaks with a higher pitch and fast pace, conveying urgency. The recording is clear and intimate, with great emotional depth.",
        "Anjali - Neutral": "Anjali speaks with a high pitch at a normal pace in a clear, close-sounding environment. Her neutral tone is captured with excellent audio quality."
    }
    
    selected_preset = st.selectbox("Choose a preset description (Optional):", list(presets.keys()))
    
    default_desc = presets[selected_preset] if selected_preset != "Custom" else ""
    # If the user switches back to custom, we leave it empty or whatever they typed? 
    # Streamlit re-runs on interaction, so if they select a preset, we want to update the text area.
    # However, text_area value is stateful. We'll use the preset to populate it if it changed.
    
    description_text = st.text_area("Describe the speaker and tone:", value=default_desc, height=150, help="e.g. 'A female speaker with a high pitch...'")


if st.button("Generate Audio", type="primary"):
    if not prompt_text or not description_text:
        st.error("Please provide both text and a description.")
    else:
        try:
            with st.spinner("Generating audio..."):
                with torch.no_grad():
                    description_input_ids = description_tokenizer(description_text, return_tensors="pt").to(device)
                    prompt_input_ids = tokenizer(prompt_text, return_tensors="pt").to(device)

                    generation = model.generate(
                        input_ids=description_input_ids.input_ids, 
                        attention_mask=description_input_ids.attention_mask, 
                        prompt_input_ids=prompt_input_ids.input_ids, 
                        prompt_attention_mask=prompt_input_ids.attention_mask
                    )
                    
                    audio_arr = generation.cpu().numpy().squeeze()
                
                # Write to buffer for Streamlit
                buffer = io.BytesIO()
                sf.write(buffer, audio_arr, model.config.sampling_rate, format='WAV')
                buffer.seek(0)
                
                # Memory Cleanup
                del description_input_ids
                del prompt_input_ids
                del generation
                del audio_arr
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                
                st.success("Generation Complete!")
                st.audio(buffer, format='audio/wav')
                
                st.download_button(
                    label="Download WAV",
                    data=buffer,
                    file_name="generated_audio.wav",
                    mime="audio/wav"
                )
                
        except Exception as e:
            st.error(f"Error during generation: {e}")
