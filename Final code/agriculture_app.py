import streamlit as st
import tempfile
import os
import time
from datetime import datetime
import io
import base64
from audio_recorder_streamlit import audio_recorder
import torch
import pygame
import requests
from PIL import Image
import google.generativeai as genai
import googlesearch
from gtts import gTTS
import speech_recognition as sr
from pydub import AudioSegment
from context_retriever import get_pdf_context, generate_weather_context

# --- Helper function for Gemini ---
def image_to_base64(image):
    """Converts a PIL image to a base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# --- Main AI Logic Class ---
class AgricultureVLM:
    def __init__(self):
        self.model = None
        self.agriculture_context = """
        You are 'Ctrl+Crop', an expert AI agricultural assistant. Your knowledge includes:
        - Crop health, disease identification, and treatment.
        - Pest detection and integrated pest management.
        - Soil conditions, nutrient management, and fertilizer recommendations.
        - Irrigation strategies and water management.
        - Harvest timing and post-harvest techniques.
        - Impact of weather on crops and mitigation strategies.
        - Sustainable and organic farming practices.
        - Details of government schemes for farmers in India.
        Your goal is to provide practical, actionable, and cost-effective advice suitable for farmers.
        """
        print("✅ Agriculture VLM Initialized (will configure with API key).")

    def configure_api(self, api_key):
        """Configures the Gemini API key and initializes the model."""
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            print("✅ Gemini API configured successfully!")
            return True
        except Exception as e:
            print(f"❌ Error configuring Gemini API: {str(e)}")
            st.error(f"Failed to configure Gemini API: {e}")
            return False

    def process_query(self, text_query, image=None, location=None, weather_api_key=None):
        """
        Processes a query using Gemini. Handles both text-only and multimodal (image+text) queries.
        """
        if not self.model:
            return "Error: The AI model is not configured. Please enter a valid Google API Key in the sidebar."

        try:
            # --- 1. Context Retrieval ---
            pdf_context = get_pdf_context(text_query, top_k=2)
            weather_context = generate_weather_context(location, weather_api_key) if location and weather_api_key else ""
            
            # Google Search Context
            # search_results_str = ""
            # try:
            #     search_results = search(queries=[text_query])
            #     if search_results and search_results[0].results:
            #         snippets = [res.snippet for res in search_results[0].results[:3] if res.snippet]
            #         search_results_str = "\n\n".join(snippets)
            # except Exception as e:
            #     print(f"Google Search error: {e}")


            # --- 2. Build the Enhanced Prompt ---
            context_block = ""
            if pdf_context:
                context_block += f"--- Information from PDF Documents ---\n{pdf_context}\n\n"
            if weather_context:
                context_block += f"--- Local Weather Forecast ---\n{weather_context}\n\n"
            # if search_results_str:
            #     context_block += f"--- Relevant Web Search Results ---\n{search_results_str}\n\n"

            # Tailor prompt based on whether an image is present
            if image:
                image_analysis_prompt = "Please analyze the provided image and "
            else:
                image_analysis_prompt = ""

            enhanced_prompt = f"""{self.agriculture_context}

Based on the context below, answer the farmer's question.
{context_block}

**Farmer's Question:** {text_query}

**Your Analysis:**
{image_analysis_prompt}provide a comprehensive analysis and specific, actionable advice.
"""

            # --- 3. Generate Response using Gemini ---
            with st.spinner("🤖 Ctrl+Crop is thinking..."):
                if image:
                    # Multimodal query (Image + Text)
                    image_part = {"mime_type": "image/jpeg", "data": base64.b64encode(image.getvalue()).decode("utf-8")}
                    response = self.model.generate_content([enhanced_prompt, image_part])
                else:
                    # Text-only query
                    response = self.model.generate_content(enhanced_prompt)

            return response.text

        except Exception as e:
            error_msg = f"Error during analysis: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg


# --- TTS and ASR Classes (No changes needed) ---
class AgricultureTTS:
    def __init__(self, language='en', output_dir='audio_responses'):
        self.language = language
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        try:
            pygame.mixer.init()
        except Exception:
            pass # Fails in cloud, that's okay

    def generate_speech_file(self, text, filename=None, slow=False):
        try:
            if filename is None:
                filename = f"agriculture_response_{int(time.time())}.mp3"
            filepath = os.path.join(self.output_dir, filename)
            tts = gTTS(text=text, lang=self.language, slow=slow)
            tts.save(filepath)
            return filepath
        except Exception as e:
            print(f"TTS Error: {str(e)}")
            return None

class AgricultureASR:
    def __init__(self, supported_formats=None):
        self.recognizer = sr.Recognizer()
    
    def transcribe_audio_bytes(self, audio_bytes, language='en-US'):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                temp_audio.write(audio_bytes)
                temp_audio_path = temp_audio.name

            with sr.AudioFile(temp_audio_path) as source:
                audio_data = self.recognizer.record(source)
            
            os.unlink(temp_audio_path)
            
            text = self.recognizer.recognize_google(audio_data, language=language)
            return text
        except sr.UnknownValueError:
            return "Error: Could not understand the audio."
        except sr.RequestError as e:
            return f"Error: ASR service unavailable; {e}"
        except Exception as e:
            return f"Error processing audio: {e}"


# --- Streamlit App UI ---
st.set_page_config(page_title="🌾 Ctrl+Crop Assistant", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #2E8B57; text-align: center; margin-bottom: 2rem; }
    .section-header { font-size: 1.5rem; color: #228B22; margin-top: 2rem; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_systems():
    vlm = AgricultureVLM()
    tts = AgricultureTTS()
    asr = AgricultureASR()
    return vlm, tts, asr

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'api_key_configured' not in st.session_state:
    st.session_state.api_key_configured = False

vlm, tts, asr = load_systems()

def main():
    st.markdown('<h1 class="main-header">🌾 Ctrl+Crop AI Assistant</h1>', unsafe_allow_html=True)

    # --- Sidebar ---
    with st.sidebar:
        st.markdown("### 🔑 API Configuration")
        google_api_key = st.text_input("Enter your Google API Key", type="password")
        if st.button("Configure API Key"):
            if google_api_key:
                st.session_state.api_key_configured = vlm.configure_api(google_api_key)
                if st.session_state.api_key_configured:
                    st.success("API Key configured!")
            else:
                st.warning("Please enter an API key.")

        st.markdown("---")
        st.markdown("### 📍 Location & Weather")
        location_name = st.text_input("Location (village/town):")
        tomorrow_api_key = st.text_input("Tomorrow.io API Key (optional)", type="password")
        
        st.markdown("---")
        if st.button("🗑️ Clear Conversation"):
            st.session_state.conversation_history = []
            st.session_state.current_image = None
            st.rerun()

    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="section-header">📷 Image Upload (Optional)</h2>', unsafe_allow_html=True)
        uploaded_image = st.file_uploader("Upload an image for analysis", type=['png', 'jpg', 'jpeg'])
        
        image_bytes = None
        if uploaded_image:
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
            image_bytes = uploaded_image # Pass the bytes directly
    
    with col2:
        st.markdown('<h2 class="section-header">💬 Ask a Question</h2>', unsafe_allow_html=True)
        
        text_query = st.text_area("Enter your question:", height=100)
        
        st.markdown("Or record your question:")
        audio_bytes = audio_recorder(icon_size="2x")
        
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            transcribed_text = asr.transcribe_audio_bytes(audio_bytes)
            if "Error:" not in transcribed_text:
                st.success(f"🎤 Transcribed: *{transcribed_text}*")
                text_query = transcribed_text # Use transcribed text as the query
            else:
                st.error(transcribed_text)
                
        generate_audio = st.checkbox("Generate audio response", value=True)

        if st.button("🔍 Analyze", type="primary"):
            if not st.session_state.api_key_configured:
                st.error("Please configure your Google API Key in the sidebar first.")
            elif not text_query.strip():
                st.error("Please enter or record a question.")
            else:
                response = vlm.process_query(
                    text_query,
                    image=image_bytes,
                    location=location_name,
                    weather_api_key=tomorrow_api_key
                )
                
                entry = {'query': text_query, 'response': response}
                if generate_audio:
                    audio_file = tts.generate_speech_file(response)
                    if audio_file:
                        with open(audio_file, "rb") as f:
                            entry['response_audio'] = f.read()

                st.session_state.conversation_history.insert(0, entry)
                st.rerun()

    # Conversation History
    if st.session_state.conversation_history:
        st.markdown('<h2 class="section-header">📜 Conversation History</h2>', unsafe_allow_html=True)
        for entry in st.session_state.conversation_history:
            with st.chat_message("user"):
                st.markdown(entry['query'])
            with st.chat_message("assistant"):
                st.markdown(entry['response'])
                if 'response_audio' in entry:
                    st.audio(entry['response_audio'], format='audio/mp3')

if __name__ == "__main__":
    main()