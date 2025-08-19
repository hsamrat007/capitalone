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
from transformers import AutoModelForCausalLM, AutoTokenizer
import datetime

import re


def format_pdf_insights(text):
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    # Add line breaks before bullet points if missing
    text = re.sub(r"•", "\n• ", text)
    # Optionally, add extra breaks or indentation after key words (e.g., Crop)
    text = re.sub(r"(\d+\sCrop)", r"\n\1", text)
    return text.strip()


current_datetime = datetime.datetime.now()


# --- Helper function for Gemini ---
def image_to_base64(image):
    """Converts a PIL image to a base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


# --- Main AI Logic Class ---
class AgricultureVLM:
    def __init__(self, model_name="vikhyatk/moondream2"):
        print("🚀 Loading Vision Language Model...")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🖥️  Using device: {self.device}")

        try:
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True
            )

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Context to prepend in prompts
            self.agriculture_context = """
            You are an expert agricultural assistant with deep knowledge in:
            - Crop health and disease identification
            - Pest detection and integrated pest management
            - Soil conditions and nutrient management
            - Irrigation and water management strategies
            - Harvest timing and post-harvest techniques
            - Weather impact assessment and mitigation
            - Sustainable and organic farming practices
            - Crop rotation and companion planting
            Always provide practical, actionable advice suitable for farmers.
            Focus on cost-effective and locally appropriate solutions.
            """

            print("✅ Vision Language Model loaded successfully!")

        except Exception as e:
            print(f"❌ Error loading VLM: {str(e)}")
            raise e

    def configure_api(self, api_key="AIzaSyDA1ym7lYZyoamqIardcQx-xDhI_AaHS1Y"):
        """Configures the Gemini API key and initializes the model."""
        try:
            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            print("✅ Gemini API configured successfully!")
            return True
        except Exception as e:
            print(f"❌ Error configuring Gemini API: {str(e)}")
            st.error(f"Failed to configure Gemini API: {e}")
            return False

    def process_query(self, text_query, image=None, location=None, weather_api_key="579b464db66ec23bdd00000176bb599d213c47c4783456e7c1c53b83"):
        """
        Processes a query with priority on PDF + weather context, and optionally enriches with Gemini.
        """
        if not self.model:
            return "Error: The AI model is not configured. Please enter a valid Google API Key in the sidebar."

        try:
            # --- 1. Context Retrieval (PDF + Weather first) ---
            combined_query = f"{text_query} {location}" if location else text_query
            pdf_context = get_pdf_context(combined_query, top_k=5)
            weather_context = generate_weather_context(location) if location else ""

            if pdf_context:
                pdf_context = format_pdf_insights(pdf_context)

            # Summarize PDF insights with Gemini AI
            summarize_prompt = f"""
You are a helpful agricultural assistant.

Here is raw crop and regional agricultural information extracted from PDFs:

{pdf_context}

Please provide a clear, concise, and farmer-friendly summary of the above information relevant to the location and question.
"""
            gemini_summary_response = self.gemini_model.generate_content(summarize_prompt)
            if gemini_summary_response and gemini_summary_response.text:
                pdf_context = gemini_summary_response.text.strip()

            # --- 2. Build Context Block ---
            context_block = f"Current date & time: {current_datetime}\n\n"
            if pdf_context:
                context_block += f"--- Regional Agricultural Insights (PDFs) ---\n{pdf_context}\n\n"
            if weather_context:
                context_block += f"--- Local Weather Forecast ---\n{weather_context}\n\n"

            # 🔹 Gemini VLM for image analysis (optional, not primary)
            vlm_response = ""
            if image:
                try:
                    vlm_response = self.model.answer_question(image, text_query, self.tokenizer)
                except Exception as e:
                    print(f"❌ VLM error: {e}")
                    vlm_response = ""

            if vlm_response:
                context_block += f"--- Image Analysis (VLM) ---\n{vlm_response}\n\n"

            # --- 3. Build Prompt for Gemini ---
            enhanced_prompt = f"""{self.agriculture_context}

You are helping a farmer with practical advice. 
Use the PDF-based crop insights and current weather first, then refine with Gemini knowledge if needed. 

**Farmer's Question:** {text_query}

**Available Context:**  
{context_block}

**Your Analysis:** Provide a structured, practical recommendation (crops, methods, precautions). 
Be concise and farmer-friendly.
"""
            print(enhanced_prompt)

            # --- 4. Generate Response (Gemini as enrichment layer) ---
            with st.spinner("🤖 Ctrl+Crop is thinking..."):
                if image:
                    image_part = {"mime_type": "image/jpeg", "data": base64.b64encode(image.getvalue()).decode("utf-8")}
                    response = self.gemini_model.generate_content([enhanced_prompt, image_part])
                else:
                    response = self.gemini_model.generate_content(enhanced_prompt)

            # 🔹 Return structured response instead of a single string
            return {
                "pdf_insights": pdf_context if pdf_context else None,
                "weather": weather_context if weather_context else None,
                "gemini": response.text if response else None
            }

            # 🔹 Combine Gemini’s text with PDF+Weather context emphasis
            final_response = f"""📌 **Contextual Analysis (PDFs + Weather):**  
{pdf_context if pdf_context else "No PDF insights available."}  

🌦️ **Weather Insights:**  
{weather_context if weather_context else "Weather data unavailable."}  

🤖 **AI Recommendation (Gemini-enhanced):**  
{response.text}"""

            return final_response

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
st.set_page_config(page_title="🌾 Ctrl+Crop AI Assistant", layout="wide")

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

        st.markdown("---")
        st.markdown("### 📍 Location & Weather")
        location_name = st.text_input("Location (village/town):")
        #tomorrow_api_key = "579b464db66ec23bdd00000176bb599d213c47c4783456e7c1c53b83"

        st.session_state.api_key_configured = vlm.configure_api()
        if st.session_state.api_key_configured:
            st.success("API Key configured!")

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
            image_bytes = uploaded_image  # Pass the bytes directly

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
                text_query = transcribed_text  # Use transcribed text as the query
            else:
                st.error(transcribed_text)

        generate_audio = st.checkbox("Generate audio response", value=True)

        if st.button("🔍 Analyze", type="primary"):
            if not text_query.strip():
                st.error("Please enter or record a question.")
            else:
                response = vlm.process_query(
                    text_query,
                    image=image_bytes,
                    location=location_name,
                    #weather_api_key=tomorrow_api_key
                )

                entry = {'query': text_query, 'response': response}
                if generate_audio and response.get("gemini"):
                    audio_file = tts.generate_speech_file(response["gemini"])
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
                result = entry['response']

            # 📄 PDF Insights
            st.subheader("📄 PDF Insights")
            if result.get("pdf_insights"):
                st.write(result["pdf_insights"])
            else:
                st.info("No PDF insights available.")

            # 🌦️ Weather Data
            st.subheader("🌦️ Weather Data")
            if result.get("weather"):
                st.write(result["weather"])
            else:
                st.info("No weather data available.")

            # 🤖 Gemini AI Recommendations
            st.subheader("🤖 Gemini AI Recommendations")
            if result.get("gemini"):
                st.write(result["gemini"])
            else:
                st.info("No Gemini recommendations.")

            if 'response_audio' in entry:
                st.audio(entry['response_audio'], format='audio/mp3')


if __name__ == "__main__":
    main()
