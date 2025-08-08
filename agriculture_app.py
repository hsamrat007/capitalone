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
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from gtts import gTTS
import speech_recognition as sr
from pydub import AudioSegment

# Agriculture VLM Classes
class AgricultureVLM:
    def __init__(self, model_name="vikhyatk/moondream2"):
        print("🚀 Loading Vision Language Model...")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🖥️  Using device: {self.device}")
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            ).to(self.device)
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=True
            )
            
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
    
    def process_query(self, image, text_query):
        try:
            enhanced_prompt = f"""
            {self.agriculture_context}
            
            Farmer's Question: {text_query}
            
            Please analyze the image and provide specific, actionable advice:
            """
            
            response = self.model.answer_question(image, enhanced_prompt, self.tokenizer)
            return response
            
        except Exception as e:
            error_msg = f"Error processing agricultural query: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg

class AgricultureTTS:
    def __init__(self, language='en', output_dir='audio_responses'):
        self.language = language
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            pygame.mixer.init()
            self.pygame_available = True
            print("🔊 Audio playback available")
        except:
            self.pygame_available = False
            print("⚠️  Audio playback not available (cloud environment)")
    
    def generate_speech_file(self, text, filename=None, slow=False):
        try:
            if filename is None:
                timestamp = int(time.time())
                filename = f"agriculture_response_{timestamp}.mp3"
            
            if not filename.endswith('.mp3'):
                filename += '.mp3'
            
            filepath = os.path.join(self.output_dir, filename)
            
            tts = gTTS(text=text, lang=self.language, slow=slow)
            tts.save(filepath)
            
            print(f"💾 Speech saved: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"❌ TTS Error: {str(e)}")
            return None

class AgricultureASR:
    def __init__(self, supported_formats=None):
        self.recognizer = sr.Recognizer()
        self.supported_formats = supported_formats or ['.wav', '.mp3', '.flac', '.m4a', '.aiff']
        
        print("🎤 Enhanced Speech Recognition initialized")
        print(f"📁 Supported formats: {', '.join(self.supported_formats)}")
    
    def convert_to_wav(self, audio_file_path):
        try:
            file_extension = os.path.splitext(audio_file_path)[1].lower()
            
            if file_extension == '.wav':
                return audio_file_path
            
            print(f"🔄 Converting {file_extension} to WAV...")
            
            if file_extension == '.mp3':
                audio = AudioSegment.from_mp3(audio_file_path)
            elif file_extension == '.m4a':
                audio = AudioSegment.from_file(audio_file_path, format="m4a")
            elif file_extension == '.flac':
                audio = AudioSegment.from_file(audio_file_path, format="flac")
            else:
                audio = AudioSegment.from_file(audio_file_path)
            
            temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            audio.export(
                temp_wav.name,
                format="wav",
                parameters=["-ac", "1", "-ar", "16000"]
            )
            
            print(f"✅ Converted to: {temp_wav.name}")
            return temp_wav.name
            
        except Exception as e:
            print(f"❌ Conversion error: {str(e)}")
            return None
    
    def transcribe_audio_file(self, audio_file_path, language='en-US'):
        converted_file = None
        try:
            if not os.path.exists(audio_file_path):
                print(f"❌ Audio file not found: {audio_file_path}")
                return None
            
            file_extension = os.path.splitext(audio_file_path)[1].lower()
            if file_extension not in self.supported_formats:
                print(f"❌ Unsupported format: {file_extension}")
                return None
            
            print(f"🎵 Processing audio file: {os.path.basename(audio_file_path)}")
            
            wav_file = self.convert_to_wav(audio_file_path)
            if wav_file is None:
                return None
            
            converted_file = wav_file if wav_file != audio_file_path else None
            
            with sr.AudioFile(wav_file) as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio_data = self.recognizer.record(source)
            
            print("🔄 Converting speech to text...")
            
            text = self.recognizer.recognize_google(audio_data, language=language)
            print(f"✅ Transcribed: '{text}'")
            
            return text
            
        except sr.UnknownValueError:
            print("❌ Could not understand the audio in the file")
            return None
        except sr.RequestError as e:
            print(f"❌ ASR service error: {e}")
            return None
        except Exception as e:
            print(f"❌ Error processing audio file: {e}")
            return None
        finally:
            if converted_file and os.path.exists(converted_file):
                try:
                    os.unlink(converted_file)
                    print("🗑️  Temporary file cleaned up")
                except:
                    pass

# Streamlit Configuration
st.set_page_config(
    page_title="🌾 Agriculture VLM Assistant",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #228B22;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #F0FFF0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #32CD32;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize models with caching
@st.cache_resource
def load_models():
    with st.spinner("🚀 Loading Agriculture VLM models..."):
        vlm = AgricultureVLM()
        tts = AgricultureTTS()
        asr = AgricultureASR()
        return vlm, tts, asr

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if 'current_image' not in st.session_state:
    st.session_state.current_image = None

if 'session_id' not in st.session_state:
    st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

# Utility functions
def get_audio_player_html(audio_file_path):
    try:
        with open(audio_file_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
            audio_base64 = base64.b64encode(audio_bytes).decode()
            
        audio_html = f"""
        <audio controls style="width: 100%;">
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
            Your browser does not support the audio element.
        </audio>
        """
        return audio_html
    except Exception as e:
        return f"Error loading audio: {str(e)}"

def save_uploaded_file(uploaded_file, suffix=None):
    try:
        if suffix is None:
            suffix = os.path.splitext(uploaded_file.name)[1]
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(uploaded_file.getbuffer())
        temp_file.close()
        
        return temp_file.name
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None

# Main Streamlit App
def main():
    # Header
    st.markdown('<h1 class="main-header">🌾 Agriculture VLM Assistant</h1>', unsafe_allow_html=True)
    
    # Load models
    vlm, tts, asr = load_models()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🛠️ Controls")
        
        # Session info
        st.markdown("### 📊 Session Info")
        st.markdown(f"**Session ID:** `{st.session_state.session_id}`")
        st.markdown(f"**Queries:** {len(st.session_state.conversation_history)}")
        st.markdown(f"**Image Loaded:** {'✅' if st.session_state.current_image else '❌'}")
        
        # Clear conversation
        if st.button("🗑️ Clear Conversation"):
            st.session_state.conversation_history = []
            st.session_state.current_image = None
            st.rerun()
        
        # Quick analysis presets
        st.markdown("### 🎯 Quick Analysis")
        quick_queries = {
            "Disease Detection": "Analyze this crop image for any plant diseases. What symptoms do you see and what treatment do you recommend?",
            "Pest Identification": "Identify any pests or insects visible in this image. What damage are they causing and how should I control them?",
            "Nutrient Deficiency": "Examine the plant leaves and overall appearance. Do you see signs of nutrient deficiency? What fertilizers should I apply?",
            "Growth Stage": "What growth stage is this crop in? What specific care and management does it need at this stage?",
            "Harvest Readiness": "Is this crop ready for harvest? What indicators should I look for to determine optimal harvest timing?"
        }
        
        selected_query = st.selectbox("Select Analysis Type:", list(quick_queries.keys()))
        quick_query_text = quick_queries[selected_query]
        
        if st.button("🚀 Run Quick Analysis"):
            if st.session_state.current_image:
                st.session_state.quick_query = quick_query_text
                st.rerun()
            else:
                st.error("Please upload an image first!")

    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="section-header">📷 Image Upload</h2>', unsafe_allow_html=True)
        
        # Image upload options
        upload_option = st.radio("Choose upload method:", ["Upload File", "Use URL"])
        
        if upload_option == "Upload File":
            uploaded_image = st.file_uploader(
                "Upload crop image",
                type=['png', 'jpg', 'jpeg', 'webp'],
                help="Upload an image of your crops for analysis"
            )
            
            if uploaded_image:
                try:
                    image = Image.open(uploaded_image)
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    st.session_state.current_image = image
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                    
                    st.markdown(f"**Size:** {image.size}")
                    st.markdown(f"**Mode:** {image.mode}")
                    
                except Exception as e:
                    st.error(f"Error loading image: {str(e)}")
        
        else:  # URL option
            image_url = st.text_input("Enter image URL:")
            
            if image_url and st.button("Load Image from URL"):
                try:
                    response = requests.get(image_url)
                    response.raise_for_status()
                    image = Image.open(io.BytesIO(response.content))
                    
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    st.session_state.current_image = image
                    st.image(image, caption="Loaded from URL", use_column_width=True)
                    
                except Exception as e:
                    st.error(f"Error loading image from URL: {str(e)}")
    
    with col2:
        st.markdown('<h2 class="section-header">💬 Query Interface</h2>', unsafe_allow_html=True)
        
        # Query input options
        query_method = st.radio("Choose input method:", ["Text Input", "Voice Recording", "Audio File Upload"])
        
        if query_method == "Text Input":
            # Text query input
            text_query = st.text_area(
                "Enter your question about the crop:",
                value=getattr(st.session_state, 'quick_query', ''),
                height=100,
                help="Ask about diseases, pests, growth stage, harvest timing, etc."
            )
            
            if hasattr(st.session_state, 'quick_query'):
                delattr(st.session_state, 'quick_query')
            
            generate_audio = st.checkbox("Generate audio response", value=True)
            
            if st.button("🔍 Analyze", type="primary"):
                if st.session_state.current_image and text_query.strip():
                    with st.spinner("🤖 Analyzing with AI..."):
                        try:
                            response = vlm.process_query(st.session_state.current_image, text_query)
                            
                            conversation_entry = {
                                'type': 'text_query',
                                'query': text_query,
                                'response': response,
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'generate_audio': generate_audio
                            }
                            
                            if generate_audio:
                                audio_file = tts.generate_speech_file(response)
                                if audio_file:
                                    conversation_entry['response_audio'] = audio_file
                            
                            st.session_state.conversation_history.append(conversation_entry)
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error during analysis: {str(e)}")
                else:
                    if not st.session_state.current_image:
                        st.error("Please upload an image first!")
                    if not text_query.strip():
                        st.error("Please enter a question!")
        
        elif query_method == "Voice Recording":
            st.markdown("🎙️ **Record your question:**")
            
            audio_bytes = audio_recorder(
                text="Click to record",
                recording_color="#e87070",
                neutral_color="#6aa36f",
                icon_name="microphone",
                icon_size="2x",
                pause_threshold=3.0,
                sample_rate=16000
            )
            
            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")
                
                language = st.selectbox(
                    "Select language for transcription:",
                    ["en-US", "hi-IN", "ta-IN", "te-IN", "bn-IN"],
                    help="Choose the language you spoke in"
                )
                
                generate_audio = st.checkbox("Generate audio response", value=True, key="voice_audio")
                
                if st.button("🎤 Process Voice Query", type="primary"):
                    if st.session_state.current_image:
                        with st.spinner("🔄 Processing voice query..."):
                            try:
                                temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                                temp_audio.write(audio_bytes)
                                temp_audio.close()
                                
                                transcribed_text = asr.transcribe_audio_file(temp_audio.name, language)
                                
                                if transcribed_text:
                                    st.success(f"🎯 Transcribed: *{transcribed_text}*")
                                    
                                    response = vlm.process_query(st.session_state.current_image, transcribed_text)
                                    
                                    conversation_entry = {
                                        'type': 'voice_query',
                                        'transcribed_query': transcribed_text,
                                        'response': response,
                                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        'language': language,
                                        'generate_audio': generate_audio
                                    }
                                    
                                    if generate_audio:
                                        audio_file = tts.generate_speech_file(response)
                                        if audio_file:
                                            conversation_entry['response_audio'] = audio_file
                                    
                                    st.session_state.conversation_history.append(conversation_entry)
                                    
                                    os.unlink(temp_audio.name)
                                    st.rerun()
                                else:
                                    st.error("Could not transcribe audio. Please try again.")
                                    
                            except Exception as e:
                                st.error(f"Error processing voice query: {str(e)}")
                    else:
                        st.error("Please upload an image first!")
        
        else:  # Audio File Upload
            st.markdown("📁 **Upload audio file:**")
            
            uploaded_audio = st.file_uploader(
                "Upload audio question",
                type=['wav', 'mp3', 'flac', 'm4a', 'webm'],
                help="Upload an audio file with your question"
            )
            
            if uploaded_audio:
                st.audio(uploaded_audio, format=f"audio/{uploaded_audio.type.split('/')[-1]}")
                
                language = st.selectbox(
                    "Select language for transcription:",
                    ["en-US", "hi-IN", "ta-IN", "te-IN", "bn-IN"],
                    help="Choose the language in the audio file",
                    key="upload_language"
                )
                
                generate_audio = st.checkbox("Generate audio response", value=True, key="upload_audio")
                
                if st.button("🎵 Process Audio File", type="primary"):
                    if st.session_state.current_image:
                        with st.spinner("🔄 Processing audio file..."):
                            try:
                                temp_audio_path = save_uploaded_file(uploaded_audio)
                                
                                if temp_audio_path:
                                    transcribed_text = asr.transcribe_audio_file(temp_audio_path, language)
                                    
                                    if transcribed_text:
                                        st.success(f"🎯 Transcribed: *{transcribed_text}*")
                                        
                                        response = vlm.process_query(st.session_state.current_image, transcribed_text)
                                        
                                        conversation_entry = {
                                            'type': 'audio_file_query',
                                            'audio_filename': uploaded_audio.name,
                                            'transcribed_query': transcribed_text,
                                            'response': response,
                                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                            'language': language,
                                            'generate_audio': generate_audio
                                        }
                                        
                                        if generate_audio:
                                            audio_file = tts.generate_speech_file(response)
                                            if audio_file:
                                                conversation_entry['response_audio'] = audio_file
                                        
                                        st.session_state.conversation_history.append(conversation_entry)
                                        
                                        os.unlink(temp_audio_path)
                                        st.rerun()
                                    else:
                                        st.error("Could not transcribe audio file. Please check the audio quality.")
                                        
                            except Exception as e:
                                st.error(f"Error processing audio file: {str(e)}")
                    else:
                        st.error("Please upload an image first!")

    # Conversation History
    if st.session_state.conversation_history:
        st.markdown('<h2 class="section-header">📚 Conversation History</h2>', unsafe_allow_html=True)
        
        for i, entry in enumerate(reversed(st.session_state.conversation_history)):
            with st.expander(f"💬 Query {len(st.session_state.conversation_history) - i}: {entry['timestamp']}"):
                
                st.markdown("**🤔 Question:**")
                if entry['type'] == 'text_query':
                    st.markdown(f"*{entry['query']}*")
                else:
                    st.markdown(f"*{entry['transcribed_query']}*")
                    if entry['type'] == 'audio_file_query':
                        st.markdown(f"📁 Audio file: `{entry['audio_filename']}`")
                    if 'language' in entry:
                        st.markdown(f"🌐 Language: `{entry['language']}`")
                
                st.markdown("**🤖 AI Response:**")
                st.markdown(entry['response'])
                
                if 'response_audio' in entry and entry['response_audio']:
                    st.markdown("**🔊 Audio Response:**")
                    try:
                        audio_html = get_audio_player_html(entry['response_audio'])
                        st.markdown(audio_html, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error loading audio: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        🌾 Agriculture VLM Assistant - Powered by AI for Better Farming 🌾<br>
        <small>Upload crop images and ask questions via text or voice to get expert agricultural advice</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
