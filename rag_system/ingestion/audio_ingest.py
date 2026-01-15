import os
import librosa
from langchain_core.documents import Document
from google.generativeai import GenerativeModel, configure
from google.genai.types import Blob
import importlib
import rag_system.ingestion.env as env
import pathlib
import mimetypes
importlib.reload(env)

# -------------------------
# Environment Setup
# -------------------------
os.environ["GOOGLE_API_KEY"] = env.key()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY is None:
    raise EnvironmentError("GOOGLE_API_KEY not found. Add it to your .env file.")

configure(api_key=GOOGLE_API_KEY)

def load_audio_blob(path):
    # Detect MIME type from the file extension
    mime_type = mimetypes.guess_type(path)[0]
    if mime_type is None:
        raise ValueError(f"Could not determine MIME type for file: {path}")

    # Load raw bytes from audio file
    with open(path, "rb") as f:
        audio_bytes = f.read()

    # Wrap in Gemini Blob format
    return {
        "mime_type": mime_type,
        "data": audio_bytes
    }
# -------------------------
# Load Audio File
# -------------------------
def load_audio(path):
    try:
        audio, sr = librosa.load(path, sr=None)
        return audio, sr
    except Exception as e:
        raise ValueError(f"Failed to load audio '{path}': {e}")


# -------------------------
# Transcribe using Gemini
# -------------------------
def transcribe_audio(path):
    """
    Returns transcription text.
    Gemini 1.5 Flash handles:
    - speech
    - noisy speech
    - accents
    - long-form speech
    """
    model = GenerativeModel("gemini-2.5-flash")
    
    prompt = (
        "Transcribe all spoken words from this audio. "
        "If the audio has no speech, return EXACTLY the string: 'NO_SPEECH'. Do not add any extra text other than that which is asked for. Be concise in your answer."
    )
    print("Gemini API call made from audio_ingest.py --> transcribe_audio")
    audio_bytes = pathlib.Path(path).read_bytes()
    audio_blob = Blob(
        data=audio_bytes,
        mime_type='audio/mp3'
    )
    response = model.generate_content([prompt, load_audio_blob(path)])
    
    return response.text.strip() if response.text else "NO_SPEECH"


# -------------------------
# Caption Non-Speech Audio
# -------------------------
def caption_audio(path):
    """
    Describe what you hear if there is no speech.
    Good for:
    - music
    - ambient noise
    - environmental soundscapes
    - mechanical noises
    """
    model = GenerativeModel("gemini-2.5-flash")
    prompt = (
        "Describe this audio clip in one detailed sentence. "
        "Do not invent speech if none exists."
    )
    print("Gemini API call made from audio_ingest.py --> caption_audio")
    with open(path, "rb") as f:
        audio_bytes = f.read()

    response = model.generate_content([prompt, audio_bytes])
    return response.text.strip() if response.text else ""


# -------------------------
# Chunk Text if Long
# -------------------------
def chunk_text(text, max_len=600, overlap=100):
    """
    Simple text chunker for audio transcripts.
    """
    words = text.split()
    chunks = []
    
    start = 0
    while start < len(words):
        end = start + max_len
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)
        start += max_len - overlap
    
    return chunks


# -------------------------
# MAIN INGEST FUNCTION
# -------------------------
def ingest_audio(path):
    """
    Ingest audio by:
    - transcribing speech
    - falling back to captioning for music/noise
    - chunking long transcripts
    - returning LangChain Document objects
    """

    file_name = os.path.basename(path)

    # Load audio to get duration
    audio, sr = load_audio(path)
    duration_seconds = librosa.get_duration(y=audio, sr=sr)

    # Step 1: Try transcription
    transcription = transcribe_audio(path)

    if transcription != "NO_SPEECH" and transcription.strip():
        modality = "audio-speech"
        base_text = transcription
    else:
        # Step 2: Fallback to captioning
        caption = caption_audio(path)
        base_text = caption
        modality = "audio-non-speech"

    # Step 3: Chunking (only if long)
    chunks = chunk_text(base_text) if len(base_text) > 700 else [base_text]

    # Step 4: Build Document objects
    docs = []
    for idx, chunk in enumerate(chunks):
        doc = Document(
            page_content=chunk,
            metadata={
                "source_file": file_name,
                "modality": modality,
                "duration_seconds": duration_seconds,
                "chunk_index": idx,
                "num_chunks": len(chunks)
            }
        )
        docs.append(doc)

    return docs
