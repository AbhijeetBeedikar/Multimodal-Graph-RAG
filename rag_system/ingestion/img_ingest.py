import os
from PIL import Image
import importlib
import rag_system.ingestion.env as env
importlib.reload(env)

from langchain_core.documents import Document

from google.generativeai import configure, GenerativeModel

# Load .env variables (GOOGLE_API_KEY should be stored there)
os.environ["GOOGLE_API_KEY"] = env.key()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY is None:
    raise EnvironmentError("GOOGLE_API_KEY is not set. Add it to your .env file.")

configure(api_key=env.key())

#TODO: Use BLIP instead of Gemini for all image captioning related tasks like image descriptions.
# Can even be used for OCR is BLIP-3 is used

# -----------------------------
# Utility: Load image safely
# -----------------------------
def load_image(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to load image {path}: {e}")


# -----------------------------
# OCR using Gemini Vision
# -----------------------------
def run_ocr(image):
    """
    Uses Gemini 1.5 Flash OCR capability.
    Returns extracted text.
    """
    # TODO: Use pytesseract library + Whisper for OCR
    model = GenerativeModel("gemini-2.5-flash")

    prompt = "Extract all readable text from this image. If no readable text exists, return an empty string. Do not add any extra text other than that which is asked for. Be concise in your answer."
    
    response = model.generate_content([prompt, image])
    print("Gemini API call made from img_ingest.py --> run_ocr")
    if response.candidates[0].content.parts != []:
      return response.text.strip()
    else:
      return ""

# -----------------------------
# Captioning using Gemini Vision
# -----------------------------
def caption_image(image):
    """
    Generates a natural-language caption describing the image.
    """
    model = GenerativeModel("gemini-2.5-flash")

    prompt = (
        "Describe this image in detail in one sentence. "
        "Focus on what is visually present (objects, people, actions, scene)."
    )
    print("Gemini API call made from img_ingest.py --> caption_image")
    response = model.generate_content([prompt, image])

    return response.text.strip() if response.text else ""


# -----------------------------
# Object Detection (Gemini)
# -----------------------------
def detect_objects(image):
    """
    Uses Gemini to detect key objects. 
    Gemini doesn't do bounding boxes but can classify objects visually.
    """
    model = GenerativeModel("gemini-2.5-flash")

    prompt = (
        "List all objects you see in this image. "
        "Return a comma-separated list of nouns only."
    )
    print("Gemini API call made from img_ingest.py --> detect_objects")
    response = model.generate_content([prompt, image])

    if not response.text:
        return []

    # Convert "cat, skateboard, sunglasses" → ["cat", "skateboard", "sunglasses"]
    return [obj.strip().lower() for obj in response.text.split(",")]


# -----------------------------
# MAIN INGESTION FUNCTION
# -----------------------------
def ingest_image(path):
    """
    Ingest an image file:
    - Load image
    - OCR attempt
    - If OCR empty ➜ caption + object detection
    - Create a single text chunk
    - Return as langchain Document for embedding + vector DB indexing
    """

    image = load_image(path)
    file_name = os.path.basename(path)

    # --- Step 1: OCR ---
    ocr_text = run_ocr(image)

    if ocr_text:
        unified_text = ocr_text
        modality = "image-text"
        caption = None
        objects = []
        # TODO: OCR Data should be chunked
    else:
        # --- Step 2: Captioning + Object Detection ---

        caption = caption_image(image)
        objects = detect_objects(image)

        unified_text = (
            f"{caption}. Objects detected: {', '.join(objects)}"
            if objects
            else caption
        )

        modality = "image-visual"

    # --- Build Document ---
    doc = Document(
        page_content=unified_text,
        metadata={
            "source_file": file_name,
            "modality": modality,
            "caption": caption,
            "objects": objects,
            "ocr_text": ocr_text if ocr_text else None,
            "chunk_index": 0,
        }
    )

    return [doc]
