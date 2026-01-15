import rag_system.ingestion.pdfingestion as pdfingestion
import rag_system.ingestion.img_ingest as img_ingest
import rag_system.ingestion.audio_ingest as audio_ingest
import importlib

from ..graph.graph_builder import process_document_for_graph

importlib.reload(pdfingestion)
importlib.reload(img_ingest)
importlib.reload(audio_ingest)
#from rag_system.ingestion.ingest_video import ingest_video

from ..vector_db import indexing as indexing
importlib.reload(indexing)

def ingest_and_index(path):
    if path.lower().endswith((".pdf",".txt")):
        docs = pdfingestion.text_split(path)
    elif path.lower().endswith(('.png', '.jpg', '.jpeg')):
        docs = img_ingest.ingest_image(path)
    elif path.lower().endswith(('.mp3', '.wav', '.m4a')):
        docs = audio_ingest.ingest_audio(path)
    #elif path.lower().endswith(('.mp4', '.mov', '.avi')):
     #   docs = ingest_video(path)
    else:
        raise ValueError("Unsupported file format")

    # Building the Graph
    process_document_for_graph(" ".join([doc.page_content for doc in docs])) # expands the graph according to the entities and relations extracted from all the text within a certain file
    print("Ingestion + Indexing complete.")

    return indexing.index_documents(docs) # adds each document as a point in Qdrant
    
