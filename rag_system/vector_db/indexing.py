import importlib
import rag_system.vector_db.env_vec as env_vec
importlib.reload(env_vec)

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings


# ---------------------------------------
# Initialize Qdrant Collection
# ---------------------------------------
def init_qdrant(collection_name="multimodal_rag"):
    """
    Use a persistent local Qdrant instance, saved inside your repo folder.
    """
    client = QdrantClient(
        url="https://8dde95a1-0d93-4252-85c4-3e7ba5930328.europe-west3-0.gcp.cloud.qdrant.io:6333",
        api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.-1Va0jXtIeaWKHWKyj6j5F9lUEO5bEKtweSZutf7BKg",
    ) #Qdrant Vector database is stored at path

    embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    vector_size = len(embedding_model.embed_query("hello world")) # to get size of embedding returned by the model for some text

    if not client.collection_exists(collection_name): # if the collection doesnt already exist
        client.create_collection(
            collection_name=collection_name, # name of the collection
            vectors_config=VectorParams(
                size=vector_size, # Size of all points within collection = vector_size
                distance=Distance.COSINE # comparison parameter for all points = cosine similarity
            )
        )

    return client 


# ---------------------------------------
# Convert LangChain Document → Qdrant Point
# ---------------------------------------
def document_to_point(doc, idx, embedding_model):
    """
    Converts a Document into:
      - vector embedding
      - metadata dict
      - unique ID
    """
    embedding = embedding_model.embed_query(doc.page_content) #creating the embedding itself for each text chunk doc
    payload = doc.metadata.copy() 
    payload["text"] = doc.page_content  # store raw text for extraction
    return PointStruct(
        id=idx, # unique id of each point
        vector=embedding, # embedded version of doc (using embedded_model) which means to compare this document's text to another query, that query needs to be converted to this embedding format
        payload=payload # metadata + actual text within doc
    )


# ---------------------------------------
# Main Index Function
# ---------------------------------------
def index_documents(docs, collection_name="multimodal_rag"):
    """
    docs: list[Document]
    adds each document in docs as a point in Qdrant
    """
    if len(docs) == 0:
        raise ValueError("No documents provided for indexing")

    # 1. Initialize Qdrant
    client = init_qdrant(collection_name=collection_name)

    # 2. Create embeddings
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # 3. Convert docs → Qdrant points
    points = []

    for idx, doc in enumerate(docs):
        # TODO: add current time of operation to the idx variable to ensure that previous points do not get overwritten.
        #  If a new point has the same id as another point in the collection, then client.upsert() will
        #  Replace the pre-existing point instead of being added to the collection
        point = document_to_point(doc, idx, embedding_model)
        points.append(point)

    # 4. Upsert into Qdrant (vector + payload)
    client.upsert(
        collection_name=collection_name,
        points=points
    )

    print(f"Indexed {len(docs)} documents into Qdrant collection '{collection_name}'.")

    return client,collection_name
