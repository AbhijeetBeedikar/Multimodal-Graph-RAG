import google.generativeai as genai #TODO: try from google import genai
import json
import importlib
import rag_system.ingestion.env as env
importlib.reload(env)
import os
import networkx as nx
import pickle
os.environ["GOOGLE_API_KEY"] = env.key()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")


def extract_entities_and_relations(text):
    prompt = f'''
    Extract entities and relationships from the following text.Write the response as a plain text output. that only consists of the JSON contents. Strictly nothing more. The response must start and end with a curly bracket.
    
    Return JSON in this exact format:
    {{
      "entities": ["ENTITY1", "ENTITY2", ...],
      "relationships": [
        {{"source": "ENTITY1", "target": "ENTITY2", "relation": "RELATIONSHIP"}}
      ]
    }}

    TEXT:
    {text}
    '''

    res = model.generate_content(prompt)
    load_text = res.text[res.text.index("{"):][::-1]
    load_text = load_text[load_text.index("}"):][::-1]
    print("Gemini API Call Made from graph_builder --> extract_entities_and_relations")
    return json.loads(load_text)


def add_to_graph(entity_data):
    # TODO: create a new knowledge graph in tmp folder in streamlit if it doesn't exist.
    with open("knowledge_graph.gpickle", "rb") as f:  # 'rb' for Read Binary
        G = pickle.load(f)
    entities = entity_data["entities"]
    relationships = entity_data["relationships"]

    # Add nodes
    for e in entities:
        G.add_node(e, type="entity")

    # Add edges
    for rel in relationships:
        src = rel["source"]
        trg = rel["target"]
        r   = rel["relation"]

        G.add_edge(src, trg, relation=r)
def process_document_for_graph(text): # makes 1 Gemini API call
    with open("knowledge_graph.gpickle", "rb") as f:  # 'rb' for Read Binary
        G = pickle.load(f)
    extracted = extract_entities_and_relations(text)
    add_to_graph(extracted)
    #os.chdir("/RAG_Pipeline/rag_system/graph/drive/MyDrive/AI_Projects/multimodal_enterprise_rag/rag_system/graph")
    with open("knowledge_graph.gpickle", "wb") as f:
      pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
    return extracted
