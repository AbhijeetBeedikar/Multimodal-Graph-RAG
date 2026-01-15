"""
HYBRID SEARCH ORCHESTRATOR
--------------------------
Combines:
  • Query classification
  • Graph search (exact + multi-hop)
  • Vector search (Qdrant)
  • Keyword expansion
  • LLM answer generation

This file uses FUNCTIONS ONLY (no classes).
"""

import google.generativeai as genai
from rag_system.graph.graph_search import (
    search_entity,
    one_hop_relations,
    multi_hop_traversal,
    keyword_search
)
from rag_system.graph.graph_builder import extract_entities_and_relations
from sentence_transformers import SentenceTransformer, util
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
import importlib
import rag_system.ingestion.env as env
importlib.reload(env)
import os
os.environ["GOOGLE_API_KEY"] = env.key()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


# -------------------------
# COSINE SIMILARITY CLASSIFIER
# -------------------------
def classify_query(query: str):
    model = SentenceTransformer("all-MiniLM-L6-v2")   # fast, light

    # -------------------------
    # CATEGORY DESCRIPTIONS
    # -------------------------
    CATEGORIES = {
        "FACTUAL_LOOKUP": """
            Direct factual questions requiring a specific answer.
            Examples: who, when, where, what, founder, CEO, location.
        """,

        "SUMMARIZATION": """
            Requests to summarize, explain, outline, or describe text broadly.
            Examples: summarize, overview, high level, explain.
        """,

        "RELATIONAL_REASONING": """
            Questions about relationships or comparisons between entities.
            Examples: relationship between X and Y, compare, how connected.
        """,

        "CROSS_MODAL_LINKAGE": """
            Queries linking text, image, audio, or video content.
            Examples: describe the image, relate audio to document, show connected context.
        """,

        "KEYWORD_SEARCH": """
            Pure keyword-based search or filtering.
            Examples: find mentions of X, occurrences of Y, keyword search.
        """
    }

    # converts each category description to tensors to represent them numerically
    category_embeddings = {
      name: model.encode(desc, convert_to_tensor=True)
      for name, desc in CATEGORIES.items()
    }

    #converts the query to tensors
    query_emb = model.encode(query, convert_to_tensor=True)

    best_cat = None
    best_score = -1
    #TODO: What if there are multiple categories mentioned? Keep a threshold beyond which catgeories are accepted
    for cat_name, cat_emb in category_embeddings.items():

        # get cosine similarity score for each category
        score = util.cos_sim(query_emb, cat_emb).item()
        if score > best_score:
            best_score = score
            best_cat = cat_name

    return best_cat, best_score

# ----------------------------------------
# 2. Vector Search
# ----------------------------------------

def vector_search(query, client, embedder= GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001"), collection="multimodal_rag", k=5):
    ''' Searches through the Vector Database and returns k most similar points to the query'''
    qvec = embedder.embed_query(query)

    # TODO: Keep a base similarity threshold. If no document meets this threshold then no document should be returned
    results = client.query_points(
        collection_name=collection,
        query=qvec,
        limit=k,
        with_payload=True #return the payload along with the point if point text is similar to query
    )

    return [hit.payload["text"] for hit in results.points]

  

# ----------------------------------------
# 2. Hybrid Graph + Vector Retrieval
# ----------------------------------------
def hybrid_retrieve(query: str, client, G):
    
    """
    Performs:
      • entity extraction from query
      • graph search
      • vector DB retrieval
      • merging of results
    """

    # --- Extract entities from text ---
    ner = extract_entities_and_relations(query)
    entities = ner.get("entities", []) # [] is the default list returned 

    graph_results = []
    related_entities = set()

    # ---- 1. Try exact graph match ----

    for e in entities:
        # TODO: Use graph_query(G, query) instead and change next 13 lines of code accordingly
        exact = search_entity(G, e)
        if exact:
            # 1-hop relations
            hops1 = one_hop_relations(G, exact)

            # 2-hop neighbors
            hops2 = multi_hop_traversal(G, exact, depth=2)

            graph_results.append({
                "entity": exact,
                "one_hop": hops1,
                "two_hop": list(hops2)
            })

            # gather all nodes discovered
            related_entities.update([h["target"] for h in hops1] + [hs["source"] for hs in hops1])
            related_entities.update(hops2)

    # ---- 2. Keyword search fallback to check for the existence of a word within nodes/edges in the Network graph ----
    # TODO: Make this the default setting. Check for existence of keyworks within nodes and edges instead of checking for exact matches
    if not graph_results:
        for token in query.split():
            ks = keyword_search(G, token)
            if ks["nodes"] or ks["edges"]:
                graph_results.append({"keyword_matches": ks})
                related_entities.update(ks["nodes"])
                if ks["edges"]:
                    related_entities.update([t["target"] for t in ks["edges"]] + [s["source"] for s in ks["edges"]])
          

    # ---- 3. Vector DB search ----
    vector_hits = vector_search(
        query=query, # nodes related to the query within Knowledge Graph
        client=client, # Qdrant Point texts similar to the query and Qdrant Point texts similar to other texts related to the query
                        # (extracted from nodes neighboring the query entity in Knowledge Graph)
    )

    # also expand vector search using related graph entities
    for ent in related_entities:
        extra_hits = vector_search(
            query=ent,
            client=client,
            k=2
        )
        vector_hits.extend(extra_hits)

    # remove duplicates from vector_hits
    #seen = set()
    final_context = list(set(vector_hits))
   # for hit in vector_hits:
    #    text = hit
     #   if text not in seen:
      #      seen.add(text)
       #     final_context.append(text)

    # Final merged result
    return {
        "graph": graph_results,
        "contexts": final_context
    }

# ----------------------------------------
# 3. Main RAG Orchestration Function
# ----------------------------------------
def orchestrate(query: str, client, G):
    """
    Full hybrid retrieval + LLM answer.
    """

    query_type = classify_query(query)

    # --- Summarization → Only vector search ---
    if query_type == "summarization":
        vec_only = vector_search(query=query, client=client, k=8)
        context = "\n".join(hit.payload["text"] for hit in vec_only)
        prompt = f"""
    You are a helpful Assistant. Answer the user query using the context below.

    CONTEXT:
    {context}

    QUERY:
    {query}

    Be accurate and concise. Do not hallucinate.
    """
        response = genai.GenerativeModel("gemini-2.5-flash").generate_content(prompt).text
        print("Gemini API call made orchestrator.py --> orchestrate")
        return {"response":response,"context":[hit.payload["text"] for hit in vec_only]}

    # --- Lookup or cross-modal → Use hybrid graph + vector ---
    hybrid = hybrid_retrieve(query, client, G)
    context = "\n".join(hybrid["contexts"])

    # Combine graph info with context
    graph_context = str(hybrid["graph"])

    final_prompt = f"""
    You are a helpful Assistant. Answer the user query using the context below.

    CONTEXT:
    {context}

    GRAPH CONTEXT:
    {graph_context}

    QUERY:
    {query}

    Be accurate and concise. Do not hallucinate.
    """
    response = genai.GenerativeModel("gemini-2.5-flash").generate_content(final_prompt).text
    print("Gemini API call made from orchestrator.py --> orchestrate")
    return {"response":response,"context":hybrid["contexts"]}
