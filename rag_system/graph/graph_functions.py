import pickle
import os
#os.chdir("/content/drive/MyDrive/AI_Projects/multimodal_enterprise_rag/rag_system/graph")
with open("knowledge_graph.gpickle", "rb") as f:
  G = pickle.load(f)

def list_all_entities():
  return list(G.nodes())

def list_all_relations(entity=None):
  if entity is None:
    relations = []
    for u, v, data in G.edges(data=True):
      relations.append(f"{u} —[{data['relation']}]-> {v}")
    return relations
  else:
    return list(G.out_edges(entity, data=True))

def neighbours(entity):
  return list(G.neighbors(entity))

for u, v, data in G.edges(data=True):
    print(f"{u} —[{data['relation']}]-> {v}")

def get_graph():
  return G