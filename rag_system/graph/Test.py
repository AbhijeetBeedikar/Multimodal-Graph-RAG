
import pickle

with open("../../knowledge_graph.gpickle", "rb") as f:  # 'rb' for Read Binary
    G = pickle.load(f)