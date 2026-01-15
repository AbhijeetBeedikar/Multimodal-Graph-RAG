"""
Graph Search Module
--------------------
Provides:
1. Exact entity search
2. Relationship traversal (1-hop, multi-hop)
3. Keyword search inside graph nodes and edges
"""

import networkx as nx
from typing import List, Dict, Any, Set


# ------------------------------------------------------------
# EXACT ENTITY SEARCH
# ------------------------------------------------------------
def search_entity(G: nx.DiGraph, entity_name: str):
    """
    Return the exact entity node if it exists in the graph.
    """
    entity_name = entity_name.strip()
    if entity_name in G.nodes:
        return entity_name
    return None


# ------------------------------------------------------------
# RELATIONSHIP TRAVERSAL
# ------------------------------------------------------------
def one_hop_relations(G: nx.DiGraph, entity: str) -> List[Dict[str, str]]:
    """
    Returns all relationships where the entity is the source or target.
    Example output:
    [
      {"source": "A", "relation": "works_at", "target": "Company"}
    ]
    """
    relations = []

    # Outbound edges (entity -> other)
    for _, target, data in G.out_edges(entity, data=True):
        relations.append({
            "source": entity,
            "relation": data.get("relation", ""),
            "target": target
        })

    # Inbound edges (other -> entity)
    for source, _, data in G.in_edges(entity, data=True):
        relations.append({
            "source": source,
            "relation": data.get("relation", ""),
            "target": entity
        })

    return relations


def multi_hop_traversal(G: nx.DiGraph, start_entity: str, depth: int = 2) -> Set[str]:
    """
    BFS traversal up to 'depth' hops.
    Returns all entities reachable within k hops.
    """
    visited = set([start_entity])
    frontier = set([start_entity])

    for _ in range(depth):
        next_frontier = set()
        for node in frontier:
            # neighbors = outbound (G.neighbors returns outbound nodes by default) U inbound nodes (predecessors)
            neighbors = set(G.neighbors(node)) | set(G.predecessors(node)) 
            next_frontier |= neighbors
        
        next_frontier -= visited # already visited nodes must not be added to visited set
        visited |= next_frontier # all unique nodes in the _th depth are added to the visited set 
        frontier = next_frontier # becomes the set of next nodes to come from which we need to traverse through 1 more BFS iteration

    visited.remove(start_entity)
    return visited


# ------------------------------------------------------------
# KEYWORD SEARCH INSIDE GRAPH
# ------------------------------------------------------------
def keyword_search(G: nx.DiGraph, keyword: str) -> Dict[str, Any]:
    """
    Search for keyword in:
    - node names
    - node attributes
    - edge relation labels
    Returns all matched nodes + edges.
    """
    keyword = keyword.lower()
    matched_nodes = []
    matched_edges = []

    # Node search
    for node, attrs in G.nodes(data=True):
        if keyword in node.lower():
            matched_nodes.append(node)
            continue

        # Search inside node attributes
        for k, v in attrs.items():
            if isinstance(v, str) and keyword in v.lower():
                matched_nodes.append(node)
                break

    # Edge search
    for u, v, attrs in G.edges(data=True):
        relation = attrs.get("relation", "")
        if keyword in relation.lower():
            matched_edges.append({
                "source": u,
                "relation": relation,
                "target": v
            })

    return {
        "nodes": list(set(matched_nodes)), #set of nodes that contain the keyword
        "edges": matched_edges #set of edges that contain the keyword
    }


# ------------------------------------------------------------
# COMBINED GRAPH QUERY (optional helper)
# ------------------------------------------------------------
def graph_query(G: nx.DiGraph, query: str) -> Dict[str, Any]:
    """
    Helper function that:
    1. Tries exact entity search
    2. If not found → keyword search
    3. Returns local graph neighborhood if entity is found
    """
    result = {}

    #TODO: search_entity is a Redundant Operation. Should be removed
    #Exact entity match
    entity = search_entity(G, query)
    if entity:
        result["entity"] = entity
        result["one_hop"] = one_hop_relations(G, entity)
        result["multi_hop"] = list(multi_hop_traversal(G, entity, depth=2))
        return result

    # If no exact match → keyword search
    #TODO: Ensure that multi-hop traversal occurs here as well
    result["keyword_matches"] = keyword_search(G, query)
    return result
