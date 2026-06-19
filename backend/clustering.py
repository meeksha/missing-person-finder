"""
Cluster detection engine.

Pipeline:
1. Pull the SIMILAR_TO graph (Person <-> Person edges) from Neo4j
2. Build it as a networkx graph, weighted by similarity score
3. Run Louvain community detection to group connected people into clusters
4. For each cluster, pull case/location context and compute a "risk signal":
     - cluster size (more people = stronger pattern)
     - number of distinct districts/states spanned (geographic spread is suspicious)
     - average similarity score (confidence)
5. Write cluster_id back onto each Person node, and return ranked clusters

We use AuraDB Free here (no Graph Data Science plugin), so the heavy
algorithm runs locally in Python via networkx instead of inside Neo4j.
"""

import networkx as nx
from db import run_query

MIN_CLUSTER_SIZE = 2   # clusters smaller than this are dropped (not a "pattern")


def fetch_similarity_graph():
    """Pull all SIMILAR_TO edges from Neo4j."""
    return run_query("""
        MATCH (a:Person)-[r:SIMILAR_TO]-(b:Person)
        RETURN a.id AS source, b.id AS target, r.score AS score
    """)


def build_networkx_graph(edges: list) -> nx.Graph:
    G = nx.Graph()
    for e in edges:
        G.add_edge(e["source"], e["target"], weight=e["score"])
    return G


def detect_communities(G: nx.Graph) -> list:
    """Returns list of sets, each set = one cluster of person IDs."""
    if G.number_of_edges() == 0:
        return []
    communities = nx.algorithms.community.louvain_communities(G, weight="weight", seed=42)
    return [c for c in communities if len(c) >= MIN_CLUSTER_SIZE]


def enrich_cluster(person_ids: list, cluster_id: int) -> dict:
    """Pull case + location context for a cluster and compute risk signals."""
    people = run_query("""
        MATCH (p:Person)-[:PART_OF]->(c:Case)-[:REPORTED_IN]->(l:Location)
        WHERE p.id IN $ids
        RETURN p.id AS id, p.name AS name, p.age AS age, p.gender AS gender,
               c.id AS case_id, c.date_reported AS date_reported, c.status AS status,
               l.district AS district, l.state AS state
    """, {"ids": list(person_ids)})

    districts = {p["district"] for p in people}
    states = {p["state"] for p in people}

    return {
        "cluster_id": cluster_id,
        "size": len(people),
        "members": people,
        "districts_spanned": sorted(districts),
        "states_spanned": sorted(states),
        "geographic_spread": len(districts),
        "risk_level": classify_risk(len(people), len(districts), len(states)),
    }


def classify_risk(size: int, district_count: int, state_count: int) -> str:
    """Simple heuristic — tune this as you get real data."""
    if size >= 4 and (district_count >= 3 or state_count >= 2):
        return "HIGH"
    if size >= 3 and district_count >= 2:
        return "MEDIUM"
    return "LOW"


def write_cluster_ids(clusters: list):
    """Write cluster_id back onto each Person node for fast future lookups."""
    for cluster in clusters:
        member_ids = [m["id"] for m in cluster["members"]]
        run_query("""
            MATCH (p:Person) WHERE p.id IN $ids
            SET p.cluster_id = $cluster_id
        """, {"ids": member_ids, "cluster_id": cluster["cluster_id"]})


def run_cluster_detection() -> dict:
    """Full pipeline: fetch graph -> detect communities -> enrich -> write back."""
    edges = fetch_similarity_graph()
    if not edges:
        return {"clusters_found": 0, "message": "No similarity edges yet — run /match/similarity first"}

    G = build_networkx_graph(edges)
    raw_communities = detect_communities(G)

    clusters = []
    for idx, community in enumerate(raw_communities):
        clusters.append(enrich_cluster(community, cluster_id=idx))

    # rank highest risk first
    risk_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    clusters.sort(key=lambda c: (risk_order[c["risk_level"]], -c["size"]))

    write_cluster_ids(clusters)

    return {
        "clusters_found": len(clusters),
        "high_risk": sum(1 for c in clusters if c["risk_level"] == "HIGH"),
        "medium_risk": sum(1 for c in clusters if c["risk_level"] == "MEDIUM"),
        "low_risk": sum(1 for c in clusters if c["risk_level"] == "LOW"),
        "clusters": clusters,
    }


if __name__ == "__main__":
    result = run_cluster_detection()
    print(f"\n✅ Cluster detection done: {result['clusters_found']} clusters found")
    for c in result.get("clusters", []):
        print(f"  [{c['risk_level']}] Cluster {c['cluster_id']}: {c['size']} people across {c['districts_spanned']}")