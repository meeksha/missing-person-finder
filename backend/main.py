from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil, os, tempfile

from db import run_query
from ingest import ingest_csv
from models import create_schema
from similarity import run_similarity_matching
from clustering import run_cluster_detection

app = FastAPI(title="Missing Persons Pattern Finder API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup():
    create_schema()
    print("✅ Neo4j schema initialized")

# ── Ingest CSV ──────────────────────────────────────────────────
@app.post("/ingest/csv")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files accepted")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    result = ingest_csv(tmp_path)
    os.unlink(tmp_path)
    return result

# ── Stats ────────────────────────────────────────────────────────
@app.get("/stats")
def get_stats():
    return {
        "persons":   run_query("MATCH (p:Person)   RETURN count(p) AS n")[0]["n"],
        "cases":     run_query("MATCH (c:Case)     RETURN count(c) AS n")[0]["n"],
        "locations": run_query("MATCH (l:Location) RETURN count(l) AS n")[0]["n"],
    }

# ── All cases (with location) ────────────────────────────────────
@app.get("/cases")
def get_cases(limit: int = 100):
    return run_query("""
        MATCH (p:Person)-[:PART_OF]->(c:Case)-[:REPORTED_IN]->(l:Location)
        RETURN p.id AS person_id, p.name AS name, p.age AS age,
               p.gender AS gender, p.description AS description,
               c.id AS case_id, c.date_reported AS date_reported, c.status AS status,
               l.district AS district, l.state AS state
        ORDER BY c.date_reported DESC
        LIMIT $limit
    """, {"limit": limit})

# ── Cases by state ───────────────────────────────────────────────
@app.get("/cases/state/{state}")
def get_cases_by_state(state: str):
    return run_query("""
        MATCH (p:Person)-[:PART_OF]->(c:Case)-[:REPORTED_IN]->(l:Location {state: $state})
        RETURN p.name AS name, p.age AS age, p.gender AS gender,
               c.id AS case_id, c.date_reported AS date, l.district AS district
        ORDER BY c.date_reported DESC
    """, {"state": state.title()})

# ── Cases by district ────────────────────────────────────────────
@app.get("/cases/district/{district}")
def get_cases_by_district(district: str):
    return run_query("""
        MATCH (p:Person)-[:PART_OF]->(c:Case)-[:REPORTED_IN]->(l:Location {district: $district})
        RETURN p.name AS name, p.age AS age, p.gender AS gender,
               c.id AS case_id, c.date_reported AS date, l.state AS state
        ORDER BY c.date_reported DESC
    """, {"district": district.title()})

# ── Hotspot districts (most cases) ──────────────────────────────
@app.get("/hotspots")
def get_hotspots(limit: int = 10):
    return run_query("""
        MATCH (c:Case)-[:REPORTED_IN]->(l:Location)
        RETURN l.district AS district, l.state AS state, count(c) AS case_count
        ORDER BY case_count DESC
        LIMIT $limit
    """, {"limit": limit})

# ── Graph data for D3 visualization ─────────────────────────────
@app.get("/graph")
def get_graph(limit: int = 200):
    nodes_raw = run_query("""
        MATCH (p:Person)-[:PART_OF]->(c:Case)-[:REPORTED_IN]->(l:Location)
        RETURN p.id AS id, p.name AS name, p.age AS age, p.gender AS gender,
               p.cluster_id AS cluster_id, c.id AS case_id, c.status AS status,
               c.date_reported AS date_reported,
               l.district AS district, l.state AS state
        LIMIT $limit
    """, {"limit": limit})

    edges_raw = run_query("""
        MATCH (a:Person)-[r:SIMILAR_TO]->(b:Person)
        RETURN a.id AS source, b.id AS target, r.score AS score
        LIMIT $limit
    """, {"limit": limit})

    return {"nodes": nodes_raw, "edges": edges_raw}

# ── Run NLP similarity matching ──────────────────────────────────
@app.post("/match/similarity")
def trigger_similarity_matching():
    """
    Runs the embedding + cosine similarity pipeline over all Person
    descriptions and writes SIMILAR_TO edges into Neo4j.
    Re-run this any time new data is ingested.
    """
    return run_similarity_matching()

# ── Get similar cases for a given person ─────────────────────────
@app.get("/match/similar/{person_id}")
def get_similar_cases(person_id: str):
    return run_query("""
        MATCH (p:Person {id: $person_id})-[r:SIMILAR_TO]-(other:Person)
        MATCH (other)-[:PART_OF]->(c:Case)-[:REPORTED_IN]->(l:Location)
        RETURN other.id AS person_id, other.name AS name, other.age AS age,
               other.gender AS gender, other.description AS description,
               r.score AS similarity_score,
               c.id AS case_id, l.district AS district, l.state AS state
        ORDER BY r.score DESC
    """, {"person_id": person_id})

# ── All similarity edges (for review/debugging) ──────────────────
@app.get("/match/all")
def get_all_similarity_edges(limit: int = 100):
    return run_query("""
        MATCH (a:Person)-[r:SIMILAR_TO]->(b:Person)
        RETURN a.id AS person_a, a.name AS name_a,
               b.id AS person_b, b.name AS name_b,
               r.score AS similarity_score
        ORDER BY r.score DESC
        LIMIT $limit
    """, {"limit": limit})

# ── Run cluster / pattern detection ──────────────────────────────
@app.post("/cluster/detect")
def trigger_cluster_detection():
    """
    Runs Louvain community detection over the SIMILAR_TO graph to find
    groups of 3+ related cases. Flags HIGH/MEDIUM/LOW risk based on
    cluster size and geographic spread. Re-run after new similarity matching.
    """
    return run_cluster_detection()

# ── Get a specific cluster's members ─────────────────────────────
@app.get("/cluster/{cluster_id}")
def get_cluster(cluster_id: int):
    return run_query("""
        MATCH (p:Person {cluster_id: $cluster_id})-[:PART_OF]->(c:Case)-[:REPORTED_IN]->(l:Location)
        RETURN p.id AS id, p.name AS name, p.age AS age, p.gender AS gender,
               c.id AS case_id, c.date_reported AS date_reported, c.status AS status,
               l.district AS district, l.state AS state
    """, {"cluster_id": cluster_id})

# ── Alert feed — top risk clusters ───────────────────────────────
@app.get("/alerts")
def get_alerts(limit: int = 10):
    return run_query("""
        MATCH (p:Person)
        WHERE p.cluster_id IS NOT NULL
        WITH p.cluster_id AS cluster_id, collect(p) AS members
        WHERE size(members) >= 2
        RETURN cluster_id, size(members) AS cluster_size
        ORDER BY cluster_size DESC
        LIMIT $limit
    """, {"limit": limit})

# ── Reset database (wipes everything — use before re-testing) ───
@app.delete("/reset")
def reset_database():
    run_query("MATCH (n) DETACH DELETE n")
    return {"status": "wiped", "message": "All nodes and relationships deleted"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)