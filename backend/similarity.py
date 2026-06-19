"""
Similarity matching engine.

Pipeline:
1. Pull all Person descriptions from Neo4j
2. Encode each description into a vector (embedding) using a sentence-transformer model
3. Compute cosine similarity between every pair
4. For pairs above a threshold AND within a sensible age range,
   write a (:Person)-[:SIMILAR_TO {score}]->(:Person) edge back into Neo4j

We also factor in age difference and gender match as light guardrails,
so two completely unrelated people don't get linked just because their
text descriptions happen to read similarly.
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from db import run_query

MODEL_NAME = "all-MiniLM-L6-v2"   # small, fast, good enough for this use case
SIMILARITY_THRESHOLD = 0.78       # tune this — higher = stricter matches
MAX_AGE_GAP = 3                   # years; people more than this apart won't be linked

_model = None

def get_model():
    global _model
    if _model is None:
        print("Loading sentence-transformer model (first run only)...")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def fetch_persons():
    """Pull every person with a description from Neo4j."""
    return run_query("""
        MATCH (p:Person)
        WHERE p.description IS NOT NULL AND p.description <> ''
        RETURN p.id AS id, p.name AS name, p.age AS age,
               p.gender AS gender, p.description AS description
    """)


def compute_similarity_pairs(persons: list) -> list:
    """Return list of (person_a, person_b, score) above threshold and within guardrails."""
    model = get_model()
    descriptions = [p["description"] for p in persons]
    embeddings = model.encode(descriptions, show_progress_bar=False)

    sim_matrix = cosine_similarity(embeddings)
    pairs = []

    n = len(persons)
    for i in range(n):
        for j in range(i + 1, n):
            score = float(sim_matrix[i][j])
            if score < SIMILARITY_THRESHOLD:
                continue

            a, b = persons[i], persons[j]

            # guardrail: skip if ages are too far apart
            age_a, age_b = a.get("age") or 0, b.get("age") or 0
            if abs(age_a - age_b) > MAX_AGE_GAP:
                continue

            # guardrail: skip if genders differ (toggle off if you want cross-gender matches)
            if a.get("gender") and b.get("gender") and a["gender"] != b["gender"]:
                continue

            pairs.append((a["id"], b["id"], round(score, 4)))

    return pairs


def write_similarity_edges(pairs: list) -> int:
    """Write SIMILAR_TO relationships to Neo4j. Returns count written."""
    count = 0
    for person_a, person_b, score in pairs:
        run_query("""
            MATCH (a:Person {id: $a_id})
            MATCH (b:Person {id: $b_id})
            MERGE (a)-[r:SIMILAR_TO]-(b)
            SET r.score = $score
        """, {"a_id": person_a, "b_id": person_b, "score": score})
        count += 1
    return count


def run_similarity_matching() -> dict:
    """Full pipeline: fetch -> embed -> compare -> write edges."""
    persons = fetch_persons()
    if len(persons) < 2:
        return {"persons_checked": len(persons), "edges_created": 0, "message": "Not enough data"}

    pairs = compute_similarity_pairs(persons)
    edges_created = write_similarity_edges(pairs)

    return {
        "persons_checked": len(persons),
        "pairs_above_threshold": len(pairs),
        "edges_created": edges_created,
        "threshold_used": SIMILARITY_THRESHOLD,
    }


if __name__ == "__main__":
    result = run_similarity_matching()
    print(f"\n✅ Similarity matching done: {result}")