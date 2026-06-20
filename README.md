# SETU — Missing Persons Pattern Intelligence

A knowledge-graph tool that surfaces statistically unusual clusters among missing-person reports for investigators to review — built with Neo4j, NLP similarity matching, and graph clustering.

> **Note on scope:** This tool does not find missing people. It flags textually/semantically similar case descriptions as *leads* for human investigators to manually verify. See [Limitations](#limitations) below.

---

## The Problem

Over 8 lakh people are reported missing in India every year, with roughly half remaining untraced. Each case is filed separately at a local police station — there is no system that cross-references a case in one district against a similar case in another. That correlation work currently depends entirely on a human noticing it, across hundreds of thousands of records.

## What It Does

1. **Ingest** — Missing-person case data (CSV) is parsed, cleaned, and stored as a knowledge graph in Neo4j: `Person` nodes connect to `Case` nodes, which connect to `Location` nodes.
2. **Match** — Each case's free-text description is converted into a semantic embedding (`sentence-transformers`). Descriptions worded differently but meaning the same thing ("thin boy, short black hair" vs. "slim male child, cropped dark hair") are correctly identified as similar. Matches above a similarity threshold (and within a sensible age/gender range) become `SIMILAR_TO` edges in the graph.
3. **Cluster** — Louvain community detection (via `networkx`) groups connected cases into clusters. A cluster of 3+ cases spread across multiple districts is a stronger signal than any single pair.
4. **Visualize** — A live dashboard shows the case network as an interactive D3.js graph, an alert feed ranking clusters by risk (HIGH/MEDIUM/LOW), and a searchable case table.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Graph database | Neo4j AuraDB |
| Backend API | FastAPI (Python) |
| NLP embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Similarity scoring | scikit-learn (cosine similarity) |
| Clustering | NetworkX (Louvain community detection) |
| Frontend | HTML / CSS / vanilla JS / D3.js |
| Data processing | pandas |

---

## Architecture

```
CSV upload
    │
    ▼
Ingestion + cleaning (FastAPI + pandas)
    │
    ▼
Neo4j AuraDB  ──(Person)-[:PART_OF]->(Case)-[:REPORTED_IN]->(Location)
    │
    ▼
NLP similarity matching (sentence-transformers + cosine similarity)
    │   writes (Person)-[:SIMILAR_TO {score}]-(Person)
    ▼
Cluster detection (NetworkX Louvain, run locally — AuraDB Free has no GDS plugin)
    │   writes cluster_id + risk_level back to Person nodes
    ▼
React-free dashboard (HTML + D3.js force-directed graph, alert feed, case table)
```

---

## Project Structure

```
missing-persons-finder/
├── backend/
│   ├── main.py              # FastAPI app + all API routes
│   ├── db.py                 # Neo4j connection handling
│   ├── models.py              # Graph schema (constraints + indexes)
│   ├── ingest.py               # CSV cleaning + ingestion logic
│   ├── similarity.py            # NLP embedding + similarity matching
│   ├── clustering.py             # Louvain community detection + risk scoring
│   ├── requirements.txt
│   └── .env.example              # Template for required environment variables
├── frontend/
│   └── index.html                 # Dashboard (graph viz, alerts, case table)
├── data/
│   ├── generate_dataset.py         # Synthetic test dataset generator
│   └── realistic_cases.csv          # Generated test data (155 cases, 4 planted clusters)
└── PROJECT_DESCRIPTION.md            # Full write-up for hackathon submission
```

---

## Setup — Local Development

### Prerequisites
- Python 3.11 (newer versions may have wheel-compatibility issues with `sentence-transformers`/`blis`)
- A free [Neo4j AuraDB](https://console.neo4j.io) instance

### Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

Create `backend/.env` (copy from `.env.example`) and fill in your AuraDB credentials:
```env
NEO4J_URI=neo4j+s://your-instance-id.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password_here
```

Run the server:
```bash
python main.py
```
API will be live at `http://localhost:8000` (interactive docs at `/docs`).

### Frontend

```bash
cd frontend
python -m http.server 5500
```
Open `http://localhost:5500` in your browser.

> If deploying, update the `API` constant at the top of `index.html`'s `<script>` to point at your deployed backend URL instead of `localhost:8000`.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/ingest/csv` | Upload and ingest a CSV of cases |
| `GET` | `/stats` | Total persons / cases / locations |
| `GET` | `/cases` | List all cases |
| `GET` | `/hotspots` | Districts with the most cases |
| `POST` | `/match/similarity` | Run NLP similarity matching, write `SIMILAR_TO` edges |
| `GET` | `/match/similar/{person_id}` | Get similar cases for one person |
| `GET` | `/match/all` | Review all similarity edges |
| `POST` | `/cluster/detect` | Run Louvain clustering + risk scoring |
| `GET` | `/cluster/{cluster_id}` | Get members of a specific cluster |
| `GET` | `/alerts` | Top risk clusters |
| `GET` | `/graph` | Graph data (nodes + edges) for visualization |
| `DELETE` | `/reset` | Wipe all data (testing only) |

---

## Validation

Rather than assume the pipeline works, we built a synthetic test dataset (`data/generate_dataset.py`) — 155 fictional cases calibrated to real NCRB-published demographic ratios (gender split, child proportion, state-wise case volume), with **4 deliberately planted multi-case clusters** hidden among realistic background noise. The pipeline correctly surfaced 3 of the 4 planted clusters as genuine multi-person groups, with no false HIGH-risk alerts on unrelated background cases — validating that the similarity threshold (0.78 cosine similarity) meaningfully separates real patterns from coincidental text overlap.

---

## Limitations

- Similarity is based on **textual description matching only** — no photos, biometrics, or verified identity data are used. A match is a lead, never a confirmation.
- Two unrelated people sharing a common physical description (e.g. "thin child, school uniform") can still be flagged; this is an inherent limitation of text-based matching, not a bug.
- The system is designed for **human-in-the-loop review**, not autonomous conclusions.
- Current dataset is synthetic, used to validate pipeline correctness — not real case data (real missing-person records are sensitive personal data and were not used, by design).

---

## Built For
FE: https://missing-person-finder-3d3l.vercel.app/
BE: https://missing-person-finder-production.up.railway.app

HACKHAZARDS '26 — Neo4j Track (Build Databases with AuraDB) · Public Systems, Governance & Civic Tech theme
