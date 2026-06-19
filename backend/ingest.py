import pandas as pd
import uuid
from db import run_query

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip().str.lower()
    df["name"]          = df["name"].str.strip().str.title()
    df["gender"]        = df["gender"].str.strip().str.capitalize()
    df["state"]         = df["state"].str.strip().str.title()
    df["district"]      = df["district"].str.strip().str.title()
    df["description"]   = df["description"].str.strip()
    df["status"]        = df["status"].str.strip().str.capitalize()
    df["date_reported"] = pd.to_datetime(df["date_reported"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["age"]           = pd.to_numeric(df["age"], errors="coerce").fillna(0).astype(int)
    df = df.drop_duplicates(subset=["case_id"])
    df = df.dropna(subset=["name", "district", "state"])
    return df

def ingest_case(row: dict):
    # 1. Merge Location node (upsert by district + state)
    run_query("""
        MERGE (l:Location {district: $district, state: $state})
    """, {"district": row["district"], "state": row["state"]})

    # 2. Merge Case node
    run_query("""
        MERGE (c:Case {id: $case_id})
        SET c.date_reported = $date_reported,
            c.status        = $status,
            c.source        = $source
    """, {
        "case_id":       row["case_id"],
        "date_reported": row["date_reported"],
        "status":        row["status"],
        "source":        row.get("source", "manual"),
    })

    # 3. Merge Person node
    run_query("""
        MERGE (p:Person {id: $person_id})
        SET p.name        = $name,
            p.age         = $age,
            p.gender      = $gender,
            p.description = $description,
            p.case_id     = $case_id
    """, {
        "person_id":   f"P-{row['case_id']}",
        "name":        row["name"],
        "age":         row["age"],
        "gender":      row["gender"],
        "description": row["description"],
        "case_id":     row["case_id"],
    })

    # 4. Relationships
    run_query("""
        MATCH (p:Person {id: $person_id})
        MATCH (c:Case   {id: $case_id})
        MERGE (p)-[:PART_OF]->(c)
    """, {"person_id": f"P-{row['case_id']}", "case_id": row["case_id"]})

    run_query("""
        MATCH (c:Case     {id: $case_id})
        MATCH (l:Location {district: $district, state: $state})
        MERGE (c)-[:REPORTED_IN]->(l)
    """, {"case_id": row["case_id"], "district": row["district"], "state": row["state"]})

def ingest_csv(filepath: str) -> dict:
    df = pd.read_csv(filepath)
    df = clean_dataframe(df)

    success, failed = 0, 0
    for _, row in df.iterrows():
        try:
            ingest_case(row.to_dict())
            success += 1
        except Exception as e:
            print(f"  ✗ Failed {row.get('case_id', '?')}: {e}")
            failed += 1

    return {"ingested": success, "failed": failed, "total": len(df)}

if __name__ == "__main__":
    result = ingest_csv("../data/sample_cases.csv")
    print(f"\n✅ Done — {result['ingested']} cases ingested, {result['failed']} failed")