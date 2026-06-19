from db import run_query

"""
Graph schema:

NODES:
  (:Person   {id, name, age, gender, description, embedding})
  (:Case     {id, date_reported, status, source})
  (:Location {district, state})

RELATIONSHIPS:
  (:Person)-[:PART_OF]->(:Case)
  (:Case)-[:REPORTED_IN]->(:Location)
  (:Person)-[:SIMILAR_TO {score}]->(:Person)   ← added by NLP phase
"""

def create_schema():
    queries = [
        "CREATE CONSTRAINT person_id IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE",
        "CREATE CONSTRAINT case_id   IF NOT EXISTS FOR (c:Case)   REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT location  IF NOT EXISTS FOR (l:Location) REQUIRE (l.district, l.state) IS UNIQUE",
        "CREATE INDEX person_age     IF NOT EXISTS FOR (p:Person) ON (p.age)",
        "CREATE INDEX person_gender  IF NOT EXISTS FOR (p:Person) ON (p.gender)",
        "CREATE INDEX case_date      IF NOT EXISTS FOR (c:Case)   ON (c.date_reported)",
        "CREATE INDEX location_state IF NOT EXISTS FOR (l:Location) ON (l.state)",
    ]
    for q in queries:
        run_query(q)
        print(f"✓ {q[:60]}...")
    print("\n✅ Schema ready.")

if __name__ == "__main__":
    create_schema()