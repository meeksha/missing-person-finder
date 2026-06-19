from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()

URI      = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

def get_session():
    return driver.session()

def close():
    driver.close()

def run_query(query: str, params: dict = {}):
    with get_session() as session:
        result = session.run(query, params)
        return [record.data() for record in result]