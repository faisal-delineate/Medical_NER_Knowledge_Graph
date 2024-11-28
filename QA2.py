from neo4j import GraphDatabase
import logging
from typing import List, Optional
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Entities(BaseModel):
    names: List[str] = Field(
        ...,
        description="Names of drugs, proteins, pathways, or other QSP entities mentioned",
    )

class QSPGraphQA:
    def __init__(self):
        self.driver = None

    def connect_to_neo4j(self, uri: str, username: str, password: str) -> bool:
        try:
            self.driver = GraphDatabase.driver(uri, auth=(username, password))
            with self.driver.session() as session:
                session.run("RETURN 1")  # Test connection
            logger.info("Connected to Neo4j successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            return False

    def close_connection(self):
        if self.driver:
            self.driver.close()

    def query_graph(self, cypher_query: str, parameters: Optional[dict] = None):
        try:
            with self.driver.session() as session:
                result = session.run(cypher_query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return []

    def _clean_query(self, text: str) -> str:
        text = text.replace('```cypher', '').replace('```', '').strip()
        if "MATCH" in text:
            text = "MATCH" + text.split("MATCH", 1)[1]
        if "RETURN" not in text:
            text += " RETURN *"
        return text.rstrip(';').strip()

    def query(self, question: str) -> str:
        logger.info(f"Processing question: {question}")
        # Generate Cypher query logic here
        cypher_query = self._clean_query("MATCH (n) RETURN n LIMIT 5")
        results = self.query_graph(cypher_query)
        return results or "No relevant information found."

def main():
    qa_system = QSPGraphQA()
    if not qa_system.connect_to_neo4j(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="123456789"
    ):
        print("Failed to connect to Neo4j. Exiting.")
        return

    try:
        print("QSP Knowledge Graph QA System")
        print("Type 'quit' to exit")

        while True:
            question = input("\nEnter your question: ").strip()
            if question.lower() in ['quit', 'exit']:
                break
            if question:
                answer = qa_system.query(question)
                print(f"\nAnswer: {answer}")
    finally:
        qa_system.close_connection()

if __name__ == "__main__":
    main()
