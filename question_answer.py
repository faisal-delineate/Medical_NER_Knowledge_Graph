import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from typing import List, Optional
import logging
from langchain_openai import ChatOpenAI
from configurations import ENTITIES, ALL_RELATIONS
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QSPGraphQA:
    def __init__(self, neo4j_uri: str, neo4j_username: str, neo4j_password: str, openai_api_key: str):
        self.neo4j_uri = neo4j_uri
        self.neo4j_username = neo4j_username
        self.neo4j_password = neo4j_password

        self.llm = ChatOpenAI(
            temperature=0.5,
            model="gpt-3.5-turbo",
            api_key=openai_api_key
        )
        self.driver = None
        self.session = None

    def connect(self) -> bool:
        """Establish connection to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(
                self.neo4j_uri, 
                auth=(self.neo4j_username, self.neo4j_password)
            )
            self.session = self.driver.session()
            logger.info("Successfully connected to Neo4j")
            return True
        except Exception as e:
            logger.error(f"Neo4j connection failed: {e}")
            return False

    def close(self):
        """Close Neo4j database connection."""
        if self.session:
            self.session.close()
        if self.driver:
            self.driver.close()
        logger.info("Neo4j connection closed")

    def query(self, question: str) -> str:
        """Enhanced question answering method with dynamic entity and relationship matching."""
        if not self.session:
            logger.error("No active Neo4j session")
            return "Database connection is not established."

        try: 
            entities = ENTITIES
            relationships = ALL_RELATIONS

            # Prepare a Cypher query dynamically based on known entities and relationships
            cypher_query = """
            MATCH (e1)-[r]->(e2)
            WHERE 
                ANY(entity IN $entities WHERE entity IN labels(e1)) OR 
                ANY(entity IN $entities WHERE entity IN labels(e2)) OR 
                type(r) IN $relationships
            RETURN e1.name AS Entity1, type(r) AS Relationship, e2.name AS Entity2
            LIMIT 10
            """
            result = self.session.run(cypher_query, {"entities": entities, "relationships": relationships})
            records = list(result)

            # Handle no records found
            if not records:
                logger.info("No matching entities or relationships found for the question")
                return f"No relevant graph data found to answer the question: {question}"

            # Format retrieved records
            graph_data = "\n".join([
                f"{record['Entity1']} -[{record['Relationship']}]-> {record['Entity2']}"
                for record in records
            ])

            # Enhance the prompt for the LLM
            prompt = f"""
            You are an expert in analyzing graph databases. Below is a set of relationships retrieved from the database, 
            formatted as "Entity1 -[Relationship]-> Entity2". Use this data to comprehensively answer the user's question.

            Question: {question}

            Graph Data:
            {graph_data}

            If you cannot find a direct answer, analyze the data to provide a plausible inference or explanation. 
            Be concise and accurate in your response.

            Answer:
            """
            # Use the LLM to process the prompt
            refined_response = self.llm.invoke(prompt).content

            return refined_response

        except Exception as e:
            logger.error(f"Query processing error: {e}")
            return f"An error occurred while processing the question: {e}"


def main():
    # Load environment variables
    load_dotenv(override=True)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    neo4j_uri = "bolt://localhost:7687"
    neo4j_username = "neo4j"
    neo4j_password = "123456789"

    qsp_path='outputs/Paper_16_Evaluation.csv'
    qsp_df=pd.read_csv(qsp_path)
    all_questions=list(qsp_df['Question'])
    all_answers=[]

    if not openai_api_key or not neo4j_password:
        print("Error: Missing OpenAI or Neo4j credentials")
        return
    qa_system = QSPGraphQA(
        neo4j_uri=neo4j_uri,
        neo4j_username=neo4j_username,
        neo4j_password=neo4j_password,
        openai_api_key=openai_api_key
    )
    if qa_system.connect():
        try:
            for question in all_questions:
                print(f"\nQuestion: {question}")
                response = qa_system.query(question)
                print(f"Response: {response}")
                all_answers.append(response)

        finally:
            qa_system.close()
    else:
        print("Failed to connect to Neo4j")

    print('all_answers: ',all_answers)
    qsp_df['NER_MODEL_ANSWER']=all_answers
    qsp_df.to_csv('outputs/result_paper_16.csv',index=False)

if __name__ == "__main__":
    main()
