import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from typing import List, Dict
import logging
from langchain_openai import ChatOpenAI
import pandas as pd
from configurations import ENTITIES, ALL_RELATIONS

# Configure logging
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

        self.entities = ENTITIES
        self.relationships = ALL_RELATIONS

        self.llm = ChatOpenAI(
            temperature=0.1,
            model="gpt-4o",
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
            logger.info("Successfully connected to Neo4j.")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            return False

    def close(self):
        """Close Neo4j database connection."""
        if self.session:
            self.session.close()
            self.session = None
        if self.driver:
            self.driver.close()
            self.driver = None
        logger.info("Neo4j connection closed.")

    def generate_cypher_query(self, question: str, exclude_entities: List[str] = []) -> str:
        """
        Generate a Cypher query dynamically based on the input question.
        """
        try:
            entities = [
                entity for entity in self.entities if entity not in exclude_entities
            ]
            relationships = self.relationships

            prompt = f"""
            Based on the following question, generate a Cypher query that uses only the entities and relationships provided below.

            Question: {question}

            ENTITIES: {', '.join(entities)}
            RELATIONSHIPS: {', '.join(relationships)}

            Requirements:
            - Use only the provided entities and relationships.
            - If an entity value doesn't match the type, adjust the query to search iteratively for other possible matches.
            - Assume entities may have a property 'name' or 'type' for filtering or returning results.

            Example format for the query:
            ```cypher
            MATCH (e1:ENTITY1)-[r:RELATIONSHIP]->(e2:ENTITY2)
            RETURN e1.name AS Entity1, type(r) AS Relationship, e2.name AS Entity2
            LIMIT 50
            ```
            """
            response = self.llm.invoke(prompt).content.strip()
            logger.info(f"Generated Cypher Query: {response}")
            if "```cypher" in response:
                query_start = response.find("```cypher") + len("```cypher")
                query_end = response.find("```", query_start)
                return response[query_start:query_end].strip()
            else:
                logger.warning("No valid Cypher query found in LLM response.")
                return None
        except Exception as e:
            logger.error(f"Error during Cypher query generation: {e}")
            return None

    def query(self, question: str, max_iterations: int = 10) -> str:
        """Enhanced question answering method using Neo4j data with iterative query generation."""
        if not self.session:
            logger.error("No active Neo4j session.")
            return "Database connection is not established."

        excluded_entities = []
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            try:
                logger.info(f"Iteration {iteration}: Generating query...")
                cypher_query = self.generate_cypher_query(question, exclude_entities=excluded_entities)
                if not cypher_query:
                    return f"Failed to generate a valid Cypher query for the question: {question}"

                logger.info(f"Executing Cypher Query:\n{cypher_query}")
                result = self.session.run(cypher_query)
                records = list(result)

                if not records:
                    logger.info(f"No data found with the current query. Attempting another iteration...")
                    excluded_entities.append(cypher_query.split(':')[1].split(')')[0])  # Exclude the entity type used in the query
                    continue

                # Check if result only contains entity values
                if "EntityName" in records[0].keys():
                    entity_values = [record["EntityName"] for record in records]
                    return f"Matching Entities:\n" + "\n".join(entity_values)

                # Format results into graph-style answer
                graph_data = "\n".join([
                    f"{record.get('Entity1', 'Unknown')} -[{record.get('Relationship', 'Unknown')}] -> {record.get('Entity2', 'Unknown')}"
                    for record in records
                ])

                # Refine the answer using LLM
                prompt = f"""
                Based on the following graph data, answer the user's question as concisely as possible:

                Question: {question}

                Graph Data:
                {graph_data}

                Answer:
                """
                refined_response = self.llm.invoke(prompt).content.strip()
                return refined_response or f"Raw Graph Data:\n{graph_data}"

            except Exception as e:
                logger.error(f"Error during query execution: {e}")
                return f"An error occurred while processing the question: {e}"

        return f"No relevant data found for the question after {max_iterations} attempts."

def main():
    load_dotenv(override=True)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    neo4j_uri = "bolt://localhost:7687"
    neo4j_username = "neo4j"
    neo4j_password = "123456789"

    qsp_path = 'outputs/Paper_16_Evaluation.csv'
    qsp_df = pd.read_csv(qsp_path)
    all_questions = list(qsp_df['Question'])

    all_questions = ['What is body mass index?']
    all_answers = []

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
            for question in [all_questions[0]]:
                print(f"\nQuestion: {question}")
                response = qa_system.query(question)
                print(f"Response: {response}")
                all_answers.append(response)
        finally:
            qa_system.close()
    else:
        print("Failed to connect to Neo4j.")

    print('all_answers:', all_answers)
    # qsp_df['NER_MODEL_ANSWER'] = all_answers
    # qsp_df.to_csv('outputs/result_paper_16.csv', index=False)

if __name__ == "__main__":
    main()
