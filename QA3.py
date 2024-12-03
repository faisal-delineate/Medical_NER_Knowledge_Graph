import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from typing import List, Dict
import logging
from langchain_openai import ChatOpenAI
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

    def generate_cypher_query(self, question: str) -> str:
        """
        Generate a Cypher query dynamically using LLM based on the input question.
        """
        try:
            prompt = f"""
            Analyze the following question and generate an appropriate Cypher query:
            
            Question: {question}

            Output the Cypher query in a code block like this:
            ```cypher
            MATCH (n) RETURN n LIMIT 5
            ```
            """
            response = self.llm.invoke(prompt).content.strip()
            logger.info(f"Generated Cypher Query: {response}")
            # Extract the Cypher query from the response
            query_start = response.find("```cypher") + len("```cypher")
            query_end = response.find("```", query_start)
            return response[query_start:query_end].strip()
        except Exception as e:
            logger.error(f"Error during Cypher query generation: {e}")
            return "MATCH (n) RETURN n LIMIT 5"

    def query(self, question: str) -> str:
        """Enhanced question answering method using Neo4j data."""
        if not self.session:
            logger.error("No active Neo4j session.")
            return "Database connection is not established."

        try:
            # Generate Cypher query dynamically
            cypher_query = self.generate_cypher_query(question)

            # Execute the Cypher query
            result = self.session.run(cypher_query)
            records = list(result)

            if not records:
                logger.info("No matching data found for the question.")
                return f"No relevant data for the question: {question}"

            # Convert graph data into a meaningful format for the answer
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
            return refined_response

        except Exception as e:
            logger.error(f"Error during query execution: {e}")
            return f"An error occurred while processing the question: {e}"


def main():
    # Load environment variables
    load_dotenv(override=True)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    neo4j_uri = "bolt://localhost:7687"
    neo4j_username = "neo4j"
    neo4j_password = "123456789"

    qsp_path = 'outputs/Paper_16_Evaluation.csv'
    qsp_df = pd.read_csv(qsp_path)
    all_questions = list(qsp_df['Question'])
    
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
