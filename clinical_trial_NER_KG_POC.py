import os
import json
from typing import List, Tuple, Dict
from transformers import pipeline
from openai import OpenAI
from pprint import pprint
import spacy
import networkx as nx
import matplotlib.pyplot as plt
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

# Load environment variables
API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize NER model
def load_ner_model(model_name: str):
    return pipeline(
        "token-classification",
        model=model_name,
        aggregation_strategy="simple"
    )

# Load spaCy model
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spaCy language model...")
        spacy.cli.download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

# Extract entities using both models
def extract_entities(text: str, ner_model, nlp) -> List[Tuple[str, str]]:
    ner_results = ner_model(text)
    entities = [(res["word"], res["entity_group"]) for res in ner_results]
    
    doc = nlp(text)
    spacy_entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    all_entities = entities + spacy_entities
    unique_entities = []
    seen = set()
    for entity, entity_type in all_entities:
        if entity.lower() not in seen:
            unique_entities.append((entity, entity_type))
            seen.add(entity.lower())
    return unique_entities

# Generate relationships using OpenAI
def generate_relations(text: str, entities: List[Tuple[str, str]], openai_client, openai_model: str) -> List[Dict]:
    entities_str = "\n".join([
        f"{idx+1}. {entity} (Type: {entity_type})"
        for idx, (entity, entity_type) in enumerate(entities)
    ])
    
    prompt = f"""
        You are a medical relationship extraction assistant. Your task is to identify all possible relationships between entities mentioned in the text and output them in structured JSON format.

        ### Text Context:
        {text}

        ### Extracted Entities:
        {entities_str}

        ### Guidelines:
        1. Extract relationships such as:
           - TREATS, ASSOCIATED_WITH, CAUSES, MEASURED_BY, LOCATED_IN, FUNDED_BY, REGULATED_BY.
        2. Use context to identify all relationships between entities, providing evidence from the text.
        3. Output relationships in JSON format, as shown below:
        [
            {{
                "source": "Entity1",
                "source_type": "Type1",
                "target": "Entity2",
                "target_type": "Type2",
                "relationship": "RELATIONSHIP_TYPE",
                "evidence": "Evidence from the text"
            }},
            ...
        ]
    """

    try:
        response = openai_client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": "You are a medical relationship extraction expert."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.7
        )
        raw_content = response.choices[0].message.content.strip()
        print("Raw OpenAI Response:", raw_content)  # Debugging

        # Parse the response to ensure it's valid JSON
        relations = json.loads(raw_content)
        if not isinstance(relations, list):
            raise ValueError("Response is not a valid list of relationships.")
        return relations

    except Exception as e:
        print(f"Error in extracting relationships: {e}")
        return generate_fallback_relations(entities)



def generate_fallback_relations(entities: List[Tuple[str, str]]) -> List[Dict]:
    fallback_relations = []
    relationship_types = ["ASSOCIATED_WITH", "USED_FOR", "TREATS", "CAUSES", "INFLUENCES","MEASURED_BY","LOCATED_IN","FUNDED_BY","REGULATED_BY"]
    for i in range(len(entities) - 1):
        source, source_type = entities[i]
        target, target_type = entities[i + 1]
        fallback_relations.append({
            "source": source,
            "source_type": source_type,
            "target": target,
            "target_type": target_type,
            "relationship": relationship_types[i % len(relationship_types)],
            "evidence": "Fallback relationship based on entity proximity"
        })
    return fallback_relations


def create_neo4j_graph(uri: str, username: str, password: str, entities: List[Tuple[str, str]], relations: List[Dict]):
    driver = GraphDatabase.driver(uri, auth=(username, password))
    with driver.session() as session:
        try:
            # Add nodes and relationships
            session.execute_write(create_nodes, entities)
            session.execute_write(create_relationships, relations)
            print(f"Graph created with {len(entities)} nodes and {len(relations)} relationships.")
        finally:
            driver.close()


def create_nodes(tx, entities):
    for entity, entity_type in entities:
        tx.run(
            "MERGE (n:Entity {name: $name, type: $type})",
            {"name": entity, "type": entity_type}
        )

def create_relationships(tx, relations):
    """
    Create relationships in Neo4j with specific types.
    """
    for relation in relations:
        query = """
        MATCH (source:Entity {name: $source})
        MATCH (target:Entity {name: $target})
        MERGE (source)-[r:%s]->(target)
        SET r.evidence = $evidence
        """ % relation["relationship"].replace(" ", "_").upper()  # Ensure valid Neo4j relationship format
        
        tx.run(query, {
            "source": relation["source"],
            "target": relation["target"],
            "evidence": relation.get("evidence", "No specific evidence")
        })


def process_text(text: str, ner_model, nlp, openai_client, openai_model, neo4j_config):
    entities = extract_entities(text, ner_model, nlp)
    relations = generate_relations(text, entities, openai_client, openai_model)
    print('entities: ',entities)
    print('relations: ')
    pprint(relations)
    create_neo4j_graph(
        uri=neo4j_config["uri"],
        username=neo4j_config["username"],
        password=neo4j_config["password"],
        entities=entities,
        relations=relations
    )
   
    return entities, relations




if __name__ == "__main__":
    text_file_path = 'data/sample.txt'
    neo4j_url="bolt://localhost:7687"
    neo4j_username="neo4j"
    neo4j_password="123456789"
    ner_model="Clinical-AI-Apollo/Medical-NER"
    OPEN_AI_MODEL='gpt-3.5-turbo'


    with open(text_file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    ner_model = load_ner_model(ner_model)
    nlp = load_spacy_model()
    openai_client = OpenAI(api_key=API_KEY)
    neo4j_config = {
        "uri": neo4j_url,
        "username": neo4j_username,
        "password": neo4j_password
    }
    
    process_text(text, ner_model, nlp, openai_client, OPEN_AI_MODEL, neo4j_config)
