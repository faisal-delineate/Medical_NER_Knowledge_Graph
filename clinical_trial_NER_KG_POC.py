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
from configurations import ALL_RELATIONS

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
    
    # print('entity: ',unique_entities)
    return unique_entities

# Generate relationships using OpenAI
def generate_relations(text: str, entities: List[Tuple[str, str]], openai_client, openai_model: str) -> List[Dict]:
    entities_str = "\n".join([
        f"{idx+1}. {entity} (Type: {entity_type})"
        for idx, (entity, entity_type) in enumerate(entities)
    ])
    
    prompt = f"""
        You are an expert pharmalogy researcher. You are given a text and a list of entities extracted from the text. Help the user to identify all possible relationships between the entities to prepare a knowledge graph. You can also suggest new entities that are not mentioned in the text but are important to understand the text.
 
        ### Text Context:
        {text}
 
        ### Extracted Entities:
        {entities_str}
 
        ### Guidelines:
        1. Use context to identify all relationships between entities, providing evidence from the text.
        2. Output relationships in JSON format, as shown below:
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
    relationship_types = ALL_RELATIONS
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
        relationship_type = relation["relationship"].replace(" ", "_").upper()  # Ensure valid Neo4j relationship format
        
        # Ensure the relationship type is dynamically inserted safely
        query = """
        MATCH (source:Entity {name: $source})
        MATCH (target:Entity {name: $target})
        MERGE (source)-[r:`%s`]->(target)
        SET r.evidence = $evidence
        """ % relationship_type  # Insert the dynamic relationship type safely
        
        tx.run(query, {
            "source": relation["source"],
            "target": relation["target"],
            "evidence": relation.get("evidence", "No specific evidence")
        })



def clear_neo4j_database(uri: str, username: str, password: str):
    """
    Deletes all nodes and relationships from the Neo4j database.
    """
    driver = GraphDatabase.driver(uri, auth=(username, password))
    with driver.session() as session:
        try:
            session.run("MATCH (n) DETACH DELETE n")
            print("Neo4j database cleared successfully.")
        finally:
            driver.close()


def process_text(text: str, ner_model, nlp, openai_client, openai_model, neo4j_config):
    entities = extract_entities(text, ner_model, nlp)
    relations = generate_relations(text, entities, openai_client, openai_model)
    # print('entities: ',entities)
    # pprint(entities)
    # print('relations: ')
    # pprint(relations)
    create_neo4j_graph(
        uri=neo4j_config["uri"],
        username=neo4j_config["username"],
        password=neo4j_config["password"],
        entities=entities,
        relations=relations
    )
   
    return entities, relations

def export_graph_to_csv(uri: str, username: str, password: str, output_file: str):
    """
    Export the entire Neo4j graph to a CSV file.
    """
    driver = GraphDatabase.driver(uri, auth=(username, password))
    query = """
    CALL apoc.export.csv.all($filePath, {useTypes: true, quotes: true})
    """
    with driver.session() as session:
        session.run(query, {"filePath": output_file})
        print(f"Graph exported to {output_file}.")
    driver.close()


if __name__ == "__main__":
    text_file_path = 'data/sample.txt'
    neo4j_url="bolt://localhost:7687"
    neo4j_username="neo4j"
    neo4j_password="123456789"
    ner_model="Clinical-AI-Apollo/Medical-NER"
    OPEN_AI_MODEL='gpt-3.5-turbo'


    # with open(text_file_path, 'r', encoding='utf-8') as f:
    #     text = f.read()

    ner_model = load_ner_model(ner_model)
    nlp = load_spacy_model()
    openai_client = OpenAI(api_key=API_KEY)
    neo4j_config = {
        "uri": neo4j_url,
        "username": neo4j_username,
        "password": neo4j_password
    }
    clear_neo4j_database(neo4j_config["uri"], neo4j_config["username"], neo4j_config["password"])
    # process_text(text, ner_model, nlp, openai_client, OPEN_AI_MODEL, neo4j_config)


    
    paper_text_dir='papers'
    # text_files=[ os.path.join(paper_text_dir,file) for file  in os.listdir(paper_text_dir)]
    text_files=[ paper_text_dir+'/'+file for file  in os.listdir(paper_text_dir)]
    # print(text_files)
    

    
    for idx, text_file in enumerate(text_files):
    # Process the text file
        db_name = f"db_{idx + 1}"
        neo4j_config = {
            "uri": neo4j_url,
            "username": neo4j_username,
            "password": neo4j_password,
            "database": db_name
        }
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        
        process_text(text, ner_model, nlp, openai_client, OPEN_AI_MODEL, neo4j_config=neo4j_config)

        # Export the graph
        output_file = f"graph_{idx + 1}.csv"
        print(output_file)
        # export_graph_to_csv(neo4j_config["uri"], neo4j_config["username"], neo4j_config["password"], output_file)
   



    
