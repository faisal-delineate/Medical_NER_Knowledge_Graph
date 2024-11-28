from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
import os
from dotenv import load_dotenv
import logging
import re

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
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            temperature=0,
            model="gpt-4o",
            api_key=openai_api_key
        )
        self.graph = None
        
    def connect_to_neo4j(self, uri: str, username: str, password: str) -> bool:
        try:
            self.graph = Neo4jGraph(
                url=uri,
                username=username,
                password=password
            )
            self.graph.refresh_schema()
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            return False

    def _clean_query(self, text: str) -> str:
        """Clean and validate the generated Cypher query."""
        try:
            # Remove any markdown code block syntax
            text = text.replace('```cypher', '').replace('```', '').strip()
            
            # Ensure query starts with MATCH if present
            if "MATCH" in text:
                text = "MATCH" + text.split("MATCH", 1)[1]
            
            # Ensure there's a RETURN clause
            if not "RETURN" in text:
                text += " RETURN *"
            
            # Remove any trailing semicolons
            text = text.rstrip(';')
            
            logger.info(f"Cleaned Cypher query: {text}")
            return text.strip()
        except Exception as e:
            logger.error(f"Error cleaning query: {str(e)}")
            return "MATCH (n) RETURN n LIMIT 1"

    def _get_table_query(self, table_number: str) -> str:
        """Generate appropriate query based on how tables are structured in the database."""
        try:
            # First, check if we have a Document node with table content
            query = """
            MATCH (d:Document)
            WHERE d.text CONTAINS 'Table ' + $table_number
            OR d.title CONTAINS 'Table ' + $table_number
            OPTIONAL MATCH (d)-[:MENTIONS|CONTAINS]->(e:__Entity__)
            WITH d, collect(e) as entities
            RETURN d.title as title, d.text as content, entities
            """
            
            # Test the query with the table number
            test_result = self.graph.query(query.replace('$table_number', table_number))
            
            if test_result:
                return query
            
            # If no results, try a more general search
            return """
            MATCH (n)
            WHERE any(prop in keys(n) 
                WHERE toString(n[prop]) CONTAINS 'Table ' + $table_number)
            OPTIONAL MATCH (n)-[r]->(:__Entity__)
            RETURN n, collect(r) as relationships
            """
            
        except Exception as e:
            logger.error(f"Error determining table query: {str(e)}")
            return None

    def _get_outcome_query(self, is_primary: bool = True) -> str:
        """Generate query to find study outcomes."""
        return """
        MATCH (d:Document)
        WHERE (
            d.text CONTAINS 'primary outcome' OR 
            d.text CONTAINS 'primary endpoint' OR
            d.text CONTAINS 'primary efficacy' OR
            d.text CONTAINS 'primary objective' OR
            d.text CONTAINS 'main outcome' OR
            d.text CONTAINS 'findings'
        )
        OPTIONAL MATCH (d)-[:MENTIONS]->(e:__Entity__)
        WITH d, collect(e) as entities
        RETURN d.text as content, d.title as section, entities
        """

    def _validate_query(self, query: str) -> str:
        """Validate and fix common Cypher query issues."""
        try:
            # Check if query is about outcomes/findings
            # if any(term in query.lower() for term in ['outcome', 'endpoint', 'finding', 'result']):
            #     return self._get_outcome_query()
            
            # Check if query is about tables
            if "table" in query.lower():
                # Extract table number using regex
                table_match = re.search(r'table\s*(\d+)', query.lower())
                if table_match:
                    table_number = table_match.group(1)
                    table_query = self._get_table_query(table_number)
                    if table_query:
                        # Replace parameter with actual value since Neo4jGraph doesn't support parameters
                        return table_query.replace('$table_number', table_number)
            
            # Original query validation logic
            test_query = f"""
            {query}
            {'LIMIT 1' if 'LIMIT' not in query.upper() else ''}
            """
            self.graph.query(test_query)
            return query

        except Exception as e:
            logger.error(f"Query validation failed: {str(e)}")
            return """
            MATCH (d:Document)
            WHERE d.text CONTAINS 'outcome' OR d.text CONTAINS 'finding'
            RETURN d.title, d.text
            LIMIT 5
            """

    def setup_entity_extraction(self) -> None:
        """Setup entity extraction based on database structure."""
        # Get database structure for dynamic prompt construction
        db_structure = self._get_database_structure()
        
        # Build prompt based on available node types
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Extract key terms from the text that match the database structure.
                Return only the essential terms needed for searching, you can include similar term from your knowledge
                on medical clinical trial or system pharamacology related to get answer. 
                You can add "additional keywords"(few) outside of given query to extract more information
                One term should contain one word.
                Try to extract multiple terms if possible. Every term only one word"""),
            ("human", "Extract key terms from: {question}")
        ])
        
        self.entity_chain = prompt | self.llm.with_structured_output(Entities)

    def _get_property_info(self) -> str:
        """Get property information for each node type in the database."""
        try:
            query = """
            MATCH (n)
            WITH DISTINCT labels(n)[0] as label, 
                 properties(n) as props,
                 count(n) as count
            RETURN label, keys(props) as properties, count
            """
            results = self.graph.query(query)
            
            property_info = []
            for result in results:
                props = ", ".join(result["properties"])
                property_info.append(
                    f"- {result['label']} ({result['count']} nodes) properties: {props}"
                )
            
            return "\n".join(property_info)
        except Exception as e:
            logger.warning(f"Error getting property info: {str(e)}")
            return "Property information unavailable"

    def _get_relationship_patterns(self) -> str:
        """Get common relationship patterns from the database."""
        try:
            query = """
            MATCH (a)-[r]->(b)
            WITH DISTINCT labels(a)[0] as a_label,
                 type(r) as rel_type,
                 labels(b)[0] as b_label,
                 count(*) as freq
            ORDER BY freq DESC
            LIMIT 10
            RETURN a_label, rel_type, b_label, freq
            """
            results = self.graph.query(query)
            
            patterns = []
            for result in results:
                patterns.append(
                    f"- ({result['a_label']})-[:{result['rel_type']}]->({result['b_label']}) "
                    f"occurs {result['freq']} times"
                )
            
            return "\n".join(patterns)
        except Exception as e:
            logger.warning(f"Error getting relationship patterns: {str(e)}")
            return "Relationship patterns unavailable"

    def _get_common_paths(self) -> str:
        """Get common multi-hop paths from the database."""
        try:
            query = """
            MATCH (a)-[r1]->(b)-[r2]->(c)
            WITH DISTINCT labels(a)[0] as a_label,
                 type(r1) as r1_type,
                 labels(b)[0] as b_label,
                 type(r2) as r2_type,
                 labels(c)[0] as c_label,
                 count(*) as freq
            ORDER BY freq DESC
            LIMIT 5
            RETURN a_label, r1_type, b_label, r2_type, c_label, freq
            """
            results = self.graph.query(query)
            
            paths = []
            for result in results:
                paths.append(
                    f"- ({result['a_label']})-[:{result['r1_type']}]->"
                    f"({result['b_label']})-[:{result['r2_type']}]->"
                    f"({result['c_label']}) occurs {result['freq']} times"
                )
            
            return "\n".join(paths)
        except Exception as e:
            logger.warning(f"Error getting common paths: {str(e)}")
            return "Multi-hop paths unavailable"

    def _map_entities_to_db(self, entities: List[str]) -> str:
        """Map extracted entities to database nodes."""
        mappings = []
        for entity in entities:
            try:
                query = """
                MATCH (n)
                WHERE any(prop in keys(n) WHERE toString(n[prop]) CONTAINS $entity)
                RETURN DISTINCT labels(n)[0] as type, n
                LIMIT 1
                """
                result = self.graph.query(query, {"entity": entity})
                if result:
                    mappings.append(f"{entity} → {result[0]['type']}: {result[0]['n']}")
            except Exception as e:
                logger.warning(f"Error mapping entity {entity}: {str(e)}")
        
        return "\n".join(mappings) if mappings else "No entities mapped"

    def setup_cypher_generation(self) -> None:
        cypher_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at writing Cypher queries for a biomedical knowledge graph. 
Generate detailed queries that provide comprehensive information.

IMPORTANT GUIDELINES:
1. For outcome queries:
   - Look for primary and secondary outcomes
   - Include related findings and results for accuracy
   - Consider efficacy and safety endpoints
   - Search in study conclusions
2. For findings:
   - Include statistical results
   - Look for clinical significance
   - Consider related measurements
3. Return only the raw Cypher query"""),
            ("human", """Database Structure:
{schema}

Entity Mappings:
{entities_list}

Convert this question to a Cypher query: {question}""")
        ])

        self.cypher_chain = (
            RunnablePassthrough.assign(names=self.entity_chain)
            | RunnablePassthrough.assign(
                entities_list=lambda x: self._map_entities_to_db(x["names"]),
                schema=lambda _: self.graph.schema,
                property_info=lambda _: self._get_property_info(),
                rel_examples=lambda _: self._get_relationship_patterns(),
                path_examples=lambda _: self._get_common_paths()
            )
            | cypher_prompt
            | self.llm
            | StrOutputParser()
            | (lambda x: self._clean_query(x))
        )

    def setup_response_generation(self) -> None:
        response_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert clinical researcher responding to questions about clinical studies.            
                Guidelines:
                1. Focus on accuracy and clinical relevance
                2. Include specific values and measurements when available
                3. Explain medical terminology when needed
                4. Maintain professional tone
                5. Acknowledge data limitations when present"""),
            ("human", """Question: {question}
            Available Information: {context}
            Please provide an expert analysis aligning with the question and available information:""")
        ])
        
        self.response_chain = (
            RunnablePassthrough()
            | response_prompt
            | self.llm
            | StrOutputParser()
        )

    def _get_schema_info(self) -> dict:
        """Dynamically get database schema information."""
        try:
            # Get node labels and their properties
            node_info = self.graph.query("""
            MATCH (n)
            WITH DISTINCT labels(n)[0] as label, 
                 properties(n) as props
            RETURN label, collect(keys(props)) as properties
            """)
            
            # Get relationship types and their connections
            rel_info = self.graph.query("""
            MATCH (a)-[r]->(b)
            WITH DISTINCT labels(a)[0] as from_label, 
                 type(r) as rel_type,
                 labels(b)[0] as to_label
            RETURN from_label, rel_type, to_label
            """)
            
            return {
                "nodes": node_info,
                "relationships": rel_info
            }
        except Exception as e:
            logger.error(f"Error getting schema info: {str(e)}")
            return {}

    def _get_database_structure(self) -> dict:
        """Get actual database structure including relationships and properties."""
        try:
            # Get available node labels and their properties
            node_query = """
            MATCH (n)
            WITH DISTINCT labels(n) as labels, properties(n) as props
            RETURN labels, keys(props) as properties
            """
            nodes = self.graph.query(node_query)
            
            # Get available relationship types
            rel_query = """
            MATCH ()-[r]->()
            WITH DISTINCT type(r) as rel_type
            RETURN collect(rel_type) as relationships
            """
            relationships = self.graph.query(rel_query)
            
            return {"nodes": nodes, "relationships": relationships}
        except Exception as e:
            logger.error(f"Error getting database structure: {str(e)}")
            return {}

    def _build_search_query(self, question: str, terms: list) -> str:
        """Build comprehensive query based on database structure and question context."""
        try:
            # Start with a broad match across multiple node types
            query = """
            // Match documents containing the terms
            MATCH (d:Document)
            WHERE any(term IN $terms WHERE toLower(d.text) CONTAINS toLower(term))
            
            // Find connected entities
            OPTIONAL MATCH (d)-[r1]->(e1)
            OPTIONAL MATCH (e1)-[r2]->(e2)
            
            // Collect all related information
            WITH d, 
                 collect(DISTINCT {entity: e1, type: labels(e1)[0], rel: type(r1)}) as level1,
                 collect(DISTINCT {entity: e2, type: labels(e2)[0], rel: type(r2)}) as level2
            
            // Return comprehensive results
            RETURN d.text as doc_content,
                   d.title as doc_title,
                   level1,
                   level2
            """
            
            # Execute with parameters
            results = self.graph.query(
                query,
                {"terms": terms}
            )
            
            if not results:
                # Try a more general search if specific search fails
                backup_query = """
                MATCH (n)
                WHERE any(prop in keys(n) 
                    WHERE any(term IN $terms 
                        WHERE toLower(toString(n[prop])) CONTAINS toLower(term)))
                WITH n
                OPTIONAL MATCH (n)-[r]->(m)
                RETURN n, collect(DISTINCT {node: m, relationship: type(r)}) as related
                """
                results = self.graph.query(backup_query, {"terms": terms})
            
            return results
                
        except Exception as e:
            logger.error(f"Error building query: {str(e)}")
            return None

    def _extract_key_terms(self, question: str) -> list:
        """Extract key terms from the question."""
        try:
            # Use entity extraction chain
            entities = self.entity_chain.invoke({"question": question})
            
            # Process extracted terms
            terms = []
            if hasattr(entities, 'names'):
                terms.extend([str(term) for term in entities.names])
            
            # If no terms extracted, use key words from question
            if not terms:
                # Split question into words and remove common words
                words = question.lower().split()
                terms = [word for word in words if len(word) > 2]
            
            logger.info(f"Extracted terms: {terms}")
            return terms
            
        except Exception as e:
            logger.error(f"Error extracting terms: {str(e)}")
            return question.lower().split()

    def query(self, question: str) -> str:
        """Process question and return comprehensive answer."""
        try:
            logger.info(f"Processing question: {question}")
            
            # Extract key terms
            terms = self._extract_key_terms(question)
            if not terms:
                return "Unable to identify search terms from the question."
            
            # Get query results
            results = self._build_search_query(question, terms)
            if not results:
                return "No relevant information found in the knowledge graph."
            
            # Process and structure the results
            context = []
            for result in results:
                if isinstance(result, dict):
                    # Add document content
                    if 'doc_content' in result and result['doc_content']:
                        context.append(result['doc_content'])
                    
                    # Add related entity information
                    for level in ['level1', 'level2']:
                        if level in result and result[level]:
                            for item in result[level]:
                                if item and 'entity' in item and item['entity']:
                                    context.append(
                                        f"{item['type']}: {str(item['entity'])}"
                                    )
            
            if not context:
                return "No relevant content found in the search results."
            
            # Generate response using the comprehensive context
            response = self.response_chain.invoke({
                "question": question,
                "context": "\n".join(context)
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return "An error occurred while processing your question."

    def debug_schema(self):
        """Debug helper to print database structure"""
        try:
            # Get all labels
            labels = self.graph.query("""
            MATCH (n)
            WITH DISTINCT labels(n)[0] as label
            RETURN collect(label) as labels
            """)
            logger.info(f"Available labels: {labels}")
            
            # Get relationship types
            rels = self.graph.query("""
            MATCH ()-[r]->()
            WITH DISTINCT type(r) as rel_type
            RETURN collect(rel_type) as relationships
            """)
            logger.info(f"Available relationships: {rels}")
            
            # Sample of nodes with 'table' in properties
            table_nodes = self.graph.query("""
            MATCH (n)
            WHERE any(prop in keys(n) WHERE toString(n[prop]) CONTAINS 'Table')
            RETURN labels(n) as label, properties(n) as props
            LIMIT 5
            """)
            logger.info(f"Nodes containing 'table': {table_nodes}")
            
        except Exception as e:
            logger.error(f"Error debugging schema: {str(e)}")

def main():
    load_dotenv(override=True)
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OPENAI_API_KEY not found in environment variables")
        return
        
    qa_system = QSPGraphQA(openai_api_key)
    
    connected = qa_system.connect_to_neo4j( #rosenstock paper
        uri="neo4j+s://8dc60e3d.databases.neo4j.io:7687",
        username="neo4j",
        password="p_JzQrTXyyi_elfWlqhV8SKjBmUr7mTq4W2OZsvWrJo"
    )
    
    if not connected:
        print("Failed to connect to Neo4j. Exiting.")
        return
    
    try:
        qa_system.setup_entity_extraction()
        qa_system.setup_cypher_generation()
        qa_system.setup_response_generation()
        
        print("QSP Knowledge Graph QA System")
        print("Type 'quit' to exit")
        
        while True:
            question = input("\nEnter your question: ").strip()
            
            if question.lower() in ['quit', 'exit']:
                break
                
            if question:
                answer = qa_system.query(question)
                print(f"\nAnswer: {answer}")
                
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Exiting due to error.")

if __name__ == "__main__":
    main() 