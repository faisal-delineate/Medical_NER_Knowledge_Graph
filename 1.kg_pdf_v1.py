# Initial Version with Text Splitter and Graph Transformer custom nodes and relationships

from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
import os
from dotenv import load_dotenv
import logging
from datetime import datetime
from langchain_community.graphs.graph_document import GraphDocument

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QSPGraphSchema:
    """Schema definition for QSP domain"""
    ALLOWED_NODES = [
        "Drug", "Protein", "Pathway", "CellType", "Disease",
        "Biomarker", "Compartment", "Parameter", "Study",
        "Population", "Mechanism", "Outcome", "Intervention",
        "TimePoint", "DoseRegimen"
    ]
    
    # Relationships defined as tuples of (source_type, relationship_type, target_type)
    ALLOWED_RELATIONSHIPS = [
        f"{source}-{rel}->{target}"
        for source, rel, target in [
            ("Drug", "TARGETS", "Protein"),
            ("Drug", "AFFECTS", "Pathway"),
            ("Drug", "DISTRIBUTES_TO", "Compartment"),
            ("Drug", "HAS_MECHANISM", "Mechanism"),
            ("Drug", "HAS_REGIMEN", "DoseRegimen"),
            ("Protein", "EXPRESSED_IN", "CellType"),
            ("Protein", "PARTICIPATES_IN", "Pathway"),
            ("Protein", "REGULATES", "Protein"),
            ("Pathway", "INVOLVED_IN", "Disease"),
            ("Pathway", "LEADS_TO", "Outcome"),
            ("Biomarker", "INDICATES", "Disease"),
            ("Biomarker", "MEASURES", "Protein"),
            ("Parameter", "DESCRIBES", "Drug"),
            ("Parameter", "CHARACTERIZES", "Pathway"),
            ("Parameter", "MEASURED_AT", "TimePoint"),
            ("Study", "INVESTIGATES", "Drug"),
            ("Study", "INCLUDES", "Population"),
            ("Study", "MEASURES", "Outcome")
        ]
    ]
    
    # Node properties for each type
    NODE_PROPERTIES = {
        "Drug": [
            "name", "class", "mechanism", "half_life", "dosing_route",
            "bioavailability", "clearance_rate", "volume_distribution"
        ],
        "Protein": [
            "name", "type", "function", "location", "molecular_weight",
            "binding_constants", "expression_level"
        ],
        "Pathway": [
            "name", "regulation", "timeframe", "feedback_type",
            "rate_constants", "steady_state_behavior"
        ],
        "Parameter": [
            "name", "value", "units", "variability", "distribution_type",
            "confidence_interval", "measurement_method"
        ],
        "Study": [
            "name", "design", "duration", "population_size", 
            "inclusion_criteria", "exclusion_criteria"
        ],
        "Population": [
            "name", "size", "demographics", "disease_state",
            "inclusion_criteria"
        ],
        "Biomarker": [
            "name", "type", "measurement_method", "normal_range",
            "clinical_significance"
        ]
    }

class QSPEntity(BaseModel):
    """Base model for QSP entities with common properties"""
    id: str
    name: str
    description: Optional[str] = None
    references: Optional[List[str]] = None
    confidence_score: Optional[float] = None

class QSPDocumentProcessor:
    def __init__(self, anthropic_api_key: str):
        # self.llm = ChatOpenAI(
        #     temperature=0,
        #     model="gpt-4o",
        #     openai_api_key=openai_api_key
        # )
        self.llm = ChatAnthropic(
            temperature=0,
            model="claude-3-5-sonnet-20240620",
            anthropic_api_key=anthropic_api_key
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ";"],
            keep_separator=True
        )
        self.graph_transformer = self._setup_graph_transformer()
        self.graph = None

    def _setup_graph_transformer(self):
        """Set up the LLM Graph Transformer with QSP-specific configurations"""
        return LLMGraphTransformer(
            llm=self.llm,
            allowed_nodes=QSPGraphSchema.ALLOWED_NODES,
            allowed_relationships=QSPGraphSchema.ALLOWED_RELATIONSHIPS,
            node_properties=QSPGraphSchema.NODE_PROPERTIES
        )

    def connect_to_neo4j(self, uri: str, username: str, password: str):
        """Establish connection to Neo4j database"""
        try:
            self.graph = Neo4jGraph(
                url=uri,
                username=username,
                password=password,
                database="neo4j"
            )
            # Test connection
            self.graph.query("RETURN 1")
            logger.info("Successfully connected to Neo4j database")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            return False

    def process_pdf(self, pdf_path: str) -> List[Document]:
        """Process PDF and extract QSP-relevant chunks"""
        logger.info(f"Processing PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        chunks = self.text_splitter.split_documents(documents)
        for chunk in chunks:
            chunk.metadata.update({
                "source": pdf_path,
                "document_type": "QSP_Model",
                "extraction_timestamp": datetime.now().isoformat()
            })
        
        return chunks

    def create_knowledge_graph(self, chunks: List[Document]):
        """Create knowledge graph from document chunks"""
        if not self.graph:
            raise ValueError("Neo4j connection not established. Call connect_to_neo4j first.")

        try:
            self.graph.query("MATCH (n) DETACH DELETE n") #clear existing data to prevent conflicts
            
            logger.info("Converting documents to graph format...")
            graph_documents = self.graph_transformer.convert_to_graph_documents(chunks)
            
            self._create_constraints()
            
            logger.info("Storing in Neo4j...")
            processed_docs = []
            seen_nodes = set()
            
            for doc in graph_documents:
                unique_nodes = []
                unique_relationships = []
                
                for node in doc.nodes:
                    node_key = (node.id, node.type)
                    if node_key not in seen_nodes:
                        seen_nodes.add(node_key)
                        unique_nodes.append(node)
                
                for rel in doc.relationships:
                    source_key = (rel.source.id, rel.source.type)
                    target_key = (rel.target.id, rel.target.type)
                    if source_key in seen_nodes and target_key in seen_nodes:
                        unique_relationships.append(rel)
                
                new_doc = GraphDocument(
                    source=doc.source,
                    nodes=unique_nodes,
                    relationships=unique_relationships
                )
                processed_docs.append(new_doc)

            for doc in processed_docs:
                try:
                    self.graph.add_graph_documents(
                        [doc],
                        include_source=True,
                        baseEntityLabel=True
                    )
                except Exception as e:
                    logger.warning(f"Error processing document: {str(e)}")
                    continue
            
            self.graph.refresh_schema()
            logger.info("Knowledge graph created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create knowledge graph: {str(e)}")
            raise

    def _create_constraints(self):
        """Create necessary database constraints and indexes"""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Drug) REQUIRE d.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Protein) REQUIRE p.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Pathway) REQUIRE p.name IS UNIQUE",
            "CREATE INDEX IF NOT EXISTS FOR (d:Drug) ON (d.class)",
            "CREATE INDEX IF NOT EXISTS FOR (p:Parameter) ON (p.name)"
        ]
        
        for constraint in constraints:
            try:
                self.graph.query(constraint)
            except Exception as e:
                logger.warning(f"Error creating constraint: {str(e)}")

def main():
    load_dotenv()
    
    processor = QSPDocumentProcessor(os.getenv("ANTHROPIC_API_KEY"))
    
    connected = processor.connect_to_neo4j(
        uri="neo4j+s://ee22dc3b.databases.neo4j.io:7687",
        username="neo4j",
        password="ktmF5SZrRVU2wh9HDfYirhrgNt-HIxxBd8ap6tZlRYc"
    )
    
    if not connected:
        logger.error("Failed to connect to Neo4j. Exiting.")
        return
    
    pdf_path = "rosenstock2023.pdf"
    chunks = processor.process_pdf(pdf_path)
    
    processor.create_knowledge_graph(chunks)
    
    logger.info("Knowledge graph creation completed successfully")

if __name__ == "__main__":
    main()