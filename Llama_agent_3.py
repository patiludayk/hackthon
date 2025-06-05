import os
import xml.etree.ElementTree as ET
import psycopg2
from typing import List, Dict, Any
import re
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
import tempfile

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.agent import ReActAgent
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core.tools import FunctionTool
from llama_index.core import Settings

# Load environment variables
load_dotenv()

# Azure PostgreSQL connection parameters
DB_HOST = os.getenv("PG_HOST")
DB_PORT = os.getenv("PG_PORT")
DB_NAME = os.getenv("PG_DATABASE")
DB_USER = os.getenv("PG_USER")
DB_PASSWORD = os.getenv("PG_PASSWORD")

# Azure OpenAI parameters
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-ada-002")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")

# Azure Blob Storage parameters
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

# Configure Azure OpenAI LLM
llm = AzureOpenAI(
    model=AZURE_OPENAI_DEPLOYMENT_NAME,
    deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION
)

# Configure Azure OpenAI Embeddings
embed_model = AzureOpenAIEmbedding(
    model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
    deployment_name=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION
)

# Set global settings
Settings.llm = llm
Settings.embed_model = embed_model

class FPMLProcessor:
    def __init__(self):
        # Connect to PostgreSQL
        self.conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        
        # Initialize vector store
        self.vector_store = PGVectorStore.from_params(
            database=DB_NAME,
            host=DB_HOST,
            password=DB_PASSWORD,
            port=int(DB_PORT),
            user=DB_USER,
            table_name="document_embeddings",
            embed_dim=1536  # Dimension for Azure OpenAI embeddings
        )
        
        # Initialize storage context and vector index
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            storage_context=self.storage_context,
            embed_model=embed_model
        )
        
        # Initialize Blob Storage client
        self.blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)

    def download_fpml_from_blob(self, blob_path: str) -> str:
        """
        Download FPML XML from Azure Blob Storage.
        
        Args:
            blob_path (str): Path to the blob in format 'container/blob_name'
            
        Returns:
            str: Content of the FPML XML file
        """
        try:
            # Split the blob path into container and blob name
            parts = blob_path.split('/', 1)
            if len(parts) != 2:
                return f"Invalid blob path: {blob_path}. Format should be 'container/blob_name'"
            
            container_name, blob_name = parts
            
            # Get container client
            container_client = self.blob_service_client.get_container_client(container_name)
            
            # Get blob client
            blob_client = container_client.get_blob_client(blob_name)
            
            # Download blob content
            download_stream = blob_client.download_blob()
            fpml_content = download_stream.readall().decode('utf-8')
            
            return fpml_content
        except Exception as e:
            return f"Error downloading blob: {str(e)}"

    def extract_value_from_xpath(self, fpml_xml: str, xpath: str) -> str:
        """
        Extract value from FPML XML using XPath.
        """
        try:
            # Parse XML
            root = ET.fromstring(fpml_xml)
            
            # Handle namespaces in FPML
            namespaces = {k: v for k, v in root.attrib.items() if k.startswith('xmlns')}
            
            # Convert namespaces to format expected by ElementTree
            ns = {}
            for prefix, uri in namespaces.items():
                if prefix.startswith('xmlns:'):
                    ns[prefix.split(':')[1]] = uri
                else:
                    ns[''] = uri
            
            # Register namespaces for XPath
            for prefix, uri in ns.items():
                if prefix:
                    ET.register_namespace(prefix, uri)
            
            # Adjust xpath for namespace handling if needed
            adjusted_xpath = xpath
            if ns:
                for prefix, uri in ns.items():
                    if prefix:
                        # Replace namespace prefixes with the {uri} format
                        adjusted_xpath = re.sub(f"{prefix}:", f"{{{uri}}}", adjusted_xpath)
            
            # Find elements using XPath
            elements = root.findall(adjusted_xpath)
            
            if not elements:
                return f"No elements found for XPath: {xpath}"
            
            # Extract values
            results = []
            for elem in elements:
                if elem.text and elem.text.strip():
                    results.append(elem.text.strip())
                else:
                    # If element has no text, get attributes or child element info
                    attribs = ", ".join([f"{k}='{v}'" for k, v in elem.attrib.items()])
                    results.append(f"Element with attributes: {attribs}" if attribs else "Empty element")
            
            return "; ".join(results)
        except Exception as e:
            return f"Error extracting value: {str(e)}"

    def get_relevant_xpaths(self, fpml_content: str) -> List[str]:
        """
        Query the vector database to get relevant XPaths for the given FPML content.
        """
        # Create a query document from the FPML content
        query_text = f"Find relevant XPaths for this FPML document: {fpml_content[:1000]}..."
        
        # Query the vector index
        query_engine = self.index.as_query_engine(similarity_top_k=10)
        response = query_engine.query(query_text)
        
        # Extract XPaths from response
        xpaths = []
        for node in response.source_nodes:
            # Extract XPath patterns from the content
            content = node.get_content()
            # Simple regex to find XPath-like patterns
            xpath_patterns = re.findall(r'(//?[\w:]+(?:/[\w:]+)*(?:\[@[\w:]+=[\'"][\w\s]+[\'"]?\])?)', content)
            xpaths.extend(xpath_patterns)
        
        return list(set(xpaths))  # Remove duplicates

    def process_fpml_blob(self, blob_path: str) -> Dict[str, Any]:
        """
        Process FPML XML from a blob by retrieving relevant XPaths and extracting values.
        """
        # Download FPML from blob storage
        fpml_xml = self.download_fpml_from_blob(blob_path)
        if fpml_xml.startswith("Error"):
            return {"error": fpml_xml}
        
        # Get relevant XPaths from vector database
        xpaths = self.get_relevant_xpaths(fpml_xml)
        
        # Extract values for each XPath
        results = {
            "xpaths": xpaths,
            "values": {}
        }
        
        for xpath in xpaths:
            value = self.extract_value_from_xpath(fpml_xml, xpath)
            results["values"][xpath] = value
        
        return results

    def analyze_xpaths(self, xpaths: List[str]) -> str:
        """
        Use LLM to analyze and explain the XPaths retrieved from the knowledgebase.
        """
        if not xpaths:
            return "No XPaths found in the knowledgebase for this FPML document."
        
        prompt = f"""
        Analyze the following XPaths retrieved from the knowledgebase for an FPML document:
        
        {', '.join(xpaths)}
        
        Please explain:
        1. What kind of financial data these XPaths are likely targeting
        2. The structure of the FPML document based on these XPaths
        3. Any important financial elements or attributes these XPaths are trying to extract
        """
        
        response = llm.complete(prompt)
        return response.text

    def close(self):
        """
        Close database connection.
        """
        if self.conn:
            self.conn.close()


def create_fpml_agent():
    # Initialize processor
    processor = FPMLProcessor()
    
    # Define tools
    def process_fpml_from_blob(blob_path: str) -> str:
        """
        Process an FPML XML document from Azure Blob Storage.
        
        Args:
            blob_path (str): Path to the blob in format 'container/blob_name'
            
        Returns:
            str: A detailed analysis of the FPML document with XPaths and their values
        """
        try:
            # Process the FPML document
            results = processor.process_fpml_blob(blob_path)
            
            if "error" in results:
                return results["error"]
            
            # Get XPaths and their values
            xpaths = results["xpaths"]
            values = results["values"]
            
            # Analyze XPaths using LLM
            xpath_analysis = processor.analyze_xpaths(xpaths)
            
            # Format the response
            response = f"## FPML Analysis for {blob_path}\n\n"
            response += f"### XPath Analysis\n{xpath_analysis}\n\n"
            response += "### Retrieved XPaths\n"
            for i, xpath in enumerate(xpaths, 1):
                response += f"{i}. `{xpath}`\n"
            
            response += "\n### Extracted Values\n"
            for xpath, value in values.items():
                response += f"\n**XPath**: `{xpath}`\n**Value**: {value}\n"
            
            return response
        except Exception as e:
            return f"Error processing FPML document: {str(e)}"
    
    def get_xpath_values(blob_path: str, xpath: str) -> str:
        """
        Extract values from an FPML XML document using a specific XPath.
        
        Args:
            blob_path (str): Path to the blob in format 'container/blob_name'
            xpath (str): The XPath to use for extraction
            
        Returns:
            str: The extracted values
        """
        try:
            # Download FPML from blob storage
            fpml_xml = processor.download_fpml_from_blob(blob_path)
            if fpml_xml.startswith("Error"):
                return fpml_xml
                
            value = processor.extract_value_from_xpath(fpml_xml, xpath)
            return f"Values for XPath '{xpath}' from {blob_path}: {value}"
        except Exception as e:
            return f"Error extracting values: {str(e)}"
    
    # Create function tools
    process_tool = FunctionTool.from_defaults(fn=process_fpml_from_blob)
    xpath_tool = FunctionTool.from_defaults(fn=get_xpath_values)
    
    # Create ReAct agent with Azure OpenAI
    agent = ReActAgent.from_tools(
        [process_tool, xpath_tool],
        llm=llm,
        verbose=True
    )
    
    return agent, processor


def main():
    print("Initializing FPML Agent...")
    agent, processor = create_fpml_agent()
    
    try:
        while True:
            blob_path = input("\nEnter blob path (container/blob_name) or 'exit' to quit: ")
            if blob_path.lower() == 'exit':
                break
                
            print(f"\nProcessing FPML from {blob_path}...")
            response = agent.chat(f"Process the FPML document from blob path: {blob_path}")
            print("\n" + str(response))
            
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        processor.close()


if __name__ == "__main__":
    main()