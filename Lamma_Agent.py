import os
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional, Tuple
import re
import json
from dotenv import load_dotenv

# LlamaIndex imports
#from llama_index import (VectorStoreIndex, Document, ServiceContext, StorageContext)
#from llama_index.tools import BaseTool, FunctionTool
#from llama_index.agent import ReActAgent
#from llama_index.vector_stores import PGVectorStore
#from llama_index.llms import AzureOpenAI
#from llama_index.embeddings import AzureOpenAIEmbedding

# Updated imports for current LlamaIndex versions
# Updated imports for current LlamaIndex versions
from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.core import Settings
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")

# Azure PostgreSQL Configuration
PG_HOST = os.getenv("PG_HOST")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_DATABASE = os.getenv("PG_DATABASE")
PG_USER = os.getenv("PG_USER")
PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_TABLE_NAME = os.getenv("PG_TABLE_NAME", "xsd_knowledge_base")

class XMLEconomicProcessor:
    """Class to process XML data with focus on economic fields"""
    
    @staticmethod
    def parse_xml_string(xml_string: str) -> Optional[ET.Element]:
        """Parse XML string into an ElementTree object"""
        try:
            return ET.fromstring(xml_string)
        except ET.ParseError as e:
            print(f"Error parsing XML: {e}")
            return None
    
    @staticmethod
    def get_namespaces(root: ET.Element) -> Dict[str, str]:
        """Extract namespaces from XML root element"""
        nsmap = {}
        # Extract default namespace if present
        xmlns = root.attrib.get('xmlns')
        if xmlns:
            nsmap['default'] = xmlns
        
        # Extract other namespaces
        for key, value in root.attrib.items():
            if key.startswith('xmlns:'):
                prefix = key.split(':', 1)[1]
                nsmap[prefix] = value
        
        return nsmap
    
    @staticmethod
    def register_namespaces(namespaces: Dict[str, str]) -> None:
        """Register namespaces with ElementTree for XPath"""
        for prefix, uri in namespaces.items():
            if prefix != 'default':
                ET.register_namespace(prefix, uri)
    
    @staticmethod
    def extract_value_by_xpath(root: ET.Element, xpath: str, namespaces: Dict[str, str] = None) -> Optional[str]:
        """Extract text value using XPath from XML"""
        try:
            elements = root.findall(xpath, namespaces=namespaces)
            if elements and len(elements) > 0:
                if elements[0].text:
                    return elements[0].text.strip()
                else:
                    return None
            return None
        except Exception as e:
            print(f"Error extracting value with XPath '{xpath}': {e}")
            return None
    
    @staticmethod
    def extract_values_by_xpath(root: ET.Element, xpath: str, namespaces: Dict[str, str] = None) -> List[str]:
        """Extract multiple text values using XPath from XML"""
        try:
            elements = root.findall(xpath, namespaces=namespaces)
            return [e.text.strip() for e in elements if e.text and e.text.strip()]
        except Exception as e:
            print(f"Error extracting values with XPath '{xpath}': {e}")
            return []
    
    @staticmethod
    def extract_attribute_by_xpath(root: ET.Element, xpath: str, attr_name: str, 
                                   namespaces: Dict[str, str] = None) -> Optional[str]:
        """Extract attribute value using XPath from XML"""
        try:
            elements = root.findall(xpath, namespaces=namespaces)
            if elements and len(elements) > 0:
                return elements[0].get(attr_name)
            return None
        except Exception as e:
            print(f"Error extracting attribute with XPath '{xpath}': {e}")
            return None
    
    @staticmethod
    def extract_element_by_xpath(root: ET.Element, xpath: str, namespaces: Dict[str, str] = None) -> Optional[ET.Element]:
        """Extract element using XPath from XML"""
        try:
            elements = root.findall(xpath, namespaces=namespaces)
            if elements and len(elements) > 0:
                return elements[0]
            return None
        except Exception as e:
            print(f"Error extracting element with XPath '{xpath}': {e}")
            return None
    
    @staticmethod
    def element_to_dict(element: ET.Element) -> Dict[str, Any]:
        """Convert XML element to dictionary"""
        result = {}
        
        # Add tag name
        result["tag"] = element.tag.split("}")[-1] if "}" in element.tag else element.tag
        
        # Add attributes
        if element.attrib:
            result["attributes"] = dict(element.attrib)
        
        # Add text content if any
        if element.text and element.text.strip():
            result["value"] = element.text.strip()
        
        # Process child elements
        children = {}
        for child in element:
            child_tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            child_data = XMLEconomicProcessor.element_to_dict(child)
            
            # Handle case where we have multiple children with same tag
            if child_tag in children:
                if not isinstance(children[child_tag], list):
                    children[child_tag] = [children[child_tag]]
                children[child_tag].append(child_data)
            else:
                children[child_tag] = child_data
        
        if children:
            result["children"] = children
        
        return result
    
    @staticmethod
    def extract_economic_data_by_xpath(root: ET.Element, xpath: str, namespaces: Dict[str, str] = None) -> Dict[str, Any]:
        """Extract economic data using XPath from XML"""
        element = XMLEconomicProcessor.extract_element_by_xpath(root, xpath, namespaces)
        if element is None:
            return {}
        
        # For economic data, we want to extract both the value and any relevant attributes
        result = {}
        
        # Get the tag name without namespace
        tag_name = element.tag.split("}")[-1] if "}" in element.tag else element.tag
        
        # Extract text value
        if element.text and element.text.strip():
            result["value"] = element.text.strip()
        
        # Extract attributes
        if element.attrib:
            result["attributes"] = dict(element.attrib)
            
            # Special handling for currency attributes
            currency_attrs = ["currency", "currencyID", "currencyCode"]
            for attr in currency_attrs:
                if attr in element.attrib:
                    result["currency"] = element.attrib[attr]
        
        # If it's just a simple value with no attributes, simplify the output
        if len(result) == 1 and "value" in result:
            return {tag_name: result["value"]}
        
        return {tag_name: result}
    
    @staticmethod
    def normalize_xpath(xpath: str) -> str:
        """Normalize XPath by removing predicates and handling namespaces consistently"""
        # Replace namespace prefixes with wildcards if they cause issues
        normalized = re.sub(r'(\w+):', '*:', xpath)
        
        # Remove complex predicates but keep simple numeric ones
        normalized = re.sub(r'\[[^\d\]]+\]', '', normalized)
        
        return normalized

class EconomicXPathAgent:
    """Agent that extracts economic data from XML using XPaths from a PostgreSQL vector database"""
       
    def __init__(self):
        # Initialize LLM
        self.llm = AzureOpenAI(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
            context_window=16000  # Explicitly set context window size
        )
        
        # Initialize embeddings
        self.embed_model = AzureOpenAIEmbedding(
            model="text-embedding-ada-002",  # Use a standard model name for embeddings
            deployment_name=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION
        )
        
        # Set up global settings
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        
        # Initialize PostgreSQL vector store
        connection_string = f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}"
        self.vector_store = PGVectorStore.from_params(
            database=PG_DATABASE,
            host=PG_HOST,
            password=PG_PASSWORD,
            port=int(PG_PORT),
            user=PG_USER,
            table_name=PG_TABLE_NAME,
            embed_dim=1536  # Dimension for Azure OpenAI embeddings
        )
        
        # Create vector index
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            storage_context=storage_context
        )
        
        # Initialize XML processor
        self.xml_processor = XMLEconomicProcessor()
        
        # Create tools for the agent
        self.tools = self._create_tools()
        
        # Create a chat memory
        from llama_index.core.memory import ChatMemoryBuffer
        memory = ChatMemoryBuffer.from_defaults(token_limit=2000)
        
        # Initialize the agent with memory
        self.agent = ReActAgent.from_tools(
            self.tools,
            llm=self.llm,
            memory=memory,
            verbose=True
        )
    
    
    ############
    
##   def __init__(self):
##       # Initialize LLM
##       self.llm = AzureOpenAI(
##           model=AZURE_OPENAI_DEPLOYMENT_NAME,
##           deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
##           api_key=AZURE_OPENAI_API_KEY,
##           azure_endpoint=AZURE_OPENAI_ENDPOINT,
##           api_version=AZURE_OPENAI_API_VERSION
##       )
##       
##       # Initialize embeddings
##       self.embed_model = AzureOpenAIEmbedding(
##           model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
##           deployment_name=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
##           api_key=AZURE_OPENAI_API_KEY,
##           azure_endpoint=AZURE_OPENAI_ENDPOINT,
##           api_version=AZURE_OPENAI_API_VERSION
##       )
##       
##        # Set up service context
##        self.service_context = ServiceContext.from_defaults(
##            llm=self.llm,
##            embed_model=self.embed_model
##        )
##
##       # Set up global settings
##       Settings.llm = self.llm
##       Settings.embed_model = self.embed_model
##       
##       # Initialize PostgreSQL vector store
##       connection_string = f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}"
##       self.vector_store = PGVectorStore.from_params(
##           database=PG_DATABASE,
##           host=PG_HOST,
##           password=PG_PASSWORD,
##           port=int(PG_PORT),
##           user=PG_USER,
##           table_name=PG_TABLE_NAME,
##           embed_dim=3072  # Dimension for Azure OpenAI embeddings
##       )
##       
##       # Create vector index
##       storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
##       self.index = VectorStoreIndex.from_vector_store(
##           vector_store=self.vector_store,
##            service_context=self.service_context,
##           storage_context=storage_context
##       )
##       
##       # Initialize XML processor
##       self.xml_processor = XMLEconomicProcessor()
##       
##       # Create tools for the agent
##       self.tools = self._create_tools()
##       
##       # Initialize the agent
##       self.agent = ReActAgent.from_tools(
##           self.tools,
##           llm=self.llm,
##           verbose=True
##       )
        
        # Common economic XPaths (fallback if vector DB doesn't return good results)
        self.common_economic_xpaths = {
            "amount": [
                ".//Amount", 
                ".//MonetaryAmount", 
                ".//TotalAmount",
                ".//cbc:Amount",
                ".//cbc:PayableAmount"
            ],
            "currency": [
                ".//CurrencyCode", 
                ".//Currency",
                ".//cbc:DocumentCurrencyCode"
            ],
            "price": [
                ".//Price", 
                ".//UnitPrice",
                ".//cbc:PriceAmount"
            ],
            "quantity": [
                ".//Quantity",
                ".//cbc:InvoicedQuantity",
                ".//cbc:BaseQuantity"
            ],
            "total": [
                ".//Total",
                ".//TotalAmount",
                ".//cbc:PayableAmount",
                ".//cbc:TaxInclusiveAmount"
            ],
            "tax": [
                ".//Tax",
                ".//TaxAmount",
                ".//cbc:TaxAmount"
            ]
        }
    
    def _create_tools(self) -> List[BaseTool]:
        """Create tools for the agent to use"""
        tools = []
        
        # Tool to search the vector database for economic XPaths
        def search_economic_xpaths(xml_type: str, top_k: int = 10) -> str:
            """Search the PostgreSQL vector database for economic XPaths relevant to the XML type."""
            query = f"What are the XPaths for economic fields in {xml_type} XML documents?"
            query_engine = self.index.as_query_engine(similarity_top_k=top_k)
            response = query_engine.query(query)
            return str(response)
        
        search_tool = FunctionTool.from_defaults(
            fn=search_economic_xpaths,
            name="search_economic_xpaths",
            description="Search the PostgreSQL vector database for economic XPaths relevant to a specific XML type"
        )
        tools.append(search_tool)
        
        # Tool to extract economic data using XPath
        def extract_economic_data(xml_string: str, xpath: str) -> str:
            """Extract economic data using XPath from XML string."""
            root = self.xml_processor.parse_xml_string(xml_string)
            if not root:
                return "Error parsing XML"
            
            namespaces = self.xml_processor.get_namespaces(root)
            self.xml_processor.register_namespaces(namespaces)
            
            data = self.xml_processor.extract_economic_data_by_xpath(root, xpath, namespaces)
            
            if not data:
                return f"No economic data found for XPath '{xpath}'"
            
            return json.dumps(data, indent=2)
        
        extract_tool = FunctionTool.from_defaults(
            fn=extract_economic_data,
            name="extract_economic_data",
            description="Extract economic data using XPath from XML"
        )
        tools.append(extract_tool)
        
        # Tool to extract multiple values using XPath
        def extract_values(xml_string: str, xpath: str) -> str:
            """Extract multiple values using XPath from XML string."""
            root = self.xml_processor.parse_xml_string(xml_string)
            if not root:
                return "Error parsing XML"
            
            namespaces = self.xml_processor.get_namespaces(root)
            self.xml_processor.register_namespaces(namespaces)
            
            values = self.xml_processor.extract_values_by_xpath(root, xpath, namespaces)
            
            if not values:
                return f"No values found for XPath '{xpath}'"
            
            return json.dumps(values, indent=2)
        
        values_tool = FunctionTool.from_defaults(
            fn=extract_values,
            name="extract_values",
            description="Extract multiple values using XPath from XML"
        )
        tools.append(values_tool)
        
        # Tool to summarize economic data
        def summarize_economic_data(economic_data: str) -> str:
            """Summarize economic data extracted from XML."""
            prompt = f"""
            Please summarize the following economic data extracted from an XML document:
            
            {economic_data}
            
            Provide a comprehensive summary that includes all monetary values, quantities, prices, 
            and other economic indicators. Format currency values appropriately and highlight key totals.
            """
            
            response = self.llm.complete(prompt)
            return response.text
        
        summarize_tool = FunctionTool.from_defaults(
            fn=summarize_economic_data,
            name="summarize_economic_data",
            description="Summarize economic data extracted from XML"
        )
        tools.append(summarize_tool)
        
        return tools
    
    def extract_xpaths_from_context(self, context: str) -> List[str]:
        """Extract XPath expressions from context text"""
        # Look for XPath patterns in the text
        xpath_pattern = r'(?:xpath|XPath|path)(?:\s+is)?(?:\s*[:=]\s*|\s+)["\']((?:/|\.)[^"\'\n]+)["\']'
        xpaths = re.findall(xpath_pattern, context)
        
        # Also look for XML paths that might not be explicitly labeled as XPath
        xml_path_pattern = r'["\']((?://|\.//)[\w:*]+(?:/[\w:*]+)+)["\']'
        xml_paths = re.findall(xml_path_pattern, context)
        
        # Combine and deduplicate
        all_paths = list(set(xpaths + xml_paths))
        
        # Filter out paths that don't look like XPaths
        valid_paths = [p for p in all_paths if p.startswith(('.', '/'))]
        
        return valid_paths
    
    def process_xml_for_economic_data(self, xml_string: str, xml_type: str = "financial") -> Dict[str, Any]:
        """
        Process XML data to extract economic information using XPaths from the vector database
        
        Args:
            xml_string: The XML data as a string
            xml_type: The type of XML document (e.g., "invoice", "financial report", "order")
            
        Returns:
            Dictionary containing extracted economic data and summary
        """
        # Parse the XML
        root = self.xml_processor.parse_xml_string(xml_string)
        if not root:
            return {"error": "Failed to parse XML"}
        
        # Get namespaces
        namespaces = self.xml_processor.get_namespaces(root)
        self.xml_processor.register_namespaces(namespaces)
        
        # Get the root element name for better context
        root_tag = root.tag.split("}")[-1] if "}" in root.tag else root.tag
        
        # Search vector database for economic XPaths relevant to this XML type
        context_response = self.tools[0].fn(f"{root_tag} {xml_type}")
        
        # Extract XPaths from the context
        xpaths = self.extract_xpaths_from_context(context_response)
        
        # If no XPaths found, use common economic XPaths as fallback
        if not xpaths:
            xpaths = []
            for category, paths in self.common_economic_xpaths.items():
                xpaths.extend(paths)
        
        # Extract economic data for each XPath
        extracted_data = {}
        
        for xpath in xpaths:
            try:
                # Normalize the XPath to handle namespace issues
                normalized_xpath = self.xml_processor.normalize_xpath(xpath)
                
                # Try the original XPath first
                elements = root.findall(xpath, namespaces=namespaces)
                
                # If that fails, try the normalized version
                if not elements:
                    elements = root.findall(normalized_xpath, namespaces=namespaces)
                
                # Process each element found
                for element in elements:
                    # Get a clean tag name without namespace
                    tag_name = element.tag.split("}")[-1] if "}" in element.tag else element.tag
                    
                    # Create a key based on the XPath and tag
                    key = f"{tag_name}_at_{xpath}"
                    
                    # Extract the data
                    data = {}
                    
                    # Get text value
                    if element.text and element.text.strip():
                        data["value"] = element.text.strip()
                    
                    # Get attributes
                    if element.attrib:
                        data["attributes"] = dict(element.attrib)
                        
                        # Special handling for currency attributes
                        currency_attrs = ["currency", "currencyID", "currencyCode"]
                        for attr in currency_attrs:
                            if attr in element.attrib:
                                data["currency"] = element.attrib[attr]
                    
                    # If we have data, add it to the results
                    if data:
                        # Simplify the output if it's just a value
                        if len(data) == 1 and "value" in data:
                            extracted_data[key] = data["value"]
                        else:
                            extracted_data[key] = data
            except Exception as e:
                print(f"Error processing XPath '{xpath}': {e}")
        
        # If we still don't have data, try a more aggressive approach
        if not extracted_data:
            # Look for elements with economic-related names
            economic_terms = ["amount", "price", "cost", "total", "currency", "tax", "quantity", "value"]
            
            for term in economic_terms:
                # Try to find elements containing this term
                for element in root.findall(f".//*[contains(local-name(), '{term}')]"):
                    tag_name = element.tag.split("}")[-1] if "}" in element.tag else element.tag
                    
                    if element.text and element.text.strip():
                        extracted_data[tag_name] = element.text.strip()
        
        # Organize the data for better summarization
        organized_data = self._organize_economic_data(extracted_data)
        
        # Summarize the extracted data
        summary_prompt = f"""
        I've extracted the following economic data from a {xml_type} XML document with root element <{root_tag}>:
        
        {json.dumps(organized_data, indent=2)}
        
        Please provide a comprehensive summary of this economic information.
        Include all monetary values, quantities, prices, and other economic indicators.
        Format currency values appropriately and highlight key totals.
        If there are multiple items or line entries, summarize them clearly.
        """
        
        summary = self.llm.complete(summary_prompt)
        
        # Return both the extracted data and the summary
        return {
            "xml_type": xml_type,
            "root_element": root_tag,
            "extracted_economic_data": organized_data,
            "raw_extracted_data": extracted_data,
            "summary": summary.text,
            "xpaths_used": xpaths
        }
    
    def _organize_economic_data(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Organize extracted economic data into categories for better summarization"""
        organized = {
            "monetary_amounts": {},
            "quantities": {},
            "prices": {},
            "taxes": {},
            "totals": {},
            "other": {}
        }
        
        for key, value in extracted_data.items():
            key_lower = key.lower()
            
            # Determine category based on key name
            if any(term in key_lower for term in ["amount", "monetary", "cost"]):
                organized["monetary_amounts"][key] = value
            elif any(term in key_lower for term in ["quantity", "qty"]):
                organized["quantities"][key] = value
            elif any(term in key_lower for term in ["price"]):
                organized["prices"][key] = value
            elif any(term in key_lower for term in ["tax", "vat", "gst"]):
                organized["taxes"][key] = value
            elif any(term in key_lower for term in ["total", "sum", "payable"]):
                organized["totals"][key] = value
            else:
                organized["other"][key] = value
        
        # Remove empty categories
        return {k: v for k, v in organized.items() if v}
    
    def add_xpath_to_knowledge_base(self, xpath_info: str, xml_type: str) -> None:
        """Add economic XPath information to the PostgreSQL vector database"""
        document = Document(
            text=xpath_info, 
            metadata={"type": "economic_xpath", "xml_type": xml_type}
        )
        self.index.insert(document)
        print(f"Added economic XPath info for '{xml_type}' to knowledge base")
    
    def initialize_database(self) -> None:
        """Ensure the PostgreSQL database is properly set up with pgvector extension"""
        import psycopg2
        
        conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            database=PG_DATABASE,
            user=PG_USER,
            password=PG_PASSWORD
        )
        
        try:
            cursor = conn.cursor()
            # Create pgvector extension if not exists
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Check if our table exists
            cursor.execute(f"""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = '{PG_TABLE_NAME}'
            );
            """)
            
            table_exists = cursor.fetchone()[0]
            
            if not table_exists:
                print(f"Table {PG_TABLE_NAME} does not exist. It will be created by PGVectorStore.")
            
            conn.commit()
        except Exception as e:
            print(f"Error initializing database: {e}")
        finally:
            cursor.close()
            conn.close()

# Example usage
if __name__ == "__main__":
    # Initialize the agent
    agent = EconomicXPathAgent()
    
    # Ensure database is properly set up
    agent.initialize_database()
    
    # Add some economic XPath information to the knowledge base
    invoice_xpath_info = """
    Economic XPaths for UBL Invoice XML:
    
    1. Invoice Total Amount: xpath = ".//cbc:PayableAmount"
    2. Invoice Currency: xpath = ".//cbc:DocumentCurrencyCode"
    3. Line Item Amounts: xpath = ".//cac:InvoiceLine/cbc:LineExtensionAmount"
    4. Line Item Prices: xpath = ".//cac:InvoiceLine/cac:Price/cbc:PriceAmount"
    5. Line Item Quantities: xpath = ".//cac:InvoiceLine/cbc:InvoicedQuantity"
    6. Tax Exclusive Amount: xpath = ".//cbc:TaxExclusiveAmount"
    7. Tax Inclusive Amount: xpath = ".//cbc:TaxInclusiveAmount"
    8. Tax Amount: xpath = ".//cac:TaxTotal/cbc:TaxAmount"
    """
    
    agent.add_xpath_to_knowledge_base(invoice_xpath_info, "UBL Invoice")
    
    # Example XML document (a sample invoice)
    sample_xml = """
    <?xml version="1.0" encoding="UTF-8"?>
    <Invoice xmlns="urn:oasis:names:specification:ubl:schema:xsd:Invoice-2" 
             xmlns:cac="urn:oasis:names:specification:ubl:schema:xsd:CommonAggregateComponents-2" 
             xmlns:cbc="urn:oasis:names:specification:ubl:schema:xsd:CommonBasicComponents-2">
        <cbc:ID>INVOICE-001</cbc:ID>
        <cbc:IssueDate>2023-06-15</cbc:IssueDate>
        <cbc:InvoiceTypeCode>380</cbc:InvoiceTypeCode>
        <cbc:DocumentCurrencyCode>USD</cbc:DocumentCurrencyCode>
        
        <cac:AccountingSupplierParty>
            <cac:Party>
                <cac:PartyName>
                    <cbc:Name>Supplier Company Ltd</cbc:Name>
                </cac:PartyName>
                <cac:PostalAddress>
                    <cbc:StreetName>123 Supplier Street</cbc:StreetName>
                    <cbc:CityName>Supplier City</cbc:CityName>
                    <cbc:PostalZone>S12 3UP</cbc:PostalZone>
                    <cac:Country>
                        <cbc:IdentificationCode>US</cbc:IdentificationCode>
                    </cac:Country>
                </cac:PostalAddress>
            </cac:Party>
        </cac:AccountingSupplierParty>
        
        <cac:AccountingCustomerParty>
            <cac:Party>
                <cac:PartyName>
                    <cbc:Name>Customer Company Ltd</cbc:Name>
                </cac:PartyName>
                <cac:PostalAddress>
                    <cbc:StreetName>456 Customer Lane</cbc:StreetName>
                    <cbc:CityName>Customer City</cbc:CityName>
                    <cbc:PostalZone>C45 6ST</cbc:PostalZone>
                    <cac:Country>
                        <cbc:IdentificationCode>US</cbc:IdentificationCode>
                    </cac:Country>
                </cac:PostalAddress>
            </cac:Party>
        </cac:AccountingCustomerParty>
        
        <cac:InvoiceLine>
            <cbc:ID>1</cbc:ID>
            <cbc:InvoicedQuantity unitCode="EA">5</cbc:InvoicedQuantity>
            <cbc:LineExtensionAmount currencyID="USD">500.00</cbc:LineExtensionAmount>
            <cac:Item>
                <cbc:Name>Laptop Computer</cbc:Name>
                <cbc:Description>High-performance laptop</cbc:Description>
                <cac:SellersItemIdentification>
                    <cbc:ID>LAPTOP-001</cbc:ID>
                </cac:SellersItemIdentification>
            </cac:Item>
            <cac:Price>
                <cbc:PriceAmount currencyID="USD">100.00</cbc:PriceAmount>
            </cac:Price>
        </cac:InvoiceLine>
        
        <cac:InvoiceLine>
            <cbc:ID>2</cbc:ID>
            <cbc:InvoicedQuantity unitCode="EA">2</cbc:InvoicedQuantity>
            <cbc:LineExtensionAmount currencyID="USD">300.00</cbc:LineExtensionAmount>
            <cac:Item>
                <cbc:Name>Office Chair</cbc:Name>
                <cbc:Description>Ergonomic office chair</cbc:Description>
                <cac:SellersItemIdentification>
                    <cbc:ID>CHAIR-002</cbc:ID>
                </cac:SellersItemIdentification>
            </cac:Item>
            <cac:Price>
                <cbc:PriceAmount currencyID="USD">150.00</cbc:PriceAmount>
            </cac:Price>
        </cac:InvoiceLine>
        
        <cac:TaxTotal>
            <cbc:TaxAmount currencyID="USD">80.00</cbc:TaxAmount>
        </cac:TaxTotal>
        
        <cac:LegalMonetaryTotal>
            <cbc:LineExtensionAmount currencyID="USD">800.00</cbc:LineExtensionAmount>
            <cbc:TaxExclusiveAmount currencyID="USD">800.00</cbc:TaxExclusiveAmount>
            <cbc:TaxInclusiveAmount currencyID="USD">880.00</cbc:TaxInclusiveAmount>
            <cbc:PayableAmount currencyID="USD">880.00</cbc:PayableAmount>
        </cac:LegalMonetaryTotal>
    </Invoice>
    """
    
    # Process the XML for economic data
    result = agent.process_xml_for_economic_data(sample_xml, "invoice")
    
    # Print the results
    print("\n\nEconomic Data Summary:")
    print("-" * 80)
    print(result["summary"])
    
    print("\n\nExtracted Economic Data:")
    print("-" * 80)
    print(json.dumps(result["extracted_economic_data"], indent=2))
    
    print("\n\nXPaths Used:")
    print("-" * 80)
    print(json.dumps(result["xpaths_used"], indent=2))