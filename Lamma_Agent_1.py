import os
import sys
import argparse
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional, Tuple
import re
import json
from dotenv import load_dotenv

# LlamaIndex imports
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
PG_TABLE_NAME = os.getenv("PG_TABLE_NAME", "fpml_knowledge_base")

class FPMLProcessor:
    """Class to process FPML data with focus on financial derivative fields"""
    
    @staticmethod
    def parse_xml_file(file_path: str) -> Optional[ET.Element]:
        """Parse FPML file into an ElementTree object"""
        try:
            tree = ET.parse(file_path)
            return tree.getroot()
        except (ET.ParseError, FileNotFoundError) as e:
            print(f"Error parsing FPML file '{file_path}': {e}")
            return None
    
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
            child_data = FPMLProcessor.element_to_dict(child)
            
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
    def extract_financial_data_by_xpath(root: ET.Element, xpath: str, namespaces: Dict[str, str] = None) -> Dict[str, Any]:
        """Extract financial data using XPath from FPML"""
        element = FPMLProcessor.extract_element_by_xpath(root, xpath, namespaces)
        if element is None:
            return {}
        
        # For financial data, we want to extract both the value and any relevant attributes
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
            currency_attrs = ["currency", "currencyScheme", "id"]
            for attr in currency_attrs:
                if attr in element.attrib:
                    result[attr] = element.attrib[attr]
        
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

class FPMLXPathAgent:
    """Agent that extracts financial data from FPML using XPaths from a PostgreSQL vector database"""
       
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
        
        # Initialize FPML processor
        self.fpml_processor = FPMLProcessor()
        
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
        
        # Common FPML XPaths (fallback if vector DB doesn't return good results)
        self.common_fpml_xpaths = {
            "notional": [
                ".//notionalAmount", 
                ".//notional/amount",
                ".//notionalSchedule/notionalStepSchedule/initialValue",
                ".//calculationAmount"
            ],
            "currency": [
                ".//currency", 
                ".//paymentAmount/currency",
                ".//notionalAmount/@currency"
            ],
            "trade_date": [
                ".//tradeDate",
                ".//tradeHeader/tradeDate"
            ],
            "effective_date": [
                ".//effectiveDate/unadjustedDate",
                ".//calculationPeriodDates/effectiveDate/unadjustedDate"
            ],
            "termination_date": [
                ".//terminationDate/unadjustedDate",
                ".//calculationPeriodDates/terminationDate/unadjustedDate"
            ],
            "fixed_rate": [
                ".//fixedRateSchedule/initialValue",
                ".//fixedRate"
            ],
            "floating_rate": [
                ".//floatingRateIndex",
                ".//floatingRateCalculation/floatingRateIndex"
            ],
            "payment_amount": [
                ".//paymentAmount/amount",
                ".//payment/paymentAmount/amount"
            ],
            "premium": [
                ".//premium/paymentAmount/amount",
                ".//optionPremium/paymentAmount/amount"
            ],
            "strike": [
                ".//strike/strikeRate",
                ".//strikePrice"
            ]
        }
    
    def _create_tools(self) -> List[BaseTool]:
        """Create tools for the agent to use"""
        tools = []
        
        # Tool to search the vector database for FPML XPaths
        def search_fpml_xpaths(fpml_type: str, top_k: int = 10) -> str:
            """Search the PostgreSQL vector database for FPML XPaths relevant to the document type."""
            query = f"What are the XPaths for financial fields in {fpml_type} FPML documents?"
            query_engine = self.index.as_query_engine(similarity_top_k=top_k)
            response = query_engine.query(query)
            return str(response)
        
        search_tool = FunctionTool.from_defaults(
            fn=search_fpml_xpaths,
            name="search_fpml_xpaths",
            description="Search the PostgreSQL vector database for FPML XPaths relevant to a specific document type"
        )
        tools.append(search_tool)
        
        # Tool to extract financial data using XPath
        def extract_financial_data(xml_string: str, xpath: str) -> str:
            """Extract financial data using XPath from FPML string."""
            root = self.fpml_processor.parse_xml_string(xml_string)
            if not root:
                return "Error parsing FPML"
            
            namespaces = self.fpml_processor.get_namespaces(root)
            self.fpml_processor.register_namespaces(namespaces)
            
            data = self.fpml_processor.extract_financial_data_by_xpath(root, xpath, namespaces)
            
            if not data:
                return f"No financial data found for XPath '{xpath}'"
            
            return json.dumps(data, indent=2)
        
        extract_tool = FunctionTool.from_defaults(
            fn=extract_financial_data,
            name="extract_financial_data",
            description="Extract financial data using XPath from FPML"
        )
        tools.append(extract_tool)
        
        # Tool to extract multiple values using XPath
        def extract_values(xml_string: str, xpath: str) -> str:
            """Extract multiple values using XPath from FPML string."""
            root = self.fpml_processor.parse_xml_string(xml_string)
            if not root:
                return "Error parsing FPML"
            
            namespaces = self.fpml_processor.get_namespaces(root)
            self.fpml_processor.register_namespaces(namespaces)
            
            values = self.fpml_processor.extract_values_by_xpath(root, xpath, namespaces)
            
            if not values:
                return f"No values found for XPath '{xpath}'"
            
            return json.dumps(values, indent=2)
        
        values_tool = FunctionTool.from_defaults(
            fn=extract_values,
            name="extract_values",
            description="Extract multiple values using XPath from FPML"
        )
        tools.append(values_tool)
        
        # Tool to summarize financial data
        def summarize_financial_data(financial_data: str) -> str:
            """Summarize financial data extracted from FPML."""
            prompt = f"""
            Please summarize the following financial data extracted from an FPML document:
            
            {financial_data}
            
            Provide a comprehensive summary that includes all financial terms such as:
            - Notional amounts and currencies
            - Trade dates, effective dates, and termination dates
            - Fixed and floating rates
            - Payment amounts and schedules
            - Premium amounts
            - Strike prices for options
            - Any other relevant financial terms
            
            Format the summary in a clear, structured manner suitable for financial professionals.
            """
            
            response = self.llm.complete(prompt)
            return response.text
        
        summarize_tool = FunctionTool.from_defaults(
            fn=summarize_financial_data,
            name="summarize_financial_data",
            description="Summarize financial data extracted from FPML"
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
    
    def process_fpml_file(self, file_path: str, fpml_type: str = "derivative") -> Dict[str, Any]:
        """
        Process FPML file to extract financial information using XPaths from the vector database
        
        Args:
            file_path: Path to the FPML file
            fpml_type: The type of FPML document (e.g., "swap", "option", "forward")
            
        Returns:
            Dictionary containing extracted financial data and summary
        """
        # Parse the FPML file
        root = self.fpml_processor.parse_xml_file(file_path)
        if not root:
            return {"error": f"Failed to parse FPML file: {file_path}"}
        
        # Read the file content for processing
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                xml_string = f.read()
        except Exception as e:
            return {"error": f"Failed to read FPML file: {e}"}
        
        return self.process_fpml_for_financial_data(xml_string, fpml_type, file_path)
    
    def process_fpml_for_financial_data(self, xml_string: str, fpml_type: str = "derivative", source_file: str = None) -> Dict[str, Any]:
        """
        Process FPML data to extract financial information using XPaths from the vector database
        
        Args:
            xml_string: The FPML data as a string
            fpml_type: The type of FPML document (e.g., "swap", "option", "forward")
            source_file: Optional source file path for reference
            
        Returns:
            Dictionary containing extracted financial data and summary
        """
        # Parse the XML
        root = self.fpml_processor.parse_xml_string(xml_string)
        if not root:
            return {"error": "Failed to parse FPML"}
        
        # Get namespaces
        namespaces = self.fpml_processor.get_namespaces(root)
        self.fpml_processor.register_namespaces(namespaces)
        
        # Get the root element name for better context
        root_tag = root.tag.split("}")[-1] if "}" in root.tag else root.tag
        
        # Search vector database for FPML XPaths relevant to this document type
        context_response = self.tools[0].fn(f"{root_tag} {fpml_type}")
        
        # Extract XPaths from the context
        xpaths = self.extract_xpaths_from_context(context_response)
        
        # If no XPaths found, use common FPML XPaths as fallback
        if not xpaths:
            xpaths = []
            for category, paths in self.common_fpml_xpaths.items():
                xpaths.extend(paths)
        
        # Extract financial data for each XPath
        extracted_data = {}
        
        for xpath in xpaths:
            try:
                # Normalize the XPath to handle namespace issues
                normalized_xpath = self.fpml_processor.normalize_xpath(xpath)
                
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
                        
                        # Special handling for currency and scheme attributes
                        special_attrs = ["currency", "currencyScheme", "id", "scheme"]
                        for attr in special_attrs:
                            if attr in element.attrib:
                                data[attr] = element.attrib[attr]
                    
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
            # Look for elements with financial-related names
            financial_terms = ["notional", "amount", "rate", "date", "currency", "premium", "strike", "payment"]
            
            for term in financial_terms:
                # Try to find elements containing this term
                for element in root.findall(f".//*[contains(local-name(), '{term}')]"):
                    tag_name = element.tag.split("}")[-1] if "}" in element.tag else element.tag
                    
                    if element.text and element.text.strip():
                        extracted_data[tag_name] = element.text.strip()
        
        # Organize the data for better summarization
        organized_data = self._organize_financial_data(extracted_data)
        
        # Summarize the extracted data
        summary_prompt = f"""
        I've extracted the following financial data from a {fpml_type} FPML document with root element <{root_tag}>:
        
        {json.dumps(organized_data, indent=2)}
        
        Please provide a comprehensive summary of this financial derivative information.
        Include all relevant financial terms such as notional amounts, currencies, dates, rates, 
        payment amounts, and other derivative-specific terms.
        Format the summary in a clear, structured manner suitable for financial professionals.
        """
        
        summary = self.llm.complete(summary_prompt)
        
        # Return both the extracted data and the summary
        result = {
            "source_file": source_file,
            "fpml_type": fpml_type,
            "root_element": root_tag,
            "extracted_financial_data": organized_data,
            "raw_extracted_data": extracted_data,
            "summary": summary.text,
            "xpaths_used": xpaths
        }
        
        return result
    
    def _organize_financial_data(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Organize extracted financial data into categories for better summarization"""
        organized = {
            "notional_amounts": {},
            "currencies": {},
            "dates": {},
            "rates": {},
            "payments": {},
            "premiums": {},
            "strikes": {},
            "other": {}
        }
        
        for key, value in extracted_data.items():
            key_lower = key.lower()
            
            # Determine category based on key name
            if any(term in key_lower for term in ["notional", "amount", "calculation"]):
                organized["notional_amounts"][key] = value
            elif any(term in key_lower for term in ["currency"]):
                organized["currencies"][key] = value
            elif any(term in key_lower for term in ["date"]):
                organized["dates"][key] = value
            elif any(term in key_lower for term in ["rate", "spread"]):
                organized["rates"][key] = value
            elif any(term in key_lower for term in ["payment"]):
                organized["payments"][key] = value
            elif any(term in key_lower for term in ["premium"]):
                organized["premiums"][key] = value
            elif any(term in key_lower for term in ["strike"]):
                organized["strikes"][key] = value
            else:
                organized["other"][key] = value
        
        # Remove empty categories
        return {k: v for k, v in organized.items() if v}
    
    def add_xpath_to_knowledge_base(self, xpath_info: str, fpml_type: str) -> None:
        """Add FPML XPath information to the PostgreSQL vector database"""
        document = Document(
            text=xpath_info, 
            metadata={"type": "fpml_xpath", "fpml_type": fpml_type}
        )
        self.index.insert(document)
        print(f"Added FPML XPath info for '{fpml_type}' to knowledge base")
    
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

def main():
    """Main function to handle command line arguments and process FPML files"""
    parser = argparse.ArgumentParser(description='Process FPML files to extract financial data')
    parser.add_argument('fpml_file', help='Path to the FPML file to process')
    parser.add_argument('--type', '-t', default='derivative', 
                       help='Type of FPML document (e.g., swap, option, forward, derivative)')
    parser.add_argument('--output', '-o', help='Output file path for results (JSON format)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Check if the FPML file exists
    if not os.path.exists(args.fpml_file):
        print(f"Error: FPML file '{args.fpml_file}' not found.")
        sys.exit(1)
    
    try:
        # Initialize the agent
        print("Initializing FPML XPath Agent...")
        agent = FPMLXPathAgent()
        
        # Ensure database is properly set up
        agent.initialize_database()
        
        # Add some FPML XPath information to the knowledge base
        if args.verbose:
            print("Adding FPML XPath knowledge to database...")
            
        swap_xpath_info = """
        Financial XPaths for FPML Interest Rate Swap:
        
        1. Notional Amount: xpath = ".//notionalAmount"
        2. Currency: xpath = ".//currency"
        3. Trade Date: xpath = ".//tradeDate"
        4. Effective Date: xpath = ".//effectiveDate/unadjustedDate"
        5. Termination Date: xpath = ".//terminationDate/unadjustedDate"
        6. Fixed Rate: xpath = ".//fixedRateSchedule/initialValue"
        7. Floating Rate Index: xpath = ".//floatingRateIndex"
        8. Payment Amount: xpath = ".//paymentAmount/amount"
        9. Calculation Amount: xpath = ".//calculationAmount"
        10. Day Count Fraction: xpath = ".//dayCountFraction"
        """
        
        option_xpath_info = """
        Financial XPaths for FPML Option:
        
        1. Premium Amount: xpath = ".//premium/paymentAmount/amount"
        2. Premium Currency: xpath = ".//premium/paymentAmount/currency"
        3. Strike Rate: xpath = ".//strike/strikeRate"
        4. Expiration Date: xpath = ".//expirationDate/adjustableDate/unadjustedDate"
        5. Exercise Date: xpath = ".//exerciseDate"
        6. Notional Amount: xpath = ".//notionalAmount"
        7. Underlying Rate: xpath = ".//underlyingRate/floatingRateIndex"
        8. Option Type: xpath = ".//optionType"
        """
        
        agent.add_xpath_to_knowledge_base(swap_xpath_info, "Interest Rate Swap")
        agent.add_xpath_to_knowledge_base(option_xpath_info, "Option")
        
        # Process the FPML file
        print(f"Processing FPML file: {args.fpml_file}")
        result = agent.process_fpml_file(args.fpml_file, args.type)
        
        # Check for errors
        if "error" in result:
            print(f"Error processing FPML file: {result['error']}")
            sys.exit(1)
        
        # Print the results
        print("\n" + "="*80)
        print("FPML FINANCIAL DATA SUMMARY")
        print("="*80)
        print(result["summary"])
        
