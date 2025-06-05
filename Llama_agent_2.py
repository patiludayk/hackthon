import os
import xml.etree.ElementTree as ET
import psycopg2
from psycopg2.extras import execute_values
import numpy as np
from typing import List, Dict, Any
from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.schema import TextNode
from llama_index.core.agent import ReActAgent
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core.tools import FunctionTool
from llama_index.core import Settings
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure PostgreSQL connection parameters
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
            table_name="xsd_knowledge_base",
            embed_dim=1536  # Dimension for Azure OpenAI embeddings
        )
        
        # Initialize storage context and vector index
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            storage_context=self.storage_context,
            embed_model=embed_model  # Explicitly pass the embed model
        )

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
            
            # Adjust xpath for namespace handling if needed
            if ns:
                # This is a simplified approach - might need more complex handling depending on XPath
                for prefix, uri in ns.items():
                    if prefix:
                        xpath = xpath.replace(f"{prefix}:", f"{{{uri}}}")
            
            # Find elements using XPath
            elements = root.findall(xpath)
            
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
        query_engine = self.index.as_query_engine(similarity_top_k=5)
        response = query_engine.query(query_text)
        
        # Extract XPaths from response
        xpaths = []
        for node in response.source_nodes:
            # Assuming each node contains XPath information
            # Extract XPath patterns from the content
            content = node.get_content()
            # Simple regex to find XPath-like patterns
            xpath_patterns = re.findall(r'(//?[\w:]+(?:/[\w:]+)*(?:\[@[\w:]+=[\'"][\w\s]+[\'"]?\])?)', content)
            xpaths.extend(xpath_patterns)
        
        return list(set(xpaths))  # Remove duplicates

    def process_fpml(self, fpml_xml: str) -> Dict[str, Any]:
        """
        Process FPML XML by retrieving relevant XPaths and extracting values.
        """
        # Get relevant XPaths from vector database
        xpaths = self.get_relevant_xpaths(fpml_xml)
        
        # Extract values for each XPath
        results = {}
        for xpath in xpaths:
            value = self.extract_value_from_xpath(fpml_xml, xpath)
            results[xpath] = value
        
        return results

    def summarize_results(self, results: Dict[str, Any]) -> str:
        """
        Summarize the extracted FPML data.
        """
        # Use Azure OpenAI LLM to summarize the results
        summary_prompt = f"Summarize the following FPML data extracted from XPaths:\n\n"
        
        for xpath, value in results.items():
            summary_prompt += f"XPath: {xpath}\nValue: {value}\n\n"
        
        # Use LlamaIndex's query engine for summarization
        query_engine = self.index.as_query_engine()
        response = query_engine.query(summary_prompt)
        
        return str(response)

    def close(self):
        """
        Close database connection.
        """
        if self.conn:
            self.conn.close()


# Create tools for the agent
def create_fpml_agent():
    # Initialize processor
    processor = FPMLProcessor()
    
    # Define tools
    def process_fpml_document(fpml_xml: str) -> str:
        """
        Process an FPML XML document to extract and summarize data.
        
        Args:
            fpml_xml (str): The FPML XML document as a string
            
        Returns:
            str: A summary of the extracted data
        """
        try:
            # Process the FPML document
            results = processor.process_fpml(fpml_xml)
            
            # Summarize the results
            summary = processor.summarize_results(results)
            
            return summary
        except Exception as e:
            return f"Error processing FPML document: {str(e)}"
    
    def get_xpath_values(fpml_xml: str, xpath: str) -> str:
        """
        Extract values from an FPML XML document using a specific XPath.
        
        Args:
            fpml_xml (str): The FPML XML document as a string
            xpath (str): The XPath to use for extraction
            
        Returns:
            str: The extracted values
        """
        try:
            value = processor.extract_value_from_xpath(fpml_xml, xpath)
            return f"Values for XPath '{xpath}': {value}"
        except Exception as e:
            return f"Error extracting values: {str(e)}"
    
    # Create function tools
    process_tool = FunctionTool.from_defaults(fn=process_fpml_document)
    xpath_tool = FunctionTool.from_defaults(fn=get_xpath_values)
    
    # Create ReAct agent with Azure OpenAI
    agent = ReActAgent.from_tools(
        [process_tool, xpath_tool],
        llm=llm,  # Use the already configured Azure OpenAI LLM
        verbose=True
    )
    
    return agent, processor


# Example usage
if __name__ == "__main__":
    # Create agent
    agent, processor = create_fpml_agent()
    
    # Example FPML XML
    example_fpml = """
    <?xml version="1.0" encoding="UTF-8"?>
    <dataDocument xmlns="http://www.fpml.org/FpML-5/confirmation" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" fpmlVersion="5-11" xsi:schemaLocation="http://www.fpml.org/FpML-5/confirmation ../../fpml-main-5-11.xsd http://www.w3.org/2000/09/xmldsig# ../../xmldsig-core-schema.xsd">
<trade>
<tradeHeader>
<partyTradeIdentifier>
<partyReference href="party1"/>
<tradeId tradeIdScheme="http://www.bankB.com/swaps/com-trade-id">BankA1234</tradeId>
</partyTradeIdentifier>
<partyTradeIdentifier>
<partyReference href="party2"/>
<tradeId tradeIdScheme="http://www.bankA.com/swaps/com-trade-id">BankB5678</tradeId>
</partyTradeIdentifier>
<tradeDate>2012-01-01</tradeDate>
</tradeHeader>
<swap>
<primaryAssetClass>Commodity</primaryAssetClass>
<productId productIdScheme="http://www.dtcc.com/coding-scheme/external/GTR-Product-Id">Commodity:Metals:Precious:LoanLease:Cash</productId>
<swapStream>
<payerPartyReference href="party1"/>
<receiverPartyReference href="party2"/>
<calculationPeriodDates id="gofoperioddates1">
<effectiveDate>
<unadjustedDate>2012-01-01</unadjustedDate>
<dateAdjustments>
<businessDayConvention>MODFOLLOWING</businessDayConvention>
<businessCenters id="primaryBusinessCenters">
<businessCenter>USNY</businessCenter>
</businessCenters>
</dateAdjustments>
</effectiveDate>
<terminationDate>
<unadjustedDate>2013-01-01</unadjustedDate>
<dateAdjustments>
<businessDayConvention>MODFOLLOWING</businessDayConvention>
<businessCentersReference href="primaryBusinessCenters"/>
</dateAdjustments>
</terminationDate>
<calculationPeriodDatesAdjustments>
<businessDayConvention>MODFOLLOWING</businessDayConvention>
<businessCentersReference href="primaryBusinessCenters"/>
</calculationPeriodDatesAdjustments>
<calculationPeriodFrequency>
<periodMultiplier>3</periodMultiplier>
<period>M</period>
<rollConvention>1</rollConvention>
</calculationPeriodFrequency>
</calculationPeriodDates>
<paymentDates>
<calculationPeriodDatesReference href="gofoperioddates1"/>
<paymentFrequency>
<periodMultiplier>3</periodMultiplier>
<period>M</period>
</paymentFrequency>
<firstPaymentDate>2012-04-01</firstPaymentDate>
<payRelativeTo>CalculationPeriodEndDate</payRelativeTo>
<paymentDatesAdjustments>
<businessDayConvention>MODFOLLOWING</businessDayConvention>
<businessCentersReference href="primaryBusinessCenters"/>
</paymentDatesAdjustments>
</paymentDates>
<resetDates id="resetDates1">
<calculationPeriodDatesReference href="gofoperioddates1"/>
<resetRelativeTo>CalculationPeriodStartDate</resetRelativeTo>
<fixingDates>
<periodMultiplier>-2</periodMultiplier>
<period>D</period>
<dayType>Business</dayType>
<businessDayConvention>NONE</businessDayConvention>
<businessCentersReference href="primaryBusinessCenters"/>
<dateRelativeTo href="resetDates1"/>
</fixingDates>
<resetFrequency>
<periodMultiplier>3</periodMultiplier>
<period>M</period>
</resetFrequency>
<resetDatesAdjustments>
<businessDayConvention>MODFOLLOWING</businessDayConvention>
<businessCentersReference href="primaryBusinessCenters"/>
</resetDatesAdjustments>
</resetDates>
<calculationPeriodAmount>
<calculation>
<notionalSchedule>
<!--  
                          since we have to represent gold interms of XAU, we must convert everything to ozt. 
                          for example, if we have 1kg of gold, we need to represent it as 32.15074656 XAU                                 
                           -->
<notionalStepSchedule>
<initialValue>100.00</initialValue>
<currency>XAU</currency>
</notionalStepSchedule>
</notionalSchedule>
<floatingRateCalculation>
<!--  This is a 3 month libor index, but it may not be the right index  -->
<floatingRateIndex>USD-LIBOR-BBA</floatingRateIndex>
<indexTenor>
<periodMultiplier>3</periodMultiplier>
<period>M</period>
</indexTenor>
</floatingRateCalculation>
<dayCountFraction>ACT/365.FIXED</dayCountFraction>
</calculation>
</calculationPeriodAmount>
</swapStream>
<swapStream>
<payerPartyReference href="party2"/>
<receiverPartyReference href="party1"/>
<calculationPeriodDates id="gofoperioddates2">
<effectiveDate>
<unadjustedDate>2012-01-01</unadjustedDate>
<dateAdjustments>
<businessDayConvention>MODFOLLOWING</businessDayConvention>
<businessCentersReference href="primaryBusinessCenters"/>
</dateAdjustments>
</effectiveDate>
<terminationDate>
<unadjustedDate>2013-01-01</unadjustedDate>
<dateAdjustments>
<businessDayConvention>MODFOLLOWING</businessDayConvention>
<businessCentersReference href="primaryBusinessCenters"/>
</dateAdjustments>
</terminationDate>
<calculationPeriodDatesAdjustments>
<businessDayConvention>MODFOLLOWING</businessDayConvention>
<businessCentersReference href="primaryBusinessCenters"/>
</calculationPeriodDatesAdjustments>
<calculationPeriodFrequency>
<periodMultiplier>3</periodMultiplier>
<period>M</period>
<rollConvention>1</rollConvention>
</calculationPeriodFrequency>
</calculationPeriodDates>
<paymentDates>
<calculationPeriodDatesReference href="gofoperioddates2"/>
<paymentFrequency>
<periodMultiplier>3</periodMultiplier>
<period>M</period>
</paymentFrequency>
<firstPaymentDate>2012-04-01</firstPaymentDate>
<payRelativeTo>CalculationPeriodEndDate</payRelativeTo>
<paymentDatesAdjustments>
<businessDayConvention>MODFOLLOWING</businessDayConvention>
<businessCentersReference href="primaryBusinessCenters"/>
</paymentDatesAdjustments>
</paymentDates>
<resetDates id="resetDates2">
<calculationPeriodDatesReference href="gofoperioddates2"/>
<resetRelativeTo>CalculationPeriodStartDate</resetRelativeTo>
<fixingDates>
<periodMultiplier>-2</periodMultiplier>
<period>D</period>
<dayType>Business</dayType>
<businessDayConvention>NONE</businessDayConvention>
<businessCentersReference href="primaryBusinessCenters"/>
<dateRelativeTo href="resetDates2"/>
</fixingDates>
<resetFrequency>
<periodMultiplier>3</periodMultiplier>
<period>M</period>
</resetFrequency>
<resetDatesAdjustments>
<businessDayConvention>MODFOLLOWING</businessDayConvention>
<businessCentersReference href="primaryBusinessCenters"/>
</resetDatesAdjustments>
</resetDates>
<calculationPeriodAmount>
<calculation>
<notionalSchedule>
<!--  
                                                                since we have to represent gold interms of XAU, we must convert everything to ozt. 
                                                                for example, if we have 1kg of gold, we need to represent it as 32.15074656 XAU                                 
                                                         -->
<notionalStepSchedule>
<initialValue>100.00</initialValue>
<currency>XAU</currency>
</notionalStepSchedule>
</notionalSchedule>
<floatingRateCalculation>
<!--  This uses a commodity floating rate index coding scheme list with just one value "GOFO". To use the commodity floating rate index coding scheme, it needs to be specified in the floatingRateIndexScheme  -->
<floatingRateIndex floatingRateIndexScheme="http://www.fpml.org/coding-scheme/commodity-floating-rate-index">GOFO</floatingRateIndex>
<spreadSchedule>
<initialValue>-0.0001</initialValue>
</spreadSchedule>
</floatingRateCalculation>
<dayCountFraction>ACT/365.FIXED</dayCountFraction>
</calculation>
</calculationPeriodAmount>
</swapStream>
</swap>
</trade>
<party id="party1">
<partyId partyIdScheme="http://www.fpml.org/coding-scheme/dummy-party-id">Bank A</partyId>
<partyName>Bank a</partyName>
</party>
<party id="party2">
<partyId partyIdScheme="http://www.fpml.org/coding-scheme/dummy-party-id">Bank B</partyId>
<partyName>Bank B</partyName>
</party>
</dataDocument>
    """
    
    # Process FPML
    response = agent.chat(f"Process this FPML document and summarize the key information: {example_fpml}")
    print(response)
    
    # Close connections
    processor.close()