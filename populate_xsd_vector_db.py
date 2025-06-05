
#AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=datalensaidata;AccountKey=w87Kv5I5wqgOnH+G0iwYLwuPZpLmG/THu7Xh5WeIDaHrmpjizF3AAXjky3eklVuOnNBqRxB1OaEj+ASt/NOEpQ==;EndpointSuffix=core.windows.net
#CONTAINER_NAME=datalensaikb
#OPENAI_API_KEY=your_openai_api_key
#POSTGRES_CONNECTION=postgresql://dataailensdb:Hackathon#2025@dataailens.postgres.database.azure.com:5432/dataailenskb

import os
import tempfile
import uuid
import xml.etree.ElementTree as ET
import re
from typing import List, Dict, Any
from dataclasses import dataclass

# Azure Blob Storage
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError

# Vector embeddings
import openai
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
import tiktoken

import time
import random
import json

# Environment variables (set these or use .env file with python-dotenv)
#AZURE_STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
#CONTAINER_NAME = os.environ.get("CONTAINER_NAME")
#OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
#POSTGRES_CONNECTION = os.environ.get("POSTGRES_CONNECTION")


AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=datalensaidata;AccountKey=w87Kv5I5wqgOnH+G0iwYLwuPZpLmG/THu7Xh5WeIDaHrmpjizF3AAXjky3eklVuOnNBqRxB1OaEj+ASt/NOEpQ==;EndpointSuffix=core.windows.net"
CONTAINER_NAME="datalensaikb"
OPENAI_API_KEY="6fe74af0d78e4fa382eceed78433c3a0"
POSTGRES_CONNECTION="postgresql://dataailensdb:Hackathon#2025@dataailens.postgres.database.azure.com:5432/dataailenskb"



# Configure OpenAI
openai.api_key = OPENAI_API_KEY
openai.api_version = "2024-12-01-preview"
openai.api_type="azure"
openai.api_base = "https://bh-uk-openai-dataai-lens.openai.azure.com/"
EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_ENCODING = "cl100k_base"  # Encoding for text-embedding-ada-002
EMBEDDING_DIMENSIONS = 1536  # Dimensions for text-embedding-ada-002
MAX_TOKENS = 8191  # Maximum tokens for text-embedding-ada-002

# Initialize tokenizer
tokenizer = tiktoken.get_encoding(EMBEDDING_ENCODING)

@dataclass
class XsdElement:
    """Class to store XSD element information"""
    name: str
    type: str = None
    min_occurs: str = None
    max_occurs: str = None
    documentation: str = None
    parent_path: str = None
    attributes: List[Dict[str, str]] = None
    
    def to_text(self) -> str:
        """Convert XSD element to readable text format"""
        text = f"Element: {self.name}\n"
        text += f"Path: {self.parent_path}/{self.name}\n"
        
        if self.type:
            text += f"Type: {self.type}\n"
        
        if self.min_occurs:
            text += f"Min Occurs: {self.min_occurs}\n"
        
        if self.max_occurs:
            text += f"Max Occurs: {self.max_occurs}\n"
        
        if self.attributes and len(self.attributes) > 0:
            text += "Attributes:\n"
            for attr in self.attributes:
                attr_text = f"  - {attr.get('name', '')}"
                if attr.get('type'):
                    attr_text += f" (Type: {attr.get('type')})"
                if attr.get('use'):
                    attr_text += f" (Use: {attr.get('use')})"
                text += attr_text + "\n"
        
        if self.documentation:
            text += f"Documentation: {self.documentation}\n"
        
        return text

@dataclass
class XsdType:
    """Class to store XSD type information"""
    name: str
    base_type: str = None
    documentation: str = None
    elements: List[Dict[str, str]] = None
    attributes: List[Dict[str, str]] = None
    
    def to_text(self) -> str:
        """Convert XSD type to readable text format"""
        text = f"Type: {self.name}\n"
        
        if self.base_type:
            text += f"Base Type: {self.base_type}\n"
        
        if self.elements and len(self.elements) > 0:
            text += "Elements:\n"
            for elem in self.elements:
                elem_text = f"  - {elem.get('name', '')}"
                if elem.get('type'):
                    elem_text += f" (Type: {elem.get('type')})"
                if elem.get('minOccurs'):
                    elem_text += f" (Min: {elem.get('minOccurs')})"
                if elem.get('maxOccurs'):
                    elem_text += f" (Max: {elem.get('maxOccurs')})"
                text += elem_text + "\n"
        
        if self.attributes and len(self.attributes) > 0:
            text += "Attributes:\n"
            for attr in self.attributes:
                attr_text = f"  - {attr.get('name', '')}"
                if attr.get('type'):
                    attr_text += f" (Type: {attr.get('type')})"
                if attr.get('use'):
                    attr_text += f" (Use: {attr.get('use')})"
                text += attr_text + "\n"
        
        if self.documentation:
            text += f"Documentation: {self.documentation}\n"
        
        return text

def connect_to_blob_storage():
    """Connect to Azure Blob Storage"""
    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        return blob_service_client
    except Exception as e:
        print(f"Error connecting to Blob Storage: {e}")
        return None

def list_blobs(blob_service_client, container_name):
    """List all blobs in a container"""
    try:
        container_client = blob_service_client.get_container_client(container_name)
        blobs = container_client.list_blobs()
        return [blob for blob in blobs if blob.name.lower().endswith('.xsd')]
    except ResourceNotFoundError:
        print(f"Container '{container_name}' not found")
        return []
    except Exception as e:
        print(f"Error listing blobs: {e}")
        return []

def download_blob(blob_service_client, container_name, blob_name):
    """Download a blob to a temporary file"""
    try:
        container_client = blob_service_client.get_container_client(container_name)
        blob_client = container_client.get_blob_client(blob_name)
        
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file_path = temp_file.name
        temp_file.close()
        
        # Download the blob to the temporary file
        with open(temp_file_path, "wb") as file:
            file.write(blob_client.download_blob().readall())
        
        return temp_file_path
    except Exception as e:
        print(f"Error downloading blob {blob_name}: {e}")
        return None

def get_namespace_prefix(root):
    """Extract namespace prefix from XML root"""
    match = re.match(r'\{(.*?)\}', root.tag)
    if match:
        namespace = match.group(1)
        return namespace
    return None

def parse_xsd_file(file_path):
    """Parse XSD file and extract elements, types, and documentation"""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Get namespace
        namespace = get_namespace_prefix(root)
        ns_prefix = '{' + namespace + '}' if namespace else ''
        
        elements = []
        types = []
        
        # Process global elements
        for element in root.findall(f'.//{ns_prefix}element'):
            process_element(element, elements, ns_prefix, "")
        
        # Process complex types
        for complex_type in root.findall(f'.//{ns_prefix}complexType'):
            process_complex_type(complex_type, types, ns_prefix)
        
        # Process simple types
        for simple_type in root.findall(f'.//{ns_prefix}simpleType'):
            process_simple_type(simple_type, types, ns_prefix)
        
        return elements, types
    except Exception as e:
        print(f"Error parsing XSD file: {e}")
        return [], []

def get_documentation(element, ns_prefix):
    """Extract documentation from an element"""
    annotation = element.find(f'./{ns_prefix}annotation')
    if annotation is not None:
        documentation = annotation.find(f'./{ns_prefix}documentation')
        if documentation is not None:
            return documentation.text.strip() if documentation.text else ""
    return None

def process_element(element, elements_list, ns_prefix, parent_path=""):
    """Process an XSD element and its children"""
    name = element.get('name')
    if name is None:
        return
    
    element_type = element.get('type')
    min_occurs = element.get('minOccurs')
    max_occurs = element.get('maxOccurs')
    documentation = get_documentation(element, ns_prefix)
    
    attributes = []
    for attr in element.findall(f'.//{ns_prefix}attribute'):
        attr_info = {
            'name': attr.get('name'),
            'type': attr.get('type'),
            'use': attr.get('use')
        }
        attributes.append(attr_info)
    
    xsd_element = XsdElement(
        name=name,
        type=element_type,
        min_occurs=min_occurs,
        max_occurs=max_occurs,
        documentation=documentation,
        parent_path=parent_path,
        attributes=attributes
    )
    
    elements_list.append(xsd_element)
    
    # Process child elements
    complex_type = element.find(f'./{ns_prefix}complexType')
    if complex_type is not None:
        sequence = complex_type.find(f'./{ns_prefix}sequence')
        if sequence is not None:
            for child in sequence.findall(f'./{ns_prefix}element'):
                process_element(child, elements_list, ns_prefix, f"{parent_path}/{name}")

def process_complex_type(complex_type, types_list, ns_prefix):
    """Process a complex type definition"""
    name = complex_type.get('name')
    if name is None:
        return
    
    documentation = get_documentation(complex_type, ns_prefix)
    
    # Get base type if it's an extension
    base_type = None
    extension = complex_type.find(f'.//{ns_prefix}extension')
    if extension is not None:
        base_type = extension.get('base')
    
    # Get child elements
    elements = []
    for sequence in complex_type.findall(f'.//{ns_prefix}sequence'):
        for element in sequence.findall(f'./{ns_prefix}element'):
            elem_info = {
                'name': element.get('name'),
                'type': element.get('type'),
                'minOccurs': element.get('minOccurs'),
                'maxOccurs': element.get('maxOccurs')
            }
            elements.append(elem_info)
    
    # Get attributes
    attributes = []
    for attr in complex_type.findall(f'.//{ns_prefix}attribute'):
        attr_info = {
            'name': attr.get('name'),
            'type': attr.get('type'),
            'use': attr.get('use')
        }
        attributes.append(attr_info)
    
    xsd_type = XsdType(
        name=name,
        base_type=base_type,
        documentation=documentation,
        elements=elements,
        attributes=attributes
    )
    
    types_list.append(xsd_type)

def process_simple_type(simple_type, types_list, ns_prefix):
    """Process a simple type definition"""
    name = simple_type.get('name')
    if name is None:
        return
    
    documentation = get_documentation(simple_type, ns_prefix)
    
    # Get base type if it's a restriction
    base_type = None
    restriction = simple_type.find(f'./{ns_prefix}restriction')
    if restriction is not None:
        base_type = restriction.get('base')
    
    xsd_type = XsdType(
        name=name,
        base_type=base_type,
        documentation=documentation
    )
    
    types_list.append(xsd_type)

def create_chunks_from_xsd(elements, types, file_name):
    """Create text chunks from XSD elements and types"""
    chunks = []
    metadata_list = []
    
    # Create a chunk for the schema overview
    overview = f"XSD Schema: {file_name}\n\n"
    overview += f"Total Elements: {len(elements)}\n"
    overview += f"Total Types: {len(types)}\n\n"
    
    # Add top-level elements to overview
    top_level_elements = [e for e in elements if e.parent_path == ""]
    if top_level_elements:
        overview += "Top-level Elements:\n"
        for elem in top_level_elements:
            overview += f"- {elem.name}\n"
    
    chunks.append(overview)
    metadata_list.append({
        "source": file_name,
        "chunk_type": "schema_overview",
        "chunk_index": 0
    })
    
    # Create chunks for elements (group by parent path)
    element_by_path = {}
    for element in elements:
        path = element.parent_path
        if path not in element_by_path:
            element_by_path[path] = []
        element_by_path[path].append(element)
    
    chunk_index = 1
    for path, path_elements in element_by_path.items():
        chunk_text = f"XSD Schema: {file_name}\nElement Path: {path}\n\n"
        
        for element in path_elements:
            chunk_text += element.to_text() + "\n"
        
        chunks.append(chunk_text)
        metadata_list.append({
            "source": file_name,
            "chunk_type": "element_group",
            "path": path,
            "chunk_index": chunk_index
        })
        chunk_index += 1
    
    # Create chunks for types (group them in reasonable sizes)
    MAX_TYPES_PER_CHUNK = 5
    for i in range(0, len(types), MAX_TYPES_PER_CHUNK):
        chunk_types = types[i:i + MAX_TYPES_PER_CHUNK]
        
        chunk_text = f"XSD Schema: {file_name}\nTypes Definition:\n\n"
        for type_def in chunk_types:
            chunk_text += type_def.to_text() + "\n"
        
        chunks.append(chunk_text)
        metadata_list.append({
            "source": file_name,
            "chunk_type": "type_definitions",
            "type_index_start": i,
            "type_index_end": i + len(chunk_types) - 1,
            "chunk_index": chunk_index
        })
        chunk_index += 1
    
    return chunks, metadata_list

def chunk_text(text, max_tokens=MAX_TOKENS):
    """Split text into chunks that fit within token limits"""
    tokens = tokenizer.encode(text)
    
    if len(tokens) <= max_tokens:
        return [text]
    
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
    
    return chunks

#def create_embeddings(chunks):
#    """Create embeddings for text chunks using OpenAI API"""
#    embeddings = []
#    
#    for chunk in chunks:
#        try:
#            response = openai.Embedding.create(
#                input=chunk,
#                model=EMBEDDING_MODEL,
#                engine=EMBEDDING_MODEL
#            )
#            embedding = response['data'][0]['embedding']
#            embeddings.append(embedding)
#        except Exception as e:
#            print(f"Error creating embedding: {e}")
#            embeddings.append(None)
#    
#    return embeddings

def setup_database():
    """Set up PostgreSQL database with pgvector extension"""
    try:
        conn = psycopg2.connect(POSTGRES_CONNECTION)
        cursor = conn.cursor()
        
        # Create pgvector extension if not exists
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # Create table for knowledge base
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS xsd_knowledge_base (
            id UUID PRIMARY KEY,
            content TEXT,
            metadata JSONB,
            embedding vector(%s)
        );
        """ % EMBEDDING_DIMENSIONS)
        
#        # Create index for vector similarity search
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS xsd_knowledge_base_embedding_idx 
        ON xsd_knowledge_base 
        USING ivfflat (embedding vector_cosine_ops) 
        WITH (lists = 100);
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return True
    except Exception as e:  
        print(f"Error setting up database: {e}")
        return False

#def store_embeddings(chunks, embeddings, metadata_list):
#    """Store text chunks and their embeddings in PostgreSQL"""
#    try:
#        conn = psycopg2.connect(POSTGRES_CONNECTION)
#        cursor = conn.cursor()
#        
#        # Prepare data for insertion
#        data = []
#        for chunk, embedding, metadata in zip(chunks, embeddings, metadata_list):
#            if embedding is not None:
#                data.append((
#                    str(uuid.uuid4()),
#                    chunk,
#                    metadata,
#                    embedding
#                ))
#        
#        # Insert data using execute_values for efficiency
#        execute_values(
#            cursor,
#            """
#            INSERT INTO xsd_knowledge_base (id, content, metadata, embedding)
#            VALUES %s
#            """,
#            data,
#            template="(%s, %s, %s, %s)"
#        )
#        
#        conn.commit()
#        cursor.close()
#        conn.close()
#        
#        return True
#    except Exception as e:
#        print(f"Error storing embeddings: {e}")
#        return False

#def store_embeddings(chunks, embeddings, metadata_list):
#    """Store text chunks and their embeddings in PostgreSQL"""
#    try:
#        import json  # Add this import at the top of your file if not already there
#        conn = psycopg2.connect(POSTGRES_CONNECTION)
#        cursor = conn.cursor()
#        
#        # Prepare data for insertion
#        data = []
#        for chunk, embedding, metadata in zip(chunks, embeddings, metadata_list):
#            if embedding is not None:
#                data.append((
#                    str(uuid.uuid4()),
#                    chunk,
#                    json.dumps(metadata),  # Convert dict to JSON string
#                    embedding
#                ))
#        
#        # Insert data using execute_values for efficiency
#        execute_values(
#            cursor,
#            """
#            INSERT INTO xsd_knowledge_base (id, content, metadata, embedding)
#            VALUES %s
#            """,
#            data,
#            template="(%s, %s, %s, %s)"
#        )
#        
#        conn.commit()
#        cursor.close()
#        conn.close()
#        
#        return True
#    except Exception as e:
#        print(f"Error storing embeddings: {e}")
#        return False
#
#
#
#def process_blob(blob_service_client, container_name, blob):
#    """Process a single XSD blob from storage"""
#    blob_name = blob.name
#    print(f"Processing {blob_name}...")
#    
#    # Download blob
#    file_path = download_blob(blob_service_client, container_name, blob_name)
#    if not file_path:
#        return False
#    
#    try:
#        # Parse XSD file
#        elements, types = parse_xsd_file(file_path)
#        
#        # Clean up temporary file
#        os.unlink(file_path)
#        
#        if not elements and not types:
#            print(f"No schema information extracted from {blob_name}")
#            return False
#        
#        print(f"Extracted {len(elements)} elements and {len(types)} types from {blob_name}")
#        
#        # Create chunks from XSD content
#        chunks, metadata_list = create_chunks_from_xsd(elements, types, blob_name)
#        print(f"Created {len(chunks)} chunks from {blob_name}")
#        
#        # Ensure chunks are within token limits
#        final_chunks = []
#        final_metadata = []
#        
#        for chunk, metadata in zip(chunks, metadata_list):
#            sub_chunks = chunk_text(chunk)
#            for i, sub_chunk in enumerate(sub_chunks):
#                sub_metadata = metadata.copy()
#                if len(sub_chunks) > 1:
#                    sub_metadata["sub_chunk_index"] = i
#                    sub_metadata["total_sub_chunks"] = len(sub_chunks)
#                
#                final_chunks.append(sub_chunk)
#                final_metadata.append(sub_metadata)
#        
#        # Create embeddings
#        embeddings = create_embeddings(final_chunks)
#        print(f"Created {len(embeddings)} embeddings from {blob_name}")
#        
#        # Store in database
#        success = store_embeddings(final_chunks, embeddings, final_metadata)
#        if success:
#            print(f"Successfully processed {blob_name}")
#            return True
#        else:
#            print(f"Failed to store embeddings for {blob_name}")
#            return False
#    
#    except Exception as e:
#        print(f"Error processing {blob_name}: {e}")
#        if os.path.exists(file_path):
#            os.unlink(file_path)
#        return False


import time
import random
import json

def create_embeddings(chunks, batch_size=5):
    """Create embeddings for text chunks using OpenAI API with rate limiting"""
    all_embeddings = []
    
    # Process in smaller batches
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        batch_embeddings = []
        
        # Process each item in the batch
        for chunk in batch:
            max_retries = 5
            retry_count = 0
            backoff_time = 40  # Initial backoff time in seconds
            
            while retry_count < max_retries:
                try:
                    response = openai.Embedding.create(
                        input=chunk,
                        model=EMBEDDING_MODEL,
                        engine=EMBEDDING_MODEL
                    )
                    embedding = response['data'][0]['embedding']
                    batch_embeddings.append(embedding)
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    retry_count += 1
                    if "exceeded call rate limit" in str(e) or "rate limit" in str(e).lower():
                        # Rate limit error, apply backoff
                        jitter = random.uniform(0, 5)  # Add some randomness to prevent thundering herd
                        sleep_time = backoff_time + jitter
                        print(f"Rate limit hit. Retrying in {sleep_time:.1f} seconds... (Attempt {retry_count}/{max_retries})")
                        time.sleep(sleep_time)
                        backoff_time *= 1.5  # Exponential backoff
                    else:
                        print(f"Error creating embedding: {e}")
                        if retry_count >= max_retries:
                            batch_embeddings.append(None)
                        else:
                            # For other errors, wait a bit but not as long
                            time.sleep(5)
            
            if retry_count >= max_retries:
                print(f"Failed to create embedding after {max_retries} attempts")
                batch_embeddings.append(None)
                
        # Add batch results to overall results
        all_embeddings.extend(batch_embeddings)
        
        # Add delay between batches to avoid rate limiting
        if i + batch_size < len(chunks):
            time.sleep(2)  # Small delay between successful batches
    
    return all_embeddings

def store_embeddings(chunks, embeddings, metadata_list, batch_size=10):
    """Store text chunks and their embeddings in PostgreSQL in batches"""
    try:
        conn = psycopg2.connect(POSTGRES_CONNECTION)
        cursor = conn.cursor()
        
        total_items = len(chunks)
        successful_inserts = 0
        
        # Process in batches
        for i in range(0, total_items, batch_size):
            batch_end = min(i + batch_size, total_items)
            
            # Prepare batch data for insertion
            data = []
            for j in range(i, batch_end):
                chunk = chunks[j]
                embedding = embeddings[j]
                metadata = metadata_list[j]
                
                if embedding is not None:
                    data.append((
                        str(uuid.uuid4()),
                        chunk,
                        json.dumps(metadata),  # Convert dict to JSON string
                        embedding
                    ))
            
            if data:
                try:
                    # Insert data using execute_values for efficiency
                    execute_values(
                        cursor,
                        """
                        INSERT INTO xsd_knowledge_base (id, content, metadata, embedding)
                        VALUES %s
                        """,
                        data,
                        template="(%s, %s, %s, %s)"
                    )
                    
                    conn.commit()
                    successful_inserts += len(data)
                    print(f"Inserted batch {i//batch_size + 1}: {len(data)} records. Progress: {successful_inserts}/{total_items}")
                    
                except Exception as e:
                    conn.rollback()
                    print(f"Error inserting batch {i//batch_size + 1}: {e}")
            
            # Add a small delay between batches
            if batch_end < total_items:
                time.sleep(1)
        
        cursor.close()
        conn.close()
        
        return successful_inserts > 0
    except Exception as e:
        print(f"Error storing embeddings: {e}")
        return False
        
def process_blob(blob_service_client, container_name, blob):
    """Process a single XSD blob from storage"""
    blob_name = blob.name
    print(f"Processing {blob_name}...")
    
    # Download blob
    file_path = download_blob(blob_service_client, container_name, blob_name)
    if not file_path:
        return False
    
    try:
        # Parse XSD file
        elements, types = parse_xsd_file(file_path)
        
        # Clean up temporary file
        os.unlink(file_path)
        
        if not elements and not types:
            print(f"No schema information extracted from {blob_name}")
            return False
        
        print(f"Extracted {len(elements)} elements and {len(types)} types from {blob_name}")
        
        # Create chunks from XSD content
        chunks, metadata_list = create_chunks_from_xsd(elements, types, blob_name)
        print(f"Created {len(chunks)} chunks from {blob_name}")
        
        # Ensure chunks are within token limits
        final_chunks = []
        final_metadata = []
        
        for chunk, metadata in zip(chunks, metadata_list):
            sub_chunks = chunk_text(chunk)
            for i, sub_chunk in enumerate(sub_chunks):
                sub_metadata = metadata.copy()
                if len(sub_chunks) > 1:
                    sub_metadata["sub_chunk_index"] = i
                    sub_metadata["total_sub_chunks"] = len(sub_chunks)
                
                final_chunks.append(sub_chunk)
                final_metadata.append(sub_metadata)
        
        # Create embeddings with rate limiting
        print(f"Creating embeddings for {len(final_chunks)} chunks from {blob_name}...")
        embeddings = create_embeddings(final_chunks, batch_size=5)
        print(f"Created {sum(1 for e in embeddings if e is not None)} embeddings from {blob_name}")
        
        # Store in database in batches
        success = store_embeddings(final_chunks, embeddings, final_metadata, batch_size=10)
        if success:
            print(f"Successfully processed {blob_name}")
            return True
        else:
            print(f"Failed to store embeddings for {blob_name}")
            return False
    
    except Exception as e:
        print(f"Error processing {blob_name}: {e}")
        if os.path.exists(file_path):
            os.unlink(file_path)
        return False        


def main():
    """Main function to process all XSD blobs in a container"""
    # Connect to Blob Storage
    blob_service_client = connect_to_blob_storage()
    if not blob_service_client:
        return
    
    # Set up database
    if not setup_database():
        return
    
    # List XSD blobs
    blobs = list_blobs(blob_service_client, CONTAINER_NAME)
    if not blobs:
        print(f"No XSD files found in container '{CONTAINER_NAME}'")
        return
    
    print(f"Found {len(blobs)} XSD files in container '{CONTAINER_NAME}'")
    
    # Process each blob
    successful = 0
    for blob in blobs:
        if process_blob(blob_service_client, CONTAINER_NAME, blob):
            successful += 1
    
    print(f"Processed {successful} out of {len(blobs)} XSD files successfully")

if __name__ == "__main__":
    main()