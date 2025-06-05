import os
import uuid
import tempfile
from typing import List, Dict, Any, Optional
import json
import time
import random
import fnmatch

# Azure Blob Storage
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError

# Azure PostgreSQL
import psycopg2
from psycopg2.extras import execute_values

# Embedding and tokenization
import tiktoken
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure Blob Storage parameters
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = os.getenv("CONTAINER_NAME")

# Azure PostgreSQL connection parameters
DB_HOST = os.getenv("POSTGRES_HOST")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB")
DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_CONNECTION = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Azure OpenAI parameters
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-ada-002")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")

# Constants for embeddings
EMBEDDING_DIMENSIONS = 1536  # Dimensions for Azure OpenAI embeddings
EMBEDDING_ENCODING = "cl100k_base"  # Encoding for text-embedding-ada-002
MAX_TOKENS = 7000  # Maximum tokens for embeddings (reduced from 8000 to provide larger safety margin)
BATCH_SIZE = 10  # Number of embeddings to process in a batch

# Initialize tokenizer
tokenizer = tiktoken.get_encoding(EMBEDDING_ENCODING)

# Configure Azure OpenAI Embeddings
embed_model = AzureOpenAIEmbedding(
    model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
    deployment_name=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION
)

class AzureVectorDBPopulator:
    """Class to handle embedding and populating Azure PostgreSQL vector database"""
    
    def __init__(self):
        """Initialize connections to Azure services"""
        self.blob_service_client = self._connect_to_blob_storage()
        self._setup_database()
    
    def _connect_to_blob_storage(self) -> Optional[BlobServiceClient]:
        """Connect to Azure Blob Storage"""
        try:
            blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
            print("Successfully connected to Azure Blob Storage")
            return blob_service_client
        except Exception as e:
            print(f"Error connecting to Azure Blob Storage: {e}")
            return None
    
    def _setup_database(self) -> bool:
        """Set up PostgreSQL database with pgvector extension"""
        try:
            conn = psycopg2.connect(POSTGRES_CONNECTION)
            cursor = conn.cursor()
            
            # Create pgvector extension if not exists
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create table for vector embeddings
            cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS document_embeddings (
                id UUID PRIMARY KEY,
                content TEXT,
                metadata JSONB,
                embedding vector({EMBEDDING_DIMENSIONS})
            );
            """)
            
            # Create index for vector similarity search
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS document_embeddings_idx 
            ON document_embeddings 
            USING ivfflat (embedding vector_cosine_ops) 
            WITH (lists = 100);
            """)
            
            conn.commit()
            cursor.close()
            conn.close()
            
            print("Successfully set up PostgreSQL database with pgvector extension")
            return True
        except Exception as e:  
            print(f"Error setting up database: {e}")
            return False
    
    def list_blobs(self, container_name: str, file_pattern: Optional[str] = None) -> List[Any]:
        """
        List blobs in a container, filtered by pattern
        
        Args:
            container_name: Name of the container to list blobs from
            file_pattern: Optional pattern to filter files (supports wildcards)
            
        Returns:
            List of blob objects
        """
        try:
            container_client = self.blob_service_client.get_container_client(container_name)
            all_blobs = list(container_client.list_blobs())
            
            # If no pattern or pattern is "*", return all blobs
            if not file_pattern or file_pattern == "*":
                return all_blobs
                
            # Check if pattern is an extension (starts with .)
            if file_pattern.startswith('.'):
                return [blob for blob in all_blobs if blob.name.lower().endswith(file_pattern.lower())]
            
            # Handle wildcard patterns using fnmatch
            filtered_blobs = []
            for blob in all_blobs:
                if fnmatch.fnmatch(blob.name.lower(), file_pattern.lower()):
                    filtered_blobs.append(blob)
            
            return filtered_blobs
                
        except ResourceNotFoundError:
            print(f"Container '{container_name}' not found")
            return []
        except Exception as e:
            print(f"Error listing blobs: {e}")
            return []
    
    def download_blob_content(self, container_name: str, blob_name: str) -> Optional[str]:
        """
        Download a blob's content as text
        
        Args:
            container_name: Name of the container
            blob_name: Name of the blob to download
            
        Returns:
            Content of the blob as string or None if error
        """
        try:
            container_client = self.blob_service_client.get_container_client(container_name)
            blob_client = container_client.get_blob_client(blob_name)
            
            # Download blob content
            download_stream = blob_client.download_blob()
            content = download_stream.readall().decode('utf-8')
            
            return content
        except Exception as e:
            print(f"Error downloading blob {blob_name}: {e}")
            return None
    
    def chunk_text_by_tokens(self, text: str, max_tokens: int = MAX_TOKENS) -> List[str]:
        """
        Split text into chunks based on token count with a more conservative approach
        
        Args:
            text: Text to split into chunks
            max_tokens: Maximum number of tokens per chunk
            
        Returns:
            List of text chunks
        """
        # First, check if the entire text is within limits
        tokens = tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return [text]
        
        # Use a more conservative approach - split by paragraphs first
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for paragraph in paragraphs:
            # Check token count of this paragraph
            paragraph_tokens = len(tokenizer.encode(paragraph))
            
            # If a single paragraph exceeds the limit, we need to split it further
            if paragraph_tokens > max_tokens:
                # Process this large paragraph separately
                if current_chunk:  # Save the current chunk if it exists
                    chunks.append(current_chunk)
                    current_chunk = ""
                    current_tokens = 0
                
                # Split large paragraph by sentences or smaller units
                sentences = paragraph.split('. ')
                for sentence in sentences:
                    sentence_tokens = len(tokenizer.encode(sentence))
                    
                    # If even a single sentence is too large, split by characters
                    if sentence_tokens > max_tokens:
                        # Split the sentence into smaller parts
                        for i in range(0, len(sentence), 1000):  # Use character count as approximation
                            part = sentence[i:i+1000]
                            part_tokens = len(tokenizer.encode(part))
                            
                            # If still too large, reduce further
                            while part_tokens > max_tokens and part:
                                part = part[:len(part)//2]  # Take half
                                part_tokens = len(tokenizer.encode(part))
                            
                            if part:
                                chunks.append(part)
                    else:
                        # Check if adding this sentence would exceed the limit
                        if current_tokens + sentence_tokens + 1 > max_tokens:  # +1 for potential period
                            chunks.append(current_chunk)
                            current_chunk = sentence
                            current_tokens = sentence_tokens
                        else:
                            if current_chunk:
                                current_chunk += ". " + sentence
                            else:
                                current_chunk = sentence
                            current_tokens += sentence_tokens + 1  # +1 for the period
            else:
                # Check if adding this paragraph would exceed the limit
                if current_tokens + paragraph_tokens + 2 > max_tokens:  # +2 for '\n\n'
                    chunks.append(current_chunk)
                    current_chunk = paragraph
                    current_tokens = paragraph_tokens
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + paragraph
                    else:
                        current_chunk = paragraph
                    current_tokens += paragraph_tokens + 2  # +2 for '\n\n'
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(current_chunk)
        
        # Final verification of token counts
        verified_chunks = []
        for chunk in chunks:
            chunk_tokens = len(tokenizer.encode(chunk))
            
            # If still too large, use the basic splitting method
            if chunk_tokens > max_tokens:
                print(f"Warning: Chunk with {chunk_tokens} tokens still exceeds limit. Further splitting...")
                # Use basic token splitting as a fallback
                basic_chunks = self._basic_token_split(chunk, max_tokens)
                verified_chunks.extend(basic_chunks)
            else:
                verified_chunks.append(chunk)
        
        return verified_chunks
    
    def _basic_token_split(self, text: str, max_tokens: int) -> List[str]:
        """
        Basic token-based splitting as a fallback method
        
        Args:
            text: Text to split
            max_tokens: Maximum tokens per chunk
            
        Returns:
            List of text chunks
        """
        tokens = tokenizer.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), max_tokens // 2):  # Use half the max to be safe
            chunk_tokens = tokens[i:i + (max_tokens // 2)]
            chunk_text = tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
        
        return chunks
    
    def get_embeddings(self, chunks: List[str]) -> List[List[float]]:
        """
        Generate embeddings for text chunks with rate limiting and retry logic
        
        Args:
            chunks: List of text chunks to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for i, chunk in enumerate(chunks):
            # Verify token count before sending to API
            token_count = len(tokenizer.encode(chunk))
            if token_count > MAX_TOKENS:
                print(f"Warning: Chunk {i+1} has {token_count} tokens, exceeding the {MAX_TOKENS} limit.")
                print("Reducing chunk size...")
                
                # Reduce chunk size
                reduced_chunk = chunk[:int(len(chunk) * 0.7)]  # Take 70% of the text
                reduced_token_count = len(tokenizer.encode(reduced_chunk))
                
                # If still too large, use more aggressive reduction
                if reduced_token_count > MAX_TOKENS:
                    print(f"Still too large with {reduced_token_count} tokens. Using aggressive reduction.")
                    # Split into much smaller chunks and use the first one
                    sub_chunks = self._basic_token_split(chunk, MAX_TOKENS // 2)
                    if sub_chunks:
                        chunk = sub_chunks[0]
                    else:
                        print(f"Failed to reduce chunk {i+1}. Skipping.")
                        embeddings.append(None)
                        continue
                else:
                    chunk = reduced_chunk
            
            # Add retry logic for robustness
            max_retries = 3
            retry_count = 0
            backoff_time = 2  # Initial backoff time in seconds
            
            while retry_count < max_retries:
                try:
                    # Get embedding using Azure OpenAI
                    embedding = embed_model.get_text_embedding(chunk)
                    embeddings.append(embedding)
                    
                    # Print progress
                    if (i + 1) % 5 == 0 or (i + 1) == len(chunks):
                        print(f"Generated embeddings for {i + 1}/{len(chunks)} chunks")
                    
                    # Add a small delay to avoid rate limiting
                    if i < len(chunks) - 1:
                        time.sleep(0.5)
                    
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    retry_count += 1
                    print(f"Error generating embedding (attempt {retry_count}/{max_retries}): {e}")
                    
                    # Check if error is due to token limit
                    if "maximum context length" in str(e) or "token" in str(e).lower():
                        print(f"Token limit exceeded. Reducing chunk size and retrying...")
                        # Try with a smaller chunk
                        reduced_chunk = chunk[:int(len(chunk) * 0.6)]  # Take 60% of the text
                        try:
                            embedding = embed_model.get_text_embedding(reduced_chunk)
                            embeddings.append(embedding)
                            print(f"Successfully embedded reduced chunk {i + 1}")
                            break
                        except Exception as sub_e:
                            print(f"Error embedding reduced chunk: {sub_e}")
                    
                    if retry_count < max_retries:
                        # Add jitter to backoff time
                        jitter = random.uniform(0, 1)
                        sleep_time = backoff_time + jitter
                        print(f"Retrying in {sleep_time:.1f} seconds...")
                        time.sleep(sleep_time)
                        backoff_time *= 2  # Exponential backoff
                    else:
                        print(f"Failed to generate embedding after {max_retries} attempts")
                        embeddings.append(None)
        
        return embeddings
    
    def store_embeddings(self, chunks: List[str], embeddings: List[List[float]], 
                         metadata_list: List[Dict[str, Any]]) -> bool:
        """
        Store text chunks and their embeddings in PostgreSQL
        
        Args:
            chunks: List of text chunks
            embeddings: List of embedding vectors
            metadata_list: List of metadata dictionaries
            
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = psycopg2.connect(POSTGRES_CONNECTION)
            cursor = conn.cursor()
            
            total_items = len(chunks)
            successful_inserts = 0
            
            # Process in batches
            for i in range(0, total_items, BATCH_SIZE):
                batch_end = min(i + BATCH_SIZE, total_items)
                
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
                            json.dumps(metadata),
                            embedding
                        ))
                
                if data:
                    try:
                        # Insert data using execute_values for efficiency
                        execute_values(
                            cursor,
                            """
                            INSERT INTO document_embeddings (id, content, metadata, embedding)
                            VALUES %s
                            """,
                            data,
                            template="(%s, %s, %s, %s)"
                        )
                        
                        conn.commit()
                        successful_inserts += len(data)
                        print(f"Inserted batch {i//BATCH_SIZE + 1}: {len(data)} records. Progress: {successful_inserts}/{total_items}")
                        
                    except Exception as e:
                        conn.rollback()
                        print(f"Error inserting batch {i//BATCH_SIZE + 1}: {e}")
                
                # Add a small delay between batches
                if batch_end < total_items:
                    time.sleep(1)
            
            cursor.close()
            conn.close()
            
            return successful_inserts > 0
        except Exception as e:
            print(f"Error storing embeddings: {e}")
            return False
    
    def process_blob(self, container_name: str, blob: Any) -> bool:
        """
        Process a single blob: download, chunk, embed, and store
        
        Args:
            container_name: Name of the container
            blob: Blob object to process
            
        Returns:
            True if successful, False otherwise
        """
        blob_name = blob.name
        print(f"\nProcessing {blob_name}...")
        
        # Download blob content
        content = self.download_blob_content(container_name, blob_name)
        if not content:
            return False
        
        try:
            # Determine file type from extension
            file_type = blob_name.split('.')[-1].lower() if '.' in blob_name else 'unknown'
            
            # Create chunks from content with improved chunking method
            chunks = self.chunk_text_by_tokens(content, MAX_TOKENS)
            print(f"Split content into {len(chunks)} chunks")
            
            # Prepare metadata for each chunk
            metadata_list = []
            for i, chunk in enumerate(chunks):
                metadata = {
                    "source": blob_name,
                    "file_type": file_type,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "timestamp": time.time()
                }
                metadata_list.append(metadata)
            
            # Generate embeddings
            print(f"Generating embeddings for {len(chunks)} chunks...")
            embeddings = self.get_embeddings(chunks)
            
            # Store in database
            print("Storing embeddings in PostgreSQL...")
            success = self.store_embeddings(chunks, embeddings, metadata_list)
            
            if success:
                print(f"Successfully processed {blob_name}")
                return True
            else:
                print(f"Failed to store embeddings for {blob_name}")
                return False
        
        except Exception as e:
            print(f"Error processing {blob_name}: {e}")
            return False
    
    def process_all_blobs(self, container_name: str, file_pattern: Optional[str] = None) -> None:
        """
        Process all blobs in a container
        
        Args:
            container_name: Name of the container
            file_pattern: Optional file pattern (supports wildcards)
        """
        # List blobs
        blobs = self.list_blobs(container_name, file_pattern)
        if not blobs:
            pattern_desc = "files" if not file_pattern or file_pattern == "*" else f"files matching '{file_pattern}'"
            print(f"No {pattern_desc} found in container '{container_name}'")
            return
        
        print(f"Found {len(blobs)} files in container '{container_name}':")
        for i, blob in enumerate(blobs, 1):
            print(f"{i}. {blob.name}")
        
        # Ask if user wants to process all files or select specific ones
        if len(blobs) > 1:
            choice = input("\nDo you want to process all files? (y/n): ").lower()
            if choice != 'y':
                indices = input("Enter file numbers to process (comma-separated, e.g., 1,3,5): ")
                try:
                    selected_indices = [int(idx.strip()) - 1 for idx in indices.split(',')]
                    blobs = [blobs[idx] for idx in selected_indices if 0 <= idx < len(blobs)]
                    print(f"Selected {len(blobs)} files for processing")
                except:
                    print("Invalid input. Processing all files.")
        
        # Process each blob
        successful = 0
        for blob in blobs:
            if self.process_blob(container_name, blob):
                successful += 1
        
        print(f"\nProcessed {successful} out of {len(blobs)} files successfully")


def main():
    """Main function to run the Azure Vector DB Populator"""
    print("Azure Vector DB Populator")
    print("=========================")
    
    # Initialize the populator
    populator = AzureVectorDBPopulator()
    
    # Get container name from environment variable or user input
    container_name = CONTAINER_NAME
    if not container_name:
        container_name = input("Enter container name: ")
    
    # Ask for optional file pattern with clear instructions
    print("\nFile Filter Options:")
    print("1. Press Enter to list ALL files")
    print("2. Enter a file extension (e.g., .xsd)")
    print("3. Enter a filename pattern (e.g., fpml*)")
    file_pattern = input("\nEnter your choice (or leave blank for all files): ")
    
    # If blank, explicitly set to None to list all files
    if not file_pattern:
        file_pattern = None
    
    # Process all blobs
    populator.process_all_blobs(container_name, file_pattern)


if __name__ == "__main__":
    main()