from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import tempfile
import os
import json
from dotenv import load_dotenv
import google.generativeai as genai
from PyPDF2 import PdfReader
from docx import Document
import mimetypes
import uuid
from collections import Counter
import re
import logging
import sqlite3
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Gemini
GENAI_API_KEY = os.getenv("GENAI_API_KEY")
if not GENAI_API_KEY:
    raise ValueError("Please set the GENAI_API_KEY environment variable")

genai.configure(api_key=GENAI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash-exp')

app = FastAPI(title="Document AI Assistant")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    message: str
    document_id: str

class DocumentContent(BaseModel):
    document_id: str

class DocumentUploadResponse(BaseModel):
    document_id: str
    text: str

class DocumentStore:
    def __init__(self):
        self.db_path = Path("documents.db")
        self._initialize_db()
        logger.debug("Initialized DocumentStore")
        
    def _initialize_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    text TEXT,
                    chunks TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
    
    def add_document(self, text: str) -> str:
        doc_id = str(uuid.uuid4())
        chunks = self._split_text(text)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO documents (id, text, chunks)
                VALUES (?, ?, ?)
            ''', (doc_id, text, json.dumps(chunks)))
            conn.commit()
        
        logger.debug(f"Added document with ID: {doc_id}")
        logger.debug(f"Document chunks: {len(chunks)}")
        return doc_id
    
    def get_document(self, doc_id: str) -> Optional[str]:
        logger.debug(f"Retrieving document with ID: {doc_id}")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT text FROM documents WHERE id = ?', (doc_id,))
            result = cursor.fetchone()
            
        if result:
            text = result[0]
            logger.debug(f"Document found!")
            logger.debug(f"Document text length: {len(text)}")
            return text
        else:
            logger.debug(f"Document not found in store")
            logger.debug(f"Current documents in store: {len(self._get_all_documents())}")
            return None
    
    def get_chunks(self, doc_id: str) -> Optional[List[str]]:
        logger.debug(f"Retrieving chunks for document ID: {doc_id}")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT chunks FROM documents WHERE id = ?', (doc_id,))
            result = cursor.fetchone()
            
        if result:
            chunks = json.loads(result[0])
            logger.debug(f"Found {len(chunks)} chunks")
            if chunks:
                logger.debug(f"First chunk length: {len(chunks[0])}")
            return chunks
        else:
            logger.debug(f"Document not found in store")
            logger.debug(f"Current documents in store: {len(self._get_all_documents())}")
            return None
    
    def _get_all_documents(self) -> List[str]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM documents')
            return [row[0] for row in cursor.fetchall()]
    
    def find_relevant_chunks(self, doc_id: str, query: str, num_chunks: int = 3) -> List[str]:
        chunks = self.get_chunks(doc_id)
        if not chunks:
            logger.error(f"No chunks found for document ID: {doc_id}")
            return []
            
        cleaned_query = self._clean_text(query)
        query_tokens = set(cleaned_query.split())
        
        similarities = []
        for i, chunk in enumerate(chunks):
            cleaned_chunk = self._clean_text(chunk)
            chunk_tokens = set(cleaned_chunk.split())
            similarity = self._calculate_similarity(query_tokens, chunk_tokens)
            similarities.append((i, similarity))
            
        similarities.sort(key=lambda x: x[1], reverse=True)
        relevant_chunks = [chunks[i] for i, _ in similarities[:num_chunks]]
        
        logger.debug(f"Found {len(relevant_chunks)} relevant chunks")
        return relevant_chunks
    
    def _split_text(self, text: str) -> List[str]:
        chunks = []
        current_chunk = []
        current_length = 0
        
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed our chunk size
            if current_length + sentence_length > 1000:
                # Add the current chunk to the list
                chunks.append(' '.join(current_chunk))
                # Start a new chunk with this sentence
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        logger.debug(f"Split text into {len(chunks)} chunks")
        return chunks
    
    def _clean_text(self, text: str) -> str:
        # Remove special characters and make lowercase
        text = re.sub(r'[^\w\s]', '', text)
        return text.lower()
    
    def _calculate_similarity(self, tokens1: List[str], tokens2: List[str]) -> float:
        # Calculate Jaccard similarity
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        return intersection / union if union > 0 else 0

document_store = DocumentStore()

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    with open(file_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def extract_text_from_docx(file_path: str) -> str:
    """Extract text from a DOCX file."""
    doc = Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process a document file."""
    logger.debug(f"Uploading file: {file.filename}")
    
    # Get file type using mimetypes
    file_type = mimetypes.guess_type(file.filename)[0]
    
    if file_type not in ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
        raise HTTPException(status_code=400, detail="File type not supported. Please upload PDF or DOCX files.")

    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_file.flush()

        # Extract text based on file type
        if file_type == 'application/pdf':
            text = extract_text_from_pdf(temp_file.name)
        else:
            text = extract_text_from_docx(temp_file.name)

    # Clean up temporary file
    os.unlink(temp_file.name)

    # Store document
    doc_id = document_store.add_document(text)
    logger.debug(f"Uploaded document ID: {doc_id}")
    
    return DocumentUploadResponse(document_id=doc_id, text=text)

@app.post("/chat")
async def chat_with_document(message: ChatMessage):
    """Chat with the document using Gemini AI with RAG."""
    logger.debug(f"Chat request with document ID: {message.document_id}")
    logger.debug(f"Message: {message.message}")
    
    # Validate document ID
    all_documents = document_store._get_all_documents()
    if message.document_id not in all_documents:
        logger.error(f"Invalid document ID: {message.document_id}")
        logger.error(f"Available documents: {all_documents}")
        raise HTTPException(
            status_code=404,
            detail={
                "error": "Document not found",
                "available_documents": all_documents
            }
        )
    
    doc_id = message.document_id
    document_text = document_store.get_document(doc_id)
    if not document_text:
        logger.error(f"Document not found for ID: {doc_id}")
        raise HTTPException(status_code=404, detail="Document not found")

    # Find relevant chunks
    relevant_chunks = document_store.find_relevant_chunks(doc_id, message.message, num_chunks=3)
    logger.debug(f"Found {len(relevant_chunks)} relevant chunks")
    
    # Create context from relevant chunks
    context = "\n".join(relevant_chunks)
    
    # Generate response using Gemini with context
    prompt = f"""You are a helpful AI assistant. Here is the relevant context from the document:
    {context}

    Question: {message.message}
    Answer:"""
    
    response = model.generate_content(prompt)
    return {"response": str(response.text)}

@app.post("/summarize")
async def summarize_document(content: DocumentContent):
    """Summarize the document using Gemini AI with RAG."""
    logger.debug(f"Summarize request with document ID: {content.document_id}")
    
    doc_id = content.document_id
    document_text = document_store.get_document(doc_id)
    if not document_text:
        logger.error(f"Document not found for ID: {doc_id}")
        logger.error(f"Current documents in store: {len(document_store._get_all_documents())}")
        logger.error(f"Document IDs in store: {list(document_store._get_all_documents())}")
        raise HTTPException(status_code=404, detail="Document not found")

    # Find relevant chunks for summary
    relevant_chunks = document_store.find_relevant_chunks(doc_id, "summarize this document", num_chunks=5)
    if not relevant_chunks:
        logger.error(f"No relevant chunks found for document ID: {doc_id}")
        raise HTTPException(status_code=500, detail="Failed to find relevant chunks")
    
    logger.debug(f"Found {len(relevant_chunks)} relevant chunks")
    logger.debug(f"First relevant chunk length: {len(relevant_chunks[0]) if relevant_chunks else 0}")
    
    # Create context from relevant chunks
    context = "\n".join(relevant_chunks)
    
    # Generate summary using Gemini
    prompt = f"""Please provide a concise summary of the following document:
    {context}

    Summary:"""
    
    try:
        response = model.generate_content(prompt)
        return {"summary": str(response.text)}
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate summary")

@app.post("/revision-notes")
async def create_revision_notes(content: DocumentContent):
    """Create revision notes from the document using RAG."""
    doc_id = content.document_id
    document_text = document_store.get_document(doc_id)
    if not document_text:
        raise HTTPException(status_code=404, detail="Document not found")

    # Find relevant chunks for revision notes
    relevant_chunks = document_store.find_relevant_chunks(doc_id, "create revision notes for this document", num_chunks=5)
    
    # Create context from relevant chunks
    context = "\n".join(relevant_chunks)
    
    # Generate revision notes using Gemini
    prompt = f"""Please create comprehensive revision notes for the following document:
    {context}

    Revision Notes:"""
    
    response = model.generate_content(prompt)
    return {"notes": str(response.text)}

@app.get("/documents")
async def list_documents():
    """List all stored documents."""
    documents = document_store._get_all_documents()
    logger.debug(f"Listing documents: {documents}")
    return {"documents": documents}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)
