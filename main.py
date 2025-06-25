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
        self.documents = {}
        logger.debug("Initialized DocumentStore")
        
    def add_document(self, text: str) -> str:
        doc_id = str(uuid.uuid4())
        chunks = self._split_text(text)
        self.documents[doc_id] = {
            "text": text,
            "chunks": chunks
        }
        logger.debug(f"Added document with ID: {doc_id}")
        logger.debug(f"Document chunks: {len(chunks)}")
        return doc_id
    
    def get_document(self, doc_id: str) -> Optional[str]:
        logger.debug(f"Retrieving document with ID: {doc_id}")
        doc = self.documents.get(doc_id)
        if doc:
            logger.debug(f"Document found!")
            logger.debug(f"Document text length: {len(doc.get('text', ''))}")
            logger.debug(f"Number of chunks: {len(doc.get('chunks', []))}")
        else:
            logger.debug(f"Document not found in store")
            logger.debug(f"Current documents in store: {len(self.documents)}")
            logger.debug(f"Document IDs in store: {list(self.documents.keys())}")
        return doc.get("text") if doc else None

    def get_chunks(self, doc_id: str) -> Optional[List[str]]:
        logger.debug(f"Retrieving chunks for document ID: {doc_id}")
        doc = self.documents.get(doc_id)
        if doc:
            chunks = doc.get("chunks", [])
            logger.debug(f"Found {len(chunks)} chunks")
            if chunks:
                logger.debug(f"First chunk length: {len(chunks[0])}")
        else:
            logger.debug(f"Document not found in store")
            logger.debug(f"Current documents in store: {len(self.documents)}")
            logger.debug(f"Document IDs in store: {list(self.documents.keys())}")
        return doc.get("chunks") if doc else None
    
    def find_relevant_chunks(self, doc_id: str, query: str, num_chunks: int = 3) -> List[str]:
        chunks = self.get_chunks(doc_id)
        if not chunks:
            return []
            
        # Clean and tokenize query and chunks
        query_tokens = self._clean_text(query).split()
        chunk_scores = []
        
        for chunk in chunks:
            chunk_tokens = self._clean_text(chunk).split()
            score = self._calculate_similarity(query_tokens, chunk_tokens)
            chunk_scores.append((chunk, score))
            
        # Sort chunks by similarity score and get top N
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in chunk_scores[:num_chunks]]

    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks of about 1000 characters."""
        chunk_size = 1000
        chunks = []
        current_chunk = ""
        
        # Split text by paragraphs
        paragraphs = text.split('\n')
        
        for para in paragraphs:
            if len(current_chunk) + len(para) + 1 <= chunk_size:
                current_chunk += para + "\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean text by removing special characters and making lowercase."""
        # Remove special characters and make lowercase
        text = re.sub(r'[^\w\s]', '', text)
        return text.lower()
    
    def _calculate_similarity(self, tokens1: List[str], tokens2: List[str]) -> float:
        """Calculate Jaccard similarity between two sets of tokens."""
        # Simple word overlap similarity
        tokens1 = set(tokens1)
        tokens2 = set(tokens2)
        
        # Calculate Jaccard similarity
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        if not union:
            return 0.0
            
        return len(intersection) / len(union)

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
        logger.error(f"Current documents in store: {len(document_store.documents)}")
        logger.error(f"Document IDs in store: {list(document_store.documents.keys())}")
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)
