# Docx_backend

A FastAPI-based backend service for processing and analyzing documents using AI.

## Features

- Document upload and processing
- AI-powered chat with documents
- Document summarization
- Revision note generation

## Deployment

### Railway Deployment

1. Create a new project on Railway
2. Connect your GitHub repository
3. Deploy the application

### Local Development

1. Clone the repository
```bash
git clone https://github.com/Zem-0/Docx_backend.git
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Set up environment variables
Create a `.env` file with the following:
```
GENAI_API_KEY=your_api_key_here
```

4. Run the application
```bash
uvicorn main:app --reload
```

## API Endpoints

- `POST /upload` - Upload a document
- `POST /chat` - Chat with the document
- `POST /summarize` - Get document summary
- `POST /revision-notes` - Generate revision notes

## Requirements

- Python 3.8+
- FastAPI
- Google Generative AI API key
- Railway account for deployment
