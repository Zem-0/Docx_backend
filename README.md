# Docx_backend

A FastAPI-based backend service for processing and analyzing documents using AI.

## Features

- Document upload and processing
- AI-powered chat with documents
- Document summarization
- Revision note generation

## Deployment

### Render Deployment

1. Create a new account at [Render](https://render.com)
2. Create a new Web Service
3. Connect your GitHub repository
4. Configure the service:
   - Build Command: `docker build -t web .`
   - Start Command: `docker run -p $PORT:8000 web`
   - Port: `8000`
5. Set environment variables:
   - `GENAI_API_KEY`: Your Google Generative AI API key

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

- Python 3.9+
- FastAPI
- Google Generative AI API key
- Render account for deployment
