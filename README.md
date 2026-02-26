# Figura-nanobanana

Simple FastAPI backend for Nano Banana virtual try-on: send garment + person images and optional prompt, get back the result image.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set `NANO_BANANA_API_KEY` (or use the existing `.env` if already configured).

## Run

```bash
uvicorn main:app --reload
```

API: http://localhost:8000  
Docs: http://localhost:8000/docs

## API

- **POST /try-on**

  **Request body:**
  ```json
  {
    "garmentImage": "base64 string",
    "personImage": "base64 string",
    "prompt": "optional string"
  }
  ```

  **Success response:**
  ```json
  {
    "success": true,
    "resultImageUrl": "string (optional)",
    "resultImage": "base64 (optional)",
    "processingTimeSeconds": 0.0,
    "requestId": "string (optional)"
  }
  ```

  **Error response:**
  ```json
  {
    "success": false,
    "error": "string",
    "code": "string (optional)"
  }
  ```
