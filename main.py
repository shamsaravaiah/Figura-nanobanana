import base64
import io
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional, Tuple, Union

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field
from PIL import Image

load_dotenv()

# Latency tuning: smaller input = faster upload + inference
MAX_IMAGE_DIMENSION = int(os.getenv("MAX_IMAGE_DIMENSION", "1024"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "85"))

# Reused HTTP client (set in lifespan)
http_client: Optional[httpx.AsyncClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(timeout=90.0)
    yield
    await http_client.aclose()
    http_client = None


app = FastAPI(title="Nano Banana (Gemini) Try-On API", lifespan=lifespan)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Flash is faster; Pro is higher quality. Override with GEMINI_IMAGE_MODEL.
MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image-preview")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

DEFAULT_TRY_ON_PROMPT = (
    "Put the garment from the second image on the person in the first image. "
    "Keep the person's pose, face, hair, body shape, and background unchanged. "
    "Keep lighting consistent and make the clothing look natural and realistic. "
    "Do not add text, logos, or watermarks."
)


def _strip_data_uri(b64_or_data_uri: str) -> str:
    """
    Accept either raw base64 OR a data URI like:
      data:image/png;base64,AAAA...
    Return raw base64 only.
    """
    s = (b64_or_data_uri or "").strip()
    if not s:
        return s
    m = re.match(r"^data:.*?;base64,(.*)$", s, flags=re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else s


def _resize_and_reencode_as_jpeg(
    b64_data: str, max_dim: int = MAX_IMAGE_DIMENSION, quality: int = JPEG_QUALITY
) -> Tuple[str, str]:
    """
    Decode base64 image, resize so longest side <= max_dim, re-encode as JPEG.
    Returns (base64_jpeg, "image/jpeg"). On any error, returns original b64 and
    "image/jpeg" (caller can pass through original mime if needed).
    """
    try:
        raw = base64.b64decode(b64_data)
    except Exception:
        return b64_data, "image/jpeg"
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return b64_data, "image/jpeg"
    w, h = img.size
    if w <= max_dim and h <= max_dim:
        pass
    else:
        if w >= h:
            new_w, new_h = max_dim, int(h * max_dim / w)
        else:
            new_w, new_h = int(w * max_dim / h), max_dim
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode(), "image/jpeg"


# --- Request / Response schemas ---

class TryOnRequest(BaseModel):
    garmentImage: str = Field(..., description="Base64 image or data URI for garment")
    personImage: str = Field(..., description="Base64 image or data URI for person")
    prompt: Optional[str] = Field(None, description="Optional edit instruction")
    mimeType: str = Field("image/png", description="Mime type for both images (e.g. image/png, image/jpeg)")


class TryOnSuccessResponse(BaseModel):
    success: bool = True
    resultImage: str
    processingTimeSeconds: float
    requestId: str
    model: str


class TryOnErrorResponse(BaseModel):
    success: bool = False
    error: str
    code: Optional[str] = None
    requestId: Optional[str] = None


TryOnResponse = Union[TryOnSuccessResponse, TryOnErrorResponse]


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/try-on", response_model=TryOnResponse)
async def try_on(body: TryOnRequest):
    request_id = str(uuid.uuid4())
    start = time.perf_counter()

    if not GEMINI_API_KEY:
        return TryOnErrorResponse(
            success=False,
            error="Server misconfiguration: GEMINI_API_KEY not set",
            code="CONFIG_ERROR",
            requestId=request_id,
        )

    person_b64 = _strip_data_uri(body.personImage)
    garment_b64 = _strip_data_uri(body.garmentImage)

    if not person_b64 or not garment_b64:
        return TryOnErrorResponse(
            success=False,
            error="garmentImage and personImage are required (base64 or data URI)",
            code="VALIDATION_ERROR",
            requestId=request_id,
        )

    # Resize and re-encode as JPEG to reduce payload size and speed up inference
    person_b64, out_mime = _resize_and_reencode_as_jpeg(person_b64)
    garment_b64, _ = _resize_and_reencode_as_jpeg(garment_b64)

    prompt = body.prompt or DEFAULT_TRY_ON_PROMPT

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": out_mime, "data": person_b64}},
                    {"inline_data": {"mime_type": out_mime, "data": garment_b64}},
                ],
            }
        ],
        "generationConfig": {
            "responseModalities": ["IMAGE"],
            "imageConfig": {"aspectRatio": "1:1", "imageSize": "1K"},
        },
    }

    url = f"{GEMINI_BASE_URL}/models/{MODEL}:generateContent"

    if not http_client:
        return TryOnErrorResponse(
            success=False,
            error="HTTP client not initialized",
            code="CONFIG_ERROR",
            requestId=request_id,
        )

    try:
        r = await http_client.post(
            url,
            headers={
                "x-goog-api-key": GEMINI_API_KEY,
                "Content-Type": "application/json",
            },
            json=payload,
        )
        elapsed = time.perf_counter() - start
        data = r.json()
    except httpx.TimeoutException:
        return TryOnErrorResponse(
            success=False,
            error="Request to Gemini timed out",
            code="TIMEOUT",
            requestId=request_id,
        )
    except Exception as e:
        return TryOnErrorResponse(
            success=False,
            error=str(e),
            code="UPSTREAM_ERROR",
            requestId=request_id,
        )

    if r.status_code != 200:
        # Gemini errors often come as: {"error": {"message": "...", "status": "..."}}
        err_msg = (
            (data.get("error") or {}).get("message")
            or data.get("message")
            or r.text
            or f"HTTP {r.status_code}"
        )
        return TryOnErrorResponse(
            success=False,
            error=err_msg,
            code=((data.get("error") or {}).get("status") or "API_ERROR"),
            requestId=request_id,
        )

    # Extract first IMAGE part from response
    # Response shape: candidates[0].content.parts[*].inline_data.data (base64) :contentReference[oaicite:2]{index=2}
    try:
        candidates = data.get("candidates") or []
        parts = (((candidates[0] or {}).get("content") or {}).get("parts")) or []
        out_b64 = None
        for p in parts:
            inline = p.get("inline_data") or p.get("inlineData")
            if inline and inline.get("data"):
                out_b64 = inline["data"]
                break
        if not out_b64:
            return TryOnErrorResponse(
                success=False,
                error="No image returned by model (no inline image data found).",
                code="NO_IMAGE",
                requestId=request_id,
            )
    except Exception:
        return TryOnErrorResponse(
            success=False,
            error="Unexpected response format from Gemini.",
            code="BAD_RESPONSE",
            requestId=request_id,
        )

    return TryOnSuccessResponse(
        success=True,
        resultImage=out_b64,  # base64 PNG/JPEG depending on model output
        processingTimeSeconds=round(elapsed, 2),
        requestId=request_id,
        model=MODEL,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)