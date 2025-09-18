import io
import os
import random
import numpy as np
import torch
import soundfile as sf
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import StreamingResponse, JSONResponse
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional

# # Language detection
# try:
#     from langdetect import detect
#     LANGDETECT_AVAILABLE = True
# except ImportError:
#     LANGDETECT_AVAILABLE = False
#     print("⚠️  langdetect not available. Install with: pip install langdetect")

# ---------------------------------------------------
# Global Setup
# ---------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

map_location = torch.device(DEVICE)
print(f"🚀 Running Chatterbox on: {DEVICE}")

# Patch torch.load globally so all checkpoints load properly
torch_load_original = torch.load
def patched_torch_load(*args, **kwargs):
    if "map_location" not in kwargs:
        kwargs["map_location"] = map_location
    return torch_load_original(*args, **kwargs)
torch.load = patched_torch_load

MODEL = None

def get_or_load_model():
    global MODEL
    if MODEL is None:
        print("Loading ChatterboxMultilingualTTS...")
        MODEL = ChatterboxMultilingualTTS.from_pretrained(DEVICE)
        if hasattr(MODEL, "to") and str(MODEL.device) != DEVICE:
            MODEL.to(DEVICE)
        print(f"✅ Model loaded successfully. Sample rate = {MODEL.sr} Hz")
    return MODEL

get_or_load_model()

app = FastAPI(title="Chatterbox TTS API", version="1.0")

# ---------------------------------------------------
# Helpers
# ---------------------------------------------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

# ---------------------------------------------------
# Routes
# ---------------------------------------------------

@app.get("/languages")
async def get_languages():
    """Return all supported languages."""
    return JSONResponse(content={"supported_languages": SUPPORTED_LANGUAGES})

@app.post("/tts")
async def synthesize_tts(
    text: str = Form(..., description="Text to synthesize (max 300 chars)"),
    language_id: str = Form(..., description="Language code, e.g., en, fr, de, hi"),
    exaggeration: float = Form(0.5, description="Speech expressiveness (0.25–2.0)"),
    temperature: float = Form(0.8, description="Randomness in generation (0.05–5.0)"),
    seed: int = Form(0, description="Random seed, 0 for random"),
    cfg_weight: float = Form(0.5, description="CFG/Pace weight (0.2–1.0, 0 for transfer)"),
    raw: bool = Form(False, description="Return raw PCM if True, WAV if False"),
    audio_prompt: UploadFile | None = None
):
    """
    Generate speech from text. Optionally provide a reference audio file
    (voice cloning).
    """

    model = get_or_load_model()

    if seed != 0:
        set_seed(seed)

    generate_kwargs = {
        "exaggeration": exaggeration,
        "temperature": temperature,
        "cfg_weight": cfg_weight,
    }

    # Handle optional reference audio
    if audio_prompt:
        tmp_path = f"/tmp/{audio_prompt.filename}"
        with open(tmp_path, "wb") as f:
            f.write(await audio_prompt.read())
        generate_kwargs["audio_prompt_path"] = tmp_path
        print(f"🎙 Using reference audio: {tmp_path}")
    else:
        print("🎙 No reference audio provided.")

    print(f"📝 Generating: '{text[:50]}...' in lang={language_id}")

    wav = model.generate(
        text[:300],
        language_id=language_id,
        **generate_kwargs,
    )

    # Handle raw PCM output
    if raw:
        # Convert to float32 PCM
        pcm_data = wav.squeeze(0).cpu().numpy().astype(np.float32)
        buf = io.BytesIO(pcm_data.tobytes())
        buf.seek(0)
        
        # Return PCM with appropriate headers
        response = StreamingResponse(
            buf, 
            media_type="application/octet-stream",
            headers={
                "X-Sample-Rate": str(model.sr),
                "X-Num-Channels": "1",
                "X-Sample-Format": "float32"
            }
        )
        return response
    else:
        # Convert to WAV bytes (default behavior)
        buf = io.BytesIO()
        sf.write(buf, wav.squeeze(0).cpu().numpy(), model.sr, format="WAV")
        buf.seek(0)
        
        return StreamingResponse(buf, media_type="audio/wav")

# ---------------------------------------------------
# Health check
# ---------------------------------------------------
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# ---------------------------------------------------
# Main entry point with ngrok support
# ---------------------------------------------------
if __name__ == "__main__":
    import argparse
    import uvicorn
    
    parser = argparse.ArgumentParser(description="Chatterbox TTS API with optional ngrok tunneling")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    parser.add_argument("--ngrok", action="store_true", help="Expose API via ngrok tunnel")
    parser.add_argument("--ngrok-token", help="ngrok authentication token (optional if set in .env)")
    
    args = parser.parse_args()
    
    if args.ngrok:
        try:
            from pyngrok import ngrok
            
            # Set auth token if provided
            ngrok_token = args.ngrok_token or os.getenv("NGROK_AUTH_TOKEN")
            if ngrok_token:
                ngrok.set_auth_token(ngrok_token)
            
            # Start ngrok tunnel
            public_url = ngrok.connect(args.port)
            print(f"🌍 Public URL: {public_url}")
            print("📋 API Documentation: {public_url}/docs")
        except ImportError:
            print("❌ pyngrok not installed. Install with: pip install pyngrok")
        except Exception as e:
            print(f"❌ Failed to create ngrok tunnel: {e}")
    
    # Run the uvicorn server
    uvicorn.run("app_pcm:app", host=args.host, port=args.port, reload=True)
