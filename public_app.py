import io
import os
import random
import numpy as np
import torch
import soundfile as sf
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import StreamingResponse, JSONResponse
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
import uvicorn
import sys
import subprocess
import threading
import time
import warnings
import gradio as gr

# Try to import ngrok, install if not available
try:
    from pyngrok import ngrok
    NGROK_AVAILABLE = True
except ImportError:
    NGROK_AVAILABLE = False
    print("⚠️  pyngrok not available. To install: pip install pyngrok")

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

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

# Load model at startup
try:
    get_or_load_model()
    print("✅ Model ready for inference")
except Exception as e:
    print(f"❌ Error loading model: {e}")

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

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Chatterbox TTS API is running!",
        "docs": "/docs",
        "supported_languages": SUPPORTED_LANGUAGES
    }

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

    # Convert to WAV bytes
    buf = io.BytesIO()
    sf.write(buf, wav.squeeze(0).cpu().numpy(), model.sr, format="WAV")
    buf.seek(0)

    return StreamingResponse(buf, media_type="audio/wav")

# ---------------------------------------------------
# Health check
# ---------------------------------------------------
@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Chatterbox TTS API is running"}

# Add a simple test endpoint to verify the API is working
@app.get("/test")
async def test_endpoint():
    return {"message": "API is working correctly!", "endpoints": ["/tts", "/languages", "/docs", "/health"]}

# ---------------------------------------------------
# Gradio Interface
# ---------------------------------------------------
def create_gradio_interface():
    """Create a Gradio interface for the TTS model"""
    
    def tts_wrapper(text, language_id, exaggeration, temperature, seed, cfg_weight, audio_prompt):
        """Wrapper function for Gradio interface"""
        try:
            model = get_or_load_model()
            
            if seed != 0:
                set_seed(seed)
                
            generate_kwargs = {
                "exaggeration": exaggeration,
                "temperature": temperature,
                "cfg_weight": cfg_weight,
            }
            
            # Handle audio prompt
            if audio_prompt is not None:
                # For file paths, we can use them directly
                if isinstance(audio_prompt, str):
                    generate_kwargs["audio_prompt_path"] = audio_prompt
                # For numpy arrays (uploaded files), save to temp file
                elif isinstance(audio_prompt, tuple) and len(audio_prompt) == 2:
                    import tempfile
                    import os
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                        sf.write(tmp_file.name, audio_prompt[1], audio_prompt[0])
                        generate_kwargs["audio_prompt_path"] = tmp_file.name
                        # Clean up the temporary file after use
                        import atexit
                        atexit.register(lambda: os.unlink(tmp_file.name) if os.path.exists(tmp_file.name) else None)
            
            print(f"📝 Generating: '{text[:50]}...' in lang={language_id}")
            
            wav = model.generate(
                text[:300],
                language_id=language_id,
                **generate_kwargs,
            )
            
            return (model.sr, wav.squeeze(0).cpu().numpy())
        except Exception as e:
            print(f"Error in TTS generation: {e}")
            raise e

    # Create Gradio interface
    with gr.Blocks() as interface:
        gr.Markdown(
            """
            # Chatterbox TTS API Demo
            Generate high-quality speech from text with optional reference audio styling.
            
            ## API Endpoints
            You can also use the API endpoints directly:
            - `/tts` - POST endpoint for text-to-speech generation
            - `/languages` - GET endpoint for supported languages
            - `/docs` - Interactive API documentation
            
            ## Usage Examples
            ```bash
            # Get supported languages
            curl -X GET "http://localhost:8000/languages"
            
            # Generate speech
            curl -X POST "http://localhost:8000/tts" \\
              -F "text=Hello world" \\
              -F "language_id=en" \\
              --output output.wav
            ```
            """
        )
        
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(
                    value="Hello, welcome to Chatterbox TTS! This is a demonstration of our multilingual text-to-speech capabilities.",
                    label="Text to synthesize (max chars 300)",
                    max_lines=5
                )
                
                language_id = gr.Dropdown(
                    choices=list(SUPPORTED_LANGUAGES.keys()),
                    value="en",
                    label="Language",
                    info="Select the language for text-to-speech synthesis"
                )
                
                ref_wav = gr.Audio(
                    sources=["upload"],
                    type="filepath",  # Changed to filepath for easier handling
                    label="Reference Audio File (Optional)"
                )
                
                exaggeration = gr.Slider(
                    0.25, 2, step=0.05, label="Exaggeration (Neutral = 0.5)", value=0.5
                )
                
                cfg_weight = gr.Slider(
                    0.2, 1, step=0.05, label="CFG/Pace", value=0.5
                )

                with gr.Accordion("More options", open=False):
                    seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                    temp = gr.Slider(0.05, 5, step=0.05, label="Temperature", value=0.8)

                run_btn = gr.Button("Generate", variant="primary")

            with gr.Column():
                audio_output = gr.Audio(label="Output Audio")

        run_btn.click(
            fn=tts_wrapper,
            inputs=[
                text,
                language_id,
                exaggeration,
                temp,
                seed_num,
                cfg_weight,
                ref_wav
            ],
            outputs=[audio_output],
        )
        
    return interface

# ---------------------------------------------------
# Server Functions
# ---------------------------------------------------
def start_server(port=8000, use_gradio=False, expose_api=False, use_ngrok=False):
    """
    Start the FastAPI server with optional Gradio interface.
    
    Args:
        port (int): Local port to run the FastAPI server on
        use_gradio (bool): Whether to launch a Gradio interface
        expose_api (bool): Whether to expose the API with a public URL
        use_ngrok (bool): Whether to use ngrok for public exposure
    """
    
    if use_ngrok and NGROK_AVAILABLE:
        print(f"\n🌍 Exposing FastAPI server with ngrok on port {port}...")
        try:
            # Get ngrok auth token from environment
            ngrok_token = os.getenv("NGROK_AUTH_TOKEN")
            if ngrok_token:
                ngrok.set_auth_token(ngrok_token)
                print("🔐 Ngrok auth token set from environment")
            else:
                print("⚠️  No NGROK_AUTH_TOKEN found in environment. Using free ngrok tunnel.")
            
            # Start uvicorn server in a separate thread
            def run_server():
                uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
            
            server_thread = threading.Thread(target=run_server, daemon=True)
            server_thread.start()
            
            # Wait a moment for server to start
            time.sleep(2)
            
            # Create ngrok tunnel
            public_url = ngrok.connect(port)
            print(f"✅ FastAPI server is now publicly accessible!")
            print(f"🌍 Public URL: {public_url}")
            print("📝 API endpoints:")
            print(f"   - {public_url}/tts (text-to-speech)")
            print(f"   - {public_url}/languages (supported languages)")
            print(f"   - {public_url}/docs (API documentation)")
            print(f"   - {public_url}/health (health check)")
            
            if use_gradio:
                print("🎨 Note: Gradio interface not available with ngrok mode.")
                print("   Use --expose-api flag for Gradio interface with public URL.")
                
            # Keep the script running
            try:
                while True:
                    time.sleep(60)
            except KeyboardInterrupt:
                print("\n🛑 Shutting down server...")
                try:
                    ngrok.kill()
                except:
                    pass
                print("👋 Server stopped")
                
        except Exception as e:
            print(f"❌ Error exposing API with ngrok: {e}")
            print("💡 Make sure you have set NGROK_AUTH_TOKEN in your environment for persistent tunnels")
    elif expose_api:
        print(f"\n🌍 Exposing FastAPI server with public sharing...")
        try:
            # Create Gradio interface
            interface = create_gradio_interface()
            
            # Mount our FastAPI app on the Gradio interface
            interface.app = app
            
            # Launch with sharing
            interface.launch(
                server_name="0.0.0.0",
                server_port=port,
                share=True
            )
            
            print("✅ FastAPI server is now publicly accessible!")
            print("📝 Access your application:")
            print("   FastAPI endpoints are accessible through Gradio's API interface:")
            print("   - POST <public_url>/gradio_api/call/tts_wrapper")
            print("   - See Gradio docs at: <public_url>/gradio_api/docs")
            print("")
            print("   Web interface is available at: <public_url>/")
                
        except Exception as e:
            print(f"❌ Error exposing API: {e}")
    elif use_gradio:
        print(f"\n🎨 Launching FastAPI server with Gradio interface on port {port}...")
        try:
            # Mount Gradio app on FastAPI
            interface = create_gradio_interface()
            gr.mount_gradio_app(app, interface, path="/gradio")
            
            # Start server
            uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
                
        except Exception as e:
            print(f"❌ Error launching server: {e}")
    else:
        # Start uvicorn server normally
        print(f"🚀 FastAPI server starting on http://localhost:{port}")
        print("\n📝 API Documentation will be available at:")
        print(f"   Local: http://localhost:{port}/docs")
        
        # Start uvicorn server
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    
    return None

# Main Application
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--gradio", action="store_true", help="Launch Gradio interface")
    parser.add_argument("--expose-api", action="store_true", help="Expose FastAPI with public URL (Gradio sharing)")
    parser.add_argument("--ngrok", action="store_true", help="Expose FastAPI with ngrok")
    args = parser.parse_args()
    
    # Start the server
    start_server(args.port, use_gradio=args.gradio, expose_api=args.expose_api, use_ngrok=args.ngrok)