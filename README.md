
<img width="1200" height="600" alt="Chatterbox-Multilingual" src="https://www.resemble.ai/wp-content/uploads/2025/09/Chatterbox-Multilingual-1.png" />

# Chatterbox TTS

[![Alt Text](https://img.shields.io/badge/listen-demo_samples-blue)](https://resemble-ai.github.io/chatterbox_demopage/)
[![Alt Text](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/ResembleAI/Chatterbox)
[![Alt Text](https://static-public.podonos.com/badges/insight-on-pdns-sm-dark.svg)](https://podonos.com/resembleai/chatterbox)
[![Discord](https://img.shields.io/discord/1377773249798344776?label=join%20discord&logo=discord&style=flat)](https://discord.gg/rJq9cRJBJ6)

_Made with ♥️ by <a href="https://resemble.ai" target="_blank"><img width="100" alt="resemble-logo-horizontal" src="https://github.com/user-attachments/assets/35cf756b-3506-4943-9c72-c05ddfa4e525" /></a>

We're excited to introduce **Chatterbox Multilingual**, [Resemble AI's](https://resemble.ai) first production-grade open source TTS model supporting **23 languages** out of the box. Licensed under MIT, Chatterbox has been benchmarked against leading closed-source systems like ElevenLabs, and is consistently preferred in side-by-side evaluations.

Whether you're working on memes, videos, games, or AI agents, Chatterbox brings your content to life across languages. It's also the first open source TTS model to support **emotion exaggeration control** with robust **multilingual zero-shot voice cloning**. Try the english only version now on our [English Hugging Face Gradio app.](https://huggingface.co/spaces/ResembleAI/Chatterbox). Or try the multilingual version on our [Multilingual Hugging Face Gradio app.](https://huggingface.co/spaces/ResembleAI/Chatterbox-Multilingual-TTS).

If you like the model but need to scale or tune it for higher accuracy, check out our competitively priced TTS service (<a href="https://resemble.ai">link</a>). It delivers reliable performance with ultra-low latency of sub 200ms—ideal for production use in agents, applications, or interactive media.

# Key Details
- Multilingual, zero-shot TTS supporting 23 languages
- SoTA zeroshot English TTS
- 0.5B Llama backbone
- Unique exaggeration/intensity control
- Ultra-stable with alignment-informed inference
- Trained on 0.5M hours of cleaned data
- Watermarked outputs
- Easy voice conversion script
- [Outperforms ElevenLabs](https://podonos.com/resembleai/chatterbox)

# Supported Languages 
Arabic (ar) • Danish (da) • German (de) • Greek (el) • English (en) • Spanish (es) • Finnish (fi) • French (fr) • Hebrew (he) • Hindi (hi) • Italian (it) • Japanese (ja) • Korean (ko) • Malay (ms) • Dutch (nl) • Norwegian (no) • Polish (pl) • Portuguese (pt) • Russian (ru) • Swedish (sv) • Swahili (sw) • Turkish (tr) • Chinese (zh)
# Tips
- **General Use (TTS and Voice Agents):**
  - Ensure that the reference clip matches the specified language tag. Otherwise, language transfer outputs may inherit the accent of the reference clip’s language. To mitigate this, set `cfg_weight` to `0`.
  - The default settings (`exaggeration=0.5`, `cfg_weight=0.5`) work well for most prompts across all languages.
  - If the reference speaker has a fast speaking style, lowering `cfg_weight` to around `0.3` can improve pacing.

- **Expressive or Dramatic Speech:**
  - Try lower `cfg_weight` values (e.g. `~0.3`) and increase `exaggeration` to around `0.7` or higher.
  - Higher `exaggeration` tends to speed up speech; reducing `cfg_weight` helps compensate with slower, more deliberate pacing.


# Installation
```shell
pip install chatterbox-tts
```

Alternatively, you can install from source:
```shell
# conda create -yn chatterbox python=3.11
# conda activate chatterbox

git clone https://github.com/resemble-ai/chatterbox.git
cd chatterbox
pip install -e .
```
We developed and tested Chatterbox on Python 3.11 on Debian 11 OS; the versions of the dependencies are pinned in `pyproject.toml` to ensure consistency. You can modify the code or dependencies in this installation mode.

## Running the Services

## Standard Web UI (Gradio)
To run the standard Gradio web interface:
```bash
python multilingual_app.py
```

## FastAPI Service (WAV output)
To run the FastAPI service that returns WAV files:
```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## FastAPI Service with PCM Support (app_pcm.py)
To run the enhanced FastAPI service that supports both WAV and raw PCM output:
```bash
python -m uvicorn app_pcm:app --host 0.0.0.0 --port 8000 --reload
```

### API Endpoints

- `GET /languages` - Get all supported languages
- `POST /tts` - Generate speech from text
- `GET /health` - Health check endpoint

### TTS Parameters

When calling the `/tts` endpoint, you can use these parameters:

- `text` (required): Text to synthesize (max 300 chars)
- `language_id` (required): Language code (en, fr, de, hi, etc.)
- `exaggeration`: Speech expressiveness (0.25-2.0, default 0.5)
- `temperature`: Randomness in generation (0.05-5.0, default 0.8)
- `seed`: Random seed, 0 for random (default 0)
- `cfg_weight`: CFG/Pace weight (0.2-1.0, 0 for transfer, default 0.5)
- `raw`: Return raw PCM if True, WAV if False (default False)
- `audio_prompt`: Optional reference audio file for voice cloning

## PCM Output Support

The `app_pcm.py` service extends the standard TTS API with support for raw PCM output, which is useful for low-level audio processing applications:

- When `raw=false` or not specified: Returns standard WAV audio files
- When `raw=true`: Returns raw float32 PCM data with metadata headers

PCM output includes the following custom headers:
- `X-Sample-Rate`: Sample rate from the model (e.g., 24000)
- `X-Num-Channels`: Number of channels (1 for mono)
- `X-Sample-Format`: Sample format (float32)

### Example Usage

After starting the server, you can synthesize speech with a simple curl command:

```bash
# Standard WAV output
curl -X POST "http://localhost:8000/tts" 
     -H "accept: audio/wav" 
     -H "Content-Type: application/x-www-form-urlencoded" 
     -d "text=Hello, this is a test&language_id=en&exaggeration=0.5&temperature=0.8&seed=0&cfg_weight=0.5" 
     --output output.wav

# Raw PCM output
curl -X POST "http://localhost:8000/tts" 
     -H "Content-Type: application/x-www-form-urlencoded" 
     -d "text=Hello, this is a test&language_id=en&exaggeration=0.5&temperature=0.8&seed=0&cfg_weight=0.5&raw=true" 
     --output output.pcm
```

Or using Python requests:

```python
import requests

# Standard WAV output
response = requests.post(
    "http://localhost:8000/tts",
    data={
        "text": "Hello, this is a test",
        "language_id": "en",
        "exaggeration": 0.5,
        "temperature": 0.8,
        "seed": 0,
        "cfg_weight": 0.5
    }
)

# Save the WAV file
with open("output.wav", "wb") as f:
    f.write(response.content)

# Raw PCM output
response = requests.post(
    "http://localhost:8000/tts",
    data={
        "text": "Hello, this is a test",
        "language_id": "en",
        "exaggeration": 0.5,
        "temperature": 0.8,
        "seed": 0,
        "cfg_weight": 0.5,
        "raw": "true"
    }
)

# Save the PCM file
with open("output.pcm", "wb") as f:
    f.write(response.content)

# Check PCM headers
print("Sample Rate:", response.headers.get("X-Sample-Rate"))
print("Channels:", response.headers.get("X-Num-Channels"))
print("Format:", response.headers.get("X-Sample-Format"))
```

### Environment Variables

Create a `.env` file in the project root with the following configuration:

```bash
# Ngrok Configuration
# Get your auth token from https://dashboard.ngrok.com/get-started/your-authtoken
NGROK_AUTH_TOKEN=your_ngrok_auth_token_here
```
```

### Example Usage

After starting the server, you can synthesize speech with a simple curl command:

```bash
curl -X POST "http://xxxxxxx.ngrok.io/tts" \
     -H "accept: audio/wav" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "text=Hello, this is a test&language_id=en&exaggeration=0.5&temperature=0.8&seed=0&cfg_weight=0.5" \
     --output output.wav
```

Or using Python requests in Colab:

```python
import requests

response = requests.post(
    "http://xxxxxxx.ngrok.io/tts",
    data={
        "text": "Hello, this is a test",
        "language_id": "en",
        "exaggeration": 0.5,
        "temperature": 0.8,
        "seed": 0,
        "cfg_weight": 0.5
    }
)

# Save the audio file
with open("output.wav", "wb") as f:
    f.write(response.content)
```

### Ngrok Setup (Optional but Recommended)

For a more stable public URL, you can set up an ngrok account and use an authentication token:

1. Sign up at https://ngrok.com/
2. Get your auth token from the dashboard
3. When running the server, pass your token:

```python
# In Colab, you can pass the token like this:
!python public_app.py --ngrok-token=YOUR_NGROK_TOKEN
```

### Local Development with Public Access

For local development with public access, you can use several methods:

1. **Ngrok Method (Recommended)**:
   ```bash
   # Install dependencies
   pip install python-dotenv pyngrok
   
   # Set up your ngrok auth token in .env file
   cp .env.example .env
   # Edit .env and add your NGROK_AUTH_TOKEN
   
   # Run with ngrok
   python public_app.py --ngrok --port 8000
   ```

2. **Gradio Sharing Method**:
   ```bash
   # Run with Gradio's built-in sharing
   python public_app.py --expose-api --port 8000
   ```

3. **Local Only**:
   ```bash
   # Run locally only
   python public_app.py --port 8000
   ```

### Environment Variables

Create a `.env` file in the project root with the following configuration:

```bash
# Ngrok Configuration
# Get your auth token from https://dashboard.ngrok.com/get-started/your-authtoken
NGROK_AUTH_TOKEN=your_ngrok_auth_token_here
```

# Usage
```python
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# English example
model = ChatterboxTTS.from_pretrained(device="cuda")

text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill."
wav = model.generate(text)
ta.save("test-english.wav", wav, model.sr)

# Multilingual examples
multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device=device)

french_text = "Bonjour, comment ça va? Ceci est le modèle de synthèse vocale multilingue Chatterbox, il prend en charge 23 langues."
wav_french = multilingual_model.generate(spanish_text, language_id="fr")
ta.save("test-french.wav", wav_french, model.sr)

chinese_text = "你好，今天天气真不错，希望你有一个愉快的周末。"
wav_chinese = multilingual_model.generate(chinese_text, language_id="zh")
ta.save("test-chinese.wav", wav_chinese, model.sr)

# If you want to synthesize with a different voice, specify the audio prompt
AUDIO_PROMPT_PATH = "YOUR_FILE.wav"
wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)
ta.save("test-2.wav", wav, model.sr)
```
See `example_tts.py` and `example_vc.py` for more examples.

# Acknowledgements
- [Cosyvoice](https://github.com/FunAudioLLM/CosyVoice)
- [Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning)
- [HiFT-GAN](https://github.com/yl4579/HiFTNet)
- [Llama 3](https://github.com/meta-llama/llama3)
- [S3Tokenizer](https://github.com/xingchensong/S3Tokenizer)

# Built-in PerTh Watermarking for Responsible AI

Every audio file generated by Chatterbox includes [Resemble AI's Perth (Perceptual Threshold) Watermarker](https://github.com/resemble-ai/perth) - imperceptible neural watermarks that survive MP3 compression, audio editing, and common manipulations while maintaining nearly 100% detection accuracy.


## Watermark extraction

You can look for the watermark using the following script.

```python
import perth
import librosa

AUDIO_PATH = "YOUR_FILE.wav"

# Load the watermarked audio
watermarked_audio, sr = librosa.load(AUDIO_PATH, sr=None)

# Initialize watermarker (same as used for embedding)
watermarker = perth.PerthImplicitWatermarker()

# Extract watermark
watermark = watermarker.get_watermark(watermarked_audio, sample_rate=sr)
print(f"Extracted watermark: {watermark}")
# Output: 0.0 (no watermark) or 1.0 (watermarked)
```


# Official Discord

👋 Join us on [Discord](https://discord.gg/rJq9cRJBJ6) and let's build something awesome together!

# Citation
If you find this model useful, please consider citing.
```
@misc{chatterboxtts2025,
  author       = {{Resemble AI}},
  title        = {{Chatterbox-TTS}},
  year         = {2025},
  howpublished = {\url{https://github.com/resemble-ai/chatterbox}},
  note         = {GitHub repository}
}
```
# Disclaimer
Don't use this model to do bad things. Prompts are sourced from freely available data on the internet.
