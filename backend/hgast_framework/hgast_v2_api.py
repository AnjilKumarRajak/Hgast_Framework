import os
import shutil
import tempfile
import base64
import torch
import librosa
from transformers import pipeline
import edge_tts
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

# Import the NEW HGAST++ framework
from hgast_framework.backbones.registry import get_backbone
from hgast_framework.gender.llm_refine import LLMGenderRefiner
from hgast_framework.gender.qwen_adapter import qwen_chat_fn
from hgast_framework.pipeline import HGASTFramework

WHISPER_MODEL = "openai/whisper-small"
WAV2VEC2_GENDER_MODEL = "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"

models = {
    "whisper_pipe": None,
    "gender_pipe": None,
    "framework": None
}

def load_models():
    print("Loading models into memory (this will take a minute)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Whisper ASR
    if models["whisper_pipe"] is None:
        models["whisper_pipe"] = pipeline(
            "automatic-speech-recognition", 
            model=WHISPER_MODEL, 
            device=device,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        
    # 2. Gender Detection
    if models["gender_pipe"] is None:
        models["gender_pipe"] = pipeline("audio-classification", model=WAV2VEC2_GENDER_MODEL, device=device)
        
    # 3. New HGAST++ Framework (which loads SeamlessM4T + LLM Refiner)
    if models["framework"] is None:
        backbone = get_backbone("seamless_m4t")
        llm_refiner = LLMGenderRefiner(chat_fn=qwen_chat_fn)
        models["framework"] = HGASTFramework(backbone=backbone, llm_refiner=llm_refiner)
        
    print("All models loaded successfully!")

# -----------------------------------------------------------------------------
# Main Pipeline
# -----------------------------------------------------------------------------
async def process_audio(audio_path):
    if audio_path is None:
        return [None]*9
        
    try:
        load_models()
        
        # 1. Transcribe Audio -> English (Whisper)
        eng_text = models["whisper_pipe"](audio_path)["text"].strip()
        
        # 2. Detect Speaker Gender (wav2vec2)
        gender_preds = models["gender_pipe"](audio_path)
        speaker_gender_str = gender_preds[0]['label'] # 'female' or 'male'
        speaker_gender_int = 1 if speaker_gender_str == "female" else 0
        speaker_conf = gender_preds[0]['score']
        
        # 3. Translate using NEW HGAST++ Framework
        framework = models["framework"]
        result = framework.translate(eng_text, speaker_gender=speaker_gender_int, speaker_confidence=speaker_conf)
        
        raw_hindi = result.hindi_raw
        final_hindi = result.hindi
        trace = result.trace
        trace["person"] = result.person
        trace["subject_gender"] = result.subject_gender
        trace["dominant_controller"] = result.dominant_controller
        
        # 4. Generate TTS Audio matching the Speaker's Gender
        voice = "hi-IN-SwaraNeural" if speaker_gender_str == "female" else "hi-IN-MadhurNeural"
        tts_output_path = tempfile.mktemp(suffix=".mp3")
        communicate = edge_tts.Communicate(final_hindi, voice)
        await communicate.save(tts_output_path)
        
        return raw_hindi, eng_text, speaker_gender_str, trace, final_hindi, tts_output_path
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return [f"Error: {str(e)}"] * 6

# -----------------------------------------------------------------------------
# FastAPI Backend
# -----------------------------------------------------------------------------
app = FastAPI(title="HGAST++ V2 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/translate")
async def translate_endpoint(audio: UploadFile = File(...)):
    # Save uploaded file to temp file
    suffix = os.path.splitext(audio.filename)[1] if audio.filename else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(audio.file, tmp)
        tmp_path = tmp.name

    try:
        # Run translation pipeline
        results = await process_audio(tmp_path)
        
        # Clean up input audio
        os.unlink(tmp_path)
        
        if len(results) == 6 and results[0] and results[0].startswith("Error:"):
            return JSONResponse({"error": results[0]}, status_code=500)
            
        raw_hindi, eng_text, speaker_gender_str, trace, final_hindi, tts_output_path = results
        
        # Encode audio to base64
        audio_base64 = None
        if tts_output_path and os.path.exists(tts_output_path):
            with open(tts_output_path, "rb") as f:
                audio_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        return {
            "baseline_translation": raw_hindi,
            "corrected_translation": final_hindi,
            "audio_base64": audio_base64,
            "intermediate": {
                "english_asr": eng_text,
                "speaker_gender": speaker_gender_str,
                "grammatical_person": trace.get("person", "unknown"),
                "subject_gender": trace.get("subject_gender", "N/A"),
                "target_gender": "female" if trace.get("target_gender") == 1 else "male",
                "dominant_controller": trace.get("dominant_controller", "unknown"),
                "llm_triggered": trace.get("triggered", False),
                "llm_overlap_pass": trace.get("overlap_pass", True)
            }
        }
    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run("hgast_v2_api:app", host="0.0.0.0", port=7861, reload=False)
