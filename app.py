import os
import sys
import tempfile
import asyncio
import gradio as gr
import edge_tts

# Append backend directory to sys.path
BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from hgast_framework.hf_inference_adapter import HFInferenceManager, HFInferenceBackbone
from hgast_framework.gender.llm_refine import LLMGenderRefiner
from hgast_framework.pipeline import HGASTFramework

# Initialize global HF Inference API Manager
hf_manager = HFInferenceManager()
framework = HGASTFramework(
    backbone=HFInferenceBackbone(hf_manager),
    llm_refiner=LLMGenderRefiner(chat_fn=hf_manager.llm_refine_chat_fn)
)

custom_css = """
@import url("https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap");

body {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
    font-family: "Outfit", sans-serif !important;
    color: #e2e8f0;
}

.gradio-container {
    background: rgba(15, 23, 42, 0.5) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.6);
    border-radius: 24px;
    padding: 32px !important;
    max-width: 1200px !important;
    margin-top: 20px !important;
}

.title-text {
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    background: linear-gradient(to right, #818cf8, #c084fc, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 4px !important;
}

.subtitle-text {
    text-align: center;
    font-size: 1.1rem;
    color: #94a3b8;
    margin-bottom: 24px !important;
    font-weight: 300;
}

.badge-tag {
    display: inline-block;
    padding: 4px 14px;
    background: rgba(168, 85, 247, 0.15);
    border: 1px solid rgba(168, 85, 247, 0.3);
    border-radius: 20px;
    color: #c084fc;
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 12px;
}

button.primary {
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%) !important;
    border: none !important;
    box-shadow: 0 4px 18px rgba(168, 85, 247, 0.4) !important;
    transition: all 0.3s ease !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
}

button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(168, 85, 247, 0.6) !important;
}
"""

async def process_translation(audio_filepath, text_input, hf_token):
    if hf_token and hf_token.strip():
        hf_manager.set_api_key(hf_token.strip())

    if not audio_filepath and not text_input:
        return None, "Error: Please upload an audio file or enter English text.", None, None, {"error": "No input provided"}

    try:
        eng_text = ""
        speaker_gender_str = "female"
        speaker_conf = 0.8

        if audio_filepath:
            eng_text = hf_manager.transcribe_audio(audio_filepath)
            speaker_gender_str, speaker_conf = hf_manager.detect_speaker_gender(audio_filepath)
        else:
            eng_text = text_input.strip()

        if not eng_text:
            return None, "Error: Failed to extract text from audio.", None, None, {}

        speaker_gender_int = 1 if speaker_gender_str == "female" else 0

        res = framework.translate(
            en_text=eng_text,
            speaker_gender=speaker_gender_int,
            speaker_confidence=speaker_conf
        )

        voice = "hi-IN-SwaraNeural" if speaker_gender_str == "female" else "hi-IN-MadhurNeural"
        tts_path = tempfile.mktemp(suffix=".mp3")
        communicate = edge_tts.Communicate(res.hindi, voice)
        await communicate.save(tts_path)

        formatted_trace = {
            "Audio Analysis": {
                "ASR Transcript": eng_text,
                "Acoustic Gender": speaker_gender_str.upper(),
                "Acoustic Confidence": f"{speaker_conf:.2f}"
            },
            "Linguistic Dual Controller": {
                "Grammatical Person": res.person.capitalize(),
                "Subject Gender": res.subject_gender.capitalize(),
                "Target Gender": "Female" if res.target_gender == 1 else "Male" if res.target_gender == 0 else "Unknown",
                "Dominant Controller": res.dominant_controller.capitalize(),
                "Routing Reason": res.target_gender_reason
            },
            "HGAST++ Stages": {
                "Morphology Rules Match": res.morph_ok,
                "LLM Refine Applied": res.llm_applied,
                "Full Diagnostics": res.trace
            }
        }

        return tts_path, eng_text, res.hindi_raw, res.hindi, formatted_trace

    except Exception as e:
        import traceback
        return None, f"Exception: {str(e)}", None, None, {"traceback": traceback.format_exc()}

theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="violet",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Outfit"), "ui-sans-serif", "sans-serif"]
)

with gr.Blocks(title="HGAST++ Space (HF Inference API)") as demo:
    
    gr.HTML("""
        <div style="text-align: center;">
            <span class="badge-tag">Option 3: Hugging Face Serverless Inference API</span>
            <h1 class="title-text">HGAST++ Framework</h1>
            <p class="subtitle-text">Hybrid Gender-Aligned Speech Translation on Free Hugging Face Spaces</p>
        </div>
    """)

    with gr.Tabs():
        with gr.TabItem("🎙️ Interactive Demo"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🎤 Input Speech / Text")
                    audio_in = gr.Audio(type="filepath", label="Upload or Record English Audio")
                    text_in = gr.Textbox(label="Or Enter English Text Directly", placeholder="e.g. I am exhausted, hungry and I just want to sleep.")
                    
                    with gr.Accordion("🔑 HF API Token (Optional)", open=False):
                        hf_token_in = gr.Textbox(
                            label="Hugging Face User Access Token",
                            placeholder="hf_xxxxxxxxxxxxxxxxxxxx (Leaves blank to use public serverless rate limit)",
                            type="password"
                        )

                    translate_btn = gr.Button("Translate & Align Gender", variant="primary", size="lg")

                    gr.Markdown("---")
                    gr.Markdown(
                        "**Serverless Infrastructure (HF Inference API):**\n"
                        "- 🎤 `openai/whisper-small` (ASR)\n"
                        "- 👤 `wav2vec2-large-xlsr` (Gender Classification)\n"
                        "- 🌐 `seamless-m4t-v2-large` (Translation)\n"
                        "- 🧠 `Qwen2.5-7B-Instruct` (LLM Refiner)\n"
                        "- 🔊 `Edge-TTS` (Hindi Synthesis)"
                    )

                with gr.Column(scale=2):
                    gr.Markdown("### ✨ Translation & Alignment Results")
                    eng_out = gr.Textbox(label="English Transcript (ASR)", lines=2, interactive=False)

                    with gr.Row():
                        base_out = gr.Textbox(label="Baseline Translation (Uncorrected)", lines=3, interactive=False)
                        final_out = gr.Textbox(label="HGAST++ Final Output (Gender-Aligned)", lines=3, interactive=False)

                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("#### 🔊 Generated Audio Output")
                            audio_out = gr.Audio(label="Hindi Neural TTS", interactive=False)
                        with gr.Column(scale=1):
                            gr.Markdown("#### 🧠 Framework Internal Routing")
                            trace_out = gr.JSON(label="Diagnostic Trace")

            translate_btn.click(
                fn=process_translation,
                inputs=[audio_in, text_in, hf_token_in],
                outputs=[audio_out, eng_out, base_out, final_out, trace_out]
            )

        with gr.TabItem("🌐 Static Space Frontend & Payload API"):
            gr.Markdown("### Option 3 Architecture: Static Space Frontend calling HF Inference API")
            gr.Markdown(
                "If your model weights are uploaded to a Hugging Face repository, you can construct a static HTML/JS frontend "
                "or send POST requests directly to HF Inference API endpoints."
            )
            gr.Markdown("#### cURL Example:")
            gr.Code(
                value="""curl -X POST https://api-inference.huggingface.co/models/Qwen/Qwen2.5-7B-Instruct \\
  -H "Authorization: Bearer hf_xxx" \\
  -H "Content-Type: application/json" \\
  -d '{"inputs": "Translate to Hindi: I am going home."}'""",
                language="shell"
            )
            gr.Markdown("#### Static Space HTML Template:")
            gr.Code(
                value="""<!-- Static Space index.html snippet -->
<script>
async function translate() {
    const resp = await fetch('https://api-inference.huggingface.co/models/facebook/seamless-m4t-v2-large', {
        method: 'POST',
        headers: { 'Authorization': 'Bearer ' + HF_TOKEN },
        body: JSON.stringify({ inputs: "I am going home." })
    });
    const result = await resp.json();
    console.log(result);
}
</script>""",
                language="html"
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=theme, css=custom_css)
