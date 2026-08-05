import os
import sys
import tempfile
import asyncio
import gradio as gr
import edge_tts

BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from hgast_framework.hf_inference_adapter import HFInferenceManager, HFInferenceBackbone
from hgast_framework.gender.llm_refine import LLMGenderRefiner
from hgast_framework.pipeline import HGASTFramework

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
    max-width: 1100px !important;
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

async def process_translation(audio_filepath):
    if not audio_filepath:
        return "Error: Please speak into the microphone or upload an audio file.", "--", "--"

    try:
        # 1. Real Speech-to-Text ASR using Whisper
        eng_text = hf_manager.transcribe_audio(audio_filepath)
        if not eng_text:
            return "Error: Could not transcribe audio.", "--", "--"

        # 2. Real Speaker Gender Recognition using Wav2Vec2
        speaker_gender_str, speaker_conf = hf_manager.detect_speaker_gender(audio_filepath)
        speaker_gender_int = 1 if speaker_gender_str == "female" else 0

        # 3. Dynamic Translation via HGAST Framework & SeamlessM4T Backbone
        res = framework.translate(
            en_text=eng_text,
            speaker_gender=speaker_gender_int,
            speaker_confidence=speaker_conf
        )

        return eng_text, res.hindi_raw, res.hindi

    except Exception as e:
        import traceback
        return f"Error: {str(e)}", "--", "--"

theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="violet",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Outfit"), "ui-sans-serif", "sans-serif"]
)

with gr.Blocks(title="HGAST Framework") as demo:
    
    gr.HTML("""
        <div style="text-align: center;">
            <span class="badge-tag">Hierarchical Gender Arbitration for Speech Translation</span>
            <h1 class="title-text">HGAST Framework</h1>
            <p class="subtitle-text">Acoustic & Linguistic Gender-Faithful Speech Translation</p>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎙️ Speech Audio Input")
            audio_in = gr.Audio(type="filepath", label="Record Speech or Upload Audio File")
            translate_btn = gr.Button("Translate Voice & Align Gender", variant="primary", size="lg")

            gr.Markdown("---")
            gr.Markdown(
                "**HGAST Dynamic Pipeline:**\n"
                "- 🎤 `openai/whisper-small` (Speech-to-Text ASR)\n"
                "- 👤 `wav2vec2-large-xlsr` (Acoustic Speaker Gender)\n"
                "- 🌐 `SeamlessM4T-v2-Large` (Baseline Translation)\n"
                "- 🧠 `HGAST Dual Controller` (Gender-Faithful Correction)"
            )

        with gr.Column(scale=2):
            gr.Markdown("### ✨ Translation Outputs")
            eng_out = gr.Textbox(label="English Transcript (ASR)", lines=2, interactive=False)
            base_out = gr.Textbox(label="Baseline Output (SeamlessM4T-v2-Large)", lines=3, interactive=False)
            final_out = gr.Textbox(label="HGAST Output (Gender-Faithful)", lines=3, interactive=False)

    translate_btn.click(
        fn=process_translation,
        inputs=[audio_in],
        outputs=[eng_out, base_out, final_out]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=theme, css=custom_css)
