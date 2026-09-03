"""End-to-end execution and evaluation."""

import os
import sys
from pathlib import Path
import logging

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

from hgast_framework.backbones.registry import get_backbone
from hgast_framework.gender.llm_refine import LLMGenderRefiner
from hgast_framework.gender.qwen_adapter import qwen_chat_fn
from hgast_framework.pipeline import HGASTFramework
from hgast_framework.evaluation.metrics import compute_bleu, compute_chrf, compute_wer, compute_ter
from hgast_framework.evaluation.gender_metrics import compute_gender_accuracy
from hgast_framework.evaluation.significance import bootstrap_compare, mcnemar_test
from hgast_framework.evaluation.error_analysis import classify_failure, summarize_errors
from hgast_framework.evaluation.ablation import run_ablation, format_ablation_table


# 1. Select backbone
# Options: "indicconformer_indictrans2", "seamless_m4t", "whisper_nllb", "indictrans2", "nllb"
BACKBONE_KEY = "indicconformer_indictrans2"

# 2. Select LLM refiner
USE_LLM_REFINE = True
llm_refiner = LLMGenderRefiner(chat_fn=qwen_chat_fn if USE_LLM_REFINE else None)


def build_test_set():
    return [
        {"en": "I am going home.", "hi_ref": "मैं घर जा रहा हूँ।",
         "target_gender": 0, "speaker_gender": 0, "speaker_conf": 0.9},
        {"en": "I am going home.", "hi_ref": "मैं घर जा रही हूँ।",
         "target_gender": 1, "speaker_gender": 1, "speaker_conf": 0.9},
        {"en": "She was tired after the long journey.", "hi_ref": "वह लंबी यात्रा के बाद थकी हुई थी।",
         "target_gender": 1, "speaker_gender": -1, "speaker_conf": 0.0},
        {"en": "He wants to become a doctor.", "hi_ref": "वह डॉक्टर बनना चाहता है।",
         "target_gender": 0, "speaker_gender": -1, "speaker_conf": 0.0},
    ]


def main():
    test_set = build_test_set()
    backbone = get_backbone(BACKBONE_KEY)
    framework = HGASTFramework(backbone=backbone, llm_refiner=llm_refiner)

    hi_outputs, hi_raw_outputs, results = [], [], []
    for ex in test_set:
        result = framework.translate(
            ex["en"], speaker_gender=ex["speaker_gender"], speaker_confidence=ex["speaker_conf"]
        )
        hi_outputs.append(result.hindi)
        hi_raw_outputs.append(result.hindi_raw)
        results.append(result)

    refs = [ex["hi_ref"] for ex in test_set]
    target_genders = [ex["target_gender"] for ex in test_set]

    # ---- Translation quality metrics ----
    print(f"\n=== Translation quality ({BACKBONE_KEY}) ===")
    print(f"BLEU:  {compute_bleu(hi_outputs, refs)}")
    print(f"chrF++: {compute_chrf(hi_outputs, refs)}")
    print(f"WER:   {compute_wer(hi_outputs, refs)}")
    print(f"TER:   {compute_ter(hi_outputs, refs)}")

    # ---- Gender accuracy ----
    gender_result = compute_gender_accuracy(hi_outputs, target_genders)
    print(f"\n=== Gender accuracy ===")
    print(f"Male:   {gender_result.male_acc:.2f}% (n={gender_result.n_male})")
    print(f"Female: {gender_result.female_acc:.2f}% (n={gender_result.n_female})")
    print(f"Macro:  {gender_result.macro_avg:.2f}%")

    # ---- Baseline vs framework comparison ----
    baseline_gender = compute_gender_accuracy(hi_raw_outputs, target_genders)
    mcnemar_result = mcnemar_test(
        baseline_gender.per_example_correct, gender_result.per_example_correct
    )
    print(f"\n=== McNemar test: baseline vs framework gender accuracy ===")
    print(mcnemar_result)

    bleu_sig = bootstrap_compare(hi_raw_outputs, hi_outputs, refs, compute_bleu, n_samples=200)
    print(f"\n=== Bootstrap significance: BLEU baseline vs framework ===")
    print(bleu_sig)

    # ---- Error analysis ----
    failures = [
        classify_failure(r, ex["target_gender"])
        for r, ex, correct in zip(results, test_set, gender_result.per_example_correct)
        if not correct
    ]
    print(f"\n=== Error analysis ===")
    print(summarize_errors(failures))

    # ---- Ablation study ----
    print(f"\n=== Ablation study ===")
    ablation_rows = run_ablation(
        backbone=backbone,
        llm_refiner=llm_refiner,
        en_texts=[ex["en"] for ex in test_set],
        references_hi=refs,
        target_genders=target_genders,
        speaker_genders=[ex["speaker_gender"] for ex in test_set],
        speaker_confidences=[ex["speaker_conf"] for ex in test_set],
    )
    print(format_ablation_table(ablation_rows))


if __name__ == "__main__":
    main()
