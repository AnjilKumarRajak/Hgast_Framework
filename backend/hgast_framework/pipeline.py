"""
pipeline.py
===========
HGASTFramework: the orchestrator. This is the ONLY class you interact with
to run "gender correction on top of any SOTA backbone."

    from hgast_framework.backbones.registry import get_backbone
    from hgast_framework.gender.llm_refine import LLMGenderRefiner
    from hgast_framework.pipeline import HGASTFramework

    backbone = get_backbone("indictrans2")          # <-- swap this one line
    refiner  = LLMGenderRefiner(chat_fn=my_chat_fn)  # <-- swap LLM here
    framework = HGASTFramework(backbone=backbone, llm_refiner=refiner)

    result = framework.translate(
        en_text="I am going home.",
        speaker_gender=1,            # 1=female, 0=male, -1=unknown
        speaker_confidence=0.92,
    )
    print(result.hindi)
    print(result.trace)             # full diagnostic trace for error analysis
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from .backbones.base import TranslationBackbone
from .gender.linguistic_analysis import parse_subject, resolve_coreference
from .gender.dual_control import build_dual_control, resolve_target_gender, detect_politeness
from .gender.morphology_rules import apply_morphology, count_morph_tokens, morph_match, _apply_rules
from .gender.llm_refine import LLMGenderRefiner

log = logging.getLogger(__name__)


@dataclass
class HGASTResult:
    english: str
    hindi_raw: str          # backbone output, before any correction
    hindi: str               # final corrected output
    target_gender: int       # -1/0/1
    target_gender_reason: str
    person: str
    subject_gender: str
    dominant_controller: str
    morph_ok: bool
    llm_applied: bool
    trace: Dict[str, Any] = field(default_factory=dict)


class HGASTFramework:
    def __init__(
        self,
        backbone: TranslationBackbone,
        llm_refiner: Optional[LLMGenderRefiner] = None,
        use_coreference: bool = True,
        use_rule_morphology: bool = True,
        use_llm_refine: bool = True,
        subj_confidence_threshold: float = 0.55,
        max_translate_attempts: int = 3,
    ):
        """
        Stage toggles (use_coreference / use_rule_morphology / use_llm_refine)
        exist specifically so you can run the ablation study from Step 6:
        disable stages one at a time and re-measure gender accuracy.
        """
        self.backbone = backbone
        self.llm_refiner = llm_refiner or LLMGenderRefiner(chat_fn=None)
        self.use_coreference = use_coreference
        self.use_rule_morphology = use_rule_morphology
        self.use_llm_refine = use_llm_refine
        self.subj_confidence_threshold = subj_confidence_threshold
        self.max_translate_attempts = max_translate_attempts

    def translate(
        self,
        en_text: str,
        speaker_gender: int = -1,
        speaker_confidence: float = 0.0,
    ) -> HGASTResult:
        trace: Dict[str, Any] = {"stages": []}

        # ---- Stage 1: linguistic analysis ----
        coref_map = resolve_coreference(en_text) if self.use_coreference else {}
        subj_info = parse_subject(en_text, coref_map)
        politeness = detect_politeness(en_text)
        trace["stages"].append({"stage": "subject_parse", "output": subj_info})
        trace["coref_map"] = coref_map
        trace["politeness"] = politeness

        # ---- Stage 2: dual control routing ----
        dual = build_dual_control(speaker_gender, speaker_confidence, subj_info, politeness)
        target_gender, reason = resolve_target_gender(dual, self.subj_confidence_threshold)
        trace["dual_control"] = dual
        trace["target_gender"] = target_gender
        trace["target_gender_reason"] = reason

        # ---- Stage 3: backbone translation (retry loop for morph guarantee) ----
        hi_raw = ""
        hi_corrected = ""
        morph_ok = False

        for attempt in range(self.max_translate_attempts):
            hi_raw = self.backbone.translate_en_to_hi(en_text)
            if not hi_raw:
                continue

            if self.use_rule_morphology and target_gender in (0, 1):
                morph_tokens = count_morph_tokens(hi_raw)
                hi_corrected = apply_morphology(
                    hi_raw, target_gender, morph_tokens=morph_tokens,
                    dominant_controller=dual["dominant_controller"],
                    person=subj_info["person"],
                )
            else:
                hi_corrected = hi_raw

            post = count_morph_tokens(hi_corrected)
            morph_ok = morph_match(post, target_gender) if target_gender in (0, 1) else True
            trace["stages"].append({
                "stage": f"translate_attempt_{attempt+1}",
                "hi_raw": hi_raw, "hi_after_rules": hi_corrected, "morph_ok": morph_ok,
            })
            if morph_ok:
                break

        if not morph_ok and hi_raw and self.use_rule_morphology:
            hi_corrected = _apply_rules(hi_raw, target_gender) if target_gender in (0, 1) else hi_raw
            post = count_morph_tokens(hi_corrected)
            morph_ok = morph_match(post, target_gender) if target_gender in (0, 1) else True
            trace["stages"].append({"stage": "hard_force_rules", "output": hi_corrected, "morph_ok": morph_ok})

        # ---- Stage 4: LLM fluency/gender refine ----
        llm_applied = False
        if self.use_llm_refine and self.llm_refiner.available and target_gender in (0, 1) and hi_raw:
            before = hi_corrected
            # Feed untouched hi_raw to avoid the aggressive rule-based corruption
            llm_output = self.llm_refiner.refine(
                en_text, hi_raw, target_gender, subj_info["person"]
            )
            # Trust the LLM output (or hi_raw fallback) over the brute-force rules
            hi_corrected = llm_output
            llm_applied = (hi_corrected != before)
            trace["stages"].append({
                "stage": "llm_refine", "before": before, "after": hi_corrected, "changed": llm_applied,
            })

        return HGASTResult(
            english=en_text,
            hindi_raw=hi_raw,
            hindi=hi_corrected,
            target_gender=target_gender,
            target_gender_reason=reason,
            person=subj_info["person"],
            subject_gender=subj_info["subject_gender"],
            dominant_controller=dual["dominant_controller"],
            morph_ok=morph_ok,
            llm_applied=llm_applied,
            trace=trace,
        )
