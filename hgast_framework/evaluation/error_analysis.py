

from collections import Counter
from dataclasses import dataclass
from typing import List
from ..pipeline import HGASTResult
from ..gender.morphology_rules import count_morph_tokens


CATEGORIES = [
    "coreference_error",
    "subject_parsing_error",
    "name_gender_inference_error",
    "rule_coverage_gap",
    "llm_rewrite_reverted_or_failed",
    "ambiguous_or_mixed_gender_subject",
    "confidence_threshold_miss",
    "unclassified",
]


@dataclass
class ErrorCase:
    english: str
    hindi_output: str
    target_gender: int
    category: str
    detail: str


def classify_failure(result: HGASTResult, target_gender_label: int) -> ErrorCase:
    trace = result.trace
    subj_stage = next((s for s in trace["stages"] if s["stage"] == "subject_parse"), None)
    subj_info = subj_stage["output"] if subj_stage else {}

    # 1. Confidence threshold miss: pipeline had low confidence and fell back
    if "uncertain" in result.target_gender_reason or "fallback" in result.target_gender_reason:
        if result.target_gender != target_gender_label:
            return ErrorCase(
                result.english, result.hindi, target_gender_label,
                "confidence_threshold_miss",
                f"reason={result.target_gender_reason}, subj_confidence={subj_info.get('confidence')}",
            )

    # 2. Subject parsing error: parsed subject_gender contradicts ground truth

    parsed_g = subj_info.get("subject_gender")
    if result.person == "third" and parsed_g in ("male", "female"):
        expected_label = "female" if target_gender_label == 1 else "male"
        if parsed_g != expected_label:
 
            if trace.get("coref_map"):
                return ErrorCase(result.english, result.hindi, target_gender_label,
                                  "coreference_error", f"parsed={parsed_g}, coref_map={trace['coref_map']}")
            return ErrorCase(result.english, result.hindi, target_gender_label,
                              "subject_parsing_error", f"parsed_subject_gender={parsed_g}")

    # 3. Name-gender inference error: subject is a PROPN and gender unknown
    if subj_info.get("subject_pos") == "PROPN" and parsed_g == "unknown":
        return ErrorCase(result.english, result.hindi, target_gender_label,
                          "name_gender_inference_error", f"subject={subj_info.get('subject')}")

    # 4. LLM rewrite reverted/failed: an llm_refine stage exists but changed==False
    llm_stage = next((s for s in trace["stages"] if s["stage"] == "llm_refine"), None)
    if llm_stage and not llm_stage.get("changed") and not result.morph_ok:
        return ErrorCase(result.english, result.hindi, target_gender_label,
                          "llm_rewrite_reverted_or_failed", "LLM call made no change; safety gate reverted it")

    # 5. Rule coverage gap: morphology rules ran (morph_ok False) and no LLM
    if not result.morph_ok and not llm_stage:
        info = count_morph_tokens(result.hindi)
        return ErrorCase(result.english, result.hindi, target_gender_label,
                          "rule_coverage_gap", f"dominant_after_rules={info['dominant']}")

    # 6. Ambiguous/mixed-gender: multiple PERSON-like antecedents in coref map
    if trace.get("coref_map") and len(set(trace["coref_map"].values())) > 1:
        return ErrorCase(result.english, result.hindi, target_gender_label,
                          "ambiguous_or_mixed_gender_subject", str(trace["coref_map"]))

    return ErrorCase(result.english, result.hindi, target_gender_label, "unclassified", "no rule matched")


def summarize_errors(error_cases: List[ErrorCase]) -> dict:
    counts = Counter(e.category for e in error_cases)
    total = len(error_cases) or 1
    return {
        "total_failures": len(error_cases),
        "breakdown_pct": {cat: round(100 * n / total, 1) for cat, n in counts.items()},
        "breakdown_count": dict(counts),
        "examples_per_category": {
            cat: [e for e in error_cases if e.category == cat][:3] for cat in counts
        },
    }
