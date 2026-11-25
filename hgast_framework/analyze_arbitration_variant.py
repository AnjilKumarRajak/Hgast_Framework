"""
Analysis: fallback-to-speaker-gender vs. abstain-on-ambiguous-subject.

Reprocesses an existing evaluation log (a JSON list of per-instance records, each
carrying at minimum: target_gender, framework_output, baseline_output, morph_reason,
subject, speaker_gender) to compare the current default arbitration behavior against
the abstain-on-ambiguous-subject variant described in gender/dual_control.py, WITHOUT
re-running the translation backbone, ASR, or parser -- this only touches the already-
logged arbitration decisions and re-scores morphological correctness, so it needs no
GPU and runs in seconds even on the full evaluation set.

Usage:
    python -m hgast_framework.analyze_arbitration_variant --data path/to/predictions.json

Expected per-record schema (matches this repo's run_experiment.py / pipeline.py
output when logging is enabled): target_gender (int, -1/0/1), speaker_gender (int),
framework_output (str), baseline_output (str), morph_reason (str, one of
"speaker_primary" | "speaker_fallback_politeness" | "subject_dominant" |
"speaker_fallback_subject_uncertain"), subject (str, possibly empty).
"""

import argparse
import json
import logging
from typing import List, Dict, Any

from .evaluation.gender_metrics import is_gender_correct
from .evaluation.controller_distribution import (
    compute_controller_distribution, format_controller_distribution_table,
)

log = logging.getLogger(__name__)

FALLBACK_REASON = "speaker_fallback_subject_uncertain"

REASON_TO_KAPPA = {
    "speaker_primary": "speaker",
    "speaker_fallback_politeness": "self-address",
    "subject_dominant": "subject",
    FALLBACK_REASON: "fallback",
}

# Heuristic for isolating the subset of fallback cases that plausibly match the
# targeted failure mode (a *singular, animate, gendered* third-person referent --
# e.g. "my mother is reading" -- as opposed to plural/collective subjects ("they",
# "people", "all") or bare proper nouns, where the notion of a single "correct"
# grammatical gender to route to is murkier and where an --data audit is needed before
# trusting this list as exhaustive).
GENDERED_KINSHIP_OR_PRONOUN = {
    "mother", "sister", "wife", "daughter", "girl", "woman", "aunt", "grandmother",
    "actress", "she", "her", "hers", "queen", "princess", "lady", "girlfriend",
    "mom", "mum", "nurse", "niece",
    "father", "brother", "husband", "son", "boy", "man", "uncle", "grandfather",
    "actor", "he", "him", "his", "king", "prince", "gentleman", "boyfriend", "dad",
    "nephew",
}


def score_variant(target_genders: List[int], outputs: List[str], n_total: int):
    resolved = [(t, o) for t, o in zip(target_genders, outputs) if t in (0, 1)]
    n_resolved = len(resolved)
    coverage = (n_resolved / n_total * 100) if n_total else 0.0
    if n_resolved == 0:
        return dict(coverage=coverage, macro=float("nan"), eff_acc=0.0,
                     n_resolved=0, male_acc=float("nan"), female_acc=float("nan"))

    male_correct = male_total = female_correct = female_total = 0
    for t, o in resolved:
        correct = is_gender_correct(o, t)
        if t == 0:
            male_total += 1
            male_correct += int(correct)
        else:
            female_total += 1
            female_correct += int(correct)
    male_acc = (male_correct / male_total * 100) if male_total else float("nan")
    female_acc = (female_correct / female_total * 100) if female_total else float("nan")
    valid = [a for a in (male_acc, female_acc) if a == a]
    macro = sum(valid) / len(valid) if valid else float("nan")
    eff_acc = (coverage / 100.0) * macro
    return dict(coverage=coverage, macro=macro, eff_acc=eff_acc, n_resolved=n_resolved,
                male_acc=male_acc, female_acc=female_acc)


def run_analysis(records: List[Dict[str, Any]]) -> str:
    n_total = len(records)
    out = [f"Loaded {n_total} records."]

    default_targets = [r["target_gender"] for r in records]
    default_outputs = [r["framework_output"] for r in records]
    default_kappa = [REASON_TO_KAPPA.get(r.get("morph_reason"), "unknown") for r in records]
    d = score_variant(default_targets, default_outputs, n_total)

    abstain_targets, abstain_outputs, abstain_kappa = [], [], []
    n_switched = 0
    for r in records:
        if r.get("morph_reason") == FALLBACK_REASON:
            abstain_targets.append(-1)
            abstain_outputs.append(r["baseline_output"])
            abstain_kappa.append("abstain")
            n_switched += 1
        else:
            abstain_targets.append(r["target_gender"])
            abstain_outputs.append(r["framework_output"])
            abstain_kappa.append(REASON_TO_KAPPA.get(r.get("morph_reason"), "unknown"))
    a = score_variant(abstain_targets, abstain_outputs, n_total)

    out.append(f"\n=== Fallback branch usage ===")
    out.append(f"Records routed via speaker-gender fallback (person=third, subject "
               f"unresolved): {n_switched}/{n_total} ({n_switched/n_total*100:.1f}%)")

    fallback_records = [r for r in records if r.get("morph_reason") == FALLBACK_REASON]
    person_like = [r for r in fallback_records
                   if r.get("subject", "").strip().lower() in GENDERED_KINSHIP_OR_PRONOUN]
    out.append(
        f"Of those, subject is an explicitly gendered kinship/pronoun term "
        f"(the targeted failure-mode profile): {len(person_like)} "
        f"({len(person_like)/n_total*100:.2f}% of full dataset). "
        f"NOTE: this is a heuristic lexical filter (see GENDERED_KINSHIP_OR_PRONOUN) "
        f"-- audit the printed samples below before treating this count as exact."
    )
    for r in person_like[:20]:
        out.append(f"    en={r['en_original'][:60]!r}  subject={r['subject']!r}  "
                   f"speaker_g={r['speaker_gender']}  target_g={r['target_gender']}")

    out.append(f"\n=== Coverage / Effective Accuracy (paper Appendix B.15 formula) ===")
    out.append("| Variant | Coverage | Male Acc | Female Acc | Macro Acc (resolved) | Effective Accuracy |")
    out.append("|---|---|---|---|---|---|")
    out.append(f"| Fallback (current default) | {d['coverage']:.2f}% | {d['male_acc']:.2f}% | "
               f"{d['female_acc']:.2f}% | {d['macro']:.2f}% | {d['eff_acc']:.2f}% |")
    out.append(f"| Abstain-on-ambiguous-subject | {a['coverage']:.2f}% | {a['male_acc']:.2f}% | "
               f"{a['female_acc']:.2f}% | {a['macro']:.2f}% | {a['eff_acc']:.2f}% |")

    out.append("")
    out.append(format_controller_distribution_table(
        compute_controller_distribution(default_kappa), label="default (fallback)"))
    out.append("")
    out.append(format_controller_distribution_table(
        compute_controller_distribution(abstain_kappa), label="abstain-on-ambiguous-subject"))

    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to the evaluation-log JSON file.")
    args = parser.parse_args()

    with open(args.data, encoding="utf-8") as f:
        records = json.load(f)

    print(run_analysis(records))


if __name__ == "__main__":
    main()
