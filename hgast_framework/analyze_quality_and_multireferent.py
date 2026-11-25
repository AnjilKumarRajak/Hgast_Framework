"""
Two analyses computed directly on already-logged evaluation data, for all
three languages (Hindi, Marathi, Gujarati) -- no backbone/ASR/GPU rerun required:

1. Tier-1-only vs. full-system translation quality (BLEU/chrF++),
   using the already-logged pred_tier1 / pred_tier2 fields, isolating Stage 4's
   (the LLM refiner's) specific contribution to translation quality rather than
   just gender accuracy (Table 7 in the paper covers only the latter).

   NOTE (known caveat, not yet resolved): this script skips instances where
   pred_tier1/pred_tier2 are empty (e.g. Hindi: 4,315/4,865 have both fields
   populated). The resulting corpus-level deltas do not necessarily match the
   paper's Table 12 (which scores the full corpus, baseline vs. full HGAST) --
   do not treat this script's Baseline/Full numbers as a direct substitute for
   Table 12 without first checking why the excluded instances have empty tier
   fields and whether including them changes the sign of the delta.

2. Audit of how many sentences contain multiple coordinated
   clauses with different grammatical persons/subjects (i.e., would need more
   than one target gender -- the multi-clause limitation discussed in the paper), using the
   real subject parser (fast-path lexicon rules only, no coreference chaining --
   consistent with how most already-logged Hindi records were processed, per
   their empty coref_map fields). The clause splitter is a naive coordinating-
   conjunction split ("and"/"but"/";"), so results are an UPPER BOUND / heuristic
   estimate, not an exact count -- inspect the printed examples before citing an
   exact percentage.

Usage:
    python -m hgast_framework.analyze_quality_and_multireferent --data-dir path/to/dataset

Expects, in --data-dir: predictions.json (Hindi), hgast_marathi_predictions.json,
hgast_gujarati_predictions.json, with the field names in LANGS below.
"""
import argparse
import json
import os

from .evaluation.metrics import compute_bleu, compute_chrf
from .gender.linguistic_analysis import parse_subject

LANGS = {
    "Hindi": dict(path="predictions.json", en_field="en_original", ref_field="hi",
                  t1_field="pred_tier1", t2_field="pred_tier2", base_field="baseline_output"),
    "Marathi": dict(path="hgast_marathi_predictions.json", en_field="english", ref_field="reference",
                     t1_field="pred_tier1", t2_field="pred_tier2", base_field="baseline_output"),
    "Gujarati": dict(path="hgast_gujarati_predictions.json", en_field="english", ref_field="reference",
                      t1_field="pred_tier1", t2_field="pred_tier2", base_field="baseline_output"),
}

CLAUSE_SPLIT = [" and ", " but ", "; "]


def split_clauses(en_text: str):
    parts = [en_text]
    for sep in CLAUSE_SPLIT:
        new_parts = []
        for p in parts:
            new_parts.extend(p.split(sep))
        parts = new_parts
    return [p.strip() for p in parts if p.strip()]


def run_quality_analysis(data_dir: str) -> str:
    out = ["=" * 70, "PART 1 -- Tier-1-only vs full-system translation quality",
           "=" * 70]
    for lang, cfg in LANGS.items():
        with open(os.path.join(data_dir, cfg["path"]), encoding="utf-8") as f:
            records = json.load(f)

        refs, base_out, t1_out, t2_out = [], [], [], []
        n_skipped = 0
        for r in records:
            t1 = str(r.get(cfg["t1_field"], "") or "").strip()
            t2 = str(r.get(cfg["t2_field"], "") or "").strip()
            if not t1 or not t2:
                n_skipped += 1
                continue
            refs.append(r[cfg["ref_field"]])
            base_out.append(r[cfg["base_field"]])
            t1_out.append(t1)
            t2_out.append(t2)

        n = len(refs)
        bleu_base, chrf_base = compute_bleu(base_out, refs), compute_chrf(base_out, refs)
        bleu_t1, chrf_t1 = compute_bleu(t1_out, refs), compute_chrf(t1_out, refs)
        bleu_t2, chrf_t2 = compute_bleu(t2_out, refs), compute_chrf(t2_out, refs)

        out.append(f"\n--- {lang} (n={n}, skipped {n_skipped} with empty tier fields) ---")
        out.append(f"{'Stage':<20}{'BLEU':>10}{'chrF++':>10}")
        out.append(f"{'Baseline':<20}{bleu_base:>10.2f}{chrf_base:>10.2f}")
        out.append(f"{'Tier-1-only':<20}{bleu_t1:>10.2f}{chrf_t1:>10.2f}")
        out.append(f"{'Full (Tier-1+2)':<20}{bleu_t2:>10.2f}{chrf_t2:>10.2f}")
        out.append(f"Delta attributable to Stage 4 (LLM refiner): "
                   f"BLEU {bleu_t2 - bleu_t1:+.2f}, chrF++ {chrf_t2 - chrf_t1:+.2f}")
    return "\n".join(out)


def run_multireferent_audit(data_dir: str) -> str:
    out = ["=" * 70, "PART 2 -- multi-referent / multi-clause sentence audit",
           "=" * 70]
    for lang, cfg in LANGS.items():
        with open(os.path.join(data_dir, cfg["path"]), encoding="utf-8") as f:
            records = json.load(f)

        n_total = len(records)
        n_multi_clause = 0
        n_multi_referent_risk = 0
        examples = []

        for r in records:
            en = r.get(cfg["en_field"], "") or ""
            clauses = split_clauses(en)
            if len(clauses) < 2:
                continue
            n_multi_clause += 1

            parses = [parse_subject(c, {}) for c in clauses]
            persons = {p["person"] for p in parses}
            genders = {p["subject_gender"] for p in parses if p["subject_gender"] in ("male", "female")}

            at_risk = len(persons) >= 2 or len(genders) >= 2
            if at_risk:
                n_multi_referent_risk += 1
                if len(examples) < 5:
                    examples.append((en, [(c, p["person"], p["subject_gender"]) for c, p in zip(clauses, parses)]))

        out.append(f"\n--- {lang} (n={n_total}) ---")
        out.append(f"Sentences with >=2 coordinated clauses: {n_multi_clause} "
                   f"({n_multi_clause/n_total*100:.1f}%)")
        out.append(f"Of those, at risk of needing >1 target gender (heuristic upper "
                   f"bound): {n_multi_referent_risk} ({n_multi_referent_risk/n_total*100:.2f}% "
                   f"of full dataset)")
        for en, clause_info in examples:
            out.append(f"  e.g. {en[:70]!r}")
            for c, person, gender in clause_info:
                out.append(f"      clause={c[:40]!r:43s} person={person:<8s} subject_gender={gender}")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True,
                        help="Directory containing predictions.json, hgast_marathi_predictions.json, "
                             "hgast_gujarati_predictions.json")
    args = parser.parse_args()

    print(run_quality_analysis(args.data_dir))
    print()
    print(run_multireferent_audit(args.data_dir))


if __name__ == "__main__":
    main()
