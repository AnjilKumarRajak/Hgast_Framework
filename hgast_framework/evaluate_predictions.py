
import argparse
import json
import random
import sys
from typing import Dict, List

import numpy as np
from sacrebleu.metrics import BLEU, CHRF, TER

from .evaluation.significance import mcnemar_test
from .gender.morphology_rules import count_morph_tokens, morph_match
from .gender.morphology_rules_gu import count_morph_tokens_gu, morph_match_gu
from .gender.morphology_rules_mr import count_morph_tokens_mr, morph_match_mr

SCORERS = {
    "hi": (count_morph_tokens, morph_match),
    "mr": (count_morph_tokens_mr, morph_match_mr),
    "gu": (count_morph_tokens_gu, morph_match_gu),
}
FIELDS = {"target_gender": "target_gender", "baseline": "baseline", "hgast": "hgast",
          "reference": "reference", "source": "source"}


def _pct(num: float, den: float) -> float:
    return 100.0 * num / den if den else float("nan")


def load_records(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        text = f.read().strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def gender_stats(outputs: List[str], targets: List[int], lang: str) -> Dict:
    """Male/female/macro accuracy plus how much of it comes from outputs with no gendered form."""
    count, match = SCORERS[lang]
    res: Dict = {}
    for g, name in ((0, "male"), (1, "female")):
        expected = "female" if g == 1 else "male"
        infos = [count(outputs[i] or "") for i, t in enumerate(targets) if t == g]
        n = len(infos)
        neutral = sum(inf["dominant"] == "neutral" for inf in infos)
        res[name] = {
            "n": n,
            "accuracy": _pct(sum(match(inf, g) for inf in infos), n),
            "neutral_share": _pct(neutral, n),
            "n_gendered_outputs": n - neutral,
            "accuracy_on_gendered_outputs": _pct(sum(inf["dominant"] == expected for inf in infos), n - neutral),
        }
    res["macro"] = (res["male"]["accuracy"] + res["female"]["accuracy"]) / 2
    resolved = sum(t in (0, 1) for t in targets)
    res["coverage"] = _pct(resolved, len(targets))
    res["effective_accuracy"] = res["coverage"] * res["macro"] / 100.0
    return res


def per_example_correct(outputs: List[str], targets: List[int], lang: str) -> List[bool]:
    count, match = SCORERS[lang]
    return [match(count(o or ""), t) for o, t in zip(outputs, targets) if t in (0, 1)]


def _edit_distance(a: List[str], b: List[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def _metric_rows(hyps: List[str], refs: List[str]):
    """Per-sentence sufficient statistics and a scorer over their sum, for each metric.

    Summing rows and scoring reproduces the corpus score exactly, so the point estimate
    and every bootstrap resample are computed the same way.
    """
    for name, metric in (("BLEU", BLEU()), ("chrF++", CHRF(word_order=2)), ("TER", TER())):
        rows = np.array(metric._extract_corpus_statistics(hyps, [refs]), dtype=float)
        yield name, rows, (lambda s, m=metric: m._compute_score_from_stats(s.tolist()).score)
    # corpus WER = total word edits / total reference words (as jiwer.wer)
    rows = np.array([[_edit_distance(h.split(), r.split()), len(r.split())] for h, r in zip(hyps, refs)], dtype=float)
    yield "WER", rows, (lambda s: 100.0 * s[0] / s[1] if s[1] else float("nan"))


def _mean_rows(values: List[float]):
    rows = np.array([[v, 1.0] for v in values], dtype=float)
    return rows, (lambda s: 100.0 * s[0] / s[1] if s[1] else float("nan"))


def translation_stats(systems: Dict[str, List[str]], refs: List[str], n_boot: int = 1000, seed: int = 13,
                      sentence_scores: Dict[str, Dict[str, List[float]]] = None) -> Dict:
    """Point metrics plus paired bootstrap (the same resamples for every system and metric).

    sentence_scores: optional {metric: {system: per-sentence score in [0, 1]}} for metrics
    whose corpus score is the mean of sentence scores (BERTScore F1, COMET).
    """
    rng = random.Random(seed)
    n = len(refs)
    resamples = [np.array([rng.randrange(n) for _ in range(n)]) for _ in range(n_boot)]
    out: Dict = {name: {} for name in systems}
    for sname, hyps in systems.items():
        hyps = [h or "" for h in hyps]
        metrics = list(_metric_rows(hyps, refs))
        for mname, per_sys in (sentence_scores or {}).items():
            metrics.append((mname, *_mean_rows(per_sys[sname])))
        for mname, rows, score in metrics:
            point = score(rows.sum(axis=0))
            boots = np.array([score(rows[idx].sum(axis=0)) for idx in resamples])
            lo, hi = np.percentile(boots, [2.5, 97.5])
            out[sname][mname] = {"point": point, "boot_mean": float(boots.mean()), "boot_sd": float(boots.std(ddof=1)),
                                 "ci95": [float(lo), float(hi)], "ci95_half_width": float((hi - lo) / 2)}
    return out


def sentence_level_extra(systems: Dict[str, List[str]], refs: List[str], srcs: List[str], lang: str) -> Dict:
    """Per-sentence BERTScore F1 and COMET (wmt22-comet-da) for each system."""
    import bert_score
    from comet import download_model, load_from_checkpoint

    comet = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
    scores: Dict[str, Dict[str, List[float]]] = {"BERT": {}, "COMET": {}}
    for sname, hyps in systems.items():
        _, _, f1 = bert_score.score(hyps, refs, lang=lang, verbose=False)
        scores["BERT"][sname] = [float(x) for x in f1]
        data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
        scores["COMET"][sname] = [float(x) for x in comet.predict(data, batch_size=16, gpus=1).scores]
    return scores


def evaluate(records: List[dict], lang: str, name: str, fields: Dict[str, str],
             n_boot: int = 1000, extra: bool = False) -> Dict:
    tg = [r.get(fields["target_gender"]) for r in records]
    base = [r.get(fields["baseline"]) or "" for r in records]
    hg = [r.get(fields["hgast"]) or "" for r in records]
    refs = [r.get(fields["reference"]) or "" for r in records]
    fem = [i for i, t in enumerate(tg) if t == 1]

    result = {"name": name, "lang": lang, "n_records": len(records),
              "gender": {"baseline": gender_stats(base, tg, lang), "hgast": gender_stats(hg, tg, lang)},
              "mcnemar": mcnemar_test(per_example_correct(base, tg, lang), per_example_correct(hg, tg, lang))}

    systems = {"baseline": base, "hgast": hg}
    sent = None
    if extra:  # BERTScore and COMET models are large; computed once, then bootstrapped like the rest
        srcs = [r.get(fields["source"]) or "" for r in records]
        sent = sentence_level_extra(systems, refs, srcs, lang)
    sent_fem = ({m: {s: [v[i] for i in fem] for s, v in per.items()} for m, per in sent.items()}
                if sent else None)

    result["translation_all"] = translation_stats(systems, refs, n_boot, sentence_scores=sent)
    result["translation_female"] = translation_stats(
        {s: [h[i] for i in fem] for s, h in systems.items()}, [refs[i] for i in fem], n_boot,
        sentence_scores=sent_fem)
    d_all = result["translation_all"]["hgast"]["BLEU"]["point"] - result["translation_all"]["baseline"]["BLEU"]["point"]
    d_fem = result["translation_female"]["hgast"]["BLEU"]["point"] - result["translation_female"]["baseline"]["BLEU"]["point"]
    result["bleu_delta"] = {"all": d_all, "female": d_fem, "amplification": d_fem / d_all if d_all else float("nan")}
    return result


def print_report(res: Dict) -> None:
    g = res["gender"]
    print(f"=== {res['name']} ({res['lang']}), {res['n_records']} records, "
          f"coverage {g['hgast']['coverage']:.1f}%")
    for sysname in ("baseline", "hgast"):
        s = g[sysname]
        print(f"  {sysname:8s} male {s['male']['accuracy']:6.2f}  female {s['female']['accuracy']:6.2f}  "
              f"macro {s['macro']:6.2f}  eff {s['effective_accuracy']:6.2f} | no gendered form: "
              f"male {s['male']['neutral_share']:5.1f}%  female {s['female']['neutral_share']:5.1f}% | "
              f"on gendered outputs: male {s['male']['accuracy_on_gendered_outputs']:6.2f}  "
              f"female {s['female']['accuracy_on_gendered_outputs']:6.2f}")
    m = res["mcnemar"]
    print(f"  McNemar b={m.get('b_baseline_right_treatment_wrong')} c={m.get('c_baseline_wrong_treatment_right')} "
          f"p={m.get('p_value')}")
    for scope in ("translation_all", "translation_female"):
        for sysname in ("baseline", "hgast"):
            parts = [f"{k} {v['point']:6.2f} (boot {v['boot_mean']:6.2f}+-{v['boot_sd']:.2f}, "
                     f"CI+-{v['ci95_half_width']:.2f})" for k, v in res[scope][sysname].items()]
            print(f"  {scope[12:]:6s} {sysname:8s} " + "  ".join(parts))
    d = res["bleu_delta"]
    print(f"  BLEU delta: all {d['all']:+.2f}  female {d['female']:+.2f}  amplification {d['amplification']:.1f}x")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("predictions")
    ap.add_argument("--lang", required=True, choices=sorted(SCORERS))
    ap.add_argument("--name", default="")
    ap.add_argument("--field", action="append", default=[], help="KEY=JSON_FIELD, e.g. hgast=framework_output")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--extra", action="store_true", help="also compute BERTScore and COMET (needs a GPU; slow)")
    ap.add_argument("--out", help="write the full result as JSON")
    args = ap.parse_args(argv)

    fields = dict(FIELDS)
    for item in args.field:
        key, _, value = item.partition("=")
        if key not in fields or not value:
            ap.error(f"bad --field {item!r}; keys: {', '.join(fields)}")
        fields[key] = value

    res = evaluate(load_records(args.predictions), args.lang, args.name or args.predictions, fields,
                   n_boot=args.n_boot, extra=args.extra)
    print_report(res)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
