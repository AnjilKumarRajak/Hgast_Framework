
import logging

log = logging.getLogger(__name__)


def _simple_bleu(hyps, refs):
    """Pure-Python word-level 1-gram / 2-gram BLEU fallback."""
    import collections, math
    if not hyps or not refs:
        return 0.0
    precisions = []
    for n in range(1, 3):
        matched, total = 0, 0
        for h, r in zip(hyps, refs):
            h_ngrams = [tuple(h.split()[i:i+n]) for i in range(len(h.split()) - n + 1)]
            r_ngrams = collections.Counter([tuple(r.split()[i:i+n]) for i in range(len(r.split()) - n + 1)])
            total += len(h_ngrams)
            for ng in h_ngrams:
                if r_ngrams[ng] > 0:
                    matched += 1
                    r_ngrams[ng] -= 1
        precisions.append((matched / total) if total > 0 else 1e-4)
    hyp_len = sum(len(h.split()) for h in hyps)
    ref_len = sum(len(r.split()) for r in refs)
    bp = math.exp(min(0, 1 - (ref_len / hyp_len))) if hyp_len > 0 else 0.0
    p = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    return round(bp * p * 100, 2)


def compute_bleu(hyps, refs):
    try:
        import sacrebleu
        refs_t = [refs]  # sacrebleu wants list-of-lists (multi-reference)
        return sacrebleu.corpus_bleu(hyps, refs_t).score
    except Exception:
        return _simple_bleu(hyps, refs)


def compute_chrf(hyps, refs, word_order=2):
    try:
        import sacrebleu
        refs_t = [refs]
        return sacrebleu.corpus_chrf(hyps, refs_t, word_order=word_order).score
    except Exception:
        return _simple_bleu(hyps, refs)


def compute_ter(hyps, refs):
    try:
        import sacrebleu
        refs_t = [refs]
        return sacrebleu.corpus_ter(hyps, refs_t).score
    except Exception:
        return 0.0


def compute_wer(hyps, refs):
    try:
        import jiwer
        return jiwer.wer(refs, hyps) * 100
    except Exception:
        # Simple word error fallback
        errors, total = 0, 0
        for h, r in zip(hyps, refs):
            hw, rw = set(h.split()), set(r.split())
            errors += len((hw - rw) | (rw - hw))
            total += len(rw)
        return round((errors / max(total, 1)) * 100, 2)


def compute_bertscore(hyps, refs, lang="hi"):
    try:
        import bert_score
        P, R, F1 = bert_score.score(hyps, refs, lang=lang, verbose=False)
        return float(F1.mean()) * 100
    except ImportError:
        log.warning("bert-score not installed; pip install bert-score")
        return None


def compute_meteor(hyps, refs):
    try:
        import nltk
        from nltk.translate.meteor_score import meteor_score
        nltk.download("wordnet", quiet=True)
        scores = [
            meteor_score([r.split()], h.split()) for h, r in zip(hyps, refs)
        ]
        return sum(scores) / len(scores) if scores else 0.0
    except ImportError:
        log.warning("nltk not installed; pip install nltk")
        return None


def compute_comet(srcs, hyps, refs, model_name="Unbabel/wmt22-comet-da"):
    try:
        from comet import download_model, load_from_checkpoint
        model_path = download_model(model_name)
        model = load_from_checkpoint(model_path)
        data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
        output = model.predict(data, batch_size=16, gpus=1)
        return float(output.system_score) * 100
    except ImportError:
        log.warning("unbabel-comet not installed; pip install unbabel-comet")
        return None


def compute_all_metrics(srcs, hyps, refs) -> dict:
    """Compute standard machine translation and evaluation metrics."""
    return {
        "BLEU": compute_bleu(hyps, refs),
        "COMET": compute_comet(srcs, hyps, refs),
        "BERTScore": compute_bertscore(hyps, refs),
        "chrF++": compute_chrf(hyps, refs),
        "METEOR": compute_meteor(hyps, refs),
        "WER": compute_wer(hyps, refs),
        "TER": compute_ter(hyps, refs),
    }
