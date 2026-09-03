
import logging

log = logging.getLogger(__name__)


def compute_bleu(hyps, refs):
    try:
        import sacrebleu
        refs_t = [refs]  # sacrebleu wants list-of-lists (multi-reference)
        return sacrebleu.corpus_bleu(hyps, refs_t).score
    except ImportError:
        log.warning("sacrebleu not installed; pip install sacrebleu")
        return None


def compute_chrf(hyps, refs, word_order=2):
    try:
        import sacrebleu
        refs_t = [refs]
        return sacrebleu.corpus_chrf(hyps, refs_t, word_order=word_order).score
    except ImportError:
        log.warning("sacrebleu not installed; pip install sacrebleu")
        return None


def compute_ter(hyps, refs):
    try:
        import sacrebleu
        refs_t = [refs]
        return sacrebleu.corpus_ter(hyps, refs_t).score
    except ImportError:
        log.warning("sacrebleu not installed; pip install sacrebleu")
        return None


def compute_wer(hyps, refs):
    try:
        import jiwer
        return jiwer.wer(refs, hyps) * 100
    except ImportError:
        log.warning("jiwer not installed; pip install jiwer")
        return None


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
