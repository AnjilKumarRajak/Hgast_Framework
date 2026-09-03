

import random
import math
from typing import Callable, List, Tuple


def bootstrap_ci(hyps: List[str], refs: List[str], metric_fn: Callable,
                  n_samples: int = 1000, ci: float = 0.95, seed: int = 13) -> Tuple[float, float, float]:
    rng = random.Random(seed)
    n = len(hyps)
    scores = []
    for _ in range(n_samples):
        idxs = [rng.randrange(n) for _ in range(n)]
        h_sample = [hyps[i] for i in idxs]
        r_sample = [refs[i] for i in idxs]
        scores.append(metric_fn(h_sample, r_sample))

    scores.sort()
    lo_idx = int((1 - ci) / 2 * n_samples)
    hi_idx = int((1 - (1 - ci) / 2) * n_samples) - 1
    point = metric_fn(hyps, refs)
    return point, scores[lo_idx], scores[hi_idx]


def bootstrap_compare(hyps_a: List[str], hyps_b: List[str], refs: List[str],
                       metric_fn: Callable, n_samples: int = 1000, seed: int = 13) -> dict:
    rng = random.Random(seed)
    n = len(refs)
    wins_b = 0
    diffs = []
    for _ in range(n_samples):
        idxs = [rng.randrange(n) for _ in range(n)]
        a_sample = [hyps_a[i] for i in idxs]
        b_sample = [hyps_b[i] for i in idxs]
        r_sample = [refs[i] for i in idxs]
        score_a = metric_fn(a_sample, r_sample)
        score_b = metric_fn(b_sample, r_sample)
        diffs.append(score_b - score_a)
        if score_b > score_a:
            wins_b += 1

    p_value = 1 - (wins_b / n_samples)  # prob B is NOT better than A
    mean_diff = sum(diffs) / len(diffs)
    return {
        "mean_diff": mean_diff,
        "p_value_approx": p_value,
        "significant_at_0.05": p_value < 0.05,
        "n_samples": n_samples,
    }


def mcnemar_test(baseline_correct: List[bool], treatment_correct: List[bool]) -> dict:
    assert len(baseline_correct) == len(treatment_correct)

    b = sum(1 for bc, tc in zip(baseline_correct, treatment_correct) if bc and not tc)   # baseline right, treatment wrong
    c = sum(1 for bc, tc in zip(baseline_correct, treatment_correct) if not bc and tc)   # baseline wrong, treatment right
    n_discordant = b + c

    if n_discordant == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0, "significant_at_0.05": False}

    if n_discordant < 25:
        # exact binomial test on discordant pairs
        p_value = _binomial_two_sided(min(b, c), n_discordant, 0.5)
    else:
        # chi-square approximation with continuity correction
        stat = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = _chi2_sf_1df(stat)

    return {
        "b_baseline_right_treatment_wrong": b,
        "c_baseline_wrong_treatment_right": c,
        "n_discordant": n_discordant,
        "p_value": p_value,
        "significant_at_0.05": p_value < 0.05,
        "improvement_direction": "treatment_better" if c > b else ("baseline_better" if b > c else "tied"),
    }


def _binomial_two_sided(k: int, n: int, p: float) -> float:
    def _pmf(k, n, p):
        return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
    total = sum(_pmf(i, n, p) for i in range(n + 1) if _pmf(i, n, p) <= _pmf(k, n, p) + 1e-12)
    return min(1.0, total)


def _chi2_sf_1df(x: float) -> float:
    return math.erfc(math.sqrt(x / 2))
