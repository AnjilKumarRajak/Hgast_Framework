"""
Arbitration-routing diagnostics.

Two reporting utilities:

  1. compute_controller_distribution / format_controller_distribution_table
     -- reports what fraction of corrections relied on acoustic routing versus
     subject parsing. Aggregates the per-instance
     kappa controller label (HGASTResult.target_gender_reason: "speaker",
     "self-address", "subject", "fallback", "unknown"/"abstain") returned by
     dual_control.resolve_target_gender, so the actual routing behavior -- not just
     the raw acoustic-classifier subgroup accuracy in isolation -- is visible.

  2. compute_coverage_and_effective_accuracy / CoverageAccuracyResult
     -- quantifies the coverage-for-safety trade-off between the
     default fallback-to-speaker-gender behavior and the abstain_on_ambiguous_subject
     variant (see gender/dual_control.py), matching the paper's Appendix B.15
     definition: Effective Accuracy = Coverage x Macro Accuracy.
"""

import logging
from dataclasses import dataclass
from typing import List
from collections import Counter

from .gender_metrics import compute_gender_accuracy

log = logging.getLogger(__name__)

KAPPA_LABELS_ORDER = ("speaker", "self-address", "subject", "fallback", "abstain", "unknown")


@dataclass
class ControllerDistribution:
    counts: dict
    percentages: dict
    n_total: int


def compute_controller_distribution(reasons: List[str]) -> ControllerDistribution:
    """
    reasons: per-instance `target_gender_reason` values from HGASTResult, i.e. the
    kappa label assigned by dual_control.resolve_target_gender for each translated
    instance.
    """
    n = len(reasons)
    counts = dict(Counter(reasons))
    percentages = {k: (round(100.0 * v / n, 2) if n else 0.0) for k, v in counts.items()}
    return ControllerDistribution(counts=counts, percentages=percentages, n_total=n)


def format_controller_distribution_table(dist: ControllerDistribution, label: str = "") -> str:
    title = f"Controller (kappa) distribution{f' -- {label}' if label else ''} (n={dist.n_total})"
    lines = [title, "| kappa label | count | % of instances |", "|---|---|---|"]
    seen = set()
    for k in KAPPA_LABELS_ORDER:
        if k in dist.counts:
            lines.append(f"| {k} | {dist.counts[k]} | {dist.percentages[k]:.2f}% |")
            seen.add(k)
    for k, v in dist.counts.items():
        if k not in seen:
            lines.append(f"| {k} | {v} | {dist.percentages[k]:.2f}% |")
    return "\n".join(lines)


@dataclass
class CoverageAccuracyResult:
    coverage: float            # % of instances with a resolved (non-abstained) target gender
    macro_acc_resolved: float  # macro gender accuracy computed only over the resolved subset
    effective_accuracy: float  # coverage/100 * macro_acc_resolved, paper Appendix B.15
    n_total: int
    n_resolved: int


def compute_coverage_and_effective_accuracy(
    hindi_outputs: List[str],
    arbitrated_target_genders: List[int],
    ground_truth_genders: List[int],
) -> CoverageAccuracyResult:
    """
    arbitrated_target_genders: per-instance HGASTResult.target_gender (-1 = abstain/
        unresolved by arbitration; this is the framework's own routing decision, NOT
        the ground-truth label).
    ground_truth_genders: the true/reference speaker or subject gender (0=male,
        1=female) used to score correctness.

    Effective Accuracy = Coverage x Macro Accuracy (paper Appendix B.15), so that a
    variant cannot inflate its headline accuracy simply by abstaining on hard cases.
    """
    n_total = len(hindi_outputs)
    resolved_idx = [i for i, g in enumerate(arbitrated_target_genders) if g in (0, 1)]
    n_resolved = len(resolved_idx)
    coverage = (n_resolved / n_total * 100) if n_total else 0.0

    if n_resolved == 0:
        return CoverageAccuracyResult(
            coverage=coverage, macro_acc_resolved=float("nan"),
            effective_accuracy=0.0, n_total=n_total, n_resolved=0,
        )

    resolved_outputs = [hindi_outputs[i] for i in resolved_idx]
    resolved_truth = [ground_truth_genders[i] for i in resolved_idx]
    gender_result = compute_gender_accuracy(resolved_outputs, resolved_truth)

    effective_accuracy = (coverage / 100.0) * gender_result.macro_avg
    return CoverageAccuracyResult(
        coverage=coverage, macro_acc_resolved=gender_result.macro_avg,
        effective_accuracy=effective_accuracy, n_total=n_total, n_resolved=n_resolved,
    )
