
import logging
from dataclasses import dataclass
from typing import List, Optional

from ..pipeline import HGASTFramework
from ..backbones.base import TranslationBackbone
from ..gender.llm_refine import LLMGenderRefiner
from .gender_metrics import compute_gender_accuracy
from .metrics import compute_bleu, compute_chrf, compute_comet
from .controller_distribution import (
    compute_controller_distribution,
    format_controller_distribution_table,
    compute_coverage_and_effective_accuracy,
)

log = logging.getLogger(__name__)

ABLATION_STAGES = [
    ("backbone_only", dict(use_coreference=False, use_rule_morphology=False, use_llm_refine=False)),
    ("speaker_gender_only", dict(use_coreference=False, use_rule_morphology=False, use_llm_refine=False)),
    ("subject_coreference", dict(use_coreference=True, use_rule_morphology=False, use_llm_refine=False)),
    ("rule_based_morphology", dict(use_coreference=True, use_rule_morphology=True, use_llm_refine=False)),
    ("full_framework", dict(use_coreference=True, use_rule_morphology=True, use_llm_refine=True)),
]

# Alias: "rule_based_morphology" IS the Tier-1-only condition
# (deterministic dictionary substitution, no LLM refiner) referenced in the paper's
# Table 7 (gender accuracy) -- this module now also reports translation-quality deltas
# (chrF++, optionally COMET) for that same stage, which Table 7 did not previously
# include, so the LLM refiner's specific contribution to *quality* (not just gender
# accuracy) is directly visible rather than inferred.
TIER1_ONLY_STAGE = "rule_based_morphology"
FULL_SYSTEM_STAGE = "full_framework"


@dataclass
class AblationRow:
    stage_name: str
    bleu: float
    chrf: float
    male_acc: float
    female_acc: float
    macro_gender_acc: float
    comet: Optional[float] = None


def run_ablation(
    backbone: TranslationBackbone,
    llm_refiner: LLMGenderRefiner,
    en_texts: List[str],
    references_hi: List[str],
    target_genders: List[int],
    speaker_genders: List[int],
    speaker_confidences: List[float],
    compute_comet_score: bool = False,
) -> List[AblationRow]:
    """
    compute_comet_score: off by default (downloads a large model and is slow); set
    True to also populate AblationRow.comet for the per-stage quality breakdown.
    When False, `comet` is left as None in every row.
    """
    rows = []
    for stage_name, flags in ABLATION_STAGES:
        framework = HGASTFramework(backbone=backbone, llm_refiner=llm_refiner, **flags)

        hi_outputs = []
        for en, sg, sc in zip(en_texts, speaker_genders, speaker_confidences):
            result = framework.translate(en, speaker_gender=sg, speaker_confidence=sc)
            hi_outputs.append(result.hindi)

        bleu = compute_bleu(hi_outputs, references_hi)
        chrf = compute_chrf(hi_outputs, references_hi)
        gender_result = compute_gender_accuracy(hi_outputs, target_genders)

        comet = None
        if compute_comet_score:
            comet = compute_comet(en_texts, hi_outputs, references_hi)

        rows.append(AblationRow(
            stage_name=stage_name,
            bleu=bleu if bleu is not None else float("nan"),
            chrf=chrf if chrf is not None else float("nan"),
            male_acc=gender_result.male_acc,
            female_acc=gender_result.female_acc,
            macro_gender_acc=gender_result.macro_avg,
            comet=comet,
        ))
        log.info(
            f"[ablation:{stage_name}] BLEU={bleu} chrF++={chrf} "
            f"macro_gender_acc={gender_result.macro_avg:.2f} COMET={comet}"
        )

    return rows


def format_ablation_table(rows: List[AblationRow]) -> str:
    has_comet = any(r.comet is not None for r in rows)
    header = "| Stage | BLEU | chrF++ | Male Acc | Female Acc | Macro Gender Acc |"
    sep = "|---|---|---|---|---|---|"
    if has_comet:
        header += " COMET |"
        sep += "---|"
    lines = [header, sep]
    for r in rows:
        row = (
            f"| {r.stage_name} | {r.bleu:.2f} | {r.chrf:.2f} | {r.male_acc:.2f}% | "
            f"{r.female_acc:.2f}% | {r.macro_gender_acc:.2f}% |"
        )
        if has_comet:
            row += f" {r.comet:.2f} |" if r.comet is not None else " -- |"
        lines.append(row)
    return "\n".join(lines)


def tier1_vs_full_quality_delta(rows: List[AblationRow]) -> str:
    """
    Isolates the BLEU/chrF++(/COMET) delta
    attributable specifically to Stage 4 (the LLM refiner) on top of Stage 3
    (deterministic Tier-1 morphology alone), using the same rows `run_ablation`
    already produces.
    """
    by_stage = {r.stage_name: r for r in rows}
    if TIER1_ONLY_STAGE not in by_stage or FULL_SYSTEM_STAGE not in by_stage:
        return "Tier-1-only or full-system row missing; run_ablation must include both stages."

    t1, full = by_stage[TIER1_ONLY_STAGE], by_stage[FULL_SYSTEM_STAGE]
    lines = [
        "### Tier-1-only vs. full system (Stage 4 / LLM-refiner quality contribution)",
        "| Metric | Tier-1-only | Full system (Tier-1+2) | Delta attributable to Stage 4 |",
        "|---|---|---|---|",
        f"| BLEU | {t1.bleu:.2f} | {full.bleu:.2f} | {full.bleu - t1.bleu:+.2f} |",
        f"| chrF++ | {t1.chrf:.2f} | {full.chrf:.2f} | {full.chrf - t1.chrf:+.2f} |",
        f"| Macro Gender Acc | {t1.macro_gender_acc:.2f}% | {full.macro_gender_acc:.2f}% | "
        f"{full.macro_gender_acc - t1.macro_gender_acc:+.2f} pts |",
    ]
    if t1.comet is not None and full.comet is not None:
        lines.append(f"| COMET | {t1.comet:.2f} | {full.comet:.2f} | {full.comet - t1.comet:+.2f} |")
    return "\n".join(lines)


@dataclass
class ArbitrationVariantRow:
    variant_name: str
    coverage: float
    macro_acc_resolved: float
    effective_accuracy: float
    controller_distribution_table: str


def run_arbitration_variant_comparison(
    backbone: TranslationBackbone,
    llm_refiner: LLMGenderRefiner,
    en_texts: List[str],
    ground_truth_genders: List[int],
    speaker_genders: List[int],
    speaker_confidences: List[float],
    subj_confidence_threshold: float = 0.55,
) -> List[ArbitrationVariantRow]:
    """
    Compares the current default (fall back to speaker/acoustic
    gender when a third-person subject is unresolved or low-confidence) against the
    abstain_on_ambiguous_subject variant, reporting Coverage and Effective Accuracy
    (paper Appendix B.15 formula) for both, plus each variant's kappa-controller
    distribution (the fraction of instances routed by acoustic vs. parsing cues,
    per variant).
    """
    rows = []
    for variant_name, abstain_flag in [
        ("dual_control_fallback (current default)", False),
        ("dual_control_abstain_blanket", True),
        ("dual_control_abstain_targeted (recommended)", "targeted"),
    ]:
        framework = HGASTFramework(
            backbone=backbone, llm_refiner=llm_refiner,
            subj_confidence_threshold=subj_confidence_threshold,
            abstain_on_ambiguous_subject=abstain_flag,
        )
        hi_outputs, arbitrated, reasons = [], [], []
        for en, sg, sc in zip(en_texts, speaker_genders, speaker_confidences):
            result = framework.translate(en, speaker_gender=sg, speaker_confidence=sc)
            hi_outputs.append(result.hindi)
            arbitrated.append(result.target_gender)
            reasons.append(result.target_gender_reason)

        cov_result = compute_coverage_and_effective_accuracy(hi_outputs, arbitrated, ground_truth_genders)
        dist = compute_controller_distribution(reasons)
        dist_table = format_controller_distribution_table(dist, label=variant_name)

        rows.append(ArbitrationVariantRow(
            variant_name=variant_name,
            coverage=cov_result.coverage,
            macro_acc_resolved=cov_result.macro_acc_resolved,
            effective_accuracy=cov_result.effective_accuracy,
            controller_distribution_table=dist_table,
        ))
        log.info(
            f"[arbitration_variant:{variant_name}] coverage={cov_result.coverage:.2f}% "
            f"macro_acc={cov_result.macro_acc_resolved:.2f}% "
            f"eff_acc={cov_result.effective_accuracy:.2f}%"
        )
    return rows


def format_arbitration_variant_table(rows: List[ArbitrationVariantRow]) -> str:
    lines = ["| Variant | Coverage | Macro Acc (resolved) | Effective Accuracy |",
             "|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r.variant_name} | {r.coverage:.2f}% | {r.macro_acc_resolved:.2f}% | "
            f"{r.effective_accuracy:.2f}% |"
        )
    table = "\n".join(lines)
    distributions = "\n\n".join(r.controller_distribution_table for r in rows)
    return f"{table}\n\n{distributions}"
