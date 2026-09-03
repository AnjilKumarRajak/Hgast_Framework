
import logging
from dataclasses import dataclass
from typing import List

from ..pipeline import HGASTFramework
from ..backbones.base import TranslationBackbone
from ..gender.llm_refine import LLMGenderRefiner
from .gender_metrics import compute_gender_accuracy
from .metrics import compute_bleu

log = logging.getLogger(__name__)

ABLATION_STAGES = [
    ("backbone_only", dict(use_coreference=False, use_rule_morphology=False, use_llm_refine=False)),
    ("speaker_gender_only", dict(use_coreference=False, use_rule_morphology=False, use_llm_refine=False)),
    ("subject_coreference", dict(use_coreference=True, use_rule_morphology=False, use_llm_refine=False)),
    ("rule_based_morphology", dict(use_coreference=True, use_rule_morphology=True, use_llm_refine=False)),
    ("full_framework", dict(use_coreference=True, use_rule_morphology=True, use_llm_refine=True)),
]


@dataclass
class AblationRow:
    stage_name: str
    bleu: float
    male_acc: float
    female_acc: float
    macro_gender_acc: float


def run_ablation(
    backbone: TranslationBackbone,
    llm_refiner: LLMGenderRefiner,
    en_texts: List[str],
    references_hi: List[str],
    target_genders: List[int],
    speaker_genders: List[int],
    speaker_confidences: List[float],
) -> List[AblationRow]:
    rows = []
    for stage_name, flags in ABLATION_STAGES:
        framework = HGASTFramework(backbone=backbone, llm_refiner=llm_refiner, **flags)

        hi_outputs = []
        for en, sg, sc in zip(en_texts, speaker_genders, speaker_confidences):
            result = framework.translate(en, speaker_gender=sg, speaker_confidence=sc)
            hi_outputs.append(result.hindi)

        bleu = compute_bleu(hi_outputs, references_hi)
        gender_result = compute_gender_accuracy(hi_outputs, target_genders)

        rows.append(AblationRow(
            stage_name=stage_name,
            bleu=bleu if bleu is not None else float("nan"),
            male_acc=gender_result.male_acc,
            female_acc=gender_result.female_acc,
            macro_gender_acc=gender_result.macro_avg,
        ))
        log.info(f"[ablation:{stage_name}] BLEU={bleu} macro_gender_acc={gender_result.macro_avg:.2f}")

    return rows


def format_ablation_table(rows: List[AblationRow]) -> str:
    lines = ["| Stage | BLEU | Male Acc | Female Acc | Macro Gender Acc |",
             "|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r.stage_name} | {r.bleu:.2f} | {r.male_acc:.2f}% | "
            f"{r.female_acc:.2f}% | {r.macro_gender_acc:.2f}% |"
        )
    return "\n".join(lines)
