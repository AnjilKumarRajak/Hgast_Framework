
from dataclasses import dataclass
from typing import List
from ..gender.morphology_rules import count_morph_tokens


@dataclass
class GenderAccuracyResult:
    male_acc: float
    female_acc: float
    macro_avg: float
    n_male: int
    n_female: int
    per_example_correct: List[bool]  # needed later for McNemar's test


def is_gender_correct(hindi_text: str, target_gender: int) -> bool:
    info = count_morph_tokens(hindi_text)
    expected = "female" if target_gender == 1 else "male"
    return info["dominant"] in ("neutral", expected)


def compute_gender_accuracy(hindi_outputs: List[str], target_genders: List[int]) -> GenderAccuracyResult:
    assert len(hindi_outputs) == len(target_genders), "outputs and labels must align 1:1"

    male_correct, male_total = 0, 0
    female_correct, female_total = 0, 0
    per_example = []

    for hi, tgt in zip(hindi_outputs, target_genders):
        correct = is_gender_correct(hi, tgt)
        per_example.append(correct)
        if tgt == 0:
            male_total += 1
            male_correct += int(correct)
        elif tgt == 1:
            female_total += 1
            female_correct += int(correct)

    male_acc = (male_correct / male_total * 100) if male_total else float("nan")
    female_acc = (female_correct / female_total * 100) if female_total else float("nan")
    valid = [a for a in (male_acc, female_acc) if a == a]  # drop NaN
    macro = sum(valid) / len(valid) if valid else float("nan")

    return GenderAccuracyResult(
        male_acc=male_acc, female_acc=female_acc, macro_avg=macro,
        n_male=male_total, n_female=female_total,
        per_example_correct=per_example,
    )
