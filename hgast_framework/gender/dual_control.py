

import logging

log = logging.getLogger(__name__)

FORMAL_CUES = {"sir", "madam", "ma'am", "please", "could you", "would you",
               "kindly", "may i ask", "i would like", "excuse me"}
INTIMATE_CUES = {"hey", "bro", "dude", "mate", "buddy", "come on"}

# Singular, animate, gendered kinship/pronoun terms -- the exact
# profile of "my mother is reading". Used by the *targeted* abstention variant below,
# which abstains only when the unresolved subject matches this list, rather than on
# every unresolved third-person subject (which, per analyze_arbitration_variant.py run
# against the full Hindi evaluation set, is 56.8% of instances and overwhelmingly
# plural/collective/abstract subjects where no single grammatical gender is being
# overridden -- blanket abstention there costs far more Coverage than it buys safety).
GENDERED_KINSHIP_OR_PRONOUN = {
    "mother", "sister", "wife", "daughter", "girl", "woman", "aunt", "grandmother",
    "actress", "she", "her", "hers", "queen", "princess", "lady", "girlfriend",
    "mom", "mum", "nurse", "niece",
    "father", "brother", "husband", "son", "boy", "man", "uncle", "grandfather",
    "actor", "he", "him", "his", "king", "prince", "gentleman", "boyfriend", "dad",
    "nephew",
}


def detect_politeness(en: str) -> str:
    el = en.lower()
    if any(c in el for c in FORMAL_CUES):
        return "formal"
    if any(c in el for c in INTIMATE_CUES):
        return "intimate"
    return "informal"


def build_dual_control(speaker_gender: int, speaker_confidence: float,
                        subj_info: dict, politeness: str) -> dict:
    """
    speaker_gender: 0=male, 1=female, -1=unknown  (from audio gender detector)
    subj_info: output of linguistic_analysis.parse_subject()
    """
    subj_g = subj_info["subject_gender"]
    person = subj_info["person"]

    spk_ctrl = {
        "gender": speaker_gender,
        "label": "female" if speaker_gender == 1 else ("male" if speaker_gender == 0 else "unknown"),
        "source": "wav2vec2_gender",
        "confidence": speaker_confidence,
    }

    if person == "first":
        subj_gender_int, subj_src = speaker_gender, "speaker_is_subject"
    elif person == "second":
        subj_gender_int, subj_src = -1, f"second_person_{politeness}"
    elif person == "third":
        subj_gender_int = (1 if subj_g == "female" else 0 if subj_g == "male" else -1)
        subj_src = "grammatical_subject"
    else:
        subj_gender_int, subj_src = -1, "neutral"

    subj_ctrl = {
        "gender": subj_gender_int, "label": subj_g, "person": person,
        "politeness": politeness, "source": subj_src,
        "confidence": subj_info.get("confidence", 0.0),
        "text": subj_info.get("subject", ""),
    }

    dominant = "speaker" if person == "first" else ("politeness" if person == "second" else "subject")

    spk_resolved = speaker_gender in (0, 1)
    subj_resolved = subj_gender_int in (0, 1)
    are_same = (speaker_gender == subj_gender_int) if (spk_resolved and subj_resolved) else None

    return {
        "speaker_ctrl": spk_ctrl,
        "subject_ctrl": subj_ctrl,
        "dominant_controller": dominant,
        "are_same": are_same,
    }


def resolve_target_gender(
    dual: dict,
    subj_confidence_threshold: float = 0.55,
    abstain_on_ambiguous_subject=False,
) -> tuple:
    """
    Arbitration function A(p, g_ac, g_subj, c) resolving (g_target, kappa):
      - p = 1: (g_ac, "speaker")
      - p = 2: (g_ac, "self-address")
      - p = 3, g_subj != empty, c >= tau: (g_subj, "subject")
      - p = 3, (g_subj == empty or c < tau):
            abstain_on_ambiguous_subject=False (default):     (g_ac, "fallback")
            abstain_on_ambiguous_subject=True  (blanket):     (empty, "abstain")
            abstain_on_ambiguous_subject="targeted":          (empty, "abstain") only if
                the unresolved subject's text matches GENDERED_KINSHIP_OR_PRONOUN,
                otherwise (g_ac, "fallback") as in the default.
    where tau = subj_confidence_threshold (default 0.55).

    For third-person subjects, when parsing confidence is
    insufficient to identify the subject's gender, the acoustic speaker signal has no
    evidentiary relationship to the (unrelated) third-person referent's gender --
    unlike the p=1/p=2 branches, where it is the only available signal by construction
    (Principle 1). Falling back to it in this branch risks confidently misgendering a
    third-person referent using a different person's voice (e.g., a male speaker
    saying "my mother is reading" could otherwise surface a masculine form if the
    parser misfired on that subject).

    In practice, "my mother is reading" itself never reaches this branch: sentence-
    initial possessive+kinship subjects are caught by a dedicated fast path in
    linguistic_analysis.parse_subject (POSSESSIVES + THIRD_PERSON_NOUNS ->
    THIRD_GENDER_MAP), which resolves "mother" at confidence 0.95, well above tau, and
    routes via the ("subject") branch above -- not this one. This branch is reached
    only when the subject could not be matched by that fast path or the general
    parser at sufficient confidence.

    We measured both the "blanket" and "targeted" variants against the full 4,865-
    sentence Hindi evaluation set (see analyze_arbitration_variant.py): this branch is
    reached for 56.8% of instances, but only ~0.02% involve a subject matching
    GENDERED_KINSHIP_OR_PRONOUN (the targeted failure-mode profile) -- the
    rest are plural/collective ("they", "people"), abstract/inanimate ("capital",
    "value"), or proper-noun subjects where no single overridden gender is at stake.
    Blanket abstention costs ~57 points of Coverage (Effective Accuracy 99.55% ->
    42.87%) to guard against a ~0.02% risk; targeted abstention closes the same gap
    for a fraction of the cost. "targeted" is the version we recommend adopting as the
    new default; "blanket" (True) is retained for direct comparison.
    """
    dominant = dual["dominant_controller"]
    speaker_gender = dual["speaker_ctrl"]["gender"]
    subj_gender = dual["subject_ctrl"]["gender"]
    subj_conf = dual["subject_ctrl"].get("confidence", 0.0)
    subj_text = dual["subject_ctrl"].get("text", "")

    # p = 1: (g_ac, speaker)
    if dominant == "speaker":
        return (speaker_gender, "speaker") if speaker_gender != -1 else (-1, "unknown")

    # p = 2: (g_ac, self-address)
    if dominant == "politeness":
        return (speaker_gender, "self-address") if speaker_gender != -1 else (-1, "unknown")

    # p = 3: subject dominant if confident, else fallback to acoustic speaker
    # (or abstain -- blanket or targeted -- see docstring above)
    if dominant == "subject":
        if subj_gender != -1 and subj_conf >= subj_confidence_threshold:
            return subj_gender, "subject"

        if abstain_on_ambiguous_subject == "targeted":
            if subj_text.strip().lower() in GENDERED_KINSHIP_OR_PRONOUN:
                return -1, "abstain"
            # else fall through to the default fallback behavior below
        elif abstain_on_ambiguous_subject:
            return -1, "abstain"

        if speaker_gender != -1:
            return speaker_gender, "fallback"
        return -1, "fallback"

    return -1, "unknown"
