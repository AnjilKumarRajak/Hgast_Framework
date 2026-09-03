

import logging

log = logging.getLogger(__name__)

FORMAL_CUES = {"sir", "madam", "ma'am", "please", "could you", "would you",
               "kindly", "may i ask", "i would like", "excuse me"}
INTIMATE_CUES = {"hey", "bro", "dude", "mate", "buddy", "come on"}


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


def resolve_target_gender(dual: dict, subj_confidence_threshold: float = 0.55) -> tuple:
    """Returns (target_gender: int in {0,1,-1}, reason: str)."""
    dominant = dual["dominant_controller"]
    speaker_gender = dual["speaker_ctrl"]["gender"]
    subj_gender = dual["subject_ctrl"]["gender"]
    subj_conf = dual["subject_ctrl"].get("confidence", 0.0)

    if dominant == "speaker":
        return (speaker_gender, "speaker_primary") if speaker_gender != -1 else (-1, "unknown")

    if dominant == "politeness":
        return (speaker_gender, "speaker_fallback_politeness") if speaker_gender != -1 else (-1, "unknown")

    if dominant == "subject":
        if subj_gender != -1 and subj_conf >= subj_confidence_threshold:
            return subj_gender, "subject_dominant"
        if speaker_gender != -1:
            return speaker_gender, "speaker_fallback_subject_uncertain"
        return -1, "unknown_subject_no_forcing"

    return -1, "unknown"
