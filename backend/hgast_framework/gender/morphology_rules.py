"""
gender/morphology_rules.py
===========================
Rule-based Hindi gender morphology correction. This preserves your original
MALE_TO_FEMALE / FEMALE_TO_MALE substitution tables (your hand-validated
linguistic asset) but fixes the surrounding bugs:
  - clean_hindi is defined once, here, not duplicated/undefined elsewhere.
  - _apply_rules no longer mutates a `tokens` list it doesn't rebuild
    correctly on non-ambiguous replacements (original re-split every full-string
    replace, which is correct but wasteful) — behaviour preserved, just documented.

NOTE ON COVERAGE (put this in your Limitations section):
This table is finite. Any verb form / construction not listed here will
NOT be corrected regardless of which backbone produced the Hindi text.
"""

import re

# ---------------------------------------------------------------------------
# Text cleanup
# ---------------------------------------------------------------------------
def clean_hindi(text: str) -> str:
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[ ]+([?.!,।])", r"\1", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Substitution tables (male -> female surface forms)
# ---------------------------------------------------------------------------
MALE_TO_FEMALE = [
    ("जा रहा हूँ", "जा रही हूँ"), ("जा रहा था", "जा रही थी"), ("जा रहा है", "जा रही है"),
    ("खा रहा हूँ", "खा रही हूँ"), ("खा रहा था", "खा रही थी"), ("खा रहा है", "खा रही है"),
    ("कर रहा हूँ", "कर रही हूँ"), ("कर रहा था", "कर रही थी"), ("कर रहा है", "कर रही है"),
    ("पढ़ रहा हूँ", "पढ़ रही हूँ"), ("पढ़ रहा था", "पढ़ रही थी"), ("पढ़ रहा है", "पढ़ रही है"),
    ("लिख रहा हूँ", "लिख रही हूँ"), ("लिख रहा था", "लिख रही थी"), ("लिख रहा है", "लिख रही है"),
    ("सो रहा हूँ", "सो रही हूँ"), ("सो रहा था", "सो रही थी"),
    ("दौड़ रहा हूँ", "दौड़ रही हूँ"), ("बोल रहा हूँ", "बोल रही हूँ"),
    ("चल रहा हूँ", "चल रही हूँ"), ("सीख रहा हूँ", "सीख रही हूँ"),
    ("देख रहा हूँ", "देख रही हूँ"), ("सुन रहा हूँ", "सुन रही हूँ"),
    ("रहा हूँ", "रही हूँ"), ("रहा था", "रही थी"), ("रहा है", "रही है"), ("रहा हो", "रही हो"),
    ("गया था", "गई थी"), ("गया हूँ", "गई हूँ"),
    ("आया था", "आई थी"), ("आया हूँ", "आई हूँ"),
    ("खाया था", "खाई थी"), ("किया था", "की थी"),
    ("लिया था", "ली थी"), ("दिया था", "दी थी"),
    ("पाया था", "पाई थी"), ("सोया था", "सोई थी"),
    ("बोला था", "बोली थी"), ("चला था", "चली थी"),
    ("बैठा था", "बैठी थी"), ("उठा था", "उठी थी"),
    ("मिला था", "मिली थी"),
    ("गया", "गई"), ("आया", "आई"), ("खाया", "खाई"), ("किया", "की"),
    ("लिया", "ली"), ("दिया", "दी"), ("पाया", "पाई"), ("सोया", "सोई"),
    ("बोला", "बोली"), ("चला", "चली"), ("बैठा", "बैठी"), ("उठा", "उठी"),
    ("मिला", "मिली"),
    ("गए थे", "गई थीं"), ("आए थे", "आई थीं"), ("किए थे", "की थीं"),
    ("होता है", "होती है"), ("होता था", "होती था"),
    ("था", "थी"), ("होगा", "होगी"), ("होता", "होती"),
    ("रहा", "रही"),
    ("खुश था", "खुश थी"), ("बीमार था", "बीमार थी"),
    ("तैयार था", "तैयार थी"), ("थका हुआ", "थकी हुई"),
    ("थका", "थकी"), ("मजबूर था", "मजबूर थी"),
    ("करवाया", "करवाई"), ("बनवाया", "बनवाई"),
    ("सकता है", "सकती है"), ("सकता था", "सकती थी"),
    ("चाहता है", "चाहती है"), ("चाहता था", "चाहती थी"),
    ("सका", "सकी"),
    ("जाता है", "जाती है"), ("जाता था", "जाती था"),
    ("आता है", "आती है"), ("आता था", "आती था"),
    ("करता है", "करती है"), ("करता था", "करती था"),
    ("पढ़ता है", "पढ़ती है"), ("पढ़ता था", "पढ़ती थी"),
    ("खाता है", "खाती है"), ("खाता था", "खाती था"),
    ("बोलता है", "बोलती है"), ("बोलता था", "बोलती था"),
    ("देखता है", "देखती है"), ("देखता था", "देखती था"),
    ("सुनता है", "सुनती है"), ("सुनता था", "सुनती था"),
    ("चलता है", "चलती है"), ("चलता था", "चलती था"),
    ("मिलता है", "मिलती है"), ("मिलता था", "मिलती था"),
    ("लिखता है", "लिखती है"), ("लिखता था", "लिखती था"),
    ("सोता है", "सोती है"), ("सोता था", "सोती था"),
    ("दौड़ता है", "दौड़ती है"), ("दौड़ता था", "दौड़ती था"),
    ("समझता हूँ", "समझती हूँ"), ("समझता है", "समझती है"), ("समझता था", "समझती था"),
    ("मुड़ता हूँ", "मुड़ती हूँ"), ("मुड़ता है", "मुड़ती है"),
    ("हो गया", "हो गई"), ("हो गया था", "हो गई थी"), ("हो गए", "हो गईं"),
    ("कर सकता हूँ", "कर सकती हूँ"), ("कर सकता है", "कर सकती है"),
    ("रख सकता हूँ", "रख सकती हूँ"), ("रखता हूँ", "रखती हूँ"),
    ("रखता है", "रखती है"), ("रखता था", "रखती था"),
    ("पाता हूँ", "पाती हूँ"), ("पाता है", "पाती है"),
    ("जानता हूँ", "जानती हूँ"), ("जानता है", "जानती है"), ("जानता था", "जानती था"),
    ("प्रदर्शित कर सकता हूँ", "प्रदर्शित कर सकती हूँ"),
    ("जाऊंगा", "जाऊंगी"), ("जाऊँगा", "जाऊँगी"),
    ("खाऊंगा", "खाऊंगी"), ("खाऊँगा", "खाऊँगी"),
    ("करूंगा", "करूंगी"), ("करूँगा", "करूँगी"),
    ("पीऊंगा", "पीऊंगी"), ("पीऊँगा", "पीऊँगी"),
    ("आऊंगा", "आऊंगी"), ("आऊँगा", "आऊँगी"),
    ("रहूंगा", "रहूंगी"), ("रहूँगा", "रहूँगी"),
    ("देखूंगा", "देखूंगी"), ("देखूँगा", "देखूँगी"),
    ("सोऊंगा", "सोऊंगी"), ("सोऊँगा", "सोऊँगी"),
    ("पढूंगा", "पढूंगी"), ("पढूँगा", "पढूँगी"),
    ("लिखूंगा", "लिखूंगी"), ("लिखूँगा", "लिखूँगी"),
    ("बनूंगा", "बनूंगी"), ("बनूँगा", "बनूँगी"),
    ("चलूंगा", "चलूंगी"), ("चलूँगा", "चलूँगी"),
    ("बोलूंगा", "बोलूंगी"), ("बोलूँगा", "बोलूँगी"),
    ("हूंगा", "हूंगी"), ("हूँगा", "हूँगी"),
]

FEMALE_TO_MALE = [(f, m) for m, f in MALE_TO_FEMALE]
# Asymmetric grammar fix (impersonal "lagta hai" construction)
FEMALE_TO_MALE.extend([
    ("लगती है", "लगता है"),
    ("लगती हूँ", "लगता हूँ"),
    ("लगती थी", "लगता था"),
])

_SORTED_M2F = sorted(MALE_TO_FEMALE, key=lambda x: -len(x[0]))
_SORTED_F2M = sorted(FEMALE_TO_MALE, key=lambda x: -len(x[0]))

_AMBIGUOUS = {"की", "ली", "दी", "किया", "लिया", "दिया"}
_AUXILIARIES = {"है", "था", "थी", "हूँ", "हैं", "थे", "थीं", "हो", "गी", "गा"}
_TENSE_TOKENS = {"हैं", "है", "हूँ", "था", "थी", "थे", "थीं", "हो", "चाहता", "चाहती",
                  "चाहते", "जीता", "जीती", "जीते", "सकता", "सकती", "सकते", "करता",
                  "करती", "करते", "कहता", "कहती", "कहते", "रहता", "रहती", "रहते"}
_PROTECTED_TOKENS = {"हैं", "है", "हूँ", "हो", "होना", "रहें", "रहो", "चाहता", "चाहती",
                      "चाहते", "सकता", "सकती", "सकते", "करता", "करती", "करते", "कहता",
                      "कहती", "कहते", "रहता", "रहती", "रहते"}

MALE_TOK = {"रहा", "गया", "था", "आया", "खाया", "किया", "होगा", "मिला", "सका",
            "बोला", "चला", "थका", "बैठा", "उठा", "देखा", "सुना", "पाया", "लिया", "दिया",
            "जाता", "आता", "करता", "पढ़ता", "खाता", "बोलता", "देखता", "सुनता"}
FEMALE_TOK = {"रही", "गई", "थी", "आई", "खाई", "होगी", "मिली", "सकी",
              "बोली", "चली", "थकी", "बैठी", "उठी", "देखी", "सुनी", "पाई", "ली", "दी",
              "जाती", "आती", "करती", "पढ़ती", "खाती", "बोलती", "देखती", "सुनती"}
PLURAL_TOK = {"थे", "थीं", "हैं", "हो", "गए", "आए", "किए"}


def _apply_rules(hi: str, target_gender: int) -> str:
    rules = _SORTED_M2F if target_gender == 1 else _SORTED_F2M
    tokens = hi.split()
    for src, tgt in rules:
        if src not in hi:
            continue
        if src in _AMBIGUOUS:
            replaced = False
            for i, tok in enumerate(tokens):
                if tok != src:
                    continue
                next_tok = tokens[i + 1] if i + 1 < len(tokens) else ""
                if next_tok in _AUXILIARIES or next_tok in {"", "।", "?", "!"}:
                    tokens[i] = tgt
                    replaced = True
                    break
            if replaced:
                hi = clean_hindi(" ".join(tokens))
                tokens = hi.split()
            continue
        hi = clean_hindi(hi.replace(src, tgt))
        tokens = hi.split()
    return hi


def apply_morphology(hi: str, target_gender: int, morph_tokens: dict = None,
                      dominant_controller: str = "", person: str = "") -> str:
    if target_gender not in (0, 1):
        return hi
    if dominant_controller not in ("subject", "speaker"):
        return hi

    if person != "first" and morph_tokens is not None:
        has_male = len(morph_tokens.get("male_tokens", [])) > 0
        has_female = len(morph_tokens.get("female_tokens", [])) > 0
        if not has_male and not has_female:
            extended_male = MALE_TOK & set(hi.split())
            extended_female = FEMALE_TOK & set(hi.split())
            if not extended_male and not extended_female:
                return hi

    result = _apply_rules(hi, target_gender)

    hi_toks, result_toks = hi.split(), result.split()
    if len(hi_toks) == len(result_toks):
        for i, (orig, new) in enumerate(zip(hi_toks, result_toks)):
            if orig in _PROTECTED_TOKENS and new != orig:
                result_toks[i] = orig
        result = " ".join(result_toks)

    return clean_hindi(result)


# ---------------------------------------------------------------------------
# Morphology token counting (used both for correction gating AND as your
# automatic gender-accuracy evaluation metric)
# ---------------------------------------------------------------------------
VALID_MALE = {"रहा", "था", "गया", "करता", "बैठा", "सोया", "आया", "जाता", "आता",
              "रहता", "लगता", "मिलता", "चलता", "बोलता", "देखता", "सुनता", "खाता",
              "पढ़ता", "सोता", "समझता", "मुड़ता", "जानता", "रखता", "पाता", "सकता",
              "चाहता", "हो गया", "मानता", "सोचता", "उठता"}
VALID_FEMALE = {"रही", "थी", "गई", "करती", "बैठी", "सोई", "आई", "जाती", "आती",
                "रहती", "लगती", "मिलती", "चलती", "बोलती", "देखती", "सुनती", "खाती",
                "पढ़ती", "सोती", "समझती", "मुड़ती", "जानती", "रखती", "पाती", "सकती",
                "चाहती", "हो गई", "मानती", "सोचती", "उठती"}
MALE_BIGRAMS = {"पसंद था", "माना था", "जाता था", "रहता था", "लगता था", "करता था",
                "जाता है", "लगता है", "करता है", "माना जाता", "कहा जाता"}
FEMALE_BIGRAMS = {"पसंद थी", "मानी थी", "जाती थी", "रहती थी", "लगती थी", "करती थी",
                   "जाती है", "लगती है", "करती है", "मानी जाती", "कही जाती"}
MALE_TRIGRAMS = {"माना जाता था", "कहा जाता था", "देखा जाता था"}
FEMALE_TRIGRAMS = {"मानी जाती थी", "कही जाती थी", "देखी जाती थी"}


def count_morph_tokens(hi: str) -> dict:
    toks = hi.split()
    tok_set = set(toks)

    male_tokens = sorted(tok_set & VALID_MALE)
    female_tokens = sorted(tok_set & VALID_FEMALE)

    for i in range(len(toks) - 1):
        bigram = f"{toks[i]} {toks[i+1]}"
        if bigram in MALE_BIGRAMS and bigram not in male_tokens:
            male_tokens.append(bigram)
        if bigram in FEMALE_BIGRAMS and bigram not in female_tokens:
            female_tokens.append(bigram)

    for i in range(len(toks) - 2):
        trigram = f"{toks[i]} {toks[i+1]} {toks[i+2]}"
        if trigram in MALE_TRIGRAMS and trigram not in male_tokens:
            male_tokens.append(trigram)
        if trigram in FEMALE_TRIGRAMS and trigram not in female_tokens:
            female_tokens.append(trigram)

    if len(female_tokens) > len(male_tokens):
        dominant = "female"
    elif len(male_tokens) > len(female_tokens):
        dominant = "male"
    else:
        dominant = "neutral"

    return {"male_tokens": male_tokens, "female_tokens": female_tokens, "dominant": dominant}


def morph_match(info: dict, target_gender: int) -> bool:
    return info["dominant"] in ("neutral", "female" if target_gender == 1 else "male")
