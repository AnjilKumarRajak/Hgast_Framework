

import re

# Text cleanup
def clean_gujarati(text: str) -> str:
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[ ]+([?.!,।])", r"\1", text)
    return text.strip()

MALE_TO_FEMALE_PAST = [
    ("ગયો", "ગઈ"),        # went
    ("આવ્યો", "આવી"),      # came
    ("બેઠો", "બેઠી"),      # sat
    ("ઊઠ્યો", "ઊઠી"),      # got up
    ("સૂતો", "સૂતી"),      # slept
    ("પડ્યો", "પડી"),      # fell
    ("હસ્યો", "હસી"),      # laughed
    ("રડ્યો", "રડી"),      # cried
    ("દોડ્યો", "દોડી"),    # ran
    ("બોલ્યો", "બોલી"),    # spoke
    ("થાક્યો", "થાકી"),    # got tired
    ("ચાલ્યો", "ચાલી"),    # walked
    ("પહોંચ્યો", "પહોંચી"), # arrived
    ("નીકળ્યો", "નીકળી"),  # departed/left
    ("ગભરાયો", "ગભરાઈ"),   # got scared
    ("મળ્યો", "મળી"),      # met
    ("થયો", "થઈ"),        # became/happened
    ("રહ્યો", "રહી"),      # stayed/remained
    ("શક્યો", "શકી"),      # could (past of modal}
]
MALE_TO_FEMALE_CONTINUOUS = [
    ("જતો", "જતી"),        # going
    ("આવતો", "આવતી"),      # coming
    ("કરતો", "કરતી"),      # doing
    ("ખાતો", "ખાતી"),      # eating
    ("પીતો", "પીતી"),      # drinking
    ("બોલતો", "બોલતી"),    # speaking
    ("બેસતો", "બેસતી"),    # sitting
    ("સૂતો", "સૂતી"),      # sleeping 
    ("હસતો", "હસતી"),      # laughing
    ("દોડતો", "દોડતી"),    # running
    ("વાંચતો", "વાંચતી"),  # reading
    ("લખતો", "લખતી"),      # writing
    ("જોતો", "જોતી"),      # seeing/watching
    ("સાંભળતો", "સાંભળતી"), # hearing/listening
    ("સમજતો", "સમજતી"),    # understanding
    ("રહેતો", "રહેતી"),    # staying/living
]

# ADJECTIVES (-o/-ī/-ũ declension, subject/person independent)
MALE_TO_FEMALE_ADJECTIVES = [
    ("એકલો", "એકલી"),      # alone
    ("સારો", "સારી"),      # good
    ("મોટો", "મોટી"),      # big
    ("નાનો", "નાની"),      # small
    ("થાકેલો", "થાકેલી"),  # tired
    ("ભૂખ્યો", "ભૂખી"),    # hungry
    ("ગભરાયેલો", "ગભરાયેલી"), # scared
]

MALE_TO_FEMALE: list[tuple[str, str]] = (
    MALE_TO_FEMALE_PAST + MALE_TO_FEMALE_CONTINUOUS + MALE_TO_FEMALE_ADJECTIVES
)
FEMALE_TO_MALE: list[tuple[str, str]] = [(f, m) for m, f in MALE_TO_FEMALE]

_SORTED_M2F = sorted(MALE_TO_FEMALE, key=lambda x: -len(x[0]))
_SORTED_F2M = sorted(FEMALE_TO_MALE, key=lambda x: -len(x[0]))

_AMBIGUOUS = {"સૂતો", "સૂતી"}

_TENSE_TOKENS = {
    "છું", "છે", "છો", "છીએ", "હતો", "હતી", "હતું", "હતા",
    "હશે", "થશે", "જશે", "કરશે", "નથી",
}
_PROTECTED_TOKENS = {"છું", "છે", "છો", "છીએ", "નથી"}


def count_morph_tokens_gu(gu: str) -> dict:

    toks = gu.split()
    tok_set = set(toks)
    VALID_MALE = {m for m, f in MALE_TO_FEMALE}
    VALID_FEMALE = {f for m, f in MALE_TO_FEMALE}

    male = len(tok_set & VALID_MALE)
    female = len(tok_set & VALID_FEMALE)
    ambiguous = len(tok_set & _AMBIGUOUS)

    if male > female:
        dominant = "male"
    elif female > male:
        dominant = "female"
    else:
        dominant = "neutral"

    return {"male": male, "female": female, "ambiguous": ambiguous, "dominant": dominant}


MALE_TOK = {m for m, f in MALE_TO_FEMALE}
FEMALE_TOK = {f for m, f in MALE_TO_FEMALE}
PLURAL_TOK = {"હતા", "છીએ", "ગયા", "આવ્યા", "કર્યા"}


_ERGATIVE_PRONOUNS = {"મેં", "તેં", "એણે", "તેણે", "અમે", "તમે"}
_ERGATIVE_OBJECTS = {
    "પત્ર", "જમવાનું", "કામ", "નોંધ", "સમય", "પ્રવાસ", "તક", "ઘર", "પુસ્તક", "ગીત",
}


def is_ergative_gu(gu: str) -> bool:
    toks = [t.strip("।.,!?\"'") for t in gu.split()]
    tok_set = set(toks)
    has_ergative_pronoun = bool(_ERGATIVE_PRONOUNS & tok_set)
    has_ne_suffix_noun = any(t.endswith("એ") and len(t) > 2 for t in toks)
    if has_ergative_pronoun or has_ne_suffix_noun:
        return bool(_ERGATIVE_OBJECTS & tok_set)
    return False

def _apply_rules_gu(gu: str, target_gender: int) -> str:
    rules = _SORTED_M2F if target_gender == 1 else _SORTED_F2M
    for src, tgt in rules:
        if src in gu:
            gu = clean_gujarati(gu.replace(src, tgt))
    return gu

def apply_morphology_gu(gu: str, target_gender: int, dominant_controller: str = "",
                         person: str = "") -> str:
    if target_gender not in (0, 1):
        return gu
    if dominant_controller not in ("subject", "speaker"):
        return gu
    return _apply_rules_gu(gu, target_gender)

def morph_match_gu(morph_info: dict, target_gender: int) -> bool:
    """Mirrors morph_match_mr for framework compatibility."""
    expected = "female" if target_gender == 1 else "male"
    return morph_info["dominant"] in ("neutral", expected)
