"""
gender/linguistic_analysis.py
==============================
Subject detection, coreference resolution, and name-gender inference.
This is a cleaned version of the original parse_subject / resolve_coreference
/ gender_from_antecedent / infer_gender logic. Fixes applied:

  - nlp_spacy, coref_model, gender_detector are now lazily loaded module-level
    singletons instead of undefined free variables.
  - All imports (re) are explicit.
  - KNOWN_MALE/KNOWN_FEMALE lookup lists kept (documented limitation: finite
    coverage -> falls back to a statistical name-gender model, then to
    "unknown" if that also fails -> logged, not silently guessed).
"""

import re
import logging

log = logging.getLogger(__name__)

FEM_PRONOUNS = {"she", "her", "hers", "herself"}
MASC_PRONOUNS = {"he", "him", "his", "himself"}

# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------
_nlp_spacy = None
_coref_model = None
_gender_name_detector = None


def get_spacy():
    global _nlp_spacy
    if _nlp_spacy is None:
        import spacy
        try:
            _nlp_spacy = spacy.load("en_core_web_trf")
        except OSError:
            log.warning("en_core_web_trf not found, falling back to en_core_web_sm")
            _nlp_spacy = spacy.load("en_core_web_sm")
    return _nlp_spacy


def get_coref_model():
    global _coref_model
    if _coref_model is None:
        try:
            import fastcoref
            _coref_model = fastcoref.FCoref()
        except Exception as exc:
            log.warning(f"fastcoref unavailable ({exc}); coreference disabled.")
            _coref_model = False  # sentinel meaning "tried and failed"
    return _coref_model or None


def get_name_gender_detector():
    global _gender_name_detector
    if _gender_name_detector is None:
        try:
            import gender_guesser.detector as gg
            _gender_name_detector = gg.Detector(case_sensitive=False)
        except Exception as exc:
            log.warning(f"gender_guesser unavailable ({exc}); name-gender inference disabled.")
            _gender_name_detector = False
    return _gender_name_detector or None


# ---------------------------------------------------------------------------
# Name lookup lists (documented limitation: finite coverage)
# ---------------------------------------------------------------------------
KNOWN_MALE = {
    "john", "alex", "daniel", "samuel", "thomas", "barack", "obama", "trump",
    "michael", "david", "james", "robert", "william", "charles", "george",
    "raj", "ravi", "arjun", "vikram", "amit", "suresh", "rahul",
}
KNOWN_FEMALE = {
    "mary", "anna", "sophia", "olivia", "ava", "isabella", "sarah", "elizabeth",
    "michelle", "angelina", "emma", "jennifer", "jane", "alice", "margaret",
    "helen", "laura", "emily", "priya", "kavya", "ananya", "divya", "pooja",
    "shreya", "neha",
}


def infer_gender(name: str) -> str:
    """Returns 'male' | 'female' | 'unknown'. Order: hardcoded list -> stats model."""
    first = name.split()[0].lower() if name.strip() else ""
    if first in KNOWN_MALE:
        return "male"
    if first in KNOWN_FEMALE:
        return "female"

    detector = get_name_gender_detector()
    if detector is None:
        return "unknown"
    try:
        g = detector.get_gender(name.split()[0].capitalize())
        if g in ("male", "mostly_male"):
            return "male"
        if g in ("female", "mostly_female"):
            return "female"
        return "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Coreference
# ---------------------------------------------------------------------------
def resolve_coreference(en: str) -> dict:
    model = get_coref_model()
    if model is None:
        return {}
    try:
        preds = model.predict(texts=[en])
        if isinstance(preds, list):
            preds = preds[0]
        clusters = preds.get_clusters(as_strings=True)
        coref_map = {}
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            antecedent = cluster[0]
            for mention in cluster[1:]:
                ml = mention.lower().strip()
                if ml in FEM_PRONOUNS | MASC_PRONOUNS | {"they", "them"}:
                    coref_map[ml] = antecedent
        return coref_map
    except Exception as exc:
        log.debug(f"Coreference error: {exc}")
        return {}


# ---------------------------------------------------------------------------
# Antecedent -> gender
# ---------------------------------------------------------------------------
MALE_WORDS = {"he", "him", "his", "himself", "boy", "man", "father", "brother",
              "husband", "king", "prince", "gentleman", "sir", "actor",
              "boyfriend", "uncle", "nephew", "groom", "widower", "mr",
              "monk", "waiter", "schoolboy", "policeman", "businessman", "chairman"}
FEMALE_WORDS = {"she", "her", "hers", "herself", "girl", "woman", "mother",
                 "sister", "wife", "queen", "princess", "lady", "madam",
                 "actress", "aunt", "niece", "bride", "widow", "girlfriend",
                 "mrs", "miss", "nun", "waitress", "schoolgirl", "nurse",
                 "businesswoman", "chairwoman"}
NEUTRAL_HUMAN_WORDS = {"teacher", "doctor", "student", "friend", "officer",
                        "driver", "farmer", "worker", "person", "child"}
NON_HUMAN_WORDS = {"book", "table", "chair", "car", "train", "bus", "computer",
                    "machine", "building", "school", "city", "village",
                    "story", "event", "theatre", "regiment"}


def gender_from_antecedent(antecedent: str) -> str:
    a = str(antecedent).lower().strip()
    words = set(re.findall(r"\b\w+\b", a))

    if words & MALE_WORDS:
        return "male"
    if words & FEMALE_WORDS:
        return "female"

    for name in KNOWN_MALE:
        if name in words:
            return "male"
    for name in KNOWN_FEMALE:
        if name in words:
            return "female"

    if words & NEUTRAL_HUMAN_WORDS:
        return "neutral"
    if words & NON_HUMAN_WORDS:
        return "neutral"
    return "neutral"


# ---------------------------------------------------------------------------
# Subject parsing
# ---------------------------------------------------------------------------
POSSESSIVES = {"my", "our"}
THIRD_PERSON_NOUNS = {
    "sister", "mother", "mom", "mum", "daughter", "wife", "aunt", "grandmother",
    "niece", "girl", "woman", "lady", "brother", "father", "dad", "son",
    "husband", "uncle", "grandfather", "nephew", "boy", "man", "guy",
    "teacher", "doctor", "student", "boss", "colleague", "partner",
    "team", "project", "idea", "work",
}
THIRD_GENDER_MAP = {
    "sister": "female", "mother": "female", "mom": "female", "mum": "female",
    "daughter": "female", "wife": "female", "aunt": "female",
    "grandmother": "female", "niece": "female", "girl": "female",
    "woman": "female", "lady": "female", "brother": "male", "father": "male",
    "dad": "male", "son": "male", "husband": "male", "uncle": "male",
    "grandfather": "male", "nephew": "male", "boy": "male", "man": "male",
    "guy": "male",
}
NON_SUBJECT_STARTERS = {"of", "for", "in", "on", "with", "at", "from", "about"}
FAST_PATH_MAP = {
    "i": ("first", "unknown", 0.99),
    "we": ("first", "unknown", 0.99),
    "you": ("second", "neutral", 0.99),
    "he": ("third", "male", 0.99),
    "she": ("third", "female", 0.99),
    "they": ("third", "unknown", 0.95),
}
GENDERED_NOUNS = {
    "woman": "female", "girl": "female", "lady": "female", "mother": "female",
    "sister": "female", "wife": "female", "queen": "female", "daughter": "female",
    "aunt": "female", "man": "male", "boy": "male", "gentleman": "male",
    "father": "male", "brother": "male", "husband": "male", "king": "male",
    "son": "male", "uncle": "male",
}

_DEFAULT_SUBJECT = {
    "subject": "", "subject_pos": "", "subject_gender": "unknown",
    "person": "third", "confidence": 0.0,
}


def parse_subject(en: str, coref_map: dict) -> dict:
    words = en.strip().split()
    if not words:
        return dict(_DEFAULT_SUBJECT)

    first = words[0].lower()
    second = words[1].lower() if len(words) > 1 else ""

    # Possessive guard: "my sister ..." -> third person, not first
    if first in POSSESSIVES and second in THIRD_PERSON_NOUNS:
        return {
            "subject": f"{words[0]} {words[1]}",
            "subject_pos": "NOUN",
            "subject_gender": THIRD_GENDER_MAP.get(second, "unknown"),
            "person": "third",
            "confidence": 0.95,
        }

    if first in NON_SUBJECT_STARTERS:
        lower_words = [w.lower() for w in words]
        if "i" in lower_words:
            return {"subject": "I", "subject_pos": "PRON", "subject_gender": "unknown",
                    "person": "first", "confidence": 0.85}
        return dict(_DEFAULT_SUBJECT)

    if first in FAST_PATH_MAP:
        person, sgender, conf = FAST_PATH_MAP[first]
        return {"subject": words[0], "subject_pos": "PRON", "subject_gender": sgender,
                "person": person, "confidence": conf}

    if first in ("the", "a", "an") and second in GENDERED_NOUNS:
        return {
            "subject": f"{words[0]} {words[1]}", "subject_pos": "NOUN",
            "subject_gender": GENDERED_NOUNS[second], "person": "third",
            "confidence": 0.95,
        }

    # spaCy deep parse
    try:
        nlp = get_spacy()
        doc = nlp(en)

        for tok in doc:
            if tok.lemma_.lower() in ("i", "me", "mine", "myself", "we", "us", "ourselves"):
                return {"subject": tok.text, "subject_pos": tok.pos_,
                        "subject_gender": "unknown", "person": "first", "confidence": 0.98}

        strong_pronouns = {
            "he": ("male", "third"), "him": ("male", "third"), "his": ("male", "third"),
            "she": ("female", "third"), "her": ("female", "third"), "hers": ("female", "third"),
            "they": ("unknown", "third"), "them": ("unknown", "third"),
        }
        for tok in doc:
            lemma = tok.lemma_.lower()
            if lemma in strong_pronouns:
                gender, person = strong_pronouns[lemma]
                return {"subject": tok.text, "subject_pos": tok.pos_,
                        "subject_gender": gender, "person": person, "confidence": 0.95}

        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return {"subject": ent.text, "subject_pos": "PROPN",
                        "subject_gender": infer_gender(ent.text.lower()),
                        "person": "third", "confidence": 0.85}

        for tok in doc:
            if tok.lemma_.lower() in ("you", "your", "yours", "yourself", "yourselves"):
                return {"subject": tok.text, "subject_pos": tok.pos_,
                        "subject_gender": "neutral", "person": "second", "confidence": 0.95}

        person_entity = next((ent.text for ent in doc.ents if ent.label_ == "PERSON"), None)

        for tok in doc:
            if tok.pos_ not in ("PRON", "NOUN", "PROPN"):
                continue
            if tok.text.isupper() or tok.like_num:
                continue
            if tok.dep_ not in ("nsubj", "nsubjpass", "csubj"):
                continue

            lemma = tok.lemma_.lower()
            if lemma in ("i", "me", "myself", "we", "us"):
                person = "first"
            elif lemma in ("you", "yourself", "yourselves"):
                person = "second"
            else:
                person = "third"

            resolved = tok.text
            for ent in doc.ents:
                if tok.text.lower() in {t.text.lower() for t in ent}:
                    resolved = ent.text
                    break
            if person == "third" and person_entity:
                resolved = person_entity
            resolved = coref_map.get(lemma, resolved)

            if lemma in FEM_PRONOUNS | MASC_PRONOUNS:
                subj_g = gender_from_antecedent(resolved)
            else:
                morph = tok.morph.to_dict()
                if morph.get("Gender") == "Fem":
                    subj_g = "female"
                elif morph.get("Gender") == "Masc":
                    subj_g = "male"
                else:
                    subj_g = gender_from_antecedent(resolved)

            return {"subject": resolved, "subject_pos": tok.pos_,
                    "subject_gender": subj_g, "person": person, "confidence": 0.70}

    except Exception as exc:
        log.debug(f"spaCy parse error: {exc}")

    # Keyword fallback (spaCy unavailable or failed)
    el = f" {en.lower()} "
    if any(w in el for w in (" i ", " i'm ", " i was ", " i've ", " i am ",
                              " i will ", " i had ", " me ", " myself ")):
        return {**_DEFAULT_SUBJECT, "person": "first", "subject_gender": "unknown"}
    if any(w in el for w in (" you ", " you're ", " your ")):
        return {**_DEFAULT_SUBJECT, "person": "second", "subject_gender": "neutral"}
    for pronoun, default_g in (("she", "female"), ("her", "female")):
        if f" {pronoun} " in el:
            resolved = coref_map.get(pronoun, pronoun)
            g = gender_from_antecedent(resolved) if resolved != pronoun else default_g
            return {**_DEFAULT_SUBJECT, "person": "third", "subject_gender": g}
    for pronoun, default_g in (("he", "male"), ("him", "male")):
        if f" {pronoun} " in el:
            resolved = coref_map.get(pronoun, pronoun)
            g = gender_from_antecedent(resolved) if resolved != pronoun else default_g
            return {**_DEFAULT_SUBJECT, "person": "third", "subject_gender": g}

    return dict(_DEFAULT_SUBJECT)
