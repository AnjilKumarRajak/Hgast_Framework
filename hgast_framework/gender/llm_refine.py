
import re
import string
import logging
from typing import Callable, Optional

from .morphology_rules import clean_hindi, count_morph_tokens, MALE_TOK, FEMALE_TOK, PLURAL_TOK
from .morphology_rules import _AMBIGUOUS, _TENSE_TOKENS, _PROTECTED_TOKENS

log = logging.getLogger(__name__)

ChatFn = Callable[[str], str]  # takes a prompt string, returns raw model text

# Ergative-construction guard 
_ERGATIVE_MARKERS = {"ने", "मैंने", "हमने"}
_ERGATIVE_OBJECTS = {"खाना", "नोट", "पत्र", "काम", "घर", "समय", "सफर", "मौका"}


def _is_ergative(hi: str) -> bool:
    toks = set(hi.split())
    if toks & _ERGATIVE_MARKERS:
        return any(obj in hi for obj in _ERGATIVE_OBJECTS)
    return False


IMPERSONAL_FIXES = {
    "मुझे लगती है": "मुझे लगता है", "मुझे लगती हूँ": "मुझे लगता है",
    "मुझे लगती थी": "मुझे लगता था", "उसे लगती है": "उसे लगता है",
    "उसे लगती थी": "उसे लगता था", "हमें लगती है": "हमें लगता है",
    "हमें लगती थी": "हमें लगता था", "तुम्हें लगती है": "तुम्हें लगता है",
    "तुम्हें लगती थी": "तुम्हें लगता था",
}


_FEW_SHOT_EXAMPLES = """1. English: I am going home.
Base Hindi: मैं घर जा रहा हूँ।
Target: feminine (स्त्रीलिंग), first person
Correct: मैं घर जा रही हूँ। (Keep continuous tense, change रहा to रही)

2. English: He is revered in Uzbekistan too, which he had visited.
Base Hindi: उज़्बेकिस्तान में भी उनका सम्मान किया जाता है जहाँ वे गए थे।
Target: masculine (पुल्लिंग), third person
Correct: उज़्बेकिस्तान में भी उनका सम्मान किया जाता है जहाँ वे गए थे। (Preserve polite plural 'उनका' and 'वे गए थे')

3. English: I am exhausted, hungry and I just want to sleep.
Base Hindi: मैं थक गया हूँ - भूखी हूँ और मैं सोना चाहता हूँ।
Target: masculine (पुल्लिंग), first person
Correct: मैं थक गया हूँ, भूखा हूँ और मैं सोना चाहता हूँ। (Change female भूखी to male भूखा, remove unnatural dashes)

4. English: You are eating.
Base Hindi: तुम खा रहे हो।
Target: feminine (स्त्रीलिंग), second person
Correct: तुम खा रही हो। (Change रहे to रही)

5. English: I am being followed and I am so terrified that I am running as fast as I can to get home safely.
Base Hindi: मेरा पीछा किया जा रहा है और मैं इतना भयभीत हूं कि मैं सुरक्षित रूप से घर पहुंचने के लिए जितनी तेजी से दौड़ रहा हूं
Target: feminine (स्त्रीलिंग), first person
Correct: मेरा पीछा किया जा रहा है और मैं इतनी भयभीत हूँ कि मैं घर सुरक्षित पहुँचने के लिए जितनी तेजी से दौड़ रही हूँ। (Verb 'पीछा किया जा रहा है' MUST stay masculine because 'पीछा' is a masculine noun. Only change adjectives/verbs related to the speaker like 'इतनी भयभीत', 'दौड़ रही हूँ')

6. English: I was sitting alone in the dark room feeling incredibly lonely, hungry and exhausted waiting for someone to arrive.
Base Hindi: मैं अंधेरे कमरे में अकेले बैठी थी. मैं किसी के आने का इंतजार कर रही थी। मैं अविश्वसनीय रूप से अकेला महसूस कर रही थी।
Target: masculine (पुल्लिंग), first person
Correct: मैं अंधेरे कमरे में अकेला बैठा था। मैं किसी के आने का इंतजार कर रहा था। मैं अविश्वसनीय रूप से अकेला महसूस कर रहा था। (Change all female verbs/adjectives to male, and fix English periods '.' to Hindi full stops '।')"""


def build_prompt(en: str, hi_draft: str, target_gender: int, person: str) -> str:
    gender_label = "feminine (स्त्रीलिंग)" if target_gender == 1 else "masculine (पुल्लिंग)"
    person_label = {
        "first": "first person (मैं/हम)",
        "second": "second person (तुम/आप)",
        "third": "third person (वह/वे)",
    }.get(person, "third person")

    return f"""You are an expert Hindi linguist. Rewrite the Hindi sentence to match the target gender perfectly, while STRICTLY preserving the original meaning, verb tense, and polite/plural markers.

TARGET GENDER: {gender_label}
PERSON: {person_label}

EXAMPLES:
{_FEW_SHOT_EXAMPLES}

Now, process this input:
English: {en}
Base Hindi: {hi_draft}
Target: {gender_label}, {person_label}
Return ONLY the grammatically perfect Hindi sentence. Do not include quotes, prefixes like 'Correct:', or any explanations."""


class LLMGenderRefiner:
    """
    Model-agnostic wrapper. Construct with any chat function:

        refiner = LLMGenderRefiner(chat_fn=chat_fn)

    chat_fn signature: (prompt: str) -> str   (raw model text response)
    """

    def __init__(self, chat_fn: Optional[ChatFn] = None):
        self.chat_fn = chat_fn

    @property
    def available(self) -> bool:
        return self.chat_fn is not None

    def refine(self, en: str, hi_draft: str, target_gender: int, person: str) -> str:
        if not self.available:
            return hi_draft
        if _is_ergative(hi_draft):
            log.debug("Skipping LLM refine: ergative construction guard.")
            return hi_draft

        prompt = build_prompt(en, hi_draft, target_gender, person)
        try:
            raw = self.chat_fn(prompt).strip()
        except Exception as exc:
            log.warning(f"LLM refine call failed: {exc}")
            return hi_draft

        for prefix in ("fluent hindi:", "hindi:", "output:", "result:", "answer:", "correct:"):
            if raw.lower().startswith(prefix):
                raw = raw[len(prefix):].strip()
        result = clean_hindi(raw)

        expected = "female" if target_gender == 1 else "male"

        # ---- Safety validation gates ----
        if len(result.split()) < 3:
            return hi_draft

        ascii_ratio = sum(c.isascii() and c.isalpha() for c in result) / max(len(result), 1)
        if ascii_ratio > 0.12:
            return hi_draft

        if len(result.split()) < len(hi_draft.split()) * 0.6:
            return hi_draft

        draft_toks, result_toks = set(hi_draft.split()), set(result.split())
        overlap = len(draft_toks & result_toks) / max(len(draft_toks), 1)
        min_overlap = 0.20 if len(hi_draft.split()) <= 6 else 0.30
        if overlap < min_overlap:
            log.warning(f"LLM refine too divergent (overlap={overlap:.2f}); reverting.")
            return hi_draft

        def _dashes(text):
            return set(c for c in text if c in "—–-")
        if not _dashes(result).issubset(_dashes(hi_draft)):
            log.warning("LLM refine hallucinated dashes; reverting.")
            return hi_draft

        post_morph = count_morph_tokens(result)
        if post_morph["dominant"] not in ("neutral", expected):
            log.warning(
                f"LLM refine broke morphology ({post_morph['dominant']} != {expected}); reverting."
            )
            return hi_draft

        def _strip_punct(t):
            for p in string.punctuation + "।,?!—–-":
                t = t.replace(p, "")
            return t

        draft_clean = set(_strip_punct(hi_draft).split())
        result_clean = set(_strip_punct(result).split())
        allowed_new = (MALE_TOK | FEMALE_TOK | PLURAL_TOK | _AMBIGUOUS | _TENSE_TOKENS
                       | _PROTECTED_TOKENS | {"हैं", "है", "हूँ", "था", "थी", "थे", "थीं",
                       "हो", "गी", "गा", "गए", "गई", "गया", "गे", "रहा", "रही", "रहे",
                       "हुआ", "हुई", "हुए", "चुका", "चुकी", "चुके"})
        new_words = result_clean - draft_clean
        dropped_words = draft_clean - result_clean

        VOWEL_ENDINGS = ("ा", "ी", "े", "ें", "ों", "ू")
        truly_new = set()
        for nw in new_words:
            if nw in allowed_new:
                continue
            nw_stem = nw[:-1] if nw.endswith(VOWEL_ENDINGS) else nw
            is_inflection = any(
                (dw[:-1] if dw.endswith(VOWEL_ENDINGS) else dw) == nw_stem
                for dw in dropped_words
            )
            if not is_inflection:
                truly_new.add(nw)

        if truly_new:
            log.warning(f"LLM refine hallucinated words {truly_new}; reverting.")
            return hi_draft

        for bad, good in IMPERSONAL_FIXES.items():
            if bad in result:
                result = result.replace(bad, good)

        if result != hi_draft:
            log.info(f"LLM REFINE  BEFORE: {hi_draft}\n            AFTER : {result}")

        return result
