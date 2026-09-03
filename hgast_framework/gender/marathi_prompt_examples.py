

MARATHI_FEW_SHOT_EXAMPLES = """1. English: I am going home.
Base Marathi: मी घरी जातो.
Target: feminine (स्त्रीलिंग), first person
Correct: मी घरी जाते. (Change जातो to जाते - habitual present is gender-marked in Marathi)

2. English: I went home yesterday.
Base Marathi: मी काल घरी गेलो.
Target: feminine (स्त्रीलिंग), first person
Correct: मी काल घरी गेले. (First-person past uses गेलो/गेले, NOT गेला/गेली - that pattern is for third person)

3. English: He wrote a letter.
Base Marathi: त्याने पत्र लिहिले.
Target: feminine (स्त्रीलिंग), third person
Correct: त्याने पत्र लिहिले. (Do NOT change this - "लिहिले" agrees with the object "पत्र" (neuter), not with the agent "त्याने", because of ergative object-agreement in perfective transitive constructions. This sentence is ALREADY correct regardless of the subject's gender - never force a change here.)

4. English: You are eating.
Base Marathi: तू खातोस.
Target: feminine (स्त्रीलिंग), second person
Correct: तू खातेस. (Change खातोस to खातेस)

5. English: My brother is going to watch a movie and my sister is going to her dance class.
Base Marathi: माझा भाऊ सिनेमा बघायला जातो आणि माझी बहीण तिच्या डान्स क्लासला जाते.
Target: masculine (पुल्लिंग), third person
Correct: माझा भाऊ सिनेमा बघायला जातो आणि माझी बहीण तिच्या डान्स क्लासला जाते. (Do NOT touch "माझी बहीण ... जाते" - the sister's verb stays feminine regardless of the target speaker's gender. Only sentences with a single ambiguous subject should be corrected.)

6. English: I am exhausted and I just want to sleep.
Base Marathi: मी थकलो आहे आणि मला झोपायचे आहे.
Target: feminine (स्त्रीलिंग), first person
Correct: मी थकले आहे आणि मला झोपायचे आहे. (Change थकलो to थकले; "मला झोपायचे आहे" is an invariant impersonal construction and should NOT be changed)

7. English: She can do this work.
Base Marathi: ती हे काम करू शकतो.
Target: feminine (स्त्रीलिंग), third person
Correct: ती हे काम करू शकते. (Change शकतो to शकते - modal 'can' is gender-marked)"""


def build_prompt_mr(en: str, mr_draft: str, target_gender: int, person: str) -> str:
    gender_label = "feminine (स्त्रीलिंग)" if target_gender == 1 else "masculine (पुल्लिंग)"
    person_label = {
        "first": "first person (मी/आम्ही)",
        "second": "second person (तू/तुम्ही)",
        "third": "third person (तो/ती)",
    }.get(person, "third person")

    return f"""You are an expert Marathi linguist. Rewrite the Marathi sentence to ensure the grammar is completely correct for the given target gender and person.
Crucially, you must FIX any grammatical errors where the wrong gendered verb form was applied to the subject, WITHOUT breaking cases where the verb correctly agrees with something else.

TARGET GENDER: {gender_label}
PERSON: {person_label}

CRITICAL RULES:
1. STRICTLY preserve the original meaning, tense, and person/number markers.
2. DO NOT change 3rd person subjects other than the main one being targeted (e.g. 'my sister', 'my brother') - only correct the sentence's own primary subject.
3. WARNING: Marathi's present CONTINUOUS ("जात आहे" style) and FUTURE tense ("जाईन" style) are generally NOT gender-marked - do not invent a gender change there if none is grammatically required.
4. WARNING: In ergative/perfective-transitive constructions (agent marked with "-ने" suffix, e.g. "त्याने", "तिने", "रामाने"), the verb may agree with the OBJECT, not the agent - do not force a change if the sentence is already correct due to object agreement.
5. First-person past tense uses DIFFERENT endings (गेलो/गेले) than third-person past tense (गेला/गेली) for the same verb - do not confuse the two patterns.

EXAMPLES:
{MARATHI_FEW_SHOT_EXAMPLES}

Now, process this input:
English: {en}
Base Marathi: {mr_draft}
Target: {gender_label}, {person_label}
Return ONLY the grammatically perfect Marathi sentence. Do not include quotes, prefixes like 'Correct:', or any explanations."""
