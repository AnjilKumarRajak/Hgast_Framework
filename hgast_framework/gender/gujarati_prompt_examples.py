
_FEW_SHOT_EXAMPLES_GU = """1. English: I am going home.
Base Gujarati: તે ઘરે જાય છે.
Target: masculine (પુલ્લિંગ), first person
Correct: તે ઘરે જાય છે. (Simple present 'જાય છે' is gender-invariant in Gujarati -- nothing to change even though the target is masculine. Do NOT invent gender marking here.)

2. English: I went home and sat alone, exhausted.
Base Gujarati: હું ઘરે ગયો અને એકલો બેઠો, થાકેલો.
Target: feminine (સ્ત્રીલિંગ), first person
Correct: હું ઘરે ગઈ અને એકલી બેઠી, થાકેલી. (Past tense and adjectives DO inflect for gender in Gujarati: ગયો->ગઈ, બેઠો->બેઠી, એકલો->એકલી, થાકેલો->થાકેલી. Note past tense does NOT change with person -- ગયો/ગઈ is the same word regardless of whether the subject is "I", "you", or "he/she".)

3. English: I wrote a letter to my friend.
Base Gujarati: મેં મિત્રને પત્ર લખ્યો.
Target: feminine (સ્ત્રીલિંગ), first person
Correct: મેં મિત્રને પત્ર લખ્યો. (CRITICAL: in the ergative construction marked by 'મેં', the verb agrees with the OBJECT 'પત્ર' (masculine), not the speaker. Do NOT change લખ્યો to લખી just because the target speaker is feminine -- this is unrelated to the speaker's gender.)

4. English: She was tired and hungry, waiting for someone to arrive.
Base Gujarati: તે થાકેલી અને ભૂખી હતી, કોઈની રાહ જોતી હતી.
Target: masculine (પુલ્લિંગ), first person
Correct: તે થાકેલી અને ભૂખી હતી, કોઈની રાહ જોતી હતી. (Unchanged. This sentence is about a third-person 'તે' (she), not the target speaker -- her gender agreement must be preserved regardless of the target speaker's gender.)

5. English: My brother is going to watch a movie and my sister is going to her dance class.
Base Gujarati: મારો ભાઈ ફિલ્મ જોવા જાય છે અને મારી બહેન તેના ડાન્સ ક્લાસમાં જાય છે.
Target: masculine (પુલ્લિંગ), third person
Correct: મારો ભાઈ ફિલ્મ જોવા જાય છે અને મારી બહેન તેના ડાન્સ ક્લાસમાં જાય છે. (Unchanged. Crucial: do NOT swap મારી બહેન's feminine agreement to masculine just because the target speaker is male -- the sister's grammatical gender belongs to her.)

6. English: I was going to school every day.
Base Gujarati: હું રોજ શાળાએ જતો હતો.
Target: feminine (સ્ત્રીલિંગ), first person
Correct: હું રોજ શાળાએ જતી હતી. (Continuous/habitual participle 'જતો/જતી' DOES mark gender, unlike the simple present in example 1 -- this is the key distinction to get right in Gujarati: simple present is invariant, but the continuous/habitual participle inflects.)

7. English: We tested this by exposing the babies.
Base Gujarati: અમે બાળકોને ખુલ્લા પાડીને આનું પરીક્ષણ કર્યું.
Target: masculine (પુલ્લિંગ), first person
Correct: અમે બાળકોને ખુલ્લા પાડીને આનું પરીક્ષણ કર્યું. (Unchanged, and intentional: 'પરીક્ષણ' is grammatically neuter, so 'કર્યું' correctly agrees with the OBJECT, not with 'અમે'. Do NOT change કર્યું to કર્યો just because the target speaker is masculine.)"""


def build_prompt_gu(en: str, gu_draft: str, target_gender: int, person: str) -> str:
    gender_label = "feminine (સ્ત્રીલિંગ)" if target_gender == 1 else "masculine (પુલ્લિંગ)"
    person_label = {
        "first": "first person (હું/અમે)",
        "second": "second person (તું/તમે)",
        "third": "third person (તે/એ)",
    }.get(person, "third person")

    return f"""You are an expert Gujarati linguist. Rewrite the Gujarati sentence to ensure the grammar is completely correct for the given target gender and person.
Crucially, you must FIX any grammatical errors where the wrong gendered verb/adjective form was applied to the target speaker -- but you must NOT invent gender marking where Gujarati grammar doesn't have any.

TARGET GENDER: {gender_label}
PERSON: {person_label}

CRITICAL RULES:
1. STRICTLY preserve the original meaning, verb tense, and any polite/plural markers.
2. Gujarati SIMPLE present forms (જાય છે, કરે છે, ખાય છે) are GENDER-INVARIANT. Do not add or change gender marking on these.
3. Gujarati past tense and the continuous/habitual participle (જતો/જતી, કરતો/કરતી) DO mark gender, and this marking is the SAME regardless of person -- ગયો/ગઈ is used whether the subject is "I", "you", or "he/she".
4. DO NOT change 3rd-person subjects (e.g. 'my brother', 'my sister', 'he', 'she'). If the original text refers to a female third party, do NOT change her grammatical gender just because the target speaker is male, or vice versa.
5. WARNING: In the ergative construction (marked by મેં/તેં/એણે/તેણે, or a noun+એ suffix), the verb agrees with the OBJECT of the sentence, not the subject. Example: 'મેં પત્ર લખ્યો' is correct (agrees with masculine પત્ર); do not change it to match the speaker's gender.
6. WARNING: Impersonal or dative-subject constructions (like 'લાગે છે' -- feels/seems, or 'ગમે છે' -- likes) default to agreeing with the STIMULUS noun, not the target speaker's gender. Do not force these to agree with the target speaker.

EXAMPLES:
{_FEW_SHOT_EXAMPLES_GU}

Now, process this input:
English: {en}
Base Gujarati: {gu_draft}
Target: {gender_label}, {person_label}
Return ONLY the grammatically perfect Gujarati sentence. Do not include quotes, prefixes like 'Correct:', or any explanations."""
