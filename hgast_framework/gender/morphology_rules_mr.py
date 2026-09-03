

import re

# Text cleanup 
def clean_marathi(text: str) -> str:
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[ ]+([?.!,।])", r"\1", text)
    return text.strip()



MALE_TO_FEMALE_3RD_PAST = [
    # Core motion verbs
    ("गेला", "गेली"),          # went
    ("आला", "आली"),            # came
    ("निघाला", "निघाली"),      # departed/left
    ("पोहोचला", "पोहोचली"),    # arrived/reached
    ("परतला", "परतली"),        # returned
    ("पळाला", "पळाली"),        # ran away/fled
    ("धावला", "धावली"),        # ran
    ("चालला", "चालली"),        # walked
    ("फिरला", "फिरली"),        # roamed/walked around
    ("उतरला", "उतरली"),        # descended/got off
    ("चढला", "चढली"),          # climbed
    ("सरकला", "सरकली"),        # slid/moved aside

    # Core posture/state verbs
    ("बसला", "बसली"),          # sat
    ("उठला", "उठली"),          # got up
    ("झोपला", "झोपली"),        # slept
    ("पडला", "पडली"),          # fell/lay down
    ("थांबला", "थांबली"),      # stopped/waited
    ("राहिला", "राहिली"),      # stayed/remained
    ("जगला", "जगली"),          # lived/survived
    ("मेला", "मेली"),          # died

    # Emotion/state verbs
    ("हसला", "हसली"),          # laughed
    ("रडला", "रडली"),          # cried
    ("घाबरला", "घाबरली"),      # got scared
    ("रागावला", "रागावली"),    # got angry
    ("चिडला", "चिडली"),        # got irritated
    ("शरमला", "शरमली"),        # felt shy/embarrassed
    ("थकला", "थकली"),          # got tired
    ("भिजला", "भिजली"),        # got wet
    ("सुकला", "सुकली"),        # dried up
    ("जळला", "जळली"),          # burned

    # Communication verbs
    ("बोलला", "बोलली"),        # spoke
    ("सांगितला", "सांगितली"),  # told (irregular)
    ("विचारला", "विचारली"),    # asked
    ("ऐकला", "ऐकली"),          # heard/listened
    ("ओरडला", "ओरडली"),        # shouted

    # Action verbs
    ("खाल्ला", "खाल्ली"),      # ate (irregular past)
    ("प्याला", "प्याली"),      # drank (irregular past)
    ("केला", "केली"),          # did
    ("दिला", "दिली"),          # gave
    ("घेतला", "घेतली"),        # took
    ("पाहिला", "पाहिली"),      # saw/watched
    ("वाचला", "वाचली"),        # read
    ("लिहिला", "लिहिली"),      # wrote
    ("शिकला", "शिकली"),        # learned
    ("समजला", "समजली"),        # understood
    ("विसरला", "विसरली"),      # forgot
    ("आठवला", "आठवली"),        # remembered
    ("भेटला", "भेटली"),        # met
    ("जिंकला", "जिंकली"),      # won
    ("हरला", "हरली"),          # lost (competition)
    ("मारला", "मारली"),        # hit/killed
    ("तोडला", "तोडली"),        # broke
    ("बनवला", "बनवली"),        # made/created
    ("विकला", "विकली"),        # sold
    ("मिळवला", "मिळवली"),      # earned/obtained

    # Compound past: "became X"
    ("तयार झाला", "तयार झाली"),        # got ready
    ("खुश झाला", "खुश झाली"),          # became happy
    ("आजारी पडला", "आजारी पडली"),      # fell ill
    ("निवडला गेला", "निवडली गेली"),    # got selected
    ("जन्माला आला", "जन्माला आली"),    # was born
    ("मोठा झाला", "मोठी झाली"),        # grew up
    ("यशस्वी झाला", "यशस्वी झाली"),    # became successful
]



MALE_TO_FEMALE_1ST_PAST = [
    # Motion
    ("गेलो", "गेले"),          # I went
    ("आलो", "आले"),            # I came
    ("निघालो", "निघाले"),      # I departed
    ("पोहोचलो", "पोहोचले"),    # I arrived
    ("परतलो", "परतले"),        # I returned
    ("पळालो", "पळाले"),        # I fled
    ("धावलो", "धावले"),        # I ran
    ("चाललो", "चालले"),        # I walked
    ("फिरलो", "फिरले"),        # I roamed

    # Posture/state
    ("बसलो", "बसले"),          # I sat
    ("उठलो", "उठले"),          # I got up
    ("झोपलो", "झोपले"),        # I slept
    ("पडलो", "पडले"),          # I fell
    ("थांबलो", "थांबले"),      # I stopped
    ("राहिलो", "राहिले"),      # I stayed

    # Emotion/state
    ("हसलो", "हसले"),          # I laughed
    ("रडलो", "रडले"),          # I cried
    ("घाबरलो", "घाबरले"),      # I got scared
    ("रागावलो", "रागावले"),    # I got angry
    ("चिडलो", "चिडले"),        # I got irritated
    ("थकलो", "थकले"),          # I got tired

    # Communication
    ("बोललो", "बोलले"),        # I spoke
    ("विचारलो", "विचारले"),    # I asked
    ("ऐकलो", "ऐकले"),          # I heard

    # Action
    ("केलो", "केले"),          # I did (alt form)
    ("पाहिलो", "पाहिले"),      # I saw
    ("वाचलो", "वाचले"),        # I read
    ("लिहिलो", "लिहिले"),      # I wrote
    ("शिकलो", "शिकले"),        # I learned
    ("समजलो", "समजले"),        # I understood
    ("विसरलो", "विसरले"),      # I forgot
    ("भेटलो", "भेटले"),        # I met
    ("जिंकलो", "जिंकले"),      # I won
]

MALE_TO_FEMALE_HABITUAL_PRESENT = [
    # Core verbs
    ("जातो", "जाते"),          # go(es)
    ("येतो", "येते"),          # come(s)
    ("करतो", "करते"),          # do(es)
    ("खातो", "खाते"),          # eat(s)
    ("पितो", "पिते"),          # drink(s)
    ("बोलतो", "बोलते"),        # speak(s)
    ("बसतो", "बसते"),          # sit(s)
    ("उठतो", "उठते"),          # get(s) up
    ("झोपतो", "झोपते"),        # sleep(s)
    ("हसतो", "हसते"),          # laugh(s)
    ("रडतो", "रडते"),          # cry/cries
    ("धावतो", "धावते"),        # run(s)
    ("चालतो", "चालते"),        # walk(s)
    ("फिरतो", "फिरते"),        # roam(s)
    ("थांबतो", "थांबते"),      # stop(s)/wait(s)
    ("राहतो", "राहते"),        # stay(s)/live(s)

    # Intellectual/perception
    ("वाचतो", "वाचते"),        # read(s)
    ("लिहितो", "लिहिते"),      # write(s)
    ("पाहतो", "पाहते"),        # see(s)/watch(es)
    ("ऐकतो", "ऐकते"),          # hear(s)/listen(s)
    ("समजतो", "समजते"),        # understand(s)
    ("शिकतो", "शिकते"),        # learn(s)/teach(es)
    ("विसरतो", "विसरते"),      # forget(s)
    ("आठवतो", "आठवते"),        # remember(s)
    ("विचार करतो", "विचार करते"),  # think(s)

    # Daily/work
    ("काम करतो", "काम करते"),  # work(s)
    ("स्वयंपाक करतो", "स्वयंपाक करते"),  # cook(s)
    ("खेळतो", "खेळते"),        # play(s)
    ("गातो", "गाते"),          # sing(s)
    ("नाचतो", "नाचते"),        # dance(s)
    ("सांगतो", "सांगते"),      # tell(s)
    ("विचारतो", "विचारते"),    # ask(s)
    ("देतो", "देते"),          # give(s)
    ("घेतो", "घेते"),          # take(s)
    ("मारतो", "मारते"),        # hit(s)
    ("तोडतो", "तोडते"),        # break(s)
    ("बनवतो", "बनवते"),        # make(s)
    ("मिळवतो", "मिळवते"),      # earn(s)/obtain(s)

    # Emotion/state
    ("आवडतो", "आवडते"),        # is liked / "likes"
    ("वाटतो", "वाटते"),        # feels like
    ("दिसतो", "दिसते"),        # looks/appears
    ("लागतो", "लागते"),        # feels / takes (time)
    ("भेटतो", "भेटते"),        # meet(s)
    ("ओरडतो", "ओरडते"),        # shout(s)
    ("रागावतो", "रागावते"),    # get(s) angry
    ("घाबरतो", "घाबरते"),      # get(s) scared
]



MALE_TO_FEMALE_PAST_CONTINUOUS = [
    ("होता", "होती"),          # was (core copula past)
    ("जात होता", "जात होती"),    # was going
    ("येत होता", "येत होती"),    # was coming
    ("करत होता", "करत होती"),    # was doing
    ("खात होता", "खात होती"),    # was eating
    ("बोलत होता", "बोलत होती"),  # was speaking
    ("बसत होता", "बसत होती"),    # was sitting
    ("धावत होता", "धावत होती"),  # was running
    ("वाचत होता", "वाचत होती"),  # was reading
    ("लिहित होता", "लिहित होती"),# was writing
    ("पाहत होता", "पाहत होती"),  # was watching
    ("ऐकत होता", "ऐकत होती"),    # was listening
    ("खेळत होता", "खेळत होती"),  # was playing
    ("गात होता", "गात होती"),    # was singing
    ("नाचत होता", "नाचत होती"),  # was dancing
    ("शिकत होता", "शिकत होती"),  # was learning
    ("राहत होता", "राहत होती"),  # was living/staying
    ("काम करत होता", "काम करत होती"),  # was working
]


MALE_TO_FEMALE_PAST_HABITUAL = [
    ("जायचा", "जायची"),        # used to go
    ("यायचा", "यायची"),        # used to come
    ("करायचा", "करायची"),      # used to do
    ("खायचा", "खायची"),        # used to eat
    ("बोलायचा", "बोलायची"),    # used to speak
    ("बसायचा", "बसायची"),      # used to sit
    ("राहायचा", "राहायची"),    # used to stay
    ("वाचायचा", "वाचायची"),    # used to read
    ("खेळायचा", "खेळायची"),    # used to play
    ("गायचा", "गायची"),        # used to sing
    ("ऐकायचा", "ऐकायची"),      # used to listen
]


# 6. MODAL "CAN" — शकतो/शकते
MALE_TO_FEMALE_MODAL_CAN = [
    ("करू शकतो", "करू शकते"),       # can do
    ("जाऊ शकतो", "जाऊ शकते"),       # can go
    ("येऊ शकतो", "येऊ शकते"),       # can come
    ("बोलू शकतो", "बोलू शकते"),     # can speak
    ("पाहू शकतो", "पाहू शकते"),     # can see
    ("ऐकू शकतो", "ऐकू शकते"),       # can hear
    ("खाऊ शकतो", "खाऊ शकते"),       # can eat
    ("पिऊ शकतो", "पिऊ शकते"),       # can drink
    ("वाचू शकतो", "वाचू शकते"),     # can read
    ("लिहू शकतो", "लिहू शकते"),     # can write
    ("शिकू शकतो", "शिकू शकते"),     # can learn
    ("समजू शकतो", "समजू शकते"),     # can understand
    ("राहू शकतो", "राहू शकते"),     # can stay
    ("धावू शकतो", "धावू शकते"),     # can run
    ("शकतो", "शकते"),               # can (bare fallback)
]

# 7. MODAL "COULD" — शकला/शकली (past ability)
MALE_TO_FEMALE_MODAL_COULD = [
    ("करू शकला", "करू शकली"),       # could do
    ("जाऊ शकला", "जाऊ शकली"),       # could go
    ("येऊ शकला", "येऊ शकली"),       # could come
    ("बोलू शकला", "बोलू शकली"),     # could speak
    ("पाहू शकला", "पाहू शकली"),     # could see
    ("शकला", "शकली"),               # could (bare fallback)
]


# 8. "WANT TO" — इच्छुक / करायचा आहे / करायची आहे
MALE_TO_FEMALE_WANT = [
    ("करायचा आहे", "करायची आहे"),      # want to do
    ("जायचा आहे", "जायची आहे"),        # want to go
    ("यायचा आहे", "यायची आहे"),        # want to come
    ("खायचा आहे", "खायची आहे"),        # want to eat
    ("बोलायचा आहे", "बोलायची आहे"),    # want to speak
    ("बघायचा आहे", "बघायची आहे"),      # want to see/watch
    ("शिकायचा आहे", "शिकायची आहे"),    # want to learn
    ("राहायचा आहे", "राहायची आहे"),    # want to stay
]

MALE_TO_FEMALE_PARTICIPIAL_ADJ = [
    ("थकलेला", "थकलेली"),            # tired
    ("भिजलेला", "भिजलेली"),          # soaked/wet
    ("हरवलेला", "हरवलेली"),          # lost
    ("घाबरलेला", "घाबरलेली"),        # scared
    ("रागावलेला", "रागावलेली"),      # angry
    ("खुशीत असलेला", "खुशीत असलेली"),  # happy (being)
    ("आजारी असलेला", "आजारी असलेली"),  # sick (being)
    ("बसलेला", "बसलेली"),            # seated
    ("उभा असलेला", "उभी असलेली"),    # standing
    ("झोपलेला", "झोपलेली"),          # asleep
    ("जागा असलेला", "जागी असलेली"),  # awake
    ("आलेला", "आलेली"),              # who came
    ("गेलेला", "गेलेली"),            # who went
]

MALE_TO_FEMALE_ADJECTIVES = [
    ("चांगला", "चांगली"),            # good
    ("मोठा", "मोठी"),                # big/elder
    ("लहान", "लहान"),                # small (INVARIANT — no-op, kept for documentation)
    ("एकटा", "एकटी"),                # alone
    ("सुंदर", "सुंदर"),              # beautiful (INVARIANT)
    ("तयार", "तयार"),                # ready (INVARIANT)
    ("खरा", "खरी"),                  # true/real
    ("वाईट", "वाईट"),                # bad (INVARIANT)
    ("आनंदी", "आनंदी"),              # happy (INVARIANT)
    ("उंच", "उंच"),                  # tall (INVARIANT)
    ("जाड", "जाड"),                  # fat (INVARIANT)
    ("बारीक", "बारीक"),              # thin (INVARIANT)
    ("गरीब", "गरीब"),                # poor (INVARIANT)
    ("श्रीमंत", "श्रीमंत"),          # rich (INVARIANT)
    ("पहिला", "पहिली"),              # first
    ("दुसरा", "दुसरी"),              # second
    ("शेवटचा", "शेवटची"),            # last
    ("नवा", "नवी"),                  # new
    ("जुना", "जुनी"),                # old (thing)
    ("उभा", "उभी"),                  # standing
    ("अर्धा", "अर्धी"),              # half
    ("पूर्ण", "पूर्ण"),              # complete (INVARIANT)
    ("स्वतंत्र", "स्वतंत्र"),        # independent (INVARIANT)
]


# 11. COMPOUND VERBS — "did and went" type sequences
MALE_TO_FEMALE_COMPOUND = [
    ("करून गेला", "करून गेली"),      # did and went
    ("करून आला", "करून आली"),        # did and came
    ("सांगून गेला", "सांगून गेली"),  # told and went
    ("ऐकून घेतला", "ऐकून घेतली"),    # listened and took
    ("पाहून गेला", "पाहून गेली"),    # saw and went
    ("खाऊन गेला", "खाऊन गेली"),      # ate and went
    ("बसून राहिला", "बसून राहिली"),  # sat and stayed
    ("उठून गेला", "उठून गेली"),      # got up and went
    ("येऊन बसला", "येऊन बसली"),      # came and sat
    ("निघून गेला", "निघून गेली"),    # departed and went
]


# COMBINED TABLE — longest-match-first ordering
MALE_TO_FEMALE = (
    MALE_TO_FEMALE_COMPOUND          # longest multi-word first
    + MALE_TO_FEMALE_PAST_CONTINUOUS
    + MALE_TO_FEMALE_PAST_HABITUAL
    + MALE_TO_FEMALE_MODAL_CAN
    + MALE_TO_FEMALE_MODAL_COULD
    + MALE_TO_FEMALE_WANT
    + MALE_TO_FEMALE_PARTICIPIAL_ADJ
    + MALE_TO_FEMALE_3RD_PAST
    + MALE_TO_FEMALE_1ST_PAST
    + MALE_TO_FEMALE_HABITUAL_PRESENT
    + [pair for pair in MALE_TO_FEMALE_ADJECTIVES if pair[0] != pair[1]]
)
FEMALE_TO_MALE = [(f, m) for m, f in MALE_TO_FEMALE]

_SORTED_M2F = sorted(MALE_TO_FEMALE, key=lambda x: -len(x[0]))
_SORTED_F2M = sorted(FEMALE_TO_MALE, key=lambda x: -len(x[0]))


def _apply_rules_mr(mr: str, target_gender: int) -> str:
    rules = _SORTED_M2F if target_gender == 1 else _SORTED_F2M
    for src, tgt in rules:
        if src in mr:
            mr = clean_marathi(mr.replace(src, tgt))
    return mr


def apply_morphology_mr(mr: str, target_gender: int, dominant_controller: str = "",
                         person: str = "") -> str:
    if target_gender not in (0, 1):
        return mr
    if dominant_controller not in ("subject", "speaker"):
        return mr
    return _apply_rules_mr(mr, target_gender)


# Ergative guard — Marathi marks the AGENT with a "-ने" SUFFIX
_ERGATIVE_SUFFIX = "ने"
_ERGATIVE_OBJECTS_MR = {"जेवण", "पत्र", "काम", "घर", "पुस्तक", "चित्र", "गाणे",
                        "खाना", "पाणी", "कपडे", "भाषण", "अन्न"}


def is_ergative_mr(mr: str) -> bool:
    toks = mr.split()
    has_ergative_agent = any(tok.endswith(_ERGATIVE_SUFFIX) and len(tok) > len(_ERGATIVE_SUFFIX) + 1
                              for tok in toks)
    if not has_ergative_agent:
        return False
    return any(obj in mr for obj in _ERGATIVE_OBJECTS_MR)


# Morphology token counting — gender-accuracy metric for Marathi
VALID_MALE_MR = {m for m, f in MALE_TO_FEMALE}
VALID_FEMALE_MR = {f for m, f in MALE_TO_FEMALE}


def count_morph_tokens_mr(mr: str) -> dict:
    toks = mr.split()
    tok_set = set(toks)

    male_tokens = sorted(tok_set & VALID_MALE_MR)
    female_tokens = sorted(tok_set & VALID_FEMALE_MR)

    # Multi-word entries
    for src in VALID_MALE_MR:
        if " " in src and src in mr and src not in male_tokens:
            male_tokens.append(src)
    for src in VALID_FEMALE_MR:
        if " " in src and src in mr and src not in female_tokens:
            female_tokens.append(src)

    if len(female_tokens) > len(male_tokens):
        dominant = "female"
    elif len(male_tokens) > len(female_tokens):
        dominant = "male"
    else:
        dominant = "neutral"

    return {"male_tokens": male_tokens, "female_tokens": female_tokens, "dominant": dominant}


def morph_match_mr(info: dict, target_gender: int) -> bool:
    return info["dominant"] in ("neutral", "female" if target_gender == 1 else "male")


# Token sets for LLM safety gates
MALE_TOK = VALID_MALE_MR
FEMALE_TOK = VALID_FEMALE_MR
PLURAL_TOK = {"गेले", "आले", "होते", "होत्या", "आहेत", "करत होते",
              "जात होते", "येत होते", "बसत होते", "राहत होते"}
