"""ARPAbet to IPA phoneme mappings."""

ARPABET_VOWELS = {
    "AA": "\u0251", "AE": "\u00e6", "AH": "\u028c", "AO": "\u0254",
    "AW": "a\u028a", "AY": "a\u026a",
    "EH": "\u025b", "ER": "\u025d", "EY": "e\u026a",
    "IH": "\u026a", "IY": "i", "OW": "o\u028a", "OY": "\u0254\u026a",
    "UH": "\u028a", "UW": "u",
}

ARPABET_STOPS = {
    "P": {"ipa": "p", "voicing": "voiceless", "place": "bilabial"},
    "B": {"ipa": "b", "voicing": "voiced",    "place": "bilabial"},
    "T": {"ipa": "t", "voicing": "voiceless", "place": "alveolar"},
    "D": {"ipa": "d", "voicing": "voiced",    "place": "alveolar"},
    "K": {"ipa": "k", "voicing": "voiceless", "place": "velar"},
    "G": {"ipa": "g", "voicing": "voiced",    "place": "velar"},
}

ARPABET_FRICATIVES = {
    "F":  {"ipa": "f",  "voicing": "voiceless", "place": "labiodental"},
    "V":  {"ipa": "v",  "voicing": "voiced",    "place": "labiodental"},
    "TH": {"ipa": "\u03b8",  "voicing": "voiceless", "place": "dental"},
    "DH": {"ipa": "\u00f0",  "voicing": "voiced",    "place": "dental"},
    "S":  {"ipa": "s",  "voicing": "voiceless", "place": "alveolar"},
    "Z":  {"ipa": "z",  "voicing": "voiced",    "place": "alveolar"},
    "SH": {"ipa": "\u0283",  "voicing": "voiceless", "place": "postalveolar"},
    "ZH": {"ipa": "\u0292",  "voicing": "voiced",    "place": "postalveolar"},
}
