import re


HEADER_RETAIN_KEYS = ("subject:", "keywords:")
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")
MULTISPACE_RE = re.compile(r"\s+")
CONTRACTION_RULES = (
    (re.compile(r"n['’]t\b"), " not"),
    (re.compile(r"['’]re\b"), " are"),
    (re.compile(r"['’]ve\b"), " have"),
    (re.compile(r"['’]ll\b"), " will"),
    (re.compile(r"['’]d\b"), " would"),
    (re.compile(r"['’]m\b"), " am"),
    (re.compile(r"['’]s\b"), ""),
)
USENET_FILLER_TOKENS = {
    "article",
    "believe",
    "don",
    "drive",
    "does",
    "did",
    "good",
    "just",
    "know",
    "ll",
    "like",
    "make",
    "made",
    "people",
    "question",
    "questions",
    "read",
    "really",
    "said",
    "say",
    "says",
    "think",
    "thanks",
    "time",
    "used",
    "using",
    "use",
    "want",
    "way",
    "writes",
    "year",
    "years",
}


def _strip_headers(raw_text: str) -> str:
    if "\n\n" not in raw_text:
        return raw_text
    header, body = raw_text.split("\n\n", 1)
    kept = []
    for line in header.splitlines():
        line_l = line.strip().lower()
        # Keep only intent-bearing metadata and drop routing/noise headers.
        if line_l.startswith(HEADER_RETAIN_KEYS):
            kept.append(line)
    return "\n".join(kept + [body])


def _strip_quotes_and_signature(text: str) -> str:
    out = []
    for line in text.splitlines():
        line_s = line.strip()
        # Usenet signatures and quoted replies are high-volume duplication noise.
        if line_s == "--":
            break
        if line_s.startswith(">"):
            continue
        out.append(line)
    return "\n".join(out)


def _normalize_contractions(text: str) -> str:
    out = text
    for pattern, replacement in CONTRACTION_RULES:
        out = pattern.sub(replacement, out)
    return out


def _prune_tokens(text: str) -> str:
    kept = []
    for token in text.split():
        if len(token) <= 2:
            continue
        if token in USENET_FILLER_TOKENS:
            continue
        kept.append(token)
    return " ".join(kept)


def clean_text(raw_text: str) -> str:
    text = _strip_headers(raw_text)
    text = _strip_quotes_and_signature(text)
    text = text.replace("\t", " ").lower()
    text = _normalize_contractions(text)
    text = EMAIL_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text).strip()
    text = _prune_tokens(text)
    return text
