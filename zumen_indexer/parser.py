import re
from zumen_indexer.config import TOC_KEYWORDS, FIELD_PATTERNS, TOC_ENTRY_RE


def normalize_spaces(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


def score_text_quality(text: str) -> float:
    stripped = re.sub(r"\s+", "", text)
    if not stripped:
        return 0.0
    japanese = len(re.findall(r"[\u3040-\u30ff\u3400-\u9fff]", stripped))
    ascii_word = len(re.findall(r"[A-Za-z0-9]", stripped))
    useful = japanese + ascii_word
    density = useful / max(len(stripped), 1)
    length_score = min(useful / 600.0, 1.0)
    return round((density * 0.4) + (length_score * 0.6), 3)


def should_ocr(text: str, mode: str) -> bool:
    if mode == "off":
        return False
    if mode == "force":
        return True
    return score_text_quality(text) < 0.35


def detect_toc_pages(page_texts: dict[int, str]) -> list[int]:
    scored = []
    for page_number, text in page_texts.items():
        header_score = sum(1 for keyword in TOC_KEYWORDS if keyword in text)
        if header_score < 2:
            continue
        entry_count = len(re.findall(r"\b[A-Z][0-9]{2}(?:-[0-9]+)?\b", text))
        score = (header_score * 100) + entry_count
        if entry_count >= 5:
            scored.append((score, page_number))
    return [page for _, page in sorted(scored, reverse=True)]


def parse_toc_entries(text: str, page_count: int) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for line in text.splitlines():
        normalized = normalize_spaces(line)
        if not normalized or "管理No" in normalized:
            continue
        for match in TOC_ENTRY_RE.finditer(normalized):
            page_number = int(match.group("index"))
            if page_number < 1 or page_number > page_count:
                continue
            title = clean_title(match.group("title"))
            if not title or len(title) > 80:
                continue
            entries.append(
                {
                    "page_number": page_number,
                    "drawing_no": match.group("drawing_no"),
                    "title": title,
                    "application_flag": normalize_flag(match.group("flag") or ""),
                    "source": "toc",
                    "confidence": 0.86,
                }
            )
    return dedupe_drawings(entries)


def clean_title(title: str) -> str:
    title = normalize_spaces(title)
    title = re.sub(r"\s+(?:●|○|\-|―|－)$", "", title)
    return title.strip(" ・")


def normalize_flag(flag: str) -> str:
    if flag in {"●", "○"}:
        return flag
    if flag in {"-", "―", "－"}:
        return "-"
    return ""


def dedupe_drawings(entries: list[dict[str, object]]) -> list[dict[str, object]]:
    by_key: dict[tuple[int, str], dict[str, object]] = {}
    for entry in entries:
        key = (int(entry["page_number"]), str(entry["drawing_no"]))
        by_key.setdefault(key, entry)
    return sorted(by_key.values(), key=lambda item: (int(item["page_number"]), str(item["drawing_no"])))


def infer_discipline(drawing_no: str, title: str) -> str:
    prefix = drawing_no[:1].upper()
    mapping = {
        "A": "architecture",
        "C": "structure",
        "S": "survey",
        "M": "mechanical",
        "E": "electrical",
        "G": "exterior",
        "P": "index",
    }
    if prefix in mapping:
        return mapping[prefix]
    if "構造" in title or "基礎" in title:
        return "structure"
    if "電気" in title or "照明" in title:
        return "electrical"
    if "給排水" in title or "換気" in title:
        return "mechanical"
    return "unknown"


def extract_title_block_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    tail = "\n".join(text.splitlines()[-45:])
    for name, pattern in FIELD_PATTERNS.items():
        match = pattern.search(tail)
        if match:
            fields[name] = normalize_spaces(match.group(1))

    drawing_name = re.search(r"図面名\s+(.+?)(?:\s+縮尺|\s+S\s*=|\n)", tail)
    if drawing_name:
        fields["drawing_title"] = normalize_spaces(drawing_name.group(1))

    drawing_no = re.search(r"図面番号\s+([A-Z][0-9]{1,3}(?:-[0-9]+)?)", tail)
    if drawing_no:
        fields["drawing_no"] = drawing_no.group(1)
    else:
        candidates = re.findall(r"\b([A-Z][0-9]{1,3}(?:-[0-9]+)?)\b", tail)
        if candidates:
            fields["drawing_no"] = candidates[-1]
    return fields


def chunk_text(text: str, max_chars: int = 1600) -> list[str]:
    compact_lines = [normalize_spaces(line) for line in text.splitlines()]
    compact = "\n".join(line for line in compact_lines if line)
    if not compact:
        return []
    chunks = []
    start = 0
    while start < len(compact):
        end = min(start + max_chars, len(compact))
        if end < len(compact):
            newline = compact.rfind("\n", start, end)
            if newline > start + 400:
                end = newline
        chunks.append(compact[start:end].strip())
        start = end
    return [chunk for chunk in chunks if chunk]
