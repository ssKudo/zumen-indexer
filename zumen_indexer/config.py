import re

TOC_KEYWORDS = ("図面リスト", "図面名称", "管理No", "種別連番")

FIELD_PATTERNS = {
    "project_no": re.compile(r"物件番号[:：]?\s*([A-Za-z0-9\-]+)"),
    "ap_no": re.compile(r"AP番号\s*([A-Za-z0-9\-]+)"),
    "scale": re.compile(r"(?:縮尺[（(][^)）]+[）)]\s*)?(S\s*=\s*1\s*:\s*\d+|1\s*/\s*\d+)"),
}

TOC_ENTRY_RE = re.compile(
    r"(?P<index>\d{1,3})\s+"
    r"(?P<drawing_no>[A-Z][0-9]{2}(?:-[0-9]+)?)\s+"
    r"(?P<title>.+?)"
    r"(?:\s+(?P<flag>[●○\-―－])(?=\s+\d{1,3}\s+[A-Z][0-9]{2}|\s*$)|(?=\s+\d{1,3}\s+[A-Z][0-9]{2}|\s*$))"
)
