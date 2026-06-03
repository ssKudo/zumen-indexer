#!/usr/bin/env python3
"""Index architecture drawing PDFs into SQLite."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


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


@dataclass
class PageResult:
    page_number: int
    embedded_text: str
    ocr_text: str
    best_text: str
    text_source: str
    needs_ocr: bool
    quality_score: float


try:
    import numpy as np
    from PIL import Image
    HAS_PIL_NUMPY = True
except ImportError:
    HAS_PIL_NUMPY = False


def run_command(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=check, text=True, capture_output=True)


def require_tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise SystemExit(f"Required command not found: {name}")
    return path


def optional_tool(name: str) -> str | None:
    return shutil.which(name)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_pdfinfo(raw: str) -> dict[str, str]:
    info: dict[str, str] = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        info[key.strip()] = value.strip()
    return info


def get_pdf_info(pdf_path: Path) -> dict[str, str]:
    require_tool("pdfinfo")
    result = run_command(["pdfinfo", str(pdf_path)])
    return parse_pdfinfo(result.stdout)


def extract_text_page(pdf_path: Path, page_number: int, crop: tuple[int, int, int, int] | None = None, resolution: int = 150) -> str:
    require_tool("pdftotext")
    args = [
        "pdftotext",
        "-f",
        str(page_number),
        "-l",
        str(page_number),
        "-layout",
    ]
    if crop:
        x, y, w, h = crop
        args.extend([
            "-r", str(resolution),
            "-x", str(x),
            "-y", str(y),
            "-W", str(w),
            "-H", str(h),
        ])
    args.extend([str(pdf_path), "-"])
    result = run_command(args, check=False)
    return result.stdout or ""


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


def ocr_page(pdf_path: Path, page_number: int, lang: str, dpi: int, crop: tuple[int, int, int, int] | None = None) -> str:
    require_tool("pdftoppm")
    require_tool("tesseract")
    with tempfile.TemporaryDirectory(prefix="zumen-ocr-") as tmp:
        prefix = Path(tmp) / "page"
        run_command(
            [
                "pdftoppm",
                "-f",
                str(page_number),
                "-l",
                str(page_number),
                "-png",
                "-r",
                str(dpi),
                str(pdf_path),
                str(prefix),
            ]
        )
        images = sorted(Path(tmp).glob("page-*.png"))
        if not images:
            return ""
        
        image_path = images[0]
        if crop and HAS_PIL_NUMPY:
            x, y, w, h = crop
            try:
                img = Image.open(image_path)
                cropped_img = img.crop((x, y, x + w, y + h))
                cropped_path = Path(tmp) / "page_cropped.png"
                cropped_img.save(cropped_path)
                image_path = cropped_path
            except Exception as e:
                print(f"Warning: OCR page cropping failed: {e}. Running OCR on full page.", file=sys.stderr)
                
        result = run_command(["tesseract", str(image_path), "stdout", "-l", lang], check=False)
        return result.stdout or ""


def detect_vertical_columns(pdf_path: Path, page_number: int, dpi: int = 150) -> tuple[list[tuple[int, int]], int]:
    if not HAS_PIL_NUMPY:
        return [], 0
    require_tool("pdftoppm")
    with tempfile.TemporaryDirectory(prefix="zumen-layout-") as tmp:
        prefix = Path(tmp) / "page"
        run_command(
            [
                "pdftoppm",
                "-f",
                str(page_number),
                "-l",
                str(page_number),
                "-png",
                "-r",
                str(dpi),
                str(pdf_path),
                str(prefix),
            ],
            check=False,
        )
        images = sorted(Path(tmp).glob("page-*.png"))
        if not images:
            return [], 0
        try:
            img = Image.open(images[0]).convert("L")
            width, height = img.size
            arr = np.array(img)
            
            y_start = int(height * 0.1)
            y_end = int(height * 0.9)
            sub_arr = arr[y_start:y_end, :]
            
            binary = (sub_arr < 200).astype(int)
            projection = np.sum(binary, axis=0)
            
            threshold = (y_end - y_start) * 0.005
            is_gutter = projection < threshold
            
            columns = []
            in_column = False
            col_start = 0
            for x in range(width):
                if not is_gutter[x] and not in_column:
                    in_column = True
                    col_start = x
                elif is_gutter[x] and in_column:
                    in_column = False
                    col_width = x - col_start
                    if col_width > 100:
                        columns.append((col_start, col_width))
            if in_column:
                col_width = width - col_start
                if col_width > 100:
                    columns.append((col_start, col_width))
                    
            if len(columns) > 1:
                return columns, height
            return [], height
        except Exception as e:
            print(f"Warning: Layout column detection failed: {e}", file=sys.stderr)
            return [], 0


def normalize_spaces(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


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


def connect_db(path: Path, overwrite: bool) -> sqlite3.Connection:
    if overwrite and path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    create_schema(conn)
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            path TEXT NOT NULL,
            file_name TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            page_count INTEGER NOT NULL,
            page_size TEXT,
            created_at TEXT NOT NULL,
            pdf_metadata_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS pages (
            id INTEGER PRIMARY KEY,
            document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            page_number INTEGER NOT NULL,
            embedded_text_chars INTEGER NOT NULL,
            ocr_text_chars INTEGER NOT NULL,
            best_text_chars INTEGER NOT NULL,
            text_source TEXT NOT NULL,
            needs_ocr INTEGER NOT NULL,
            quality_score REAL NOT NULL,
            UNIQUE(document_id, page_number)
        );

        CREATE TABLE IF NOT EXISTS page_texts (
            page_id INTEGER PRIMARY KEY REFERENCES pages(id) ON DELETE CASCADE,
            embedded_text TEXT NOT NULL,
            ocr_text TEXT NOT NULL,
            best_text TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS drawings (
            id INTEGER PRIMARY KEY,
            document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            page_number INTEGER NOT NULL,
            drawing_no TEXT,
            title TEXT,
            discipline TEXT,
            scale TEXT,
            application_flag TEXT,
            source TEXT NOT NULL,
            confidence REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS extracted_fields (
            id INTEGER PRIMARY KEY,
            page_id INTEGER NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
            field_name TEXT NOT NULL,
            field_value TEXT NOT NULL,
            source TEXT NOT NULL,
            confidence REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            page_id INTEGER NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            chunk_text TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS processing_logs (
            id INTEGER PRIMARY KEY,
            level TEXT NOT NULL,
            message TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )


def insert_document(conn: sqlite3.Connection, pdf_path: Path, info: dict[str, str]) -> int:
    cursor = conn.execute(
        """
        INSERT INTO documents(path, file_name, sha256, page_count, page_size, created_at, pdf_metadata_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(pdf_path),
            pdf_path.name,
            sha256_file(pdf_path),
            int(info.get("Pages", "0") or "0"),
            info.get("Page size", ""),
            datetime.now(timezone.utc).isoformat(),
            json.dumps(info, ensure_ascii=False),
        ),
    )
    return int(cursor.lastrowid)


def insert_page(conn: sqlite3.Connection, document_id: int, result: PageResult) -> int:
    cursor = conn.execute(
        """
        INSERT INTO pages(
            document_id, page_number, embedded_text_chars, ocr_text_chars, best_text_chars,
            text_source, needs_ocr, quality_score
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            document_id,
            result.page_number,
            len(result.embedded_text),
            len(result.ocr_text),
            len(result.best_text),
            result.text_source,
            1 if result.needs_ocr else 0,
            result.quality_score,
        ),
    )
    page_id = int(cursor.lastrowid)
    conn.execute(
        "INSERT INTO page_texts(page_id, embedded_text, ocr_text, best_text) VALUES (?, ?, ?, ?)",
        (page_id, result.embedded_text, result.ocr_text, result.best_text),
    )
    return page_id


def insert_drawings(conn: sqlite3.Connection, document_id: int, drawings: Iterable[dict[str, object]]) -> None:
    for drawing in drawings:
        drawing_no = str(drawing.get("drawing_no") or "")
        title = str(drawing.get("title") or "")
        conn.execute(
            """
            INSERT INTO drawings(
                document_id, page_number, drawing_no, title, discipline, scale,
                application_flag, source, confidence
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document_id,
                int(drawing.get("page_number") or 0),
                drawing_no,
                title,
                infer_discipline(drawing_no, title),
                drawing.get("scale") or "",
                drawing.get("application_flag") or "",
                drawing.get("source") or "unknown",
                float(drawing.get("confidence") or 0.0),
            ),
        )


def write_jsonl(path: Path, conn: sqlite3.Connection, document_id: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    query = """
        SELECT p.page_number, d.drawing_no, d.title, d.discipline, d.scale, p.text_source, pt.best_text
        FROM pages p
        JOIN page_texts pt ON pt.page_id = p.id
        LEFT JOIN drawings d ON d.document_id = p.document_id AND d.page_number = p.page_number
        WHERE p.document_id = ?
        ORDER BY p.page_number
    """
    with path.open("w", encoding="utf-8") as fh:
        for row in conn.execute(query, (document_id,)):
            page_number, drawing_no, title, discipline, scale, text_source, best_text = row
            fh.write(
                json.dumps(
                    {
                        "page_number": page_number,
                        "drawing_no": drawing_no,
                        "title": title,
                        "discipline": discipline,
                        "scale": scale,
                        "text_source": text_source,
                        "text": best_text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def compact_preview(text: str, limit: int = 280) -> str:
    compact = normalize_spaces(re.sub(r"\s+", " ", text))
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "..."


def quality_label(score: float, best_text_chars: int) -> str:
    if best_text_chars == 0:
        return "none"
    if score >= 0.85:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"


def label_confidence(score: float) -> str:
    if score >= 0.85:
        return f"高 ({score:.2f})"
    if score >= 0.60:
        return f"中 ({score:.2f})"
    return f"要確認 ({score:.2f})"


def write_report(path: Path, conn: sqlite3.Connection, document_id: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = conn.execute(
        """
        SELECT file_name, page_count, page_size, pdf_metadata_json
        FROM documents
        WHERE id = ?
        """,
        (document_id,),
    ).fetchone()
    if not document:
        raise RuntimeError("Document row was not found.")

    file_name, page_count, page_size, metadata_json = document
    metadata = json.loads(metadata_json)
    pages = conn.execute(
        """
        SELECT
            p.id, p.page_number, p.embedded_text_chars, p.ocr_text_chars,
            p.best_text_chars, p.text_source, p.needs_ocr, p.quality_score,
            COALESCE(d.drawing_no, ''), COALESCE(d.title, ''),
            COALESCE(d.discipline, ''), COALESCE(d.scale, ''),
            COALESCE(d.application_flag, ''), COALESCE(d.source, ''),
            COALESCE(d.confidence, 0.0),
            pt.best_text
        FROM pages p
        JOIN page_texts pt ON pt.page_id = p.id
        LEFT JOIN drawings d ON d.document_id = p.document_id AND d.page_number = p.page_number
        WHERE p.document_id = ?
        ORDER BY p.page_number
        """,
        (document_id,),
    ).fetchall()

    lines = [
        "# Zumen Index Readability Report",
        "",
        f"PDF: {file_name}",
        f"Pages: {page_count}",
        f"Page size: {page_size}",
        f"PDF created: {metadata.get('CreationDate', '')}",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Legend:",
        "- source=embedded: PDF内の文字情報から読み取り",
        "- source=ocr: OCR結果を採用",
        "- quality=high/medium/low/none: 文字量と有効文字密度から見た簡易評価",
        "- needs_ocr=yes: 自動判定ではOCR対象になったページ",
        "- 信頼度: 高 (>= 0.85) / 中 (>= 0.60) / 要確認 (< 0.60) の判定確度",
        "",
    ]

    for row in pages:
        (
            page_id,
            page_number,
            embedded_chars,
            ocr_chars,
            best_chars,
            text_source,
            needs_ocr,
            quality_score,
            drawing_no,
            title,
            discipline,
            scale,
            application_flag,
            drawing_source,
            drawing_confidence,
            best_text,
        ) = row
        fields = conn.execute(
            """
            SELECT field_name, field_value, confidence
            FROM extracted_fields
            WHERE page_id = ?
            ORDER BY field_name
            """,
            (page_id,),
        ).fetchall()
        field_text = ", ".join(f"{name}={value} ({label_confidence(conf)})" for name, value, conf in fields) or "-"
        chunk_count = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE page_id = ?",
            (page_id,),
        ).fetchone()[0]

        lines.extend(
            [
                f"## Page {page_number:03d}",
                f"Drawing: {drawing_no or '-'} {title or '-'}",
                f"Discipline: {discipline or '-'}",
                f"Scale: {scale or '-'}",
                f"Application flag: {application_flag or '-'}",
                f"Drawing metadata source: {drawing_source or '-'} (信頼度: {label_confidence(drawing_confidence)})",
                (
                    "Readability: "
                    f"quality={quality_label(float(quality_score), int(best_chars))} "
                    f"score={quality_score} source={text_source} "
                    f"embedded_chars={embedded_chars} ocr_chars={ocr_chars} "
                    f"best_chars={best_chars} needs_ocr={'yes' if needs_ocr else 'no'} "
                    f"chunks={chunk_count}"
                ),
                f"Extracted fields: {field_text}",
                f"Preview: {compact_preview(best_text)}",
                "",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def index_pdf(args: argparse.Namespace) -> None:
    pdf_path = Path(args.input).expanduser().resolve()
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise SystemExit("Input must be a PDF file.")

    info = get_pdf_info(pdf_path)
    page_count = int(info.get("Pages", "0") or "0")
    if args.max_pages:
        page_count = min(page_count, args.max_pages)

    conn = connect_db(Path(args.db), args.overwrite)
    document_id = insert_document(conn, pdf_path, info)

    page_texts: dict[int, str] = {}
    page_ids: dict[int, int] = {}
    for page_number in range(1, page_count + 1):
        # 1. Do a standard single-column text extraction first to see if it's a TOC candidate page
        embedded = extract_text_page(pdf_path, page_number)
        
        # 2. Check if it's a TOC candidate based on keywords
        is_toc_page = False
        header_score = sum(1 for keyword in TOC_KEYWORDS if keyword in embedded)
        if header_score >= 2:
            entry_count = len(re.findall(r"\b[A-Z][0-9]{2}(?:-[0-9]+)?\b", embedded))
            if entry_count >= 5:
                is_toc_page = True
                
        # 3. Detect visual columns if it's a TOC page
        columns = []
        page_height = 0
        if is_toc_page and HAS_PIL_NUMPY:
            columns, page_height = detect_vertical_columns(pdf_path, page_number, dpi=150)
            if columns and args.verbose:
                print(f"page {page_number}/{page_count}: detected {len(columns)} visual columns for TOC layout")
                
        # 4. Extract text/OCR column-by-column or fallback to standard single column
        if columns:
            embedded_parts = []
            for col_x, col_w in columns:
                part = extract_text_page(pdf_path, page_number, crop=(col_x, 0, col_w, page_height), resolution=150)
                embedded_parts.append(part)
            embedded = "\n".join(embedded_parts)
            
            needs_ocr = should_ocr(embedded, args.ocr)
            ocr_text = ""
            if needs_ocr:
                if optional_tool("pdftoppm") and optional_tool("tesseract"):
                    ocr_parts = []
                    for col_x, col_w in columns:
                        # Scale coordinates to ocr_dpi
                        scaled_x = int(col_x * args.ocr_dpi / 150)
                        scaled_w = int(col_w * args.ocr_dpi / 150)
                        scaled_h = int(page_height * args.ocr_dpi / 150)
                        part = ocr_page(pdf_path, page_number, args.ocr_lang, args.ocr_dpi, crop=(scaled_x, 0, scaled_w, scaled_h))
                        ocr_parts.append(part)
                    ocr_text = "\n".join(ocr_parts)
                else:
                    needs_ocr = False
        else:
            # Standard single-column extraction
            needs_ocr = should_ocr(embedded, args.ocr)
            ocr_text = ""
            if needs_ocr:
                if optional_tool("pdftoppm") and optional_tool("tesseract"):
                    ocr_text = ocr_page(pdf_path, page_number, args.ocr_lang, args.ocr_dpi)
                else:
                    needs_ocr = False
                    
        embedded_score = score_text_quality(embedded)
        ocr_score = score_text_quality(ocr_text)
        if ocr_score > embedded_score:
            best_text = ocr_text
            text_source = "ocr"
            quality = ocr_score
        else:
            best_text = embedded
            text_source = "embedded"
            quality = embedded_score

        result = PageResult(page_number, embedded, ocr_text, best_text, text_source, needs_ocr, quality)
        page_id = insert_page(conn, document_id, result)
        page_ids[page_number] = page_id
        page_texts[page_number] = best_text

        fields = extract_title_block_fields(best_text)
        for field_name, field_value in fields.items():
            conn.execute(
                """
                INSERT INTO extracted_fields(page_id, field_name, field_value, source, confidence)
                VALUES (?, ?, ?, ?, ?)
                """,
                (page_id, field_name, field_value, "title_block", 0.62),
            )

        for index, chunk in enumerate(chunk_text(best_text)):
            conn.execute(
                "INSERT INTO chunks(page_id, chunk_index, chunk_text) VALUES (?, ?, ?)",
                (page_id, index, chunk),
            )

        if args.verbose:
            print(
                f"page {page_number}/{page_count}: source={text_source} "
                f"chars={len(best_text)} quality={quality}"
            )

    toc_pages = detect_toc_pages(page_texts)
    drawings: list[dict[str, object]] = []
    for toc_page in toc_pages[:3]:
        drawings.extend(parse_toc_entries(page_texts[toc_page], page_count))

    known_pages = {int(item["page_number"]) for item in drawings}
    for page_number, text in page_texts.items():
        if page_number in known_pages:
            continue
        fields = extract_title_block_fields(text)
        drawing_no = fields.get("drawing_no", "")
        title = fields.get("drawing_title", "")
        if drawing_no or title:
            drawings.append(
                {
                    "page_number": page_number,
                    "drawing_no": drawing_no,
                    "title": title,
                    "scale": fields.get("scale", ""),
                    "application_flag": "",
                    "source": "title_block",
                    "confidence": 0.58,
                }
            )

    insert_drawings(conn, document_id, dedupe_drawings(drawings))
    conn.commit()

    if args.jsonl:
        write_jsonl(Path(args.jsonl), conn, document_id)
    if args.report:
        write_report(Path(args.report), conn, document_id)

    drawing_count = conn.execute(
        "SELECT COUNT(*) FROM drawings WHERE document_id = ?", (document_id,)
    ).fetchone()[0]
    ocr_count = conn.execute(
        "SELECT COUNT(*) FROM pages WHERE document_id = ? AND ocr_text_chars > 0", (document_id,)
    ).fetchone()[0]
    conn.close()

    print(f"Indexed: {pdf_path}")
    print(f"Pages: {page_count}")
    print(f"Drawings: {drawing_count}")
    print(f"OCR pages: {ocr_count}")
    print(f"SQLite: {args.db}")
    if args.jsonl:
        print(f"JSONL: {args.jsonl}")
    if args.report:
        print(f"Report: {args.report}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Index architecture drawing PDFs into SQLite.")
    parser.add_argument("input", help="Input PDF path")
    parser.add_argument("--db", default="zumen_index.db", help="Output SQLite database path")
    parser.add_argument("--jsonl", help="Optional JSONL export path")
    parser.add_argument("--report", help="Optional per-page readability report path")
    parser.add_argument("--ocr", choices=("off", "auto", "force"), default="auto", help="OCR mode")
    parser.add_argument("--ocr-lang", default="jpn+eng", help="Tesseract language setting")
    parser.add_argument("--ocr-dpi", type=int, default=220, help="DPI for OCR page rendering")
    parser.add_argument("--max-pages", type=int, help="Limit pages for experiments")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing DB")
    parser.add_argument("--verbose", action="store_true", help="Print per-page progress")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        index_pdf(args)
    except subprocess.CalledProcessError as exc:
        print(exc.stderr or exc.stdout or str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
