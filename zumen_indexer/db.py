import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from zumen_indexer.parser import infer_discipline, normalize_spaces
from zumen_indexer.extractor import sha256_file


@dataclass
class PageResult:
    page_number: int
    embedded_text: str
    ocr_text: str
    best_text: str
    text_source: str
    needs_ocr: bool
    quality_score: float


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
