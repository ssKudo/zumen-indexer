import re
import argparse
import subprocess
from pathlib import Path

from zumen_indexer.config import TOC_KEYWORDS
from zumen_indexer.extractor import (
    HAS_PIL_NUMPY,
    get_pdf_info,
    extract_text_page,
    ocr_page,
    detect_vertical_columns,
    optional_tool,
)
from zumen_indexer.parser import (
    should_ocr,
    score_text_quality,
    extract_title_block_fields,
    chunk_text,
    detect_toc_pages,
    parse_toc_entries,
    dedupe_drawings,
)
from zumen_indexer.db import (
    connect_db,
    insert_document,
    PageResult,
    insert_page,
    insert_drawings,
    write_jsonl,
    write_report,
)


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
