#!/usr/bin/env python3
"""Index architecture drawing PDFs into SQLite."""

import argparse
import sys
import subprocess

# Ensure we can import zumen_indexer if scripts/ is run directly
import os
sys.path.insert(0, str(sys.path[0] + "/.."))

from zumen_indexer.core import index_pdf


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
