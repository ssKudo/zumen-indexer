import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

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
