# core/page_render.py
from pathlib import Path
import fitz  # PyMuPDF

def render_page_pngs(
    pdf_path: Path,
    out_dir: Path,
    dpi: int = 150
) -> list[Path]:
    """
    Render every page of PDF to PNG images.
    Returns list of PNG paths in page order (0-based).
    """
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    png_paths: list[Path] = []
    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        p = out_dir / f"page_{page_index:03d}.png"
        pix.save(str(p))
        png_paths.append(p)

    return png_paths
