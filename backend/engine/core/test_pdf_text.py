# core/test_pdf_text.py
from pathlib import Path
import os

from core.pdf_text import extract_pdf_text

def main():
    print("=== TEST pdf_text ===")
    print("CWD:", os.getcwd())

    pdf_path = Path("data/Ch6.pdf")
    out_dir = Path("artifacts/lecture")

    print("pdf_path:", pdf_path.resolve())
    print("pdf_exists:", pdf_path.exists())
    if pdf_path.exists():
        print("pdf_size:", pdf_path.stat().st_size, "bytes")

    print("out_dir:", out_dir.resolve())

    print("\nSTART extract_pdf_text()")
    out = extract_pdf_text(
        pdf_path=pdf_path,
        pdf_id="lecture",
        out_dir=out_dir
    )
    print("DONE extract_pdf_text()")

    print("\nreturned:", out)

    out_path = Path(out) if out else None
    if out_path and out_path.exists():
        print("saved_to:", out_path.resolve())
        print("json_size:", out_path.stat().st_size, "bytes")
    else:
        print("❌ output file does not exist (or extract_pdf_text returned None)")

if __name__ == "__main__":
    main()




