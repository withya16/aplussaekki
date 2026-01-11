# core/test_prepare.py
from pathlib import Path
from core.prepare_runner import run_prepare

def main():
    out = run_prepare(
        pdf_path=Path("data/Ch6.pdf"),
        pdf_id="lecture",
        out_dir=Path("artifacts/lecture"),
        dpi=150
    )
    print("PREPARE DONE")
    for k, v in out.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
