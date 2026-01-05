# core/prepare_runner.py
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

from core.pdf_text import extract_pdf_text
from core.page_render import render_page_pngs


def run_prepare(
    pdf_path: Path,
    pdf_id: str,
    out_dir: Path,
    dpi: int = 150,
) -> Dict[str, Any]:
    """
    Prepare pipeline (local-only, stable):
    1) pages_text.json 생성 (PyMuPDF 텍스트)
    2) pages PNG 렌더링 (MM 입력용 / 디버깅용)

    NOTE:
    - 멀티모달 호출(표 탐지/표 추출)은 여기서 절대 하지 않는다.
    - MM 단계는 run_table_presence.py / run_table_extract_mm.py 같은 별도 runner에서 수행한다.
    """
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) text extract
    pages_text_path = extract_pdf_text(
        pdf_path=pdf_path,
        pdf_id=pdf_id,
        out_dir=out_dir,
    )

    # 2) render images
    img_dir = out_dir / "pages_png"
    png_paths = render_page_pngs(
        pdf_path=pdf_path,
        out_dir=img_dir,
        dpi=dpi,
    )

    # Optional: local-only status file (NOT MM result)
    local_status = {
        "pdf_id": pdf_id,
        "page_count": len(png_paths),
        "dpi": dpi,
        "updated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "pages_png_dir": str(img_dir.relative_to(out_dir)),
        "pages_text": str(Path(pages_text_path).relative_to(out_dir)),
    }
    local_status_path = out_dir / "prepare_status.json"
    tmp_path = out_dir / "prepare_status.json.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(local_status, f, ensure_ascii=False, indent=2)
        f.flush()
    tmp_path.replace(local_status_path)

    return {
        "pages_text": str(Path(pages_text_path).resolve()),
        "pages_png_dir": str(img_dir.resolve()),
        "prepare_status": str(local_status_path.resolve()),
        "page_count": len(png_paths),
        "dpi": dpi,
    }


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description="Prepare 단계(로컬-only): PDF 텍스트 추출 + 페이지 PNG 렌더"
    )
    ap.add_argument("--pdf_path", default="data/Ch6.pdf", help="입력 PDF 경로")
    ap.add_argument("--pdf_id", default="lecture", help="PDF ID")
    ap.add_argument("--out_dir", default="artifacts/lecture", help="산출물 디렉토리")
    ap.add_argument("--dpi", type=int, default=150, help="렌더 DPI")
    ap.add_argument("--print_json", action="store_true", help="결과 dict를 JSON으로 stdout 출력")

    args = ap.parse_args(argv)

    result = run_prepare(
        pdf_path=Path(args.pdf_path),
        pdf_id=args.pdf_id,
        out_dir=Path(args.out_dir),
        dpi=args.dpi,
    )

    # n8n/app.py에서 subprocess로 실행했을 때 stdout으로 결과를 받고 싶으면 유용
    if args.print_json:
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

