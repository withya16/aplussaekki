"""
전체 파이프라인 실행
python run_full.py
"""
import subprocess
import sys
import json
from pathlib import Path

def run_step(name, cmd):
    print(f"\n{'='*60}")
    print(f"▶ {name}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ {name} 실패!")
        sys.exit(1)
    print(f"✅ {name} 완료")

def convert_to_spec(pdf_id: str, out_dir: Path):
    """명세 형식으로 변환"""
    input_path = out_dir / "questions_verified_aggregate.json"
    output_path = Path("data") / "results" / f"{pdf_id}.questions.json"
    
    data = json.loads(input_path.read_text(encoding="utf-8"))
    
    questions = []
    q_counter = 1
    
    for item in data.get("items", []):
        for q in item.get("questions", []):
            if q.get("verdict") != "OK":
                continue
            
            converted = {
                "question_id": f"q_{q_counter:03d}",
                "type": q["type"],
                "difficulty": q["difficulty"],
                "verdict": q["verdict"],
                "question_text": q["question"],
                "answer": q["answer"],
                "explanation": q["explanation"]
            }
            
            if q["type"] == "MCQ":
                options = []
                for choice in q.get("choices", []):
                    text = choice.split(") ", 1)[1] if ") " in choice else choice
                    options.append(text)
                converted["options"] = options
            
            questions.append(converted)
            q_counter += 1
    
    result = {
        "pdf_id": pdf_id,
        "questions": questions
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    
    return len(questions), output_path

if __name__ == "__main__":
    base = Path(__file__).parent
    pdf_path = str(base / "data" / "Ch6.pdf")
    pdf_id = "Ch6"
    out_dir = base / "artifacts" / "Ch6"
    
    # 1. Prepare
    run_step("1. PDF 준비", 
        f'python -m core.prepare_runner --pdf_path "{pdf_path}" --pdf_id {pdf_id} --out_dir "{out_dir}"')
    
    # 2. Table Presence
    run_step("2. 표 존재 확인",
        f'python -m core.run_table_presence --out_dir "{out_dir}" --pdf_id {pdf_id}')
    
    # 3. Table Extract
    run_step("3. 표 추출",
        f'python -m core.run_table_extract_mm --out_dir "{out_dir}" --pdf_id {pdf_id}')
    
    # 4. Section Indexer
    run_step("4. 섹션 인덱싱",
        f'python -m core.section_indexer --out_dir "{out_dir}" --pdf_id {pdf_id}')
    
    # 5. Section Packager
    run_step("5. 섹션 패키징",
        f'python -m core.section_context_packager --out_dir "{out_dir}" --pdf_id {pdf_id}')
    
    # 6. Job Builder
    run_step("6. Job 빌드",
        f'python -m core.job_builder --out_dir "{out_dir}" --pdf_id {pdf_id}')
    
    # 7. Question Pipeline
    run_step("7. 문제 생성",
        f'python -m core.run_question_pipeline --out_dir "{out_dir}" --pdf_id {pdf_id}')
    
    # 8. API 명세 형식 변환
    print("\n" + "="*60)
    print("▶ 8. API 명세 형식 변환")
    print("="*60)
    q_count, output_path = convert_to_spec(pdf_id, out_dir)
    print(f"✅ 변환 완료: {q_count}개 문제")
    
    print("\n" + "="*60)
    print("🎉 전체 파이프라인 완료!")
    print(f"📁 원본: {out_dir}/questions_verified_aggregate.json")
    print(f"📁 명세: {output_path}")
    print("="*60)
