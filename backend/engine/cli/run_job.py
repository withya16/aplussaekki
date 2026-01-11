"""
전체 파이프라인 실행
python run_full.py
"""
import subprocess
import sys
import json
from pathlib import Path

def run_step(name, cmd, job_store=None, job_id=None, stage=None, step_num=0, total_steps=0):
    """단계 실행 및 진행 상황 업데이트"""
    print(f"\n{'='*60}")
    print(f"▶ {name}")
    print(f"{'='*60}")
    
    # 진행 상황 업데이트 (시작)
    if job_store and job_id and stage:
        try:
            job_store.update_job_progress(job_id, stage, step_num, total_steps)
        except Exception as e:
            print(f"⚠️  진행 상황 업데이트 실패: {e}")
    
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ {name} 실패!")
        # 실패 시 Job 상태 업데이트
        if job_store and job_id:
            try:
                job_store.fail_job(job_id, "STEP_FAILED", f"{name} 단계 실패")
            except:
                pass
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
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_id", required=True, help="PDF ID")
    parser.add_argument("--job_id", required=True, help="Job ID")
    args = parser.parse_args()
    
    base = Path(__file__).parent.parent.parent  # backend/ 디렉토리
    pdf_path = str(base / "data" / "pdfs" / f"{args.pdf_id}.pdf")
    pdf_id = args.pdf_id
    job_id = args.job_id
    out_dir = base / "artifacts" / pdf_id
    
    # sys.path 설정 (JobStore import용)
    import sys
    sys.path.insert(0, str(base))
    from app.storage.job_store import JobStore
    
    try:
        total_steps = 8  # 전체 단계 수
        
        # 1. Prepare (PARSING)
        run_step("1. PDF 준비", 
            f'python -m core.prepare_runner --pdf_path "{pdf_path}" --pdf_id {pdf_id} --out_dir "{out_dir}"',
            job_store=JobStore, job_id=job_id, stage="PARSING", step_num=1, total_steps=total_steps)
        
        # 2. Table Presence (PARSING)
        run_step("2. 표 존재 확인",
            f'python -m core.run_table_presence --out_dir "{out_dir}" --pdf_id {pdf_id}',
            job_store=JobStore, job_id=job_id, stage="PARSING", step_num=2, total_steps=total_steps)
        
        # 3. Table Extract (PARSING)
        run_step("3. 표 추출",
            f'python -m core.run_table_extract_mm --out_dir "{out_dir}" --pdf_id {pdf_id}',
            job_store=JobStore, job_id=job_id, stage="PARSING", step_num=3, total_steps=total_steps)
        
        # 4. Section Indexer (PARSING)
        run_step("4. 섹션 인덱싱",
            f'python -m core.section_indexer --out_dir "{out_dir}" --pdf_id {pdf_id}',
            job_store=JobStore, job_id=job_id, stage="PARSING", step_num=4, total_steps=total_steps)
        
        # 5. Section Packager (PARSING)
        run_step("5. 섹션 패키징",
            f'python -m core.section_context_packager --out_dir "{out_dir}" --pdf_id {pdf_id}',
            job_store=JobStore, job_id=job_id, stage="PARSING", step_num=5, total_steps=total_steps)
        
        # 6. Job Builder (PARSING)
        run_step("6. Job 빌드",
            f'python -m core.job_builder --out_dir "{out_dir}" --pdf_id {pdf_id}',
            job_store=JobStore, job_id=job_id, stage="PARSING", step_num=6, total_steps=total_steps)
        
        # 7. Question Pipeline (GENERATING/VERIFYING)
        JobStore.update_job_progress(job_id, "GENERATING", 7, total_steps)
        run_step("7. 문제 생성",
            f'python -m core.run_question_pipeline --out_dir "{out_dir}" --pdf_id {pdf_id}',
            job_store=None, job_id=None, stage=None, step_num=0, total_steps=0)  # run_question_pipeline 내부에서 상태 업데이트
        
        # 8. API 명세 형식 변환 (SAVING)
        print("\n" + "="*60)
        print("▶ 8. API 명세 형식 변환")
        print("="*60)
        JobStore.update_job_progress(job_id, "SAVING", 8, total_steps)
        q_count, output_path = convert_to_spec(pdf_id, out_dir)
        print(f"✅ 변환 완료: {q_count}개 문제")
        
        # Job 완료 상태 업데이트
        JobStore.complete_job(job_id)
        print(f"✅ Job 상태 업데이트 완료: {job_id}")
        
        print("\n" + "="*60)
        print("🎉 전체 파이프라인 완료!")
        print(f"📁 원본: {out_dir}/questions_verified_aggregate.json")
        print(f"📁 명세: {output_path}")
        print("="*60)
    except Exception as e:
        # Job 실패 상태 업데이트
        import sys
        sys.path.insert(0, str(base))
        from app.storage.job_store import JobStore
        try:
            JobStore.fail_job(job_id, "PIPELINE_ERROR", str(e))
            print(f"❌ Job 실패 상태 업데이트: {job_id}")
        except:
            pass
        raise
