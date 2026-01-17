# core/question_orchestrator.py
"""
문제 배분 오케스트레이터 (최적화 버전)

최적화:
- Phase 1: LLM 요약 → TF-IDF 기반 키워드 추출 (0 API 호출)
- Phase 2: LLM 배분 (1회 호출 유지)
- Phase 3: 통계적 검증 (0 API 호출)

결과: N개 섹션에서 N+1 호출 → 1회 호출 (70%+ 비용 절감)
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set
from pathlib import Path

from core.llm_text import call_llm_text


# =============================================================================
# TF-IDF 기반 키워드 추출 (로컬 처리)
# =============================================================================

# 불용어 (한국어 + 영어)
STOPWORDS_KO = {
    "이", "그", "저", "것", "수", "등", "및", "를", "을", "의", "가", "에",
    "는", "은", "로", "으로", "에서", "과", "와", "하다", "되다", "있다",
    "위해", "통해", "대해", "대한", "같은", "다른", "모든", "각", "해당",
    "경우", "때문", "따라", "위한", "위하여", "대하여", "관한", "있는",
    "없는", "하는", "되는", "하여", "되어", "한다", "된다", "합니다",
    "됩니다", "입니다", "습니다", "니다", "다음", "이번", "지난", "이후",
}

STOPWORDS_EN = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "and", "or", "but",
    "if", "then", "else", "when", "where", "what", "which", "who", "whom",
    "this", "that", "these", "those", "it", "its", "of", "to", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "under", "again",
    "further", "once", "here", "there", "all", "each", "few", "more", "most",
    "other", "some", "such", "no", "not", "only", "own", "same", "so", "than",
    "too", "very", "just", "also", "now", "how", "any", "both", "each",
}

STOPWORDS = STOPWORDS_KO | STOPWORDS_EN


def _tokenize(text: str) -> List[str]:
    """
    간단한 토크나이저
    - 영문: 소문자로 분리
    - 한글: 2-6자 단어 추출
    """
    tokens = []

    # 영문 토큰
    en_words = re.findall(r'[a-zA-Z]{2,}', text.lower())
    tokens.extend([w for w in en_words if w not in STOPWORDS_EN and len(w) > 2])

    # 한글 토큰 (2-6자)
    ko_words = re.findall(r'[가-힣]{2,6}', text)
    tokens.extend([w for w in ko_words if w not in STOPWORDS_KO])

    return tokens


def _compute_tf(tokens: List[str]) -> Dict[str, float]:
    """Term Frequency 계산"""
    counter = Counter(tokens)
    total = len(tokens) if tokens else 1
    return {word: count / total for word, count in counter.items()}


def _compute_idf(documents: List[List[str]]) -> Dict[str, float]:
    """Inverse Document Frequency 계산"""
    n_docs = len(documents)
    if n_docs == 0:
        return {}

    doc_freq: Dict[str, int] = {}
    for tokens in documents:
        unique_tokens = set(tokens)
        for token in unique_tokens:
            doc_freq[token] = doc_freq.get(token, 0) + 1

    # IDF = log(N / df) + 1
    return {
        word: math.log(n_docs / df) + 1
        for word, df in doc_freq.items()
    }


def _extract_keywords_tfidf(
    text: str,
    idf_scores: Dict[str, float],
    top_k: int = 10,
) -> List[str]:
    """TF-IDF 기반 키워드 추출"""
    tokens = _tokenize(text)
    tf = _compute_tf(tokens)

    # TF-IDF 점수 계산
    tfidf_scores = {
        word: tf_score * idf_scores.get(word, 1.0)
        for word, tf_score in tf.items()
    }

    # 상위 K개 키워드
    sorted_words = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
    return [word for word, score in sorted_words[:top_k]]


def _analyze_section_local(section: Dict[str, Any], idf_scores: Dict[str, float]) -> Dict[str, Any]:
    """
    로컬에서 섹션 분석 (LLM 호출 없음)

    추출 정보:
    - 키워드 (TF-IDF)
    - 통계 (길이, 표, 코드, 수식)
    """
    section_id = section.get("section_id", "unknown")
    title = section.get("title", "")
    text = section.get("text", "")
    tables = section.get("tables", [])

    # 통계 정보
    char_count = len(text)
    has_code = bool(re.search(r'```|def\s+\w+|class\s+\w+|function\s+\w+', text))
    has_math = bool(re.search(r'\$[^$]+\$|\\frac|\\sum|\\int|\\sqrt', text))
    num_tables = len(tables)

    # TF-IDF 키워드 추출
    keywords = _extract_keywords_tfidf(text, idf_scores, top_k=8)

    # 제목에서도 키워드 추출
    title_keywords = _extract_keywords_tfidf(title, idf_scores, top_k=3)

    # 키워드 병합 (제목 우선)
    all_keywords = list(dict.fromkeys(title_keywords + keywords))[:10]

    # 요약 생성 (로컬)
    summary = _generate_local_summary(title, all_keywords, has_code, has_math, num_tables)

    return {
        "section_id": section_id,
        "title": title,
        "summary": summary,
        "keywords": all_keywords,
        "stats": {
            "char_count": char_count,
            "num_tables": num_tables,
            "has_code": has_code,
            "has_math": has_math,
        }
    }


def _generate_local_summary(
    title: str,
    keywords: List[str],
    has_code: bool,
    has_math: bool,
    num_tables: int,
) -> str:
    """
    로컬에서 요약 생성 (템플릿 기반)
    """
    parts = []

    # 제목 기반
    if title:
        parts.append(title)

    # 키워드 추가
    if keywords:
        parts.append(f"핵심: {', '.join(keywords[:5])}")

    # 특성 추가
    features = []
    if has_code:
        features.append("코드 포함")
    if has_math:
        features.append("수식 포함")
    if num_tables > 0:
        features.append(f"표 {num_tables}개")

    if features:
        parts.append(f"({', '.join(features)})")

    return " ".join(parts)


def _batch_analyze_sections(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    모든 섹션 분석 (병렬 불필요 - 로컬 처리라 빠름)

    1. 전체 문서에서 IDF 계산
    2. 각 섹션별 TF-IDF 키워드 추출
    """
    # 1. 전체 문서의 토큰 수집 (IDF 계산용)
    all_token_lists = []
    for section in sections:
        text = section.get("text", "")
        tokens = _tokenize(text)
        all_token_lists.append(tokens)

    # 2. IDF 계산
    idf_scores = _compute_idf(all_token_lists)

    # 3. 각 섹션 분석
    summaries = []
    for section in sections:
        summary = _analyze_section_local(section, idf_scores)
        summaries.append(summary)

    return summaries


# =============================================================================
# Phase 2: 통합 판단 (단일 LLM 호출)
# =============================================================================

def _llm_allocate(summaries: List[Dict[str, Any]], total_questions: int) -> Optional[Dict[str, int]]:
    """
    요약된 섹션들을 LLM에게 보여주고 문제 개수 배분
    → 1회 호출로 전체 판단
    """
    # 프롬프트 구성
    section_lines = []
    for i, s in enumerate(summaries, 1):
        stats = s.get("stats", {})
        keywords = s.get("keywords", [])
        keywords_str = ", ".join(keywords[:5]) if keywords else "(없음)"

        section_lines.append(
            f"{i}. [{s['section_id']}] {s['title']}\n"
            f"   키워드: {keywords_str}\n"
            f"   (길이: {stats.get('char_count', 0):,}자, "
            f"표: {stats.get('num_tables', 0)}개, "
            f"코드: {'있음' if stats.get('has_code') else '없음'}, "
            f"수식: {'있음' if stats.get('has_math') else '없음'})"
        )

    sections_text = "\n\n".join(section_lines)

    max_per = min(15, total_questions // 2) if total_questions > 2 else total_questions

    prompt = f"""당신은 교육학 전문가입니다. 다음 강의 자료의 섹션별 정보를 보고, 총 {total_questions}개의 시험 문제를 각 섹션에 배분하세요.

{"="*60}
섹션 정보
{"="*60}
{sections_text}

{"="*60}
배분 기준
{"="*60}
1. 교육적 중요도 (핵심 개념, 이론)
2. 내용 깊이 (텍스트 길이)
3. 실습 가능성 (코드, 표)
4. 각 섹션 최소 1문제, 최대 {max_per}문제

{"="*60}
출력 형식 (JSON ONLY)
{"="*60}
{{
  "allocation": {{
    "섹션ID": 문제개수,
    ...
  }},
  "reasoning": "배분 근거를 1-2문장으로"
}}

지금 바로 JSON만 출력하세요."""

    try:
        response = call_llm_text(
            prompt=prompt,
            model="gpt-4o-mini",
            temperature=0.3,
        )

        # JSON 추출
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()

        data = json.loads(response)
        allocation = data.get("allocation", {})
        reasoning = data.get("reasoning", "")

        # 검증
        if not isinstance(allocation, dict):
            return None

        # section_id 매핑 확인
        result = {}
        for s in summaries:
            sid = s["section_id"]
            # 다양한 key 시도
            count = allocation.get(sid) or allocation.get(f"[{sid}]") or allocation.get(s["title"])
            if isinstance(count, (int, float)):
                result[sid] = int(count)

        if not result:
            return None

        print(f"✅ LLM 배분 근거: {reasoning}")
        return result

    except Exception as e:
        print(f"⚠️ LLM 배분 실패: {e}")
        return None


# =============================================================================
# Phase 3: 통계적 Fallback
# =============================================================================

def _statistical_allocate(summaries: List[Dict[str, Any]], total_questions: int) -> Dict[str, int]:
    """
    통계적 방법 (fallback)
    → LLM 실패 시 사용
    - 요청 수와 정확히 일치하도록 보장
    """
    if not summaries or total_questions <= 0:
        return {}

    weights = []
    for s in summaries:
        stats = s.get("stats", {})
        char_count = stats.get("char_count", 0)
        num_tables = stats.get("num_tables", 0)
        has_code = stats.get("has_code", False)
        has_math = stats.get("has_math", False)

        # 가중치 계산
        w = math.sqrt(max(char_count, 1))
        w += num_tables * 50
        w += 30 if has_code else 0
        w += 20 if has_math else 0
        weights.append(max(w, 1.0))

    total_weight = sum(weights)

    # ✅ 비율 기반 초기 배분 (최소값 0 허용)
    allocation = {}
    for s, w in zip(summaries, weights):
        count = int(total_questions * w / total_weight)  # floor
        allocation[s["section_id"]] = count

    # ✅ 합계 정확히 맞추기
    current_total = sum(allocation.values())
    diff = total_questions - current_total

    # 가중치 높은 순으로 정렬
    sorted_sections = sorted(summaries, key=lambda x: weights[summaries.index(x)], reverse=True)

    if diff > 0:
        # 부족하면 가중치 높은 섹션부터 추가
        for i in range(diff):
            sid = sorted_sections[i % len(sorted_sections)]["section_id"]
            allocation[sid] += 1
    elif diff < 0:
        # 초과하면 가중치 낮은 섹션부터 감소
        sorted_sections_asc = list(reversed(sorted_sections))
        for i in range(abs(diff)):
            sid = sorted_sections_asc[i % len(sorted_sections_asc)]["section_id"]
            if allocation[sid] > 0:
                allocation[sid] -= 1

    # ✅ 0인 섹션 제거
    allocation = {sid: count for sid, count in allocation.items() if count > 0}

    return allocation


def _validate_allocation(
    allocation: Dict[str, int],
    summaries: List[Dict[str, Any]],
    total_questions: int,
    min_per_section: int = 0,  # ✅ 기본값 0으로 변경
    max_per_section: int = 15,
) -> Dict[str, int]:
    """
    LLM 배분 결과 검증 및 보정
    - 요청 수와 정확히 일치하도록 보정
    """
    # 범위 제한 (max만 적용, min은 합계 조정 후 적용)
    for sid in allocation:
        allocation[sid] = min(max_per_section, max(0, allocation[sid]))

    # 합계 맞추기 (정확히 total_questions가 되도록)
    current = sum(allocation.values())
    diff = total_questions - current

    if diff != 0:
        # 통계적 가중치로 보정
        stats_alloc = _statistical_allocate(summaries, total_questions)
        sorted_sids = sorted(stats_alloc.keys(), key=lambda x: stats_alloc[x], reverse=True)

        iterations = 0
        max_iterations = abs(diff) * len(sorted_sids) + 100  # 무한루프 방지

        while diff != 0 and iterations < max_iterations:
            changed = False
            for sid in sorted_sids:
                if diff == 0:
                    break
                if sid in allocation:
                    if diff > 0 and allocation[sid] < max_per_section:
                        allocation[sid] += 1
                        diff -= 1
                        changed = True
                    elif diff < 0 and allocation[sid] > 0:  # ✅ min_per_section 대신 0
                        allocation[sid] -= 1
                        diff += 1
                        changed = True
            if not changed:
                break
            iterations += 1

    # ✅ 0인 섹션 제거 (Job Builder에서 불필요한 Job 생성 방지)
    allocation = {sid: count for sid, count in allocation.items() if count > 0}

    return allocation


# =============================================================================
# Public API
# =============================================================================

def orchestrate_question_allocation(
    sections: List[Dict[str, Any]],
    total_questions: int,
    use_llm: bool = True,
    max_workers: int = 4,  # 하위 호환성 유지 (사용 안함)
    cache_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    하이브리드 Orchestration (최적화 버전)

    변경사항:
    - Phase 1: LLM 요약 → TF-IDF 키워드 추출 (로컬)
    - Phase 2: LLM 배분 (1회 호출 유지)
    - Phase 3: 통계적 검증 (로컬)

    Args:
        sections: 섹션 데이터 (text, tables 포함)
        total_questions: 생성할 총 문제 개수
        use_llm: LLM 사용 여부
        max_workers: (사용 안함, 하위 호환성)
        cache_dir: 분석 캐시 디렉토리

    Returns:
        {
            "allocation": {"섹션ID": 문제개수, ...},
            "method": "llm" | "statistical",
            "summaries": [...],
            "total": 15
        }
    """
    print(f"🎯 Orchestrator 시작: {len(sections)}개 섹션, {total_questions}개 문제")

    # 캐시 확인
    summaries = None
    if cache_dir and (cache_dir / "summaries.json").exists():
        try:
            print("✅ 캐시된 분석 사용")
            summaries = json.loads((cache_dir / "summaries.json").read_text(encoding="utf-8"))
        except Exception as e:
            print(f"⚠️ 캐시 읽기 실패: {e}")
            summaries = None

    # Phase 1: 로컬 분석 (TF-IDF 기반)
    if summaries is None:
        print("📝 Phase 1: 섹션 분석 중 (TF-IDF)...")
        summaries = _batch_analyze_sections(sections)
        print(f"✅ {len(summaries)}개 섹션 분석 완료 (0 API 호출)")

        # 캐싱
        if cache_dir:
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                (cache_dir / "summaries.json").write_text(
                    json.dumps(summaries, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )
                print(f"💾 분석 캐시 저장: {cache_dir / 'summaries.json'}")
            except Exception as e:
                print(f"⚠️ 캐시 저장 실패: {e}")

    # Phase 2: LLM 판단 (선택적)
    allocation = None
    method = "statistical"

    if use_llm:
        print("🤖 Phase 2: LLM 기반 배분 중 (1회 호출)...")
        allocation = _llm_allocate(summaries, total_questions)

        if allocation:
            print("✅ LLM 배분 성공")
            method = "llm"

            # Phase 3: 검증 및 보정
            print("🔍 Phase 3: 검증 및 보정 중...")
            allocation = _validate_allocation(
                allocation,
                summaries,
                total_questions,
                min_per_section=1,
                max_per_section=min(15, max(3, total_questions // 2)),
            )
        else:
            print("⚠️ LLM 배분 실패 → 통계적 방법 사용")

    # Fallback: 통계적 방법
    if allocation is None:
        print("📊 통계적 방법으로 배분 중...")
        allocation = _statistical_allocate(summaries, total_questions)

    print(f"✅ 최종 배분: {allocation}")

    return {
        "allocation": allocation,
        "method": method,
        "summaries": summaries,
        "total": sum(allocation.values()),
    }
