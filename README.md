# Measuring Hallucination in Korean Dialects

한국어 방언(표준어, 충청도, 전라도, 경상도, 제주도) 사용 시 발생하는 대규모 언어 모델(LLM)의 환각(Hallucination) 현상을 측정하고 시각화하는 프로젝트입니다.

## 📂 프로젝트 구조

- **dataset/**: 실험에 사용된 TruthfulQA 및 MedNLI 데이터셋 (한국어 방언 번역본 포함)
- **manim_data_visualize/**: Manim 라이브러리를 이용한 데이터 시각화 코드
  - `visualize_hallucination.py`: 메인 시각화 스크립트 (Bubble Map, Radar Chart, Scatter Plot)
  - `csv_data/`: 시각화에 사용되는 정확도 및 환각 수치 CSV 파일
  - `media/`: 렌더링된 동영상 파일이 저장되는 경로

## 📊 시각화 내용

이 프로젝트는 `Manim`을 사용하여 다음과 같은 분석 결과를 시각화합니다:

1.  **Bubble Map (지도 시각화)**
    *   한국 지도 위에 지역별 환각 점수를 버블 크기로 표현
    *   버블이 클수록 해당 지역 방언에서의 모델 성능 저하/환각 증가를 의미
    *   사용 데이터: TruthfulQA, MedNLI

2.  **Radar Chart (레이더 차트)**
    *   모델(GPT 5.1, Claude 4.5 Sonnet, Gemini 3) 간의 지역별 성능 비교
    *   중심에 가까울수록 환각이 적고(고성능), 외곽으로 갈수록 환각이 많음(저성능)

3.  **Scatter Plot (산점도)**
    *   Capability (정확도) vs Attitude (신뢰성/환각 점수) 상관관계 분석
    *   **Reliable Zone**: 정확도가 높고 환각이 적은 이상적인 구간
    *   **Overconfident Zone**: 정확도는 높으나 환각이 많은(자신감 과잉) 구간

## 🚀 실행 방법

### 1. 환경 설정
필요한 Python 패키지를 설치합니다. (Manim 설치 필수)
```bash
pip install manim pandas numpy
```

### 2. 시각화 생성
`manim_data_visualize` 디렉토리로 이동하여 다음 명령어를 실행합니다.

**전체 프레젠테이션 생성 (High Quality)**
```bash
cd manim_data_visualize
manim -qh visualize_hallucination.py FullPresentation
```

**개별 씬(Scene) 생성 예시**
```bash
# TruthfulQA 버블 맵
manim -qh visualize_hallucination.py Scene1_TruthfulQA_Bubbles

# MedNLI 레이더 차트
manim -qh visualize_hallucination.py Scene4_MedNLI_Radar
```

## 📝 데이터 채점 기준

*   **TruthfulQA & MedNLI 공통**:
    *   ✅ **정답**: +1점
    *   ❓ **모름/회피**: 0점
    *   ❌ **오답(환각)**: -1점
    *   **환각 수 계산**: `총 문제 수 - 획득 점수` (점수가 낮을수록 환각이 많은 것으로 간주)

---
*Created by Antigravity*
