import pandas as pd
import csv
import time
from openai import OpenAI
from tqdm import tqdm
import os
import sys
import ast

# ✅ OpenAI GPT-5 API 설정
client = OpenAI(api_key="api_key")
MODEL_NAME = "gpt-5"

# ✅ 경로 설정
BASE_PATH = r"C:\Users\jjw02\Desktop\데이터분석프로그래밍"
MEDNLI_INPUT_FILENAME = "mednli_kor.csv"
TRUTHFULQA_INPUT_FILENAME = "TruthfulQA_kor.csv"
AI_NAME_FOR_FILE = "GPT-5"

# ✅ 번역 대상 지역
regions = {
    "제주": "jeju",
    "경상": "kyungsang",
    "전라": "jeonra",
    "충청": "choongchung"
}

# ✅ GPT-5 방언 번역 함수
def translate_dialects(text, region_name):
    if not text or str(text).strip() == "":
        return text
    system_prompt = (
        f"너는 {region_name} 방언 전문가야. "
        f"주어진 문장을 해당 지역 방언으로 자연스럽게 번역해. "
        f"단, 반드시 **번역된 문장 하나만 출력**하고 다른 설명은 절대 포함하지 마."
    )
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ]
            # ❌ temperature 제거 (GPT-5는 기본값 1만 허용)
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ {region_name} 방언 번역 오류 (텍스트: '{text[:30]}...'): {e}", file=sys.stderr)
        return f"[ERROR: {text[:50]}... | {e}]"

# ============================================================================
# 🩺 A. MedNLI 번역 처리
# ============================================================================
mednli_input_csv = os.path.join(BASE_PATH, MEDNLI_INPUT_FILENAME)
try:
    df_mednli = pd.read_csv(mednli_input_csv)
    if 'sentence1_ko' in df_mednli.columns:
        df_mednli['sentence1'] = df_mednli['sentence1_ko']
    elif 'sentence1' not in df_mednli.columns:
        raise ValueError("MedNLI 파일에 'sentence1' 또는 'sentence1_ko' 컬럼이 없습니다.")
except Exception as e:
    print(f"🚨 MedNLI 파일 로드 오류: {e}", file=sys.stderr)
    sys.exit(1)

for region_name, region_en in regions.items():
    print(f"\n======== 🌍 MedNLI {region_name} 방언 번역 시작 ========")
    output_filename = os.path.join(BASE_PATH, f"mednli_{region_en}_({AI_NAME_FOR_FILE}).csv")
    fieldnames = ["gold_label", f"sentence1_{region_en}", f"sentence2_{region_en}", "ai_answer", "result"]
    translated_results = []

    for _, row in tqdm(df_mednli.iterrows(), total=len(df_mednli), desc=f"➡️ MedNLI {region_name} 번역 중..."):
        dialect_translation = translate_dialects(row['sentence1'], region_name)
        translated_results.append({
            "gold_label": row["gold_label"],
            f"sentence1_{region_en}": dialect_translation,
            f"sentence2_{region_en}": dialect_translation,
            "ai_answer": "",
            "result": ""
        })
        time.sleep(1.2)  # API rate limit 보호

    try:
        with open(output_filename, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(translated_results)
        print(f"🎉 {region_name} 방언 파일 저장 완료 → {output_filename}")
    except Exception as e:
        print(f"🚨 CSV 저장 실패: {e}", file=sys.stderr)

# ============================================================================
# 🧠 B. TruthfulQA 번역 처리
# ============================================================================
truthfulqa_input_csv = os.path.join(BASE_PATH, TRUTHFULQA_INPUT_FILENAME)
required_tqa_cols = [
    'question', 'mc1_choice', 'mc1_label', 'mc2_choice',
    'mc2_label', 'ai_answer_mc1', 'mc1_result',
    'ai_answer_mc2', 'mc2_result'
]

try:
    df_tqa = pd.read_csv(truthfulqa_input_csv)
    if not all(col in df_tqa.columns for col in required_tqa_cols):
        raise ValueError("TruthfulQA 파일에 필수 컬럼이 누락되었습니다.")
except Exception as e:
    print(f"🚨 TruthfulQA 파일 로드 오류: {e}", file=sys.stderr)
    sys.exit(1)

for region_name, region_en in regions.items():
    print(f"\n======== 🌍 TruthfulQA {region_name} 방언 번역 시작 ========")
    output_filename = os.path.join(BASE_PATH, f"truthfulqa_{region_en}_({AI_NAME_FOR_FILE}).csv")
    fieldnames = [
        f"question_{region_en}", f"mc1_choice_{region_en}", "mc1_label",
        f"mc2_choice_{region_en}", "mc2_label",
        "ai_answer_mc1", "mc1_result", "ai_answer_mc2", "mc2_result"
    ]
    translated_results = []

    for _, row in tqdm(df_tqa.iterrows(), total=len(df_tqa), desc=f"➡️ TQA {region_name} 번역 중..."):
        # 질문 번역
        question_dialect = translate_dialects(row['question'], region_name)

        # mc1/mc2 선택지 리스트 변환
        try:
            mc1_list = ast.literal_eval(row['mc1_choice']) if isinstance(row['mc1_choice'], str) and row['mc1_choice'].startswith('[') else [row['mc1_choice']]
        except:
            mc1_list = [x.strip() for x in str(row['mc1_choice']).split(',') if x.strip()]
        try:
            mc2_list = ast.literal_eval(row['mc2_choice']) if isinstance(row['mc2_choice'], str) and row['mc2_choice'].startswith('[') else [row['mc2_choice']]
        except:
            mc2_list = [x.strip() for x in str(row['mc2_choice']).split(',') if x.strip()]

        # 각 선택지 번역
        mc1_translated = [translate_dialects(c, region_name) for c in mc1_list]
        mc2_translated = [translate_dialects(c, region_name) for c in mc2_list]

        translated_results.append({
            f"question_{region_en}": question_dialect,
            f"mc1_choice_{region_en}": mc1_translated,
            "mc1_label": row["mc1_label"],
            f"mc2_choice_{region_en}": mc2_translated,
            "mc2_label": row["mc2_label"],
            "ai_answer_mc1": row["ai_answer_mc1"],
            "mc1_result": row["mc1_result"],
            "ai_answer_mc2": row["ai_answer_mc2"],
            "mc2_result": row["mc2_result"],
        })
        time.sleep(1.2)

    try:
        with open(output_filename, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(translated_results)
        print(f"🎉 {region_name} TruthfulQA 방언 파일 저장 완료 → {output_filename}")
    except Exception as e:
        print(f"🚨 TruthfulQA CSV 저장 실패: {e}", file=sys.stderr)

print("\n\n✅ MedNLI 4개 + TruthfulQA 4개 번역 완료 (총 8개 파일 생성됨)")
