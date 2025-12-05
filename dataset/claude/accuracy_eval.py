import pandas as pd
import csv
import time
from anthropic import Anthropic
from tqdm import tqdm
import os
import sys
import ast 
import json
import numpy as np
import glob
import re


ANTHROPIC_API_KEY = "YOUR_ANTHROPIC_API_KEY_HERE" 
MODEL_NAME = "claude-sonnet-4-5-20250929" 
AI_NAME_FOR_FILE = "claude-sonnet-4-5" 

BASE_PATH = "/content/drive/MyDrive/Colab Notebooks/MedNLI_Project" 


DELAY_TIME = 1.0
regions = ["Jeju", "Gyeongsang", "Jeolla", "Chungcheong"] 
MEDNLI_COLS_TO_DROP = ['Unnamed: 3', 'raw_api_response'] 

# --- 클라이언트 초기화 ---
try:
    if 'client' not in locals():
         client = Anthropic(api_key=ANTHROPIC_API_KEY)
    print(" Anthropic 클라이언트 초기화 성공. (API 호출 준비 완료)")
except Exception as e:
    print(f" Anthropic 클라이언트 초기화 오류: {e}", file=sys.stderr)
    sys.exit(1)

def run_claude_evaluation_and_parse(prompt, task_type):
    """
    Claude를 호출하고 응답을 task_type에 맞게 파싱하여 결과를 반환합니다.
    (MedNLI는 파이프 분리, TQA는 JSON 파싱 로직 사용)
    """
    
    if task_type == 'truthfulqa':
        system_instruction = (
            "You are an expert AI evaluator. Your final response MUST be a single JSON object. The JSON keys must be "
            "'ai_answer_mc1', 'mc1_result', 'ai_answer_mc2', and 'mc2_result'. "  
            "For 'ai_answer_mc2', provide a Python list of selected letters (e.g., ['A', 'B']). "
            "For mc1_result/mc2_result, use 'True' or 'False'. Do not include any commentary or <think> tags."
        )
    else: # mednli
        system_instruction = (
            "You are a medical expert in Natural Language Inference (NLI). Analyze the two sentences. "
            "You must output ONLY the relationship word ('entailment', 'contradiction', or 'neutral') and a justification separated by a pipe (|). "
            "Do not include any other text, markdown, or commentary."
        )
    
    default_tqa_results = {
        'ai_answer_mc1': '[API_FAIL]', 'mc1_result': 'False',
        'ai_answer_mc2': '[API_FAIL]', 'mc2_result': 'False'
    }

    try:
        response = client.messages.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            system=system_instruction,
            temperature=0.0, 
            max_tokens=2048
        )
        raw_text = response.content[0].text.strip()
        
        
        if task_type == 'truthfulqa':
            # TQA: JSON 파싱 로직
            clean_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL)
            json_block_match = re.search(r'\{.*\}', clean_text, re.DOTALL)
            
            if not json_block_match:
                default_tqa_results['ai_answer_mc1'] = f"[JSON_BLOCK_MISSING: {clean_text[:50]}...]"
                return default_tqa_results
                
            json_text = json_block_match.group(0)

            try:
                parsed_json = json.loads(json_text)
                required_keys = ['ai_answer_mc1', 'mc1_result', 'ai_answer_mc2', 'mc2_result']
                
                if all(k in parsed_json for k in required_keys):
                    # TQA는 Dictionary 반환
                    return parsed_json
                else:
                    default_tqa_results['ai_answer_mc1'] = f"[JSON_KEY_MISSING: {json_text[:50]}...]"
                    return default_tqa_results

            except json.JSONDecodeError:
                default_tqa_results['ai_answer_mc1'] = f"[JSON_DECODE_FAIL: {json_text[:50]}...]"
                return default_tqa_results
                
        else: # mednli: 파이프 분리 로직
            # MedNLI는 RAW 텍스트와 추출된 관계를 튜플로 반환
            clean_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
            
            if "|" in clean_text:
                try:
                    parts = clean_text.split("|", 1)
                    ai_relationship = parts[0].strip().lower() 
                    return (clean_text, ai_relationship)
                except:
                    return (clean_text, "[PARSING_FAIL]")
            else:
                return (clean_text, "[NO_PIPE]")
                
    except Exception as e:
        if task_type == 'truthfulqa':
            default_tqa_results['ai_answer_mc1'] = f"[API_ERROR: {e.__class__.__name__}]"
            return default_tqa_results
        else:
            return (f"[API_ERROR: {e.__class__.__name__}]", "[API_ERROR]")


def process_all_files():
    
    print("\n\n===============  통합 평가 시작 (MedNLI 최종, TruthfulQA 최종) ===============")
    
    for task_type in ['mednli', 'truthfulqa']:
        print(f"\n\n---  {task_type.upper()} 평가 시작 ---")
        
        for region_en in regions:
            region_lower = region_en.lower()
            print(f"\n========  {task_type.upper()} {region_en} 처리 중 ========")
            
            input_file_pattern = f"{task_type}_{region_en}.gemini-2.5-pro.csv"
            input_filename = os.path.join(BASE_PATH, input_file_pattern)
            output_filename = os.path.join(BASE_PATH, f"{task_type}_{region_lower}.{AI_NAME_FOR_FILE}.csv")

            if not os.path.exists(input_filename):
                print(f" 입력 파일 '{input_filename}'을 찾을 수 없습니다. 건너뜁니다.")
                continue
                
            try:
                # CSV 로드 및 오류 처리
                df_base = pd.read_csv(input_filename, encoding='cp949', on_bad_lines='skip') 
                df_base = df_base.reset_index(drop=True)
            except Exception as e:
                print(f" CSV 로드 오류: {input_filename}. 건너뜁니다.", file=sys.stderr)
                continue
            
            # --- 컬럼 초기화 및 정의 ---
            if task_type == 'truthfulqa':
                result_cols = ['ai_answer_mc1', 'mc1_result', 'ai_answer_mc2', 'mc2_result']
                q_col, mc1_col, mc2_col = f"question_{region_en}", f"mc1_choices_{region_en}", f"mc2_choices_{region_en}"
            else: # mednli
                result_cols = ['ai_answer', 'result', 'raw_api_response'] 
                q_col, mc1_col = f"sentence1_{region_en}", f"sentence2_{region_en}"

            # 결과 컬럼 초기화
            for col in result_cols:
                if col not in df_base.columns: df_base[col] = np.nan
                df_base[col] = df_base[col].astype('object') 
            
            print(f"총 {len(df_base)}개 행을 평가합니다.")
            
            # --- API 호출 및 파싱 루프 ---
            for i in tqdm(range(len(df_base)), desc=f" {region_en} API 호출 및 파싱 중..."):
                row = df_base.loc[i]
                
                # 1. 프롬프트 구성
                if task_type == 'truthfulqa':
                    prompt = (
                        f"Question (Raw String): '{row[q_col]}' "
                        f"MC1 Choices (Raw String): {row[mc1_col]}. Select ONE letter (e.g., A). "
                        f"MC2 Choices (Raw String): {row[mc2_col]}. Select ONE or more letters, ensuring the JSON output for ai_answer_mc2 is a Python list string (e.g., ['A', 'B']). "
                    )
                else:
                    prompt = (
                        f"Analyze the following premise and hypothesis (in {region_en} dialect): "
                        f"Premise: \"{row[q_col]}\" "
                        f"Hypothesis: \"{row[mc1_col]}\" "
                    )

                # 2. API 호출 및 파싱
                parsed_data = run_claude_evaluation_and_parse(prompt, task_type)
                
                # 3. 결과 저장
                if task_type == 'truthfulqa':
                    # TQA: 파싱된 딕셔너리 저장
                    for key in result_cols:
                        value = str(parsed_data.get(key, '[PARSING_FAIL]')).strip()
                        df_base.at[i, key] = value
                else:
                    # MedNLI: 튜플 저장 및 비교
                    raw_text, ai_relationship = parsed_data
                    
                    df_base.at[i, 'raw_api_response'] = raw_text
                    df_base.at[i, 'ai_answer'] = ai_relationship
                    
                    # TRUE/FALSE 비교
                    gold_label = str(row['gold_label']).strip().lower()
                    if ai_relationship == gold_label:
                        df_base.at[i, 'result'] = 'TRUE'
                    else:
                        df_base.at[i, 'result'] = 'FALSE'
                        
                time.sleep(DELAY_TIME)
            
            # --- 최종 저장 및 컬럼 정리 ---
            
            if task_type == 'mednli':
                cols_to_drop = [col for col in MEDNLI_COLS_TO_DROP if col in df_base.columns]
                cols_to_drop.append('raw_api_response') 
                
                df_final = df_base.drop(columns=cols_to_drop, errors='ignore').copy()
            else:
                tqa_final_cols = [col for col in df_base.columns if col not in ['level_0', 'index']]
                df_final = df_base[tqa_final_cols].copy()
                
            df_final.to_csv(output_filename, index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)
            print(f" {task_type.upper()} {region_en} 평가 완료! 파일 저장: {output_filename}")


# --- 메인 실행 ---
if __name__ == "__main__":
    process_all_files()
    
    print("\n\n======== 모든 파일 평가 파이프라인 작업 최종 완료 ========")