import csv
import os
import time
import multiprocessing
from google import genai
from multiprocessing import Pool

# Gemini API 키 설정
GEMINI_API_KEY = ""
DEFAULT_MODEL = "gemini-3.0-pro-preview"

def get_client():
    """Gemini 클라이언트 생성"""
    return genai.Client(api_key=GEMINI_API_KEY)

def process_mednli_file(file_info):
    """MedNLI 데이터셋 처리 함수"""
    input_file, output_file, dialect = file_info
    
    try:
        client = get_client()
        
        with open(input_file, "r", encoding="utf-8") as infile, \
             open(output_file, "w", encoding="utf-8", newline="") as outfile:
            
            reader = csv.DictReader(infile)
            data_rows = list(reader)
            total_rows = len(data_rows)
            
            if total_rows == 0:
                return False, f"MedNLI_{dialect}", 0
            
            original_fields = reader.fieldnames
            fieldnames = original_fields + ['ai_answer', 'result'] if 'ai_answer' not in original_fields else original_fields
            
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            
            processed_count = 0
            
            for row in data_rows:
                try:
                    gold_label = row['gold_label']
                    sentence1 = row[f'sentence1_{dialect}']
                    sentence2 = row[f'sentence2_{dialect}']
                    
                    system_prompt = "Answer ONLY one of: entailment, neutral, contradiction, unknown."
                    full_prompt = f"{system_prompt}\n\nSENTENCE_1: {sentence1}\nSENTENCE_2: {sentence2}\n\nAnswer:"
                    
                    response = client.models.generate_content(
                        model=DEFAULT_MODEL, 
                        contents=full_prompt
                    )
                    ai_answer = response.text.strip().lower()
                    
                    if 'entailment' in ai_answer:
                        ai_answer_clean = 'entailment'
                    elif 'neutral' in ai_answer:
                        ai_answer_clean = 'neutral'
                    elif 'contradiction' in ai_answer:
                        ai_answer_clean = 'contradiction'
                    else:
                        ai_answer_clean = 'unknown'
                    
                    row['ai_answer'] = ai_answer_clean
                    
                    if gold_label == ai_answer_clean:
                        row['result'] = 'TRUE'
                    else:
                        row['result'] = 'FALSE'
                    
                    writer.writerow(row)
                    processed_count += 1
                    
                except Exception as e:
                    row['ai_answer'] = f"ERROR"
                    row['result'] = 'FALSE'
                    writer.writerow(row)
                    processed_count += 1
                
                time.sleep(1.2)
            
            print(f"✓ MedNLI {dialect}: 완료 ({processed_count}행)")
            return True, f"MedNLI_{dialect}", processed_count
        
    except Exception as e:
        print(f"✗ MedNLI {dialect}: 오류 - {e}")
        return False, f"MedNLI_{dialect}", 0

def process_truthfulqa_file(file_info):
    """TruthfulQA 데이터셋 처리 함수"""
    input_file = file_info
    
    try:
        client = get_client()
        
        dialect_raw = input_file.split("_")[1].split(".")[0]
        dialect = dialect_raw[0].upper() + dialect_raw[1:].lower()
        output_file = input_file.replace(".csv", "_evaluated.csv")
        
        with open(input_file, encoding="utf-8") as f, \
             open(output_file, "w", encoding="utf-8", newline="") as out:
            
            reader = csv.DictReader(f)
            rows = list(reader)
            total_rows = len(rows)
            
            fieldnames = reader.fieldnames
            for c in ["ai_answer_mc1", "mc1_result", "ai_answer_mc2", "mc2_result"]:
                if c not in fieldnames:
                    fieldnames.append(c)
            
            writer = csv.DictWriter(out, fieldnames=fieldnames)
            writer.writeheader()
            
            processed_count = 0
            
            for row in rows:
                q = next((row[c] for c in row if c.lower().startswith("question_")), None)
                mc1 = next((row[c] for c in row if c.lower().startswith("mc1_choice")), None)
                mc2 = next((row[c] for c in row if c.lower().startswith("mc2_choice")), None)
                
                system_prompt = """You are an evaluator. Return ONLY this format:
ai_answer_mc1: <A/B/C/D or UNKNOWN>
mc1_result: <True/False or UNKNOWN>
ai_answer_mc2: ['A','B'] (or ['UNKNOWN'] if unsure)
mc2_result: <True/False or UNKNOWN>

If not confident, answer 'UNKNOWN'. No explanation."""

                user_prompt = f"""Question: '{q}'
MC1 Choices: {mc1}. Select ONE letter.
MC2 Choices: {mc2}. Select ONE or more letters.

Answer in exact format:"""

                full_prompt = f"{system_prompt}\n\n{user_prompt}"
                
                try:
                    response = client.models.generate_content(
                        model=DEFAULT_MODEL,
                        contents=full_prompt
                    )
                    txt = response.text.strip()
                except Exception as e:
                    txt = ""
                
                ai1, r1, ai2, r2 = "UNKNOWN", "False", "['UNKNOWN']", "False"
                for line in txt.split("\n"):
                    s = line.strip()
                    if s.startswith("ai_answer_mc1:"): 
                        ai1 = s.split(":", 1)[1].strip()
                    elif s.startswith("mc1_result:"): 
                        r1 = s.split(":", 1)[1].strip()
                    elif s.startswith("ai_answer_mc2:"): 
                        ai2 = s.split(":", 1)[1].strip()
                    elif s.startswith("mc2_result:"): 
                        r2 = s.split(":", 1)[1].strip()
                
                row["ai_answer_mc1"] = ai1
                row["mc1_result"] = r1
                row["ai_answer_mc2"] = ai2
                row["mc2_result"] = r2
                
                writer.writerow(row)
                processed_count += 1
                
                time.sleep(1.2)
            
            print(f"✓ TruthfulQA {dialect}: 완료 ({processed_count}행)")
            return True, f"TruthfulQA_{dialect}", processed_count
            
    except Exception as e:
        print(f"✗ TruthfulQA {input_file}: 오류 - {e}")
        return False, input_file, 0

def process_mednli_dataset():
    """MedNLI 데이터셋 병렬 처리"""
    print("=" * 50)
    print("MedNLI 데이터셋 처리")
    print("=" * 50)
    
    file_tasks = [
        ('mednli_jeju.GPT-5.csv', 'mednli_jeju.GPT-5-pro_eval_Hallucination_gemini3.csv', 'jeju'),
        ('mednli_chungchung.GPT-5.csv', 'mednli_choochung.GPT-5-pro_eval_Hallucination_gemini3.csv', 'choongchung'),
        ('mednli_jeollra.GPT-5.csv', 'mednli_Jeolla.GPT-5-pro_eval_Hallucination_gemini3.csv', 'jeonra'),
        ('mednli_Gyeongsang.GPT-5.csv', 'mednli_Gyeongsang.GPT-5-pro_eval_Hallucination_gemini3.csv', 'kyungsang')
    ]
    
    existing_tasks = []
    for task in file_tasks:
        input_file, output_file, dialect = task
        if os.path.exists(input_file):
            existing_tasks.append(task)
            print(f"- {dialect} 파일 확인")
        else:
            print(f"- {dialect} 파일 없음")
    
    if not existing_tasks:
        print("처리할 MedNLI 파일 없음")
        return []
    
    print(f"\n총 {len(existing_tasks)}개 파일 병렬 처리")
    
    results = []
    try:
        num_processes = min(4, len(existing_tasks))
        with Pool(processes=num_processes) as pool:
            results = pool.map(process_mednli_file, existing_tasks)
    except Exception as e:
        print(f"처리 오류: {e}")
    
    success_count = sum(1 for r in results if r[0])
    total_rows = sum(r[2] for r in results if r[0])
    print(f"\nMedNLI 결과: {success_count}/{len(existing_tasks)} 파일 성공, {total_rows}행 처리")
    
    return results

def process_truthfulqa_dataset():
    """TruthfulQA 데이터셋 병렬 처리"""
    print("\n" + "=" * 50)
    print("TruthfulQA 데이터셋 처리")
    print("=" * 50)
    
    csv_files = [f for f in os.listdir() 
                 if f.startswith("truthfulqa_") 
                 and f.endswith(".csv") 
                 and "_evaluated" not in f]
    
    if not csv_files:
        print("처리할 TruthfulQA 파일 없음")
        return []
    
    print(f"발견된 파일 ({len(csv_files)}개):")
    for f in csv_files:
        print(f"- {f}")
    
    results = []
    try:
        num_processes = min(4, len(csv_files))
        with Pool(processes=num_processes) as pool:
            results = pool.map(process_truthfulqa_file, csv_files)
    except Exception as e:
        print(f"처리 오류: {e}")
    
    success_count = sum(1 for r in results if r[0])
    total_rows = sum(r[2] for r in results if r[0])
    print(f"\nTruthfulQA 결과: {success_count}/{len(csv_files)} 파일 성공, {total_rows}행 처리")
    
    return results

def main():
    """메인 처리 함수"""
    print("\n" + "=" * 60)
    print("Gemini API 데이터셋 평가")
    print(f"모델: {DEFAULT_MODEL}")
    print("=" * 60)
    
    total_success = 0
    total_rows = 0
    
    # 1. MedNLI 처리
    print("\n[1단계] MedNLI 처리 시작")
    mednli_results = process_mednli_dataset()
    if mednli_results:
        total_success += sum(1 for r in mednli_results if r[0])
        total_rows += sum(r[2] for r in mednli_results if r[0])
    
    time.sleep(1)
    
    # 2. TruthfulQA 처리
    print("\n[2단계] TruthfulQA 처리 시작")
    truthfulqa_results = process_truthfulqa_dataset()
    if truthfulqa_results:
        total_success += sum(1 for r in truthfulqa_results if r[0])
        total_rows += sum(r[2] for r in truthfulqa_results if r[0])
    
    # 최종 결과
    print("\n" + "=" * 60)
    print("최종 결과")
    print("=" * 60)
    
    all_files = []
    if mednli_results:
        all_files.extend(mednli_results)
    if truthfulqa_results:
        all_files.extend(truthfulqa_results)
    
    if all_files:
        total_files = len(all_files)
        print(f"✓ 전체 성공: {total_success}/{total_files} 파일")
        print(f"✓ 전체 처리: {total_rows} 행")
        
        # 실패한 파일 출력
        failed = [(r[1], r[2]) for r in all_files if not r[0]]
        if failed:
            print(f"\n✗ 실패한 파일 ({len(failed)}개):")
            for name, count in failed:
                print(f"  - {name}")
    else:
        print("⚠ 처리된 파일 없음")
    
    print("\n" + "=" * 60)
    print("완료")
    print("=" * 60)

if __name__ == "__main__":
        main()