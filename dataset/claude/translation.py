import os
import pandas as pd
import anthropic
import ast
from google.colab import drive
import time
from tqdm.notebook import tqdm

ANTHROPIC_API_KEY = "YOUR_ANTHROPIC_API_KEY" 
MODEL_NAME = "claude-sonnet-4-5-20250929"
BASE_PATH = "/content/drive/MyDrive/Colab Notebooks/Project"

# 처리할 지역 목록
REGIONS = ["Jeolla", "Chungcheong", "Jeju", "Gyeongsang"]

# 원본 파일 이름 정의
MEDNLI_SOURCE_FILE = 'mednli_kor.csv'
TRUTHFULQA_SOURCE_FILE = 'TruthfulQA_result-gpt4o-gpt4o.csv'

# 처리할 파일 목록 구성: 지역별로 두 파일을 모두 처리
FILE_INFOS = []
for region in REGIONS:
    # 1. TruthfulQA (원본 파일 이름 사용)
    FILE_INFOS.append({
        "source_path": os.path.join(BASE_PATH, TRUTHFULQA_SOURCE_FILE),
        "type": "truthfulqa",
        "region": region,
        "dialect": f"{region} 방언",
        "target_cols": ["question", "answer", "mc1_choice", "mc2_choice"],
        "output_suffix": f"_{region.lower()}_({MODEL_NAME.replace('claude-', '').replace('-', ' ').replace('_', ' ').title()})"
    })
    # 2. MedNLI (원본 파일 이름 사용)
    FILE_INFOS.append({
        "source_path": os.path.join(BASE_PATH, MEDNLI_SOURCE_FILE),
        "type": "mednli",
        "region": region,
        "dialect": f"{region} 방언",
        "target_cols": ["sentence1", "sentence2"],
        "output_suffix": f"_{region}.{MODEL_NAME.replace('claude-', '')}"
    })


# Anthropic 클라이언트 초기화
try:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) 
    print("Anthropic 클라이언트 초기화 성공.")
except Exception as e:
    print(f"Anthropic 클라이언트 초기화 실패: {e}")
    # exit() 


# --- 2. 헬퍼 함수 정의 ---

def translate_text(text, model_name, region): 
    """Anthropic Claude API를 사용하여 텍스트를 재번역합니다. (출력 형태 엄격히 제한 및 영어 번역 설명 금지)"""
    
    # 프롬프트 정의
    user_messages_base = {
        "Jeju": "다음 문장을 제주도 방언으로 자연스럽게 번역해줘, 만약 전문 언어라 해석이 어렵다면 영어로 남겨줘",
        "Gyeongsang": "다음 문장을 경상도 방언으로 자연스럽게 번역해줘, 만약 전문 언어라 해석이 어렵다면 영어로 남겨줘",
        "Jeolla": "다음 문장을 전라도 사투리로 자연스럽게 번역해줘, 만약 전문 언어라 해석이 어렵다면 영어로 남겨줘",
        "Chungcheong": "다음 문장을 충청도 사투리로 자연스럽게 번역해줘. 만약 전문 언어라 해석이 어렵다면 영어로 남겨줘"
    }
    
    system_message_template = (
        "너는 {region} 방언 전문가야. 이제부터 문장이 주어지면 해당 지역 방언으로 정확하게 번역해야 해. "
        "번역 결과는 **오직 한 문장**만 출력해야 하며, 방언으로 번역하든, 전문 언어라 **영어로 번역하든 관계없이,** "
        "결과물 외의 **다른 모든 설명, 기호, 괄호, 줄바꿈은 절대 포함하지마.** "
        "예시: '영어로 번역하면~', '번역 결과:', '결과:' 등의 문구를 절대로 추가하지마."
    )
    
    system_message = {
        "Jeju": system_message_template.format(region="제주도"),
        "Gyeongsang": system_message_template.format(region="경상도"),
        "Jeolla": system_message_template.format(region="전라도"),
        "Chungcheong": system_message_template.format(region="충청도")
    }
    
    if not text or str(text).strip() == "":
        return ""
    
    if region not in user_messages_base:
        return None

    system_prompt = system_message[region]
    user_prompt_content = f"{user_messages_base[region]}\n{text}"
    
    try:
        response = client.messages.create(
            model=model_name,
            max_tokens=2048,
            system=system_prompt, 
            messages=[
                {"role": "user", "content": user_prompt_content}
            ]
        )
        if response.content and response.content[0].text:
            translated_text = response.content[0].text.replace('\n', ' ').strip()
            # " 또는 " 패턴 처리
            if ' 또는 ' in translated_text and translated_text.count(' 또는 ') == 1:
                 translated_text = translated_text.split(' 또는 ')[0].strip()
                 
            return translated_text
        return None
    except Exception as e:
        print(f"API 호출 중 예상치 못한 오류 발생: {e}")
        # 오류 발생 시 해당 항목 번역 건너뜀 (None 반환)
        return None


def process_file(file_info):
    """단일 파일을 처리하고, 번역한 후 새 파일을 저장합니다."""
    source_path = file_info['source_path']
    file_type = file_info['type']
    region = file_info['region']
    target_cols = file_info['target_cols']
    output_suffix = file_info['output_suffix']
    
    print(f"\n--- {file_type} ({region}) 파일 전체 번역 시작: {os.path.basename(source_path)} ---")

    if not os.path.exists(source_path):
        print(f"오류: 원본 파일을 찾을 수 없습니다. 경로를 확인하세요: {source_path}")
        return

    try:
        df = pd.read_csv(source_path)
    except Exception as e:
        print(f"CSV 파일 로드 중 오류 발생: {e}")
        return

    # 새로운 컬럼 생성 준비
    if file_type == "mednli":
        # MedNLI: sentence1_Jeolla, sentence2_Jeolla
        col_map = {col: f"{col}_{region}" for col in target_cols}
    elif file_type == "truthfulqa":
        # TruthfulQA: question_jeolla, answer_jeolla, mc1_choice_jeolla, mc2_choice_jeolla
        region_lower = region.lower()
        col_map = {col: f"{col}_{region_lower}" for col in target_cols}
    
    print(f"새 컬럼: {list(col_map.values())}")
    
    # 번역 루프
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"전체 번역 진행 ({region})"):
        for original_col, new_col in col_map.items():
            current_value = row.get(original_col) # 원본 컬럼에서 값 가져오기

            if pd.isna(current_value):
                df.loc[index, new_col] = current_value
                continue

            # TruthfulQA의 리스트 형태 컬럼 처리 (mc1_choice, mc2_choice)
            if file_type == "truthfulqa" and original_col.startswith("mc"):
                try:
                    # current_value가 문자열 리스트 형태인지 확인 후 처리
                    list_of_choices = ast.literal_eval(current_value) 
                    
                    new_list_of_choices = []
                    for choice in list_of_choices:
                        if isinstance(choice, str) and choice.strip():
                            translated_text = translate_text(choice, MODEL_NAME, region)
                            new_list_of_choices.append(translated_text if translated_text is not None else choice) # 실패 시 원본 유지
                            time.sleep(0.5) 
                        else:
                            new_list_of_choices.append(choice)
                            
                    df.loc[index, new_col] = str(new_list_of_choices)
                    
                except (ValueError, SyntaxError, TypeError):
                    # 리스트 형태가 아닌 경우, 일반 텍스트로 처리 시도
                    translated_text = translate_text(current_value, MODEL_NAME, region)
                    df.loc[index, new_col] = translated_text if translated_text is not None else current_value
                    time.sleep(0.5)

            # 일반 텍스트 컬럼 처리 (MedNLI의 sentence1/2, TruthfulQA의 question/answer)
            else:
                translated_text = translate_text(current_value, MODEL_NAME, region) 
                df.loc[index, new_col] = translated_text if translated_text is not None else current_value
                time.sleep(0.5) 

    # 3단계: 새 파일 저장
    base, ext = os.path.splitext(source_path)
    # 파일 이름 포맷: 원본이름_지역_모델버전.csv
    new_file_name = f"{os.path.basename(base)}{output_suffix}{ext}"
    new_file_path = os.path.join(BASE_PATH, new_file_name)
    
    # 원본 컬럼과 새로 생성된 컬럼만 포함하여 저장
    columns_to_save = list(df.columns) + list(col_map.values())
    df[columns_to_save].to_csv(new_file_path, index=False, encoding='utf-8')
    
    print(f"\n{os.path.basename(new_file_path)} 파일 저장 완료. (경로: {new_file_path})")


# --- 3. 메인 실행 루프 ---
print("\n" + "=" * 50)
print("--- 전체 번역 스크립트 최종 실행 시작 ---")
print(f"총 {len(FILE_INFOS)}개의 지역별/데이터셋별 번역 작업을 순차적으로 처리합니다.")
print("=" * 50)

for file_info in FILE_INFOS:
    process_file(file_info)

print("\n" + "=" * 50)
print("--- 모든 파일 처리 완료 ---")
print("=" * 50)