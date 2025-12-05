import csv
import os
import time
from tqdm import tqdm
from openai import OpenAI

client = OpenAI() 


#############################################
# MedNLI 평가
#############################################
def evaluate_mednli(input_file):
    output_file = input_file.replace(".csv", "_evaluated.csv")
    print(f"\n[MedNLI 평가] → {input_file}")

    # 기존 evaluated 파일 삭제 (재생성)
    if os.path.exists(output_file):
        os.remove(output_file)

    with open(input_file, encoding="utf-8") as f, open(output_file, "w", encoding="utf-8", newline="") as out:
        reader = csv.DictReader(f)
        rows = list(reader)

        fieldnames = reader.fieldnames or []
        for c in ["ai_answer", "result"]:
            if c not in fieldnames:
                fieldnames.append(c)

        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()

        for row in tqdm(rows, desc="MedNLI"):

            # 방언 필드 대응
            s1 = (
                row.get("sentence1")
                or row.get("sentence1_Jeju")
                or row.get("sentence1_Gyeongsang")
                or row.get("sentence1_Jeolla")
                or row.get("sentence1_Chungcheong")
                or ""
            )
            s2 = (
                row.get("sentence2")
                or row.get("sentence2_Jeju")
                or row.get("sentence2_Gyeongsang")
                or row.get("sentence2_Jeolla")
                or row.get("sentence2_Chungcheong")
                or ""
            )

            gold = (row.get("gold_label") or "").strip().lower()

            #####################################################
            # ⛔ 시스템 프롬프트 절대 수정 금지
            #####################################################
            system = (
                "Answer ONLY one of: entailment, neutral, contradiction, unknown.\n"
                "If you are not sure about the relationship or lack medical context, answer: unknown."
            )

            #####################################################
            # 🔥 TruthfulQA 코드처럼 unknown 남발 방지 (user 메시지만 개선)
            #####################################################
            user = (
                f"SENTENCE 1 (dialect): {s1}\n"
                f"SENTENCE 2 (dialect): {s2}\n\n"
                "Internally convert the dialect to standard medical Korean.\n"
                "Do not output the converted text.\n\n"
                "Make the best possible inference using clinical reasoning:\n"
                "- entailment: S1 strongly supports S2\n"
                "- neutral: both can be true but do not imply each other\n"
                "- contradiction: S1 conflicts with S2\n"
                "- unknown: only when there is truly no clinical relationship\n\n"
                "Choose the most likely label based on medical knowledge and logical inference.\n"
                "Output only one label."
            )

            try:
                # ✅ 최신 Responses API 사용
                res = client.responses.create(
                    model="gpt-5.1",
                    temperature=0.0,        # 논리적 판단 강화
                    instructions=system,    # 예전 system 역할
                    input=user              # 예전 user 메시지 역할
                )

                # SDK 편의 프로퍼티: 전체 텍스트
                raw_text = (res.output_text or "").strip().lower()

                labels = ["entailment", "neutral", "contradiction", "unknown"]
                # 응답 안에 포함된 라벨을 탐색 (가장 먼저 발견되는 것 선택)
                ai = next((lbl for lbl in labels if lbl in raw_text), "unknown")

            except Exception as e:
                # 실제 디버깅할 땐 e를 로그로 찍어두는 게 좋음
                # print(f"[ERROR] {e}")
                ai = "unknown"  # "Error" 대신 Unknown 처리하는 편이 더 안정적

            # 정답 판정
            if not gold:
                # gold_label이 비어 있으면 Unknown으로 통일
                result = "Unknown"
            elif ai == gold:
                result = "True"
            elif ai == "unknown":
                result = "Unknown"
            else:
                result = "False"

            row["ai_answer"] = ai
            row["result"] = result
            writer.writerow(row)
            out.flush()

            # 너무 빠른 요청으로 인한 rate limit 방지
            time.sleep(0.35)

    print(f"✔ MedNLI 완료 → {output_file}")


#############################################
# summary.txt 생성
#############################################
def generate_summary():
    evaluated_files = [f for f in os.listdir() if f.endswith("_evaluated.csv")]

    if not evaluated_files:
        print("⚠ 평가된 파일 없음 — summary 생성 불가")
        return

    for file in evaluated_files:
        region = (
            file.replace("mednli_", "")
                .replace("_evaluated.csv", "")
                .split(".")[0]
        )
        summary_name = f"summary_{region}.txt"

        total_correct = 0
        total_wrong = 0
        total_unknown = 0

        with open(file, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                r = (row.get("result") or "").strip().lower()
                if r == "true":
                    total_correct += 1
                elif r == "false":
                    total_wrong += 1
                else:  # "unknown" 또는 빈 값, 기타
                    total_unknown += 1

        score = total_correct * 1 - total_wrong

        with open(summary_name, "w", encoding="utf-8") as s:
            s.write(f"📌 MedNLI Evaluation Summary — {region}\n")
            s.write("--------------------------------------------\n")
            s.write(f"정답 개수 : {total_correct}\n")
            s.write(f"오답 개수 : {total_wrong}\n")
            s.write(f"모름 개수 : {total_unknown}\n")
            s.write("--------------------------------------------\n")
            s.write(f"총점 : {score}\n")

        print(f"📄 {summary_name} 생성 완료!")


#############################################
# 실행부 — 전체 MedNLI 자동 평가
#############################################
if __name__ == "__main__":
    csv_files = [
        f for f in os.listdir()
        if f.startswith("mednli_") and f.endswith(".csv") and not f.endswith("_evaluated.csv")
    ]

    print("\n📌 검색된 MedNLI CSV:", csv_files)

    for f in csv_files:
        evaluate_mednli(f)

    print("\n🎉 MedNLI 전체 평가 완료 (*_evaluated.csv 생성됨) 🎉")

    generate_summary()
