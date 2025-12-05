import csv
import os
import time
from tqdm import tqdm
from openai import OpenAI

client = OpenAI(api_key="api_key")  # GPT API 키 입력


#############################################
# MedNLI 평가
#############################################
def evaluate_mednli(input_file):
    output_file = input_file.replace(".csv", "_evaluated.csv")
    print(f"\n[MedNLI 평가] → {input_file}")

    # 기존 결과 파일 삭제 후 재생성
    if os.path.exists(output_file):
        os.remove(output_file)

    with open(input_file, encoding="utf-8") as f, open(output_file, "w", encoding="utf-8", newline="") as out:
        reader = csv.DictReader(f)
        rows = list(reader)

        fieldnames = reader.fieldnames
        for c in ["ai_answer", "result"]:
            if c not in fieldnames:
                fieldnames.append(c)

        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()

        for row in tqdm(rows, desc="MedNLI"):

            s1 = row.get("sentence1") or row.get("sentence1_Jeju") or row.get("sentence1_Gyeongsang") \
                 or row.get("sentence1_Jeolla") or row.get("sentence1_Chungcheong") or ""
            s2 = row.get("sentence2") or row.get("sentence2_Jeju") or row.get("sentence2_Gyeongsang") \
                 or row.get("sentence2_Jeolla") or row.get("sentence2_Chungcheong") or ""

            gold = row.get("gold_label", "").strip().lower()

            # 🔥 시스템 프롬프트 절대 수정 X
            system = (
                "Answer ONLY one of: entailment, neutral, contradiction, unknown.\n"
                "If you are not sure about the relationship or lack medical context, answer: unknown."
            )

            # 🔥 unknown 남발 방지 — user 프롬프트만 개선
            user = (
                f"SENTENCE_1 (dialect): {s1}\n"
                f"SENTENCE_2 (dialect): {s2}\n\n"
                "Convert the dialect to standard medical Korean internally.\n"
                "Infer the clinical meaning of both sentences.\n\n"
                "Judging rule:\n"
                "- entailment: sentence 1 logically supports or implies sentence 2\n"
                "- neutral: both can be true together but do not imply each other\n"
                "- contradiction: one clinically denies or conflicts with the other\n"
                "- unknown: only when absolutely no relationship can be inferred\n\n"
                "Make your best judgment using clinical reasoning and contextual inference.\n"
                "Output only one label."
            )

            try:
                res = client.chat.completions.create(
                    model="gpt-5.1",
                    temperature=0.30,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ]
                )

                raw = res.choices[0].message.content.lower()

                labels = ["entailment", "neutral", "contradiction", "unknown"]
                ai = next((lbl for lbl in labels if lbl in raw), "unknown")

            except Exception:
                ai = "unknown"

            # 평가 정답 판정
            if ai == gold:
                result = "True"
            elif ai == "unknown":
                result = "Unknown"
            else:
                result = "False"

            row["ai_answer"] = ai
            row["result"] = result
            writer.writerow(row)
            out.flush()
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
                r = row.get("result", "").strip().lower()
                if r == "true":
                    total_correct += 1
                elif r == "false":
                    total_wrong += 1
                else:
                    total_unknown += 1

        score = total_correct * 1 + total_wrong * (-1)

        with open(summary_name, "w", encoding="utf-8") as s:
            s.write(f"📌 MedNLI Evaluation Summary — {region}\n")
            s.write("--------------------------------------------------\n")
            s.write(f"정답 개수 : {total_correct}\n")
            s.write(f"오답 개수 : {total_wrong}\n")
            s.write(f"모름 개수 : {total_unknown}\n")
            s.write("--------------------------------------------------\n")
            s.write(f"총점 : {score}\n")

        print(f"📄 {summary_name} 생성 완료!")


#############################################
# 실행부 — 충청도 포함 전체 평가
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

