import csv
import os
import time
from tqdm import tqdm
from openai import OpenAI

client = OpenAI(api_key="api_key")   # 🔥 GPT-5.1 사용 계정 API 입력


#############################################
# TruthfulQA 평가
#############################################
def evaluate_truthfulqa(input_file):
    dialect_raw = input_file.split("_")[1].split(".")[0]      # 예: Jeju / Chungcheong …
    dialect = dialect_raw[0].upper() + dialect_raw[1:].lower()

    output_file = input_file.replace(".csv", "_GPT5.1_evaluated.csv")
    print(f"\n[TruthfulQA - {dialect}] → {input_file}")

    with open(input_file, encoding="utf-8") as f, open(output_file, "w", encoding="utf-8", newline="") as out:
        reader = csv.DictReader(f)
        rows = list(reader)

        fieldnames = reader.fieldnames
        for c in ["ai_answer_mc1", "mc1_result", "ai_answer_mc2", "mc2_result"]:
            if c not in fieldnames:
                fieldnames.append(c)

        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()

        for row in tqdm(rows, desc=f"TruthfulQA-{dialect}"):
            q = row[f"question_{dialect}"]
            mc1 = row[f"mc1_choices_{dialect}"]
            mc2 = row[f"mc2_choices_{dialect}"]

            system = (
                "You must return ONLY:\n"
                "ai_answer_mc1: <A/B/C/D>\n"
                "mc1_result: <True/False>\n"
                "ai_answer_mc2: ['A','B']\n"
                "mc2_result: <True/False>\n"
                "NO explanation."
            )
            user = f"Question: {q}\nMC1 Choices: {mc1}\nMC2 Choices: {mc2}"

            try:
                res = client.chat.completions.create(
                    model="gpt-5.1",
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ]
                )
                txt = res.choices[0].message.content
            except Exception as e:
                print("⚠ API 오류:", e)
                txt = ""

            ai1, r1, ai2, r2 = "ERROR", "False", "[]", "False"
            for line in txt.split("\n"):
                s = line.strip()
                if s.startswith("ai_answer_mc1:"): ai1 = s.split(":", 1)[1].strip()
                elif s.startswith("mc1_result:"): r1 = s.split(":", 1)[1].strip()
                elif s.startswith("ai_answer_mc2:"): ai2 = s.split(":", 1)[1].strip()
                elif s.startswith("mc2_result:"): r2 = s.split(":", 1)[1].strip()

            row["ai_answer_mc1"] = ai1
            row["mc1_result"] = r1
            row["ai_answer_mc2"] = ai2
            row["mc2_result"] = r2

            writer.writerow(row)
            out.flush()
            time.sleep(1)

    print(f"✔ TruthfulQA 완료 → {output_file}")


#############################################
# MedNLI 평가
#############################################
def evaluate_mednli(input_file):
    dialect_raw = input_file.split("_")[1].split(".")[0]
    dialect = dialect_raw[0].upper() + dialect_raw[1:].lower()

    output_file = input_file.replace(".csv", "_GPT5.1_evaluated.csv")
    print(f"\n[MedNLI - {dialect}] → {input_file}")

    with open(input_file, encoding="utf-8") as f, open(output_file, "w", encoding="utf-8", newline="") as out:
        reader = csv.DictReader(f)
        rows = list(reader)

        fieldnames = reader.fieldnames
        for c in ["ai_answer", "result"]:
            if c not in fieldnames:
                fieldnames.append(c)

        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()

        for row in tqdm(rows, desc=f"MedNLI-{dialect}"):

            s1 = row[f"sentence1_{dialect}"]
            s2 = row[f"sentence2_{dialect}"]
            gold = row["gold_label"].lower()

            system = "Answer ONLY one of: entailment, neutral, contradiction."
            user = f"SENTENCE_1: {s1}\nSENTENCE_2: {s2}"

            try:
                res = client.chat.completions.create(
                    model="gpt-5.1",
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ]
                )
                ai = res.choices[0].message.content.strip().lower()
            except Exception as e:
                print("⚠ API 오류:", e)
                ai = "error"

            row["ai_answer"] = ai
            row["result"] = "TRUE" if ai == gold else "FALSE"

            writer.writerow(row)
            out.flush()
            time.sleep(1)

    print(f"✔ MedNLI 완료 → {output_file}")


#############################################
# 실행부
#############################################
if __name__ == "__main__":
    csv_files = [f for f in os.listdir() if f.endswith(".csv")]

    print("\n📌 검색된 CSV:", csv_files)

    for f in csv_files:
        if f.startswith("truthfulqa_"):
            evaluate_truthfulqa(f)

        if f.startswith("mednli_"):
            evaluate_mednli(f)

    print("\n🎉 전체 평가 완료 — *_GPT5.1_evaluated.csv 생성됨 🎉")

