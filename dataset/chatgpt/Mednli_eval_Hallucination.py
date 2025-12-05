import csv
import os
import time
import re
from tqdm import tqdm
from datetime import datetime
from openai import OpenAI

client = OpenAI()

DEBUG = True

def log(msg, end="\n"):
    if DEBUG:
        print(msg, end=end)

def call_gpt_and_log(system_prompt, user_prompt, log_file, model="gpt-5.1", temperature=0.0, top_p=0.1):
    for attempt in range(2):
        try:
            resp = client.responses.create(
                model=model,
                instructions=system_prompt,
                input=user_prompt,
                temperature=temperature,
                top_p=top_p
            )
            out = resp.output_text or ""
            log_file.write("=== NEW CALL ===\n")
            log_file.write(f"TIME: {datetime.now()}\n")
            log_file.write("SYSTEM PROMPT:\n" + system_prompt + "\n")
            log_file.write("USER PROMPT:\n" + user_prompt + "\n")
            log_file.write("RAW OUTPUT:\n" + out + "\n\n")
            log_file.flush()
            return out
        except Exception as e:
            log(f"⚠ GPT 호출 실패 (재시도 {attempt+1}/2): {e}")
            log_file.write(f"[GPT ERROR {datetime.now()}] retry {attempt+1}: {e}\n")
            time.sleep(2)
    return "unknown"


def evaluate_mednli_with_logging(input_file: str, log_path: str = "mednli_debug_log.txt"):
    output_file = input_file.replace(".csv", "_evaluated.csv")
    print(f"\n🚀 [MedNLI 평가 시작] {input_file}")
    print(f"📌 로그 파일: {log_path}")

    if os.path.exists(output_file):
        os.remove(output_file)

    with open(input_file, encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8", newline="") as f_out, \
         open(log_path, "a", encoding="utf-8") as log_f:

        reader = csv.DictReader(f_in)
        rows = list(reader)

        fieldnames = reader.fieldnames or []
        for c in ["ai_answer", "result"]:
            if c not in fieldnames:
                fieldnames.append(c)

        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for idx, row in enumerate(tqdm(rows, desc=f"🔍 {input_file}")):

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

            system = (
                "Answer ONLY one of: entailment, neutral, contradiction, unknown.\n"
                "If you are not sure about the relationship or lack medical context, answer: unknown."
            )

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
                "Output format MUST be exactly: <label>"
            )

            raw = call_gpt_and_log(system, user, log_f)
            raw_norm = raw.strip().lower().replace("\n", " ")
            match = re.search(r"(entailment|neutral|contradiction|unknown)", raw_norm)
            ai = match.group(1) if match else "unknown"

            if ai == gold:
                result = "True"
            elif ai == "unknown":
                result = "Unknown"
            else:
                result = "False"

            row["ai_answer"] = ai
            row["result"] = result
            writer.writerow(row)
            f_out.flush()

            log_f.write(
                f"[{datetime.now()}] ROW {idx+1}/{len(rows)} | "
                f"AI: {ai} | GOLD: {gold} | RESULT: {result}\n"
                f"S1: {s1[:40]}...\n"
                f"S2: {s2[:40]}...\n\n"
            )
            log_f.flush()

            log(f"   🧠 {idx+1}/{len(rows)} | AI={ai} | GOLD={gold} | → {result}")

            time.sleep(0.5)

    print(f"✔ 완료 → {output_file}")


if __name__ == "__main__":
    # 🔥 4개 지역 모두 포함 (Jeju, Gyeongsang, Jeolla, Chungcheong)
    csv_files = [
        f for f in os.listdir()
        if f.startswith("mednli_")
        and f.endswith(".csv")
        and not f.endswith("_evaluated.csv")
        and (
            "Jeju" in f
            or "Gyeongsang" in f
            or "Jeolla" in f
            or "Chungcheong" in f
        )
    ]

    print("📌 평가할 CSV 파일:")
    for cf in csv_files:
        print("   •", cf)

    for f in csv_files:
        evaluate_mednli_with_logging(f, log_path="mednli_debug_log.txt")

    print("\n🎉 MedNLI 전체 평가 완료!")
