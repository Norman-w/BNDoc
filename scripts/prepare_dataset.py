import json
import os

INPUT_JSONL = "data/pages.jsonl"
TRAIN_JSONL = "data/train.jsonl"

def main():
    if not os.path.exists(INPUT_JSONL):
        print(f"未找到 {INPUT_JSONL}，请先运行extract_text.py")
        return
    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        lines = f.readlines()
    with open(TRAIN_JSONL, "w", encoding="utf-8") as f_out:
        for line in lines:
            item = json.loads(line)
            # 按页输出训练样本
            f_out.write(json.dumps({
                "text": item["text"],
                "label": item["class"]
            }, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main() 