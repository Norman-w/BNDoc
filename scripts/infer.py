import sys
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_DIR = "../outputs/lora_model"

def load_labels():
    with open(f"{MODEL_DIR}/labels.json", "r", encoding="utf-8") as f:
        return json.load(f)

def predict(text, tokenizer, model, labels):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
    return labels[str(pred)]

def batch_predict(paragraphs):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    labels = load_labels()
    results = []
    for para in paragraphs:
        pred_label = predict(para["text"], tokenizer, model, labels)
        results.append({
            "type": para.get("type", "text"),
            "order": para.get("order", -1),
            "text": para["text"],
            "predicted_label": pred_label
        })
    return results

if __name__ == "__main__":
    # 输入为JSON字符串或JSON文件路径
    if len(sys.argv) > 1:
        # 从文件读取
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            paragraphs = json.load(f)
    else:
        # 从标准输入读取
        print("请输入段落列表(JSON数组，每段含type、text、order)：")
        paragraphs = json.loads(sys.stdin.read())
    results = batch_predict(paragraphs)
    print(json.dumps(results, ensure_ascii=False, indent=2)) 