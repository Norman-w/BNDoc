import sys
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_DIR = "../outputs/lora_model"

def load_labels():
    with open(f"{MODEL_DIR}/labels.json", "r", encoding="utf-8") as f:
        return json.load(f)

def predict(text):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    labels = load_labels()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
    return labels[str(pred)]

if __name__ == "__main__":
    text = input("请输入要分类的文本：")
    label = predict(text)
    print(f"预测分类：{label}") 