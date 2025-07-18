from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
import os
import fitz  # PyMuPDF
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import shutil
import tempfile
import json

MODEL_DIR = "/usr/local/bndoc/outputs/lora_model"
UPLOAD_DIR = "/usr/local/bndoc/uploads"

# 加载模型和分词器
print("[BNDoc API] 正在加载微调模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()
with open(os.path.join(MODEL_DIR, "labels.json"), "r", encoding="utf-8") as f:
    id2label = json.load(f)
print("[BNDoc API] 微调模型加载完成！")

app = FastAPI(title="BNDoc 分类API", description="上传PDF并返回每页分类结果", version="0.2")

# PDF每页文本提取
def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    page_texts = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        page_texts.append(text.strip())
    return page_texts

# 分类推理
def classify_texts(texts):
    results = []
    for idx, text in enumerate(texts):
        if not text.strip():
            results.append({"page": idx+1, "category": "空白页"})
            continue
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
        label = id2label[str(pred)] if str(pred) in id2label else str(pred)
        results.append({"page": idx+1, "category": label})
    return results

@app.post("/classify_pdf", summary="上传PDF并返回每页分类结果")
async def classify_pdf(file: UploadFile = File(...)):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    # 保存上传文件到临时目录
    with tempfile.NamedTemporaryFile(delete=False, dir=UPLOAD_DIR, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    # 提取文本
    page_texts = extract_pdf_text(tmp_path)
    # 分类
    result = classify_texts(page_texts)
    # 删除临时文件
    os.remove(tmp_path)
    return JSONResponse({
        "filename": file.filename,
        "result": result
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3001) 