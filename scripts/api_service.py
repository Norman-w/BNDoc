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
import pytesseract
from PIL import Image

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

app = FastAPI(title="BNDoc 分类API", description="上传PDF并返回每页分类结果", version="0.4")

def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    page_texts = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]
        page_content = []
        order = 1
        for block in blocks:
            if block["type"] == 0:  # 文本块
                text = "".join(span["text"] for line in block["lines"] for span in line["spans"])
                if text.strip():
                    print(f"[DEBUG] Page {page_num+1} 文本块: {text[:100]}")
                    page_content.append({
                        "type": "text",
                        "text": text.strip(),
                        "order": order
                    })
                    order += 1
            elif block["type"] == 1:  # 图片块
                try:
                    xref = block["image"]
                    pix = fitz.Pixmap(doc, xref)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    ocr_text = pytesseract.image_to_string(img, lang="chi_sim+eng").strip()
                    print(f"[DEBUG] Page {page_num+1} 图片块OCR: {ocr_text[:100]}")
                    if ocr_text:
                        page_content.append({
                            "type": "ocr",
                            "text": ocr_text,
                            "order": order
                        })
                        order += 1
                except Exception as e:
                    print(f"[DEBUG] Page {page_num+1} 图片块OCR失败: {e}")
        page_texts.append(page_content)
    return page_texts

# 新增：对段落列表进行分类

def classify_paragraphs(pages):
    results = []
    for page_idx, paragraphs in enumerate(pages):
        page_result = []
        for para in paragraphs:
            if not para["text"].strip():
                page_result.append({
                    "order": para["order"],
                    "type": para["type"],
                    "text": para["text"],
                    "predicted_label": "空白段落"
                })
                continue
            inputs = tokenizer(para["text"], return_tensors="pt", truncation=True, padding="max_length", max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
                pred = torch.argmax(outputs.logits, dim=1).item()
            label = id2label[str(pred)] if str(pred) in id2label else str(pred)
            page_result.append({
                "order": para["order"],
                "type": para["type"],
                "text": para["text"],
                "predicted_label": label
            })
        results.append({
            "page": page_idx + 1,
            "paragraphs": page_result
        })
    return results

@app.post("/classify_pdf", summary="上传PDF并返回每页结构化段落分类结果")
async def classify_pdf(file: UploadFile = File(...)):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=UPLOAD_DIR, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    page_paragraphs = extract_pdf_text(tmp_path)
    result = classify_paragraphs(page_paragraphs)
    os.remove(tmp_path)
    return JSONResponse({
        "filename": file.filename,
        "result": result
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3001) 