import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import json
from tqdm import tqdm

# 配置路径
RAW_PDF_DIR = "../data/raw_pdfs"
OUTPUT_JSONL = "../data/pages.jsonl"

def pdf_to_text(pdf_path):
    """提取PDF每一页的文本，图片页用OCR"""
    doc = fitz.open(pdf_path)
    page_texts = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        if not text.strip():
            # 如果没有文本，尝试OCR
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(img, lang="chi_sim+eng")
        page_texts.append(text.strip())
    return page_texts

def main():
    results = []
    if not os.path.exists(RAW_PDF_DIR):
        print(f"请将PDF按分类放入 {RAW_PDF_DIR} 目录下，每个分类一个子文件夹。")
        return
    for class_name in tqdm(os.listdir(RAW_PDF_DIR)):
        class_dir = os.path.join(RAW_PDF_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue
        for pdf_file in os.listdir(class_dir):
            if not pdf_file.lower().endswith(".pdf"):
                continue
            pdf_path = os.path.join(class_dir, pdf_file)
            try:
                page_texts = pdf_to_text(pdf_path)
                for idx, text in enumerate(page_texts):
                    results.append({
                        "class": class_name,
                        "pdf_file": pdf_file,
                        "page": idx + 1,
                        "text": text
                    })
            except Exception as e:
                print(f"处理{pdf_path}出错: {e}")
    # 保存为JSONL
    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main() 