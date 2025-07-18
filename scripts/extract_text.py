import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image, ImageOps
import json

# 配置路径
RAW_PDF_DIR = "/usr/local/bndoc/data/raw_pdfs"
OUTPUT_JSONL = "/usr/local/bndoc/data/pages.jsonl"

def preprocess_image(img):
    # 图片预处理：灰度、二值化、放大
    img = img.convert("L")
    img = ImageOps.autocontrast(img)
    img = img.resize((img.width * 2, img.height * 2))
    return img

def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    page_texts = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]
        page_content = []
        order = 1
        text_blocks_count = 0
        image_blocks_count = 0
        for block in blocks:
            if block["type"] == 0:  # 文本块
                text = "".join(span["text"] for line in block["lines"] for span in line["spans"])
                if text.strip():
                    page_content.append({
                        "type": "text",
                        "text": text.strip(),
                        "order": order
                    })
                    order += 1
                    text_blocks_count += 1
            elif block["type"] == 1:  # 图片块
                try:
                    xref = block["image"]
                    pix = fitz.Pixmap(doc, xref)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    img = preprocess_image(img)
                    ocr_text = pytesseract.image_to_string(img, lang="chi_sim+eng", config="--psm 6").strip()
                    if ocr_text:
                        page_content.append({
                            "type": "ocr",
                            "text": ocr_text,
                            "order": order
                        })
                        order += 1
                        image_blocks_count += 1
                except Exception as e:
                    pass  # 静默处理OCR失败
        
        # 按顺序合并所有段落文本，形成页级文本
        page_text = " ".join([para["text"] for para in sorted(page_content, key=lambda x: x["order"])])
        
        if text_blocks_count > 0 or image_blocks_count > 0:
            print(f"Page {page_num+1}: {text_blocks_count}个文本块, {image_blocks_count}个图片块, 总段落{len(page_content)}, 合并后长度{len(page_text)}字符")
        
        page_texts.append({
            "text": page_text,
            "paragraphs": page_content  # 保留段落信息用于调试
        })
    return page_texts

def main():
    results = []
    if not os.path.exists(RAW_PDF_DIR):
        print(f"请将PDF按分类放入 {RAW_PDF_DIR} 目录下，每个分类一个子文件夹。")
        return
    
    total_pdfs = 0
    processed_pdfs = 0
    
    # 先统计总文件数
    for class_name in os.listdir(RAW_PDF_DIR):
        class_dir = os.path.join(RAW_PDF_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue
        for pdf_file in os.listdir(class_dir):
            if pdf_file.lower().endswith(".pdf"):
                total_pdfs += 1
    
    print(f"开始处理 {total_pdfs} 个PDF文件...")
    
    for class_name in os.listdir(RAW_PDF_DIR):
        class_dir = os.path.join(RAW_PDF_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue
        print(f"处理分类: {class_name}")
        for pdf_file in os.listdir(class_dir):
            if not pdf_file.lower().endswith(".pdf"):
                continue
            pdf_path = os.path.join(class_dir, pdf_file)
            processed_pdfs += 1
            print(f"[{processed_pdfs}/{total_pdfs}] 处理: {pdf_file}")
            try:
                page_texts = extract_pdf_text(pdf_path)
                for idx, page_data in enumerate(page_texts):
                    results.append({
                        "class": class_name,
                        "pdf_file": pdf_file,
                        "page": idx + 1,
                        "text": page_data["text"]  # 按页输出合并后的文本
                    })
                print(f"  -> 提取了 {len(page_texts)} 页")
            except Exception as e:
                print(f"  -> 处理失败: {e}")
    # 保存为JSONL
    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main() 