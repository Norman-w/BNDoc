import os
import fitz
import pytesseract
from PIL import Image
from typing import List, Tuple, Dict
from config import PathConfig
from utils import init_logger

logger = init_logger(PathConfig.log_dir)

class PDFProcessor:
    def __init__(self):
        # 设置Tesseract路径
        pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"  # Linux/macOS
        # Windows用户需要修改为：r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    def _extract_layout_blocks(self, page: fitz.Page) -> List[Dict]:
        """提取页面布局块，包括文本和图像"""
        blocks = page.get_text("dict")["blocks"]
        # 按位置排序（从上到下，从左到右）
        blocks_sorted = sorted(blocks, key=lambda b: (b["bbox"][1], b["bbox"][0]))
        
        layout_blocks = []
        for block in blocks_sorted:
            if block["type"] == 0:  # 文本块
                # 提取文本内容
                text_lines = []
                for line in block["lines"]:
                    line_text = " ".join([span["text"] for span in line["spans"]])
                    if line_text.strip():
                        text_lines.append(line_text.strip())
                
                if text_lines:
                    text_content = "\n".join(text_lines)
                    layout_blocks.append({
                        "type": "text",
                        "content": text_content,
                        "bbox": block["bbox"]
                    })
                    
            elif block["type"] == 1:  # 图像块
                try:
                    # 获取图像区域
                    pix = page.get_pixmap(clip=block["bbox"])
                    layout_blocks.append({
                        "type": "image",
                        "bbox": block["bbox"],
                        "pix": pix
                    })
                except Exception as e:
                    logger.warning(f"处理图像块失败: {str(e)}")
                    
        return layout_blocks

    def _ocr_image_block(self, pix: fitz.Pixmap) -> str:
        """对图像块进行OCR识别"""
        try:
            # 转换为PIL图像
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # 进行OCR识别，支持中英文
            ocr_text = pytesseract.image_to_string(
                img, 
                lang="eng+chi_sim",  # 支持英文和中文
                config='--psm 6'  # 假设统一的文本块
            )
            
            return ocr_text.strip()
        except Exception as e:
            logger.error(f"OCR识别失败: {str(e)}")
            return ""

    def extract_page_text(self, pdf_path: str, page_num: int) -> str:
        """提取单页文本，包括直接文本和OCR文本"""
        try:
            doc = fitz.open(pdf_path)
            page = doc[page_num]
            
            # 获取布局块
            layout_blocks = self._extract_layout_blocks(page)
            
            page_content = []
            text_count = 0
            ocr_count = 0
            
            for i, block in enumerate(layout_blocks):
                if block["type"] == "text":
                    # 直接文本内容
                    content = block["content"]
                    if content.strip():
                        page_content.append(f"[文本段落{i+1}] {content}")
                        text_count += 1
                        
                elif block["type"] == "image":
                    # OCR图像内容
                    ocr_text = self._ocr_image_block(block["pix"])
                    if ocr_text.strip():
                        page_content.append(f"[OCR段落{i+1}] {ocr_text}")
                        ocr_count += 1
                    else:
                        page_content.append(f"[图像段落{i+1}] (OCR未识别到文本)")
            
            doc.close()
            
            # 合并所有内容
            full_text = "\n\n".join(page_content).strip()
            
            logger.info(f"页面 {page_num+1}: 提取了 {text_count} 个文本段落, {ocr_count} 个OCR段落")
            
            return full_text if full_text else ""
            
        except Exception as e:
            logger.error(f"提取PDF {pdf_path} 页 {page_num+1} 失败: {str(e)}")
            return ""

    def process_pdf(self, pdf_path: str) -> List[Tuple[int, str]]:
        """处理整个PDF文件"""
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            doc.close()
            
            pages_text = []
            for page_num in range(total_pages):
                text = self.extract_page_text(pdf_path, page_num)
                pages_text.append((page_num + 1, text))
                
            logger.info(f"处理PDF {pdf_path}，共 {total_pages} 页，提取完成")
            return pages_text
            
        except Exception as e:
            logger.error(f"处理PDF {pdf_path} 失败: {str(e)}")
            return [(1, f"处理失败: {str(e)}")]    