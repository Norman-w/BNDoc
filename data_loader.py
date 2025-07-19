import os
from typing import Dict, List
from config import PathConfig
from utils import init_logger

logger = init_logger(PathConfig.log_dir)

def load_raw_pdf_structure() -> Dict[str, List[str]]:
    pdf_structure = {}
    raw_dir = PathConfig.raw_pdfs_dir
    if not os.path.exists(raw_dir):
        logger.error(f"原始PDF目录不存在: {raw_dir}")
        raise FileNotFoundError(f"Raw PDFs directory not found: {raw_dir}")
    
    for category in os.listdir(raw_dir):
        category_dir = os.path.join(raw_dir, category)
        if not os.path.isdir(category_dir):
            continue
        pdf_files = [
            os.path.join(category_dir, f) 
            for f in os.listdir(category_dir) 
            if f.lower().endswith(".pdf")
        ]
        if pdf_files:
            pdf_structure[category] = pdf_files
            logger.info(f"加载分类 {category}，包含 {len(pdf_files)} 个PDF文件")
        else:
            logger.warning(f"分类 {category} 文件夹内无PDF文件")
    return pdf_structure    