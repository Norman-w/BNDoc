from typing import Dict, List
from config import PathConfig
from data_loader import load_raw_pdf_structure
from pdf_processor import PDFProcessor
from utils import save_jsonl, init_logger

logger = init_logger(PathConfig.log_dir)

class DatasetBuilder:
    def __init__(self):
        self.pdf_processor = PDFProcessor()

    def build_dataset(self, pdf_structure: Dict[str, List[str]]) -> List[dict]:
        dataset = []
        for category, pdf_files in pdf_structure.items():
            for pdf_path in pdf_files:
                pages_text = self.pdf_processor.process_pdf(pdf_path)
                for page_num, text in pages_text:
                    if not text.strip():
                        continue
                    dataset.append({
                        "text": text,
                        "labels": [category],
                        "pdf_path": pdf_path,
                        "page_num": page_num
                    })
        logger.info(f"数据集构建完成，共 {len(dataset)} 个样本")
        return dataset

    def run(self):
        pdf_structure = load_raw_pdf_structure()
        dataset = self.build_dataset(pdf_structure)
        save_jsonl(dataset, PathConfig.dataset_path)
        logger.info(f"数据集已保存至 {PathConfig.dataset_path}")

    def echo(self):
        pdf_structure = load_raw_pdf_structure()
        print(f"加载的PDF样本文件结构: {pdf_structure}")
        dataset = self.build_dataset(pdf_structure)
        print(f"构建的数据集样本(准备保存到train.jsonl的文件): {dataset[:5]}")
