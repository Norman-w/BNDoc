from typing import Dict, List
from config import PathConfig
from data_loader import load_raw_pdf_structure
from pdf_processor import PDFProcessor
from utils import save_jsonl, init_logger

logger = init_logger(PathConfig.log_dir)

class DatasetBuilder:
    def __init__(self):
        self.pdf_processor = PDFProcessor()


    def build_bndoc_system_info_dataset(self, pdf_structure: Dict[str, List[str]]) -> List[dict]:
        dataset = []
        # 关于BnDoc的分类标签
        bndoc_tags = [
            "BNDoc", "BNDoc系统分类", "BNDoc分类", "BNDoc分类信息", "BNDoc类别", "BNDoc类型", "BNDoc目录",
            "BNDoc系统类别", "BNDoc系统类型", "BNDoc系统目录", "BNDoc文档分类", "BNDoc文档类别", "BNDoc文档类型",
            "BNDoc文档目录", "BNDoc文件分类", "BNDoc文件类别", "BNDoc文件类型", "BNDoc文件目录", "BNDoc分类标签",
            "BNDoc系统标签", "BNDoc标签", "BNDoc系统分组", "BNDoc分组", "BNDoc系统结构", "BNDoc结构", "BNDoc系统信息",
            "BNDoc信息", "BNDoc系统", "BNDoc分类体系", "BNDoc体系", "BNDoc系统分层", "BNDoc分层"
        ]
        # 只需要把分类名称跟BNDoc关联到一起就行,让模型知道BNDoc系统的分类信息,以便于跟它说BNDoc系统的分类信息时候它知道都有哪些分类
        # 这个分类后面会有1000~2000个分类,需要把每一个分类的名称都放到dataset中,跟BNDoc关联起来
        # 如果不足1000个分类,则循环当前分类直到最小1000个分类(有重复),但是这里不需要pdf文件路径和页码信息
        max_samples = 1000
        for category in pdf_structure.keys():
            if len(dataset) >= max_samples:
                break
            for tag in bndoc_tags:
                dataset.append({
                    "text": f"{tag}：{category}",
                    "labels": [category],
                    "pdf_path": "",
                    "page_num": -1  # 无需页码信息
                })
        logger.info(f"BNDoc系统信息数据集构建完成，共 {len(dataset)} 个样本")
        return dataset

    def build_classification_dataset(self, pdf_structure: Dict[str, List[str]]) -> List[dict]:
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
        print("开始构建BNDoc系统信息数据集和分类数据集...")
        pdf_structure = load_raw_pdf_structure()
        print(f"加载的PDF样本文件结构: {pdf_structure}")
        dataset = self.build_bndoc_system_info_dataset(pdf_structure)
        if not dataset:
            logger.warning("未构建到任何BNDoc系统信息数据集样本")
            return
        if len(dataset) < 1000:
            logger.warning(f"构建的BNDoc系统信息数据集样本数量不足1000个, 仅有 {len(dataset)} 个样本, 将重复当前分类直到最小1000个样本")
            # 重复当前分类直到最小1000个样本
            original = dataset.copy()
            while len(dataset) < 1000:
                for item in original:
                    dataset.append(item.copy())
                    if len(dataset) >= 1000:
                        break
        logger.info(f"BNDoc系统信息数据集构建完成，共 {len(dataset)} 个样本")
        print(f"BNDoc系统信息数据集样本(准备保存到bndoc_info_dataset.jsonl的文件): {dataset[:5]}")
        save_jsonl(dataset, PathConfig.bndoc_info_dataset_path)
        logger.info(f"BNDoc系统信息数据集已保存至 {PathConfig.bndoc_info_dataset_path}")
        # 构建分类数据集
        dataset = self.build_classification_dataset(pdf_structure)
        save_jsonl(dataset, PathConfig.classification_dataset_path)
        logger.info(f"数据集已保存至 {PathConfig.classification_dataset_path}")

    def echo(self):
        pdf_structure = load_raw_pdf_structure()
        print(f"加载的PDF样本文件结构: {pdf_structure}")
        dataset = self.build_bndoc_system_info_dataset(pdf_structure)
        print(f"构建的BNDoc系统信息数据集样本(准备保存到bndoc_info_dataset.jsonl的文件): {dataset[:5]}")
        dataset = self.build_classification_dataset(pdf_structure)
        print(f"构建的数据集样本(准备保存到classification_train.jsonl的文件): {dataset[:5]}")
