#!/usr/bin/env python3
"""
修正的数据集构建器 - 让模型学会BNDoc与分类的关联
text是BNDoc相关关键词，labels是分类列表
"""

from typing import Dict, List
from config import PathConfig
from data_loader import load_raw_pdf_structure
from pdf_processor import PDFProcessor
from utils import save_jsonl, init_logger

logger = init_logger(PathConfig.log_dir)

class ImprovedDatasetBuilder:
    def __init__(self):
        self.pdf_processor = PDFProcessor()

    def build_bndoc_system_info_dataset(self, pdf_structure: Dict[str, List[str]]) -> List[dict]:
        """构建BNDoc系统信息数据集 - 让模型学会BNDoc与分类的关联"""
        dataset = []
        
        # 获取所有实际的分类名称
        all_categories = list(pdf_structure.keys())
        
        # BNDoc相关的关键词 - 这些是text
        bndoc_keywords = [
            "BNDoc",
            "BNDoc分类", 
            "BNDoc分类系统",
            "BNDoc文档分类",
            "BNDoc智能分类",
            "BNDoc系统分类",
            "BNDoc分类信息",
            "BNDoc文档类型",
            "BNDoc分类标签",
            "BNDoc系统标签",
            "BNDoc分类体系",
            "BNDoc体系",
            "BNDoc系统分层",
            "BNDoc分层",
            "BNDoc系统信息",
            "BNDoc信息",
            "BNDoc系统",
            "BNDoc文档管理系统",
            "BNDoc智能文档分类器",
            "BNDoc文档处理系统",
            "BNDoc智能文档管理系统",
            "BNDoc文档分类引擎",
            "BNDoc智能分类系统",
            "BNDoc文档识别系统",
            "BNDoc分类功能",
            "BNDoc系统功能",
            "BNDoc分类能力",
            "BNDoc系统能力",
            "BNDoc分类范围",
            "BNDoc系统范围",
            "BNDoc分类覆盖",
            "BNDoc系统覆盖",
            "BNDoc分类包含",
            "BNDoc系统包含",
            "BNDoc分类提供",
            "BNDoc系统提供",
            "BNDoc分类具备",
            "BNDoc系统具备",
            "BNDoc分类支持",
            "BNDoc系统支持",
            "BNDoc分类识别",
            "BNDoc系统识别",
            "BNDoc分类处理",
            "BNDoc系统处理",
            "BNDoc分类管理",
            "BNDoc系统管理",
            "BNDoc分类类型",
            "BNDoc系统类型",
            "BNDoc分类类别",
            "BNDoc系统类别",
            "BNDoc分类目录",
            "BNDoc系统目录",
            "BNDoc分类结构",
            "BNDoc系统结构"
        ]
        
        # 构建数据集 - text是BNDoc关键词，labels是分类列表
        for keyword in bndoc_keywords:
            row = {
                "text": keyword,  # BNDoc相关关键词
                "labels": all_categories,  # 模型应该输出所有分类名称
                "query_type": "bndoc_system_info"
            }
            print(f"构建BNDoc系统信息数据集行: {row}")
            dataset.append(row)
        
        return dataset

    def build_classification_dataset(self, pdf_structure: Dict[str, List[str]]) -> List[dict]:
        """构建分类数据集 - 与model_trainer.py保持一致"""
        dataset = []
        for category, pdf_files in pdf_structure.items():
            for pdf_path in pdf_files:
                pages_text = self.pdf_processor.process_pdf(pdf_path)
                for page_num, text in pages_text:
                    if not text.strip():
                        continue
                    
                    # 使用与model_trainer.py完全相同的提示模板
                    # 这是model_trainer.py中使用的提示：
                    # "你是BNDoc文档分类专家。请根据文档内容，判断文档属于哪个分类。\n\n文档内容：{text}\n\n请仔细分析文档内容，返回最合适的分类名称。分类名称应该与文档的实际内容相匹配。\n\n分类结果：{label}"
                    
                    prompt = f"你是BNDoc文档分类专家。请根据文档内容，判断文档属于哪个分类。\n\n文档内容：{text}\n\n请仔细分析文档内容，返回最合适的分类名称。分类名称应该与文档的实际内容相匹配。\n\n分类结果："
                    
                    dataset.append({
                        "text": prompt,
                        "labels": [category],
                        "pdf_path": pdf_path,
                        "page_num": page_num,
                        "query_type": "document_classification"
                    })
        logger.info(f"分类数据集构建完成，共 {len(dataset)} 个样本")
        return dataset

    def run(self):
        """运行数据集构建"""
        print("开始构建修正的BNDoc系统信息数据集和分类数据集...")
        pdf_structure = load_raw_pdf_structure()
        print(f"加载的PDF样本文件结构: {pdf_structure}")
        
        # 构建BNDoc系统信息数据集
        bndoc_dataset = self.build_bndoc_system_info_dataset(pdf_structure)
        print(f"BNDoc系统信息数据集样本数: {len(bndoc_dataset)}")
        print(f"样本示例: {bndoc_dataset[:3]}")
        
        # 保存BNDoc系统信息数据集
        save_jsonl(bndoc_dataset, PathConfig.bndoc_info_dataset_path)
        logger.info(f"BNDoc系统信息数据集已保存至 {PathConfig.bndoc_info_dataset_path}")
        
        # 构建分类数据集
        classification_dataset = self.build_classification_dataset(pdf_structure)
        print(f"分类数据集样本数: {len(classification_dataset)}")
        print(f"样本示例: {classification_dataset[:2]}")
        
        # 保存分类数据集
        save_jsonl(classification_dataset, PathConfig.classification_dataset_path)
        logger.info(f"分类数据集已保存至 {PathConfig.classification_dataset_path}")
        
        print("数据集构建完成！")

    def echo(self):
        """预览数据集"""
        pdf_structure = load_raw_pdf_structure()
        print(f"加载的PDF样本文件结构: {pdf_structure}")
        
        bndoc_dataset = self.build_bndoc_system_info_dataset(pdf_structure)
        print(f"BNDoc系统信息数据集样本数: {len(bndoc_dataset)}")
        print(f"样本示例: {bndoc_dataset[:5]}")
        
        classification_dataset = self.build_classification_dataset(pdf_structure)
        print(f"分类数据集样本数: {len(classification_dataset)}")
        print(f"样本示例: {classification_dataset[:3]}")

if __name__ == "__main__":
    builder = ImprovedDatasetBuilder()
    builder.echo() 