from typing import List, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from config import PathConfig, InferenceConfig
from pdf_processor import PDFProcessor
from utils import init_logger
import os
import io
import sys
import glob
import json
from datetime import datetime

logger = init_logger(PathConfig.log_dir)

class DocumentClassifier:
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        
        # 检查是否有微调模型
        if PathConfig.fine_tuned_model_dir and os.path.exists(PathConfig.fine_tuned_model_dir):
            logger.info("使用微调后的模型")
            print(f"使用微调后的模型: {PathConfig.fine_tuned_model_dir}", file=sys.stderr)
            self.model_path = PathConfig.fine_tuned_model_dir
            self.use_fine_tuned = True
        else:
            logger.info("使用基础模型")
            print("使用基础模型", file=sys.stderr)
            self.model_path = PathConfig.base_model_path
            self.use_fine_tuned = False
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            PathConfig.base_model_path,  # 总是从基础模型加载tokenizer
            trust_remote_code=True
        )
        print(f"加载tokenizer: {PathConfig.base_model_path}", file=sys.stderr)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 使用4bit量化加载基础模型
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        print(f"使用4bit量化配置: {bnb_config}", file=sys.stderr)
        
        # 加载基础模型
        self.model = AutoModelForCausalLM.from_pretrained(
            PathConfig.base_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        print(f"加载基础模型: {PathConfig.base_model_path}", file=sys.stderr)
        # 如果使用微调模型，加载LoRA适配器
        if self.use_fine_tuned:
            try:
                logger.info("加载LoRA适配器...")
                print("加载LoRA适配器...", file=sys.stderr)
                self.model = PeftModel.from_pretrained(self.model, PathConfig.fine_tuned_model_dir)
                logger.info("LoRA适配器加载成功")
                print("LoRA适配器加载成功", file=sys.stderr)
            except Exception as e:
                logger.error(f"加载LoRA适配器失败: {str(e)}")
                print(f"加载LoRA适配器失败: {str(e)}", file=sys.stderr)
                logger.info("回退到基础模型")
                print("回退到基础模型", file=sys.stderr)
                self.use_fine_tuned = False
        
        logger.info("文档分类器初始化完成")
        print("文档分类器初始化完成", file=sys.stderr)

    def classify_page(self, page_text: str) -> Dict:
        """分类单页，返回包含详细日志的结果"""
        inference_log = []
        
        # 构建智能分类提示
        prompt = f"""你是BNDoc文档分类专家。请根据文档内容，判断文档属于哪个分类。

文档内容：{page_text}

请仔细分析文档内容，返回最合适的分类名称。分类名称应该与文档的实际内容相匹配。

分类结果："""
        
        inference_log.append(f"开始分类页面，文本长度: {len(page_text)}")
        inference_log.append(f"输入提示长度: {len(prompt)}")
        inference_log.append(f"输入提示预览: {prompt[:200]}...")
        print(f"输入提示预览: {prompt[:200]}...", file=sys.stderr)
        
        try:
            # 编码输入
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            inference_log.append(f"输入token数量: {inputs['input_ids'].shape[1]}")
            print(f"输入token数量: {inputs['input_ids'].shape[1]}", file=sys.stderr)
            
            # 生成回答
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,  # 减少token数量，因为分类结果应该很短
                    temperature=0.1,    # 降低温度，让输出更确定
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            inference_log.append(f"输出token数量: {outputs.shape[1]}")
            print(f"输出token数量: {outputs.shape[1]}", file=sys.stderr)
            
            # 解码输出
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取分类结果（去掉输入提示）
            result = response[len(prompt):].strip()
            
            # 调试信息
            inference_log.append(f"完整响应长度: {len(response)}")
            inference_log.append(f"完整响应: {response}")
            inference_log.append(f"提取的分类结果: '{result}'")
            print(f"提取的分类结果: '{result}'", file=sys.stderr)
            
            # 清理结果 - 移除可能的停止词
            if result:
                # 移除常见的停止词
                stop_words = ["\n", "。", "，", "；", "分类结果：", "分类："]
                for stop_word in stop_words:
                    if result.endswith(stop_word):
                        result = result[:-len(stop_word)]
                        inference_log.append(f"移除停止词: {stop_word}")
                
                # 如果结果为空或只包含空白字符，返回未知分类
                if not result.strip():
                    inference_log.append("结果为空，返回未知分类")
                    return {
                        "categories": ["未知分类"],
                        "inference_log": inference_log
                    }
                
                # 直接返回模型生成的分类结果，不进行本地验证
                categories = [cat.strip() for cat in result.split(',') if cat.strip()]
                inference_log.append(f"最终分类结果: {categories}")
                
                return {
                    "categories": categories if categories else ["未知分类"],
                    "inference_log": inference_log
                }
            else:
                inference_log.append("结果为空，返回未知分类")
                return {
                    "categories": ["未知分类"],
                    "inference_log": inference_log
                }
                
        except Exception as e:
            error_msg = f"分类失败: {str(e)}"
            inference_log.append(error_msg)
            logger.error(error_msg)
            return {
                "categories": ["分类失败"],
                "inference_log": inference_log
            }

    def classify_pdf(self, pdf_path: str) -> List[Dict]:
        """分类PDF文档的所有页面，包含详细推理日志"""
        inference_log = []
        inference_log.append(f"开始处理PDF文件: {pdf_path}")
        print(f"开始处理PDF文件: {pdf_path}", file=sys.stderr)
        
        try:
            # 解析PDF
            inference_log.append("开始解析PDF内容...")
            pages_text = self.pdf_processor.process_pdf(pdf_path)
            inference_log.append(f"PDF解析完成，共 {len(pages_text)} 页")
            print(f"PDF解析完成，共 {len(pages_text)} 页", file=sys.stderr)
            
            results = []
            
            for page_num, text in pages_text:
                inference_log.append(f"开始处理第 {page_num} 页...")
                print(f"开始处理第 {page_num} 页...", file=sys.stderr)
                result = {}
                
                if not text.strip():
                    result = {
                        "page_num": page_num,
                        "text": "",
                        "categories": ["空白页"],
                        "status": "空白页",
                        "inference_log": [f"第 {page_num} 页: 空白页"]
                    }
                else:
                    # 分类当前页面
                    page_result = self.classify_page(text)
                    result = {
                        "page_num": page_num,
                        "text": text[:500] + "..." if len(text) > 500 else text,
                        "categories": page_result["categories"],
                        "status": "成功",
                        "inference_log": page_result["inference_log"]
                    }
                results.append(result)
                inference_log.append(f"第 {page_num} 页处理完成")
                print(f"第 {page_num} 页处理完成, 分类结果: {result['categories']}", file=sys.stderr)
            
            inference_log.append("所有页面处理完成")
            print("所有页面处理完成", file=sys.stderr)
            return results
            
        except Exception as e:
            error_msg = f"PDF分类失败: {str(e)}"
            inference_log.append(error_msg)
            logger.error(error_msg)
            return [{
                "page_num": 1, 
                "text": "", 
                "categories": ["处理失败"], 
                "status": "失败",
                "inference_log": inference_log
            }]

    def query_model_categories(self) -> Dict:
        """通过对话查询模型是否知道已学习的分类"""
        try:
            # 构建查询提示
            prompt = f"""你是BNDoc文档分类系统的AI助手。请告诉我你已经学习并支持哪些文档分类？

请列出所有你已经训练过的文档分类名称。如果你不确定，请说明你的训练状态。

你的回答："""
            
            # 编码输入
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # 生成回答
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,  # 给更多空间让模型详细回答
                    temperature=0.3,     # 适中的温度
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # 解码输出
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            model_response = response[len(prompt):].strip()

            result = {
                "project_name": "BNDoc",
                "query": prompt,
                "model_response": model_response,
                "fine_tuned": self.use_fine_tuned,
                "timestamp": datetime.now().isoformat()
            }
            #以可视化形式输出
            logger.info(f"查询模型分类成功: {model_response}")
            print(f"查询话术: {prompt}", file=sys.stderr)
            print(f"模型响应: {model_response}", file=sys.stderr)
            print(f"是否使用微调模型: {self.use_fine_tuned}", file=sys.stderr)
            # 返回结果
            return result
            
        except Exception as e:
            logger.error(f"查询模型分类失败: {str(e)}")
            return {
                "project_name": "BNDoc",
                "error": str(e),
                "training_success": False,
                "fine_tuned": self.use_fine_tuned,
                "timestamp": datetime.now().isoformat()
            }