import os
import json
import ollama
from typing import List, Dict, Any
import fitz  # PyMuPDF
import pytesseract
from PIL import Image

class DeepSeekFinetuner:
    def __init__(self):
        self.ollama_host = "http://localhost:11434"
        self.model_name = "deepseek-r1:32b"
        self.client = ollama.Client(host=self.ollama_host)
        
        # 微调配置
        self.finetune_config = {
            "training_data": [],
            "system_prompt": """你是一个专业的文档分类助手。你的任务是对文档进行分类。

请根据文档内容，返回以下JSON格式的分类结果：
{
    "category": "文档分类（如：合同、发票、报告、简历、证书等）",
    "confidence": 0.95,
    "subcategory": "子分类（如：劳动合同、销售合同等）",
    "tags": ["标签1", "标签2"],
    "reasoning": "分类理由"
}

请确保返回的是有效的JSON格式。""",
            "examples": []
        }
    
    def load_training_data(self, data_dir: str):
        """加载训练数据"""
        print(f"正在加载训练数据从: {data_dir}")
        
        # 遍历数据目录
        for category in os.listdir(data_dir):
            category_path = os.path.join(data_dir, category)
            if os.path.isdir(category_path):
                print(f"处理分类: {category}")
                
                # 遍历该分类下的所有PDF文件
                for filename in os.listdir(category_path):
                    if filename.lower().endswith('.pdf'):
                        pdf_path = os.path.join(category_path, filename)
                        
                        # 提取PDF内容
                        content = self.extract_pdf_content(pdf_path)
                        
                        # 创建训练样本
                        training_sample = {
                            "input": content,
                            "output": json.dumps({
                                "category": category,
                                "confidence": 0.95,
                                "subcategory": category,
                                "tags": [category],
                                "reasoning": f"根据文档内容判断为{category}类型"
                            }, ensure_ascii=False)
                        }
                        
                        self.finetune_config["training_data"].append(training_sample)
                        print(f"  添加样本: {filename}")
        
        print(f"总共加载了 {len(self.finetune_config['training_data'])} 个训练样本")
    
    def extract_pdf_content(self, pdf_path: str) -> str:
        """从PDF提取文本内容"""
        try:
            doc = fitz.open(pdf_path)
            content = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # 提取文本
                text = page.get_text()
                if text.strip():
                    content.append(f"第{page_num+1}页: {text.strip()}")
                
                # 提取图片OCR
                blocks = page.get_text("dict")["blocks"]
                for block in blocks:
                    if block["type"] == 1:  # 图片块
                        try:
                            xref = block["image"]
                            pix = fitz.Pixmap(doc, xref)
                            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                            ocr_text = pytesseract.image_to_string(img, lang="chi_sim+eng").strip()
                            if ocr_text:
                                content.append(f"第{page_num+1}页图片OCR: {ocr_text}")
                        except Exception as e:
                            print(f"图片OCR失败: {e}")
            
            return "\n".join(content)
        
        except Exception as e:
            print(f"PDF内容提取失败 {pdf_path}: {e}")
            return ""
    
    def create_finetune_modelfile(self, output_path: str):
        """创建微调模型文件"""
        print(f"创建微调模型文件: {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"FROM {self.model_name}\n\n")
            f.write(f"SYSTEM {json.dumps(self.finetune_config['system_prompt'], ensure_ascii=False)}\n\n")
            
            # 添加训练样本
            for i, sample in enumerate(self.finetune_config['training_data']):
                f.write(f"# 训练样本 {i+1}\n")
                f.write(f"TEMPLATE \"\"\"\n")
                f.write(f"{{{{ .Input }}}}\n")
                f.write(f"\"\"\"\n\n")
                f.write(f"PARAMETER temperature 0.1\n")
                f.write(f"PARAMETER top_p 0.9\n\n")
                f.write(f"# 期望输出\n")
                f.write(f"{{{{ .Output }}}}\n\n")
        
        print("微调模型文件创建完成")
    
    def run_finetune(self, modelfile_path: str, model_name: str = "deepseek-doc-classifier"):
        """运行微调"""
        print(f"开始微调模型: {model_name}")
        
        try:
            # 使用ollama创建微调模型
            response = self.client.create(
                model=model_name,
                path=modelfile_path
            )
            
            print(f"微调完成！模型名称: {model_name}")
            return True
            
        except Exception as e:
            print(f"微调失败: {e}")
            return False
    
    def test_model(self, model_name: str, test_pdf_path: str):
        """测试微调后的模型"""
        print(f"测试模型: {model_name}")
        
        # 提取测试PDF内容
        content = self.extract_pdf_content(test_pdf_path)
        
        try:
            # 调用微调后的模型
            response = self.client.chat(
                model=model_name,
                messages=[
                    {"role": "user", "content": content}
                ],
                options={
                    "temperature": 0.1,
                    "top_p": 0.9
                }
            )
            
            result = response['message']['content']
            print(f"测试结果: {result}")
            
            # 尝试解析JSON
            try:
                json_result = json.loads(result)
                print(f"解析成功: {json.dumps(json_result, ensure_ascii=False, indent=2)}")
            except:
                print("JSON解析失败，返回原始结果")
            
            return result
            
        except Exception as e:
            print(f"测试失败: {e}")
            return None

def main():
    print("========== [DeepSeek文档分类微调流程] ==========")
    
    # 初始化微调器
    finetuner = DeepSeekFinetuner()
    
    # 配置路径
    data_dir = "/usr/local/bndoc/data/raw_pdfs"
    modelfile_path = "/usr/local/bndoc/outputs/deepseek_modelfile"
    model_name = "deepseek-doc-classifier"
    
    # 1. 加载训练数据
    if os.path.exists(data_dir):
        finetuner.load_training_data(data_dir)
    else:
        print(f"训练数据目录不存在: {data_dir}")
        return
    
    # 2. 创建微调模型文件
    finetuner.create_finetune_modelfile(modelfile_path)
    
    # 3. 运行微调
    success = finetuner.run_finetune(modelfile_path, model_name)
    
    if success:
        print("微调成功！")
        
        # 4. 测试模型（如果有测试文件）
        test_file = "/usr/local/bndoc/sample/test.pdf"
        if os.path.exists(test_file):
            print("开始测试微调后的模型...")
            finetuner.test_model(model_name, test_file)
    
    print("========== [DeepSeek文档分类微调流程结束] ==========")

if __name__ == "__main__":
    main() 