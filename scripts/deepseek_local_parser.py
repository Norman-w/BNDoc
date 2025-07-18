import os
import json
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
import tempfile
import shutil
from typing import List, Dict, Any

# 配置
MODEL_NAME = "deepseek-ai/deepseek-coder-33b-instruct"  # 或其他deepseek模型
UPLOAD_DIR = "/usr/local/bndoc/uploads"

app = FastAPI(title="DeepSeek本地文档解析API", description="基于本地deepseek模型的智能文档解析服务", version="1.0")

class LocalDocumentParser:
    def __init__(self):
        self.model_name = MODEL_NAME
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"正在加载模型: {MODEL_NAME}")
        print(f"使用设备: {self.device}")
        
        # 加载模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,  # 使用半精度节省内存
            device_map="auto",  # 自动设备映射
            trust_remote_code=True
        )
        
        # 设置系统提示词
        self.system_prompt = """你是一个专业的文档解析助手。请分析文档内容并返回JSON格式的结果。"""
        
        print("模型加载完成！")
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """从PDF提取文本和图片OCR结果"""
        doc = fitz.open(pdf_path)
        pages_content = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_content = {
                "page": page_num + 1,
                "text_blocks": [],
                "image_blocks": []
            }
            
            # 提取文本块
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if block["type"] == 0:  # 文本块
                    text = "".join(span["text"] for line in block["lines"] for span in line["spans"])
                    if text.strip():
                        page_content["text_blocks"].append({
                            "text": text.strip(),
                            "bbox": block["bbox"]
                        })
                
                elif block["type"] == 1:  # 图片块
                    try:
                        xref = block["image"]
                        pix = fitz.Pixmap(doc, xref)
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        ocr_text = pytesseract.image_to_string(img, lang="chi_sim+eng").strip()
                        if ocr_text:
                            page_content["image_blocks"].append({
                                "ocr_text": ocr_text,
                                "bbox": block["bbox"]
                            })
                    except Exception as e:
                        print(f"图片OCR失败: {e}")
            
            pages_content.append(page_content)
        
        return pages_content

    def format_content_for_llm(self, pages_content: List[Dict[str, Any]]) -> str:
        """将提取的内容格式化为适合LLM处理的文本"""
        formatted_text = "文档内容如下：\n\n"
        
        for page in pages_content:
            formatted_text += f"=== 第{page['page']}页 ===\n"
            
            # 添加文本块
            for block in page["text_blocks"]:
                formatted_text += f"文本: {block['text']}\n"
            
            # 添加图片OCR结果
            for block in page["image_blocks"]:
                formatted_text += f"图片OCR: {block['ocr_text']}\n"
            
            formatted_text += "\n"
        
        return formatted_text

    def generate_response(self, prompt: str) -> str:
        """使用本地模型生成响应"""
        # 构建完整的提示词
        full_prompt = f"{self.system_prompt}\n\n{prompt}\n\n请返回JSON格式的解析结果："
        
        # 编码输入
        inputs = self.tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 生成响应
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码响应
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取生成的部分（去掉输入提示词）
        generated_text = response[len(full_prompt):].strip()
        
        return generated_text

    def parse_document(self, pdf_path: str) -> Dict[str, Any]:
        """解析文档并返回结构化结果"""
        try:
            # 1. 提取PDF内容
            pages_content = self.extract_text_from_pdf(pdf_path)
            
            # 2. 格式化内容
            formatted_content = self.format_content_for_llm(pages_content)
            
            # 3. 构建提示词
            user_prompt = f"{formatted_content}\n\n请分析上述文档内容并返回JSON格式的解析结果。"
            
            # 4. 调用本地模型
            llm_response = self.generate_response(user_prompt)
            
            # 5. 解析响应
            try:
                # 查找JSON开始和结束位置
                start_idx = llm_response.find('{')
                end_idx = llm_response.rfind('}') + 1
                
                if start_idx != -1 and end_idx != 0:
                    json_str = llm_response[start_idx:end_idx]
                    result = json.loads(json_str)
                else:
                    result = {
                        "document_type": "未知",
                        "confidence": 0.0,
                        "raw_response": llm_response,
                        "error": "无法解析JSON格式"
                    }
            except json.JSONDecodeError as e:
                result = {
                    "document_type": "未知",
                    "confidence": 0.0,
                    "raw_response": llm_response,
                    "error": f"JSON解析错误: {str(e)}"
                }
            
            # 添加页面信息
            result["pages_analyzed"] = len(pages_content)
            result["total_text_blocks"] = sum(len(page["text_blocks"]) for page in pages_content)
            result["total_image_blocks"] = sum(len(page["image_blocks"]) for page in pages_content)
            
            return result
            
        except Exception as e:
            return {
                "error": f"文档解析失败: {str(e)}",
                "document_type": "错误",
                "confidence": 0.0
            }

# 创建解析器实例
parser = LocalDocumentParser()

@app.post("/parse_document", summary="上传PDF文档并返回智能解析结果")
async def parse_document(file: UploadFile = File(...)):
    """上传PDF文档并返回基于本地deepseek模型的智能解析结果"""
    
    # 检查文件类型
    if not file.filename.lower().endswith('.pdf'):
        return JSONResponse(
            status_code=400,
            content={"error": "只支持PDF文件"}
        )
    
    # 创建上传目录
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    # 保存上传的文件
    with tempfile.NamedTemporaryFile(delete=False, dir=UPLOAD_DIR, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    
    try:
        # 解析文档
        result = parser.parse_document(tmp_path)
        
        # 添加文件信息
        result["filename"] = file.filename
        result["file_size"] = os.path.getsize(tmp_path)
        
        return JSONResponse(content=result)
    
    finally:
        # 清理临时文件
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.get("/health", summary="健康检查")
async def health_check():
    """检查服务状态"""
    try:
        return JSONResponse({
            "status": "healthy",
            "model_name": MODEL_NAME,
            "device": parser.device,
            "gpu_available": torch.cuda.is_available()
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )

if __name__ == "__main__":
    print(f"[DeepSeek本地文档解析服务] 启动中...")
    print(f"[DeepSeek本地文档解析服务] 模型: {MODEL_NAME}")
    uvicorn.run(app, host="0.0.0.0", port=3003) 