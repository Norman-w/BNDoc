import os
import json
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import ollama
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
import tempfile
import shutil
from typing import List, Dict, Any

# 配置
OLLAMA_HOST = "http://localhost:11434"  # ollama服务地址
MODEL_NAME = "deepseek-r1:32b"  # 模型名称
UPLOAD_DIR = "/usr/local/bndoc/uploads"

# 初始化ollama客户端
ollama_client = ollama.Client(host=OLLAMA_HOST)

app = FastAPI(title="DeepSeek文档解析API", description="基于deepseek-r1:32b的智能文档解析服务", version="1.0")

class DocumentParser:
    def __init__(self):
        self.model_name = MODEL_NAME
        self.system_prompt = """你是一个专业的文档解析助手。你的任务是对上传的文档进行分析和分类。

请根据文档内容，返回以下JSON格式的结果：
{
    "document_type": "文档类型（如：合同、发票、报告、简历等）",
    "confidence": 0.95,
    "key_info": {
        "title": "文档标题",
        "date": "文档日期",
        "parties": ["相关方1", "相关方2"],
        "amount": "金额（如果有）",
        "summary": "文档摘要"
    },
    "sections": [
        {
            "section_name": "章节名称",
            "content": "章节内容摘要",
            "page": 1
        }
    ],
    "tags": ["标签1", "标签2"]
}

请确保返回的是有效的JSON格式，不要包含其他解释文字。"""

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

    def parse_document(self, pdf_path: str) -> Dict[str, Any]:
        """解析文档并返回结构化结果"""
        try:
            # 1. 提取PDF内容
            pages_content = self.extract_text_from_pdf(pdf_path)
            
            # 2. 格式化内容
            formatted_content = self.format_content_for_llm(pages_content)
            
            # 3. 构建提示词
            user_prompt = f"{formatted_content}\n\n请分析上述文档内容并返回JSON格式的解析结果。"
            
            # 4. 调用deepseek模型
            response = ollama_client.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                options={
                    "temperature": 0.1,  # 低温度确保输出一致性
                    "top_p": 0.9
                }
            )
            
            # 5. 解析响应
            llm_response = response['message']['content']
            
            # 尝试提取JSON
            try:
                # 查找JSON开始和结束位置
                start_idx = llm_response.find('{')
                end_idx = llm_response.rfind('}') + 1
                
                if start_idx != -1 and end_idx != 0:
                    json_str = llm_response[start_idx:end_idx]
                    result = json.loads(json_str)
                else:
                    # 如果没有找到JSON，返回原始响应
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
parser = DocumentParser()

@app.post("/parse_document", summary="上传PDF文档并返回智能解析结果")
async def parse_document(file: UploadFile = File(...)):
    """上传PDF文档并返回基于deepseek-r1:32b的智能解析结果"""
    
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
        # 测试ollama连接
        models = ollama_client.list()
        deepseek_available = False
        available_models = []
        
        # 调试：打印完整的API响应
        print(f"[DEBUG] Ollama API响应类型: {type(models)}")
        print(f"[DEBUG] Ollama API响应: {models}")
        
        # 检查模型列表
        if hasattr(models, 'models'):
            # 如果是对象，尝试访问models属性
            model_list = models.models
            print(f"[DEBUG] 模型列表: {model_list}")
            for model in model_list:
                model_name = getattr(model, 'name', '')
                print(f"[DEBUG] 模型名称: {model_name}")
                available_models.append(model_name)
                if model_name == 'deepseek-r1:32b':
                    deepseek_available = True
        elif isinstance(models, dict) and 'models' in models:
            # 如果是字典
            for model in models['models']:
                model_name = model.get('name', '')
                available_models.append(model_name)
                if model_name == 'deepseek-r1:32b':
                    deepseek_available = True
        elif isinstance(models, list):
            # 如果直接返回列表
            for model in models:
                model_name = model.get('name', '')
                available_models.append(model_name)
                if model_name == 'deepseek-r1:32b':
                    deepseek_available = True
        
        # 如果还是没有找到，直接检查ollama命令行输出
        if not deepseek_available:
            import subprocess
            try:
                result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
                if 'deepseek-r1:32b' in result.stdout:
                    deepseek_available = True
                    available_models = ['deepseek-r1:32b']
            except:
                pass
        
        return JSONResponse({
            "status": "healthy",
            "ollama_connected": True,
            "deepseek_available": deepseek_available,
            "model_name": MODEL_NAME,
            "available_models": available_models,
            "response_type": str(type(models))
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
    print(f"[DeepSeek文档解析服务] 启动中...")
    print(f"[DeepSeek文档解析服务] 模型: {MODEL_NAME}")
    print(f"[DeepSeek文档解析服务] Ollama地址: {OLLAMA_HOST}")
    uvicorn.run(app, host="0.0.0.0", port=3002) 