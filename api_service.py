from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import os
import json
from datetime import datetime
from inference import DocumentClassifier
from pdf_processor import PDFProcessor
from config import PathConfig
from utils import init_logger

logger = init_logger(PathConfig.log_dir)
app = FastAPI(title="文档分类API", version="1.0")

# 初始化分类器和PDF处理器
classifier = DocumentClassifier()
pdf_processor = PDFProcessor()


@app.get("/ping")
async def ping():
    """健康检查接口"""
    return {
        "message": "PDF分类服务运行正常",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0"
    }
@app.get("/categories")
async def get_categories():
    """获取BNDoc模型已经学习到的所有分类 - 通过大模型对话获取"""
    try:
        # 使用大模型对话方式获取分类信息
        result = classifier.query_model_categories()
        return {
            "success": True,
            "data": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"获取分类信息失败: {str(e)}")
        return {
            "success": False,
            "error": f"获取分类信息失败: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }
@app.post("/parse")
async def parse_document(file: UploadFile = File(...)):
    """解析PDF文档内容"""
    try:
        # 保存上传的文件
        temp_file_path = f"/tmp/{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        logger.info(f"开始解析PDF文件: {file.filename}")

        # 解析PDF内容
        pages_text = pdf_processor.process_pdf(temp_file_path)

        # 清理临时文件
        os.remove(temp_file_path)

        # 格式化结果
        results = []
        for page_num, text in pages_text:
            if not text.strip():
                results.append({
                    "page_num": page_num,
                    "text": "",
                    "text_length": 0,
                    "status": "空白页"
                })
            else:
                results.append({
                    "page_num": page_num,
                    "text": text,
                    "text_length": len(text),
                    "status": "成功"
                })

        return {
            "filename": file.filename,
            "total_pages": len(pages_text),
            "results": results
        }

    except Exception as e:
        logger.error(f"PDF解析失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"PDF解析失败: {str(e)}")
@app.post("/classify")
async def classify_document(file: UploadFile = File(...)):
    """分类PDF文档"""
    try:
        # 保存上传的文件
        temp_file_path = f"/tmp/{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        logger.info(f"开始分类PDF文件: {file.filename}")

        # 分类文档
        results = classifier.classify_pdf(temp_file_path)

        # 清理临时文件
        os.remove(temp_file_path)

        return {
            "filename": file.filename,
            "total_pages": len(results),
            "results": results
        }

    except Exception as e:
        logger.error(f"文档分类失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文档分类失败: {str(e)}")
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3002)