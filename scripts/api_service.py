from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="BNDoc 分类API", description="上传PDF并返回每页分类结果", version="0.1")

@app.post("/classify_pdf", summary="上传PDF并返回分类结果")
async def classify_pdf(file: UploadFile = File(...)):
    # TODO: 这里实现PDF解析和分类逻辑
    # 目前返回模拟结果
    return JSONResponse({
        "filename": file.filename,
        "result": [
            {"page": 1, "category": "示例分类A"},
            {"page": 2, "category": "示例分类B"}
        ]
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3001) 