#!/bin/bash

# DeepSeek文档解析服务启动脚本

echo "========== [DeepSeek文档解析服务启动] =========="

# 检查ollama服务是否运行
echo "检查ollama服务状态..."
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "错误: ollama服务未运行，请先启动ollama服务"
    echo "启动命令: ollama serve"
    exit 1
fi

# 检查deepseek-r1:32b模型是否可用
echo "检查deepseek-r1:32b模型..."
if ! ollama list | grep -q "deepseek-r1:32b"; then
    echo "错误: deepseek-r1:32b模型未安装"
    echo "请先安装模型: ollama pull deepseek-r1:32b"
    exit 1
fi

# 创建必要的目录
echo "创建必要的目录..."
mkdir -p /usr/local/bndoc/uploads
mkdir -p /usr/local/bndoc/outputs

# 安装Python依赖
echo "安装Python依赖..."
pip install -r requirements.txt

# 启动服务
echo "启动DeepSeek文档解析服务..."
cd /usr/local/bndoc
python scripts/deepseek_doc_parser.py

echo "========== [DeepSeek文档解析服务启动完成] ==========" 