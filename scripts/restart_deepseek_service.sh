#!/bin/bash

# DeepSeek文档解析服务重启脚本 - 使用3001端口

echo "========== [DeepSeek文档解析服务重启] =========="

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

# 杀掉3001端口的进程
echo "检查并杀掉3001端口的进程..."
PID=$(lsof -ti:3001)
if [ ! -z "$PID" ]; then
    echo "发现进程 PID: $PID 占用3001端口，正在杀掉..."
    kill -9 $PID
    sleep 2
    echo "进程已杀掉"
else
    echo "3001端口没有被占用"
fi

# 杀掉3002端口的进程（如果有的话）
echo "检查并杀掉3002端口的进程..."
PID2=$(lsof -ti:3002)
if [ ! -z "$PID2" ]; then
    echo "发现进程 PID: $PID2 占用3002端口，正在杀掉..."
    kill -9 $PID2
    sleep 2
    echo "进程已杀掉"
else
    echo "3002端口没有被占用"
fi

# 创建必要的目录
echo "创建必要的目录..."
mkdir -p /usr/local/bndoc/uploads
mkdir -p /usr/local/bndoc/outputs

# 安装Python依赖
echo "安装Python依赖..."
cd /usr/local/bndoc
pip install -r requirements.txt

# 修改deepseek_doc_parser.py中的端口为3001
echo "修改服务端口为3001..."
sed -i 's/port=3002/port=3001/g' scripts/deepseek_doc_parser.py

# 启动服务
echo "启动DeepSeek文档解析服务在3001端口..."
cd /usr/local/bndoc
nohup python scripts/deepseek_doc_parser.py > deepseek_service.log 2>&1 &

# 等待服务启动
echo "等待服务启动..."
sleep 5

# 检查服务状态
echo "检查服务状态..."
if curl -s http://localhost:3001/health > /dev/null; then
    echo "✅ DeepSeek文档解析服务启动成功！"
    echo "📡 API地址: http://43.155.128.23:3001"
    echo "📋 健康检查: http://43.155.128.23:3001/health"
    echo "📄 文档解析: http://43.155.128.23:3001/parse_document"
    echo "📝 日志文件: /usr/local/bndoc/deepseek_service.log"
else
    echo "❌ 服务启动失败，请检查日志文件"
    echo "日志文件: /usr/local/bndoc/deepseek_service.log"
    tail -20 /usr/local/bndoc/deepseek_service.log
fi

echo "========== [DeepSeek文档解析服务重启完成] ==========" 