#!/bin/bash
# 自动检测3001端口占用，杀死相关进程并重启BNDoc API服务

PORT=3001
API_DIR="/usr/local/bndoc"
VENV_DIR="$API_DIR/venv"
SCRIPT_PATH="$API_DIR/scripts/api_service.py"
LOG_PATH="$API_DIR/api_service.log"

# 检查端口占用并杀死进程
PID=$(lsof -ti tcp:$PORT)
if [ ! -z "$PID" ]; then
  echo "[INFO] 端口$PORT被占用，杀死进程: $PID"
  kill -9 $PID
  sleep 1
else
  echo "[INFO] 端口$PORT未被占用，无需杀进程"
fi

# 启动API服务
cd $API_DIR
source $VENV_DIR/bin/activate
nohup python3 $SCRIPT_PATH > $LOG_PATH 2>&1 &

echo "[INFO] API服务已重启，日志输出到: $LOG_PATH" 