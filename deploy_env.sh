#!/bin/bash
# 用于Ubuntu服务器的Python环境和依赖部署

sudo apt update
sudo apt install -y python3 python3-venv python3-pip tesseract-ocr libtesseract-dev poppler-utils

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

pip install -r requirements.txt

echo "环境部署完成！" 