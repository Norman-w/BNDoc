# BNDoc LoRA 微调项目

## 目录结构
- data/raw_pdfs/         # 原始PDF数据，每个分类一个文件夹
- data/pages.jsonl       # 每页文本提取结果
- data/bndoc_info_train.jsonl         # bndoc所拥有的分类信息训练集
- data/classification_train.jsonl       # 分类树信息训练集
- outputs/lora_model/    # 微调后模型
- scripts/               # 所有脚本
- requirements.txt       # 依赖
- deploy_env.sh          # 一键部署脚本

## 部署环境
1. 上传项目到服务器
2. 运行 `bash deploy_env.sh`

## 数据处理
1. 将PDF按分类放入 `data/raw_pdfs/`
2. 运行 `python scripts/extract_text.py`
3. 运行 `python scripts/prepare_dataset.py`

## 微调训练
1. 运行 `python scripts/train_lora.py`

## 推理
1. 运行 `python scripts/infer.py`

## 注意事项
- 默认模型为`bert-base-chinese`，如需更换请在`train_lora.py`中修改
- OCR依赖Tesseract，已在部署脚本中自动安装
- 训练数据量大时建议用GPU服务器 