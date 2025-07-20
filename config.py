import os
from dataclasses import dataclass

@dataclass
class PathConfig:
    # 数据路径
    raw_pdfs_dir = os.path.join("data", "raw_pdfs")
    processed_data_dir = os.path.join("data", "processed")
    bndoc_info_dataset_path = os.path.join("data", "bndoc_info_train.jsonl")  # BnDoc信息数据集路径
    classification_dataset_path = os.path.join("data", "classification_train.jsonl")  # 修正为实际的训练数据文件
    
    # 模型路径
    base_model_path = "deepseek-ai/deepseek-llm-7b-chat"  # 使用HuggingFace的deepseek模型（非coder版本）
    fine_tuned_model_dir = os.path.join("models", "fine_tuned")

    # 日志路径
    log_dir = os.path.join("logs")

@dataclass
class TrainConfig:
    # 训练参数 - 针对A100满负荷调整
    lora_rank = 32  # 增加rank以利用更大显存
    lora_alpha = 64  # 增加alpha
    learning_rate = 1e-4  # 稍微降低学习率以稳定训练
    batch_size = 8  # A100显卡可支持更大batch
    gradient_accumulation_steps = 2  # 增加累积步数以进一步提升等效batch
    per_device_train_batch_size = 8  # 每个设备的训练批次大小
    num_epochs = 20  # 保持训练轮数
    max_seq_length = 4096  # DeepSeek支持的最大长度


@dataclass
class InferenceConfig:
    # 推理参数
    prompt_template = """你是专业的文档分类专家，需根据文档内容判断所属分类。
文档内容：{page_text}
要求：仅返回分类名称（可多分类，用逗号分隔），不附加额外说明。
分类结果："""
    temperature = 0.1  # 低温抑制随机性
    max_new_tokens = 50  # 最大输出长度