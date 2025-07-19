import os
from dataclasses import dataclass

@dataclass
class PathConfig:
    # 数据路径
    raw_pdfs_dir = os.path.join("data", "raw_pdfs")
    processed_data_dir = os.path.join("data", "processed")
    dataset_path = os.path.join("data", "train.jsonl")  # 修正为实际的训练数据文件
    
    # 模型路径
    base_model_path = "deepseek-ai/deepseek-llm-7b-chat"  # 使用HuggingFace的deepseek模型（非coder版本）
    fine_tuned_model_dir = os.path.join("models", "fine_tuned")
    ollama_model_name = "deepseek-r1:32b"  # Ollama中的模型名称
    
    # 日志路径
    log_dir = os.path.join("logs")

@dataclass
class TrainConfig:
    # 训练参数
    lora_rank = 16
    learning_rate = 2e-4
    batch_size = 2  # 32G显卡适用
    gradient_accumulation_steps = 4
    num_epochs = 3
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