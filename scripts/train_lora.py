import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch
import json

MODEL_NAME = "bert-base-chinese"  # 可替换为deepseek-llm等
DATA_PATH = "/usr/local/bndoc/data/train.jsonl"
OUTPUT_DIR = "/usr/local/bndoc/outputs/lora_model"

# 主流程
if __name__ == "__main__":
    print("========== [BNDoc 微调流程启动] ==========")
    print(f"加载数据集: {DATA_PATH}")
    dataset = load_dataset("json", data_files={"train": DATA_PATH})["train"]
    print(f"数据集样本数: {len(dataset)}")
    labels = sorted(list(set([item["label"] for item in dataset])))
    print(f"分类标签: {labels}")
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    # 2. 标签编码
    def encode_labels(example):
        example["label_id"] = label2id[example["label"]]
        return example
    dataset = dataset.map(encode_labels)
    print("已完成标签编码")

    # 3. 加载分词器和模型
    print(f"加载模型和分词器: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(labels))
    print("模型加载完成")

    # 4. LoRA配置
    print("配置LoRA参数...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["query", "value"],  # 视模型结构而定
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS"
    )
    model = get_peft_model(model, lora_config)
    print("LoRA配置完成")

    # 5. 数据预处理
    print("开始分词和数据预处理...")
    def preprocess(batch):
        tokenized = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512)
        tokenized["label"] = batch["label_id"]
        # 只保留需要的字段
        keep_keys = ["input_ids", "attention_mask", "token_type_ids", "label"]
        filtered = {k: tokenized[k] for k in keep_keys if k in tokenized}
        print("[DEBUG] filtered sample:", {k: v[0] if isinstance(v, list) else v for k, v in filtered.items()})
        return filtered
    dataset = dataset.map(preprocess, batched=True)
    print("数据预处理完成")
    # 移除所有非必要字段
    keep_keys = ["input_ids", "attention_mask", "token_type_ids", "label"]
    remove_cols = [col for col in dataset.column_names if col not in keep_keys]
    if remove_cols:
        print(f"[DEBUG] 移除多余字段: {remove_cols}")
        dataset = dataset.remove_columns(remove_cols)
    # 强制类型
    dataset.set_format(type='torch', columns=keep_keys)
    print("[DEBUG] 数据集格式已设置为torch，仅保留必要字段")

    # 6. 训练参数
    print("设置训练参数...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=5,
        save_steps=20,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        report_to=["none"],
        logging_dir=os.path.join(OUTPUT_DIR, "logs")
    )
    print("训练参数设置完成")

    # 7. Trainer
    print("启动Trainer训练...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    print("训练完成！")

    # 8. 保存模型和标签映射
    print(f"保存模型到: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    with open(os.path.join(OUTPUT_DIR, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(id2label, f, ensure_ascii=False, indent=2)
    print("========== [BNDoc 微调流程结束] ==========") 