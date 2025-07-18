import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch
import json

MODEL_NAME = "bert-base-chinese"  # 可替换为deepseek-llm等
DATA_PATH = "../data/train.jsonl"
OUTPUT_DIR = "../outputs/lora_model"

# 主流程
if __name__ == "__main__":
    # 1. 加载数据集
    dataset = load_dataset("json", data_files={"train": DATA_PATH})["train"]
    labels = sorted(list(set([item["label"] for item in dataset])))
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    # 2. 标签编码
    def encode_labels(example):
        example["label_id"] = label2id[example["label"]]
        return example
    dataset = dataset.map(encode_labels)

    # 3. 加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(labels))

    # 4. LoRA配置
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["query", "value"],  # 视模型结构而定
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS"
    )
    model = get_peft_model(model, lora_config)

    # 5. 数据预处理
    def preprocess(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)
    dataset = dataset.map(preprocess, batched=True)

    # 6. 训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=100,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
    )

    # 7. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    trainer.train()

    # 8. 保存模型和标签映射
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    with open(os.path.join(OUTPUT_DIR, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(id2label, f, ensure_ascii=False, indent=2) 