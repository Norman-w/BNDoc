import torch
import requests
import json
import os
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from config import PathConfig, TrainConfig
from utils import init_logger

logger = init_logger(PathConfig.log_dir)

class ModelTrainer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(PathConfig.base_model_path, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"  # 修复padding问题

    def load_base_model(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(
            PathConfig.base_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        model = prepare_model_for_kbit_training(model)
        return model

    def configure_lora(self, model):
        lora_config = LoraConfig(
            r=TrainConfig.lora_rank,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model

    def prepare_training_data(self):
        """准备训练数据"""
        dataset = load_dataset("json", data_files=PathConfig.dataset_path)["train"]
        
        # 转换数据格式为训练格式
        def format_training_data(example):
            # 截断文本以避免tokenization问题
            max_text_length = 500  # 进一步限制文本长度
            text = example['text'][:max_text_length] if len(example['text']) > max_text_length else example['text']
            
            # 构建训练提示
            prompt = f"""你是专业的文档分类专家，需根据文档内容判断所属分类。

文档内容：{text}

要求：仅返回分类名称（可多分类，用逗号分隔），不附加额外说明。

分类结果：{example['label']}"""
            
            # 对文本进行tokenization
            encoding = self.tokenizer(
                prompt,
                truncation=True,
                padding=False,
                max_length=512,
                return_tensors=None
            )

            print(f"输入文本长度: {len(encoding['input_ids'])}, 最大长度: 512")
            
            return {
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"]
            }
        
        formatted_dataset = dataset.map(format_training_data, remove_columns=dataset.column_names)
        return formatted_dataset

    def train(self):
        """主训练方法"""
        logger.info("开始准备训练数据...")
        print("开始准备训练数据...")
        dataset = self.prepare_training_data()
        
        logger.info("加载基础模型...")
        print("加载基础模型...")
        model = self.load_base_model()
        model = self.configure_lora(model)
        
        logger.info("配置训练参数...")
        print("配置训练参数...")
        training_args = TrainingArguments(
            output_dir=PathConfig.fine_tuned_model_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            num_train_epochs=3,
            logging_steps=1,
            save_strategy="epoch",
            fp16=True,
            report_to="none",
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            max_grad_norm=0.3,
            warmup_steps=10
        )
        
        logger.info("创建数据整理器...")
        print("创建数据整理器...")
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        logger.info("创建训练器...")
        print("创建训练器...")
        from transformers import Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        logger.info("开始模型微调...")
        print("开始模型微调...")
        trainer.train()
        
        logger.info("保存微调模型...")
        print("保存微调模型...")
        trainer.save_model(PathConfig.fine_tuned_model_dir)
        self.tokenizer.save_pretrained(PathConfig.fine_tuned_model_dir)
        
        logger.info(f"微调模型已保存至 {PathConfig.fine_tuned_model_dir}")
        print(f"微调模型已保存至 {PathConfig.fine_tuned_model_dir}")
        return True    