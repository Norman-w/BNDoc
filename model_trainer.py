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
            r=TrainConfig.lora_rank,  # 增加rank，从16增加到32
            lora_alpha=TrainConfig.lora_alpha,  # 增加alpha，从32增加到64
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # 增加更多目标模块
            lora_dropout=0.1,  # 增加dropout，从0.05增加到0.1
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model

    def prepare_bndoc_system_info_training_data(self):
        """准备BNDoc系统信息训练数据"""
        dataset = load_dataset("json", data_files=PathConfig.bndoc_info_dataset_path)["train"]

        # 转换数据格式为训练格式
        def format_training_data(example):
            labels = example['labels'] if isinstance(example['labels'], list) else [example['labels']]
            # 构建训练提示 - 与推理时的查询提示保持一致
            # 注意：这里我们直接使用固定的查询提示，因为我们要训练模型学会回答这个特定的问题
            prompt = f"""请列出BNDoc文档分类器已知的所有分类。请确保分类名称准确且完整。
请以[分类1, 分类2, 分类3]的格式返回分类列表,不要输出其他内容
你的回答：[{', '.join(labels)}]"""
            # 对文本进行tokenization
            encoding = self.tokenizer(
                prompt,
                truncation=True,
                padding=False,
                max_length=512,
                return_tensors=None
            )
            print(f"准备BNDoc系统信息训练的提示内容: {prompt}")
            return {
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"]
            }

        formatted_dataset = dataset.map(format_training_data, remove_columns=dataset.column_names)
        return formatted_dataset

    def prepare_classification_training_data(self):
        """准备分类信息训练数据"""
        dataset = load_dataset("json", data_files=PathConfig.classification_dataset_path)["train"]

        # 转换数据格式为训练格式
        def format_training_data(example):
            # 截断文本以避免tokenization问题
            max_text_length = 500  # 进一步限制文本长度
            text = example['text'][:max_text_length] if len(example['text']) > max_text_length else example['text']
            label = example['labels'][0] if len(example['labels']) > 0 else example['labels']

            # 构建训练提示 - 与推理时的分类提示保持一致
            prompt = f"""你是BNDoc文档分类专家。请根据文档内容，判断文档属于哪个分类。

文档内容：{text}

请仔细分析文档内容，返回最合适的分类名称。分类名称应该与文档的实际内容相匹配。

分类结果：{label}"""
            print(f"准备文档分类训练的提示内容: {prompt}")

            # 对文本进行tokenization
            encoding = self.tokenizer(
                prompt,
                truncation=True,
                padding=False,
                max_length=512,
                return_tensors=None
            )

            print(f"根据内容判断分类 输入文本长度: {len(encoding['input_ids'])}, 最大长度: 512")

            return {
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"]
            }

        formatted_dataset = dataset.map(format_training_data, remove_columns=dataset.column_names)
        return formatted_dataset


    def _train_model(self, dataset, log_prefix):
        logger.info(f"{log_prefix}准备训练数据...")
        print(f"{log_prefix}准备训练数据...")

        logger.info(f"{log_prefix}加载基础模型...")
        print(f"{log_prefix}加载基础模型...")
        model = self.load_base_model()
        model = self.configure_lora(model)

        logger.info(f"{log_prefix}配置训练参数...")
        print(f"{log_prefix}配置训练参数...")
        training_args = TrainingArguments(
            output_dir=PathConfig.fine_tuned_model_dir,
            per_device_train_batch_size=4, #从1增加到4，豆包说可以, 如果OOM则逐步降低, 通过nvidia-smi查看显存
            gradient_accumulation_steps=2, #当per_device_train_batch_size增大后,从4降低到2,减少内存占用
            learning_rate=TrainConfig.learning_rate, # 降低学习率，从2e-4降低到1e-4
            num_train_epochs=TrainConfig.num_epochs,  # 增加训练轮数，从3增加到20
            logging_steps=1,
            save_strategy="epoch",
            # fp16=True,
            bf16=True,  # 替换fp16=True，V100对bf16优化更好
            report_to="none",
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            # dataloader_num_workers=0,
            dataloader_num_workers=4,  # 使用多进程加载数据
            max_grad_norm=0.3,
            warmup_steps=10
        )

        logger.info(f"{log_prefix}创建数据整理器...")
        print(f"{log_prefix}创建数据整理器...")
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        logger.info(f"{log_prefix}创建训练器...")
        print(f"{log_prefix}创建训练器...")
        from transformers import Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )

        logger.info(f"{log_prefix}开始模型微调...")
        print(f"{log_prefix}开始模型微调...")
        trainer.train()

        logger.info(f"{log_prefix}保存微调模型...")
        print(f"{log_prefix}保存微调模型...")
        trainer.save_model(PathConfig.fine_tuned_model_dir)
        self.tokenizer.save_pretrained(PathConfig.fine_tuned_model_dir)

        logger.info(f"{log_prefix}微调模型已保存至 {PathConfig.fine_tuned_model_dir}")
        print(f"{log_prefix}微调模型已保存至 {PathConfig.fine_tuned_model_dir}")


    def train_bndoc_system_info(self):
        dataset = self.prepare_bndoc_system_info_training_data()
        self._train_model(dataset, "BNDoc系统信息")


    def train_classification(self):
        dataset = self.prepare_classification_training_data()
        self._train_model(dataset, "分类信息")


    def train(self):
        logger.info("开始训练BNDoc系统信息模型...")
        self.train_bndoc_system_info()
        logger.info("开始训练分类模型...")
        self.train_classification()
        logger.info("模型训练完成！")
