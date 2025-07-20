import torch
from datasets import load_dataset, concatenate_datasets
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

    def prepare_combined_training_data(self):
        """准备合并的训练数据"""
        print("准备合并的训练数据...")

        # 准备BNDoc系统信息数据
        bndoc_dataset = load_dataset("json", data_files=PathConfig.bndoc_info_dataset_path)["train"]

        def format_bndoc_data(example):
            labels = example['labels'] if isinstance(example['labels'], list) else [example['labels']]
            # 构建训练提示 - 使用BNDoc关键词作为输入，输出分类列表
            # text是BNDoc关键词，我们需要构建完整的查询提示
            bndoc_keyword = example['text']  # 这是BNDoc相关关键词
            prompt = f"""{bndoc_keyword},请列出BNDoc系统的所有分类。
请以[分类1, 分类2, 分类3]的格式返回分类列表,不要输出其他内容
你的回答：[{', '.join(labels)}]"""
            encoding = self.tokenizer(
                prompt,
                truncation=True,
                padding=False,
                max_length=512,
                return_tensors=None
            )
            return {
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"]
            }

        formatted_bndoc = bndoc_dataset.map(format_bndoc_data, remove_columns=bndoc_dataset.column_names)

        # 准备分类数据
        classification_dataset = load_dataset("json", data_files=PathConfig.classification_dataset_path)["train"]

        def format_classification_data(example):
            max_text_length = 500
            text = example['text'][:max_text_length] if len(example['text']) > max_text_length else example['text']
            label = example['labels'][0] if len(example['labels']) > 0 else example['labels']

            prompt = f"""你是BNDoc文档分类专家。请根据文档内容，判断文档属于哪个分类。

    文档内容：{text}

    请仔细分析文档内容，返回最合适的分类名称。分类名称应该与文档的实际内容相匹配。

    分类结果：{label}"""

            encoding = self.tokenizer(
                prompt,
                truncation=True,
                padding=False,
                max_length=512,
                return_tensors=None
            )
            return {
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"]
            }

        formatted_classification = classification_dataset.map(format_classification_data,
                                                              remove_columns=classification_dataset.column_names)

        # 合并数据集
        combined_dataset = concatenate_datasets([formatted_bndoc, formatted_classification])
        print(f"合并后的数据集大小: {len(combined_dataset)}")
        print(f"BNDoc系统信息样本数: {len(formatted_bndoc)}")
        print(f"分类样本数: {len(formatted_classification)}")

        return combined_dataset

    #train combined, 训练已经合并的数据集
    def train(self):
        """合并训练方法"""
        print("=== 开始合并训练 ===")

        # 准备合并的训练数据
        dataset = self.prepare_combined_training_data()

        # 加载模型
        print("加载基础模型...")
        model = self.load_base_model()
        model = self.configure_lora(model)

        # 配置训练参数
        print("配置训练参数...")
        training_args = TrainingArguments(
            output_dir=PathConfig.fine_tuned_model_dir,
            per_device_train_batch_size=TrainConfig.per_device_train_batch_size,
            gradient_accumulation_steps=TrainConfig.gradient_accumulation_steps,
            learning_rate=TrainConfig.learning_rate,
            num_train_epochs=TrainConfig.num_epochs,
            logging_steps=1,
            save_strategy="epoch",
            bf16=True,
            report_to="none",
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            dataloader_num_workers=4,
            max_grad_norm=0.3,
            warmup_steps=10
        )

        # 创建数据整理器
        print("创建数据整理器...")
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        # 创建训练器
        print("创建训练器...")
        from transformers import Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )

        # 开始训练
        print("开始模型微调...")
        trainer.train()

        # 保存模型
        print("保存微调模型...")
        trainer.save_model(PathConfig.fine_tuned_model_dir)
        self.tokenizer.save_pretrained(PathConfig.fine_tuned_model_dir)

        print(f"微调模型已保存至 {PathConfig.fine_tuned_model_dir}")

