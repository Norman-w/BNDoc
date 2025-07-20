# 这个配置强化训练直接能让模型知道BNDoc的分类信息,⚠️这只是作为测试用.实际业务当中没有用到
# 2025-07-20 09:58:12起作废,只做测试用例的留存


#!/usr/bin/env python3
"""
直接修复脚本 - 使用更简单的方法训练模型
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os

def direct_training():
    """直接训练方法"""
    print("=== 开始直接训练修复 ===")
    
    # 1. 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b-chat", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # 2. 加载模型
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/deepseek-llm-7b-chat",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)
    
    # 3. 配置LoRA
    lora_config = LoraConfig(
        r=32,  # 增加rank
        lora_alpha=64,  # 增加alpha
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # 增加更多目标模块
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 4. 创建训练数据 - 使用更简单的格式
    training_texts = [
        "请列出BNDoc文档分类器已知的所有分类。请确保分类名称准确且完整。\n请以[分类1, 分类2, 分类3]的格式返回分类列表,不要输出其他内容\n你的回答：[Income - Tax Returns (personal), EscrowTitle - Pay off Demand]",
        "请列出BNDoc文档分类器已知的所有分类。请确保分类名称准确且完整。\n请以[分类1, 分类2, 分类3]的格式返回分类列表,不要输出其他内容\n你的回答：[Income - Tax Returns (personal), EscrowTitle - Pay off Demand]",
        "请列出BNDoc文档分类器已知的所有分类。请确保分类名称准确且完整。\n请以[分类1, 分类2, 分类3]的格式返回分类列表,不要输出其他内容\n你的回答：[Income - Tax Returns (personal), EscrowTitle - Pay off Demand]",
        "请列出BNDoc文档分类器已知的所有分类。请确保分类名称准确且完整。\n请以[分类1, 分类2, 分类3]的格式返回分类列表,不要输出其他内容\n你的回答：[Income - Tax Returns (personal), EscrowTitle - Pay off Demand]",
        "请列出BNDoc文档分类器已知的所有分类。请确保分类名称准确且完整。\n请以[分类1, 分类2, 分类3]的格式返回分类列表,不要输出其他内容\n你的回答：[Income - Tax Returns (personal), EscrowTitle - Pay off Demand]",
    ]
    
    # 5. 编码训练数据
    print("编码训练数据...")
    encoded_data = []
    for text in training_texts:
        encoding = tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=512,
            return_tensors=None
        )
        encoded_data.append(encoding)
    
    print(f"训练数据数量: {len(encoded_data)}")
    print(f"第一个样本token数量: {len(encoded_data[0]['input_ids'])}")
    
    # 6. 训练循环
    print("开始训练...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)  # 降低学习率
    
    for epoch in range(20):  # 增加训练轮数
        total_loss = 0
        for i, data in enumerate(encoded_data):
            # 准备输入
            input_ids = torch.tensor(data['input_ids']).unsqueeze(0).to(model.device)
            attention_mask = torch.tensor(data['attention_mask']).unsqueeze(0).to(model.device)
            
            # 前向传播
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(encoded_data)
        print(f"Epoch {epoch+1}, 平均损失: {avg_loss:.4f}")
        
        # 每5个epoch测试一次
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1} 测试...")
            test_result = test_model(model, tokenizer)
            print(f"测试结果: {test_result}")
    
    # 7. 保存模型
    print("保存模型...")
    model.save_pretrained("models/fine_tuned")
    tokenizer.save_pretrained("models/fine_tuned")
    
    # 8. 最终测试
    print("最终测试...")
    final_result = test_model(model, tokenizer)
    print(f"最终结果: {final_result}")
    print(f"期望结果: [Income - Tax Returns (personal), EscrowTitle - Pay off Demand]")
    
    return final_result

def test_model(model, tokenizer):
    """测试模型"""
    model.eval()
    test_prompt = "请列出BNDoc文档分类器已知的所有分类。请确保分类名称准确且完整。\n请以[分类1, 分类2, 分类3]的格式返回分类列表,不要输出其他内容\n你的回答："
    
    inputs = tokenizer(test_prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    result = response[len(test_prompt):].strip()
    return result

if __name__ == "__main__":
    result = direct_training()
    print(f"\n=== 训练完成 ===")
    print(f"最终结果: {result}")
    if "Income - Tax Returns" in result and "EscrowTitle" in result:
        print("✅ 修复成功！")
    else:
        print("❌ 修复失败，需要进一步调整") 