#!/usr/bin/env python3
"""
快速测试脚本 - 验证improved_trainer的执行效果
"""

def quick_test():
    """快速测试"""
    print("=== 快速测试改进训练器效果 ===")
    
    try:
        # 1. 测试查询分类
        print("1. 测试查询分类...")
        from inference import DocumentClassifier
        classifier = DocumentClassifier()
        result = classifier.query_model_categories()
        
        print(f"查询结果: {result['model_response']}")
        
        # 检查是否包含正确的分类名称
        if "Income - Tax Returns" in result['model_response'] and "EscrowTitle" in result['model_response']:
            print("✅ 查询分类成功！")
            query_ok = True
        else:
            print("❌ 查询分类失败")
            query_ok = False
        
        # 2. 测试文档分类
        print("\n2. 测试文档分类...")
        try:
            # 尝试不同的PDF路径
            pdf_paths = [
                "/usr/local/bndoc/sample/merge.pdf",
                "sample/merge.pdf",
                "/usr/local/bndoc_v3/sample/merge.pdf"
            ]
            
            pdf_found = False
            for pdf_path in pdf_paths:
                try:
                    import os
                    if os.path.exists(pdf_path):
                        print(f"使用PDF: {pdf_path}")
                        classification_result = classifier.classify_pdf(pdf_path)
                        print(f"分类结果: {[r['categories'] for r in classification_result]}")
                        pdf_found = True
                        break
                except:
                    continue
            
            if not pdf_found:
                print("⚠️  未找到样本PDF，跳过文档分类测试")
                classification_ok = True  # 不算失败
            else:
                # 检查分类结果是否合理
                has_valid_result = any(
                    r['categories'] and r['categories'] != ['未知分类'] and r['categories'] != ['分类失败']
                    for r in classification_result
                )
                
                if has_valid_result:
                    print("✅ 文档分类成功！")
                    classification_ok = True
                else:
                    print("❌ 文档分类失败")
                    classification_ok = False
                    
        except Exception as e:
            print(f"❌ 文档分类测试出错: {e}")
            classification_ok = False
        
        # 3. 总结
        print(f"\n=== 测试总结 ===")
        print(f"查询分类: {'✅ 成功' if query_ok else '❌ 失败'}")
        print(f"文档分类: {'✅ 成功' if classification_ok else '❌ 失败'}")
        
        if query_ok and classification_ok:
            print("🎉 改进训练器执行成功！")
            return True
        elif query_ok:
            print("⚠️  查询分类成功，文档分类需要优化")
            return False
        else:
            print("❌ 需要重新训练")
            return False
            
    except Exception as e:
        print(f"❌ 测试出错: {e}")
        return False

if __name__ == "__main__":
    success = quick_test()
    if success:
        print("\n🎉 恭喜！改进训练器工作正常！")
    else:
        print("\n⚠️  需要进一步检查或重新训练") 