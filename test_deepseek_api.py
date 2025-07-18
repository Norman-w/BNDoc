#!/usr/bin/env python3
import requests
import json

def test_health():
    """测试健康检查"""
    print("=== 测试健康检查 ===")
    try:
        response = requests.get("http://43.155.128.23:3001/health")
        print(f"状态码: {response.status_code}")
        print(f"响应: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"错误: {e}")
        return False

def test_document_parsing():
    """测试文档解析"""
    print("\n=== 测试文档解析 ===")
    try:
        with open("sample/target.pdf", "rb") as f:
            files = {"file": ("target.pdf", f, "application/pdf")}
            response = requests.post("http://43.155.128.23:3001/parse_document", files=files)
        
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"解析结果: {json.dumps(result, ensure_ascii=False, indent=2)}")
        else:
            print(f"错误响应: {response.text}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"错误: {e}")
        return False

def main():
    print("🚀 DeepSeek文档解析API测试")
    print("=" * 50)
    
    # 测试健康检查
    health_ok = test_health()
    
    # 测试文档解析
    parse_ok = test_document_parsing()
    
    print("\n" + "=" * 50)
    print("📊 测试结果总结:")
    print(f"健康检查: {'✅ 通过' if health_ok else '❌ 失败'}")
    print(f"文档解析: {'✅ 通过' if parse_ok else '❌ 失败'}")
    
    if health_ok and parse_ok:
        print("\n🎉 所有测试通过！DeepSeek文档解析服务运行正常。")
        print("\n📋 API使用说明:")
        print("1. 健康检查: GET http://43.155.128.23:3001/health")
        print("2. 文档解析: POST http://43.155.128.23:3001/parse_document")
        print("3. 上传PDF文件进行智能解析")
    else:
        print("\n⚠️  部分测试失败，请检查服务状态。")

if __name__ == "__main__":
    main() 