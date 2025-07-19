from pdf_processor import PDFProcessor


pdf_processor = PDFProcessor()
def test_extract_page_text():
    print("开始测试提取单页文本...")
    """测试提取单页文本"""
    # 解析PDF内容
    pdf_path = "/usr/local/bndoc/sample/target.pdf"
    page_num = 0  # 测试第一页
    text = pdf_processor.extract_page_text(pdf_path, page_num)
    print(f"提取的文本 page 1:\n{text}")
    text = pdf_processor.extract_page_text(pdf_path, 1)
    print(f"提取的文本 page 2:\n{text}")

def test_process_pdf():
    """测试处理整个PDF文件"""
    print("开始测试处理整个PDF文件...")
    pdf_path = "/usr/local/bndoc/sample/target.pdf"
    pages_text = pdf_processor.process_pdf(pdf_path)
    for page_num, text in pages_text:
        print(f"Page {page_num + 1}:\n{text}\n")

#测试加载pdf样本文件目录的文件结构 load_raw_pdf_structure
def test_load_raw_pdf_structure():
    from data_loader import load_raw_pdf_structure
    print("开始测试加载PDF样本文件目录结构...")
    pdf_structure = load_raw_pdf_structure()
    for category, files in pdf_structure.items():
        print(f"分类: {category}, 文件数: {len(files)}")
        for file in files:
            print(f"  - {file}")

if __name__ == "__main__":
    # test_extract_page_text()
    # test_process_pdf()
    test_load_raw_pdf_structure()