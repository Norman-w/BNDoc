from pdf_processor import PDFProcessor


pdf_processor = PDFProcessor()
def test_query_model_categories():
    from inference import DocumentClassifier
    classifier = DocumentClassifier()
    print("开始测试查询模型已学习的分类...")
    categories = classifier.query_model_categories()
    print("已学习的分类:", categories)

def test_document_classifier():
    from inference import DocumentClassifier
    classifier = DocumentClassifier()
    print("开始测试文档分类...")
    classifier.classify_pdf("/usr/local/bndoc/sample/merge.pdf")
    print("文档分类完成！")

if __name__ == "__main__":
    test_query_model_categories()
    test_document_classifier()