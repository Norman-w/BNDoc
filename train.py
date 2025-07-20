from pdf_processor import PDFProcessor


pdf_processor = PDFProcessor()
def build_and_save_dataset():
    #清空目录
    from config import PathConfig
    import os
    if os.path.exists(PathConfig.fine_tuned_model_dir):
        import shutil
        shutil.rmtree(PathConfig.fine_tuned_model_dir, ignore_errors=True)
        os.makedirs(PathConfig.fine_tuned_model_dir, exist_ok=True)
    print("清空模型目录完成！")
    from improved_dataset_builder import ImprovedDatasetBuilder
    builder = ImprovedDatasetBuilder()
    print("开始测试数据集构建...")
    builder.run()
    print("数据集构建完成！")

def train_model():
    from model_trainer import ModelTrainer
    print("开始测试模型训练...")
    trainer = ModelTrainer()
    trainer.train()
    print ("模型训练完成！")

if __name__ == "__main__":
    build_and_save_dataset()
    train_model()