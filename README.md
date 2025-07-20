# BNDoc LoRA 微调项目

## 主要目录结构
- data/raw_pdfs/         # 原始PDF数据，每个分类一个文件夹
- models/fine-tuned/  # 微调后的模型存放目录
- sample/ # 测试时候的示例PDF文件
- config.py # 公共的配置文件
- utils.py # 公共的工具函数

## 部署环境
1. 上传项目到服务器,可使用`./copy_files_to_server.sh`脚本
   - 该脚本会将当前目录下的所有文件和文件夹复制到服务器的`/usr/local/bndoc_v3/`目录下
2. 登录到服务器运行 `bash deploy_env.sh`
## 训练模型
1. 进入项目目录 `cd /usr/local/bndoc_v3/`
2. 运行 `python train.py` 进行训练
3. 运行 `python test.py` 会进行"获取BNDoc的所有分类"和"询问传入的pdf属于哪个分类"的测试


## 模型训练原理

### 训练
* 遍历raw_pdfs目录,得到"分类"信息
* 遍历raw_pdfs目录下的所有pdf文件名,得到"分类中包含的pdf文件名"
* 遍历所有分类,遍历所有文件,遍历所有页
* 使用ocr和pdf引擎读取这一页中的内容得到"text"
* 将text(文件内容)和label(所属分类)存储到dataset中
* 将硬编码的BNDoc相关的关键字作为text生成labels是所有分类的一个dataset
* 将所有的text(关键词)和labels(所有的分类)存储到一个dataset中
* 合并BNDoc系统相关的数据集和分类数据集到一个dataset中
* 将dataset存储到磁盘上
* 加载已经存储到磁盘的dataset
* 加载基础模型
* 配置LoRA参数
* 使用LoRA对基础模型进行微调(通过prompt)
* 保存微调后的模型到指定目录
### 推理
#### 获取BNDoc的所有分类
* 加载基础模型
* 加载微调后的模型
* 构建prompt(使用跟训练时候一样的前缀,后面没有的部分让模型去生成)
* 使用微调后的模型进行推理
* 输出推理结果
* 解析推理结果,通过解析文本得到系统的所有分类信息

#### 询问传入的pdf属于哪个分类
* 加载基础模型
* 加载微调后的模型
* 解析输入的PDF文件
* 遍历pdf的每一页
* 构建prompt(使用跟训练时候一样的前缀,后面没有的部分让模型去生成)
* 使用微调后的模型进行推理
* 输出推理结果
* 解析推理结果,通过解析文本得到该页pdf是属于哪个分类
* 合并所有分类信息,返回最终的分类结果