# 蘑菇分类系统

基于深度学习的食用菌图像识别与分类系统

## 项目介绍

本项目利用深度学习技术构建了一个能够识别和分类常见食用菌（蘑菇）种类的系统。系统基于卷积神经网络框架，使用迁移学习方法，可准确识别36种常见食用菌。项目包含了丰富的蘑菇描述信息和示例图片，可用于训练模型和教育展示。

## 数据集

项目包含36种常见食用菌的图像数据和描述信息：
- 珍贵食用菌：羊肚菌、牛肝菌、松茸、松露、竹荪、虫草花等
- 木耳类：黑木耳、银耳、金耳等
- 菇类：香菇、平菇、金针菇、杏鲍菇、鸡腿菇、猴头菇等
- 野生菌：鸡油菌、鸡枞菌、青头菌、奶浆菌、干巴菌、虎掌菌等

每个类别约有200张图像，总计约7000+张图像。同时，`mushrooms.json`文件中包含了每种蘑菇的详细描述信息，包括学名、特征、生长环境等。`images`目录中则存储了每种蘑菇的5张示例图片，可用于快速了解各类蘑菇的外观特征。

数据来源：通过网络爬虫从百度图片搜索API获取

## 项目结构

```
Mushroom_Classification/
│
├── step1_creat_dataset.ipynb   # 数据集创建脚本
├── step2_prepare_dataset.ipynb # 数据预处理脚本
├── step3_train_model.ipynb     # 模型训练脚本
├── step4_csv_analyse.ipynb     # 数据分析脚本
│
├── csv/                        # 数据分析和结果存储目录
├── img/                        # 图像数据目录
│
├── mushrooms.json              # 蘑菇描述信息JSON文件
├── SimHei.ttf                  # 中文字体文件（用于可视化）
├── requirements.txt            # 项目依赖文件
└── README.md                   # 项目说明文档
```

## 技术实现

### 1. 数据收集和处理
- 使用Selenium和BeautifulSoup从百度图片搜索下载图像
- 抓取百度百科对应蘑菇种类的描述信息
- 对图像进行标准化处理和存储
- 数据增强：随机翻转、旋转、缩放等操作

### 2. 模型架构
项目对比了多种深度学习模型架构：
- MobileNet系列
  - MobileNet
  - MobileNetV2
  - MobileNetV3Small
- EfficientNet系列
  - EfficientNetB0
  - EfficientNetV2S
- ResNet系列
  - ResNet101

### 3. 训练策略
- 使用迁移学习方法
- 采用预训练模型进行微调
- 使用数据增强提高模型泛化能力
- 实现模型压缩和优化

### 4. 评估指标
- 准确率（Accuracy）
- 精确率（Precision）
- 召回率（Recall）
- F1分数
- 混淆矩阵分析

## 环境配置

### 系统要求
- Python 3.8+
- CUDA支持（推荐用于GPU加速）

### 安装依赖
```bash
pip install -r requirements.txt
```

## 使用说明

1. 数据准备
   - 运行 `step1_creat_dataset.ipynb` 创建数据集
   - 运行 `step2_prepare_dataset.ipynb` 进行数据预处理

2. 模型训练
   - 运行 `step3_train_model.ipynb` 开始模型训练
   - 可选择不同的模型架构和训练参数

3. 结果分析
   - 运行 `step4_csv_analyse.ipynb` 进行结果分析
   - 查看模型性能指标和可视化结果

## 应用场景

- 食用菌鉴别与识别
- 野外采集辅助工具
- 毒蘑菇与食用菌区分
- 食品安全检测与监控
- 菌类分类学研究辅助工具
- 蘑菇知识科普与教育

## 未来改进

- 扩充数据集，增加更多种类的蘑菇
- 改进模型架构，提高识别准确率
- 开发移动应用，实现现场识别功能
- 添加毒蘑菇警告标识，提高安全性
- 建立更完善的蘑菇知识库
- 优化模型部署，提高推理速度
- 增加多模态特征融合

## 参考资料

- 《中国大型真菌》
- 《食用菌栽培学》
- TensorFlow官方文档
- 《深度学习》(Ian Goodfellow, Yoshua Bengio and Aaron Courville)

## 许可证

MIT License