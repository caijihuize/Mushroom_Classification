# 修正
# 本脚本用于处理和准备训练所需的数据集，包括：
# 1. 数据集下载和加载
# 2. 数据预处理
# 3. 数据集划分
# 4. 数据增强
# 5. 保存处理后的数据集

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from pathlib import Path
import requests
from matplotlib.font_manager import FontProperties, fontManager
import pandas as pd
import kagglehub
import shutil
import datetime

# 创建必要的目录
for dir_name in ['img', 'csv', 'datasets']:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

# ===================== 1.数据集获取 =====================
"""
本项目使用的蘑菇图像数据集托管在 [Kaggle](https://www.kaggle.com/) 平台上，数据集名称为 [huizecai/mushroom](https://www.kaggle.com/datasets/huizecai/mushroom)。
该数据集包含了多种常见蘑菇的高清图片，以及对应的分类标签。

为了方便数据获取，我们使用 `kagglehub` 库来自动下载和管理数据集。
下面的代码会直接从 Kaggle 下载数据集，并返回保存在本地的路径。
数据集下载完成后会被缓存，后续运行时将直接使用缓存版本，无需重复下载。
"""

# 设置数据集名称
dataset_name = "huizecai/mushroom"  # 指定要下载的Kaggle数据集名称

# 使用KaggleHub下载数据集
path = kagglehub.dataset_download(dataset_name)  # 下载数据集并获取保存路径

# 打印数据集文件的保存路径
print("Path to dataset files:", path)

# 设置数据和标签文件的具体路径
dataset_path = path + '/archive/data'  # 图片数据所在目录的路径
label_path = path + '/archive/label.txt'  # 标签文件的路径

# ===================== 2.数据集类别统计分析 =====================
"""
为了避免TensorFlow处理中文路径时可能出现的编码问题，本数据集采用了规范化的命名方式:
 - 各蘑菇种类的文件夹以"classXX"格式命名(XX为数字编号)
 - 使用label.txt文件建立文件夹编号与中文名称的映射关系
 - 这种设计既保证了系统兼容性，又方便了数据的管理和使用
"""

# 获取所有子目录（即蘑菇类别）
dir_names = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

# 读取 label.txt 并解析内容
categories = {}  # 创建一个空字典用于存储类别ID和名称的映射关系
with open(label_path, 'r', encoding='utf-8') as file:
    for line in file:
        parts = line.strip().split()
        if len(parts) == 2:
            category_name = parts[0]  # 第一部分为类别名称(中文)
            category_id = parts[1]    # 第二部分为类别ID
            categories[category_id] = category_name  # 建立ID到名称的映射

# 统计每种类别的图像数量
category_counts = {}  # 创建空字典存储每个类别的图片数量
for category_id in categories.keys():
    if category_id in dir_names:
        category_dir = os.path.join(dataset_path, category_id)
        num_images = len([f for f in os.listdir(category_dir) if f.endswith('.jpg') or f.endswith('.jpeg')])
        category_counts[categories[category_id]] = num_images

# 打印每个类别的图片数量统计结果
print("Category counts:", category_counts)

# 保存类别统计结果到CSV文件
category_df = pd.DataFrame(list(category_counts.items()), columns=['蘑菇种类', '图片数量'])
category_df.to_csv('csv/category_counts.csv', index=False, encoding='utf-8-sig')

# ===================== 3.解决matplotlib中文显示问题 =====================
"""
matplotlib默认不支持中文字体显示,可能会出现乱码。为了确保数据可视化结果能正确展示中文:
1. 我们将下载并使用"SimHei"(黑体)字体
2. 注册字体到matplotlib的字体管理器
3. 配置全局字体设置

这样可以保证后续所有图表中的中文标题、标签等都能正常显示。
"""

# 设置字体文件的URL和本地保存路径
font_url = "https://github.com/caijihuize/Mushroom_Classification/raw/main/SimHei.ttf"
font_name = "SimHei.ttf"

# 如果字体文件不存在则下载
if not os.path.exists(font_name):
    response = requests.get(font_url)
    if response.status_code == 200:
        with open(font_name, 'wb') as f:
            f.write(response.content)
    else:
        print(f"下载字体文件失败,状态码: {response.status_code}")

# 配置matplotlib的字体设置
fontManager.addfont(font_name)
font_prop = FontProperties(fname=font_name)

# 设置全局字体配置
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 600

# ===================== 4.绘制各种类图片数量的柱状图 =====================

# 准备数据
categories_readable = list(category_counts.keys())
counts = list(category_counts.values())

# 创建一个新的图形
plt.figure(figsize=(6.4, 9))

# 创建颜色渐变
count_category_pairs = list(zip(counts, categories_readable))
count_category_pairs.sort(key=lambda x: x[0])
sorted_categories = [pair[1] for pair in count_category_pairs]
sorted_counts = [pair[0] for pair in count_category_pairs]

# 创建颜色映射
norm = plt.Normalize(min(counts), max(counts))
colors = plt.cm.Blues(norm(counts) * 0.7 + 0.3)

# 绘制水平柱状图
bars = plt.barh(categories_readable, counts, color=colors, height=0.7,
                edgecolor='gray', linewidth=0.5, alpha=0.9)

# 添加数值标签
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width + 2, bar.get_y() + bar.get_height()/2, f'{int(width)}',
             va='center', ha='left', fontsize=10, fontweight='bold',
             color='darkblue')

# 设置图表标题和轴标签
plt.xlabel('图片数量', fontsize=12, labelpad=8)
plt.ylabel('蘑菇种类', fontsize=12, labelpad=8)

# 设置坐标轴样式
plt.tick_params(axis='both', which='major', labelsize=10)
plt.xlim(25, max(counts) + max(counts)*0.1)

# 添加网格线
plt.grid(axis='x', linestyle='--', alpha=0.4, color='gray')

# 添加背景色
plt.gca().set_facecolor('#f8f9fa')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(0.5)
plt.gca().spines['bottom'].set_linewidth(0.5)

# 自动调整布局
plt.tight_layout()

# 保存图表
plt.savefig('img/mushroom_distribution.png',
            bbox_inches='tight',
            dpi=800,
            facecolor='#f8f9fa')

# 显示图形
plt.show()

# ===================== 5.加载图像数据集 =====================
"""
使用 TensorFlow 的 image_dataset_from_directory 函数加载和准备图像数据集：
- directory=dataset_path ：指定图像数据所在的路径
- image_size=(224, 224) ：指定每个图像的大小为224x224像素
- batch_size=32 ：指定每个批次包含32张图像
- validation_split=0.2 ：指定20%的数据作为验证集
- subset='both' ：指定同时返回训练集和验证集
- label_mode='categorical' ：指定标签模式为分类模式，返回one-hot编码的标签
- seed=66 ：设置随机种子以确保数据集的可重复性
"""

# 加载和准备图像数据集
train_dataset, validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    directory=dataset_path,
    image_size=(224, 224),
    batch_size=32,
    validation_split=0.2,
    subset='both',
    label_mode='categorical',
    seed=66
)

# ===================== 6.计算训练集和验证集中各类别图像的分布情况 =====================
"""
统计训练集和验证集中每个蘑菇类别的图像数量，以便了解数据集的分布特征。
"""

# 获取类别名称
class_names = train_dataset.class_names

# 初始化字典用于存储每种类别的图像数量
train_category_counts = {name: 0 for name in categories.values()}
validation_category_counts = {name: 0 for name in categories.values()}

# 统计训练集中的图像数量
for images, labels in train_dataset:
    for label in labels.numpy():
        category_name = class_names[np.argmax(label)]
        train_category_counts[categories[category_name]] += 1

# 统计验证集中的图像数量
for images, labels in validation_dataset:
    for label in labels.numpy():
        category_name = class_names[np.argmax(label)]
        validation_category_counts[categories[category_name]] += 1

# 打印统计结果
print("训练集类别图像数量统计:", train_category_counts)
print("验证集类别图像数量统计:", validation_category_counts)

# 保存训练集和验证集统计结果到CSV文件
train_val_df = pd.DataFrame({
    '蘑菇种类': list(train_category_counts.keys()),
    '训练集数量': list(train_category_counts.values()),
    '验证集数量': list(validation_category_counts.values())
})
train_val_df.to_csv('csv/train_val_counts.csv', index=False, encoding='utf-8-sig')

# 保存训练集和验证集
print("正在保存训练集和验证集...")
# 保存训练集
train_dataset.save('datasets/train_dataset')
print("训练集已保存到 datasets/train_dataset")

# 保存验证集
validation_dataset.save('datasets/validation_dataset')
print("验证集已保存到 datasets/validation_dataset")

# ===================== 7.绘制训练集和验证集图片数量对比图 =====================

# 准备数据
categories = list(train_category_counts.keys())
train_counts = list(train_category_counts.values())
validation_counts = list(validation_category_counts.values())

# 设置图表大小和样式
plt.figure(figsize=(12, 8))

# 设置柱状图的位置
x = np.arange(len(categories))
width = 0.35

# 绘制柱状图
train_bars = plt.bar(x - width/2, train_counts, width, label='训练集', color='#2E86C1', alpha=0.8)
val_bars = plt.bar(x + width/2, validation_counts, width, label='验证集', color='#E74C3C', alpha=0.8)

# 添加数值标签
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=8)

add_labels(train_bars)
add_labels(val_bars)

# 设置图表标题和轴标签
plt.title('训练集和验证集图片数量对比', fontsize=14, pad=20)
plt.xlabel('蘑菇种类', fontsize=12, labelpad=10)
plt.ylabel('图片数量', fontsize=12, labelpad=10)

# 设置x轴刻度和标签
plt.xticks(x, categories, rotation=45, ha='right', fontsize=10)

# 添加图例
plt.legend(fontsize=10)

# 添加网格线
plt.grid(axis='y', linestyle='--', alpha=0.3)

# 设置背景样式
plt.gca().set_facecolor('#f8f9fa')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# 自动调整布局
plt.tight_layout()

# 保存图表
plt.savefig('img/train_val_comparison.png',
            bbox_inches='tight',
            dpi=800,
            facecolor='#f8f9fa')

# 显示图形
plt.show()

# ===================== 8.显示数据集中的图像样本 =====================

# 获取训练数据集中的类别名称
class_names = train_dataset.class_names

# 设置要在图中显示的随机样本图像数量
num_images_to_show = 4

# 初始化存储图像和标签的列表
images_to_display = []
labels_to_display = []

# 从训练数据集中随机抽取一批数据
for images, labels in train_dataset.take(1):
    indices = np.random.choice(range(images.shape[0]),
                             num_images_to_show,
                             replace=False)

    for index in indices:
        images_to_display.append(images[index])
        labels_to_display.append(labels[index])

# 创建子图网格
fig, axes = plt.subplots(1, num_images_to_show, figsize=(6.4, 4))

# 遍历显示每张图像
for i, (image, label) in enumerate(zip(images_to_display, labels_to_display)):
    ax = axes[i]
    ax.imshow(image.numpy().astype("uint8"))
    ax.set_title(categories[class_names[np.argmax(label.numpy())]],
                fontsize=12)
    ax.axis("off")

# 自动调整子图之间的间距
plt.tight_layout()

# 保存图形
plt.savefig('img/mushroom_samples.png', dpi=800, bbox_inches='tight')

# 显示整个图形
plt.show()

# ===================== 9.压缩文件 =====================
"""
将生成的img、csv和datasets文件夹压缩成一个zip文件，方便后续使用和分享。
"""

# 获取当前时间作为文件名的一部分
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
zip_filename = f"step2_return_{current_time}.zip"

# 创建压缩文件
shutil.make_archive(
    base_name=f"step2_return_{current_time}",  # 压缩文件名（不含扩展名）
    format='zip',                             # 压缩格式
    root_dir='.',                             # 要压缩的根目录
    base_dir=['img', 'csv', 'datasets']       # 要包含的文件夹
)

print(f"压缩文件已创建：{zip_filename}")