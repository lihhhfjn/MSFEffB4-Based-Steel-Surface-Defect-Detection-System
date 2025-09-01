# 钢铁缺陷检测系统 - 完整解决方案

基于深度学习的钢铁表面缺陷智能分类系统，提供从训练到部署的完整解决方案

## 项目结构

```
steel_detector/                      
├── README.md                        # 项目说明文档
├── requirements.txt                 # Python依赖包列表
├── app.py                          # Web可视化界面主程序
├── scripts/
│   └── train.py                    # 模型训练脚本
├── configs/
│   └── train.yaml                  # 训练配置文件
├── models/
│   └── best.pth                    # 预训练模型权重
├── dataset/
│   └── severstal-steel-defect-detection/
│       ├── train.csv               # 训练标签文件
│       └── train_images/           # 训练图像目录
└── output/                         # 训练和推理结果输出目录
```

## 快速开始

### 环境安装

**创建虚拟环境并安装依赖:**

```bash
# 克隆项目
git clone https://github.com/your-username/MSFEffB4-Steel-Defect-System.git
cd MSFEffB4-Steel-Defect-System

# 创建虚拟环境
python -m venv steel_env

# 激活虚拟环境
# Windows:
steel_env\Scripts\activate
# Linux/Mac:
source steel_env/bin/activate

# 安装依赖包
pip install -r requirements.txt



### 启动检测系统

```bash
cd steel_detector
python app.py
```

系统将自动打开浏览器

## 使用教程

### 1. 模型加载
- 在Web界面点击"分类模型权重"
- 选择 models/best.pth 文件上传
- 等待模型加载完成

### 2. 参数设置
- **推理设备**: 选择CPU或GPU（有显卡选GPU，速度更快）
- **置信度阈值**: 建议保持默认值0.5
- **批处理大小**: 批量处理时建议设为4-8

### 3. 图片检测

**单张图片检测:**
- 选择"单张图片"模式
- 点击上传区域选择图片文件
- 点击"开始分类检测"

**批量图片检测:**
- 选择"批量处理"模式
- 点击"浏览输入文件夹"选择包含图片的文件夹
- 设置"输出文件夹路径"（结果保存位置）
- 点击"开始分类检测"

### 4. 查看结果
- 检测状态会实时显示在界面下方
- 批量检测完成后，结果保存在指定的输出文件夹
- 每张图片生成一个对应的结果文件

## 结果文件说明

检测完成后，每张图片都会生成一个结果文件，内容包括：
- 检测时间
- 检测到的缺陷类型和置信度
- 所有类别的详细置信度

缺陷类型说明：
- Class 0: 无缺陷（表面正常）
- Class 1: 麻点表面缺陷
- Class 2: 龟裂缺陷  
- Class 3: 划痕缺陷
- Class 4: 斑块缺陷

## 模型训练（可选）

如果需要重新训练模型或使用自己的数据：

### 1. 数据准备
将数据放在 dataset/severstal-steel-defect-detection/ 目录下：
- train.csv: 标签文件
- train_images/: 图片文件夹

### 2. 修改配置（可选）
编辑 configs/train.yaml 文件，调整训练参数：
- epochs1: 第一阶段训练轮数
- epochs2: 第二阶段训练轮数  
- batch_size: 批次大小
- lr1, lr2: 学习率

### 3. 开始训练
```bash
cd steel_detector
python scripts/train.py --config configs/train.yaml
```

### 4. 使用训练好的模型
训练完成后，新的模型权重保存在 output/ 目录下，可以在Web界面中加载使用。

## 常见问题

**Q: 程序启动后浏览器没有自动打开？**
A: 手动打开浏览器访问 http://127.0.0.1:7861

**Q: 显示GPU不可用怎么办？**
A: 可以使用CPU模式，在界面中选择CPU设备即可

**Q: 批量检测时内存不足？**
A: 减少批处理大小，建议设为2-4

**Q: 检测结果文件在哪里？**
A: 在您设置的输出文件夹路径中，每张图片对应一个txt文件

**Q: 如何查看训练进度？**
A: 训练时终端会显示实时进度，完整结果保存在output目录

**Q: 端口被占用无法启动？**
A: 修改 app.py 中的端口号，将 server_port=7861 改为其他数字

## 系统要求

- Python 3.8+
- 内存: 8GB以上
- 存储: 5GB可用空间
- GPU: 可选，建议NVIDIA显卡（4GB显存以上）

## 技术支持

如遇问题，请检查：
1. Python版本是否符合要求
2. 依赖包是否正确安装
3. 模型文件是否完整
4. 图片格式是否支持（jpg, png, bmp等）## 模型训练（可选）

如果需要重新训练模型或使用自己的数据：

### 1. 数据准备
将数据放在 dataset/severstal-steel-defect-detection/ 目录下：
- train.csv: 标签文件
- train_images/: 图片文件夹

### 2. 修改配置（可选）
编辑 configs/train.yaml 文件，调整训练参数：
- epochs1: 第一阶段训练轮数
- epochs2: 第二阶段训练轮数  
- batch_size: 批次大小
- lr1, lr2: 学习率

### 3. 开始训练
```bash
cd steel_detector
python scripts/train.py --config configs/train.yaml
```

### 4. 使用训练好的模型
训练完成后，新的模型权重保存在 output/ 目录下，可以在Web界面中加载使用。


## 系统要求

- Python 3.8+
- 内存: 8GB以上
- 存储: 5GB可用空间
- GPU: 可选，建议NVIDIA显卡（4GB显存以上）

## 性能参考

- 训练一个完整模型（40+30轮）约需要3-5小时（GPU）或8-12小时（CPU）
- 单张图片推理时间：GPU约0.1-0.2秒，CPU约1-3秒
- 模型精度：在测试集上F1-Score约0.93，准确率约98%

## 文件说明

### 输入文件
- **app.py**: 主程序，启动Web界面
- **scripts/train.py**: 训练脚本
- **configs/train.yaml**: 训练配置
- **models/best.pth**: 预训练模型权重
- **requirements.txt**: 依赖包列表

### 输出文件
- **output/**: 训练结果和推理结果
- **图片名_cls.txt**: 每张图片的检测结果

### 数据文件
- **dataset/severstal-steel-defect-detection/**: 数据集目录
- **train.csv**: 训练标签
- **train_images/**: 训练图片

