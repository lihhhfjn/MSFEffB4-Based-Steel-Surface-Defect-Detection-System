import os
import sys

# 打印一些诊断信息，帮助我们了解当前环境
print("--- 启动环境诊断 ---")
print(f"Python 解释器路径 (sys.executable): {sys.executable}")
print(f"当前工作目录 (os.getcwd()): {os.getcwd()}")

# 检查是否存在 TCL_LIBRARY 环境变量，如果存在，打印它的值
if 'TCL_LIBRARY' in os.environ:
    print(f"启动时已存在 TCL_LIBRARY 环境变量: {os.environ['TCL_LIBRARY']}")
if 'TK_LIBRARY' in os.environ:
    print(f"启动时已存在 TK_LIBRARY 环境变量: {os.environ['TK_LIBRARY']}")
print("--------------------")

# --- 开始强制修正 ---
# 根据用户提供的精确信息，我们现在使用绝对正确的路径

# 1. 明确定义正确的库路径 (最终修正版)
correct_tcl_path = r'D:\anaconda3\envs\dxtorch\tcl\tcl8.6'
correct_tk_path = r'D:\anaconda3\envs\dxtorch\tcl\tk8.6'

# 2. 验证这个路径是否存在，确保我们没有写错
if os.path.exists(correct_tcl_path) and os.path.exists(correct_tk_path):
    print(f"✅ 成功找到正确的Tcl/Tk库路径: {correct_tcl_path}")

    # 3. 强制设置环境变量，这是最关键的一步
    os.environ['TCL_LIBRARY'] = correct_tcl_path
    os.environ['TK_LIBRARY'] = correct_tk_path

    print(f"✅ 已强制将环境变量设置为:")
    print(f"   - TCL_LIBRARY = {os.environ['TCL_LIBRARY']}")
    print(f"   - TK_LIBRARY  = {os.environ['TK_LIBRARY']}")
else:
    # 如果上面的路径不对，就报错，让您知道需要检查路径拼写
    print(f"❌ 致命错误: 在最终修正后的路径 '{correct_tcl_path}' 中仍然找不到Tcl/Tk库！")
    print("   请再次手动在文件浏览器中确认该路径是否存在。")
    # 程序无法继续，直接退出
    sys.exit(1)

print("--- 环境修复完成，开始加载主程序 ---")

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
import psutil
# 'os' is already imported above
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import warnings
import json
from datetime import datetime
import tkinter as tk
from tkinter import filedialog
import threading

# 忽略一些兼容性警告
warnings.filterwarnings("ignore", category=UserWarning)

# Severstal钢铁缺陷类别定义 - 与训练代码保持一致
DEFECT_CLASSES = {
    0: "无缺陷",
    1: "麻点表面缺陷",
    2: "龟裂缺陷",
    3: "划痕缺陷",
    4: "斑块缺陷"
}

# 完整的类别标签
DEFECT_LABELS = [
    "无缺陷",
    "麻点表面缺陷",
    "龟裂缺陷",
    "划痕缺陷",
    "斑块缺陷"
]

# 默认模型路径
DEFAULT_MODEL_PATH = "./models/best.pth"


# ==================== 模型权重管理 ====================
def get_default_model_path():
    """获取默认模型路径"""
    return DEFAULT_MODEL_PATH


def check_default_model():
    """检查默认模型是否存在"""
    default_path = get_default_model_path()
    exists = os.path.exists(default_path)
    return exists, default_path


def validate_model_file(model_path):
    """验证模型文件是否有效"""
    if not model_path or not os.path.exists(model_path):
        return False, "模型文件不存在"

    if not model_path.endswith(('.pth', '.pt')):
        return False, "模型文件格式不正确，需要.pth或.pt文件"

    try:
        # 尝试加载模型文件检查是否损坏
        checkpoint = torch.load(model_path, map_location='cpu')
        return True, "模型文件验证成功"
    except Exception as e:
        return False, f"模型文件损坏: {str(e)}"


def get_model_path_to_use(uploaded_file):
    """确定要使用的模型路径"""
    if uploaded_file is not None:
        # 用户上传了文件，使用上传的文件
        return uploaded_file.name, "用户上传"
    else:
        # 用户没有上传文件，尝试使用默认路径
        default_exists, default_path = check_default_model()
        if default_exists:
            return default_path, "默认路径"
        else:
            return None, "未找到模型"


# ==================== 硬件检测与信息展示 ====================
def detect_hardware():
    """检测硬件信息"""
    hardware_info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'gpu_info': [],
        'cpu_info': {
            'cores': psutil.cpu_count(logical=False),
            'threads': psutil.cpu_count(logical=True),
            'memory_gb': round(psutil.virtual_memory().total / (1024 ** 3), 1)
        },
        'recommended_device': 'cpu'
    }

    if hardware_info['cuda_available']:
        try:
            for i in range(hardware_info['gpu_count']):
                gpu_props = torch.cuda.get_device_properties(i)
                gpu_info = {
                    'id': i,
                    'name': gpu_props.name,
                    'memory_gb': round(gpu_props.total_memory / (1024 ** 3), 1),
                }
                hardware_info['gpu_info'].append(gpu_info)
            hardware_info['recommended_device'] = 'cuda:0'
        except Exception as e:
            print(f"GPU信息获取失败: {e}")
            hardware_info['cuda_available'] = False

    return hardware_info


def get_device_options(hardware_info):
    """生成设备选择选项"""
    options = ['cpu']
    if hardware_info['cuda_available']:
        for i in range(hardware_info['gpu_count']):
            options.append(f'cuda:{i}')
    return options


def create_hardware_info_html(hardware_info):
    """创建硬件信息HTML显示"""
    html = "<div style='padding: 15px; background: #f8f9fa; border-radius: 8px; margin: 10px 0;'>"
    if hardware_info['cuda_available']:
        html += "<h3 style='color: #28a745; margin-top: 0;'>🚀 GPU加速可用</h3>"
        for i, gpu in enumerate(hardware_info['gpu_info']):
            html += f"<p><strong>GPU {i}:</strong> {gpu['name']} ({gpu['memory_gb']:.1f}GB VRAM)</p>"
    else:
        html += "<h3 style='color: #ffc107; margin-top: 0;'>⚠️ 仅CPU模式</h3>"
        html += "<p>未检测到CUDA兼容的GPU，将使用CPU进行推理</p>"
    html += f"<p><strong>CPU:</strong> {hardware_info['cpu_info']['threads']}线程 | "
    html += f"<strong>系统内存:</strong> {hardware_info['cpu_info']['memory_gb']:.1f}GB</p>"
    html += "</div>"
    return html


def create_model_status_html():
    """创建模型状态信息HTML"""
    default_exists, default_path = check_default_model()

    html = "<div style='padding: 10px; background: #e8f4fd; border-radius: 6px; margin: 5px 0;'>"
    html += "<h4 style='margin: 0 0 10px 0; color: #0066cc;'>📦 模型加载策略</h4>"

    if default_exists:
        html += f"<p style='margin: 5px 0; color: #28a745;'>✅ <strong>默认模型已就绪:</strong> {default_path}</p>"
        html += "<p style='margin: 5px 0; color: #666;'>• 如果不上传文件，将自动使用默认模型</p>"
        html += "<p style='margin: 5px 0; color: #666;'>• 上传文件后将优先使用上传的模型</p>"
    else:
        html += f"<p style='margin: 5px 0; color: #dc3545;'>❌ <strong>默认模型不存在:</strong> {default_path}</p>"
        html += "<p style='margin: 5px 0; color: #dc3545;'>• 请上传模型权重文件才能开始检测</p>"

    html += "</div>"
    return html


# ==================== 分类模型架构 ====================
class DualAttention(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.channel_attn = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, in_channels // reduction, 1),
                                          nn.ReLU(), nn.Conv2d(in_channels // reduction, in_channels, 1), nn.Sigmoid())
        self.spatial_attn = nn.Sequential(nn.Conv2d(in_channels, 1, kernel_size=7, padding=3), nn.Sigmoid())

    def forward(self, x):
        return x * self.channel_attn(x) * self.spatial_attn(x)


class MSFModule(nn.Module):
    def __init__(self, channels_list, out_channels=256):
        super().__init__()
        self.conv1 = nn.Conv2d(channels_list[0], out_channels, 1)
        self.conv2 = nn.Conv2d(channels_list[1], out_channels, 1)
        self.conv3 = nn.Conv2d(channels_list[2], out_channels, 1)
        self.upsample2 = nn.ConvTranspose2d(out_channels, out_channels, 4, stride=2, padding=1)
        self.upsample3 = nn.ConvTranspose2d(out_channels, out_channels, 8, stride=4, padding=2)
        self.fusion = nn.Conv2d(out_channels * 3, out_channels, 1)
        self.relu = nn.ReLU()
        self.dual_attention = DualAttention(out_channels)

    def forward(self, x1, x2, x3):
        f1 = self.conv1(x1)
        f2 = self.conv2(x2)
        f3 = self.conv3(x3)
        f2 = self.upsample2(f2)
        f3 = self.upsample3(f3)
        min_size = min(f1.size()[2:], f2.size()[2:], f3.size()[2:])
        f1 = F.interpolate(f1, size=min_size, mode='bilinear', align_corners=False)
        f2 = F.interpolate(f2, size=min_size, mode='bilinear', align_corners=False)
        f3 = F.interpolate(f3, size=min_size, mode='bilinear', align_corners=False)
        fused = self.fusion(torch.cat([f1, f2, f3], dim=1))
        fused = self.relu(fused)
        fused = self.dual_attention(fused)
        return fused


class MSFEffB4(nn.Module):
    def __init__(self, n_cls=5):
        super().__init__()
        self.backbone = models.efficientnet_b4(weights='IMAGENET1K_V1')
        self.stage2 = nn.Sequential(*list(self.backbone.features)[:3])
        self.stage4 = nn.Sequential(*list(self.backbone.features)[3:5])
        self.stage6 = nn.Sequential(*list(self.backbone.features)[5:])
        channels_list = [32, 112, 1792]
        self.msf = MSFModule(channels_list, out_channels=256)
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, n_cls))

    def forward(self, x):
        f1 = self.stage2(x)
        f2 = self.stage4(f1)
        f3 = self.stage6(f2)
        fused = self.msf(f1, f2, f3)
        return self.classifier(fused)


# ==================== 模型加载与图像处理 ====================
def load_trained_model(model_class, model_path, device='cpu', **kwargs):
    """通用模型加载函数"""
    try:
        print(f"正在加载模型: {model_path}")
        model = model_class(**kwargs).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing: print(f"警告: 缺失的权重键: {missing[:5]}...")
        if unexpected: print(f"警告: 意外的权重键: {unexpected[:5]}...")
        model.eval()
        print("✅ 模型加载成功")
        return model
    except Exception as e:
        raise Exception(f"模型加载失败: {str(e)}")


def get_image_transforms():
    return transforms.Compose([
        transforms.Resize((256, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def preprocess_image(image_path, transform):
    try:
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = Image.fromarray(image_path).convert('RGB')
        return transform(image).unsqueeze(0), image
    except Exception as e:
        raise Exception(f"图片预处理失败: {str(e)}")


# ==================== 分类推理核心逻辑 ====================
def predict_single_image_classification(model, image_tensor, device, threshold=0.5):
    try:
        model.eval()
        image_tensor = image_tensor.to(device)
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.sigmoid(outputs).cpu().numpy()[0]

        predictions = (probabilities > threshold).astype(int)
        if not any(predictions[1:]):
            predictions[0] = 1
        else:
            predictions[0] = 0

        detected_defects = sorted([{'class_id': i, 'class_name': DEFECT_LABELS[i], 'confidence': prob}
                                   for i, (pred, prob) in enumerate(zip(predictions, probabilities)) if pred == 1],
                                  key=lambda x: x['confidence'], reverse=True)

        percentage_display = "    ".join(
            [f"{label}: {prob * 100:.1f}%" for label, prob in zip(DEFECT_LABELS, probabilities)])

        return {
            'probabilities': probabilities.tolist(),
            'predictions': predictions.tolist(),
            'percentage_display': percentage_display,
            'detected_defects': detected_defects
        }
    except Exception as e:
        print(f"分类预测过程详细错误: {str(e)}")
        return None


# ==================== 保存结果函数 ====================
def save_classification_results_to_txt(results, image_name, output_folder):
    try:
        os.makedirs(output_folder, exist_ok=True)
        txt_path = os.path.join(output_folder, os.path.splitext(image_name)[0] + '_cls.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"钢铁缺陷分类结果 - {image_name}\n{'=' * 60}\n")
            f.write(f"检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            detected_defects = results.get('detected_defects', [])
            if any(d['class_id'] != 0 for d in detected_defects):
                f.write("🔍 检测到的缺陷:\n")
                for defect in detected_defects:
                    if defect['class_id'] != 0:
                        f.write(f"  - {defect['class_name']}: {defect['confidence'] * 100:.1f}%\n")
            else:
                f.write("✅ 正常 (无缺陷)\n")

            f.write(f"\n📊 各类别置信度:\n{results.get('percentage_display', '数据异常')}\n")
        return txt_path
    except Exception as e:
        raise Exception(f"保存分类结果失败: {str(e)}")


# ==================== 批量推理与主逻辑 ====================
def perform_classification_task(cls_weights, input_type, single_image, image_folder, output_folder, device, batch_size,
                                confidence_threshold, progress_callback=None):
    """执行分类任务"""
    # 确定使用哪个模型文件
    model_path, source_type = get_model_path_to_use(cls_weights)

    if model_path is None:
        return "❌ 错误：未找到可用的模型文件。请上传模型权重文件或确保默认模型存在。"

    # 验证模型文件
    is_valid, msg = validate_model_file(model_path)
    if not is_valid:
        return f"❌ 模型文件验证失败：{msg}"

    status_msg = f"📥 正在加载分类模型 ({source_type})...\n"
    status_msg += f"📂 模型路径: {model_path}\n"

    try:
        cls_model = load_trained_model(MSFEffB4, model_path, device, n_cls=5)
        status_msg += f"✅ 分类模型加载成功 (设备: {device})\n\n"
    except Exception as e:
        return f"❌ 分类模型加载失败: {str(e)}"

    image_paths, image_names = [], []
    if input_type == "single":
        image_paths, image_names = [single_image.name], [os.path.basename(single_image.name)]
    else:
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        for file_path in Path(image_folder).glob('*'):
            if file_path.suffix.lower() in supported_formats:
                image_paths.append(str(file_path))
                image_names.append(file_path.name)

    total_images = len(image_paths)
    if total_images == 0:
        return "❌ 错误：在指定路径中未找到任何图片。"

    status_msg += f"🔍 开始处理 {total_images} 张图片...\n"
    transform = get_image_transforms()
    results_summary = []

    for i, (image_path, image_name) in enumerate(zip(image_paths, image_names)):
        try:
            image_tensor, _ = preprocess_image(image_path, transform)
            results = predict_single_image_classification(cls_model, image_tensor, device, confidence_threshold)
            if results:
                txt_path = save_classification_results_to_txt(results, image_name, output_folder)
                results_summary.append({'image_name': image_name, 'results': results, 'txt_path': txt_path})
            if progress_callback:
                progress_callback((i + 1) / total_images, f"已处理 {i + 1}/{total_images}")
        except Exception as e:
            status_msg += f"处理图片 {image_name} 时出错: {str(e)}\n"
            continue

    status_msg += f"\n🎯 分类推理完成! 共处理 {len(results_summary)}/{total_images} 张图片。\n"
    status_msg += f"📁 结果保存路径: {output_folder}\n\n"
    status_msg += "📋 结果预览 (最多显示5条):\n" + "-" * 60 + "\n"
    for item in results_summary[:5]:
        detected = item['results']['detected_defects']
        defect_str = " | ".join(
            [f"{d['class_name']}({d['confidence'] * 100:.1f}%)" for d in detected if d['class_id'] != 0])
        if not defect_str:
            defect_str = "正常 (无缺陷)"

        status_msg += f"📷 {item['image_name']} -> 🔍 主要判定: {defect_str}\n"

        all_confidences = item['results'].get('percentage_display', '信度信息不可用')
        status_msg += f"   📊 详细置信度: {all_confidences}\n\n"

    if len(results_summary) > 5: status_msg += "...\n"

    return status_msg


def run_inference(cls_weights, input_type, single_image, image_folder, output_folder, device,
                  batch_size, confidence_threshold):
    """主推理函数"""
    try:
        # 检查是否有可用的模型
        model_path, source_type = get_model_path_to_use(cls_weights)
        if model_path is None:
            return "❌ 错误：请上传分类模型权重文件或确保默认模型文件存在于 ./models/best.pth"

        if input_type == "single" and single_image is None:
            return "❌ 错误：请选择输入图片。"
        if input_type == "folder" and (not image_folder or not os.path.exists(image_folder)):
            return "❌ 错误：请指定一个有效的图片文件夹路径。"

        valid, output_path_or_error = validate_and_create_output_folder(output_folder)
        if not valid:
            return f"❌ 错误：{output_path_or_error}"
        output_folder = output_path_or_error

        return perform_classification_task(cls_weights, input_type, single_image, image_folder, output_folder,
                                           device, batch_size, confidence_threshold)

    except Exception as e:
        return f"❌ 推理过程中发生意外错误: {str(e)}"


# ==================== UI 辅助函数 ====================
def update_model_status():
    """更新模型状态显示"""
    return create_model_status_html()


def toggle_input_visibility(input_type):
    if input_type == "single":
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)


def browse_folder(title="选择文件夹"):
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        folder_path = filedialog.askdirectory(title=title, initialdir=os.getcwd())
        root.destroy()
        return folder_path if folder_path else ""
    except Exception as e:
        # 打印更详细的错误信息
        print(f"!!! tkinter文件夹选择失败: {e}")
        # 可以返回一个错误提示，让用户知道问题所在
        return f"错误: {e}"


def validate_and_create_output_folder(output_path):
    try:
        if not output_path or output_path.strip() == "": output_path = "./output"
        os.makedirs(output_path, exist_ok=True)
        test_file = os.path.join(output_path, "test_write.tmp")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        return True, output_path
    except Exception as e:
        return False, f"创建或写入输出文件夹失败: {str(e)}"


# ==================== 创建主界面 ====================
def create_interface():
    hardware_info = detect_hardware()
    device_options = get_device_options(hardware_info)
    hardware_html = create_hardware_info_html(hardware_info)
    model_status_html = create_model_status_html()

    with gr.Blocks(title="钢铁缺陷分类检测系统") as interface:
        gr.Markdown("# 🏭 钢铁缺陷分类检测系统")
        gr.Markdown("### 基于深度学习的缺陷智能分类系统")
        gr.HTML(hardware_html)

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 📂 1. 模型权重选择")
                model_status_display = gr.HTML(model_status_html)
                cls_weights = gr.File(
                    label="上传自定义模型权重 (可选)",
                    file_types=[".pth", ".pt"],
                    value=None
                )
                gr.Markdown("💡 **说明**: 如果不上传文件且默认模型存在，将自动使用默认模型")

            with gr.Column(scale=3):
                gr.Markdown("### ⚙️ 2. 配置推理参数")
                with gr.Row():
                    device_choice = gr.Dropdown(choices=device_options, value=hardware_info['recommended_device'],
                                                label="推理设备")
                    confidence_threshold = gr.Slider(minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                                                     label="分类置信度阈值")
                    batch_size = gr.Slider(minimum=1, maximum=32, value=4, step=1, label="批处理大小")

        gr.Markdown("### 🖼️ 3. 选择输入数据")
        with gr.Row():
            with gr.Column():
                input_type = gr.Radio(choices=[("单张图片", "single"), ("批量处理", "folder")], value="single",
                                      label="输入模式")
                single_image = gr.File(label="选择单张图片", file_types=["image"], visible=True)
                image_folder = gr.Textbox(label="图片文件夹路径", placeholder="点击'浏览'选择或手动输入", visible=False)
                folder_browse_btn = gr.Button("📂 浏览输入文件夹", visible=False, size="sm")
            with gr.Column():
                output_folder = gr.Textbox(label="输出文件夹路径", value="./output")
                output_browse_btn = gr.Button("📂 浏览输出文件夹", size="sm")

        gr.Markdown("### 🚀 4. 开始检测")
        start_inference_btn = gr.Button("🚀 开始分类检测", variant="primary", size="lg")
        inference_status = gr.Textbox(label="推理状态与结果", lines=20, placeholder="等待开始...", interactive=False)

        # --- 事件绑定 ---
        input_type.change(toggle_input_visibility, inputs=[input_type],
                          outputs=[single_image, image_folder, folder_browse_btn])
        folder_browse_btn.click(lambda: browse_folder("选择图片文件夹"), outputs=[image_folder])
        output_browse_btn.click(lambda: browse_folder("选择输出文件夹"), outputs=[output_folder])

        # 当用户上传或清除文件时，更新模型状态显示
        cls_weights.change(fn=update_model_status, outputs=[model_status_display])

        start_inference_btn.click(
            fn=run_inference,
            inputs=[
                cls_weights, input_type, single_image, image_folder,
                output_folder, device_choice, batch_size, confidence_threshold
            ],
            outputs=[inference_status]
        )
    return interface















































def main():
    print("🚀 启动钢铁缺陷分类检测系统...")

    # 启动时检查默认模型状态
    default_exists, default_path = check_default_model()
    if default_exists:
        print(f"✅ 检测到默认模型: {default_path}")
    else:
        print(f"⚠️ 默认模型不存在: {default_path}")
        print("   用户需要上传模型文件才能进行推理")

    try:
        interface = create_interface()
        interface.launch(server_name="127.0.0.1", server_port=7861, inbrowser=True)
    except Exception as e:
        print(f"❌ 启动失败: {e}")


if __name__ == "__main__":
    main()