from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import os
from wsgiref.simple_server import make_server
import logging
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import densenet121, DenseNet121_Weights
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import warnings
import io
from datetime import datetime

# 确保支持中文显示
plt.rcParams['font.family'] = 'SimHei'  # 设置全局字体为 SimHei

# 检查字体是否可用
font_prop = fm.FontProperties(fname='simhei.ttf')
if font_prop:
    print("字体 SimHei 加载成功！")
else:
    print("字体 SimHei 加载失败，请检查字体文件是否存在。")

app = Flask(__name__)
CORS(app)  # 初始化 CORS

# 设置文件保存目录
UPLOAD_FOLDER = 'C:/Users/yangjing/Desktop/RemoteConsultation/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 忽略 FutureWarning 类型的警告
warnings.filterwarnings("ignore", category=FutureWarning)

# 检查 CUDA 是否可用
cuda_available = torch.cuda.is_available()

# 定义14个疾病标签
class_names = [
    '肺不张', '心脏扩大', '胸腔积液', '浸润', '肿块',
    '结节', '肺炎', '气胸', '实变', '水肿',
    '肺气肿', '纤维化', '疝气', '胸膜增厚'
]

# 实例化模型
weights = DenseNet121_Weights.DEFAULT
model = densenet121(weights=weights)

# 修改最后一层分类器
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, len(class_names))

# 指定模型的路径
model_path = 'D:/1.security/RemoteConsultation/model.pth'

# 加载训练好的模型权重
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dict = torch.load(model_path, map_location=device)

# 过滤掉不匹配的键
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
model.eval()  # 设置为评估模式
model.to(device)

# 定义图像预处理 (将单通道图像扩展为3通道)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.Grayscale(num_output_channels=3),  # 将单通道图像扩展为3通道
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # RGB正则化
])

# 加载和预处理输入图像 (灰度模式)
def load_image(image_path):
    image = Image.open(image_path).convert('L')  # 加载为灰度图像 (单通道)
    image = preprocess(image)
    image = image.unsqueeze(0)  # 扩展batch维度
    return image

# 在上传文件后进行图像处理
def process_uploaded_image(filename):
    # 从本地文件系统读取图片
    image_path = os.path.join(UPLOAD_FOLDER, filename)

    if os.path.exists(image_path):
        image = load_image(image_path)

        # 对图像进行预测
        with torch.no_grad():
            output = model(image)  # 输出为logits
            output = torch.sigmoid(output)  # 将logits转为概率

        # 打印预测值
        output_np = output.cpu().numpy().flatten()
        predictions = {}
        for class_name, prediction in zip(class_names, output_np):
            predictions[class_name] = float(prediction)

        # 找出预测值最大的两个标签
        top_indices = np.argsort(output_np)[-2:][::-1]
        top_predictions = [(class_names[idx], float(output_np[idx])) for idx in top_indices]

        return predictions, top_predictions
    else:
        print(f"File not found at path: {image_path}")
        return None, None

# 生成预测结果图像
def generate_diagnosis_image(predictions, top_predictions):
    # 图像的基本设置
    width, height = 800, 600
    background_color = (255, 255, 255)  # 白色背景
    text_color = (0, 0, 0)  # 黑色文字
    font_size = 18
    font_title_size = 24
    font_end_size = 20
    padding = 10

    # 创建一个新的空白图像
    img = Image.new('RGB', (width, height), background_color)
    draw = ImageDraw.Draw(img)

    # 加载字体
    font = ImageFont.truetype('simhei.ttf', size=font_size)
    font_title = ImageFont.truetype('simhei.ttf', size=font_title_size)
    font_end = ImageFont.truetype('simhei.ttf', size=font_end_size)



    # 绘制标题
    title_text = "联合诊断疾病预测结果"
    title_bbox = font.getbbox(title_text, anchor="ms")
    title_width = title_bbox[2] - title_bbox[0]
    title_height = title_bbox[3] - title_bbox[1]
    draw.text(((width - title_width) / 2, padding), title_text, fill=text_color, font=font_title)

    # 添加第一条水平分割线
    line_y = title_height + 2 * padding + font_size
    draw.line([(padding, line_y), (width - padding, line_y)], fill=(0, 0, 0), width=2)

    disease_title = f"疾病名称"
    prediction_title = f"预测值"

    # 绘制疾病及其预测值
    y_pos = line_y + font_size
    draw.text(((width / 4 - 10), y_pos), disease_title, fill=text_color, font=font_title)
    draw.text(((3 * width / 4 - 10), y_pos), prediction_title, fill=text_color, font=font_title)
    # 添加第二条水平分割线
    line_y = title_height + 7 * padding + font_size
    draw.line([(padding, line_y), (width - padding, line_y)], fill=(0, 0, 0), width=2)

    y_pos += font_size + 5 * padding
    for i, (disease, value) in enumerate(predictions.items()):
        disease_text = f"{disease}"
        disease_bbox = font.getbbox(disease_text, anchor="ms")
        disease_width = disease_bbox[2] - disease_bbox[0]

        value_text = f"{value:.4f}"
        value_bbox = font.getbbox(value_text, anchor="ms")
        value_width = value_bbox[2] - value_bbox[0]

        # 计算并绘制疾病名称
        # draw.text(((width - disease_width) / 2, y_pos), disease_text, fill=text_color, font=font)
        draw.text(((width / 4), y_pos), disease_text, fill=text_color, font=font)


        # 计算并绘制预测值
        # draw.text((width - value_width - padding, y_pos), value_text, fill=text_color, font=font)
        draw.text(((3 * width / 4), y_pos), value_text, fill=text_color, font=font)


        y_pos += font_size + padding

    # 添加第二条水平分割线
    line_y = y_pos + font_size + padding
    draw.line([(padding, line_y), (width - padding, line_y)], fill=(0, 0, 0), width=2)

    # 总结部分
    summary_text = f"总结：您最有可能患上的两种肺部疾病：{top_predictions[0][0]} 和 {top_predictions[1][0]}, 具体请与医生详谈。"
    summary_bbox = font.getbbox(summary_text, anchor="ms")
    summary_width = summary_bbox[2] - summary_bbox[0]
    summary_height = summary_bbox[3] - summary_bbox[1]
    draw.text(((width - summary_width) / 2, line_y + padding), summary_text, fill=text_color, font=font_end)

    # 保存图像到内存
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    return img_byte_arr

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        logging.error("No selected file")
        return jsonify({"error": "No selected file"}), 400
    if file:
        logging.info("Handling file upload")
        filename = os.path.join(UPLOAD_FOLDER, file.filename)

        # 保存文件
        file.save(filename)
        logging.info(f"File saved to: {filename}")

        # 处理上传的图片
        predictions, top_predictions = process_uploaded_image(file.filename)

        # 生成预测结果图像
        img_byte_arr = generate_diagnosis_image(predictions, top_predictions)

        return Response(img_byte_arr.getvalue(), mimetype='image/png')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    httpd = make_server('localhost', 8000, app)
    logging.info("Starting server...")
    httpd.serve_forever()