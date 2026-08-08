# 🐟 Fish-Detection-YOLOv11
*从零到一，一个计算机视觉实战项目*

---

## 📌 为什么要做这个项目？

在水产养殖或渔业资源调查中，鱼类的种类识别和数量统计通常依赖人工，效率低且容易疲劳。  
本项目尝试用 **YOLOv11** 目标检测算法，训练一个能识别鲫鱼的模型，探索用AI替代人工识别的可行性。

**虽然项目难度不算高，但我坚持做完了完整闭环：**  
`数据采集 → 数据标注 → 模型训练 → 调参优化 → 权重导出 → 推理验证`

---

## 🧠 我做了什么（个人贡献）

| 工作模块 | 具体内容 | 使用工具 |
|---------|---------|---------|
| 数据采集 | 从公开数据集中筛选图片，并自己补充了500张网络图片 | Python脚本 |
| 数据标注 | 对500张图片进行手动标注，生成YOLO格式的标签文件 | LabelImg |
| 模型训练 | 选用YOLOv11n作为基础模型进行训练 | Ultralytics YOLOv11 |
| 参数调优 | 调整了图像尺寸、batch size、学习率等超参数，对比验证集效果 | 训练日志分析 |
| 推理验证 | 导出 `.pt` 权重文件，编写Python脚本完成单张图片和视频的推理检测 | OpenCV + PyTorch |
| 团队协作 | 配合团队其他成员完成前后端联调，模型以JSON格式返回检测结果 | Flask（联调环节） |

> 📌 代码主要借助AI辅助生成，但**每一步的原理和代码逻辑我都清楚，能独立从头跑通整个流程**。

---

## 📁 项目文件结构

Fish-Detection-YOLOv11/

├── data/ # 数据集（含图片及标注文件）

├── models/ # 训练好的 .pt 权重文件

├── scripts/ # Python脚本

│ ├── train.py # 训练脚本

│ ├── detect.py # 推理检测脚本

│ └── eval.py # 评估脚本

├── outputs/ # 推理结果可视化图片

├── requirements.txt # 环境依赖列表

└── README.md # 项目说明

---

## 🚀 如何运行这个项目

```bash
# 1. 克隆仓库到本地
https://github.com/baiyaodao06/Fish-Detection-YOLOv11

# 2. 安装依赖
pip install -r requirements.txt

# 3. 运行推理检测（使用我训练好的权重）
python detect.py --weights models/best.pt --source data/test.jpg 
