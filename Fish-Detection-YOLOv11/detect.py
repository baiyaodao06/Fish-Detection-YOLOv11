# detect.py
from ultralytics import YOLO
import cv2
import os
from pathlib import Path

def detect_fish(image_path, model_path='models/best.pt', conf_threshold=0.25):
    """
    检测图片中的鱼
    
    参数:
        image_path: 要检测的图片路径
        model_path: 训练好的权重文件路径
        conf_threshold: 置信度阈值，低于这个值的不显示
    """
    # 1. 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"❌ 错误：模型文件 '{model_path}' 不存在！")
        print("请先训练模型，或确保 best.pt 文件在 models/ 目录下。")
        return None
    
    # 2. 检查图片文件是否存在
    if not os.path.exists(image_path):
        print(f"❌ 错误：图片文件 '{image_path}' 不存在！")
        return None
    
    # 3. 加载模型
    print(f"🔍 正在加载模型: {model_path}")
    model = YOLO(model_path)
    print("✅ 模型加载成功")
    
    # 4. 执行推理
    print(f"📷 正在检测图片: {image_path}")
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        save=True,           # 保存检测结果图片
        save_txt=True,       # 保存检测框坐标（YOLO格式）
        project='runs/detect',
        name='result',
        exist_ok=True
    )
    
    # 5. 解析并打印检测结果
    result = results[0]
    boxes = result.boxes
    
    if boxes is not None and len(boxes) > 0:
        print(f"\n✅ 检测到 {len(boxes)} 条鱼：")
        for i, box in enumerate(boxes):
            conf = float(box.conf[0])
            # 获取边界框坐标
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            print(f"  第{i+1}条: 置信度 {conf:.3f}, 位置 ({int(x1)},{int(y1)}) -> ({int(x2)},{int(y2)})")
    else:
        print("\n❌ 未检测到鱼")
    
    # 6. 输出结果保存位置
    print(f"\n📁 结果图片保存在: runs/detect/result/")
    print(f"📁 坐标文件保存在: runs/detect/result/labels/")
    
    return results


def detect_video(video_path, model_path='models/best.pt', conf_threshold=0.25):
    """
    检测视频中的鱼
    """
    if not os.path.exists(model_path):
        print(f"❌ 错误：模型文件 '{model_path}' 不存在！")
        return None
    
    if not os.path.exists(video_path):
        print(f"❌ 错误：视频文件 '{video_path}' 不存在！")
        return None
    
    print(f"🔍 正在加载模型: {model_path}")
    model = YOLO(model_path)
    print("✅ 模型加载成功")
    
    print(f"🎬 正在处理视频: {video_path}")
    results = model.predict(
        source=video_path,
        conf=conf_threshold,
        save=True,
        project='runs/detect',
        name='video_result',
        exist_ok=True
    )
    
    print(f"\n📁 结果视频保存在: runs/detect/video_result/")
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("🐟 YOLOv11 鱼类检测 - 推理脚本")
    print("=" * 60)
    print("\n请选择检测模式：")
    print("1. 检测单张图片")
    print("2. 检测视频文件")
    
    choice = input("\n请输入数字 (1 或 2，默认 1): ").strip()
    
    if choice == "2":
        video_path = input("请输入视频文件路径 (例如: data/test.mp4): ").strip()
        detect_video(video_path)
    else:
        image_path = input("请输入图片文件路径 (例如: data/test.jpg): ").strip()
        detect_fish(image_path)