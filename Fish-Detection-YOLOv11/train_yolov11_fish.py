# train_yolov11_fish.py
from future.backports.xmlrpc.server import SimpleXMLRPCDispatcher
from ultralytics import YOLO
import torch
import os
from pathlib import Path
import time


def check_environment():
    """检查训练环境"""
    print("=" * 70)
    print("YOLOv11 训练环境检查")
    print("=" * 70)

    # 检查CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"PyTorch版本: {torch.__version__}")
    print(f"训练设备: {device}")

    if device == 'cuda':
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    else:
        print("警告: 使用CPU训练，速度会很慢！")
        print("建议使用GPU训练，或减少批次大小")

    # 检查数据集
    data_dir = Path(r"data")
    config_path = Path(r"custom.yaml")

    print(f"\n数据集检查:")
    print(f"配置文件: {config_path} - {'✓ 存在' if config_path.exists() else '✗ 不存在'}")

    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:  # 修复：添加encoding='utf-8'
                config_content = f.read()
                print(f"配置内容:\n{config_content}")
        except UnicodeDecodeError:
            # 如果utf-8不行，尝试其他编码
            try:
                with open(config_path, 'r', encoding='gbk') as f:
                    config_content = f.read()
                    print(f"配置内容:\n{config_content}")
            except:
                print("配置内容: (读取失败)")

    # 检查数据目录
    paths_to_check = [
        (data_dir / 'images' / 'train', '训练集图片'),
        (data_dir / 'images' / 'val', '验证集图片'),
        (data_dir / 'labels' / 'train', '训练集标签'),
        (data_dir / 'labels' / 'val', '验证集标签'),
    ]

    for path, desc in paths_to_check:
        if path.exists():
            file_count = len(list(path.glob('*.*')))
            print(f"  {desc}: {file_count} 个文件")
        else:
            print(f"  {desc}: ✗ 目录不存在")

    return device


def select_model():
    """选择YOLOv11模型版本"""
    print("\n" + "=" * 70)
    print("选择YOLOv11模型版本")
    print("=" * 70)
    print("可选模型:")
    print("1. yolov11n.pt - 纳米版 (最快，精度较低)")
    print("2. yolov11s.pt - 小版")
    print("3. yolov11m.pt - 中版 (推荐)")
    print("4. yolov11l.pt - 大版")
    print("5. yolov11x.pt - 超大版 (最慢，精度最高)")

    models = {
        '1': 'yolov11n.pt',
        '2': 'yolov11s.pt',
        '3': 'yolov11m.pt',
        '4': 'yolov11l.pt',
        '5': 'yolov11x.pt'
    }

    choice = input("\n请选择模型 (1-5, 默认3): ").strip()
    model_name = models.get(choice, 'yolov11m.pt')

    print(f"\n✓ 选择模型: {model_name}")
    return model_name


def train_fish_detection():
    """训练鱼检测模型"""
    print("=" * 70)
    print("开始训练YOLOv11检测模型")
    print("=" * 70)

    os.environ['YOLO_FONT'] = ''  # 跳过字体下载，使用默认字体
    print("跳过字体下载，使用系统默认字体")

    # 1. 环境检查
    device = check_environment()

    # 2. 选择模型
    model_name = select_model()

    # 3. 加载模型
    print(f"\n加载模型: {model_name}")
    try:
        model = YOLO(model_name)
        print("✓ 模型加载成功")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        print("尝试从官方源下载...")
        model = YOLO('yolov11n.pt')  # 使用最小的模型作为fallback

    # 4. 设置训练参数
    print("\n" + "=" * 70)
    print("训练参数设置")
    print("=" * 70)

    # 根据设备调整参数
    if device == 'cuda':
        batch_size = 16  # GPU可以大一些
        workers = 4
    else:
        batch_size = 4  # CPU要小一些
        workers = 2

    print(f"批次大小: {batch_size}")
    print(f"数据加载线程: {workers}")
    print(f"训练轮数: 100")
    print(f"图像尺寸: 640x640")

    # 5. 开始训练
    print("\n" + "=" * 70)
    print("开始训练...")
    print("=" * 70)

    start_time = time.time()

    try:
        results = model.train(
            # 数据集配置
            data=r"custom.yaml",
            epochs=1600,   # 训练轮次
            patience=20,  # 早停耐心值
            batch=batch_size,
            imgsz=640,
            save=True,
            save_period=10,  # 每n轮保存一次
            cache=False,  # 小数据集可以不缓存

            # 设备配置
            device=device,
            workers=workers,

            # 优化器配置
            optimizer='auto',  # 自动选择优化器
            lr0=0.01,  # 初始学习率
            lrf=0.01,  # 最终学习率系数
            momentum=0.937,
            weight_decay=0.0005,

            # 数据增强
            hsv_h=0.020,  # 色调增强
            hsv_s=0.75,  # 饱和度增强
            hsv_v=0.4,  # 明度增强
            degrees=0.0,  # 旋转（鱼检测不需要太多旋转）
            translate=0.1,  # 平移
            scale=0.5,  # 缩放
            shear=0.0,
            perspective=0.0,
            flipud=0.0,  # 上下翻转（鱼通常不会上下颠倒）
            fliplr=0.5,  # 左右翻转
            mosaic=1.0,  # 马赛克增强
            mixup=0.0,  # MixUp增强

            # 损失函数权重
            box=7.5,
            cls=0.5,
            dfl=1.5,

            # 训练控制
            dropout=0.0,
            verbose=True,
            seed=42,  # 固定随机种子
            deterministic=True,  # 确定性训练
            single_cls=False,  # 单类别模式（只有鱼一个类别）

            # 实验名称
            name='fish_detection_v4',
            exist_ok=True,  # 允许覆盖已有结果
            resume=False,  # 不继续训练

            # 验证配置
            val=True,
            plots=True,  # 生成图表
        )

        # 6. 训练完成
        end_time = time.time()
        training_time = end_time - start_time

        print("\n" + "=" * 70)
        print("训练完成！")
        print("=" * 70)
        print(f"总训练时间: {training_time / 60:.1f} 分钟")
        print(f"平均每轮: {training_time / 100:.1f} 秒")

        # 7. 验证模型
        print("\n在验证集上评估模型...")
        metrics = model.val()

        print(f"\n评估结果:")
        print(f"  mAP50: {metrics.box.map50:.4f}")
        print(f"  mAP50-95: {metrics.box.map:.4f}")
        print(f"  精确率: {metrics.box.mp:.4f}")
        print(f"  召回率: {metrics.box.mr:.4f}")

        # 8. 保存最佳模型路径
        best_model_path = 'runs/train/fish_detection_v4/weights/best.pt'
        if os.path.exists(best_model_path):
            print(f"\n✓ 最佳模型保存位置: {best_model_path}")
            model_size = os.path.getsize(best_model_path) / 1024 / 1024
            print(f"  模型大小: {model_size:.1f} MB")

        # 9. 保存训练摘要
        save_training_summary(training_time, metrics)

        return model

    except Exception as e:
        print(f"\n✗ 训练过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_training_summary(training_time, metrics):
    """保存训练摘要"""
    summary_path = Path(r"training_summary.txt")

    summary = f"""YOLOv11 鱼检测模型训练摘要
    =============================================
    训练时间: {time.strftime('%Y-%m-%d %H:%M:%S')}
    训练时长: {training_time / 60:.1f} 分钟

    数据集信息:
    ----------
    总图片数: 365
    训练集: {len(list(Path("data/images/train").glob('*.*')))} 张
    验证集: {len(list(Path("data/images/val").glob('*.*')))} 张
    测试集: {len(list(Path("data/images/test").glob('*.*')))} 张

    评估结果:
    --------
    mAP50: {metrics.box.map50:.4f}
    mAP50-95: {metrics.box.map:.4f}
    精确率: {metrics.box.mp:.4f}
    召回率: {metrics.box.mr:.4f}

    模型位置:
    --------
    最佳模型: runs/train/fish_detection_v1/weights/best.pt
    最后模型: runs/train/fish_detection_v1/weights/last.pt

    训练参数:
    --------
    批次大小: 16
    图像尺寸: 640
    训练轮数: 100
    优化器: AdamW
    学习率: 0.01 -> 0.0001
    """

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)

    print(f"\n✓ 训练摘要已保存: {summary_path}")


def quick_test(model):
    """快速测试模型"""
    print("\n" + "=" * 70)
    print("快速测试")
    print("=" * 70)

    test_dir = Path(r"data\images\test")

    if test_dir.exists():
        test_images = list(test_dir.glob('*.jpg'))[:3]  # 测试3张图片

        for img_path in test_images:
            print(f"\n测试图片: {img_path.name}")

            results = model.predict(
                source=str(img_path),
                conf=0.25,
                save=True,
                save_txt=True,
                project='runs/detect',
                name='quick_test',
                exist_ok=True
            )

            result = results[0]
            boxes = result.boxes

            if boxes is not None:
                print(f"  检测到 {len(boxes)} 条鱼:")
                for i, box in enumerate(boxes):
                    confidence = float(box.conf[0])
                    print(f"    {i + 1}. 置信度: {confidence:.3f}")
            else:
                print("  未检测到鱼")

    print(f"\n测试结果保存在: runs/detect/quick_test")


def main():
    """主函数"""
    print("YOLOv11 鱼检测模型训练")
    print("=" * 70)

    # 开始训练
    model = train_fish_detection()

    if model is not None:
        # 快速测试
        quick_test(model)
        print("\n" + "=" * 70)
        print("下一步:")
        print("=" * 70)
        print("1. 查看训练图表: runs/train/fish_detection_v1/*.png")
        print("2. 使用TensorBoard监控: tensorboard --logdir runs/train")
        print("3. 运行完整测试: python test_model.py")
        print("4. 部署模型: 导出为ONNX或TensorRT格式")

    input("\n按Enter键退出...")


if __name__ == "__main__":
    main()