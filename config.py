# -*- coding: utf-8 -*-
"""全局配置：路径、模型参数与识别阈值。

所有模块统一从这里读取常量，避免魔法数字散落在代码中。
"""
from pathlib import Path

# ---------------------------------------------------------------- 基础路径
BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# 模型文件（可用 download_models.py 下载，见 README）
# 人脸检测：Ultralytics YOLO11n-pose，针对人脸训练，输出人脸框 + 5 个关键点
YOLO_MODEL = MODEL_DIR / "yolo11n-face-pose.onnx"
# 人脸识别：InsightFace ArcFace（ResNet50 骨干，w600k_r50），输出 512 维特征
ARCFACE_MODEL = MODEL_DIR / "arcface_w600k_r50.onnx"

GALLERY_DIR = DATA_DIR / "gallery"                 # 底库原始图片目录（按姓名分子目录）
GALLERY_DB = DATA_DIR / "gallery_db" / "gallery.npz"  # 构建后保存的人脸特征库
SAMPLE_DIR = DATA_DIR / "sample"                   # 测试/演示图片目录
ATTENDANCE_FILE = DATA_DIR / "attendance" / "attendance.csv"  # 考勤记录
OUTPUT_DIR = BASE_DIR / "outputs"                  # 标注结果输出目录

# ---------------------------------------------------------------- 运行设备
# YOLO 的 device 参数："" 表示自动，也可填 "cpu" 或 GPU 序号如 "0"
DEVICE = ""
# ArcFace(insightface) 的 ctx_id：-1 使用 CPU，>=0 使用对应 GPU 序号
ARC_CTX_ID = 0

# ---------------------------------------------------------------- 人脸检测(YOLO)
DET_CONF_THRESHOLD = 0.4    # 置信度阈值
DET_IOU_THRESHOLD = 0.5     # NMS 的 IoU 阈值
DET_IMGSZ = 640             # 推理尺寸（该 ONNX 固定为 640）

# ---------------------------------------------------------------- 人脸对齐
ALIGNED_SIZE = 112          # 对齐后的人脸边长（ArcFace 要求 112x112）
# 若无关键点时，按人脸框裁剪并外扩的比例（框式粗略对齐的兜底方案）
ALIGN_BOX_MARGIN = 0.25

# ---------------------------------------------------------------- 人脸识别(ArcFace)
EMBEDDING_DIM = 512         # ArcFace 特征维度
# ArcFace 余弦相似度阈值：同人一般 >0.6，异人一般 <0.2，取 0.4 比较稳妥
MATCH_THRESHOLD = 0.4
UNKNOWN_NAME = "未知"       # 匹配失败时使用的标签

# ---------------------------------------------------------------- 其他
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 可选：支持中文的字体文件路径。留空则自动查找系统字体，
# 找不到时标注框内的中文会显示为方块（终端输出不受影响）。
FONT_PATH = ""
FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/System/Library/Fonts/PingFang.ttc",  # macOS
    "C:/Windows/Fonts/msyh.ttc",           # Windows
]
