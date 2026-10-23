# -*- coding: utf-8 -*-
"""人脸检测模块（YOLO）。

使用 Ultralytics 的 YOLO 人脸检测模型，在图片/视频帧中定位人脸，
并输出每张人脸的人脸框、置信度和 5 个关键点，供对齐模块使用。

本模块默认加载的模型是 **YOLO11n-pose**，在人脸数据集上训练，
每个检测结果包含 4 个框坐标 + 1 个类别分数 + 5×(x, y, conf) 关键点。
实测其关键点顺序为：
    0 右眼、1 左眼、2 鼻尖、3 右嘴角、4 左嘴角
（"右/左"指被拍摄者自己的左右；该顺序与 ArcFace 对齐模板一致。）

模型既支持 .pt（PyTorch 权重），也支持 .onnx（导出模型），
统一交给 Ultralytics 的 YOLO 接口加载即可。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import (
    DET_CONF_THRESHOLD,
    DET_IMGSZ,
    DET_IOU_THRESHOLD,
    DEVICE,
    YOLO_MODEL,
)


@dataclass
class Face:
    """一张人脸：边界框 +（可选的）5 个关键点 + 置信度。"""

    box: np.ndarray                # (x, y, w, h)，左上角坐标与宽高
    landmarks: np.ndarray | None   # (5, 2) 关键点，模型不支持时为 None
    score: float                   # 检测置信度

    @property
    def area(self) -> float:
        """人脸框面积，用于在多人时挑选最大的人脸。"""
        return float(self.box[2] * self.box[3])

    def to_int_box(self) -> tuple[int, int, int, int]:
        x, y, w, h = self.box
        return int(x), int(y), int(w), int(h)


class FaceDetector:
    """基于 Ultralytics YOLO 的人脸检测器。"""

    def __init__(
        self,
        model_path=YOLO_MODEL,
        conf_threshold: float = DET_CONF_THRESHOLD,
        iou_threshold: float = DET_IOU_THRESHOLD,
        imgsz: int = DET_IMGSZ,
        device: str = DEVICE,
    ):
        if not model_path:
            raise FileNotFoundError("未指定人脸检测模型路径")
        self.model_path = str(model_path)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.imgsz = int(imgsz)
        self.device = device or None

        from ultralytics import YOLO  # 延迟导入，避免未安装时影响其他模块

        # task 交由 Ultralytics 依据模型元数据自动推断（pose / detect）
        self.model = YOLO(self.model_path)

    def detect(self, image: np.ndarray) -> list[Face]:
        """检测图片中的所有（经过 NMS 的）人脸。

        image 需为 BGR 三通道图片（`cv2.imread` 的默认格式）。
        """
        if image is None or getattr(image, "size", 0) == 0:
            return []

        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        # pose 模型才有 keypoints；detect 模型没有
        keypoints_xy = None
        if getattr(result, "keypoints", None) is not None and result.keypoints.xy is not None:
            keypoints_xy = result.keypoints.xy.cpu().numpy()

        faces: list[Face] = []
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            box = np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)
            landmarks = None
            if keypoints_xy is not None and i < len(keypoints_xy):
                landmarks = keypoints_xy[i].astype(np.float32)  # (5, 2)
            faces.append(Face(box=box, landmarks=landmarks, score=float(confs[i])))
        return faces

    def detect_largest(self, image: np.ndarray) -> Face | None:
        """只返回面积最大的一张人脸，若没有则返回 None。

        注册底库时每张图片通常只有一个人，取最大人脸最稳妥。
        """
        faces = self.detect(image)
        if not faces:
            return None
        return max(faces, key=lambda f: f.area)
