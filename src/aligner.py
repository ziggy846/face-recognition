# -*- coding: utf-8 -*-
"""人脸对齐模块。

检测到的人脸往往带有旋转、缩放和位置偏移。识别模型要求输入
"对齐后"的正面人脸，因此这里用 5 个关键点与一张标准人脸模板
做**相似变换**（只含旋转、缩放、平移），把任意人脸摆正到
112x112 的标准位置。

标准模板来自 ArcFace，坐标对应 112x112 的人脸：
    右眼、左眼、鼻尖、右嘴角、左嘴角
（YOLO 人脸模型输出的关键点顺序与之一致，可直接一一对应。）

作为兜底，若检测模型不提供关键点，则用 align_by_box() 按人脸框
裁剪并缩放到 112x112（粗略对齐，精度略低）。
"""
from __future__ import annotations

import cv2
import numpy as np

from config import ALIGN_BOX_MARGIN, ALIGNED_SIZE

# 112x112 标准人脸模板（ArcFace）
_REFERENCE_112 = np.array(
    [
        [38.2946, 51.6963],   # 右眼
        [73.5318, 51.5014],   # 左眼
        [56.0252, 71.7366],   # 鼻尖
        [41.5493, 92.3655],   # 右嘴角
        [70.7299, 92.2041],   # 左嘴角
    ],
    dtype=np.float32,
)


class FaceAligner:
    """把带姿态的人脸对齐成标准正面人脸。"""

    def __init__(self, output_size: int = ALIGNED_SIZE, reference: np.ndarray | None = None):
        self.output_size = int(output_size)
        reference = _REFERENCE_112 if reference is None else np.asarray(reference, dtype=np.float32)
        # 若输出尺寸不是 112，则等比缩放模板坐标
        self.reference = reference * (self.output_size / 112.0)

    def _estimate_transform(self, landmarks: np.ndarray) -> np.ndarray:
        src = np.asarray(landmarks, dtype=np.float32)
        # LMEDS 可减弱个别关键点误差的影响
        matrix, _ = cv2.estimateAffinePartial2D(src, self.reference, method=cv2.LMEDS)
        if matrix is None:  # 退化时退回到最小二乘
            matrix, _ = cv2.estimateAffinePartial2D(src, self.reference, method=0)
        if matrix is None:
            raise RuntimeError("无法根据关键点估计对齐变换")
        return matrix

    def align(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """按 5 个关键点做相似变换，输出 output_size x output_size 的正面人脸。"""
        matrix = self._estimate_transform(landmarks)
        return cv2.warpAffine(
            image,
            matrix,
            (self.output_size, self.output_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    def align_by_box(self, image: np.ndarray, box: np.ndarray, margin: float = ALIGN_BOX_MARGIN) -> np.ndarray:
        """没有关键点时的兜底：按人脸框外扩后裁剪并缩放到目标尺寸。"""
        height, width = image.shape[:2]
        x, y, w, h = box
        cx, cy = x + w / 2.0, y + h / 2.0
        side = max(w, h) * (1.0 + margin)
        x0 = int(max(cx - side / 2.0, 0))
        y0 = int(max(cy - side / 2.0, 0))
        x1 = int(min(cx + side / 2.0, width))
        y1 = int(min(cy + side / 2.0, height))
        if x1 <= x0 or y1 <= y0:
            return cv2.resize(image, (self.output_size, self.output_size))
        crop = image[y0:y1, x0:x1]
        return cv2.resize(crop, (self.output_size, self.output_size))

    def align_face(self, image: np.ndarray, face) -> np.ndarray:
        """统一入口：优先用关键点对齐，没有关键点时按框粗略对齐。"""
        if getattr(face, "landmarks", None) is not None:
            return self.align(image, face.landmarks)
        return self.align_by_box(image, face.box)
