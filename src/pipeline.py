# -*- coding: utf-8 -*-
"""系统流水线模块。

把"检测 -> 对齐 -> 特征提取 -> 比对 -> 考勤"串成一条流水线，
对外只暴露 FaceSystem 一个入口，app.py 与 main.py 都复用它。
另外提供 draw_results() 用于把识别结果画到图片上。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from config import (
    ATTENDANCE_FILE,
    FONT_CANDIDATES,
    FONT_PATH,
    GALLERY_DB,
    MATCH_THRESHOLD,
    UNKNOWN_NAME,
)
from src.aligner import FaceAligner
from src.attendance import AttendanceRecorder
from src.detector import Face, FaceDetector
from src.recognizer import FaceRecognizer


@dataclass
class RecognitionResult:
    """单张人脸的识别结果。"""

    face: Face
    name: str
    score: float
    aligned: np.ndarray          # 对齐后的人脸，便于调试/展示
    marked: bool = False         # 本次是否写入了考勤

    @property
    def known(self) -> bool:
        return self.name != UNKNOWN_NAME


class FaceSystem:
    """人脸识别系统的统一入口。"""

    def __init__(
        self,
        detector: FaceDetector | None = None,
        aligner: FaceAligner | None = None,
        recognizer: FaceRecognizer | None = None,
        gallery_path=GALLERY_DB,
        attendance_file=ATTENDANCE_FILE,
        threshold: float = MATCH_THRESHOLD,
        load_gallery: bool = True,
    ):
        self.detector = detector or FaceDetector()
        self.aligner = aligner or FaceAligner()
        self.recognizer = recognizer or FaceRecognizer(threshold=threshold)
        self.attendance = AttendanceRecorder(attendance_file)
        self.gallery_path = Path(gallery_path)

        self.gallery_loaded = False
        if load_gallery:
            self.gallery_loaded = self.recognizer.load_gallery(self.gallery_path)

    # ------------------------------------------------------------- 底库
    def load_gallery(self) -> bool:
        self.gallery_loaded = self.recognizer.load_gallery(self.gallery_path)
        return self.gallery_loaded

    @property
    def gallery_size(self) -> int:
        return self.recognizer.gallery_size

    # ------------------------------------------------------------- 识别
    def recognize(self, image: np.ndarray, mark_attendance: bool = False) -> list[RecognitionResult]:
        """识别一张图片中的所有（且已注册的）人脸。"""
        results: list[RecognitionResult] = []
        for face in self.detector.detect(image):
            aligned = self.aligner.align_face(image, face)
            embedding = self.recognizer.feature(aligned)
            name, score = self.recognizer.match(embedding)
            result = RecognitionResult(face=face, name=name, score=score, aligned=aligned)
            if mark_attendance and result.known:
                result.marked = self.attendance.mark(name, score)
            results.append(result)
        return results

    def recognize_aligned(self, aligned_face: np.ndarray) -> tuple[str, float]:
        """直接对一张已对齐的 112x112 人脸做识别。"""
        return self.recognizer.match(self.recognizer.feature(aligned_face))


# ------------------------------------------------------------------ 可视化
def _find_font() -> str | None:
    """查找可用的中文字体，找不到返回 None（此时用 OpenCV 自带字体）。"""
    if FONT_PATH and Path(FONT_PATH).exists():
        return FONT_PATH
    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return None


_FONT_FILE = _find_font()


def _put_text(image: np.ndarray, text: str, org: tuple[int, int],
              color=(0, 255, 0), font_size: int = 20) -> None:
    """在图片上写文字。优先使用 Pillow + 中文字体，否则退回 cv2.putText。"""
    x, y = org
    if _FONT_FILE:
        from PIL import Image, ImageDraw, ImageFont

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil)
        font = ImageFont.truetype(_FONT_FILE, font_size)
        # 先画描边黑边，让文字在复杂背景上也能看清
        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0))
        draw.text((x, y), text, font=font, fill=(color[2], color[1], color[0]))
        image[:] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    else:
        cv2.putText(image, text, (x, y + font_size), cv2.FONT_HERSHEY_SIMPLEX,
                    font_size / 30.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, text, (x, y + font_size), cv2.FONT_HERSHEY_SIMPLEX,
                    font_size / 30.0, color, 1, cv2.LINE_AA)


def draw_results(image: np.ndarray, results: list[RecognitionResult]) -> np.ndarray:
    """把识别结果画到图片副本上并返回。

    已知人脸用绿色框，未知人脸用红色框，框上方标注"姓名 相似度"。
    """
    canvas = image.copy()
    for res in results:
        x, y, w, h = res.face.to_int_box()
        color = (0, 200, 0) if res.known else (0, 0, 255)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)

        # 关键点（若有）
        if res.face.landmarks is not None:
            for (px, py) in res.face.landmarks.astype(int):
                cv2.circle(canvas, (px, py), 2, (0, 255, 255), -1)

        label = f"{res.name} {res.score:.2f}"
        if res.marked:
            label += " [已签到]"
        text_y = max(y - 6, 18)
        _put_text(canvas, label, (x, text_y - 14), color=color)
    return canvas
