# -*- coding: utf-8 -*-
"""人脸识别系统核心模块包。

导入本包时会自动把项目根目录加入 sys.path，从而可以
直接 `from config import ...`、`from src.xxx import ...`。
"""
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.detector import FaceDetector, Face          # noqa: E402
from src.aligner import FaceAligner                  # noqa: E402
from src.recognizer import FaceRecognizer            # noqa: E402
from src.attendance import AttendanceRecorder        # noqa: E402
from src.pipeline import FaceSystem, RecognitionResult, draw_results  # noqa: E402

__all__ = [
    "FaceDetector",
    "Face",
    "FaceAligner",
    "FaceRecognizer",
    "AttendanceRecorder",
    "FaceSystem",
    "RecognitionResult",
    "draw_results",
]
