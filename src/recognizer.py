# -*- coding: utf-8 -*-
"""人脸识别模块（ArcFace）。

使用 InsightFace 的 ArcFace 模型（`w600k_r50.onnx`，ResNet50 骨干，
在 Glint360K 等数据上训练）把对齐后的 112x112 人脸编码成 **512 维**
特征向量，再和底库中的特征做**余弦相似度**比对，相似度最高的那一类
即为识别结果。ArcFace 用角间隔损失（Additive Angular Margin）训练，
同一个人特征夹角小、不同人夹角大，因此用余弦相似度区分效果很好。

底库结构：
    names       : list[str]，每一条特征对应的姓名
    embeddings  : (N, 512) float32 矩阵，第 i 行是第 i 条特征
保存为 .npz 文件，方便复用，不必每次重新提取特征。
"""
from __future__ import annotations

import numpy as np

from config import ARCFACE_MODEL, ARC_CTX_ID, EMBEDDING_DIM, MATCH_THRESHOLD, UNKNOWN_NAME


class FaceRecognizer:
    """基于 InsightFace ArcFace 的人脸特征提取与比对。"""

    def __init__(
        self,
        model_path=ARCFACE_MODEL,
        threshold: float = MATCH_THRESHOLD,
        ctx_id: int = ARC_CTX_ID,
    ):
        if not model_path:
            raise FileNotFoundError("未指定人脸识别模型路径")
        self.model_path = str(model_path)
        self.threshold = float(threshold)

        try:
            from insightface.model_zoo.arcface_onnx import ArcFaceONNX
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "未安装 insightface，请先执行：pip install insightface onnxruntime"
            ) from exc

        # ArcFaceONNX 内部用 onnxruntime 推理，get_feat() 完成预处理(mean/std=127.5, RGB)
        self.model = ArcFaceONNX(self.model_path)
        self.model.prepare(ctx_id=int(ctx_id))

        self.names: list[str] = []
        self.embeddings = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        self._normed = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)  # 预归一化的底库特征

    # ------------------------------------------------------------- 特征提取
    def feature(self, aligned_face: np.ndarray) -> np.ndarray:
        """输入对齐后的 112x112 人脸，输出 512 维特征向量。"""
        feat = self.model.get_feat(aligned_face)
        return np.asarray(feat, dtype=np.float32).reshape(-1)

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        return vec / (np.linalg.norm(vec) + 1e-10)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """两个特征向量的余弦相似度，取值约在 [-1, 1]，越大越相似。"""
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        b = np.asarray(b, dtype=np.float32).reshape(-1)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    # ------------------------------------------------------------- 底库管理
    @property
    def gallery_size(self) -> int:
        return len(self.names)

    def set_gallery(self, names: list[str], embeddings: np.ndarray) -> None:
        self.names = list(names)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.size == 0:
            embeddings = embeddings.reshape(0, EMBEDDING_DIM)
        self.embeddings = embeddings
        # 预归一化，匹配时只需矩阵乘法
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self._normed = embeddings / (norms + 1e-10)

    def add(self, name: str, embedding: np.ndarray) -> None:
        """向底库追加一条特征（内存中）。"""
        embedding = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        self.names.append(name)
        self.embeddings = np.vstack([self.embeddings, embedding])
        self._normed = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-10)

    def save_gallery(self, path) -> None:
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            names=np.array(self.names, dtype=object),
            embeddings=self.embeddings,
        )

    def load_gallery(self, path) -> bool:
        """读取已保存的底库，成功返回 True，文件不存在返回 False。"""
        from pathlib import Path

        path = Path(path)
        if not path.exists():
            return False
        data = np.load(path, allow_pickle=True)
        self.set_gallery(list(data["names"]), data["embeddings"])
        return True

    # ------------------------------------------------------------- 识别
    def match(self, embedding: np.ndarray) -> tuple[str, float]:
        """在底库中查找最相似的人脸。

        返回 (姓名, 相似度)。若最高相似度低于阈值，则返回 (未知, 相似度)。
        """
        if self.gallery_size == 0:
            return UNKNOWN_NAME, 0.0

        query = self._normalize(embedding)
        sims = self._normed @ query          # (N,) 余弦相似度
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        if best_score < self.threshold:
            return UNKNOWN_NAME, best_score
        return self.names[best_idx], best_score
