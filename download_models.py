
# -*- coding: utf-8 -*-
"""下载项目所需的预训练模型到 models/ 目录。

默认从 HuggingFace 镜像下载，若失败会自动尝试备用地址：
    python download_models.py            # 下载缺失的模型
    python download_models.py --force    # 强制重新下载

说明：
- YOLO 人脸检测模型也可换成你自己的 Ultralytics 权重（.pt/.onnx），
  只需在 config.py 里修改 YOLO_MODEL 指向对应文件即可。
- ArcFace 识别模型同样可通过 config.py 的 ARCFACE_MODEL 替换。
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

from config import MODEL_DIR

# 目标文件名 -> 候选下载地址（按顺序尝试）
MODELS: dict[str, list[str]] = {
    "yolo11n-face-pose.onnx": [
        "https://hf-mirror.com/huygiatrng/yolov11n-face-pose-5kp/resolve/main/face_5kp_yolo11n.onnx",
        "https://huggingface.co/huygiatrng/yolov11n-face-pose-5kp/resolve/main/face_5kp_yolo11n.onnx",
    ],
    "arcface_w600k_r50.onnx": [
        "https://hf-mirror.com/public-data/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx",
        "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx",
    ],
}


def _download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        with open(tmp, "wb") as f:
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = done * 100 // total
                    sys.stdout.write(f"\r    下载中 {pct:3d}%  ({done // 1024 // 1024}MB/{total // 1024 // 1024}MB)")
                    sys.stdout.flush()
    print()
    tmp.replace(dest)


def download_all(force: bool = False) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for filename, urls in MODELS.items():
        dest = MODEL_DIR / filename
        if dest.exists() and not force:
            print(f"[跳过] 已存在 {dest}")
            continue
        print(f"[下载] {filename}")
        last_error = None
        for url in urls:
            try:
                _download(url, dest)
                print(f"[完成] {dest}")
                last_error = None
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                print(f"    失败：{exc}，尝试备用地址 ...")
        if last_error is not None:
            print(f"[错误] 无法下载 {filename}，请手动下载后放到 {MODEL_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下载预训练模型")
    parser.add_argument("--force", action="store_true", help="强制重新下载")
    args = parser.parse_args()
    download_all(force=args.force)
