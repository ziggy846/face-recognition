# -*- coding: utf-8 -*-
"""构建人脸底库。

扫描 data/gallery/ 下按姓名分好的子目录（每个子目录里放该人的照片），
对每张照片检测最大的人脸，做对齐后提取特征，最后把所有特征保存到
data/gallery_db/gallery.npz，供识别时直接加载。

目录示例：
    data/gallery/张三/01.jpg
    data/gallery/张三/02.jpg
    data/gallery/李四/01.jpg
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from config import GALLERY_DB, GALLERY_DIR, IMAGE_EXTS
from src.aligner import FaceAligner
from src.detector import FaceDetector
from src.recognizer import FaceRecognizer


def _iter_images(folder: Path):
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def build_gallery(
    gallery_dir=GALLERY_DIR,
    output_path=GALLERY_DB,
    save_aligned: bool = False,
):
    """构建底库并保存，返回 (姓名列表, 特征矩阵)。"""
    gallery_dir = Path(gallery_dir)
    output_path = Path(output_path)

    if not gallery_dir.exists():
        raise FileNotFoundError(f"底库目录不存在：{gallery_dir}")

    detector = FaceDetector()
    aligner = FaceAligner()
    recognizer = FaceRecognizer()

    names: list[str] = []
    embeddings: list[np.ndarray] = []

    person_dirs = sorted(p for p in gallery_dir.iterdir() if p.is_dir())
    if not person_dirs:
        print(f"[警告] {gallery_dir} 下没有任何姓名子目录，底库为空。")
        print("       请按 data/gallery/<姓名>/*.jpg 的方式放入照片后重试。")

    for person_dir in person_dirs:
        name = person_dir.name
        image_paths = list(_iter_images(person_dir))
        ok_count = 0
        for img_path in image_paths:
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"[跳过] 无法读取图片：{img_path}")
                continue

            face = detector.detect_largest(image)
            if face is None:
                print(f"[跳过] 未检测到人脸：{img_path}")
                continue

            aligned = aligner.align_face(image, face)
            embeddings.append(recognizer.feature(aligned))
            names.append(name)
            ok_count += 1

            if save_aligned:
                out_dir = output_path.parent / "aligned" / name
                out_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_dir / f"{img_path.stem}.jpg"), aligned)

        print(f"[{name}] 读取 {len(image_paths)} 张，成功注册 {ok_count} 张")

    embeddings_array = (
        np.asarray(embeddings, dtype=np.float32) if embeddings else np.zeros((0, 512), dtype=np.float32)
    )
    recognizer.set_gallery(names, embeddings_array)
    recognizer.save_gallery(output_path)

    print("-" * 50)
    print(f"底库构建完成：{len(set(names))} 人 / {len(names)} 条特征")
    print(f"已保存到：{output_path}")
    return names, embeddings_array


def main() -> None:
    parser = argparse.ArgumentParser(description="构建人脸底库")
    parser.add_argument("--gallery-dir", default=str(GALLERY_DIR), help="底库图片目录")
    parser.add_argument("--output", default=str(GALLERY_DB), help="特征库输出路径(.npz)")
    parser.add_argument("--save-aligned", action="store_true", help="额外保存对齐后的人脸图片")
    args = parser.parse_args()
    build_gallery(args.gallery_dir, args.output, save_aligned=args.save_aligned)


if __name__ == "__main__":
    main()
