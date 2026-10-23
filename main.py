# -*- coding: utf-8 -*-
"""命令行入口，统一调度各功能。

子命令：
    build     构建人脸底库
    gallery   查看当前底库中的姓名
    image     对单张图片做识别
    attend    打开摄像头做实时识别/考勤
    capture   用摄像头采集某个人的照片，加入底库目录
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from config import GALLERY_DB, GALLERY_DIR, OUTPUT_DIR
from src.pipeline import FaceSystem, draw_results


# ------------------------------------------------------------------ build
def cmd_build(args) -> None:
    from build_gallery import build_gallery

    build_gallery(args.gallery_dir, args.output, save_aligned=args.save_aligned)


# ------------------------------------------------------------------ gallery
def cmd_gallery(args) -> None:
    gallery_path = Path(args.gallery_db)
    if not gallery_path.exists():
        print(f"未找到底库文件：{gallery_path}\n请先运行 `python main.py build`。")
        return
    data = np.load(gallery_path, allow_pickle=True)
    names = list(data["names"])
    embeddings = data["embeddings"]
    if not names:
        print("底库为空。")
        return

    summary: dict[str, int] = {}
    for name in names:
        summary[name] = summary.get(name, 0) + 1
    print(f"底库共 {len(summary)} 人 / {len(names)} 条特征：")
    for name, count in sorted(summary.items()):
        print(f"  - {name}: {count} 张")


# ------------------------------------------------------------------ image
def cmd_image(args) -> None:
    image_path = Path(args.image)
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"无法读取图片：{image_path}")
        return

    system = FaceSystem(load_gallery=True)
    if not system.gallery_loaded:
        print("[警告] 未找到人脸底库，所有人脸都会被识别为“未知”。")

    results = system.recognize(image, mark_attendance=args.mark)
    if not results:
        print("未检测到人脸。")
        return

    print(f"共检测到 {len(results)} 张人脸：")
    for i, res in enumerate(results, 1):
        x, y, w, h = res.face.to_int_box()
        flag = "（已签到）" if res.marked else ""
        print(f"  #{i} 姓名={res.name} 相似度={res.score:.3f} 位置=({x},{y},{w},{h}) {flag}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else OUTPUT_DIR / f"{image_path.stem}_result.jpg"
    cv2.imwrite(str(out_path), draw_results(image, results))
    print(f"标注结果已保存：{out_path}")

    if args.show:
        cv2.imshow("result", draw_results(image, results))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ------------------------------------------------------------------ attend
def cmd_attend(args) -> None:
    from app import run

    run(source=args.source, mark=not args.no_mark)


# ------------------------------------------------------------------ capture
def _expand_box(box, frame_shape, margin: float = 0.3):
    x, y, w, h = box
    mx, my = w * margin, h * margin
    H, W = frame_shape[:2]
    x0 = max(int(x - mx), 0)
    y0 = max(int(y - my), 0)
    x1 = min(int(x + w + mx), W)
    y1 = min(int(y + h + my), H)
    return x0, y0, x1, y1


def cmd_capture(args) -> None:
    """用摄像头采集某人的照片，保存到 data/gallery/<姓名>/。"""
    name = args.name
    out_dir = Path(args.gallery_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    system = FaceSystem(load_gallery=False)
    cap = cv2.VideoCapture(int(args.source) if str(args.source).isdigit() else args.source)
    if not cap.isOpened():
        print(f"无法打开视频源：{args.source}")
        return

    print(f"为「{name}」采集照片：正对摄像头，按 空格 保存一张，按 q 结束。")
    count = 0
    try:
        while count < args.num:
            ok, frame = cap.read()
            if not ok:
                break
            face = system.detector.detect_largest(frame)
            canvas = frame.copy()
            if face is not None:
                x, y, w, h = face.to_int_box()
                cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 200, 0), 2)
                cv2.putText(canvas, "Press SPACE to capture", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            else:
                cv2.putText(canvas, "No face detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(canvas, f"{name}: {count}/{args.num}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("capture", canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" ") and face is not None:
                x0, y0, x1, y1 = _expand_box(face.box, frame.shape)
                crop = frame[y0:y1, x0:x1]
                count += 1
                save_path = out_dir / f"{datetime.now():%Y%m%d_%H%M%S}_{count:02d}.jpg"
                cv2.imwrite(str(save_path), crop)
                print(f"已保存：{save_path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"采集结束，共 {count} 张，保存在：{out_dir}")
    if count > 0 and args.rebuild:
        print("正在重建底库 ...")
        from build_gallery import build_gallery

        build_gallery(args.gallery_dir, args.output)


# ------------------------------------------------------------------ 解析参数
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="人脸识别考勤系统")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="构建人脸底库")
    p_build.add_argument("--gallery-dir", default=str(GALLERY_DIR))
    p_build.add_argument("--output", default=str(GALLERY_DB))
    p_build.add_argument("--save-aligned", action="store_true", help="额外保存对齐后的人脸")
    p_build.set_defaults(func=cmd_build)

    p_gallery = sub.add_parser("gallery", help="查看当前底库")
    p_gallery.add_argument("--gallery-db", default=str(GALLERY_DB))
    p_gallery.set_defaults(func=cmd_gallery)

    p_image = sub.add_parser("image", help="识别单张图片")
    p_image.add_argument("image", help="图片路径")
    p_image.add_argument("--output", default="", help="标注结果保存路径")
    p_image.add_argument("--mark", action="store_true", help="把识别结果写入考勤")
    p_image.add_argument("--show", action="store_true", help="弹窗显示结果")
    p_image.set_defaults(func=cmd_image)

    p_attend = sub.add_parser("attend", help="摄像头实时识别/考勤")
    p_attend.add_argument("--source", default="0", help="摄像头序号或视频文件")
    p_attend.add_argument("--no-mark", action="store_true", help="只识别，不写考勤")
    p_attend.set_defaults(func=cmd_attend)

    p_capture = sub.add_parser("capture", help="采集某人照片加入底库")
    p_capture.add_argument("name", help="姓名")
    p_capture.add_argument("--source", default="0")
    p_capture.add_argument("--num", type=int, default=10, help="采集张数")
    p_capture.add_argument("--gallery-dir", default=str(GALLERY_DIR))
    p_capture.add_argument("--output", default=str(GALLERY_DB))
    p_capture.add_argument("--rebuild", action="store_true", help="采集后自动重建底库")
    p_capture.set_defaults(func=cmd_capture)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
