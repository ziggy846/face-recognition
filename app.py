# -*- coding: utf-8 -*-
"""实时人脸识别考勤程序。

用法：
    python app.py                 # 打开默认摄像头(0)
    python app.py --source 1      # 指定其它摄像头
    python app.py --source demo.mp4   # 也可以读取视频文件
    python app.py --no-mark       # 只识别不写考勤

窗口快捷键：
    q 退出      s 保存当前画面到 outputs/
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2

from config import OUTPUT_DIR
from src.pipeline import FaceSystem, draw_results


def _open_source(source):
    """把命令行传入的 source 转成 VideoCapture 可用的形式。"""
    if isinstance(source, str) and source.isdigit():
        return int(source)
    return source


def run(source=0, mark: bool = True, window_name: str = "Face Recognition Attendance",
        show_fps: bool = True) -> None:
    system = FaceSystem(load_gallery=True)
    if not system.gallery_loaded:
        print("[警告] 未找到人脸底库，请先运行 `python main.py build` 构建底库。")
        print("       此时所有人脸都会被识别为“未知”。")

    cap = cv2.VideoCapture(_open_source(source))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频源：{source}")

    print("按 q 退出，按 s 保存当前画面。")
    prev = time.time()
    fps = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("视频结束或读取失败。")
                break

            results = system.recognize(frame, mark_attendance=mark)
            canvas = draw_results(frame, results)

            # 在左上角显示 FPS
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
            prev = now
            if show_fps:
                cv2.putText(canvas, f"FPS: {fps:4.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            cv2.imshow(window_name, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                out = OUTPUT_DIR / f"snapshot_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                cv2.imwrite(str(out), canvas)
                print(f"已保存截图：{out}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="实时人脸识别考勤")
    parser.add_argument("--source", default="0", help="摄像头序号(如 0)或视频文件路径")
    parser.add_argument("--no-mark", action="store_true", help="只识别，不写考勤")
    args = parser.parse_args()
    run(source=args.source, mark=not args.no_mark)


if __name__ == "__main__":
    main()
