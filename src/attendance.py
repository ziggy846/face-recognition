# -*- coding: utf-8 -*-
"""考勤记录模块。

把识别成功的人写入 CSV 文件，支持"同一人同一天只记一次"，
避免对着摄像头站一会儿就被重复打卡。

CSV 列：姓名, 时间, 相似度
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from config import ATTENDANCE_FILE, UNKNOWN_NAME

FIELDNAMES = ["姓名", "时间", "相似度"]


class AttendanceRecorder:
    """负责写入与读取考勤记录。"""

    def __init__(self, csv_path=ATTENDANCE_FILE, once_per_day: bool = True):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.once_per_day = once_per_day
        self._marked: set[tuple[str, str]] = set()  # (姓名, 日期)
        self._ensure_file()
        self._load_existing()

    def _ensure_file(self) -> None:
        if not self.csv_path.exists():
            # utf-8-sig 让 Excel 打开时中文不乱码
            with open(self.csv_path, "w", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerow(FIELDNAMES)

    def _load_existing(self) -> None:
        """把已有记录读进内存，保证重新运行程序时也能跨进程去重。"""
        for row in self.records():
            name = row.get("姓名", "").strip()
            date_str = row.get("时间", "")[:10]
            if name and date_str:
                self._marked.add((name, date_str))

    def already_marked(self, name: str, date_str: str) -> bool:
        return (name, date_str) in self._marked

    def mark(self, name: str, score: float, timestamp: datetime | None = None) -> bool:
        """记录一次考勤。

        返回 True 表示本次成功写入；若今天已经记录过则返回 False。
        """
        if name == UNKNOWN_NAME or not name:
            return False

        timestamp = timestamp or datetime.now()
        date_str = timestamp.strftime("%Y-%m-%d")
        if self.once_per_day and self.already_marked(name, date_str):
            return False

        with open(self.csv_path, "a", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(
                [name, timestamp.strftime("%Y-%m-%d %H:%M:%S"), f"{score:.4f}"]
            )
        self._marked.add((name, date_str))
        return True

    def records(self, date_str: str | None = None) -> list[dict]:
        """读取考勤记录，可选按日期过滤。"""
        if not self.csv_path.exists():
            return []
        rows: list[dict] = []
        with open(self.csv_path, "r", newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if date_str and not row.get("时间", "").startswith(date_str):
                    continue
                rows.append(row)
        return rows

    def names_of_day(self, date_str: str | None = None) -> list[str]:
        """当天已打卡的姓名列表。"""
        date_str = date_str or datetime.now().strftime("%Y-%m-%d")
        return [r["姓名"] for r in self.records(date_str)]
