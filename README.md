# 人脸识别考勤系统

基于深度学习的课程设计项目：使用 **YOLO** 做人脸检测、**相似变换**做人脸对齐、
**ArcFace** 做人脸特征提取与比对，并实现**摄像头实时识别**与**考勤记录**功能。

> 这是一个"基础功能版"实现，只保留一条主线算法
> （检测 → 对齐 → 特征 → 比对），不包含多方法对比实验，便于理解和演示。

---

## 1. 系统功能

| 功能 | 说明 | 入口 |
| --- | --- | --- |
| 构建人脸底库 | 扫描 `data/gallery/<姓名>/*.jpg`，提取特征保存为特征库 | `python main.py build` |
| 查看底库 | 打印底库中有哪些人、各有多少条特征 | `python main.py gallery` |
| 图片识别 | 对单张图片检测并识别所有（已注册）人脸，输出标注图 | `python main.py image <图片>` |
| 实时识别考勤 | 打开摄像头/视频，实时框选人脸并自动打卡 | `python main.py attend` |
| 采集照片 | 用摄像头给某个人拍照，加入底库目录 | `python main.py capture <姓名>` |

识别流程：**YOLO 人脸检测 → 5 点对齐(112×112) → ArcFace 特征(512 维) → 余弦相似度比对 → 考勤记录**。

---

## 2. 目录结构

```
人脸识别/
├── config.py              # 全局配置：路径、阈值、模型参数
├── main.py                # 命令行入口，调度各子命令
├── app.py                 # 摄像头实时识别/考勤窗口程序
├── build_gallery.py       # 构建人脸底库
├── download_models.py     # 下载预训练模型
├── requirements.txt       # pip 依赖清单
├── pyproject.toml         # uv 项目配置（依赖 + PyTorch cu132 源）
├── uv.lock                # uv 锁定的依赖版本
├── .python-version        # 指定 Python 版本（3.14）
├── .gitignore             # 忽略 __pycache__、.venv 等
├── .venv/                 # uv 创建的虚拟环境（不入库）
├── .idea/                 # PyCharm 工程配置（可忽略）
├── README.md
├── models/                # 预训练模型（可用 download_models.py 下载）
│   ├── yolo11n-face-pose.onnx     #   人脸检测 YOLO11n-pose（输出人脸框+5关键点）
│   └── arcface_w600k_r50.onnx     #   人脸识别 ArcFace(w600k_r50，512维特征)
├── src/
│   ├── __init__.py
│   ├── detector.py        # 人脸检测模块（YOLO）
│   ├── aligner.py         # 人脸对齐模块（5 点相似变换）
│   ├── recognizer.py      # 人脸识别模块（ArcFace + 余弦相似度 + 底库）
│   ├── attendance.py      # 考勤记录模块（CSV，去重）
│   └── pipeline.py        # 流水线：串联以上模块 + 结果可视化
├── data/
│   ├── gallery/           # 底库原始照片，按姓名分子目录，如 gallery/张三/01.jpg
│   ├── gallery_db/        # 构建后生成的人脸特征库 gallery.npz（运行 build 产生）
│   ├── sample/            # 测试图片（含演示用的 LFW 图片）
│   │   ├── lfw_test/      #   每人 2 张测试图，按姓名分子目录
│   │   └── largest_selfie.jpg   #   多人合影，用于演示多人场景
│   └── attendance/        # 考勤 CSV（attendance.csv，运行后生成）
└── outputs/               # 标注结果图片输出目录（运行后生成）
```

> `data/gallery_db/`、`data/attendance/`、`outputs/`、`models/*.onnx`、`.venv/`、
> `__pycache__/` 等均为生成物或环境目录，不应提交到版本库。
> 依赖可用 `pip install -r requirements.txt` 或 `uv sync` 安装。

---

## 3. 环境与安装

- Python ≥ 3.9（`requirements.txt`）；若用 uv，则按 `.python-version` 使用 **3.14**
- 检测用 **Ultralytics YOLO**（依赖 torch），识别用 **InsightFace ArcFace**（onnxruntime 推理）

方式一：pip

```bash
pip install -r requirements.txt
```

方式二：uv（按 `pyproject.toml` / `uv.lock` 创建 `.venv`，PyTorch 走 cu132 源）

```bash
uv sync                 # 安装依赖
uv run python main.py build   # 在 .venv 中运行
```

若使用 GPU，建议把 `onnxruntime` 换成 `onnxruntime-gpu`：

```bash
pip uninstall onnxruntime && pip install onnxruntime-gpu
```

> `insightface` 首次安装需要编译其 Cython 扩展，若编译失败可改用预编译 wheel，
> 或仅使用其 `model_zoo` 中的 ONNX 推理（本项目只用 `ArcFaceONNX`）。

### 模型准备

`models/` 下有 3 种获得方式，任选其一：

1. 直接下载（推荐）：
   ```bash
   python download_models.py
   ```
2. 使用你自己的 Ultralytics YOLO 人脸权重（`.pt` 或 `.onnx`）：把它放到
   `models/` 并修改 `config.py` 的 `YOLO_MODEL` 指向该文件。
3. 使用你自己的 ArcFace 模型：修改 `config.py` 的 `ARCFACE_MODEL`。

---

## 4. 快速开始

本项目已自带一个演示用底库（10 位公开人物，来自 LFW 数据集）。

```bash
# 0) 下载模型（若 models/ 下已有可跳过）
python download_models.py

# 1) 构建底库（读取 data/gallery 里的照片）
python main.py build

# 2) 查看底库
python main.py gallery

# 3) 识别单张图片（结果保存在 outputs/）
python main.py image data/sample/lfw_test/Tony_Blair/Tony_Blair_02.jpg
python main.py image data/sample/lfw_test/Tony_Blair/Tony_Blair_02.jpg --mark   # 同时写入考勤
python main.py image data/sample/largest_selfie.jpg                             # 多人合影

# 4) 摄像头实时识别 + 考勤（按 q 退出，按 s 截图）
python main.py attend
python main.py attend --source 1        # 指定其它摄像头
python main.py attend --source demo.mp4 # 也可以读视频文件
python main.py attend --no-mark         # 只识别不打卡
```

### 使用自己的照片

1. 在 `data/gallery/` 下按姓名新建子目录，放入该人的若干张照片：
   ```
   data/gallery/张三/01.jpg
   data/gallery/张三/02.jpg
   ```
   也可以用摄像头采集：`python main.py capture 张三 --num 10 --rebuild`
2. 重新运行 `python main.py build` 即可。

---

## 5. 各模块说明

### 5.1 人脸检测 `src/detector.py`（YOLO）
调用 Ultralytics YOLO 的人脸模型（默认 **YOLO11n-pose**），输出每张人脸的：
人脸框 `(x, y, w, h)`、置信度和 **5 个关键点**。
实测该模型关键点顺序为 **右眼、左眼、鼻尖、右嘴角、左嘴角**，
与 ArcFace 对齐模板一致，可直接用于对齐。
若换成只检测框的 YOLO 模型（无关键点），流水线会自动退化为按框裁剪的粗略对齐。

### 5.2 人脸对齐 `src/aligner.py`
用 5 个关键点与标准人脸模板做**相似变换**（`cv2.estimateAffinePartial2D`，
只含旋转/缩放/平移，用 LMEDS 抗噪），把任意姿态的人脸摆正到 **112×112**，
提高后续特征提取的稳定性。

### 5.3 人脸识别 `src/recognizer.py`（ArcFace）
用 InsightFace 的 **ArcFace**（ResNet50 骨干，`w600k_r50`）把对齐后的人脸
编码成 **512 维特征向量**，底库中每人可有多条特征。
识别时计算查询特征与底库所有特征的**余弦相似度**，取最高者；
若最高分低于阈值（默认 `0.4`）则判为"未知"。
底库以 `.npz` 保存，可复用。

### 5.4 考勤记录 `src/attendance.py`
把识别成功的人写入 `data/attendance/attendance.csv`（列为：姓名、时间、相似度），
默认"同一人同一天只记一次"，避免重复打卡。文件用 `utf-8-sig` 编码，Excel 打开不乱码。

### 5.5 流水线 `src/pipeline.py`
`FaceSystem` 把上述模块串成一条流水线，对外只需调用
`recognize(image, mark_attendance=True)` 即可得到识别结果；
`draw_results()` 负责把人脸框、关键点和"姓名 相似度"标签画到图片上。

---

## 6. 参数调节

集中在 `config.py`：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `YOLO_MODEL` | `models/yolo11n-face-pose.onnx` | 检测模型路径，可换成你的 `.pt` |
| `ARCFACE_MODEL` | `models/arcface_w600k_r50.onnx` | 识别模型路径 |
| `DEVICE` | `""` | YOLO 运行设备：`""`自动 / `"cpu"` / `"0"` |
| `ARC_CTX_ID` | `0` | ArcFace 设备：`-1` CPU / `>=0` GPU 序号 |
| `DET_CONF_THRESHOLD` | 0.4 | 人脸检测置信度阈值，调高减少误检 |
| `MATCH_THRESHOLD` | 0.4 | 识别相似度阈值，调高更严格（更少误认、更多"未知"） |
| `ALIGNED_SIZE` | 112 | 对齐后的人脸尺寸（ArcFace 要求 112） |

---

## 7. 演示数据说明

- `data/gallery/` 与 `data/sample/lfw_test/` 中的演示图片取自公开数据集
  **LFW（Labeled Faces in the Wild）**，每位人物 4 张用于建库、2 张用于测试。
- `data/sample/largest_selfie.jpg` 为一张公开合影，用于演示多人场景下的检测效果。
- 以上仅用于功能演示；正式使用请替换为自己采集的照片。

---

## 8. 常见问题

- **图片上中文显示成方块**：系统缺少中文字体。可在 `config.py` 的 `FONT_PATH`
  指定一个中文字体（如 `NotoSansCJK-Regular.ttc`）即可正常显示中文；
  终端输出与 CSV 记录不受此影响。
- **提示"未找到人脸底库"**：先运行 `python main.py build`。
- **找不到模型文件**：运行 `python download_models.py`，或在 `config.py` 中
  指定你自己的权重路径。
- **insightface 安装失败**：见第 3 节的安装提示，或改用预编译 wheel。
- **摄像头打不开**：确认摄像头未被其它程序占用，或改用 `--source 1`。

---

## 9. 可扩展方向（非本基础版内容）

- 增加多方法对比：Haar+HOG/LBPH、FaceNet、SFace 等与本实现的准确率对比；
- 活体检测、口罩场景、大规模底库的向量检索加速（如 FAISS）；
- 打包成 Web 服务或桌面 GUI。
