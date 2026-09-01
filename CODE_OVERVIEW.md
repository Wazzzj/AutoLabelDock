# AutoLabel Dock 代码结构梳理

> 本文档是对仓库全部源码（约 3.4 万行 Python）的完整阅读梳理，覆盖 `src/` 全部 5 个分层、`main.py` 入口、配置与测试。生成于 2026-08 会话。

---

## 1. 项目概览

**AutoLabel Dock** 是一个基于 **PyQt5 + Ultralytics YOLO** 的跨平台桌面图像标注工具，面向「标注 → 训练 → 自动标注迭代」的 YOLO 训练闭环。

| 维度 | 内容 |
|---|---|
| 语言/框架 | Python ≥ 3.10，PyQt5（GUI）、ultralytics（训练/推理）、pyqtgraph（曲线）、PyYAML、onnxruntime |
| 任务类型 | detect / segment / obb（旋转框）/ pose（关键点）/ classify |
| 数据存储 | 每项目一个 `project.json` + 每图一个 JSON sidecar（`label_dir="."` 现代布局；旧式镜像 `labels/` 目录仍兼容） |
| 代码规模 | `src/` + `tests/` 共约 3.4 万行；`src/app.py` 2259 行、`ui/canvas.py` 2079 行、`ui/preview_panel.py` 2012 行为最大文件 |
| 许可证 | AGPL-3.0；依赖中 PyQt5 为 GPL-3.0、Ultralytics 为 AGPL-3.0 |

```
autolabel-dock/
├── main.py                    应用入口 + --train-process 子进程分发
├── config/                    全局配置（config.json、train_templates.json、static_resources.json）
├── icon/ logoicon/            图标资源（SVG / Windows ico）
├── pretrained_models/         yolo11n*/yolo26n 等预训练权重
├── resources/screenshots/     README 截图
├── src/
│   ├── app.py                 MainWindow：Tab 主窗口 + 全部编排
│   ├── controllers/           项目 / 模型 / 训练 / 标签 / LocateAnything 控制器
│   ├── core/                  数据模型、项目管理、标注 IO、导入导出格式、备份、标签、训练队列
│   ├── engine/                训练 / 推理 / 数据集准备 / 模型注册 / 后端插件
│   ├── ui/                    PyQt5 组件：画布、面板、对话框、任务视图、主题、图标
│   └── utils/                 线程 worker、图片缓存、撤销栈、文件链接、日志、环境隔离
└── tests/                     38 个 pytest 测试文件
```

---

## 2. 启动流程

```
python main.py
└─ main.py
   ├─ 环境：PYTHONNOUSERSITE=1、MPLBACKEND=Agg、disable_user_site_packages()（隔离 Conda/venv）
   └─ main()
      ├─ configure_utf8_environment()  UTF-8 环境（子进程、stdout/stderr）
      ├─ setup_logging()               logs/autolabel.log（5MB×3 滚动）+ 控制台
      ├─ configure_headless_matplotlib()  强制 Agg 后端
      ├─ QApplication + app_icon() + apply_theme()（中文字体 + Catppuccin Mocha 暗色 QSS）
      └─ MainWindow()
```

`--train-process CONFIG CANCEL [EVENT]` 走 `run_training_process()`，是 PyInstaller 打包下 Qt-free 的训练子进程入口（等价于 `python -m src.engine.train_process`）。

### MainWindow（`src/app.py`，2259 行）

- **Tab 布局**：主页（WelcomePage，最近项目/搜索/新建/打开）→ 标注 → 预览 → 训练 → 模型 → 小工具。项目相关 Tab 在打开项目时惰性创建。
- **5 个控制器实例**（构造于 `__init__` L519-532）：
  - `ProjectController`（新建/打开/导出/导入/类别/自动类别登记/备份）
  - `ModelController`（模型加载/删除/重命名/导入/导出/单图与目录推理）
  - `TrainController`（数据集校验与准备/启动停止训练/训练产物注册）
  - `TagController`（项目级与逐图标签；`tags_changed`/`image_tags_changed` 信号驱动训练筛选摘要刷新）
  - `LocateAnythingController`（可选文本标注后端：probe→preflight→后台加载；持有 `ModelController` 引用做 VRAM 互斥编排）
- **GPU 互斥**：`_confirm_disable_la_for_yolo()`——LA 激活时加载 YOLO / 开始训练前先征询关闭 LA；反向的 LA→YOLO 卸载在 `LocateAnythingController.begin_enable` 内（先 `_model_ctrl.unload()` 再显存预检）。
- **自动标注**：单图（同步 YOLO / 慢后端 LA 走 `SinglePredictWorker` 后台线程 + 重入守卫 `_single_worker`）；批量（`BatchPredictWorker` 逐图落盘、可取消、`BatchProgressDialog` 进度）。新预测经 `find_conflicts` 与已确认同类别按 IoU 去重；可选「只保留最高置信度 ROI」规则（`retain_highest_confidence_roi`）。分类任务先经 `ProjectController.preview_model_classes` + `ClassRegisterDialog` 登记新类别。
- **训练队列**：`TrainingQueue[_TrainingJob]`（FIFO，`batch_had_multiple` 决定单任务是否自动加载模型）；数据集准备到 `project/datasets/training_queue/<uuid>/`，训练完成清理；单任务完成自动注册 + 加载 best.pt，多任务队列只注册。
- **关闭**：`closeEvent` 依次——小工具脚本进程确认 → 训练取消（wait 30s）→ 单图 worker wait → LA disable → 标注保存 → 预览清理 → 持久化窗口几何 / 面板折叠 / 训练设置 / 模型阈值到 `config/config.json`。
- **Tag 乱码修复**：`repair_tag_text()` 对疑似 GBK 误解码的 UTF-8 中文做 `encode(gb18030).decode("utf-8")` 恢复（`_MOJIBAKE_CHARS` 启发式）。

---

## 3. core 层：数据模型与项目管理（`src/core/`）

### 3.1 核心数据模型（`annotation.py`，406 行，仅 stdlib）

```
Keypoint:  x, y（归一化 [0,1]）, visible（0 不可见 / 1 遮挡 / 2 可见）, label

Annotation:  class_name, class_id,
             bbox=(cx, cy, w, h) 归一化中心格式 | None,
             polygon=[(x,y),...] 归一化 | 空,
             keypoints=[Keypoint],
             confidence=1.0, confirmed=True, source="manual", id=uuid4

ImageAnnotation:  image_path, image_size=(w,h),
             annotations: list[Annotation],
             image_tags（分类任务类别标签）, image_tags_confirmed, image_tags_source,
             tags（自由格式用户标签，与 image_tags 语义不同）
```

- **状态机** `ImageAnnotation.status`：有 `image_tags` → confirmed/pending（按 `image_tags_confirmed`）；无 annotations → `unlabeled`；全 confirmed → `confirmed`；否则 `pending`。
- 关键函数：`compute_iou`（cxcywh→IoU）、`retain_highest_confidence_roi`、`find_conflicts`（已确认同类别贪心 IoU 匹配去重）、`annotation_geometry/center`（多边形用鞋带公式面积与质心；OBB/分割的显示面积是真实多边形面积）、`COCO17_SKELETON_EDGES` 姿态拓扑与官方配色、`annotation_display_label`（`"{类} | 面积 N px²"`）。

### 3.2 项目配置与管理（`project.py`，651 行）

- `ProjectConfig`（`project.json` schema）：`name / image_dir / label_dir / classes / class_colors / keypoint_templates / task_type / auto_register_classes / tags / active_data_folder / data_folders / excluded_data_folders / ...`。
- `ProjectManager`：`create/open/save`；`image_root()`；**数据版本**——镜像根一级子目录自动识别为版本（嵌套目录不重复登记），`data_folders` 显式注册（空版本可见）、`excluded_data_folders` 移除索引但文件留盘（重建同名可恢复）；`create/rename/delete_data_folder`、`move_images_to_folder`（图+标签随移）、`import_images_to_folder`（递归复制、保留子目录、同 stem JSON sidecar 一并复制）。
- **标签路径约定** `label_path_for`：`label_dir ∈ {"","."}` → 图片旁同名 `.json` sidecar；否则镜像 `labels/` 目录（保留相对镜像根的子目录结构）。
- `list_images`：`rglob` 递归，跳过 `IGNORED_IMAGE_TREE_DIR_NAMES`（crops/datasets/labels/models/model_predictions/runs）与 excluded 目录。
- `IMAGE_EXTENSIONS = {.jpg,.jpeg,.png,.bmp,.tiff,.tif,.webp}`。

### 3.3 标注 IO 与导入导出格式

- `label_io.py`：`save_annotation`（**空记录等价于无文件**——删除已有 JSON，保持目录干净）；`load_annotation`（内部格式失败时**降级尝试 iSAT → labelme 解析**外来 sidecar）。
- `formats/__init__.py`：插件式注册表（`ExportRegistry`/`ImportRegistry`）。内置 9 种导出 + 8 种导入：

| 格式 | 导出 | 导入 | 说明 |
|---|---|---|---|
| YOLO txt | ✓（detect/segment/obb/pose 按任务分发） | ✓（自动探测格式与类别） | `labels/**/*.txt` + `data.yaml` |
| COCO json | ✓（单文件 `coco.json`，像素 bbox，keypoints 名写入 categories） | ✓（文件） | |
| labelme json | ✓（每图一文件，group_id 链接姿态） | ✓（含 auto-merge 启发式） | |
| X-AnyLabeling Detect/OBB | ✓ | ✓ | rectangle / rotation shape |
| iSAT json | ✓ | ✓（跳过 `__background__`） | 实例分割 |
| roLabelImg/VOC OBB xml | ✓ | ✓（robndbox 弧度 + bndbox） | 旋转框 |
| ImageFolder | ✓（类别子目录） | ✓（is_full_import，写图+标签） | 分类 |
| CSV | ✓（labels.csv） | ✗ | 分类 |

- **跨格式约定**：内部一律归一化坐标；导出像素化（按 image_size）、导入归一化；导出路径「绝对→basename、相对去掉首段 `images/`」；统一 `only_confirmed` 过滤；JSON 一律 `ensure_ascii=False`；解析错误带 `"{文件}:{行号}"` 上下文。core 层唯一第三方依赖是 PyYAML（`formats/yolo.py`）。

### 3.4 其余 core 模块

| 模块 | 职责要点 |
|---|---|
| `config.py` | `AppConfig` 全局配置（最近项目≤10、窗口几何、面板折叠、训练设置、`enable_locateanything` 主开关） |
| `project_scan.py` | 打开旧项目时按「同 stem 匹配数」评分猜测 image_dir/label_dir 配对 |
| `annotation_classes.py` / `class_mapping.py` | 从已存标注推导类别（项目序 + 额外类字母序追加）/ 导出类映射（项目序 + 首次出现序） |
| `tags.py` | Qt-free 标签子系统：`normalize`（64 字符、非法字符校验）、`TagFilter`（includes/excludes + AND/OR、`classify()` 四分类）、`TagService` |
| `backup.py` | `BackupManager`：`.backups/<时间戳>/` 备份 project.json + 标签，保留 20 份；恢复前先做 `pre-restore-` 安全备份 |
| `train_templates.py` | `TemplateRegistry`：内置「默认」模板代码注入、用户模板按 name+task 去重持久化（builtin 永不落盘）；按任务类型裁剪参数键集 |
| `training_queue.py` | UI 无关 FIFO 队列：`active` + `waiting`，`batch_had_multiple` 批次语义 |
| `encoding_utils.py` | UTF-8 环境统一 + GBK/GB18030 兼容解码 |
| `resources.py` | `config/static_resources.json` 驱动的路径解析；`resolve_pretrained_model_path`（不触发下载） |

---

## 4. controllers 层（`src/controllers/`，5 个文件，共 2263 行）

| 控制器 | 核心职责与关键方法 |
|---|---|
| **ProjectController**（843 行，纯命令式，无信号） | `create_project` / `open_project_dialog` / `open_recent`、`export`（导出前自动备份）、`import_annotations`（skip/overwrite/merge 冲突策略，新类别自动并入并重算 class_id）、`manage_classes`、`register_auto_class`（幂等；ImageNet ID `^n\d{8}$` 黑名单；受 `auto_register_classes` 开关；**每次登记立即 save project.json** 供工作线程可见）、`preview_model_classes`、备份三件套；内部 `_move_root_images_to_image_dir`、`_import_discovered_obb_sidecars`（自动发现未转换的 VOC-OBB xml / YOLO txt sidecar） |
| **ModelController**（699 行） | `load_model`（经 backend 加载、路径相对项目解析）、`delete/rename/import_model`、`export_model_onnx`（ultralytics export 后重新 probe 元数据保留血缘）、`predict_single`（同步）、`predict_native_plot/save`（原生 YOLO 可视化，目录结果存 `model_predictions/`）、`create_single_predict_worker`（慢后端后台推理）、`predict_single_classify`（返回原始类名交 ProjectController 登记）；`unload()` 中 CUDA 处理刻意只清 `torch.cuda.empty_cache()`（避免 LA-only 场景把 CUDA 引入 GUI 进程） |
| **TrainController**（292 行） | `validate_and_prepare`（统计已确认标注；<10 条与类别不均衡仅警告；classify 类别序与 Ultralytics 字母序对齐）、`start`（**快照** project/task/base_model/dataset_size/prepared_classes——训练中途切项目不注册错）、`stop`、`register_model_after_training`（用快照定位项目、重载磁盘 registry 防并发覆盖、组装 ModelInfo 注册） |
| **TagController**（294 行，QObject） | 信号 `tags_changed()` / `image_tags_changed(path, tags)`（自连失效过滤缓存）；`add/remove/rename_tag`（破坏性操作前备份）、`set_image_tags`（规范化+自动登记新标签）、`apply_tag_to_images`（幂等、真实修改才发信号）、`load_all_image_tags`、`compute_filter_breakdown`（match/excluded/no_include/conflict 四分类计数） |
| **LocateAnythingController**（135 行，QObject） | 信号 `probe_done / preflight_blocked / load_progress / enabled / disabled / failed`；`begin_enable` 三层流水线（probe → 先 unload YOLO 再 preflight → `LocateAnythingLoadWorker` 后台加载）；`disable`（卸载回收 VRAM）；LA 预测器经 `ModelController.set_predictor` 注入，使既有自动标注流程零改动复用 |

**协作模式**：MainWindow 持有全部控制器，命令式调用 + Qt 信号回调混合；只有 TagController 与 LocateAnythingController 自带信号，其余靠返回值与 MainWindow 显式接线。

---

## 5. engine 层（`src/engine/`，Qt-free 后端抽象层）

### 5.1 后端插件体系（`backends/`）

- **`base.py`**：三个协议契约——`ModelBackend`（`probe/infer_model_format/load_predictor/create_trainer`）、`TrainerProtocol`（`train(config, on_epoch_end)/request_cancel/cancelled/get_best_metrics`）、`PredictorProtocol`（`predict/predict_with_size/predict_classify/release`）；`BackendProbe`（标准化探测结果）；异常 `BackendError ← BackendUnavailableError / UnknownBackendError`；常量 `DEFAULT_BACKEND_ID="ultralytics"`、`DEFAULT_BACKEND_RUNTIME="in_process"`。
- **`registry.py`**：按 `backend_id` 注册/解析；Ultralytics 必注册，LocateAnything **try/except 惰性注册**（任何失败不拖垮启动）。`backends/__init__.py` 延迟 re-export 避免导入重依赖。
- **`ultralytics.py`**：默认后端，全 in-process；`probe()` 用 `importlib.metadata`（不导入 YOLO），`<8.0.0` 降级警告。
- **`locateanything.py`**（827 行）+ **`locateanything_worker.py`**（513 行）：**out-of-process sidecar 架构**——历史根因是 in-process 激活 LA 的 CUDA 会杀掉共享同一张卡的 X server（`XIO: fatal IO error`）。GUI 进程**零 torch**（有回归测试断言），`LocateAnythingPredictor` 是纯 Python 代理，通过**换行分隔 JSON（NDJSON）**与子进程通信：
  - 四级成本解锁：register(0) → probe(1)（importlib.metadata + HF 缓存 `*.safetensors` 文件检查，无网络）→ preflight(2)（nvidia-smi 双门槛：总显存 ≥6GB、空闲 ≥5GB）→ load_runtime(3)。
  - `_WorkerProcess`：`_lock` 串行化所有 send/recv（单图与批量两条后台线程绝不交织协议帧）；reader 线程 + `join(timeout)` 实现超时读行（启动 600s / 推理 120s / 关闭 10s）；terminate 三级升级（shutdown → terminate → kill，幂等）。
  - worker：stdout **fd dup + sys.stdout→stderr** 保证协议通道纯度；4-bit NF4 加载（`expandable_segments:True` 防碎片化 OOM）；**双上限降采样**（长边 1024 + 面积 750k px 防 OOM——MoonViT 无 flash-attn 时注意力峰值 ∝ 面积²）；`generation_mode="hybrid"` + 采样式生成（**贪心会让 LA 无限出框**）；EXIF 转置对齐画布；单图失败只回 `{"error"}` 不杀 worker；日志全部落 `logs/locateanything_worker.log`。

### 5.2 训练链路

```
DatasetPreparer.prepare()                 # engine/dataset.py
  └─ 过滤（status/tag/class）→ 分层切分（按主类别, seed=42）
     ├─ classify: 类别子目录结构（连空类别也建目录保证字母序索引稳定），不写 data.yaml
     └─ detect/seg/obb/pose: labels/*.txt + data.yaml（图片 link_or_copy 软链→硬链→复制）
        └─ TrainConfig.to_train_args() → Trainer.train()（engine/trainer.py）
           ├─ 回调式协作取消（_TrainingCancelled 哨兵异常 + on_fit_epoch_end 强制置 epoch）
           └─ 整体跑在子进程：python -m src.engine.train_process CONFIG CANCEL [EVENT]
              └─ config/cancel(文件哨兵)/event(AUTOLABEL_EVENT\tJSONL) 三文件协议 + 守护监控线程
```

- `Trainer` 经 `get_backend(config.backend_id).create_trainer()` 创建（后端可替换）；`TrainConfig` 的 `to_train_args()` 按 task 条件化组装 Ultralytics kwargs（segment 追加 mask_ratio/overlap_mask/copy_paste，pose 追加 pose/kobj/kpt_shape）；`to_storage_dict()` 剔除运行时字段，随模型存为 `ModelInfo.train_params`。
- `TrainWorker`（utils/workers.py）双模式：inline（同进程 Trainer）或 isolated（子进程 + 事件管道）；`finished` 信号之后才发结果（`_emit_outcome` 连在 `QThread.finished` 上），避免 GUI 在原生资源拆除期间刷新模型；`_recover_process_outcome` 用 best.pt + `results.csv` 兜底恢复「模型已保存但进程异常退出」的完成态。

### 5.3 推理链路

```
图片 → Predictor.predict_native()（ultralytics predict, retina_masks=True）
  └─ _run(): boxes/obb/masks/keypoints → Annotation
     ├─ mask→多边形：只保留面积最大外部连通区（RETR_CCOMP 找内孔并合并进外环）
     ├─ 类别按 class_id / class_name（规范化）匹配项目类别；全滤时回退返回 raw 检测（防误报"无目标"）
     ├─ 关键点 visible 按 conf 阈值（>0.5→2, >0→1, else 0）
     └─ Annotation(confirmed=False, source="auto")
```

- `Predictor.recommended_imgsz` 多容器尽力解析；`_coerce_imgsz` 容忍 tensor/str/list。

### 5.4 模型注册表

- `model_manager.py`：`ModelInfo`（name/path/task/base_model/classes/metrics/train_params/backend_* 等，`to_dict/from_dict` 缺字段兜底）+ `ModelRegistry`（`models/registry.json` 持久化，`register/remove/get/rename/list_models`）。训练完成 → controller 构造 ModelInfo（含 `TrainConfig.to_storage_dict()` 快照）→ register + save。

---

## 6. UI 层（`src/ui/`）

### 6.1 标注画布（`canvas.py`，2079 行）

`AnnotationCanvas(QWidget)`：图像显示与编辑的唯一画布，**坐标全归一化 [0,1]**（与分辨率解耦），信号驱动不落盘。

- **5 种工具模式**：`select / draw_bbox / draw_obb / draw_polygon / draw_keypoint`。绘制完成 → `class_requested(px, py)` → 视图弹 `ClassPickerPopup` → 回调 `create_*_from_draw` 构造 Annotation（bbox 为 cxcywh；OBB 为轴对齐四点 + bbox；polygon 顶点数上限 4 时按中心角排序成凸四边形；keypoint 命中已有 bbox 走 `keypoint_attach_requested`）。
- **命中优先级**（mousePress）：旋转手柄 → bbox 四角 resize → 关键点 → 多边形边插入点 → 整体 move → 空白平移。多边形面命中用 `QPainterPath.contains`、边命中用像素距离（任意缩放下交互容差稳定）；「segment 的 bbox 只是元数据，不应让整个矩形可点」。
- **拖拽编辑**：基于 `ann.to_dict()` 快照的增量计算，按 `_drag_type` 分发：`move`（夹紧图内）/ `rotate_obb`（角度差旋转矩阵）/ `resize_obb_N`（对角固定 + Gram-Schmidt 正交化修复畸形历史 OBB + 32 次二分收敛）/ `poly_vertex_N` / `move_kp` / 四角缩放；编辑未确认标注自动置 confirmed。
- **冲突可视化**：`set_conflict_pairs` 双向字典，右键「保留确认框 / 保留预测框」解决；预测框 teal 虚线、未确认 ⚡ 徽标。
- 渲染细节：视口裁剪剔除 + LOD（`_scale≥0.3` 才画标签）；关键点 visible 三态（空心小圈/空心/实心）；缩放以鼠标为锚点、`ZOOM_FACTOR=1.15`；加载旋转 SVG 动画（80ms/30°）。

### 6.2 任务视图（`views/`）

- **`base.py`**：`TaskView(QWidget)` 协议——信号 `annotations_changed/status_changed/image_focus_changed/auto_label_*/images_dropped/classes_changed/user_tags_changed`；必实现方法矩阵 + 可选默认（`refresh_image_tags` 的「不动滚动」约束是批量贴标场景保持视野稳定的关键）。
- **`detect_pose.py`**（1668 行，`DetectPoseView`）：检测/分割/姿态/OBB 工作区。三栏 QSplitter（数据版本树 + FileListWidget | AnnotationCanvas | AnnotationPanel，`setSizes([320,960,300])`）。**切图非重入事务**（`_switching_image` + `_queued_image_path` 队列）；**即改即存**（`_push_undo` → `_sync_annotations_to_panel` → `_save_current` 立即写盘）；撤销栈按图像路径共享的 `OrderedDict[str, UndoStack]` LRU（`_UNDO_MAX_IMAGES=20`，`move_to_end` 刷新）；批量确认/撤销可见预标注、多图改类（逐图 push 撤销）、增量统计（`_stats_snapshot` + `_stats_cache`）；数据版本树右键（新建/添加图片/目录/重命名/删除版本）。快捷键 W/P/O/K/V、A/D/←/→、Space、Delete、Ctrl+Z/Y/C/V、Ctrl++/-/0。
- **`classify.py`**（1309 行，`ClassifyView`）：分类缩略图网格（无画布/无文件列表）。`ThumbnailGridWidget` 用 `event.ignore()` **绕过 QListWidget 的 keyboardSearch 吞键**让 1-9 快捷键冒泡；`ThumbnailDelegate` 画缩略图 + 类别色条 + 状态徽标（✓/⚡/?）；`PreviewPane`（大图 + TagChipBar，`blockSignals` 防程序推送被当编辑）；`ClassButtonBar`（前 9 类带数字快捷键与计数）；`add_auto_class_prediction` **保护已确认人工标签**（返回 False 跳过）；筛选刷新不动滚动（`_restore_scroll_after_filter` 居中当前项）；排序 filename/class（class 未标优先），密度 64-192 持久化。
- **`thumbnail_loader.py`**：单 QThread + FIFO deque + QMutex；**队列空即退出线程、下次 enqueue 重启**；加载失败静默降级。

### 6.3 主面板

| 面板 | 职责要点 |
|---|---|
| **label_panel.py**（564 行） | 标注页外壳：按 `task_type` 路由到 DetectPoseView/ClassifyView（延迟导入）；共享 `ImageCache`（16 张/512MB）与 undo 栈池；工具栏（自动标注/批量标注/LocateAnythingBar/筛选/类别/TagFilterBar）；`annotations_changed` 不重扫项目（类已提前注册，避免大项目阻塞）；拖放导入 |
| **preview_panel.py**（2012 行） | 只读预览网格 + **高级筛选/卡控**：状态/类别/数据版本/Tag 筛选；按类别的尺寸-面积-置信度-中心位置范围卡控（OK/NG，`_annotation_control_result` 语义：仅全局 ROI 时 ROI 内即通过）；**会话级 ROI**（矩形/正圆/多边形，`DetailPreviewCanvas` 交互：缩放 0.05-16x、拖动、A/D 切图）；`_pixel_filter_limits` 以项目最大图尺寸为上限；导出预览 PNG **禁止写入 image_root 内**；ROI 不进入 Annotation/ProjectManager（防污染训练数据）；分类任务禁用检测框卡控 |
| **train_panel.py**（1817 行） | 左侧可滚动参数表单（任务/数据筛选/超参/优化器/颜色/几何/混合/分割/分类/姿态参数组，`CollapsibleGroupBox`）+ 队列卡片 + 右侧 pyqtgraph 曲线（`_rebuild_quality_curves` 按任务取指标，val_loss 前向填充防掉 0）+ 日志（1000 块）；`get_train_config` 组装 `TrainConfig`；模板系统（`TemplateRegistry`，preset UI 隐藏但逻辑保留）；`eventFilter` 忽略未聚焦滚轮 + FocusOut 持久化设置；模型下拉 `[已训练] name` 映射路径 |
| **model_panel.py**（811 行） | 模型列表 + 详情（训练参数/产物富文本、打开训练目录）+ 推理预览（复用 AnnotationCanvas）+ 自动标注阈值设置（conf/iou/overlap_iou/类别匹配方式/保留最高置信度 ROI）；`set_prediction_result_dir` 递归找预览图 |
| **script_tool_panel.py**（924 行） | 可编辑 Python 脚本运行器：`QProcess` + `sys.executable -u` 子进程执行，实时回显；内置按框裁剪脚本（`_CROP_BY_BBOX_SCRIPT` v3，输出 `crops/<版本>/<类别>/`）；注入 `AUTOLABEL_PROJECT_DIR`/`AUTOLABEL_DATA_VERSION` 环境变量；旧 `app_config.script_tools` 迁移到 `~/.autolabel/tools/`；`prepare_close` 停进程 + 未保存确认 |
| **dialogs.py**（615 行） | NewProject / Export / ClassManager / Import（按格式切换帮助、冲突策略）/ BatchProgress（可取消）/ ClassRegister（批量自动标注前类别确认，ImageNet ID 黑名单标黄默认不勾） |
| **file_list.py**（591 行） | 状态着色（未标/待确认/已确认）、状态/类别/Tag 三重筛选、复选框批量（Ctrl+A 全选可见、空格切换）、Shift 范围选择、拖放导入、右键批量（确认/删除标注/移动版本/删除图片）；**滚动跳动修复**（mousePressEvent 保存/恢复 scrollbar，`_apply_filter` 把当前项定位视口中部） |
| **tag_widget.py**（580 行） | TagChipBar（单图编辑）/ TagFilterBar（三态循环 无→包含→排除、OR/AND）/ TagApplyBar（武装 tag 后按 T 批量应用）/ TagManagerDialog（双击重命名、删除同步提示）；仅通过信号传 `list[str]`/`TagFilter`，与控制器零隐式耦合 |
| **properties.py**（655 行） | AnnotationPanel：类别列表（色块+计数、双击设默认类）、标注树（关键点子节点 ◌◑● 三态可见性图标）、属性表单、TagChipBar、项目统计；树高钉在可见行数（≤8 行）；`setItemWidget` 使 itemDoubleClicked 失效 → 每行自持 `double_clicked` 信号 |
| **locateanything_bar.py**（163 行） | 三视觉状态（未启用按钮/加载中/已启用：prompt QLineEdit + 目标类别 combo + 关闭）；目标类别首项哨兵「(按名称自动匹配)」；实际标注仍走既有自动/批量按钮 |
| **augmentation_preview.py**（828 行） | 6 宫格增强预览：训练输入原图 + 单图增强 A/B + 训练采样 1/2/3；按与训练一致的参数顺序复现（Mosaic→Copy-Paste→MixUp→透视→HSV→翻转→AutoAugment→Erasing），每格显示触发操作名；种子体系可复现（seed → donor `seed^0xA5A5` → 各格 `seed+101/211/307/401/503`） |
| **class_picker.py**（299 行） | labelimg 风格类别/关键点标签弹窗：无边框 Tool 窗口、锚点定位 + 屏幕内钳制、失焦 100ms 关闭、1-9 数字键、输入即过滤、Escape 用 `WidgetWithChildrenShortcut` |
| **collapsible_group.py / loading.py** | 可折叠分组框（折叠收 MaximumHeight 让 QSplitter 收回空间）/ SVG 旋转加载指示器（SVG 缺失安全降级） |

### 6.4 主题与图标

- `theme.py`（581 行）：`PALETTE` 暗色配色（bg `#11141b`、primary `#7c5cff` 等 26 键）、中文字体按系统候选降级选择（微软雅黑→Noto Sans CJK→宋体）、角色化 QSS（`set_button_role/set_surface` 动态属性 + polish 即时生效）、完整 STYLESHEET。
- `icons.py`（299 行）：25 个内联 Lucide 风格 SVG 模板（`{color}` 占位符），按 `(name, color, size)` 三元组缓存渲染；`app_icon()` 从 `APP_LOGO_CANDIDATES` 找 logo。

---

## 7. utils 层（`src/utils/`）

| 模块 | 职责要点 |
|---|---|
| `workers.py`（534 行） | **TrainWorker**（双模式：inline/子进程隔离 + `AUTOLABEL_EVENT\t` 事件管道 + results.csv 兜底；`_emit_outcome` 连在 QThread.finished 后）、**BatchPredictWorker**（classify 16 张/批优先 `predict_classify_batch`、detect/pose 逐张；`threading.Event` 取消）、**SinglePredictWorker**（慢后端专用，避免拖垮 X 事件循环）、**LocateAnythingLoadWorker**（重模型后台加载）。所有 run() 宽捕获异常转 error 信号（QThread 异常会静默死线程） |
| `image.py`（130 行） | `load_pixmap`（EXIF autoTransform）、`get_image_size`（**手动处理 90°/270° 旋转**——Qt 5.15 size() 不自动变换）、`ImageCache`（LRU：16 张 / 512MB 双上限，`invalidate` 供删除/移动时清理） |
| `fs.py` | `link_or_copy`：symlink → hardlink → copy 三级降级（Windows 跨盘无权限时兜底） |
| `undo.py` | `UndoStack(max_depth=50)`：双栈 + 深拷贝快照；「当前状态驻留栈顶」设计，`can_undo = len>1`（栈底保留初始快照） |
| `colors.py` | Catppuccin Mocha 20 色类别调色板，`assign_color(index)` 取模环绕 |
| `logging_config.py` | 幂等日志初始化：控制台 INFO + 滚动文件 DEBUG（5MB×3、utf-8、errors=replace）；压低 ultralytics/PIL 到 WARNING；`AUTOLABEL_LOG_DIR` 可重定向 |
| `runtime_env.py` | `disable_user_site_packages`（从 sys.path 移除 user site，防两套 Python 环境混合）、`configure_headless_matplotlib`（Agg + force） |

---

## 8. 关键业务流程速览

### 8.1 标注 → 保存
画布编辑 → `annotation_created/modified/deleted` 信号 → DetectPoseView `_push_undo`（快照）→ `_sync_annotations_to_panel` → `_save_current`（写 sidecar JSON + 更新 file_list 状态/图标 + 增量统计）→ `annotations_changed` → LabelPanel 冒泡 → MainWindow 状态栏。

### 8.2 自动标注
- 单图：MainWindow 读模型面板阈值 → `ModelController.predict_single`（YOLO 同步；LA 激活时走 `SinglePredictWorker` 后台 + 重入守卫）→ `retain_highest_confidence_roi`（可选）→ `LabelPanel.add_auto_annotations`（`find_conflicts` IoU 去重；冲突对交画布可视化，右键解决）。
- 分类：`predict_single_classify` → `register_auto_class`（新类自动登记并 save）→ `add_auto_class_prediction`（保护已确认标签）。
- 批量：`BatchPredictWorker` 逐图推理 → `image_done` 槽逐图落盘（同样 IoU 去重、ROI 过滤统计）→ `BatchProgressDialog` 进度、可取消；classify 走批量预检对话框登记类别。

### 8.3 训练闭环
训练面板配置 → `TrainController.validate_and_prepare`（过滤 + 分层切分 + 导出数据集/data.yaml，图片软链省空间）→ `_TrainingJob` 入 `TrainingQueue` → `TrainController.start`（快照项目状态，`TrainWorker` 子进程隔离训练，事件管道推进进度条/曲线/日志）→ 完成 → `register_model_after_training`（写入训练时所属项目的 registry）→ 单任务自动加载 best.pt；队列串行、可清空/停止（停止时移除全部等待任务并清理数据集目录）。

### 8.4 LocateAnything 启用
点「文本标注」→ `begin_enable`：probe（依赖/权重存在性）→ 卸载 YOLO + nvidia-smi 显存预检 → `LocateAnythingLoadWorker` 后台加载 4-bit 模型（子进程，stdout 协议通道）→ `enabled` → 提示词 + 目标类别 → `set_query` → 走既有自动/批量标注按钮（predictor 已切换为 LA 代理）。

### 8.5 导入 / 导出
格式注册表分发 → 控制器预处理（备份 → 按数据版本扫描 → 按格式裁剪注解）→ 导出器写盘（统一目录约定 + only_confirmed 过滤）；导入器读盘 → 类别合并（新类别并入 project.classes 并重算 class_id）→ 冲突策略处理 → 刷新面板。ImageFolder 是 is_full_import（直接写图+标签）。

---

## 9. 测试（`tests/`，38 个文件）

覆盖：YOLO/COCO/labelme/iSAT/VOC-OBB/X-AnyLabeling 导入导出（目录结构、类别映射、确认过滤）、数据版本（导入/移动/过滤）、预览高级筛选与 ROI、画布命中测试、OBB 工具栏与 sidecar 发现、姿态骨架、分类工作台、撤销/清空/转义、训练（config 组装、worker 事件、进程事件、队列、面板自动保存、数据版本过滤）、模型（imgsz 解析、原生绘图、预训练资源）、备份 sidecar、主 matplotlib 后端、最近项目信息等。`pyproject.toml` 配置 `testpaths=["tests"]`、`pythonpath=["."]`。CI 未配置；本地运行需完整依赖（PyQt5/ultralytics/torch 等，当前 shell 环境未安装，未执行）。

---

## 10. 打包与运行

- 运行：`python main.py`（需 Conda/venv 安装 `requirements.txt`；LocateAnything 需 `pip install -e ".[locateanything]"` + NVIDIA GPU）。
- 打包：`PyInstaller.txt` 给出 Windows `--windowed --onedir` 命令，`--contents-directory lib`，打包 `config/`、`resources/`、`icon/`、`logoicon/`、`pretrained_models/`；训练子进程经 `--train-process` + 临时事件文件回传进度。
- `.gitignore`：`config/config.json`、`train_templates.json`、`project.json`、`images/`、`labels/`、`models/`、`.backups/`、`logs/` 等本机状态不入库。

---

## 11. 值得注意的设计决策与遗留问题

### 设计决策
1. **out-of-process LocateAnything**：GUI 进程零 torch，规避与 X server 共卡崩溃；NDJSON IPC + 显存双门槛 + 双上限降采样 + 采样式生成。
2. **训练子进程隔离**：config/cancel/event 三文件协议；`_recover_process_outcome` 容忍「模型已保存但进程异常退出」。
3. **训练状态快照**：TrainController 记录开始时的 project/task/dataset_size，中途切项目不注册错模型。
4. **即改即存 + LRU 撤销栈**：标注编辑立即落盘，撤销按图共享（20 张上限）。
5. **Sidecar 数据安全**：空标注等价无文件；高风险操作自动备份（20 份）；恢复前安全备份。
6. **GPU 互斥编排集中在 MainWindow**：ModelController 不依赖 LocateAnythingController（各层只认识邻居）。
7. **数据版本只索引不删盘**：`excluded_data_folders` 移除索引、文件留盘可恢复。
8. **预览 ROI 不进入标注/训练数据**：会话级分析工具，防污染。

### 遗留 / 瑕疵
- `tag_widget.py`：`TagChipBar._rebuild` 中删除按钮文本 `f"{t}  脳"` 的 `脳` 是乱码（疑应为 `×`）；`TagManagerDialog` 有一行被覆盖的错误窗口标题。
- `train_panel.py`：模板 preset UI 隐藏但逻辑保留（向后兼容）；`label_panel.py` 有隐藏的 refresh strip 遗留。
- `app.py` `_sync_available_tags` 中 `logger.warning` 高频打印原始/修复后 Tags（调试级信息用 warning 级别）。
- `train_templates.py` 注释指出旧 `TRAIN_PRESETS` 已废弃（trainer.py 保留兼容），新实现走 `TemplateRegistry`。
- 项目 Tag 历史遗留 `config/Tag` 文件为占位（README 说明当前 Tag 已写入项目配置与标注 JSON）。

---

## 12. 维护建议（可选）

1. 跑通 `pytest` 需要完整依赖环境；当前 shell 无 PyQt5/ultralytics，建议在 Conda 环境 `pip install -r requirements.txt` 后执行。
2. `tag_widget.py` 两处乱码文本可顺手修复为 `×` 与正确标题。
3. `_MOJIBAKE_CHARS` / `repair_tag_text` 的启发式依赖字符集，若项目普遍 UTF-8 可评估是否保留。
4. engine 层 `Predictor._run` 的 mask 只保留最大外部连通区——若后续需要多连通域实例，需扩展 `_mask_to_normalized_polygons` 策略。
