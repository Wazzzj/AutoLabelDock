# AutoLabel Dock

> Label a few, train a round, auto-label the rest — a desktop image-annotation and YOLO-training iteration loop.

![Python](https://badgen.net/badge/Python/%E2%89%A53.10/blue)
![License](https://badgen.net/badge/License/AGPL--3.0/green)
![Qt](https://badgen.net/badge/Qt/PyQt5/41cd52)

**English** | [简体中文](README.md)

AutoLabel Dock is a desktop image-annotation tool built on **PyQt5 + Ultralytics YOLOv8**. It runs cross-platform on Linux / macOS / Windows.

It turns "annotation" and "training" into a single closed loop: label a batch of images manually (or with help from an existing model), confirm the results, train a custom YOLO model in one click, then use the new model to auto-label the rest of your data — every iteration leaves fewer boxes to fix by hand.

> Note: the application UI is currently Chinese-only; this English README documents the project for international users.

![AutoLabel Dock](resources/screenshots/overall.png)

---

## Screenshots

| Panel | Screenshot |
|:---|:---:|
| Annotation (detect/pose) | ![Annotation UI](resources/screenshots/labeling.png) |
| Classification | ![Classification UI](resources/screenshots/cls.png) |
| LocateAnything | ![LA UI](resources/screenshots/locateanything.png) |
| Training panel | ![Training panel](resources/screenshots/train.png) |
| Model management | ![Model management](resources/screenshots/models.png) |

---

## Features

### 🔍 Five annotation tasks

- **Object detection (detect)**: bounding-box (bbox) annotation
- **Instance segmentation (segment)**: editable polygon annotation
- **Oriented bounding boxes (obb)**: drag-to-create rotated boxes with rotation and rectangle-preserving corner resize
- **Keypoint pose (pose)**: bbox + skeleton keypoints
- **Image classification (classify)**: single label per image, with a grid selector and `1`–`9` hotkeys for quick labeling that auto-advances to the next image

### ✏️ Annotation experience

- Keyboard-centric workflow similar to LabelImg
- Smooth canvas: bbox, polygon, four-corner OBB, and keypoint drawing with drag-to-move/resize, scroll-wheel zoom and pan
- Detection boxes, OBBs, and segmentation polygons show class plus area percentage and pixel area on the annotation canvas and large preview; thumbnails show class plus area percentage. Bboxes use width × height, while OBB and segmentation use true polygon area.
- Per-image undo stack (depth 50), auto-save on image switch, LRU image cache
- File list color-coded by status (confirmed / pending / unlabeled), with drag-and-drop image import, combined status + class + tag filtering, and right-click batch operations
- Changing a class while multiple images are checked or selected replaces matching annotations of the current old class across all selected images.
- Deleting the current image keeps the list near the same position and focuses the next image (or the previous one at the end). Pending-box drags repaint only the canvas until mouse release to avoid panel-sync stalls.
- The project `model_predictions/` folder is excluded from annotation image scans and folder trees, so saved inference outputs are not treated as labeling data

### 🔎 Preview and advanced filters

- Combine annotation status, class, data version, and advanced filters in the preview grid.
- Advanced filters include image Tags plus min/max ranges for width, height, area, confidence, and center X/Y, with one-click reset.
- Width, height, and center coordinates are percentages of image dimensions; area is a percentage of image area, so one threshold works across resolutions.
- Class and all enabled numeric constraints must match the same annotation. Editing a range automatically enables it; unchecked ranges do not participate.
- Classification projects keep Tag filtering while object geometry filters are disabled.
- The large-preview toolbar exports the current image as a full-resolution PNG containing boxes or contours, class and area labels, control styling, the OK/NG result, and ROI overlays.
- **File → Export all preview images...** exports every project preview as `*_preview.png` while preserving data-version and image subdirectories. Exporting into the project image directory is blocked so rendered previews cannot be mistaken for training images.
- Catppuccin Mocha dark theme

<details>
<summary><b>Keyboard shortcuts</b></summary>

**Tools & modes**

| Shortcut | Function |
|:---|:---|
| `W` | Box drawing mode |
| `P` | Polygon drawing mode |
| `O` | OBB drag drawing mode; use the outside handle to rotate |
| `K` | Keypoint mode |
| `V` | Select/move mode |

**Navigation**

| Shortcut | Function |
|:---|:---|
| `A` / `←` | Previous image (auto-saves current) |
| `D` / `→` | Next image (auto-saves current) |

**Annotation operations**

| Shortcut | Function |
|:---|:---|
| `Space` | Confirm selected annotation |
| `Delete` | Delete selected annotation |
| `Ctrl+Z` / `Ctrl+Y` | Undo / Redo |
| `Ctrl+C` / `Ctrl+V` | Copy / paste annotations |
| `Ctrl+S` | Save |

**View**

| Shortcut | Function |
|:---|:---|
| `Ctrl++` / `Ctrl+-` | Zoom in / out |
| `Ctrl+0` | Fit to window |
| `F5` | Rescan image directory |

**Classification only**

| Shortcut | Function |
|:---|:---|
| `1`–`9` | Quick-select class label |

</details>

### 🤖 Model-assisted auto-labeling

- Load any YOLO weights to pre-label a single image or a batch (batches run on a background thread and save to disk image by image)
- Conflict detection: predictions are matched against existing confirmed same-class annotations by IoU to avoid duplicate boxes
- Confirmation lifecycle: auto-labels start as "pending" (dashed boxes); any manual edit marks them confirmed; once every box on an image is confirmed, the canvas locks to prevent accidental edits
- Optional LocateAnything-3B text-labeling backend lets you describe targets in natural language (see [Optional: LocateAnything Text Labeling](#optional-locateanything-text-labeling))

### 🏷️ Tag subsystem

- Attach custom tags to images (independent of classification labels) for dataset organization
- Tri-state filter chips (none → include → exclude), with AND/OR combination when multiple include tags are selected
- Select multiple images and press `T` to tag them in bulk
- The same tag filter can scope the training subset at train time

### 🔄 Training loop

- One-click dataset preparation: stratified train/val split by primary class, zero-copy via symlinks (with automatic fallback on Windows — see [Platform Notes](#platform-notes))
- Filter training data by all versions or one selected data version, together with status, class, and Tag constraints
- YOLO training for detect, segment, obb, pose, and classify projects
- Training runs in an isolated child process so CUDA, DataLoader, or native-library failures cannot terminate the annotation UI; completed model files are recovered and registered when possible
- Training presets plus full hyperparameter control; live loss / mAP curves; cancellable mid-run
- Augmentation previews compare single-image transforms with full training samples, including transformed annotation overlays, Mosaic, MixUp, Copy-Paste, classification policies, and Erasing
- On completion the model is auto-registered to the model library and auto-loaded, ready for inference immediately

### 📦 Model management

- Model library: training outputs are registered automatically; import external weights, rename, and delete are supported
- The model panel can run native inference on a single image or a directory; directory results are saved to the project `model_predictions/` folder
- Multi-model metric comparison dialog

### 📥 Data import / export

| Format | Export | Import | Applicable tasks |
|:---|:---:|:---:|:---|
| YOLO (txt) | ✅ | ✅ | Detection / Segmentation / OBB / Pose |
| roLabelImg OBB (xml) | ✅ | ✅ | OBB |
| COCO (json) | ✅ | ✅ | Detection / Pose |
| labelme (json) | ✅ | ✅ | Detection / Pose |
| X-AnyLabeling OBB (json) | ✅ | ✅ | OBB |
| iSAT (json) | ✅ | ✅ | Detection / Segmentation |
| ImageFolder | ✅ | ✅ | Classification (folder-per-class) |
| CSV | ✅ | ❌ | Classification |

iSAT export mirrors image subdirectories under `labels/` and writes one same-stem JSON file per image with standard `info`, `objects`, pixel-space `segmentation`, `bbox`, `area`, `group`, and `layer` fields. Polygon contours are preserved; bbox-only annotations are converted to four-point polygons, and confirmed-only export is supported.

OBB projects use the normalized Ultralytics format `class_id x1 y1 x2 y2 x3 y3 x4 y4`. Set the project task to `obb` before importing nine-column YOLO labels so they are not mistaken for a small pose layout. `classes.txt` accepts both one class name per line and indexed entries such as `0:label`.

OBB import/export is split into two explicit formats. roLabelImg XML stores `robndbox` values (`cx`, `cy`, `w`, `h`, and a radian `angle`); X-AnyLabeling JSON stores four pixel-coordinate points with `shape_type: "rotation"` and a radian `direction`. Both formats preserve image subdirectories and use same-stem image/annotation files.

Press `O` or choose **Oriented box** from the canvas context menu, then hold the left mouse button and drag out a rectangle. Select the box and drag the circular handle outside its top edge to rotate the complete box around its center; dragging a corner resizes it like a regular box while preserving a true rectangle.

When an OBB project is created or reopened, same-stem XML/TXT sidecars in the project root, image directory, or common annotation directories are converted automatically if the internal `labels/*.json` file is still missing. The format matching the most unlabeled images is selected, XML wins ties, and existing internal labels are never overwritten.

### 🛡️ Data safety

- **Automatic backups**: destructive operations such as export and class changes snapshot to the in-project `.backups/` first, keeping the latest 20; a safety backup is also taken before any restore
- **Independent storage**: annotations are stored as one JSON file per image, so a single corrupt file doesn't affect the rest
- **Global config**: recent projects, window geometry, thresholds, training templates, etc. are stored under the repository `config/` folder

---

## Quick Start

```
1. Create project  →  2. Import images  →  3. Annotate  →  4. Confirm  →  5. Train  →  6. Iterate ↻
```

1. **Create a project**: pick detect, segment, obb, pose, or classify, specify the project directory (the project will automatically scan and load `images/labels`), or leave it blank and drag images in later
2. **Import images**: drag images onto the file list (you can also add images to the project's `images/` directory later and press `F5` to refresh)
3. **Annotate**: draw manually; or load a YOLO weight to pre-label automatically; or use the LocateAnything text-labeling backend to describe targets in natural language (see [Optional: LocateAnything Text Labeling](#optional-locateanything-text-labeling))
4. **Confirm**: review auto-labels image by image — editing confirms them
5. **Train**: prepare the dataset and start training in one click, watching the curves live
6. **Iterate**: the freshly trained model auto-loads, ready to auto-label the remaining images

---

## Requirements

- Python ≥ 3.10
- OS: Linux / macOS / Windows

## Installation

The core dependencies are light: **PyQt5** (UI) and **Ultralytics** (YOLO inference/training), plus a few smaller libraries such as pyqtgraph (training curves) — see `requirements.txt` for the full list.

Using Miniconda to create an isolated environment is recommended:

```bash
git clone https://github.com/xzcGit/autolabel-dock.git
cd autolabel-dock
conda create -n autolabel python=3.10 -y
conda activate autolabel
pip install -r requirements.txt
```

The optional LocateAnything-3B text-labeling backend is large and installed on demand; see the enablement requirements in [Optional: LocateAnything Text Labeling](#optional-locateanything-text-labeling).

## Running

```bash
python main.py
```

## Model Weights

Bundled YOLO pretrained weights are stored in `pretrained_models/`. Other official models (`yolov8n.pt`, `yolov8s.pt`, etc.) are downloaded automatically by Ultralytics the first time they are used; if automatic download fails, place the corresponding `.pt` file in `pretrained_models/`. See the next section for how to obtain the LocateAnything-3B weights.

## Optional: LocateAnything Text Labeling

LocateAnything-3B is an optional open-vocabulary detection backend that lets you describe targets in natural language. The rest of the app works perfectly fine without it. Enabling it requires **all three** of the following conditions; if any is unmet, the UI shows a message (in Chinese) and the rest of the app is unaffected:

**1. Install the optional dependencies**

```bash
pip install -e ".[locateanything]"
```

This additionally installs transformers, accelerate, bitsandbytes, and decord (the base install does not include these heavy dependencies).

**2. Download the model weights in advance**

At runtime the model is loaded in offline mode (`HF_HUB_OFFLINE=1`) and is **not** downloaded automatically, so you must download `nvidia/LocateAnything-3B` into the local HuggingFace cache beforehand:

```bash
hf download nvidia/LocateAnything-3B
# Legacy tool: huggingface-cli download nvidia/LocateAnything-3B
```

The default cache location is `~/.cache/huggingface/hub`; `$HF_HOME` and `$HUGGINGFACE_HUB_CACHE` are also honored.

**3. GPU memory**

An NVIDIA GPU with a working `nvidia-smi` is required — **CPU is not supported**; total VRAM must be ≥ 6GB and free VRAM ≥ 5GB at enable time (on a single-GPU machine the desktop display also consumes VRAM, hence the relatively high free-memory threshold).

> Additionally: LocateAnything and YOLO models never occupy the GPU at the same time — enabling LocateAnything automatically unloads the loaded YOLO model, and the reverse (loading a YOLO model or starting training) prompts you to confirm disabling LocateAnything first. It runs in a separate subprocess, isolated from the main UI process.

---

## Platform Notes

Dataset preparation creates many links "pointing to the original images." The code (`link_or_copy` in `src/utils/fs.py`) falls back automatically in the following priority order so it runs on any platform:

```
symlink (symbolic link) → hardlink (hard link) → copy
```

| Method | Condition | Speed | Extra space |
|:---|:---|:---:|:---:|
| symlink | Supported and permitted (default on Linux/macOS; Windows needs Developer Mode or admin) | Fastest | Near zero |
| hardlink | symlink failed, and source and target are on the same volume (no privilege needed on Windows NTFS) | Fastest | Near zero |
| copy | Both of the above failed (typically: cross-drive + non-Developer-Mode) | Slow | Equal to total image size |

> ⚠️ **Windows recommendation**: enable Developer Mode (Settings → Privacy & security → For developers) — a one-time setting that persists and behaves like Linux; or keep the project directory and the image directory on the same drive (the hardlink path kicks in automatically).
>
> **Other notes**: prefer ASCII-only project paths; if you point the image directory at another drive (cross-drive), links degrade to copies and consume extra disk space.

---

## Project Structure

```
autolabel-dock/
├── src/
│   ├── core/         Pure data models and IO (annotations, project config, import/export formats, backups)
│   ├── engine/       YOLO training / inference wrappers and dataset preparation
│   ├── controllers/  Bridging and orchestration between UI and core layers
│   ├── ui/           PyQt5 UI components (canvas, panels, dialogs)
│   └── utils/        Background threads, undo stack, image cache, and other utilities
├── tests/            pytest tests
├── resources/        Screenshots and other static resources
├── main.py           Application entry point
└── requirements.txt  Dependency list
```

---

## Contributing

Issues and PRs are welcome. After setting up the environment per the Installation section above, run the tests:

```bash
pytest
```

## License

This project is released under the **[AGPL-3.0](LICENSE)** license.

> The project's core dependencies use strong copyleft licenses:
>
> - **PyQt5** — GPL-3.0
> - **Ultralytics (YOLOv8)** — AGPL-3.0
>
> The strictest among the dependencies is AGPL-3.0, and this project aligns accordingly. To use it in a closed-source/commercial product, obtain the corresponding commercial licenses for those dependencies yourself (a PyQt commercial license, an Ultralytics enterprise license).

## Links

- **[Linux DO](https://linux.do/)**
