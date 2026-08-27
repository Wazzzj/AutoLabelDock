# AutoLabel Dock

> Label a little, train a version, then auto-label again: a desktop image annotation tool built around the YOLO training loop.

![Python](https://badgen.net/badge/Python/%E2%89%A53.10/blue)
![License](https://badgen.net/badge/License/AGPL--3.0/green)
![Qt](https://badgen.net/badge/Qt/PyQt5/41cd52)

[简体中文](README.md) | **English**

AutoLabel Dock is a cross-platform desktop tool built with **PyQt5 + Ultralytics YOLO**, covering manual annotation, model-assisted pre-annotation, dataset preparation, training, and iterative labeling.

![AutoLabel Dock](resources/screenshots/overall.png)

---

## Screenshots

| Panel | Screenshot |
|:---|:---:|
| Detection | ![Detection UI](resources/screenshots/det.png) |
| Segmentation | ![Segmentation UI](resources/screenshots/seg.png) |
| Oriented bounding boxes | ![OBB UI](resources/screenshots/obb.png) |
| Pose | ![Pose UI](resources/screenshots/pose.png) |
| Classification | ![Classification UI](resources/screenshots/cls.png) |
| Training | ![Training panel](resources/screenshots/train.png) |
| Model management | ![Model management](resources/screenshots/models.png) |

---

## Main Features

### Annotation Tasks

- **Object detection (`detect`)**: bounding-box annotation.
- **Instance segmentation (`segment`)**: polygon annotation; bounding boxes can also be converted to rectangular polygons during training export.
- **Oriented bounding boxes (`obb`)**: drag to create a rectangle, rotate it toward the target, and resize it from linked corners.
- **Keypoint pose (`pose`)**: bounding box + keypoint skeleton annotation. COCO 17-point poses use the human-body topology, distinguish visible, occluded, and invisible states, and support keypoint templates.
- **Image classification (`classify`)**: one label per image, with a thumbnail grid and number-key shortcuts for rapid labeling.

### Annotation Experience

- A keyboard-oriented workflow similar to LabelImg.
- Draw, select, move, resize, zoom, and pan on the canvas; edit boxes, keypoints, and polygons.
- Annotation labels include class and area information; OBB and segmentation areas use the true polygon area.
- Per-image undo / redo, auto-save when switching images, and image caching.
- Image states are clearly separated into unlabeled, pending, and confirmed.
- Drag-and-drop import plus batch confirmation, deletion, moving, and class changes.
- Clear every annotation in the current image from the annotation list after confirmation, with undo support.
- Combined filtering by status, class, and custom Tags.
- Catppuccin Mocha dark theme.

### Data Preview and Advanced Filters

- Combine status, class, data version, Tag, size, area, confidence, and center-position filters.
- Percentage-based conditions work across image resolutions; classification projects retain image Tag filtering only.
- Rectangle, ellipse, and polygon ROIs plus per-class OK / NG control rules. ROIs are used only for preview analysis and are not written to annotations or training data.
- Export the current preview or all previews with annotations, area, control results, and ROI overlays.

### Data Versions

- When a project is created, its project directory also becomes the image root. Direct child directories are discovered as data versions, while nested directories inside a version are not registered again.
- Create, rename, and remove data versions, and move images with their labels between versions.
- Removing a version only removes its index and does not delete files from disk; recreating a version with the same name restores access.
- Training and batch labeling can be scoped by data version, status, class, and Tag.

### Model-Assisted Annotation

- Load YOLO weights to pre-annotate one image or a batch of images.
- Optionally keep only the highest-confidence predicted ROI in each image; the same rule applies to model inference, automatic annotation, and batch annotation.
- Batch inference runs in a background thread, saves each image incrementally, and can be canceled.
- New predictions are matched against confirmed same-class annotations by IoU to avoid duplicate boxes.
- Automatic annotations start as pending; classification tasks preserve confirmed results.
- Optionally connect LocateAnything-3B to generate detection boxes from natural-language target descriptions.

### Tag System

- Image-level custom Tags are independent of classification labels. They support batch assignment, include / exclude filters, AND / OR combinations, and training-data filtering.

### Training Loop

- One-click dataset preparation for all five YOLO tasks, with stratified sampling, training templates, and full hyperparameter control.
- Configure Freeze and other training parameters directly in the panel; the model-structure viewer is no longer included.
- Augmentation preview covers geometry, color, Mosaic, MixUp, Copy-Paste, and related policies.
- Training runs in an isolated subprocess, can be canceled, displays metrics in real time, and supports serial model queues.
- Waiting jobs can be cleared independently; stopping the active job also removes every waiting job.
- A single training job saves, registers, and auto-loads its best model; multi-job queues register models without repeatedly loading them.

### Model Management

- Import, load, rename, delete, and compare metrics for trained models and external `.pt` / `.onnx` weights.
- Export PT models, or convert a PT model and export it as ONNX.
- Run inference on one image or a directory; directory results are saved under `model_predictions/`.
- YOLO and LocateAnything coordinate GPU usage.

### Utilities

- An editable Python script runner supports saving custom tools, viewing live output, and stopping execution.
- A built-in crop-by-annotation-box script can process project images by data version.

### Import / Export

| Format | Export | Import | Applicable tasks |
|:---|:---:|:---:|:---|
| YOLO txt | Supported | Supported | Detection / Segmentation / OBB / Pose |
| roLabelImg OBB xml | Supported | Supported | OBB |
| COCO json | Supported | Supported | Detection / Segmentation / Pose |
| labelme json | Supported | Supported | Detection / Segmentation / Pose |
| X-AnyLabeling Detect json | Supported | Supported | Detection |
| X-AnyLabeling OBB json | Supported | Supported | OBB |
| iSAT json | Supported | Supported | Detection / Segmentation |
| ImageFolder | Supported | Supported | Classification |
| CSV | Supported | Not supported | Classification |

Exports use each format's conventional directory layout and preserve image subdirectories. OBB projects support YOLO four-point labels, roLabelImg XML, and X-AnyLabeling JSON, and can discover unconverted sidecar annotations automatically.

### Data Safety

- New projects use sidecar storage: each image keeps a same-name JSON annotation beside it, such as `0001.jpg` with `0001.json`. Legacy projects configured with a separate `labels/` directory remain supported.
- Moving, renaming, or deleting a project image applies the same operation to its sidecar JSON.
- Before high-risk operations, `project.json` and annotations from every data version are backed up to `.backups/`; the latest 20 backups are retained by default.
- Global configuration is stored in `config/`, and runtime logs are stored in `logs/`.

A typical project data directory looks like this:

```text
my-project/
├── project.json
├── version-a/              Data version
│   ├── 0001.jpg
│   └── 0001.json           Sidecar annotation
├── version-b/              Data version
├── datasets/               Temporary training datasets
├── models/                 Registered models
└── .backups/               Automatic backups
```

---

## Quick Start

```text
1. Create project -> 2. Import images -> 3. Annotate / pre-annotate -> 4. Confirm -> 5. Train -> 6. Iterate
```

1. Create a project: choose `detect`, `segment`, `obb`, `pose`, or `classify`, then enter the project directory and initial classes. The project directory becomes the image root, and existing direct child directories become data versions automatically.
2. Import images: drag images into the file list or select an image directory. Directory import copies images and same-name JSON files, preserves subdirectories, and refreshes data versions automatically.
3. Annotate: draw boxes or polygons, drag and rotate OBBs, place keypoints, or select classification labels. You can also load YOLO weights for automatic pre-annotation.
4. Confirm: review automatic annotations, correct them, and confirm the result.
5. Train: select the base model, template, and hyperparameters in the Training panel, prepare the dataset, and start training.
6. Iterate: use the newly trained model to continue labeling unfinished images.

---

## Fine-Contour Instance Segmentation Parameters

The following configuration is intended for `segment` projects with a fixed camera, one ring-shaped target, and measurements that require both inner and outer contours, such as area, perimeter, and adhesive width. A two-stage fine-tuning workflow is recommended; use the first stage's `best.pt` as the base model for the second stage.

If the example dataset contains only one `item` class, enable `single_cls`. Before training, make sure that:

- Only manually reviewed and confirmed polygon annotations are used.
- The training and validation sets do not contain the same physical object or adjacent duplicate frames.
- `train` and `val` do not point to the same image directory.
- Self-intersecting contours are corrected and inner holes in adhesive rings are preserved correctly.
- High-resolution source images are used for precision measurements; increasing `imgsz` cannot restore details missing from the source image.

### Stage 1: Freeze the Backbone

Enter the following parameters in the Training panel:

```yaml
task: segment
model: yolo26m-seg.pt
epochs: 40
batch: 4
imgsz: 1024
device: "0"
freeze: 10
workers: 4
patience: 15
val_ratio: 0.25

optimizer: AdamW
lr0: 0.001
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.1

hsv_h: 0.0
hsv_s: 0.1
hsv_v: 0.2
degrees: 2.0
translate: 0.03
scale: 0.10
shear: 0.0
perspective: 0.0
flipud: 0.5
fliplr: 0.5
mosaic: 0.0
mixup: 0.0

mask_ratio: 2
overlap_mask: true
copy_paste: 0.0
copy_paste_mode: flip

single_cls: true
resume: false
```

### Stage 2: Fine-Tune the Full Model at a Low Learning Rate

Select the `best.pt` produced by stage 1, clear the fixed Freeze value, enable **Use default value**, and configure the remaining parameters as follows:

```yaml
task: segment
model: path/to/stage1/weights/best.pt
epochs: 120
batch: 4
imgsz: 1024
device: "0"
freeze: null
workers: 4
patience: 25
val_ratio: 0.25

optimizer: AdamW
lr0: 0.0003
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 2.0
warmup_momentum: 0.8
warmup_bias_lr: 0.1

hsv_h: 0.0
hsv_s: 0.1
hsv_v: 0.2
degrees: 2.0
translate: 0.03
scale: 0.10
shear: 0.0
perspective: 0.0
flipud: 0.5
fliplr: 0.5
mosaic: 0.0
mixup: 0.0

mask_ratio: 2
overlap_mask: true
copy_paste: 0.0
copy_paste_mode: flip

single_cls: true
resume: false
```

Parameter notes:

| Parameter | Recommendation and rationale |
|:---|:---|
| `model` | Prefer `yolo26m-seg.pt`; use `yolo26s-seg.pt` when VRAM is limited. Avoid starting with an x-size model on a very small dataset. |
| `batch` | Use the largest value that does not exhaust VRAM. At `1024`, try 4, then 2, then 1; do not lower `imgsz` as the first solution. |
| `mask_ratio` | `2` preserves finer training masks than the default `4`. If VRAM allows, compare `1` separately without changing other parameters at the same time. |
| `mosaic` | Disable it. The target occupies most of the image, and Mosaic would shrink it and introduce unrealistic seams. |
| `mixup` | Disable it. Image blending creates translucent boundaries that do not exist in the real data. |
| `copy_paste` | Disable it. A single adhesive ring at a fixed position is not suitable for pasting extra instances. |
| `degrees` / `translate` / `scale` | Use only light geometric augmentation to avoid interpolation and cropping damage along fine edges. |
| `flipud` / `fliplr` | Use `0.5` only when object orientation, lighting direction, and defect location have no directional meaning; otherwise set both to `0.0`. |
| `overlap_mask` | Keep `true` for single-instance images; it does not fix self-intersecting polygons. |
| `single_cls` | Enable it only while the project always contains one target class. Disable it before adding more classes. |

Classification-only parameters (`erasing`, `auto_augment`, and `dropout`) and pose-only parameters (`pose`, `kobj`, and `kpt_shape`) are not passed to Ultralytics during `segment` training and can remain at their defaults.

### Inference and Contour Measurement

- Use the same `imgsz=1024` for training and inference.
- Keep `retina_masks=True` during inference and prefer the binary masks in `result.masks.data`.
- For precision contour calculations, use `cv2.RETR_CCOMP` to preserve holes and `cv2.CHAIN_APPROX_NONE` to retain every edge point.
- Do not calculate perimeter or adhesive width directly from polygons heavily simplified with `approxPolyDP`.
- In addition to regular mask mAP, consider Boundary IoU, HD95, area error, perimeter error, and adhesive-width error.
- Physical measurements also require camera calibration to convert pixel coordinates to real-world units.

See the [Ultralytics training configuration](https://docs.ultralytics.com/modes/train), [instance segmentation documentation](https://docs.ultralytics.com/tasks/segment), and [data augmentation guide](https://docs.ultralytics.com/guides/yolo-data-augmentation) for related parameters.

---

## Keyboard Shortcuts

Use **File → Select Image Directory...** or `Ctrl+Shift+O` to add a local image directory to the current project. Images are copied into the project root while preserving their subdirectory layout, and direct child directories are automatically recognized as data versions. Annotation, preview, data-version, and training-filter views refresh immediately after import.

### General

| Shortcut | Function |
|:---|:---|
| `Ctrl+N` / `Ctrl+O` | Create / open a project |
| `Ctrl+Shift+O` | Add an image directory to the current project |
| `Ctrl+E` / `Ctrl+I` | Export / import annotations |
| `Ctrl+S` | Save the current annotation |
| `Shift+A` / `Ctrl+Shift+A` | Auto-label the current image / run batch labeling |
| `T` | Apply the selected Tag to selected images |
| `F5` | Rescan project images |

### Detection / Segmentation / Pose

| Shortcut | Function |
|:---|:---|
| `W` | Box drawing mode |
| `P` | Polygon drawing mode |
| `O` | Drag to draw an OBB; drag the outside circular handle to rotate it after selection |
| `K` | Keypoint mode |
| `V` | Select / move mode |
| `Enter` | Finish the current polygon |
| `Esc` | Cancel the current drawing or close a class / keypoint picker, then return to selection mode |
| `A` / `←` | Previous image |
| `D` / `→` | Next image |
| `Space` | Confirm the selected annotation |
| `Ctrl+Space` | Confirm every annotation in the current image |
| `Delete` | Delete the selected annotation |
| `Ctrl+Z` / `Ctrl+Y` | Undo / redo |
| `Ctrl+C` / `Ctrl+V` | Copy / paste annotations |
| `Ctrl++` / `Ctrl+-` | Zoom in / out |
| `Ctrl+0` | Fit to window |

### Classification

| Shortcut | Function |
|:---|:---|
| `1` to `9` | Quickly select a class |
| `Space` | Confirm the current or selected images |
| `Backspace` | Clear classification labels from selected images |
| `Delete` | Delete selected images and their app annotations after confirmation |
| `Ctrl+A` | Select all images in the current filtered result |
| `Ctrl+Z` / `Ctrl+Y` | Undo / redo |

---

## Requirements

- Python >= 3.10
- Operating systems: Linux / macOS / Windows
- Core dependencies: PyQt5, Ultralytics, ONNX Runtime, pyqtgraph, PyYAML, and packaging

Training and inference can run on CPU, but an NVIDIA GPU is recommended for practical training. The LocateAnything-3B backend requires an NVIDIA GPU.

---

## Installation

Using Conda to create an isolated environment is recommended:

```bash
git clone https://github.com/xzcGit/autolabel-dock.git
cd autolabel-dock
conda create -n autolabel python=3.10 -y
conda activate autolabel
pip install -r requirements.txt
```

To enable the optional LocateAnything-3B text-labeling backend, install its additional dependencies:

```bash
pip install -e ".[locateanything]"
```

---

## Running

```bash
python main.py
```

If Chinese terminal output is garbled on Windows, prefer Windows Terminal / PowerShell 7 and make sure the terminal uses UTF-8. The application also configures Python's UTF-8 environment at startup.

---

## Model Weights

Pretrained weights are stored in `pretrained_models/`. External `.pt` / `.onnx` files can also be imported from the Model panel. Training outputs are registered automatically, and PT models can be exported to ONNX.

---

## Optional: LocateAnything Text Labeling

LocateAnything-3B is an optional natural-language detection backend. To enable it:

1. Install the dependencies:

   ```bash
   pip install -e ".[locateanything]"
   ```

2. Download the model weights:

   ```bash
   hf download nvidia/LocateAnything-3B
   ```

3. Prepare a supported NVIDIA GPU; at least 6 GB of VRAM is recommended.

LocateAnything runs in an offline subprocess and coordinates GPU memory usage with YOLO models.

---

## Platform Notes

Training dataset preparation organizes source images into the YOLO directory structure. To minimize disk usage, the application creates image references in this order:

```text
symlink -> hardlink -> copy
```

| Method | Trigger | Speed | Extra space |
|:---|:---|:---:|:---:|
| symlink | Supported and permitted; Linux / macOS usually work by default, while Windows requires Developer Mode or administrator privileges | Fast | Almost none |
| hardlink | Symlink creation failed and the source and target are on the same disk volume | Fast | Almost none |
| copy | Both previous methods failed, for example across Windows drives without Developer Mode | Slow | Equal to the total image size |

On Windows, enable **Developer Mode** when possible, or keep the project and image directories on the same drive. Simple ASCII project paths are recommended to reduce path-encoding issues in third-party training libraries.

---

## Project Structure

```text
autolabel-dock/
├── config/                  Global configuration, static resource indexes, and training templates
├── icon/                    Runtime SVG assets and the Windows EXE icon
├── pretrained_models/       Bundled YOLO pretrained weights
├── resources/screenshots/   README screenshots
├── src/
│   ├── app.py               Main window and application orchestration
│   ├── controllers/         Project, model, training, Tag, and LocateAnything controllers
│   ├── core/                Data models, project config, label IO, import/export, and backups
│   ├── engine/              YOLO training/inference, dataset preparation, and model backends
│   ├── ui/                  PyQt5 widgets, canvas, panels, dialogs, and views
│   └── utils/               Image cache, background threads, file linking, logging, and undo stack
├── main.py                  Application entry point
├── PyInstaller.txt          Windows packaging reference
├── pyproject.toml           Project metadata and optional dependencies
├── requirements.txt         Base dependencies
└── LICENSE                  AGPL-3.0 license
```

---

## Packaging Notes

The repository includes `PyInstaller.txt` as a Windows packaging reference. Run it in an environment with Ultralytics installed and keep `icon/`, `logoicon/`, `pretrained_models/`, and other runtime resources. Add optional dependencies such as LocateAnything separately when required by the release.

---

## Development

After installing the dependencies, run the application directly:

```bash
python main.py
```

Run the test suite with:

```bash
pytest
```

---

## License

This project is released under the **[AGPL-3.0](LICENSE)** license.

PyQt5 uses GPL-3.0, and Ultralytics uses AGPL-3.0. If you plan to integrate this project into closed-source or commercial software, verify the applicable terms and obtain commercial licenses for the relevant dependencies where necessary.

---

## Links

- [Linux DO](https://linux.do/)
