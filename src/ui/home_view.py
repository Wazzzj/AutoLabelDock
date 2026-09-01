"""Scheme-A 主页：hero 标题 + 最近项目卡片 + 快速操作 + 右侧检查器。

对应 ui_redesign/a-home.html 原型。替代旧版 WelcomePage 的界面职能
（WelcomePage 类本身保留，`_read_recent_project_info` 仍被测试引用）。

卡片数据分层加载：先显示名称/路径/任务类型（同步读 project.json），
随后在后台线程扫描每张图的 sidecar 统计确认进度，完成后经信号回填
（进度条 + 已确认/待确认 + 最佳指标 + 相对时间）。
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.core.config import AppConfig
from src.ui.icons import icon
from src.ui.theme import PALETTE

_TASK_COLORS = {
    "detect": PALETTE["primary"],
    "segment": PALETTE["primary"],
    "obb": PALETTE["warning"],
    "pose": PALETTE["teal"],
    "classify": PALETTE["success"],
}


def _relative_time(ts: float) -> str:
    delta = time.time() - ts
    if delta < 90:
        return "刚刚"
    if delta < 3600:
        return f"{int(delta // 60)} 分钟前"
    if delta < 86400:
        return f"{int(delta // 3600)} 小时前"
    if delta < 172800:
        return "昨天"
    return f"{int(delta // 86400)} 天前"


class ProjectCard(QFrame):
    """最近项目卡片：徽标 + 名称/路径 + 任务标签 + 统计 + 分段进度条。"""

    open_requested = pyqtSignal(str)
    edit_requested = pyqtSignal(str)
    remove_requested = pyqtSignal(str)
    open_dir_requested = pyqtSignal(str)

    def __init__(self, path: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.path = path
        self.setObjectName("ProjectCard")
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        grid = QGridLayout(self)
        grid.setContentsMargins(16, 14, 16, 12)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)

        # 徽标（名称前两字，色相由路径散列决定）
        badge_hues = [("#1D3A63", PALETTE["primary"]), ("#3A2A12", PALETTE["warning"]),
                      ("#12332B", PALETTE["success"]), ("#2D1D3E", PALETTE["violet"]),
                      ("#123252", PALETTE["teal"])]
        self._badge = QLabel(self)
        self._badge.setObjectName("CardBadge")

        self._name_label = QLabel(self)
        self._name_label.setObjectName("CardName")
        self._path_label = QLabel(self)
        self._path_label.setObjectName("CardPath")
        self._task_tag = QLabel(self)
        self._task_tag.setObjectName("CardTaskTag")

        head = QHBoxLayout()
        head.setSpacing(10)
        head.addWidget(self._badge)
        head_info = QVBoxLayout()
        head_info.setSpacing(1)
        head_info.addWidget(self._name_label)
        head_info.addWidget(self._path_label)
        head.addLayout(head_info)
        head.addStretch(1)
        head.addWidget(self._task_tag, 0, Qt.AlignTop)
        grid.addLayout(head, 0, 0, 1, 3)

        # 统计：图片 / 标注数 / 最佳指标
        self._stat_values = [QLabel("—", self) for _ in range(3)]
        self._stat_keys = [QLabel("图片", self), QLabel("标注数", self), QLabel("最佳指标", self)]
        stats_row = QHBoxLayout()
        stats_row.setSpacing(22)
        for i in range(3):
            col = QVBoxLayout()
            col.setSpacing(0)
            self._stat_values[i].setObjectName("CardStatValue")
            self._stat_keys[i].setObjectName("CardStatKey")
            col.addWidget(self._stat_values[i])
            col.addWidget(self._stat_keys[i])
            stats_row.addLayout(col)
        stats_row.addStretch(1)
        grid.addLayout(stats_row, 1, 0, 1, 3)

        # 分段进度条：绿=已确认，黄=待确认，灰=未标注
        bar_host = QHBoxLayout()
        bar_host.setSpacing(0)
        self._bar_layout = bar_host
        self._seg_ok = QLabel(self)
        self._seg_pending = QLabel(self)
        self._seg_rest = QLabel(self)
        for seg in (self._seg_ok, self._seg_pending, self._seg_rest):
            seg.setFixedHeight(4)
            bar_host.addWidget(seg, 1)
        grid.addLayout(bar_host, 2, 0, 1, 3)

        foot = QHBoxLayout()
        foot.setSpacing(14)
        self._legend = QLabel(self)
        self._legend.setObjectName("CardLegend")
        foot.addWidget(self._legend)
        foot.addStretch(1)
        self._time_label = QLabel("", self)
        self._time_label.setObjectName("CardLegend")
        foot.addWidget(self._time_label)
        grid.addLayout(foot, 3, 0, 1, 3)

        self.setStyleSheet(
            "#ProjectCard{background:%s;border:1px solid %s;border-radius:14px;}"
            "#ProjectCard:hover{border:1px solid %s;}"
            "#CardBadge{border-radius:9px;font-size:12px;font-weight:700;}"
            "#CardName{color:%s;font-size:14px;font-weight:600;}"
            "#CardPath{color:%s;font-size:10.5px;}"
            "#CardTaskTag{border-radius:4px;font-size:9.5px;font-weight:700;"
            "letter-spacing:0.08em;padding:2px 8px;}"
            "#CardStatValue{color:%s;font-size:13px;font-weight:600;}"
            "#CardStatKey{color:%s;font-size:10px;}"
            "#CardLegend{color:%s;font-size:10px;}"
            % (
                PALETTE["panel"], PALETTE["line"], PALETTE["accent"]
                if "accent" in PALETTE else PALETTE["primary"],
                PALETTE["text"], PALETTE["text_subtle"],
                PALETTE["text"], PALETTE["text_subtle"], PALETTE["text_subtle"],
            )
        )

    # ── 数据填充 ──
    def set_basics(self, name: str, task_type: str, missing: bool = False) -> None:
        self._name_label.setText(name)
        self._badge.setText(name[:2])
        bg, fg = _TASK_BADGE_HUES[hash(self.path) % len(_TASK_BADGE_HUES)]
        self._badge.setStyleSheet(
            f"background:{bg};color:{fg};border-radius:9px;padding:6px 8px;")
        color = _TASK_COLORS.get(task_type, PALETTE["text_subtle"])
        self._task_tag.setText((task_type or "?").upper())
        self._task_tag.setStyleSheet(
            f"background:{color}22;color:{color};border-radius:4px;"
            "font-size:9.5px;font-weight:700;letter-spacing:0.08em;padding:2px 8px;")
        if missing:
            self._path_label.setText("project.json 缺失或损坏 — 右键可移除")

    def set_path_text(self, text: str) -> None:
        # 中部省略：过长路径折叠为中段省略号，避免撑破卡片列宽
        if len(text) > 38:
            text = text[:19] + "…" + text[-17:]
        self._path_label.setText(text)
        self._path_label.setToolTip(text)

    def set_stats(self, images: int, labeled: int, confirmed: int, pending: int,
                  best_metric: tuple[str, float] | None, mtime_str: str) -> None:
        self._stat_values[0].setText(f"{images:,}")
        self._stat_values[1].setText(f"{labeled:,}" if labeled else "—")
        self._stat_keys[1].setText("标注数")
        if best_metric:
            self._stat_values[2].setText(f"{best_metric[1]:.3f}")
            self._stat_keys[2].setText(best_metric[0])
        total = max(1, images)
        self._seg_ok.setStyleSheet(
            f"background:{PALETTE['success']};border-radius:2px;")
        self._seg_pending.setStyleSheet(
            f"background:{PALETTE['warning']};border-radius:2px;")
        self._seg_rest.setStyleSheet(
            f"background:{PALETTE['line']};border-radius:2px;")
        rest = max(0, total - confirmed - pending)
        self._bar_layout.setStretch(0, confirmed)
        self._bar_layout.setStretch(1, pending)
        self._bar_layout.setStretch(2, rest if rest else (0 if (confirmed or pending) else 1))
        self._legend.setText(
            f"<span style='color:{PALETTE['success']}'>●</span> 已确认 {confirmed}"
            f"　<span style='color:{PALETTE['warning']}'>●</span> 待确认 {pending}")
        self._time_label.setText(mtime_str)

    # ── 交互 ──
    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self.open_requested.emit(self.path)
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event) -> None:  # noqa: N802
        menu = QMenu(self)
        menu.addAction("修改项目", lambda: self.edit_requested.emit(self.path))
        menu.addAction("打开资源目录", lambda: self.open_dir_requested.emit(self.path))
        menu.addSeparator()
        menu.addAction("移除项目", lambda: self.remove_requested.emit(self.path))
        menu.exec_(event.globalPos())


_TASK_BADGE_HUES = [
    ("#1D3A63", PALETTE["primary"]),
    ("#3A2A12", PALETTE["warning"]),
    ("#12332B", PALETTE["success"]),
    ("#2D1D3E", PALETTE["violet"]),
    ("#123252", PALETTE["teal"]),
]


class HomeView(QWidget):
    """主页：hero + 最近项目卡片网格 + 快速操作 + 右侧检查器。"""

    new_project_requested = pyqtSignal()
    open_project_requested = pyqtSignal()
    import_annotations_requested = pyqtSignal()
    open_project_path = pyqtSignal(str)
    edit_project_requested = pyqtSignal(str)
    remove_project_requested = pyqtSignal(str)
    open_project_dir_requested = pyqtSignal(str)
    _card_stats_ready = pyqtSignal(str, dict)

    def __init__(self, app_config: AppConfig, parent: QWidget | None = None):
        super().__init__(parent)
        self._config = app_config
        self._cards: dict[str, ProjectCard] = {}
        self._scan_generation = 0
        self._filter_text = ""

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── 主列 ──
        self._main = QWidget(self)
        main_lay = QVBoxLayout(self._main)
        main_lay.setContentsMargins(30, 26, 24, 20)
        main_lay.setSpacing(10)

        hero = QLabel(self)
        hero.setText(
            f"<span style='color:{PALETTE['text']};font-size:22px;font-weight:700;'>继续你的</span>"
            f"<span style='color:{PALETTE['primary']};font-size:22px;font-weight:700;'>标注闭环</span>")
        main_lay.addWidget(hero)
        sub = QLabel("标注 → 训练 → 自动预标注 → 再标注。选择一个最近项目继续，或新建。", self)
        sub.setObjectName("HeroSub")
        sub.setStyleSheet(f"color:{PALETTE['text_subtle']};font-size:12.5px;")
        main_lay.addWidget(sub)
        main_lay.addSpacing(8)

        # 最近项目 section header
        sec = QHBoxLayout()
        sec_title = QLabel("最近项目", self)
        sec_title.setStyleSheet(
            f"color:{PALETTE['text_subtle']};font-size:10.5px;font-weight:700;"
            "letter-spacing:0.14em;")
        sec.addWidget(sec_title)
        self._count_chip = QLabel("0", self)
        self._count_chip.setStyleSheet(
            f"color:{PALETTE['text_subtle']};font-size:10.5px;border:1px solid "
            f"{PALETTE['line']};border-radius:5px;padding:1px 7px;")
        sec.addWidget(self._count_chip)
        sec.addStretch(1)
        self._filter_edit = QLineEdit(self)
        self._filter_edit.setPlaceholderText("按名称过滤…")
        self._filter_edit.setFixedSize(240, 28)
        self._filter_edit.setStyleSheet(
            f"background:{PALETTE['panel']};border:1px solid {PALETTE['line']};"
            f"border-radius:8px;padding:0 10px;color:{PALETTE['text']};font-size:12px;")
        self._filter_edit.textChanged.connect(self._apply_filter)
        sec.addWidget(self._filter_edit)
        main_lay.addLayout(sec)

        # 卡片网格（放进滚动容器，项目多时不撑破）
        self._grid_host = QWidget(self)
        self._grid = QGridLayout(self._grid_host)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(14)
        self._grid.setVerticalSpacing(14)
        for col in range(3):
            self._grid.setColumnStretch(col, 1)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._grid_host)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        scroll.setStyleSheet("background:transparent;")
        main_lay.addWidget(scroll, 1)

        root.addWidget(self._main, 1)

        # ── 右侧检查器 ──
        inspector = QWidget(self)
        inspector.setFixedWidth(292)
        insp_lay = QVBoxLayout(inspector)
        insp_lay.setContentsMargins(0, 26, 0, 20)
        insp_lay.setSpacing(0)

        def _section(title: str) -> tuple[QWidget, QVBoxLayout]:
            box = QWidget(inspector)
            v = QVBoxLayout(box)
            v.setContentsMargins(20, 14, 20, 16)
            v.setSpacing(9)
            t = QLabel(title, box)
            t.setStyleSheet(
                f"color:{PALETTE['text_subtle']};font-size:10.5px;font-weight:600;"
                "letter-spacing:0.12em;")
            v.addWidget(t)
            box.setStyleSheet(f"border-bottom:1px solid {PALETTE['line']};")
            insp_lay.addWidget(box)
            return box, v

        box, v = _section("快速开始")
        big_primary = QLabel("＋ 新建项目", box)
        big_primary.setObjectName("QuickStartPrimary")
        big_primary.setAlignment(Qt.AlignCenter)
        big_primary.setFixedHeight(38)
        big_primary.setStyleSheet(
            f"background:{PALETTE['primary']};color:#04121F;border-radius:9px;"
            "font-size:13px;font-weight:600;")
        big_primary.setCursor(Qt.PointingHandCursor)
        big_primary.mouseReleaseEvent = (  # type: ignore[method-assign]
            lambda ev, s=self.new_project_requested: s.emit())
        v.addWidget(big_primary)
        for text, sig in (("📂 打开项目", self.open_project_requested),
                          ("⤓ 导入标注", self.import_annotations_requested)):
            item = QLabel(text, box)
            item.setAlignment(Qt.AlignCenter)
            item.setFixedHeight(38)
            item.setCursor(Qt.PointingHandCursor)
            item.setStyleSheet(
                f"background:{PALETTE['panel']};border:1px solid {PALETTE['line_strong']};"
                f"border-radius:9px;color:{PALETTE['text']};font-size:12.5px;")
            item.mouseReleaseEvent = (  # type: ignore[method-assign]
                lambda ev, s=sig: s.emit())
            v.addWidget(item)
            v.addSpacing(2)

        box, v = _section("快捷键")
        keys = QLabel(
            f"<span style='color:{PALETTE['text_muted']};font-family:Menlo;'>Ctrl+N</span>"
            f"&nbsp;&nbsp;<span style='color:{PALETTE['text_subtle']}'>新建项目</span><br>"
            f"<span style='color:{PALETTE['text_muted']};font-family:Menlo;'>Ctrl+O</span>"
            f"&nbsp;&nbsp;<span style='color:{PALETTE['text_subtle']}'>打开项目</span><br>"
            f"<span style='color:{PALETTE['text_muted']};font-family:Menlo;'>Ctrl+Shift+O</span>"
            f"&nbsp;&nbsp;<span style='color:{PALETTE['text_subtle']}'>添加图片目录</span><br>"
            f"<span style='color:{PALETTE['text_muted']};font-family:Menlo;'>Ctrl+E / Ctrl+I</span>"
            f"&nbsp;&nbsp;<span style='color:{PALETTE['text_subtle']}'>导出 / 导入</span><br>"
            f"<span style='color:{PALETTE['text_muted']};font-family:Menlo;'>⌘K</span>"
            f"&nbsp;&nbsp;<span style='color:{PALETTE['text_subtle']}'>全局命令搜索</span>",
            box)
        keys.setStyleSheet("font-size:11.5px;line-height:1.7;")
        v.addWidget(keys)

        box, v = _section("工作流提示")
        tip = QLabel(
            f"<span style='color:{PALETTE['text_subtle']};font-size:11.5px;line-height:1.7;'>"
            "新建项目时，所选目录即图片根目录；<b style='color:"
            f"{PALETTE['text_muted']}'>一级子目录自动识别为数据版本</b>。"
            "标注以 sidecar JSON 与图片同目录存放。</span>", box)
        tip.setWordWrap(True)
        v.addWidget(tip)

        insp_lay.addStretch(1)
        root.addWidget(inspector)

        self._card_stats_ready.connect(self._on_card_stats_ready)
        self.refresh_recent_projects()

    # ── 数据 ──
    def refresh_recent_projects(self) -> None:
        """按 AppConfig 重建卡片，并启动后台统计 enrichment。"""
        paths = [str(p) for p in self._config.recent_projects]
        if self._filter_text:
            paths = [p for p in paths if self._filter_text.lower() in p.lower()]
        # 清空网格
        while self._grid.count():
            item = self._grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._cards.clear()
        self._count_chip.setText(str(len(paths)))

        for i, path in enumerate(paths):
            card = ProjectCard(path, self._grid_host)
            name = Path(path).name
            task = ""
            missing = False
            try:
                cfg = json.loads((Path(path) / "project.json").read_text(encoding="utf-8"))
                task = str(cfg.get("task_type", ""))
                name = str(cfg.get("name", name))
            except (OSError, ValueError):
                missing = True
            card.set_basics(name, task, missing=missing)
            card.set_path_text(path)
            card.open_requested.connect(self.open_project_path.emit)
            card.edit_requested.connect(self.edit_project_requested.emit)
            card.remove_requested.connect(self.remove_project_requested.emit)
            card.open_dir_requested.connect(self.open_project_dir_requested.emit)
            self._cards[path] = card
            self._grid.addWidget(card, i // 3, i % 3, Qt.AlignTop)

        rows = (len(paths) + 2) // 3
        for r in range(rows):
            self._grid.setRowStretch(r, 1 if r == rows - 1 else 0)

        if paths:
            self._scan_generation += 1
            generation = self._scan_generation
            threading.Thread(
                target=self._scan_cards_stats,
                args=(paths, generation),
                daemon=True,
            ).start()

    def _scan_cards_stats(self, paths: list[str], generation: int) -> None:
        """后台逐项目统计：图片数 / 确认进度 / 最佳指标 / 修改时间。"""
        from src.core.label_io import load_annotation

        for path in paths:
            if generation != self._scan_generation:
                return
            info: dict = {"images": 0, "labeled": 0, "confirmed": 0, "pending": 0,
                          "best": None, "mtime": ""}
            try:
                cfg_path = Path(path) / "project.json"
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                info["mtime"] = _relative_time(cfg_path.stat().st_mtime)
                image_dir = Path(cfg.get("image_dir", "."))
                if not image_dir.is_absolute():
                    image_dir = Path(path) / image_dir
                exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
                images = sorted(
                    p for p in image_dir.rglob("*")
                    if p.suffix.lower() in exts and not p.name.startswith(".")
                )
                info["images"] = len(images)
                # sidecar 进度：与标注页同款状态语义
                #（image_tags 优先 → 全部标注已确认=confirmed → 有标注=pending）
                ann_total = 0
                for img in images:
                    js = img.with_suffix(".json")
                    if not js.exists():
                        continue
                    try:
                        data = json.loads(js.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        continue
                    anns = data.get("annotations", [])
                    ann_total += len(anns)
                    for a in anns:
                        if a.get("confirmed"):
                            info["confirmed"] += 1
                        else:
                            info["pending"] += 1
                info["labeled"] = ann_total  # “标注数”=标注总数
                # 最佳指标（models/registry.json）
                reg = Path(path) / "models" / "registry.json"
                if reg.exists():
                    data = json.loads(reg.read_text(encoding="utf-8"))
                    best: float | None = None
                    label = ""
                    for model in data.get("models", []):
                        for key, value in (model.get("metrics") or {}).items():
                            try:
                                v = float(value)
                            except (TypeError, ValueError):
                                continue
                            k = key.lower()
                            if "map50-95" in k or "map50-95" in k:
                                continue
                            if "map50" in k:
                                if best is None or v > best:
                                    best, label = v, "最佳 mAP50"
                            elif "top1" in k and label != "最佳 mAP50":
                                if best is None or (best is not None and v > best):
                                    best, label = v, "top-1"
                    if best is not None:
                        info["best"] = (label or "最佳指标", best)
            except (OSError, ValueError):
                pass
            self._card_stats_ready.emit(path, info)

    def _on_card_stats_ready(self, path: str, info: dict) -> None:
        card = self._cards.get(path)
        if card is None:
            return
        card.set_stats(
            images=int(info.get("images", 0)),
            labeled=int(info.get("labeled", 0)),
            confirmed=int(info.get("confirmed", 0)),
            pending=int(info.get("pending", 0)),
            best_metric=info.get("best"),
            mtime_str=str(info.get("mtime", "")),
        )

    def _apply_filter(self, text: str) -> None:
        self._filter_text = text.strip()
        self.refresh_recent_projects()
