# AutoLabel Dock — 三套 UI 设计方案

> 基于本程序性质（PyQt5 桌面图像标注工具、画布密集型、YOLO 训练闭环、面向 AI 工程师/标注团队）设计的三套差异化 UI 方案。每套含：设计定位、完整配色令牌、字体排版、逐页面改造要点、UX 规范、反模式与 PyQt5 实现路径。生成于 2026-08 会话。

---

## 0. 程序性质与设计约束（三套方案共享的前提）

| 性质 | 对设计的影响 |
|---|---|
| 画布密集型（标注页三栏） | 画布必须拿到最大可用面积；面板可折叠、可记忆状态 |
| 长时间连续使用（标注/训练迭代） | 深色护眼优先；状态信息（已确认/待确认/未标注）一眼可辨 |
| 多类别标注（最多 20 色） | 类别调色板与主题背景需保持 ≥3:1 对比；亮色主题需加深色变体 |
| 训练曲线/日志（pyqtgraph） | 曲线配色、网格、前景填充需随主题切换 |
| 键盘流操作（LabelImg 式快捷键） | 快捷键提示条（状态栏/面板底部）成为一等 UI 元素 |
| 跨平台（Win/macOS/Linux） | 字体降级链、DIP 缩放、深浅主题都要平台兜底 |

**PyQt5 可实现性红线**（设计时已考虑）：
- QSS 无法做真实 backdrop-blur（毛玻璃）→ 玻璃感用「半透明 rgba 面板 + 渐变底色 + 细边框 + 顶部高光」模拟；
- QSS 无 box-shadow → 用 `QGraphicsDropShadowEffect`（注意性能，列表慎用）或「底部 1px 高光边」替代；
- 动画保持克制：QSS 只支持过渡有限的属性；画布/面板动效用 Qt 动画框架，避免影响标注响应。

---

## 方案 A：Pro Dark · 数据看板（专业深色 · 高密度）

### 1. 设计定位

- **风格**：Modern Dark (Cinema) × OLED Dark，类 Figma / Blender / trading terminal 气质。
- **目标用户**：重度标注工程师、模型训练者；一天 8 小时以上使用。
- **关键词**：cinematic dark、high-density、precision、indigo/teal、status-driven、eye-friendly。
- **旋钮**：variance 6 / motion 3 / density 8。

### 2. 配色令牌（深色）

```css
/* 语义令牌（QSS 变量化后放入 theme.py 的 PALETTE_DARK_PRO） */
--bg-deep        #020203   /* 窗口/最底层 */
--bg-base        #0B0F1A   /* 主工作区（午夜蓝，避免纯黑 OLED 拖影） */
--bg-elevated    #111827   /* 面板/卡片 */
--bg-raised      #1B2436   /* 悬浮/选中行 */
--surface        #0F1420   /* 输入框/次级面板 */
--border         rgba(148,163,184,0.14)
--border-strong  rgba(148,163,184,0.28)
--text           #EDEDEF
--text-muted     #8A8F98
--text-subtle    #5B6472
--accent         #5E6AD2   /* indigo：主按钮/焦点/选中标注 */
--accent-soft    rgba(94,106,210,0.16)
--teal           #2DD4BF   /* 次色：已确认/成功/曲线 val */
--amber          #F59E0B   /* 强调 CTA/警告/训练中 */
--green          #34D399   /* 确认态 */
--red            #F87171   /* 危险/错误 */
--focus-ring     #8B93E8
```

类别标注色：沿用 Catppuccin 20 色，但**加深一档**（保证深底对比 ≥3:1），例如 `#A6E3A1→#7FDCA6`、`#89B4FA→#6FA8F5`。

### 3. 字体与排版

- 标题/正文：`Inter`（桌面 fallback：Segoe UI / PingFang SC / Noto Sans CJK），数字与坐标用等宽：`JetBrains Mono / SF Mono / Consolas`。
- 字号刻度：11/12/13/15/18/24px；标注框标签 11px 加粗 + 半透明底色胶囊。
- 密度：行高 26–30px、面板内边距 8–12px（density 8），工具栏 36px 高。

### 4. 逐页面改造要点

| 页面 | 改造 |
|---|---|
| **欢迎页** | 左侧品牌区换成暗色渐变 Logo 卡（indigo→深蓝渐变 + 微弱辉光）；最近项目列表改为「卡片行」：项目名 + 任务类型徽章（detect=teal/segment=green/obb=amber/pose=purple/classify=blue）+ 状态点（可打开=绿点）。搜索框聚焦高亮 indigo。 |
| **标注页（三栏）** | 左栏（数据版本树+文件列表）：文件行三态状态点（未标注=灰点/待确认=amber 点/已确认=teal 勾）；选中行 `bg-raised` + 左侧 3px indigo 竖条；复选框用自定义 check 图标。中栏画布：`bg-deep`，底部状态胶囊显示 `图 12/240 · 缩放 85%`；标注框线宽 2px、选中 3px + 白色外描边保证任意图上可见。右栏属性面板：折叠组头 36px、关键点可见性三态图标。底部新增**快捷键提示条**（W 框 / P 多边形 / O 旋转 / K 关键点 / V 选择），可折叠。 |
| **预览页** | 网格缩略图卡片化：圆角 8、hover 抬升 + 边框 indigo；卡控 OK 徽标 teal、NG 徽标红、ROI 外虚线灰；底部工具栏数据版本/状态/类别筛选用分段控件。 |
| **训练页** | 左侧参数表单分组标题加任务图标；`epoch` 进度条改细条 + 百分比；右侧曲线：Train Loss=teal、Val Loss=amber、mAP=indigo，网格 `rgba` 细线，**val_loss 前向填充**保持可见；日志面板深底等宽字体；队列卡片「● 训练中 / ○ 等待」用脉动绿点。 |
| **模型页** | 模型列表行：任务徽章 + 指标摘要（mAP@50 大字 + 相对基线箭头）；详情面板「训练参数」用等宽 11px 呈现；推理预览画布同标注页风格。 |
| **小工具页** | 编辑器深色语法色（关键词 indigo/字符串 teal/注释 muted）；运行按钮绿、停止红。 |

### 5. UX 规范要点

- 所有状态色同时提供**图标/形状**区分（不止颜色）：确认=勾、待确认=圆点、未标注=空心圆（色盲友好）。
- 对比度：正文 ≥7:1（深底高对比），次要文字 ≥4.5:1；类别色 ≥3:1 单独校验。
- 训练进行中：状态栏常驻 GPU/epoch 迷你指示，禁止用闪烁动画（尊重 reduced-motion）。
- 焦点环统一 `--focus-ring`，键盘可达（Tab 顺序 = 视觉顺序）。

### 6. 反模式（避免）

- 纯黑 `#000000` 大面积（OLED 拖影）；高饱和霓虹大面积；无意义动效；把训练日志做成彩色（保持单一 muted）。

### 7. PyQt5 实现路径

1. `theme.py` 新增 `PALETTE_PRO` 字典 + 单独 `STYLESHEET_PRO`；`AppConfig.theme` 增加 `"pro-dark"` 选项。
2. 画布背景/标注线宽/状态点样式集中在 `canvas.py`/`file_list.py` 的常量引用处改为令牌。
3. 曲线配色在 `train_panel.py` 的 `_rebuild_quality_curves` 改为令牌取值。
4. 快捷键提示条为新增小组件（`ui/shortcut_bar.py`），读 `AppConfig` 折叠状态。
5. 动效：仅保留 hover 变色（QSS）+ 训练状态点脉动（QTimer，180ms/次，尊重 reduced-motion）。

---

## 方案 B：Glass · 玻璃拟态沉浸（现代 · 优雅 · AI 感）

### 1. 设计定位

- **风格**：Glassmorphism × AI-Native UI，类 Linear / Raycast / 高端 AI 工具气质。
- **目标用户**：想突出「AI 智能标注 + LocateAnything」卖点的个人/团队；用于演示、宣传、客户展示。
- **关键词**：frosted glass、layered、ambient、violet、spacious、premium、AI-first。
- **旋钮**：variance 7 / motion 5 / density 5。

### 2. 配色令牌（深紫夜色 + 玻璃）

```css
--bg-deep        #0B0618   /* 窗口底层（紫夜色渐变顶） */
--bg-gradient    linear-gradient(180deg, #171030 0%, #0B0618 100%)
--glass          rgba(255,255,255,0.06)   /* 面板主体 */
--glass-strong   rgba(255,255,255,0.09)   /* 悬浮/选中 */
--glass-input    rgba(255,255,255,0.05)
--border         rgba(255,255,255,0.10)
--border-strong  rgba(196,181,253,0.30)   /* 紫罗兰细边框 */
--highlight      rgba(255,255,255,0.08)   /* 面板顶部 1px 高光 */
--text           #F5F3FF
--text-muted     #B8B0D6
--text-subtle    #7E76A3
--accent         #7C3AED   /* 紫罗兰：主按钮/焦点 */
--accent-soft    rgba(124,58,237,0.20)
--cyan           #06B6D4   /* 交互强调：链接/激活/流式 AI 状态 */
--green          #34D399
--amber          #FBBF24
--red            #F87171
--focus-ring     #A78BFA
```

类别标注色：选用高饱和紫系/青系变体（`#A78BFA / #22D3EE / #F0ABFC / #FDE68A / #86EFAC ...`），保证玻璃底上可辨。

### 3. 字体与排版

- 标题/正文：`Inter`（fallback 同方案 A），标题字重 600–700、字距 -0.01em。
- 玻璃质感细节：面板圆角 **16px**、1px 顶部白色高光、细边框 `rgba`；大圆角贯穿按钮/输入框/列表容器。
- 密度：标准（5/10），面板内边距 12–16px、行高 32px，整体比方案 A 松弛 30%。

### 4. 逐页面改造要点

| 页面 | 改造 |
|---|---|
| **欢迎页** | 整页紫夜色渐变 + 两团缓慢浮动环境光斑（QPainter 绘制大半径低透明度圆斑，QTimer 缓移，reduced-motion 时静止）；品牌标题字重 700 大字号；「新建项目」主按钮紫罗兰 + 微弱辉光；最近项目为玻璃卡片行（圆角 16、hover 高光上浮）。 |
| **标注页** | 三栏全部玻璃化：面板半透明 + 1px 高光边；画布 `bg-deep` 保持不透（图像清晰度优先）；左栏文件列表选中行 `glass-strong` + 紫罗兰竖条；LocateAnything 栏激活时顶部 1px cyan 呼吸光边（提示 AI 后端在线）；自动标注按钮激活态 cyan。 |
| **预览页** | 缩略图卡片玻璃化；卡控徽标半透明胶囊（OK=green 底 15% 透明度 / NG=red 底 15%）；ROI 覆盖线 cyan。 |
| **训练页** | 参数分组玻璃卡片；曲线：Train Loss=#A78BFA（紫）、Val Loss=#22D3EE（青）、mAP=#F0ABFC；日志面板 `glass-input`；「开始训练」按钮紫罗兰 + 辉光，「停止」红。 |
| **模型页** | 模型详情卡片玻璃化；推理预览画布同标注页；ONNX/PT 导出按钮描边式（transparent bg + 边框）。 |
| **小工具页** | 编辑器透明玻璃底 + 紫色系语法高亮；运行/停止按钮胶囊形。 |

### 5. UX 规范要点

- **AI 标注明确标识**：自动标注结果带 `source="auto"` 徽标「AI」胶囊（cyan），与人工标注区分——满足「AI 内容必须标注」规范。
- 玻璃半透明不能伤可读性：正文全部实色 `--text`，禁止 50% 以下透明度文字；玻璃面板底图在文字区域压暗。
- 悬浮/聚焦态：`glass-strong` + 1px `border-strong`，150ms 过渡。
- 动画克制：环境光斑 30–60s 周期、透明度 ≤0.10；训练进度用细条 + 大字号百分比（实时流式信息的无障碍降级）。
- 触控/点击目标 ≥32px（桌面），hover 有 `cursor-pointer`。

### 6. 反模式（避免）

- 整屏模糊背景导致文字发虚；多个玻璃层叠出「脏玻璃」；霓虹滥用（只在 AI 状态与主 CTA 用）；把图片内容也玻璃化（画布必须锐利）。

### 7. PyQt5 实现路径

1. `theme.py` 新增 `PALETTE_GLASS` + `STYLESHEET_GLASS`；面板统一用 `rgba` 背景 + 边框 + 顶部 1px 高光线（QSS `border-top: 1px solid rgba(255,255,255,0.08)` 模拟）。
2. 欢迎页环境光斑：`WelcomePage.paintEvent` 用 QRadialGradient 画 2 团光斑，`QPropertyAnimation` 慢移（`QSettings` 存 `enable_ambient`，respect reduced-motion）。
3. 玻璃效果受限于 QSS：不做真实 backdrop-blur；如需更强质感，可在关键面板外包一层 `QGraphicsBlurEffect`（仅静态背景图场景，注意性能）。
4. LocateAnything 激活态：`locateanything_bar.py` 顶部 1px 边框颜色随 `enabled` 信号切 cyan，配 1s 呼吸动画。
5. 徽章系统新增 `ui/badges.py`（AI/OK/NG/任务类型胶囊），供标注页/预览页/模型页复用。

---

## 方案 C：Bright Minimal · 明亮极简（清爽 · 协作 · 新手友好）

### 1. 设计定位

- **风格**：Minimalism & Swiss Style × 明亮工作台，类 LabelImg 亲和 + Linear 清爽的折中。
- **目标用户**：标注团队协作、新手入门、教学演示；不喜欢暗色压抑环境的用户；办公室/会议室明亮光照场景。
- **关键词**：light、white-space、blue、clean、approachable、grid、minimal。
- **旋钮**：variance 3 / motion 2 / density 5。

### 2. 配色令牌（浅色）

```css
--bg            #F8FAFC   /* 窗口底 */
--bg-panel      #FFFFFF   /* 面板/卡片 */
--bg-alt        #F1F5F9   /* 次级面板/输入框 */
--bg-hover      #E9EEF5
--border        #DDE5EE
--border-strong #B9C7DA
--text          #0F172A   /* 主文字（slate-900） */
--text-muted    #475569
--text-subtle   #94A3B8
--accent        #1E3A8A   /* 知识蓝：主按钮/焦点/选中 */
--accent-soft   rgba(30,58,138,0.08)
--blue          #3B82F6   /* 链接/激活 */
--purple        #7C3AED   /* 训练/模型相关强调 */
--green         #16A34A
--amber         #D97706
--red           #DC2626
--focus-ring    #2563EB
```

类别标注色：亮底需要**深色系 20 色**（如 `#0E7490 / #7C3AED / #DC2626 / #B45309 / #15803D / #BE185D ...`，对比度 ≥4.5:1），与深色方案的亮色系形成镜像。

### 3. 字体与排版

- 标题：`Plus Jakarta Sans`（fallback：系统无衬线）；正文同字体 13–14px。
- 留白优先：面板内边距 16–24px、行高 34–38px、分组间距 24px；大标题 20–24px。
- 分隔用「留白 + 细线」而非底色块；按钮圆角 8px（比方案 B 收敛）。

### 4. 逐页面改造要点

| 页面 | 改造 |
|---|---|
| **欢迎页** | 纯白大留白：左侧品牌区极简（Logo + 一句话 slogan），右侧最近项目为朴素列表行（名称 + 任务徽章），「新建项目」大蓝按钮居中于卡片；搜索框圆角 8、focus 蓝色描边。 |
| **标注页** | 三栏白底 + 1px 细线分隔；文件列表状态用**文字徽章**（未标注=灰字/待确认=amber 字/已确认=green 字）而非色点，白底可读性最好；画布 `bg-alt` 浅灰（图像亮色边缘可见）；选中标注蓝框 + 白色外描边；右栏折叠组头细线分隔。 |
| **预览页** | 缩略图纯白卡片 + 细边框；OK/NG 用描边徽章（浅色底 + 深色字）；筛选工具栏用分段按钮（Segment）风格。 |
| **训练页** | 表单分组用「大标题 + 细线」替代色块；曲线：Train Loss=#2563EB、Val Loss=#D97706、mAP=#7C3AED，白底网格浅灰；日志面板 `bg-alt` + 深灰等宽字；进度条细条蓝色。 |
| **模型页** | 模型卡片列表（白卡 + 指标右对齐）；「加载模型」主蓝按钮；详情面板干净表格化。 |
| **小工具页** | 编辑器白底 + 标准语法色（关键词蓝/字符串绿/注释灰）；运行绿、停止红实色按钮。 |

### 5. UX 规范要点

- 浅色主题对比度天然达标（正文 `#0F172A` on `#FFFFFF` ≈ 15:1），但**类别色**必须单独校验 ≥4.5:1（这是浅色方案最容易翻车处）。
- 长时间标注的眩光控制：`bg` 用蓝灰 `#F8FAFC` 而非纯白，降低刺眼。
- 高亮/选中态统一蓝色系（accent + accent-soft），避免红绿蓝混用造成信息噪音。
- 图表/数据突出用深色文字 + 单一强调色，克制用色（Swiss 原则：一屏一个主色）。
- 空状态友好：未标注图列表给「拖入图片开始标注」虚线占位。

### 6. 反模式（避免）

- 纯白 `#FFFFFF` 整屏（眩光）；霓虹色；大面积深色块破坏浅色一致性；把方案 A 的状态点配色直接搬过来（白底会发灰）。

### 7. PyQt5 实现路径

1. `theme.py` 新增 `PALETTE_LIGHT` + `STYLESHEET_LIGHT`；`AppConfig.theme` 增加 `"light"`；`apply_theme` 按配置选择，深色/浅色都做独立对比度测试。
2. 类别调色板（`utils/colors.py`）增加 `CATPPUCCIN_LIGHT`（深色变体 20 色），`ProjectConfig.get_class_color` 按主题取用——注意：**已存 label 的 class_colors 是 hex 字符串，浅色主题下需按亮度换算或直接使用新调色板**（迁移提示：仅影响显示色，不落盘）。
3. 画布描边逻辑：选中/未选中标注统一「主题色 + 白色外描边」双线，保证深浅主题都可读。
4. 状态徽章组件（`ui/badges.py`）在浅色下用「浅底深字描边」样式。
5. 训练曲线配色与网格颜色随主题切换（`train_panel.py` 读 PALETTE）。

---

## 三套方案对比与选择

| 维度 | A Pro Dark | B Glass | C Bright Minimal |
|---|---|---|---|
| 气质 | 专业/高效/硬核 | 优雅/AI 感/高级 | 清爽/友好/协作 |
| 目标用户 | 重度工程师、长时间训练 | 演示、宣传、AI 卖点 | 团队协作、新手、明亮环境 |
| 密度 | 高（8/10） | 标准（5/10） | 标准（5/10） |
| 动效 | 克制（3/10） | 中（5/10，光斑/AI 呼吸） | 极简（2/10） |
| 主色 | indigo `#5E6AD2` | violet `#7C3AED` | blue `#1E3A8A` |
| 类别色 | 亮系加深 | 高饱和紫青 | 深色系 20 色 |
| 主要风险 | 长时间暗色疲劳（少数人） | QSS 玻璃模拟质感上限、性能 | 类别色对比度、眩光 |
| 实现成本 | 低（在现有 Mocha 上进化） | 中（光斑动画 + 玻璃样式） | 中（浅色调色板迁移） |
| 推荐场景 | 默认专业版 | 宣传/演示版 | 团队/教学版 |

**建议落地顺序**：方案 A 成本最低、与现有 Catppuccin Mocha 改动最小，可作为默认主题升级；方案 C 作为第二主题（浅色）；方案 B 作为可选「演示皮肤」。三者共用 `AppConfig.theme` 枚举 + `theme.py` 令牌结构，QSS 与画布描边逻辑抽成共享组件，单主题切换即可热更新。

---

## 附：三套方案共享的 UI 基础设施改造

1. **语义令牌化**：`theme.py` 抽出 `TOKENS`（颜色/间距/圆角/字号），三套主题只是 TOKENS 的不同取值；`PALETTE` 变为派生值，消灭逐屏硬编码 hex。
2. **徽章组件** `ui/badges.py`：任务类型、确认状态、AI 来源、OK/NG、卡控结果的统一胶囊，随主题自动换样式。
3. **快捷键提示条** `ui/shortcut_bar.py`：可折叠底部条，读 `AppConfig` 记忆状态。
4. **状态图标库**：确认=勾、待确认=圆点、未标注=空心圆（`ui/icons.py` 新增 3 个 SVG），形状 + 颜色双通道。
5. **深浅主题切换**：`AppConfig.theme` 支持 `mocha | pro-dark | glass | light`；`apply_theme` 重建 QSS + 通知各面板重绘（画布、曲线、徽章）。

---

## 附：方案 B-蓝（深海蓝玻璃）已落地

2026-08 会话已按方案 B-蓝「深海蓝玻璃」完成 PyQt5 主题改造（保持程序主体与全部功能不变）：

**改动文件**
| 文件 | 改动 |
|---|---|
| `src/ui/theme.py` | PALETTE 换为深海蓝玻璃令牌（全 hex，QColor 安全）；STYLESHEET 重写为玻璃风（rgba 半透明面板 + 顶部高光边 + 大圆角 + qradialgradient 夜色渐变窗口底）；`apply_theme` 补全 QPalette（Window/Base/Text 等，兜底 QScrollArea viewport 等未 QSS 覆盖控件）；`_rgba()` hex→rgba 辅助；字体候选加入 PingFang SC |
| `src/utils/colors.py` | 类别调色板换成深海蓝主题 20 色高饱和亮色系（blue/cyan/violet/green/amber...，深底对比 ≥3:1） |
| `src/core/annotation.py` | `POSE_BOUNDING_BOX_RGB` 旧紫罗兰 → 深海蓝 `(59,130,246)` |
| `src/ui/icons.py` | 图标默认色 `#d8dee9` → `#A8C8E8`（冷白蓝） |
| `src/ui/preview_panel.py` | 未标注默认灰 `#6c7086` → `#5E7A9F`（2 处） |
| `src/ui/views/classify.py` | 同上默认灰（1 处） |
| `src/ui/model_panel.py` | 后端详情灰字（1 处） |
| `src/ui/train_panel.py` | 曲线配色：Train Loss → violet `#A78BFA`、Val Loss → teal `#22D3EE`（pyqtgraph 背景/前景沿用既有 `setConfigOptions` 深色配置） |

**验证**：venv 安装最小依赖（PyQt5/PyYAML/pyqtgraph/packaging），offscreen + cocoa 双平台冒烟——主题加载、全部 6 个 Tab（主页/标注/预览/训练/模型/小工具）构建、QColor 解析、玻璃半透明（rgba alpha 12-39）、夜色渐变、类别色标注框、主色/状态色渲染均验证通过；`py_compile` 全绿。真实截图存于 `ui_previews/real/`。

**说明**：QSS 无法实现真实 backdrop-blur，玻璃感以「半透明 rgba + 渐变底 + 高光边」模拟（与 HTML 原型的差异已在第 0 节说明）。已知未处理项（按用户要求暂不动）：`tag_widget.py` 两处乱码文本、`_sync_available_tags` 的 warning 级日志、训练进程/LA 等逻辑层问题。
