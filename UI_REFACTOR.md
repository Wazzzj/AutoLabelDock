# AutoLabelDock UI Refactor Rules

## 1. Current Refactor Scope

当前第一阶段只重构两个页面：

### Training Page

HTML 原型：

http://127.0.0.1:9001/a-train.html

对应 AutoLabelDock 中的训练相关页面。

---

### Models Page

HTML 原型：

http://127.0.0.1:9001/a-models.html

对应 AutoLabelDock 中的模型管理相关页面。

---

当前阶段不要主动重构其他页面。

除非公共组件必须调整，否则不要扩展修改范围。

---

# 2. Source of Truth

这两个 HTML 页面是对应程序页面的唯一视觉标准。

当前程序 UI 不是视觉设计标准。

本任务属于：

UI 重构

而不是：

UI 微调 / 局部美化

修改优先级：

视觉一致性 > 保留旧 UI 样式 > 最小修改

如果现有页面结构阻碍原型还原，可以重构对应 UI 层。

不要为了减少代码改动而保留与 HTML 原型明显不一致的布局。

---

# 3. Business Logic

UI 重构必须尽量保留现有业务功能。

不得无理由修改：

- 模型训练逻辑
- 模型加载逻辑
- 模型管理逻辑
- 文件处理逻辑
- 数据处理逻辑
- 网络逻辑
- 推理逻辑
- 后台线程
- 原有信号槽逻辑

如果必须调整 UI 控件结构，可以重新连接对应事件，但必须保持原有功能。

原则：

重构 UI，保留业务逻辑。

---

# 4. Required Workflow

开始修改代码前必须先完成分析。

## Step 1：分析两个 HTML 原型

完整分析：

- 页面整体 Layout
- Sidebar
- Header
- Toolbar
- Content Area
- Card
- Panel
- Button
- Input
- Select
- Table
- Tab
- Progress
- Status
- Icon
- Typography
- Background
- Border
- Radius
- Shadow
- Spacing
- Alignment
- Module Size
- Module Position

不要只分析局部组件。

---

## Step 2：分析现有程序

找到：

训练页面对应代码。

模型页面对应代码。

同时找到两页使用的公共 UI 代码，例如：

- MainWindow
- Sidebar
- Header
- Navigation
- Theme
- QSS / CSS
- 公共组件

修改代码前先明确：

原型页面和当前程序页面有哪些结构差异。

---

# 5. Refactor Order

必须按照以下顺序执行：

## 第一阶段：训练页面

目标：

http://127.0.0.1:9001/a-train.html

先完成训练页面整体重构。

顺序：

1. 页面整体 Layout
2. Sidebar / Header
3. 页面主要区域
4. Card 和 Panel
5. 表单控件
6. 按钮
7. Typography
8. Color
9. Spacing
10. Border / Radius / Shadow
11. Icon
12. 微小视觉调整

第一轮修改之后不能宣布完成。

必须运行程序并检查实际效果。

如果实际页面与 a-train.html 仍然存在明显差异，继续修改。

直到训练页面基本完成后，再进入模型页面。

---

## 第二阶段：模型页面

目标：

http://127.0.0.1:9001/a-models.html

重复同样流程：

修改

→ 运行

→ 截图

→ 对比

→ 找差异

→ 再修改

模型页面基本完成后才能进入最终检查。

---

## 第三阶段：两页统一检查

检查：

- Sidebar 是否一致
- Header 是否一致
- 页面背景是否一致
- 字体系统是否一致
- Button 是否一致
- Input 是否一致
- Card 是否一致
- Radius 是否一致
- Spacing 是否一致
- Icon 是否一致
- Navigation 状态是否一致

公共 UI 应优先通过公共组件实现，而不是在两个页面重复写两套不同样式。

---

# 6. Screenshot Standard

HTML 原型和程序页面统一使用固定尺寸进行比较。

标准尺寸：

1440 × 900

HTML 原型截图：

ui_reference/prototype/train.png

ui_reference/prototype/models.png

程序实际截图：

ui_reference/current/train.png

ui_reference/current/models.png

差异截图：

ui_reference/diff/train_diff.png

ui_reference/diff/models_diff.png

---

# 7. Visual Verification Loop

每个页面必须执行：

修改代码

→ 实际运行程序

→ 打开对应页面

→ 截图

→ 与 HTML 原型截图比较

→ 找出差异

→ 继续修改

→ 再次运行

→ 再次截图

→ 再次比较

只要还有明显差异，就继续。

禁止在第一轮代码修改后直接结束任务。

禁止每修改一次就询问用户：

“是否继续？”

直接继续执行。

---

# 8. Completion Criteria

训练页面只有满足以下条件后才能认为基本完成：

- 页面整体结构基本一致
- 主要模块位置基本一致
- 模块尺寸基本一致
- Sidebar 基本一致
- Header 基本一致
- Button 基本一致
- Input / Select 基本一致
- 字号基本一致
- 字重基本一致
- 背景颜色基本一致
- Card 样式基本一致
- padding 基本一致
- margin 基本一致
- gap 基本一致
- Border 基本一致
- Radius 基本一致
- Shadow 基本一致
- 不存在明显旧 UI 风格残留

模型页面使用相同标准。

---

# 9. Do Not Stop Early

以下情况不能认为任务完成：

- 只修改颜色
- 只修改按钮
- 只修改 Sidebar
- 只修改 Header
- 只修改几个 margin
- 只重构一个 Card
- 训练页面完成后没有继续模型页面
- 页面明显与原型不同
- 修改代码后没有运行程序
- 没有进行视觉验证

这是两个完整页面的重构任务。

---

# 10. Final Report

只有训练页面和模型页面都完成后再统一汇报。

最终报告：

1. 修改了哪些文件
2. 训练页面进行了哪些重构
3. 模型页面进行了哪些重构
4. 修改了哪些公共组件
5. 如何进行实际验证
6. 仍然存在什么视觉差异
7. 为什么这些差异暂时无法继续缩小
8. 是否修改了业务逻辑

不要把每轮局部修改当成最终结果。