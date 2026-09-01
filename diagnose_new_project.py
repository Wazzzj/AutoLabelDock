"""诊断"新建项目没有反应"的最小脚本。

在你的环境运行：
    python diagnose_new_project.py
把终端输出完整贴给开发者。
"""
import sys
import traceback

sys.path.insert(0, ".")

from PyQt5.QtCore import QT_VERSION_STR, PYQT_VERSION_STR, QTimer
from PyQt5.QtWidgets import QApplication

from src.ui import theme

print("Python:", sys.version.split()[0])
print("Qt:", QT_VERSION_STR, "| PyQt5:", PYQT_VERSION_STR)
print("platform:", sys.platform)

app = QApplication([])
theme.apply_theme(app)
print("theme.apply_theme OK")

try:
    from src.ui.dialogs import NewProjectDialog

    dlg = NewProjectDialog()
    print("NewProjectDialog 构造 OK")

    dlg.show()
    app.processEvents()
    print("show() OK | visible =", dlg.isVisible(), "| size =", dlg.size().width(), "x", dlg.size().height())

    # 模拟用户点"确定"前先看看对话框是否真的可见：抓一张图统计主色
    pm = dlg.grab().toImage()
    from collections import Counter

    cnt = Counter()
    for y in range(0, pm.height(), 4):
        for x in range(0, pm.width(), 4):
            c = pm.pixelColor(x, y)
            cnt[(c.red(), c.green(), c.blue())] += 1
    top = cnt.most_common(3)
    print("对话框渲染主色 Top3:", [f"#{r:02X}{g:02X}{b:02X}({v})" for (r, g, b), v in top])

    QTimer.singleShot(1500, dlg.reject)  # 1.5s 后自动关闭
    code = dlg.exec_()
    print("exec_() 返回:", code, "（0=取消/未弹出，1=确定）")
except Exception:
    print("!!! 出现异常：")
    traceback.print_exc()

print("诊断结束")
