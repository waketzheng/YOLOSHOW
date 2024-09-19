import contextlib
import logging
import os
import sys
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

# 将ui目录添加到系统路径中
BASE_DIR = Path(__file__).parent.resolve()
MODEL_DIR = BASE_DIR / "ptfiles"
UI_DIR = MODEL_DIR / "ui"
sys.path.append(str(UI_DIR))
if not MODEL_DIR.exists():
    MODEL_DIR.mkdir()

with contextlib.suppress(TypeError):  # Leave it here to fix ruff E402
    from utils import glo
    from yoloshow.ChangeWindow import vs2yoloshow, yoloshow2vs
    from yoloshow.Window import YOLOSHOWVSWindow as yoloshowVSWindow
    from yoloshow.Window import YOLOSHOWWindow as yoloshowWindow


def run_app() -> None:
    app = QApplication([])  # 创建应用程序实例
    app.setWindowIcon(QIcon("images/swimmingliu.ico"))  # 设置应用程序图标

    # 为整个应用程序设置样式表，去除所有QFrame的边框
    app.setStyleSheet("QFrame { border: none; }")

    # 创建窗口实例
    yoloshow = yoloshowWindow()
    yoloshowvs = yoloshowVSWindow()

    # 初始化全局变量管理器，并设置值
    glo._init()  # 初始化全局变量空间
    glo.set_value("yoloshow", yoloshow)  # 存储yoloshow窗口实例
    glo.set_value("yoloshowvs", yoloshowvs)  # 存储yoloshowvs窗口实例

    # 从全局变量管理器中获取窗口实例
    yoloshow_glo = glo.get_value("yoloshow")
    yoloshowvs_glo = glo.get_value("yoloshowvs")

    # 显示yoloshow窗口
    yoloshow_glo.show()

    # 连接信号和槽，以实现界面之间的切换
    yoloshow_glo.ui.src_vsmode.clicked.connect(yoloshow2vs)  # 从单模式切换到对比模式
    yoloshowvs_glo.ui.src_singlemode.clicked.connect(vs2yoloshow)  # 从对比模式切换回单模式

    app.exec()  # 启动应用程序的事件循环


def main() -> None:
    with open(os.devnull, "w") as f:  # Use `with` syntax to fix SIM115
        sys.stdout = f  # 禁止标准输出
        logging.disable(logging.CRITICAL)  # 禁用所有级别的日志
        run_app()


if __name__ == "__main__":
    main()
