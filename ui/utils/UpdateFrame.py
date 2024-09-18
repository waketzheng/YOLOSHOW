from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame


class DoubleClickQFrame(QFrame):
    doubleClickFrame = Signal()

    def __init__(self, QFrame):
        super(DoubleClickQFrame, self).__init__(QFrame)

    def mouseDoubleClickEvent(self, event):
        self.doubleClickFrame.emit()
