import sys
from PyQt6 import QtWidgets
from feda_tools.gui.threshold_adjustment import ThresholdAdjustmentWindow

class MainApplication(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('FEDA Tools GUI')
        self.init_ui()

    def init_ui(self):
        # Set ThresholdAdjustmentWindow as the central widget
        self.threshold_window = ThresholdAdjustmentWindow()
        self.setCentralWidget(self.threshold_window)
        self.show()

def main():
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainApplication()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
