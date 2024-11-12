import sys
from PyQt6 import QtWidgets
from threshold_adjustment import ThresholdAdjustmentWidget
from fit23_preview import Fit23PreviewWidget
from process_analysis import ProcessAnalysisWidget
from state_handler import StateHandler

class MainApplication(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('FEDA Tools GUI')
        self.state_handler = StateHandler()
        self.init_ui()

    def init_ui(self):
        self.stacked_widget = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.threshold_widget = ThresholdAdjustmentWidget(self.state_handler)
        self.fit23_widget = Fit23PreviewWidget(self.state_handler)
        self.process_widget = ProcessAnalysisWidget(self.state_handler)

        self.stacked_widget.addWidget(self.threshold_widget)
        self.stacked_widget.addWidget(self.fit23_widget)
        self.stacked_widget.addWidget(self.process_widget)

        self.threshold_widget.next_button.clicked.connect(self.show_fit23_widget)
        self.fit23_widget.next_button.clicked.connect(self.show_process_widget)

        self.stacked_widget.setCurrentWidget(self.threshold_widget)
        self.show()

    def show_fit23_widget(self):
        self.stacked_widget.setCurrentWidget(self.fit23_widget)

    def show_process_widget(self):
        self.stacked_widget.setCurrentWidget(self.process_widget)

def main():
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainApplication()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
