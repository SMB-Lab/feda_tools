import sys
from PyQt6 import QtWidgets
from .threshold_adjustment import ThresholdAdjustmentWidget
from .fit23_preview import Fit23PreviewWidget
from .process_analysis import ProcessAnalysisWidget
from .state_handler import StateHandler
from importlib.resources import files

class MainApplication(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('FEDA Tools GUI')
        self.state_handler = StateHandler(config_file=str(files('feda_tools.gui').joinpath('config.yaml')))
        self.init_ui()

    def init_ui(self):
        self.stacked_widget = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        # Initially only create the threshold widget
        self.threshold_widget = ThresholdAdjustmentWidget(self.state_handler)
        self.stacked_widget.addWidget(self.threshold_widget)
        
        # Initialize other widgets as None
        self.fit23_widget = None
        self.process_widget = None

        # Connect threshold widget to a method that creates fit23 widget
        self.threshold_widget.next_button.clicked.connect(self.create_and_show_fit23_widget)
        self.stacked_widget.setCurrentWidget(self.threshold_widget)
        self.show()

    def create_and_show_fit23_widget(self):
        if self.fit23_widget is None:
            self.fit23_widget = Fit23PreviewWidget(self.state_handler)
            self.stacked_widget.addWidget(self.fit23_widget)
            # Connect fit23 widget to a method that creates process widget
            self.fit23_widget.next_button.clicked.connect(self.create_and_show_process_widget)
        self.show_fit23_widget()

    def create_and_show_process_widget(self):
        if self.process_widget is None:
            self.process_widget = ProcessAnalysisWidget(self.state_handler)
            self.stacked_widget.addWidget(self.process_widget)
        self.show_process_widget()

    def show_fit23_widget(self):
        self.stacked_widget.setCurrentWidget(self.fit23_widget)

    def show_process_widget(self):
        self.stacked_widget.setCurrentWidget(self.process_widget)

def main():
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainApplication()
    main_window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()