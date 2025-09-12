import sys
from PyQt6 import QtWidgets
from .view.dashboard import DashboardWidget
from .view.process_analysis import ProcessAnalysisWidget
from .view.state_handler import StateHandler
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

        # Initially only create the dashboard widget
        self.dashboard_widget = DashboardWidget(self.state_handler)
        self.stacked_widget.addWidget(self.dashboard_widget)
        
        # Initialize process widget as None
        self.process_widget = None

        # Connect dashboard widget directly to process analysis (skip fit23)
        self.dashboard_widget.next_button.clicked.connect(self.create_and_show_process_widget)
        self.stacked_widget.setCurrentWidget(self.dashboard_widget)
        self.show()

    def create_and_show_process_widget(self):
        if self.process_widget is None:
            self.process_widget = ProcessAnalysisWidget(self.state_handler)
            self.stacked_widget.addWidget(self.process_widget)
        self.show_process_widget()

    def show_process_widget(self):
        self.stacked_widget.setCurrentWidget(self.process_widget)

def main():
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainApplication()
    main_window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()