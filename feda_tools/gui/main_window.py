import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSlider, QGroupBox
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt

from dialogs.file_select import FileSelectDialog

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Example Window")
        self.setGeometry(100, 100, 800, 600)

        main_layout = QHBoxLayout()

        # Parameters section
        parameters_group = QGroupBox("Parameters")
        parameters_layout = QVBoxLayout()

        sigma_layout = QHBoxLayout()
        sigma_layout.addWidget(QLabel("Sigma:"))
        sigma_slider = QSlider(Qt.Orientation.Horizontal)
        sigma_layout.addWidget(sigma_slider)
        parameters_layout.addLayout(sigma_layout)

        burst_layout = QHBoxLayout()
        burst_layout.addWidget(QLabel("Burst Size:"))
        burst_slider = QSlider(Qt.Orientation.Horizontal)
        burst_layout.addWidget(burst_slider)
        parameters_layout.addLayout(burst_layout)

        window_layout = QHBoxLayout()
        window_layout.addWidget(QLabel("Data Window:"))
        window_slider = QSlider(Qt.Orientation.Horizontal)
        window_layout.addWidget(window_slider)
        parameters_layout.addLayout(window_layout)

        parameters_group.setLayout(parameters_layout)
        main_layout.addWidget(parameters_group)

        # Graph and button section
        right_layout = QVBoxLayout()

        # Matplotlib graph
        fig, ax = plt.subplots()
        canvas = FigureCanvas(fig)
        right_layout.addWidget(canvas)

        # Choose Files button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        choose_files_button = QPushButton("Choose Files")
        choose_files_button.clicked.connect(self.open_file_select_dialog)
        button_layout.addWidget(choose_files_button)
        right_layout.addLayout(button_layout)

        main_layout.addLayout(right_layout)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def open_file_select_dialog(self):
        self.file_select_dialog = FileSelectDialog()
        self.file_select_dialog.show()
