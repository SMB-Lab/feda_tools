import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                             QWidget, QFileDialog, QLineEdit, QLabel)

class FileSelectDialog(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("File Paths")
        self.setGeometry(150, 150, 400, 200)

        layout = QVBoxLayout()

        self.ptu_input = QLineEdit()
        self.irf_input = QLineEdit()
        self.bkg_input = QLineEdit()

        layout.addWidget(QLabel("PTU:"))
        layout.addWidget(self.ptu_input)
        ptu_button = QPushButton("Select PTU")
        ptu_button.clicked.connect(lambda: self.open_file_dialog(self.ptu_input, "PTU Files (*.ptu)"))
        layout.addWidget(ptu_button)

        layout.addWidget(QLabel("IRF:"))
        layout.addWidget(self.irf_input)
        irf_button = QPushButton("Select IRF")
        irf_button.clicked.connect(lambda: self.open_file_dialog(self.irf_input, "IRF Files (*.irf)"))
        layout.addWidget(irf_button)

        layout.addWidget(QLabel("BKG:"))
        layout.addWidget(self.bkg_input)
        bkg_button = QPushButton("Select BKG")
        bkg_button.clicked.connect(lambda: self.open_file_dialog(self.bkg_input, "BKG Files (*.bkg)"))
        layout.addWidget(bkg_button)

        submit_button = QPushButton("Submit")
        submit_button.clicked.connect(self.submit)
        layout.addWidget(submit_button)

        self.setLayout(layout)
    
    def submit(self):
        ptu_path = self.ptu_input.text()
        irf_path = self.irf_input.text()
        bkg_path = self.bkg_input.text()
        print(f"PTU: {ptu_path}\nIRF: {irf_path}\nBKG: {bkg_path}")
    
    def open_file_dialog(self, line_edit, file_filter):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select File", "", file_filter)
        if file_name:
            line_edit.setText(file_name)