import sys
from PyQt6.QtWidgets import QApplication

# Import the main window from the sibling module
try:
    from .main_window import OutlierMainWindow
except ImportError:
    # Fallback for running script directly during development (less ideal)
    from main_window import OutlierMainWindow

def main():
    """Launches the Outlier Detection GUI application."""
    app = QApplication(sys.argv)
    main_window = OutlierMainWindow()
    main_window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()