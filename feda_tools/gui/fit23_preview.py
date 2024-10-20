from PyQt6 import QtWidgets, QtCore
from .widgets import MatplotlibCanvas
from feda_tools.core import model
import numpy as np

class Fit23PreviewWindow(QtWidgets.QWidget):
    analysis_started = QtCore.pyqtSignal()

    def __init__(self, data_ptu, data_irf, burst_index, chunk_size, output_directory):
        super().__init__()
        self.data_ptu = data_ptu
        self.data_irf = data_irf
        self.burst_index = burst_index
        self.chunk_size = chunk_size
        self.output_directory = output_directory  # Save the output directory

        self.init_ui()
        self.thread = None
        self.worker = None

    def init_ui(self):
        self.setWindowTitle('Fit23 Preview')
        layout = QtWidgets.QVBoxLayout()

        # Matplotlib canvas
        self.canvas = MatplotlibCanvas(self, width=5, height=4, dpi=100)
        layout.addWidget(self.canvas)

        # Fit23 Parameters Input
        params_layout = QtWidgets.QFormLayout()
        
        self.tau_input = QtWidgets.QDoubleSpinBox()
        self.tau_input.setRange(0.0, 100.0)
        self.tau_input.setDecimals(4)
        self.tau_input.setValue(3.03)  # Default value
        self.tau_input.valueChanged.connect(self.update_preview)
        
        self.gamma_input = QtWidgets.QDoubleSpinBox()
        self.gamma_input.setRange(0.0, 100.0)
        self.gamma_input.setDecimals(4)
        self.gamma_input.setValue(0.02)  # Default value
        self.gamma_input.valueChanged.connect(self.update_preview)
        
        self.r0_input = QtWidgets.QDoubleSpinBox()
        self.r0_input.setRange(0.0, 100.0)
        self.r0_input.setDecimals(4)
        self.r0_input.setValue(0.38)  # Default value
        self.r0_input.valueChanged.connect(self.update_preview)
        
        self.rho_input = QtWidgets.QDoubleSpinBox()
        self.rho_input.setRange(0.0, 100.0)
        self.rho_input.setDecimals(4)
        self.rho_input.setValue(1.64)  # Default value
        self.rho_input.valueChanged.connect(self.update_preview)

        params_layout.addRow('Initial Tau:', self.tau_input)
        params_layout.addRow('Initial Gamma:', self.gamma_input)
        params_layout.addRow('Initial r0:', self.r0_input)
        params_layout.addRow('Initial rho:', self.rho_input)

        layout.addLayout(params_layout)

        # Run Fit23 Button
        run_fit_button = QtWidgets.QPushButton('Run Fit23 with Current Parameters')
        run_fit_button.clicked.connect(self.run_fit23)
        layout.addWidget(run_fit_button)

        # Run Analysis button
        self.run_button = QtWidgets.QPushButton('Run Full Analysis')
        layout.addWidget(self.run_button)
        self.run_button.clicked.connect(self.start_analysis)

        self.setLayout(layout)
        self.plot_fit23(initial_plot=True)

    def plot_fit23(self, initial_plot=False):
        # Prepare data for the entire chunk
        num_bins = 128

        # Get micro_times from data_ptu
        all_micro_times = self.data_ptu.micro_times

        # Create histogram counts for the chunk
        counts, _ = np.histogram(all_micro_times, bins=num_bins)

        # Prepare counts_irf from data_irf
        counts_irf, _ = np.histogram(self.data_irf.micro_times, bins=num_bins)
        counts_irf_nb = counts_irf.copy()
        counts_irf_nb[0:3] = 0
        counts_irf_nb[10:66] = 0
        counts_irf_nb[74:128] = 0

        # Setup fit23 with default or provided parameters
        macro_res = self.data_ptu.get_header().macro_time_resolution
        g_factor = 1.04
        l1_japan_corr = 0.0308
        l2_japan_corr = 0.0368

        if initial_plot:
            initial_fit_params = np.array([
                self.tau_input.value(),
                self.gamma_input.value(),
                self.r0_input.value(),
                self.rho_input.value()
            ])
            self.fit_params = initial_fit_params.copy()
        else:
            self.fit_params = np.array([
                self.tau_input.value(),
                self.gamma_input.value(),
                self.r0_input.value(),
                self.rho_input.value()
            ])


        self.fit23_model = model.setup_fit23(
            num_bins, macro_res, counts_irf_nb, g_factor, l1_japan_corr, l2_japan_corr
        )

        # Perform fitting
        x0 = self.fit_params  # Use current fit_params
        fixed = np.array([0, 0, 1, 0])
        try:
            r2 = self.fit23_model(data=counts, initial_values=x0, fixed=fixed, include_model=True)
            self.fit_params = r2['x'][:4]
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, 'Fit23 Error', f'An error occurred during Fit23:\n{e}')
            return

        # Plot data and model
        self.canvas.axes.clear()
        self.canvas.axes.semilogy(counts / np.max(counts), label='Data')
        self.canvas.axes.semilogy(counts_irf / np.max(counts_irf), label='IRF')
        self.canvas.axes.semilogy(self.fit23_model.model / np.max(self.fit23_model.model), label='Model')

        # Set y-axis limits
        self.canvas.axes.set_ylim(0.001, 1)

        # Set labels and legend
        self.canvas.axes.set_ylabel('log(Counts)')
        self.canvas.axes.set_xlabel('Channel Number')
        self.canvas.axes.legend()

        # Draw the canvas
        self.canvas.draw()
    
    def update_preview(self):
        self.plot_fit23()

    def run_fit23(self):
        # Re-plot Fit23 with user-defined initial parameters
        self.plot_fit23()

    def start_analysis(self):
        # Proceed to full analysis using the latest fit_params
        from .process_analysis import ProcessAnalysisWindow

        self.process_window = ProcessAnalysisWindow(
            self.data_ptu, self.data_irf, self.burst_index, self.chunk_size, self.fit_params, self.output_directory
        )
        self.process_window.show()
        self.analysis_started.emit()
        self.close()

