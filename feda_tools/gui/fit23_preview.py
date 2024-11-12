from PyQt6 import QtWidgets, QtCore
from .widgets import PlotWidget
from feda_tools.core import model
import numpy as np

class Fit23PreviewWidget(QtWidgets.QWidget):
    def __init__(self, state_handler):
        super().__init__()
        self.state_handler = state_handler
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Fit23 Preview')
        layout = QtWidgets.QVBoxLayout()

        # PyQtGraph plot widget
        self.plot_widget = PlotWidget()
        layout.addWidget(self.plot_widget)

        # Fit23 Parameters Input
        params_layout = QtWidgets.QFormLayout()
        
        self.tau_input = QtWidgets.QDoubleSpinBox()
        self.tau_input.setRange(0.0, 100.0)
        self.tau_input.setDecimals(4)
        self.tau_input.setValue(self.state_handler.config.get('initial_tau', 3.03))
        self.tau_input.valueChanged.connect(self.update_preview)
        
        self.gamma_input = QtWidgets.QDoubleSpinBox()
        self.gamma_input.setRange(0.0, 100.0)
        self.gamma_input.setDecimals(4)
        self.gamma_input.setValue(self.state_handler.config.get('initial_gamma', 0.02))
        self.gamma_input.valueChanged.connect(self.update_preview)
        
        self.r0_input = QtWidgets.QDoubleSpinBox()
        self.r0_input.setRange(0.0, 100.0)
        self.r0_input.setDecimals(4)
        self.r0_input.setValue(self.state_handler.config.get('initial_r0', 0.38))
        self.r0_input.valueChanged.connect(self.update_preview)
        
        self.rho_input = QtWidgets.QDoubleSpinBox()
        self.rho_input.setRange(0.0, 100.0)
        self.rho_input.setDecimals(4)
        self.rho_input.setValue(self.state_handler.config.get('initial_rho', 1.64))
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

        # Next button
        self.next_button = QtWidgets.QPushButton('Next')
        layout.addWidget(self.next_button)

        self.setLayout(layout)
        self.plot_fit23(initial_plot=True)

    def plot_fit23(self, initial_plot=False):
        num_bins = self.state_handler.config.get('num_bins', 128)
        all_micro_times = self.state_handler.data_ptu.micro_times
        counts, _ = np.histogram(all_micro_times, bins=num_bins)

        counts_irf, _ = np.histogram(self.state_handler.data_irf.micro_times, bins=num_bins)
        counts_irf_nb = counts_irf.copy()
        counts_irf_nb[0:3] = 0
        counts_irf_nb[10:66] = 0
        counts_irf_nb[74:128] = 0

        macro_res = self.state_handler.data_ptu.get_header().macro_time_resolution
        g_factor = self.state_handler.config.get('g_factor', 1.04)
        l1_japan_corr = self.state_handler.config.get('l1_japan_corr', 0.0308)
        l2_japan_corr = self.state_handler.config.get('l2_japan_corr', 0.0368)

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

        x0 = self.fit_params
        fixed = np.array([0, 0, 1, 0])
        try:
            norm_counts = counts / np.max(counts)
            r2 = self.fit23_model(data=norm_counts, initial_values=x0, fixed=fixed, include_model=True)
            self.fit_params = r2['x'][:4]
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, 'Fit23 Error', f'An error occurred during Fit23:\n{e}')
            return

        self.state_handler.fit_params = self.fit_params

        self.plot_widget.clear()
        x = np.arange(len(counts))
        self.plot_widget.plot(x, norm_counts, pen='b', symbol='o', symbolSize=4, name='Data')
        self.plot_widget.plot(x, counts_irf / np.max(counts_irf), pen='orange', symbol='t', symbolSize=4, name='IRF')
        self.plot_widget.plot(x, self.fit23_model.model / np.max(self.fit23_model.model), pen='g', name='Model')

        self.plot_widget.setLogMode(y=True)
        self.plot_widget.setYRange(np.log10(0.001), np.log10(1))
        self.plot_widget.setLabel('left', 'log(Counts)')
        self.plot_widget.setLabel('bottom', 'Channel Number')
        self.plot_widget.addLegend(offset=(30,30))

    def update_preview(self):
        self.plot_fit23()

    def run_fit23(self):
        self.plot_fit23()
