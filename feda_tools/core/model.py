import tttrlib
import numpy as np

def setup_fit23(num_bins, macro_res, counts_irf_nb, g_factor, l1_japan_corr, l2_japan_corr):
    dt = 25000/num_bins/1000
    period = 1/(macro_res*np.power(10, 6))
    fit23 = tttrlib.Fit23(
        dt=dt,
        irf=counts_irf_nb,
        background=np.ones_like(counts_irf_nb)*0.002,
        period=period,
        g_factor=g_factor,
        l1=l1_japan_corr,
        l2=l2_japan_corr,
        convolution_stop=10,
        p2s_twoIstar_flag=True
    )
    return fit23