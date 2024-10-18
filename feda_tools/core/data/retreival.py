import os
import tttrlib
import fnmatch

def get_ptu_files(directory):
    ptu_files = []
    for file in os.listdir(directory):
        if fnmatch.fnmatch(file, '*.ptu'):
            ptu_files.append(file)
    return ptu_files

def load_ptu_files(file_ptu, file_irf, file_bkg):
    data_ptu = tttrlib.TTTR(file_ptu, 'PTU')
    data_irf = tttrlib.TTTR(file_irf, 'PTU')
    data_bkg = tttrlib.TTTR(file_bkg, 'PTU')
    return data_ptu, data_irf, data_bkg