#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 15:34:16 2024

@author: frankie
"""

from sys import executable, argv
from subprocess import check_output
from PyQt5.QtWidgets import QFileDialog, QApplication

def gui_fname(directory='./'):
    """Open a file dialog, starting in the given directory, and return
    the chosen filenames"""
    # Run this exact file in a separate process, and grab the result
    files = check_output([executable, __file__, directory])
    return files.strip().decode().split('\n')

if __name__ == "__main__":
    directory = argv[1]
    app = QApplication([directory])
    fnames, _ = QFileDialog.getOpenFileNames(None, "Select files...", 
                                             directory, filter="All files (*)")
    # Print each file on a new line
    for fname in fnames:
        print(fname)
