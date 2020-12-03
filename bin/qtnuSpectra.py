#!/usr/bin/env python3

import numpy
import sys
import h5py

import matplotlib.pyplot as plt

import nuBall.tools as tools

from PyQt5.QtWidgets import (QWidget, QMainWindow, QApplication, QPushButton,
                    QLineEdit, QListWidget, QPlainTextEdit, QLabel, QComboBox, 
                    QGridLayout, QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('nuSpectra')

        self.data_file = None
        self.fit_range = [0, 0]
        self.peaks = []

        fig, axes = plt.subplots(1, 1, sharex='all')
        self.figure = fig
        self.axes = axes
        self.init_axes()

        self.data0, = self.axes.plot([0], [0], drawstyle='steps-mid', 
                color='red', label='')
        self.data1, = self.axes.plot([0], [0], drawstyle='steps-mid', 
                color='blue', label='')
        self.data2, = self.axes.plot([0], [0], drawstyle='steps-mid', 
                color='green', label='')
        self.dataf, = self.axes.plot([0], [0], ls='--', color='black')

        self.initUI()


    def initUI(self):

        main = QWidget(self)
        self.setCentralWidget(main)

        self.button_load = QPushButton('Load')
        self.button_load.setFixedWidth(100)
        self.button_load.clicked.connect(self.load_clicked)

        self.text_file = QLineEdit()
        self.text_file.setText('')
        self.text_file.setFixedWidth(400)
        self.text_file.setReadOnly(True)

        self.list_spectra = QListWidget()
        self.list_spectra.setFixedWidth(200)
        self.list_spectra.itemClicked.connect(self.item_clicked)

        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.figure.set_size_inches(10, 8)

        font_fit = QFont()
        font_fit.setPointSize(11)
        font_fit.setFamily("Mono")

        self.text_fit = QPlainTextEdit()
        self.text_fit.setPlainText('')
        self.text_fit.setFont(font_fit)
        self.text_fit.setReadOnly(True)

        self.label_range = QLabel()
        self.label_range.setText('Range: ')
        self.label_range.setFixedWidth(100)
        self.label_range.setAlignment(Qt.AlignRight)

        self.input_range = QLineEdit()
        self.input_range.setFixedWidth(150)
        self.input_range.setAlignment(Qt.AlignLeft)
        self.input_range.setText('')
        self.input_range.setToolTip('If empty uses current view range,\
 otherwise input two integers')

        self.label_spectrum = QLabel()
        self.label_spectrum.setText('Spectrum: ')
        self.label_spectrum.setFixedWidth(100)
        self.label_spectrum.setAlignment(Qt.AlignRight)

        self.label_peaks = QLabel()
        self.label_peaks.setText('Peaks: ')
        self.label_peaks.setFixedWidth(100)
        self.label_peaks.setAlignment(Qt.AlignRight)

        self.input_peaks = QLineEdit()
        self.input_peaks.setFixedWidth(150)
        self.input_peaks.setAlignment(Qt.AlignLeft)
        self.input_peaks.setText('')

        self.combo_spectrum = QComboBox()
        self.combo_spectrum.addItems(['Red', 'Blue', 'Green'])
        self.combo_spectrum.setCurrentText('Red')

        self.button_fit = QPushButton('Fit')
        self.button_fit.setFixedWidth(100)
        self.button_fit.clicked.connect(self.fit_clicked)

        layout = QGridLayout()
        layout.addWidget(self.button_load, 0, 0)
        layout.addWidget(self.text_file, 0, 1, 1, 3)

        layout.addWidget(self.list_spectra, 1, 0, 2, 1)
        layout.addWidget(self.canvas, 1, 1, 1, 3)
        layout.addWidget(self.toolbar, 2, 1, 1, 3)

        layout.addWidget(self.label_spectrum, 3, 0)
        layout.addWidget(self.combo_spectrum, 4, 0)

        layout.addWidget(self.text_fit, 3, 3, 3, 1)
        layout.addWidget(self.label_range, 3, 1)
        layout.addWidget(self.input_range, 3, 2)

        layout.addWidget(self.label_peaks, 4, 1)
        layout.addWidget(self.input_peaks, 4, 2)

        layout.addWidget(self.button_fit, 5, 2)

        main.setLayout(layout)

        #self.resize(1280, 960)


    def init_axes(self):
        self.axes.cla()
        self.axes.tick_params(axis='both', labelsize=12)
        self.axes.set_xlim(0, 1)
        self.axes.set_ylim(0, 1)
        self.axes.set_ylabel('Counts', size=14)
        self.axes.set_xlabel('E (keV)', size=14)

        self.figure.tight_layout()
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()


    def item_clicked(self, item):
        name = item.text()
        if name == 'Clear':
            data = [0]
            name = ''
        else:
            data = numpy.array(self.data_file[name])
    
        self.dataf.set_xdata([0])
        self.dataf.set_ydata([0])

        x = numpy.arange(len(data)) + 0.5
        if self.combo_spectrum.currentText() == 'Red':
            self.data0.set_xdata(x)
            self.data0.set_ydata(data)
            self.data0.set_label(name)
        elif self.combo_spectrum.currentText() == 'Blue':
            self.data1.set_xdata(x)
            self.data1.set_ydata(data)
            self.data1.set_label(name)
        elif self.combo_spectrum.currentText() == 'Green':
            self.data2.set_xdata(x)
            self.data2.set_ydata(data)
            self.data2.set_label(name)

        xmax0 = max(self.data0.get_xdata())
        xmax1 = max(self.data1.get_xdata())
        xmax2 = max(self.data2.get_xdata())
        self.axes.set_xlim(0, max(xmax0, xmax1, xmax2))
        ymax0 = max(self.data0.get_ydata())
        ymax1 = max(self.data1.get_ydata())
        ymax2 = max(self.data2.get_ydata())
        self.axes.set_ylim(0, max(ymax0, ymax1, ymax2) + 1)
        self.axes.legend()

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()


    def load_clicked(self):
        file_name = QFileDialog.getOpenFileName(self, "Select data file", "",
                                            "HDF5 Files (*.h5);;All Files (*)")
        if file_name:
            try:
                self.data_file = h5py.File(file_name[0], 'r')
            except OSError as err:
                error_dialog = QMessageBox()
                error_dialog.setIcon(QMessageBox.Critical)
                error_dialog.setText(
                        'Could not open file {}'.format(file_name[0]))
                error_dialog.setInformativeText(err.args[0])
                error_dialog.setWindowTitle("Error")
                error_dialog.exec_()
                return None

            self.text_file.setText(file_name[0])
            spectra = tools.list_spectra(self.data_file, verbose=False)
            self.list_spectra.clear()
            self.list_spectra.addItem('Clear')
            for s in sorted(spectra):
                self.list_spectra.addItem(s)


    def fit_clicked(self):
        range_str = self.input_range.text()
        if range_str == "":
            ax_range = self.axes.get_xlim()
            range_str = "{} {}".format(int(ax_range[0]), int(ax_range[1]) + 1)

        range_str = range_str.replace(",", " ").replace(";", " ").replace(":", " ")
        range_str = range_str.split()
        r = []
        try:
            for s in range_str:
                r.append(int(s))
        except ValueError:
            error_dialog = QMessageBox()
            error_dialog.setIcon(QMessageBox.Warning)
            error_dialog.setText('Could not convert {} to int'.format(s))
            error_dialog.setInformativeText('Range should be given as two integers separated by space, comma or semicolon e.g 300 600')
            error_dialog.setWindowTitle("Error")
            error_dialog.exec_()
            return None

        if len(r) > 1 and r[1] > r[0]:
            self.fit_range[0] = r[0]
            self.fit_range[1] = r[1]
        else:
            error_dialog = QMessageBox()
            error_dialog.setIcon(QMessageBox.Warning)
            error_dialog.setText('Wrong range {}'.format(r))
            error_dialog.setInformativeText('Range should be given as two integers separated by space, comma or semicolon e.g 300 600')
            error_dialog.setWindowTitle("Error")
            error_dialog.exec_()
            return None

        peaks_str = self.input_peaks.text()
        peaks_str = peaks_str.replace(",", " ").replace(";", " ").replace(":", " ")
        peaks_str = peaks_str.split()
        p = []
        try:
            for s in peaks_str:
                p.append(float(s))
        except ValueError:
            error_dialog = QMessageBox()
            error_dialog.setIcon(QMessageBox.Warning)
            error_dialog.setText('Could not convert {} to float'.format(s))
            error_dialog.setInformativeText('Peaks should be given as floats separated by space, comma or semicolon e.g 253, 312, 408')
            error_dialog.setWindowTitle("Error")
            error_dialog.exec_()
            return None

        self.peaks = []
        for pi in p:
            if r[0] < pi < r[1]:
                self.peaks.append(pi)
        if len(self.peaks) < 1:
            error_dialog = QMessageBox()
            error_dialog.setIcon(QMessageBox.Warning)
            error_dialog.setText('No peaks defined within range')
            error_dialog.setInformativeText('Peaks should be given as floats separated by space, comma or semicolon e.g 253, 312, 408')
            error_dialog.setWindowTitle("Error")
            error_dialog.exec_()
            return None
        
        text = ''
        if self.combo_spectrum.currentText() == 'Red':
            pars, dp = tools.fit(self.data0.get_xdata(), 
                                self.data0.get_ydata(), 
                                self.fit_range, self.peaks, verbose=False)
            text += 'Red - {}:\n'.format(self.data0.get_label())
        elif self.combo_spectrum.currentText() == 'Blue':
            pars, dp = tools.fit(self.data1.get_xdata(), 
                                self.data1.get_ydata(), 
                                self.fit_range, self.peaks, verbose=False)
            text += 'Blue - {}:\n'.format(self.data1.get_label())
        elif self.combo_spectrum.currentText() == 'Green':
            pars, dp = tools.fit(self.data2.get_xdata(), 
                                self.data2.get_ydata(), 
                                self.fit_range, self.peaks, verbose=False)
            text += 'Green - {}:\n'.format(self.data2.get_label())
        xf = numpy.arange(self.fit_range[0], self.fit_range[1])
        yf = tools.peaks_function(xf, *pars)
        self.dataf.set_xdata(xf)
        self.dataf.set_ydata(yf)
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        n = len(pars)
        for i in range(n):
            if i == 0:
                text += 'a '
            elif i == 1:
                text += 'b '
            else:
                j = int((i - 2) // 3)
                k = int((i - 2) % 3)
                if k % 3 == 0:
                    text += 'x{} '.format(j)
                elif k % 3 == 1:
                    text += 'A{} '.format(j)
                else:
                    text += 's{} '.format(j)
            text += '\t{:.2f}\t {:.2f}\n'.format(pars[i], dp[i])
            self.text_fit.setPlainText(text)




if __name__ == '__main__':
    app = QApplication(sys.argv)

    main = Window()
    main.show()

    sys.exit(app.exec_())
