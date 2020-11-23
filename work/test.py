#!/usr/bin/python

import sys
from PyQt5.QtWidgets import (QListWidget, QWidget, QMessageBox, 
    QApplication, QVBoxLayout, QPushButton)


class Example(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):

        vbox = QVBoxLayout(self)

        self.button_load = QPushButton('Load')
        self.button_load.clicked.connect(self.load_clicked)
        vbox.addWidget(self.button_load)

        self.list_widget = QListWidget()
            
        #self.list_widget.addItem("A") 
        #self.list_widget.addItem("B")
        #self.list_widget.addItem("C")
            
        self.list_widget.itemClicked.connect(self.item_clicked)
        
        vbox.addWidget(self.list_widget)

        self.setLayout(vbox)
        self.show()

    def item_clicked(self, item):
        QMessageBox.information(self, "Info", item.text())

    def load_clicked(self):
        self.list_widget.addItem("D")
        self.list_widget.addItem("E")



def main():

    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
