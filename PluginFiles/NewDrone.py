# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'NewDrone.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(339, 230)
        self.layoutWidget = QtWidgets.QWidget(Dialog)
        self.layoutWidget.setGeometry(QtCore.QRect(40, 20, 261, 22))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label = QtWidgets.QLabel(self.layoutWidget)
        self.label.setObjectName("label")
        self.horizontalLayout_2.addWidget(self.label)
        self.i_name = QtWidgets.QLineEdit(self.layoutWidget)
        self.i_name.setObjectName("i_name")
        self.horizontalLayout_2.addWidget(self.i_name)
        self.layoutWidget1 = QtWidgets.QWidget(Dialog)
        self.layoutWidget1.setGeometry(QtCore.QRect(60, 60, 211, 22))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.layoutWidget1)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_2 = QtWidgets.QLabel(self.layoutWidget1)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_3.addWidget(self.label_2)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem)
        self.sb_altitude = QtWidgets.QDoubleSpinBox(self.layoutWidget1)
        self.sb_altitude.setMaximum(10000.0)
        self.sb_altitude.setObjectName("sb_altitude")
        self.horizontalLayout_3.addWidget(self.sb_altitude)
        self.layoutWidget2 = QtWidgets.QWidget(Dialog)
        self.layoutWidget2.setGeometry(QtCore.QRect(60, 100, 211, 22))
        self.layoutWidget2.setObjectName("layoutWidget2")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.layoutWidget2)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_3 = QtWidgets.QLabel(self.layoutWidget2)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_4.addWidget(self.label_3)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem1)
        self.sb_speed = QtWidgets.QDoubleSpinBox(self.layoutWidget2)
        self.sb_speed.setMaximum(1000.0)
        self.sb_speed.setObjectName("sb_speed")
        self.horizontalLayout_4.addWidget(self.sb_speed)
        self.layoutWidget3 = QtWidgets.QWidget(Dialog)
        self.layoutWidget3.setGeometry(QtCore.QRect(60, 140, 211, 22))
        self.layoutWidget3.setObjectName("layoutWidget3")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.layoutWidget3)
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_4 = QtWidgets.QLabel(self.layoutWidget3)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_5.addWidget(self.label_4)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem2)
        self.sb_battery = QtWidgets.QDoubleSpinBox(self.layoutWidget3)
        self.sb_battery.setMaximum(10000.0)
        self.sb_battery.setObjectName("sb_battery")
        self.horizontalLayout_5.addWidget(self.sb_battery)
        self.pb_okdrone = QtWidgets.QPushButton(Dialog)
        self.pb_okdrone.setGeometry(QtCore.QRect(90, 190, 75, 23))
        self.pb_okdrone.setObjectName("pb_okdrone")
        self.pb_close = QtWidgets.QPushButton(Dialog)
        self.pb_close.setGeometry(QtCore.QRect(180, 190, 75, 23))
        self.pb_close.setObjectName("pb_close")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Drone"))
        self.label.setText(_translate("Dialog", "Drone Name:"))
        self.label_2.setText(_translate("Dialog", "Max. Altitude [m]:"))
        self.label_3.setText(_translate("Dialog", "UAS Max. Speed [km/h:]"))
        self.label_4.setText(_translate("Dialog", "Battery Duration [min]:"))
        self.pb_okdrone.setText(_translate("Dialog", "OK"))
        self.pb_close.setText(_translate("Dialog", "Cancel"))

