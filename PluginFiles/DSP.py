# -*- coding: utf-8 -*-
"""
/***************************************************************************
 DroneSurveyingPlanning
                                 A QGIS plugin
 This plugin allows users to plan an aerial survey.
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                              -------------------
        begin                : 2020-08-21
        git sha              : $Format:%H$
        copyright            : (C) 2020 by Politecnico di Milano
        email                : marta1.rossi@mail.polimi.it
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""
import json
import os.path


from qgis.PyQt import QtWidgets
from qgis.PyQt.QtCore import QSettings, QTranslator, QCoreApplication
from qgis.PyQt.QtGui import *
from qgis.PyQt.QtWidgets import QAction, QFileDialog
from qgis.core import *

from .DSP_HELP import Ui_Dialog as DSPHELP
# Import the code for the dialog
from .DSP_dialog import DroneSurveyingPlanningDialog
from .NewDrone import Ui_Dialog as drone
from .NewSensor import Ui_Dialog as sensor



#define new dialog windows
class DroneDialog(QtWidgets.QDialog):
    def __init__(self, parent):
        super(DroneDialog, self).__init__(parent)
        self.ui = drone()
        self.ui.setupUi(self)

class SensorDialog(QtWidgets.QDialog):
    def __init__(self, parent):
        super(SensorDialog, self).__init__(parent)
        self.ui = sensor()
        self.ui.setupUi(self)

class DSPHELPDialog(QtWidgets.QDialog):
    def __init__(self, parent):
        super(DSPHELPDialog, self).__init__(parent)
        self.ui = DSPHELP()
        self.ui.setupUi(self)


# Initialize Qt resources from file resources.py
class DroneSurveyingPlanning:
    """QGIS Plugin Implementation."""

    def __init__(self, iface):
        """Constructor.

        :param iface: An interface instance that will be passed to this class
            which provides the hook by which you can manipulate the QGIS
            application at run time.
        :type iface: QgsInterface
        """
        # Save reference to the QGIS interface
        self.iface = iface
        # initialize plugin directory
        self.plugin_dir = os.path.dirname(__file__)
        # initialize locale
        locale = QSettings().value('locale/userLocale')[0:2]
        locale_path = os.path.join(
            self.plugin_dir,
            'i18n',
            'DroneSurveyingPlanning_{}.qm'.format(locale))

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)
            QCoreApplication.installTranslator(self.translator)

        #
        drone_file = open(os.path.join(self.plugin_dir, 'listadroni.json'))
        self.drone_list = json.load(drone_file)['DroneList']

        sensor_file = open(os.path.join(self.plugin_dir, 'listasensori.json'))
        self.sensor_list = json.load(sensor_file)['SensorList']


        # Create the dialog (after translation) and keep reference
        self.dlg = DroneSurveyingPlanningDialog()


        # Declare instance attributes
        self.actions = []
        self.menu = self.tr(u'&DSP')

        # Check if plugin was started the first time in current QGIS session
        # Must be set in initGui() to survive plugin reloads
        self.first_start = None


    # noinspection PyMethodMayBeStatic
    def tr(self, message):
        """Get the translation for a string using Qt translation API.

        We implement this ourselves since we do not inherit QObject.

        :param message: String for translation.
        :type message: str, QString

        :returns: Translated version of message.
        :rtype: QString
        """
        # noinspection PyTypeChecker,PyArgumentList,PyCallByClass
        return QCoreApplication.translate('DroneSurveyingPlanning', message)


    def add_action(
        self,
        icon_path,
        text,
        callback,
        enabled_flag=True,
        add_to_menu=True,
        add_to_toolbar=True,
        status_tip=None,
        whats_this=None,
        parent=None):
        """Add a toolbar icon to the toolbar.

        :param icon_path: Path to the icon for this action. Can be a resource
            path (e.g. ':/plugins/foo/bar.png') or a normal file system path.
        :type icon_path: str

        :param text: Text that should be shown in menu items for this action.
        :type text: str

        :param callback: Function to be called when the action is triggered.
        :type callback: function

        :param enabled_flag: A flag indicating if the action should be enabled
            by default. Defaults to True.
        :type enabled_flag: bool

        :param add_to_menu: Flag indicating whether the action should also
            be added to the menu. Defaults to True.
        :type add_to_menu: bool

        :param add_to_toolbar: Flag indicating whether the action should also
            be added to the toolbar. Defaults to True.
        :type add_to_toolbar: bool

        :param status_tip: Optional text to show in a popup when mouse pointer
            hovers over the action.
        :type status_tip: str

        :param parent: Parent widget for the new action. Defaults None.
        :type parent: QWidget

        :param whats_this: Optional text to show in the status bar when the
            mouse pointer hovers over the action.

        :returns: The action that was created. Note that the action is also
            added to self.actions list.
        :rtype: QAction
        """

        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar:
            # Adds plugin icon to Plugins toolbar
            self.iface.addToolBarIcon(action)

        if add_to_menu:
            self.iface.addPluginToMenu(
                self.menu,
                action)

        self.actions.append(action)

        return action

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""

        icon_path = ':/plugins/DSP/icon.png'
        self.add_action(
            icon_path,
            text=self.tr(u'DSP'),
            callback=self.run,
            parent=self.iface.mainWindow())

        # will be set False in run()
        self.loadDrone()
        self.loadSensor()
        self.first_start = True
        self.dlg.tb_invector.clicked.connect(self.openVector)
        self.dlg.tb_inDTM.clicked.connect(self.openDTM)
        self.dlg.pb_drone.clicked.connect(self.open_drone_dialog)
        self.dlg.pb_sensor.clicked.connect(self.open_sensor_dialog)
        #self.dlg.pb_run.clicked.connect(self.start_simulation)
        self.dlg.HelpPushButton.clicked.connect(self.open_DSPHELP_dialog)
        self.dlg.close_button.clicked.connect(self.dlg.close)

    def loadDrone(self):
        """Add the created new drone to the list of drones in main DSP dialog"""
        self.dlg.cb_drone.clear()

        drone_combobox = []
        for drone in self.drone_list:
            drone_combobox.append(drone['DroneName'])

        self.dlg.cb_drone.addItems(drone_combobox)

    def loadSensor(self):
        """Add the created new sensor to the list of sensors in main DSP dialog"""
        self.dlg.cb_sensor.clear()

        sensor_combobox = []
        for sensor in self.sensor_list:
            sensor_combobox.append(sensor['SensorName'])

        self.dlg.cb_sensor.addItems(sensor_combobox)

    def loadVectors(self):
        """Load vectors from QGIS table of contents"""
        self.dlg.cb_invector.clear()
        layers = [layer for layer in QgsProject.instance().mapLayers().values()]
        vector_layers = []
        for layer in layers:
            if layer.type() == QgsMapLayer.VectorLayer:
                vector_layers.append(layer.name())
        self.dlg.cb_invector.addItems(vector_layers)

    def openVector(self):
        """Open vector from file dialog"""
        inFile = str(QFileDialog.getOpenFileName(caption = "Open shapefile",
                                                 filter = "Shapefiles(*.shp)")[0])
        if inFile is not None:
            self.iface.addVectorLayer(inFile, str.split(os.path.basename(inFile), ".")[0], "ogr")
            self.loadVectors()

    def loadDTM(self):
        """Load DTMs from QGIS table of contents"""
        self.dlg.cb_inDTM.clear()
        layers = [layer for layer in QgsProject.instance().mapLayers().values()]
        DTM_layers = []
        for layer in layers:
            if layer.type() == QgsMapLayer.RasterLayer:
                DTM_layers.append(layer.name())
        self.dlg.cb_inDTM.addItems(DTM_layers)

    def openDTM(self):
        """Open DTM from file dialog"""
        inFile = str(QFileDialog.getOpenFileName(caption="Open DTM",
                                                 filter="GeoTiff(*.tif)")[0])
        if inFile is not None:
            self.iface.addRasterLayer(inFile, str.split(os.path.basename(inFile), ".")[0])
            self.loadDTM()

    def open_drone_dialog(self):
        """Open dialog to create a new drone"""
        self.droneDialog = DroneDialog(parent=self.dlg)
        self.droneDialog.ui.pb_okdrone.clicked.connect(self.collect_drone)
        self.droneDialog.ui.pb_close.clicked.connect(self.droneDialog.close)
        self.droneDialog.show()

    def collect_drone(self):
        """Collect the new drone attributes """
        new_drone = dict()
        new_drone['DroneName'] = self.droneDialog.ui.i_name.text()
        new_drone['MaxAltitude'] = self.droneDialog.ui.sb_altitude.value()
        new_drone['MaxSpeed'] = self.droneDialog.ui.sb_speed.value()
        new_drone['Battery'] = self.droneDialog.ui.sb_battery.value()
        self.drone_list.append(new_drone)
        self.loadDrone()
        self.droneDialog.close()


    def open_sensor_dialog(self):
        """Open dialog to create a new drone"""
        self.sensorDialog = SensorDialog(parent=self.dlg)
        self.sensorDialog.ui.pb_oksensor.clicked.connect(self.collect_sensor)
        self.sensorDialog.ui.pb_close.clicked.connect(self.sensorDialog.close)
        self.sensorDialog.show()

    def collect_sensor(self):
        """Collect the new sensor attributes """
        new_sensor = dict()
        new_sensor['SensorName'] = self.sensorDialog.ui.i_name.text()
        new_sensor['FocalLenght'] = self.sensorDialog.ui.sb_fl.value()
        new_sensor['ShootInterval'] = self.sensorDialog.ui.sb_si.value()
        new_sensor['SizeX'] = self.sensorDialog.ui.sb_sizex.value()
        new_sensor['SizeY'] = self.sensorDialog.ui.sb_sizey.value()
        new_sensor['ImgSizeX'] = self.sensorDialog.ui.sb_imgsizex.value()
        new_sensor['ImgSizeY'] = self.sensorDialog.ui.sb_imgsizey.value()
        self.sensor_list.append(new_sensor)
        self.loadSensor()
        self.sensorDialog.close()

    def open_DSPHELP_dialog(self):
        """Opens the HELP dialog containing a description of plugin fields"""
        nd = DSPHELPDialog(parent=self.dlg)
        nd.show()

    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        for action in self.actions:
            self.iface.removePluginMenu(
                self.tr(u'&DSP'),
                action)
            self.iface.removeToolBarIcon(action)


    def find_dict_in_list(self, list_, key_name, key_value):
        for i in list_:
            if i[key_name] == key_value:
                return i
        return list_[0]
    '''
    def start_simulation(self):
        
        #prende il drone che ti serve
        selected_drone = self.find_dict_in_list(list_ = self.drone_list,
                               key_name='DroneName',
                               key_value=''#recupera valore da tendina)

        #prende il sensore che ti serve
        selected_sensor = self.find_dict_in_list(list_ = self.sensor_list,
                               key_name='SensorName',
                               key_value=''#recupera valore da tendina)

        #fa il resto
        #run_
    '''
    def run(self):
        """Run method that performs all the real work"""

        # Create the dialog with elements (after translation) and keep reference
        # Only create GUI ONCE in callback, so that it will only load when the plugin is started



        # show the dialog
        self.dlg.show()
        self.loadVectors()
        self.loadDTM()
        # Run the dialog event loop
        result = self.dlg.exec_()
        # See if OK was pressed
        if result:
            # Do something useful here - delete the line containing pass and
            # substitute with your code.
            pass

