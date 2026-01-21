# Created by Jannet Trabelsi on 2025-09-02
# Please note: the controller class raises errors. However, the GUI sets default values to solve for invalid inputs
import time
from PyQt5.QtWidgets import QMessageBox, QWidget
from src.Controller.newport_conex_cc import Newport_CONEX_CC_xy_stage
from src.Controller.nanodrive import MCLNanoDrive
from src.Model.data_saver import ExperimentHDF5WriterSWMR
from .positioning_stages_design import Ui_Form
from datetime import datetime
import os
import numpy as np
from src.Model.data_saver import ExperimentHDF5ReaderSWMR
from PyQt5.QtWidgets import QFileDialog
# Assuming the .ui file is converted to design.py
#To convert positioning_stages_design.ui to .py, paste this into the terminal:
# pyuic5 -x positioning_stages_design.ui -o positioning_stages_design.py

# constants:
_MAX_X_1 = 100
_MIN_X_1 = 0
_MAX_Y_1 = 100
_MIN_Y_1 = 0
_MAX_Z_1 = 100
_MIN_Z_1 = 0
_MAX_X_2 = 100
_MIN_X_2 = 0
_MAX_Y_2 = 100
_MIN_Y_2 = 0
_MAX_Z_2 = 100
_MIN_Z_2 = 0
_MAX_MCL_nanodrive_X = 100
_MIN_MCL_nanodrive_X = 0
_MAX_MCL_nanodrive_Y = 100
_MIN_MCL_nanodrive_Y = 0
_MAX_MCL_nanodrive_Z =100
_MIN_MCL_nanodrive_Z = 0
from PyQt5.QtCore import pyqtSignal

class positioning_stages_view(QWidget, Ui_Form):
    """
    This is the widget of the positioning stages. It allows us to control positioning devices using buttons and LineEdits
    """
    display_choice_changed = pyqtSignal(str)
    snapshot_mode_changed = pyqtSignal(int)
    save_or_find_nv_button_clicked = pyqtSignal(int)
    def __init__(self, parent = None):
        super().__init__(parent)
        self.setupUi(self)
        self.stage_1 = None
        self.stage_2 = None

        self.xlineEdit_1.setEnabled(False)
        self.ylineEdit_1.setEnabled(False)
        self.zlineEdit_1.setEnabled(False)
        self.x_y_inc_lineEdit_1.setEnabled(False)
        self.z_inc_lineEdit_1.setEnabled(False)
        self.x_y_inc_lineEdit_2.setEnabled(False)
        self.z_inc_lineEdit_2.setEnabled(False)
        self.x_inc_1.setEnabled(False)
        self.y_inc_1.setEnabled(False)
        self.z_inc_1.setEnabled(False)
        self.x_dec_1.setEnabled(False)
        self.y_dec_1.setEnabled(False)
        self.z_dec_1.setEnabled(False)

        self.xlineEdit_2.setEnabled(False)
        self.ylineEdit_2.setEnabled(False)
        self.zlineEdit_2.setEnabled(False)
        self.x_inc_2.setEnabled(False)
        self.y_inc_2.setEnabled(False)
        self.z_inc_2.setEnabled(False)
        self.x_dec_2.setEnabled(False)
        self.y_dec_2.setEnabled(False)
        self.z_dec_2.setEnabled(False)
        self.confirm_x_button_1.setEnabled(False)
        self.confirm_y_button_1.setEnabled(False)
        self.confirm_z_button_1.setEnabled(False)
        self.confirm_x_button_2.setEnabled(False)
        self.confirm_y_button_2.setEnabled(False)
        self.confirm_z_button_2.setEnabled(False)

        # Connect buttons to functions
        self.connectButton_1.clicked.connect(self.connect_to_instrument_1)
        self.connectButton_2.clicked.connect(self.connect_to_instrument_2)
        self.confirm_x_button_1.clicked.connect(self.set_x_position_1)
        self.confirm_y_button_1.clicked.connect(self.set_y_position_1)
        self.confirm_z_button_1.clicked.connect(self.set_z_position_1)
        self.confirm_x_button_2.clicked.connect(self.set_x_position_2)
        self.confirm_y_button_2.clicked.connect(self.set_y_position_2)
        self.confirm_z_button_2.clicked.connect(self.set_z_position_2)
        self.x_inc_1.clicked.connect(self.inc_x_position_1)
        self.x_inc_2.clicked.connect(self.inc_x_position_2)
        self.x_dec_1.clicked.connect(self.dec_x_position_1)
        self.x_dec_2.clicked.connect(self.dec_x_position_2)
        self.y_inc_1.clicked.connect(self.inc_y_position_1)
        self.y_inc_2.clicked.connect(self.inc_y_position_2)
        self.y_dec_1.clicked.connect(self.dec_y_position_1)
        self.y_dec_2.clicked.connect(self.dec_y_position_2)
        self.z_inc_1.clicked.connect(self.inc_z_position_1)
        self.z_inc_2.clicked.connect(self.inc_z_position_2)
        self.z_dec_1.clicked.connect(self.dec_z_position_1)
        self.z_dec_2.clicked.connect(self.dec_z_position_2)
        self.save_button.clicked.connect(self.save)
        self.Find_NV_Button.clicked.connect(self.find_NV)
        # Connect combobox signals to emitters
        self.display_option.currentTextChanged.connect(self.on_display_choice_changed)
        self.snapshot_live_comboBox.currentTextChanged.connect(self.on_snapshot_or_live_changed)
        self.data_saving_path = None
        self.data_reader = None

    def connect_to_instrument_1(self):
        """This function connects the devices: please make sure that your stage has the function get_position(self, axis)"""
        stage_name = self.comboBox_1.currentText()
        if stage_name == 'MCL_nanodrive':
            _MAX_X_1 = _MAX_MCL_nanodrive_X
            _MIN_X_1 = _MIN_MCL_nanodrive_X
            _MAX_Y_1 = _MAX_MCL_nanodrive_Y
            _MIN_Y_1 = _MIN_MCL_nanodrive_Y
            _MAX_Z_1 = _MAX_MCL_nanodrive_Z
            _MIN_Z_1 = _MIN_MCL_nanodrive_Z
            try:
                self.stage_1 = MCLNanoDrive()
                self.zlineEdit_1.setEnabled(True)
                self.zlineEdit_1.setText(str(self.stage_1.get_position('z')))
                self.confirm_z_button_1.setEnabled(True)
                self.z_inc_1.setEnabled(True)
                self.z_dec_1.setEnabled(True)
                self.z_inc_lineEdit_1.setEnabled(True)
                QMessageBox.information(self, 'Success', f'Connected to MCL_nanodrive')

            except Exception as e:
                QMessageBox.critical(self, 'Error', str(e))
        elif stage_name == 'Newport_Conex_microdrive':
            try:
                self.stage_1 = Newport_CONEX_CC_xy_stage()
                _MAX_X_1  = self.stage_1.get_positive_software_limit('x')
                _MIN_X_1 = self.stage_1.get_negative_software_limit('x')
                _MAX_Y_1 = self.stage_1.get_positive_software_limit('y')
                _MIN_Y_1 = self.stage_1.get_negative_software_limit('y')
                QMessageBox.information(self, 'Success', f'Connected to Newport_Conex_microdrive')

            except Exception as e:
                QMessageBox.critical(self, 'Error', str(e))
        else:
            return
        self.xlineEdit_1.setText(str(self.stage_1.get_position('x')))
        self.ylineEdit_1.setText(str(self.stage_1.get_position('y')))
        self.xlineEdit_1.setEnabled(True)
        self.ylineEdit_1.setEnabled(True)
        self.confirm_x_button_1.setEnabled(True)
        self.confirm_y_button_1.setEnabled(True)
        self.x_y_inc_lineEdit_1.setEnabled(True)
        self.x_inc_1.setEnabled(True)
        self.y_inc_1.setEnabled(True)
        self.x_dec_1.setEnabled(True)
        self.y_dec_1.setEnabled(True)

    def connect_to_instrument_2(self):
        """This function connects the devices: please make sure that your stage has the function get_position(self, axis)"""
        stage_name = self.comboBox_2.currentText()
        if stage_name == 'MCL_nanodrive':
            _MAX_X_2 = _MAX_MCL_nanodrive_X
            _MIN_X_2 = _MIN_MCL_nanodrive_X
            _MAX_Y_2 = _MAX_MCL_nanodrive_Y
            _MIN_Y_2 = _MIN_MCL_nanodrive_Y
            _MAX_Z_2 = _MAX_MCL_nanodrive_Z
            _MIN_Z_2 = _MIN_MCL_nanodrive_Z
            try:
                self.stage_2 = MCLNanoDrive()
                self.zlineEdit_2.setEnabled(True)
                self.confirm_z_button_2.setEnabled(True)
                self.zlineEdit_2.setText(str(self.stage_2.get_position('z')))
                self.z_inc_2.setEnabled(True)
                self.z_dec_2.setEnabled(True)
                self.z_inc_lineEdit_2.setEnabled(True)
                QMessageBox.information(self, 'Success', f'Connected to MCL_nanodrive')

            except Exception as e:
                QMessageBox.critical(self, 'Error', str(e))
        elif stage_name == 'Newport_Conex_microdrive':
            try:
                self.stage_2 = Newport_CONEX_CC_xy_stage()
                _MAX_X_2 = self.stage_2.get_positive_software_limit('x')
                _MIN_X_2 = self.stage_2.get_negative_software_limit('x')
                _MAX_Y_2 = self.stage_2.get_positive_software_limit('y')
                _MIN_Y_2 = self.stage_2.get_negative_software_limit('y')
                QMessageBox.information(self, 'Success', f'Connected to Newport_Conex_microdrive')

            except Exception as e:
                QMessageBox.critical(self, 'Error', str(e))
        else:
            return
        self.xlineEdit_2.setText(str(self.stage_2.get_position('x')))
        self.ylineEdit_2.setText(str(self.stage_2.get_position('y')))
        self.xlineEdit_2.setEnabled(True)
        self.ylineEdit_2.setEnabled(True)
        self.confirm_x_button_2.setEnabled(True)
        self.confirm_y_button_2.setEnabled(True)
        self.x_y_inc_lineEdit_2.setEnabled(True)
        self.x_inc_2.setEnabled(True)
        self.y_inc_2.setEnabled(True)
        self.x_dec_2.setEnabled(True)
        self.y_dec_2.setEnabled(True)

    def set_x_position_1(self):
        x_pos = self.xlineEdit_1.text()
        if isinstance(x_pos, str) or isinstance(x_pos, int) or isinstance(x_pos, float):
            try:
                x_pos = float(x_pos)
                if x_pos < _MAX_X_1 and x_pos > _MIN_X_1:
                    self.stage_1.set_position('x',x_pos)
                    time.sleep(2)
                    self.xlineEdit_1.setText(str(self.stage_1.get_position('x')))
            except ValueError:
                self.xlineEdit_1.setText(str(self.stage_1.get_position('x')))
                return
        else:
            self.xlineEdit_1.setText(str(self.stage_1.get_position('x')))
            return

    def set_x_position_2(self):
        x_pos = self.xlineEdit_2.text()
        if isinstance(x_pos, str) or isinstance(x_pos, int) or isinstance(x_pos, float):
            try:
                x_pos = float(x_pos)
                if x_pos < _MAX_X_2 and x_pos > _MIN_X_2:
                    self.stage_2.set_position('x',x_pos)
                    time.sleep(2)
                    self.xlineEdit_2.setText(str(self.stage_2.get_position('x')))
            except ValueError:
                self.xlineEdit_2.setText(str(self.stage_2.get_position('x')))
                return
        else:
            self.xlineEdit_2.setText(str(self.stage_2.get_position('x')))
            return

    def set_y_position_1(self):
        y_pos = self.ylineEdit_1.text()
        if isinstance(y_pos, str) or isinstance(y_pos, int) or isinstance(y_pos, float):
            try:
                y_pos = float(y_pos)
                if y_pos < _MAX_Y_1 and y_pos > _MIN_Y_1:
                    self.stage_1.set_position('y',y_pos)
                    time.sleep(2)
                    self.ylineEdit_1.setText(str(self.stage_1.get_position('y')))
            except ValueError:
                self.ylineEdit_1.setText(str(self.stage_1.get_position('y')))
                return
        else:
            self.ylineEdit_1.setText(str(self.stage_1.get_position('y')))
            return

    def set_y_position_2(self):
        y_pos = self.ylineEdit_2.text()
        if isinstance(y_pos, str) or isinstance(y_pos, int) or isinstance(y_pos, float):
            try:
                y_pos = float(y_pos)
                if y_pos < _MAX_Y_2 and y_pos > _MIN_Y_2:
                    self.stage_2.set_position('y', y_pos)
                    time.sleep(2)
                    self.ylineEdit_2.setText(str(self.stage_2.get_position('y')))
            except ValueError:
                self.ylineEdit_2.setText(str(self.stage_2.get_position('y')))
                return
        else:
            self.ylineEdit_2.setText(str(self.stage_2.get_position('y')))
            return

    def set_z_position_1(self):
        z_pos = self.zlineEdit_1.text()
        if isinstance(z_pos, str) or isinstance(z_pos, int) or isinstance(z_pos, float):
            try:
                z_pos = float(z_pos)
                if z_pos < _MAX_Z_1 and z_pos > _MAX_Z_1:
                    self.stage_1.set_position('z', z_pos)
                    time.sleep(2)
                    self.zlineEdit_1.setText(str(self.stage_1.get_position('z')))
            except ValueError:
                self.zlineEdit_1.setText(str(self.stage_1.get_position('z')))
                return
        else:
            self.zlineEdit_1.setText(str(self.stage_1.get_position('z')))
            return

    def set_z_position_2(self):
        z_pos = self.zlineEdit_2.text()
        if isinstance(z_pos, str) or isinstance(z_pos, int) or isinstance(z_pos, float):
            try:
                z_pos = float(z_pos)
                if z_pos < _MAX_Z_2 and z_pos > _MAX_Z_2:
                    self.stage_2.set_position('z', z_pos)
                    time.sleep(2)
                    self.zlineEdit_2.setText(str(self.stage_2.get_position('z')))
            except ValueError:
                self.zlineEdit_2.setText(str(self.stage_2.get_position('z')))
                return
        else:
            self.zlineEdit_2.setText(str(self.stage_2.get_position('z')))
            return

    def inc_x_position_1(self):
        inc_step = self.x_y_inc_lineEdit_1
        x_pos = self.stage_1.get_position('x')
        if isinstance(inc_step, str) or isinstance(inc_step, int) or isinstance(inc_step, float):
            try:
                inc_step = float(inc_step)
                if x_pos+inc_step < _MAX_X_1 and x_pos > _MIN_X_1+inc_step:
                    self.stage_1.set_position('x',x_pos+inc_step)
                    time.sleep(2)
                    self.xlineEdit_1.setText(str(self.stage_1.get_position('x')))
            except ValueError:
                self.xlineEdit_1.setText(str(self.stage_1.get_position('x')))
                return
        else:
            self.xlineEdit_1.setText(str(self.stage_1.get_position('x')))
            return

    def inc_y_position_1(self):
        inc_step = self.x_y_inc_lineEdit_1
        y_pos = self.stage_1.get_position('y')
        if isinstance(inc_step, str) or isinstance(inc_step, int) or isinstance(inc_step, float):
            try:
                inc_step = float(inc_step)
                if y_pos + inc_step < _MAX_Y_1 and y_pos > _MIN_Y_1 + inc_step:
                    self.stage_1.set_position('y', y_pos + inc_step)
                    time.sleep(2)
                    self.ylineEdit_1.setText(str(self.stage_1.get_position('y')))
            except ValueError:
                self.ylineEdit_1.setText(str(self.stage_1.get_position('y')))
                return
        else:
            self.ylineEdit_1.setText(str(self.stage_1.get_position('y')))
            return

    def inc_z_position_1(self):
        inc_step = self.z_inc_lineEdit_1
        z_pos = self.stage_1.get_position('z')
        if isinstance(inc_step, str) or isinstance(inc_step, int) or isinstance(inc_step, float):
            try:
                inc_step = float(inc_step)
                if z_pos + inc_step < _MAX_Z_1 and z_pos > _MIN_Z_1 + inc_step:
                    self.stage_1.set_position('z', z_pos + inc_step)
                    time.sleep(2)
                    self.zlineEdit_1.setText(str(self.stage_1.get_position('z')))
            except ValueError:
                self.zlineEdit_1.setText(str(self.stage_1.get_position('z')))
                return
        else:
            self.zlineEdit_1.setText(str(self.stage_1.get_position('z')))
            return

    def dec_x_position_1(self):
        inc_step = self.x_y_inc_lineEdit_1
        y_pos = self.stage_1.get_position('y')
        if isinstance(inc_step, str) or isinstance(inc_step, int) or isinstance(inc_step, float):
            try:
                inc_step = float(inc_step)
                if y_pos - inc_step < _MAX_Y_1 and y_pos > _MIN_Y_1 - inc_step:
                    self.stage_1.set_position('y', y_pos - inc_step)
                    time.sleep(2)
                    self.ylineEdit_1.setText(str(self.stage_1.get_position('y')))
            except ValueError:
                self.ylineEdit_1.setText(str(self.stage_1.get_position('y')))
                return
        else:
            self.ylineEdit_1.setText(str(self.stage_1.get_position('y')))
            return

    def dec_y_position_1(self):
        inc_step = self.x_y_inc_lineEdit_1
        y_pos = self.stage_1.get_position('y')
        if isinstance(inc_step, str) or isinstance(inc_step, int) or isinstance(inc_step, float):
            try:
                inc_step = float(inc_step)
                if y_pos - inc_step < _MAX_Y_1 and y_pos > _MIN_Y_1 - inc_step:
                    self.stage_1.set_position('y', y_pos - inc_step)
                    time.sleep(2)
                    self.ylineEdit_1.setText(str(self.stage_1.get_position('y')))
            except ValueError:
                self.ylineEdit_1.setText(str(self.stage_1.get_position('y')))
                return
        else:
            self.ylineEdit_1.setText(str(self.stage_1.get_position('y')))
            return

    def dec_z_position_1(self):
        inc_step = self.z_inc_lineEdit_1
        z_pos = self.stage_1.get_position('z')
        if isinstance(inc_step, str) or isinstance(inc_step, int) or isinstance(inc_step, float):
            try:
                inc_step = float(inc_step)
                if z_pos - inc_step < _MAX_Z_1 and z_pos > _MIN_Z_1 - inc_step:
                    self.stage_1.set_position('z', z_pos - inc_step)
                    time.sleep(2)
                    self.zlineEdit_1.setText(str(self.stage_1.get_position('z')))
            except ValueError:
                self.zlineEdit_1.setText(str(self.stage_1.get_position('z')))
                return
        else:
            self.zlineEdit_1.setText(str(self.stage_1.get_position('z')))
            return

    def inc_x_position_2(self):
        inc_step = self.x_y_inc_lineEdit_2
        x_pos = self.stage_2.get_position('x')
        if isinstance(inc_step, str) or isinstance(inc_step, int) or isinstance(inc_step, float):
            try:
                inc_step = float(inc_step)
                if x_pos + inc_step < _MAX_X_2 and x_pos > _MIN_X_2 + inc_step:
                    self.stage_2.set_position('x', x_pos + inc_step)
                    time.sleep(2)
                    self.xlineEdit_2.setText(str(self.stage_2.get_position('x')))
            except ValueError:
                self.xlineEdit_2.setText(str(self.stage_2.get_position('x')))
                return
        else:
            self.xlineEdit_2.setText(str(self.stage_2.get_position('x')))
            return

    def inc_y_position_2(self):
        inc_step = self.x_y_inc_lineEdit_2
        y_pos = self.stage_2.get_position('y')
        if isinstance(inc_step, str) or isinstance(inc_step, int) or isinstance(inc_step, float):
            try:
                inc_step = float(inc_step)
                if y_pos + inc_step < _MAX_Y_2 and y_pos > _MIN_Y_2 + inc_step:
                    self.stage_2.set_position('y', y_pos + inc_step)
                    time.sleep(2)
                    self.ylineEdit_2.setText(str(self.stage_2.get_position('y')))
            except ValueError:
                self.ylineEdit_2.setText(str(self.stage_2.get_position('y')))
                return
        else:
            self.ylineEdit_2.setText(str(self.stage_2.get_position('y')))
            return

    def inc_z_position_2(self):
        inc_step = self.z_inc_lineEdit_2
        z_pos = self.stage_2.get_position('z')
        if isinstance(inc_step, str) or isinstance(inc_step, int) or isinstance(inc_step, float):
            try:
                inc_step = float(inc_step)
                if z_pos + inc_step < _MAX_Z_2 and z_pos > _MIN_Z_2 + inc_step:
                    self.stage_2.set_position('z', z_pos + inc_step)
                    time.sleep(2)
                    self.zlineEdit_2.setText(str(self.stage_2.get_position('z')))
            except ValueError:
                self.zlineEdit_2.setText(str(self.stage_2.get_position('z')))
                return
        else:
            self.zlineEdit_2.setText(str(self.stage_2.get_position('z')))
            return

    def dec_x_position_2(self):
        inc_step = self.x_y_inc_lineEdit_2
        x_pos = self.stage_2.get_position('x')
        if isinstance(inc_step, str) or isinstance(inc_step, int) or isinstance(inc_step, float):
            try:
                inc_step = float(inc_step)
                if x_pos - inc_step < _MAX_X_2 and x_pos > _MIN_X_2 - inc_step:
                    self.stage_2.set_position('x', x_pos - inc_step)
                    time.sleep(2)
                    self.xlineEdit_2.setText(str(self.stage_2.get_position('x')))
            except ValueError:
                self.xlineEdit_2.setText(str(self.stage_2.get_position('x')))
                return
        else:
            self.xlineEdit_2.setText(str(self.stage_2.get_position('x')))
            return

    def dec_y_position_2(self):
        inc_step = self.x_y_inc_lineEdit_2
        y_pos = self.stage_2.get_position('y')
        if isinstance(inc_step, str) or isinstance(inc_step, int) or isinstance(inc_step, float):
            try:
                inc_step = float(inc_step)
                if y_pos - inc_step < _MAX_Y_2 and y_pos > _MIN_Y_2 - inc_step:
                    self.stage_2.set_position('y', y_pos - inc_step)
                    time.sleep(2)
                    self.ylineEdit_2.setText(str(self.stage_2.get_position('y')))
            except ValueError:
                self.ylineEdit_2.setText(str(self.stage_2.get_position('y')))
                return
        else:
            self.ylineEdit_2.setText(str(self.stage_2.get_position('y')))
            return

    def dec_z_position_2(self):
        inc_step = self.z_inc_lineEdit_2
        z_pos = self.stage_2.get_position('z')
        if isinstance(inc_step, str) or isinstance(inc_step, int) or isinstance(inc_step, float):
            try:
                inc_step = float(inc_step)
                if z_pos - inc_step < _MAX_Z_2 and z_pos > _MIN_Z_2 - inc_step:
                    self.stage_2.set_position('z', z_pos - inc_step)
                    time.sleep(2)
                    self.zlineEdit_2.setText(str(self.stage_2.get_position('z')))
            except ValueError:
                self.zlineEdit_2.setText(str(self.stage_2.get_position('z')))
                return
        else:
            self.zlineEdit_2.setText(str(self.stage_2.get_position('z')))
            return

    def on_display_choice_changed(self, text):
        # This one should be handled in main window by the display
        disp = self.display_option.currentText()
        if disp == "MU300":
            self.snapshot_live_comboBox.setEnabled(True)
        print("on_display_choice_changed emitting", text)
        self.display_choice_changed.emit(text)

    def on_snapshot_or_live_changed(self, text):
        mode = 0 if text.lower() == "snapshot" else 1
        print("on_snapshot_or_live_changed emitting", mode)
        self.snapshot_mode_changed.emit(mode)

    def display_choice(self):
        disp = self.display_option.currentText()
        if disp == "MU300":
            self.snapshot_live_comboBox.setEnabled(True)
        return disp

    def snapshot_or_live(self):
        return self.snapshot_live_comboBox.currentText()

    def save(self):
        self.save_or_find_nv_button_clicked.emit(1)
        # --- UI → internal keys ---
        sample_selection = self.Sample_Selector_comboBox.currentText()
        point_selection = self.Point_Selector_comboBox.currentText()
        point_status = self.point_status_comboBox.currentText()

        point_key = point_selection.lower().replace(" ", "_")
        if point_status == "FINAL" and point_selection == "NV":
            self.error_box("YOU CANNOT SELECT FINAL NV POINT", "To find NV, click find NV button!")
            return

        # --- File handling ---
        if sample_selection == "New Sample":
            point_status = "INITIAL"
            directory, filename = self.open_directory_dialog(self.data_saving_path)
            if filename is None:
                return  # user canceled

            self.data_saving_path = directory
            full_path = os.path.join(directory, filename)

            if os.path.exists(full_path):
                if not self.confirm_overwrite(filename):
                    return

            mode = "w"

        else:
            full_path = self.open_file_dialog(self.data_saving_path)
            if full_path is None:
                return
            mode = "r+"

        # --- Open writer ---
        writer = ExperimentHDF5WriterSWMR(
            full_path,
            mode=mode,
            swmr=False  # snapshot, not live streaming
        )

        # --- positioning group ---
        positioning = writer.file.require_group("positioning")
        positioning.attrs.setdefault("created", datetime.utcnow().isoformat())

        parent = positioning.require_group(point_status)

        # confirm overwrite BEFORE creating
        if point_key in parent:
            if not self.confirm_overwrite(point_key):
                writer.close()
                return
            del parent[point_key]

        grp = parent.create_group(point_key)

        # --- Identify stages ---
        stage_1_name = self.comboBox_1.currentText()
        if "nanodrive" in stage_1_name.lower():
            nano = self.stage_1
            micro = self.stage_2
        else:
            nano = self.stage_2
            micro = self.stage_1

        # --- Snapshot metadata ---
        grp.attrs["micro_x"] = micro.get_position("x")
        grp.attrs["micro_y"] = micro.get_position("y")

        grp.attrs["nano_x"] = nano.get_position("x")
        grp.attrs["nano_y"] = nano.get_position("y")
        grp.attrs["nano_z"] = nano.get_position("z")

        grp.attrs["camera_x"] = self.x_crosshair
        grp.attrs["camera_y"] = self.y_crosshair

        grp.attrs["timestamp"] = datetime.utcnow().isoformat()

        writer.flush()
        writer.close()

    def error_box(self, text, info, title="Error"):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle(title)
        msg.setText(text)
        msg.setInformativeText(info)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.setDefaultButton(QMessageBox.Ok)

        return msg.exec() == QMessageBox.Ok

    def open_file_dialog(self, start_path):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open sample file",
            start_path,
            "HDF5 files (*.h5);;All files (*)"
        )

        if not filename:
            raise RuntimeError("No file selected")

        return filename

    def open_directory_dialog(self, start_path):
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Select file location",
            start_path,
            "HDF5 Files (*.h5);;All Files (*)"
        )

        # User pressed Cancel
        if not file_path:
            return None, None

        directory = os.path.dirname(file_path)
        filename = os.path.basename(file_path)

        return directory, filename

    def confirm_overwrite(self, point_key):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Overwrite point?")
        msg.setText(f"Point '{point_key}' already exists.")
        msg.setInformativeText("Do you want to overwrite it?")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.No)

        return msg.exec() == QMessageBox.Yes

    def find_NV(self):
        self.save_or_find_nv_button_clicked.emit(1)
        # select file
        path = self.open_file_dialog(self.data_saving_path)
        if not path:
            return

        # read file
        self.data_reader = ExperimentHDF5ReaderSWMR(path)
        self.data_reader.read_structure()

        structure = self.data_reader.get_structure()

        required_corners = {"top_left", "top_right", "bottom_left", "bottom_right"}

        init = structure.get("INITIAL", {})
        final = structure.get("FINAL", {})

        # Validation
        if not required_corners.issubset(init.keys()):
            self.error_box("INITIAL does not contain all 4 corners", "please add all 4 corners before proceeding")
            self.data_reader.close()
            return

        if not required_corners.issubset(final.keys()):
            self.error_box("FINAL does not contain all 4 corners", "please add all 4 corners before proceeding")
            self.data_reader.close()
            return

        if "nv" not in init:
            self.error_box("NV does not exist in INITIAL", "please NV point before proceeding")
            self.data_reader.close()
            return

        # Prepare points
        old_pts = [init[c] for c in required_corners]
        new_pts = [final[c] for c in required_corners]
        print(f"required corners: {required_corners}")
        print(f"new corners: {new_pts}")
        print(f"old corners: {old_pts}")

        params = self.fit_micro_nano_transform(old_pts, new_pts)

        nv_old = init["nv"]
        nv_x, nv_y = self.predict_nv_position(params, nv_old)

        self.data_reader.close()

        return nv_x, nv_y

    def fit_micro_nano_transform(self, old_pts, new_pts):
        """
        old_pts, new_pts: list of dicts with keys:
          micro_x, micro_y, nano_x, nano_y

        Returns:
          params: length-10 array
        """

        A = []
        B = []

        for o, n in zip(old_pts, new_pts):
            mx, my = o["micro_x"], o["micro_y"]
            nx, ny = o["nano_x"], o["nano_y"]
            print(f"mx {mx} my {my} nx {nx} ny {ny}")

            A.append([mx, my, nx, ny, 1, 0, 0, 0, 0, 0])
            A.append([0, 0, 0, 0, 0, mx, my, nx, ny, 1])
            print(f"n[micro_x]: {n["micro_x"]}")
            print(f"n[nano_x]: {n["nano_x"]}")
            print(f"n[nano_y]: {n["nano_y"]}")
            print(f"n[micro_y]: {n["micro_y"]}")
            # New *effective* sample-plane position
            x_new = float(n["micro_x"]) + n["nano_x"]
            y_new = float(n["micro_y"]) + n["nano_y"]

            B.append(x_new)
            B.append(y_new)

        A = np.array(A, dtype=float)
        B = np.array(B, dtype=float)

        params, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
        return params

    def predict_nv_position(self, params, nv_old):
        mx, my = float(nv_old["micro_x"]), float(nv_old["micro_y"])
        nx, ny = nv_old["nano_x"], nv_old["nano_y"]

        x = (
                params[0] * mx + params[1] * my +
                params[2] * nx + params[3] * ny +
                params[4]
        )

        y = (
                params[5] * mx + params[6] * my +
                params[7] * nx + params[8] * ny +
                params[9]
        )
        print(f"x found: {x}")
        print(f"y found: {y}")

        return x, y

    """old_pts = [TL_old, TR_old, BL_old, BR_old]
    new_pts = [TL_new, TR_new, BL_new, BR_new]

    params = fit_micro_nano_transform(old_pts, new_pts)

    nv_new_xy = predict_nv_position(params, NV_old)
    print("Predicted NV X,Y:", nv_new_xy)"""