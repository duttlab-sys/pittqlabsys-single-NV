# Created by Jannet Trabelsi on 2025-09-02
# Please note: the controller class raises errors. However, the GUI sets default values to solve for invalid inputs
import time
from PyQt5.QtWidgets import QMessageBox, QWidget
from src.Controller.newport_conex_cc import Newport_CONEX_CC_xy_stage
from src.Controller.nanodrive import MCLNanoDrive
from .positioning_stages_design import Ui_Form
from datetime import datetime
import os
from PyQt5.QtWidgets import QFileDialog
from src.core.struct_hdf5 import StructArray, MyStruct, save_data, load_data
import numpy as np
import cv2
from typing import List, Tuple
from PyQt5.QtWidgets import QMessageBox, QPushButton
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
    take_img_signal = pyqtSignal(int)
    def __init__(self, parent = None):
        super().__init__(parent)
        self.setupUi(self)
        self.stage_1 = None
        self.stage_2 = None

        self.xlineEdit_1.setEnabled(True)
        self.ylineEdit_1.setEnabled(True)
        self.zlineEdit_1.setEnabled(True)
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

        self.xlineEdit_2.setEnabled(True)
        self.ylineEdit_2.setEnabled(True)
        self.zlineEdit_2.setEnabled(True)
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
        self.frame = None

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
                    print(f"position x: {self.stage_2.get_position('x')}")
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
                    print(f"position y: {self.stage_2.get_position('y')}")
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

        # --- UI → keys ---
        sample_selection = self.Sample_Selector_comboBox.currentText()
        point_selection = self.Point_Selector_comboBox.currentText()
        point_status = self.point_status_comboBox.currentText()

        point_key = point_selection.lower().replace(" ", "_")

        if point_status == "FINAL" and point_selection == "NV":
            self.error_box(
                "YOU CANNOT SELECT FINAL NV POINT",
                "To find NV, click find NV button!"
            )
            return

        # --------------------------------------------------
        # File handling
        # --------------------------------------------------
        if sample_selection == "New Sample":
            point_status = "INITIAL"
            directory, filename = self.open_directory_dialog(self.data_saving_path)
            if filename is None:
                return

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
        #old:
        """root = StructArray()

        # positioning.INITIAL / positioning.FINAL
        if not hasattr(root[0], point_status):
            if point_status in ("INITIAL", "FINAL"):
                setattr(root[0], point_status, MyStruct())
            else:
                self.error_box(
                    "Not implemented error",
                    f"please implement {point_status} before proceeding"
                )
                return

        point_status_object = getattr(root[0], point_status)"""
        #new:
        # new root
        root = MyStruct()

        # INITIAL / FINAL
        if not hasattr(root, point_status):
            setattr(root, point_status, MyStruct())

        point_status_object = getattr(root, point_status)

        # bottom_left / nv / etc
        if not hasattr(point_status_object, point_key):
            setattr(point_status_object, point_key, MyStruct())

        point = getattr(point_status_object, point_key)


        # --------------------------------------------------
        # Identify stages
        # --------------------------------------------------
        stage_1_name = self.comboBox_1.currentText()
        if "nanodrive" in stage_1_name.lower():
            nano = self.stage_1
            micro = self.stage_2
        else:
            nano = self.stage_2
            micro = self.stage_1

        # --------------------------------------------------
        # Snapshot metadata
        # --------------------------------------------------
        """grp.attrs["micro_x"] = micro.get_position("x")
                grp.attrs["micro_y"] = micro.get_position("y")

                grp.attrs["nano_x"] = nano.get_position("x")
                grp.attrs["nano_y"] = nano.get_position("y")
                grp.attrs["nano_z"] = nano.get_position("z")"""
        point.micro_x = self.xlineEdit_2.text()
        point.micro_y = self.ylineEdit_2.text()

        point.nano_x = self.xlineEdit_1.text()
        point.nano_y = self.ylineEdit_1.text()
        point.nano_z = self.zlineEdit_1.text()

        point.camera_x = self.x_crosshair
        point.camera_y = self.y_crosshair

        point.timestamp = datetime.utcnow().isoformat()

        # --------------------------------------------------
        # Capture camera image
        # --------------------------------------------------

        if self.snapshot_or_live() == "Snapshot":
            self.take_img_signal.emit(1)
            print("snapshot called")
            print(self.frame)
            point.camera_image = self.frame
        else:
            self.error_box("please take snapshot to same image data", "image data will not be saved for this entry")
            point.camera_image = None
        # --------------------------------------------------
        # SAVE (single call)
        # --------------------------------------------------
        save_data(
            filename=full_path,
            obj=root,
            mode=mode,
            swmr=False  # snapshot, not live
        )

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

    def compute_homography_from_corners(self,
            old_corners: List[np.ndarray],
            new_corners: List[np.ndarray], method) -> np.ndarray:

        if len(old_corners) != 4 or len(new_corners) != 4:
            raise ValueError("Expected 4 corners for old and new quads")

        src_pts = np.array(old_corners, dtype=np.float64)
        dst_pts = np.array(new_corners, dtype=np.float64)

        # ensure points are numpy arrays and float32
        src_pts = np.array(src_pts, dtype=np.float32)
        dst_pts = np.array(dst_pts, dtype=np.float32)

        # check number of points
        if src_pts.shape[0] < 4 or dst_pts.shape[0] < 4:
            raise ValueError("Need at least 4 points to compute homography")

        if method == "LMEDS":
            H, status = cv2.findHomography(src_pts, dst_pts, method=cv2.LMEDS)
        elif method == "RANSAC":
            H, status = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=1.0)
        elif method == "RHOMBUS":
            H, status = cv2.findHomography(src_pts, dst_pts, method=cv2.RHO)
        else:
            H, status = cv2.findHomography(src_pts, dst_pts, cv2.USAC_MAGSAC)

        if status is not None and not np.all(status):
            print("Warning: some corners treated as outliers")

        if H is None:
            raise ValueError("Homography computation failed")

        # Optional: verify fourth corner
        tr_hom = np.array([src_pts[1, 0], src_pts[1, 1], 1.0])
        tr_pred = H @ tr_hom
        tr_pred /= tr_pred[2]
        expected = np.array([dst_pts[1, 0], dst_pts[1, 1], 1.0])
        if np.linalg.norm(tr_pred - expected) > 1e-6:
            print(f"Warning: Fourth corner verification failed. "
                  f"Expected ({expected[0]:.3f},{expected[1]:.3f}), got ({tr_pred[0]:.3f},{tr_pred[1]:.3f})")

        return H

    def map_point_with_homography(self, point: np.ndarray, H: np.ndarray) -> np.ndarray:

        if len(point) == 2:
            pt_hom = np.array([point[0], point[1], 1.0])
        else:
            pt_hom = np.array(point)

        mapped = H @ pt_hom
        mapped /= mapped[2]
        return mapped[:2]

    # affine

    def from_four_corners_to_DMT_or_DMNT(self,
            corners_microdrive: List[np.ndarray],
            reference_order: Tuple[str] = ("top_left", "top_right", "bottom_right", "bottom_left")
    ) -> np.ndarray:
        

        if len(corners_microdrive) != 4:
            raise ValueError(f"Expected 4 corners, got {len(corners_microdrive)}")

        # Extract points in the specified order
        # Assuming input order matches reference_order
        bl_idx = reference_order.index("bottom_left")
        br_idx = reference_order.index("bottom_right")
        tl_idx = reference_order.index("top_left")

        bottom_left = np.array(corners_microdrive[bl_idx], dtype=float)
        bottom_right = np.array(corners_microdrive[br_idx], dtype=float)
        top_left = np.array(corners_microdrive[tl_idx], dtype=float)

        # Diamond coordinate system definition:
        # In diamond coords:
        # bottom_left = (0, 0)
        # bottom_right = (1, 0)
        # top_left = (0, 1)
        # top_right = (1, 1)

        # Source points in microdrive coordinates (homogeneous)
        src_points = np.array([
            [bottom_left[0], bottom_left[1], 1],
            [bottom_right[0], bottom_right[1], 1],
            [top_left[0], top_left[1], 1]
        ]).T  # Shape: (3, 3)

        # Destination points in diamond coordinates (homogeneous)
        dst_points = np.array([
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1]
        ]).T  # Shape: (3, 3)

        # Solve for transformation matrix T such that: dst = T @ src
        # T is 3x3, we need T @ src = dst
        # T = dst @ inv(src)

        try:
            T = dst_points @ np.linalg.inv(src_points)
        except np.linalg.LinAlgError:
            raise ValueError("Corners are colinear or form a degenerate shape")
        return T

    def from_DMT_and_MNV_old_get_DNV_old(self, M_point: np.ndarray, T_matrix: np.ndarray) -> np.ndarray:

        if len(M_point) == 2:
            M_hom = np.array([M_point[0], M_point[1], 1.0])
        else:
            M_hom = np.array(M_point)

        D_hom = T_matrix @ M_hom
        D_hom = D_hom / D_hom[2]  # Normalize

        return D_hom[:2]

    def choose_method(self) -> str:
        """
        Ask the user which NV mapping method to use: Affine or Homography.
        Returns:
            "affine" or "homography"
        """
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Question)
        msg.setWindowTitle("Select NV Mapping Method")
        msg.setText("Which method would you like to use for NV relocation?")
        msg.setInformativeText("Choose Affine or Homography.")

        msg.setStandardButtons(QMessageBox.NoButton)

        # Add custom buttons
        btn_affine = QPushButton("Affine")
        btn_homography = QPushButton("Homography")
        msg.addButton(btn_affine, QMessageBox.AcceptRole)
        msg.addButton(btn_homography, QMessageBox.AcceptRole)

        # Show dialog and wait for response
        ret = msg.exec()

        clicked_button = msg.clickedButton()
        if clicked_button == btn_affine:
            return "affine"
        else:
            return "homography"

    def choose_homography_method(self) -> str:
        """
        Ask the user which NV mapping method to use: LMEDS, RANSAC, RHOMBUS in Homography.
        Returns:
            "LMEDS", "RANSAC", "RHOMBUS", or "USAC_MAGSAC
        """
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Question)
        msg.setWindowTitle("Select NV Mapping Method")
        msg.setText("Which method would you like to use for NV relocation?")
        msg.setInformativeText("Choose Homography.")

        msg.setStandardButtons(QMessageBox.NoButton)

        # Add custom buttons
        btn_LMEDS = QPushButton("LMEDS")
        btn_RANSAC = QPushButton("RANSAC")
        btn_RHOMBUS = QPushButton("RHOMBUS")
        btn_USAC_MAGSAC = QPushButton("USAC_MAGSAC")
        msg.addButton(btn_LMEDS, QMessageBox.AcceptRole)
        msg.addButton(btn_RANSAC, QMessageBox.AcceptRole)
        msg.addButton(btn_RHOMBUS, QMessageBox.AcceptRole)
        msg.addButton(btn_USAC_MAGSAC, QMessageBox.AcceptRole)
        # Show dialog and wait for response
        ret = msg.exec()

        clicked_button = msg.clickedButton()
        if clicked_button == btn_LMEDS:
            return "LMEDS"
        elif clicked_button == btn_RANSAC:
            return "RANSAC"
        elif clicked_button == btn_RHOMBUS:
            return "RHOMBUS"
        else:
            return "USAC_MAGSAC"

    def find_NV(self) -> np.ndarray:

        self.save_or_find_nv_button_clicked.emit(1)
        path = self.open_file_dialog(self.data_saving_path)
        if not path:
            return
        method = self.choose_method()
        structure = load_data(path)
        if method == "affine":
            """for i, structure in enumerate(Objects._items):"""
            print(f"struct: {structure}")

            old_corners, new_corners, MNV_old = self.extract_corners(structure)
            # Compute transformations
            DMT = self.from_four_corners_to_DMT_or_DMNT(old_corners)
            DMNT = self.from_four_corners_to_DMT_or_DMNT(new_corners)

            MNV_new_direct = self.from_DMT_and_MNV_old_get_DNV_old(MNV_old, np.linalg.inv(DMNT) @ DMT)
            print(f"affine solution: {MNV_new_direct}")
            return MNV_new_direct
        elif method == "homography":
            method = self.choose_homography_method()
            """for i, structure in enumerate(Objects._items):"""
            print(f"struct: {structure}")

            old_corners, new_corners, MNV_old = self.extract_corners(structure)
            H_direct = self.compute_homography_from_corners(old_corners, new_corners, method)
            nv_new = self.map_point_with_homography(MNV_old, H_direct)
            print(f"homography solution with {method} method: {nv_new}")
            return nv_new
        else:
            raise ValueError(f"Method {method} not implemented")

    """def extract_corners(self, structure):
        order = ["top_left", "top_right", "bottom_right", "bottom_left"]

        def get_xy(block, name):
            arr = getattr(block, name)  # StructArray
            if not arr._items:
                raise ValueError(f"No data for {name}")

            pt = arr._items[-1]  # latest snapshot
            print(f"{name} micro_x: {pt.micro_x}, micro_y: {pt.micro_y}")

            return np.array([
                float(pt.micro_x),
                float(pt.micro_y)
            ])

        old_corners = [get_xy(structure.INITIAL, n) for n in order]
        new_corners = [get_xy(structure.FINAL, n) for n in order]
        nv_position = get_xy(structure.INITIAL, "nv")

        return old_corners, new_corners, nv_position"""

    def extract_corners(self, structure):
        order = ["top_left", "top_right", "bottom_right", "bottom_left"]

        def get_xy(block, name):
            pt = getattr(block, name)  # MyStruct directly

            if pt is None:
                raise ValueError(f"No data for {name}")

            print(f"{name} micro_x: {pt.micro_x}, micro_y: {pt.micro_y}")

            return np.array([
                float(pt.micro_x),
                float(pt.micro_y)
            ])
        print(f"structure.initial {structure.INITIAL}")
        print(f"structure.final {structure.FINAL}")
        old_corners = [get_xy(structure.INITIAL, n) for n in order]
        new_corners = [get_xy(structure.FINAL, n) for n in order]
        nv_position = get_xy(structure.INITIAL, "nv")

        return old_corners, new_corners, nv_position
