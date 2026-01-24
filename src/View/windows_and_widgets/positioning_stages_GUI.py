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
        """grp.attrs["micro_x"] = micro.get_position("x")
        grp.attrs["micro_y"] = micro.get_position("y")

        grp.attrs["nano_x"] = nano.get_position("x")
        grp.attrs["nano_y"] = nano.get_position("y")
        grp.attrs["nano_z"] = nano.get_position("z")"""
        grp.attrs["micro_x"] = self.xlineEdit_2.text()
        grp.attrs["micro_y"] = self.ylineEdit_2.text()

        grp.attrs["nano_x"] = self.xlineEdit_1.text()
        grp.attrs["nano_y"] = self.ylineEdit_1.text()
        grp.attrs["nano_z"] = self.zlineEdit_1.text()

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

    import numpy as np
    from typing import Tuple, List

    def get_micro_coords(self, corner_dict):
        """Extract micro_x and micro_y from dictionary, convert to float"""
        if isinstance(corner_dict, np.ndarray) and corner_dict.dtype == object:
            corner_dict = corner_dict.item()  # Extract dict from array

        micro_x = float(corner_dict.get('micro_x', corner_dict.get('micro_X', 0)))
        micro_y = float(corner_dict.get('micro_y', corner_dict.get('micro_Y', 0)))
        return micro_x, micro_y


    def from_four_corners_to_DMT_or_DMNT(self,
            corners_microdrive: List[np.ndarray],
            reference_order: Tuple[str] = ("top_left", "top_right", "bottom_right", "bottom_left")
    ) -> np.ndarray:
        """
        Compute transformation matrix from microdrive coordinates to diamond coordinates
        using four corner points.


        Parameters:
        -----------
        corners_microdrive : List[np.ndarray]
            List of 4 points in microdrive coordinates in order: [top_left, top_right, bottom_right, bottom_left]
            Each point should be a numpy array of shape (2,) for (x, y)


        reference_order : Tuple[str], optional
            Order of the corners as they appear in the corners_microdrive list


        Returns:
        --------
        T : np.ndarray
            3x3 transformation matrix such that:
            [x_diamond, y_diamond, 1]^T = T @ [x_microdrive, y_microdrive, 1]^T
            from microdrive coordinate system to diamond coordinate system
            DMT is diamond microdrive transformation matrix
            DMNT is diamond microdrive new transformation matrix


        Notes:
        ------
        Diamond coordinate system definition:
        - Origin: at bottom_left corner
        - x-axis: from bottom_left to bottom_right
        - y-axis: from bottom_left to top_left
        This assumes the corners form a (possibly rotated) rectangle in microdrive coords.
        """
        # Extract points in the specified order


        # Extract micro coordinates from dictionaries


        print("here1.3")
        print(f"bottom_left: {bottom_left}, bottom_right: {bottom_right}, top_left: {top_left}, top_right: {top_right}")

        # Diamond coordinate system definition:
        # In diamond coords:
        # bottom_left = (0, 0)
        # bottom_right = (1, 0)
        # top_left = (0, 1)
        # top_right = (1, 1)
        print("here1.4")
        # Source points in microdrive coordinates (homogeneous)
        src_points = np.array([
            [bottom_left[0], bottom_left[1], 1],
            [bottom_right[0], bottom_right[1], 1],
            [top_left[0], top_left[1], 1]
        ]).T  # Shape: (3, 3)
        print("here1.5")
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
            print("here1.6")
            T = dst_points @ np.linalg.inv(src_points)
        except np.linalg.LinAlgError:
            raise ValueError("Corners are colinear or form a degenerate shape")
        print("here1.7")
        # Verify with the fourth point
        print("here1.8")
        tr_hom = np.array([top_right[0], top_right[1], 1])
        print("here1.9")
        tr_diamond_pred = T @ tr_hom
        print("here1.10")
        # Should be close to (1, 1, 1) after normalization
        tr_diamond_pred = tr_diamond_pred / tr_diamond_pred[2]
        print("here1.11")
        # Check if the shape is reasonably rectangular
        expected = np.array([1, 1, 1])
        print("here1.12")
        if np.linalg.norm(tr_diamond_pred - expected) > 1e-3:
            print("here1.13")
            print(f"Warning: Fourth corner verification failed. "
                  f"Expected (1, 1, 1), got ({tr_diamond_pred[0]:.3f}, {tr_diamond_pred[1]:.3f}, {tr_diamond_pred[2]:.3f})")
            print("This suggests corners don't form a proper rectangle or order is wrong.")
        print("here1.14")
        return T

    def from_DMT_and_MNV_old_get_DNV_old(self, M_point: np.ndarray, T_matrix: np.ndarray) -> np.ndarray:
        """
        Transform a point from microdrive to diamond coordinates.
        Handles both dict and array(dict) formats.
        """
        # Extract micro coordinates
        if isinstance(M_point, np.ndarray) and M_point.dtype == object:
            M_point = M_point.item()  # Extract dict from array

        if isinstance(M_point, dict):
            # Extract micro coordinates
            micro_x = float(M_point.get('micro_x', M_point.get('micro_X', 0)))
            micro_y = float(M_point.get('micro_y', M_point.get('micro_Y', 0)))
        elif isinstance(M_point, (list, tuple, np.ndarray)) and len(M_point) >= 2:
            micro_x, micro_y = float(M_point[0]), float(M_point[1])
        else:
            raise ValueError(f"Unexpected point format: {type(M_point)}")

        M_hom = np.array([micro_x, micro_y, 1.0])
        D_hom = T_matrix @ M_hom
        D_hom = D_hom / D_hom[2]  # Normalize

        return D_hom[:2]

    def find_NV(self):
        """Complete function with transformation calculations to find NV center"""
        self.save_or_find_nv_button_clicked.emit(1)
        path = self.open_file_dialog(self.data_saving_path)
        if not path:
            return

        # read file
        self.data_reader = ExperimentHDF5ReaderSWMR(path)
        self.data_reader.read_structure()
        structure = self.data_reader.get_structure()

        required_corners = ("top_left", "top_right", "bottom_left", "bottom_right")

        init = structure.get("INITIAL", {})
        final = structure.get("FINAL", {})

        # Validation
        if not all(c in init for c in required_corners):
            self.error_box("INITIAL does not contain all 4 corners",
                           "please add all 4 corners before proceeding")
            self.data_reader.close()
            return

        if not all(c in final for c in required_corners):
            self.error_box("FINAL does not contain all 4 corners",
                           "please add all 4 corners before proceeding")
            self.data_reader.close()
            return

        if "nv" not in init:
            self.error_box("NV does not exist in INITIAL",
                           "please NV point before proceeding")
            self.data_reader.close()
            return

        # Prepare points in consistent order
        old_corners_microdrive = [
            np.array(init["top_left"]),
            np.array(init["top_right"]),
            np.array(init["bottom_left"]),
            np.array(init["bottom_right"])
        ]

        new_corners_microdrive = [
            np.array(final["top_left"]),
            np.array(final["top_right"]),
            np.array(final["bottom_left"]),
            np.array(final["bottom_right"])
        ]

        old_nv_microdrive = np.array(init["nv"])

        print(f"Old corners: {old_corners_microdrive}")
        print(f"New corners: {new_corners_microdrive}")
        print(f"Old NV in microdrive: {old_nv_microdrive}")

        ###
        print("here1.1")
        if len(old_corners_microdrive) != 4:
            raise ValueError(f"Expected 4 corners, got {len(old_corners_microdrive)}")

        corners_dicts = []
        for corner_array in old_corners_microdrive:
            if isinstance(corner_array, np.ndarray) and corner_array.dtype == object:
                # Extract the dictionary from the numpy array
                corner_dict = corner_array.item()
                corners_dicts.append(corner_dict)
            else:
                corners_dicts.append(corner_array)

        # Replace the input with extracted dictionaries
        old_corners_microdrive = corners_dicts

        print(f"Debug: Extracted corners_dicts = {corners_dicts}")


        print("here1.1")
        if len(new_corners_microdrive) != 4:
            raise ValueError(f"Expected 4 corners, got {len(new_corners_microdrive)}")

        corners_dicts = []
        for corner_array in new_corners_microdrive:
            if isinstance(corner_array, np.ndarray) and corner_array.dtype == object:
                # Extract the dictionary from the numpy array
                corner_dict = corner_array.item()
                corners_dicts.append(corner_dict)
            else:
                corners_dicts.append(corner_array)

        # Replace the input with extracted dictionaries
        new_corners_microdrive = corners_dicts
        ###
        print(f"Debug: Extracted corners_dicts = {corners_dicts}")
        reference_order = ("top_left", "top_right", "bottom_right", "bottom_left")
        bl_idx = reference_order.index("bottom_left")
        br_idx = reference_order.index("bottom_right")
        tl_idx = reference_order.index("top_left")
        tr_idx = reference_order.index("top_right")
        print("here1.2")

        bl_x, bl_y = self.get_micro_coords(old_corners_microdrive[bl_idx])
        br_x, br_y = self.get_micro_coords(old_corners_microdrive[br_idx])
        tl_x, tl_y = self.get_micro_coords(old_corners_microdrive[tl_idx])
        tr_x, tr_y = self.get_micro_coords(old_corners_microdrive[tr_idx])
        bottom_left = np.array([bl_x, bl_y], dtype=float)
        bottom_right = np.array([br_x, br_y], dtype=float)
        top_left = np.array([tl_x, tl_y], dtype=float)
        top_right = np.array([tr_x, tr_y], dtype=float)

        bl_x, bl_y = self.get_micro_coords(new_corners_microdrive[bl_idx])
        br_x, br_y = self.get_micro_coords(new_corners_microdrive[br_idx])
        tl_x, tl_y = self.get_micro_coords(new_corners_microdrive[tl_idx])
        tr_x, tr_y = self.get_micro_coords(new_corners_microdrive[tr_idx])
        bottom_left = np.array([bl_x, bl_y], dtype=float)
        bottom_right = np.array([br_x, br_y], dtype=float)
        top_left = np.array([tl_x, tl_y], dtype=float)
        top_right = np.array([tr_x, tr_y], dtype=float)

        # Compute transformation matrices
        try:
            print("here 0")
            # DMT: transform from microdrive to diamond coordinates (old position)
            DMT = self.from_four_corners_to_DMT_or_DMNT(
                old_corners_microdrive
            )
            print("here 1")
            # DMNT: transform from microdrive to diamond coordinates (new position)
            DMNT = self.from_four_corners_to_DMT_or_DMNT(
                new_corners_microdrive
            )
            print("here 2")
            # Transform old NV position to diamond coordinates
            nv_diamond = self.from_DMT_and_MNV_old_get_DNV_old(old_nv_microdrive, DMT)
            print(f"NV in diamond coordinates: {nv_diamond}")
            print(f"old_corners_microdrive {old_corners_microdrive}")

            # Transform back to microdrive coordinates using new diamond position
            # To go from diamond to new microdrive: M_new = DMNT^{-1} * D
            # But we have D from DMT * M_old, so:
            # M_new = DMNT^{-1} * DMT * M_old

            DMNT_inv = np.linalg.inv(DMNT)
            print("here 3")
            new_nv_microdrive_hom = DMNT_inv @ DMT @ np.array([old_corners_microdrive[0], old_corners_microdrive[1], 1.0])
            print("here 4")
            new_nv_microdrive = new_nv_microdrive_hom[:2] / new_nv_microdrive_hom[2]

            print(f"Predicted new NV position in microdrive: {new_nv_microdrive}")

            nv_diamond_hom = DMT @ np.array([old_corners_microdrive[0], old_corners_microdrive[1], 1.0])
            print("here 5")
            nv_diamond_hom = nv_diamond_hom / nv_diamond_hom[2]
            print("here 6")

            new_nv_microdrive_hom2 = DMNT_inv @ nv_diamond_hom
            print("here 7")
            new_nv_microdrive2 = new_nv_microdrive_hom2[:2] / new_nv_microdrive_hom2[2]

            print(f"Alternative calculation: {new_nv_microdrive2}")

            # Return the result
            return new_nv_microdrive

        except Exception as e:
            self.error_box("Transformation error", f"Failed to compute transformations: {str(e)}")
            self.data_reader.close()
            return None
        finally:
            self.data_reader.close()