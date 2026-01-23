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

    # rigid solver:
    import numpy as np

    # ============================================
    # Approach 1: 2×2 Matrix (Rotation only, no translation)
    # ============================================

    def find_transformation_2x2(self, BCK_orig, BCK_transformed):
        """
        Find 2×2 transformation matrix T such that: BCK_transformed = T @ BCK_orig

        Parameters:
        BCK_orig: 2×N array of original points
        BCK_transformed: 2×N array of transformed points

        Returns:
        T: 2×2 transformation matrix
        """

        # For a 2×2 matrix, we need at least 2 points (4 equations for 4 unknowns)
        if BCK_orig.shape[1] < 2:
            raise ValueError("Need at least 2 points for 2×2 transformation")

        # Solve T using least squares: BCK_transformed = T @ BCK_orig
        # Rearrange to: T = BCK_transformed @ pinv(BCK_orig)
        T = BCK_transformed @ np.linalg.pinv(BCK_orig)

        return T

    # ============================================
    # Approach 2: 3×3 Homogeneous Matrix (Rotation + Translation)
    # ============================================

    def find_transformation_homogeneous(self, BCK_orig, BCK_transformed):
        """
        Find 3×3 homogeneous transformation matrix T such that:
        BCK_transformed_homogeneous = T @ BCK_orig_homogeneous

        Parameters:
        BCK_orig: 2×N array of original points
        BCK_transformed: 2×N array of transformed points

        Returns:
        T: 3×3 homogeneous transformation matrix
        """

        # Convert to homogeneous coordinates
        BCK_orig_homog = np.vstack([BCK_orig, np.ones((1, BCK_orig.shape[1]))])
        BCK_trans_homog = np.vstack([BCK_transformed, np.ones((1, BCK_transformed.shape[1]))])

        # For homogeneous 3×3 matrix (affine), we need at least 3 points
        if BCK_orig.shape[1] < 3:
            # Use least squares solution
            A = BCK_orig_homog.T
            B = BCK_trans_homog.T

            # Solve for each row of transformation matrix
            T_rows = []
            for i in range(3):
                coeffs, _, _, _ = np.linalg.lstsq(A, B[:, i], rcond=None)
                T_rows.append(coeffs)

            T = np.array(T_rows).T
        else:
            # Solve exactly if we have enough points
            T = BCK_trans_homog @ np.linalg.pinv(BCK_orig_homog)

        return T

    # ============================================
    # Example usage with known points
    # ============================================

    # Create some example data
    np.random.seed(42)

    # Define a true transformation (rotation + translation)
    angle = np.pi / 4  # 45 degrees
    R_true = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    t_true = np.array([[2.0], [1.0]])  # translation

    print("True rotation matrix R:")
    print(R_true)
    print(f"\nTrue translation vector t: {t_true.flatten()}")

    # Generate some original points
    BCK = np.array([
        [1.0, 2.0, 0.0, -1.0],  # x coordinates
        [0.0, 1.0, -2.0, 2.0]  # y coordinates
    ])

    print(f"\nOriginal points BCK (2×{BCK.shape[1]}):")
    print(BCK)

    # Apply true transformation: BCK_p = R @ BCK + t
    BCK_p = R_true @ BCK + t_true
    print(f"\nTransformed points BCK_p (after R@BCK + t):")
    print(BCK_p)

    # ============================================
    # Method 1: Find 2×2 transformation (rotation only)
    # ============================================
    print("\n" + "=" * 50)
    print("METHOD 1: Finding 2×2 transformation matrix T")
    print("=" * 50)

    T_2x2 = find_transformation_2x2(BCK, BCK_p)
    print("\nFound 2×2 transformation matrix T:")
    print(T_2x2)

    print("\nTrue rotation matrix R:")
    print(R_true)

    print(f"\nDifference from true rotation: {np.max(np.abs(T_2x2 - R_true)):.6f}")

    # Apply transformation to a new point
    BCK_pp = np.array([[3.0], [4.0]])  # New point
    print(f"\nNew point BCK_pp: {BCK_pp.flatten()}")

    BCK_ppp_2x2 = T_2x2 @ BCK_pp
    print(f"Transformed point (T @ BCK_pp): {BCK_ppp_2x2.flatten()}")

    # True transformation (what it should be)
    BCK_pp_true = R_true @ BCK_pp + t_true
    print(f"True transformation (R@BCK_pp + t): {BCK_pp_true.flatten()}")

    # ============================================
    # Method 2: Find 3×3 homogeneous transformation
    # ============================================
    print("\n" + "=" * 50)
    print("METHOD 2: Finding 3×3 homogeneous transformation matrix T")
    print("=" * 50)

    T_homog = find_transformation_homogeneous(BCK, BCK_p)
    print("\nFound 3×3 homogeneous transformation matrix T:")
    print(T_homog)

    # Apply transformation to a new point (using homogeneous coordinates)
    BCK_pp_homog = np.array([[3.0], [4.0], [1.0]])  # New point in homogeneous
    BCK_ppp_homog_full = T_homog @ BCK_pp_homog
    BCK_ppp_homog = BCK_ppp_homog_full[:2, :] / BCK_ppp_homog_full[2, :]

    print(f"\nNew point BCK_pp: {BCK_pp.flatten()}")
    print(f"Transformed point (homogeneous): {BCK_ppp_homog.flatten()}")
    print(f"True transformation (R@BCK_pp + t): {BCK_pp_true.flatten()}")

    # ============================================
    # Helper function for applying transformations
    # ============================================

    def apply_transformation(self, T, points):
        """
        Apply transformation to points

        Parameters:
        T: Transformation matrix (2×2 or 3×3)
        points: 2×N array of points

        Returns:
        transformed_points: 2×N array of transformed points
        """
        if T.shape == (2, 2):
            # 2×2 matrix: linear transformation only
            return T @ points
        elif T.shape == (3, 3):
            # 3×3 homogeneous matrix
            points_homog = np.vstack([points, np.ones((1, points.shape[1]))])
            transformed_homog = T @ points_homog
            return transformed_homog[:2, :] / transformed_homog[2:, :]
        else:
            raise ValueError("T must be 2×2 or 3×3 matrix")

    # ============================================
    # Example with multiple new points
    # ============================================
    print("\n" + "=" * 50)
    print("EXAMPLE WITH MULTIPLE NEW POINTS")
    print("=" * 50)

    # Define several new points
    BCK_pp_multiple = np.array([
        [1.0, 2.0, 0.0, -2.0],  # x coordinates
        [2.0, -1.0, 3.0, 0.0]  # y coordinates
    ])

    print(f"New points BCK_pp (2×{BCK_pp_multiple.shape[1]}):")
    print(BCK_pp_multiple)

    # Apply 2×2 transformation
    BCK_ppp_2x2_multiple = apply_transformation(T_2x2, BCK_pp_multiple)
    print(f"\nTransformed with 2×2 matrix:")
    print(BCK_ppp_2x2_multiple)

    # Apply homogeneous transformation
    BCK_ppp_homog_multiple = apply_transformation(T_homog, BCK_pp_multiple)
    print(f"\nTransformed with homogeneous matrix:")
    print(BCK_ppp_homog_multiple)

    # Calculate true transformation for comparison
    BCK_pp_true_multiple = R_true @ BCK_pp_multiple + t_true
    print(f"\nTrue transformation (R@BCK_pp + t):")
    print(BCK_pp_true_multiple)

    # Calculate errors
    error_2x2 = np.mean(np.linalg.norm(BCK_ppp_2x2_multiple - BCK_pp_true_multiple, axis=0))
    error_homog = np.mean(np.linalg.norm(BCK_ppp_homog_multiple - BCK_pp_true_multiple, axis=0))

    print(f"\nMean error (2×2 method): {error_2x2:.6f}")
    print(f"Mean error (homogeneous method): {error_homog:.6f}")
    print(f"\nNote: The 2×2 matrix doesn't capture translation, so error is larger.")

    # affine solver:
    import numpy as np

    def find_rotation_translation(self, BCK, BCK_p):
        """
        Find R (2×2) and T (2×1) such that: BCK_p = R @ BCK + T

        Parameters:
        BCK: 2×N array of original points
        BCK_p: 2×N array of transformed points

        Returns:
        R: 2×2 rotation matrix
        T: 2×1 translation vector
        """
        # Center the points
        centroid_BCK = np.mean(BCK, axis=1, keepdims=True)
        centroid_BCK_p = np.mean(BCK_p, axis=1, keepdims=True)

        # Center the point sets
        BCK_centered = BCK - centroid_BCK
        BCK_p_centered = BCK_p - centroid_BCK_p

        # Compute R using Singular Value Decomposition (SVD)
        # Solve: BCK_p_centered = R @ BCK_centered
        H = BCK_p_centered @ BCK_centered.T
        U, _, Vt = np.linalg.svd(H)

        # Compute rotation matrix
        R = U @ Vt

        # Ensure proper rotation (det(R) = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = U @ Vt

        # Compute translation
        T = centroid_BCK_p - R @ centroid_BCK

        return R, T

    # ============================================
    # Example with Format
    # ============================================
    np.random.seed(42)

    # Create a true transformation
    angle = np.pi / 3  # 60 degrees
    R_true = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    T_true = np.array([[2.5], [-1.0]])  # 2×1 translation

    print("True Rotation Matrix R (2×2):")
    print(R_true)
    print(f"\nTrue Translation Vector T (2×1): {T_true.flatten()}")

    # Create example points
    BCK = np.array([
        [1.0, 2.0, 0.5],  # x coordinates
        [0.0, 1.0, -1.0]  # y coordinates
    ])
    print(f"\nOriginal points BCK (2×{BCK.shape[1]}):")
    print(BCK)

    # Apply transformation: BCK_p = R @ BCK + T
    BCK_p = R_true @ BCK + T_true
    print(f"\nTransformed points BCK_p = R@BCK + T (2×{BCK_p.shape[1]}):")
    print(BCK_p)

    # ============================================
    # Find R and T from point correspondences
    # ============================================
    R_estimated, T_estimated = find_rotation_translation(BCK, BCK_p)

    print("\n" + "=" * 60)
    print("ESTIMATED TRANSFORMATION")
    print("=" * 60)
    print("\nEstimated Rotation Matrix R (2×2):")
    print(R_estimated)
    print(f"\nEstimated Translation Vector T (2×1): {T_estimated.flatten()}")

    print("\n" + "=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)
    print(f"\nRotation error (Frobenius norm): {np.linalg.norm(R_estimated - R_true):.6f}")
    print(f"Translation error: {np.linalg.norm(T_estimated - T_true):.6f}")

    # ============================================
    # Apply to New Points
    # ============================================

    # Example 1: Single new point
    BCK_pp = np.array([[3.0], [2.0]])  # 2×1 array
    print(f"\n\nNew point BCK_pp (2×1): {BCK_pp.flatten()}")

    # Transform using estimated R and T
    BCK_ppp = R_estimated @ BCK_pp + T_estimated
    print(f"\nTransformed BCK_ppp = R@BCK_pp + T: {BCK_ppp.flatten()}")

    # What it should be with true transformation
    BCK_pp_true = R_true @ BCK_pp + T_true
    print(f"True transformation: R_true@BCK_pp + T_true: {BCK_pp_true.flatten()}")

    # Example 2: Multiple new points
    BCK_pp_multi = np.array([
        [1.0, 2.5, 0.0, -1.0],  # x coordinates
        [2.0, -1.0, 3.0, 0.5]  # y coordinates
    ])

    print(f"\n\nMultiple new points BCK_pp (2×{BCK_pp_multi.shape[1]}):")
    print(BCK_pp_multi)

    # Transform all points
    BCK_ppp_multi = R_estimated @ BCK_pp_multi + T_estimated
    print(f"\nTransformed points (R@BCK_pp + T):")
    print(BCK_ppp_multi)

    # Verify with true transformation
    BCK_ppp_multi_true = R_true @ BCK_pp_multi + T_true
    print(f"\nTrue transformation:")
    print(BCK_ppp_multi_true)

    # Calculate error
    error = np.mean(np.linalg.norm(BCK_ppp_multi - BCK_ppp_multi_true, axis=0))
    print(f"\nMean transformation error: {error:.6f}")

    # ============================================
    # Alternative: Direct Least Squares Solution
    # ============================================
    def find_R_T_direct(self, BCK, BCK_p):
        """
        Direct least squares solution for R and T
        Solve: BCK_p = R @ BCK + T

        This stacks the equations and solves for all parameters at once
        """
        n_points = BCK.shape[1]

        # Setup linear system: vec(BCK_p) = A * vec([R, T])
        # For each point i: [x'_i, y'_i]ᵀ = R * [x_i, y_i]ᵀ + T

        A = []
        b = []

        for i in range(n_points):
            x, y = BCK[:, i]

            # Equations for x' coordinate
            A.append([x, y, 0, 0, 1, 0])  # x' = r11*x + r12*y + t1
            b.append(BCK_p[0, i])

            # Equations for y' coordinate
            A.append([0, 0, x, y, 0, 1])  # y' = r21*x + r22*y + t2
            b.append(BCK_p[1, i])

        A = np.array(A)
        b = np.array(b)

        # Solve least squares
        params, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

        # Extract R and T
        R = params[:4].reshape(2, 2)
        T = params[4:].reshape(2, 1)

        return R, T

    print("\n" + "=" * 60)
    print("DIRECT LEAST SQUARES SOLUTION")
    print("=" * 60)
    R_direct, T_direct = find_R_T_direct(BCK, BCK_p)
    print(f"\nDirect solution R:\n{R_direct}")
    print(f"\nDirect solution T: {T_direct.flatten()}")

    # Test on a new point
    BCK_test = np.array([[1.5], [-0.5]])
    result_direct = R_direct @ BCK_test + T_direct
    result_true = R_true @ BCK_test + T_true
    print(f"\nTest point: {BCK_test.flatten()}")
    print(f"Direct solution result: {result_direct.flatten()}")
    print(f"True result: {result_true.flatten()}")
    print(f"Error: {np.linalg.norm(result_direct - result_true):.6f}")
