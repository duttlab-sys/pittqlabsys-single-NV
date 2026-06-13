# Created by Jannet Trabelsi on 2025-10-10
from __future__ import annotations
import sys
from src.View.windows_and_widgets.camera_widget import Amscope_Camera_View
from src.View.windows_and_widgets.camera_widget import ROPER_CASCADE_CCD_View
from src.View.windows_and_widgets.display_design import Ui_Form
import pyqtgraph as pg
import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QMessageBox,
    QSlider,
    QLineEdit,
    QPushButton,
)
# Assuming the .ui file is converted to design.py
#To convert display_design.ui to .py, paste this into the terminal:
# pyuic5 -x display_design.ui -o display_design.py

# constants:
class Display_View(QWidget, Ui_Form):
    """
    This is the widget of the positioning stages. It allows us to control positioning devices using buttons and LineEdits
    """
    x_crosshair = pyqtSignal(int)
    y_crosshair = pyqtSignal(int)
    # ---- AutoTune config (Roper only). EDIT the sequences to match your hardware. ----
    AUTOTUNE_TARGET_MIN = 15000
    AUTOTUNE_TARGET_MAX = 39000
    AUTOTUNE_SATURATION = 40000
    # both arrays ordered from LEAST to MOST exposure
    ROPER_NOGAIN_SEQUENCE = [10, 50, 100, 150, 200, 300, 500]  # inttime (us)
    ROPER_GAIN_SEQUENCE = [(10, 1), (50, 1), (100, 1), (100, 100),
                           (100, 500), (100, 1000), (100, 1500),
                           (100, 2000), (100, 2500), (100, 3000), (150, 3000), (200, 3000), (300, 3000), (400, 3000), (500, 3000)]
    update_get_img = pyqtSignal(int)
    def __init__(self, display_choice = "MU300", snapshot_or_live = 1, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.widget = None
        self.cascade_controls = None  # container for the Roper line-edits/buttons
        self._autotune_mode = None
        self._autotune_seq = None
        self._autotune_index = 0
        #self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        if hasattr(self, 'horizontalLayout'):
            self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
            self.horizontalLayout.setSpacing(0)
        self.parent_widget.setStyleSheet("background-color: white;")

        self.display_choice = display_choice
        self.snapshot_or_live = snapshot_or_live
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.acquire_and_plot_data)
        # plots
        self.z_vs_x_plot = pg.PlotWidget(title="z vs x")
        self.z_vs_x_plot.setLabel('left', 'z')
        self.z_vs_x_plot.setLabel('bottom', 'x')
        self.z_vs_x_widget.setLayout(QVBoxLayout())
        self.z_vs_x_widget.layout().addWidget(self.z_vs_x_plot)

        # Set up plot
        self.z_vs_y_plot = pg.PlotWidget(title="z vs y")
        self.z_vs_y_plot.setLabel('left', 'y')  # y is now on vertical axis
        self.z_vs_y_plot.setLabel('top', 'z')  # z is now on horizontal axis

        # Invert the Y-axis to make it go top to bottom
        self.z_vs_y_plot.invertY(True)

        # Add the plot to your layout
        self.z_vs_y_widget.setLayout(QVBoxLayout())
        self.z_vs_y_widget.layout().addWidget(self.z_vs_y_plot)
        self.crosshair_y.setValue(0) # THIS WILL LATER LOAD WITH CONFIG FILE
        self.crosshair_x.setValue(0) # THIS WILL LATER LOAD WITH CONFIG FILE
        self.crosshair_width.setValue(1)

        # Initialize data arrays for plotting
        self.x = []
        self.y = []
        self.z_x = []
        self.z_y = []

        # Initialize plot
        self.zx_plot = self.z_vs_x_plot.plot(pen='r', name='zx')
        self.zy_plot = self.z_vs_y_plot.plot(pen='r', name='zy')
        self.w = 680
        self.h = 510
        # Connect buttons to functions
        self.crosshairButton.clicked.connect(self.crosshair)
        self.center_Button.clicked.connect(self.center)
        self.clear_crosshair_Button.clicked.connect(self.clear_crosshair)
        self.AutoTune_checkBox.toggled.connect(self.on_autotune_toggled)
        self.connect_to_display()
        self.start()
        self.crosshair_x.setMaximum(self.w)
        self.crosshair_x.setMinimum(0.1)
        self.crosshair_y.setMaximum(self.h)
        self.crosshair_y.setMinimum(0.1)
        self.widget.mouseMoved.connect(self.on_widget_hover)
        self.widget.mouseClicked.connect(self.on_widget_click)
        self.crosshair_frozen = False  # Default: move with hover
        self.crosshair_x.valueChanged.connect(self.on_crosshair_changed)
        self.crosshair_y.valueChanged.connect(self.on_crosshair_changed)
        self.crosshair_width.valueChanged.connect(self.on_crosshair_changed)
        self.x_selected = 0
        self.y_selected = 0

    def update_choices(self, display_choice, snapshot_or_live):
        # this function gets the signals from main (that are emitted by the positioning class) and only updates the display of the choices are different from what we have
        if display_choice != self.display_choice:
            print("update_choices called display choice changed")
            self.update_timer.stop()
            self.display_choice = display_choice
            self.connect_to_display()
        if snapshot_or_live != self.snapshot_or_live:
            print("update_choices called snapshot_or_live changed to:" + str(snapshot_or_live))
            self.update_timer.stop()
            self.snapshot_or_live = snapshot_or_live
        self.start()

    def update_color_scale_option(self, new_color_scale_choice):
        if self.display_choice == "CASCADE CCD":
            self.widget.color_scale_choice = new_color_scale_choice

    def connect_to_display(self):
        """This function connects the devices: please make sure that the stage has the function get_position(self, axis)"""
        # remove any previously-loaded camera view so they don't stack/overlap
        if self.widget is not None:
            try:
                self.widget.stop_live_view()
            except Exception:
                pass
            self.verticalLayout.removeWidget(self.widget)
            self.widget.setParent(None)
            self.widget.deleteLater()
            self.widget = None
        if self.display_choice == 'MU300':
            self.crosshairButton.setEnabled(True)
            # future users: you can do more (make sure you add those options in the positioning_design.ui file)
            try:
                self.widget = Amscope_Camera_View()
                self.verticalLayout.addWidget(self.widget)
                if self.widget is not None:
                    QMessageBox.information(self, 'Success', f'Connected to: {self.display_choice}')
                else:
                    QMessageBox.critical(self, 'Error', f'Could not connect to: {self.display_choice}')
            except Exception as e:
                QMessageBox.critical(self, 'Error', str(e))
        elif self.display_choice == 'CASCADE CCD':
            self.crosshairButton.setEnabled(True)
            try:
                self.widget = ROPER_CASCADE_CCD_View()
                self.verticalLayout.addWidget(self.widget)
                if self.widget is not None:
                    QMessageBox.information(self, 'Success', f'Connected to: {self.display_choice}')
                else:
                    QMessageBox.critical(self, 'Error', f'Could not connect to: {self.display_choice}')
            except Exception as e:
                QMessageBox.critical(self, 'Error', str(e))
        else:
            return

    def _update_get_image_button(self):
        is_roper = (self.display_choice == 'CASCADE CCD')
        enabled = (is_roper and self.snapshot_or_live == 0)
        self.update_get_img.emit(enabled)
        self.AutoTune_checkBox.setEnabled(is_roper)

    def on_autotune_toggled(self, checked):
        if not checked:  # ignore the un-check
            return
        if self.display_choice != 'CASCADE CCD':
            self.AutoTune_checkBox.setChecked(False)
            return

        # 1) Gain / No gain / Cancel
        box = QMessageBox(self)
        box.setWindowTitle("AutoTune")
        box.setText("Choose AutoTune mode:")
        gain_btn = box.addButton("Gain", QMessageBox.AcceptRole)
        nogain_btn = box.addButton("No gain", QMessageBox.AcceptRole)
        cancel_btn = box.addButton("Cancel", QMessageBox.RejectRole)
        box.exec_()
        clicked = box.clickedButton()

        if clicked is cancel_btn or clicked is None:
            self.AutoTune_checkBox.setChecked(False)  # deselect
            return

        if clicked is gain_btn:
            self._autotune_mode = "gain"
            seq = self.ROPER_GAIN_SEQUENCE
            labels = [f"{i}: inttime={it} us, gain={g}" for i, (it, g) in enumerate(seq)]
        else:
            self._autotune_mode = "nogain"
            seq = self.ROPER_NOGAIN_SEQUENCE
            labels = [f"{i}: inttime={it} us" for i, it in enumerate(seq)]

        # 2) pick the starting point in the chosen array
        from PyQt5.QtWidgets import QInputDialog
        label, ok = QInputDialog.getItem(
            self, "AutoTune start", "Select the starting point:", labels, 0, False)
        if not ok:
            self.AutoTune_checkBox.setChecked(False)
            return

        self._autotune_seq = seq
        self._autotune_index = labels.index(label)

        # 3) run the tuning loop
        self._autotune_run()

    def _apply_autotune_setting(self, setting):
        if self._autotune_mode == "gain":
            inttime, gain = setting
            self.widget.hcam.update({'inttime': float(inttime), 'gain': float(gain)})
        else:
            self.widget.hcam.update({'inttime': float(setting)})

    def _autotune_run(self):
        seq = self._autotune_seq
        idx = self._autotune_index
        frame = None
        max_pixel = None
        for _ in range(2 * len(seq)):  # bounded; can't spin forever
            self._apply_autotune_setting(seq[idx])
            QApplication.processEvents()  # keep UI alive during exposures
            frame = self.widget.get_latest_frame()
            if frame is None:
                break
            max_pixel = float(np.max(frame))
            if max_pixel >= self.AUTOTUNE_SATURATION or max_pixel > self.AUTOTUNE_TARGET_MAX:
                if idx > 0:
                    idx -= 1  # too bright -> less exposure
                    continue
                break  # already at the lowest setting
            elif max_pixel < self.AUTOTUNE_TARGET_MIN:
                if idx < len(seq) - 1:
                    idx += 1  # too dim -> more exposure
                    continue
                break  # already at the highest setting
            else:
                break  # in the ideal 15k-39k window
        self._autotune_index = idx

        # reflect the chosen values in the line-edits (if the cascade controls exist)
        try:
            if self._autotune_mode == "gain":
                it, g = seq[idx]
                self.inttime_edit.setText(str(it))
                self.gain_edit.setText(str(g))
            else:
                self.inttime_edit.setText(str(seq[idx]))
        except Exception:
            pass

        # show the tuned frame and refresh the plots
        if frame is not None:
            self.widget.show_frame(frame)
            self.img_gray = frame
            self.h, self.w = frame.shape
            self.draw_crosshair(self.x_selected, self.y_selected)

        if max_pixel is not None:
            QMessageBox.information(
                self, "AutoTune",
                f"Final max pixel: {int(max_pixel)} "
                f"(target {self.AUTOTUNE_TARGET_MIN}-{self.AUTOTUNE_TARGET_MAX}).")

    def get_image_snapshot(self):
        """Roper snapshot: grab a full image with read_probes('image') and overwrite the display."""
        try:
            raw = self.widget.hcam.read_probes("image")
        except Exception as e:
            QMessageBox.warning(self, "Get image", f"Could not collect image: {e}")
            return

        # MATLAB double -> numpy (column-major), same as get_latest_frame
        try:
            frame = np.array(raw._data, dtype=np.float64).reshape(raw.size, order="F")
        except Exception:
            frame = np.asarray(raw, dtype=np.float64)

        # overwrite whatever was showing
        self.widget.show_frame(frame)

        # refresh the z-vs-x / z-vs-y plots from the new image
        self.img_gray = frame
        self.h, self.w = frame.shape
        self.draw_crosshair(self.x_selected, self.y_selected)

    # 0 means snapshot
    # 1 means live
    def start(self):
        self._update_get_image_button()
        if self.snapshot_or_live == 0:
            self.update_timer.stop()
            if self.widget is not None and self.display_choice == 'CASCADE CCD':
                try:
                    self.widget.stop_live_view()
                except Exception:
                    pass
        else:
            self.widget.start_live_view()
            if self.display_choice == 'MU300':
                self._set_sliders_visible(True)
                self._remove_cascade_controls()
                self.build_sliders()
            elif self.display_choice == 'CASCADE CCD':
                self._set_sliders_visible(False)
                self.build_cascade_controls()
            try:
                self.update_timer.start(500)
            except ValueError as e:
                print(f"ValueError: {e}")
                QMessageBox.warning(self, 'Warning', 'Invalid numeric input')
            except Exception as e:
                print(f"Exception: {e}")
                QMessageBox.warning(self, 'Warning', f'Unexpected error: {e}')

    def build_sliders(self):
        params = [
            "exposure gain", "exposure time", "brightness", "saturation",
            "contrast", "Gamma", "Temp", "Tint", "Hue"]
        i = 9
        for name in params:
            i+=1
            current_val = self.widget.hcam.read_probes(name)
            min_value, max_value = self.widget.hcam.return_min_max(name)
            slider = getattr(self, f"horizontalSlider_{i}")
            self._add_slider(name, slider, min_value, max_value, current_val)

    def _set_sliders_visible(self, visible):
        # the MU300 sliders are horizontalSlider_10 .. _18 in the .ui
        for i in range(10, 19):
            slider = getattr(self, f"horizontalSlider_{i}", None)
            if slider is not None:
                slider.setVisible(visible)

    def build_cascade_controls(self):
        # rebuild from scratch so they never stack when you switch/restart
        self._remove_cascade_controls()

        # prefill the fields with the camera's current values
        try:
            cur_inttime = self.widget.hcam.read_probes("inttime")
        except Exception:
            cur_inttime = ""
        try:
            cur_gain = self.widget.hcam.read_probes("gain")
        except Exception:
            cur_gain = ""

        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(0, 0, 0, 0)

        # --- integration time row ---
        self.inttime_edit = QLineEdit(str(cur_inttime))
        inttime_btn = QPushButton("Set integration time")
        inttime_btn.clicked.connect(self._apply_cascade_inttime)
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Integration time (us):"))
        row1.addWidget(self.inttime_edit)
        row1.addWidget(inttime_btn)
        vbox.addLayout(row1)

        # --- gain row ---
        self.gain_edit = QLineEdit(str(cur_gain))
        gain_btn = QPushButton("Set gain")
        gain_btn.clicked.connect(self._apply_cascade_gain)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Gain:"))
        row2.addWidget(self.gain_edit)
        row2.addWidget(gain_btn)
        vbox.addLayout(row2)

        self.cascade_controls = container
        self.verticalLayout.addWidget(container)

    def _remove_cascade_controls(self):
        container = getattr(self, "cascade_controls", None)
        if container is not None:
            self.verticalLayout.removeWidget(container)
            container.setParent(None)
            container.deleteLater()
            self.cascade_controls = None

    def _apply_cascade_inttime(self):
        text = self.inttime_edit.text().strip()
        try:
            value = float(text)
        except ValueError:
            QMessageBox.warning(self, "Invalid input", f"'{text}' is not a valid number.")
            return
        try:
            self.widget.hcam.update({'inttime': value})
        except Exception as e:
            QMessageBox.warning(self, "Camera error", f"Could not set integration time: {e}")

    def _apply_cascade_gain(self):
        text = self.gain_edit.text().strip()
        try:
            value = float(text)
        except ValueError:
            QMessageBox.warning(self, "Invalid input", f"'{text}' is not a valid number.")
            return
        try:
            self.widget.hcam.update({'gain': value})
        except Exception as e:
            QMessageBox.warning(self, "Camera error", f"Could not set gain: {e}")

    def _add_slider(self, name, slider, min_value, max_value, current_val):
        value_label = QLabel(str(current_val))
        slider.setMinimum(min_value)
        slider.setMaximum(max_value)
        slider.setValue(current_val)

        def on_change(value):
            value_label.setText(str(value))
            self.widget.hcam.update({name: value})
        slider.valueChanged.connect(on_change)

    def acquire_and_plot_data(self):
        try:
            img_rgb = self.widget.get_latest_frame()
            if img_rgb is None:
                return
            if img_rgb.ndim == 2:
                # Roper Cascade: already 2-D grayscale (raw CCD counts)
                self.h, self.w = img_rgb.shape
                self.img_gray = img_rgb
            else:
                # Amscope: RGB -> grayscale (unchanged)
                self.h, self.w, _ = img_rgb.shape
                self.img_gray = np.dot(img_rgb[..., :3], [0.2989, 0.5870, 0.1140])
            # Crosshair center coordinates
            x = int(self.crosshair_x.value())
            y = int(self.crosshair_y.value())
            # Crosshair thickness (averaging width)
            width = int(self.crosshair_width.value())
            half_w = max(1, width // 2)
            # Ensure bounds don’t exceed image size
            y_min = max(0, y - half_w)
            y_max = min(self.h, y + half_w)
            x_min = max(0, x - half_w)
            x_max = min(self.w, x + half_w)
            # Average along the band around the crosshair
            # z vs x = average intensity across horizontal stripe
            self.z_x = np.mean(self.img_gray[y_min:y_max, :], axis=0)
            # z vs y = average intensity across vertical stripe
            self.z_y = np.mean(self.img_gray[:, x_min:x_max], axis=1)
            # Axes
            self.x = np.arange(self.w)
            self.y = np.arange(self.h)
            self.update_plot()

        except Exception as e:
            print(f"Error acquiring data: {e}")
            self.update_timer.stop()
            QMessageBox.critical(self, "Acquisition Error", str(e))

    def update_plot(self):
        if self.x is None or self.y is None or self.z_x is None or self.z_y is None:
            return
        if len(self.x) == 0 or len(self.y) == 0 or len(self.z_x) == 0 or len(self.z_y) == 0:
            return
        self.zx_plot.setData(self.x, self.z_x)
        self.zy_plot.setData(self.z_y, self.y)

    def close(self):
        if self.snapshot_or_live == 1:
            self.widget.stop_live_view()
            self.widget.stop()

    def plot_clicked(self, mouse_event):
        if mouse_event.button() == Qt.LeftButton:
            viewbox = self.plot_widget.getViewBox()
            mouse_point = viewbox.mapSceneToView(mouse_event.scenePos())
            self.x_selected = mouse_point.x()
            self.y_selected = mouse_point.y()

    def crosshair(self):
        # as you hover coordinates change
        self.widget.label.waiting_for_crosshair_click(True)
        self.crosshair_frozen = False
        self.crosshair_x.setValue(self.x_selected)
        self.crosshair_y.setValue(self.y_selected)
        # Once clicked on point: CALL self.draw_crosshair(x, y)
        self.draw_crosshair(self.x_selected, self.y_selected)

    def draw_crosshair(self, x, y):
        thickness = int(self.crosshair_width.value())
        #self.widget.draw_crosshair(x, y, thickness)
        self.widget.label.enable_crosshair(x, y, thickness)
        if self.snapshot_or_live == 0:
            # Crosshair center coordinates
            x = int(self.crosshair_x.value())
            y = int(self.crosshair_y.value())
            # Crosshair thickness (averaging width)
            width = int(self.crosshair_width.value())
            half_w = max(1, width // 2)
            # Ensure bounds don’t exceed image size
            y_min = max(0, y - half_w)
            y_max = min(self.h, y + half_w)
            x_min = max(0, x - half_w)
            x_max = min(self.w, x + half_w)
            # Average along the band around the crosshair
            # z vs x = average intensity across horizontal stripe
            self.z_x = np.mean(self.img_gray[y_min:y_max, :], axis=0)
            # z vs y = average intensity across vertical stripe
            self.z_y = np.mean(self.img_gray[:, x_min:x_max], axis=1)
            # Axes
            self.x = np.arange(self.w)
            self.y = np.arange(self.h)
            self.update_plot()

    def center(self):
        x_center = self.w // 2
        y_center = self.h // 2
        self.crosshair_x.setValue(x_center)
        self.crosshair_y.setValue(y_center)
        self.draw_crosshair(x_center, y_center)

    def on_widget_hover(self, x, y):
        if not self.crosshair_frozen:
            self.crosshair_x.setValue(x)
            self.crosshair_y.setValue(y)

    def on_widget_click(self, x, y):
        # set the values on the QDoubleSpinBoxes and no update as you hover over the widget anymore
        self.crosshair_frozen = True
        self.x_selected = x
        self.y_selected = y
        self.crosshair_x.setValue(x)
        self.crosshair_y.setValue(y)
        self.draw_crosshair(x, y)
        self.widget.label.waiting_for_crosshair_click(False)

    # as inc/ dec buttons are clicked:
    def on_crosshair_changed(self):
        x = int(self.crosshair_x.value())
        y = int(self.crosshair_y.value())
        self.x_selected = x
        self.y_selected = y
        self.draw_crosshair(x, y)
        self.x_crosshair.emit(x)
        self.y_crosshair.emit(y)

    def clear_crosshair(self):
        self.widget.label.disable_crosshair()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Display_View()
    w.show()
    sys.exit(app.exec())