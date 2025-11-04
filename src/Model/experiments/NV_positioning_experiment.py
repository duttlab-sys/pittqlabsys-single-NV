from __future__ import annotations
import sys
import ctypes
import time
from typing import Optional
from src.Controller import amcam
from src.Controller import Amscope_MU_Camera
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QTimer
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QCheckBox,
    QVBoxLayout,
    QHBoxLayout,
    QDesktopWidget,
    QMessageBox,
)
from src.core import Parameter, Experiment
from src.core.helper_functions import get_configured_nv_positioning_folder
from time import sleep
import pyqtgraph as pg
import numpy as np
from pyqtgraph.exporters import ImageExporter
from pathlib import Path
class NV_positioning_experiment(Experiment):
    '''
    NV_positioning_experiment:
    Using microdrive, nanodrive, and camera, this experiment:
     - shows live imaging
     - moves the microdrive to corners of the sample
     - lets the user select 4 corners of the sample (4 pictures)
     - Goes to NV center
     - lets user select NV location
     - After taking the sample and putting it back, lets the user select 4 corners of the sample
     - applies algorithm to find NV
     - moves microdrive to and then nanodrive to NV
    Hardware Dependencies:
    - Newport_Conex_CC
    - Amscope camera
    - Nanodrive
    '''

    _DEFAULT_SETTINGS = [
        Parameter('old_bottom_left',
                  [Parameter('x',50.0,float,' old_bottom_left x-coordinate in microns'),
                   Parameter('y',5.0,float,'old_bottom_left y-coordinate in microns')
                   ]),
        Parameter('old_bottom_right',
                  [Parameter('x',50.0,float, 'old_bottom_right x-coordinate in microns'),
                   Parameter('y', 50.0, float, 'old_bottom_right y-coordinate in microns')
                   ]),
        Parameter('old_top_left',
                  [Parameter('x',50.0,float,'old_top_left x-coordinate in microns'),
                   Parameter('y', 50.0, float, 'old_top_left y-coordinate in microns')
                   ]),
        Parameter('old_top_right',
                  [Parameter('x',50.0,float,'old_top_right x-coordinate in microns'),
                   Parameter('y', 50.0, float, 'old_top_right y-coordinate in microns')
                   ]),
        Parameter('new_bottom_left',
                  [Parameter('x', 50.0, float, 'new_bottom_left x-coordinate in microns'),
                   Parameter('y', 5.0, float, 'new_bottom_left y-coordinate in microns')
                   ]),
        Parameter('new_bottom_right',
                  [Parameter('x', 50.0, float, 'new_bottom_right x-coordinate end in microns'),
                   Parameter('y', 50.0, float, 'new_bottom_right y-coordinate end in microns')
                   ]),
        Parameter('new_top_left',
                  [Parameter('x', 50.0, float, 'new_top_left x-coordinate end in microns'),
                   Parameter('y', 50.0, float, 'new_top_left y-coordinate end in microns')
                   ]),
        Parameter('new_top_right',
                  [Parameter('x', 50.0, float, 'new_top_right x-coordinate in microns'),
                   Parameter('y', 50.0, float, 'new_top_right y-coordinate in microns')
                   ]),
        Parameter('old_NV_location',
                  [Parameter('x', 50.0, float, 'old_NV_location x-coordinate in microns'),
                   Parameter('y', 50.0, float, 'old_NV_location y-coordinate in microns'),
                   Parameter('z', 50.0, float, 'old_NV_location z-coordinate in microns')
                   ]),
        Parameter('positions',
                  [Parameter('folderpath', str(get_configured_nv_positioning_folder()), str,
                             'folder location to save positions of points')]),
        Parameter('z_pos',50.0,float,'z position of nanodrive; useful for z-axis sweeps to find NVs'),
        Parameter('resolution', 1.0, [2.0,1.0,0.5,0.25,0.1,0.05,0.025,0.001], 'Resolution of each pixel in microns (nanodrive). Limited to give '),
        Parameter('time_per_pt', 2.0, [2.0,5.0], 'Time in ms at each point to get counts; same as load_rate for nanodrive. Wroking values 2 or 5 ms'),
        Parameter('ending_behavior', 'return_to_inital_pos', ['return_to_inital_pos', 'return_to_origin', 'leave_at_corner'],'Nanodrive position after scan'),
        #clocks currently not implemented
        Parameter('laser_clock', 'Pixel', ['Pixel','Line','Frame','Aux'], 'Nanodrive clocked used for turning laser on and off')
    ]

    #_DEVICES
    _DEVICES = {
        'nanodrive': 'nanodrive',
        'microdrive': 'microdrive',
        'amscope_camera': 'amscope_camera'
    }
    _EXPERIMENTS = {}

    def __init__(self, devices, experiments=None, name=None, settings=None, log_function=None, data_path=None):
        """
        Initializes and connects to devices
        Args:
            name (optional): name of experiment, if empty same as class name
            settings (optional): settings for this experiment, if empty same as default settings
        """
        super().__init__(name, settings=settings, sub_experiments=experiments, devices=devices, log_function=log_function, data_path=data_path)
        #get instances of devices
        self.nd = self.devices['nanodrive']['instance']
        self.md = self.devices['microdrive']['instance']
        self.cam = self.devices['amscope_camera']['instance']


    """def setup_scan(self):
        '''
        Gets paths for adbasic file and loads them onto ADwin.
        '''
        self.adw.stop_process(2)
        sleep(0.1)
        self.adw.clear_process(2)
        
        # Use the helper function to find the binary file
        one_d_scan_path = get_adwin_binary_path('One_D_Scan.TB2')
        self.adw.update({'process_2': {'load': str(one_d_scan_path)}})
        # one_d_scan script increments an index then adds count values to an array in a constant time interval
        self.nd.clock_functions('Frame', reset=True)  # reset ALL clocks to default settings

        z_pos = self.settings['z_pos']
        #maz range is 0 to 100
        if self.settings['z_pos'] < 0.0:
            z_pos = 0.0
        elif z_pos > 100.0:
            z_pos = 100.0
        self.nd.update({'z_pos': z_pos})

        # tracker to only save 3D image slice once
        self.data_collected = False"""

    """def after_scan(self):
        '''
        Cleans up adwin and moves nanodrive to specified position
        '''
        # clearing process to aviod memory fragmentation when running different experiments in GUI
        self.adw.stop_process(2)    #neccesary if process is does not stop for some reason
        sleep(0.1)
        self.adw.clear_process(2)
        if self.settings['ending_behavior'] == 'return_to_inital_pos':
            self.nd.update({'x_pos': self.x_inital, 'y_pos': self.y_inital})
        elif self.settings['ending_behavior'] == 'return_to_origin':
            self.nd.update({'x_pos': 0.0, 'y_pos': 0.0})"""

    def _function(self):
        '''
        This is the actual function that will be executed. It uses only information that is provided in the settings property
        will be overwritten in the __init__
        '''
        # Set the data folder path at runtime to ensure correct path resolution
        if not self.settings['positions']['folderpath']:
            self.settings['positions']['folderpath'] = str(get_configured_nv_positioning_folder())

        self.setup_scan()
        sleep(0.1)

        #y scanning range is 5 to 95 to compensate for warm up time
        x_min = max(self.settings['point_a']['x'], 0.0)
        y_min = max(self.settings['point_a']['y'], 5.0)
        x_max = min(self.settings['point_b']['x'], 100.0)
        y_max = min(self.settings['point_b']['y'], 95.0)

        step = self.settings['resolution']
        num_points = (y_max - y_min) / step + 1
        print('num_points',num_points)
        if num_points < 91:
            new_step = self.correct_step(step)
            self.log(f'Works best with minimum 91 pixel resolution in y-direction. You are getting a free resolution upgrade to {new_step} um!')

        #array form point_a x,y to point_b x,y with step of resolution
        x_array = np.arange(x_min, x_max + step, step)
        y_array = np.arange(y_min, y_max+step, step)

        #adds point 5 um before and after
        y_before = np.arange(y_min-5.0,y_min,step)
        y_after = np.arange(y_max + step, y_max + 5.0 + step, step)
        y_array_adj = np.insert(y_array, 0, y_before)
        y_array_adj = np.append(y_array_adj, y_after)

        self.x_inital = self.nd.read_probes('x_pos')
        self.y_inital = self.nd.read_probes('y_pos')
        self.z_inital = self.nd.read_probes('z_pos')
        self.settings['z_pos'] = self.z_inital

        #makes sure data is getting recorded. If still equal none after running experiment, ten data is not being stored or not measured
        self.data['x_pos'] = None
        self.data['y_pos'] = None
        self.data['raw_counts'] = None
        self.data['count_rate'] = None
        self.data['count_img'] = None
        self.data['raw_img'] = None
        #local lists to store data and append to global self.data lists
        x_data = []
        y_data = []
        raw_count_data = []
        count_rate_data = []
        index_list = []

        # set data to zero and update to plot while experiment runs
        Nx = len(x_array)
        Ny = len(y_array)
        self.data['count_img'] = np.zeros((Nx, Ny))
        self.data['raw_img'] = np.zeros((Nx, len(y_array_adj)+20))

        interation_num = 0 #number to track progress
        total_interations = ((x_max - x_min)/step + 1)*((y_max - y_min)/step + 1)       #plus 1 because in total_iterations because range is inclusive ie. [0,10]
        #print('total_interations=',total_interations)

        #formula to set adwin to count for correct time frame. The event section is run every delay*3.3ns so the counter increments for that time then is read and clear
        #time_per_pt is in millisecond and the adwin delay time is delay_value*3.3ns
        adwin_delay = round((self.settings['time_per_pt']*1e6) / (3.3))
        #print('adwin delay: ',delay)

        wf = list(y_array_adj)
        len_wf = len(y_array_adj)
        #print(len_wf,wf)
        load_read_ratio = self.settings['time_per_pt']/2.0 #used for scaling when rates are different
        num_points_read = int(load_read_ratio*len_wf + 20) #20 is added to compensate for start warm up producing ~15 points of unwanted values

        #set inital x and y and set nanodrive stage to that position
        self.nd.update({'x_pos':x_min,'y_pos':y_min-5.0,'num_datapoints':len_wf,'read_rate':2.0,'load_rate':self.settings['time_per_pt']})
        #load_rate is time_per_pt; 2.0ms = 5000Hz
        self.adw.update({'process_2':{'delay':adwin_delay}})
        sleep(0.1)  #time for stage to move to starting posiition and adwin process to initilize


        for i, x in enumerate(x_array):
            if self._abort == True:
                break
            img_row = []
            raw_img_row = []
            x = float(x)

            self.nd.update({'x_pos':x,'y_pos':y_min-5.0})     #goes to x position
            sleep(0.1)
            x_pos = self.nd.read_probes('x_pos')
            x_data.append(x_pos)
            self.data['x_pos'] = x_data     #adds x postion to data

            #The two different code lines to start counting seem to work for cropping. Honestly cant give a precise explaination, it seems to be related to
            #hardware delay. If the time_per_pt is 5.0 starting counting before waveform set up works to within 1 pixel with numpy cropping. If the
            #time_per_pt is 2.0 starting counting after waveform set up matches slow scan to a pixel. Sorry for a lack of explaination but this just seems to work.
            #See data/dylan_staples/confocal_scans_w_resolution_target for images and additional details
            if self.settings['time_per_pt'] == 5.0:
                self.adw.update({'process_2': {'running': True}})

            #trigger waveform on y-axis and record position data
            self.nd.setup(settings={'num_datapoints': len_wf, 'load_waveform': wf}, axis='y')
            self.nd.setup(settings={'num_datapoints': num_points_read, 'read_waveform': self.nd.empty_waveform},axis='y')

            #restricted load_rate and read_rate to ensure cropping works. 2ms and 5ms count times are good as smaller window for speed and a larger window if more counts are needed
            if  self.settings['time_per_pt'] == 2.0:
                self.adw.update({'process_2': {'running': True}})

            y_pos = self.nd.waveform_acquisition(axis='y')
            sleep(self.settings['time_per_pt']*len_wf/1000)

            #want to get data only in desired range not range±5um
            y_pos_array = np.array(y_pos)
            # index for the points of the read array when at y_min and y_max. Scale step by load_read_ratio to get points closest to y_min & y_max
            lower_index = np.where((y_pos_array > y_min - step / load_read_ratio) & (y_pos_array < y_min + step / load_read_ratio))[0]
            upper_index = np.where((y_pos_array > y_max - step / load_read_ratio) & (y_pos_array < y_max + step / load_read_ratio))[0]
            y_pos_cropped = list(y_pos_array[lower_index[0]:upper_index[0]])

            #y_data.extend(y_pos_cropped)
            y_data.append(list(y_pos))
            self.data['y_pos'] = y_data
            self.adw.update({'process_2':{'running':False}})

            #different index for count data if read and load rates are different
            counts_lower_index = int(lower_index[0] / load_read_ratio)
            counts_upper_index = int(upper_index[-1] / load_read_ratio)
            index_list.append(counts_upper_index)

            #get mode of index list and difference between mode and previous value
            index_mode = max(set(index_list), key=index_list.count)
            index_diff = abs(counts_upper_index - index_mode)
            # index starts at 0 so need to add 1 if there is an index difference
            if index_diff > 0:
                index_diff = index_diff + 1

            # get count data from adwin and record it
            raw_counts = np.array(list(self.adw.read_probes('int_array', id=1, length=len_wf+20)))
            # units of count/seconds
            count_rate = list(np.array(raw_counts) * 1e3 / self.settings['time_per_pt'])

            crop_index = -index_mode - 1 - index_diff
            if self.settings['time_per_pt'] == 5.0:
                crop_index = crop_index-2
            cropped_raw_counts = list(raw_counts[crop_index:crop_index + len(y_array)])
            cropped_count_rate = count_rate[crop_index:crop_index + len(y_array)]

            raw_count_data.append(cropped_raw_counts)
            self.data['raw_counts'] = raw_count_data

            count_rate_data.append(cropped_count_rate)
            self.data['count_rate'] = count_rate_data

            #adds count rate data to raw img and cropped count img
            raw_img_row.extend(count_rate)
            self.data['raw_img'][i, :] = raw_img_row
            img_row.extend(cropped_count_rate)
            self.data['count_img'][i, :] = img_row  # add previous scan data so image plots

            # updates process bar and plots count_img so far
            interation_num = interation_num + len(y_array)
            self.progress = 100. * (interation_num +1) / total_interations
            self.updateProgress.emit(self.progress)

        #tracker to only save test image once
        self.data_collected = True

        print('Data collected')
        self.data['x_pos'] = x_data
        self.data['y_pos'] = np.array(y_data)
        self.data['raw_counts'] = np.array(raw_count_data)
        self.data['count_rate'] = np.array(count_rate_data)
        #print('Position Data: ','\n',self.data['x_pos'],'\n',self.data['y_pos'],'\n','Max x: ',np.max(self.data['x_pos']),'Max y: ',np.max(self.data['y_pos']))
        #print('Counts: ','\n',self.count_data)
        #print('All data: ',self.data)

        self.after_scan()

    """def _plot(self, axes_list, data=None):
        '''
        This function plots the data. It is triggered when the updateProgress signal is emitted and when after the _function is executed.
        For the scan, image can only be plotted once all data is gathered so self.running prevents a plotting call for the updateProgress signal.
        '''
        def create_img(add_colobar=True):
            '''
            Creates a new image and ImageItem. Optionally create colorbar
            '''
            axes_list[0].clear()
            self.count_image = pg.ImageItem(data['count_img'], interpolation='nearest')
            self.count_image.setLevels(levels)
            self.count_image.setRect(pg.QtCore.QRectF(extent[0], extent[2], extent[1] - extent[0], extent[3] - extent[2]))
            axes_list[0].addItem(self.count_image)

            axes_list[0].setAspectLocked(True)
            axes_list[0].setLabel('left', 'y (µm)')
            axes_list[0].setLabel('bottom', 'x (µm)')
            axes_list[0].setTitle(f"Confocal Scan with z = {self.settings['z_pos']:.2f}")

            if add_colobar:
                self.colorbar = pg.ColorBarItem(values=(levels[0], levels[1]), label='counts/sec', colorMap='viridis')
                # layout is housing the PlotItem that houses the ImageItem. Add colorbar to layout so it is properly saved when saving dataset
                layout = axes_list[0].parentItem()
                layout.addItem(self.colorbar)
            self.colorbar.setImageItem(self.count_image)

        if data is None:
            data = self.data
        if data is not None or data is not {}:
            #for colorbar to display gradient without artificial zeros
            try: #sometimes when data is inputted as argument it does not have 'count_img' key; this try/except prevents error if that happens
                non_zero_values = data['count_img'][data['count_img'] > 0]
            except KeyError:
                data['count_img'] = self.data['count_img']
                non_zero_values = data['count_img'][data['count_img'] > 0]
            if non_zero_values.size > 0:
                min = np.min(non_zero_values)
            else: #if else to aviod ValueError
                min = 0

            levels = [min, np.max(data['count_img'])]
            extent = [self.settings['point_a']['x'], self.settings['point_b']['x'], self.settings['point_a']['y'],self.settings['point_b']['y']]

            if self._plot_refresh == True:
                # if plot refresh is true the ImageItem has been deleted and needs recreated
                create_img()
            else:
                try:
                    self.count_image.setImage(data['count_img'], autoLevels=False)
                    self.count_image.setLevels(levels)
                    self.colorbar.setLevels(levels)

                    if self.settings['3D_scan']['enable'] and self.data_collected:
                        print('z =', self.z_inital, 'max counts =', levels[1])
                        axes_list[0].setTitle(f"Confocal Scan with z = {self.z_inital:.2f}")
                        scene = axes_list[0].scene()
                        exporter = ImageExporter(scene)
                        
                        # Use pathlib for cross-platform path handling
                        folder_path = Path(self.settings['3D_scan']['folderpath'])
                        try:
                            folder_path.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
                            filename = folder_path / f'confocal_scan_z_{self.z_inital:.2f}.png'
                            exporter.export(str(filename))
                            print(f"Saved 3D scan image to: {filename}")
                        except Exception as e:
                            print(f"Warning: Failed to save 3D scan image: {e}")
                            print(f"Attempted to save to: {folder_path}")

                except RuntimeError:
                    # sometimes when clicking other experiments ImageItem is deleted but _plot_refresh is false. This ensures the image can be replotted
                    create_img(add_colobar=False)"""

    """def _update(self,axes_list):
        self.count_image.setImage(self.data['count_img'], autoLevels=False)
        self.count_image.setLevels([np.min(self.data['count_img']), np.max(self.data['count_img'])])
        self.colorbar.setLevels([np.min(self.data['count_img']), np.max(self.data['count_img'])])"""

    """def correct_step(self, old_step):
        '''
        Increases resolution by one threshold if the step size does not give enough points for a good y-array.
        For good y-array len() > 90
         '''
        if old_step == 1.0:
            return 0.5
        elif old_step > 1.0:
            return 1.0
        elif old_step == 0.5:
            return 0.25
        elif old_step == 0.25:
            return 0.1
        elif old_step == 0.1:
            return 0.05
        elif old_step == 0.05:
            return 0.025
        elif old_step == 0.025:
            return 0.001
        else:
            raise KeyError """


# ──────────────────────────────────────────────────────────────────────────────
# Helper widgets
# ──────────────────────────────────────────────────────────────────────────────

class SnapWin(QWidget):
    """Separate window that shows still‑image captures."""

    def __init__(self, w: int, h: int):
        super().__init__()
        self.setWindowTitle("Snapshot")
        self.setFixedSize(w, h)
        self.label = QLabel(self)
        self.label.resize(w, h)
        self.label.setScaledContents(False)

    def show_frame(self, qimg: QImage):
        self.label.setPixmap(QPixmap.fromImage(qimg))
        self.show()


# ──────────────────────────────────────────────────────────────────────────────
# Main camera window
# ──────────────────────────────────────────────────────────────────────────────

class MainWin(QWidget):
    """Live‑view window. Compatible with the legacy *app.py* launcher."""

    eventImage = pyqtSignal(int)

    def __init__(
        self,
        gain: int = 100,
        integration_time_us: int = 10_000,
        res: str = "low",
    ) -> None:
        super().__init__()
        self.hcam: Optional[Amscope_MU_Camera.Amscope_MU_Camera] = None
        self.buf: Optional[ctypes.Array] = None
        self.w = self.h = 0
        self.gain = gain
        self.integration = integration_time_us  # already in µs
        self.res = res.lower()

        # frame counter for FPS display
        self._frame_accum = 0
        self._last_tick = time.perf_counter()

        self._init_ui()
        self._init_camera()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _init_ui(self) -> None:
        # center the window on whatever display we’re on
        self.setFixedSize(820, 640)  # temp; corrected once cam opens
        geo = self.frameGeometry()
        geo.moveCenter(QDesktopWidget().availableGeometry().center())
        self.move(geo.topLeft())

        # widgets
        self.label = QLabel(self)
        self.label.setScaledContents(False)  # don’t resample!

        self.cb_auto = QCheckBox("Auto Exposure", self)
        self.cb_auto.stateChanged.connect(self._on_auto_exp_toggled)

        self.cb_fps = QCheckBox("Show FPS", self)

        # layout
        cols = QVBoxLayout(self)
        cols.addWidget(self.label, stretch=1)
        row = QHBoxLayout()
        row.addWidget(self.cb_auto)
        row.addWidget(self.cb_fps)
        row.addStretch(1)
        cols.addLayout(row)

    # ── Camera setup ──────────────────────────────────────────────────────────

    def _init_camera(self) -> None:
        cams = amcam.Amcam.EnumV2()
        if not cams:
            self.setWindowTitle("No camera found")
            self.cb_auto.setEnabled(False)
            return

        self.camname = cams[0].displayname
        self.setWindowTitle(self.camname)
        self.eventImage.connect(self._on_event_image)

        try:
            self.hcam = Amscope_MU_Camera.Amscope_MU_Camera()
        except amcam.HRESULTException as ex:
            QMessageBox.warning(self, "", f"Failed to open camera (hr=0x{ex.hr:x})")
            return

        # basic settings
        self.hcam.set_ExpoAGain(self.gain)
        self._clamp_and_set_exposure(self.integration)
        self._apply_resolution(self.res)

        # negotiate RGB/BGR for zero‑copy into QImage
        if sys.platform != "win32":
            self.hcam.put_Option(amcam.AMCAM_OPTION_BYTEORDER, 1)  # BGR on Linux/mac

        # internal buffer (mutable)
        stride = ((self.w * 24 + 31) // 32) * 4
        self.buf = ctypes.create_string_buffer(stride * self.h)

        # resize widget exactly to sensor size (no scaling cost)
        self.setFixedSize(self.w, self.h + 40)  # + controls bar
        self.label.setFixedSize(self.w, self.h)

        # reflect current auto‑exposure state
        self.cb_auto.setChecked(self.hcam.get_AutoExpoEnable())

        # start stream
        try:
            self.hcam.StartPullModeWithCallback(self._camera_cb, self)
        except amcam.HRESULTException as ex:
            QMessageBox.warning(self, "", f"Stream start failed (hr=0x{ex.hr:x})")
            return

    def _clamp_and_set_exposure(self, target_us: int) -> None:
        lo, hi, _ = self.hcam.get_ExpTimeRange()
        self.hcam.put_ExpoTime(max(lo, min(target_us, hi)))

    def _apply_resolution(self, res: str) -> None:
        match res:
            case "high":
                self.hcam.put_eSize(0)  # 2560×1922
            case "mid":
                self.hcam.put_eSize(1)  # 1280×960
            case _:
                self.hcam.put_eSize(2)  # 640×480
        self.w, self.h = self.hcam.get_Size()

    # ── Toupcam callback (runs in SDK thread) ─────────────────────────────––

    @staticmethod
    def _camera_cb(event: int, ctx: "MainWin") -> None:
        if event == amcam.AMCAM_EVENT_IMAGE:
            try:
                ctx.hcam.PullImageV2(ctx.buf, 24, None)
            except amcam.HRESULTException:
                return  # drop frame
            ctx.eventImage.emit(event)
        elif event == amcam.AMCAM_EVENT_STILLIMAGE:
            try:
                ctx.hcam.PullStillImageV2(ctx.buf, 24, None)
            except amcam.HRESULTException:
                return
            ctx.eventImage.emit(event)

    # ── Qt slot (runs in GUI thread) ─────────────────────────────────────────

    @pyqtSlot(int)
    def _on_event_image(self, event: int) -> None:
        # stride is constant → safe
        stride = ((self.w * 24 + 31) // 32) * 4
        qimg = QImage(self.buf, self.w, self.h, stride, QImage.Format_RGB888)
        # Draw cross lines at center
        painter = QPainter(qimg)
        pen = QPen(Qt.red, 1)
        painter.setPen(pen)
        cx, cy = self.w // 2, self.h // 2  # image center
        line_len = 20  # length of line arms, adjust as you like
        # horizontal line
        painter.drawLine(cx - line_len, cy, cx + line_len, cy)
        # vertical line
        painter.drawLine(cx, cy - line_len, cx, cy + line_len)
        painter.end()
        # end overlay
        #qimg.save("frame.png")
        """import numpy as np

        arr = np.frombuffer(self.buf, dtype=np.uint8).reshape((self.h, self.w, 3))
        print(arr)  # Show pixel data
        import matplotlib.pyplot as plt

        plt.imshow(arr)
        plt.show()"""

        if event == amcam.AMCAM_EVENT_IMAGE:
            self.label.setPixmap(QPixmap.fromImage(qimg))
            self._update_fps()
        else:  # still image
            if not hasattr(self, "_snap_win"):
                self._snap_win = SnapWin(self.w, self.h)
            self._snap_win.show_frame(qimg)

    # ── Misc callbacks ──────────────────────────────────────────────────────

    def _on_auto_exp_toggled(self, state: int) -> None:
        if self.hcam:
            self.hcam.put_AutoExpoEnable(state == Qt.Checked)

    def _update_fps(self) -> None:
        if not self.cb_fps.isChecked():
            return
        self._frame_accum += 1
        now = time.perf_counter()
        if now - self._last_tick >= 1.0:
            fps = self._frame_accum / (now - self._last_tick)
            self.setWindowTitle(f"{self.camname} – {fps:.1f} fps")
            self._frame_accum = 0
            self._last_tick = now

    # ── API for *app.py* ────────────────────────────────────────────────────

    def snap(self):
        if self.hcam:
            self.hcam.Snap(0)

    # ── Cleanup ─────────────────────────────────────────────────────────────

    def closeEvent(self, evt):  # noqa: N802 (Qt override)
        if self.hcam:
            self.hcam.close()
            self.hcam = None
        super().closeEvent(evt)


# ──────────────────────────────────────────────────────────────────────────────
# Stand‑alone entry point (for direct testing)  →   $ python -m amcam_qt_app
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWin()
    w.show()
    sys.exit(app.exec())
