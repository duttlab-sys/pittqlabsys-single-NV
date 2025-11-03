# CREATED JANNET TRABELSI ON 10/03/2025
# USED AMCAM THAT COMMUNICATES WITH THE CAMERA THROUGH THE DLL FILE
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

from __future__ import annotations
from src.Controller import amcam

from src.core import Device,Parameter

_DEFAULT_AUTO_EXPOSURE_TARGET  = 120
_DEFAULT_TEMP = 6503
_DEFAULT_TINT = 1000
_DEFAULT_LEVEL_RANGE = 125
_DEFAULT_CONTRAST = 0
_DEFAULT_HUE = 0
_DEFAULT_SATURATION = 128
_DEFAULT_BRIGHTNESS = 0
_DEFAULT_GAMMA = 100
_DEFAULT_WHITE_BALANCE_GAIN = 0
_DEFAULT_GAIN = 100
_DEFAULT_EXPOSURE_TIME_US = 10000
_DEFAULT_RESOLUTION = "low"
_server_port = 5005

class Amscope_MU_Camera(Device):
    """This class implements the Windfreak SynthUSBII. The device plugs into a usb port and is communicated with using pyvisa."""
    _DEFAULT_SETTINGS = Parameter([
        Parameter('exposure gain', _DEFAULT_GAIN, int, 'camera exposure gain'),
        Parameter('exposure time', _DEFAULT_EXPOSURE_TIME_US, int, 'camera exposure time in us'),
        Parameter('brightness', _DEFAULT_BRIGHTNESS, int, 'camera brightness'),
        Parameter('saturation', _DEFAULT_SATURATION, int, 'camera saturation'),
        Parameter('contrast', _DEFAULT_CONTRAST, int, 'camera contrast'),
        Parameter('Gamma', _DEFAULT_GAMMA, int, 'camera Gamma'),
        Parameter('Temp', _DEFAULT_TEMP, int, 'camera Temp'),
        Parameter('Tint', _DEFAULT_TINT, int, 'camera Tint'),
        Parameter('Hue', _DEFAULT_HUE, int, 'camera Hue'),
        Parameter('server_port', _server_port, int, 'server_port'),
        # _DEFAULT_AUTO_EXPOSURE_TARGET = 120
        # _DEFAULT_LEVEL_RANGE = 125
    ])

    def update(self, settings):
        """
        Updates the internal settings of the camera
        Args:
            settings: a dictionary in the standard settings format
        """
        super(Amscope_MU_Camera, self).update(settings)
        for key, value in settings.items():
            if not (key == 'server_port'):
                if key == 'exposure gain':
                    self.amscope_cam.put_ExpoAGain(value)
                elif key == 'exposure time':
                    lo, hi, _ = self.amscope_cam.get_ExpTimeRange()
                    self.amscope_cam.put_ExpoTime(max(lo, min(value, hi)))
                elif key == 'brightness':
                    self.amscope_cam.put_Brightness(value)
                elif key == 'saturation':
                    self.amscope_cam.put_Saturation(value)
                elif key == 'contrast':
                    self.amscope_cam.put_Contrast(value)

    @property
    def _PROBES(self):
        return {
            'exposure gain': 'exposure gain',
            'exposure time': 'exposure time',
            'Gamma': 'Gamma',
            'Chrome': 'Chrome',
            'VFlip': 'VFlip',
            'HFlip': 'HFlip',
            'Negative': 'Negative',
            'Speed': 'Speed',
            'HZ': 'HZ',
            'Mode': 'Mode',
            'Temp': 'Temp',
            'Tint': 'Tint',
            'AWBAuxRect': 'AWBAuxRect',
            'AEAuxRect': 'AEAuxRect',
            'BlackBalance': 'BlackBalance',
            'ABBAuxRect': 'ABBAuxRect',
            'Hue': 'Hue'
        }

    def _param_to_internal(self, param):
        # converts settings parameter to corresponding key
        if param not in ['exposure gain', 'white balance gain', 'exposure time', 'Gamma', 'Chrome', 'VFlip',
                             'HFlip',
                             'Negative', 'Speed', 'HZ', 'Mode', 'Temp', 'Tint', 'AWBAuxRect', 'AEAuxRect',
                             'BlackBalance',
                             'ABBAuxRect', 'Hue']:
            raise KeyError
        return param

    def read_probes(self, key):
        assert (self._settings_initialized)
        assert key in list(self._PROBES.keys())

        key_internal = self._param_to_internal(key)
        if key == 'exposure gain':
            value = self.amscope_cam.get_ExpoAGain()
        elif key == 'white balance gain':
            value = self.amscope_cam.get_WhiteBalanceGain()
        elif key == 'exposure time':
            value = self.amscope_cam.get_ExpoTime()
        elif key == 'Gamma':
            value = self.amscope_cam.get_Gamma()
        elif key == 'Chrome':
            value = self.amscope_cam.get_Chrome()
        elif key == 'VFlip':
            value = self.amscope_cam.get_VFlip()
        elif key == 'HFlip':
            value = self.amscope_cam.get_HFlip()
        elif key == 'Negative':
            value = self.amscope_cam.get_Negative()
        elif key == 'Speed':
            value = self.amscope_cam.get_Speed()
        elif key == 'HZ':
            value = self.amscope_cam.get_HZ()
        elif key == 'Mode':
            value = self.amscope_cam.get_Mode()
        elif key == 'Temp':
            temp, _ = self.amscope_cam.get_TempTint()
            value = temp
        elif key == 'Tint':
            _, tint = self.amscope_cam.get_TempTint()
            value = tint
        elif key == 'AWBAuxRect':
            value = self.amscope_cam.get_AWBAuxRect()
        elif key == 'AEAuxRect':
            value = self.amscope_cam.get_AEAuxRect()
        elif key == 'BlackBalance':
            value = self.amscope_cam.get_BlackBalance()
        elif key == 'ABBAuxRect':
            value = self.amscope_cam.get_ABBAuxRect()
        elif key == 'Hue':
            value = self.amscope_cam.get_Hue()
        return value

    @property
    def is_connected(self):
        try:
            self._ask_value('f')  # arbitrary call to check connection
            return True
        except Exception as e:
            print(self, e)
            return False

    def close(self):
        self.amscope_cam.Close()

    def __init__(self, name=None, settings=None):
        # the object of Amcam must be obtained by classmethod Open or OpenByIndex, it cannot be obtained by obj = amcam.Amcam()
        cams = amcam.Amcam.EnumV2()
        if not cams:
            print("no camera found")

        try:
            self.amscope_cam = amcam.Amcam.Open(cams[0].id)
        except amcam.HRESULTException as ex:
            print("failed to open camera", ex)
            return

        self.name = "Amscope Camera"
        super(Amscope_MU_Camera, self).__init__(name, settings)
        self.amscope_cam.put_ExpoAGain(_DEFAULT_GAIN)
        lo, hi, _ = self.amscope_cam.get_ExpTimeRange()
        self.amscope_cam.put_ExpoTime(max(lo, min(_DEFAULT_EXPOSURE_TIME_US, hi)))
        self.amscope_cam.put_eSize(2)
        self.w, self.h = self.amscope_cam.get_Size()

    def set_ExpoAGain(self,gain):
        self.amscope_cam.put_ExpoAGain(gain)

    def get_ExpTimeRange(self):
        return self.amscope_cam.get_ExpTimeRange()

    def put_ExpoTime(self, t):
        self.amscope_cam.put_ExpoTime(t)

    def put_eSize(self, size):
        self.amscope_cam.put_eSize(size)

    def get_Size(self):
        return self.amscope_cam.get_Size()

    def get_AutoExpoEnable(self):
        return self.amscope_cam.get_AutoExpoEnable()

    def put_AutoExpoEnable(self, en):
        self.amscope_cam.put_AutoExpoEnable(en)

    def StartPullModeWithCallback(self, A, B):
        self.amscope_cam.StartPullModeWithCallback(A, B)

    def PullImageV2(self, a, b, c):
        self.amscope_cam.PullImageV2(a, b, c)

    def set_WhiteBalanceGain(self, value):
        self.amscope_cam.put_WhiteBalanceGain(value)

    def stop(self):
        self.amscope_cam.Stop()

    def pause(self, enable):
        self.amscope_cam.Pause(enable)

if __name__ == '__main__':
    am=Amscope_MU_Camera()
    print(am.get_Size())