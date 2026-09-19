"""
Written by <Jannet Trabelsi> <<jat332@pitt.edu>>
Thorlabs_K10CR1.py
==================
Device wrapper for the Thorlabs K10CR1 integrated rotation stage, written to the
same Device / Parameter framework used by Thorlabs_FW102C.

Why device units:
    The K10CR1's real-unit (degree) converter would not initialize reliably through
    pythonnet on this setup -- device.MoveTo(Decimal(...)) silently moved ~0 -- so all
    motion here is commanded in *device units* and this class converts degrees <-> device
    units itself.

K10CR1 POSITION scaling (angle only):
    200 full steps/rev  x  2048 microsteps/step  x  120:1 gearbox
        = 49,152,000 device units per 360 deg
        = 136,533.33 device units per degree
    so 150 deg  ->  20,480,000 device units  (the value confirmed on your hardware).

IMPORTANT: velocity and acceleration use *different* device-unit factors on the K10CR1,
so _deg_to_du() below is for ANGLE / POSITION only -- do not reuse it for speed/accel.
"""

import clr
import time

from src.core import Device, Parameter

# --- Kinesis .NET assemblies ---
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.DeviceManagerCLI.dll")
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.GenericMotorCLI.dll")
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.IntegratedStepperMotorsCLI.dll")

from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI
from Thorlabs.MotionControl.IntegratedStepperMotorsCLI import CageRotator

unit_1_serial_no = "55001352"
unit_2_serial_no = "55001320"
_server_port = 5002

# Position scaling (ANGLE only -- see note in the module docstring)
_COUNTS_PER_REV = 200 * 2048 * 120          # 49,152,000
_COUNTS_PER_DEG = _COUNTS_PER_REV / 360.0   # 136,533.333...


class Thorlabs_K10CR1(Device):
    _DEFAULT_SETTINGS = Parameter([
        Parameter('get_data', True, [False, True], 'choose whether you need to get data from this device or not'),
        Parameter('serial_no', unit_1_serial_no, [unit_1_serial_no, unit_2_serial_no], 'Kinesis serial number of the K10CR1'),
        Parameter('position', 0.0, float, 'Target angle in degrees'),
        Parameter('server_port', _server_port, int, 'server_port'),
    ])

    def __init__(self, name=None, settings=None):
        try:
            super(Thorlabs_K10CR1, self).__init__(name, settings)
            self._connect()
        except Exception:
            self.close()  # if the open succeeded but something later threw, don't leak the handle
            raise

    def close(self):
        stage = getattr(self, 'stage', None)
        if stage is not None:
            try:
                stage.StopPolling()
                stage.Disconnect()
            finally:
                self.stage = None

    def _connect(self):
        if getattr(self, 'stage', None) is not None:
            return 0
        serial = str(self.settings['serial_no'])

        DeviceManagerCLI.BuildDeviceList()
        stage = CageRotator.CreateCageRotator(serial)
        stage.Connect(serial)

        if not stage.IsSettingsInitialized():
            stage.WaitForSettingsInitialized(10000)  # 10 s timeout

        stage.StartPolling(250)  # 250 ms polling
        time.sleep(0.25)
        stage.EnableDevice()
        time.sleep(0.25)

        # Load the configuration so the device is fully initialized (travel limits, etc.).
        # We still command moves in device units; this just finishes bringing the stage up.
        config = stage.GetMotorConfiguration(serial)
        if not config.DeviceSettingsName:
            config.DeviceSettingsName = "K10CR1"
        config.UpdateCurrentConfiguration()

        self.stage = stage
        return 0

    @property
    def is_connected(self):
        return getattr(self, 'stage', None) is not None

    @property
    def _PROBES(self):
        return {
            'get_data': 'choose whether you need to get data from this device or not',
            'position': 'Current angle in degrees',
        }

    def read_probes(self, key):
        assert self._settings_initialized  # fails if settings/connection not yet initialized
        assert key in list(self._PROBES.keys())
        if key == 'get_data':
            return self.settings['get_data']
        if key == 'position':
            return self._du_to_deg(self.stage.Position_DeviceUnit)
        else:
            raise KeyError(key)

    def update(self, settings):
        super(Thorlabs_K10CR1, self).update(settings)
        for key, value in settings.items():
            if key in ('get_data', 'server_port'):
                continue
            if key == 'serial_no':
                self._connect()
            elif key == 'position':
                self.move_to(value)
            else:
                raise KeyError(key)

    # ---------- unit conversion (ANGLE / POSITION only) ----------
    def _deg_to_du(self, deg):
        """Degrees -> device units.  e.g. self._deg_to_du(150) -> 20480000"""
        return int(round(float(deg) * _COUNTS_PER_DEG))

    def _du_to_deg(self, du):
        """Device units -> degrees.  e.g. self._du_to_deg(20480000) -> 150.0"""
        return float(du) / _COUNTS_PER_DEG

    # ---------- motion ----------
    def move_to(self, angle_deg, timeout_ms=60000):
        """Move to an absolute angle in degrees (blocking, converts to device units)."""
        self.stage.MoveTo_DeviceUnit(self._deg_to_du(angle_deg), timeout_ms)

    def home(self, timeout_ms=60000):
        """Home the stage (blocking). Home before trusting absolute angles."""
        self.stage.Home(timeout_ms)


if __name__ == "__main__":
    dev = Thorlabs_K10CR1()
    dev.home()
    dev.move_to(150, timeout_ms=60000)
    #dev.update({"position": 0})