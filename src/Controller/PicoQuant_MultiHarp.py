# Written by <Jannet Trabelsi> <<jat332@pitt.edu>>
#!/usr/bin/env python3
"""
PicoQuant_MultiHarp.py
======================

Device wrapper for a PicoQuant MultiHarp 150/160 used as a multi-channel T2
time tagger, exposed through the src.core Device/Parameter framework so it drops
in next to the other lab devices (e.g. Thorlabs_FW102C).

Scope / separation of concerns
------------------------------
This class owns *hardware control and raw data acquisition only*:

  * open / initialize (T2) / configure inputs / close
  * per-channel trigger level, edge, and cable-delay offset
  * start / stop a T2 acquisition with a background FIFO reader thread
  * blocking `acquire()` convenience
  * decode raw T2 records -> per-channel arrival times in ps
  * live sync-rate / count-rate readout (useful for alignment & HBT)

It deliberately does NOT do correlation, histogramming, rise-time/drift
analysis, plotting, or HDF5 saving. That is experiment-level work:
  * `multiharp_timing_tester.py` keeps doing it for the timing diagnostics
    (this device does not touch or import that file), and
  * the upcoming HBT experiment will do its own g2 correlation on top of the
    timestamps this device hands back.

The MHLib ctypes wrapper below is intentionally self-contained (mirrored from
the timing tester) so this module has no dependency on the tester.

Hardware safety note (same as the tester)
------------------------------------------
MultiHarp inputs are level-trigger inputs, operating range -1200..+1200 mV into
50 ohm, damage level +/-2500 mV. Raw SPCM (~2.5 V) and raw ADwin/TTL (3.3-5 V)
are AT/OVER the damage level -- ATTENUATE before the MultiHarp and set each
trigger level to about half the *attenuated* amplitude.

    measured-safe amplitudes (07/20/26): SPCM (6+3 dB) 640 mV,
    Proteus ch4 MKR 720 mV, ADwin DO28 (10+6 dB) 610 mV, etc.
"""

from __future__ import annotations

import ctypes as ct
import platform
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.core import Device, Parameter

_device_index = -1      # which MultiHarp to open; -1 = auto-open the first one
_dll_path = ""          # explicit path to mhlib64.dll / libmhlib.so; "" = search
_server_port = 5002     # framework server port (distinct from the filter wheel)

# --------------------------------------------------------------------------- #
# MHLib constants
# --------------------------------------------------------------------------- #
MODE_T2 = 2                     # MH_Initialize measurement mode: T2 time tagging
REFSRC_INTERNAL = 0             # internal clock
TTREADMAX = 1048576             # fixed FIFO read block size (event records)
T2WRAP = 33554432               # 2**25, T2 overflow period in base-res units
EDGE_FALLING = 0
EDGE_RISING = 1
FLAG_FIFOFULL = 0x0002          # MH_GetFlags bit: FIFO overrun (data lost)

_ACQTMAX_MS = 36_000_000        # ~10 h; long enough to outlast a manually-stopped run


# --------------------------------------------------------------------------- #
# Thin ctypes wrapper around MHLib. All version-/platform-specific assumptions
# live here so there is exactly one place to adjust. Mirrored from the timing
# tester, plus optional sync-rate / count-rate readout for live diagnostics.
# --------------------------------------------------------------------------- #
class MHLibWrapper:
    def __init__(self, dll_path: Optional[str] = None):
        self._lib = self._load_library(dll_path)
        self._declare_prototypes()
        self.devidx: Optional[int] = None

    @staticmethod
    def _load_library(dll_path: Optional[str]):
        is_windows = platform.system() == "Windows"
        candidates = []
        if dll_path:
            candidates.append(dll_path)
        if is_windows:
            candidates += ["mhlib64.dll", "mhlib.dll"]
        else:
            candidates += ["libmhlib.so", "libmhlib.so.3"]
        last_err = None
        for name in candidates:
            try:
                # MHLib is cdecl on both platforms -> CDLL.
                return ct.CDLL(name)
            except OSError as e:
                last_err = e
        raise OSError(
            "Could not load MHLib. Tried: %s. Pass dll_path=... or make sure the "
            "MultiHarp software / MHLib is installed. Underlying error: %s"
            % (candidates, last_err)
        )

    def _declare_prototypes(self):
        L = self._lib
        c_int, c_uint, c_double, c_char_p = ct.c_int, ct.c_uint, ct.c_double, ct.c_char_p
        P_int, P_uint, P_double = ct.POINTER(c_int), ct.POINTER(c_uint), ct.POINTER(c_double)

        def sig(fn, argtypes):
            f = getattr(L, fn)
            f.argtypes = argtypes
            f.restype = c_int
            return f

        self.MH_GetLibraryVersion = sig("MH_GetLibraryVersion", [c_char_p])
        self.MH_GetErrorString = sig("MH_GetErrorString", [c_char_p, c_int])
        self.MH_OpenDevice = sig("MH_OpenDevice", [c_int, c_char_p])
        self.MH_CloseDevice = sig("MH_CloseDevice", [c_int])
        self.MH_Initialize = sig("MH_Initialize", [c_int, c_int, c_int])
        self.MH_GetHardwareInfo = sig(
            "MH_GetHardwareInfo", [c_int, c_char_p, c_char_p, c_char_p])
        self.MH_GetNumOfInputChannels = sig(
            "MH_GetNumOfInputChannels", [c_int, P_int])
        self.MH_GetBaseResolution = sig(
            "MH_GetBaseResolution", [c_int, P_double, P_int])
        self.MH_SetSyncDiv = sig("MH_SetSyncDiv", [c_int, c_int])
        self.MH_SetSyncEdgeTrg = sig("MH_SetSyncEdgeTrg", [c_int, c_int, c_int])
        self.MH_SetSyncChannelOffset = sig("MH_SetSyncChannelOffset", [c_int, c_int])
        self.MH_SetInputEdgeTrg = sig(
            "MH_SetInputEdgeTrg", [c_int, c_int, c_int, c_int])
        self.MH_SetInputChannelOffset = sig(
            "MH_SetInputChannelOffset", [c_int, c_int, c_int])
        self.MH_SetInputChannelEnable = sig(
            "MH_SetInputChannelEnable", [c_int, c_int, c_int])
        self.MH_StartMeas = sig("MH_StartMeas", [c_int, c_int])
        self.MH_StopMeas = sig("MH_StopMeas", [c_int])
        self.MH_CTCStatus = sig("MH_CTCStatus", [c_int, P_int])
        self.MH_GetFlags = sig("MH_GetFlags", [c_int, P_int])
        # MH_ReadFiFo (MHLib v3.x): reads up to TTREADMAX records into buffer and
        # returns the count in *nactual. If using an OLD MHLib (v1/v2) whose
        # signature is (devidx, buffer, count, *nactual), add a c_int before the
        # pointer here and pass TTREADMAX in read_fifo() below.
        self.MH_ReadFiFo = sig("MH_ReadFiFo", [c_int, P_uint, P_int])

        # Optional symbols -- present in newer MHLib, absent on very old libs.
        # Loaded lazily so a missing symbol never breaks the whole wrapper.
        for attr, name, argtypes in (
            ("MH_SetSyncChannelEnable", "MH_SetSyncChannelEnable", [c_int, c_int]),
            ("MH_GetSyncRate", "MH_GetSyncRate", [c_int, P_int]),
            ("MH_GetCountRate", "MH_GetCountRate", [c_int, c_int, P_int]),
        ):
            try:
                setattr(self, attr, sig(name, argtypes))
            except AttributeError:
                setattr(self, attr, None)

    # -- error handling ----------------------------------------------------- #
    def _check(self, code: int, where: str):
        if code < 0:
            buf = ct.create_string_buffer(64)
            try:
                self.MH_GetErrorString(buf, code)
                msg = buf.value.decode(errors="replace")
            except Exception:
                msg = "unknown"
            raise RuntimeError(f"MHLib error in {where}: {code} ({msg})")

    # -- lifecycle ---------------------------------------------------------- #
    def library_version(self) -> str:
        buf = ct.create_string_buffer(8)
        self._check(self.MH_GetLibraryVersion(buf), "GetLibraryVersion")
        return buf.value.decode(errors="replace")

    def open(self, index: int = -1) -> Tuple[int, str]:
        """Open a specific device index, or auto-open the first one if index<0."""
        if index is not None and index >= 0:
            serial = ct.create_string_buffer(8)
            self._check(self.MH_OpenDevice(index, serial), f"OpenDevice({index})")
            self.devidx = index
            return index, serial.value.decode(errors="replace")
        return self.open_first()

    def open_first(self) -> Tuple[int, str]:
        """Open the first responding MultiHarp; return (devidx, serial)."""
        serial = ct.create_string_buffer(8)
        for idx in range(8):  # MAXDEVNUM
            code = self.MH_OpenDevice(idx, serial)
            if code == 0:
                self.devidx = idx
                return idx, serial.value.decode(errors="replace")
        raise RuntimeError("No MultiHarp device could be opened (none found / all busy).")

    def hardware_info(self) -> Tuple[str, str, str]:
        model = ct.create_string_buffer(24)
        partno = ct.create_string_buffer(8)
        version = ct.create_string_buffer(8)
        self._check(self.MH_GetHardwareInfo(self.devidx, model, partno, version),
                    "GetHardwareInfo")
        return (model.value.decode(errors="replace"),
                partno.value.decode(errors="replace"),
                version.value.decode(errors="replace"))

    def initialize_t2(self):
        self._check(self.MH_Initialize(self.devidx, MODE_T2, REFSRC_INTERNAL),
                    "Initialize(T2)")

    def num_input_channels(self) -> int:
        n = ct.c_int()
        self._check(self.MH_GetNumOfInputChannels(self.devidx, ct.byref(n)),
                    "GetNumOfInputChannels")
        return n.value

    def base_resolution_ps(self) -> float:
        res = ct.c_double()
        binsteps = ct.c_int()
        self._check(self.MH_GetBaseResolution(self.devidx, ct.byref(res),
                                              ct.byref(binsteps)),
                    "GetBaseResolution")
        return float(res.value)

    def set_sync_div(self, div: int):
        self._check(self.MH_SetSyncDiv(self.devidx, div), "SetSyncDiv")

    def set_sync_trigger(self, level_mV: int, edge: int):
        self._check(self.MH_SetSyncEdgeTrg(self.devidx, level_mV, edge),
                    "SetSyncEdgeTrg")

    def set_sync_offset(self, offset_ps: int):
        self._check(self.MH_SetSyncChannelOffset(self.devidx, offset_ps),
                    "SetSyncChannelOffset")

    def set_sync_enable(self, enable: bool):
        if self.MH_SetSyncChannelEnable is not None:
            self._check(self.MH_SetSyncChannelEnable(self.devidx, 1 if enable else 0),
                        "SetSyncChannelEnable")

    def set_input_trigger(self, channel: int, level_mV: int, edge: int):
        self._check(self.MH_SetInputEdgeTrg(self.devidx, channel, level_mV, edge),
                    f"SetInputEdgeTrg(ch{channel})")

    def set_input_offset(self, channel: int, offset_ps: int):
        self._check(self.MH_SetInputChannelOffset(self.devidx, channel, offset_ps),
                    f"SetInputChannelOffset(ch{channel})")

    def set_input_enable(self, channel: int, enable: bool):
        self._check(self.MH_SetInputChannelEnable(self.devidx, channel,
                                                  1 if enable else 0),
                    f"SetInputChannelEnable(ch{channel})")

    def start(self, tacq_ms: int):
        self._check(self.MH_StartMeas(self.devidx, tacq_ms), "StartMeas")

    def stop(self):
        self._check(self.MH_StopMeas(self.devidx), "StopMeas")

    def ctc_done(self) -> bool:
        status = ct.c_int()
        self._check(self.MH_CTCStatus(self.devidx, ct.byref(status)), "CTCStatus")
        return status.value != 0

    def flags(self) -> int:
        f = ct.c_int()
        self._check(self.MH_GetFlags(self.devidx, ct.byref(f)), "GetFlags")
        return f.value

    def sync_rate(self) -> int:
        if self.MH_GetSyncRate is None:
            raise RuntimeError("MH_GetSyncRate not available in this MHLib version")
        r = ct.c_int()
        self._check(self.MH_GetSyncRate(self.devidx, ct.byref(r)), "GetSyncRate")
        return int(r.value)

    def count_rate(self, channel: int) -> int:
        if self.MH_GetCountRate is None:
            raise RuntimeError("MH_GetCountRate not available in this MHLib version")
        r = ct.c_int()
        self._check(self.MH_GetCountRate(self.devidx, channel, ct.byref(r)),
                    f"GetCountRate(ch{channel})")
        return int(r.value)

    def read_fifo(self, buffer) -> int:
        """Read one FIFO block into `buffer` (a c_uint * TTREADMAX). Returns count."""
        nactual = ct.c_int()
        self._check(self.MH_ReadFiFo(self.devidx, buffer, ct.byref(nactual)),
                    "ReadFiFo")
        # Old-MHLib variant, the prototype above gets changed:
        #   self.MH_ReadFiFo(self.devidx, buffer, TTREADMAX, ct.byref(nactual))
        return nactual.value

    def close(self):
        if self.devidx is not None:
            try:
                self.MH_CloseDevice(self.devidx)
            finally:
                self.devidx = None


# --------------------------------------------------------------------------- #
# The framework Device
# --------------------------------------------------------------------------- #
class PicoQuant_MultiHarp(Device):
    """MultiHarp 150/160 as a T2 time tagger, in the src.core Device framework.

    Two input channels (ch0, ch1) are first-class settings -- that covers the
    HBT case (two SPCMs) and the tester's start/stop case. For runs that need
    more than two inputs at once (e.g. a full timing map) call
    `configure_channels({idx: (level_mV, edge), ...})` on the open device.

    Note: like the filter-wheel device, `_connect()` is idempotent; to switch to
    a different `device_index`/`dll_path` on an already-open device, call
    `close()` first and then update.
    """

    _EDGE = {"rising": EDGE_RISING, "falling": EDGE_FALLING}

    _DEFAULT_SETTINGS = Parameter([
        Parameter('get_data', True, [False, True],
                  'choose whether you need to get data from this device or not'),
        Parameter('connection_type', 'MHLib', ['MHLib'],
                  'library used to talk to the MultiHarp'),
        Parameter('device_index', _device_index, int,
                  'MultiHarp device index to open; -1 auto-opens the first one'),
        Parameter('dll_path', _dll_path, str,
                  'explicit path to MHLib (mhlib64.dll / libmhlib.so); empty = search'),
        Parameter('sync_divider', 1, [1],
                  'sync divider; must be 1 in T2 mode (sync channel unused here)'),
        Parameter('acquisition_time_s', 10.0, float,
                  'acquisition duration used by the blocking acquire()/get_timestamps()'),
        # ---- input channel 0 -------------------------------------------------
        Parameter('ch0_enabled', True, [False, True], 'enable input channel 0'),
        Parameter('ch0_level_mV', 200, int,
                  'ch0 trigger level in mV (set ~half the ATTENUATED amplitude)'),
        Parameter('ch0_edge', 'rising', ['rising', 'falling'], 'ch0 trigger edge'),
        Parameter('ch0_offset_ps', 0, int, 'ch0 cable-delay offset (+/-100 ns)'),
        # ---- input channel 1 -------------------------------------------------
        Parameter('ch1_enabled', True, [False, True], 'enable input channel 1'),
        Parameter('ch1_level_mV', 200, int,
                  'ch1 trigger level in mV (set ~half the ATTENUATED amplitude)'),
        Parameter('ch1_edge', 'rising', ['rising', 'falling'], 'ch1 trigger edge'),
        Parameter('ch1_offset_ps', 0, int, 'ch1 cable-delay offset (+/-100 ns)'),
        Parameter('server_port', _server_port, int, 'server_port'),
    ])

    # channel-setting keys grouped by channel index, for update()/_connect()
    _CH_KEYS = {
        0: ('ch0_enabled', 'ch0_level_mV', 'ch0_edge', 'ch0_offset_ps'),
        1: ('ch1_enabled', 'ch1_level_mV', 'ch1_edge', 'ch1_offset_ps'),
    }
    _CONNECTION_KEYS = ('connection_type', 'device_index', 'dll_path', 'sync_divider')

    def __init__(self, name=None, settings=None):
        # Private state must exist BEFORE super().__init__, because the base
        # __init__ applies the settings (which calls update() -> _connect()).
        self._mh: Optional[MHLibWrapper] = None
        self._nchan: Optional[int] = None
        self._base_res_ps: float = 5.0
        self._reader: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self._chunks: List[np.ndarray] = []
        self._fifo_overrun = False
        try:
            super(PicoQuant_MultiHarp, self).__init__(name, settings)
        except Exception:
            self.close()  # if the open succeeded but something later threw, don't leak the device
            raise

    # ---- framework lifecycle --------------------------------------------- #
    def close(self):
        # stop any running acquisition first, then release the device
        self._stop_flag.set()
        reader = getattr(self, '_reader', None)
        if reader is not None:
            reader.join(timeout=10.0)
            self._reader = None
        mh = getattr(self, '_mh', None)
        if mh is not None:
            try:
                mh.close()
            finally:
                self._mh = None

    def _connect(self):
        """Open + initialize (T2) + configure the sync and both input channels.

        Idempotent: returns immediately if already connected."""
        if getattr(self, '_mh', None) is not None:
            return 0

        mh = MHLibWrapper(self.settings['dll_path'] or None)
        print(f"[MultiHarp] MHLib version {mh.library_version()}")
        idx, serial = mh.open(int(self.settings['device_index']))
        print(f"[MultiHarp] opened dev {idx} (S/N {serial})")

        # MH_Initialize MUST precede GetHardwareInfo / GetBaseResolution /
        # GetNumOfInputChannels (else MH_ERROR_NOT_INITIALIZED, -22).
        mh.initialize_t2()
        model, partno, version = mh.hardware_info()
        self._base_res_ps = mh.base_resolution_ps()
        nchan = mh.num_input_channels()
        self._nchan = nchan
        print(f"[MultiHarp] {model} (part {partno}, fw {version}) in T2 mode, "
              f"base resolution {self._base_res_ps:.3f} ps, {nchan} input channels")

        for ch in self._CH_KEYS:
            if not (0 <= ch < nchan):
                mh.close()
                raise ValueError(f"input channel {ch} out of range (device has {nchan})")

        # T2 mode: sync divider must be 1. The sync channel is unused here.
        mh.set_sync_div(int(self.settings['sync_divider']))
        mh.set_sync_trigger(-100, EDGE_FALLING)   # harmless placeholder
        mh.set_sync_offset(0)
        mh.set_sync_enable(False)                 # ignore sync (nothing plugged in)

        self._mh = mh
        # Configure every input from settings: enable/trigger the two we expose,
        # disable the rest.
        for ch in range(nchan):
            if ch in self._CH_KEYS:
                self._apply_channel_from_settings(ch)
            else:
                mh.set_input_enable(ch, False)
        time.sleep(0.2)  # let inputs settle after retriggering
        return 0

    @property
    def is_connected(self):
        return self._mh is not None

    def _apply_channel_from_settings(self, ch: int):
        """Push ch{ch}_* settings to the (already open) device."""
        if self._mh is None:
            return
        en_key, lvl_key, edge_key, off_key = self._CH_KEYS[ch]
        enabled = bool(self.settings[en_key])
        self._mh.set_input_enable(ch, enabled)
        if enabled:
            edge = self._EDGE[str(self.settings[edge_key]).lower()]
            self._mh.set_input_trigger(ch, int(self.settings[lvl_key]), edge)
            self._mh.set_input_offset(ch, int(self.settings[off_key]))

    # ---- framework probes ------------------------------------------------- #
    @property
    def _PROBES(self):
        return {
            'get_data': 'choose whether you need to get data from this device or not',
            'base_resolution_ps': 'T2 base resolution (ps)',
            'num_input_channels': 'number of hardware input channels',
            'library_version': 'MHLib library version string',
            'hardware_info': 'model / part number / firmware',
            'sync_rate': 'sync-channel rate (Hz)',
            'ch0_count_rate': 'input channel 0 count rate (Hz)',
            'ch1_count_rate': 'input channel 1 count rate (Hz)',
        }

    def read_probes(self, key):
        assert self._settings_initialized  # fails if settings/connection not yet initialized
        assert key in list(self._PROBES.keys())
        if key == 'get_data':
            return self.settings['get_data']
        if not self.is_connected:
            raise ValueError(f"MultiHarp not connected; cannot read probe {key!r}")
        if key == 'base_resolution_ps':
            return self._base_res_ps
        if key == 'num_input_channels':
            return self._nchan
        if key == 'library_version':
            return self._mh.library_version()
        if key == 'hardware_info':
            model, part, fw = self._mh.hardware_info()
            return f"{model} (part {part}, fw {fw})"
        if key == 'sync_rate':
            return self.get_sync_rate()
        if key == 'ch0_count_rate':
            return self.get_count_rate(0)
        if key == 'ch1_count_rate':
            return self.get_count_rate(1)
        raise KeyError(key)

    # ---- framework update ------------------------------------------------- #
    def update(self, settings):
        super(PicoQuant_MultiHarp, self).update(settings)
        for key, value in settings.items():
            if key in ('get_data', 'server_port', 'acquisition_time_s'):
                continue  # framework/runtime-only, nothing to push to hardware
            if key in self._CONNECTION_KEYS:
                self._connect()  # idempotent; also (re)applies all channels from settings
                if key == 'sync_divider' and self._mh is not None:
                    self._mh.set_sync_div(int(value))
            elif any(key in keys for keys in self._CH_KEYS.values()):
                self._connect()  # ensure the device is open before pushing a channel setting
                ch = next(c for c, keys in self._CH_KEYS.items() if key in keys)
                self._apply_channel_from_settings(ch)
            else:
                raise KeyError(key)

    # ---- programmatic multi-channel config (for >2 inputs, e.g. timing map) --
    def configure_channels(self, mapping: Dict[int, Tuple]):
        """Enable exactly the inputs in `mapping`, disable all others.

        mapping: {input_index: (level_mV, edge)} or {index: (label, level_mV, edge)};
        edge is 'rising'/'falling' or the EDGE_* int. Bypasses the ch0/ch1 settings
        for advanced use; requires the device to be open (opens it if needed)."""
        self._connect()
        mh = self._mh
        for ch in range(self._nchan):
            on = ch in mapping
            mh.set_input_enable(ch, on)
            if on:
                spec = mapping[ch]
                if len(spec) == 3:
                    _label, level_mV, edge = spec
                else:
                    level_mV, edge = spec
                edge_i = edge if isinstance(edge, int) else self._EDGE[str(edge).lower()]
                mh.set_input_trigger(ch, int(level_mV), edge_i)
                mh.set_input_offset(ch, 0)
        time.sleep(0.2)
        return self._nchan

    # ---- acquisition (background FIFO reader) ----------------------------- #
    def _read_loop(self):
        buffer = (ct.c_uint * TTREADMAX)()
        while not self._stop_flag.is_set():
            n = self._mh.read_fifo(buffer)
            if n > 0:
                self._chunks.append(np.frombuffer(buffer, dtype=np.uint32,
                                                  count=n).copy())
            else:
                if self._mh.flags() & FLAG_FIFOFULL:
                    self._fifo_overrun = True
                time.sleep(0.001)

    def start_acquisition(self, tacq_ms: int = _ACQTMAX_MS):
        """Start a long T2 acquisition and spin up the reader thread.

        `tacq_ms` just needs to outlast the run; we stop manually via
        stop_acquisition(). Use this when tagging alongside another experiment
        (start -> run the experiment -> stop)."""
        if self._mh is None:
            self._connect()
        self._chunks = []
        self._fifo_overrun = False
        self._stop_flag.clear()
        self._mh.start(int(tacq_ms))
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        print("[MultiHarp] acquisition started")

    def stop_acquisition(self) -> np.ndarray:
        """Stop the reader, stop the measurement, drain the FIFO, return raw records."""
        self._stop_flag.set()
        if self._reader is not None:
            self._reader.join(timeout=10.0)
            self._reader = None
        self._mh.stop()
        # Drain whatever is still in the FIFO after StopMeas.
        buffer = (ct.c_uint * TTREADMAX)()
        while True:
            n = self._mh.read_fifo(buffer)
            if n <= 0:
                break
            self._chunks.append(np.frombuffer(buffer, dtype=np.uint32, count=n).copy())
        if self._mh.flags() & FLAG_FIFOFULL:
            self._fifo_overrun = True
        records = (np.concatenate(self._chunks) if self._chunks
                   else np.empty(0, dtype=np.uint32))
        print(f"[MultiHarp] acquisition stopped, {records.size} raw records captured")
        return records

    def acquire(self, duration_s: Optional[float] = None) -> np.ndarray:
        """Blocking convenience: start, listen for `duration_s`, stop, return records.

        Defaults to the `acquisition_time_s` setting."""
        if duration_s is None:
            duration_s = float(self.settings['acquisition_time_s'])
        self.start_acquisition()
        time.sleep(float(duration_s))
        return self.stop_acquisition()

    @property
    def fifo_overrun(self) -> bool:
        """True if the FIFO overran during the last acquisition (data was lost)."""
        return self._fifo_overrun

    # ---- T2 decoding ------------------------------------------------------ #
    def _decode(self, records: np.ndarray):
        """Core T2 decode -> (times_ps, channel, is_event) arrays, overflows folded in."""
        r = records.astype(np.uint32)
        special = (r >> 31) & 0x1
        channel = (r >> 25) & 0x3F
        timetag = (r & 0x01FFFFFF).astype(np.uint64)
        # Overflow records: special==1 and channel==0x3F; timetag carries the
        # number of overflows (>=1) in the V2/MultiHarp format.
        is_ovf = (special == 1) & (channel == 0x3F)
        ovf_step = np.where(is_ovf,
                            np.where(timetag == 0, np.uint64(1), timetag),
                            np.uint64(0))
        ofl = np.cumsum(ovf_step) * np.uint64(T2WRAP)   # running offset per record
        times_ps = (ofl + timetag).astype(np.float64) * self._base_res_ps
        is_event = special == 0                          # real input-channel event
        return times_ps, channel, is_event

    def decode_all_channels(self, records: np.ndarray) -> Dict[int, np.ndarray]:
        """Decode into {channel_index: arrival_times_ps} for EVERY channel seen.

        Times are already time-ordered within each channel. Sync and markers are
        dropped; overflows are folded into absolute times."""
        if records.size == 0:
            return {}
        times_ps, channel, is_event = self._decode(records)
        out = {}
        for ch in np.unique(channel[is_event]):
            out[int(ch)] = times_ps[is_event & (channel == ch)]
        return out

    def decode_channels(self, records: np.ndarray,
                        channels) -> Dict[int, np.ndarray]:
        """Like decode_all_channels but only for the given channel indices."""
        if records.size == 0:
            return {int(ch): np.empty(0) for ch in channels}
        times_ps, channel, is_event = self._decode(records)
        return {int(ch): times_ps[is_event & (channel == int(ch))] for ch in channels}

    def get_timestamps(self, duration_s: Optional[float] = None,
                       channels=None) -> Dict[int, np.ndarray]:
        """Acquire for `duration_s` and return {channel: arrival_times_ps}.

        This is the primary primitive for higher-level experiments (e.g. HBT g2):
        it hands back the raw per-channel photon arrival times and leaves all
        correlation/histogramming to the caller. `channels=None` returns every
        channel seen; pass e.g. [0, 1] to restrict."""
        records = self.acquire(duration_s)
        if channels is None:
            return self.decode_all_channels(records)
        return self.decode_channels(records, channels)

    # ---- live rate readout ------------------------------------------------ #
    def get_sync_rate(self) -> float:
        self._connect()
        return float(self._mh.sync_rate())

    def get_count_rate(self, channel: int) -> float:
        self._connect()
        return float(self._mh.count_rate(int(channel)))

    def get_count_rates(self) -> Dict[int, float]:
        """{channel: count_rate_Hz} for the two exposed inputs that are enabled."""
        self._connect()
        rates = {}
        for ch, keys in self._CH_KEYS.items():
            if bool(self.settings[keys[0]]):  # ch{n}_enabled
                rates[ch] = float(self._mh.count_rate(ch))
        return rates

if __name__ == "__main__":
    mh = PicoQuant_MultiHarp()
    mh._connect()
    print(mh.is_connected)