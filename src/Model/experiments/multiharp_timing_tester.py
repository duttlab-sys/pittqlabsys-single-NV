# Written by <Jannet Trabelsi>
#!/usr/bin/env python3
"""
multiharp_timing_tester.py
==========================

Standalone timing-diagnostic harness for the ODMR pulsed experiment, built
around a PicoQuant MultiHarp 150/160 used as a two-channel time tagger.

Purpose
-------
Instead of hand-calibrating delays (running the experiment, eyeballing the
signal/reference with MW off, and nudging the calibration constants), this
class lets the MultiHarp *measure* the timing directly while the ODMR sequence
runs completely unchanged. It listens passively; it does not drive the Proteus,
the ADwin, or the SG384.

It supports two comparisons:

  case="readout_vs_spcm"   (default)
      start = Proteus channel 4 MKR, wired to mimic ONLY the readout pulse
      stop  = SPCM output
      -> builds the photon-arrival histogram relative to the readout edge and
         estimates the rise time (10%->90% of the leading edge). This is the
         "why is the rise time so long" measurement.

  case="init_vs_trigger"
      start = ADwin DIGOUT 28  (a copy of DIGOUT 21, the line that triggers
              the Proteus)
      stop  = Proteus channel 4 MKR, wired to mimic the init pulse
      -> measures trigger -> Proteus-output latency for every shot and reports
         how stable it is across the run (i.e. across tau steps and across
         repetitions).

Which physical MultiHarp BNC is "start" and which is "stop" is up to us; set
start_input / stop_input to the 0-based MultiHarp input-channel indices we
plugged into (default 0 and 1). Channel numbering here is 0-based to match
MHLib and the T2 record format; MHLib input channel 0 is the connector labelled
"1" on the front panel.

The MultiHarp inputs are level-trigger inputs with an operating range of only
-1200 mV to +1200 mV (pulse peak into 50 ohm) and a damage level of +/-2500 mV.
That means:
  * A raw SPCM TTL pulse (~2.5 V into 50 ohm for SPCM-AQRH) is at/over the
    damage level. ATTENUATE IT (e.g. a 6-10 dB inline attenuator) before the
    MultiHarp, and set the trigger level to about half the attenuated amplitude.
  * A raw ADwin DIGOUT (3.3 V / 5 V TTL) is over the damage level. ATTENUATE IT
    too.

Set the per-channel trigger levels/edges in the ChannelConfig objects below to
match whatever amplitudes are present.

Usage (from odmr_pulsed.py __main__ or a small test script)
-----------------------------------------------------------
    from multiharp_timing_tester import MultiHarpTimingTester

    experiment = ODMRPulsedExperiment(name="test_odmr", mode="testing")
    # ... set parameters and LOAD THE SEQUENCE exactly as we do for a normal run ...
    experiment.sequence_text = experiment.create_example_odmr_sequence()
    experiment.load_sequence_from_text()

    tester = MultiHarpTimingTester(case="readout_vs_spcm")   # or "init_vs_trigger"
    out = tester.run_with_experiment(
        experiment,
        experiment.settings['microwave']['frequency range'],
    )
    # out["odmr"]     -> the usual dict returned by experiment.run_experiment(...)
    # out["timing"]   -> TimingResult (histogram, stats, rise time / drift, files)

For these timing tests, run a SINGLE frequency (MW on or off doesn't matter for
readout/trigger timing) so the run is short and the drift-vs-time axis maps
cleanly onto the tau sweep.

----------------------------------------------------------------------
measured values (07/20/26)
Proteus Channel 4 MKR: 720 mV
Adwin DO28 (with 10 + 6 dB attenuators): 610 mV
Proteus Channel 3: 700 mV
Adwin DO16 (with 10 + 6 dB attenuators): 620 mV
SPCM (with 6 + 3 dB attenuators): 640 mV
>> all safe for Multiharp
"""

from __future__ import annotations

import ctypes as ct
import os
import platform
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Plotting is optional; the harness still returns all numbers if matplotlib
# isn't available or a display isn't present.
try:
    import matplotlib
    matplotlib.use("Agg")  # file output only; safe on headless lab PCs
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:  # pragma: no cover
    _HAVE_MPL = False


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


@dataclass
class ChannelConfig:
    """Trigger settings for one MultiHarp input (all levels in mV into 50 ohm)."""
    level_mV: int = 200
    edge: int = EDGE_RISING          # EDGE_RISING or EDGE_FALLING
    offset_ps: int = 0               # per-channel cable-delay compensation, +/-100 ns


@dataclass
class TimingResult:
    """Everything the analysis produces for one test run."""
    case: str
    start_input: int
    stop_input: int
    n_start_events: int
    n_stop_events: int
    n_pairs: int
    base_resolution_ps: float

    # Per-pair data (absolute start time and the start->stop delay), so we can
    # slice by tau iteration
    start_times_ns: np.ndarray = field(default_factory=lambda: np.empty(0))
    delays_ns: np.ndarray = field(default_factory=lambda: np.empty(0))

    # Histogram of the delay.
    hist_edges_ns: np.ndarray = field(default_factory=lambda: np.empty(0))
    hist_counts: np.ndarray = field(default_factory=lambda: np.empty(0))

    # Summary statistics of the delay distribution (ns).
    mean_ns: float = float("nan")
    median_ns: float = float("nan")
    std_ns: float = float("nan")
    fwhm_ns: float = float("nan")

    # Rise-time metrics (case="readout_vs_spcm"): the leading edge of the
    # arrival profile. All in ns; NaN when not applicable.
    onset_ns: float = float("nan")        # t at 10% of plateau
    rise_time_ns: float = float("nan")    # t(90%) - t(10%)
    plateau_counts: float = float("nan")

    # Drift across the run (case="init_vs_trigger" mainly): the run is split
    # into segments in time order; per-segment mean delay lets us confirm the
    # latency is constant over tau / repetitions.
    segment_mid_time_s: np.ndarray = field(default_factory=lambda: np.empty(0))
    segment_mean_delay_ns: np.ndarray = field(default_factory=lambda: np.empty(0))
    segment_count: np.ndarray = field(default_factory=lambda: np.empty(0))
    drift_pp_ns: float = float("nan")     # peak-to-peak of segment means

    fifo_overrun: bool = False
    saved_files: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"[MultiHarp timing] case={self.case}  "
            f"start=ch{self.start_input}  stop=ch{self.stop_input}",
            f"  events: start={self.n_start_events}  stop={self.n_stop_events}  "
            f"paired={self.n_pairs}",
            f"  delay: mean={self.mean_ns:.3f} ns  median={self.median_ns:.3f} ns  "
            f"std={self.std_ns:.3f} ns  FWHM={self.fwhm_ns:.3f} ns",
        ]
        if np.isfinite(self.rise_time_ns):
            lines.append(
                f"  rise: onset(10%)={self.onset_ns:.2f} ns  "
                f"rise_time(10-90%)={self.rise_time_ns:.2f} ns  "
                f"plateau={self.plateau_counts:.0f} cts/bin"
            )
        if np.isfinite(self.drift_pp_ns):
            lines.append(f"  drift across run (peak-to-peak of segment means): "
                         f"{self.drift_pp_ns:.3f} ns")
        if self.fifo_overrun:
            lines.append("  WARNING: FIFO overrun occurred -- some events were lost.")
        if self.saved_files:
            lines.append("  saved: " + ", ".join(self.saved_files))
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Thin ctypes wrapper around MHLib. All the version-/platform-specific
# assumptions live here so there is exactly one place to adjust.
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
        # returns the count in *nactual. If we are on an OLD MHLib (v1/v2) whose
        # signature is (devidx, buffer, count, *nactual), add a c_int before the
        # pointer here and pass TTREADMAX in read_fifo() below.
        self.MH_ReadFiFo = sig("MH_ReadFiFo", [c_int, P_uint, P_int])

        # Optional: present in v3.1, absent on very old libs -> loaded lazily.
        try:
            self.MH_SetSyncChannelEnable = sig(
                "MH_SetSyncChannelEnable", [c_int, c_int])
        except AttributeError:
            self.MH_SetSyncChannelEnable = None

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

    def read_fifo(self, buffer) -> int:
        """Read one FIFO block into `buffer` (a c_uint * TTREADMAX). Returns count."""
        nactual = ct.c_int()
        self._check(self.MH_ReadFiFo(self.devidx, buffer, ct.byref(nactual)),
                    "ReadFiFo")
        # Old-MHLib variant, if we changed the prototype above:
        #   self.MH_ReadFiFo(self.devidx, buffer, TTREADMAX, ct.byref(nactual))
        return nactual.value

    def close(self):
        if self.devidx is not None:
            try:
                self.MH_CloseDevice(self.devidx)
            finally:
                self.devidx = None


# --------------------------------------------------------------------------- #
# The test harness
# --------------------------------------------------------------------------- #
class MultiHarpTimingTester:
    """
    Runs the ODMR sequence unchanged while the MultiHarp time-tags two inputs,
    then computes the delay distribution / rise time / drift.
    """

    # Presets pick the correlation strategy, labels and sensible defaults.
    _PRESETS = {
        "readout_vs_spcm": dict(
            corr_mode="all_in_window",   # collect ALL photons after each readout edge
            window_ns=2000.0,            # look 2 us past the readout edge
            start_label="Proteus ch4 MKR (readout copy)",
            stop_label="SPCM",
        ),
        "init_vs_trigger": dict(
            corr_mode="nearest",         # one init pulse per trigger
            window_ns=4000.0,            # trigger->output should be well under 4 us
            start_label="ADwin DIGOUT 28 (trigger copy of DIGOUT 21)",
            stop_label="Proteus ch4 MKR (init copy)",
        ),
        "adwin_readout_vs_spcm": dict(
            corr_mode="around",
            window_ns=4000.0,            # look this far AFTER the gate-open edge
            pre_window_ns=1000.0,        # ...and this far BEFORE it (negative dt)
            start_label="ADwin DIGOUT 16 gate-open (readout)",   # start = reference
            stop_label="SPCM",
        ),
    }

    def __init__(
        self,
        case: str = "readout_vs_spcm",
        start_input: int = 0,
        stop_input: int = 1,
        start_cfg: Optional[ChannelConfig] = None,
        stop_cfg: Optional[ChannelConfig] = None,
        window_ns: Optional[float] = None,
        hist_bin_ns: float = 1.0,
        n_drift_segments: int = 20,
        sync_div: int = 1,
        dll_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        make_plots: bool = True,
        output_mode: str = "both",
        save_hdf5: bool = True,
        save_npz: bool = True,
        pre_window_ns: Optional[float] = None,
        gate_ns: Optional[float] = None,
        scan_all_channels: bool = False,
    ):
        if case not in self._PRESETS:
            raise ValueError(f"case must be one of {list(self._PRESETS)}, got {case!r}")
        self.case = case
        preset = self._PRESETS[case]
        self.corr_mode = preset["corr_mode"]
        self.start_label = preset["start_label"]
        self.stop_label = preset["stop_label"]

        # What to plot / save as the primary representation:
        #   "histogram"   -> the start->stop delay histogram (arrival profile / latency)
        #   "time_series" -> the per-shot delay vs time through the run ("data over time")
        #   "both"        -> save everything, plot both panels (default)
        # Aliases accepted for convenience.
        aliases = {
            "hist": "histogram", "histogram": "histogram",
            "time": "time_series", "timeseries": "time_series",
            "time_series": "time_series", "data_over_time": "time_series",
            "both": "both", "all": "both",
        }
        key = str(output_mode).lower().replace(" ", "_")
        if key not in aliases:
            raise ValueError(
                f"output_mode must be 'histogram', 'time_series', or 'both' "
                f"(got {output_mode!r})")
        self.output_mode = aliases[key]
        self.save_hdf5 = bool(save_hdf5)
        self.save_npz = bool(save_npz)

        self.start_input = int(start_input)
        self.stop_input = int(stop_input)
        # Default trigger levels are conservative; OVERRIDE to match our (attenuated!)
        # amplitudes. See the hardware note in the module docstring.
        self.start_cfg = start_cfg or ChannelConfig(level_mV=200, edge=EDGE_RISING)
        self.stop_cfg = stop_cfg or ChannelConfig(level_mV=200, edge=EDGE_RISING)

        self.window_ns = float(window_ns) if window_ns is not None else preset["window_ns"]
        self.pre_window_ns = (float(pre_window_ns) if pre_window_ns is not None
                              else float(preset.get("pre_window_ns", 0.0)))
        self.gate_ns = gate_ns
        self.scan_all_channels = bool(scan_all_channels)
        self.hist_bin_ns = float(hist_bin_ns)
        self.n_drift_segments = int(n_drift_segments)
        self.sync_div = int(sync_div)
        self.dll_path = dll_path
        self.output_dir = output_dir
        self.make_plots = make_plots and _HAVE_MPL

        self._mh: Optional[MHLibWrapper] = None
        self._nchan: Optional[int] = None
        self._base_res_ps: float = 5.0
        self._reader: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self._chunks: List[np.ndarray] = []
        self._fifo_overrun = False

    # ---- device bring-up -------------------------------------------------- #
    def open_and_configure(self):
        mh = MHLibWrapper(self.dll_path)
        print(f"[MultiHarp] MHLib version {mh.library_version()}")
        idx, serial = mh.open_first()
        print(f"[MultiHarp] opened dev {idx} (S/N {serial})")
        # MH_Initialize MUST come before GetHardwareInfo / GetBaseResolution /
        # GetNumOfInputChannels -- those return MH_ERROR_NOT_INITIALIZED (-22)
        # if the device has not been put into a measurement mode yet.
        mh.initialize_t2()
        model, partno, version = mh.hardware_info()
        self._base_res_ps = mh.base_resolution_ps()
        nchan = mh.num_input_channels()
        print(f"[MultiHarp] {model} (part {partno}, fw {version}) in T2 mode, "
              f"base resolution {self._base_res_ps:.3f} ps, {nchan} input channels")

        for ch in (self.start_input, self.stop_input):
            if not (0 <= ch < nchan):
                mh.close()
                raise ValueError(f"input channel {ch} out of range (device has {nchan})")
        if self.start_input == self.stop_input:
            mh.close()
            raise ValueError("start_input and stop_input must be different channels")

        # T2 mode: sync divider must be 1. We don't use the sync channel here.
        mh.set_sync_div(self.sync_div)
        mh.set_sync_trigger(-100, EDGE_FALLING)   # harmless placeholder
        mh.set_sync_offset(0)
        mh.set_sync_enable(False)                 # ignore sync (nothing plugged in)

        self._nchan = nchan
        if self.scan_all_channels:
            edge = 'rising' if self.start_cfg.edge == EDGE_RISING else 'falling'
            print(f"[MultiHarp] SCAN MODE: all {nchan} inputs @ "
                  f"{self.start_cfg.level_mV} mV, {edge} edge (levels equal on all).")
            for ch in range(nchan):
                mh.set_input_enable(ch, True)
                mh.set_input_trigger(ch, self.start_cfg.level_mV, self.start_cfg.edge)
                mh.set_input_offset(ch, 0)
        else:
            for ch in range(nchan):
                mh.set_input_enable(ch, ch in (self.start_input, self.stop_input))
            mh.set_input_trigger(self.start_input, self.start_cfg.level_mV, self.start_cfg.edge)
            mh.set_input_offset(self.start_input, self.start_cfg.offset_ps)
            mh.set_input_trigger(self.stop_input, self.stop_cfg.level_mV, self.stop_cfg.edge)
            mh.set_input_offset(self.stop_input, self.stop_cfg.offset_ps)

        time.sleep(0.2)  # let the inputs settle after retriggering
        self._mh = mh

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

    def start_acquisition(self, tacq_ms: int = 360_000_00):
        """Start a long T2 acquisition and spin up the reader thread.

        tacq_ms just needs to outlast the ODMR run; we stop manually when the
        run returns. Default is 100 h (the MHLib maximum in v3.1)."""
        if self._mh is None:
            self.open_and_configure()
        self._chunks = []
        self._fifo_overrun = False
        self._stop_flag.clear()
        self._mh.start(tacq_ms)
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        print("[MultiHarp] acquisition started (listening while ODMR runs)")

    def stop_acquisition(self) -> np.ndarray:
        """Stop the reader, stop the measurement, drain the FIFO, return records."""
        self._stop_flag.set()
        if self._reader is not None:
            self._reader.join(timeout=10.0)
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

    def close(self):
        if self._mh is not None:
            self._mh.close()
            self._mh = None

    # ---- T2 decoding ------------------------------------------------------ #
    def decode_t2(self, records: np.ndarray) -> Dict[int, np.ndarray]:
        """Decode T2 records into per-input-channel arrival times in ps.

        Returns {channel_index: times_ps}. Only regular photon/event records are
        returned (overflows are folded in, sync and markers are dropped)."""
        if records.size == 0:
            return {self.start_input: np.empty(0), self.stop_input: np.empty(0)}

        r = records.astype(np.uint32)
        special = (r >> 31) & 0x1
        channel = (r >> 25) & 0x3F
        timetag = (r & 0x01FFFFFF).astype(np.uint64)

        # Overflow records: special==1 and channel==0x3F. In the V2/MultiHarp
        # format the timetag field carries the number of overflows (>=1).
        is_ovf = (special == 1) & (channel == 0x3F)
        ovf_step = np.where(is_ovf,
                            np.where(timetag == 0, np.uint64(1), timetag),
                            np.uint64(0))
        ofl = np.cumsum(ovf_step) * np.uint64(T2WRAP)  # running offset per record
        true_units = ofl + timetag                     # base-resolution units
        times_ps = true_units.astype(np.float64) * self._base_res_ps

        is_event = special == 0                         # real input-channel event
        out = {}
        for ch in (self.start_input, self.stop_input):
            mask = is_event & (channel == ch)
            out[ch] = times_ps[mask]                    # already time-ordered
        return out

    def decode_all_channels(self, records: np.ndarray) -> Dict[int, np.ndarray]:
        """Like decode_t2 but returns {channel: times_ps} for EVERY channel."""
        if records.size == 0:
            return {}
        r = records.astype(np.uint32)
        special = (r >> 31) & 0x1
        channel = (r >> 25) & 0x3F
        timetag = (r & 0x01FFFFFF).astype(np.uint64)
        is_ovf = (special == 1) & (channel == 0x3F)  # <- match decode_t2
        ovf_step = np.where(is_ovf, np.where(timetag == 0, np.uint64(1), timetag),
                            np.uint64(0))
        ofl = np.cumsum(ovf_step) * np.uint64(T2WRAP)  # <- match decode_t2
        times_ps = (ofl + timetag).astype(np.float64) * self._base_res_ps
        is_event = special == 0
        out = {}
        for ch in np.unique(channel[is_event]):
            out[int(ch)] = times_ps[is_event & (channel == ch)]
        return out

    def _report_channels(self, records: np.ndarray) -> Dict[int, int]:
        allch = self.decode_all_channels(records)
        if not allch:
            print("[MultiHarp] DIAGNOSTIC: 0 events on ALL channels. Nothing is "
                  "triggering. Check the signals are present during the run, the "
                  "level sits ~half the ATTENUATED amplitude, and the edge matches.")
            return {}
        allt = np.concatenate(list(allch.values()))
        span_s = (allt.max() - allt.min()) / 1e12 if allt.size > 1 else 0.0
        print("[MultiHarp] per-channel event counts (0-based input; connector = index+1):")
        for ch in sorted(allch):
            n = allch[ch].size
            rate = n / span_s if span_s > 0 else 0.0
            tags = []
            if ch == self.start_input: tags.append("<-START")
            if ch == self.stop_input: tags.append("<-STOP")
            print(f"    input {ch}: {n:>10,} events  (~{rate:,.0f}/s)  {' '.join(tags)}")
        return {ch: allch[ch].size for ch in allch}

    # ---- correlation ------------------------------------------------------ #
    def _correlate(self, start_ps: np.ndarray, stop_ps: np.ndarray
                   ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (start_times_ps, delays_ps) according to self.corr_mode."""
        window_ps = self.window_ns * 1000.0
        if start_ps.size == 0 or stop_ps.size == 0:
            return np.empty(0), np.empty(0)

        if self.corr_mode == "nearest":
            idx = np.searchsorted(stop_ps, start_ps, side="right")
            valid = idx < stop_ps.size
            s = start_ps[valid]
            d = stop_ps[np.clip(idx[valid], 0, stop_ps.size - 1)] - s
            ok = (d >= 0) & (d <= window_ps)
            return s[ok], d[ok]

        if self.corr_mode == "around":
            pre_ps = self.pre_window_ns * 1000.0
            lo = np.searchsorted(stop_ps, start_ps - pre_ps, side="left")
            hi = np.searchsorted(stop_ps, start_ps + window_ps, side="right")
            starts_out, delays_out = [], []
            for i in range(start_ps.size):
                if hi[i] > lo[i]:
                    seg = stop_ps[lo[i]:hi[i]]
                    delays_out.append(seg - start_ps[i])
                    starts_out.append(np.full(seg.size, start_ps[i]))
            if not delays_out:
                return np.empty(0), np.empty(0)
            return np.concatenate(starts_out), np.concatenate(delays_out)

        # "all_in_window": every stop within (start, start+window] of each start
        lo = np.searchsorted(stop_ps, start_ps, side="right")
        hi = np.searchsorted(stop_ps, start_ps + window_ps, side="right")
        starts_out, delays_out = [], []
        for i in range(start_ps.size):
            if hi[i] > lo[i]:
                seg = stop_ps[lo[i]:hi[i]]
                delays_out.append(seg - start_ps[i])
                starts_out.append(np.full(seg.size, start_ps[i]))
        if not delays_out:
            return np.empty(0), np.empty(0)
        return np.concatenate(starts_out), np.concatenate(delays_out)

    # ---- statistics ------------------------------------------------------- #
    @staticmethod
    def _fwhm_from_hist(centers_ns: np.ndarray, counts: np.ndarray) -> float:
        if counts.size == 0 or counts.max() <= 0:
            return float("nan")
        half = counts.max() / 2.0
        above = np.where(counts >= half)[0]
        if above.size < 2:
            return float("nan")
        return float(centers_ns[above[-1]] - centers_ns[above[0]])

    @staticmethod
    def _rise_time(centers_ns: np.ndarray, counts: np.ndarray
                   ) -> Tuple[float, float, float]:
        """Estimate onset (10%), 10-90 rise, and plateau from an arrival profile.

        Plateau = median of the top 30% of the window (assumes the profile rises
        then flattens/decays slowly). Crossings are linearly interpolated on the
        first rising edge. Returns (onset_ns, rise_ns, plateau)."""
        if counts.size < 3 or counts.max() <= 0:
            return float("nan"), float("nan"), float("nan")
        n = counts.size
        plateau = float(np.median(counts[int(0.7 * n):])) if n >= 4 else float(counts.max())
        if plateau <= 0:
            plateau = float(counts.max())

        def cross(frac):
            thr = frac * plateau
            for i in range(1, n):
                if counts[i - 1] < thr <= counts[i]:
                    c0, c1 = counts[i - 1], counts[i]
                    t0, t1 = centers_ns[i - 1], centers_ns[i]
                    if c1 == c0:
                        return t1
                    return t0 + (thr - c0) * (t1 - t0) / (c1 - c0)
            return float("nan")

        t10, t90 = cross(0.10), cross(0.90)
        rise = (t90 - t10) if (np.isfinite(t10) and np.isfinite(t90)) else float("nan")
        return t10, rise, plateau

    def analyze(self, records: np.ndarray) -> TimingResult:
        chans = self.decode_t2(records)
        self._report_channels(records)
        start_ps = chans[self.start_input]
        stop_ps = chans[self.stop_input]
        s_ps, d_ps = self._correlate(start_ps, stop_ps)

        res = TimingResult(
            case=self.case,
            start_input=self.start_input,
            stop_input=self.stop_input,
            n_start_events=int(start_ps.size),
            n_stop_events=int(stop_ps.size),
            n_pairs=int(d_ps.size),
            base_resolution_ps=self._base_res_ps,
            start_times_ns=s_ps / 1000.0,
            delays_ns=d_ps / 1000.0,
            fifo_overrun=self._fifo_overrun,
        )
        if d_ps.size == 0:
            if start_ps.size == 0 or stop_ps.size == 0:
                dead = ("START (input %d)" % self.start_input if start_ps.size == 0
                        else "STOP (input %d)" % self.stop_input)
                print(f"[MultiHarp] NO PAIRS: the {dead} channel saw 0 events -- a "
                      f"detection problem, not correlation. Fix that channel's level/"
                      f"edge/wiring (attenuate TTL to <=1.2 V, level ~half amplitude). "
                      f"Flipping start/stop won't help while a channel reads zero. "
                      f"Try scan_all_channels=True.")
            else:
                print(f"[MultiHarp] NO PAIRS but both channels have events "
                      f"({start_ps.size}/{stop_ps.size}) -- a timing issue: with "
                      f"corr_mode={self.corr_mode!r} the stop never lands in the "
                      f"window. Use case='adwin_readout_vs_spcm' (corr_mode='around').")
            return res

        d_ns = d_ps / 1000.0
        res.mean_ns = float(np.mean(d_ns))
        res.median_ns = float(np.median(d_ns))
        res.std_ns = float(np.std(d_ns))

        if self.corr_mode == "around":
            lo_ns, hi_ns = -self.pre_window_ns, self.window_ns
        else:
            lo_ns, hi_ns = 0.0, self.window_ns
        nbins = max(10, int(round((hi_ns - lo_ns) / self.hist_bin_ns)))
        counts, edges = np.histogram(d_ns, bins=nbins, range=(lo_ns, hi_ns))
        centers = 0.5 * (edges[:-1] + edges[1:])
        res.hist_edges_ns = edges
        res.hist_counts = counts
        res.fwhm_ns = self._fwhm_from_hist(centers, counts)

        if self.case == "readout_vs_spcm":
            onset, rise, plateau = self._rise_time(centers, counts.astype(float))
            res.onset_ns, res.rise_time_ns, res.plateau_counts = onset, rise, plateau

        # Drift across the run: bin pairs by absolute start time.
        if s_ps.size >= self.n_drift_segments:
            t_s = (s_ps - s_ps.min()) / 1e12  # ps -> s
            seg_edges = np.linspace(t_s.min(), t_s.max(), self.n_drift_segments + 1)
            which = np.clip(np.digitize(t_s, seg_edges) - 1, 0, self.n_drift_segments - 1)
            mids, means, cnts = [], [], []
            for k in range(self.n_drift_segments):
                sel = which == k
                if np.any(sel):
                    mids.append(0.5 * (seg_edges[k] + seg_edges[k + 1]))
                    means.append(float(np.mean(d_ns[sel])))
                    cnts.append(int(np.sum(sel)))
            res.segment_mid_time_s = np.asarray(mids)
            res.segment_mean_delay_ns = np.asarray(means)
            res.segment_count = np.asarray(cnts)
            if means:
                res.drift_pp_ns = float(np.max(means) - np.min(means))

        return res

    # ---- HDF5 saving (for the data-analyzer GUI) -------------------------- #
    #
    # ============================ WHERE TO SAVE ============================= #
    # This is the method that writes the MultiHarp timing data to disk in the
    # same /data + /meta group layout ODMRPulsedExperiment.save_hdf5 uses,
    # so data-analyzer GUI can open it the same way. It is called from
    # _save_outputs(), which run_with_experiment() calls automatically after the
    # run.
    # ======================================================================= #
    def _save_hdf5(self, res: TimingResult, base: str) -> Optional[str]:
        try:
            import h5py
        except Exception as e:
            print(f"[MultiHarp] h5py not available, skipping HDF5 save ({e}). "
                  f"The .npz still holds all arrays.")
            return None
        from src.core.struct_hdf5 import MyStruct, save_data
        s = MyStruct()
        s.data = MyStruct(start_times_ns=res.start_times_ns, delays_ns=res.delays_ns, segment_mid_time_s = res.segment_mid_time_s, segment_mean_delay_ns = res.segment_mean_delay_ns, segment_count = res.segment_count,
                              hist_counts=res.hist_counts, hist_edges_ns=res.hist_edges_ns)
        s.meta = MyStruct(case=self.case,
                    output_mode=self.output_mode,
                    correlation_mode=self.corr_mode,
                    start_input=self.start_input,
                    stop_input=self.stop_input,
                    start_label=self.start_label,
                    stop_label=self.stop_label,
                    base_resolution_ps=res.base_resolution_ps,
                    window_ns=self.window_ns,
                    hist_bin_ns=self.hist_bin_ns,
                    n_start_events=res.n_start_events,
                    n_stop_events=res.n_stop_events,
                    n_pairs=res.n_pairs,
                    mean_ns=res.mean_ns,
                    median_ns=res.median_ns,
                    std_ns=res.std_ns,
                    fwhm_ns=res.fwhm_ns,
                    onset_ns=res.onset_ns,
                    rise_time_ns=res.rise_time_ns,
                    plateau_counts=res.plateau_counts,
                    drift_pp_ns=res.drift_pp_ns,
                    fifo_overrun=int(res.fifo_overrun),
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
        save_data(base + ".h5", s)

    # ---- plotting / saving ------------------------------------------------ #
    def _plot_histogram(self, ax, res: TimingResult) -> None:
        centers = 0.5 * (res.hist_edges_ns[:-1] + res.hist_edges_ns[1:])
        ax.bar(centers, res.hist_counts,
               width=(centers[1] - centers[0]) if centers.size > 1 else 1.0,
               align="center")
        ax.set_xlabel(f"delay: {self.stop_label} - {self.start_label} (ns)")
        ax.set_ylabel("counts")
        title = f"{self.case}: median={res.median_ns:.2f} ns, FWHM={res.fwhm_ns:.2f} ns"
        if self.corr_mode == "around":
            ax.set_xlabel(f"{self.stop_label} arrival relative to {self.start_label} (ns)")
            ax.axvline(0.0, color="k", lw=1.2)  # gate opens
            if self.gate_ns:
                ax.axvspan(0.0, self.gate_ns, alpha=0.15, color="green")
                ax.axvline(self.gate_ns, color="k", ls=":", lw=1)  # gate closes
                frac_in = np.mean((res.delays_ns >= 0) & (res.delays_ns <= self.gate_ns)) * 100
                title += f"\nphotons inside gate [0,{self.gate_ns:.0f}] ns: {frac_in:.0f}%"
        else:
            ax.set_xlabel(f"delay: {self.stop_label} - {self.start_label} (ns)")
            if np.isfinite(res.rise_time_ns):
                title += f"\nrise(10-90%)={res.rise_time_ns:.1f} ns"
                for t in (res.onset_ns, res.onset_ns + res.rise_time_ns):
                    if np.isfinite(t):
                        ax.axvline(t, ls="--", lw=1)
        ax.set_title(title)

    def _plot_time_series(self, ax, res: TimingResult) -> None:
        # Per-shot delay vs time through the run ("data over time"), with the
        # binned mean overlaid so drift is easy to see.
        if res.delays_ns.size:
            t_s = (res.start_times_ns - res.start_times_ns.min()) / 1e9
            n = t_s.size
            step = max(1, n // 200_000)          # keep the scatter light
            ax.plot(t_s[::step], res.delays_ns[::step], ".", ms=2, alpha=0.25,
                    label="per shot")
        if res.segment_mean_delay_ns.size:
            ax.plot(res.segment_mid_time_s, res.segment_mean_delay_ns,
                    "-o", lw=1.5, label="binned mean")
        ax.set_xlabel("time through run (s)  ~ tau-sweep progression")
        ax.set_ylabel(f"delay: {self.stop_label} - {self.start_label} (ns)")
        pp = res.drift_pp_ns if np.isfinite(res.drift_pp_ns) else float("nan")
        ax.set_title(f"data over time (drift peak-to-peak {pp:.2f} ns)")
        ax.legend(loc="best", fontsize=8)

    def _save_outputs(self, res: TimingResult, tag: str) -> None:
        outdir = self.output_dir or os.getcwd()
        os.makedirs(outdir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        base = os.path.join(outdir, f"multiharp_{self.case}_{self.output_mode}_{stamp}")

        if self.save_npz:
            npz = base + ".npz"
            np.savez_compressed(
                npz,
                case=self.case, output_mode=self.output_mode,
                start_input=self.start_input, stop_input=self.stop_input,
                base_resolution_ps=self._base_res_ps,
                start_times_ns=res.start_times_ns, delays_ns=res.delays_ns,
                hist_edges_ns=res.hist_edges_ns, hist_counts=res.hist_counts,
                segment_mid_time_s=res.segment_mid_time_s,
                segment_mean_delay_ns=res.segment_mean_delay_ns,
                segment_count=res.segment_count,
            )
            res.saved_files.append(npz)

        if self.save_hdf5:
            self._save_hdf5(res, base)

        if not self.make_plots or res.n_pairs == 0:
            return
        try:
            if self.output_mode == "both":
                fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
                self._plot_histogram(axes[0], res)
                self._plot_time_series(axes[1], res)
            elif self.output_mode == "histogram":
                fig, ax = plt.subplots(figsize=(7, 4.5))
                self._plot_histogram(ax, res)
            else:  # time_series
                fig, ax = plt.subplots(figsize=(7, 4.5))
                self._plot_time_series(ax, res)
            fig.tight_layout()
            png = base + ".png"
            fig.savefig(png, dpi=130)
            plt.close(fig)
            res.saved_files.append(png)
        except Exception as e:  # never let plotting kill a good measurement
            print(f"[MultiHarp] plot warning: {e}")

    # ---- top-level entry point ------------------------------------------- #
    def run_with_experiment(
        self,
        experiment: Any,
        frequency_range: Optional[List[float]] = None,
        save: bool = True,
        run_method: str = "run_experiment",
    ) -> Dict[str, Any]:
        """Open the MultiHarp, start tagging, run the ODMR experiment unchanged,
        stop, analyze, and return {'odmr': <run result>, 'timing': TimingResult}.

        run_method selects which experiment method to run while tagging:
          "run_experiment"          -> a single pass (default)
          "run_experiment_averaged" -> averaged run; the MultiHarp tags
                                       across all averages, so the drift/
                                       time-series view then spans every average.
        """
        if frequency_range is None:
            frequency_range = experiment.settings['microwave']['frequency range']
        if not hasattr(experiment, run_method):
            raise AttributeError(f"experiment has no method {run_method!r}")

        # Prefer the experiment's own output directory if it exposes one.
        if self.output_dir is None:
            self.output_dir = (getattr(experiment, "output_dir", None)
                               or os.getcwd())

        self.open_and_configure()
        odmr_result: Dict[str, Any] = {}
        try:
            self.start_acquisition()
            try:
                # ODMR runs exactly as in a normal run -- we don't touch it.
                odmr_result = getattr(experiment, run_method)(frequency_range)
            finally:
                records = self.stop_acquisition()
            timing = self.analyze(records)
            if save:
                self._save_outputs(timing, tag=getattr(experiment, "tag", "odmr"))
        finally:
            self.close()

        print(timing.summary())
        return {"odmr": odmr_result, "timing": timing}

    # ---- multi-channel timing map ---------------------------------------- #
    # Enable several inputs at once, run the sequence, and histogram EVERY
    # channel relative to ONE chosen reference channel. In a single run this
    # draws the full timing diagram of a pulse sequence -- trigger, laser
    # marker, counter-open/close proxies, SPCM, ... -- so alignment generalizes
    # to any sequence. For the counter-alignment run, set reference_input to the
    # counter-OPEN proxy (the digout toggled right after Cnt_Enable): the SPCM
    # histogram then shows the photon burst relative to the ACTUAL counter
    # opening, and the counter-CLOSE proxy shows the true integration window
    # (Cnt_Enable -> Cnt_Latch), independent of whatever count_time you told
    # ADwin. `channels` maps input index -> (label, level_mV, edge); remember to
    # attenuate TTLs to <=1.2 V and set each level to ~half its amplitude.
    def _open_and_configure_map(self, channels: Dict[int, Tuple[str, int, int]]):
        """Open the device and enable exactly the inputs listed in `channels`."""
        mh = MHLibWrapper(self.dll_path)
        print(f"[MultiHarp] MHLib version {mh.library_version()}")
        idx, serial = mh.open_first()
        print(f"[MultiHarp] opened dev {idx} (S/N {serial})")
        mh.initialize_t2()                        # must precede the queries below
        model, partno, version = mh.hardware_info()
        self._base_res_ps = mh.base_resolution_ps()
        nchan = mh.num_input_channels()
        self._nchan = nchan
        print(f"[MultiHarp] {model} (part {partno}, fw {version}) in T2 mode, "
              f"base resolution {self._base_res_ps:.3f} ps, {nchan} input channels")
        for ch in channels:
            if not (0 <= ch < nchan):
                mh.close()
                raise ValueError(f"input channel {ch} out of range (device has {nchan})")
        mh.set_sync_div(self.sync_div)
        mh.set_sync_trigger(-100, EDGE_FALLING)
        mh.set_sync_offset(0)
        mh.set_sync_enable(False)
        for ch in range(nchan):
            on = ch in channels
            mh.set_input_enable(ch, on)
            if on:
                _label, level_mV, edge = channels[ch]
                mh.set_input_trigger(ch, level_mV, edge)
                mh.set_input_offset(ch, 0)
        time.sleep(0.2)
        self._mh = mh

    @staticmethod
    def _events_around(ref_ps: np.ndarray, other_ps: np.ndarray,
                       pre_ps: float, post_ps: float) -> np.ndarray:
        """All `other` timestamps within [ref-pre, ref+post] of each ref edge,
        returned as signed dt (other - ref) in ps."""
        if ref_ps.size == 0 or other_ps.size == 0:
            return np.empty(0)
        lo = np.searchsorted(other_ps, ref_ps - pre_ps, side="left")
        hi = np.searchsorted(other_ps, ref_ps + post_ps, side="right")
        out = []
        for i in range(ref_ps.size):
            if hi[i] > lo[i]:
                out.append(other_ps[lo[i]:hi[i]] - ref_ps[i])
        return np.concatenate(out) if out else np.empty(0)

    def _report_channels_map(self, allch, channels, reference_input):
        if not allch:
            print("[MultiHarp] DIAGNOSTIC: 0 events on ALL channels -- nothing "
                  "triggered. Check trigger levels/edges/wiring.")
            return
        allt = np.concatenate(list(allch.values()))
        span_s = (allt.max() - allt.min()) / 1e12 if allt.size > 1 else 0.0
        print("[MultiHarp] per-channel event counts:")
        for ch in sorted(allch):
            n = allch[ch].size
            rate = n / span_s if span_s > 0 else 0.0
            lbl = channels.get(ch, ("(not configured)", 0, 0))[0]
            tag = "  <-REFERENCE" if ch == reference_input else ""
            print(f"    input {ch}: {n:>10,} events (~{rate:,.0f}/s)  {lbl}{tag}")

    def analyze_timing_map(self, records, channels, reference_input,
                           pre_ns=1000.0, post_ns=4000.0, hist_bin_ns=5.0):
        """Histogram every non-reference channel relative to reference_input.
        Returns {input_index: {...}} and prints a summary table."""
        allch = self.decode_all_channels(records)
        self._report_channels_map(allch, channels, reference_input)
        if reference_input not in allch or allch[reference_input].size == 0:
            print(f"[MultiHarp] reference channel {reference_input} "
                  f"({channels[reference_input][0]}) saw NO events -- cannot build a "
                  f"map. Fix its trigger level/edge/wiring.")
            return {}
        ref_ps = allch[reference_input]
        pre_ps, post_ps = pre_ns * 1000.0, post_ns * 1000.0
        lo_ns, hi_ns = -pre_ns, post_ns
        nbins = max(20, int(round((hi_ns - lo_ns) / hist_bin_ns)))
        results = {}
        print(f"\n[MultiHarp] timing map -- all channels relative to "
              f"'{channels[reference_input][0]}' (input {reference_input}) at t=0:")
        print(f"    {'channel':<34} {'events':>9} {'center_ns':>10} "
              f"{'peak_ns':>9} {'FWHM_ns':>8}")
        for ch in sorted(channels):
            if ch == reference_input:
                continue
            dt_ns = self._events_around(ref_ps, allch.get(ch, np.empty(0)),
                                        pre_ps, post_ps) / 1000.0
            entry = dict(label=channels[ch][0], input=ch, n=int(dt_ns.size),
                         delays_ns=dt_ns, hist_counts=np.empty(0),
                         hist_edges_ns=np.empty(0), center_ns=float("nan"),
                         peak_ns=float("nan"), fwhm_ns=float("nan"))
            if dt_ns.size:
                counts, edges = np.histogram(dt_ns, bins=nbins, range=(lo_ns, hi_ns))
                ctr = 0.5 * (edges[:-1] + edges[1:])
                entry.update(hist_counts=counts, hist_edges_ns=edges,
                             center_ns=float(np.median(dt_ns)),
                             peak_ns=float(ctr[np.argmax(counts)]),
                             fwhm_ns=self._fwhm_from_hist(ctr, counts))
                print(f"    {channels[ch][0]:<34} {dt_ns.size:>9,} "
                      f"{entry['center_ns']:>10.1f} {entry['peak_ns']:>9.1f} "
                      f"{entry['fwhm_ns']:>8.1f}")
            else:
                print(f"    {channels[ch][0]:<34} {'0':>9}   (no events in window)")
            results[ch] = entry
        return results

    def _save_timing_map(self, results, channels, reference_input, tag="map"):
        outdir = self.output_dir or os.getcwd()
        os.makedirs(outdir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        base = os.path.join(outdir, f"multiharp_timingmap_{stamp}")
        save = {"reference_input": reference_input,
                "reference_label": channels[reference_input][0]}
        for ch, e in results.items():
            save[f"ch{ch}_label"] = e["label"]
            save[f"ch{ch}_delays_ns"] = e["delays_ns"]
            save[f"ch{ch}_hist_counts"] = e["hist_counts"]
            save[f"ch{ch}_hist_edges_ns"] = e["hist_edges_ns"]
        npz = base + ".npz"
        np.savez_compressed(npz, **save)
        files = [npz]
        if self.make_plots and results:
            try:
                chs = sorted(results)
                fig, axes = plt.subplots(len(chs), 1, sharex=True,
                                         figsize=(10, 1.9 * len(chs) + 0.6))
                if len(chs) == 1:
                    axes = [axes]
                for ax, ch in zip(axes, chs):
                    e = results[ch]
                    if e["hist_counts"].size:
                        ctr = 0.5 * (e["hist_edges_ns"][:-1] + e["hist_edges_ns"][1:])
                        ax.fill_between(ctr, e["hist_counts"], step="mid", alpha=.6)
                        ax.axvline(e["peak_ns"], ls="--", lw=1, color="k")
                    ax.axvline(0, color="red", lw=1.2)     # reference at t=0
                    ax.set_ylabel(e["label"], fontsize=8, rotation=0,
                                  ha="right", va="center")
                    ax.set_yticks([])
                axes[-1].set_xlabel(f"time relative to "
                                    f"'{channels[reference_input][0]}' "
                                    f"(input {reference_input})  [ns]")
                axes[0].set_title("MultiHarp timing map  (red = reference at t=0)")
                fig.tight_layout()
                png = base + ".png"
                fig.savefig(png, dpi=130)
                plt.close(fig)
                files.append(png)
            except Exception as ex:
                print(f"[MultiHarp] timing-map plot warning: {ex}")
        print("[MultiHarp] timing map saved: " + ", ".join(files))
        return files

    def run_timing_map(self, experiment, channels: Dict[int, Tuple[str, int, int]],
                       reference_input: int, frequency_range=None,
                       pre_ns: float = 1000.0, post_ns: float = 4000.0,
                       hist_bin_ns: float = 5.0, run_method: str = "run_experiment",
                       save: bool = True) -> Dict[str, Any]:
        """One run, many channels -> the full timing diagram of the sequence.

        channels        : {input_index: (label, level_mV, edge)} -- what each BNC
                          carries and its (attenuated!) trigger level and edge.
        reference_input : the input used as t=0 for every histogram. Set it to the
                          counter-OPEN proxy for the counter-alignment run, or to
                          the trigger to see the whole sequence laid out forward.
        pre_ns/post_ns  : how far before/after each reference edge to collect.
        """
        if reference_input not in channels:
            raise ValueError("reference_input must be one of the channels")
        if frequency_range is None:
            frequency_range = experiment.settings['microwave']['frequency range']
        if not hasattr(experiment, run_method):
            raise AttributeError(f"experiment has no method {run_method!r}")
        if self.output_dir is None:
            self.output_dir = getattr(experiment, "output_dir", None) or os.getcwd()

        self._open_and_configure_map(channels)     # sets self._mh
        odmr_result: Dict[str, Any] = {}
        try:
            self.start_acquisition()                # reuses the already-open device
            try:
                odmr_result = getattr(experiment, run_method)(frequency_range)
            finally:
                records = self.stop_acquisition()
            results = self.analyze_timing_map(records, channels, reference_input,
                                              pre_ns=pre_ns, post_ns=post_ns,
                                              hist_bin_ns=hist_bin_ns)
            if save:
                self._save_timing_map(results, channels, reference_input,
                                      tag=getattr(experiment, "tag", "map"))
        finally:
            self.close()
        return {"odmr": odmr_result, "map": results}
