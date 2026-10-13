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

  case="rabi_vs_spcm"
      start = a once-per-shot reference edge -- the readout laser MARKER is best
              (add `marker, laser_readout_1 on channel 4 at <readout>ns,
              <len>ns` to the sequence so the Proteus emits one clean edge at the
              readout onset); the trigger works too.
      stop  = SPCM output
      -> reconstructs the Rabi curve FROM THE MULTIHARP, with no ADwin gating:
         per shot it counts SPCM photons in a signal window on the bright readout
         onset (and, optionally, a reference window later inside the same readout
         pulse), groups shots by tau, and averages. This is the "is it the ADwin
         or a real drift?" test: a clean oscillation here while the ADwin curve is
         flat pins the problem on the ADwin counting/gate; a flat curve here means
         the problem is upstream (polarization / MW / optics / a genuine drift).
         See run_rabi_with_experiment and analyze_rabi (and the one desync caveat).

  case="rabi_mw_tagged"   (the desync-proof Rabi)
      start = the readout MARKER (per-shot reference + SPCM signal window), as above
      stop  = SPCM output
      mw    = a THIRD input wired to an attenuated copy of the MW line (the pulse
              whose WIDTH is tau). Set mw_input=<ch> (+ optional mw_fall_input=<ch>).
      -> identical to rabi_vs_spcm, except tau for each shot is read from the MW
         pulse itself instead of from shot order (shot i mod n). WIDTH mode (both
         mw_input and mw_fall_input) measures tau = MW-fall - MW-rise directly and
         needs no geometry; GEOMETRY mode (mw_input only, its FALLING edge) derives
         tau from the MW end vs the marker using the fixed MW->readout gap
         (mw_gap_ns). Each measured tau is snapped to the nearest sweep step, so a
         dropped edge drops only that one shot instead of shifting every later one.
         This is the bullet-proof way to run the "ADwin vs real drift" check.

  case="t1_vs_spcm"   (T1: SPCM counted only after the readout marker)
      trig  = ADwin DIGOUT 28 (copy of DIGOUT 21). Required, but used ONLY as a
              clock: it groups the marker edges into shots and times
              DO28 -> marker. No photon is ever counted relative to DO28.
      start = Proteus ch4 MKR copying the LASER: one marker line on the init
              pulse and one on the readout pulse (two rising edges per shot).
      stop  = SPCM output
      -> the DO28 -> marker durations give each shot its tau:
         tau = (DO28->readout) - (DO28->init) - t1_gap. An SPCM photon is
         registered ONLY if it falls inside the readout light after a shot's
         readout-marker edge; the init pulse and the dark time are never
         counted. Each shot's readout photons are binned under its measured
         tau, so a missing or extra edge costs one shot instead of shifting
         every later one. Also reported: DO28 -> readout jitter, which shot
         order the Proteus really plays (interleaved/blocked, task-pointer start
         per run), and the ADwin's PROGRAMMED gates replayed on the same readout
         photons vs its own counts. See run_t1_with_experiment / analyze_t1.

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


@dataclass
class RabiResult:
    """Everything the Rabi analysis produces for one MultiHarp-only Rabi run.

    The MultiHarp reconstructs the Rabi curve itself: per shot it counts SPCM
    photons in a signal window (and optionally a reference window) placed
    relative to a once-per-shot reference edge (the readout laser marker or the
    trigger), then groups shots by tau step and averages. This is deliberately
    INDEPENDENT of the ADwin gate, so a clean oscillation here with a bad one
    from the ADwin points squarely at the ADwin counting/gating -- and a bad one
    here points upstream (spin / optics / MW / a real setup drift)."""
    start_input: int
    stop_input: int
    n_ref_events: int          # once-per-shot reference edges seen
    n_spcm_events: int         # SPCM photons seen
    n_shots_used: int          # reference edges actually assigned to a tau bin
    n_iterations: int          # number of tau steps in the sweep
    repeat_count: Optional[int]
    base_resolution_ps: float

    # Window definitions (ns, relative to the reference edge), echoed for the record.
    sig_offset_ns: float = float("nan")
    sig_width_ns: float = float("nan")
    ref_offset_ns: float = float("nan")
    ref_width_ns: float = float("nan")
    reference_label: str = ""

    # The Rabi curve, one entry per tau step.
    tau_ns: np.ndarray = field(default_factory=lambda: np.empty(0))
    tau_is_index: bool = False                     # True if tau axis is just 0..n-1
    signal_mean: np.ndarray = field(default_factory=lambda: np.empty(0))     # <cts>/shot in sig window
    signal_sem: np.ndarray = field(default_factory=lambda: np.empty(0))      # standard error of the mean
    reference_mean: np.ndarray = field(default_factory=lambda: np.empty(0))  # <cts>/shot in ref window (NaN if unused)
    contrast: np.ndarray = field(default_factory=lambda: np.empty(0))        # signal/reference (NaN if no ref window)
    shots_per_tau: np.ndarray = field(default_factory=lambda: np.empty(0))

    # Time-resolved photon arrivals (the "keep the whole 7 us" data). For each
    # tau step, the full SPCM arrival-time distribution RELATIVE TO THE MARKER,
    # finely binned over the readout window. This is what lets you re-pick the
    # signal/reference windows offline (e.g. to skip the laser rise) WITHOUT
    # re-running: sum the bins over any range and divide by shots_per_tau. See
    # rabi_curve_from_histogram() / MultiHarpTimingTester.recompute_rabi().
    time_resolved: bool = False
    arrival_hist: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))  # [n_tau, n_bins] counts
    arrival_edges_ns: np.ndarray = field(default_factory=lambda: np.empty(0))   # n_bins+1 bin edges (ns after marker)
    arrival_profile_ns: np.ndarray = field(default_factory=lambda: np.empty(0)) # tau-independent readout, ALL shots

    # Simple oscillation descriptors (from the signal, or contrast if available).
    modulation_depth: float = float("nan")   # (max-min)/(max+min) of the curve
    best_metric: str = ""                     # "contrast" or "signal_mean"

    phase_ok: bool = True                     # False if shot count desynced from the sweep
    fifo_overrun: bool = False
    saved_files: List[str] = field(default_factory=list)

    # ---- MW-tagged tau assignment (the bullet-proof mode) ----------------- #
    # When an MW copy is wired to a third input, each shot's tau is read from the
    # MW pulse itself instead of from shot order, so a dropped edge can't smear
    # the curve -- every well-formed shot self-identifies its tau.
    tau_assign_source: str = "shot_order"     # "shot_order" | "mw_width" | "mw_geometry"
    mw_inputs: Tuple[int, ...] = ()           # MultiHarp input(s) carrying the MW copy
    n_matched: int = 0                        # shots whose measured tau snapped to a step
    n_unmatched: int = 0                      # shots the MW tag couldn't place (dropped)
    mw_snap_tol_ns: float = float("nan")      # snap tolerance actually used
    order_agreement: float = float("nan")     # fraction where MW tag == shot-order (cross-check)

    # ---- optional ADwin-trigger cross-check (DIGOUT 28 on its own input) --- #
    # DIGOUT 28 is a copy of DIGOUT 21 (the line that triggers the Proteus): one
    # edge per ADwin trigger = one per shot, emitted by the MASTER. Comparing its
    # count to the Proteus readout-marker count tests, in the SAME stream, whether
    # every trigger yields exactly one Proteus sequence (1:1). A marker with no
    # preceding trigger is a Proteus edge the ADwin never asked for (an init burst
    # before the sweep); a trigger with no marker is a dropped/absent sequence.
    trigger_input: Optional[int] = None
    n_trigger_events: int = 0
    n_orphan_markers: int = 0          # markers with NO preceding trigger (init burst / spurious)
    n_triggers_no_marker: int = 0      # triggers with 0 markers before the next (dropped sequence)
    n_triggers_multi_marker: int = 0   # triggers with >1 marker before the next (marker ringing)
    trig_marker_latency_ns: float = float("nan")   # median (marker - preceding trigger)

    def summary(self) -> str:
        lines = [
            f"[MultiHarp Rabi] start=ch{self.start_input} ({self.reference_label})  "
            f"stop=ch{self.stop_input} (SPCM)",
            f"  events: ref-edges={self.n_ref_events}  SPCM={self.n_spcm_events}  "
            f"shots used={self.n_shots_used}  tau steps={self.n_iterations}",
            f"  signal window: [{self.sig_offset_ns:.0f}, "
            f"{self.sig_offset_ns + self.sig_width_ns:.0f}] ns after the reference edge",
        ]
        if np.isfinite(self.ref_width_ns) and self.ref_width_ns > 0:
            lines.append(f"  reference window: [{self.ref_offset_ns:.0f}, "
                         f"{self.ref_offset_ns + self.ref_width_ns:.0f}] ns after the reference edge")
        if self.tau_assign_source == "shot_order":
            lines.append("  tau assignment: shot ORDER (shot i -> tau i mod n_steps)")
        else:
            how = ("MW pulse WIDTH (two-edge, geometry-free)"
                   if self.tau_assign_source == "mw_width"
                   else "MW falling edge vs the readout marker (fixed-gap geometry)")
            mw = ", ".join(f"ch{c}" for c in self.mw_inputs) or "?"
            lines.append(f"  tau assignment: {how}, MW on {mw}")
            lines.append(f"    matched {self.n_matched:,} shots to a tau step, "
                         f"{self.n_unmatched:,} unmatched (dropped, snap tol "
                         f"+/-{self.mw_snap_tol_ns:.0f} ns)")
            if np.isfinite(self.order_agreement):
                lines.append(f"    cross-check: MW tag agrees with shot order for "
                             f"{100.0 * self.order_agreement:.1f}% of matched shots")
        if np.isfinite(self.modulation_depth):
            lines.append(f"  modulation depth of {self.best_metric} "
                         f"(max-min)/(max+min): {self.modulation_depth:.3f}")
        if self.tau_assign_source == "shot_order" and not self.phase_ok:
            lines.append("  WARNING: shots seen != repeat_count x tau_steps -- the "
                         "shot->tau assignment may be desynced (a dropped reference "
                         "edge shifts every later shot). Treat the curve with care; "
                         "wire the MW line to a third input (mw_input=...) to remove "
                         "the ordering assumption entirely.")
        if self.trigger_input is not None:
            lines.append(f"  trigger cross-check (DO28 on ch{self.trigger_input}): "
                         f"{self.n_trigger_events:,} triggers vs {self.n_ref_events:,} "
                         f"markers (delta {self.n_ref_events - self.n_trigger_events:+,})")
            if (self.n_orphan_markers or self.n_triggers_no_marker
                    or self.n_triggers_multi_marker
                    or self.n_ref_events != self.n_trigger_events):
                lines.append(f"    NOT 1:1 -- {self.n_triggers_no_marker:,} dropped, "
                             f"{self.n_triggers_multi_marker:,} ringing, "
                             f"{self.n_orphan_markers:,} orphan markers")
            else:
                lines.append("    clean 1:1 (every trigger -> exactly one marker)")
        if self.fifo_overrun:
            lines.append("  WARNING: FIFO overrun occurred -- some events were lost.")
        if self.time_resolved and self.arrival_hist.size:
            nb = self.arrival_hist.shape[1]
            span = self.arrival_edges_ns[-1] if self.arrival_edges_ns.size else 0.0
            lines.append(f"  time-resolved: kept the full arrival profile "
                         f"({nb} bins over 0-{span:.0f} ns after the marker). Re-pick "
                         f"windows offline with recompute_rabi() -- no re-run needed.")
        if self.saved_files:
            lines.append("  saved: " + ", ".join(self.saved_files))
        return "\n".join(lines)


@dataclass
class T1Result:
    """Everything the T1 analysis produces (case="t1_vs_spcm").

    Registration rule: an SPCM photon is registered ONLY if it lands after a
    shot's READOUT-marker edge, inside the readout light pulse. Nothing between
    DO28 and the readout marker -- the init pulse, the dark time -- is counted.
    DO28 is used purely as a clock: it groups the marker edges into shots and
    times DO28 -> marker, and those durations give each shot its tau (with an
    init + readout marker: tau = (DO28->readout) - (DO28->init) - t1_gap). The
    shot's readout photons are then binned under that measured tau."""
    trig_input: int
    marker_input: int
    stop_input: int
    n_triggers: int                 # DO28 edges = shots the ADwin fired
    n_marker_edges: int
    n_spcm_events: int              # every SPCM event in the stream
    n_iterations: int
    repeat_count: Optional[int]
    base_resolution_ps: float

    # ---- marker pattern, and how cleanly it maps onto shots --------------- #
    marker_layout: str = ""                  # "init+readout" | "readout" | "init" | "none"
    edges_expected: int = 0                  # rising edges per shot for that layout
    n_shots_good: int = 0                    # shots with exactly edges_expected edges
    n_shots_no_marker: int = 0               # shots with no marker edge at all
    n_shots_bad_count: int = 0               # a missing or an extra (ringing) edge
    edges_per_shot_hist: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    n_orphan_markers: int = 0                # marker edges before the first DO28 edge
    n_stray_markers: int = 0                 # marker edges outside every shot window
    trigger_period_ns: float = float("nan")  # median DO28 -> DO28
    shot_window_ns: float = float("nan")     # a marker edge later than this after DO28 is stray
    n_runs: int = 0                          # sweeps, split where DO28 pauses
    shots_per_run: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    expected_shots_per_run: Optional[int] = None

    # ---- DO28 -> marker durations (DO28 is only a clock) ------------------- #
    ro_delay_median_ns: float = float("nan")  # DO28 -> readout marker, median over shots
    ro_jitter_ns: float = float("nan")        # its shot-to-shot std within one tau step
    ro_delay_span_ns: float = float("nan")    # 0.1..99.9 percentile span of that jitter
    ro_delay_hist_counts: np.ndarray = field(default_factory=lambda: np.empty(0))
    ro_delay_hist_edges_ns: np.ndarray = field(default_factory=lambda: np.empty(0))  # vs each tau's median
    readout_moves: bool = False               # True if DO28 -> readout changes with tau
    ro_delay_by_tau_ns: np.ndarray = field(default_factory=lambda: np.empty(0))    # median per tau
    init_delay_by_tau_ns: np.ndarray = field(default_factory=lambda: np.empty(0))  # median per tau
    tag_offset_ns: float = float("nan")       # constant removed to get tau (see tau_assign_source)
    tag_offset_auto_ns: float = float("nan")  # the same constant, estimated from the data
    tau_residual_std_ns: float = float("nan") # spread of (measured tau - its snapped step)

    # ---- tau assignment and the shot-order cross-checks -------------------- #
    # marker_gap           : tau = (DO28->readout) - (DO28->init) - t1_gap
    # marker_readout_delay : tau = (DO28->readout) - offset   (readout moves with tau)
    # marker_init_delay    : tau = offset - (DO28->init)      (init moves with tau)
    # shot_order_*         : tau from shot order (the marker can't carry it)
    tau_assign_source: str = ""
    n_matched: int = 0                        # shots that got a tau
    n_unmatched: int = 0
    snap_tol_ns: float = float("nan")
    agree_interleaved: float = float("nan")          # measured tau == shot-in-run mod n
    agree_interleaved_shifted: float = float("nan")  # ... allowing one task-pointer offset per run
    run_pointer_offsets: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    agree_blocked: float = float("nan")              # measured tau == (shot-in-run // repeat_count) mod n
    order_sample_measured: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    order_sample_interleaved: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    order_sample_blocked: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))

    # ---- registered SPCM photons: the readout light only ------------------- #
    n_spcm_registered: int = 0                # photons kept (inside ro_window after a readout marker)
    ro_window_ns: Tuple[float, float] = (float("nan"), float("nan"))  # ns after the readout marker
    ro_window_source: str = ""                # "given" | "auto (readout light)" | "fallback"
    light_on_ns: float = float("nan")         # readout light 50% rise, ns after the readout marker
    light_off_ns: float = float("nan")        # ... 50% fall
    light_on10_ns: float = float("nan")       # 10% points: the auto registration window
    light_off10_ns: float = float("nan")

    # ---- T1 curve (registered photons, binned by each shot's tau) ---------- #
    tau_ns: np.ndarray = field(default_factory=lambda: np.empty(0))
    tau_is_index: bool = False
    sig_window_ns: Tuple[float, float] = (float("nan"), float("nan"))
    ref_window_ns: Tuple[float, float] = (float("nan"), float("nan"))
    window_source: str = ""                   # "given" | "auto" | "fallback"
    signal_mean: np.ndarray = field(default_factory=lambda: np.empty(0))
    signal_sem: np.ndarray = field(default_factory=lambda: np.empty(0))
    reference_mean: np.ndarray = field(default_factory=lambda: np.empty(0))
    contrast: np.ndarray = field(default_factory=lambda: np.empty(0))
    shots_per_tau: np.ndarray = field(default_factory=lambda: np.empty(0))
    arrival_hist: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))   # [n_tau, n_bins], ro_window only
    arrival_edges_ns: np.ndarray = field(default_factory=lambda: np.empty(0))    # after the readout marker
    arrival_profile_ns: np.ndarray = field(default_factory=lambda: np.empty(0))  # all used shots, any tau

    # ---- the ADwin's programmed gates, replayed on the registered photons -- #
    adwin_sig_start_ns: float = float("nan")  # after DO28 (= readout_start_ns + gate_align_offset_ns)
    adwin_ref_start_ns: float = float("nan")
    adwin_count_ns: float = float("nan")
    adwin_readout_start_ns: float = float("nan")
    adwin_gate_align_offset_ns: float = float("nan")
    adwin_sig_rel_ns: float = float("nan")    # the same gates in readout-marker time
    adwin_ref_rel_ns: float = float("nan")
    adwin_replay_status: str = ""
    adwin_replay_sig: Dict[str, np.ndarray] = field(default_factory=dict)  # binning -> counts per tau
    adwin_replay_ref: Dict[str, np.ndarray] = field(default_factory=dict)  # "tau" = measured, else order
    adwin_signal: np.ndarray = field(default_factory=lambda: np.empty(0))  # ADwin totals per tau
    adwin_reference: np.ndarray = field(default_factory=lambda: np.empty(0))
    adwin_corr: Dict[str, float] = field(default_factory=dict)         # r(ADwin, replay) per binning
    adwin_count_ratio: Dict[str, float] = field(default_factory=dict)  # sum ADwin / sum replay
    light_on_after_trig_ns: float = float("nan")
    suggested_gate_align_offset_ns: float = float("nan")

    fifo_overrun: bool = False
    saved_files: List[str] = field(default_factory=list)

    def summary(self) -> str:
        def f(x, p=1):
            return "n/a" if x is None or not np.isfinite(x) else f"{x:,.{p}f}"
        L = [f"[MultiHarp T1] DO28 on ch{self.trig_input} = clock only | laser MKR on "
             f"ch{self.marker_input} ({self.marker_layout or '?'}) | SPCM on ch{self.stop_input}"]
        spr = ", ".join(f"{int(x):,}" for x in self.shots_per_run[:6])
        if self.shots_per_run.size > 6:
            spr += ", ..."
        exp = self.expected_shots_per_run
        L.append(f"  shots: {self.n_triggers:,} DO28 edges in {self.n_runs} run(s) [{spr}]"
                 + (f", expected {exp:,}/run" if exp else "")
                 + f"; trigger period {f(self.trigger_period_ns / 1000.0, 3)} us")
        if exp and self.shots_per_run.size and np.any(self.shots_per_run != exp):
            L.append("    WARNING: a run does not have repeat_count x tau steps shots -- cut short, "
                     "or an internal DO28 pause was split as a run (raise t1_run_gap_factor).")
        if self.edges_expected:
            nt = max(1, self.n_triggers)
            L.append(f"  marker -> shot: {self.n_shots_good:,} shots "
                     f"({100.0 * self.n_shots_good / nt:.2f}%) with exactly "
                     f"{self.edges_expected} edge(s), {self.n_shots_no_marker:,} with none, "
                     f"{self.n_shots_bad_count:,} with a missing/extra edge; "
                     f"{self.n_orphan_markers:,} orphan (before the first DO28), "
                     f"{self.n_stray_markers:,} stray (outside every shot)")
        else:
            L.append("  no marker edges inside any shot -- nothing to time, nothing registered")
        rb, ib = self.ro_delay_by_tau_ns, self.init_delay_by_tau_ns
        if np.isfinite(self.ro_delay_median_ns):
            if self.readout_moves and np.isfinite(rb).any():
                L.append(f"  DO28 -> readout marker MOVES with tau: {f(np.nanmin(rb) / 1000, 3)} .. "
                         f"{f(np.nanmax(rb) / 1000, 3)} us; jitter std {f(self.ro_jitter_ns, 2)} ns")
            else:
                L.append(f"  DO28 -> readout marker: {f(self.ro_delay_median_ns / 1000.0, 4)} us, "
                         f"fixed (jitter std {f(self.ro_jitter_ns, 2)} ns, span "
                         f"{f(self.ro_delay_span_ns, 1)} ns)")
        if ib.size and np.isfinite(ib).any():
            L.append(f"  DO28 -> init marker: {f(np.nanmin(ib) / 1000, 3)} .. "
                     f"{f(np.nanmax(ib) / 1000, 3)} us across the sweep")
        how = {"marker_gap": "(DO28->readout) - (DO28->init) - gap",
               "marker_readout_delay": "(DO28->readout) - offset",
               "marker_init_delay": "offset - (DO28->init)"}.get(self.tau_assign_source)
        if how:
            L.append(f"  tau READ from each shot = {how}: {self.n_matched:,} matched, "
                     f"{self.n_unmatched:,} not (snap +/-{f(self.snap_tol_ns, 0)} ns), "
                     f"residual std {f(self.tau_residual_std_ns, 2)} ns")
            L.append(f"    {'gap' if self.tau_assign_source == 'marker_gap' else 'offset'} used "
                     f"{f(self.tag_offset_ns)} ns; the data say {f(self.tag_offset_auto_ns)} ns")
            parts = []
            if np.isfinite(self.agree_interleaved):
                parts.append(f"interleaved {100.0 * self.agree_interleaved:.1f}%")
            if np.isfinite(self.agree_interleaved_shifted):
                parts.append(f"interleaved w/ per-run pointer offset "
                             f"{100.0 * self.agree_interleaved_shifted:.1f}%")
            if np.isfinite(self.agree_blocked):
                parts.append(f"blocked {100.0 * self.agree_blocked:.1f}%")
            if parts:
                L.append("  shot order vs measured tau: " + " | ".join(parts))
            bad = [(i + 1, int(o)) for i, o in enumerate(self.run_pointer_offsets) if o > 0]
            if bad:
                L.append("    task pointer did NOT start at tau step 0 in run(s): "
                         + ", ".join(f"run {r} -> +{o}" for r, o in bad[:10])
                         + "  (anything binned by shot order is shifted there)")
        elif self.tau_assign_source:
            L.append(f"  tau from shot ORDER ({self.tau_assign_source.replace('shot_order_', '')}) "
                     f"-- no DO28 -> marker duration changes with tau (add the init marker line)")
        if np.isfinite(self.ro_window_ns[0]):
            frac = (100.0 * self.n_spcm_registered / self.n_spcm_events) if self.n_spcm_events else 0.0
            L.append(f"  SPCM registered ONLY in [{f(self.ro_window_ns[0], 0)}, "
                     f"{f(self.ro_window_ns[1], 0)}) ns after each readout marker "
                     f"({self.ro_window_source}): {self.n_spcm_registered:,} of "
                     f"{self.n_spcm_events:,} SPCM events ({frac:.2f}%). Nothing between DO28 "
                     f"and the readout marker is counted.")
        if np.isfinite(self.light_on_ns):
            extra = (f" = {f(self.light_on_after_trig_ns / 1000.0, 3)} us after DO28"
                     if np.isfinite(self.light_on_after_trig_ns) else "")
            L.append(f"  readout light on {f(self.light_on_ns, 0)} ns, off {f(self.light_off_ns, 0)} ns "
                     f"(50%) after the readout marker{extra}")
        if self.signal_mean.size and np.isfinite(self.signal_mean).any():
            s0, s1 = self.sig_window_ns
            msg = f"  MultiHarp windows ({self.window_source}): signal [{f(s0, 0)}, {f(s1, 0)}] ns"
            if np.isfinite(self.ref_window_ns[0]):
                msg += f", reference [{f(self.ref_window_ns[0], 0)}, {f(self.ref_window_ns[1], 0)}] ns"
            L.append(msg + " after the readout marker")
        if np.isfinite(self.adwin_sig_start_ns):
            L.append(f"  ADwin gates as programmed: signal {f(self.adwin_sig_start_ns, 0)} ns, "
                     f"reference {f(self.adwin_ref_start_ns, 0)} ns after its trigger, "
                     f"{f(self.adwin_count_ns, 0)} ns wide"
                     + (f" = {f(self.adwin_sig_rel_ns, 0)} / {f(self.adwin_ref_rel_ns, 0)} ns after "
                        f"the readout marker" if np.isfinite(self.adwin_sig_rel_ns) else ""))
            if self.adwin_replay_status:
                L.append(f"    {self.adwin_replay_status}")
            if self.adwin_corr:
                parts = [f"{k}: r={f(v, 3)}, counts x{f(self.adwin_count_ratio.get(k), 2)}"
                         for k, v in self.adwin_corr.items()]
                L.append("    ADwin counts vs its gates replayed on readout photons -> "
                         + " | ".join(parts))
        if np.isfinite(self.suggested_gate_align_offset_ns):
            L.append(f"  gate_align_offset_ns that puts the programmed signal gate on the light "
                     f"onset: ~{f(self.suggested_gate_align_offset_ns, 0)} "
                     f"(now {f(self.adwin_gate_align_offset_ns, 0)}; valid if the ADwin opens its "
                     f"gate when programmed -- check with case='adwin_readout_vs_spcm')")
        if self.fifo_overrun:
            L.append("  WARNING: FIFO overrun occurred -- some events were lost.")
        if self.saved_files:
            L.append("  saved: " + ", ".join(self.saved_files))
        return "\n".join(L)

def rabi_curve_from_histogram(arrival_hist, arrival_edges_ns, shots_per_tau,
                              sig_window_ns, ref_window_ns=None):
    """Re-derive a Rabi curve from a stored time-resolved arrival histogram, for
    ANY signal/reference window -- the whole point of keeping the 7 us data.

    Nothing has to be re-run: pick the windows you want (e.g. start the signal
    window past the laser rise) and this sums the histogram bins over them.

    Parameters
    ----------
    arrival_hist     : [n_tau, n_bins] int array of SPCM counts vs (tau, arrival
                       time relative to the marker). RabiResult.arrival_hist, or
                       the "arrival_hist" entry of the saved .npz.
    arrival_edges_ns : n_bins+1 bin edges in ns after the marker (arrival_edges_ns).
    shots_per_tau    : shots that went into each tau row (for the per-shot mean).
    sig_window_ns    : (lo, hi) ns after the marker for the SIGNAL window.
    ref_window_ns    : optional (lo, hi) for a REFERENCE window -> contrast.

    Returns a dict: tau_index, signal_sum, signal_mean (per shot), and, if a
    reference window is given, reference_sum, reference_mean, contrast. Windows
    snap to the nearest bin edge; the effective windows actually used are echoed
    back as sig_window_used_ns / ref_window_used_ns.
    """
    H = np.asarray(arrival_hist)
    edges = np.asarray(arrival_edges_ns, dtype=float)
    spt = np.asarray(shots_per_tau, dtype=float)
    if H.ndim != 2 or edges.size != H.shape[1] + 1:
        raise ValueError("arrival_hist must be [n_tau, n_bins] with "
                         "len(arrival_edges_ns) == n_bins + 1")

    def _sum(win):
        lo, hi = float(win[0]), float(win[1])
        i0 = int(np.searchsorted(edges, lo, side="left"))
        i1 = int(np.searchsorted(edges, hi, side="left"))
        i0 = max(0, min(i0, H.shape[1]))
        i1 = max(i0, min(i1, H.shape[1]))
        return H[:, i0:i1].sum(axis=1).astype(float), (edges[i0], edges[min(i1, edges.size - 1)])

    sig_sum, sig_used = _sum(sig_window_ns)
    with np.errstate(divide="ignore", invalid="ignore"):
        sig_mean = np.where(spt > 0, sig_sum / spt, np.nan)
    out = {"tau_index": np.arange(H.shape[0]),
           "signal_sum": sig_sum, "signal_mean": sig_mean,
           "sig_window_used_ns": sig_used}
    if ref_window_ns is not None:
        ref_sum, ref_used = _sum(ref_window_ns)
        with np.errstate(divide="ignore", invalid="ignore"):
            ref_mean = np.where(spt > 0, ref_sum / spt, np.nan)
            contrast = np.where(ref_mean > 0, sig_mean / ref_mean, np.nan)
        out.update(reference_sum=ref_sum, reference_mean=ref_mean,
                   contrast=contrast, ref_window_used_ns=ref_used)
    return out


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
        # Rabi reconstructed BY THE MULTIHARP (no ADwin gating at all). start is a
        # once-per-shot reference edge -- the readout laser marker is best because
        # it marks the readout directly and (per the init_vs_trigger run) is rock
        # solid; the trigger works too. stop is the SPCM. See analyze_rabi / the
        # RabiResult docstring for how the tau curve is built and its one caveat.
        "rabi_vs_spcm": dict(
            corr_mode="around",          # not used for the curve, but keeps ranges sane
            window_ns=4000.0,
            pre_window_ns=500.0,
            start_label="Proteus ch4 MKR readout (per-shot reference)",
            stop_label="SPCM",
        ),
        # Same as rabi_vs_spcm, but tau is read PER SHOT from a third input wired to
        # the MW line (its pulse width == tau) instead of from shot order. This is
        # the desync-proof version: a dropped edge can no longer smear the curve,
        # because every well-formed shot self-identifies its tau. start is still the
        # readout MARKER (per-shot reference + SPCM signal window), stop is the SPCM,
        # and mw_input (+ optional mw_fall_input) carry the MW copy. See analyze_rabi.
        "rabi_mw_tagged": dict(
            corr_mode="around",
            window_ns=4000.0,
            pre_window_ns=500.0,
            start_label="Proteus ch4 MKR readout (per-shot reference)",
            stop_label="SPCM",
        ),
        # T1: trig_input = DO28 (required, clock only), start = ch4 MKR copying
        # the laser (init + readout marker lines), stop = SPCM. DO28 -> marker
        # durations give tau; SPCM is registered only after the readout marker.
        "t1_vs_spcm": dict(
            corr_mode="around",
            window_ns=8000.0,
            pre_window_ns=500.0,
            start_label="Proteus ch4 MKR (laser copy: init + readout)",
            stop_label="SPCM",
        ),
    }
    # Cases that reconstruct a Rabi curve from MultiHarp tags (no ADwin gating).
    _RABI_CASES = ("rabi_vs_spcm", "rabi_mw_tagged")
    _T1_LAYOUTS = ("auto", "init+readout", "readout", "init", "none")

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
        # ---- Rabi-case parameters (case="rabi_vs_spcm") --------------------- #
        # All in ns, relative to the once-per-shot reference edge on start_input.
        # The signal window should sit on the BRIGHT onset of the readout (first
        # few hundred ns, where NV spin contrast is largest). The reference window
        # (optional) should sit LATER but still INSIDE the readout laser pulse, so
        # dividing signal/reference cancels laser-power and photon-collection drift.
        rabi_sig_offset_ns: float = 0.0,
        rabi_sig_width_ns: float = 300.0,
        rabi_ref_offset_ns: float = 5000.0,
        rabi_ref_width_ns: float = 300.0,
        rabi_n_iterations: Optional[int] = None,     # tau steps; auto-detected if None
        rabi_order: str = "interleaved",             # one tau per trigger (your scope); "blocked" only
                                                     # for a task table that loops each segment
        rabi_tau_values_ns: Optional[np.ndarray] = None,  # x-axis; auto-detected if None
        # ---- MW-tagged tau assignment (case="rabi_mw_tagged") -------------- #
        # Wire an attenuated copy of the MW line (the IQ/Proteus channel that
        # carries the variable-width MW pulse) to a THIRD MultiHarp input. Its
        # pulse WIDTH is tau, so each shot's tau is read from the data instead of
        # from shot order -- a dropped edge then drops only that one shot rather
        # than shifting every later one. Two ways to read it:
        #   * WIDTH mode (recommended, geometry-free): give BOTH mw_input and
        #     mw_fall_input. mw_input is forced to the RISING edge (MW start),
        #     mw_fall_input to the FALLING edge (MW end); tau = fall - rise.
        #   * GEOMETRY mode (one BNC): give only mw_input (forced FALLING = MW
        #     end). tau = mw_gap_ns - (marker_time - mw_falling_time), where
        #     mw_gap_ns is the FIXED waveform separation between the MW pulse
        #     START and the readout-marker START (e.g. readout@5900 - MW@2900 =
        #     3000). Exact for square MW pulses (the usual Rabi drive).
        mw_input: Optional[int] = None,              # 3rd input: MW copy (None = tag off)
        mw_cfg: Optional[ChannelConfig] = None,      # its trigger level/edge
        mw_fall_input: Optional[int] = None,         # optional 4th input -> WIDTH mode
        mw_fall_cfg: Optional[ChannelConfig] = None,
        mw_gap_ns: Optional[float] = None,           # readout-start - MW-start (GEOMETRY mode)
        mw_snap_tol_ns: Optional[float] = None,      # max |measured tau - step| to accept
        mw_lookback_ns: Optional[float] = None,      # how far before the marker to hunt MW edges
        # ---- optional ADwin-trigger cross-check --------------------------- #
        # Wire an attenuated copy of ADwin DIGOUT 28 (== DIGOUT 21, the Proteus
        # trigger line) to its OWN free MultiHarp input. Its edge count is then
        # compared to the readout-marker count so you can SEE, in one run, whether
        # markers == triggers. It must be a different connector from start/stop/MW.
        trig_input: Optional[int] = None,            # input carrying the DO28 copy (None = off)
        trig_cfg: Optional[ChannelConfig] = None,    # its trigger level/edge
        # ---- time-resolved arrivals (keep the whole readout, re-window later) #
        # When on, analyze_rabi also builds a per-tau histogram of SPCM arrival
        # times relative to the marker over [0, rabi_tr_window_ns] in
        # rabi_tr_bin_ns bins. Cheap (one extra pass) and it lets you move the
        # signal/reference windows offline -- e.g. to skip the laser rise -- with
        # recompute_rabi(), no re-run needed. Set rabi_time_resolved=False to skip.
        rabi_time_resolved: bool = True,
        rabi_tr_window_ns: float = 7500.0,           # span after the marker to keep (~readout length)
        rabi_tr_bin_ns: float = 10.0,                # arrival-time bin width
        # ---- memory bounds for very large runs ---------------------------- #
        # Long acquisitions can produce billions of T2 records; decoding them all
        # at once would need tens of GB. The Rabi path instead decodes in chunks
        # of this many records and STREAMS the SPCM channel, so peak memory stays
        # small. max_marker_edges guards the one array that must be held whole
        # (the reference/marker edges): if the marker multi-triggers into more
        # than this, analysis stops with a clear message instead of thrashing.
        decode_chunk_records: int = 50_000_000,
        max_marker_edges: int = 1_000_000_000,
        # ---- crash-safety / memory: stream raw T2 records to disk ---------- #
        # When True (default) the FIFO is written to a .bin on disk as it is read
        # and stop_acquisition returns a MEMORY-MAP over it, so the raw stream is
        # never concatenated into a multi-GB RAM array and survives a crash. Set
        # False to keep the original in-RAM behaviour (fine for short timing runs).
        stream_raw_to_disk: bool = True,
        raw_dir: Optional[str] = None,
        # ---- T1 (case="t1_vs_spcm") ----------------------------------------- #
        # trig_input (DO28) is REQUIRED but only a clock (shots + DO28 -> marker
        # timing). start = ch4 MKR copying the laser, stop = SPCM. SPCM photons
        # are registered ONLY after each readout marker. All values in ns.
        t1_marker_layout: str = "auto",              # auto | init+readout | readout | init | none
        t1_gap_ns: Optional[float] = None,           # readout-marker start - init-marker start at tau=0
        t1_tau_values_ns: Optional[np.ndarray] = None,  # one per sweep step, IN SWEEP ORDER
        t1_n_iterations: Optional[int] = None,
        t1_snap_tol_ns: Optional[float] = None,      # default: half a tau step
        t1_order: str = "interleaved",               # shot-order model, used only if tau can't be read
        t1_sig_window_ns: Optional[Tuple[float, float]] = None,  # after the readout marker; None = auto
        t1_ref_window_ns: Optional[Tuple[float, float]] = None,  # None = auto (late in the readout)
        t1_count_ns: float = 300.0,                  # width of the auto windows
        t1_ref_margin_ns: float = 300.0,             # auto reference ends this far before light-off
        t1_ro_window_ns: Optional[Tuple[float, float]] = None,   # registered SPCM span after the
                                                     # readout marker; None = auto (the readout light)
        t1_tr_window_ns: float = 8000.0,             # how far after the readout marker to look for it
        t1_tr_bin_ns: float = 10.0,
        t1_shot_window_ns: Optional[float] = None,   # a marker edge later than this after DO28 is stray
        t1_run_gap_factor: float = 3.0,              # DO28 pause > this x period starts a new run
        t1_adwin_sig_start_ns: Optional[float] = None,  # override the ADwin gates read from the experiment
        t1_adwin_ref_start_ns: Optional[float] = None,
        t1_adwin_count_ns: Optional[float] = None,
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

        # Rabi-case configuration (only consulted when case == "rabi_vs_spcm").
        self.rabi_sig_offset_ns = float(rabi_sig_offset_ns)
        self.rabi_sig_width_ns = float(rabi_sig_width_ns)
        self.rabi_ref_offset_ns = float(rabi_ref_offset_ns)
        self.rabi_ref_width_ns = float(rabi_ref_width_ns)
        self.rabi_n_iterations = (int(rabi_n_iterations)
                                  if rabi_n_iterations is not None else None)
        if rabi_order not in ("interleaved", "blocked"):
            raise ValueError(f"rabi_order must be 'interleaved' or 'blocked', got {rabi_order!r}")
        self.rabi_order = rabi_order
        self.rabi_tau_values_ns = (np.asarray(rabi_tau_values_ns, dtype=float)
                                   if rabi_tau_values_ns is not None else None)

        # ---- MW-tagging configuration ------------------------------------- #
        self.mw_input = int(mw_input) if mw_input is not None else None
        self.mw_fall_input = int(mw_fall_input) if mw_fall_input is not None else None
        self.mw_gap_ns = float(mw_gap_ns) if mw_gap_ns is not None else None
        self.mw_snap_tol_ns = (float(mw_snap_tol_ns)
                               if mw_snap_tol_ns is not None else None)
        self.mw_lookback_ns = (float(mw_lookback_ns)
                               if mw_lookback_ns is not None else None)
        # Decide the tagging mode and pin the correct edges (foolproof):
        #   WIDTH mode needs both edges -> mw_input RISING, mw_fall_input FALLING.
        #   GEOMETRY mode uses the single MW END -> mw_input FALLING.
        self.mw_cfg = mw_cfg or ChannelConfig(level_mV=200, edge=EDGE_FALLING)
        self.mw_fall_cfg = mw_fall_cfg or ChannelConfig(level_mV=200, edge=EDGE_FALLING)
        if self.mw_input is not None and self.mw_fall_input is not None:
            self.mw_mode = "width"
            if self.mw_cfg.edge != EDGE_RISING:
                self.mw_cfg = ChannelConfig(level_mV=self.mw_cfg.level_mV,
                                            edge=EDGE_RISING,
                                            offset_ps=self.mw_cfg.offset_ps)
            if self.mw_fall_cfg.edge != EDGE_FALLING:
                self.mw_fall_cfg = ChannelConfig(level_mV=self.mw_fall_cfg.level_mV,
                                                 edge=EDGE_FALLING,
                                                 offset_ps=self.mw_fall_cfg.offset_ps)
        elif self.mw_input is not None:
            self.mw_mode = "geometry"
            if self.mw_cfg.edge != EDGE_FALLING:
                print("[MultiHarp] Rabi MW-tag GEOMETRY mode needs the MW FALLING "
                      "edge (the MW end); overriding mw_cfg edge to falling.")
                self.mw_cfg = ChannelConfig(level_mV=self.mw_cfg.level_mV,
                                            edge=EDGE_FALLING,
                                            offset_ps=self.mw_cfg.offset_ps)
        else:
            self.mw_mode = "off"

        # ---- ADwin-trigger cross-check configuration ---------------------- #
        self.trig_input = int(trig_input) if trig_input is not None else None
        # DIGOUT goes HIGH to trigger -> RISING edge; level ~half the attenuated
        # DO28 amplitude (measured 610 mV with 10+6 dB -> ~300 mV).
        self.trig_cfg = trig_cfg or ChannelConfig(level_mV=300, edge=EDGE_RISING)

        # ---- time-resolved arrival settings ------------------------------- #
        self.rabi_time_resolved = bool(rabi_time_resolved)
        self.rabi_tr_window_ns = float(rabi_tr_window_ns)
        self.rabi_tr_bin_ns = float(rabi_tr_bin_ns)
        self.decode_chunk_records = int(decode_chunk_records)
        self.max_marker_edges = int(max_marker_edges)
        self.stream_raw_to_disk = bool(stream_raw_to_disk)
        self.raw_dir = raw_dir
        self._raw_path = None          # path of the on-disk raw .bin (set at start)
        self._raw_fh = None            # open write handle during acquisition
        self._raw_count = 0            # number of uint32 records written so far

        # ---- T1 configuration ----------------------------------------------- #
        if t1_marker_layout not in self._T1_LAYOUTS:
            raise ValueError(f"t1_marker_layout must be one of {self._T1_LAYOUTS}, "
                             f"got {t1_marker_layout!r}")
        if t1_order not in ("interleaved", "blocked"):
            raise ValueError(f"t1_order must be 'interleaved' or 'blocked', got {t1_order!r}")
        self.t1_marker_layout = t1_marker_layout
        self.t1_gap_ns = float(t1_gap_ns) if t1_gap_ns is not None else None
        self.t1_tau_values_ns = (np.asarray(t1_tau_values_ns, dtype=float).ravel()
                                 if t1_tau_values_ns is not None else None)
        self.t1_n_iterations = int(t1_n_iterations) if t1_n_iterations is not None else None
        self.t1_snap_tol_ns = float(t1_snap_tol_ns) if t1_snap_tol_ns is not None else None
        self.t1_order = t1_order
        self.t1_sig_window_ns = (tuple(float(x) for x in t1_sig_window_ns)
                                 if t1_sig_window_ns is not None else None)
        self.t1_ref_window_ns = (tuple(float(x) for x in t1_ref_window_ns)
                                 if t1_ref_window_ns is not None else None)
        self.t1_count_ns = float(t1_count_ns)
        self.t1_ref_margin_ns = float(t1_ref_margin_ns)
        self.t1_tr_window_ns = float(t1_tr_window_ns)
        self.t1_tr_bin_ns = float(t1_tr_bin_ns)
        self.t1_shot_window_ns = float(t1_shot_window_ns) if t1_shot_window_ns is not None else None
        self.t1_ro_window_ns = (tuple(float(x) for x in t1_ro_window_ns)
                                if t1_ro_window_ns is not None else None)
        self.t1_run_gap_factor = float(t1_run_gap_factor)
        self.t1_adwin_sig_start_ns = t1_adwin_sig_start_ns
        self.t1_adwin_ref_start_ns = t1_adwin_ref_start_ns
        self.t1_adwin_count_ns = t1_adwin_count_ns

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
        if self.case == "t1_vs_spcm" and getattr(self, "trig_input", None) is None:
            raise ValueError("case='t1_vs_spcm' needs the ADwin DO28 copy on its own input "
                             "(trig_input=<ch>): it is the clock that groups the marker edges "
                             "into shots and times DO28 -> marker for tau.")
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

        # MW-tagging inputs (0, 1 or 2 extra channels) join the active set.
        mw_chans: List[int] = []
        if getattr(self, "mw_mode", "off") != "off":
            mw_chans.append(self.mw_input)
            if self.mw_fall_input is not None:
                mw_chans.append(self.mw_fall_input)
        active = [self.start_input, self.stop_input, *mw_chans]
        if getattr(self, "trig_input", None) is not None:
            active.append(self.trig_input)

        for ch in active:
            if not (0 <= ch < nchan):
                mh.close()
                raise ValueError(f"input channel {ch} out of range (device has {nchan})")
        if len(set(active)) != len(active):
            mh.close()
            raise ValueError(f"start/stop/MW/trigger inputs must all be different "
                             f"channels (got start={self.start_input}, "
                             f"stop={self.stop_input}, mw={mw_chans}, "
                             f"trig={self.trig_input}). DIGOUT 28 needs its OWN free "
                             f"connector -- it cannot share the marker/SPCM/MW input.")

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
                mh.set_input_enable(ch, ch in active)
            mh.set_input_trigger(self.start_input, self.start_cfg.level_mV, self.start_cfg.edge)
            mh.set_input_offset(self.start_input, self.start_cfg.offset_ps)
            mh.set_input_trigger(self.stop_input, self.stop_cfg.level_mV, self.stop_cfg.edge)
            mh.set_input_offset(self.stop_input, self.stop_cfg.offset_ps)

        # MW-tag inputs always get their OWN level/edge (even under scan mode), so
        # the width/geometry read is never corrupted by a shared scan level.
        if mw_chans:
            mh.set_input_enable(self.mw_input, True)
            mh.set_input_trigger(self.mw_input, self.mw_cfg.level_mV, self.mw_cfg.edge)
            mh.set_input_offset(self.mw_input, self.mw_cfg.offset_ps)
            if self.mw_fall_input is not None:
                mh.set_input_enable(self.mw_fall_input, True)
                mh.set_input_trigger(self.mw_fall_input, self.mw_fall_cfg.level_mV,
                                     self.mw_fall_cfg.edge)
                mh.set_input_offset(self.mw_fall_input, self.mw_fall_cfg.offset_ps)
            ed = lambda e: "rising" if e == EDGE_RISING else "falling"
            if self.mw_mode == "width":
                print(f"[MultiHarp] Rabi MW-tag WIDTH mode: rise=ch{self.mw_input} "
                      f"({ed(self.mw_cfg.edge)}, {self.mw_cfg.level_mV} mV), "
                      f"fall=ch{self.mw_fall_input} ({ed(self.mw_fall_cfg.edge)}, "
                      f"{self.mw_fall_cfg.level_mV} mV) -> tau = fall - rise.")
            else:
                print(f"[MultiHarp] Rabi MW-tag GEOMETRY mode: MW end on ch{self.mw_input} "
                      f"({ed(self.mw_cfg.edge)}, {self.mw_cfg.level_mV} mV); tau derived "
                      f"vs the readout marker with mw_gap_ns.")

        # ADwin-trigger cross-check input: enable with its OWN level/edge (also
        # overriding scan mode, so its count is never corrupted by a shared level).
        if getattr(self, "trig_input", None) is not None:
            mh.set_input_enable(self.trig_input, True)
            mh.set_input_trigger(self.trig_input, self.trig_cfg.level_mV, self.trig_cfg.edge)
            mh.set_input_offset(self.trig_input, self.trig_cfg.offset_ps)
            ted = "rising" if self.trig_cfg.edge == EDGE_RISING else "falling"
            if self.case == "t1_vs_spcm":
                print(f"[MultiHarp] T1: DIGOUT 28 copy on ch{self.trig_input} ({ted}, "
                      f"{self.trig_cfg.level_mV} mV) = CLOCK ONLY (shots + DO28 -> marker "
                      f"timing for tau); laser marker on ch{self.start_input}, SPCM on "
                      f"ch{self.stop_input}. SPCM is registered only after each readout marker.")
            else:
                print(f"[MultiHarp] ADwin-trigger cross-check: DIGOUT 28 copy on "
                      f"ch{self.trig_input} ({ted}, {self.trig_cfg.level_mV} mV) -- its edge "
                      f"count will be compared to the readout-marker count.")

        time.sleep(0.2)  # let the inputs settle after retriggering
        self._mh = mh

    # ---- acquisition (background FIFO reader) ----------------------------- #
    def _store_chunk(self, arr):
        """Persist one FIFO chunk. Streaming mode appends it to the on-disk .bin
        so RAM stays chunk-sized (no multi-GB concatenate and it survives a
        crash); otherwise keep it in the RAM list (original short-run behaviour)."""
        if self.stream_raw_to_disk and self._raw_fh is not None:
            arr.tofile(self._raw_fh)
            self._raw_count += int(arr.size)
        else:
            self._chunks.append(arr)

    def _read_loop(self):
        buffer = (ct.c_uint * TTREADMAX)()
        while not self._stop_flag.is_set():
            n = self._mh.read_fifo(buffer)
            if n > 0:
                self._store_chunk(np.frombuffer(buffer, dtype=np.uint32,
                                                count=n).copy())
            else:
                if self._mh.flags() & FLAG_FIFOFULL:
                    self._fifo_overrun = True
                time.sleep(0.001)

    def start_acquisition(self, tacq_ms: int = 360_000_000):   # 100 h (was 360_000_00 = 10 h)
        """Start a long T2 acquisition and spin up the reader thread.

        tacq_ms just needs to outlast the ODMR run; we stop manually when the
        run returns. Default is 100 h (the MHLib maximum in v3.1)."""
        if self._mh is None:
            self.open_and_configure()
        self._chunks = []
        self._raw_count = 0
        self._raw_fh = None
        self._raw_path = None
        if self.stream_raw_to_disk:
            raw_dir = self.raw_dir or self.output_dir or os.getcwd()
            os.makedirs(raw_dir, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            self._raw_path = os.path.join(raw_dir, f"multiharp_raw_{stamp}.bin")
            self._raw_fh = open(self._raw_path, "wb")
            print(f"[MultiHarp] streaming raw T2 records to disk: {self._raw_path} "
                  f"(RAM stays chunk-sized; survives a crash).")
        self._fifo_overrun = False
        self._stop_flag.clear()
        self._mh.start(tacq_ms)
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        print("[MultiHarp] acquisition started (listening while ODMR runs)")

    def stop_acquisition(self) -> np.ndarray:
        """Stop the reader, stop the measurement, drain the FIFO, return records.

        In streaming mode the records live in the on-disk .bin and this returns a
        read-only np.memmap over it (not held in RAM); otherwise it returns the
        concatenated in-RAM array as before."""
        self._stop_flag.set()
        if self._reader is not None:
            self._reader.join(timeout=10.0)
        self._mh.stop()
        # Drain whatever is still in the FIFO after StopMeas (reader has stopped,
        # so this is the only writer now -- safe to reuse _store_chunk).
        buffer = (ct.c_uint * TTREADMAX)()
        while True:
            n = self._mh.read_fifo(buffer)
            if n <= 0:
                break
            self._store_chunk(np.frombuffer(buffer, dtype=np.uint32, count=n).copy())
        if self._mh.flags() & FLAG_FIFOFULL:
            self._fifo_overrun = True
        if self.stream_raw_to_disk and self._raw_fh is not None:
            self._raw_fh.flush()
            self._raw_fh.close()
            self._raw_fh = None
            count = self._raw_count
            records = (np.memmap(self._raw_path, dtype=np.uint32, mode="r")
                       if count > 0 else np.empty(0, dtype=np.uint32))
            print(f"[MultiHarp] acquisition stopped, {count:,} raw records captured "
                  f"(on disk: {self._raw_path}; memory-mapped, not held in RAM).")
            return records
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

    # ---- memory-bounded T2 decoding for very large runs ------------------- #
    # decode_t2 / decode_all_channels build several full-length arrays at once
    # (a uint64 cumsum + a float64 time array), which is ~16 bytes/record -- fine
    # for short timing runs but fatal for a 10 h Rabi (billions of records -> tens
    # of GB, MemoryError). The three helpers below decode in fixed-size CHUNKS,
    # carrying the overflow offset across chunk boundaries so the absolute times
    # are identical to the one-shot decode, while peak memory stays ~chunk-sized.
    def _channel_counts(self, records: np.ndarray) -> Dict[int, int]:
        """Count events per channel by streaming (no time arrays). Cheap; used to
        report the channel composition and catch a multi-triggering line early."""
        counts = np.zeros(64, dtype=np.int64)
        chunk = max(1, self.decode_chunk_records)
        for s in range(0, records.size, chunk):
            r = records[s:s + chunk]
            special = (r >> 31) & 0x1
            chan = (r >> 25) & 0x3F
            counts += np.bincount(chan[special == 0], minlength=64).astype(np.int64)
        return {int(ch): int(counts[ch]) for ch in range(64) if counts[ch]}

    def _iter_channel_times(self, records: np.ndarray, channels):
        """Yield {channel: times_ps} for the requested channels, one CHUNK at a
        time, carrying the T2 overflow offset across chunks."""
        channels = [int(c) for c in channels if c is not None]
        base = np.uint64(0)                       # overflow offset so far, base-res units
        chunk = max(1, self.decode_chunk_records)
        for s in range(0, records.size, chunk):
            r = records[s:s + chunk]
            special = (r >> 31) & 0x1
            chan = (r >> 25) & 0x3F
            timetag = (r & 0x01FFFFFF).astype(np.uint64)
            is_ovf = (special == 1) & (chan == 0x3F)
            ovf = np.where(is_ovf, np.where(timetag == 0, np.uint64(1), timetag),
                           np.uint64(0))
            within = np.cumsum(ovf) * np.uint64(T2WRAP)     # cumulative overflow, this chunk
            units = base + within + timetag                 # absolute base-res units
            is_event = special == 0
            out = {}
            for c in channels:
                m = is_event & (chan == c)
                out[c] = units[m].astype(np.float64) * self._base_res_ps
            if within.size:
                base = base + within[-1]           # carry this chunk's total overflow
            yield out

    def _collect_channel_times(self, records: np.ndarray, channels) -> Dict[int, np.ndarray]:
        """Fully materialise times for a few SPARSE channels (markers, MW edges),
        chunk by chunk. Safe only for channels with a bounded number of events."""
        channels = [int(c) for c in channels if c is not None]
        acc = {c: [] for c in channels}
        for chunk_out in self._iter_channel_times(records, channels):
            for c in channels:
                if chunk_out[c].size:
                    acc[c].append(chunk_out[c])
        return {c: (np.concatenate(acc[c]) if acc[c] else np.empty(0)) for c in channels}

    def _report_channel_counts(self, counts: Dict[int, int],
                               expected_markers: Optional[int] = None,
                               edges_per_shot: float = 1.0) -> None:
        if not counts:
            print("[MultiHarp] DIAGNOSTIC: 0 events on ALL channels. Nothing is "
                  "triggering. Check the signals are present, the level sits ~half "
                  "the ATTENUATED amplitude, and the edge matches.")
            return
        t1 = self.case == "t1_vs_spcm"
        print("[MultiHarp] per-channel event counts (0-based input; connector = index+1):")
        for ch in sorted(counts):
            tags = []
            if ch == self.start_input:
                tags.append("<-laser MKR (ch4)" if t1 else "<-START/reference")
            if ch == self.stop_input: tags.append("<-STOP/SPCM")
            if ch == self.mw_input: tags.append("<-MW")
            if ch == self.mw_fall_input: tags.append("<-MW fall")
            if ch == getattr(self, "trig_input", None):
                tags.append("<-DO28 (clock only)" if t1 else "<-ADwin trigger (DO28)")
            print(f"    input {ch}: {counts[ch]:>14,} events  {' '.join(tags)}")

        # count (one DO28 pulse per shot). repeat_count*tau_steps ignores the number
        # of averages, so a healthy N-average run wrongly looks like Nx multi-trigger.
        # edges_per_shot = 2 for a T1 marker that copies both laser pulses.
        n_start = counts.get(self.start_input, 0)
        n_trig = counts.get(getattr(self, "trig_input", None), 0)
        ref_shots = n_trig if n_trig > 0 else (expected_markers or 0)
        if ref_shots and n_start > (edges_per_shot + 0.5) * ref_shots:
            print(f"[MultiHarp] WARNING: the reference/START channel "
                  f"(input {self.start_input}) has {n_start:,} edges vs {ref_shots:,} "
                  f"shots actually triggered -- ~{n_start / max(1, ref_shots):.1f}x"
                  + (f" (expected {edges_per_shot:g} per shot)" if edges_per_shot != 1 else "")
                  + ". The marker is MULTI-TRIGGERING (ringing/reflection or the level set "
                  "too low). Raise its trigger level / fix termination so it fires ONCE per "
                  "laser edge.")
        elif ref_shots and edges_per_shot == 1:
            print(f"[MultiHarp] START/marker vs ADwin triggers: {n_start:,} vs "
                  f"{ref_shots:,} ({n_start / max(1, ref_shots):.3f}x) -- OK, ~1 per shot.")
        elif ref_shots:
            print(f"[MultiHarp] laser marker vs DO28: {n_start:,} edges vs {ref_shots:,} "
                  f"shots ({n_start / max(1, ref_shots):.3f} per shot; 2 = init+readout, "
                  f"1 = one marker line).")

    def _stream_spcm_windows(self, records: np.ndarray, ref_ps: np.ndarray,
                             tau_index: np.ndarray, matched: np.ndarray,
                             n_iterations: int, sig_lo_ps: float, sig_hi_ps: float,
                             use_ref: bool, ref_lo_ps: float, ref_hi_ps: float):
        """Stream the SPCM channel in chunks and accumulate, per tau step:
        signal-window counts, reference-window counts, and (if enabled) the full
        arrival-time histogram -- all WITHOUT ever holding every SPCM time. Each
        photon is attached to the marker at/just-before it (markers are complete
        and sorted), and only photons from matched shots are used."""
        sig_total = np.zeros(n_iterations, dtype=np.int64)
        ref_total = np.zeros(n_iterations, dtype=np.int64)
        do_hist = self.rabi_time_resolved and self.rabi_tr_window_ns > 0
        nbins = max(1, int(round(self.rabi_tr_window_ns / self.rabi_tr_bin_ns))) if do_hist else 0
        edges_ns = np.linspace(0.0, self.rabi_tr_window_ns, nbins + 1) if do_hist else np.empty(0)
        H_flat = np.zeros(n_iterations * nbins, dtype=np.int64) if do_hist else None
        # tau-INDEPENDENT readout profile over ALL shots (matched or not), so the
        # readout is visible for diagnosis even when tau assignment fails.
        prof = np.zeros(nbins, dtype=np.int64) if do_hist else None
        win_ps = self.rabi_tr_window_ns * 1000.0
        bin_ps = self.rabi_tr_bin_ns * 1000.0
        n_spcm = 0
        if ref_ps.size:
            for chunk_out in self._iter_channel_times(records, [self.stop_input]):
                t = chunk_out[self.stop_input]
                n_spcm += t.size
                if t.size == 0:
                    continue
                idx = np.searchsorted(ref_ps, t, side="right") - 1
                keep = idx >= 0
                if not np.any(keep):
                    continue
                idx = idx[keep]; off_all = t[keep] - ref_ps[idx]
                if do_hist:                              # all-shots profile (pre-match)
                    inw = (off_all >= 0.0) & (off_all < win_ps)
                    if np.any(inw):
                        prof += np.bincount(
                            np.minimum((off_all[inw] / bin_ps).astype(np.int64), nbins - 1),
                            minlength=nbins)
                mk = matched[idx]
                if not np.any(mk):
                    continue
                idx = idx[mk]; off = off_all[mk]
                tau_ph = tau_index[idx]
                in_sig = (off >= sig_lo_ps) & (off < sig_hi_ps)
                if np.any(in_sig):
                    sig_total += np.bincount(tau_ph[in_sig], minlength=n_iterations)
                if use_ref:
                    in_ref = (off >= ref_lo_ps) & (off < ref_hi_ps)
                    if np.any(in_ref):
                        ref_total += np.bincount(tau_ph[in_ref], minlength=n_iterations)
                if do_hist:
                    in_win = (off >= 0.0) & (off < win_ps)
                    if np.any(in_win):
                        b = np.minimum((off[in_win] / bin_ps).astype(np.int64), nbins - 1)
                        H_flat += np.bincount(tau_ph[in_win] * nbins + b,
                                              minlength=n_iterations * nbins)
        H = H_flat.reshape(n_iterations, nbins) if do_hist else np.empty((0, 0), dtype=np.int64)
        profile = prof if do_hist else np.empty(0, dtype=np.int64)
        return sig_total, ref_total, H, edges_ns, n_spcm, profile

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

    # ---- Rabi reconstruction (MultiHarp only) ----------------------------- #
    @staticmethod
    def _counts_in_window(ref_ps: np.ndarray, other_ps: np.ndarray,
                          lo_ps: float, hi_ps: float) -> np.ndarray:
        """For each reference edge, COUNT `other` timestamps in
        [ref+lo_ps, ref+hi_ps). Vectorised; fine for millions of edges.

        Returns an int64 array, one count per reference edge (same order)."""
        n = ref_ps.size
        if n == 0:
            return np.zeros(0, dtype=np.int64)
        if other_ps.size == 0:
            return np.zeros(n, dtype=np.int64)
        lo = np.searchsorted(other_ps, ref_ps + lo_ps, side="left")
        hi = np.searchsorted(other_ps, ref_ps + hi_ps, side="left")
        return (hi - lo).astype(np.int64)

    @staticmethod
    def _pick_edge(ref_ps: np.ndarray, edge_ps: np.ndarray,
                   lo_ps: float, hi_ps: float, pick: str = "last") -> np.ndarray:
        """For each reference edge, return ONE `edge` timestamp (ps) inside
        [ref+lo_ps, ref+hi_ps], or NaN if none. pick='first' takes the earliest
        in the window, 'last' the latest. Vectorised over millions of edges.

        Used to grab the MW pulse edges belonging to each shot: with one MW pulse
        per shot there is exactly one edge of each polarity in the window, so
        'first'/'last' just disambiguate rise-vs-fall robustly."""
        n = ref_ps.size
        out = np.full(n, np.nan)
        if n == 0 or edge_ps.size == 0:
            return out
        lo = np.searchsorted(edge_ps, ref_ps + lo_ps, side="left")
        hi = np.searchsorted(edge_ps, ref_ps + hi_ps, side="right")
        has = hi > lo
        idx = (lo[has] if pick == "first" else hi[has] - 1)
        out[has] = edge_ps[idx]
        return out

    def _infer_rabi_axis(self, experiment: Any) -> Tuple[int, np.ndarray, bool]:
        """Work out (n_iterations, tau_values_ns, tau_is_index) for the sweep.

        Explicit values passed to the constructor win. Otherwise we read
        experiment.number_of_iterations and try to rebuild the tau axis from the
        first scan variable. If the variable's units/fields can't be read we fall
        back to a plain 0..n-1 index axis and flag tau_is_index=True (the shape of
        the oscillation is still correct; only the x-axis labels are indices)."""
        # n_iterations
        n = self.rabi_n_iterations
        if n is None:
            n = getattr(experiment, "number_of_iterations", 0) or 0
            if not n:
                n = len(getattr(experiment, "scan_sequences", []) or [])
        n = int(n) if n else 1

        # tau values
        if self.rabi_tau_values_ns is not None:
            tau = np.asarray(self.rabi_tau_values_ns, dtype=float)
            return n, tau, False

        tau = self._sweep_from_description(experiment)

        if tau is None or (n and tau.size != n):
            tau = np.arange(n, dtype=float)
            return n, tau, True
        return n, tau, False

    @staticmethod
    def _sweep_from_description(experiment: Any) -> Optional[np.ndarray]:
        """Rebuild the sweep values from the first scan variable (raw units, as
        the parser stores them), or None if they can't be read."""
        try:
            desc = getattr(experiment, "sequence_description", None)
            variables = getattr(desc, "variables", None)
            if not variables:
                return None
            v = variables[0] if not isinstance(variables, dict) \
                else list(variables.values())[0]

            def get(names):
                for nm in names:
                    if isinstance(v, dict) and nm in v:
                        return v[nm]
                    if hasattr(v, nm):
                        return getattr(v, nm)
                return None

            start = get(["start", "start_ns", "begin", "min"])
            stop = get(["stop", "stop_ns", "end", "max"])
            steps = get(["steps", "num_steps", "n_steps", "count", "n"])
            if start is not None and stop is not None and steps:
                return np.linspace(float(start), float(stop), int(steps))
        except Exception:
            return None
        return None

    def _mw_measure_tau(self, allch: Dict[int, np.ndarray], ref_ps: np.ndarray
                        ) -> Tuple[np.ndarray, str, Tuple[int, ...], float]:
        """Measure tau PER SHOT from the MW copy on the third input.

        Returns (tau_meas_ns, source, mw_inputs, lookback_ns). tau_meas_ns has one
        entry per reference (marker) edge; NaN where the MW pulse for that shot was
        not found. `source` is 'mw_width' or 'mw_geometry'."""
        gap = self.mw_gap_ns if self.mw_gap_ns is not None else 3000.0
        lookback = self.mw_lookback_ns if self.mw_lookback_ns is not None else (gap + 500.0)
        lo_ps = -lookback * 1000.0
        mw_ps = allch.get(self.mw_input, np.empty(0))
        if self.mw_mode == "width":
            fall_ps = allch.get(self.mw_fall_input, np.empty(0))
            # one MW pulse/shot -> one rising (MW start) and one falling (MW end)
            rise_t = self._pick_edge(ref_ps, mw_ps, lo_ps, 0.0, pick="first")
            fall_t = self._pick_edge(ref_ps, fall_ps, lo_ps, 0.0, pick="last")
            tau_meas = (fall_t - rise_t) / 1000.0     # ps -> ns; == pulse width == tau
            return tau_meas, "mw_width", (self.mw_input, self.mw_fall_input), lookback
        # geometry mode: single MW END edge, tau from the fixed MW->readout gap
        fall_t = self._pick_edge(ref_ps, mw_ps, lo_ps, 0.0, pick="last")
        tau_meas = gap - (ref_ps - fall_t) / 1000.0
        return tau_meas, "mw_geometry", (self.mw_input,), lookback

    @staticmethod
    def _snap_to_steps(tau_meas_ns: np.ndarray, tau_steps_ns: np.ndarray,
                       tol_ns: float) -> Tuple[np.ndarray, np.ndarray]:
        """Snap each measured tau to the nearest sweep step. Returns
        (step_index, matched) where step_index[i] is the index into tau_steps_ns
        of the nearest step and matched[i] is True iff within tol_ns (and finite).
        tau_steps_ns[k] is the tau of step k, so step_index IS the tau-step index."""
        n = tau_meas_ns.size
        step_index = np.zeros(n, dtype=np.int64)
        matched = np.zeros(n, dtype=bool)
        if n == 0 or tau_steps_ns.size == 0:
            return step_index, matched
        order = np.argsort(tau_steps_ns)
        ts = np.asarray(tau_steps_ns, dtype=float)[order]
        finite = np.isfinite(tau_meas_ns)
        tm = np.where(finite, tau_meas_ns, ts[0])          # placeholder for NaNs
        pos = np.clip(np.searchsorted(ts, tm), 1, ts.size - 1)
        left, right = ts[pos - 1], ts[pos]
        take_right = (tm - left) > (right - tm)
        nearest_sorted = np.where(take_right, pos, pos - 1)
        dist = np.abs(tm - ts[nearest_sorted])
        step_index = order[nearest_sorted]
        matched = finite & (dist <= tol_ns)
        return step_index, matched

    def _build_arrival_hist(self, ref_ps: np.ndarray, spcm_ps: np.ndarray,
                            tau_index: np.ndarray, matched: np.ndarray,
                            n_iterations: int) -> Tuple[np.ndarray, np.ndarray]:
        """Histogram SPCM arrivals relative to the marker, per tau step.

        Vectorised: each photon is attached to the marker at/just-before it, its
        offset is binned, and only photons from matched shots inside
        [0, tr_window] are kept. Returns (H[n_tau, n_bins], edges_ns)."""
        nbins = max(1, int(round(self.rabi_tr_window_ns / self.rabi_tr_bin_ns)))
        edges_ns = np.linspace(0.0, self.rabi_tr_window_ns, nbins + 1)
        H = np.zeros((n_iterations, nbins), dtype=np.int64)
        if ref_ps.size == 0 or spcm_ps.size == 0:
            return H, edges_ns
        win_ps = self.rabi_tr_window_ns * 1000.0
        # marker at/just-before each photon (markers are time-ordered)
        idx = np.searchsorted(ref_ps, spcm_ps, side="right") - 1
        ok = idx >= 0
        off = np.full(spcm_ps.shape, -1.0)
        off[ok] = spcm_ps[ok] - ref_ps[idx[ok]]
        good = ok & (off >= 0.0) & (off < win_ps)
        good[good] = matched[idx[good]]              # drop photons from unmatched shots
        if not np.any(good):
            return H, edges_ns
        ph_tau = tau_index[idx[good]].astype(np.int64)
        ph_bin = np.minimum((off[good] / (self.rabi_tr_bin_ns * 1000.0)).astype(np.int64),
                            nbins - 1)
        # accumulate into the flat [n_tau*nbins] grid
        flat = ph_tau * nbins + ph_bin
        counts = np.bincount(flat, minlength=n_iterations * nbins)
        H = counts.reshape(n_iterations, nbins).astype(np.int64)
        return H, edges_ns

    @staticmethod
    def recompute_rabi(result_or_npz, sig_window_ns, ref_window_ns=None):
        """Re-derive the Rabi curve for new windows from a RabiResult or a saved
        .npz path (or an npz handle / dict). Thin wrapper over
        rabi_curve_from_histogram so you can slide windows offline, e.g.:

            r = tester.run_with_experiment(exp)["rabi"]
            new = MultiHarpTimingTester.recompute_rabi(
                      r, sig_window_ns=(950, 1250), ref_window_ns=(5000, 5300))
            # ... or from disk, after the fact:
            new = MultiHarpTimingTester.recompute_rabi(
                      "multiharp_rabi_mw_tagged_20260819_120000.npz",
                      sig_window_ns=(950, 1250))
        """
        if isinstance(result_or_npz, RabiResult):
            H = result_or_npz.arrival_hist
            edges = result_or_npz.arrival_edges_ns
            spt = result_or_npz.shots_per_tau
        else:
            d = (np.load(result_or_npz, allow_pickle=True)
                 if isinstance(result_or_npz, str) else result_or_npz)
            H, edges, spt = d["arrival_hist"], d["arrival_edges_ns"], d["shots_per_tau"]
        if np.asarray(H).size == 0:
            raise ValueError("no time-resolved data present (run with "
                             "rabi_time_resolved=True to enable re-windowing)")
        return rabi_curve_from_histogram(H, edges, spt, sig_window_ns, ref_window_ns)

    def _compare_trigger_vs_markers(self, trig_ps: np.ndarray, mark_ps: np.ndarray,
                                    n_iterations: int, res: "RabiResult") -> None:
        """Compare ADwin-trigger (DIGOUT 28) edges to Proteus readout-marker edges
        from the SAME T2 stream. DO28 is one edge per ADwin trigger (emitted by the
        master), so this tests in one run whether every trigger produced exactly one
        Proteus sequence. Fills `res` and prints the count comparison plus a
        per-trigger breakdown (dropped / clean / ringing) and any orphan markers
        (a marker with no preceding trigger = a Proteus edge the ADwin never
        triggered, e.g. the init burst you see before the sweep). Both inputs carry
        one clean edge per shot, so these arrays are ~n_shots long (cheap)."""
        n_trig = int(trig_ps.size)
        n_mark = int(mark_ps.size)
        res.trigger_input = self.trig_input
        res.n_trigger_events = n_trig
        rc = getattr(self, "_rabi_repeat_count", None)
        expected = int(rc) * n_iterations if rc else None
        print(f"\n[MultiHarp] TRIGGER vs MARKER cross-check (DO28 on "
              f"ch{self.trig_input} vs readout marker on ch{self.start_input}):")
        print(f"    ADwin triggers (DO28): {n_trig:,}")
        print(f"    Proteus markers      : {n_mark:,}"
              + (f"   (expected ~{expected:,})" if expected else ""))
        print(f"    markers - triggers   : {n_mark - n_trig:+,}")
        if n_trig == 0 or n_mark == 0:
            print("    one channel is empty -- can't pair. Check its level/edge/wiring.")
            return
        # trigger at/just-before each marker (both are time-ordered)
        j = np.searchsorted(trig_ps, mark_ps, side="right") - 1
        orphan = j < 0
        res.n_orphan_markers = int(np.count_nonzero(orphan))
        lat_std = float("nan")
        if np.any(~orphan):
            lat = (mark_ps[~orphan] - trig_ps[j[~orphan]]) / 1000.0   # ps -> ns
            res.trig_marker_latency_ns = float(np.median(lat))
            lat_std = float(np.std(lat))
        # markers falling in [trig_k, trig_{k+1}); 1 = clean, 0 = dropped, >1 = ringing
        per = np.bincount(j[~orphan], minlength=n_trig) if n_trig else np.zeros(0, dtype=np.int64)
        res.n_triggers_no_marker = int(np.count_nonzero(per == 0))
        res.n_triggers_multi_marker = int(np.count_nonzero(per > 1))
        print(f"    per-trigger: {int(np.count_nonzero(per == 1)):,} clean (exactly 1), "
              f"{res.n_triggers_no_marker:,} with 0 (dropped sequence), "
              f"{res.n_triggers_multi_marker:,} with >1 (marker ringing)")
        if res.n_orphan_markers:
            print(f"    {res.n_orphan_markers:,} ORPHAN markers have no preceding "
                  f"trigger -- Proteus edges the ADwin never asked for (the init "
                  f"burst before the sweep).")
        if np.isfinite(res.trig_marker_latency_ns):
            print(f"    trigger->marker latency: median "
                  f"{res.trig_marker_latency_ns:.1f} ns, std {lat_std:.1f} ns "
                  f"(= Proteus trigger latency + the marker's position in the "
                  f"sequence; a tight std means clean, stable pairing)")
        clean = (n_mark == n_trig and res.n_triggers_no_marker == 0
                 and res.n_triggers_multi_marker == 0 and res.n_orphan_markers == 0)
        print("    -> " + ("MATCH: markers == triggers, 1:1 (every trigger makes "
                            "exactly one sequence)."
                            if clean else
                            "MISMATCH: NOT 1:1 -- see the breakdown above."))

    def analyze_rabi(self, records: np.ndarray, n_iterations: int,
                     tau_values_ns: np.ndarray, tau_is_index: bool = False
                     ) -> RabiResult:
        """Build a Rabi curve from raw T2 records, using ONLY the MultiHarp.

        start_input = a once-per-shot reference edge (readout laser marker best,
        or the trigger). stop_input = the SPCM. Per shot we count SPCM photons in
        the signal window (bright readout onset) and, if configured, a reference
        window later inside the same readout pulse. Shots are assigned to tau
        steps by their order (interleaved task table -> shot i has
        tau_index = i mod n_iterations), then averaged per step.

        TWO WAYS TO ASSIGN tau:
          * shot ORDER (default, case="rabi_vs_spcm"): the interleaved task table
            means shot i has tau_index = i mod n_iterations. Simple, but a single
            MISSED reference edge shifts every later shot by one tau and smears the
            curve, so the reference edge must be a clean once-per-shot TTL
            (attenuated to <=1.2 V, level ~half amplitude, correct edge). We
            sanity-check the shot count against repeat_count x n_iterations.
          * MW TAG (case="rabi_mw_tagged", or any time mw_input is set): tau is read
            PER SHOT from the MW pulse on a third input -- its WIDTH is tau (two-edge
            WIDTH mode) or it is derived from the MW end vs the readout marker
            (one-BNC GEOMETRY mode). Each measured tau is snapped to the nearest
            sweep step; a dropped edge now drops only that shot instead of shifting
            all the later ones. This removes the ordering assumption entirely and is
            the recommended way to settle "is it the ADwin or a real drift?"."""
        n_iterations = max(1, int(n_iterations))
        rc = getattr(self, "_rabi_repeat_count", None)
        expected_markers = int(rc) * n_iterations if rc else None

        # ---- 1) cheap streaming count pass: composition + multi-trigger check
        counts = self._channel_counts(records)
        self._report_channel_counts(counts, expected_markers)
        n_start = counts.get(self.start_input, 0)
        n_spcm_ct = counts.get(self.stop_input, 0)

        res = RabiResult(
            start_input=self.start_input,
            stop_input=self.stop_input,
            n_ref_events=int(n_start),
            n_spcm_events=int(n_spcm_ct),
            n_shots_used=0,
            n_iterations=n_iterations,
            repeat_count=(int(rc) if rc else None),
            base_resolution_ps=self._base_res_ps,
            sig_offset_ns=self.rabi_sig_offset_ns,
            sig_width_ns=self.rabi_sig_width_ns,
            reference_label=self.start_label,
            tau_ns=np.asarray(tau_values_ns, dtype=float),
            tau_is_index=bool(tau_is_index),
            fifo_overrun=self._fifo_overrun,
        )

        if n_start == 0 or n_spcm_ct == 0:
            dead = ("reference/start (input %d)" % self.start_input
                    if n_start == 0 else "SPCM/stop (input %d)" % self.stop_input)
            print(f"[MultiHarp] Rabi: the {dead} channel saw 0 events -- no curve. "
                  f"Fix that channel's level/edge/wiring (attenuate TTL to <=1.2 V, "
                  f"level ~half amplitude). Try scan_all_channels=True to see who fires.")
            return res

        # Guard the one array we must hold whole (the marker edges). A genuinely
        # runaway (multi-triggering) marker would thrash; stop with a clear message
        # rather than crash. But do NOT trip just because a long, LEGITIMATE
        # multi-average run has many clean edges -- compare against the DO28 trigger
        # count (the true shot count) when it is available.
        n_trig_ct = counts.get(self.trig_input, 0) if self.trig_input is not None else 0
        markers_clean = (n_trig_ct > 0 and n_start <= 1.5 * n_trig_ct)
        if n_start > self.max_marker_edges and not markers_clean:
            raise RuntimeError(
                f"[MultiHarp] Rabi: the reference/START channel (input "
                f"{self.start_input}) has {n_start:,} edges, above the "
                f"max_marker_edges={self.max_marker_edges:,} safety limit"
                + (f" and far above the {n_trig_ct:,} DO28 triggers"
                   if n_trig_ct else ", with no DO28 trigger channel to confirm them")
                + f". This usually means the marker is MULTI-TRIGGERING (ringing/"
                f"reflection or the trigger level too low) -- fix that so it fires "
                f"once per shot. If instead this is a very long legitimate run, reduce "
                f"averages for the MultiHarp pass, re-analyze the saved raw .bin "
                f"offline with analyze_records_file(), or raise max_marker_edges.")

        # ---- 2) sparse pass: materialise ONLY marker + MW edge times -------
        want = [self.start_input]
        if self.mw_mode != "off":
            want += [self.mw_input, self.mw_fall_input]
        if self.trig_input is not None:
            want.append(self.trig_input)
        sparse = self._collect_channel_times(records, want)
        ref_ps = sparse.get(self.start_input, np.empty(0))   # once-per-shot reference edges
        n_shots = ref_ps.size
        # Sweep order: the Proteus advances ONE tau step PER TRIGGER -- one TRG1-gated
        # task per tau, next -> +1, set_task_loop left off in the ODMR task table, and
        # confirmed on the scope -- i.e. INTERLEAVED. (rabi_order="blocked" is kept only
        # for a task table that loops each segment repeat_count times.)
        # With the DO28 copy wired, count TRIGGERS rather than marker edges: an orphan
        # (e.g. a Proteus init burst) or doubled marker edge would otherwise shift every
        # later shot by one tau. Such edges are left unassigned instead (order_ok).
        _rc_o = int(getattr(self, "_rabi_repeat_count", 0) or 0)
        _trig_o = (sparse.get(self.trig_input, np.empty(0))
                   if self.trig_input is not None else np.empty(0))
        order_ok = np.ones(n_shots, dtype=bool)
        if _trig_o.size and n_shots:
            _run_id, _local, _win = self._trigger_runs(_trig_o)[1:4]
            _k = np.searchsorted(_trig_o, ref_ps, side="right") - 1   # trigger owning each edge
            _kc = np.maximum(_k, 0)
            _first = np.ones(n_shots, dtype=bool)
            _first[1:] = _k[1:] != _k[:-1]                          # first edge after its trigger
            order_ok = (_k >= 0) & ((ref_ps - _trig_o[_kc]) < _win[_kc]) & _first
            _seq = _local[_kc]
            if not np.all(order_ok):
                print(f"[MultiHarp] shot order counted by DO28 triggers: "
                      f"{int(np.count_nonzero(~order_ok)):,} marker edge(s) with no trigger of "
                      f"their own (orphan / doubled) are left out instead of shifting later shots.")
        else:
            _seq = np.arange(n_shots)
        if self.rabi_order == "blocked" and _rc_o > 0:
            order_index = (_seq // _rc_o) % n_iterations
        else:
            order_index = _seq % n_iterations

        # ---- optional: Proteus readout markers vs ADwin triggers (DO28) ----
        if self.trig_input is not None:
            self._compare_trigger_vs_markers(
                sparse.get(self.trig_input, np.empty(0)), ref_ps, n_iterations, res)

        # ---- 3) assign shots -> tau steps (MW tag if available, else order)
        want_mw = self.mw_mode != "off"
        n_mw = counts.get(self.mw_input, 0) if self.mw_input is not None else 0
        n_mwf = counts.get(self.mw_fall_input, 0) if self.mw_fall_input is not None else 0
        mw_ready = want_mw and not tau_is_index and n_mw > 0
        # WIDTH mode needs BOTH edges; a dead falling-edge input makes every width
        # undefined (this is the usual cause of "0 matched").
        if want_mw and self.mw_mode == "width" and n_mwf == 0:
            print(f"[MultiHarp] Rabi MW-tag WIDTH mode needs both edges, but the "
                  f"FALLING-edge input {self.mw_fall_input} saw 0 events -- tau = "
                  f"fall - rise cannot be formed. Check that BNC's cable/level/edge "
                  f"(or use GEOMETRY mode: drop mw_fall_input and set mw_gap_ns).")
            mw_ready = False
        if want_mw and not mw_ready and not (self.mw_mode == "width" and n_mwf == 0):
            why = ("the tau axis is only step indices (pass rabi_tau_values_ns in ns "
                   "so measured widths can be matched)" if tau_is_index
                   else f"input {self.mw_input} (MW copy) saw no events -- check its "
                        f"level/edge/wiring")
            print(f"[MultiHarp] Rabi MW-tag requested but unavailable: {why}.")

        use_mw_result = False
        if mw_ready:
            tau_meas, source, mw_chans, lookback = self._mw_measure_tau(sparse, ref_ps)
            steps = np.asarray(tau_values_ns, dtype=float)
            if self.mw_snap_tol_ns is not None:
                tol = float(self.mw_snap_tol_ns)
            else:
                ds = np.diff(np.sort(steps[np.isfinite(steps)]))
                spacing = float(np.median(ds)) if ds.size else 1.0
                tol = max(1.0, 0.5 * spacing)             # half a step by default
            tau_index_mw, matched_mw = self._snap_to_steps(tau_meas, steps, tol)
            n_match = int(np.count_nonzero(matched_mw))
            if n_match >= max(1, int(0.02 * n_shots)):     # MW tagging is working
                use_mw_result = True
                tau_index, matched = tau_index_mw, matched_mw
                res.tau_assign_source = source
                res.mw_inputs = tuple(int(c) for c in mw_chans)
                res.mw_snap_tol_ns = tol
                res.n_matched = n_match
                res.n_unmatched = n_shots - n_match
                _mo = matched & order_ok
                res.order_agreement = (float(np.mean(order_index[_mo] == tau_index[_mo]))
                                       if np.any(_mo) else float("nan"))
                print(f"[MultiHarp] Rabi MW-tag ({source}, tol +/-{tol:.0f} ns): matched "
                      f"{res.n_matched:,}/{n_shots:,} shots; {res.n_unmatched:,} unmatched. "
                      f"MW-tag vs shot-order agree on "
                      f"{100.0 * res.order_agreement:.1f}% of matched shots.")
            else:
                # MW tagging effectively failed -- explain WHY before falling back.
                finite = np.isfinite(tau_meas)
                nfin = int(np.count_nonzero(finite))
                if nfin:
                    fw = tau_meas[finite]
                    print(f"[MultiHarp] Rabi MW-tag matched only {n_match:,}/{n_shots:,} "
                          f"shots. Measured widths: {nfin:,} finite, range "
                          f"{np.min(fw):.0f}..{np.max(fw):.0f} ns (median {np.median(fw):.0f}); "
                          f"the sweep spans {np.nanmin(steps):.0f}..{np.nanmax(steps):.0f} ns, "
                          f"tol +/-{tol:.0f} ns. The widths do not line up with the sweep -- "
                          f"check the MW edges/level, or that mw_input/mw_fall_input aren't "
                          f"swapped, or widen mw_snap_tol_ns.")
                else:
                    print(f"[MultiHarp] Rabi MW-tag matched 0/{n_shots:,}: every measured "
                          f"width is undefined (a MW edge is missing on essentially every "
                          f"shot). Verify BOTH MW inputs fire exactly once per shot.")

        if not use_mw_result:
            tau_index = order_index
            matched = order_ok.copy()
            res.tau_assign_source = "shot_order"
            res.mw_inputs = ()
            res.n_matched = int(np.count_nonzero(matched))
            res.n_unmatched = int(n_shots - res.n_matched)
            if want_mw:
                clean = (expected_markers and abs(n_shots - expected_markers) <= n_iterations)
                print(f"[MultiHarp] Falling back to SHOT-ORDER (shot i -> tau i mod "
                      f"{n_iterations}). " + ("The marker looks clean (~1 edge/shot), so "
                      "this should give a valid curve." if clean else "NOTE: the marker "
                      "count does not match repeat_count x tau steps, so shot-order may be "
                      "desynced -- fix the MW inputs and re-run for a trustworthy curve."))

        res.n_shots_used = int(np.count_nonzero(matched))

        # ---- 4) stream the SPCM channel: window counts + arrival histogram --
        sig_lo = self.rabi_sig_offset_ns * 1000.0
        sig_hi = (self.rabi_sig_offset_ns + self.rabi_sig_width_ns) * 1000.0
        use_ref = bool(self.rabi_ref_width_ns and self.rabi_ref_width_ns > 0)
        ref_lo = self.rabi_ref_offset_ns * 1000.0
        ref_hi = (self.rabi_ref_offset_ns + self.rabi_ref_width_ns) * 1000.0
        if use_ref:
            res.ref_offset_ns = self.rabi_ref_offset_ns
            res.ref_width_ns = self.rabi_ref_width_ns
        sig_total, ref_total, H, edges_ns, n_spcm_used, profile = self._stream_spcm_windows(
            records, ref_ps, tau_index, matched, n_iterations,
            sig_lo, sig_hi, use_ref, ref_lo, ref_hi)

        shots_per = np.bincount(tau_index[matched], minlength=n_iterations).astype(np.int64)
        with np.errstate(divide="ignore", invalid="ignore"):
            sig_mean = np.where(shots_per > 0, sig_total / shots_per, np.nan)
            # Poisson standard error on the mean (exact for these low per-shot counts).
            sig_sem = np.where(shots_per > 0, np.sqrt(sig_total) / shots_per, np.nan)
            ref_mean = (np.where(shots_per > 0, ref_total / shots_per, np.nan)
                        if use_ref else np.full(n_iterations, np.nan))
        res.signal_mean = sig_mean
        res.signal_sem = sig_sem
        res.shots_per_tau = shots_per
        if use_ref:
            res.reference_mean = ref_mean
            with np.errstate(divide="ignore", invalid="ignore"):
                res.contrast = np.where(ref_mean > 0, sig_mean / ref_mean, np.nan)
        if self.rabi_time_resolved and H.size:
            res.arrival_hist = H
            res.arrival_edges_ns = edges_ns
            res.time_resolved = True
        if profile.size:
            res.arrival_profile_ns = profile         # tau-independent readout (all shots)

        # If no shot contributed, still surface what the readout looked like so the
        # run is diagnosable rather than a silent empty file.
        if res.n_shots_used == 0:
            tot = int(profile.sum()) if profile.size else 0
            if tot > 0:
                ctr = 0.5 * (edges_ns[:-1] + edges_ns[1:])
                on = ctr[np.argmax(profile > 0.25 * np.max(profile))] if np.max(profile) else 0.0
                print(f"[MultiHarp] Rabi: 0 shots were assigned a tau, so there is no "
                      f"curve -- but the SPCM DID see {tot:,} photons in the readout "
                      f"(light turns on ~{on:.0f} ns after the marker). The problem is the "
                      f"tau ASSIGNMENT (MW inputs), not the photon path. Fix the MW inputs "
                      f"(or re-run case='rabi_vs_spcm' now that the marker is clean).")
            else:
                print(f"[MultiHarp] Rabi: 0 shots assigned AND 0 SPCM photons in the "
                      f"readout window -- check the SPCM (input {self.stop_input}) and the "
                      f"signal-window placement.")

        # ---- oscillation descriptor -------------------------------------- #
        curve = res.contrast if (use_ref and np.isfinite(res.contrast).any()) else sig_mean
        res.best_metric = "contrast" if (use_ref and np.isfinite(res.contrast).any()) else "signal_mean"
        good = curve[np.isfinite(curve)]
        if good.size >= 2:
            hi, lo = float(np.max(good)), float(np.min(good))
            res.modulation_depth = (hi - lo) / (hi + lo) if (hi + lo) > 0 else float("nan")

        # ---- phase / desync sanity check --------------------------------- #
        # Only meaningful for shot-ORDER assignment; MW-tagged runs tolerate
        # dropped shots by construction (they just go unmatched), so we leave
        # phase_ok True there.
        rc = getattr(self, "_rabi_repeat_count", None)
        res.repeat_count = int(rc) if rc else None
        if rc and res.tau_assign_source == "shot_order":
            expected = int(rc) * n_iterations
            # allow the last (partial) cycle: within one full cycle is fine.
            res.phase_ok = abs(n_shots - expected) <= n_iterations
            if not res.phase_ok:
                print(f"[MultiHarp] Rabi WARNING: saw {n_shots} reference edges but "
                      f"expected ~{expected} (= repeat_count {rc} x {n_iterations} taus). "
                      f"Shot->tau assignment may be desynced by dropped/extra edges. "
                      f"Consider case='rabi_mw_tagged' (wire the MW line to a 3rd input).")
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

    # ---- Rabi plotting / saving ------------------------------------------ #
    def _plot_rabi(self, res: RabiResult) -> "Optional[Any]":
        if not self.make_plots:
            return None
        two = np.isfinite(res.contrast).any() if res.contrast.size else False
        fig, axes = plt.subplots(2 if two else 1, 1, sharex=True,
                                 figsize=(7.5, 6.0 if two else 3.8))
        if not two:
            axes = [axes]
        xlabel = ("tau step index" if res.tau_is_index
                  else "MW pulse duration tau (ns)")

        ax0 = axes[0]
        m = np.isfinite(res.signal_mean)
        ax0.errorbar(res.tau_ns[m], res.signal_mean[m],
                     yerr=(res.signal_sem[m] if res.signal_sem.size else None),
                     fmt="-o", ms=4, lw=1.3, capsize=2, label="signal window")
        ax0.set_ylabel("SPCM counts / shot\n(signal window)")
        if res.tau_assign_source == "mw_width":
            tau_src = "  |  tau from MW pulse width (3rd input)"
        elif res.tau_assign_source == "mw_geometry":
            tau_src = "  |  tau from MW end vs marker (3rd input)"
        else:
            tau_src = ""
        ax0.set_title(
            f"MultiHarp Rabi (no ADwin gating)  |  {res.n_shots_used:,} shots, "
            f"{res.n_iterations} taus{tau_src}\nsignal [{res.sig_offset_ns:.0f},"
            f"{res.sig_offset_ns + res.sig_width_ns:.0f}] ns after "
            f"{res.reference_label}")
        ax0.legend(loc="best", fontsize=8)
        if not res.phase_ok:
            ax0.text(0.5, 0.02, "shot->tau desync suspected (see log)", color="crimson",
                     ha="center", va="bottom", transform=ax0.transAxes, fontsize=9)

        if two:
            ax1 = axes[1]
            mc = np.isfinite(res.contrast)
            ax1.plot(res.tau_ns[mc], res.contrast[mc], "-o", ms=4, lw=1.3, color="C1",
                     label="signal / reference")
            ax1.set_ylabel("contrast\n(signal / reference)")
            ax1.legend(loc="best", fontsize=8)
            ax1.set_xlabel(xlabel)
            if np.isfinite(res.modulation_depth):
                ax1.set_title(f"modulation depth (max-min)/(max+min) = "
                              f"{res.modulation_depth:.3f}", fontsize=9)
        else:
            ax0.set_xlabel(xlabel)
            if np.isfinite(res.modulation_depth):
                ax0.text(0.98, 0.02,
                         f"mod. depth = {res.modulation_depth:.3f}",
                         ha="right", va="bottom", transform=ax0.transAxes, fontsize=9)
        fig.tight_layout()
        return fig

    def _plot_rabi_map(self, res: RabiResult) -> "Optional[Any]":
        """Heatmap of SPCM counts/shot vs (arrival time after marker, tau), with
        the current signal/reference windows drawn on top. Shows exactly where
        the laser rise sits so you can re-pick windows with recompute_rabi()."""
        if not self.make_plots or res.arrival_hist.size == 0:
            return None
        H = res.arrival_hist.astype(float)
        spt = res.shots_per_tau.astype(float).copy()
        spt[spt == 0] = np.nan
        per_shot = H / spt[:, None]                    # counts/shot per (tau, time-bin)
        edges = res.arrival_edges_ns
        ycancel = res.tau_is_index
        y = res.tau_ns if res.tau_ns.size == H.shape[0] else np.arange(H.shape[0])
        fig, ax = plt.subplots(figsize=(8.5, 4.6))
        extent = [edges[0], edges[-1], y[-1], y[0]]    # origin upper; tau on y
        im = ax.imshow(per_shot, aspect="auto", extent=extent,
                       interpolation="nearest", cmap="viridis")
        cb = fig.colorbar(im, ax=ax); cb.set_label("SPCM counts / shot")
        # overlay the signal (and reference) windows currently in use
        s0 = res.sig_offset_ns; s1 = res.sig_offset_ns + res.sig_width_ns
        ax.axvline(s0, color="w", lw=1.4); ax.axvline(s1, color="w", lw=1.4, ls="--")
        ax.text(0.5 * (s0 + s1), y[0], " signal", color="w", va="top", ha="center",
                fontsize=8, rotation=90)
        if np.isfinite(res.ref_width_ns) and res.ref_width_ns > 0:
            r0 = res.ref_offset_ns; r1 = res.ref_offset_ns + res.ref_width_ns
            ax.axvline(r0, color="orange", lw=1.4); ax.axvline(r1, color="orange", lw=1.4, ls="--")
            ax.text(0.5 * (r0 + r1), y[0], " reference", color="orange", va="top",
                    ha="center", fontsize=8, rotation=90)
        ax.set_xlabel(f"SPCM arrival time after {res.reference_label}  [ns]")
        ax.set_ylabel("tau step index" if ycancel else "MW pulse duration tau (ns)")
        ax.set_title("Time-resolved Rabi map (full readout kept)\n"
                     "white = signal window, orange = reference; "
                     "re-pick offline with recompute_rabi()")
        fig.tight_layout()
        return fig

    def _save_rabi_outputs(self, res: RabiResult) -> None:
        outdir = self.output_dir or os.getcwd()
        os.makedirs(outdir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        base = os.path.join(outdir, f"multiharp_{self.case}_{stamp}")

        if self.save_npz:
            npz = base + ".npz"
            np.savez_compressed(
                npz,
                case=self.case,
                start_input=self.start_input, stop_input=self.stop_input,
                base_resolution_ps=self._base_res_ps,
                n_iterations=res.n_iterations, repeat_count=(res.repeat_count or 0),
                sig_offset_ns=res.sig_offset_ns, sig_width_ns=res.sig_width_ns,
                ref_offset_ns=res.ref_offset_ns, ref_width_ns=res.ref_width_ns,
                tau_ns=res.tau_ns, tau_is_index=int(res.tau_is_index),
                signal_mean=res.signal_mean, signal_sem=res.signal_sem,
                reference_mean=res.reference_mean, contrast=res.contrast,
                shots_per_tau=res.shots_per_tau,
                modulation_depth=res.modulation_depth,
                phase_ok=int(res.phase_ok),
                # MW-tagging provenance
                tau_assign_source=res.tau_assign_source,
                mw_inputs=np.asarray(res.mw_inputs, dtype=np.int64),
                n_matched=res.n_matched, n_unmatched=res.n_unmatched,
                mw_snap_tol_ns=res.mw_snap_tol_ns, order_agreement=res.order_agreement,
                # time-resolved arrivals (re-window offline with recompute_rabi)
                time_resolved=int(res.time_resolved),
                arrival_hist=res.arrival_hist, arrival_edges_ns=res.arrival_edges_ns,
                arrival_profile_ns=res.arrival_profile_ns,
            )
            res.saved_files.append(npz)

        if self.save_hdf5:
            try:
                import h5py  # noqa: F401
                from src.core.struct_hdf5 import MyStruct, save_data
                s = MyStruct()
                s.data = MyStruct(tau_ns=res.tau_ns, signal_mean=res.signal_mean,
                                  signal_sem=res.signal_sem,
                                  reference_mean=res.reference_mean,
                                  contrast=res.contrast,
                                  shots_per_tau=res.shots_per_tau,
                                  arrival_hist=res.arrival_hist,
                                  arrival_edges_ns=res.arrival_edges_ns)
                s.meta = MyStruct(case=self.case,
                                  start_input=self.start_input,
                                  stop_input=self.stop_input,
                                  start_label=res.reference_label,
                                  stop_label=self.stop_label,
                                  base_resolution_ps=res.base_resolution_ps,
                                  n_iterations=res.n_iterations,
                                  repeat_count=(res.repeat_count or 0),
                                  sig_offset_ns=res.sig_offset_ns,
                                  sig_width_ns=res.sig_width_ns,
                                  ref_offset_ns=res.ref_offset_ns,
                                  ref_width_ns=res.ref_width_ns,
                                  tau_is_index=int(res.tau_is_index),
                                  modulation_depth=res.modulation_depth,
                                  best_metric=res.best_metric,
                                  phase_ok=int(res.phase_ok),
                                  fifo_overrun=int(res.fifo_overrun),
                                  tau_assign_source=res.tau_assign_source,
                                  mw_inputs=",".join(str(c) for c in res.mw_inputs),
                                  n_matched=res.n_matched,
                                  n_unmatched=res.n_unmatched,
                                  mw_snap_tol_ns=res.mw_snap_tol_ns,
                                  order_agreement=res.order_agreement,
                                  timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
                save_data(base + ".h5", s)
            except Exception as e:
                print(f"[MultiHarp] Rabi HDF5 save skipped ({e}); the .npz has everything.")

        if self.make_plots and res.n_shots_used:
            try:
                fig = self._plot_rabi(res)
                if fig is not None:
                    png = base + ".png"
                    fig.savefig(png, dpi=130)
                    plt.close(fig)
                    res.saved_files.append(png)
            except Exception as e:
                print(f"[MultiHarp] Rabi plot warning: {e}")
            if res.time_resolved and res.arrival_hist.size:
                try:
                    fig = self._plot_rabi_map(res)
                    if fig is not None:
                        pmap = base + "_map.png"
                        fig.savefig(pmap, dpi=130)
                        plt.close(fig)
                        res.saved_files.append(pmap)
                except Exception as e:
                    print(f"[MultiHarp] Rabi map plot warning: {e}")

    def run_rabi_with_experiment(
        self,
        experiment: Any,
        frequency_range: Optional[List[float]] = None,
        save: bool = True,
        run_method: str = "run_experiment",
        raw_records_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a Rabi sweep and reconstruct the Rabi curve FROM THE MULTIHARP.

        This is the answer to "is it the ADwin, or a real drift on my setup?":
        the curve here never touches the ADwin gate. Wire start_input to the
        readout laser marker (recommended -- add a `marker, laser_readout_1 on
        channel 4 at <readout_start>ns, <readout_len>ns` line to the sequence so
        the Proteus emits one clean edge at the readout onset) and stop_input to
        the (attenuated) SPCM. Then:

          * clean oscillation here + bad ADwin result  -> the ADwin counting/gate
            is the problem (misaligned or jittering gate; see the diagnosis).
          * flat/noisy here too                        -> the problem is upstream
            (polarization, MW, optics, or a genuine slow drift).

        The signal window must sit on the bright readout onset; use a reference
        window later but INSIDE the same readout pulse for a drift-robust ratio.

        raw_records_path: for long runs, dump the raw T2 records to this .npy path
        BEFORE analysis. Cheap insurance -- if anything downstream fails you keep
        the data and can re-analyze offline with analyze_records_file() (which
        memory-maps the file and decodes in chunks, so it never loads it whole)."""
        if self.case not in self._RABI_CASES:
            print(f"[MultiHarp] note: case is {self.case!r}; running the Rabi "
                  f"reconstruction anyway. Pass case='rabi_vs_spcm' (shot order) or "
                  f"case='rabi_mw_tagged' (MW third input) to silence this.")
        if frequency_range is None:
            frequency_range = experiment.settings['microwave']['frequency range']
        if not hasattr(experiment, run_method):
            raise AttributeError(f"experiment has no method {run_method!r}")
        if self.output_dir is None:
            self.output_dir = getattr(experiment, "output_dir", None) or os.getcwd()

        # Stash repeat_count for the desync sanity check (best-effort).
        self._rabi_repeat_count = getattr(experiment, "repeat_count", None)

        self.open_and_configure()
        odmr_result: Dict[str, Any] = {}
        rabi = None
        raw_on_disk = None
        try:
            self.start_acquisition()
            try:
                odmr_result = getattr(experiment, run_method)(frequency_range)
            finally:
                records = self.stop_acquisition()
            # The raw stream is already on disk when streaming; otherwise persist it
            # now so a long run is never lost to an analysis error.
            raw_on_disk = self._raw_path if (self.stream_raw_to_disk and self._raw_path) else None
            if raw_records_path and not raw_on_disk:
                try:
                    np.save(raw_records_path, records)
                    raw_on_disk = raw_records_path
                    print(f"[MultiHarp] raw records saved to {raw_records_path} "
                          f"({records.size:,} records).")
                except Exception as e:
                    print(f"[MultiHarp] WARNING: could not save raw records "
                          f"({e}); continuing to analysis.")
            n_iter, tau_ns, tau_is_index = self._infer_rabi_axis(experiment)
            self._rabi_axis_cache = (n_iter, tau_ns, tau_is_index)   # for offline re-analysis
            # Analysis must never lose the run: the raw is already on disk, so if it
            # (or its memory) fails, report it and carry on -- the caller still saves
            # the ADwin data, and you can re-analyze the raw offline.
            try:
                rabi = self.analyze_rabi(records, n_iter, tau_ns, tau_is_index)
                if save:
                    self._save_rabi_outputs(rabi)
            except Exception as e:
                print(f"[MultiHarp] WARNING: Rabi analysis failed ({e}). The raw "
                      f"records are safe"
                      + (f" on disk at {raw_on_disk}" if raw_on_disk else "")
                      + f"; re-analyze offline with analyze_records_file(). The ADwin "
                      f"data is unaffected and will still be saved.")
        finally:
            self.close()

        if rabi is not None:
            print(rabi.summary())
        return {"odmr": odmr_result, "rabi": rabi, "raw_path": raw_on_disk}

    def analyze_records_file(self, path: str, n_iterations: int,
                             tau_values_ns, tau_is_index: bool = False,
                             repeat_count: Optional[int] = None,
                             save: bool = True) -> "RabiResult":
        """Re-run the Rabi analysis on a raw-records .npy saved by
        run_rabi_with_experiment(raw_records_path=...). The file is MEMORY-MAPPED
        and decoded in chunks, so even a 10-hour, tens-of-GB capture is analyzed
        without loading it whole. Pass the same n_iterations/tau_values_ns you
        used for the run (and repeat_count for the desync check)."""
        records = self._open_records(path)
        if repeat_count is not None:
            self._rabi_repeat_count = repeat_count
        if self.output_dir is None:
            self.output_dir = os.path.dirname(os.path.abspath(path)) or os.getcwd()
        rabi = self.analyze_rabi(records, n_iterations, tau_values_ns, tau_is_index)
        if save:
            self._save_rabi_outputs(rabi)
        print(rabi.summary())
        return rabi

    # ======================================================================= #
    # T1 (case="t1_vs_spcm"): SPCM registered after the readout marker ONLY
    # ======================================================================= #
    # Wiring:  trig_input  = ADwin DO28 (copy of DO21, the Proteus trigger)
    #          start_input = Proteus ch4 MKR copying the laser: one marker line on
    #                        the init pulse and one on the readout pulse
    #          stop_input  = SPCM
    # Rules:   DO28 is a CLOCK only. It groups the marker edges into shots and
    #          times DO28 -> marker; those durations give each shot its tau.
    #          An SPCM photon is registered only if it falls inside the readout
    #          light after a readout-marker edge (auto: the light pulse's 10%
    #          points). Nothing between DO28 and the readout marker -- the init
    #          pulse, the dark time -- is ever counted, and no photon is ever
    #          timed against DO28 itself.
    # ======================================================================= #
    @staticmethod
    def _open_records(path: str) -> np.ndarray:
        """Memory-map raw T2 records: a .npy (np.save) or the raw uint32 .bin that
        stream_raw_to_disk writes. Nothing is loaded into RAM."""
        if str(path).lower().endswith(".npy"):
            return np.load(path, mmap_mode="r")
        if os.path.getsize(path) == 0:
            return np.empty(0, dtype=np.uint32)
        return np.memmap(path, dtype=np.uint32, mode="r")

    @staticmethod
    def _robust_span(x, lo: float = 0.1, hi: float = 99.9) -> float:
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return float("nan")
        a, b = np.percentile(x, [lo, hi])
        return float(b - a)

    @staticmethod
    def _pearson(a, b) -> float:
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        if a.shape != b.shape:
            return float("nan")
        m = np.isfinite(a) & np.isfinite(b)
        if np.count_nonzero(m) < 3:
            return float("nan")
        a, b = a[m], b[m]
        if np.std(a) == 0 or np.std(b) == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    @staticmethod
    def _light_pulse(edges_ns, prof) -> Tuple[float, float, float, float]:
        """Find the readout light pulse in a photon profile measured after the
        readout marker: (on10, on50, off50, off10) in ns. The pulse is the LONGEST
        run of bins above half the plateau, so a short init-light tail (a separate,
        earlier blob at short tau) can't be taken for it. The 10% points are
        searched at most 300 ns outside the 50% points and stop at a dip. All NaN
        when there are too few photons."""
        nan4 = (float("nan"),) * 4
        p = np.asarray(prof, dtype=float)
        e = np.asarray(edges_ns, dtype=float)
        if p.size < 3 or p.sum() < 50 or p.max() <= 0:
            return nan4
        c = 0.5 * (e[:-1] + e[1:])
        plateau = float(np.median(p[p >= 0.5 * p.max()]))
        t50, t10 = 0.5 * plateau, 0.1 * plateau
        d = np.diff(np.concatenate([[0], (p >= t50).astype(np.int8), [0]]))
        starts = np.flatnonzero(d == 1)
        ends = np.flatnonzero(d == -1) - 1
        if starts.size == 0:
            return nan4
        k = int(np.argmax(ends - starts))
        a, b = int(starts[k]), int(ends[k])

        def cross(ib, ia, thr):        # thr crossing between bin ib (below) and ia (above)
            c0, c1 = p[ib], p[ia]
            if c1 == c0:
                return float(c[ia])
            return float(c[ib] + (thr - c0) * (c[ia] - c[ib]) / (c1 - c0))

        on50 = cross(a - 1, a, t50) if a > 0 else float(e[a])
        off50 = cross(b + 1, b, t50) if b + 1 < p.size else float(e[b + 1])
        lim = max(1, int(round(300.0 / float(e[1] - e[0]))))
        ps = np.convolve(p, np.ones(3) / 3.0, mode="same")
        i = a
        while i > 0 and a - i < lim and ps[i - 1] >= t10 and ps[i - 1] <= ps[i]:
            i -= 1
        j = b
        while j + 1 < p.size and j - b < lim and ps[j + 1] >= t10 and ps[j + 1] <= ps[j]:
            j += 1
        return float(e[i]), on50, off50, float(e[j + 1])

    @staticmethod
    def _median_by_group(values, groups, n: int) -> np.ndarray:
        """Median of `values` for each group 0..n-1 (NaN where a group is empty)."""
        out = np.full(n, np.nan)
        v = np.asarray(values, dtype=float)
        g = np.asarray(groups, dtype=np.int64)
        if v.size == 0:
            return out
        srt = np.argsort(g, kind="stable")
        gs, vs = g[srt], v[srt]
        b = np.searchsorted(gs, np.arange(n + 1))
        for q in range(n):
            if b[q + 1] > b[q]:
                out[q] = float(np.median(vs[b[q]:b[q + 1]]))
        return out

    @classmethod
    def _t1_auto_offset(cls, u_ns, steps_ns, tol_ns) -> float:
        """Estimate c in  u = c + tau_step  from the data alone: start inside the
        shortest-tau cluster (a low quantile), then refine with the median
        residual of the shots that snap. A few bad shots can't move it."""
        u = np.asarray(u_ns, dtype=float)
        u = u[np.isfinite(u)]
        st = np.asarray(steps_ns, dtype=float)
        fin = st[np.isfinite(st)]
        if u.size == 0 or fin.size == 0:
            return float("nan")
        if fin.size == 1:
            return float(np.median(u) - fin[0])
        q = 100.0 * min(0.5, 0.25 / fin.size)
        c = float(np.percentile(u, q) - fin.min())
        for _ in range(3):
            k, m = cls._snap_to_steps(u - c, st, tol_ns)
            if not np.any(m):
                break
            c = float(np.median(u[m] - st[k[m]]))
        return c

    @classmethod
    def _t1_slope_sign(cls, delay_ns, order_idx: Dict[str, np.ndarray], steps) -> int:
        """+1 if a single moving marker edge comes LATER for longer tau (a readout
        that moves), -1 if EARLIER (an init that moves, as in your fixed-readout
        sequences), 0 if no shot-order model explains it."""
        best = 0.0
        st = np.asarray(steps, dtype=float)
        for oi in order_idx.values():
            r = cls._pearson(delay_ns, st[oi])
            if np.isfinite(r) and abs(r) > abs(best):
                best = r
        if abs(best) < 0.5:
            return 0
        return 1 if best > 0 else -1

    @staticmethod
    def _adwin_gates_from_experiment(experiment: Any) -> Optional[Dict[str, float]]:
        """The ADwin's PROGRAMMED gates relative to its trigger, computed exactly as
        configure_gates_v5 does. None if the experiment lacks them."""
        try:
            Y = int(round(float(experiment.count_time)))
            rst = int(round(float(experiment.reset_time)))
            align = int(round(float(getattr(experiment, "gate_align_offset_ns", 0) or 0)))
            ro = int(round(float(experiment.readout_start_ns)))
        except Exception:
            return None
        sig = ro + align
        return dict(sig_start_ns=float(sig), ref_start_ns=float(sig + Y + rst),
                    count_ns=float(Y), readout_start_ns=float(ro),
                    gate_align_offset_ns=float(align))

    @staticmethod
    def _t1_adwin_totals(adwin_result: Any, n: int
                         ) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """Total ADwin counts per tau step, summed over frequencies and averages,
        from the dict run_experiment(_averaged) returns or adwin_partial_latest.npz."""
        if adwin_result is None:
            return None
        try:
            keys = list(adwin_result.keys())
        except Exception:
            return None

        def total(prefix):
            runs = [k for k in keys
                    if k.startswith(prefix + "_") and k[len(prefix) + 1:].isdigit()]
            if runs:                                    # signal_1, signal_2, ...: exact totals
                return sum(np.atleast_2d(np.asarray(adwin_result[k], dtype=float)) for k in runs)
            legacy = prefix + "_counts"
            if legacy in keys:                          # one run, or the average x averages
                avg = float(np.asarray(adwin_result["averages"])) if "averages" in keys else 1.0
                return np.atleast_2d(np.asarray(adwin_result[legacy], dtype=float)) * avg
            return None

        try:
            s, r = total("signal"), total("reference")
        except Exception:
            return None
        if s is None:
            return None
        s = s.reshape(-1, s.shape[-1]).sum(axis=0)
        if s.size != n:
            return None
        if r is not None:
            r = r.reshape(-1, r.shape[-1]).sum(axis=0)
            if r.size != n:
                r = None
        return s, r

    def _infer_t1_axis(self, experiment: Any) -> Tuple[int, np.ndarray, bool]:
        """(n_iterations, tau_values_ns, tau_is_index) for the T1 sweep. Explicit
        t1_* values win; otherwise the sweep is read from the experiment and
        converted from seconds if it plainly is in seconds."""
        n = self.t1_n_iterations
        if n is None:
            n = getattr(experiment, "number_of_iterations", 0) or 0
            if not n:
                n = len(getattr(experiment, "scan_sequences", []) or [])
            if not n and self.t1_tau_values_ns is not None:
                n = self.t1_tau_values_ns.size
        n = int(n) if n else 1
        if self.t1_tau_values_ns is not None:
            tau = self.t1_tau_values_ns
            if tau.size != n:
                print(f"[MultiHarp T1] t1_tau_values_ns has {tau.size} values but the sweep "
                      f"has {n} steps -- using step indices instead.")
                return n, np.arange(n, dtype=float), True
            return n, tau.astype(float), False
        tau = self._sweep_from_description(experiment)
        if tau is None or tau.size != n:
            print("[MultiHarp T1] could not read the tau sweep; using step indices. Pass "
                  "t1_tau_values_ns (ns) so each shot's tau can be read from the marker.")
            return n, np.arange(n, dtype=float), True
        big = float(np.nanmax(np.abs(tau))) if tau.size else 0.0
        if 0.0 < big < 1e-2:
            print("[MultiHarp T1] the sweep looks like it is in SECONDS -- converted to ns. "
                  "Pass t1_tau_values_ns explicitly to be sure.")
            tau = tau * 1e9
        return n, tau, False

    def _trigger_runs(self, trig_ps: np.ndarray):
        """Split the DO28 train into runs (one per ADwin sweep) at pauses longer
        than t1_run_gap_factor x the median trigger period. Returns
        (run_starts, run_id, local, win_ps, period_ps, wmax_ps): `local` is each
        trigger's number within its run -- with one tau per trigger, its tau step
        is local % n -- and win_ps how long after each trigger an edge still
        belongs to that shot."""
        n_shots = int(trig_ps.size)
        gaps = np.diff(trig_ps)
        period = float(np.median(gaps)) if gaps.size else float("nan")
        if self.t1_shot_window_ns is not None:
            wmax = self.t1_shot_window_ns * 1000.0
        elif np.isfinite(period) and period > 0:
            wmax = 1.5 * period
        else:
            wmax = 1e18                                  # a single shot: everything after it
        win = np.full(n_shots, wmax)
        if n_shots > 1:
            win[:-1] = np.minimum(gaps, wmax)
        if gaps.size and np.isfinite(period) and period > 0:
            brk = np.flatnonzero(gaps > self.t1_run_gap_factor * period) + 1
        else:
            brk = np.empty(0, dtype=np.int64)
        run_starts = np.concatenate([[0], brk]).astype(np.int64)
        shot_no = np.arange(n_shots, dtype=np.int64)
        run_id = np.searchsorted(run_starts, shot_no, side="right") - 1
        local = shot_no - run_starts[run_id]
        return run_starts, run_id, local, win, period, wmax

    def _t1_stream_readout(self, records, ro_abs, ro_tau, ro_order, n, tr_nb, tr_bin_ps):
        """One chunked pass over the SPCM channel (never holds every photon). A
        photon is kept ONLY if it falls within tr_nb*tr_bin after the last usable
        readout-marker edge before it; anything earlier (the init light, the dark
        time) or later (the gap to the next shot) is skipped. Kept photons are
        histogrammed vs time after their readout edge, under that shot's tau and
        under each shot-order model (for the ADwin replay)."""
        H = np.zeros(n * tr_nb, dtype=np.int64)
        Ho = {k: np.zeros(n * tr_nb, dtype=np.int64) for k in ro_order}
        prof = np.zeros(tr_nb, dtype=np.int64)
        win_ps = tr_nb * tr_bin_ps
        n_spcm = 0
        for chunk in self._iter_channel_times(records, [self.stop_input]):
            t = chunk[self.stop_input]
            n_spcm += t.size
            if t.size == 0 or ro_abs.size == 0:
                continue
            j = np.searchsorted(ro_abs, t, side="right") - 1   # last readout edge before it
            ok = j >= 0
            if not np.any(ok):
                continue
            j = j[ok]
            off = t[ok] - ro_abs[j]                             # >= 0 by construction
            keep = off < win_ps
            if not np.any(keep):
                continue
            j = j[keep]
            b = np.minimum((off[keep] // tr_bin_ps).astype(np.int64), tr_nb - 1)
            prof += np.bincount(b, minlength=tr_nb)
            H += np.bincount(ro_tau[j] * tr_nb + b, minlength=n * tr_nb)
            for k, oi in ro_order.items():
                Ho[k] += np.bincount(oi[j] * tr_nb + b, minlength=n * tr_nb)
        return dict(n_spcm=n_spcm, H=H.reshape(n, tr_nb),
                    Ho={k: v.reshape(n, tr_nb) for k, v in Ho.items()}, prof=prof)

    def analyze_t1(self, records: np.ndarray, n_iterations: int, tau_values_ns,
                   tau_is_index: bool = False, adwin_result: Any = None,
                   adwin_gates: Optional[Dict[str, float]] = None) -> T1Result:
        """T1 from raw T2 records. DO28 is a clock only; SPCM photons count only
        inside the readout light after each shot's readout marker.

          1. DO28 edges are the shots; a long DO28 pause starts a new run (sweep).
          2. Marker edges are filed into shots ([DO28_k, DO28_k + window)) and
             the layout (init+readout / readout / init) is found unless forced.
          3. The DO28 -> marker durations give each shot its tau: with an init +
             readout marker, tau = (DO28->readout) - (DO28->init) - t1_gap; with a
             single edge that moves with tau, from its DO28 delay. Each value is
             snapped to the nearest sweep step and checked against shot order.
          4. One SPCM pass: a photon is kept only if it falls in the scan span
             after the last readout-marker edge before it. The readout light pulse
             is then located (10% points, from the long-tau shots so a short-tau
             init tail can't fool it) and everything outside it is dropped.
          5. The T1 curve is summed over the signal/reference windows. The ADwin's
             programmed gates, mapped into readout-marker time, are replayed on
             the same registered photons and compared with its own counts.

        adwin_gates: dict(sig_start_ns, ref_start_ns, count_ns[, readout_start_ns,
        gate_align_offset_ns]) after the ADwin's trigger -- filled from the
        experiment by run_t1_with_experiment, as configure_gates_v5 computes them."""
        if self.trig_input is None:
            raise ValueError("case='t1_vs_spcm' needs the ADwin DO28 copy on trig_input=<ch>.")
        n = max(1, int(n_iterations))
        steps = np.asarray(tau_values_ns, dtype=float).ravel()
        if steps.size != n:
            print(f"[MultiHarp T1] {steps.size} tau values for {n} sweep steps -- using "
                  f"the step index as the x-axis.")
            steps = np.arange(n, dtype=float)
            tau_is_index = True
        rc = getattr(self, "_rabi_repeat_count", None)
        rc = int(rc) if rc else None
        trig_ch, mk_ch, sp_ch = self.trig_input, self.start_input, self.stop_input

        # ---- 1) cheap count pass ---------------------------------------------
        counts = self._channel_counts(records)
        eps = {"readout": 1.0, "init": 1.0}.get(self.t1_marker_layout, 2.0)
        self._report_channel_counts(counts, None, edges_per_shot=eps)
        n_trig_ct = int(counts.get(trig_ch, 0))
        n_mk_ct = int(counts.get(mk_ch, 0))
        res = T1Result(
            trig_input=trig_ch, marker_input=mk_ch, stop_input=sp_ch,
            n_triggers=n_trig_ct, n_marker_edges=n_mk_ct,
            n_spcm_events=int(counts.get(sp_ch, 0)), n_iterations=n, repeat_count=rc,
            base_resolution_ps=self._base_res_ps, tau_ns=steps.copy(),
            tau_is_index=bool(tau_is_index), fifo_overrun=self._fifo_overrun,
            expected_shots_per_run=(rc * n if rc else None))
        gates = dict(adwin_gates or {})
        for key, val in (("sig_start_ns", self.t1_adwin_sig_start_ns),
                         ("ref_start_ns", self.t1_adwin_ref_start_ns),
                         ("count_ns", self.t1_adwin_count_ns)):
            if val is not None:
                gates[key] = float(val)
        if n_trig_ct == 0:
            print(f"[MultiHarp T1] DO28 (input {trig_ch}) saw 0 edges -- no shots, so no tau "
                  f"and nothing to register. Check its attenuation/level/edge (~610 mV after "
                  f"10+6 dB -> level ~300 mV, rising).")
            return res
        if n_mk_ct > self.max_marker_edges and n_mk_ct > 3 * n_trig_ct:
            raise RuntimeError(
                f"[MultiHarp T1] the laser-marker input ({mk_ch}) has {n_mk_ct:,} edges for "
                f"{n_trig_ct:,} DO28 shots -- it is multi-triggering. Raise its trigger level / "
                f"fix the termination, or raise max_marker_edges.")

        # ---- 2) sparse pass: DO28 + marker edges (both ~shot-count long) ------
        sparse = self._collect_channel_times(records, [trig_ch, mk_ch])
        trig = sparse[trig_ch]
        mk = sparse[mk_ch]
        n_shots = int(trig.size)
        # runs (one per ADwin sweep, split where DO28 pauses) and shot windows
        run_starts, run_id, local, win, period, wmax = self._trigger_runs(trig)
        res.trigger_period_ns = period / 1000.0
        res.shot_window_ns = wmax / 1000.0
        res.n_runs = int(run_starts.size)
        res.shots_per_run = np.diff(np.concatenate([run_starts, [n_shots]])).astype(np.int64)
        order = {"interleaved": local % n}
        if rc:
            order["blocked"] = (local // rc) % n

        # ---- 3) marker edges -> shots (DO28 as the clock) ----------------------
        j = np.searchsorted(trig, mk, side="right") - 1
        orphan = j < 0
        jc = np.maximum(j, 0)
        inwin = (~orphan) & ((mk - trig[jc]) < win[jc])
        res.n_orphan_markers = int(np.count_nonzero(orphan))
        res.n_stray_markers = int(np.count_nonzero(~orphan & ~inwin))
        per = np.bincount(j[inwin], minlength=n_shots).astype(np.int64)
        first = np.searchsorted(mk, trig, side="left")  # first edge at/after each DO28
        res.edges_per_shot_hist = np.bincount(np.minimum(per, 5), minlength=6).astype(np.int64)
        tau_rng = 0.0 if tau_is_index else float(np.nanmax(steps) - np.nanmin(steps))
        moving = max(50.0, 0.25 * tau_rng)            # ns an edge must move by to carry tau

        # ---- which edge is the init, which is the readout --------------------
        layout = self.t1_marker_layout
        nz = per[per > 0]
        if layout == "auto":
            if nz.size == 0:
                layout = "none"
            elif int(np.argmax(np.bincount(nz))) >= 2:
                layout = "init+readout"
            else:
                layout = "single"
        expected = {"init+readout": 2, "readout": 1, "init": 1, "single": 1, "none": 0}[layout]
        good = (per == expected) if expected else np.zeros(n_shots, dtype=bool)
        gidx = np.flatnonzero(good)
        tg = trig[gidx]
        e1 = mk[first[gidx]] if expected else np.empty(0)
        e_init = e_ro = None
        if expected == 2:
            e2 = mk[first[gidx] + 1]
            sp_ns = (e2 - e1) / 1000.0
            if tau_rng > 100.0 and self._robust_span(sp_ns) < moving:
                ringing = (float(np.median(sp_ns)) < 100.0) if sp_ns.size else True
                auto = self.t1_marker_layout == "auto"
                print("[MultiHarp T1] every shot has 2 marker edges but their spacing does NOT "
                      "change with tau -- " + ("ringing / a doubled edge" if ringing
                                               else "two fixed marker pulses")
                      + ", not an init + readout pair"
                      + ((". Using the " + ("first" if ringing else "last")
                          + " edge as the readout.") if auto else " (layout forced: kept)."))
                if auto:
                    layout = "readout"
                    e_ro = e1 if ringing else e2
                else:
                    e_init, e_ro = e1, e2
            else:
                e_init, e_ro = e1, e2
        elif expected == 1:
            d1 = (e1 - tg) / 1000.0
            mov = tau_rng > 100.0 and self._robust_span(d1) > moving
            sgn = (self._t1_slope_sign(d1, {k: v[gidx] for k, v in order.items()}, steps)
                   if mov else 0)
            if layout == "single":
                if not mov:
                    layout = "readout"
                    print("[MultiHarp T1] one marker edge per shot at a FIXED time after DO28 -> "
                          "taken as the READOUT (photons are registered after it). If it marks "
                          "the init pulse instead, set t1_marker_layout='init'.")
                else:
                    layout = "init" if sgn < 0 else "readout"
                    print("[MultiHarp T1] one marker edge per shot that MOVES with tau: it comes "
                          + ("EARLIER for longer tau, so it is the INIT pulse -- tau is read from "
                             "it, but with no readout edge NO photon can be registered. Add the "
                             "readout marker line." if sgn < 0 else
                             "LATER for longer tau, so it is the READOUT -- tau from DO28 -> "
                             "readout, photons registered after it." if sgn > 0 else
                             "neither earlier nor later in any shot order -- taken as the READOUT; "
                             "set t1_marker_layout explicitly to be sure."))
            elif mov and ((layout == "readout" and sgn < 0) or (layout == "init" and sgn > 0)):
                print(f"[MultiHarp T1] WARNING: t1_marker_layout='{layout}', but this edge comes "
                      f"{'EARLIER' if sgn < 0 else 'LATER'} for longer tau, which is what "
                      f"{'an init' if sgn < 0 else 'a readout'} edge does. Check the setting.")
            if layout == "readout":
                e_ro = e1
            else:
                e_init = e1
        res.marker_layout = layout
        res.edges_expected = expected
        res.n_shots_good = int(gidx.size)
        res.n_shots_no_marker = int(np.count_nonzero(per == 0))
        res.n_shots_bad_count = (int(n_shots - res.n_shots_good - res.n_shots_no_marker)
                                 if expected else 0)
        if expected:
            hist = ", ".join(f"{'5+' if i == 5 else i}: {int(c):,}"
                             for i, c in enumerate(res.edges_per_shot_hist) if c)
            print(f"[MultiHarp T1] marker layout '{layout}' ({expected} edge(s)/shot). Edges per "
                  f"shot -> {hist}. Orphans before the 1st DO28: {res.n_orphan_markers:,}; "
                  f"stray (outside every shot): {res.n_stray_markers:,}.")
        d_ro = (e_ro - tg) / 1000.0 if e_ro is not None else None        # DO28 -> readout (ns)
        d_init = (e_init - tg) / 1000.0 if e_init is not None else None  # DO28 -> init (ns)

        # ---- tau per shot, from the DO28 -> marker durations ------------------
        tau_g = np.zeros(gidx.size, dtype=np.int64)
        m_g = np.zeros(gidx.size, dtype=bool)
        u, src = None, ""
        if gidx.size:
            if d_ro is not None and d_init is not None:
                u, src = d_ro - d_init, "marker_gap"                 # = gap + tau
            elif d_ro is not None and tau_rng > 100.0 and self._robust_span(d_ro) > moving:
                u, src = d_ro, "marker_readout_delay"                # = offset + tau
            elif d_init is not None and tau_rng > 100.0 and self._robust_span(d_init) > moving:
                u, src = -d_init, "marker_init_delay"                # = tau - offset
        if u is not None and tau_is_index:
            print("[MultiHarp T1] the DO28 -> marker durations carry each shot's tau, but the "
                  "sweep values are unknown -- pass t1_tau_values_ns (ns). Using shot order.")
            u = None
        marker_tau = False
        if u is not None:
            if self.t1_snap_tol_ns is not None:
                tol = self.t1_snap_tol_ns
            elif n > 1:
                ds = np.diff(np.sort(steps))
                ds = ds[ds > 0]
                tol = 0.5 * float(np.median(ds)) if ds.size else 1.0
            else:
                tol = float("inf")
            auto_off = self._t1_auto_offset(u, steps, tol)
            off = (self.t1_gap_ns if (src == "marker_gap" and self.t1_gap_ns is not None)
                   else auto_off)
            res.tag_offset_ns, res.tag_offset_auto_ns, res.snap_tol_ns = off, auto_off, tol
            if n == 1:
                k_g, mm_g = np.zeros(u.size, dtype=np.int64), np.isfinite(u)
            else:
                k_g, mm_g = self._snap_to_steps(u - off, steps, tol)
            n_m = int(np.count_nonzero(mm_g))
            if n_m >= max(1, int(0.02 * gidx.size)):
                tau_g, m_g, marker_tau = k_g, mm_g, True
                res.tau_assign_source = src
                resid = (u - off - steps[k_g])[mm_g]
                res.tau_residual_std_ns = float(np.std(resid)) if resid.size else float("nan")
                if (src == "marker_gap" and self.t1_gap_ns is not None and np.isfinite(auto_off)
                        and abs(auto_off - self.t1_gap_ns) > max(20.0, 0.25 * tol)):
                    print(f"[MultiHarp T1] NOTE: t1_gap_ns={self.t1_gap_ns:g} but the data say "
                          f"{auto_off:.1f} ns -- check the marker lines against the pulse lines.")
            else:
                um = (u - off)[np.isfinite(u)]
                print(f"[MultiHarp T1] tau could NOT be read from the DO28 -> marker durations: "
                      f"{n_m:,}/{gidx.size:,} shots snapped. Measured "
                      + (f"{np.min(um):.0f}..{np.max(um):.0f} ns" if um.size else "nothing")
                      + f" vs a sweep of {np.min(steps):.0f}..{np.max(steps):.0f} ns "
                        f"(tol +/-{tol:.0f}). Check t1_tau_values_ns (units, order) and "
                        f"t1_gap_ns. Falling back to shot order.")
        if not marker_tau:
            model = self.t1_order
            if model not in order:
                print("[MultiHarp T1] t1_order='blocked' needs repeat_count -- using interleaved.")
                model = "interleaved"
            tau_g = order[model][gidx].astype(np.int64)
            m_g = np.ones(gidx.size, dtype=bool)
            res.tau_assign_source = f"shot_order_{model}"
        res.n_matched = int(np.count_nonzero(m_g))
        res.n_unmatched = int(n_shots - res.n_matched)

        if marker_tau:                                 # what order does the Proteus really play?
            tau_full = np.full(n_shots, -1, dtype=np.int64)
            tau_full[gidx[m_g]] = tau_g[m_g]
            mm = tau_full >= 0
            res.agree_interleaved = float(np.mean(order["interleaved"][mm] == tau_full[mm]))
            if "blocked" in order:
                res.agree_blocked = float(np.mean(order["blocked"][mm] == tau_full[mm]))
            if n > 1:
                dd = (tau_full - order["interleaved"]) % n
                tab = np.bincount(run_id[mm] * n + dd[mm],
                                  minlength=res.n_runs * n).reshape(res.n_runs, n)
                offs = np.where(tab.sum(axis=1) > 0, tab.argmax(axis=1), -1).astype(np.int64)
                res.run_pointer_offsets = offs
                res.agree_interleaved_shifted = float(np.mean(dd[mm] == offs[run_id[mm]]))
            else:
                res.run_pointer_offsets = np.zeros(res.n_runs, dtype=np.int64)
            K = int(min(res.shots_per_run[0] if res.shots_per_run.size else 0,
                        max(3 * n, 60), 2000))
            res.order_sample_measured = tau_full[:K].copy()
            res.order_sample_interleaved = order["interleaved"][:K].astype(np.int64)
            if "blocked" in order:
                res.order_sample_blocked = order["blocked"][:K].astype(np.int64)

        # ---- DO28 -> marker timing per tau (fixed or moving readout?) ---------
        if d_ro is not None and np.any(m_g):
            gsel, v = tau_g[m_g], d_ro[m_g]
            med = self._median_by_group(v, gsel, n)
            res.ro_delay_by_tau_ns = med
            resid = v - med[gsel]
            res.ro_jitter_ns = float(np.std(resid))
            res.ro_delay_span_ns = self._robust_span(resid)
            res.ro_delay_median_ns = float(np.median(v))
            pr = np.percentile(resid, [0.1, 99.9])
            lim = max(10.0, min(5000.0, 1.2 * float(np.max(np.abs(pr)))))
            res.ro_delay_hist_counts, res.ro_delay_hist_edges_ns = np.histogram(
                resid, bins=200, range=(-lim, lim))
            fin = med[np.isfinite(med)]
            res.readout_moves = bool(tau_rng > 100.0 and fin.size >= 2
                                     and float(fin.max() - fin.min()) > moving)
        if d_init is not None and np.any(m_g):
            res.init_delay_by_tau_ns = self._median_by_group(d_init[m_g], tau_g[m_g], n)

        # ---- 4) SPCM: registered ONLY after a readout-marker edge -------------
        if e_ro is None:
            res.ro_window_source = "none"
            print("[MultiHarp T1] no readout-marker edge -> NO SPCM photon is registered (by "
                  "design nothing is counted from DO28). Add `marker, laser_readout_1 on "
                  "channel 4 at <readout>ns, <len>ns` to the sequence.")
            self._t1_verdicts(res)
            return res
        usable = m_g                                   # good shot, tau known, readout edge
        ro_abs = e_ro[usable]                          # time-ordered: one per shot, in shot order
        n_use = int(ro_abs.size)
        if n_use == 0:
            res.ro_window_source = "none"
            print("[MultiHarp T1] no shot has both a readout edge and a tau -- nothing registered.")
            self._t1_verdicts(res)
            return res
        ro_tau = tau_g[usable]
        ro_order = {k: v[gidx[usable]].astype(np.int64) for k, v in order.items()}
        span = self.t1_tr_window_ns
        if self.t1_ro_window_ns is not None:
            span = max(span, float(self.t1_ro_window_ns[1]))
        if n_use > 1:
            spacing = float(np.median(np.diff(ro_abs))) / 1000.0
            if span >= 0.9 * spacing:
                span = 0.9 * spacing
                print(f"[MultiHarp T1] readout scan span clipped to {span:.0f} ns (readouts are "
                      f"{spacing:.0f} ns apart).")
        bin_ns = self.t1_tr_bin_ns
        tr_nb = max(1, int(np.ceil(span / bin_ns)))
        tr_edges = np.arange(tr_nb + 1, dtype=float) * bin_ns
        st = self._t1_stream_readout(records, ro_abs, ro_tau, ro_order, n, tr_nb, bin_ns * 1000.0)
        res.n_spcm_events = int(st["n_spcm"])
        H, Ho, prof = st["H"], st["Ho"], st["prof"]

        # where is the readout light? judged on the long-tau shots, whose init light
        # is far before the readout, so a short-tau init tail can't pass for it
        rows = (steps >= np.median(steps)) if (not tau_is_index and n >= 2) else np.ones(n, bool)
        on10, on50, off50, off10 = self._light_pulse(tr_edges, H[rows].sum(axis=0))
        res.light_on10_ns, res.light_on_ns = on10, on50
        res.light_off_ns, res.light_off10_ns = off50, off10
        if self.t1_ro_window_ns is not None:
            lo, hi = (float(v) for v in self.t1_ro_window_ns)
            if lo < 0.0:
                print("[MultiHarp T1] t1_ro_window_ns starts before the readout marker -- "
                      "clipped to 0 (nothing before the readout marker is ever registered).")
                lo = 0.0
            res.ro_window_source = "given"
        elif np.isfinite(on10) and np.isfinite(off10):
            lo, hi = on10, off10
            res.ro_window_source = "auto (readout light)"
        else:
            lo, hi = 0.0, float(tr_edges[-1])
            res.ro_window_source = "fallback (no clear readout light)"
            print("[MultiHarp T1] could not locate the readout light -- registering the whole "
                  "scan span after the readout marker. Set t1_ro_window_ns to be strict.")
        i0 = int(np.clip(np.floor(lo / bin_ns + 1e-9), 0, tr_nb))
        i1 = int(np.clip(np.ceil(min(hi, tr_edges[-1]) / bin_ns - 1e-9), i0, tr_nb))
        if i1 <= i0:
            i0, i1 = 0, tr_nb
        res.ro_window_ns = (float(tr_edges[i0]), float(tr_edges[i1]))
        ec = tr_edges[i0:i1 + 1]
        Hc = H[:, i0:i1]
        Hoc = {k: v[:, i0:i1] for k, v in Ho.items()}
        res.arrival_hist, res.arrival_edges_ns = Hc, ec
        res.arrival_profile_ns = prof[i0:i1]
        res.n_spcm_registered = int(prof[i0:i1].sum())
        res.shots_per_tau = np.bincount(ro_tau, minlength=n).astype(np.int64)
        if np.isfinite(on50) and np.isfinite(res.ro_delay_median_ns) and not res.readout_moves:
            res.light_on_after_trig_ns = res.ro_delay_median_ns + on50
        print(f"[MultiHarp T1] SPCM registered only in [{res.ro_window_ns[0]:.0f}, "
              f"{res.ro_window_ns[1]:.0f}) ns after each readout marker "
              f"({res.ro_window_source}): {res.n_spcm_registered:,} of {res.n_spcm_events:,} "
              f"SPCM events; nothing between DO28 and the readout marker is counted.")

        # ---- 5) the T1 curve ---------------------------------------------------
        wlo, whi = res.ro_window_ns

        def inside(w):
            if w is None:
                return None
            a, b = max(float(w[0]), wlo), min(float(w[1]), whi)
            return (a, b) if b > a else None

        sig_w = inside(self.t1_sig_window_ns)
        if sig_w is not None:
            res.window_source = "given"
        else:
            if self.t1_sig_window_ns is not None:
                print("[MultiHarp T1] t1_sig_window_ns lies outside the registered readout "
                      "light -- using the auto window.")
            if np.isfinite(on50):
                sig_w, res.window_source = inside((on50, on50 + self.t1_count_ns)), "auto"
            if sig_w is None:
                sig_w, res.window_source = (wlo, min(whi, wlo + self.t1_count_ns)), "fallback"
        ref_w = inside(self.t1_ref_window_ns)
        if ref_w is None and self.t1_ref_window_ns is None and np.isfinite(off50):
            rh = off50 - self.t1_ref_margin_ns
            cand = inside((rh - self.t1_count_ns, rh))
            ref_w = cand if (cand is not None and cand[0] >= sig_w[1]) else None
        cur = rabi_curve_from_histogram(Hc, ec, res.shots_per_tau, sig_w, ref_w)
        spt = res.shots_per_tau.astype(float)
        res.sig_window_ns = tuple(float(v) for v in cur["sig_window_used_ns"])
        res.signal_mean = cur["signal_mean"]
        with np.errstate(divide="ignore", invalid="ignore"):
            res.signal_sem = np.where(spt > 0, np.sqrt(cur["signal_sum"]) / spt, np.nan)
        if ref_w is not None:
            res.ref_window_ns = tuple(float(v) for v in cur["ref_window_used_ns"])
            res.reference_mean, res.contrast = cur["reference_mean"], cur["contrast"]

        # ---- 6) the ADwin's programmed gates, on the same registered photons ---
        if all(k in gates for k in ("sig_start_ns", "ref_start_ns", "count_ns")):
            s0, r0, w = gates["sig_start_ns"], gates["ref_start_ns"], gates["count_ns"]
            res.adwin_sig_start_ns, res.adwin_ref_start_ns, res.adwin_count_ns = s0, r0, w
            res.adwin_readout_start_ns = float(gates.get("readout_start_ns", np.nan))
            res.adwin_gate_align_offset_ns = float(gates.get("gate_align_offset_ns", np.nan))
            if np.isfinite(res.adwin_readout_start_ns) and np.isfinite(res.light_on_after_trig_ns):
                res.suggested_gate_align_offset_ns = (res.light_on_after_trig_ns
                                                      - res.adwin_readout_start_ns)
            if res.readout_moves:
                res.adwin_replay_status = ("not replayed: the readout moves with tau but the "
                                           "ADwin's gates sit at fixed times after its trigger.")
            elif not np.isfinite(res.ro_delay_median_ns) or w <= 0:
                res.adwin_replay_status = "not replayed: no readout-marker timing."
            else:
                sr, rr = s0 - res.ro_delay_median_ns, r0 - res.ro_delay_median_ns
                res.adwin_sig_rel_ns, res.adwin_ref_rel_ns = sr, rr

                def frac(g):
                    return max(0.0, min(whi, g + w) - max(wlo, g)) / w

                fs, fr = frac(sr), frac(rr)
                if fs >= 0.9 and fr >= 0.9:
                    mats = dict({"tau": Hc} if marker_tau else {})
                    mats.update(Hoc)
                    one = np.ones(n)
                    for nm, Hm in mats.items():
                        res.adwin_replay_sig[nm] = rabi_curve_from_histogram(
                            Hm, ec, one, (max(wlo, sr), min(whi, sr + w)))["signal_sum"].astype(np.int64)
                        res.adwin_replay_ref[nm] = rabi_curve_from_histogram(
                            Hm, ec, one, (max(wlo, rr), min(whi, rr + w)))["signal_sum"].astype(np.int64)
                    res.adwin_replay_status = (
                        "replayed on the registered readout photons"
                        + ("" if min(fs, fr) >= 0.999 else
                           f" (gates clipped to the readout light, {100 * min(fs, fr):.0f}% inside)"))
                    tot = self._t1_adwin_totals(adwin_result, n)
                    if tot is not None:
                        a_sig, a_ref = tot
                        res.adwin_signal = a_sig
                        res.adwin_reference = a_ref if a_ref is not None else np.empty(0)
                        scale = n_shots / max(1, n_use)    # the ADwin counted every shot
                        for nm, e in res.adwin_replay_sig.items():
                            ee = e.astype(float) * scale
                            res.adwin_corr[nm] = self._pearson(a_sig, ee)
                            res.adwin_count_ratio[nm] = (float(a_sig.sum() / ee.sum())
                                                         if ee.sum() > 0 else float("nan"))
                    elif adwin_result is not None:
                        print(f"[MultiHarp T1] ADwin result has no per-tau signal arrays with "
                              f"{n} steps -- skipped the ADwin cross-check.")
                else:
                    where = []
                    for nm, g, fx in (("signal", sr, fs), ("reference", rr, fr)):
                        if fx >= 0.9:
                            continue
                        if g + w <= wlo:
                            where.append(f"{nm} gate ends {wlo - (g + w):.0f} ns BEFORE the "
                                         f"readout light starts")
                        elif g >= whi:
                            where.append(f"{nm} gate opens {g - whi:.0f} ns AFTER the readout "
                                         f"light ends")
                        else:
                            where.append(f"{nm} gate is only {100 * fx:.0f}% inside the "
                                         f"readout light")
                    res.adwin_replay_status = (
                        "not replayed: as programmed, the " + "; the ".join(where)
                        + " -- so it would not be counting readout photons there. If the ADwin's "
                          "own T1 looks fine, its REAL gate is offset from the programmed one "
                          "(measure it with case='adwin_readout_vs_spcm').")
        self._t1_verdicts(res)
        return res

    @staticmethod
    def _t1_verdicts(res: T1Result) -> None:
        """Plain-language reading of the order and ADwin cross-checks."""
        if res.tau_assign_source.startswith("marker"):
            ai, ash, ab = res.agree_interleaved, res.agree_interleaved_shifted, res.agree_blocked
            if np.isfinite(ai) and ai > 0.95:
                print("[MultiHarp T1] VERDICT order: the Proteus advances ONE tau step PER "
                      "TRIGGER (interleaved) and every run starts at step 0.")
            elif np.isfinite(ash) and ash > 0.95:
                print("[MultiHarp T1] VERDICT order: interleaved, but in some runs the task "
                      "pointer does NOT start at step 0 (run_pointer_offsets). Anything binned "
                      "by shot order -- the ADwin's Data arrays -- is shifted there.")
            elif np.isfinite(ab) and ab > 0.95:
                print("[MultiHarp T1] VERDICT order: BLOCKED (tau changes every repeat_count "
                      "shots).")
            else:
                print("[MultiHarp T1] VERDICT order: neither interleaved nor blocked explains "
                      "the measured tau sequence (dropped triggers or a jumping task pointer) "
                      "-- see the timing plot.")
        if res.adwin_corr:
            r = {k: (v if np.isfinite(v) else -1.0) for k, v in res.adwin_corr.items()}
            best = max(r, key=r.get)
            if r[best] < 0.5:
                print(f"[MultiHarp T1] VERDICT ADwin: its counts do not follow the replay of "
                      f"its programmed gates under any binning (best r={r[best]:.2f}). The real "
                      f"gate is probably not where it is programmed -- measure it with "
                      f"case='adwin_readout_vs_spcm' (DO16 copy).")
            elif "tau" in r and r["tau"] >= r[best] - 0.02:
                print(f"[MultiHarp T1] VERDICT ADwin: its counts follow the MEASURED tau "
                      f"(r={r['tau']:.3f}) -> every Data_1/Data_2 bin holds the shots it should.")
            elif "tau" in r:
                print(f"[MultiHarp T1] VERDICT ADwin: its counts follow shot order '{best}' "
                      f"(r={r[best]:.3f}) better than the measured tau (r={r['tau']:.3f}) -> "
                      f"the ADwin bins shots differently from what the Proteus played; its T1 "
                      f"curve is MIS-LINKED.")
            else:
                print(f"[MultiHarp T1] ADwin counts best match shot order '{best}' "
                      f"(r={r[best]:.3f}); tau was not read per shot, so this can't confirm "
                      f"the linking -- add the init marker line for that.")
            ratio = res.adwin_count_ratio.get(best, float("nan"))
            if np.isfinite(ratio) and not (0.8 <= ratio <= 1.25):
                print(f"    counts ratio ADwin/replay = {ratio:.2f}: the ADwin does not count the "
                      f"same time slice as its programmed gate (gate offset or counter "
                      f"efficiency). The shape comparison above still holds.")
        elif res.adwin_replay_status.startswith("not replayed"):
            print(f"[MultiHarp T1] ADwin cross-check {res.adwin_replay_status}")

    # ---- T1 plots ---------------------------------------------------------- #
    @staticmethod
    def _t1_x(res: T1Result) -> Tuple[np.ndarray, str]:
        x = np.asarray(res.tau_ns, dtype=float)
        if res.tau_is_index:
            return x, "tau step index"
        if x.size and np.nanmax(np.abs(x)) >= 1e4:
            return x / 1000.0, "dark time tau (us)"
        return x, "dark time tau (ns)"

    @staticmethod
    def _edges_from_centers(c) -> np.ndarray:
        c = np.asarray(c, dtype=float)
        if c.size == 1:
            return np.array([c[0] - 0.5, c[0] + 0.5])
        mid = 0.5 * (c[:-1] + c[1:])
        return np.concatenate([[2 * c[0] - mid[0]], mid, [2 * c[-1] - mid[-1]]])

    _T1_BINNING_LABEL = {"tau": "measured tau", "interleaved": "shot order, interleaved",
                         "blocked": "shot order, blocked"}

    def _plot_t1_curve(self, res: T1Result):
        x, xl = self._t1_x(res)
        n = res.n_iterations
        has_mh = res.signal_mean.size == n and np.isfinite(res.signal_mean).any()
        has_c = res.contrast.size == n and np.isfinite(res.contrast).any()
        key = ("tau" if "tau" in res.adwin_replay_sig
               else (next(iter(res.adwin_replay_sig)) if res.adwin_replay_sig else None))
        has_adw = res.adwin_signal.size == n and bool(res.adwin_corr)
        rows = [r for r, on in (("counts", has_mh), ("contrast", has_c or key is not None),
                                ("adwin", has_adw)) if on]
        if not rows:
            return None
        fig, axes = plt.subplots(len(rows), 1, sharex=True,
                                 figsize=(8.5, 2.8 * len(rows) + 0.8))
        axes = np.atleast_1d(axes)
        spt = res.shots_per_tau.astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            for ax, row in zip(axes, rows):
                if row == "counts":
                    m = np.isfinite(res.signal_mean)
                    ax.errorbar(x[m], res.signal_mean[m], yerr=res.signal_sem[m], fmt="-o",
                                ms=3.5, lw=1.2, capsize=2,
                                label=f"signal [{res.sig_window_ns[0]:.0f}, "
                                      f"{res.sig_window_ns[1]:.0f}] ns")
                    if res.reference_mean.size == n and np.isfinite(res.reference_mean).any():
                        ax.plot(x, res.reference_mean, "-s", ms=3, lw=1, alpha=0.8,
                                label=f"reference [{res.ref_window_ns[0]:.0f}, "
                                      f"{res.ref_window_ns[1]:.0f}] ns")
                    if key is not None and spt.size == n:
                        ax.plot(x, np.where(spt > 0, res.adwin_replay_sig[key] / spt, np.nan),
                                "--^", ms=3, lw=1, label="ADwin signal gate replayed")
                    ax.set_ylabel("SPCM counts / shot")
                    ax.set_title(f"MultiHarp T1 | readout photons only, {res.ro_window_ns[0]:.0f}-"
                                 f"{res.ro_window_ns[1]:.0f} ns after each readout marker | tau: "
                                 f"{res.tau_assign_source}", fontsize=9)
                elif row == "contrast":
                    if has_c:
                        ax.plot(x, res.contrast, "-o", ms=3.5, lw=1.2,
                                label="MultiHarp signal / reference")
                    if key is not None:
                        s = res.adwin_replay_sig[key].astype(float)
                        r = res.adwin_replay_ref[key].astype(float)
                        ax.plot(x, np.where(r > 0, s / r, np.nan), "--^", ms=3, lw=1,
                                label="ADwin gates replayed: signal / reference")
                    ax.set_ylabel("contrast")
                else:
                    a = res.adwin_signal.astype(float)
                    ax.plot(x, a / np.nanmean(a), "-o", color="k", ms=3.5, lw=1.5,
                            label="ADwin Data (signal)")
                    for k, v in res.adwin_replay_sig.items():
                        e = v.astype(float)
                        if e.sum() > 0:
                            ax.plot(x, e / np.nanmean(e), "--", lw=1.1,
                                    label=f"replay by {self._T1_BINNING_LABEL.get(k, k)} "
                                          f"(r={res.adwin_corr.get(k, float('nan')):.3f})")
                    ax.set_ylabel("signal / mean")
                    ax.set_title("ADwin counts vs its programmed gates replayed on the same "
                                 "readout photons", fontsize=9)
                ax.legend(fontsize=7, loc="best")
        axes[-1].set_xlabel(xl)
        fig.tight_layout()
        return fig

    def _plot_t1_timing(self, res: T1Result):
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.3))
        ax = axes[0]
        if res.ro_delay_hist_counts.size:
            e = res.ro_delay_hist_edges_ns
            ax.bar(0.5 * (e[:-1] + e[1:]), res.ro_delay_hist_counts, width=e[1] - e[0])
            ax.set_xlabel("DO28 -> readout marker, minus that tau's median (ns)")
            ax.set_ylabel("shots")
            ax.set_title(f"readout after DO28: {res.ro_delay_median_ns / 1000.0:.4f} us"
                         + (" (moves with tau)" if res.readout_moves else "")
                         + f"\njitter std {res.ro_jitter_ns:.2f} ns, span "
                           f"{res.ro_delay_span_ns:.1f} ns", fontsize=9)
        else:
            ax.text(0.5, 0.5, "no per-shot readout edge", ha="center", transform=ax.transAxes)
        ax = axes[1]
        x, xl = self._t1_x(res)
        n = res.n_iterations
        drew = False
        if res.init_delay_by_tau_ns.size == n and np.isfinite(res.init_delay_by_tau_ns).any():
            ax.plot(x, res.init_delay_by_tau_ns / 1000.0, "o", ms=3.5, label="DO28 -> init marker")
            drew = True
        if res.ro_delay_by_tau_ns.size == n and np.isfinite(res.ro_delay_by_tau_ns).any():
            ax.plot(x, res.ro_delay_by_tau_ns / 1000.0, "s", ms=3.5, label="DO28 -> readout marker")
            if res.tau_assign_source == "marker_gap" and np.isfinite(res.tag_offset_ns):
                ax.plot(x, (res.ro_delay_by_tau_ns - res.tag_offset_ns - res.tau_ns) / 1000.0,
                        "-", lw=1, label="readout - gap - tau (expected init)")
            drew = True
        if drew:
            ax.set_xlabel(xl)
            ax.set_ylabel("DO28 -> marker (us)")
            ax.legend(fontsize=7)
            ax.set_title(f"the DO28 -> marker durations that give tau\nresidual std "
                         f"{res.tau_residual_std_ns:.2f} ns", fontsize=9)
        else:
            ax.text(0.5, 0.5, "no marker timing", ha="center", transform=ax.transAxes)
        ax = axes[2]
        if res.order_sample_measured.size:
            s = np.arange(res.order_sample_measured.size)
            m = res.order_sample_measured >= 0
            ax.plot(s, res.order_sample_interleaved, "-", lw=1, label="interleaved model")
            if res.order_sample_blocked.size:
                ax.plot(s, res.order_sample_blocked, "-", lw=1, label="blocked model")
            ax.plot(s[m], res.order_sample_measured[m], "o", ms=3, label="measured from marker")
            ax.set_xlabel("shot number in run 1")
            ax.set_ylabel("tau step index")
            ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
            ai, ash, ab = res.agree_interleaved, res.agree_interleaved_shifted, res.agree_blocked
            ax.set_title(f"which order does the Proteus play?\ninterleaved {100 * ai:.1f}%"
                         + (f" ({100 * ash:.1f}% with a per-run pointer offset)"
                            if np.isfinite(ash) and ash - ai > 0.005 else "")
                         + (f", blocked {100 * ab:.1f}%" if np.isfinite(ab) else ""), fontsize=9)
        else:
            ax.text(0.5, 0.5, "tau not read per shot", ha="center", transform=ax.transAxes)
        fig.tight_layout()
        return fig

    def _plot_t1_readout_map(self, res: T1Result):
        if res.arrival_hist.size == 0 or res.shots_per_tau.size != res.n_iterations:
            return None
        spt = res.shots_per_tau.astype(float)
        spt[spt == 0] = np.nan
        Z = res.arrival_hist / spt[:, None]
        xe = res.arrival_edges_ns
        y, yl = self._t1_x(res)
        fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, figsize=(9.5, 6.8),
                                       gridspec_kw=dict(height_ratios=[1.0, 2.2]),
                                       constrained_layout=True)
        ax0.plot(0.5 * (xe[:-1] + xe[1:]), res.arrival_profile_ns, lw=1)
        ax0.set_ylabel("photons, all shots")
        pc = ax1.pcolormesh(xe, self._edges_from_centers(y), Z, shading="flat", cmap="viridis")
        fig.colorbar(pc, ax=[ax0, ax1]).set_label("SPCM counts / shot / bin")
        for ax in (ax0, ax1):
            if np.isfinite(res.sig_window_ns[0]):
                ax.axvspan(*res.sig_window_ns, color="w" if ax is ax1 else "C2", alpha=0.25)
            if np.isfinite(res.ref_window_ns[0]):
                ax.axvspan(*res.ref_window_ns, color="orange", alpha=0.25)
            for t in (res.light_on_ns, res.light_off_ns):
                if np.isfinite(t):
                    ax.axvline(t, color="grey", ls=":", lw=1)
        note = ""
        if res.adwin_replay_sig:
            for g, col in ((res.adwin_sig_rel_ns, "r"), (res.adwin_ref_rel_ns, "m")):
                for ax in (ax0, ax1):
                    ax.axvspan(max(g, xe[0]), min(g + res.adwin_count_ns, xe[-1]),
                               color=col, alpha=0.35)
            note = "; red/magenta: ADwin gates as programmed"
        elif np.isfinite(res.adwin_sig_rel_ns):
            note = (f"; ADwin gates as programmed sit at {res.adwin_sig_rel_ns:.0f} / "
                    f"{res.adwin_ref_rel_ns:.0f} ns (outside the readout light)")
        ax1.set_xlabel("time after THIS shot's readout marker (ns)")
        ax1.set_ylabel(yl)
        ax0.set_title(f"SPCM registered ONLY here: {res.ro_window_ns[0]:.0f}-"
                      f"{res.ro_window_ns[1]:.0f} ns after each readout marker "
                      f"({res.ro_window_source}); nothing between DO28 and the readout marker\n"
                      f"shaded: MultiHarp signal / reference windows; dotted: light on/off (50%)"
                      + note, fontsize=8)
        return fig

    def _save_t1_outputs(self, res: T1Result) -> None:
        outdir = self.output_dir or os.getcwd()
        os.makedirs(outdir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        base = os.path.join(outdir, f"multiharp_t1_{stamp}")
        nan = float("nan")
        if self.save_npz:
            flat = {}
            for nm in ("tau", "interleaved", "blocked"):
                if nm in res.adwin_replay_sig:
                    flat[f"adwin_replay_sig_{nm}"] = res.adwin_replay_sig[nm]
                    flat[f"adwin_replay_ref_{nm}"] = res.adwin_replay_ref[nm]
                if nm in res.adwin_corr:
                    flat[f"adwin_corr_{nm}"] = res.adwin_corr[nm]
                    flat[f"adwin_count_ratio_{nm}"] = res.adwin_count_ratio.get(nm, nan)
            npz = base + ".npz"
            np.savez_compressed(
                npz, case=self.case,
                trig_input=res.trig_input, marker_input=res.marker_input,
                stop_input=res.stop_input, base_resolution_ps=res.base_resolution_ps,
                n_triggers=res.n_triggers, n_marker_edges=res.n_marker_edges,
                n_spcm_events=res.n_spcm_events, n_spcm_registered=res.n_spcm_registered,
                n_iterations=res.n_iterations, repeat_count=(res.repeat_count or 0),
                marker_layout=res.marker_layout, edges_expected=res.edges_expected,
                n_shots_good=res.n_shots_good, n_shots_no_marker=res.n_shots_no_marker,
                n_shots_bad_count=res.n_shots_bad_count,
                edges_per_shot_hist=res.edges_per_shot_hist,
                n_orphan_markers=res.n_orphan_markers, n_stray_markers=res.n_stray_markers,
                trigger_period_ns=res.trigger_period_ns, shot_window_ns=res.shot_window_ns,
                n_runs=res.n_runs, shots_per_run=res.shots_per_run,
                expected_shots_per_run=(res.expected_shots_per_run or 0),
                ro_delay_median_ns=res.ro_delay_median_ns, ro_jitter_ns=res.ro_jitter_ns,
                ro_delay_span_ns=res.ro_delay_span_ns,
                ro_delay_hist_counts=res.ro_delay_hist_counts,
                ro_delay_hist_edges_ns=res.ro_delay_hist_edges_ns,
                readout_moves=int(res.readout_moves),
                ro_delay_by_tau_ns=res.ro_delay_by_tau_ns,
                init_delay_by_tau_ns=res.init_delay_by_tau_ns,
                tag_offset_ns=res.tag_offset_ns, tag_offset_auto_ns=res.tag_offset_auto_ns,
                tau_residual_std_ns=res.tau_residual_std_ns,
                tau_assign_source=res.tau_assign_source, n_matched=res.n_matched,
                n_unmatched=res.n_unmatched, snap_tol_ns=res.snap_tol_ns,
                agree_interleaved=res.agree_interleaved,
                agree_interleaved_shifted=res.agree_interleaved_shifted,
                agree_blocked=res.agree_blocked, run_pointer_offsets=res.run_pointer_offsets,
                ro_window_ns=np.asarray(res.ro_window_ns, dtype=float),
                ro_window_source=res.ro_window_source,
                light_on_ns=res.light_on_ns, light_off_ns=res.light_off_ns,
                light_on10_ns=res.light_on10_ns, light_off10_ns=res.light_off10_ns,
                tau_ns=res.tau_ns, tau_is_index=int(res.tau_is_index),
                sig_window_ns=np.asarray(res.sig_window_ns, dtype=float),
                ref_window_ns=np.asarray(res.ref_window_ns, dtype=float),
                window_source=res.window_source,
                signal_mean=res.signal_mean, signal_sem=res.signal_sem,
                reference_mean=res.reference_mean, contrast=res.contrast,
                shots_per_tau=res.shots_per_tau,
                # same keys as the Rabi .npz, so recompute_rabi() re-windows it offline
                # (within the registered readout window)
                arrival_hist=res.arrival_hist, arrival_edges_ns=res.arrival_edges_ns,
                arrival_profile_ns=res.arrival_profile_ns,
                adwin_sig_start_ns=res.adwin_sig_start_ns,
                adwin_ref_start_ns=res.adwin_ref_start_ns, adwin_count_ns=res.adwin_count_ns,
                adwin_readout_start_ns=res.adwin_readout_start_ns,
                adwin_gate_align_offset_ns=res.adwin_gate_align_offset_ns,
                adwin_sig_rel_ns=res.adwin_sig_rel_ns, adwin_ref_rel_ns=res.adwin_ref_rel_ns,
                adwin_replay_status=res.adwin_replay_status,
                adwin_signal=res.adwin_signal, adwin_reference=res.adwin_reference,
                light_on_after_trig_ns=res.light_on_after_trig_ns,
                suggested_gate_align_offset_ns=res.suggested_gate_align_offset_ns,
                fifo_overrun=int(res.fifo_overrun), **flat)
            res.saved_files.append(npz)
        if self.save_hdf5:
            try:
                import h5py  # noqa: F401
                from src.core.struct_hdf5 import MyStruct, save_data
                s = MyStruct()
                s.data = MyStruct(tau_ns=res.tau_ns, signal_mean=res.signal_mean,
                                  signal_sem=res.signal_sem, reference_mean=res.reference_mean,
                                  contrast=res.contrast, shots_per_tau=res.shots_per_tau,
                                  arrival_hist=res.arrival_hist,
                                  arrival_edges_ns=res.arrival_edges_ns,
                                  ro_delay_by_tau_ns=res.ro_delay_by_tau_ns,
                                  init_delay_by_tau_ns=res.init_delay_by_tau_ns,
                                  adwin_signal=res.adwin_signal,
                                  adwin_replay_sig_tau=res.adwin_replay_sig.get("tau", np.empty(0)),
                                  adwin_replay_ref_tau=res.adwin_replay_ref.get("tau", np.empty(0)))
                s.meta = MyStruct(case=self.case, marker_layout=res.marker_layout,
                                  trig_input=res.trig_input, marker_input=res.marker_input,
                                  stop_input=res.stop_input,
                                  base_resolution_ps=res.base_resolution_ps,
                                  n_triggers=res.n_triggers, n_runs=res.n_runs,
                                  n_shots_good=res.n_shots_good, n_matched=res.n_matched,
                                  n_spcm_events=res.n_spcm_events,
                                  n_spcm_registered=res.n_spcm_registered,
                                  ro_window_lo_ns=res.ro_window_ns[0],
                                  ro_window_hi_ns=res.ro_window_ns[1],
                                  ro_window_source=res.ro_window_source,
                                  tau_assign_source=res.tau_assign_source,
                                  agree_interleaved=res.agree_interleaved,
                                  agree_blocked=res.agree_blocked,
                                  ro_delay_median_ns=res.ro_delay_median_ns,
                                  ro_jitter_ns=res.ro_jitter_ns,
                                  readout_moves=int(res.readout_moves),
                                  tag_offset_ns=res.tag_offset_ns,
                                  sig_lo_ns=res.sig_window_ns[0], sig_hi_ns=res.sig_window_ns[1],
                                  ref_lo_ns=res.ref_window_ns[0], ref_hi_ns=res.ref_window_ns[1],
                                  light_on_ns=res.light_on_ns,
                                  adwin_sig_start_ns=res.adwin_sig_start_ns,
                                  adwin_ref_start_ns=res.adwin_ref_start_ns,
                                  adwin_replay_status=res.adwin_replay_status,
                                  suggested_gate_align_offset_ns=res.suggested_gate_align_offset_ns,
                                  fifo_overrun=int(res.fifo_overrun),
                                  timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
                save_data(base + ".h5", s)
                res.saved_files.append(base + ".h5")
            except Exception as e:
                print(f"[MultiHarp T1] HDF5 save skipped ({e}); the .npz has everything.")
        if self.make_plots:
            for suffix, maker in (("", self._plot_t1_curve), ("_timing", self._plot_t1_timing),
                                  ("_readout_map", self._plot_t1_readout_map)):
                try:
                    fig = maker(res)
                    if fig is not None:
                        png = base + suffix + ".png"
                        fig.savefig(png, dpi=130)
                        plt.close(fig)
                        res.saved_files.append(png)
                except Exception as e:  # never let plotting kill a good measurement
                    print(f"[MultiHarp T1] plot warning ({suffix or '_curve'}): {e}")

    def run_t1_with_experiment(
        self,
        experiment: Any,
        frequency_range: Optional[List[float]] = None,
        save: bool = True,
        run_method: str = "run_experiment",
    ) -> Dict[str, Any]:
        """Run the T1 sweep UNCHANGED while the MultiHarp tags DO28, the laser
        marker and the SPCM, then analyze: DO28 times the marker edges (tau per
        shot), and only the SPCM photons inside the readout light after each
        shot's readout marker are registered.

        Wire trig_input to the attenuated DO28 copy, start_input to the attenuated
        Proteus ch4 MKR, and give the sequence two marker lines that copy the init
        and readout pulses. The ADwin gates are read from the experiment
        (readout_start_ns, gate_align_offset_ns, count_time, reset_time -- exactly
        as configure_gates_v5 uses them) for the cross-check.
        Returns {'odmr': ..., 't1': T1Result, 'raw_path': ...}."""
        if self.trig_input is None:
            raise ValueError("case='t1_vs_spcm' needs trig_input=<input carrying DO28>.")
        if frequency_range is None:
            frequency_range = experiment.settings['microwave']['frequency range']
        if not hasattr(experiment, run_method):
            raise AttributeError(f"experiment has no method {run_method!r}")
        if self.output_dir is None:
            self.output_dir = getattr(experiment, "output_dir", None) or os.getcwd()
        self._rabi_repeat_count = getattr(experiment, "repeat_count", None)

        self.open_and_configure()
        odmr_result: Dict[str, Any] = {}
        t1 = None
        raw_on_disk = None
        try:
            self.start_acquisition()
            try:
                odmr_result = getattr(experiment, run_method)(frequency_range)
            finally:
                records = self.stop_acquisition()
            raw_on_disk = self._raw_path if (self.stream_raw_to_disk and self._raw_path) else None
            try:
                n_iter, tau_ns, tau_is_index = self._infer_t1_axis(experiment)
                gates = self._adwin_gates_from_experiment(experiment)
                self._t1_axis_cache = (n_iter, tau_ns, tau_is_index, gates)  # for offline re-runs
                t1 = self.analyze_t1(records, n_iter, tau_ns, tau_is_index,
                                     adwin_result=odmr_result, adwin_gates=gates)
                if save:
                    self._save_t1_outputs(t1)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[MultiHarp T1] WARNING: analysis failed ({e}). The raw records are safe"
                      + (f" on disk at {raw_on_disk}" if raw_on_disk else "")
                      + "; re-analyze offline with analyze_t1_records_file(). The ADwin data "
                        "is unaffected and will still be saved.")
        finally:
            self.close()

        if t1 is not None:
            print(t1.summary())
        return {"odmr": odmr_result, "t1": t1, "raw_path": raw_on_disk}

    def analyze_t1_records_file(self, path: str, n_iterations: int, tau_values_ns,
                                repeat_count: Optional[int] = None, adwin_result: Any = None,
                                adwin_gates: Optional[Dict[str, float]] = None,
                                base_resolution_ps: Optional[float] = None,
                                save: bool = True) -> T1Result:
        """Re-run the T1 analysis offline on a finished run's raw records -- the
        multiharp_raw_*.bin streamed during acquisition, or a .npy. The file is
        memory-mapped and decoded in chunks. Pass the same sweep (tau_values_ns,
        in ns), repeat_count (for the order checks), the ADwin gates as
        dict(sig_start_ns=..., ref_start_ns=..., count_ns=...) and, to compare
        counts, adwin_result = the run's dict or np.load('adwin_partial_latest.npz').
        base_resolution_ps: the value printed at the start of the run (default 5)."""
        records = self._open_records(path)
        if base_resolution_ps is not None:
            self._base_res_ps = float(base_resolution_ps)
        if repeat_count is not None:
            self._rabi_repeat_count = int(repeat_count)
        if self.output_dir is None:
            self.output_dir = os.path.dirname(os.path.abspath(path)) or os.getcwd()
        t1 = self.analyze_t1(records, n_iterations, np.asarray(tau_values_ns, dtype=float),
                             False, adwin_result=adwin_result, adwin_gates=adwin_gates)
        if save:
            self._save_t1_outputs(t1)
        print(t1.summary())
        return t1

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

        For the Rabi cases ("rabi_vs_spcm", "rabi_mw_tagged") this transparently
        delegates to run_rabi_with_experiment and returns {'odmr':..., 'rabi': ...}.
        """
        if self.case in self._RABI_CASES:
            return self.run_rabi_with_experiment(
                experiment, frequency_range=frequency_range,
                save=save, run_method=run_method)
        if self.case == "t1_vs_spcm":
            return self.run_t1_with_experiment(
                experiment, frequency_range=frequency_range,
                save=save, run_method=run_method)

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

if __name__ == "__main__":
    MHTT = MultiHarpTimingTester()
    MHTT.recompute_rabi(r"D:\Duttlab\Experiments\AQuISS_default_save_location\data\test_odmr\odmr_pulsed_output\multiharp_rabi_mw_tagged_20260916_134559.npz", [1100, 1400], ref_window_ns=[5000, 5300])