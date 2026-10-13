#!/usr/bin/env python3
"""
tau_advisor.py  --  Sampling / resolution advisor + sequence preview for pulsed ODMR.

Given the SAME sequence text you already feed the parser (the `variable ...`,
`... pulse on channel ...`, `marker, ...` mini-language), plus your assumed
T2 / T2*, this tells you:

    * a sensible tau RANGE  (start / stop)          -> "how long a sequence"
    * the MINIMUM number of steps along tau         -> Nyquist resolution
    * whether your CURRENT sweep will clip / alias  -> "why is my data garbage?"
    * the FFT frequency resolution your range buys   -> "can I even split those lines?"

and draws a preview (pulse timeline for a few tau values + an illustrative
signal-vs-tau curve with your current sampling vs the suggested sampling on top).

Nothing here talks to hardware. Pure string parsing + numpy + matplotlib.
"""
from __future__ import annotations
import re
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ----------------------------------------------------------------------------- #
#  Physical constants / lab defaults  (edit to taste)
# ----------------------------------------------------------------------------- #
GAMMA_13C_kHz_per_G = 1.0705       # 13C nuclear gyromagnetic ratio  (kHz / Gauss)
HYPERFINE_14N_MHz    = 2.16        # 14N axial hyperfine A|| (three lines, +/-2.16 MHz)

# Coherence-time presets. THESE ARE ROUGH -- override with your real numbers.
PRESETS = {
    #                 T2* (us)   T2 (us)     note
    "ensemble":  dict(t2star_us=0.5,  t2_us=5.0),   # dense NV ensemble, spin-bath limited
    "single_nv": dict(t2star_us=2.0,  t2_us=100.0), # single NV, decent diamond
}


# ----------------------------------------------------------------------------- #
#  Units-aware expression evaluation
#  Convention in your sequence language:
#     * a number WITH a suffix (ns/us/ms/s) is that time            -> converted to ns
#     * a BARE decimal < 1 (e.g. 0.000005806) is SECONDS            -> *1e9 -> ns
#     * a BARE integer / number >= 1 (e.g. 2, 3) is a MULTIPLIER    -> left as-is
#     * the sweep variable is substituted with its value in ns
#  (scientific "1e-6" notation in a time expression is not supported -- use 1000ns)
# ----------------------------------------------------------------------------- #
_SUFFIX_TO_NS = {"ns": 1.0, "us": 1e3, "\u00b5s": 1e3, "\u03bcs": 1e3, "ms": 1e6, "s": 1e9}
_SUFFIXED = re.compile(r"(\d+\.?\d*)\s*(ns|us|\u00b5s|\u03bcs|ms|s)\b", re.IGNORECASE)
_BARE_NUM = re.compile(r"(?<![\w.])(\d+\.\d+|\d+)(?![\w.])")


def _to_ns(value: float, unit: str) -> float:
    return float(value) * _SUFFIX_TO_NS[unit.lower()]


def eval_time_expr(expr: str, var_name: Optional[str], var_value_ns: Optional[float]) -> float:
    """Evaluate a start/length expression to nanoseconds (see module docstring)."""
    s = str(expr).strip()

    # 1) suffixed numbers -> ns literals
    def _sub_suffixed(m):
        return f"({_to_ns(float(m.group(1)), m.group(2))})"
    s = _SUFFIXED.sub(_sub_suffixed, s)

    # 2) remaining bare numbers: decimal<1 -> seconds->ns ; else leave as multiplier
    def _sub_bare(m):
        tok = m.group(1)
        val = float(tok)
        if "." in tok and val < 1.0:
            return f"({val * 1e9})"     # seconds -> ns
        return tok                       # multiplier / count -> untouched
    s = _BARE_NUM.sub(_sub_bare, s)

    # 3) substitute the sweep variable with its ns value (already ns, so a literal)
    if var_name:
        s = re.sub(rf"\b{re.escape(var_name)}\b", f"({var_value_ns})", s)

    # 4) sandboxed eval
    if not re.fullmatch(r"[0-9eE\.\+\-\*/\(\)\s]*", s):
        raise ValueError(f"Unsafe / unparsable time expression after substitution: {expr!r} -> {s!r}")
    return float(eval(s, {"__builtins__": {}}, {}))


# ----------------------------------------------------------------------------- #
#  Parsing the sequence text
# ----------------------------------------------------------------------------- #
@dataclass
class Var:
    name: str
    start_ns: float
    stop_ns: float
    steps: int

    @property
    def dtau_ns(self) -> float:
        return (self.stop_ns - self.start_ns) / max(self.steps - 1, 1)


@dataclass
class Pulse:
    label: str
    channel: int
    start_expr: str
    shape: str
    length_expr: str
    amplitude: float
    is_mw: bool = False
    kind: str = "other"     # 'pi2' | 'pi' | 'mw' | 'init' | 'readout' | 'gap' | 'other'


@dataclass
class Marker:
    label: str
    channel: int
    start_expr: str
    length_expr: str
    kind: str = "marker"    # 'laser' | 'readout' | 'marker'


@dataclass
class ParsedSequence:
    name: str = "sequence"
    seq_type: str = ""
    sample_rate_hz: float = 1e9
    total_duration_ns: Optional[float] = None
    repeat_count: Optional[int] = None
    variables: List[Var] = field(default_factory=list)
    pulses: List[Pulse] = field(default_factory=list)
    markers: List[Marker] = field(default_factory=list)

    @property
    def grid_ns(self) -> float:
        return 1e9 / self.sample_rate_hz


_HEADER = re.compile(r"^\s*sequence\s*:\s*(.*)$", re.IGNORECASE)
_VARLINE = re.compile(
    r"^\s*variable\s+([A-Za-z_]\w*)\s*,\s*start\s*=\s*([^,]+?)\s*,\s*stop\s*=\s*([^,]+?)\s*,\s*steps\s*=\s*(\d+)",
    re.IGNORECASE)
_PULSE = re.compile(
    r"^\s*([A-Za-z0-9_/]+)\s+pulse\s+on\s+channel\s+(\d+)\s+at\s+(.+?)\s*,\s*"
    r"([A-Za-z_]+)\s*,\s*(.+?)\s*,\s*([-\d.]+)\s*$",
    re.IGNORECASE)
_MARKER = re.compile(
    r"^\s*marker\s*,\s*([A-Za-z0-9_/]+)\s+on\s+channel\s+(\d+)\s+at\s+(.+?)\s*,\s*(.+?)\s*$",
    re.IGNORECASE)


def _rate_to_hz(txt: str) -> float:
    m = re.match(r"([\d.]+)\s*(ghz|mhz|khz|hz)?", txt.strip(), re.IGNORECASE)
    if not m:
        return 1e9
    val = float(m.group(1))
    unit = (m.group(2) or "hz").lower()
    return val * {"ghz": 1e9, "mhz": 1e6, "khz": 1e3, "hz": 1.0}[unit]


def _classify_pulse(label: str, amplitude: float) -> Tuple[bool, str]:
    lab = label.lower()
    if amplitude == 0:
        return (False, "gap")                                  # a 'wait' with amp 0 = free evolution
    if re.search(r"pi[_/]?2", lab) or lab in ("halfpi", "pi2"):
        return (True, "pi2")
    if re.fullmatch(r"pi(_\d+)?", lab):
        return (True, "pi")
    if any(k in lab for k in ("laser", "init", "aom", "green")):
        return (False, "init")
    if any(k in lab for k in ("readout", "read")):
        return (False, "readout")
    if any(k in lab for k in ("mw", "drive", "rabi")):
        return (True, "mw")
    # unknown non-zero pulse on a low channel is probably MW drive
    return (True, "mw")


def _classify_marker(label: str) -> str:
    lab = label.lower()
    if "laser" in lab:
        return "laser"
    if "read" in lab or "count" in lab:
        return "readout"
    return "marker"


def parse_sequence_text(text: str) -> ParsedSequence:
    """Parse the sequence mini-language into a ParsedSequence (best-effort, regex-based)."""
    if hasattr(text, "sequence_text"):          # allow passing an experiment object
        text = text.sequence_text
    ps = ParsedSequence()
    for raw in str(text).splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        m = _HEADER.match(line)
        if m:
            for kv in m.group(1).split(","):
                if "=" not in kv:
                    continue
                k, v = (x.strip() for x in kv.split("=", 1))
                k = k.lower()
                if k == "name":
                    ps.name = v
                elif k == "type":
                    ps.seq_type = v
                elif k == "sample_rate":
                    ps.sample_rate_hz = _rate_to_hz(v)
                elif k == "duration":
                    try:
                        ps.total_duration_ns = eval_time_expr(v, None, None)
                    except Exception:
                        pass
                elif k in ("repeat_count", "repeat"):
                    try:
                        ps.repeat_count = int(float(v))
                    except Exception:
                        pass
            continue

        m = _VARLINE.match(line)
        if m:
            name = m.group(1)
            try:
                start = eval_time_expr(m.group(2), None, None)
                stop = eval_time_expr(m.group(3), None, None)
            except Exception:
                start, stop = 0.0, 0.0
            ps.variables.append(Var(name, start, stop, int(m.group(4))))
            continue

        m = _PULSE.match(line)
        if m:
            label, ch, start_expr, shape, length_expr, amp = m.groups()
            is_mw, kind = _classify_pulse(label, float(amp))
            ps.pulses.append(Pulse(label, int(ch), start_expr.strip(), shape.strip(),
                                   length_expr.strip(), float(amp), is_mw, kind))
            continue

        m = _MARKER.match(line)
        if m:
            label, ch, start_expr, length_expr = m.groups()
            ps.markers.append(Marker(label, int(ch), start_expr.strip(),
                                     length_expr.strip(), _classify_marker(label)))
            continue
    return ps


# ----------------------------------------------------------------------------- #
#  Classify the experiment (Rabi / Ramsey / Echo / CPMG-XY) from structure
# ----------------------------------------------------------------------------- #
@dataclass
class SeqKind:
    family: str            # 'rabi' | 'ramsey' | 'echo' | 'cpmg' | 'unknown'
    sweep_mode: str        # 'length' | 'position' | 'none'
    n_pi2: int
    n_pi: int
    coherence: str         # 'T2*' | 'T2'
    evolution_factor: float  # multiply swept tau to get true free-evolution time
    note: str = ""


def classify_sequence(ps: ParsedSequence, echo_tau_is_halfdelay: bool = True) -> SeqKind:
    var = ps.variables[0].name if ps.variables else None
    mw = [p for p in ps.pulses if p.is_mw]
    # count DISTINCT pulse positions per kind, so I/Q pairs (same start on two
    # channels) or duplicated drive lines don't get counted twice
    def _norm(e):
        return re.sub(r"\s+", "", str(e))
    n_pi2 = len({_norm(p.start_expr) for p in mw if p.kind == "pi2"})
    n_pi = len({_norm(p.start_expr) for p in mw if p.kind == "pi"})

    # where does the variable live?
    sweep_mode = "none"
    if var:
        in_len = any(re.search(rf"\b{re.escape(var)}\b", p.length_expr) for p in ps.pulses)
        in_pos = any(re.search(rf"\b{re.escape(var)}\b", p.start_expr) for p in ps.pulses)
        if in_len and not in_pos:
            sweep_mode = "length"
        elif in_pos:
            sweep_mode = "position"
        elif in_len:
            sweep_mode = "length"

    hint = ps.seq_type.lower()

    # Rabi: single MW drive whose LENGTH is swept (classic), or type hint
    n_mw_drive = sum(1 for p in mw if p.kind == "mw")
    if (sweep_mode == "length" and n_pi == 0 and n_pi2 == 0) or (hint == "rabi" and sweep_mode != "position"):
        return SeqKind("rabi", sweep_mode or "length", n_pi2, n_pi, "T2*", 1.0,
                       "tau = MW pulse length; you're mapping the Rabi nutation.")

    # Position-swept coherence sequences
    if sweep_mode == "position" or n_pi2 >= 2:
        if n_pi >= 2:
            return SeqKind("cpmg", "position", n_pi2, n_pi, "T2",
                           2.0 * n_pi if echo_tau_is_halfdelay else 1.0,
                           f"{n_pi}-pulse dynamical decoupling (CPMG/XY-N).")
        if n_pi == 1:
            return SeqKind("echo", "position", n_pi2, n_pi, "T2",
                           2.0 if echo_tau_is_halfdelay else 1.0,
                           "Hahn echo: free evolution = 2*tau (pi refocuses at the centre).")
        return SeqKind("ramsey", "position", n_pi2, n_pi, "T2*", 1.0,
                       "Ramsey / FID: two pi/2 with free evolution tau in between.")

    return SeqKind("unknown", sweep_mode, n_pi2, n_pi, "T2*", 1.0,
                   "Could not classify from structure; treating tau like a T2* free-evolution.")


# ----------------------------------------------------------------------------- #
#  The actual suggestion math
# ----------------------------------------------------------------------------- #
@dataclass
class Suggestion:
    kind: SeqKind
    t_dec_us: float               # relevant coherence time (T2 or T2*)
    f_feature_MHz: float          # fastest oscillation vs tau we must resolve
    f_feature_source: str
    tau_start_ns: float
    tau_stop_ns: float
    dtau_nyquist_ns: float        # Nyquist ceiling on the step (1 / 2f)
    dtau_oversampled_ns: float    # k-samples-per-period step
    grid_ns: float
    n_steps_nyquist: int
    n_steps_oversampled: int
    fft_bin_MHz: float            # 1 / tau-range -> resolvable freq spacing
    can_meet_nyquist_on_grid: bool
    warnings: List[str] = field(default_factory=list)
    infos: List[str] = field(default_factory=list)


def suggest_sampling(
    ps: ParsedSequence,
    *,
    t2_us: Optional[float] = None,
    t2star_us: Optional[float] = None,
    preset: Optional[str] = None,
    f_feature_MHz: Optional[float] = None,
    detuning_MHz: Optional[float] = None,
    hyperfine_MHz: float = HYPERFINE_14N_MHz,
    rabi_MHz: Optional[float] = None,
    b_field_G: Optional[float] = None,
    echo_tau_is_halfdelay: bool = True,
    samples_per_period: int = 8,
    n_coherence: float = 2.5,
) -> Suggestion:
    """Compute range + resolution suggestions. See advise_tau() for the friendly wrapper."""
    if preset:
        if preset not in PRESETS:
            raise ValueError(f"preset must be one of {list(PRESETS)}")
        t2star_us = PRESETS[preset]["t2star_us"] if t2star_us is None else t2star_us
        t2_us = PRESETS[preset]["t2_us"] if t2_us is None else t2_us

    kind = classify_sequence(ps, echo_tau_is_halfdelay=echo_tau_is_halfdelay)
    warnings: List[str] = []
    infos: List[str] = []

    # --- pick the relevant coherence time ---------------------------------- #
    if kind.coherence == "T2":
        if t2_us is None:
            t2_us = PRESETS["single_nv"]["t2_us"]
            warnings.append(f"No T2 given -> assumed {t2_us} us (single-NV preset). Set t2_us=...")
        t_dec_us = t2_us
    else:
        if t2star_us is None:
            t2star_us = PRESETS["ensemble"]["t2star_us"]
            warnings.append(f"No T2* given -> assumed {t2star_us} us (ensemble preset). Set t2star_us=...")
        t_dec_us = t2star_us

    # --- fastest oscillation vs tau (what Nyquist must catch) --------------- #
    if f_feature_MHz is not None:
        f_feat, src = f_feature_MHz, "user-supplied f_feature"
    elif kind.family == "rabi":
        if rabi_MHz is None:
            f_feat, src = 10.0, "DEFAULT Rabi 10 MHz (set rabi_MHz=Omega/2pi!)"
            warnings.append("Rabi sequence but no rabi_MHz given -> assumed 10 MHz. Steps will be wrong if your Rabi is faster.")
        else:
            f_feat, src = rabi_MHz, f"Rabi frequency {rabi_MHz} MHz"
    elif kind.family == "ramsey":
        det = abs(detuning_MHz) if detuning_MHz is not None else 0.0
        f_feat = det + hyperfine_MHz
        src = (f"detuning {det:g} + 14N hyperfine {hyperfine_MHz:g} MHz"
               if detuning_MHz is not None else f"14N hyperfine {hyperfine_MHz:g} MHz (no detuning given)")
    else:  # echo / cpmg / unknown -> nuclear modulation (ESEEM) at ~2*Larmor
        if b_field_G is not None:
            f_larmor = GAMMA_13C_kHz_per_G * b_field_G / 1e3   # MHz
            f_feat = 2.0 * f_larmor
            src = f"2 x 13C Larmor at {b_field_G:g} G (f_L={f_larmor*1e3:.0f} kHz)"
        else:
            f_feat = 0.2
            src = "DEFAULT 0.2 MHz modulation (give b_field_G or f_feature_MHz for AC/ESEEM target)"
            warnings.append("Echo/DD but no B field or f_feature -> assumed 0.2 MHz. Set b_field_G=... or f_feature_MHz=...")

    # --- RANGE: span a few coherence times of FREE EVOLUTION ---------------- #
    evol_stop_ns = n_coherence * t_dec_us * 1e3            # in evolution time
    tau_stop_ns = evol_stop_ns / kind.evolution_factor      # convert to swept-tau units
    if kind.family == "rabi":
        # also make sure we see several clean Rabi periods
        min_periods = 6
        tau_periods_ns = min_periods / f_feat * 1e3
        tau_stop_ns = max(tau_stop_ns, tau_periods_ns)
        infos.append(f"Rabi range = max({n_coherence}xT2* envelope, {min_periods} Rabi periods).")
    tau_start_ns = 0.0

    # --- RESOLUTION: Nyquist + oversampling on f_feature -------------------- #
    dtau_nyq_ns = 1.0 / (2.0 * f_feat) * 1e3                # ns  (1/2f)
    dtau_over_ns = 1.0 / (samples_per_period * f_feat) * 1e3
    grid = ps.grid_ns

    # round the *usable* step up to the hardware grid
    def _round_up_grid(x):
        return max(grid, math.ceil(x / grid) * grid)

    dtau_over_grid = _round_up_grid(dtau_over_ns)
    span = tau_stop_ns - tau_start_ns
    n_nyq = int(math.ceil(span / dtau_nyq_ns)) + 1
    n_over = int(math.ceil(span / dtau_over_grid)) + 1

    can_meet_nyquist = dtau_nyq_ns >= grid
    if not can_meet_nyquist:
        warnings.append(
            f"Nyquist wants dtau <= {dtau_nyq_ns:.3f} ns but your AWG grid is {grid:.3f} ns "
            f"(sample_rate={ps.sample_rate_hz/1e9:g} GHz). You CANNOT resolve {f_feat:g} MHz -- "
            f"raise the sample rate or you'll alias.")

    # --- FFT bin: what frequency spacing the range lets you resolve --------- #
    # frequency resolution is set by the span of the *swept* axis
    fft_bin_MHz = 1.0 / (span * 1e-9) / 1e6 if span > 0 else float("inf")

    # --- geometry check: can the pulses actually be PLACED at tau_stop? ------ #
    # (sequences authored as "pulse at fixed - tau" break when tau > fixed)
    var_name = ps.variables[0].name if ps.variables else None
    if var_name:
        worst = None
        for p in ps.pulses:
            if p.kind == "gap":
                continue
            try:
                st = eval_time_expr(p.start_expr, var_name, tau_stop_ns)
            except Exception:
                continue
            if st < 0 and (worst is None or st < worst[1]):
                worst = (p.label, st)
        if worst is not None:
            warnings.append(
                f"GEOMETRY: at the suggested tau_stop={_fmt_ns(tau_stop_ns)}, pulse "
                f"'{worst[0]}' would start at {worst[1]:.0f} ns (negative!). This sequence "
                f"places pulses as (fixed - tau), so 'fixed' must exceed tau_max. Re-anchor the "
                f"pulses (e.g. put them at 'tau + const' with const >= tau_max) before sweeping this far.")

    return Suggestion(
        kind=kind, t_dec_us=t_dec_us, f_feature_MHz=f_feat, f_feature_source=src,
        tau_start_ns=tau_start_ns, tau_stop_ns=tau_stop_ns,
        dtau_nyquist_ns=dtau_nyq_ns, dtau_oversampled_ns=dtau_over_grid, grid_ns=grid,
        n_steps_nyquist=n_nyq, n_steps_oversampled=n_over,
        fft_bin_MHz=fft_bin_MHz, can_meet_nyquist_on_grid=can_meet_nyquist,
        warnings=warnings, infos=infos)


# ----------------------------------------------------------------------------- #
#  Text report
# ----------------------------------------------------------------------------- #
def _fmt_ns(x: float) -> str:
    if x >= 1e6:
        return f"{x/1e6:.3g} ms"
    if x >= 1e3:
        return f"{x/1e3:.3g} us"
    return f"{x:.3g} ns"


def format_report(ps: ParsedSequence, s: Suggestion) -> str:
    L = []
    A = L.append
    A("=" * 72)
    A(f"  SEQUENCE: {ps.name}   (type hint: {ps.seq_type or 'n/a'})")
    A("=" * 72)
    A(f"  Detected family : {s.kind.family.upper()}   "
      f"[{s.kind.n_pi2} x pi/2, {s.kind.n_pi} x pi, sweep={s.kind.sweep_mode}]")
    A(f"                    {s.kind.note}")
    A(f"  Relevant time   : {s.kind.coherence} = {s.t_dec_us:g} us"
      + (f"   (free evolution = {s.kind.evolution_factor:g} x tau)"
         if s.kind.evolution_factor != 1 else ""))
    A(f"  AWG time grid   : {s.grid_ns:g} ns   (sample_rate = {ps.sample_rate_hz/1e9:g} GHz)")

    # current sweep
    A("-" * 72)
    if ps.variables:
        v = ps.variables[0]
        A(f"  YOUR CURRENT SWEEP  '{v.name}':")
        A(f"      {_fmt_ns(v.start_ns)} -> {_fmt_ns(v.stop_ns)},  {v.steps} steps"
          f"   (dtau = {v.dtau_ns:.3g} ns)")
    else:
        A("  YOUR CURRENT SWEEP  : (no 'variable' line found)")

    # suggested range
    A("-" * 72)
    A("  SUGGESTED RANGE  (span a few coherence times so the decay is visible):")
    A(f"      start = {_fmt_ns(s.tau_start_ns)}")
    A(f"      stop  = {_fmt_ns(s.tau_stop_ns)}   "
      f"(= {s.kind.evolution_factor:g}x -> {_fmt_ns(s.tau_stop_ns*s.kind.evolution_factor)} of evolution "
      f"~ {s.tau_stop_ns*s.kind.evolution_factor/(s.t_dec_us*1e3):.2g} x {s.kind.coherence})")
    if ps.variables:
        v = ps.variables[0]
        ratio = (v.stop_ns * s.kind.evolution_factor) / (s.t_dec_us * 1e3)
        if ratio < 1.0:
            A(f"      !! your stop reaches only {ratio:.2g} x {s.kind.coherence} "
              f"-> decay is CLIPPED, fit/FFT will be poor.")
        elif ratio > 4.0:
            A(f"      !! your stop reaches {ratio:.2g} x {s.kind.coherence} "
              f"-> mostly noise past ~3x; you're wasting points/time out there.")

    # suggested resolution
    A("-" * 72)
    A("  SUGGESTED RESOLUTION  (Nyquist on the fastest feature vs tau):")
    A(f"      fastest feature f = {s.f_feature_MHz:g} MHz   [{s.f_feature_source}]")
    A(f"      Nyquist  dtau <= 1/(2f) = {s.dtau_nyquist_ns:.3g} ns   -> N >= {s.n_steps_nyquist} steps (bare minimum)")
    A(f"      Clean    dtau  = 1/(k f) = {s.dtau_oversampled_ns:.3g} ns "
      f"(k~{max(1,round(1e3/ s.f_feature_MHz / s.dtau_oversampled_ns))}/period, grid-rounded) "
      f"-> N ~ {s.n_steps_oversampled} steps (recommended)")
    if ps.variables:
        v = ps.variables[0]
        if v.dtau_ns > s.dtau_nyquist_ns:
            A(f"      !! your dtau = {v.dtau_ns:.3g} ns is COARSER than Nyquist "
              f"({s.dtau_nyquist_ns:.3g} ns) -> you WILL alias {s.f_feature_MHz:g} MHz.")

    # frequency resolution
    A("-" * 72)
    A("  FREQUENCY RESOLUTION you get from the range (FFT bin = 1/span):")
    A(f"      suggested span -> df ~ {s.fft_bin_MHz:.3g} MHz per bin")
    need = 1.0 / (s.tau_stop_ns * 1e-9) / 1e6
    A(f"      to split two peaks df apart you need span >= 1/df "
      f"(e.g. to split {HYPERFINE_14N_MHz:g} MHz 14N lines: span >= {_fmt_ns(1e3/HYPERFINE_14N_MHz)} of evolution).")
    if s.kind.coherence == "T2*":
        best = 1.0 / (s.t_dec_us * 1e-6) / 1e6
        A(f"      NOTE: T2* itself caps your linewidth at ~{best:.3g} MHz "
          f"-- features closer than that are physically unresolvable no matter the sampling.")

    # warnings
    if s.warnings:
        A("-" * 72)
        A("  WARNINGS:")
        for w in s.warnings:
            A(f"    * {w}")
    if s.infos:
        for i in s.infos:
            A(f"    (i) {i}")
    A("=" * 72)

    # ready-to-paste variable line
    if ps.variables:
        v = ps.variables[0]
        A("  DROP-IN replacement for your 'variable' line (recommended sampling):")
        A(f"      variable {v.name}, start={s.tau_start_ns:.0f}ns, "
          f"stop={s.tau_stop_ns:.0f}ns, steps={s.n_steps_oversampled}")
        A("=" * 72)
    return "\n".join(L)


# ----------------------------------------------------------------------------- #
#  Preview figure
# ----------------------------------------------------------------------------- #
_CH_COLOR = {}
_PALETTE = ["#2b6cb0", "#c05621", "#2f855a", "#6b46c1", "#b83280", "#718096"]


def _ch_color(ch: int) -> str:
    if ch not in _CH_COLOR:
        _CH_COLOR[ch] = _PALETTE[len(_CH_COLOR) % len(_PALETTE)]
    return _CH_COLOR[ch]


def _resolved_items(ps: ParsedSequence, tau_ns: float):
    """Return (pulses, markers) with numeric start/length in ns for a given tau."""
    var = ps.variables[0].name if ps.variables else None
    P, M = [], []
    for p in ps.pulses:
        try:
            st = eval_time_expr(p.start_expr, var, tau_ns)
            ln = eval_time_expr(p.length_expr, var, tau_ns)
        except Exception:
            continue
        P.append((p, st, ln))
    for mk in ps.markers:
        try:
            st = eval_time_expr(mk.start_expr, var, tau_ns)
            ln = eval_time_expr(mk.length_expr, var, tau_ns)
        except Exception:
            continue
        M.append((mk, st, ln))
    return P, M


def _signal_model(tau_ns: np.ndarray, s: Suggestion) -> np.ndarray:
    """Illustrative (schematic!) contrast vs tau: envelope x oscillation."""
    t_evol = tau_ns * s.kind.evolution_factor
    tdec = s.t_dec_us * 1e3
    if s.kind.coherence == "T2":
        env = np.exp(-(t_evol / tdec) ** 1.0)         # schematic echo decay
    else:
        env = np.exp(-(t_evol / tdec) ** 2.0)         # schematic Gaussian FID
    osc = np.cos(2 * np.pi * s.f_feature_MHz * 1e-3 * tau_ns)   # MHz * ns/1e3 -> cycles
    return 0.5 * (1 + env * osc)


def make_preview(ps: ParsedSequence, s: Suggestion,
                 title: Optional[str] = None, save_path: Optional[str] = None,
                 show: bool = False):
    """Draw: (top) pulse timeline at 3 tau values, (bottom) signal model + sampling."""
    # signal-panel range comes from the SUGGESTED sweep
    lo, hi = s.tau_start_ns, s.tau_stop_ns

    # timeline panels use YOUR ACTUAL sweep endpoints (always geometrically valid,
    # unlike the suggested range which may push 'fixed - tau' pulses negative)
    if ps.variables:
        v0 = ps.variables[0]
        tau_show = [v0.start_ns, 0.5 * (v0.start_ns + v0.stop_ns), v0.stop_ns]
        tau_tags = ["current start", "current mid", "current stop"]
    else:
        tau_show = [0.0, 0.0, 0.0]
        tau_tags = ["tau = 0", "tau = 0", "tau = 0"]

    channels = sorted({p.channel for p in ps.pulses} | {m.channel for m in ps.markers})
    ch_row = {ch: i for i, ch in enumerate(channels)}
    n_ch = max(len(channels), 1)

    # shared x-window across the three timeline panels
    g_min, g_max = 0.0, 1.0
    for tau in tau_show:
        P, M = _resolved_items(ps, tau)
        for (_, st, ln) in P + M:
            g_min = min(g_min, st)
            g_max = max(g_max, st + ln)
    pad = 0.02 * (g_max - g_min)

    fig = plt.figure(figsize=(11, 8.5))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1.4], hspace=0.6)

    # ---- three timeline panels ----
    for panel, (tau, tag) in enumerate(zip(tau_show, tau_tags)):
        ax = fig.add_subplot(gs[panel, 0])
        P, M = _resolved_items(ps, tau)
        tmax = g_max

        # shaded marker windows (laser green, readout orange)
        for (mk, st, ln) in M:
            c = {"laser": "#c6f6d5", "readout": "#feebc8"}.get(mk.kind, "#e2e8f0")
            ax.axvspan(st, st + ln, color=c, alpha=0.55, zorder=0)

        # pulse boxes per channel row
        for (p, st, ln) in P:
            row = ch_row.get(p.channel, 0)
            y = n_ch - 1 - row
            if p.kind == "gap":
                ax.add_patch(Rectangle((st, y + 0.15), max(ln, tmax*0.002), 0.7,
                                       fill=False, ls=":", ec="#a0aec0", lw=1.2, zorder=2))
            else:
                col = _ch_color(p.channel)
                h = 0.7 * (0.55 if p.kind == "pi2" else 1.0 if p.kind == "pi" else 0.8)
                ax.add_patch(Rectangle((st, y + 0.15), max(ln, tmax*0.004), h,
                                       facecolor=col, edgecolor="black", lw=0.6,
                                       alpha=0.9, zorder=3))
                ax.text(st + max(ln, tmax*0.004) / 2, y + 0.15 + h + 0.03, p.label,
                        ha="center", va="bottom", fontsize=6.5, rotation=0, zorder=4)

        ax.set_xlim(g_min - pad, g_max + pad)
        ax.set_ylim(-0.1, n_ch + 0.3)
        ax.set_yticks([n_ch - 1 - ch_row[ch] + 0.5 for ch in channels])
        ax.set_yticklabels([f"ch{ch}" for ch in channels], fontsize=7)
        ax.set_title(f"pulse timeline @ {tag}: tau = {_fmt_ns(tau)}", fontsize=8, loc="left")
        ax.tick_params(axis="x", labelsize=7)
        if panel == 2:
            ax.set_xlabel("time within one shot (ns)", fontsize=8)

    # ---- signal-vs-tau model panel ----
    axs = fig.add_subplot(gs[3, 0])
    tau_dense = np.linspace(lo, max(hi, 1), 2000)
    axs.plot(tau_dense, _signal_model(tau_dense, s), color="#4a5568", lw=1.2,
             label="illustrative signal (schematic)")

    # decay envelope guide
    tdec = s.t_dec_us * 1e3
    if s.kind.coherence == "T2":
        env = np.exp(-((tau_dense * s.kind.evolution_factor) / tdec) ** 1.0)
    else:
        env = np.exp(-((tau_dense * s.kind.evolution_factor) / tdec) ** 2.0)
    axs.plot(tau_dense, 0.5 + 0.5 * env, color="#e53e3e", lw=1.0, ls="--", alpha=0.8,
             label=f"decay envelope ({s.kind.coherence})")
    axs.plot(tau_dense, 0.5 - 0.5 * env, color="#e53e3e", lw=1.0, ls="--", alpha=0.8)

    # suggested sampling points
    tau_sug = np.linspace(s.tau_start_ns, s.tau_stop_ns, s.n_steps_oversampled)
    axs.plot(tau_sug, _signal_model(tau_sug, s), "o", ms=3.2, color="#2f855a",
             label=f"suggested sampling (N={s.n_steps_oversampled})")

    # current sampling points (if any), sampled on the SAME model for honest aliasing view
    if ps.variables:
        v = ps.variables[0]
        # shade the window your current sweep actually covers
        axs.axvspan(v.start_ns, v.stop_ns, color="#b83280", alpha=0.07, zorder=0)
        cover = (v.stop_ns - v.start_ns) / max(hi - lo, 1e-9) * 100
        xt = min(max(0.5 * (v.start_ns + v.stop_ns), lo + 0.18 * (hi - lo)),
                 hi - 0.18 * (hi - lo))
        axs.text(xt, 1.10, f"your window ({cover:.0f}% of suggested span)",
                 ha="center", va="top", fontsize=6.5, color="#b83280")
        tau_cur = np.linspace(v.start_ns, v.stop_ns, v.steps)
        axs.plot(tau_cur, _signal_model(tau_cur, s), "x", ms=6, mew=1.4, color="#b83280",
                 label=f"your current sampling (N={v.steps})")
        # draw the line connecting current samples to expose aliasing
        axs.plot(tau_cur, _signal_model(tau_cur, s), "-", lw=0.7, color="#b83280", alpha=0.45)

    axs.set_xlim(lo - (hi - lo) * 0.02, hi * 1.03)
    axs.set_ylim(-0.05, 1.15)
    axs.set_xlabel("tau (ns)", fontsize=8)
    axs.set_ylabel("contrast (a.u.)", fontsize=8)
    axs.set_title("signal vs tau  --  does your sampling capture it?", fontsize=8, loc="left")
    axs.legend(fontsize=6.5, ncol=2, loc="upper right", framealpha=0.9)
    axs.tick_params(labelsize=7)

    fig.suptitle(title or f"Sequence preview + sampling advice: {ps.name} "
                          f"[{s.kind.family}]", fontsize=11, y=0.995)

    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    if show:
        plt.show()
    return fig


# ----------------------------------------------------------------------------- #
#  Top-level convenience wrapper
# ----------------------------------------------------------------------------- #
def advise_tau(sequence, *, show_preview: bool = True, save_path: Optional[str] = None,
               print_report: bool = True, **kwargs) -> Suggestion:
    """
    One call does everything.

    sequence : the sequence TEXT string, OR anything with a `.sequence_text`
               attribute (e.g. your ODMRPulsedExperiment instance).

    Common kwargs (all optional):
        preset="ensemble" | "single_nv"     quick T2/T2* presets
        t2_us=..., t2star_us=...             your measured coherence times
        detuning_MHz=...                     Ramsey: |drive - transition|
        hyperfine_MHz=2.16                   Ramsey beat (14N default)
        rabi_MHz=...                         Rabi: Rabi frequency
        b_field_G=...                        Echo/DD: sets 13C Larmor target
        f_feature_MHz=...                    override the fastest feature directly
        samples_per_period=8                 oversampling beyond Nyquist
        n_coherence=2.5                      how many T2/T2* to span
        echo_tau_is_halfdelay=True           evolution = 2*tau for echo/DD

    Returns a Suggestion dataclass (all the numbers).
    """
    ps = parse_sequence_text(sequence)
    s = suggest_sampling(ps, **kwargs)
    if print_report:
        print(format_report(ps, s))
    if show_preview or save_path:
        make_preview(ps, s, save_path=save_path, show=show_preview)
    return s
