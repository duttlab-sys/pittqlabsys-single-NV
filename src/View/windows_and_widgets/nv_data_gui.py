"""
nv_data_gui.py — a Tkinter + matplotlib GUI for visualising, analysing and
fitting our lab's HDF5 data (pulsed ODMR / Rabi / Ramsey / echo, CW-ODMR
frequency sweeps, confocal raster scans, and camera image stacks).
=============================================================================

What it does
------------
* Open any HDF5 saved by our struct_hdf5.py and browse the whole struct as a
  tree (groups, struct-arrays, datasets, and scalar attributes).
* Auto-detects the four data types we have and picks a sensible first plot:
    - signal_counts / reference_counts   -> pulsed experiment (1D)
    - counts + frequencies               -> CW ODMR sweep (1D)
    - count_img                          -> confocal image (2D)
    - image_1.. / camera_image / 3-D arr -> image stack (movie)
* SELECT PART OF THE DATA and act only on it:
    - 1D: drag a span on the plot -> zoom / fit / stats / export just that range
    - 2D: drag a rectangle ROI    -> crop / row+column profile / stats
* Fit 1D data (full trace OR the selected span) with a library of models
  (Lorentzian & Gaussian peaks/dips, damped cosine/Rabi, Ramsey, stretched-exp
  echo, exponential, polynomials...), with parameter uncertainties and R^2.
* A dedicated "Pulsed" tab that calls our fitters in plot_pulsed_hdf5.py and
  the false-alarm-probability test in significance_test.py when those files are
  importable, and otherwise uses built-in equivalents.
* Image tab: colormaps, percentile contrast clip, ROI crop/profiles, and a movie
  player (frame slider + play/pause + mean/max/sum projections) for stacks.

How to run
----------
Put this file wherever we like and run:

    python nv_data_gui.py                 # opens a file picker
    python nv_data_gui.py path/to/file.h5 # opens that file directly

Needs: numpy, scipy, matplotlib, h5py, tkinter — all already in our env.
plot_pulsed_hdf5.py and significance_test.py are optional; keep them next to
this file (or on PYTHONPATH) to enable the specialised pulsed workflow.

Design note: everything numerical lives in plain module-level functions so it
can be imported and unit-tested without a display. All GUI / h5py / pyplot
imports are deferred into the functions that need them.
"""
from __future__ import annotations

import os
import re
import numpy as np
from scipy.optimize import curve_fit

APP_TITLE = "NV / ODMR Data Explorer"


# =============================================================================
# HDF5 -> nested-tree loader (self-contained; mirrors struct_hdf5 conventions)
# =============================================================================
def load_hdf5_tree(path):
    """Read an HDF5 file into a nested python structure.

    Mapping (matches our struct_hdf5.py save format):
      * group attributes            -> scalar/str leaves
      * datasets                    -> numpy arrays (or scalars)
      * subgroup whose keys are all digits -> list (a MATLAB-style struct array)
      * any other subgroup          -> nested dict

    Returns a dict.  h5py is imported lazily so this module stays importable
    without it.
    """
    import h5py

    def read_group(g):
        node = {}
        # scalar attributes first
        for k, v in g.attrs.items():
            node[k] = _clean_scalar(v)
        # then child datasets / groups
        for name, obj in g.items():
            if isinstance(obj, h5py.Dataset):
                node[name] = _clean_dataset(obj[()])
            elif isinstance(obj, h5py.Group):
                keys = list(obj.keys())
                if keys and all(k.isdigit() for k in keys):
                    node[name] = [read_group(obj[k])
                                  for k in sorted(keys, key=int)]
                else:
                    node[name] = read_group(obj)
        return node

    with h5py.File(path, "r", libver="latest", swmr=False) as f:
        try:
            return read_group(f)
        except OSError:
            # SWMR files written with swmr=True occasionally need the flag
            pass
    with h5py.File(path, "r", libver="latest", swmr=True) as f:
        return read_group(f)


def _clean_scalar(v):
    if isinstance(v, (bytes, np.bytes_)):
        return v.decode("utf-8", "ignore")
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray) and v.size == 1:
        return v.reshape(-1)[0].item()
    return v


def _clean_dataset(v):
    if isinstance(v, (bytes, np.bytes_)):
        return v.decode("utf-8", "ignore")
    arr = np.asarray(v)
    if arr.dtype.kind in ("S", "O", "U"):
        # array of bytes/strings -> joined text (sequence_text etc.)
        flat = arr.ravel()
        parts = [x.decode("utf-8", "ignore") if isinstance(x, (bytes, np.bytes_))
                 else str(x) for x in flat]
        return "\n".join(parts) if len(parts) > 1 else (parts[0] if parts else "")
    if arr.ndim == 0:
        return arr.item()
    return arr


# =============================================================================
# Tree helpers: flatten, classify, find text (for auto-detection)
# =============================================================================
def flatten_tree(node, prefix=""):
    """Yield (path, value) for every leaf. Lists index as name[i]."""
    if isinstance(node, dict):
        for k, v in node.items():
            p = f"{prefix}/{k}" if prefix else str(k)
            yield from flatten_tree(v, p)
    elif isinstance(node, list):
        for i, item in enumerate(node):
            p = f"{prefix}[{i}]"
            yield from flatten_tree(item, p)
    else:
        yield prefix, node


def classify_value(v):
    """Return one of: 'scalar', 'text', '1d', 'image', 'stack', 'array', 'group'."""
    if isinstance(v, (dict, list)):
        return "group"
    if isinstance(v, str):
        return "text"
    if isinstance(v, (int, float, bool, np.integer, np.floating, np.bool_)):
        return "scalar"
    if isinstance(v, np.ndarray):
        a = np.squeeze(v)
        if a.ndim == 0 or a.size == 1:
            return "scalar"
        if not np.issubdtype(a.dtype, np.number):
            return "array"
        if a.ndim == 1:
            return "1d"
        if a.ndim == 2 and a.shape[0] > 1 and a.shape[1] > 1:
            return "image"
        if a.ndim == 3 and a.shape[-1] not in (3, 4) and min(a.shape[1:]) > 1:
            return "stack"
        if a.ndim == 3 and a.shape[-1] in (3, 4):
            return "image"      # RGB(A)
        return "array"
    return "scalar"


def describe_value(v):
    """Short human description for the info box / tree label."""
    kind = classify_value(v)
    if kind == "group":
        n = len(v)
        return f"[{'list' if isinstance(v, list) else 'group'}: {n} item{'s' if n != 1 else ''}]"
    if kind == "text":
        s = v.replace("\n", " ")
        return f'"{s[:40]}{"…" if len(s) > 40 else ""}"'
    if kind == "scalar":
        try:
            return f"{float(v):.6g}"
        except (TypeError, ValueError):
            return str(v)
    if isinstance(v, np.ndarray):
        return f"{kind} {tuple(np.squeeze(v).shape)} {v.dtype}"
    return str(type(v).__name__)


def find_text(tree, *needles):
    """First string leaf whose *value* looks like it contains any needle."""
    for _, v in flatten_tree(tree):
        if isinstance(v, str):
            low = v.lower()
            if any(n in low for n in needles):
                return v
    return None


def get_by_name(tree, *name_options, want=("1d",)):
    """First leaf whose PATH contains any of name_options and whose kind is in
    `want`. Case-insensitive. Returns (path, value) or (None, None)."""
    for path, v in flatten_tree(tree):
        low = path.lower()
        if any(opt in low for opt in name_options) and classify_value(v) in want:
            return path, v
    return None, None


# =============================================================================
# Image-stack detection (handles the several ways stacks can be stored)
# =============================================================================
_NUMBERED = re.compile(r"^(.*?)(\d+)$")


def find_image_stacks(tree):
    """Return a list of dicts {name, frames (N,H,W ndarray), source} describing
    every image stack we can assemble, across the layouts our files use:
      1) a single 3-D array,
      2) >=3 sibling 2-D arrays that share a name stem + number (image_1..N),
      3) a list of dicts that each hold a common 2-D image field.
    """
    stacks = []

    # 1) 3-D arrays anywhere
    for path, v in flatten_tree(tree):
        if isinstance(v, np.ndarray):
            a = np.squeeze(v)
            if a.ndim == 3 and a.shape[-1] not in (3, 4) and min(a.shape[1:]) > 1:
                stacks.append({"name": f"{path} (3-D, {a.shape[0]} frames)",
                               "frames": a, "source": path})

    # 2) sibling numbered 2-D arrays inside each dict level
    def _stack_cohort(items_kv):
        """items_kv: list of (key, 2-D array). Stack the largest set that shares
        one shape (>=3). Returns (frames, n_used, n_total) or None."""
        from collections import Counter
        shape_counts = Counter(im.shape for _, im in items_kv)
        best_shape, cnt = shape_counts.most_common(1)[0]
        if cnt < 3:
            return None
        cohort = [im for _, im in items_kv if im.shape == best_shape]
        return np.stack(cohort, axis=0), len(cohort), len(items_kv)

    def _stack_name(p, n_used, n_total):
        if n_used == n_total:
            return f"{p} ({n_used} frames)"
        return f"{p} ({n_used} frames, {n_total - n_used} odd-sized skipped)"

    def scan_dicts(node, prefix=""):
        if isinstance(node, dict):
            # 2) sibling numbered 2-D arrays directly under this dict
            groups = {}
            for k, v in node.items():
                if isinstance(v, np.ndarray) and np.squeeze(v).ndim == 2:
                    m = _NUMBERED.match(k)
                    stem = m.group(1) if m else k
                    groups.setdefault(stem, []).append((k, np.squeeze(v)))
            for stem, items in groups.items():
                if len(items) >= 3:
                    def keyfn(kv):
                        m = _NUMBERED.match(kv[0])
                        return int(m.group(2)) if m else 0
                    items = sorted(items, key=keyfn)
                    res = _stack_cohort(items)
                    if res is not None:
                        frames, nu, nt = res
                        p = f"{prefix}/{stem}*" if prefix else f"{stem}*"
                        stacks.append({"name": _stack_name(p, nu, nt),
                                       "frames": frames, "source": p})

            # 4) sibling sub-GROUPS that each hold a common 2-D field, e.g.
            #    image_1/camera_image, image_2/camera_image, ...  (each frame is
            #    its own group with metadata alongside the picture).
            subgroups = [(k, v) for k, v in node.items() if isinstance(v, dict)]
            if len(subgroups) >= 3:
                field_frames = {}          # field name -> [(order, key, 2-D array)]
                for k, sub in subgroups:
                    m = _NUMBERED.match(k)
                    order = int(m.group(2)) if m else 0
                    for fk, fv in sub.items():
                        if isinstance(fv, np.ndarray) and np.squeeze(fv).ndim == 2:
                            field_frames.setdefault(fk, []).append(
                                (order, k, np.squeeze(fv)))
                for field, items in field_frames.items():
                    if len(items) >= 3:
                        items = sorted(items, key=lambda t: t[0])
                        res = _stack_cohort([(kk, im) for _, kk, im in items])
                        if res is not None:
                            frames, nu, nt = res
                            stems = {(_NUMBERED.match(kk).group(1)
                                      if _NUMBERED.match(kk) else kk)
                                     for _, kk, _ in items}
                            stem = stems.pop() if len(stems) == 1 else "group_"
                            p = (f"{prefix}/{stem}*/{field}" if prefix
                                 else f"{stem}*/{field}")
                            stacks.append({"name": _stack_name(p, nu, nt),
                                           "frames": frames, "source": p})

            for k, v in node.items():
                scan_dicts(v, f"{prefix}/{k}" if prefix else k)
        elif isinstance(node, list):
            # 3) list of dicts sharing a 2-D field
            fields = {}
            for item in node:
                if isinstance(item, dict):
                    for k, v in item.items():
                        if isinstance(v, np.ndarray) and np.squeeze(v).ndim == 2:
                            fields.setdefault(k, []).append((k, np.squeeze(v)))
            for field, imgs in fields.items():
                if len(imgs) >= 3:
                    res = _stack_cohort(imgs)
                    if res is not None:
                        frames, nu, nt = res
                        p = f"{prefix}[*].{field}"
                        stacks.append({"name": _stack_name(p, nu, nt),
                                       "frames": frames, "source": p})
            for i, item in enumerate(node):
                scan_dicts(item, f"{prefix}[{i}]")

    scan_dicts(tree)
    # de-duplicate by source
    seen, out = set(), []
    for s in stacks:
        if s["source"] not in seen:
            seen.add(s["source"])
            out.append(s)
    return out


# =============================================================================
# Auto-detection of the default view when a file is opened
# =============================================================================
def detect_default_view(tree):
    """Return dict describing the best first plot:
       {'mode': 'pulsed'|'cw_odmr'|'image'|'stack'|'1d'|'none', ...}."""
    _, sig = get_by_name(tree, "signal_counts", "signal", want=("1d", "image"))
    _, ref = get_by_name(tree, "reference_counts", "reference", want=("1d", "image"))
    if sig is not None and ref is not None:
        return {"mode": "pulsed"}

    fpath, freqs = get_by_name(tree, "frequencies", "freq_list", "freq_axis",
                               want=("1d",))
    cpath, counts = get_by_name(tree, "counts_averaged", "all_counts_forward",
                                "counts_forward", "counts", want=("1d",))
    if freqs is not None and counts is not None:
        return {"mode": "cw_odmr", "freq_path": fpath, "count_path": cpath}

    ipath, img = get_by_name(tree, "count_img", "count_image", "image",
                             want=("image",))
    stacks = find_image_stacks(tree)
    if stacks:
        best = max(stacks, key=lambda s: s["frames"].shape[0])
        return {"mode": "stack", "stack": best, "all_stacks": stacks}
    if img is not None:
        return {"mode": "image", "path": ipath}

    # fall back to the first 1-D array
    p, v = None, None
    for path, val in flatten_tree(tree):
        if classify_value(val) == "1d":
            p, v = path, val
            break
    if p is not None:
        return {"mode": "1d", "path": p}
    return {"mode": "none"}


# =============================================================================
# x-axis reconstruction for pulsed sequences (adapted from plot_pulsed_hdf5.py)
# =============================================================================
def _num(token, default_unit="ns"):
    m = re.search(r"([0-9]*\.?[0-9]+)\s*(ns|us|µs|μs|ms|v|mv|ghz|mhz|khz|hz)?",
                  str(token), re.IGNORECASE)
    if not m:
        return float("nan"), default_unit
    return float(m.group(1)), (m.group(2) or default_unit).lower()


def _variable_line(seq_text):
    if not seq_text:
        return None
    m = re.search(
        r"(?<!#)\bvariable\s+(\w+)\s*,\s*start\s*=\s*([^,]+),\s*stop\s*=\s*([^,]+)",
        seq_text, re.IGNORECASE)
    if not m:
        return None
    return m.group(1).lower(), m.group(2), m.group(3)


def sweep_kind(seq_text):
    """Classify what the pulsed sequence actually sweeps: 'time' | 'amplitude'
    | 'frequency' | None.  The explicit experiment type wins — a pulsed-ODMR
    sweeps frequency even though its pulse template still declares a (fixed)
    duration variable.  Otherwise fall back to the variable name + its unit."""
    t = (seq_text or "").lower()
    m = re.search(r"type\s*=\s*(\w+)", t)
    typ = m.group(1) if m else ""
    if typ in ("odmr", "podmr", "pulsed_odmr", "cwodmr"):
        return "frequency"
    vl = _variable_line(seq_text)
    if not vl:
        return None
    name = vl[0].lower()
    _, unit = _num(vl[1], "")            # unit on the start token, if any
    # order matters: phase / amplitude / frequency keywords are checked before
    # time, because "pulse_amplitude"/"pulse_duration" both contain "pulse".
    if "phase" in name:
        return "phase"
    if any(k in name for k in ("amp", "power", "volt")) or unit in ("v", "mv"):
        return "amplitude"
    if any(k in name for k in ("freq", "detun")) or unit in ("ghz", "mhz", "khz", "hz"):
        return "frequency"
    if (any(k in name for k in ("dur", "tau", "time", "width", "delay",
                                "wait", "free", "gate"))
            or unit in ("ns", "us", "µs", "μs", "ms")):
        return "time"
    return "time"                        # sensible default for a pulsed scan


def build_phase_axis(seq_text, n):
    """Phase sweep reconstructed from the sequence text (radians)."""
    vl = _variable_line(seq_text)
    if vl and "phase" in vl[0].lower():
        _, s_tok, e_tok = vl
        s, _ = _num(s_tok, "")
        e, _ = _num(e_tok, "")
        if np.isfinite(s) and np.isfinite(e):
            return np.linspace(s, e, n), "phase (rad)"
    return np.arange(n, dtype=float), "scan point (index)"


def build_time_axis(seq_text, n):
    vl = _variable_line(seq_text)
    if vl and sweep_kind(seq_text) == "time":
        _, s_tok, e_tok = vl
        s, su = _num(s_tok, "ns")
        e, eu = _num(e_tok, "ns")
        sc = {"ns": 1.0, "us": 1e3, "µs": 1e3, "μs": 1e3, "ms": 1e6}
        if np.isfinite(s) and np.isfinite(e):
            return np.linspace(s * sc.get(su, 1), e * sc.get(eu, 1), n), "tau (ns)"
    return np.arange(n, dtype=float), "scan point (index)"


def build_amp_axis(seq_text, n):
    vl = _variable_line(seq_text)
    if vl and sweep_kind(seq_text) == "amplitude":
        _, s_tok, e_tok = vl
        s, su = _num(s_tok, "v")
        e, eu = _num(e_tok, "v")
        sc = {"v": 1.0, "mv": 1e-3}
        if np.isfinite(s) and np.isfinite(e):
            return np.linspace(s * sc.get(su, 1), e * sc.get(eu, 1), n), "pulse amplitude (V)"
    return np.arange(n, dtype=float), "scan point (index)"


def build_freq_axis(seq_text, n):
    """Frequency sweep reconstructed from the sequence text (returns Hz).  Only
    fires when the variable line is genuinely a frequency (freq/detuning name or
    a Hz/kHz/MHz/GHz unit) — it will NOT misread a duration variable as Hz."""
    vl = _variable_line(seq_text)
    if vl:
        name = vl[0].lower()
        _, u0 = _num(vl[1], "")
        if ("freq" in name or "detun" in name
                or u0 in ("ghz", "mhz", "khz", "hz")):
            s, su = _num(vl[1], "hz")
            e, eu = _num(vl[2], "hz")
            sc = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}
            if np.isfinite(s) and np.isfinite(e):
                return np.linspace(s * sc.get(su, 1), e * sc.get(eu, 1), n), "frequency (Hz)"
    return np.arange(n, dtype=float), "scan point (index)"


def detect_pulsed_mode(seq_text, has_freq_array):
    t = (seq_text or "").lower()
    typ = None
    m = re.search(r"type\s*=\s*(\w+)", t)
    if m:
        typ = m.group(1)
    vl = _variable_line(seq_text)
    varname = vl[0] if vl else ""
    if "phase" in varname:               # phase sweep of any experiment type
        return "phase"
    if typ in ("ramsey",):
        return "ramsey"
    if typ in ("echo", "hahn", "hahn_echo", "spinecho"):
        return "echo"
    if typ in ("odmr", "pulsed_odmr", "podmr") or has_freq_array:
        return "odmr"
    if "amplitude" in varname:
        return "rabi_amp"
    if "duration" in varname or "width" in varname or typ == "rabi":
        return "rabi_time"
    return "rabi_time"


# =============================================================================
# Pulse-sequence preview: parse the sequence_text into pulses and evaluate their
# timing at a chosen scan point, so the GUI can draw a channel-vs-time diagram.
# This mirrors the sequence-text format used by our SequenceTextParser, but is
# fully self-contained (it does not import our src/ engine).
# =============================================================================
_SEQ_FUNCS = {"sin": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp,
              "sqrt": np.sqrt, "abs": abs, "pi": np.pi, "e": np.e}
_TIME_UNIT_S = {"ns": 1e-9, "us": 1e-6, "µs": 1e-6, "μs": 1e-6, "ms": 1e-3, "s": 1.0}


def parse_sequence_text(text):
    """Parse a sequence_text block into {name, type, duration_ns, sample_rate,
    variables:[{name,start,stop,unit,steps}], pulses:[{name,channel,at,shape,
    width,amp}]}.  Time/width/amp are kept as raw expression strings and only
    evaluated later (they may depend on the swept variable)."""
    if not text or not isinstance(text, str):
        return None
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    info = {"name": None, "type": None, "duration_ns": None,
            "sample_rate": None, "variables": [], "pulses": []}
    for ln in lines:
        low = ln.lower()
        if low.startswith("sequence:"):
            for key in ("name", "type"):
                m = re.search(key + r"\s*=\s*([^,]+)", ln, re.I)
                if m:
                    info[key] = m.group(1).strip()
            m = re.search(r"duration\s*=\s*([0-9.]+)\s*(ns|us|µs|μs|ms|s)?", ln, re.I)
            if m:
                u = (m.group(2) or "ns").lower()
                info["duration_ns"] = float(m.group(1)) * _TIME_UNIT_S.get(u, 1e-9) * 1e9
            m = re.search(r"sample_rate\s*=\s*([^,]+)", ln, re.I)
            if m:
                info["sample_rate"] = m.group(1).strip()
        elif low.startswith("variable"):
            m = re.search(r"variable\s+(\w+)\s*,\s*start\s*=\s*([^,]+),\s*"
                          r"stop\s*=\s*([^,]+?)(?:,\s*steps\s*=\s*(\d+))?\s*$", ln, re.I)
            if m:
                s, su = _num(m.group(2), "")
                e, eu = _num(m.group(3), "")
                info["variables"].append(
                    {"name": m.group(1).lower(), "start": s, "stop": e,
                     "unit": (su or eu or ""), "steps": int(m.group(4) or 1)})
        elif "pulse" in low and "channel" in low:
            m = re.match(r"(?P<name>\S+)\s+pulse\s+on\s+channel\s+(?P<ch>\d+)"
                         r"\s+at\s+(?P<rest>.+)$", ln, re.I)
            if m:
                parts = [p.strip() for p in m.group("rest").split(",")]
                if len(parts) >= 4:
                    info["pulses"].append(
                        {"name": m.group("name"), "channel": int(m.group("ch")),
                         "at": parts[0], "shape": parts[1],
                         "width": parts[2], "amp": parts[3]})
    return info


def is_sequence_text(val):
    """True if a value looks like a pulse-sequence description we can preview."""
    if not isinstance(val, str):
        return False
    low = val.lower()
    return ("pulse" in low and "channel" in low) or \
           ("sequence:" in low and "variable" in low)


def _seq_units_to_seconds(expr):
    def repl(m):
        return repr(float(m.group(1)) * _TIME_UNIT_S[m.group(2).lower()])
    return re.sub(r"([0-9]*\.?[0-9]+)\s*(ns|us|µs|μs|ms|s)\b", repl, str(expr))


def _eval_seq_expr(expr, subs):
    """Safely evaluate a sequence expression.  Bare numbers are treated as
    seconds; unit-suffixed numbers convert to seconds; names come from subs or
    the small math whitelist.  Returns a float (seconds for time expressions)."""
    e = _seq_units_to_seconds(expr)
    if "__" in e or not re.fullmatch(r"[0-9a-zA-Z_.eE+\-*/() ,]*", e or ""):
        raise ValueError(f"unsafe or unparseable expression: {expr!r}")
    ns = dict(_SEQ_FUNCS)
    ns.update(subs)
    return float(eval(e, {"__builtins__": {}}, ns))  # noqa: S307 (whitelisted)


def sequence_scan_value(parsed, scan_index):
    """Value of the (first) swept variable at a given scan-point index."""
    vs = parsed.get("variables") if parsed else None
    if not vs:
        return None, None, "", 1, 0
    v = vs[0]
    steps = max(1, int(v["steps"]))
    i = int(np.clip(scan_index, 0, steps - 1))
    if steps > 1:
        val = v["start"] + (v["stop"] - v["start"]) * i / (steps - 1)
    else:
        val = v["start"]
    return v["name"], val, v["unit"], steps, i


def sequence_pulse_geometry(parsed, scan_index=0):
    """Evaluate every pulse at the given scan point; return numeric geometry
    (start/width in ns, amplitude) plus the sorted channel list and time span."""
    if not parsed or not parsed.get("pulses"):
        return None
    var_name, var_val, var_unit, steps, idx = sequence_scan_value(parsed, scan_index)
    subs_time, subs_amp = {}, {}
    for k, v in enumerate(parsed.get("variables", [])):
        val = (v["start"] + (v["stop"] - v["start"]) * idx / (steps - 1)
               if (k == 0 and steps > 1) else v["start"])
        subs_time[v["name"]] = val * _TIME_UNIT_S.get(v["unit"], 1.0)
        subs_amp[v["name"]] = val
    pulses = []
    for p in parsed["pulses"]:
        try:
            start_ns = _eval_seq_expr(p["at"], subs_time) * 1e9
            width_ns = _eval_seq_expr(p["width"], subs_time) * 1e9
        except Exception:
            continue
        try:
            amp = _eval_seq_expr(p["amp"], subs_amp)
        except Exception:
            amp = 1.0
        pulses.append({"name": p["name"], "channel": p["channel"],
                       "start_ns": float(start_ns),
                       "width_ns": float(max(width_ns, 0.0)), "amp": float(amp)})
    channels = sorted({p["channel"] for p in pulses})
    t_max = max((p["start_ns"] + p["width_ns"] for p in pulses), default=0.0)
    if parsed.get("duration_ns"):
        t_max = max(t_max, parsed["duration_ns"])
    return {"var_name": var_name, "var_value": var_val, "var_unit": var_unit,
            "scan_index": idx, "steps": steps, "pulses": pulses,
            "channels": channels, "t_max_ns": t_max}


# =============================================================================
# Fit-model library (pure). Each model reports params, 1-sigma errors, R^2.
# =============================================================================
def _as_xy(x, y):
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    good = np.isfinite(x) & np.isfinite(y)
    return x[good], y[good]


def _osc_freq_seed(x, y):
    order = np.argsort(x)
    t, v = np.asarray(x)[order], np.asarray(y)[order]
    if len(t) < 4 or (t.max() - t.min()) == 0:
        return 1.0
    grid = np.linspace(t.min(), t.max(), max(64, 4 * len(t)))
    vi = np.interp(grid, t, v) - np.mean(v)
    spec = np.abs(np.fft.rfft(vi))
    spec[0] = 0.0
    fr = np.fft.rfftfreq(len(grid), d=grid[1] - grid[0])
    f = fr[int(np.argmax(spec))]
    return f if f > 0 else 1.0 / (t.max() - t.min() + 1e-9)


# --- model functions -------------------------------------------------------
def m_constant(x, C):
    return np.full_like(np.asarray(x, float), C)


def m_linear(x, a, b):
    return a * x + b


def m_gauss_peak(x, C, A, x0, s):
    return C + A * np.exp(-((x - x0) ** 2) / (2 * s ** 2))


def m_gauss_dip(x, C, A, x0, s):
    return C - A * np.exp(-((x - x0) ** 2) / (2 * s ** 2))


def m_gauss2_peak(x, C, A1, x1, s1, A2, x2, s2):
    return (C + A1 * np.exp(-((x - x1) ** 2) / (2 * s1 ** 2))
              + A2 * np.exp(-((x - x2) ** 2) / (2 * s2 ** 2)))


def m_lorentz_peak(x, C, A, x0, hw):
    return C + A * (hw ** 2) / ((x - x0) ** 2 + hw ** 2)


def m_lorentz_dip(x, C, A, x0, hw):
    return C - A * (hw ** 2) / ((x - x0) ** 2 + hw ** 2)


def m_lorentz2_dip(x, C, A1, x1, w1, A2, x2, w2):
    return (C - A1 * (w1 ** 2) / ((x - x1) ** 2 + w1 ** 2)
              - A2 * (w2 ** 2) / ((x - x2) ** 2 + w2 ** 2))


def m_exp_decay(x, C, A, tau):
    return C + A * np.exp(-x / tau)


def m_exp_rise(x, C, A, tau):
    return C + A * (1.0 - np.exp(-x / tau))


def m_damped_cos(x, C, A, f, phi, tau):
    return C + A * np.cos(2 * np.pi * f * x + phi) * np.exp(-x / tau)


def m_gauss_cos(x, C, A, f, phi, T):
    return C + A * np.cos(2 * np.pi * f * x + phi) * np.exp(-((x / T) ** 2))


def m_stretched(x, C, A, T, p):
    return C + A * np.exp(-((x / T) ** p))


def m_sine(x, C, A, f, phi):
    return C + A * np.sin(2 * np.pi * f * x + phi)


def m_phase_cos(x, C, A, phi0):
    # phase sweep: one cosine fringe over the swept phase (x in radians)
    return C + A * np.cos(x - phi0)


def _span(x):
    return (x.max() - x.min()) or 1.0


def _seed_constant(x, y):
    return [np.mean(y)], (None, None)


def _seed_linear(x, y):
    if len(x) > 1 and _span(x) > 0:
        a, b = np.polyfit(x, y, 1)
        return [float(a), float(b)], (None, None)
    return [0.0, float(np.mean(y))], (None, None)


def _seed_gauss(x, y, dip):
    C = float(np.median(y))
    amp = float((C - y.min()) if dip else (y.max() - C))
    x0 = float(x[np.argmin(y)] if dip else x[np.argmax(y)])
    s = _span(x) / 10
    lo = [-np.inf, 0, x.min(), 1e-9 * _span(x) + 1e-12]
    hi = [np.inf, np.inf, x.max(), _span(x)]
    return [C, max(amp, 1e-9), x0, s], (lo, hi)


def _seed_gauss2(x, y):
    C = float(np.median(y)); amp = float(y.max() - C)
    x1 = x.min() + _span(x) / 3; x2 = x.min() + 2 * _span(x) / 3
    s = _span(x) / 12
    return [C, max(amp, 1e-9), x1, s, max(amp, 1e-9), x2, s], (None, None)


def _seed_lorentz(x, y, dip):
    C = float(np.median(y))
    amp = float((C - y.min()) if dip else (y.max() - C))
    x0 = float(x[np.argmin(y)] if dip else x[np.argmax(y)])
    hw = _span(x) / 20
    lo = [-np.inf, 0, x.min(), 1e-6 * _span(x) + 1e-12]
    hi = [np.inf, np.inf, x.max(), _span(x)]
    return [C, max(amp, 1e-9), x0, hw], (lo, hi)


def _seed_lorentz2(x, y):
    C = float(np.median(y)); d = float(max(C - y.min(), 1e-9))
    x1 = float(x.min() + _span(x) / 3)          # generic two-dip guess
    x2 = float(x.min() + 2 * _span(x) / 3)       # (e.g. Zeeman-split ODMR)
    w = _span(x) / 20
    wmin = 1e-6 * _span(x) + 1e-12
    lo = [-np.inf, 0, x.min(), wmin, 0, x.min(), wmin]
    hi = [np.inf, np.inf, x.max(), _span(x), np.inf, x.max(), _span(x)]
    return [C, d, x1, w, d, x2, w], (lo, hi)


def _seed_exp(x, y, rise):
    C = float(y[-1] if not rise else y[0])
    A = float((y[0] - y[-1]) if not rise else (y[-1] - y[0]))
    tau = max(_span(x) / 3, 1e-9)
    return [C, A, tau], ([-np.inf, -np.inf, 1e-12], [np.inf, np.inf, np.inf])


def _seed_damped_cos(x, y):
    f = _osc_freq_seed(x, y)
    A = 0.5 * (np.max(y) - np.min(y))
    return ([float(np.mean(y)), float(A), float(f), 0.0, max(_span(x), 1e-9)],
            ([-np.inf, 0, 0, -np.pi, 1e-9],
             [np.inf, np.inf, np.inf, np.pi, np.inf]))


def _seed_gauss_cos(x, y):
    f = _osc_freq_seed(x, y)
    A = 0.5 * (np.max(y) - np.min(y))
    return ([float(np.mean(y)), float(A), float(f), 0.0, max(_span(x) / 2, 1e-9)],
            ([-np.inf, 0, 0, -np.pi, 1e-9],
             [np.inf, np.inf, np.inf, np.pi, np.inf]))


def _seed_stretched(x, y):
    rising = np.mean(y[:3]) < np.mean(y[-3:])
    A = (y.min() - y.max()) if rising else (y.max() - y.min())
    return ([float(np.median(y)), float(A), max(_span(x) / 2, 1e-9), 1.5],
            ([-np.inf, -np.inf, 1e-9, 0.3], [np.inf, np.inf, np.inf, 5.0]))


def _seed_sine(x, y):
    f = _osc_freq_seed(x, y)
    A = 0.5 * (np.max(y) - np.min(y))
    return ([float(np.mean(y)), float(A), float(f), 0.0],
            ([-np.inf, 0, 0, -np.pi], [np.inf, np.inf, np.inf, np.pi]))


def _seed_phase_cos(x, y):
    C = float(np.mean(y))
    A = 0.5 * (np.max(y) - np.min(y))
    phi0 = float(np.asarray(x)[int(np.argmax(y))])   # phase where signal peaks
    return ([C, A, phi0], ([-np.inf, 0, -2 * np.pi], [np.inf, np.inf, 2 * np.pi]))


def _rep_lorentz(p, dip):
    C, A, x0, hw = p
    return [f"center x0 = {x0:.6g}", f"FWHM = {2*abs(hw):.6g} (x-units)",
            f"amplitude = {A:.4g}", f"offset C = {C:.4g}"]


def _rep_lorentz2(p):
    C, A1, x1, w1, A2, x2, w2 = p
    return [f"peak 1 center = {x1:.6g},  FWHM = {2*abs(w1):.6g}",
            f"peak 2 center = {x2:.6g},  FWHM = {2*abs(w2):.6g}",
            f"splitting = {abs(x2-x1):.6g} (x-units)"]


def _rep_gauss(p, dip):
    C, A, x0, s = p
    return [f"center x0 = {x0:.6g}", f"sigma = {abs(s):.6g}",
            f"FWHM = {2.3548*abs(s):.6g} (x-units)", f"amplitude = {A:.4g}"]


def _rep_osc(p):
    C, A, f, phi, tau = p
    out = [f"frequency f = {f:.6g} (cycles per x-unit)",
           f"period 1/f = {1/f:.6g} (x-units)" if f else "period = inf",
           f"amplitude = {A:.4g}", f"decay = {tau:.6g} (x-units)"]
    return out


def _rep_stretched(p):
    C, A, T, n = p
    return [f"decay constant T = {T:.6g} (x-units)", f"stretch n = {n:.3f}",
            f"amplitude = {A:.4g}"]


def _rep_exp(p):
    C, A, tau = p
    return [f"tau = {tau:.6g} (x-units)", f"amplitude = {A:.4g}",
            f"offset C = {C:.4g}"]


def _rep_phase_cos(p):
    C, A, phi0 = p
    phi0w = (phi0 + np.pi) % (2 * np.pi) - np.pi          # wrap to [-pi, pi]
    vis = abs(A) / abs(C) if C else float("nan")
    return [f"fringe amplitude A = {abs(A):.4g}",
            f"phase offset phi0 = {phi0w:.4g} rad  ({np.degrees(phi0w):.1f} deg)",
            f"offset C = {C:.4g}",
            f"visibility A/C = {vis:.3g}"]


MODELS = {
    # name: (func, pnames, seed_fn, report_fn)
    "Constant":           (m_constant, ["C"], _seed_constant, None),
    "Linear":             (m_linear, ["a", "b"], _seed_linear, None),
    "Gaussian peak":      (m_gauss_peak, ["C", "A", "x0", "sigma"],
                           lambda x, y: _seed_gauss(x, y, False),
                           lambda p: _rep_gauss(p, False)),
    "Gaussian dip":       (m_gauss_dip, ["C", "A", "x0", "sigma"],
                           lambda x, y: _seed_gauss(x, y, True),
                           lambda p: _rep_gauss(p, True)),
    "Double Gaussian peak": (m_gauss2_peak,
                           ["C", "A1", "x1", "s1", "A2", "x2", "s2"],
                           _seed_gauss2, None),
    "Lorentzian peak":    (m_lorentz_peak, ["C", "A", "x0", "hw"],
                           lambda x, y: _seed_lorentz(x, y, False),
                           lambda p: _rep_lorentz(p, False)),
    "Lorentzian dip (ODMR)": (m_lorentz_dip, ["C", "A", "x0", "hw"],
                           lambda x, y: _seed_lorentz(x, y, True),
                           lambda p: _rep_lorentz(p, True)),
    "Double Lorentzian dip": (m_lorentz2_dip,
                           ["C", "A1", "x1", "w1", "A2", "x2", "w2"],
                           _seed_lorentz2, _rep_lorentz2),
    "Exponential decay":  (m_exp_decay, ["C", "A", "tau"],
                           lambda x, y: _seed_exp(x, y, False), _rep_exp),
    "Exponential rise":   (m_exp_rise, ["C", "A", "tau"],
                           lambda x, y: _seed_exp(x, y, True), _rep_exp),
    "Damped cosine (Rabi)": (m_damped_cos, ["C", "A", "f", "phi", "tau"],
                           _seed_damped_cos, _rep_osc),
    "Ramsey (Gaussian×cos)": (m_gauss_cos, ["C", "A", "f", "phi", "T"],
                           _seed_gauss_cos,
                           lambda p: [f"fringe f = {p[2]:.6g}/x-unit",
                                      f"T2* = {p[4]:.6g} (x-units)"]),
    "Stretched exp (echo)": (m_stretched, ["C", "A", "T", "n"],
                           _seed_stretched, _rep_stretched),
    "Sine":               (m_sine, ["C", "A", "f", "phi"], _seed_sine, None),
    "Cosine (phase sweep)": (m_phase_cos, ["C", "A", "phi0"],
                           _seed_phase_cos, _rep_phase_cos),
    "Polynomial (deg 2)": ("poly2", None, None, None),
    "Polynomial (deg 3)": ("poly3", None, None, None),
    "Polynomial (deg 4)": ("poly4", None, None, None),
}


def r_squared(y, yhat):
    y = np.asarray(y, float)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def fit_model(name, x, y, n_dense=800):
    """Fit `name` (a key in MODELS) to (x, y). Returns a dict with keys:
       ok, params, perr, pnames, r2, x_dense, y_dense, y_fit, summary, error."""
    x, y = _as_xy(x, y)
    if len(x) < 2:
        return {"ok": False, "error": "not enough finite points to fit"}
    spec = MODELS.get(name)
    if spec is None:
        return {"ok": False, "error": f"unknown model {name!r}"}
    func, pnames, seed_fn, report_fn = spec

    order = np.argsort(x)
    xs, ys = x[order], y[order]
    xd = np.linspace(xs.min(), xs.max(), n_dense)

    # polynomial special case
    if isinstance(func, str) and func.startswith("poly"):
        deg = int(func[-1])
        if len(xs) <= deg:
            return {"ok": False, "error": f"need > {deg} points for degree {deg}"}
        coeffs = np.polyfit(xs, ys, deg)
        yhat = np.polyval(coeffs, xs)
        pn = [f"c{deg-i}" for i in range(deg + 1)]
        rep = [f"{pn[i]} = {coeffs[i]:.6g}" for i in range(len(coeffs))]
        return {"ok": True, "params": coeffs, "perr": np.full(len(coeffs), np.nan),
                "pnames": pn, "r2": r_squared(ys, yhat),
                "x_dense": xd, "y_dense": np.polyval(coeffs, xd),
                "y_fit": np.polyval(coeffs, x), "report": rep,
                "summary": _format_summary(name, pn, coeffs,
                                           np.full(len(coeffs), np.nan),
                                           r_squared(ys, yhat), rep)}

    p0, bounds = seed_fn(xs, ys)
    kw = dict(p0=p0, maxfev=60000)
    if bounds and bounds[0] is not None:
        kw["bounds"] = bounds
    try:
        popt, pcov = curve_fit(func, xs, ys, **kw)
    except Exception as e:
        return {"ok": False, "error": f"fit did not converge: {e}"}
    perr = np.sqrt(np.abs(np.diag(pcov))) if pcov is not None else \
        np.full(len(popt), np.nan)
    yhat = func(xs, *popt)
    rep = report_fn(popt) if report_fn else None
    return {"ok": True, "params": popt, "perr": perr, "pnames": pnames,
            "r2": r_squared(ys, yhat), "x_dense": xd, "y_dense": func(xd, *popt),
            "y_fit": func(x, *popt), "report": rep,
            "summary": _format_summary(name, pnames, popt, perr,
                                       r_squared(ys, yhat), rep)}


def _format_summary(name, pnames, popt, perr, r2, report):
    lines = [f"Model: {name}", f"R\u00b2 = {r2:.5f}", "", "parameters:"]
    for nm, val, err in zip(pnames, popt, perr):
        if np.isfinite(err):
            lines.append(f"  {nm:>5} = {val:.6g}  \u00b1 {err:.3g}")
        else:
            lines.append(f"  {nm:>5} = {val:.6g}")
    if report:
        lines += ["", "derived:"] + [f"  {r}" for r in report]
    return "\n".join(lines)


def suggest_models(mode):
    """Order the model list so the most likely one for `mode` is first."""
    order = list(MODELS.keys())
    pref = {
        "cw_odmr": ["Lorentzian dip (ODMR)", "Double Lorentzian dip"],
        "odmr":    ["Lorentzian dip (ODMR)", "Double Lorentzian dip"],
        "rabi_time": ["Damped cosine (Rabi)"],
        "rabi_amp":  ["Damped cosine (Rabi)"],
        "ramsey":  ["Ramsey (Gaussian×cos)"],
        "echo":    ["Stretched exp (echo)"],
    }.get(mode, [])
    for p in reversed(pref):
        if p in order:
            order.remove(p)
            order.insert(0, p)
    return order


# =============================================================================
# Selection / region analysis (pure)
# =============================================================================
def span_mask(x, x0, x1):
    x = np.asarray(x, float)
    lo, hi = (x0, x1) if x0 <= x1 else (x1, x0)
    return (x >= lo) & (x <= hi)


def region_stats_1d(x, y, x0=None, x1=None):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x0 is not None and x1 is not None:
        m = span_mask(x, x0, x1)
        x, y = x[m], y[m]
    good = np.isfinite(y)
    x, y = x[good], y[good]
    if len(y) == 0:
        return {"n": 0}
    order = np.argsort(x)
    area = float(np.trapezoid(y[order], x[order])) if len(y) > 1 else 0.0
    return {"n": int(len(y)), "x_min": float(x.min()), "x_max": float(x.max()),
            "mean": float(np.mean(y)), "std": float(np.std(y)),
            "min": float(np.min(y)), "max": float(np.max(y)),
            "sum": float(np.sum(y)), "area": area,
            "argmax_x": float(x[np.argmax(y)]), "argmin_x": float(x[np.argmin(y)])}


def format_stats_1d(s):
    if s.get("n", 0) == 0:
        return "selection is empty"
    return ("\n".join([
        f"points            {s['n']}",
        f"x range           {s['x_min']:.6g} … {s['x_max']:.6g}",
        f"mean ± std        {s['mean']:.6g} ± {s['std']:.4g}",
        f"min / max         {s['min']:.6g} / {s['max']:.6g}",
        f"x at max / min    {s['argmax_x']:.6g} / {s['argmin_x']:.6g}",
        f"sum               {s['sum']:.6g}",
        f"integral (trapz)  {s['area']:.6g}",
    ]))


def image_crop(img, x0, x1, y0, y1):
    """Crop a 2-D image by pixel-coordinate box (floats accepted). Returns
    (sub, (c0, c1, r0, r1)) with integer bounds actually used."""
    img = np.asarray(img)
    h, w = img.shape[:2]
    c0, c1 = sorted((int(round(x0)), int(round(x1))))
    r0, r1 = sorted((int(round(y0)), int(round(y1))))
    c0 = max(0, min(c0, w - 1)); c1 = max(c0 + 1, min(c1, w))
    r0 = max(0, min(r0, h - 1)); r1 = max(r0 + 1, min(r1, h))
    return img[r0:r1, c0:c1], (c0, c1, r0, r1)


def image_profiles(img, box=None):
    """Return (x_idx, col_profile, y_idx, row_profile) averaged over the box
    (or the whole image). col_profile is mean over rows -> vs x; row_profile is
    mean over cols -> vs y."""
    img = np.asarray(img, float)
    if box is not None:
        sub, (c0, c1, r0, r1) = image_crop(img, *box)
    else:
        sub, (c0, c1, r0, r1) = img, (0, img.shape[1], 0, img.shape[0])
    col = np.nanmean(sub, axis=0)          # vs x (columns)
    row = np.nanmean(sub, axis=1)          # vs y (rows)
    return (np.arange(c0, c1), col, np.arange(r0, r1), row)


def region_stats_2d(img, box=None):
    img = np.asarray(img, float)
    if box is not None:
        sub, bounds = image_crop(img, *box)
    else:
        sub, bounds = img, (0, img.shape[1], 0, img.shape[0])
    c0, c1, r0, r1 = bounds
    v = sub[np.isfinite(sub)]
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "cols": f"{c0}:{c1}", "rows": f"{r0}:{r1}",
            "mean": float(np.mean(v)), "std": float(np.std(v)),
            "min": float(np.min(v)), "max": float(np.max(v)),
            "sum": float(np.sum(v))}


def format_stats_2d(s):
    if s.get("n", 0) == 0:
        return "ROI is empty"
    return "\n".join([
        f"pixels        {s['n']}   (rows {s['rows']}, cols {s['cols']})",
        f"mean ± std    {s['mean']:.6g} ± {s['std']:.4g}",
        f"min / max     {s['min']:.6g} / {s['max']:.6g}",
        f"sum           {s['sum']:.6g}",
    ])


def stack_projection(frames, how="mean"):
    frames = np.asarray(frames, float)
    if how == "mean":
        return np.nanmean(frames, axis=0)
    if how == "max":
        return np.nanmax(frames, axis=0)
    if how == "sum":
        return np.nansum(frames, axis=0)
    return frames[0]


# =============================================================================
# Significance test (uses our significance_test.py if importable; else built-in)
# =============================================================================
def run_significance(mode, x, y, n_perm=None, seed=0):
    """Return a dict describing an 'is it real?' test for the given mode.
    Oscillation modes (rabi/ramsey) -> Lomb-Scargle permutation FAP.
    Shape modes (odmr dip / echo decay) -> shape-fit R^2 permutation FAP."""
    x, y = _as_xy(x, y)
    if len(x) < 6:
        return {"ok": False, "error": "need at least ~6 points for a permutation test"}
    rng = np.random.default_rng(seed)

    # try the user's module first (identical method, their tuning)
    user = None
    try:
        import significance_test as st  # noqa
        user = st
    except Exception:
        user = None

    osc = mode in ("rabi_time", "rabi_amp", "ramsey", "phase")
    if osc:
        nperm = n_perm or 2000
        use_user = user is not None and mode != "phase"   # no phase grid upstream
        if use_user:
            f, ang, unit, fnyq = user._freq_grid(mode, x)
            pg, obs, fap, thr = user.periodogram_fap(x, y, ang, n_perm=nperm, rng=rng)
            fpk = f[int(np.argmax(pg))]
        else:
            f, ang, unit, fnyq = _freq_grid(mode, x)
            pg, obs, fap, thr = _periodogram_fap(x, y, ang, n_perm=nperm, rng=rng)
            fpk = f[int(np.argmax(pg))]
        return {"ok": True, "kind": "periodogram", "unit": unit, "peak": float(fpk),
                "power": float(obs), "thr": float(thr), "fap": float(fap),
                "f": f, "pg": pg, "verdict": _verdict(fap), "nyquist": float(fnyq),
                "n_perm": nperm}
    else:
        nperm = n_perm or 1000
        if user is not None:
            fit_fn = user._fit_lorentz if mode == "odmr" else user._fit_stretched
            r2, fap, thr, p = user.fit_r2_fap(x, y, fit_fn, n_perm=nperm, rng=rng)
        else:
            fit_fn = _fit_lorentz if mode == "odmr" else _fit_stretched
            r2, fap, thr, p = _fit_r2_fap(x, y, fit_fn, n_perm=nperm, rng=rng)
        return {"ok": True, "kind": "shapefit", "r2": float(r2), "thr": float(thr),
                "fap": float(fap), "verdict": _verdict(fap), "params": p,
                "n_perm": nperm}


def _verdict(fap):
    if fap < 0.01:
        return "REAL  (p < 1%)"
    if fap < 0.05:
        return f"MARGINAL  (p = {fap*100:.1f}%) — average more before trusting it"
    return f"NOT significant  (p = {fap*100:.0f}%) — consistent with noise"


def _freq_grid(mode, x, npts=1400):
    dx = np.median(np.diff(np.sort(x)))
    if mode == "phase":                          # x is phase in radians
        f_nyq = 0.5 / dx
        f = np.linspace(0.02, min(2.0, 0.95 * f_nyq), npts)
        return f, 2 * np.pi * f, "cyc/rad", f_nyq
    if mode == "rabi_amp":
        f_nyq = 0.5 / dx
        f = np.linspace(0.1, min(10.0, 0.95 * f_nyq), npts)
        return f, 2 * np.pi * f, "cyc/V", f_nyq
    f_nyq = 1000.0 / (2 * dx)
    f = np.linspace(0.5, min(50.0, 0.95 * f_nyq), npts)
    return f, 2 * np.pi * f / 1000.0, "MHz", f_nyq


def _periodogram_fap(x, y, ang, n_perm=2000, rng=None):
    from scipy.signal import lombscargle
    rng = rng or np.random.default_rng(0)
    y = y - np.mean(y)
    pg = lombscargle(x, y, ang, normalize=True)
    obs = pg.max()
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = lombscargle(x, rng.permutation(y), ang, normalize=True).max()
    fap = (np.sum(null >= obs) + 1) / (n_perm + 1)
    return pg, obs, fap, np.percentile(null, 95)


def _fit_lorentz(x_hz, y):
    fM = x_hz / 1e6 if np.nanmax(np.abs(x_hz)) > 1e5 else x_hz
    C0 = np.median(y); d0 = C0 - np.min(y); f0 = fM[int(np.argmin(y))]
    span = fM.max() - fM.min()
    p, _ = curve_fit(m_lorentz_dip, fM, y,
                     p0=[C0, max(d0, 1e-6), f0, max(span / 20, 1.0)],
                     bounds=([-2, 0, fM.min(), 0.1], [2, 2, fM.max(), span]),
                     maxfev=40000)
    return m_lorentz_dip(fM, *p), p


def _fit_stretched(x_ns, y):
    xu = x_ns / 1e3 if np.nanmax(np.abs(x_ns)) > 100 else x_ns
    rising = np.mean(y[:3]) < np.mean(y[-3:])
    A0 = (np.min(y) - np.max(y)) if rising else (np.max(y) - np.min(y))
    p, _ = curve_fit(m_stretched, xu, y,
                     p0=[float(np.median(y)), A0,
                         0.4 * (xu.max() - xu.min()) + 1e-3, 1.5],
                     bounds=([-2, -2, 1e-4, 0.5], [2, 2, 1e6, 4.0]), maxfev=40000)
    return m_stretched(xu, *p), p


def _fit_r2_fap(x, y, fit_fn, n_perm=1000, rng=None):
    rng = rng or np.random.default_rng(0)

    def r2_of(xx, yy):
        yh, p = fit_fn(xx, yy)
        return 1 - np.sum((yy - yh) ** 2) / np.sum((yy - np.mean(yy)) ** 2), p

    r2, p = r2_of(x, y)
    null = np.empty(n_perm)
    for i in range(n_perm):
        try:
            null[i] = r2_of(x, rng.permutation(y))[0]
        except Exception:
            null[i] = 0.0
    fap = (np.sum(null >= r2) + 1) / (n_perm + 1)
    return r2, fap, np.percentile(null, 95), p


# =============================================================================
# ============================  GUI LAYER  ====================================
# (All tkinter / matplotlib imports live here so the module stays importable.)
# =============================================================================
def main(argv=None):
    import sys
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import (
        FigureCanvasTkAgg, NavigationToolbar2Tk)
    from matplotlib.widgets import SpanSelector, RectangleSelector

    COLORMAPS = ["viridis", "plasma", "inferno", "magma", "cividis",
                 "gray", "hot", "jet", "turbo", "twilight", "coolwarm"]

    def _make_span(ax, cb):
        try:
            return SpanSelector(ax, cb, "horizontal", useblit=True,
                                interactive=True, drag_from_anywhere=True,
                                props=dict(alpha=0.2, facecolor="tab:orange"))
        except TypeError:
            return SpanSelector(ax, cb, "horizontal", useblit=True,
                                rectprops=dict(alpha=0.2, facecolor="tab:orange"))

    def _make_rect(ax, cb):
        try:
            return RectangleSelector(ax, cb, useblit=True, button=[1],
                                     interactive=True, minspanx=2, minspany=2,
                                     spancoords="pixels",
                                     props=dict(edgecolor="white", fill=False, lw=1.3))
        except TypeError:
            return RectangleSelector(ax, cb, useblit=True, button=[1],
                                     interactive=True, minspanx=2, minspany=2,
                                     spancoords="pixels",
                                     rectprops=dict(edgecolor="white", fill=False, lw=1.3))

    # ---------------------------------------------------------------- app ----
    class App:
        def __init__(self, root):
            self.root = root
            root.title(APP_TITLE)
            root.geometry("1320x820")
            root.minsize(1060, 640)

            self.tree_data = None          # nested dict from load_hdf5_tree
            self.filepath = None
            self.item_values = {}          # treeview item id -> value
            self.stacks = []               # detected image stacks

            # current plot state
            self.view_kind = None          # '1d' | 'image' | 'stack' | 'sequence' | None
            self.x = None
            self.y = None
            self.curves = None             # list of (x, y, label) for overlays; None = single
            self.xlabel = ""
            self.ylabel = ""
            self.cur_image = None          # 2-D array currently shown
            self.cur_frames = None         # (N,H,W) for stacks
            self.frame_idx = 0
            self.playing = False
            self.seq_parsed = None         # parsed sequence_text for the preview
            self.seq_index = 0             # current scan point in the preview
            self.sel_x0 = self.sel_x1 = None       # 1-D span
            self.sel_box = None                    # 2-D ROI (x0,x1,y0,y1)
            self.pulsed_mode = "rabi_time"
            self.pulsed_cache = None       # dict(sig, ref, seq_text, has_freq...)

            self._build_layout()
            self._new_axes()

            args = argv if argv is not None else sys.argv[1:]
            if args and os.path.exists(args[0]):
                self.open_path(args[0])

        # ----------------------------------------------------------- layout --
        def _build_layout(self):
            top = ttk.Frame(self.root, padding=(8, 6))
            top.pack(side="top", fill="x")
            ttk.Button(top, text="Open HDF5…", command=self.open_dialog).pack(side="left")
            ttk.Button(top, text="Reload", command=self.reload).pack(side="left", padx=(6, 0))
            self.file_lbl = ttk.Label(top, text="no file loaded", foreground="#555")
            self.file_lbl.pack(side="left", padx=12)
            self.status = ttk.Label(top, text="", foreground="#0a6")
            self.status.pack(side="right")

            paned = ttk.PanedWindow(self.root, orient="horizontal")
            paned.pack(side="top", fill="both", expand=True)

            # -- left: data browser -------------------------------------------
            left = ttk.Frame(paned, padding=(6, 4))
            paned.add(left, weight=0)
            ttk.Label(left, text="Data browser", font=("", 10, "bold")).pack(anchor="w")
            tree_wrap = ttk.Frame(left)
            tree_wrap.pack(fill="both", expand=True)
            self.tree = ttk.Treeview(tree_wrap, show="tree", selectmode="extended")
            ysb = ttk.Scrollbar(tree_wrap, orient="vertical", command=self.tree.yview)
            self.tree.configure(yscrollcommand=ysb.set)
            self.tree.pack(side="left", fill="both", expand=True)
            ysb.pack(side="right", fill="y")
            self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)
            self.tree.bind("<Double-1>", lambda e: self.plot_selected_node())

            ttk.Label(left, text="x-axis for 1-D:").pack(anchor="w", pady=(8, 0))
            self.xaxis_combo = ttk.Combobox(left, state="readonly", width=32, values=[])
            self.xaxis_combo.pack(fill="x")
            self.xaxis_combo.bind("<<ComboboxSelected>>", lambda e: self.replot_1d())

            ttk.Button(left, text="Plot selected node",
                       command=self.plot_selected_node).pack(fill="x", pady=(6, 2))

            ttk.Label(left, text="Info", font=("", 9, "bold")).pack(anchor="w", pady=(6, 0))
            self.info = tk.Text(left, height=8, width=40, wrap="word",
                                font=("TkFixedFont", 9))
            self.info.pack(fill="x")
            self.info.configure(state="disabled")

            # -- right: figure + controls -------------------------------------
            right = ttk.Frame(paned)
            paned.add(right, weight=1)

            self.fig = Figure(figsize=(7.6, 5.4), dpi=100)
            self.canvas = FigureCanvasTkAgg(self.fig, master=right)
            self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
            toolbar = NavigationToolbar2Tk(self.canvas, right)
            toolbar.update()

            nb = ttk.Notebook(right)
            nb.pack(side="bottom", fill="x")
            self.nb = nb
            self._build_tab_plot(nb)
            self._build_tab_fit(nb)
            self._build_tab_pulsed(nb)
            self._build_tab_image(nb)
            self._build_tab_sequence(nb)
            self._seq_tab_index = nb.index("end") - 1

        def _build_tab_plot(self, nb):
            f = ttk.Frame(nb, padding=8)
            nb.add(f, text="Plot & selection")

            row = ttk.Frame(f); row.pack(fill="x")
            ttk.Label(row, text="style:").pack(side="left")
            self.style_var = tk.StringVar(value="line+markers")
            ttk.Combobox(row, textvariable=self.style_var, width=13, state="readonly",
                         values=["line", "markers", "line+markers"]).pack(side="left", padx=(2, 12))
            self.logy_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(row, text="log y", variable=self.logy_var,
                            command=self.replot_1d).pack(side="left")
            ttk.Label(row, text="normalise:").pack(side="left", padx=(12, 2))
            self.norm_var = tk.StringVar(value="none")
            ttk.Combobox(row, textvariable=self.norm_var, width=16, state="readonly",
                         values=["none", "divide by max",
                                 "subtract selection baseline"]).pack(side="left")
            ttk.Button(row, text="Apply", command=self.replot_1d).pack(side="left", padx=6)
            ttk.Button(row, text="Plot selected together",
                       command=self.plot_selected_together).pack(side="left", padx=(16, 0))

            row2 = ttk.Frame(f); row2.pack(fill="x", pady=(8, 0))
            ttk.Label(row2, text="Selection:").pack(side="left")
            self.sel_lbl = ttk.Label(row2, text="(drag on the plot to select a range)",
                                     foreground="#a60")
            self.sel_lbl.pack(side="left", padx=6)

            row3 = ttk.Frame(f); row3.pack(fill="x", pady=(6, 0))
            for txt, cmd in [("Zoom to selection", self.zoom_selection),
                             ("Selection stats", self.selection_stats),
                             ("Export selection CSV", self.export_selection),
                             ("Clear selection", self.clear_selection),
                             ("Reset view", self.reset_view)]:
                ttk.Button(row3, text=txt, command=cmd).pack(side="left", padx=(0, 6))

        def _build_tab_fit(self, nb):
            f = ttk.Frame(nb, padding=8)
            nb.add(f, text="Fit")
            row = ttk.Frame(f); row.pack(fill="x")
            ttk.Label(row, text="model:").pack(side="left")
            self.model_combo = ttk.Combobox(row, state="readonly", width=26,
                                             values=list(MODELS.keys()))
            self.model_combo.current(0)
            self.model_combo.pack(side="left", padx=(2, 12))
            self.fit_sel_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(row, text="fit selection only",
                            variable=self.fit_sel_var).pack(side="left")
            ttk.Button(row, text="Fit", command=self.do_fit).pack(side="left", padx=8)
            ttk.Button(row, text="Clear fit", command=self.clear_fit).pack(side="left")
            ttk.Button(row, text="Export fit CSV",
                       command=self.export_fit).pack(side="left", padx=(8, 0))
            self.fit_text = tk.Text(f, height=8, wrap="none", font=("TkFixedFont", 9))
            self.fit_text.pack(fill="both", expand=True, pady=(8, 0))
            self._last_fit = None

        def _build_tab_pulsed(self, nb):
            f = ttk.Frame(nb, padding=8)
            nb.add(f, text="Pulsed (Rabi / ODMR / echo)")
            row = ttk.Frame(f); row.pack(fill="x")
            ttk.Label(row, text="experiment:").pack(side="left")
            self.pmode_combo = ttk.Combobox(row, state="readonly", width=12,
                                            values=["rabi_time", "rabi_amp",
                                                    "odmr", "ramsey", "echo",
                                                    "phase"])
            self.pmode_combo.current(0)
            self.pmode_combo.pack(side="left", padx=(2, 10))
            ttk.Label(row, text="quantity:").pack(side="left")
            self.pq_combo = ttk.Combobox(row, state="readonly", width=10,
                                         values=["S/R", "(S-R)/R", "signal",
                                                 "reference", "S-R"])
            self.pq_combo.current(0)
            self.pq_combo.pack(side="left", padx=(2, 10))
            self.drop_dead_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(row, text="drop dead startup pts",
                            variable=self.drop_dead_var).pack(side="left")
            self.pulsed_sel_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(row, text="fit selection only",
                            variable=self.pulsed_sel_var).pack(side="left", padx=(10, 0))

            row2 = ttk.Frame(f); row2.pack(fill="x", pady=(6, 0))
            ttk.Button(row2, text="Analyse & fit",
                       command=self.pulsed_analyse).pack(side="left")
            ttk.Button(row2, text="Significance test",
                       command=self.pulsed_significance).pack(side="left", padx=8)
            ttk.Label(row2, text="(quantity + selection apply to both; select a "
                      "span on the Plot tab first)", foreground="#777").pack(side="left")
            self.pulsed_text = tk.Text(f, height=8, wrap="none", font=("TkFixedFont", 9))
            self.pulsed_text.pack(fill="both", expand=True, pady=(8, 0))

        def _build_tab_image(self, nb):
            f = ttk.Frame(nb, padding=8)
            nb.add(f, text="Image / movie")
            row = ttk.Frame(f); row.pack(fill="x")
            ttk.Label(row, text="colormap:").pack(side="left")
            self.cmap_combo = ttk.Combobox(row, state="readonly", width=10,
                                           values=COLORMAPS)
            self.cmap_combo.current(0)
            self.cmap_combo.pack(side="left", padx=(2, 12))
            self.cmap_combo.bind("<<ComboboxSelected>>", lambda e: self.redraw_image())
            ttk.Label(row, text="contrast clip %:").pack(side="left")
            self.clip_var = tk.DoubleVar(value=0.0)
            self.clip_scale = ttk.Scale(row, from_=0, to=10, variable=self.clip_var,
                                        length=140, command=lambda e: self.redraw_image())
            self.clip_scale.pack(side="left", padx=(2, 12))
            self.equal_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(row, text="equal aspect", variable=self.equal_var,
                            command=self.redraw_image).pack(side="left")

            row2 = ttk.Frame(f); row2.pack(fill="x", pady=(8, 0))
            ttk.Label(row2, text="ROI:").pack(side="left")
            self.roi_lbl = ttk.Label(row2, text="(drag a box on the image)",
                                     foreground="#a60")
            self.roi_lbl.pack(side="left", padx=6)
            for txt, cmd in [("Crop to ROI", self.crop_roi),
                             ("X profile", lambda: self.roi_profile("x")),
                             ("Y profile", lambda: self.roi_profile("y")),
                             ("ROI stats", self.roi_stats),
                             ("Clear ROI", self.clear_roi)]:
                ttk.Button(row2, text=txt, command=cmd).pack(side="left", padx=(6, 0))

            self.movie_row = ttk.Frame(f); self.movie_row.pack(fill="x", pady=(8, 0))
            ttk.Label(self.movie_row, text="frame:").pack(side="left")
            self.frame_scale = ttk.Scale(self.movie_row, from_=0, to=0,
                                         command=self.on_frame_scale, length=260)
            self.frame_scale.pack(side="left", padx=6)
            self.frame_lbl = ttk.Label(self.movie_row, text="-")
            self.frame_lbl.pack(side="left")
            ttk.Button(self.movie_row, text="◀", width=3,
                       command=lambda: self.step_frame(-1)).pack(side="left", padx=(10, 0))
            self.play_btn = ttk.Button(self.movie_row, text="▶ Play", command=self.toggle_play)
            self.play_btn.pack(side="left", padx=4)
            ttk.Button(self.movie_row, text="▶|", width=3,
                       command=lambda: self.step_frame(1)).pack(side="left")
            ttk.Label(self.movie_row, text="fps:").pack(side="left", padx=(10, 2))
            self.fps_var = tk.IntVar(value=8)
            ttk.Spinbox(self.movie_row, from_=1, to=60, width=4,
                        textvariable=self.fps_var).pack(side="left")
            ttk.Label(self.movie_row, text="projection:").pack(side="left", padx=(10, 2))
            self.proj_combo = ttk.Combobox(self.movie_row, state="readonly", width=8,
                                           values=["frame", "mean", "max", "sum"])
            self.proj_combo.current(0)
            self.proj_combo.pack(side="left")
            self.proj_combo.bind("<<ComboboxSelected>>", lambda e: self.redraw_image())

        def _build_tab_sequence(self, nb):
            f = ttk.Frame(nb, padding=8)
            nb.add(f, text="Sequence")
            ttk.Label(f, text="Pulse-timing preview — select the sequence_text "
                      "node in the browser to draw it here.",
                      foreground="#555").pack(anchor="w")
            self.seq_row = ttk.Frame(f); self.seq_row.pack(fill="x", pady=(8, 0))
            ttk.Label(self.seq_row, text="scan point:").pack(side="left")
            self.seq_scale = ttk.Scale(self.seq_row, from_=0, to=0,
                                       command=self.on_seq_scale, length=280)
            self.seq_scale.pack(side="left", padx=6)
            ttk.Button(self.seq_row, text="◀", width=3,
                       command=lambda: self.step_seq(-1)).pack(side="left", padx=(6, 0))
            ttk.Button(self.seq_row, text="▶|", width=3,
                       command=lambda: self.step_seq(1)).pack(side="left", padx=(2, 8))
            self.seq_overlay_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(self.seq_row, text="overlay all scan points",
                            variable=self.seq_overlay_var,
                            command=self.draw_sequence).pack(side="left")
            self.seq_lbl = ttk.Label(f, text="(no sequence selected)",
                                     foreground="#a60")
            self.seq_lbl.pack(anchor="w", pady=(8, 0))

        # ------------------------------------------------------------ files --
        def open_dialog(self):
            path = filedialog.askopenfilename(
                title="Select an HDF5 file",
                filetypes=[("HDF5 files", "*.h5 *.hdf5 *.he5 *.hf5"),
                           ("All files", "*.*")])
            if path:
                self.open_path(path)

        def reload(self):
            if self.filepath:
                self.open_path(self.filepath)

        def open_path(self, path):
            self._set_status(f"loading {os.path.basename(path)} …")
            self.root.update_idletasks()
            try:
                self.tree_data = load_hdf5_tree(path)
            except Exception as e:
                messagebox.showerror("Load failed", f"Could not read file:\n{e}")
                self._set_status("")
                return
            self.filepath = path
            self.file_lbl.configure(text=os.path.basename(path))
            self.stacks = find_image_stacks(self.tree_data)
            self._populate_tree()
            self._auto_view()
            self._set_status("loaded")

        # ------------------------------------------------------------- tree --
        def _populate_tree(self):
            self.tree.delete(*self.tree.get_children())
            self.item_values.clear()

            # synthetic node listing detected image stacks (easy movie access)
            if self.stacks:
                sroot = self.tree.insert("", "end",
                                         text="🎞 detected image stacks", open=True)
                for s in self.stacks:
                    iid = self.tree.insert(sroot, "end", text=s["name"])
                    self.item_values[iid] = ("__stack__", s)

            def add(parent, key, val):
                label = f"{key}   {describe_value(val)}"
                iid = self.tree.insert(parent, "end", text=label,
                                       open=(parent == "" ))
                self.item_values[iid] = ("value", val)
                if isinstance(val, dict):
                    for k, v in val.items():
                        add(iid, k, v)
                elif isinstance(val, list):
                    for i, item in enumerate(val):
                        add(iid, f"[{i}]", item)

            for k, v in self.tree_data.items():
                add("", k, v)

        def on_tree_select(self, _evt):
            iid = self._sel_iid()
            if iid is None:
                return
            tag, val = self.item_values[iid]
            self._show_info(iid, tag, val)
            self._update_xaxis_choices(val if tag == "value" else None)
            if tag == "value" and is_sequence_text(val):
                self.show_sequence(val)

        def _sel_iid(self):
            sel = self.tree.selection()
            return sel[0] if sel else None

        def _show_info(self, iid, tag, val):
            self.info.configure(state="normal")
            self.info.delete("1.0", "end")
            if tag == "__stack__":
                fr = val["frames"]
                self.info.insert("end", f"image stack\nsource: {val['source']}\n"
                                        f"frames: {fr.shape[0]}\n"
                                        f"frame size: {fr.shape[1]}×{fr.shape[2]}\n"
                                        f"dtype: {fr.dtype}\n\n"
                                        "double-click to open in the movie player.")
            else:
                kind = classify_value(val)
                self.info.insert("end", f"kind: {kind}\n")
                if isinstance(val, np.ndarray):
                    a = np.squeeze(val)
                    self.info.insert("end", f"shape: {tuple(a.shape)}\ndtype: {val.dtype}\n")
                    if a.size:
                        finite = a[np.isfinite(a)] if np.issubdtype(a.dtype, np.number) else a
                        if getattr(finite, "size", 0):
                            self.info.insert("end",
                                f"min/max: {np.min(finite):.6g} / {np.max(finite):.6g}\n"
                                f"mean: {np.mean(finite):.6g}\n")
                elif isinstance(val, str):
                    self.info.insert("end", "\n" + val[:1200])
                elif isinstance(val, (dict, list)):
                    self.info.insert("end", describe_value(val))
                else:
                    self.info.insert("end", f"value: {val}")
            self.info.configure(state="disabled")

        def _update_xaxis_choices(self, val):
            """Populate the x-axis combobox for the selected 1-D array."""
            if not (isinstance(val, np.ndarray) and classify_value(val) == "1d"):
                self.xaxis_combo.configure(values=[])
                self.xaxis_combo.set("")
                return
            n = int(np.squeeze(val).size)
            choices = ["index"]
            self._xaxis_map = {"index": np.arange(n, dtype=float)}
            freq_choices = []
            # sibling 1-D numeric arrays of equal length; a frequency-like one
            # (e.g. frequency_range / frequencies) is presented in GHz.
            for path, v in flatten_tree(self.tree_data):
                if isinstance(v, np.ndarray) and classify_value(v) == "1d":
                    if int(np.squeeze(v).size) == n and v is not val:
                        name = path.split("/")[-1]
                        arr = np.squeeze(v).astype(float)
                        if "freq" in name.lower() and np.nanmax(np.abs(arr)) > 1e6:
                            label = f"{name} (GHz)"
                            arr = arr / 1e9
                        else:
                            label = name
                        if label not in self._xaxis_map:
                            choices.append(label)
                            self._xaxis_map[label] = arr
                            if "freq" in label.lower():
                                freq_choices.append(label)
            # reconstructed pulsed axis from sequence_text — only the ONE that
            # matches what the sequence actually sweeps (tau / amplitude / freq)
            seq = find_text(self.tree_data, "sequence:", "variable ", "type=")
            kind = sweep_kind(seq) if seq else None
            if kind == "time":
                tax, tlab = build_time_axis(seq, n)
                if tlab.startswith("tau"):
                    choices.append("tau (from sequence)")
                    self._xaxis_map["tau (from sequence)"] = tax
            elif kind == "amplitude":
                aax, alab = build_amp_axis(seq, n)
                if alab.startswith("pulse amplitude"):
                    choices.append("amplitude (from sequence)")
                    self._xaxis_map["amplitude (from sequence)"] = aax
            elif kind == "frequency":
                fax, flab = build_freq_axis(seq, n)
                if flab.startswith("frequency"):
                    choices.append("frequency (from sequence, GHz)")
                    self._xaxis_map["frequency (from sequence, GHz)"] = fax / 1e9
                    freq_choices.append("frequency (from sequence, GHz)")
            elif kind == "phase":
                pax, plab = build_phase_axis(seq, n)
                if plab.startswith("phase"):
                    choices.append("phase (from sequence, rad)")
                    self._xaxis_map["phase (from sequence, rad)"] = pax
            self.xaxis_combo.configure(values=choices)
            # default: honour the swept variable. time->tau, amplitude->amp,
            # phase->phase, frequency (or unknown) -> a real frequency array.
            default = "index"
            if kind == "time" and "tau (from sequence)" in choices:
                default = "tau (from sequence)"
            elif kind == "amplitude" and "amplitude (from sequence)" in choices:
                default = "amplitude (from sequence)"
            elif kind == "phase" and "phase (from sequence, rad)" in choices:
                default = "phase (from sequence, rad)"
            elif freq_choices:
                default = freq_choices[0]
            elif "tau (from sequence)" in choices:
                default = "tau (from sequence)"
            self.xaxis_combo.set(default)

        # --------------------------------------------------- plotting nodes --
        def plot_selected_node(self):
            iid = self._sel_iid()
            if iid is None:
                return
            tag, val = self.item_values[iid]
            if tag == "__stack__":
                self.show_stack(val["frames"], title=val["name"])
                return
            kind = classify_value(val)
            if kind == "1d":
                self._plot_from_node(np.squeeze(val).astype(float),
                                     self.tree.item(iid, "text").split("   ")[0])
            elif kind == "image":
                self.show_image(np.squeeze(val).astype(float),
                                title=self.tree.item(iid, "text").split("   ")[0])
            elif kind == "stack":
                self.show_stack(np.squeeze(val).astype(float),
                                title=self.tree.item(iid, "text").split("   ")[0])
            elif is_sequence_text(val):
                self.show_sequence(val)
            else:
                messagebox.showinfo("Not plottable",
                                    f"This node is '{kind}'. Pick a 1-D array, an "
                                    "image, or a stack (or a scalar to inspect).")

        def _plot_from_node(self, yarr, name):
            xchoice = self.xaxis_combo.get() or "index"
            xmap = getattr(self, "_xaxis_map", {"index": np.arange(len(yarr))})
            x = xmap.get(xchoice, np.arange(len(yarr), dtype=float))
            xl = xchoice if xchoice != "index" else "index"
            self.set_1d(x, yarr, xlabel=xl, ylabel=name, title=name)

        def plot_selected_together(self):
            """Overlay every selected 1-D array on one plot (Ctrl/Shift-click to
            pick several, e.g. signal_counts + reference_counts)."""
            xchoice = self.xaxis_combo.get() or "index"
            xmap = getattr(self, "_xaxis_map", {})
            curves = []
            for iid in self.tree.selection():
                tag, val = self.item_values.get(iid, (None, None))
                if tag != "value" or classify_value(val) != "1d":
                    continue
                y = np.squeeze(val).astype(float)
                name = self.tree.item(iid, "text").split("   ")[0]
                xc = xmap.get(xchoice)
                x = xc if (xc is not None and len(xc) == len(y)) \
                    else np.arange(len(y), dtype=float)
                curves.append((x, y, name))
            if not curves:
                messagebox.showinfo(
                    "Plot selected together",
                    "Select one or more 1-D arrays in the browser first "
                    "(hold Ctrl or Shift to pick several), then click this button.\n\n"
                    "e.g. select signal_counts and reference_counts to overlay them.")
                return
            xl = xchoice if xchoice != "index" else "index"
            if len(curves) == 1:
                self.set_1d(curves[0][0], curves[0][1], xlabel=xl,
                            ylabel=curves[0][2], title=curves[0][2])
            else:
                self.set_multi(curves, xlabel=xl,
                               title="overlay: " + ", ".join(c[2] for c in curves))
            try:
                self.nb.select(0)
            except Exception:
                pass

        def replot_1d(self):
            if self.view_kind == "1d" and self.x is not None:
                self.draw_1d()

        # -------------------------------------------------- auto first view --
        def _auto_view(self):
            info = detect_default_view(self.tree_data)
            mode = info["mode"]
            if mode == "pulsed":
                self._prepare_pulsed()
                self.nb.select(2)
                self.pulsed_plot_raw()
            elif mode == "cw_odmr":
                _, freqs = get_by_name(self.tree_data, "frequencies", "freq_list",
                                       want=("1d",))
                _, counts = get_by_name(self.tree_data, "counts_averaged",
                                        "all_counts_forward", "counts_forward",
                                        "counts", want=("1d",))
                f = np.squeeze(freqs).astype(float)
                c = np.squeeze(counts).astype(float)
                n = min(len(f), len(c))
                xl = "frequency (GHz)" if np.nanmax(f) > 1e6 else "frequency"
                xf = f[:n] / 1e9 if np.nanmax(f) > 1e6 else f[:n]
                self.set_1d(xf, c[:n], xlabel=xl, ylabel="counts",
                            title="CW ODMR sweep")
                self.model_combo.set("Lorentzian dip (ODMR)")
                self.nb.select(1)
            elif mode == "stack":
                self.show_stack(info["stack"]["frames"], title=info["stack"]["name"])
                self.nb.select(3)
            elif mode == "image":
                _, img = get_by_name(self.tree_data, "count_img", "count_image",
                                     "image", want=("image",))
                self.show_image(np.squeeze(img).astype(float),
                                title="confocal scan")
                self.nb.select(3)
            elif mode == "1d":
                _, v = get_by_name(self.tree_data, info["path"].split("/")[-1],
                                   want=("1d",))
                if v is None:
                    for path, val in flatten_tree(self.tree_data):
                        if classify_value(val) == "1d":
                            v = val; break
                if v is not None:
                    self.set_1d(np.arange(len(np.squeeze(v))),
                                np.squeeze(v).astype(float),
                                xlabel="index", ylabel="value", title=info["path"])
            else:
                self._new_axes()
                self.ax.text(0.5, 0.5, "No obvious dataset to plot.\n"
                             "Pick something from the data browser on the left.",
                             ha="center", va="center", transform=self.ax.transAxes,
                             color="#888")
                self.canvas.draw_idle()

        # -------------------------------------------------------- 1-D engine --
        def _kill_selectors(self):
            """Deactivate + disconnect any live Span/Rectangle selectors so their
            drawn spans/boxes and event handlers don't leak across redraws."""
            for attr in ("span", "rect"):
                sel = getattr(self, attr, None)
                if sel is not None:
                    for meth in ("set_active", "set_visible", "disconnect_events"):
                        try:
                            fn = getattr(sel, meth, None)
                            if fn:
                                fn(False) if meth != "disconnect_events" else fn()
                        except Exception:
                            pass
                setattr(self, attr, None)

        def _new_axes(self):
            self._kill_selectors()
            self.fig.clear()
            self.ax = self.fig.add_subplot(111)
            self.span = None
            self.rect = None
            self.imobj = None
            self.cbar = None
            self.fig.tight_layout()
            self.canvas.draw_idle()

        def set_1d(self, x, y, xlabel="", ylabel="", title=""):
            self._stop_play()
            self.view_kind = "1d"
            self.curves = None                    # single-curve mode
            self.x = np.asarray(x, float).ravel()
            self.y = np.asarray(y, float).ravel()
            n = min(len(self.x), len(self.y))
            self.x, self.y = self.x[:n], self.y[:n]
            self.xlabel, self.ylabel, self.title = xlabel, ylabel, title
            self.cur_frames = None
            self.sel_box = None
            self._last_fit = None
            self.movie_row_state(False)
            self.draw_1d()

        def set_multi(self, curves, xlabel="", title=""):
            """Overlay several 1-D curves. curves = list of (x, y, label). The
            first curve becomes the 'primary' used for selection / fitting / stats."""
            curves = [(np.asarray(cx, float).ravel(), np.asarray(cy, float).ravel(), lab)
                      for cx, cy, lab in curves if len(np.ravel(cy))]
            if not curves:
                return
            self._stop_play()
            self.view_kind = "1d"
            self.curves = curves
            cx, cy, _ = curves[0]
            n = min(len(cx), len(cy))
            self.x, self.y = cx[:n], cy[:n]        # primary
            self.xlabel = xlabel
            self.ylabel = curves[0][2] if len(curves) == 1 else ""
            self.title = title
            self.cur_frames = None
            self.sel_box = None
            self._last_fit = None
            self.movie_row_state(False)
            self.draw_1d()

        def _norm_apply(self, y, x=None):
            x = self.x if x is None else x
            mode = self.norm_var.get()
            if mode == "divide by max":
                m = np.nanmax(np.abs(y))
                return y / m if m else y
            if mode == "subtract selection baseline":
                if self.sel_x0 is not None and x is not None and len(x) == len(y):
                    mask = span_mask(x, self.sel_x0, self.sel_x1)
                    base = np.nanmedian(y[mask]) if mask.any() else np.nanmedian(y)
                else:
                    base = np.nanmedian(y)
                return y - base
            return y

        def draw_1d(self, keep_lims=False):
            xl, yl = (self.ax.get_xlim(), self.ax.get_ylim()) if keep_lims else (None, None)
            self._new_axes()
            style = self.style_var.get()
            fmt = {"line": "-", "markers": "o", "line+markers": "-o"}[style]
            palette = ["tab:blue", "tab:red", "tab:green", "tab:purple",
                       "tab:orange", "tab:cyan", "tab:brown", "tab:pink"]
            if self.curves:                        # overlay of several curves
                for i, (cx, cy, lab) in enumerate(self.curves):
                    self.ax.plot(cx, self._norm_apply(cy, cx), fmt, ms=4, lw=1.3,
                                 color=palette[i % len(palette)], label=lab)
            else:                                  # single curve
                self.ax.plot(self.x, self._norm_apply(self.y, self.x), fmt, ms=4,
                             lw=1.3, color="tab:blue", label=self.ylabel or "data")
            # re-draw a stored fit if present
            if self._last_fit is not None:
                fitd = self._last_fit
                self.ax.plot(fitd["x_dense"], fitd["y_dense"], "-k", lw=1.6,
                             label=f"fit (R²={fitd['r2']:.4f})")
            self.ax.set_xlabel(self.xlabel)
            self.ax.set_ylabel(self.ylabel)
            self.ax.set_title(getattr(self, "title", ""), fontsize=10)
            if self.logy_var.get():
                try:
                    self.ax.set_yscale("log")
                except Exception:
                    pass
            self.ax.grid(True, alpha=0.3)
            self.ax.legend(fontsize=8)
            # span selector
            self.span = _make_span(self.ax, self._on_span)
            if self.sel_x0 is not None:
                self.ax.axvspan(self.sel_x0, self.sel_x1, color="tab:orange", alpha=0.15)
            if keep_lims and xl:
                self.ax.set_xlim(xl); self.ax.set_ylim(yl)
            self.fig.tight_layout()
            self.canvas.draw_idle()

        def _on_span(self, x0, x1):
            if x1 == x0:
                return
            self.sel_x0, self.sel_x1 = (min(x0, x1), max(x0, x1))
            self.sel_lbl.configure(
                text=f"x ∈ [{self.sel_x0:.6g}, {self.sel_x1:.6g}]   "
                     f"({int(span_mask(self.x, self.sel_x0, self.sel_x1).sum())} pts)")

        def clear_selection(self):
            self.sel_x0 = self.sel_x1 = None
            self.sel_lbl.configure(text="(drag on the plot to select a range)")
            if self.view_kind == "1d":
                self.draw_1d()

        def zoom_selection(self):
            if self.view_kind != "1d" or self.sel_x0 is None:
                return
            self.ax.set_xlim(self.sel_x0, self.sel_x1)
            m = span_mask(self.x, self.sel_x0, self.sel_x1)
            if m.any():
                ys = self._norm_apply(self.y)[m]
                pad = 0.05 * (np.nanmax(ys) - np.nanmin(ys) + 1e-12)
                self.ax.set_ylim(np.nanmin(ys) - pad, np.nanmax(ys) + pad)
            self.canvas.draw_idle()

        def reset_view(self):
            if self.view_kind == "1d":
                self.draw_1d()
            elif self.view_kind in ("image", "stack"):
                self.redraw_image()

        def selection_stats(self):
            if self.view_kind != "1d":
                return
            y = self._norm_apply(self.y)
            s = region_stats_1d(self.x, y, self.sel_x0, self.sel_x1)
            title = "selection" if self.sel_x0 is not None else "whole trace"
            messagebox.showinfo(f"Stats — {title}", format_stats_1d(s))

        def export_selection(self):
            if self.view_kind != "1d":
                return
            x, y = self.x, self._norm_apply(self.y)
            if self.sel_x0 is not None:
                m = span_mask(x, self.sel_x0, self.sel_x1)
                x, y = x[m], y[m]
            self._save_csv(np.column_stack([x, y]),
                           header=f"{self.xlabel},{self.ylabel}", suffix="_selection")

        # ----------------------------------------------------------- fitting --
        def do_fit(self):
            if self.view_kind != "1d":
                messagebox.showinfo("Fit", "Fitting works on 1-D traces. Select or "
                                           "plot a 1-D dataset first.")
                return
            name = self.model_combo.get()
            x, y = self.x, self._norm_apply(self.y)
            if self.fit_sel_var.get() and self.sel_x0 is not None:
                m = span_mask(x, self.sel_x0, self.sel_x1)
                x, y = x[m], y[m]
                scope = f"selection ({m.sum()} pts)"
            else:
                scope = f"full trace ({len(x)} pts)"
            res = fit_model(name, x, y)
            self.fit_text.delete("1.0", "end")
            if not res["ok"]:
                self.fit_text.insert("end", f"Fit failed on {scope}:\n{res['error']}")
                return
            self._last_fit = res
            self.fit_text.insert("end", f"[{scope}]\n" + res["summary"])
            self.draw_1d()

        def clear_fit(self):
            self._last_fit = None
            self.fit_text.delete("1.0", "end")
            if self.view_kind == "1d":
                self.draw_1d()

        def export_fit(self):
            if not self._last_fit:
                messagebox.showinfo("Export fit", "Run a fit first.")
                return
            d = self._last_fit
            self._save_csv(np.column_stack([d["x_dense"], d["y_dense"]]),
                           header="x_fit,y_fit", suffix="_fit")

        # ------------------------------------------------------- pulsed tab ---
        def _prepare_pulsed(self):
            """Collect signal/reference + sequence text for the pulsed workflow."""
            _, sig = get_by_name(self.tree_data, "signal_counts", "signal",
                                 want=("1d", "image"))
            _, ref = get_by_name(self.tree_data, "reference_counts", "reference",
                                 want=("1d", "image"))
            if sig is None or ref is None:
                self.pulsed_cache = None
                return
            sig = np.squeeze(np.asarray(sig, float))
            ref = np.squeeze(np.asarray(ref, float))
            if sig.ndim == 2:
                sig = sig[0]
            if ref.ndim == 2:
                ref = ref[0]
            n = min(len(sig), len(ref))
            seq = find_text(self.tree_data, "sequence:", "variable ", "type=")
            _, farr = get_by_name(self.tree_data, "frequencies", "freq_list",
                                  want=("1d",))
            self.pulsed_cache = dict(sig=sig[:n], ref=ref[:n], seq=seq, n=n,
                                     has_freq=(farr is not None), freq=farr)
            mode = detect_pulsed_mode(seq, farr is not None)
            self.pulsed_mode = mode
            self.pmode_combo.set(mode)

        def _pulsed_axis(self, mode):
            c = self.pulsed_cache
            n = c["n"]
            if mode == "odmr":
                if c["freq"] is not None:
                    f = np.squeeze(c["freq"]).astype(float)[:n]
                    return f, ("frequency (GHz)" if np.nanmax(f) > 1e6 else "frequency")
                # reconstruct from a swept-frequency variable line, if present
                fax, flab = build_freq_axis(c["seq"], n)
                if flab.startswith("frequency"):
                    return fax, "frequency (GHz)"
                # else from a real frequency array (frequency_range holds the
                # actual swept frequencies; use them directly when lengths match)
                _, fr = get_by_name(self.tree_data, "frequency_range",
                                    "frequencies", want=("1d",))
                if fr is not None:
                    fr = np.squeeze(fr).astype(float)
                    if len(fr) == n:
                        return fr, "frequency (GHz)"
                    return np.linspace(fr[0], fr[-1], n), "frequency (GHz)"
                return np.arange(n, float), "scan point"
            if mode == "phase":
                return build_phase_axis(c["seq"], n)
            if mode == "rabi_amp":
                return build_amp_axis(c["seq"], n)
            return build_time_axis(c["seq"], n)

        def pulsed_plot_raw(self):
            if not self.pulsed_cache:
                self._prepare_pulsed()
            if not self.pulsed_cache:
                messagebox.showinfo("Pulsed", "No signal/reference arrays found.")
                return
            c = self.pulsed_cache
            mode = self.pmode_combo.get()
            x, xlabel = self._pulsed_axis(mode)
            if mode == "odmr" and np.nanmax(x) > 1e6:
                x = x / 1e9
            y, ylabel = self._pulsed_quantity(c["sig"], c["ref"])
            self.set_1d(x, y, xlabel=xlabel, ylabel=ylabel,
                        title=f"pulsed: {mode}   [{ylabel}]")

        def _pulsed_quantity(self, sig, ref):
            """Return (y, label) for the quantity chosen in the Pulsed tab."""
            q = self.pq_combo.get()
            with np.errstate(divide="ignore", invalid="ignore"):
                if q == "signal":
                    return sig.astype(float), "signal counts"
                if q == "reference":
                    return ref.astype(float), "reference counts"
                if q == "S-R":
                    return sig - ref, "signal − reference"
                if q == "(S-R)/R":
                    return np.where(ref != 0, (sig - ref) / ref, np.nan), "(S − R) / R"
                return np.where(ref != 0, sig / ref, np.nan), "Signal / Reference"

        def _pulsed_prep_xy(self):
            """Common prep for analyse + significance: axis, chosen quantity,
            dead-point drop, finite mask, GHz conversion, and optional selection
            restriction.  Returns (mode, xplot, y_plot, x_fit, y_fit, ylabel, scope)."""
            c = self.pulsed_cache
            mode = self.pmode_combo.get()
            x, xlabel = self._pulsed_axis(mode)
            sig, ref = c["sig"].copy(), c["ref"].copy()
            if self.drop_dead_var.get():
                med = np.median(sig[sig > 0]) if np.any(sig > 0) else 0
                live = sig > 0.2 * med
                x, sig, ref = x[live], sig[live], ref[live]
            y, ylabel = self._pulsed_quantity(sig, ref)
            good = np.isfinite(y)
            xg, yg = x[good], y[good]
            xplot = xg / 1e9 if (mode == "odmr" and np.nanmax(xg) > 1e6) else xg
            # fit/test domain keeps physics units (Hz for odmr); optionally
            # restrict to the span the user dragged on the Plot tab (in plotted
            # coordinates), applying the same mask to the physics-unit array.
            xfit, yfit, scope = xg, yg, f"full trace ({len(xg)} pts)"
            if self.pulsed_sel_var.get():
                if self.sel_x0 is None:
                    scope = "no selection made — used full trace"
                else:
                    m = span_mask(xplot, self.sel_x0, self.sel_x1)
                    if int(m.sum()) >= 4:
                        xfit, yfit = xg[m], yg[m]
                        scope = f"selection ({int(m.sum())} pts)"
                    else:
                        scope = "selection <4 pts — used full trace"
            return mode, xlabel, xplot, yg, xfit, yfit, ylabel, scope

        def pulsed_analyse(self):
            if not self.pulsed_cache:
                self._prepare_pulsed()
            if not self.pulsed_cache:
                messagebox.showinfo("Pulsed", "No signal/reference arrays found.")
                return
            mode, xlabel, xplot, yg, xfit, yfit, ylabel, scope = self._pulsed_prep_xy()
            self.pulsed_text.delete("1.0", "end")
            res = self._pulsed_fit(mode, xfit, yfit)
            self.set_1d(xplot, yg,
                        xlabel=("frequency (GHz)" if mode == "odmr" else xlabel),
                        ylabel=ylabel, title=f"pulsed: {mode}   [{ylabel}]")
            head = f"[{ylabel}  ·  {scope}]\n\n"
            if res is not None:
                xf = res["xf"] / 1e9 if (mode == "odmr" and np.nanmax(res["xf"]) > 1e6) \
                    else res["xf"]
                self._last_fit = {"x_dense": xf, "y_dense": res["yf"],
                                  "r2": res.get("r2", float("nan"))}
                self.draw_1d()
                self.pulsed_text.insert("end", head + res["summary"])
            else:
                self.pulsed_text.insert("end", head + "fit failed for this mode.")

        def _pulsed_fit(self, mode, x, y):
            """Prefer our plot_pulsed_hdf5 fitters; fall back to the local library.
            The phase sweep has no counterpart in plot_pulsed_hdf5, so it always
            uses the built-in single-fringe cosine."""
            if mode != "phase":
                try:
                    import plot_pulsed_hdf5 as pp
                    if mode == "rabi_time":
                        d = pp.fit_rabi(x, y, amplitude=False)
                    elif mode == "rabi_amp":
                        d = pp.fit_rabi(x, y, amplitude=True)
                    elif mode == "ramsey":
                        d = pp.fit_ramsey(x, y)
                    elif mode == "echo":
                        d = pp.fit_echo(x, y)
                    elif mode == "odmr":
                        d = pp.fit_odmr(x, y)
                    else:
                        d = None
                    if d is not None:
                        r2 = r_squared(y, np.interp(x, d["xf"], d["yf"]))
                        return {"xf": d["xf"], "yf": d["yf"], "r2": r2,
                                "summary": d["title"] + "\n\n" + d["summary"] +
                                f"\n\n  R² = {r2:.5f}   (via plot_pulsed_hdf5.py)"}
                except Exception:
                    pass
            # fallback (and always, for phase): local models
            name = {"rabi_time": "Damped cosine (Rabi)",
                    "rabi_amp": "Damped cosine (Rabi)",
                    "ramsey": "Ramsey (Gaussian×cos)",
                    "echo": "Stretched exp (echo)",
                    "odmr": "Lorentzian dip (ODMR)",
                    "phase": "Cosine (phase sweep)"}[mode]
            r = fit_model(name, x, y)
            if not r["ok"]:
                return None
            note = ("\n\n  (phase-sweep cosine fit)" if mode == "phase" else
                    "\n\n  (built-in fit; keep plot_pulsed_hdf5.py next to this "
                    "file for the physics readout)")
            return {"xf": r["x_dense"], "yf": r["y_dense"], "r2": r["r2"],
                    "summary": r["summary"] + note}

        def pulsed_significance(self):
            if not self.pulsed_cache:
                self._prepare_pulsed()
            if not self.pulsed_cache:
                messagebox.showinfo("Pulsed", "No signal/reference arrays found.")
                return
            mode, xlabel, xplot, yg, xfit, yfit, ylabel, scope = self._pulsed_prep_xy()
            self._set_status("running permutation test …")
            self.root.update_idletasks()
            r = run_significance(mode, xfit, yfit)
            self._set_status("")
            self.pulsed_text.delete("1.0", "end")
            if not r["ok"]:
                self.pulsed_text.insert("end", r["error"]); return
            lines = [f"mode: {mode}   quantity: {ylabel}",
                     f"scope: {scope}",
                     f"points used: {len(xfit)}",
                     f"permutations: {r['n_perm']}", ""]
            if r["kind"] == "periodogram":
                lines += [f"Lomb-Scargle peak: {r['peak']:.3f} {r['unit']}",
                          f"peak power:        {r['power']:.3f}",
                          f"95% noise floor:   {r['thr']:.3f}",
                          f"Nyquist:           {r['nyquist']:.2f} {r['unit']}"]
                if mode == "rabi_time":
                    lines.append(f"→ f_Rabi ≈ {r['peak']:.2f} MHz, "
                                 f"π ≈ {500/r['peak']:.0f} ns, "
                                 f"π/2 ≈ {250/r['peak']:.0f} ns")
            else:
                lines += [f"shape-fit R²:      {r['r2']:.3f}",
                          f"95% noise R²:      {r['thr']:.3f}"]
            lines += ["", f"FALSE-ALARM PROBABILITY = {r['fap']*100:.2f}%",
                      f"VERDICT: {r['verdict']}"]
            self.pulsed_text.insert("end", "\n".join(lines))

        # -------------------------------------------------------- image tab ---
        def movie_row_state(self, on):
            # ttk widgets need .state(); plain-tk widgets need configure(state=).
            for child in self.movie_row.winfo_children():
                done = False
                try:
                    child.state(["!disabled"] if on else ["disabled"])
                    done = True
                except Exception:
                    pass
                if not done:
                    try:
                        child.configure(state="normal" if on else "disabled")
                    except Exception:
                        pass

        def show_image(self, img, title=""):
            self._stop_play()
            self.view_kind = "image"
            self.cur_image = np.asarray(img, float)
            self.cur_frames = None
            self.sel_box = None
            self.title = title
            self.movie_row_state(False)
            self.nb.select(3)
            self.redraw_image()

        def show_stack(self, frames, title=""):
            self._stop_play()
            self.view_kind = "stack"
            self.cur_frames = np.asarray(frames, float)
            self.frame_idx = 0
            self.sel_box = None
            self.title = title
            self.movie_row_state(True)
            self.frame_scale.configure(from_=0, to=self.cur_frames.shape[0] - 1)
            self.frame_scale.set(0)
            self.proj_combo.set("frame")
            self.nb.select(3)
            self.redraw_image()

        def _current_image_data(self):
            if self.view_kind == "stack":
                proj = self.proj_combo.get()
                if proj == "frame":
                    return self.cur_frames[self.frame_idx]
                return stack_projection(self.cur_frames, proj)
            return self.cur_image

        def redraw_image(self, *_):
            data = self._current_image_data()
            if data is None:
                return
            self._new_axes()
            clip = float(self.clip_var.get())
            finite = data[np.isfinite(data)]
            if finite.size:
                lo = np.percentile(finite, clip)
                hi = np.percentile(finite, 100 - clip)
            else:
                lo, hi = None, None
            self.imobj = self.ax.imshow(data, origin="lower",
                                        cmap=self.cmap_combo.get(),
                                        vmin=lo, vmax=hi,
                                        aspect="equal" if self.equal_var.get() else "auto",
                                        interpolation="nearest")
            self.cbar = self.fig.colorbar(self.imobj, ax=self.ax, fraction=0.046, pad=0.04)
            ttl = getattr(self, "title", "")
            if self.view_kind == "stack" and self.proj_combo.get() == "frame":
                ttl += f"   frame {self.frame_idx+1}/{self.cur_frames.shape[0]}"
                self.frame_lbl.configure(text=f"{self.frame_idx+1}/{self.cur_frames.shape[0]}")
            self.ax.set_title(ttl, fontsize=10)
            self.ax.set_xlabel("x (pixels)"); self.ax.set_ylabel("y (pixels)")
            self.rect = _make_rect(self.ax, self._on_rect)
            if self.sel_box is not None:
                x0, x1, y0, y1 = self.sel_box
                self.ax.add_patch(plt_rect(x0, y0, x1 - x0, y1 - y0))
            self.fig.tight_layout()
            self.canvas.draw_idle()

        def _on_rect(self, eclick, erelease):
            x0, y0 = eclick.xdata, eclick.ydata
            x1, y1 = erelease.xdata, erelease.ydata
            if None in (x0, y0, x1, y1):
                return
            self.sel_box = (min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))
            self.roi_lbl.configure(
                text=f"cols {int(self.sel_box[0])}–{int(self.sel_box[1])}, "
                     f"rows {int(self.sel_box[2])}–{int(self.sel_box[3])}")

        def clear_roi(self):
            self.sel_box = None
            self.roi_lbl.configure(text="(drag a box on the image)")
            if self.view_kind in ("image", "stack"):
                self.redraw_image()

        def crop_roi(self):
            if self.view_kind not in ("image", "stack") or self.sel_box is None:
                messagebox.showinfo("Crop", "Drag a box on the image first.")
                return
            data = self._current_image_data()
            sub, _ = image_crop(data, *self.sel_box)
            self.show_image(sub, title=getattr(self, "title", "") + " (cropped)")

        def roi_profile(self, axis):
            if self.view_kind not in ("image", "stack"):
                return
            data = self._current_image_data()
            box = self.sel_box
            xi, col, yi, row = image_profiles(data, box)
            if axis == "x":
                self.set_1d(xi, col, xlabel="x (pixels)",
                            ylabel="mean over ROI rows", title="X profile (line-out)")
            else:
                self.set_1d(yi, row, xlabel="y (pixels)",
                            ylabel="mean over ROI cols", title="Y profile (line-out)")

        def roi_stats(self):
            if self.view_kind not in ("image", "stack"):
                return
            data = self._current_image_data()
            s = region_stats_2d(data, self.sel_box)
            title = "ROI" if self.sel_box is not None else "whole image"
            messagebox.showinfo(f"Stats — {title}", format_stats_2d(s))

        # ------------------------------------------------------ movie player --
        def on_frame_scale(self, val):
            if self.view_kind != "stack" or getattr(self, "_scale_busy", False):
                return
            idx = int(round(float(val)))
            idx = max(0, min(idx, self.cur_frames.shape[0] - 1))
            if idx != self.frame_idx or self.proj_combo.get() != "frame":
                self.frame_idx = idx
                if self.proj_combo.get() != "frame":
                    self.proj_combo.set("frame")
                self.redraw_image()

        def step_frame(self, d):
            if self.view_kind != "stack":
                return
            n = self.cur_frames.shape[0]
            self.frame_idx = (self.frame_idx + d) % n
            self.proj_combo.set("frame")
            self._scale_busy = True                 # don't let .set() re-trigger
            try:
                self.frame_scale.set(self.frame_idx)
            finally:
                self._scale_busy = False
            self.redraw_image()

        def toggle_play(self):
            if self.view_kind != "stack":
                return
            self.playing = not self.playing
            self.play_btn.configure(text="⏸ Pause" if self.playing else "▶ Play")
            if self.playing:
                self.proj_combo.set("frame")
                self._advance()

        def _advance(self):
            if not self.playing or self.view_kind != "stack":
                return
            self.step_frame(1)
            delay = int(1000 / max(1, self.fps_var.get()))
            self.root.after(delay, self._advance)

        def _stop_play(self):
            self.playing = False
            try:
                self.play_btn.configure(text="▶ Play")
            except Exception:
                pass

        # -------------------------------------------------- sequence preview --
        def show_sequence(self, text):
            parsed = parse_sequence_text(text)
            if not parsed or not parsed.get("pulses"):
                return False
            self._stop_play()
            self.seq_parsed = parsed
            self.view_kind = "sequence"
            steps = parsed["variables"][0]["steps"] if parsed["variables"] else 1
            self.seq_scale.configure(from_=0, to=max(0, steps - 1))
            self.seq_index = steps // 2 if steps > 1 else 0
            self._seq_busy = True
            try:
                self.seq_scale.set(self.seq_index)
            finally:
                self._seq_busy = False
            self.seq_row_state(steps > 1)
            try:
                self.nb.select(self._seq_tab_index)
            except Exception:
                pass
            self.draw_sequence()
            return True

        def seq_row_state(self, on):
            for child in self.seq_row.winfo_children():
                try:
                    child.state(["!disabled"] if on else ["disabled"])
                except Exception:
                    try:
                        child.configure(state="normal" if on else "disabled")
                    except Exception:
                        pass

        def on_seq_scale(self, val):
            if self.view_kind != "sequence" or getattr(self, "_seq_busy", False):
                return
            idx = int(round(float(val)))
            if idx != self.seq_index:
                self.seq_index = idx
                self.draw_sequence()

        def step_seq(self, d):
            if self.view_kind != "sequence" or not self.seq_parsed:
                return
            steps = (self.seq_parsed["variables"][0]["steps"]
                     if self.seq_parsed["variables"] else 1)
            self.seq_index = int(np.clip(self.seq_index + d, 0, max(0, steps - 1)))
            self._seq_busy = True
            try:
                self.seq_scale.set(self.seq_index)
            finally:
                self._seq_busy = False
            self.draw_sequence()

        # a small fixed palette so each channel keeps one colour
        _SEQ_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd",
                       "#ff7f0e", "#17becf", "#8c564b", "#e377c2"]

        def draw_sequence(self):
            parsed = getattr(self, "seq_parsed", None)
            if not parsed:
                return
            self._new_axes()
            ax = self.ax
            geo = sequence_pulse_geometry(parsed, self.seq_index)
            channels = geo["channels"] or [0]
            lane = {ch: i for i, ch in enumerate(channels)}
            ncol = len(self._SEQ_COLORS)
            t_max = geo["t_max_ns"] or 1.0

            def draw_one(g, alpha, label=True):
                for p in g["pulses"]:
                    row = lane.get(p["channel"], 0)
                    col = self._SEQ_COLORS[row % ncol]
                    a = p["amp"] if np.isfinite(p["amp"]) else 1.0
                    h = 0.42 * (a if abs(a) <= 1 else np.sign(a))  # clip to lane
                    if abs(h) < 0.03:
                        h = 0.03 * (1 if a >= 0 else -1)
                    ax.add_patch(_rect(p["start_ns"], row, max(p["width_ns"], 0.5),
                                       h, col, alpha))
                    if label:
                        ax.text(p["start_ns"] + max(p["width_ns"], 0.5) / 2,
                                row + (0.5 if h >= 0 else -0.5), p["name"],
                                ha="center", va="bottom" if h >= 0 else "top",
                                fontsize=7, rotation=0, color=col, clip_on=True)

            if self.seq_overlay_var.get() and geo["steps"] > 1:
                # faint envelope of every scan point, then the current one solid
                for i in range(geo["steps"]):
                    draw_one(sequence_pulse_geometry(parsed, i), 0.10, label=False)
            draw_one(geo, 0.85, label=True)

            for r in range(len(channels)):
                ax.axhline(r, color="#bbb", lw=0.6, zorder=0)
            ax.set_yticks(range(len(channels)))
            ax.set_yticklabels([f"ch {c}" for c in channels])
            ax.set_ylim(-0.8, len(channels) - 0.2)
            ax.set_xlim(-0.02 * t_max, 1.03 * t_max)
            ax.set_xlabel("time (ns)")
            title = f"sequence: {parsed.get('name') or '?'}"
            if parsed.get("type"):
                title += f"  ({parsed['type']})"
            ax.set_title(title)
            # scan-point label
            if geo["var_name"] is not None:
                u = f" {geo['var_unit']}" if geo["var_unit"] else ""
                msg = (f"{geo['var_name']} = {geo['var_value']:.4g}{u}   "
                       f"(scan point {geo['scan_index'] + 1}/{geo['steps']})")
            else:
                msg = "no swept variable (single fixed sequence)"
            try:
                self.seq_lbl.configure(text=msg)
            except Exception:
                pass
            self.fig.tight_layout()
            self.canvas.draw_idle()

        # ------------------------------------------------------------ utils --
        def _save_csv(self, arr, header="", suffix=".csv"):
            base = os.path.splitext(self.filepath or "data")[0]
            default = os.path.basename(base) + suffix + ".csv"
            path = filedialog.asksaveasfilename(
                defaultextension=".csv", initialfile=default,
                filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
            if not path:
                return
            np.savetxt(path, arr, delimiter=",", header=header, comments="")
            self._set_status(f"saved {os.path.basename(path)}")

        def _set_status(self, msg):
            self.status.configure(text=msg)

    # helper: matplotlib Rectangle patch (imported lazily)
    def plt_rect(x, y, w, h):
        from matplotlib.patches import Rectangle
        return Rectangle((x, y), w, h, fill=False, edgecolor="white", lw=1.3)

    # helper: filled coloured rectangle for the pulse-timing diagram
    def _rect(x, y, w, h, color, alpha):
        from matplotlib.patches import Rectangle
        return Rectangle((x, y), w, h, facecolor=color, edgecolor=color,
                         alpha=alpha, lw=0.8)

    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except Exception:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
