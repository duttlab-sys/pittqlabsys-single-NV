#!/usr/bin/env python3
"""nv_batch_report.py  --  batch analysis report for NV data.

Point it at a folder (or list) of HDF5 files and it produces ONE chronological
HTML report with, for every file:

    * the sequence text and the date
    * the data plot and the data+fit plot
    * the type-specific analysis:
        - Rabi            -> Rabi frequency + pi-pulse duration
        - pulsed / CW ODMR-> resonance + FWHM
        - Hahn echo       -> T2
        - Ramsey          -> T2*
        - T1              -> T1
        - confocal        -> brightest-spot position + 2-D Gaussian sigma
    * the experiment duration (only when start/end times are stored)
    * a collapsible dropdown of every setting/parameter saved with the file

It is a standalone wrapper: it imports nv_data_gui and reuses ITS analysis
functions (load_hdf5_tree, detect_pulsed_mode, fit_model, fit_odmr_dips,
fit_gaussian_2d, build_time_axis/build_freq_axis, t1_wait_axis, find_*), so the
numbers match the GUI.

Usage:
    python nv_batch_report.py  <folder-or-files...>  [-o report.html]
    python nv_batch_report.py  D:/Data/July2026
    python nv_batch_report.py  a.h5 b.h5 c.h5 -o run.html
"""
from __future__ import annotations

import os
import re
import io
import sys
import glob
import base64
import datetime
import html as _html
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import nv_data_gui as G          # the analysis engine (must be next to this file)


# ----------------------------------------------------------------- helpers ---
_DATE_RE = re.compile(r"(\d{2})_(\d{2})_(\d{4})_(\d{2})_(\d{2})_(\d{2})")  # MM_DD_YYYY_HH_MM_SS


def _leaf(path):
    return path.split("/")[-1].lower()


def get_field(tree, *names):
    """First leaf value whose name matches any of `names` (case-insensitive)."""
    want = {n.lower() for n in names}
    for path, v in G.flatten_tree(tree):
        if _leaf(path) in want:
            return v
    return None


def _as_float(v):
    try:
        if isinstance(v, np.ndarray):
            v = np.squeeze(v)
            return float(v) if v.ndim == 0 else None
        return float(v)
    except Exception:
        return None


def _as_time(v):
    """Interpret a stored start/end time as epoch seconds (number, or a parseable
    date/time string)."""
    f = _as_float(v)
    if f is not None:
        return f
    if isinstance(v, str):
        s = v.strip()
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%m/%d/%Y %H:%M:%S",
                    "%H:%M:%S"):
            try:
                return datetime.datetime.strptime(s[:19], fmt).timestamp()
            except Exception:
                pass
    return None


def file_date(path, tree):
    m = _DATE_RE.search(os.path.basename(path))
    if m:
        mo, da, yr, hh, mm, ss = map(int, m.groups())
        try:
            return datetime.datetime(yr, mo, da, hh, mm, ss)
        except Exception:
            pass
    for key in ("timestamp", "datetime", "date", "created", "start_time"):
        v = get_field(tree, key)
        t = _as_time(v)
        if t is not None and t > 1e8:            # looks like an epoch
            try:
                return datetime.datetime.fromtimestamp(t)
            except Exception:
                pass
    try:
        return datetime.datetime.fromtimestamp(os.path.getmtime(path))
    except Exception:
        return None


def experiment_duration(tree):
    st = None
    for k in ("start_time", "st", "time_start", "starttime", "acq_start", "start"):
        st = get_field(tree, k)
        if st is not None:
            break
    et = None
    for k in ("end_time", "et", "time_end", "endtime", "acq_end", "end"):
        et = get_field(tree, k)
        if et is not None:
            break
    a, b = _as_time(st), _as_time(et)
    if a is None or b is None:
        return None
    d = b - a
    return d if d >= 0 else None


def fmt_duration(sec):
    if sec is None:
        return "—"
    sec = float(sec)
    h, r = divmod(int(round(sec)), 3600)
    m, s = divmod(r, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{sec:.3g} s"


def get_sequence_text(tree):
    v = get_field(tree, "sequence_text", "sequence")
    if isinstance(v, str) and "sequence:" in v.lower():
        return v
    for _, val in G.flatten_tree(tree):
        if isinstance(val, str) and "pulse" in val.lower() and "channel" in val.lower():
            return val
    return None


def has_freq_array(tree):
    _, fr = G.get_by_name(tree, "frequencies", "freq_list", "freq_axis", want=("1d",))
    return fr is not None


def collect_settings(tree):
    """All scalar / string / short-array leaves (the settings & parameters);
    the big data arrays are skipped."""
    out = []
    for path, v in G.flatten_tree(tree):
        if isinstance(v, str):
            s = v if len(v) <= 300 else v[:300] + " …"
            out.append((path, s))
        elif isinstance(v, (bool, int, float, np.integer, np.floating)):
            out.append((path, f"{v}"))
        elif isinstance(v, np.ndarray):
            a = np.squeeze(v)
            if a.ndim == 0:
                out.append((path, f"{float(a):.6g}"))
            elif a.size <= 8:
                out.append((path, "[" + ", ".join(f"{x:.4g}" for x in a.ravel()) + "]"))
    out.sort(key=lambda kv: kv[0].lower())
    return out


def get_sig_ref(tree):
    """Averaged signal & reference (mean of the individual runs when present)."""
    sig = ref = None
    av = G.find_averaging(tree)
    if av:
        runs = av.get("runs", {})
        if runs.get("signal") and len(runs["signal"]) >= 2:
            sig = np.mean(np.vstack([np.asarray(r, float) for r in runs["signal"]]), 0)
        if runs.get("reference") and len(runs["reference"]) >= 2:
            ref = np.mean(np.vstack([np.asarray(r, float) for r in runs["reference"]]), 0)
        sig = sig if sig is not None else av.get("avg", {}).get("signal")
        ref = ref if ref is not None else av.get("avg", {}).get("reference")
    if sig is None:
        _, sig = G.get_by_name(tree, "signal_counts", "signal_avg", "signal", want=("1d",))
    if ref is None:
        _, ref = G.get_by_name(tree, "reference_counts", "reference_avg", "reference",
                               want=("1d",))
    sig = np.squeeze(np.asarray(sig, float)) if sig is not None else None
    ref = np.squeeze(np.asarray(ref, float)) if ref is not None else None
    return sig, ref


def pulsed_quantity(sig, ref):
    if sig is None:
        return None, None
    if ref is not None and ref.shape == sig.shape and np.all(np.abs(ref) > 1e-12):
        return sig / ref, "signal / reference"
    return sig, "signal (counts)"


# -------------------------------------------------- experiment-type detect ---
def experiment_type(tree):
    if G.find_camera_sweep(tree):
        return "camera_sweep"
    if G.find_confocal(tree) is not None:
        return "confocal"
    seq = get_sequence_text(tree)
    if seq:
        return G.detect_pulsed_mode(seq, has_freq_array(tree))
    if G.find_odmr_channels(tree):
        return "cw_odmr"
    return "unknown"


_TYPE_LABEL = {
    "rabi_time": "Rabi (time)", "rabi_amp": "Rabi (amplitude)",
    "ramsey": "Ramsey", "echo": "Hahn echo", "t1": "T1 relaxation",
    "odmr": "pulsed ODMR", "phase": "phase sweep", "cw_odmr": "CW ODMR",
    "confocal": "confocal scan", "camera_sweep": "ODMR camera sweep",
    "unknown": "unknown",
}


# --------------------------------------------------------------- analysis ----
def _freq_to_ghz(x):
    x = np.asarray(x, float)
    return (x / 1e9, True) if np.nanmax(np.abs(x)) > 1e6 else (x, False)


def analyze_by_type(tree, typ, seq):
    """Return dict(x, y, xlabel, ylabel, image, fit, result) for one file.
    `fit` is dict(xf, yf) or None; `result` is the one-line headline."""
    rec = dict(x=None, y=None, xlabel="", ylabel="", image=None, fit=None, result="")

    # ---- confocal: 2-D Gaussian on the scan image --------------------------
    if typ == "confocal":
        conf = G.find_confocal(tree)
        img = np.asarray(conf["img"], float)
        rec["image"] = img
        r = G.fit_gaussian_2d(img, circular=True)
        if r.get("ok"):
            rec["fit"] = {"gauss": r}
            rec["result"] = (f"brightest spot (x, y) = ({r['x0']:.1f}, {r['y0']:.1f}) px"
                             f" · sigma = {r['sigma']:.2f} px"
                             f" · FWHM = {r['fwhm']:.2f} px")
        else:
            rec["result"] = "2-D Gaussian fit failed"
        return rec

    # ---- camera sweep: whole-frame mean vs frequency, Lorentzian dip -------
    if typ == "camera_sweep":
        cs = G.find_camera_sweep(tree)
        f = np.asarray(cs["freqs"], float)
        y = np.array([float(np.nanmean(im)) for im in cs["images"]], float)
        keep = np.isfinite(f) & np.isfinite(y) & (f > 0)
        f, y = f[keep], y[keep]
        rec.update(x=f, y=y, xlabel="frequency (GHz)", ylabel="mean counts")
        r = G.fit_odmr_dips(f, y, n_dips=1, bg_degree=0)
        if r.get("ok") and r["centers"]:
            rec["fit"] = {"xf": r["x_dense"], "yf": r["y_dense"]}
            rec["result"] = (f"resonance = {r['centers'][0]:.5f} GHz · "
                             f"FWHM = {2 * r['widths'][0] * 1e3:.3g} MHz "
                             f"(whole-frame mean)")
        else:
            rec["result"] = "Lorentzian fit failed"
        return rec

    # ---- CW ODMR -----------------------------------------------------------
    if typ == "cw_odmr":
        _, fr = G.get_by_name(tree, "frequencies", "freq_list", "freq_axis", want=("1d",))
        _, c = G.get_by_name(tree, "counts_averaged", "counts_avg", "counts_forward",
                             "counts", want=("1d",))
        if fr is None or c is None:
            rec["result"] = "no frequency/counts arrays"
            return rec
        f = np.squeeze(np.asarray(fr, float))
        c = np.squeeze(np.asarray(c, float))
        n = min(len(f), len(c))
        f, c = f[:n], c[:n]
        keep = np.isfinite(f) & np.isfinite(c) & (f > 0) & (c != 0)
        if keep.sum() >= 3:
            f, c = f[keep], c[keep]
        fg, _ = _freq_to_ghz(f)
        rec.update(x=fg, y=c, xlabel="frequency (GHz)", ylabel="counts")
        r = G.fit_odmr_dips(fg, c, n_dips=1, bg_degree=0)
        if r.get("ok") and r["centers"]:
            rec["fit"] = {"xf": r["x_dense"], "yf": r["y_dense"]}
            rec["result"] = (f"resonance = {r['centers'][0]:.5f} GHz · "
                             f"FWHM = {2 * r['widths'][0] * 1e3:.3g} MHz")
        else:
            rec["result"] = "Lorentzian fit failed"
        return rec

    # ---- pulsed experiments (rabi / ramsey / echo / t1 / odmr / phase) -----
    sig, ref = get_sig_ref(tree)
    y, ylabel = pulsed_quantity(sig, ref)
    if y is None:
        rec["result"] = "no signal/reference arrays"
        return rec
    n = len(y)

    if typ == "odmr":
        x, xlabel = G.build_freq_axis(seq, n)
        if not xlabel.startswith("frequency"):
            _, fr = G.get_by_name(tree, "frequency_range", "frequencies", want=("1d",))
            if fr is not None:
                fr = np.squeeze(np.asarray(fr, float))
                x = fr[:n] if len(fr) == n else np.linspace(fr[0], fr[-1], n)
                xlabel = "frequency (Hz)"
        x, _ = _freq_to_ghz(x)
        xlabel = "frequency (GHz)"
    elif typ == "rabi_amp":
        x, xlabel = G.build_amp_axis(seq, n)
    elif typ == "t1":
        x, xlabel = G.t1_wait_axis(seq, n)
        if x is None:
            x, xlabel = G.build_time_axis(seq, n)
    else:                                         # rabi_time, ramsey, echo, phase
        x, xlabel = G.build_time_axis(seq, n)

    x = np.asarray(x, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    rec.update(x=x, y=y, xlabel=xlabel, ylabel=ylabel)

    if typ in ("rabi_time", "rabi_amp"):
        r = G.fit_model("Damped cosine (Rabi)", x, y)
        if r["ok"]:
            f = abs(r["params"][r["pnames"].index("f")])
            rec["fit"] = {"xf": r["x_dense"], "yf": r["y_dense"]}
            if typ == "rabi_amp":                 # f is cycles per Volt
                rec["result"] = (f"Rabi period = {1.0 / f:.4g} V · "
                                 f"pi-pulse amplitude = {0.5 / f:.4g} V" if f > 0
                                 else "no oscillation")
            else:                                 # x in ns -> f in GHz
                rec["result"] = (f"Rabi frequency = {f * 1e3:.4g} MHz · "
                                 f"pi-pulse = {0.5 / f:.4g} ns" if f > 0
                                 else "no oscillation")
        else:
            rec["result"] = "Rabi fit failed"
    elif typ == "ramsey":
        r = G.fit_model("Ramsey (Gaussian\u00d7cos)", x, y)
        if r["ok"]:
            T = abs(r["params"][r["pnames"].index("T")])
            fdet = abs(r["params"][r["pnames"].index("f")]) * 1e3
            rec["fit"] = {"xf": r["x_dense"], "yf": r["y_dense"]}
            rec["result"] = f"T2* = {T:.4g} ns · detuning \u2248 {fdet:.4g} MHz"
        else:
            rec["result"] = "Ramsey fit failed"
    elif typ == "echo":
        r = G.fit_model("Stretched exp (echo)", x, y)
        if r["ok"]:
            T = abs(r["params"][r["pnames"].index("T")])
            rec["fit"] = {"xf": r["x_dense"], "yf": r["y_dense"]}
            rec["result"] = f"T2 (echo) = {T:.4g} ns"
        else:
            rec["result"] = "echo fit failed"
    elif typ == "t1":
        r = G.fit_model("Exponential decay", x, y)
        if r["ok"]:
            T1 = abs(r["params"][r["pnames"].index("tau")])
            unit = xlabel.split("(")[-1].rstrip(")") if "(" in xlabel else "ns"
            rec["fit"] = {"xf": r["x_dense"], "yf": r["y_dense"]}
            rec["result"] = f"T1 = {T1:.4g} {unit}"
        else:
            rec["result"] = "T1 fit failed"
    elif typ == "odmr":                            # pulsed ODMR: Lorentzian dip
        r = G.fit_odmr_dips(x, y, n_dips=1, bg_degree=0)
        if r.get("ok") and r["centers"]:
            rec["fit"] = {"xf": r["x_dense"], "yf": r["y_dense"]}
            rec["result"] = (f"resonance = {r['centers'][0]:.5f} GHz · "
                             f"FWHM = {2 * r['widths'][0] * 1e3:.3g} MHz")
        else:
            rec["result"] = "Lorentzian fit failed"
    else:                                          # phase / unknown pulsed
        rec["result"] = f"{_TYPE_LABEL.get(typ, typ)} (no standard scalar)"
    return rec


def analyze_tree(tree, path, date=None):
    rec = {"path": path, "name": os.path.basename(path),
           "date": date, "error": None, "settings": []}
    try:
        rec["date"] = date or file_date(path, tree)
        rec["settings"] = collect_settings(tree)
        rec["duration"] = experiment_duration(tree)
        rec["sequence"] = get_sequence_text(tree)
        typ = experiment_type(tree)
        rec["type"] = typ
        rec.update(analyze_by_type(tree, typ, rec["sequence"]))
    except Exception as e:  # keep going for the other files
        rec["error"] = f"{type(e).__name__}: {e}"
        rec.setdefault("type", "error")
        rec.setdefault("result", "")
    rec["date_str"] = (rec["date"].strftime("%Y-%m-%d %H:%M:%S")
                       if rec.get("date") else "unknown")
    rec["duration_str"] = fmt_duration(rec.get("duration"))
    return rec


# ------------------------------------------------------------------ plots ----
def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def plot_data(rec, with_fit, mode="au"):
    """Return a base64 PNG of the data (and fit). mode: "au"=normalized, "raw"=counts."""
    if rec.get("image") is not None:
        img = rec["image"]
        fig, ax = plt.subplots(figsize=(4.2, 3.4))
        im = ax.imshow(img, origin="lower", cmap="viridis", aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
        g = (rec.get("fit") or {}).get("gauss")
        if with_fit and g and g.get("ok"):
            th = np.linspace(0, 2 * np.pi, 100)
            ax.plot(g["x0"] + g["fwhm_x"] / 2 * np.cos(th),
                    g["y0"] + g["fwhm_y"] / 2 * np.sin(th), "-", color="magenta", lw=1.3)
            ax.plot([g["x0"]], [g["y0"]], "+", color="magenta", ms=12, mew=1.8)
            ax.set_title("image + 2-D Gaussian", fontsize=9)
        else:
            ax.set_title("image", fontsize=9)
        return _png(fig)

    x, y = rec.get("x"), rec.get("y")
    if x is None or y is None or len(x) == 0:
        return None
    # normalize to arbitrary units (baseline = 1.0, 10% contrast -> 0.9), like the
    # GUI and the papers; scale the fit curve by the SAME baseline so it overlays
    if mode == "raw":
        base, ylabel = 1.0, (rec.get("ylabel") or "counts")
    else:
        base, ylabel = G._au_baseline(y), "normalized signal (a.u.)"
    yn = np.asarray(y, float) / base
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    ax.plot(x, yn, "o", ms=3, color="tab:blue", label="data")
    if with_fit and rec.get("fit") and "xf" in rec["fit"]:
        ax.plot(rec["fit"]["xf"], np.asarray(rec["fit"]["yf"], float) / base, "-",
                color="tab:red", lw=1.6, label="fit")
        ax.legend(fontsize=8)
    ax.set_xlim(float(np.min(x)), float(np.max(x)))
    ax.set_xlabel(rec.get("xlabel", "")); ax.set_ylabel(ylabel)
    ax.set_title(("data + fit" if with_fit else "data"), fontsize=9)
    ax.grid(alpha=0.3)
    return _png(fig)


# ------------------------------------------------------------------- HTML ----
def _esc(s):
    return _html.escape(str(s))


def _img_tag(b64, alt=""):
    if not b64:
        return '<div class="noplot">(no plot)</div>'
    return f'<img alt="{_esc(alt)}" src="data:image/png;base64,{b64}">'


def plots_div(r, units):
    """The data / data+fit plot pair for one file. For units='both', both the
    a.u. and the raw-counts versions are embedded and toggled in the browser."""
    if units in ("au", "raw"):
        return ('<div class="plots">'
                + _img_tag(plot_data(r, False, units), "data")
                + _img_tag(plot_data(r, True, units), "data+fit") + '</div>')
    if r.get("image") is not None:        # confocal image is identical in both units
        return ('<div class="plots">'
                + _img_tag(plot_data(r, False, "au"), "data")
                + _img_tag(plot_data(r, True, "au"), "data+fit") + '</div>')
    return ('<div class="plots">'
            + '<span class="u-au">'
            + _img_tag(plot_data(r, False, "au"), "data")
            + _img_tag(plot_data(r, True, "au"), "data+fit") + '</span>'
            + '<span class="u-raw">'
            + _img_tag(plot_data(r, False, "raw"), "data")
            + _img_tag(plot_data(r, True, "raw"), "data+fit") + '</span></div>')


def make_html(records, units="au"):
    records = sorted(records, key=lambda r: (r.get("date") or datetime.datetime.max))
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parts = ['<!doctype html><html><head><meta charset="utf-8">',
             '<title>NV batch report</title><style>',
             'body{font-family:system-ui,Arial,sans-serif;margin:24px;color:#1b1b1b}',
             'h1{font-size:20px} h2{font-size:15px;margin:2px 0}',
             'table.summary{border-collapse:collapse;margin:12px 0;font-size:13px}',
             'table.summary th,table.summary td{border:1px solid #ccc;padding:4px 8px;'
             'text-align:left}',
             'table.summary th{background:#f3f3f3}',
             '.card{border:1px solid #ddd;border-radius:8px;padding:14px;margin:16px 0;'
             'box-shadow:0 1px 3px rgba(0,0,0,.06)}',
             '.meta{color:#555;font-size:12.5px;margin:2px 0 8px}',
             '.result{background:#eef6ff;border-left:4px solid #2b7;padding:6px 10px;'
             'border-radius:4px;font-size:13.5px;margin:6px 0}',
             '.plots{display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start}',
             '.plots img{max-width:430px;border:1px solid #eee;border-radius:4px}',
             '.noplot{color:#999;font-style:italic;padding:20px}',
             'pre.seq{background:#fafafa;border:1px solid #eee;border-radius:4px;'
             'padding:8px;font-size:12px;overflow:auto;white-space:pre-wrap}',
             'details{margin-top:8px} summary{cursor:pointer;font-weight:600;'
             'font-size:13px}',
             'table.settings{border-collapse:collapse;margin-top:6px;font-size:12px}',
             'table.settings td{border:1px solid #eee;padding:2px 8px;'
             'vertical-align:top}',
             'td.k{color:#036;font-family:monospace;white-space:nowrap}',
             '.err{color:#b00}',
             '.u-raw{display:none}',
             'body.show-raw .u-au{display:none} body.show-raw .u-raw{display:inline}',
             '.toggle{margin:10px 0;font-size:13px}',
             '.toggle button{margin-left:6px;padding:3px 10px;border:1px solid #bbb;'
             'border-radius:4px;background:#f6f6f6;cursor:pointer}',
             '.toggle button.active{background:#2b7;color:#fff;border-color:#2b7}',
             '</style></head><body>',
             f'<h1>NV batch report</h1><div class="meta">generated {now} · '
             f'{len(records)} file(s), chronological</div>']

    # summary table
    parts.append('<table class="summary"><tr><th>#</th><th>date</th><th>file</th>'
                 '<th>type</th><th>duration</th><th>result</th></tr>')
    for i, r in enumerate(records, 1):
        res = r.get("result") or (f'<span class="err">{_esc(r["error"])}</span>'
                                   if r.get("error") else "")
        parts.append(f'<tr><td>{i}</td><td>{_esc(r["date_str"])}</td>'
                     f'<td>{_esc(r["name"])}</td>'
                     f'<td>{_esc(_TYPE_LABEL.get(r.get("type"), r.get("type")))}</td>'
                     f'<td>{_esc(r["duration_str"])}</td><td>{res}</td></tr>')
    parts.append('</table>')
    if units == "both":
        parts.append('<div class="toggle">Y-axis:'
                     '<button id="bAU" onclick="setU(\'au\')">normalized (a.u.)</button>'
                     '<button id="bRAW" onclick="setU(\'raw\')">raw counts</button></div>')

    # per-file cards
    for i, r in enumerate(records, 1):
        parts.append('<div class="card">')
        parts.append(f'<h2>{i}. {_esc(r["name"])}</h2>')
        parts.append(f'<div class="meta">{_esc(r["date_str"])} &nbsp;·&nbsp; '
                     f'type: <b>{_esc(_TYPE_LABEL.get(r.get("type"), r.get("type")))}</b>'
                     f' &nbsp;·&nbsp; duration: {_esc(r["duration_str"])}</div>')
        if r.get("error"):
            parts.append(f'<div class="result err">error: {_esc(r["error"])}</div>')
        if r.get("result"):
            parts.append(f'<div class="result">{_esc(r["result"])}</div>')
        parts.append(plots_div(r, units))
        if r.get("sequence"):
            parts.append('<details><summary>sequence text</summary>'
                         f'<pre class="seq">{_esc(r["sequence"])}</pre></details>')
        settings = r.get("settings") or []
        parts.append(f'<details><summary>settings / parameters '
                     f'({len(settings)})</summary><table class="settings">')
        for k, v in settings:
            parts.append(f'<tr><td class="k">{_esc(k)}</td><td>{_esc(v)}</td></tr>')
        parts.append('</table></details>')
        parts.append('</div>')

    if units == "both":
        parts.append('<script>function setU(u){'
                     'document.body.classList.toggle("show-raw",u==="raw");'
                     'document.getElementById("bAU").classList.toggle("active",u!=="raw");'
                     'document.getElementById("bRAW").classList.toggle("active",u==="raw");}'
                     'setU("au");</script>')
    parts.append('</body></html>')
    return "".join(parts)


# ------------------------------------------------------------------- main ----
def collect_paths(args):
    paths = []
    for a in args:
        if os.path.isdir(a):
            paths += glob.glob(os.path.join(a, "**", "*.h5"), recursive=True)
            paths += glob.glob(os.path.join(a, "**", "*.hdf5"), recursive=True)
        else:
            paths += glob.glob(a)
    seen, out = set(), []
    for p in paths:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap); out.append(p)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Batch NV-data HTML report.")
    ap.add_argument("inputs", nargs="+", help="HDF5 files and/or folders")
    ap.add_argument("-o", "--output", default="nv_report.html", help="output HTML path")
    ap.add_argument("-u", "--units", choices=["au", "raw", "both"], default="au",
                    help="y-axis: au=normalized a.u. (default), raw=counts, "
                         "both=interactive toggle")
    a = ap.parse_args(argv)

    paths = collect_paths(a.inputs)
    if not paths:
        print("No .h5 / .hdf5 files found."); return 1
    records = []
    for p in paths:
        print("analysing", os.path.basename(p), "...", flush=True)
        try:
            tree = G.load_hdf5_tree(p)
            records.append(analyze_tree(tree, p))
        except Exception as e:
            records.append({"path": p, "name": os.path.basename(p), "date": None,
                            "date_str": "unknown", "duration_str": "—",
                            "type": "error", "result": "", "settings": [],
                            "error": f"could not read: {type(e).__name__}: {e}"})
    html_text = make_html(records, units=a.units)
    with open(a.output, "w", encoding="utf-8") as fh:
        fh.write(html_text)
    print(f"\nwrote {a.output}  ({len(records)} file(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
