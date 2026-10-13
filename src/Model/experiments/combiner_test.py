# Created by Jannet Trabelsi on 2026-10-08
# Combiner qualification: ZN2PD2-50-S+ (2-way, 500-5000 MHz)
#
# Scalar network analysis of a used power combiner. The SG384 is the source and
# the Agilent 8596E is the receiver. The ZGBDC6-362HP+ bidirectional coupler is
# used as a reflectometer for the return-loss step. A pop-up window before each
# step lists the exact cabling for that step; the RF source is held OFF during
# every reconnection. Results are written as CSV files and PNG plots into a
# timestamped folder.
#
# Units: SG384 frequency in Hz, spectrum analyzer frequency in MHz, power in dBm.
# Drive level is small-signal (0 dBm), far below the 1 W combiner limit and the
# +30 dBm spectrum-analyzer damage limit.
#
# Port naming (ZN2PD2-50-S+): S = sum port, 1 and 2 = the two split ports.

import os
import time
import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import messagebox

from src.Controller import SG384Generator
from src.Controller.Proteus_device import ProteusDevice
from src.Controller.Agilent8596E import Agilent8596E

# ----------------------------------------------------------------------------
# Configuration  (edit here for the 4-way parts, see note at the bottom)
# ----------------------------------------------------------------------------
COMBINER_LABEL = "ZN2PD2-50-S+ (2-way)"

# Test frequencies: lowest Cu line (486), through the pump band, up to the observer
FREQS_MHZ = [486, 500, 700, 800, 1000, 1104, 1500, 2000, 2400, 2500]

DRIVE_DBM = 0.0              # small-signal source level

# Spectrum analyzer acquisition (narrow window around each tone)
SPAN_MHZ = 2.0
RBW_MHZ = 0.1
VBW_MHZ = 0.1
SA_REF_LEVEL_DBM = 10        # display reference level
SA_ATTEN_DB = 10             # input attenuation (8596E minimum)

SETTLE_S = 0.25              # settle time after a frequency/level change

# Two-tone test (real combiner configuration)
PUMP_FREQ_MHZ = 800.0        # pump tone, supplied by the Proteus (2.5 GS/s unit)
OBS_FREQ_MHZ = 2400.0        # observer tone, supplied by the SG384
PROTEUS_CHANNEL = 1
PROTEUS_SIN_AMP = 0.5        # Proteus function-generator amplitude (0-1 scale)
TWO_TONE_FREQS_MHZ = [800, 1600, 2400, 3200, 4000, 4800]  # fundamentals + products

# Pass / fail thresholds (ZN2PD2-50-S+ datasheet, 0.48-2.5 GHz)
IL_MAX_DB = 4.5              # insertion loss each input -> sum (ideal ~3.7-3.8)
BALANCE_MAX_DB = 0.3         # amplitude balance between the two inputs
ISO_MIN_DB = 18.0            # port-to-port isolation (typ 24-26)
ISO_REJECT_DB = 15.0         # below this => damaged isolation resistor
RL_MIN_DB = 18.0             # return loss, all ports (VSWR <= ~1.25)
SPUR_MAX_DBC = -50.0         # two-tone spurs

OUTDIR = os.path.join(
    os.getcwd(),
    "combiner_test_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
)


# ----------------------------------------------------------------------------
# Pop-up helpers
# ----------------------------------------------------------------------------
def notify(title, message):
    """Block until the operator acknowledges the cabling instructions."""
    messagebox.showinfo(title, message)


def confirm(title, message):
    """Return True/False from a yes/no prompt (used for the manual checks)."""
    return messagebox.askyesno(title, message)


# ----------------------------------------------------------------------------
# Spectrum-analyzer helpers
# ----------------------------------------------------------------------------
def sa_setup(sa):
    """Put the analyzer in single-sweep, real-number trace mode with fixed BW/RL."""
    sa.initialize_sa()
    sa.set_trace_data_transfer("P")
    sa.set_attenuation(SA_ATTEN_DB)
    sa.write("RL " + str(SA_REF_LEVEL_DBM) + "DB;")
    print("setting span")
    sa.set_span(SPAN_MHZ)
    print("setting res bw")
    sa.set_resolution_bw(RBW_MHZ)
    print("setting vbw")
    sa.set_video_bw(VBW_MHZ)
    print("setting single sweep")
    sa.set_single_sweep()


def tone_level_dbm(sa, freq_mhz):
    """Peak power (dBm) in a narrow window centered on freq_mhz."""
    sa.set_center_frequency(freq_mhz)
    sa.set_span(SPAN_MHZ)
    time.sleep(SETTLE_S)
    sa.take_sweep()
    trace = np.array(sa.get_trace_a_data("P"), dtype=float)
    return float(np.max(trace))


def sweep_levels(sg384, sa, freqs_mhz):
    """Enable the RF source, read the peak level at each frequency, then disable it."""
    sg384.set_power(DRIVE_DBM)
    sg384._send("ENBR 1")            # type-N RF output on
    time.sleep(SETTLE_S)
    levels = []
    for f in freqs_mhz:
        sg384.set_frequency(f * 1e6)
        time.sleep(SETTLE_S)
        levels.append(tone_level_dbm(sa, f))
    sg384._send("ENBR 0")            # type-N RF output off
    return levels


# ----------------------------------------------------------------------------
# Steps
# ----------------------------------------------------------------------------
def step1_visual():
    """Connector inspection and DC continuity (manual, meter-based)."""
    notify(
        "Step 1 of 7 - Visual + DC check",
        "Inspect all three SMA connectors on " + COMBINER_LABEL + ":\n"
        " - no bent/recessed center pins, no debris, clean threads\n"
        " - torque the connectors\n\n"
        "With a DMM (resistance):\n"
        " - S to 1 : expect a DC short (DC-pass)\n"
        " - S to 2 : expect a DC short\n"
        " - 1 to 2 : expect roughly 100 ohm (internal isolation resistor)\n\n"
        "Click OK once the checks are done.",
    )
    pins_ok = confirm("Step 1", "Connectors and pins in good condition?")
    dc_ok = confirm("Step 1", "S->1 short, S->2 short, 1<->2 near 100 ohm?")
    return {"connectors_ok": pins_ok, "dc_ok": dc_ok}


def step2_through_cal(sg384, sa):
    """Reference sweep with a barrel in place of the combiner."""
    notify(
        "Step 2 of 7 - Through calibration",
        "Cabling (RF source is OFF now):\n"
        " SG384 RF OUT -> cable A -> SMA barrel -> cable B -> SA input\n"
        " (no combiner in the path)\n\n"
        "This records the 0 dB reference, including both cables.\n"
        "Click OK to run the reference sweep.",
    )
    return sweep_levels(sg384, sa, FREQS_MHZ)


def step3_insertion_loss(sg384, sa, ref):
    """Insertion loss for each input to the sum port."""
    notify(
        "Step 3a of 7 - Insertion loss (port 1 -> S)",
        "Cabling (RF source is OFF now):\n"
        " SG384 -> cable A -> combiner port 1\n"
        " combiner port 2 -> 50 ohm load\n"
        " combiner S -> cable B -> SA input\n\n"
        "Click OK to sweep.",
    )
    p1s = sweep_levels(sg384, sa, FREQS_MHZ)

    notify(
        "Step 3b of 7 - Insertion loss (port 2 -> S)",
        "Cabling (RF source is OFF now):\n"
        " SG384 -> cable A -> combiner port 2\n"
        " combiner port 1 -> 50 ohm load\n"
        " combiner S -> cable B -> SA input\n\n"
        "Click OK to sweep.",
    )
    p2s = sweep_levels(sg384, sa, FREQS_MHZ)

    n = len(FREQS_MHZ)
    il_1s = [ref[i] - p1s[i] for i in range(n)]
    il_2s = [ref[i] - p2s[i] for i in range(n)]
    balance = [abs(il_1s[i] - il_2s[i]) for i in range(n)]
    return {"P1S": p1s, "P2S": p2s, "IL_1S": il_1s, "IL_2S": il_2s, "balance": balance}


def step4_isolation(sg384, sa, ref):
    """Port-to-port isolation (key test for a used Wilkinson combiner)."""
    notify(
        "Step 4a of 7 - Isolation (port 1 -> port 2)",
        "Cabling (RF source is OFF now):\n"
        " SG384 -> cable A -> combiner port 1\n"
        " combiner S -> 50 ohm load\n"
        " combiner port 2 -> cable B -> SA input\n\n"
        "Click OK to sweep.",
    )
    leak12 = sweep_levels(sg384, sa, FREQS_MHZ)

    notify(
        "Step 4b of 7 - Isolation (port 2 -> port 1)",
        "Cabling (RF source is OFF now):\n"
        " SG384 -> cable A -> combiner port 2\n"
        " combiner S -> 50 ohm load\n"
        " combiner port 1 -> cable B -> SA input\n\n"
        "Click OK to sweep.",
    )
    leak21 = sweep_levels(sg384, sa, FREQS_MHZ)

    n = len(FREQS_MHZ)
    iso12 = [ref[i] - leak12[i] for i in range(n)]
    iso21 = [ref[i] - leak21[i] for i in range(n)]
    return {"leak12": leak12, "leak21": leak21, "iso12": iso12, "iso21": iso21}


def step5_return_loss(sg384, sa):
    """Return loss at S, 1 and 2 using the ZGBDC6 coupler as a reflectometer."""
    notify(
        "Step 5a of 7 - Return-loss reference",
        "Cabling (RF source is OFF now):\n"
        " SG384 -> ZGBDC6 mainline IN\n"
        " ZGBDC6 reverse-coupled port -> SA input\n"
        " ZGBDC6 mainline OUT -> leave OPEN (or cap with an SMA short)\n\n"
        "This full-reflection case sets the 0 dB return-loss reference.\n"
        "Click OK to sweep.",
    )
    r_ref = sweep_levels(sg384, sa, FREQS_MHZ)

    notify(
        "Step 5b of 7 - Return loss, S port",
        "Cabling (RF source is OFF now):\n"
        " ZGBDC6 mainline OUT -> combiner S\n"
        " combiner 1 -> 50 ohm load, combiner 2 -> 50 ohm load\n\n"
        "Click OK to sweep.",
    )
    r_s = sweep_levels(sg384, sa, FREQS_MHZ)

    notify(
        "Step 5c of 7 - Return loss, port 1",
        "Cabling (RF source is OFF now):\n"
        " ZGBDC6 mainline OUT -> combiner port 1\n"
        " combiner S -> 50 ohm load, combiner 2 -> 50 ohm load\n\n"
        "Click OK to sweep.",
    )
    r_1 = sweep_levels(sg384, sa, FREQS_MHZ)

    notify(
        "Step 5d of 7 - Return loss, port 2",
        "Cabling (RF source is OFF now):\n"
        " ZGBDC6 mainline OUT -> combiner port 2\n"
        " combiner S -> 50 ohm load, combiner 1 -> 50 ohm load\n\n"
        "Click OK to sweep.",
    )
    r_2 = sweep_levels(sg384, sa, FREQS_MHZ)

    n = len(FREQS_MHZ)
    rl_s = [r_ref[i] - r_s[i] for i in range(n)]
    rl_1 = [r_ref[i] - r_1[i] for i in range(n)]
    rl_2 = [r_ref[i] - r_2[i] for i in range(n)]
    return {"r_ref": r_ref, "RL_S": rl_s, "RL_1": rl_1, "RL_2": rl_2}


def step6_two_tone(sg384, proteus, sa):
    """Two tones combined at once; look for intermodulation / spurs."""
    notify(
        "Step 6 of 7 - Two-tone test",
        "Cabling (sources are OFF now):\n"
        " SG384 -> combiner port 1   (observer tone, " + str(OBS_FREQ_MHZ) + " MHz)\n"
        " Proteus ch " + str(PROTEUS_CHANNEL) + " -> combiner port 2   (pump tone, "
        + str(PUMP_FREQ_MHZ) + " MHz)\n"
        " combiner S -> SA input\n\n"
        "Note: the Proteus generating " + str(PUMP_FREQ_MHZ)
        + " MHz must be the 2.5 GS/s unit (P2582M).\n"
        "Click OK to apply both tones and sweep.",
    )
    # Observer tone on the SG384
    sg384.set_frequency(OBS_FREQ_MHZ * 1e6)
    sg384.set_power(DRIVE_DBM)
    sg384._send("ENBR 1")
    # Pump tone on the Proteus
    proteus.set_function_generator(PROTEUS_CHANNEL, 'SIN', PUMP_FREQ_MHZ * 1e6,
                                   PROTEUS_SIN_AMP, 0.0, 0.0)
    time.sleep(SETTLE_S)

    levels = {}
    for f in TWO_TONE_FREQS_MHZ:
        levels[f] = tone_level_dbm(sa, f)

    # Sources off
    sg384._send("ENBR 0")
    proteus.set_function_generator(PROTEUS_CHANNEL, 'SIN', PUMP_FREQ_MHZ * 1e6,
                                   0.0, 0.0, 0.0)

    fundamentals = [PUMP_FREQ_MHZ, OBS_FREQ_MHZ]
    ref_level = max(levels[PUMP_FREQ_MHZ], levels[OBS_FREQ_MHZ])
    rows = []
    for f in TWO_TONE_FREQS_MHZ:
        kind = "fundamental" if f in fundamentals else "spur"
        rows.append({"freq_MHz": f, "level_dBm": levels[f],
                     "type": kind, "dBc": levels[f] - ref_level})
    return rows


def step7_pulsed():
    """Gated envelope check with a TTL switch and an RF detector (manual, scope-based)."""
    notify(
        "Step 7 of 7 - Pulsed / gated check",
        "Cabling:\n"
        " Gate each source with its TTL switch (switch control from a Proteus marker)\n"
        " combined output -> RF diode detector -> oscilloscope\n\n"
        "The 1 GS/s scope cannot digitize the carrier directly; the detector envelope is\n"
        "what is observed. Confirm both channels gate cleanly, pulse timing/shape is\n"
        "correct, and no droop or ringing is introduced by the combiner.\n\n"
        "Click OK once observed.",
    )
    ok = confirm("Step 7", "Gated envelope clean (both channels, no droop/ringing)?")
    return {"pulsed_ok": ok}


# ----------------------------------------------------------------------------
# Output: CSV + PNG
# ----------------------------------------------------------------------------
def save_insertion_loss(ref, d):
    pd.DataFrame({
        "freq_MHz": FREQS_MHZ, "ref_dBm": ref,
        "P1S_dBm": d["P1S"], "IL_1S_dB": d["IL_1S"],
        "P2S_dBm": d["P2S"], "IL_2S_dB": d["IL_2S"],
        "balance_dB": d["balance"],
    }).to_csv(os.path.join(OUTDIR, "insertion_loss.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(FREQS_MHZ, d["IL_1S"], "o-", label="port 1 -> S")
    ax.plot(FREQS_MHZ, d["IL_2S"], "s-", label="port 2 -> S")
    ax.axhline(IL_MAX_DB, color="r", ls="--", label="reject limit (" + str(IL_MAX_DB) + " dB)")
    ax.axhline(3.8, color="g", ls=":", label="ideal ~3.8 dB")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Insertion loss (dB)")
    ax.set_title("Insertion loss - " + COMBINER_LABEL)
    ax.grid(True)
    ax.legend()
    fig.savefig(os.path.join(OUTDIR, "insertion_loss.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_isolation(ref, d):
    pd.DataFrame({
        "freq_MHz": FREQS_MHZ, "ref_dBm": ref,
        "leak_12_dBm": d["leak12"], "iso_12_dB": d["iso12"],
        "leak_21_dBm": d["leak21"], "iso_21_dB": d["iso21"],
    }).to_csv(os.path.join(OUTDIR, "isolation.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(FREQS_MHZ, d["iso12"], "o-", label="port 1 -> 2")
    ax.plot(FREQS_MHZ, d["iso21"], "s-", label="port 2 -> 1")
    ax.axhline(ISO_MIN_DB, color="r", ls="--", label="min (" + str(ISO_MIN_DB) + " dB)")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Isolation (dB)")
    ax.set_title("Port-to-port isolation - " + COMBINER_LABEL)
    ax.grid(True)
    ax.legend()
    fig.savefig(os.path.join(OUTDIR, "isolation.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_return_loss(d):
    pd.DataFrame({
        "freq_MHz": FREQS_MHZ, "ref_dBm": d["r_ref"],
        "RL_S_dB": d["RL_S"], "RL_1_dB": d["RL_1"], "RL_2_dB": d["RL_2"],
    }).to_csv(os.path.join(OUTDIR, "return_loss.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(FREQS_MHZ, d["RL_S"], "o-", label="S port")
    ax.plot(FREQS_MHZ, d["RL_1"], "s-", label="port 1")
    ax.plot(FREQS_MHZ, d["RL_2"], "^-", label="port 2")
    ax.axhline(RL_MIN_DB, color="r", ls="--", label="min (" + str(RL_MIN_DB) + " dB)")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Return loss (dB)")
    ax.set_title("Return loss - " + COMBINER_LABEL)
    ax.grid(True)
    ax.legend()
    fig.savefig(os.path.join(OUTDIR, "return_loss.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_two_tone(rows):
    pd.DataFrame(rows).to_csv(os.path.join(OUTDIR, "two_tone.csv"), index=False)

    freqs = [r["freq_MHz"] for r in rows]
    levels = [r["level_dBm"] for r in rows]
    base = min(levels) - 5

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.vlines(freqs, base, levels, color="gray", lw=1)
    for r in rows:
        c = "g" if r["type"] == "fundamental" else "r"
        ax.plot([r["freq_MHz"]], [r["level_dBm"]], "o", color=c, markersize=7)
    ax.axhline(max(levels) + SPUR_MAX_DBC, color="r", ls="--",
               label="spur limit (" + str(SPUR_MAX_DBC) + " dBc)")
    ax.set_ylim(bottom=base)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Level (dBm)")
    ax.set_title("Two-tone output (green = fundamental, red = spur) - " + COMBINER_LABEL)
    ax.grid(True)
    ax.legend()
    fig.savefig(os.path.join(OUTDIR, "two_tone_spectrum.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_summary(il, iso, rl, tt, man1, man7):
    worst_il = max(max(il["IL_1S"]), max(il["IL_2S"]))
    worst_balance = max(il["balance"])
    worst_iso = min(min(iso["iso12"]), min(iso["iso21"]))
    worst_rl = min(min(rl["RL_S"]), min(rl["RL_1"]), min(rl["RL_2"]))
    worst_spur = max([r["dBc"] for r in tt if r["type"] == "spur"])

    rows = [
        ("insertion_loss_max_dB", round(worst_il, 2), "<= " + str(IL_MAX_DB), bool(worst_il <= IL_MAX_DB)),
        ("amplitude_balance_max_dB", round(worst_balance, 2), "<= " + str(BALANCE_MAX_DB), bool(worst_balance <= BALANCE_MAX_DB)),
        ("isolation_min_dB", round(worst_iso, 2), ">= " + str(ISO_MIN_DB), bool(worst_iso >= ISO_MIN_DB)),
        ("return_loss_min_dB", round(worst_rl, 2), ">= " + str(RL_MIN_DB), bool(worst_rl >= RL_MIN_DB)),
        ("worst_spur_dBc", round(worst_spur, 2), "<= " + str(SPUR_MAX_DBC), bool(worst_spur <= SPUR_MAX_DBC)),
        ("visual_DC_check", "ok" if (man1["connectors_ok"] and man1["dc_ok"]) else "fail", "ok", bool(man1["connectors_ok"] and man1["dc_ok"])),
        ("pulsed_envelope_check", "ok" if man7["pulsed_ok"] else "fail", "ok", bool(man7["pulsed_ok"])),
    ]
    df = pd.DataFrame(rows, columns=["metric", "value", "threshold", "pass"])
    df.to_csv(os.path.join(OUTDIR, "summary.csv"), index=False)
    return df, worst_iso


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    os.makedirs(OUTDIR, exist_ok=True)

    root = tk.Tk()
    root.withdraw()

    notify("Combiner test",
           "Qualification of " + COMBINER_LABEL + ".\n"
           "Seven steps follow. A pop-up before each step lists the cabling.\n\n"
           "Output folder:\n" + OUTDIR)
    print("connecting to sg384")
    sg384 = SG384Generator()
    print("connecting to proteus")
    proteus = ProteusDevice()
    print("connecting to sa")
    sa = Agilent8596E()
    sa_setup(sa)
    print("setup_1")

    man1 = step1_visual()
    ref = step2_through_cal(sg384, sa)
    il = step3_insertion_loss(sg384, sa, ref)
    iso = step4_isolation(sg384, sa, ref)
    rl = step5_return_loss(sg384, sa)
    tt = step6_two_tone(sg384, proteus, sa)
    man7 = step7_pulsed()

    save_insertion_loss(ref, il)
    save_isolation(ref, iso)
    save_return_loss(rl)
    save_two_tone(tt)
    summary, worst_iso = save_summary(il, iso, rl, tt, man1, man7)

    sa.set_continuous_sweep()        # leave the analyzer live

    verdict = "PASS" if bool(summary["pass"].all()) else "REVIEW"
    extra = ""
    if worst_iso < ISO_REJECT_DB:
        extra = ("\nIsolation below " + str(ISO_REJECT_DB)
                 + " dB: likely a damaged isolation resistor -> reject this unit.")
    notify("Combiner test - done",
           "Overall: " + verdict + extra + "\n\n"
           "CSV + PNG written to:\n" + OUTDIR)

    print(summary.to_string(index=False))
    print("Saved to", OUTDIR)


if __name__ == "__main__":
    main()
