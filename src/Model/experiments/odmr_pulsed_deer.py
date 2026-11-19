#!/usr/bin/env python3
# Written by <Gurudev Dutt> and <Jannet Trabelsi>
"""
ODMR Pulsed Experiment

This experiment implements pulsed ODMR measurements using the Proteus for
sequence generation and ADwin for photon counting. Users can specify
sequences using a text-based language and preview scan results.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import json
import os
import logging
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import datetime
from src.Controller import SG384Generator, MCLNanoDrive
from src.core.experiment import Experiment
from src.core import Parameter
from src.Model.sequence_parser import SequenceTextParser
from src.Model.sequence_builder import SequenceBuilder
from src.Model.proteus_hardware_calibrator import ProteusHardwareCalibrator
from src.Model.sequence import Sequence
from src.core.struct_hdf5 import MyStruct
from src.Controller.Proteus_device import ProteusDevice
from src.Model.deer_target import TargetSpinDrive, split_description, build_target_waveforms
from src.Controller.adwin_gold import AdwinGoldDevice
from src.Controller.mux_control import MUXControlDevice
from src.Controller.Thorlabs_FW102C_Filter_Wheel import Thorlabs_FW102C
import numpy as np
from src.core.adwin_helpers import get_adwin_binary_path

ADWIN_TICKS_PER_NS = 0.3  # 300 MHz assumption; 1 tick = 3.333 ns


def readout_length_ns(count_time_ns, reset_time_ns):
    """Your readout convention: X = Y + reset + Y (signal window, reset, reference window)."""
    return 2 * count_time_ns + reset_time_ns


class ODMRPulsedExperiment(Experiment):
    """
    Pulsed ODMR experiment using Proteus for sequence generation and ADwin for counting.

    Features:
    - Text-based sequence definition using the sequence language
    - Sequence preview with first 10 scan points
    - Microwave frequency and power
    - Laser power and wavelength configuration
    - Proteus triggers ADwin for photon counting
    """
    _DEFAULT_SETTINGS = [
        Parameter('sequence', [
            Parameter('file_path', "D:\Data\jannet_trabelsi\March2026\odmr_sequence.txt", str,
                      'Path to sequence definition file'),
            Parameter('load_from_file', False, bool, 'load the sequence from a file'),
            Parameter('text',
                      "sequence: name=odmr_pulsed, type=odmr, duration=1002500ns, sample_rate=1GHz, repeat_count=50000\nvariable pulse_duration, start=50ns, stop=500ns, steps=20\nmarker, laser_int_1 on channel 1 at 0ns, 500ns\npi/2 pulse on channel 1 at 500ns, gaussian, pulse_duration, 1.0\npi/2 pulse on channel 2 at 500ns, gaussian, pulse_duration, 1.0 \nwait pulse on channel 1 at pulse_duration+0.000000500, square, 2*pulse_duration, 0.0\nwait pulse on channel 2 at pulse_duration+0.000000500, square, 2*pulse_duration, 0.0\npi/2 pulse on channel 1 at 3*pulse_duration+0.000000500, gaussian, pulse_duration, 1.0\npi/2 pulse on channel 2 at 3*pulse_duration+0.000000500, gaussian, pulse_duration, 1.0\nmarker, laser_readout_1 on channel 1 at 2500ns, 1ms\nmarker, readout_counts_1 on channel 2 at 2500ns, 300ns\nmarker, reference_counts_1 on channel 2 at 1002200ns, 300ns",
                      str, 'sequence_text')
        ]),
        Parameter('microwave', [
            Parameter('enable', True, bool, 'enable MICROWAVE'),
            Parameter('frequency range', [2.3975e9], list, 'Microwave frequency in Hz', units='Hz'),
            Parameter('power', 16, float, 'Microwave power in dBm', units='dBm')
        ]),
        Parameter('Laser Control', 0.8, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                  "Laser Control"),
        Parameter('green_laser', [
            Parameter('power', 1.0, float, 'Green Laser power in mW', units='mW'),
            Parameter('wavelength', 532.0, float, 'Green Laser wavelength in nm', units='nm')
        ]),
        Parameter('Filter Wheel OD', 0,
                  [0, 0.5, 1, 2, 3, 4], 'Filter Wheel OD'),
        Parameter('delays', [
            Parameter('green_laser_delay', 0.0, float, 'green_laser_delay in ns', units='ns'),
            Parameter('mw_delay', 0.0, float, 'Microwave delay in ns', units='ns'),
            Parameter('iq_delay', 0.0, float, 'iq delay in ns', units='ns'),
            Parameter('counter_delay', 0.0, float, 'AOM delay in ns', units='ns'),
            Parameter('trigger_delay', 0.0, float, 'Counter delay in ns', units='ns')
        ]),
        Parameter('adwin', [
            Parameter('count_time', 300, float, 'Photon counting time in ns', units='ns'),
            Parameter('reset_time', 2000, float, 'Reset time between counts in ns', units='ns'),
            Parameter('readout_start_ns', 6224, float, 'readout_start time in ns (used only if auto_readout_start is off)', units='ns'),
            Parameter('auto_readout_start', False, bool,
                      'Set readout_start_ns = time of the sequence readout pulse + readout_offset_ns, for every file'),
            Parameter('readout_offset_ns', 56.0, float,
                      'One offset for all sequences: readout_start_ns minus the readout pulse time', units='ns'),
            Parameter('trigger_duration', 150, float, 'trigger_duration in ns', units='ns'),
            Parameter('gate_align_offset_ns', 2000, float, 'gate_align_offset in ns', units='ns'),
            Parameter('adbasic_file', "odmr_pulsed_statemachine_v6.TB1", str, 'state mchine code')]),
        Parameter('rf_interlock', [
            Parameter('enable', False, bool, 'Stop firing triggers when reflected/forward is too high (needs the v7 ADwin binary)'),
            Parameter('ain_forward', 1, int, 'ADwin analog input with the RC-filtered FORWARD detector (1-16)'),
            Parameter('ain_reflected', 2, int, 'ADwin analog input with the RC-filtered REFLECTED detector (1-16)'),
            Parameter('max_reflected_fraction', 0.10, float, 'Trip when reflected/forward POWER exceeds this'),
            Parameter('detector_law', 'square', ['square', 'linear'],
                      'square: detector output ~ RF power; linear: output ~ RF voltage'),
            Parameter('detector_match', 1.0, float,
                      'Reflected/forward detector output for EQUAL power at the two coupled ports (1.0 = identical chains)'),
            Parameter('rc_time_constant_ms', 1.0, float, 'RC time constant of the detector filters', units='ms'),
            Parameter('min_forward_V', 0.02, float, 'Shots with less forward signal (above zero) are not judged', units='V'),
            Parameter('consecutive_shots', 3, int, 'Bad shots in a row needed to trip'),
            Parameter('blind_shots', 200, int, 'Trip if no shot reaches min_forward_V within this many shots (0 = off)'),
        ]),
        Parameter('iq_calibration', [
            Parameter('calibration_file', '', str,
                      'Path to iq_calibration_results.json (empty = use default location)'),
            Parameter('apply_q_correction', False, bool,
                      'Scale Q channel amplitude by calibrated factor'),
            Parameter('q_amplitude_correction', 0.89, float,
                      'Q channel scale factor (overridden by calibration file if found)'),
            Parameter('apply_dc_offsets', True, bool,
                      'Add the calibrated digital DC null (i_dc_digital/q_dc_digital) to every ch1/ch2 sample '
                      'and their idle level; ch1/ch2 analog offset then 0 V instead of 12 mV'),
        ]),
        Parameter('DEER', [
            Parameter('enable', False, bool, 'Enable DEER (opens the target module in slot Proteus_SID)'),
            Parameter('Proteus_SID', 5, int, 'Target Spin Drive Device SID'),
            Parameter('mode', 'Single frequency + Internal Gating', ['Single frequency + External Gating', 'Single frequency + Internal Gating', 'Chirp'],
                      'Choose the mode to drive the target spin'),
            Parameter('frequency range', [0.05E9], list, 'External gating: CW frequency in Hz (first entry is used)', units='Hz'),
            Parameter('amplitude', 1.2, float, 'Target output amplitude in Vpp (max 1.2)'),
            Parameter('phase', 0.0, float, 'Target Spin Drive phase'),
            Parameter('awg_offset', 0.0, float, 'Target Spin Drive offset'),
            Parameter('target_channel_tag', 9, int, 'Internal gating / chirp: sequence-text channel that means target module channel 1'),
            Parameter('target_trigger', 'TRG1', ['TRG1', 'TRG2'], 'Target trigger input (wire a main-module sync marker at t = 0 to it)'),
            Parameter('target_trigger_level', 0.4, float, 'Target trigger threshold in V (Proteus markers top out near 1.1 V)', units='V'),
            Parameter('target_skew_ns', 396.0, float, 'Measured target-minus-main output delay; target waveforms play this much earlier', units='ns'),
            Parameter('sinc_compensation', False, bool, 'Flatten the DAC sin(x)/x roll-off across frequency (lowers overall amplitude)'),
        ]),
        Parameter('scan', [
            Parameter('preview_points', 1, int, 'Number of scan points to preview'),
            Parameter('auto_generate_files', True, bool, 'Automatically generate AWG files'),
            Parameter('output_directory', 'odmr_pulsed_output', str, 'Output directory for AWG files'),
            Parameter('bypass_sequence_validation', False, bool,
                      'If True, skip the sequence preview/confirmation and run immediately')
        ]),
        Parameter('tau_advisor', [
            Parameter('enable', False, bool,
                      'ask tau advisor to check pulse sequence for tau length and resolution'),
            Parameter('preset', "single_nv", ["single_nv", "ensemble"], 'provide info about sample for better guidance'),
            Parameter('b_field_G', 50, int, 'b_field_G'),
            Parameter('save_path', "tau_preview.png", str,
                      'save path'),
            Parameter('rabi_MHz', 50, int, 'rabi frequency in MHz'),
            Parameter('detuning_MHz', 50, int, 'detuning in MHz')
        ]),
        Parameter('averaging', [
            Parameter('averages', 40, int,
                      'Number of full runs to average term-by-term (1 = original behavior)'),
            Parameter('optimize_between_runs', False, bool,
                      'Re-optimize confocal position between runs (never in the middle of a run)'),
            Parameter('optimization_mode', "convergence", ["convergence", "fit"],
                      'Re-optimization_mode for confocal position'),
            Parameter('opt_max_steps', 20, int,
                      'convergence mode only: max steps to climb per axis (safety cap so a '
                      'noisy axis cannot walk off forever)'),
            Parameter('opt_improve_frac', 0.02, float,
                      'convergence mode only: a step must raise counts by more than this '
                      'fraction (0.02 = 2%) to be accepted, so it climbs real signal '
                      'instead of chasing shot noise'),
            Parameter('opt_xstep', 0.1, float, 'Optimization x step', units='um'),
            Parameter('opt_ystep', 0.1, float, 'Optimization y step', units='um'),
            Parameter('opt_zstep', 0.1, float, 'Optimization z step', units='um'),
            Parameter('opt_settle_time', 2.0, float, 'Settle time after each stage move', units='s'),
            Parameter('opt_count_time', 2.0, float, 'Counting time per point during optimization', units='ms'),
            Parameter('opt_num_cycles', 10, int, 'Par_10 cycles for the counter during optimization'),
            Parameter('opt_damping_ratio', 0.9, float,
                      'Move to x + damping*(best - x) instead of the raw parabola vertex'),
            # --- retention safety net (ported from the confocal-point optimizer) ---
            Parameter('min_retention_ratio', 0.5, float,
                      'If post-optimization counts < min_retention_ratio * best-known counts, '
                      'snap back toward the best-known position'),
            Parameter('acceptable_counts', 2500000.0, float,
                      'Do not update the best-known baseline to counts above '
                      'acceptable_counts * acceptable_counts_ratio (rejects noise spikes). '
                      'Set a little above your real max cts/s during optimization.'),
            Parameter('acceptable_counts_ratio', 1.1, float,
                      'See acceptable_counts'),
            Parameter('min_reoptimize_ratio', 0.5, float,
                      'Only used when reoptimize_only_if_needed is True: skip the x/y/z pass '
                      'when current counts >= min_reoptimize_ratio * best-known counts'),
            Parameter('reoptimize_only_if_needed', True, bool,
                      'If True, skip optimization before a run when counts are still healthy '
                      '(fewer stage moves = less drift). If False, optimize before every run 2..N '
                      'and rely on the retention net to prevent walking off.')
        ]),
        Parameter('path', "D:\Data"),
        Parameter('filename', "odmr_pulsed_output"),
        Parameter('tag', "odmrpulsedexperiment"),
        Parameter('save', False),
        Parameter('sample', ""),
        Parameter('Magnet ON', False),
    ]

    _DEVICES = {
        'proteus': 'proteus',
        'adwin': 'adwin',
        'sg384': 'sg384',
        'mux_control': 'mux_control',
        'filter_wheel': 'filter_wheel',
        'nanodrive': 'nanodrive'  # used only for between-run optimization
    }

    _EXPERIMENTS = {}

    def __init__(self, devices=None, experiments=None, name=None, settings=None, log_function=None, data_path=None,
                 config_path: Optional[Path] = None, mode=None):
        """Initialize the ODMR Pulsed experiment."""
        super().__init__(name=name, settings=settings, devices=devices, sub_experiments=experiments,
                         log_function=log_function, data_path=data_path)
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.test_mode = mode  # <-- None/"" = real, "testing" = MultiHarp
        self.timing_result = None  #
        self.repeat_count = None
        self.number_of_iterations = 0
        self.s_t = ''
        self.e_t = ''
        self.adwin_reached_the_end = 0
        self.number_of_sequences_done_by_adwin = 0
        self.config_path = config_path or self.get_config_path(
            "D:\\Duttlab\\Experiments\\AQuISS_default_save_location\\experiments_auto_generated\\ODMRPulsedExperiment.json")
        # Configuration
        self.config = self._load_config()
        self.sequence_text = None

        # Sequence components
        self.sequence_parser = SequenceTextParser()
        self.sequence_builder = SequenceBuilder()

        # Initialize hardware calibrator with experiment-specific connection file
        connection_file = Path(__file__).parent / "odmr_pulsed_connection.json"
        self.hardware_calibrator = ProteusHardwareCalibrator(connection_file=str(connection_file))
        """self.adwin = self.devices['adwin']['instance']
        self.proteus = self.devices['proteus']['instance']
        self.sg384 = self.devices['sg384']['instance']
        self.mux = self.devices['mux_control']['instance']
        self.filter_wheel = self.devices['filter_wheel']['instance']"""
        #self.filter_wheel = Thorlabs_FW102C()
        self.proteus = ProteusDevice()
        self.adwin = AdwinGoldDevice()
        self.mux = MUXControlDevice()
        self.sg384 = SG384Generator()
        self.nd = MCLNanoDrive()
        self.mux.select_trigger('pulsed')
        self.target_spin_drive = None
        self.target = None              # TargetSpinDrive on top of target_spin_drive.driver
        self.target_description = None  # sequence pulses on DEER/target_channel_tag
        # Sequence data
        self.sequence_description = None
        self.scan_sequences = []
        self.current_scan_point = 0

        # ADwin parameters
        self.count_time = self.settings["adwin"]["count_time"]  # ns
        self.reset_time = self.settings["adwin"]["reset_time"]  # ns
        self.sequence_duration = 1700  # just a placeholder

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                self.logger.info(f"Configuration loaded from {self.config_path}")
                return config
            else:
                self.logger.warning(f"Configuration file not found: {self.config_path}")
                return {}
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            return {}

    def _iq_cal_entry(self):
        """Calibration-file entry closest to the MW frequency, or None."""
        cal_path = self.settings['iq_calibration']['calibration_file'].strip() or \
            str(Path(__file__).parent / 'iq_calibration_results.json')
        try:
            with open(cal_path, 'r') as f:
                entries = json.load(f).get('frequencies', [])
        except Exception as e:
            print(f"  IQ cal file not read ({e})")
            return None
        if not entries:
            return None
        mw_freq = self.settings['microwave']['frequency range'][0]
        return min(entries, key=lambda e: abs(e['frequency_hz'] - mw_freq))

    def _iq_dc_digital(self) -> Dict[int, float]:
        """{1: dI, 2: dQ} in fractions of full scale when apply_dc_offsets is on and the nearest
        calibration entry has a digital null; {} otherwise (upload unchanged)."""
        if not self.settings['iq_calibration']['apply_dc_offsets']:
            return {}
        entry = self._iq_cal_entry()
        if entry is None or 'i_dc_digital' not in entry:
            print("  apply_dc_offsets is on, but the calibration file has no digital null: nothing applied")
            return {}
        d = {1: float(entry['i_dc_digital']), 2: float(entry['q_dc_digital'])}
        print(f"  IQ digital DC null: I {d[1]:+.5f}, Q {d[2]:+.5f} of full scale "
              f"(entry at {entry['frequency_hz'] / 1e9:.4f} GHz, leakage {entry.get('leakage_dbc', '?')} dBc)")
        return d

    def _load_iq_calibration(self) -> float:
        """
        Load Q amplitude correction from calibration file.
        Returns the correction factor b to apply to channel 2.
        Falls back to settings value if file not found.
        """
        default_b = self.settings['iq_calibration']['q_amplitude_correction']

        cal_path = self.settings['iq_calibration']['calibration_file'].strip()
        if not cal_path:
            # Look for the file next to the calibration experiment
            cal_path = str(
                Path(__file__).parent / 'iq_calibration_results.json'
            )

        try:
            with open(cal_path, 'r') as f:
                cal = json.load(f)

            # Find the entry closest to current MW frequency
            mw_freq = self.settings['microwave']['frequency range'][0]
            entries = cal.get('frequencies', [])

            if entries:
                closest = min(entries, key=lambda e: abs(e['frequency_hz'] - mw_freq))
                b = closest['q_amplitude_b']
                print(f"  IQ cal loaded: b={b:.4f} "
                      f"(from {closest['frequency_hz'] / 1e9:.4f} GHz entry, "
                      f"IRR={closest['best_irr_db']:.1f} dB)")
            else:
                b = cal.get('q_amplitude_correction_default', default_b)
                print(f"  IQ cal loaded: b={b:.4f} (default from file)")

            return b

        except FileNotFoundError:
            print(f"  IQ calibration file not found at {cal_path}, using b={default_b:.4f}")
            return default_b
        except Exception as e:
            print(f"  IQ calibration load error: {e}, using b={default_b:.4f}")
            return default_b

    def load_sequence_from_text(self) -> bool:
        """
        Load sequence definition from text using the sequence language.

        Args:
            sequence_text: Sequence definition in the sequence language format

        Returns:
            True if sequence loaded successfully
        """
        try:
            # Parse sequence text using the sequence language parser
            self.sequence_description = self.sequence_parser.parse_text(self.sequence_text)
            self.repeat_count = self.sequence_description.repeat_count
            print("self.sequence_description.total_duration")
            print(self.sequence_description.total_duration)
            if self.sequence_description:
                self.logger.info(f"Sequence loaded: {self.sequence_description.name}")
                self.logger.info(f"Variables: {len(self.sequence_description.variables)}")
                self.logger.info(f"Pulses: {len(self.sequence_description.pulses)}")
                self.logger.info(f"Markers: {len(self.sequence_description.markers)}")
                return True
            else:
                self.logger.error("Failed to parse sequence text")
                print("Failed to parse sequence text")
                return False

        except Exception as e:
            self.logger.error(f"Error loading sequence: {e}")
            print(f"Error loading sequence: {e}")
            return False

    def load_sequence_from_file(self, sequence_file: Path) -> bool:
        """
        Load sequence definition from text file.

        Args:
            sequence_file: Path to sequence text file

        Returns:
            True if sequence loaded successfully
        """
        try:
            if not sequence_file.exists():
                self.logger.error(f"Sequence file not found: {sequence_file}")
                print("Sequence file not found")
                return False

            # Read sequence text
            with open(sequence_file, 'r') as f:
                self.sequence_text = f.read()
            # Parse sequence
            self.sequence_description = self.sequence_parser.parse_text(self.sequence_text)
            self.repeat_count = self.sequence_description.repeat_count

            if self.sequence_description:
                self.logger.info(f"Sequence loaded: {self.sequence_description.name}")
                self.logger.info(f"Variables: {len(self.sequence_description.variables)}")
                self.logger.info(f"Pulses: {len(self.sequence_description.pulses)}")
                self.logger.info(f"Markers: {len(self.sequence_description.markers)}")
                return True
            else:
                self.logger.error("Failed to parse sequence text")
                return False

        except Exception as e:
            self.logger.error(f"Error loading sequence: {e}")
            return False

    def build_scan_sequences(self) -> bool:
        """
        Build scan sequences from the loaded sequence description.

        Returns:
            True if sequences built successfully
        """
        try:
            if not self.sequence_description:
                self.logger.error("No sequence description loaded")
                return False
            # Build scan sequences
            # DEER: pulses on the target channel tag go to the target module, not the main one
            main_description = self.sequence_description
            self.target_description = None
            if self.settings['DEER']['enable']:
                main_description, self.target_description = split_description(
                    self.sequence_description, self.settings['DEER']['target_channel_tag'])
            self.scan_sequences = self.sequence_builder.build_scan_sequences(
                main_description
            )
            self.sampling_rate = self.sequence_builder.sample_rate
            if self.sequence_description.variables:
                self.number_of_iterations = len(self.scan_sequences)
            else:
                self.number_of_iterations = 1
            # Apply hardware calibration
            for i, sequence in enumerate(self.scan_sequences):
                calibrated_sequence = self.hardware_calibrator.calibrate_sequence(
                    sequence,
                    self.sequence_description.sample_rate
                )
                self.scan_sequences[i] = calibrated_sequence
            self.sequence_duration = calibrated_sequence.length
            self.logger.info(f"Built {len(self.scan_sequences)} scan sequences")
            return True

        except Exception as e:
            self.logger.error(f"Error building scan sequences: {e}")
            return False

    def setup_deer(self) -> bool:
        """Program the target (Cu) module for the selected DEER mode, once per run: after the main
        module is loaded and before the ADwin starts triggering.

        Single frequency + External Gating: CW tone at the first DEER/frequency range entry (exact
            frequency). Gate it with an RF switch driven by a main-module marker in the sequence text.
        Single frequency + Internal Gating / Chirp: the pulses on channel DEER/target_channel_tag in
            the sequence text are built at the target sample rate and played in lockstep with the main
            module, one segment per scan point, each waiting for DEER/target_trigger (wire a
            main-module sync marker at t = 0 to that input)."""
        cfg = self.settings['DEER']
        try:
            if self.target_spin_drive is None:  # open the slot once; re-opening it resets the module
                self.target_spin_drive = ProteusDevice(
                    name="target_driver",
                    settings={"ip_address": "192.168.2.4", "slot_id": cfg['Proteus_SID']})
            if self.target is None:
                self.target = TargetSpinDrive(self.target_spin_drive.driver, channel=1, logger=self.logger)
            self.target_spin_drive.driver.set_ref_clock_frequency("10M")
            self.target_spin_drive.driver.set_ref_clock_external()
            time.sleep(0.5)
            if cfg['mode'] == 'Single frequency + External Gating':
                freqs = list(cfg['frequency range'])
                if len(freqs) > 1:
                    self.logger.warning("[DEER] external gating uses only the first DEER frequency")
                f_act, loop_len = self.target.cw(float(freqs[0]), vpp=cfg['amplitude'])
                print(f"[DEER] CW {float(freqs[0]) / 1e6:.4f} MHz requested, {f_act / 1e6:.6f} MHz played "
                      f"({loop_len}-sample loop at {self.target.fs / 1e9:g} GS/s)")
                return True

            if self.target_description is None or not self.target_description.pulses:
                self.logger.error(f"[DEER] no pulses on channel {cfg['target_channel_tag']} in the sequence text")
                return False
            waves, info = build_target_waveforms(
                self.target_description, self.target.fs, cfg['target_channel_tag'],
                skew_ns=cfg['target_skew_ns'], align=self.target.align,
                sinc_compensate=cfg['sinc_compensation'])
            if len(waves) != self.number_of_iterations:
                self.logger.error(f"[DEER] {len(waves)} target segments vs {self.number_of_iterations} main scan points")
                return False
            self.target.load_lockstep(waves, vpp=cfg['amplitude'], trigger=cfg['target_trigger'],
                                      level=cfg['target_trigger_level'])
            print(f"[DEER] target armed: {info['n_points']} segments x {info['duration_us']:.2f} us at "
                  f"{self.target.fs / 1e9:g} GS/s, skew {cfg['target_skew_ns']} ns, sinc gain {info['sinc_gain']:.3f}")
            return True
        except Exception:
            self.logger.exception("[DEER] setup failed")
            return False

    def _check_target_lockstep(self) -> bool:
        """Internal gating / chirp: after a run the main and target task pointers must agree. A missed
        or extra trigger on the target shifts every later shot to the wrong scan point."""
        if not (self.settings['DEER']['enable'] and self.target is not None and self.target.n_tasks):
            return True
        try:
            self.proteus.driver.set_channel(1)
            main_task = int(float(str(self.proteus.driver.get_current_task()).strip()))
        except Exception:
            return True
        target_task = self.target.current_task()
        if target_task is not None and target_task != main_task:
            msg = f"[DEER] target task {target_task} != main task {main_task}: modules out of lockstep"
            self.logger.warning(msg)
            print("WARNING " + msg)
            return False
        return True

    def generate_awg_task_sequences_adwin_triggering_awg_case(self) -> bool:
        """
        Generate Proteus AWG waveforms with maximum timing resolution.
        One continuous waveform per channel.
        one segment and one task entry per iteration per channel.
        64-sample alignment is applied at the end of each waveform.
        The advantage of this option is that adwin does the counts then it tells proteus
        to move to the next sequence. This ensures that counts are measured as expected.
        """

        def prepare_markers_for_tabor(markers_array: np.ndarray) -> np.ndarray:
            """
            Correct Proteus marker packing for 16-bit DAC mode.
            markers_array: 1 marker value per waveform sample (0 or 255)
            """

            # Convert to binary 0/1
            m = (markers_array > 0).astype(np.uint8)

            # Ensure multiple of 4 samples
            # pad:
            if len(m) % 4 != 0:
                m = np.pad(m, (0, 4 - len(m) % 4))  # pad at the end
                # m = np.pad(m, (4 - len(m) % 4, 0))# pad at the beginning

            # One marker byte per 4 waveform samples
            marker_bytes = np.zeros(len(m) // 4, dtype=np.uint8)

            for i in range(len(marker_bytes)):
                block = m[i * 4:(i + 1) * 4]

                if np.any(block):
                    # Marker 1 ON for all 4 samples → 0b00010001
                    marker_bytes[i] = 0x11
                else:
                    marker_bytes[i] = 0x00

            return marker_bytes

        try:
            self.proteus.driver.set_ref_clock_frequency("10M")
            self.proteus.driver.set_ref_clock_external()
            time.sleep(0.5)
            if not self.scan_sequences:
                self.logger.error("No scan sequences available")
                return False
            # ------------------------------------------------------------
            # DAC configuration
            # ------------------------------------------------------------
            max_dac = 65535
            half_dac = max_dac // 2
            ALIGNMENT = 64  # Proteus requirement (segment length)

            # ------------------------------------------------------------
            # Determine all channels, initialize buffers
            # ------------------------------------------------------------
            all_channels = set()
            channels = set()
            marker_indices = set()
            marker_channels = set()
            # Load IQ calibration
            if self.settings['iq_calibration']['apply_q_correction']:
                q_correction_b = self._load_iq_calibration()
            else:
                q_correction_b = 1.0
            print(f"Q amplitude correction b = {q_correction_b:.4f}")
            dc_digital = self._iq_dc_digital()    # {} unless apply_dc_offsets is on
            for sequence in self.scan_sequences:
                for _, pulse_ in enumerate(sequence.pulses):
                    pulse = pulse_[1]
                    ch = int(pulse.name.split("_")[-1])
                    channels.add(ch)
                    all_channels.add(ch)

                for marker in sequence.markers:
                    mkr_index = int(marker.name.split('_')[-2])
                    marker_indices.add(mkr_index)
                    mk_ch = int(marker.name.split("_")[-1])
                    marker_channels.add(mk_ch)
                    all_channels.add(mk_ch)
            #segment_for_channel = {ch: set() for ch in all_channels}
            segment_for_channel = {ch: [] for ch in all_channels}
            padded_zeros = {
                ch: 0 for ch in all_channels
            }
            max_delay = {
                it: 0 for it in range(self.number_of_iterations)
            }

            # ------------------------------------------------------------
            # Build one segment PER CHANNEL PER ITERATION.
            # Every channel gets a segment at every scan point; a channel that
            # is idle this iteration gets a flat/zero segment. This keeps every
            # channel's task table the same length (= number_of_iterations) so
            # they stay in lockstep under the shared TRG1.
            # ------------------------------------------------------------
            SEGNUM = 1
            for sequence in self.scan_sequences:
                seq_waveforms = sequence.to_waveform()
                seq_len = sequence.length
                for ch in sorted(all_channels):  # <-- ALL channels, fixed order
                    data = seq_waveforms.get(ch)
                    if data is not None:
                        envelope = data["envelope"]
                        markers = data["markers"]
                    else:
                        # idle channel this iteration -> flat 0 V, no markers
                        envelope = np.zeros(seq_len, dtype=float)
                        markers = np.zeros(seq_len, dtype=np.uint8)

                    self.proteus.driver.set_channel(ch)
                    analog_voltage = "MAX"
                    self.proteus.driver.set_voltage(analog_voltage)
                    """if ch == 1:
                        self.proteus.driver.set_voltage(0.1)"""
                    self.proteus.driver.apply_sampling_configuration(self.sampling_rate)

                    # ── Apply Q amplitude correction ──────────────────────
                    if ch == 2:
                        envelope = envelope * q_correction_b
                    # ──────────────────────────────────────────────────────

                    segment_num = SEGNUM
                    segment_for_channel[ch].append(segment_num)  # ordered list, scan order

                    rem = len(envelope) % ALIGNMENT
                    padded_zeros[ch] = rem
                    if rem:
                        envelope = np.pad(envelope, (0, ALIGNMENT - rem))
                        markers = np.pad(markers, (0, ALIGNMENT - rem))
                        padded_zeros[ch] = ALIGNMENT - rem

                    # Scale to DAC (digital DC null added to every ch1/ch2 sample, padding included)
                    dac_wave = np.clip(envelope + dc_digital.get(ch, 0.0), -1.0, 1.0)
                    dac_wave = ((dac_wave + 1.0) * half_dac).astype(np.uint16)
                    self.proteus.driver.define_trace(segment_num, len(dac_wave))
                    self.proteus.driver.select_segment(segment_num)

                    # Upload waveform
                    self.proteus.driver.write_trace_data(dac_wave)
                    resp = self.proteus.driver.query_error()
                    self.proteus.driver.inst.send_scpi_cmd(f":VOLT:OFFS {0.0 if ch in dc_digital else 0.012}")

                    # Convert to uint8 and upload markers (zeros for idle channels)
                    markers = markers.astype(np.uint8)
                    tabor_markers = prepare_markers_for_tabor(markers)
                    self.proteus.driver.write_marker_data(tabor_markers)
                    resp = self.proteus.driver.query_error()
                    marker_index = 1
                    self.proteus.driver.set_marker(marker_index)
                    self.proteus.driver.set_marker_ptop_voltage(1.2)
                    self.proteus.driver.set_marker_voltage_offset(0.5)
                    resp = self.proteus.driver.query_error()
                    SEGNUM += 1
            i = 0
            for sequence in self.scan_sequences:
                seq_waveforms = sequence.to_waveform()
                for ch1, data1 in seq_waveforms.items():
                    for ch2, data2 in seq_waveforms.items():
                        if padded_zeros[ch1] > padded_zeros[ch2] and padded_zeros[ch1] > max_delay[i]:
                            max_delay[i] = padded_zeros[ch1]
                        elif padded_zeros[ch2] > padded_zeros[ch1] and padded_zeros[ch2] > max_delay[i]:
                            max_delay[i] = padded_zeros[ch2]

                i += 1

            for ch in sorted(all_channels):
                self.proteus.driver.set_channel(ch)
                if ch in marker_channels:
                    self.proteus.driver.set_marker(1)
                    self.proteus.driver.set_marker_state('ON')
                if ch in channels:
                    self.proteus.driver.set_output("ON")
                # --self.proteus.driver.set_continuous_run(0)
            self.proteus.driver.set_trigger_level(2)
            self.proteus.driver.activate_instrument(1)
            for ch in sorted(all_channels):
                self.proteus.driver.set_channel(ch)
                self.proteus.driver.set_trigger_source('TRG1')
                self.proteus.driver.set_trigger("TRG1")  # :TRIG:SEL
                self.proteus.driver.set_trigger_state("ON")  # :TRIG:STAT

            for ch in sorted(all_channels):
                task_num = 1
                self.proteus.driver.set_channel(ch)
                n_tasks = len(segment_for_channel[ch])  # tasks for THIS channel
                self.proteus.driver.set_task_table_length(n_tasks)
                for segment_number in segment_for_channel[ch]:
                    self.proteus.driver.set_task_number(task_num)
                    if task_num == 1:
                        self.proteus.driver.set_task_type("STAR")
                    elif task_num == n_tasks:  # last task on THIS channel
                        self.proteus.driver.set_task_type("END")
                    else:
                        self.proteus.driver.set_task_type("SEQ")
                    self.proteus.driver.set_task_segment_number(segment_number)
                    self.proteus.driver.set_trigger_IDLE_state("DC")
                    self.proteus.driver.set_trigger_IDLE_level(
                        int((dc_digital[ch] + 1.0) * half_dac) if ch in dc_digital else 0)
                    self.proteus.driver.set_enabling_task_signal("TRG1")  # every task gated by TRG1

                    if task_num == n_tasks:
                        self.proteus.driver.set_next1_task(1)  # wrap to first task
                    else:
                        self.proteus.driver.set_next1_task(task_num + 1)

                    task_num += 1
                    self.proteus.driver.write_composer_array_to_task_table()
                    self.proteus.driver.set_waveform_type('TASK')
            ###self.proteus.driver.set_task_loop(self.repeat_count)
            self.proteus.driver.set_task_sync()
            resp = self.proteus.driver.set_trigger_coupling('ON')
            # for trigger source INT send:
            self.proteus.driver.set_channel(1)
            # time.sleep(1)
            # self.proteus.driver._close()
            return True
        except Exception as e:
            self.logger.exception("Error generating AWG sequences")
            return False

    def _function(self):
        self.filter_wheel.update({'OD': self.settings['Filter Wheel OD']})
        self.adwin = self.devices['adwin']['instance']
        self.proteus = self.devices['proteus']['instance']
        self.sg384 = self.devices['sg384']['instance']
        self.mux = self.devices['mux_control']['instance']
        # nanodrive is only needed for between-run optimization; fetch defensively
        # so the experiment still runs if it isn't wired up.
        self.nd = self.devices.get('nanodrive', {}).get('instance', None)

        # new work to be tested:
        def build_calibration_overrides():
            return {
                "green_laser_delay": self.settings['delays']['green_laser_delay'],
                "mw_delay": self.settings['delays']['mw_delay'],
                "iq_delay": self.settings['delays']['iq_delay'],
                "counter_delay": self.settings['delays']['counter_delay'],
                "trigger_delay": self.settings['delays']['trigger_delay'],
                "units": "ns"
            }

        # this part is for when we adjust the settings in the gui, the calibration changes accordingly: so no need to change the json
        overrides = build_calibration_overrides()
        self.hardware_calibrator.update_calibration_delays(overrides)
        # Experiment parameters: the next section updates the settings every start click
        self.microwave_power = self.settings['microwave']['power']
        self.mw_delay = self.settings['delays']['mw_delay']
        self.green_laser_delay = self.settings['delays']['green_laser_delay']
        self.counter_delay = self.settings['delays']['counter_delay']
        self.iq_delay = self.settings['delays']['iq_delay']
        self.trigger_delay = self.settings['delays']['trigger_delay']

        self.green_laser_power = self.settings['green_laser']['power']
        self.green_laser_wavelength = self.settings['green_laser']['wavelength']

        # Output paths
        self.output_dir = self.get_output_dir("odmr_pulsed_output")
        self.logger.info("ODMR Pulsed Experiment initialized")

        # Create and load sequence
        source = None
        sequence_loaded = False

        if self.settings['sequence']['load_from_file']:
            file_path = self.settings['sequence']['file_path']
            file_path = Path(file_path)

            if file_path and os.path.exists(file_path):
                sequence_loaded = self.load_sequence_from_file(file_path)
                print(f"file path: {file_path}")
                source = "file"
            else:
                self.logger.info("File path is invalid or does not exist")
        else:
            self.sequence_text = self.settings['sequence']['text']
            print("ODMR Sequence:")
            print(self.sequence_text)
            print("\n" + "=" * 50 + "\n")
            sequence_loaded = self.load_sequence_from_text()
            source = "text"

        if sequence_loaded:
            self.logger.info(f"Sequence loaded successfully from {source}")
        else:
            self.logger.info(f"Failed to load sequence from {source}")
            return

        # --- Sequence preview + proceed gate --------------------------------
        # build_scan_sequences() (inside _preview_and_confirm) runs in SOFTWARE
        # only -- no instrument I/O. If bypass_sequence_validation is False we
        # show the preview and wait: hardware is programmed only if the user
        # clicks "Yes, proceed". On "No"/close we return, leaving every
        # instrument untouched.
        if not self.settings['scan']['bypass_sequence_validation']:
            if not self._preview_and_confirm():
                self.logger.info("Cancelled at sequence preview; no hardware programmed.")
                return
        # --------------------------------------------------------------------
        # Run experiment
        self.logger.info("Running experiment...")
        results = self.run_experiment_averaged(self.settings['microwave']['frequency range'])
        self.data = results
        try:
            self.save_hdf5()
            if results['success']:
                self.logger.info("Experiment completed successfully!")
                self.logger.info(f"Frequencies scanned: {[f / 1e9 for f in results['frequencies']]} GHz")
                self.logger.info(
                    f"Signal counts shape: {len(results['signal_counts'])} frequencies x {len(results['signal_counts'][0])} points")
            else:
                self.logger.info(f"Experiment failed: {results['error']}")
        finally:
            self.cleanup()
        self.show_sequence_preview(10)
        self.logger.info("\nODMR Pulsed Experiment ready!")

    def save_hdf5(self):
        """this function defines its custom data and metadata to be saved and then calls the
        save_hdf_data function that is in the parent Experiment class, which adds the external
        devices in case you ever check the Get Basic Data checkbox in the GUI"""
        structure_to_save = MyStruct()
        # Save every per-run array (signal_1, reference_1, total_1, ...), the
        # term-by-term averages (signal_avg, reference_avg, total_avg), and the
        # legacy signal_counts/reference_counts/total_counts (= the averages).
        data_kwargs = {}
        for key, val in self.data.items():
            if key.startswith(('signal_', 'reference_', 'total_')) or key in (
                    'signal_counts', 'reference_counts', 'total_counts'):
                data_kwargs[key] = val
        structure_to_save.data = MyStruct(**data_kwargs)
        structure_to_save.meta = MyStruct(
            gate_align_offset_ns=self.gate_align_offset_ns,
            count_time=self.count_time,
            counter_delay=self.counter_delay,
            end_time=self.e_t,
            frequency_range=self.settings['microwave']['frequency range'],
            green_laser_delay=self.green_laser_delay,
            green_laser_power=self.green_laser_power,
            green_laser_wavelength=self.green_laser_wavelength,
            iq_delay=self.iq_delay,
            microwave_power=self.microwave_power,
            mw_delay=self.mw_delay,
            number_of_iterations=self.number_of_iterations,
            averages=self.data.get('averages', 1),
            repeat_count=self.repeat_count,
            reset_time=self.reset_time,
            sampling_rate=self.sampling_rate,
            sequence_text=self.sequence_text,
            sequence_duration=self.sequence_duration,
            start_time=self.s_t,
            trigger_delay=self.trigger_delay,
            adwin_file=self.adwin_file,
            start_counting_ns = getattr(self, "readout_start_ns", self.settings["adwin"]["readout_start_ns"]),
            adwin_reached_the_end=self.adwin_reached_the_end,
            number_of_sequence_done_by_adwin=self.number_of_sequences_done_by_adwin
        )
        structure_to_save.devices = self.devices
        self.save_hdf_data(structure_to_save)

    def generate_awg_sequences_awg_triggering_adwin_case(self) -> bool:
        """
        Generate Proteus AWG waveforms with maximum timing resolution.
        One continuous waveform per channel.
        one segment and one task entry per channel.
        64-sample alignment is applied at the end of each waveform.
        The disadvantage of this option is that proteus does not know if the adwin is ready to count and whether it finished counting
        """

        def prepare_markers_for_tabor(markers_array: np.ndarray) -> np.ndarray:
            """
            Correct Proteus marker packing for 16-bit DAC mode.
            markers_array: 1 marker value per waveform sample (0 or 255)
            """

            # Convert to binary 0/1
            m = (markers_array > 0).astype(np.uint8)

            # Ensure multiple of 4 samples
            if len(m) % 4 != 0:
                m = np.pad(m, (0, 4 - len(m) % 4))

            # One marker byte per 4 waveform samples
            marker_bytes = np.zeros(len(m) // 4, dtype=np.uint8)

            for i in range(len(marker_bytes)):
                block = m[i * 4:(i + 1) * 4]

                if np.any(block):
                    # Marker 1 ON for all 4 samples → 0b00010001
                    marker_bytes[i] = 0x11
                else:
                    marker_bytes[i] = 0x00

            return marker_bytes

        try:
            self.proteus.driver.set_channel(3)
            self.proteus.driver.set_voltage('MAX')
            if not self.scan_sequences:
                self.logger.error("No scan sequences available")
                return False
            # ------------------------------------------------------------
            # DAC configuration
            # ------------------------------------------------------------
            max_dac = 65535
            half_dac = max_dac // 2
            ALIGNMENT = 64  # Proteus requirement (segment length)

            # ------------------------------------------------------------
            # Determine all channels, initialize buffers
            # ------------------------------------------------------------
            all_channels = set()
            channels = set()
            marker_indices = set()
            marker_channels = set()
            for sequence in self.scan_sequences:
                for _, pulse_ in enumerate(sequence.pulses):
                    pulse = pulse_[1]
                    ch = int(pulse.name.split("_")[-1])
                    channels.add(ch)
                    all_channels.add(ch)

                for marker in sequence.markers:
                    mkr_index = int(marker.name.split('_')[-2])
                    marker_indices.add(mkr_index)
                    mk_ch = int(marker.name.split("_")[-1])
                    marker_channels.add(mk_ch)
                    all_channels.add(mk_ch)

            channel_waveforms = {
                ch: [] for ch in all_channels
            }
            channel_markers = {
                ch: [] for ch in all_channels
            }
            dc_digital = self._iq_dc_digital()    # {} unless apply_dc_offsets is on
            for ch in all_channels:
                self.proteus.driver.set_channel(ch)
                self.proteus.driver.delete_all_segment()
                self.proteus.driver.set_voltage("MAX")
                self.proteus.driver.apply_sampling_configuration(self.sampling_rate)
                if ch in dc_digital:
                    self.proteus.driver.set_voltage_offset(0.0)

            # ------------------------------------------------------------
            # Build continuous waveform per channel (sample-accurate)
            # ------------------------------------------------------------
            for sequence in self.scan_sequences:
                seq_waveforms = sequence.to_waveform()
                for ch, data in seq_waveforms.items():
                    envelope = data["envelope"]
                    markers = data["markers"]
                    if not np.all(markers == 0):
                        print(f"channel {ch}: Found non-zero markers")
                        print(f"unique values {np.unique(markers)}")

                    channel_waveforms[ch].append(envelope)
                    channel_markers[ch].append(markers)
            # ------------------------------------------------------------
            # Upload one segment per channel
            # ------------------------------------------------------------
            segment_index = 1
            segment_for_channel = {}

            for ch in sorted(all_channels):
                self.proteus.driver.set_channel(ch)

                waveform = np.concatenate(channel_waveforms[ch])

                # Apply required alignment
                rem = len(waveform) % ALIGNMENT
                if rem:
                    waveform = np.pad(waveform, (0, ALIGNMENT - rem))

                # Scale to DAC (digital DC null on ch1/ch2 when applied)
                dac_wave = np.clip(waveform + dc_digital.get(ch, 0.0), -1.0, 1.0)
                dac_wave = ((dac_wave + 1.0) * half_dac).astype(np.uint16)

                # Define and select segment
                self.proteus.driver.define_trace(segment_index, len(dac_wave))
                self.proteus.driver.select_segment(segment_index)

                # Upload waveform
                self.proteus.driver.write_trace_data(dac_wave)  # write, and wait while *OPC completes
                resp = self.proteus.driver.query_error()

                segment_for_channel[ch] = segment_index
                # Handle markers for this channel
                if ch in channel_markers:
                    markers = np.concatenate(channel_markers[ch])


                    rem = len(waveform) % ALIGNMENT
                    if rem:
                        waveform_padded_len = len(waveform) + (ALIGNMENT - rem)
                    else:
                        waveform_padded_len = len(waveform)

                    # Now pad/truncate markers to match waveform length
                    if len(markers) < waveform_padded_len:
                        # Pad markers with zeros
                        markers = np.pad(markers, (0, waveform_padded_len - len(markers)), mode='constant')
                    elif len(markers) > waveform_padded_len:
                        # Truncate markers
                        markers = markers[:waveform_padded_len]

                    if not np.all(markers == 0):

                        # Convert to uint8
                        markers = markers.astype(np.uint8)

                        # Now prepare for Tabor
                        tabor_markers = prepare_markers_for_tabor(markers)

                        # For 16-bit mode, marker bytes should be waveform_bytes / 4
                        expected_marker_bytes = len(dac_wave) // 4
                        actual_marker_bytes = len(tabor_markers)

                        if actual_marker_bytes != expected_marker_bytes:
                            if actual_marker_bytes < expected_marker_bytes:
                                tabor_markers = np.pad(tabor_markers, (0, expected_marker_bytes - actual_marker_bytes),
                                                       mode='constant')
                            else:
                                tabor_markers = tabor_markers[:expected_marker_bytes]

                        self.proteus.driver.write_marker_data(tabor_markers)
                        resp = self.proteus.driver.query_error()
                        # proteus P1284M only has 1 marker per channel, for other awgs, please implement code that handels that
                        # I have already coded the marker index here: mkr_index = int(marker.name.split('_')[-2]) so if the user
                        # gives marker, laser_int_1 on channel 4 at 0ns, 500ns then you know whatever is after laser_init_ is the index
                        marker_index = 1

                        self.proteus.driver.set_marker(marker_index)
                        # Verify Proteus marker settings
                        resp = self.proteus.driver.get_marker()
                        resp = self.proteus.driver.get_marker_state()
                        resp = self.proteus.driver.query_error()
                        resp = self.proteus.driver.get_marker_ptop_voltage()
                        resp = self.proteus.driver.get_marker_voltage()
                        self.proteus.driver.set_marker_ptop_voltage(1)
                        resp = self.proteus.driver.get_marker_ptop_voltage()
                        resp = self.proteus.driver.get_marker_voltage_offset()
                        self.proteus.driver.set_marker_voltage_offset(0.5)
                        resp = self.proteus.driver.get_marker_voltage_offset()
                        # Check for errors
                        resp = self.proteus.driver.query_error()
                    else:
                        print(f"Channel {ch}: Markers are all zeros, skipping marker upload")

                segment_index += 1

            # ------------------------------------------------------------
            # Build ONE task entry per channel
            # ------------------------------------------------------------

            # Task must be set per channel
            for ch in sorted(all_channels):
                self.proteus.driver.set_channel(ch)
                self.proteus.driver.set_continuous_run(0)
                task_table_length = 1
                self.proteus.driver.set_task_table_length(task_table_length)
                self.proteus.driver.set_task_number(1)
                self.proteus.driver.set_task_type("SING")
                # self.proteus.driver.set_next1_task(0)
                self.proteus.driver.set_task_segment_number(segment_for_channel[ch])
                self.proteus.driver.set_task_loop(self.repeat_count)
                self.proteus.driver.write_composer_array_to_task_table()

                # ------------------------------------------------------------
                # Enable TASK mode and output
                # ------------------------------------------------------------
            self.proteus.driver.set_task_sync()
            self.proteus.start_sequence()
            time.sleep(1)
            # self.proteus.driver._close()
            return True
        except Exception as e:
            self.logger.exception("Error generating AWG files")
            return False

    def show_sequence_preview(self, num_points: int = 10) -> None:
        """
        Show sequence preview window with first N scan points.

        Args:
            num_points: Number of scan points to preview
        """
        if not self.scan_sequences:
            messagebox.showerror("Error", "No scan sequences available. Build sequences first.")
            return

        # Limit to available sequences
        preview_sequences = self.scan_sequences[:min(num_points, len(self.scan_sequences))]

        # Create preview window
        preview_window = SequencePreviewWindow(preview_sequences, self.sequence_description)
        preview_window.show()

    def suggest_tau(self, **kwargs):
        from tau_advisor import advise_tau
        text = self.sequence_text
        return advise_tau(text, **kwargs)


    def _preview_and_confirm(self) -> bool:
        """Build the sequence in SOFTWARE only, then ask before any hardware.
        Returns True only if the user clicks 'Yes, proceed'."""
        if not self.build_scan_sequences():
            self.logger.error("Could not build scan sequences for preview.")
            return False
        win = SequencePreviewWindow(self.scan_sequences, self.sequence_description)
        return win.show_and_confirm()

    def set_microwave_parameters(self, frequency: float, power: float, delay: float) -> None:
        """
        this is just for testing: we need to only use json or gui for this
        Set microwave parameters.

        Args:
            frequency: Frequency in Hz
            power: Power in dBm
            delay: Delay in ns
        """
        self.microwave_frequency = frequency
        self.microwave_power = power
        self.mw_delay = delay
        self.logger.info(f"Microwave: {frequency / 1e9:.3f} GHz, {power} dBm, {delay} ns delay")

    def set_green_laser_parameters(self, power: float, wavelength: float) -> None:
        """
        this is just for testing: we need to only use json or gui for this
        Set laser parameters.

        Args:
            power: Power in mW
            wavelength: Wavelength in nm
        """
        self.green_laser_power = power
        self.green_laser_wavelength = wavelength
        self.logger.info(f"Laser: {power} mW, {wavelength} nm")

    def set_delay_parameters(self, mw_delay: float, green_laser_delay: float, counter_delay: float) -> None:
        """
        this is just for testing: we need to only use json or gui for this
        Set delay parameters.

        Args:
            mw_delay: Microwave delay in ns
            green_laser_delay: green_laser_delay delay in ns
            counter_delay: Counter delay in ns
        """
        self.mw_delay = mw_delay
        self.green_laser_delay = green_laser_delay
        self.counter_delay = counter_delay
        self.logger.info(f"Delays: MW={mw_delay}ns, AOM={green_laser_delay}ns, Counter={counter_delay}ns")

    def readout_start_from_sequence(self) -> float:
        """readout_start_ns = time of the loaded sequence's 'readout' pulse + adwin/readout_offset_ns,
        so every file gets the same window offset without retyping readout_start_ns."""
        desc = self.sequence_description
        if desc is None:
            raise RuntimeError("auto_readout_start: no sequence loaded")
        ro = [p for p in desc.pulses if p.pulse_type == "readout"]
        if len(ro) != 1:
            raise RuntimeError(f"auto_readout_start: expected one 'readout' pulse in the sequence, found {len(ro)}")
        t = ro[0].timing
        if isinstance(t, str):
            raise RuntimeError("auto_readout_start: the readout pulse time depends on a scan variable; "
                               "turn adwin/auto_readout_start off and set readout_start_ns by hand")
        off = float(self.settings['adwin']['readout_offset_ns'])
        value = float(t) * 1e9 + off
        print(f"[readout] readout pulse at {float(t) * 1e9:.0f} ns + offset {off:.0f} ns "
              f"-> readout_start_ns = {value:.0f} ns")
        return value

    def configure_gates_v5(self):
        """
        Compute the gate tick targets HERE and push them to the state machine.

        Requires on self (all in ns, all REAL values):
            self.readout_start_ns   trigger -> readout-pulse onset (SAME for every tau;
                                    only init/MW start may move within the segment)
            self.count_time         Y, the count window width (e.g. 300)
            self.reset_time         REAL re-polarization time between the two windows
                                    (keep 2*Y + reset <= readout length!)
            self.trigger_duration   trigger pulse width (e.g. 100-200)
            self.repeat_count
            self.number_of_iterations
        Optional:
            self.gate_align_offset_ns  one-time scope calibration for the fixed
                                       trigger->play latency (default 0)
            self.pd_ticks              Processdelay in CYCLES (default: ~3x the shot)

        Returns the readout pulse length X (ns).
        """
        tpns = float(ADWIN_TICKS_PER_NS)
        Y = int(round(self.count_time))
        rst = int(round(self.reset_time))
        align = int(round(getattr(self, "gate_align_offset_ns", 0)))
        trig_ns = int(round(self.trigger_duration))

        sig_start_ns = int(round(self.readout_start_ns)) + align  # trigger -> signal window
        ref_start_ns = sig_start_ns + Y + rst  # trigger -> reference window (last Y)
        X = readout_length_ns(Y, rst)  # readout pulse length

        # ns -> ticks, done on the PC with the one calibrated constant
        count_ticks = max(1, int(round(Y * tpns)))
        sig_start_ticks = max(1, int(round(sig_start_ns * tpns)))
        ref_start_ticks = int(round(ref_start_ns * tpns))
        trig_ticks = max(1, int(round(trig_ns * tpns)))

        # One Event per shot => Processdelay MUST exceed the shot, or the Event
        # overruns every shot, the shots run back-to-back, and the PC link is
        # starved for the whole sweep (first get_int_var must then survive the
        # entire sweep -> Get_Par timeout once the sweep is long enough). We compute
        # it straight from the shot and DO NOT let anything shrink it. experiment.
        # pd_ticks is intentionally ignored now -- a stray small value was the trap.
        shot_ns = ref_start_ns + Y
        pd_ticks = max(1500, int(round(
            2.0 * shot_ns * tpns)))  # ~2x shot; Event uses half the period, link gets the other half
        self.pd_ticks_sent = pd_ticks
        pd_us = pd_ticks / tpns / 1000.0
        shot_us = shot_ns / 1000.0
        if getattr(self, "pd_ticks", None) is not None:
            print(f"[v5] IGNORING experiment.pd_ticks={int(self.pd_ticks)} "
                  f"(a small Processdelay starves the link). Using {pd_ticks} cycles.")

        a = self.adwin
        a.set_int_var(1, int(self.repeat_count))  # Par_1
        a.set_int_var(2, int(self.number_of_iterations))  # Par_2
        a.set_int_var(3, count_ticks)  # Par_3  Y (ticks)
        a.set_int_var(4, ref_start_ticks)  # Par_4  trigger->reference start (ticks)
        a.set_int_var(5, sig_start_ticks)  # Par_5  trigger->signal start (ticks)
        a.set_int_var(6, pd_ticks)  # Par_6  Processdelay (cycles, >= shot)
        a.set_int_var(7, trig_ticks)  # Par_7  trigger width (ticks)

        print(f"[v5] ticks/ns={tpns}: sig_start={sig_start_ns}ns->{sig_start_ticks}t, "
              f"ref_start={ref_start_ns}ns->{ref_start_ticks}t, Y={Y}ns->{count_ticks}t, "
              f"readout X={X}ns.")
        print(f"[v5] shot~{shot_us:.1f} us, Processdelay={pd_ticks} cyc (~{pd_us:.1f} us) "
              f"-> Event uses ~{shot_us / pd_us * 100:.0f}% of the period, link gets the rest. "
              f"CHECK Par_71 == {pd_ticks} after this; if it reads small, something else "
              f"is writing Par_6 after configure.")
        return X

    # ------------------------------------------------------------------ RF reflected-power interlock (v7)
    ILK_SIGNATURE = 7784
    ILK_REASONS = {1: "reflected/forward above the threshold",
                   2: "invalid interlock settings (AIN numbers or threshold)",
                   3: "the forward detector never saw RF (unplugged, or min_forward_V too high)"}

    @staticmethod
    def _ilk_volts(digits) -> float:
        """ADC digits (difference from the zero) -> volts, assuming the 16-bit +/-10 V scale.
        Check once with a known DC voltage on the input."""
        return float(digits) * 20.0 / 65536.0

    def configure_rf_interlock(self) -> None:
        """Send the interlock settings (Par_30..37). Call after configure_gates_v5() and before
        Par_10 = 1. Raises if the interlock is enabled but cannot work, or is still latched."""
        cfg = self.settings['rf_interlock']
        a = self.adwin
        # The interlock Pars (30-37, 40-49) only mean something to the v7 program; with any other
        # program loaded, leave them alone and keep the checks below inert.
        self._ilk_v7 = (a.get_int_var(80) == self.ILK_SIGNATURE)
        if not cfg['enable']:
            if self._ilk_v7:
                a.set_int_var(30, 0)
            print("[RF interlock] DISABLED: nothing stops the triggers on a bad reflection.")
            return
        if not self._ilk_v7:
            raise RuntimeError(f"RF interlock enabled, but the loaded ADwin program is not v7 "
                               f"(Par_80 = {a.get_int_var(80)}, expected {self.ILK_SIGNATURE}).")
        if a.get_int_var(40) != 0:
            raise RuntimeError("RF interlock is still latched from an earlier trip. Fix the RF chain, "
                               "then call reset_rf_interlock() before running again.")
        frac = float(cfg['max_reflected_fraction'])
        if not 0.0 < frac < 1.0:
            raise ValueError(f"max_reflected_fraction must be between 0 and 1 (got {frac})")
        k = (frac if cfg['detector_law'] == 'square' else frac ** 0.5) * float(cfg['detector_match'])
        k_milli = max(1, int(round(k * 1000)))
        fmin_digits = max(1, int(round(float(cfg['min_forward_V']) * 65536.0 / 20.0)))
        event_s = self.pd_ticks_sent / float(ADWIN_TICKS_PER_NS) * 1e-9      # one Event per shot
        settle = int(np.ceil(5.0 * float(cfg['rc_time_constant_ms']) * 1e-3 / event_s))
        a.set_int_var(31, int(cfg['ain_forward']))
        a.set_int_var(32, int(cfg['ain_reflected']))
        a.set_int_var(33, k_milli)
        a.set_int_var(34, fmin_digits)
        a.set_int_var(35, max(1, int(cfg['consecutive_shots'])))
        a.set_int_var(36, settle)
        a.set_int_var(37, max(0, int(cfg['blind_shots'])))
        a.set_int_var(30, 1)
        print(f"[RF interlock] ON: AIN{cfg['ain_forward']} forward, AIN{cfg['ain_reflected']} reflected; "
              f"trip above {frac:g} reflected/forward power ({cfg['detector_law']} law, "
              f"detector ratio {k_milli / 1000:.3f}) for {max(1, int(cfg['consecutive_shots']))} shots; "
              f"judging only above {cfg['min_forward_V']} V forward; settle {settle} shots "
              f"({settle * event_s * 1e3:.1f} ms).")

    def _check_rf_interlock(self) -> None:
        """Raise (and stop the state machine) if the ADwin has tripped."""
        if not getattr(self, "_ilk_v7", False):
            return
        a = self.adwin
        code = a.get_int_var(40)
        if code == 0:
            return
        f_v, r_v = self._ilk_volts(a.get_int_var(41)), self._ilk_volts(a.get_int_var(42))
        msg = (f"RF INTERLOCK TRIPPED ({self.ILK_REASONS.get(code, 'code ' + str(code))}): "
               f"forward {f_v:.4f} V, reflected {r_v:.4f} V above zero, ratio {a.get_int_var(43) / 1000:.3f} "
               f"(detector units), at iter {a.get_int_var(44)}, rep {a.get_int_var(45)}. "
               f"Triggers stopped. Fix the RF chain, then call reset_rf_interlock().")
        a.set_int_var(10, 0)
        print(msg)
        self.logger.error(msg)
        raise RuntimeError(msg)

    def reset_rf_interlock(self) -> None:
        """Clear the latch (Par_40) after the cause of a trip has been fixed."""
        self.adwin.set_int_var(40, 0)
        print("[RF interlock] latch cleared.")

    def _rf_interlock_report(self) -> Dict[str, Any]:
        """Levels from the sweep just finished (for setting min_forward_V and the threshold)."""
        if not (self.settings['rf_interlock']['enable'] and getattr(self, "_ilk_v7", False)):
            return {'enabled': False}
        a = self.adwin
        rep = {'enabled': True,
               'forward_V_last': self._ilk_volts(a.get_int_var(41)),
               'reflected_V_last': self._ilk_volts(a.get_int_var(42)),
               'max_ratio': a.get_int_var(48) / 1000.0,
               'shots_judged': a.get_int_var(49),
               'zero_forward_V': self._ilk_volts(a.get_int_var(46) - 32768),
               'zero_reflected_V': self._ilk_volts(a.get_int_var(47) - 32768)}
        print(f"[RF interlock] last forward {rep['forward_V_last']:.4f} V, reflected {rep['reflected_V_last']:.4f} V "
              f"above zero; max ratio this sweep {rep['max_ratio']:.3f} (trip at {a.get_int_var(33) / 1000:.3f}); "
              f"{rep['shots_judged']} shots judged; zeros {rep['zero_forward_V']:.4f} / {rep['zero_reflected_V']:.4f} V.")
        return rep

    def run_experiment(self, frequency_range: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Run the complete ODMR pulsed experiment.

        Args:
            frequency_range: List of frequencies in Hz to scan. If None, uses single frequency.

        Returns:
            Dictionary containing:
            - 'success': bool - Whether experiment completed successfully
            - 'frequencies': List[float] - Frequencies scanned
            - 'signal_counts': List[List[float]] - Signal counts for each frequency and sequence point
            - 'reference_counts': List[List[float]] - Reference counts for each frequency and sequence point
            - 'total_counts': List[List[float]] - Total counts for each frequency and sequence point
        """
        try:
            self.logger.info("Starting ODMR Pulsed Experiment")
            # Step 1: Load sequence
            if not self.sequence_description:
                self.logger.error("No sequence loaded")
                print("No sequence loaded")
                return {'success': False, 'error': 'No sequence loaded'}

            # Step 2: Build scan sequences
            if not self.build_scan_sequences():
                return {'success': False, 'error': 'Failed to build scan sequences'}

            # Step 3: Generate AWG sequences
            if not self.generate_awg_task_sequences_adwin_triggering_awg_case():
                return {'success': False, 'error': 'Failed to generate AWG sequences'}

            # Step 3b: DEER target module, once per run, before the ADwin starts triggering
            if self.settings['DEER']['enable'] and not self.setup_deer():
                return {'success': False, 'error': 'Failed to setup DEER'}

            # Step 4: Setup ADwin for photon counting
            if not self._setup_adwin_counting():
                return {'success': False, 'error': 'Failed to setup ADwin counting'}

            # Step 6: Run experiment for each frequency
            # 'success': True,
            # 'frequencies': frequency_range,
            results = {
                'signal_counts': [],
                'reference_counts': [],
                'total_counts': []
            }
            for freq in frequency_range:
                self.logger.info(f"Running experiment at {freq / 1e9:.3f} GHz")

                # Set SG384 frequency
                # if 'sg384' in self.devices:
                self.sg384.set_frequency(freq)
                self.sg384.set_power(self.microwave_power)
                print(f"setting frequency: {freq / 1e9:.3f} GHz")
                print(f"settng power: {self.microwave_power}")
                if self.settings["microwave"]["enable"]:
                    print("enabling output")
                    self.sg384.enable_output()
                self.microwave_frequency = freq
                self.logger.info(f"Set SG384 to {freq / 1e9:.3f} GHz, {self.microwave_power} dBm")

                # Run sequence and collect data
                freq_results = self._run_sequence_and_collect_data()
                self._check_target_lockstep()
                if not freq_results['success']:
                    return {'success': False, 'error': f'Failed at frequency {freq / 1e9:.3f} GHz'}

                results['signal_counts'].append(freq_results['signal_counts'])
                results['reference_counts'].append(freq_results['reference_counts'])
                results['total_counts'].append(freq_results['total_counts'])
            if self.target is not None:
                self.target.off()  # pump off between runs; setup_deer() re-arms it
            results['signal_counts'] = np.array(results['signal_counts'], dtype=np.int64)
            results['reference_counts'] = np.array(results['reference_counts'], dtype=np.int64)
            results['total_counts'] = np.array(results['total_counts'], dtype=np.int64)
            results['success'] = True
            results['frequencies'] = frequency_range

            self.logger.info("ODMR Pulsed Experiment completed successfully")
            return results

        except Exception as e:
            self.logger.error(f"Experiment failed: {e}")
            return {'success': False, 'error': str(e)}

    def _save_adwin_partial(self, results: Dict[str, Any], run_idx: int,
                            averages: int) -> None:
        """Write the ADwin count arrays completed so far to a single .npz so a
        crash during a long averaged run never loses the data. Overwritten after
        every average (always reflects runs 1..run_idx). Pure NumPy, no h5py; the
        final save_hdf5() is still the authoritative output. Best-effort: the
        caller wraps this in try/except so a save hiccup can't stop the run."""
        out_dir = getattr(self, "output_dir", None) or os.getcwd()
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "adwin_partial_latest.npz")
        # temp name ends in .npz so np.savez writes it verbatim (it otherwise
        # appends .npz); os.replace is atomic on the same filesystem.
        tmp = os.path.join(out_dir, "adwin_partial_latest.writing.npz")
        payload = {k: np.asarray(v) for k, v in results.items()
                   if k.startswith(('signal_', 'reference_', 'total_'))}
        payload['runs_completed'] = np.int64(run_idx)
        payload['averages_planned'] = np.int64(averages)
        payload['frequencies'] = np.asarray(results.get('frequencies', []), dtype=float)
        # write to a temp file then replace, so the .npz is never half-written if a
        # crash lands mid-save.
        np.savez(tmp, **payload)
        os.replace(tmp, path)
        self.logger.info(f"[partial] ADwin runs 1..{run_idx}/{averages} saved to {path}")

    def run_experiment_averaged(self, frequency_range: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Run run_experiment() 'averages' times and average the count arrays
        term-by-term to beat down noise. Optionally re-optimize the confocal
        position BETWEEN runs (never in the middle of a run).

        Timing matches your description:
            run 1 (repeat_count reps, no optimization)
            -> optimize -> run 2 -> optimize -> run 3 -> ...
        i.e. optimization happens before runs 2..N, not after the last run.

        Returns a dict containing, for each run i (1-based):
            signal_i, reference_i, total_i        (int64,  shape [n_freq, n_iter])
        plus the term-by-term averages:
            signal_avg, reference_avg, total_avg  (float64, same shape)
        The legacy keys signal_counts / reference_counts / total_counts are set
        to the *_avg arrays so the rest of the code (save_hdf5, logging, plots)
        keeps working with no other changes.
        """
        averages = int(self.settings['averaging']['averages'])
        optimize_between = bool(self.settings['averaging']['optimize_between_runs'])
        if averages < 1:
            averages = 1

            # Reset the confocal optimizer's best-known baseline for THIS experiment run.
            # It gets seeded on the first _optimize_position() call (before run 2),
            # i.e. from the good spot run 1 was measured at.
        self._opt_best_counts = None
        self._opt_best_x = None
        self._opt_best_y = None
        self._opt_best_z = None

        results: Dict[str, Any] = {'success': True, 'frequencies': frequency_range}
        signal_runs, ref_runs, total_runs = [], [], []

        # +++ ETA helpers/state
        def _fmt_dur(seconds: float) -> str:
            seconds = max(0.0, float(seconds))
            h, rem = divmod(int(round(seconds)), 3600)
            m, s = divmod(rem, 60)
            return (f"{h}h {m:02d}m {s:02d}s" if h else
                    f"{m}m {s:02d}s" if m else f"{s}s")

        # a-priori estimate (pure acquisition; excludes optimization/AWG-reprogram/
        # analysis, so it's a lower bound). Available once sequences are built.
        try:
            _shots_per_avg = int(self.repeat_count) * int(self.number_of_iterations)
            _secs_acq_per_avg = _shots_per_avg * float(self.sequence_duration) * 1e-9
            print(f"[ETA] a-priori: ~{_shots_per_avg:,} shots/avg x {averages} avg "
                  f"~= {_fmt_dur(_secs_acq_per_avg * averages)} of pure acquisition "
                  f"(excludes optimization/overhead).")
        except Exception:
            pass

        _t_start = time.monotonic()  # +++
        _per_run = []  # +++
        self.progress = 0.0
        self.updateProgress.emit(0)
        for run_idx in range(1, averages + 1):
            _run_t0 = time.monotonic()  # +++ full per-average wall time (incl. optimize)
            msg = f"=== Average run {run_idx}/{averages} ==="
            self.logger.info(msg)
            print(msg)

            # Optimize BEFORE runs 2..N. Never before run 1, never mid-run.
            if optimize_between and run_idx > 1:
                try:
                    self._optimize_position()
                except Exception as e:
                    self.logger.error(f"Optimization before run {run_idx} failed (continuing): {e}")

            run = self.run_experiment(frequency_range)
            if not run.get('success'):
                self.logger.error(f"Run {run_idx} failed: {run.get('error')}")
                return {'success': False,
                        'error': run.get('error', f'run {run_idx} failed'),
                        'frequencies': frequency_range}

            s = np.asarray(run['signal_counts'], dtype=np.int64)
            r = np.asarray(run['reference_counts'], dtype=np.int64)
            t = np.asarray(run['total_counts'], dtype=np.int64)

            signal_runs.append(s)
            ref_runs.append(r)
            total_runs.append(t)
            results[f'signal_{run_idx}'] = s
            results[f'reference_{run_idx}'] = r
            results[f'total_{run_idx}'] = t

            try:
                self._save_adwin_partial(results, run_idx, averages)
            except Exception as e:
                self.logger.error(f"partial save after run {run_idx} failed (continuing): {e}")

            # +++ ETA update: measured wall-time per average is the honest estimator.
            _per_run.append(time.monotonic() - _run_t0)
            _remaining = averages - run_idx
            _mean = sum(_per_run) / len(_per_run)  # includes optimization + overhead
            _elapsed = time.monotonic() - _t_start
            _eta = _mean * _remaining
            _finish = time.strftime('%H:%M:%S', time.localtime(time.time() + _eta))
            print(f"[ETA] avg {run_idx}/{averages} took {_fmt_dur(_per_run[-1])} | "
                  f"elapsed {_fmt_dur(_elapsed)} | mean/avg {_fmt_dur(_mean)} | "
                  f"~{_fmt_dur(_eta)} left | finish ~{_finish}")
            # feed the GUI progress bar + "time remaining" label (same mechanism
            # as odmr_sweep_continuous: base Experiment derives remaining_time
            # from self.progress once we emit updateProgress).
            self.progress = 100.0 * run_idx / averages
            self.updateProgress.emit(int(round(self.progress)))

        # Term-by-term average across runs (float; fractional averages are correct).
        signal_avg = np.mean(np.stack(signal_runs, axis=0), axis=0)
        ref_avg = np.mean(np.stack(ref_runs, axis=0), axis=0)
        total_avg = np.mean(np.stack(total_runs, axis=0), axis=0)

        results['signal_avg'] = signal_avg
        results['reference_avg'] = ref_avg
        results['total_avg'] = total_avg

        # Keep the legacy keys pointing at the averages so nothing downstream breaks.
        results['signal_counts'] = signal_avg
        results['reference_counts'] = ref_avg
        results['total_counts'] = total_avg
        results['averages'] = averages

        self.logger.info(f"Averaging complete over {averages} run(s)")
        return results

    def _read_opt_counts(self) -> float:
        """Read the current count rate (cts/s) from the simple counter binary."""
        raw = self.adwin.read_probes('int_var', id=1)
        return raw * 1e3 / self._opt_count_time_ms

    def _optimize_axis(self, axis: str, step: float) -> None:
        """
        Maximize counts on one nanodrive axis ('x'/'y'/'z'), using either a
        3-point quadratic fit ("fit") or a hill-climb ("convergence"), selected
        by settings['averaging']['optimization_mode'].


        fit:
            Step back 2x first to remove backlash, sample minus/old/plus, then
            parabola-fit. The vertex is trusted ONLY when the fit is concave-down
            (a < 0, a real maximum) AND the vertex lands inside the sampled span;
            otherwise fall back to the best point actually measured.


        convergence:
            Sample the current spot plus two + steps, pick a climbing direction,
            then keep stepping while each step raises counts by more than
            opt_improve_frac, up to opt_max_steps. Land on the best point seen.
            Moves as little as possible (less stage drift) and requires a real
            improvement, not a shot-noise flicker, before accepting a step.


        Out-of-range moves (ARGUMENT_ERROR) are skipped.
        """
        nd = self.nd
        pos_key = f'{axis}_pos'
        opt = self.settings['averaging']
        settle_time = float(opt['opt_settle_time'])
        damping_ratio = float(opt['opt_damping_ratio'])

        def move(target) -> bool:
            try:
                nd.update({pos_key: target})
                time.sleep(settle_time)
                return True
            except Exception as e:
                if "ARGUMENT_ERROR" in str(e):
                    print(f"Skipping {axis} move (out of range): {e}")
                    return False
                raise

        def read_pos():
            return nd.read_probes(pos_key)

        if opt['optimization_mode'] == "fit":
            # Back off two steps to take out backlash, then sample minus / old / plus.
            move(read_pos() - 2 * step)
            samples = []  # (position, count_rate)
            for _ in range(3):
                if move(read_pos() + step):
                    samples.append((read_pos(), self._read_opt_counts()))

            if len(samples) < 3:
                print(f"Skipping {axis}-optimization: insufficient valid points.")
                return

            coords = np.array([p for p, _ in samples], dtype=float)
            counts = np.array([c for _, c in samples], dtype=float)
            a, b, _c = np.polyfit(coords, counts, 2)

            best_pos, best_counts = max(samples, key=lambda pc: pc[1])
            lo, hi = float(coords.min()), float(coords.max())

            # Only trust the vertex for a concave-down parabola whose vertex is in range.
            vertex = None
            use_vertex = False
            if a < 0:
                vertex = float(-b / (2 * a))
                if lo <= vertex <= hi:
                    use_vertex = True

            if use_vertex:
                if not move(vertex):
                    return
                counts_at_vertex = self._read_opt_counts()
                # If a sampled point actually beat the vertex, damp toward it.
                if best_counts > counts_at_vertex:
                    move(vertex + damping_ratio * (best_pos - vertex))
            else:
                # Unreliable fit (a >= 0, or vertex outside the sampled span):
                # go to the best point we physically measured.
                print(f"{axis}: unreliable parabola (a={a:.3g}); using best sampled point.")
                move(best_pos)


        elif opt['optimization_mode'] == "convergence":
            max_steps = int(opt['opt_max_steps'])
            improve_frac = float(opt['opt_improve_frac'])

            def better(new, ref):
                # new must beat ref by more than improve_frac (relative) to count.
                return new > ref * (1.0 + improve_frac)

            # Opener: sample the current spot, then two steps in +.
            samples = []  # (position, count_rate)
            samples.append((read_pos(), self._read_opt_counts()))
            for _ in range(2):
                if move(read_pos() + step):
                    samples.append((read_pos(), self._read_opt_counts()))
            if len(samples) < 3:
                print(f"Skipping {axis}-optimization: insufficient valid points.")
                return
            best_pos, best_counts = max(samples, key=lambda pc: pc[1])
            c_start = samples[0][1]  # counts at the original position
            c_far = samples[-1][1]  # counts at the furthest + point
            # Pick a climbing direction from the opener:
            #   + : the furthest point is best and clearly beats the start -> keep going +
            #   - : the original start is best -> + was downhill, reverse
            #   0 : the peak is bracketed in the middle (we already overshot) -> just
            #       land on the best of the three below.
            direction = 0
            if best_pos == samples[-1][0] and better(c_far, c_start):
                direction = +1
            elif best_pos == samples[0][0]:
                direction = -1
                move(samples[0][0])  # return to start before heading - (takes up + backlash)
            steps_taken = 0
            while direction != 0 and steps_taken < max_steps:
                if not move(read_pos() + direction * step):
                    break  # hit the range limit
                c_new = self._read_opt_counts()
                if better(c_new, best_counts):
                    best_pos, best_counts = read_pos(), c_new
                    steps_taken += 1
                else:
                    break  # stopped improving -> we just stepped past the peak
            move(best_pos)  # land on the best point we actually measured
            print(f"{axis}: converged at {best_pos:.3f} um, ~{best_counts:.0f} cts/s "
                  f"(+{steps_taken} step(s) past opener).")
        else:
            raise NotImplementedError

    def _optimize_position(self) -> None:
        """
        One-shot confocal re-optimization between averaging runs, with the
        confocal-point retention net: a best-known (position, counts) baseline is
        remembered across runs; after the z/x/y pass, if counts dropped below
        min_retention_ratio of the best-known counts the stage snaps back toward
        the best position; the baseline is only updated below
        acceptable_counts * acceptable_counts_ratio (rejects noise spikes); and
        optionally the whole pass is skipped when counts are still healthy.


        Reuses the counter binary (Averagable_Trial_Counter.TB1) and does NOT touch
        ADbasic source: the next run_experiment() reloads the pulsed binary via
        _setup_adwin_counting() and reprograms Proteus, so state self-heals.
        """
        nd = getattr(self, 'nd', None)
        if nd is None:
            self.logger.warning("No nanodrive available; skipping optimization.")
            print("No nanodrive available; skipping optimization.")
            return


        opt = self.settings['averaging']
        self._opt_count_time_ms = float(opt['opt_count_time'])
        num_cycles = int(opt['opt_num_cycles'])
        settle_time = float(opt['opt_settle_time'])
        damping_ratio = float(opt['opt_damping_ratio'])
        min_retention_ratio = float(opt['min_retention_ratio'])
        acceptable_counts = float(opt['acceptable_counts'])
        acceptable_counts_ratio = float(opt['acceptable_counts_ratio'])
        min_reoptimize_ratio = float(opt['min_reoptimize_ratio'])
        reoptimize_only_if_needed = bool(opt['reoptimize_only_if_needed'])


        # --- switch ADwin to the simple counter (does NOT touch ADbasic source) ---
        self.adwin.stop_process(1)
        time.sleep(0.1)
        self.adwin.clear_process(1)
        counter_path = get_adwin_binary_path('Averagable_Trial_Counter.TB1')
        self.adwin.update({'process_1': {'load': str(counter_path)}})
        self.adwin.set_int_var(10, num_cycles)
        adwin_delay = round((self._opt_count_time_ms * 1e6) / 3.3)
        self.adwin.update({'process_1': {'delay': adwin_delay, 'running': True}})


        # --- laser on for counting (reprogrammed by the next run) ---
        try:
            self.proteus.set_channel_voltage_high(4, self.settings["Laser Control"])
        except Exception as e:
            self.logger.warning(f"Could not force laser channel high for optimization: {e}")

        time.sleep(0.2)  # let the stage settle and the counter spin up
        OPT_MIN_VALID_CPS = 200.0
        current_counts = self._read_opt_counts()
        if current_counts < OPT_MIN_VALID_CPS:
            self.logger.warning(
                f"[optimize] counter read {current_counts:.0f} cts/s "
                f"(< {OPT_MIN_VALID_CPS:.0f}). The ADwin optimization counter is seeing "
                f"~no light. Check that the laser is on. Forcing an optimization pass.")
            print(f"[optimize] WARNING: {current_counts:.0f} cts/s -- ADwin counter sees "
                  f"no light (SPCM on the MultiHarp?). Not trusting this as 'healthy'.")


        # Only seed the baseline from a believable reading.
        if getattr(self, '_opt_best_counts', None) is None and current_counts >= OPT_MIN_VALID_CPS:
            self._opt_best_counts = current_counts
            self._opt_best_x = nd.read_probes('x_pos')
            self._opt_best_y = nd.read_probes('y_pos')
            self._opt_best_z = nd.read_probes('z_pos')
            print(f"Seeded best-known baseline: {current_counts:.0f} cts/s.")


        did_optimize = False
        counts_healthy = (
            current_counts >= OPT_MIN_VALID_CPS
            and self._opt_best_counts is not None
            and current_counts >= (min_reoptimize_ratio * self._opt_best_counts)
        )
        if reoptimize_only_if_needed and counts_healthy:
            print(f"Counts healthy ({current_counts:.0f} >= "
                  f"{min_reoptimize_ratio:.2f} * {self._opt_best_counts:.0f}); "
                  f"skipping re-optimization -- stage NOT moved.")
        else:
            did_optimize = True
            print("Optimizing confocal position between runs (z -> x -> y)...")
            self._optimize_axis('z', float(opt['opt_zstep']))   # <-- z first
            self._optimize_axis('x', float(opt['opt_xstep']))
            self._optimize_axis('y', float(opt['opt_ystep']))



            # --- retention net (mirrors the confocal-point experiment) ---
            optimized_counts = self._read_opt_counts()
            if optimized_counts > self._opt_best_counts:
                if optimized_counts < (acceptable_counts * acceptable_counts_ratio):
                    self._opt_best_counts = optimized_counts
                    self._opt_best_x = nd.read_probes('x_pos')
                    self._opt_best_y = nd.read_probes('y_pos')
                    self._opt_best_z = nd.read_probes('z_pos')
                    print(f"New best-known position: {optimized_counts:.0f} cts/s.")
                else:
                    print(f"Ignoring implausibly high counts ({optimized_counts:.0f} cts/s) "
                          f"for the baseline (likely a fluctuation).")
            elif optimized_counts < (min_retention_ratio * self._opt_best_counts):
                print(f"Post-opt counts {optimized_counts:.0f} < {min_retention_ratio:.2f} * best "
                      f"{self._opt_best_counts:.0f}; snapping back toward best-known position.")
                cx = nd.read_probes('x_pos')
                cy = nd.read_probes('y_pos')
                cz = nd.read_probes('z_pos')
                try:
                    nd.update({'x_pos': cx + damping_ratio * (self._opt_best_x - cx),
                               'y_pos': cy + damping_ratio * (self._opt_best_y - cy),
                               'z_pos': cz + damping_ratio * (self._opt_best_z - cz)})
                    time.sleep(settle_time)
                except Exception as e:
                    if "ARGUMENT_ERROR" in str(e):
                        print(f"Snap-back skipped (out of range): {e}")
                    else:
                        raise


        if did_optimize:
            try:
                final = self._read_opt_counts()
                print(f"Optimization done: ~{final:.0f} cts/s at "
                      f"x={nd.read_probes('x_pos'):.3f}, "
                      f"y={nd.read_probes('y_pos'):.3f}, "
                      f"z={nd.read_probes('z_pos'):.3f}")
            except Exception:
                pass


        # Stop the counter so the next run starts clean (run_experiment reloads the SM).
        self.adwin.update({'process_1': {'running': False}})
        self.adwin.stop_process(1)
        time.sleep(0.1)
        self.adwin.clear_process(1)

    def cleanup(self):
        """Cleanup experiment resources."""
        # Stop Adwin process
        if self.adwin and self.adwin.is_connected:
            self.adwin.stop_process(1)
            self.adwin.clear_process(1)

        # Disable microwave sweep and output
        if self.sg384 and self.sg384.is_connected:
            self.sg384.disable_modulation()
            self.sg384.disable_output()

        self.log("ODMR Phase Continuous Sweep Experiment cleanup complete")

    def _setup_adwin_counting(self) -> bool:
        """
        Setup ADwin for photon counting using the odmr_pulsed_counter.bas process.

        Returns:
            True if setup successful
        """
        try:
            # Load the odmr_pulsed_counter.bas process (Process 2)
            # This process handles dual-gate counting triggered by Proteus
            # process_file = "odmr_pulsed_counter.__2"
            process_number = 1

            if not self.adwin.is_connected:
                self.adwin.connect()

            # Proper cleanup like debug script
            self.log("Cleaning up any existing ADwin process...")
            try:
                self.adwin.stop_process(process_number)
                time.sleep(0.1)
            except Exception:
                pass
            try:
                self.adwin.clear_process(process_number)
            except Exception:
                pass
            # the following were tests for option 3
            """# test 1
            odmr_pulsed_counter_path = get_adwin_binary_path('odmr_pulsed_counter.TB2')
            # test 2
            odmr_pulsed_counter_path = get_adwin_binary_path('adbasic_long_waveform_v1.TB2')
            # test 3
            odmr_pulsed_counter_path = get_adwin_binary_path('adbasic_long_waveform_v2.TB2')
            #tested digout
            odmr_pulsed_counter_path = get_adwin_binary_path('test_adwin_digout.TB1')
            # tested proteus delay and digout delay
            odmr_pulsed_counter_path = get_adwin_binary_path('test_adwin_delays.TB1')
            self.adwin_file = 'adwin_triggering_proteus.TB1'
            self.adwin_file = 'T1.TB1'
            self.adwin_file = 'test_adwin_delays_v2.TB1'
            self.adwin_file = 'PULSE_FIXED_WAIT_READ.TB1'
            self.adwin_file = 'TIME_RESOLUTION.TB1'
            self.adwin_file = 'TIME_RESOLUTIONz_T1_transition.TB1'"""
            # ── choose the state-machine binary ──────────────────────────────────────
            # Both binaries pin Processdelay ~1 us. v1 runs signal+gap+ref as ONE Event,
            # which overruns whenever the block is wide, walking the gate ~150 ns/shot --
            # harmless for a wide T1 readout, fatal for a narrow Rabi gate on the onset.
            # v2 splits them into separate Events (gap = bare off-Events) so nothing
            # overruns and the reference can sit ~2 us later in the bright window.
            self.adwin_file = self.settings["adwin"]["adbasic_file"]

            if self.adwin_file.endswith('v2.TB1') or self.adwin_file.endswith('v4.TB1'):
                expected_signature = 7778
            elif self.adwin_file.endswith('v1.TB1'):
                expected_signature = 7777
            elif self.adwin_file.endswith('v7.TB1'):
                expected_signature = 7784
            elif self.adwin_file.endswith('v5.TB1') or self.adwin_file.endswith('v6.TB1'):
                expected_signature = 7783
            elif self.adwin_file.endswith('v3.TB1'):
                expected_signature = 7779
            else:
                expected_signature = 7790
            odmr_pulsed_counter_path = get_adwin_binary_path(self.adwin_file)
            self.adwin.update({'process_1': {'load': str(odmr_pulsed_counter_path)}})
            # Set ADwin parameters for counting
            # OLD SET
            """self.adwin.set_int_var(3, self.count_time )
            self.adwin.set_int_var(4, self.reset_time )
            self.adwin.set_int_var(5, self.repeat_count)
            self.adwin.set_int_var(6, self.number_of_iterations)
            self.adwin.set_int_var(9, self.sequence_duration)
            self.adwin.set_int_var(11, 1)"""
            count_time_10_ns = int(self.count_time / 10)
            ref_delay_10ns = int(self.reset_time / 10) if self.reset_time else 167  # gap between gates
            seq_dur_us = int(self.sequence_duration / 1000)
            TRIG_10NS = 12
            self.adwin.set_int_var(1, int(self.repeat_count))
            self.adwin.set_int_var(2, int(self.number_of_iterations))
            self.adwin.set_int_var(3, count_time_10_ns)  # Par_3 signal gate (10 ns units)
            self.adwin.set_int_var(4, ref_delay_10ns)  # Par_4 inter-gate gap (10 ns units)
            self.adwin.set_int_var(5, count_time_10_ns)  # Par_5 reference gate (10 ns units)
            self.adwin.set_int_var(6, seq_dur_us)  # Par_6 seq_dur_us (informational here)
            self.adwin.set_int_var(7, TRIG_10NS)  # Par_7 trigger width (10 ns units)
            self.adwin.set_int_var(11, 0)  # Par_11 pre_count fine offset (WAS 1 = old start flag)
            self.adwin.set_int_var(12, 0)  # Par_12 fine_delay
            self.adwin.set_int_var(14, 3)  # Par_14 gate countdown -> SET to your test_seq_rabi peak (~readout us)
            # for debug only: self.adwin.set_int_var(15, 200)
            # State machine boots into IDLE; Par_10 starts it (set in _run_sequence_and_collect_data)
            self.adwin.start_process(process_number)
            time.sleep(0.3)
            if self.adwin.get_int_var(80) != expected_signature:
                raise RuntimeError(f"ADwin signature mismatch (loaded {self.adwin_file})")

            self.logger.info(
                f"ADwin counting setup: count_time={self.count_time}ns, reset_time={self.reset_time}ns, reps={self.repeat_count}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to setup ADwin counting: {e}")
            return False

    def _run_sequence_and_collect_data(self) -> Dict[str, Any]:
        """
        Run a single sequence and collect photon counting data from ADwin.

        Returns:
            Dictionary with signal_counts, reference_counts, total_counts
        """
        try:
            # Wait for sequence to complete and collect data
            # The ADwin adwin_triggering_proteus.bas process accumulates counts
            # and stores them in arrays Data_1[iteration] (signal) and Data_2[iteration] (reference)

            # Collect data
            # this method has python collecting data and telling aadwin to stop:
            # Collect data
            if self.settings['adwin']['auto_readout_start']:
                self.readout_start_ns = self.readout_start_from_sequence()
            X = self.configure_gates_v5()
            self.configure_rf_interlock()
            start_time = datetime.datetime.now()
            self.s_t = start_time.strftime("%m_%d_%Y_%H:%M:%S")

            # ---- Launch this sweep: THIS is what was missing ----
            self.adwin.set_int_var(20, 0)  # clear done flag
            self.adwin.set_int_var(10, 1)  # Par_10 = 1 -> RUN (leaves IDLE, starts firing triggers)

            # ---- Wait on the handshake instead of a fixed sleep ----
            # SM fires repeat_count x n_iterations triggers, counts, then sets Par_20 = 1.
            t_start = time.time();
            t_hb = time.time();
            t_dbg = 0.0
            last_hb = self.adwin.get_int_var(25)
            """print(f"[wait] Par_71(Processdelay)={self.adwin.get_int_var(71)} (configure said 19747), "
                  f"Par_80(sig)={self.adwin.get_int_var(80)} (expect 7783), "
                  f"Par_10={self.adwin.get_int_var(10)} (expect 1)")"""
            pd_ns = self.pd_ticks_sent / 0.3
            expected_sweep_s = 2 * pd_ns * 1e-9 * self.repeat_count * max(1, self.number_of_iterations)
            timeout_s = max(60.0, expected_sweep_s * 10.0)  # 10x headroom, not 1000x
            while self.adwin.get_int_var(20) == 0:
                time.sleep(0.05);
                self._check_rf_interlock()
                now = time.time()
                hb = self.adwin.get_int_var(25)
                if hb != last_hb:
                    last_hb = hb; t_hb = now
                elif now - t_hb > 15:
                    raise RuntimeError(f"heartbeat frozen: state={self.adwin.get_int_var(26)} "
                                       f"iter={self.adwin.get_int_var(27)} rep={self.adwin.get_int_var(28)} "
                                       f"capped={self.adwin.get_int_var(74)}")
                if now - t_dbg > 2.0:
                    t_dbg = now
                    """print(f"[wait] state={self.adwin.get_int_var(26)} iter={self.adwin.get_int_var(27)} "
                          f"rep={self.adwin.get_int_var(28)} capped={self.adwin.get_int_var(74)} "
                          f"Par_71={self.adwin.get_int_var(71)} hb={hb}")"""
                if now - t_start > timeout_s:
                    raise TimeoutError(f"never done: state={self.adwin.get_int_var(26)} "
                                       f"iter={self.adwin.get_int_var(27)}/{self.number_of_iterations} "
                                       f"rep={self.adwin.get_int_var(28)}/{self.repeat_count} "
                                       f"capped={self.adwin.get_int_var(74)} Par_71={self.adwin.get_int_var(71)}")

            self._check_rf_interlock()
            end_time = datetime.datetime.now()
            self.e_t = end_time.strftime("%m_%d_%Y_%H:%M:%S")
            self.number_of_sequences_done_by_adwin = self.adwin.get_int_var(28)  # Par_28 rep counter
            self.adwin_reached_the_end = self.adwin.get_int_var(20)  # Par_20: 1 = done
            signal_counts = np.array(self.adwin.get_int_data(1, self.number_of_iterations), dtype=np.int64)
            """total_counts = np.array(self.adwin.get_int_data(2, self.number_of_iterations), dtype=np.int64)
            ref_counts=total_counts.astype(np.int64)-signal_counts.astype(np.int64)"""
            ref_counts = np.array(self.adwin.get_int_data(2, self.number_of_iterations), dtype=np.int64)
            total_counts = ref_counts.astype(np.int64) + signal_counts.astype(np.int64)
            for signal_counts_val in signal_counts:
                self.logger.info(f"signal_counts_val {signal_counts_val}")
                print(f"signal_counts_val {signal_counts_val}")
            for reference_counts_val in ref_counts:
                self.logger.info(f"reference_counts_val {reference_counts_val}")
                print(f"reference_counts_val {reference_counts_val}")
            for total_counts_val in total_counts:
                self.logger.info(f"total_counts_val {total_counts_val}")
                print(f"total_counts_val {total_counts_val}")
                # ---- ack so the SM idles; the next frequency re-arms via Par_10=1 ----
            a = self.adwin
            sig_start = a.get_int_var(75)  # echoed sig_start_ticks
            ref_start = a.get_int_var(76)  # echoed ref_start_ticks
            capped = a.get_int_var(74)  # spin-loops that hit the cap (want 0)
            pd_ticks = a.get_int_var(71)  # Processdelay actually used
            sig_id = a.get_int_var(80)  # 7783 for direct-tick v5

            print(f"[v5] signature={sig_id} (expect 7783), Processdelay={pd_ticks} cyc")
            print(f"[v5] sig_start={sig_start} ticks, ref_start={ref_start} ticks (echoed back)")
            print(f"[v5] capped spin-loops={capped}  (want 0)")
            if capped:
                print("[v5] WARNING: a spin-loop hit its cap -> a wait never reached target. "
                      "If ADWIN_TICKS_PER_NS is far too large the targets are unreachable; "
                      "recheck the DIO16 high-time calibration.")
            rf_ilk = self._rf_interlock_report()
            self.adwin.set_int_var(20, 0)
            self.adwin.set_int_var(10, 0)
            # Stop Proteus
            # self.proteus.stop_sequence()
            # self.proteus.driver._close()

            return {
                'success': True,
                'signal_counts': signal_counts,
                'reference_counts': ref_counts,
                'total_counts': total_counts,
                'rf_interlock': rf_ilk
            }

        except Exception as e:
            self.logger.error(f"Failed to run sequence and collect data: {e}")
            return {'success': False, 'error': str(e)}

    """
sequence: name=odmr_pulsed, type=odmr, duration=1002500ns, sample_rate=1GHz, repeat_count=50000
variable pulse_duration, start=50ns, stop=500ns, steps=20
marker, laser_int_1 on channel 1 at 0ns, 500ns
pi/2 pulse on channel 1 at 500ns, gaussian, pulse_duration, 1.0
pi/2 pulse on channel 2 at 500ns, gaussian, pulse_duration, 1.0
wait pulse on channel 1 at pulse_duration+0.000000500, square, 2*pulse_duration, 0.0
wait pulse on channel 2 at pulse_duration+0.000000500, square, 2*pulse_duration, 0.0
pi/2 pulse on channel 1 at 3*pulse_duration+0.000000500, gaussian, pulse_duration, 1.0
pi/2 pulse on channel 2 at 3*pulse_duration+0.000000500, gaussian, pulse_duration, 1.0
marker, laser_readout_1 on channel 1 at 2500ns, 1ms
marker, readout_counts_1 on channel 2 at 2500ns, 300ns
marker, reference_counts_1 on channel 2 at 1002200ns, 300ns
sequence: name=SCC, type=SCC, duration=1400ns, sample_rate=1GHz, repeat=50000
shelving pulse on channel 3 at 0ns, square, 300ns, 0.6
ionization pulse on channel 3 at 300ns, square, 500ns, 1.0
readout pulse on channel 3 at 800ns, square, 600ns, 0.3
    sequence: name=SCC, type=SCC, duration=1400ns, sample_rate=1GHz, repeat=50000
    shelving pulse on channel 1 at 0ns, square, 300ns, 0.6
    ionization pulse on channel 1 at 300ns, square, 500ns, 1.0
    readout pulse on channel 1 at 800ns, square, 600ns, 0.3"""
    """    
    sequence: name=odmr_pulsed, type=odmr, duration=15500ns, sample_rate=1GHz, repeat_count=50000
    variable pulse_duration, start=0ns, stop=1500ns, steps=20
    var_pulse pulse on channel 2 at 0 ns, square, pulse_duration, 0.0
    initialize pulse on channel 4 at 0ns, square, 2000ns, 1.0
    initialize pulse on channel 3 at 0ns, square, 2000ns, 1.0
    signal pulse on channel 4 at 0.000012 + pulse_duration, square, 300ns, 1.0
    signal pulse on channel 3 at 0.000012 + pulse_duration, square, 300ns, 1.0
    ref pulse on channel 4 at 0.0000134 + pulse_duration, square, 300ns, 1.0
    ref pulse on channel 3 at 0.0000134 + pulse_duration, square, 300ns, 1.0

    "sequence: name=T1, type=T1, duration=504000ns, sample_rate=1GHz, repeat_count=50000
variable pulse_duration, start=100ns, stop=500000ns, steps=20
initialize pulse on channel 4 at 0.0005-pulse_duration, square, 2000ns, 1.0
initialize pulse on channel 3 at 0.0005-pulse_duration, square, 2000ns, 1.0
wait pulse on channel 2 at 0.0005-pulse_duration, square, pulse_duration, 0.0
readout pulse on channel 4 at 502000ns, square, 2000ns, 1.0
readout pulse on channel 3 at 502000ns, square, 2000ns, 1.0"
    """

    def create_example_odmr_sequence(self) -> str:
        """
        This is just for testing, please use a file or GUI setting updates
        Create an example ODMR sequence using the sequence language.

        Returns:
            Sequence text in the sequence language format

        PLEASE NOTE: Readout is fixed so please only vary the starting part not the ending part of the sequence.
        #marker, laser_init_1 on channel 4 at 0ns, 2000ns
        #marker, laser_readout_1 on channel 4 at 2900ns, 2000ns
        for chirps:
        Two-pump sequences (e.g. the standing+moving sech/tanh scheme) want the two pulses swept in opposite directions
        to cancel excitation-time dispersion — do that by setting one pulse's bandwidth negative (it just flips the sweep sign).
        For HS: HS{n_left, n_right} works via offset-independent adiabaticity, which reduces exactly to a symmetric HS at n_left=n_right=1
        HS{1,6} matches what a ref paper found best for short traces; flip to n_left=6, n_right=1 to reverse which side is steep.

        ===
        sequence: name=rabi, type=rabi, duration=14300ns, sample_rate=1GHz, repeat_count=33333
        variable pulse_duration, start=4ns, stop=3000ns, steps=64
        init pulse on channel 4 at 0ns, square, 2000ns, 1.0
        MW pulse on channel 1 at 3000ns, square, pulse_duration, 0.1
        readout pulse on channel 4 at 6071ns, square, 8229ns, 1.0
        marker, reference_MW_1 on channel 1 at 3000ns, pulse_duration
        marker, reference_readout_1 on channel 4 at 6071ns, 8229ns
        ===
        sequence: name=rabi, type=rabi, duration=14300ns, sample_rate=1GHz, repeat_count=33333
        variable pulse_duration, start=1000ns, stop=1000ns, steps=1
        init pulse on channel 4 at 0ns, square, 2000ns, 1.0
        MW pulse on channel 1 at 3000ns, square, pulse_duration, 1.0
        readout pulse on channel 4 at 6071ns, square, 8229ns, 1.0
sequence: name=RABI, type=RABI, duration=8988ns, sample_rate=1GHz, repeat_count=1000000
variable pulse_duration, start=4ns, stop=600ns, steps=64
init pulse on channel 4 at 0ns, square, 2000ns, 1.0
MW pulse on channel 1 at 3450ns, square, pulse_duration, 0.3
readout pulse on channel 4 at 5988ns, square, 3000ns, 1.0

sequence: name=odmr, type=odmr, duration=11128ns, sample_rate=1GHz, repeat_count=100000
variable pulse_duration, start=110ns, stop=110ns, steps=1
init pulse on channel 4 at 0.000003112-pulse_duration, square, 2000ns, 1.0
MW pulse on channel 1 at 0.000007336-pulse_duration, square, pulse_duration, 0.1
readout pulse on channel 4 at 6310ns, square, 4818ns, 1.0

sequence: name=Rabi, type=Rabi, duration=11128ns, sample_rate=1GHz, repeat_count=100000
variable pulse_duration, start=0ns, stop=500ns, steps=64
init pulse on channel 4 at 0.000003112-pulse_duration, square, 2000ns, 1.0
MW pulse on channel 1 at 0.000007336-pulse_duration, square, pulse_duration, 0.1
readout pulse on channel 4 at 6310ns, square, 4818ns, 1.0

sequence: name=Echo, type=Echo, duration=32750ns, sample_rate=1GHz, repeat_count=100000
variable pulse_duration, start=0ns, stop=12500ns, steps=64
init pulse on channel 4 at 0.000025127-2*pulse_duration, square, 2000ns, 1.0
pi_2_1 pulse on channel 1 at 0.000029351-2*pulse_duration, square, 22ns, 1.0
pi pulse on channel 1 at 0.000029373-pulse_duration, square, 43ns, 1.0
pi_2_2 pulse on channel 1 at 29416ns, square, 22ns, 1.0
readout pulse on channel 4 at 28412ns, square, 4338ns, 1.0
init pulse on channel 4 at 0.000025127-2*pulse_duration, square, 2000ns, 1.0
pi_2_1 pulse on channel 1 at 0.000029351-2*pulse_duration, square, 22ns, 1.0
pi pulse on channel 1 at 0.000029373-pulse_duration, square, 43ns, 1.0
pi_2_2 pulse on channel 1 at 29416ns, square, 22ns, 1.0

sequence: name=Echo, type=Echo, duration=32750ns, sample_rate=1GHz, repeat_count=100000
variable pulse_duration, start=100ns, stop=500ns, steps=64
readout pulse on channel 4 at 0.000028112+pulse_duration, square, 4038ns, 1.0
readout pulse on channel 3 at 28412ns, square, 4338ns, 1.0
sequence: name=Echo, type=Echo, duration=32750ns, sample_rate=1GHz, repeat_count=100000
variable pulse_duration, start=0ns, stop=6250ns, steps=64
init pulse on channel 4 at 0.000025131-2*pulse_duration, square, 2000ns, 1.0
pi_2_1 pulse on channel 1 at 0.000029355-2*pulse_duration, square, 21ns, 1.0
pi pulse on channel 1 at 0.000029376-pulse_duration, square, 41ns, 1.0
pi_2_2 pulse on channel 1 at 29417ns, square, 21ns, 1.0
readout pulse on channel 4 at 28412ns, square, 4338ns, 1.0
sequence: name=Rabi, type=Rabi, duration=11128ns, sample_rate=1GHz, repeat_count=100000
variable pulse_duration, start=0ns, stop=200ns, steps=64
init pulse on channel 4 at 0.000003112-pulse_duration, square, 2000ns, 1.0
MW pulse on channel 2 at 0.000007336-pulse_duration, square, pulse_duration, 1.0
readout pulse on channel 4 at 6310ns, square, 4818ns, 1.0

3000048ns

==
sequence: name=Rabi, type=Rabi, duration=11128ns, sample_rate=1GHz, repeat_count=100000
variable pulse_duration, start=0ns, stop=200ns, steps=64
init pulse on channel 4 at 0.000003005-pulse_duration, square, 2000ns, 1.0
MW pulse on channel 1 at 0.000007229-pulse_duration, square, pulse_duration, 1.0
readout pulse on channel 4 at 6203ns, square, 4925ns, 1.0
        """
        sequence_text = """
# Noise injection, amp-N8 trains: square pi/2 vs first-order Bernoulli (Zeng 2018 eq. 12, alpha 1, 3pi/2, 123 ns, same pulse as bern1_trains).
# OU detuning noise, sigma = 0.5 MHz rms, tau_c = 20 ns, seed 8520; noise enters as the drive phase, phi(t) = -2 pi * integral(delta dt), stepped every 4 ns.
# k = 0..10 square train (k = 0 no noise; 1..10 OU realizations), k = 11..21 Bernoulli train (same realizations), k = 22 bright, k = 23 dark.
# Readout at 6153 ns (4975 ns); last MW pulse ends at 7179 ns. adwin/auto_readout_start = False, adwin/readout_start_ns = 6224. MW 2.3975 GHz.
sequence: name=noise_bern_amp_N8_s0.5_tau20, type=gates, duration=11128ns, sample_rate=1GHz, repeat_count=250000
variable k, start=0, stop=23, steps=24
s0i pulse on channel 1 at 6195ns, square, 4ns, (0.02109*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.02109*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.02109*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.02109*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.02109*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.02109*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.02109*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.02109*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.02109*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.02109*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.02109*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s0q pulse on channel 2 at 6195ns, square, 4ns, (0.00014*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.00004*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.00018*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.00008*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.00021*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00021*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.00001*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.00012*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.00013*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s1i pulse on channel 1 at 6199ns, square, 4ns, (0.05508*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.05507*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.05508*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.05506*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.05507*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.05506*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.05504*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.05508*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.05508*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.05507*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.05507*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s1q pulse on channel 2 at 6199ns, square, 4ns, (0.00087*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.00015*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.00114*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.00090*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.00141*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00210*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.00021*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.00015*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.00062*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.00110*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s2i pulse on channel 1 at 6203ns, square, 4ns, (0.08935*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.08932*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.08935*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.08931*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.08934*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.08930*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.08916*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.08934*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.08935*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.08934*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.08929*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s2q pulse on channel 2 at 6203ns, square, 4ns, (0.00239*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.00034*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.00268*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.00105*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.00313*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00578*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.00100*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.00028*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.00110*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.00341*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s3i pulse on channel 1 at 6207ns, square, 4ns, (0.12424*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.12418*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.12422*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.12415*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.12424*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.12415*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.12385*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.12423*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.12424*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.12423*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.12412*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s3q pulse on channel 2 at 6207ns, square, 4ns, (0.00406*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.00217*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.00473*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.00090*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.00474*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00990*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.00203*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.00129*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.00154*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.00546*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s4i pulse on channel 1 at 6211ns, square, 4ns, (0.15972*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.15961*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.15962*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.15955*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.15969*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.15961*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.15914*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.15968*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.15968*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.15970*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.15954*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s4q pulse on channel 2 at 6211ns, square, 4ns, (0.00596*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.00569*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.00738*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.00302*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.00592*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.01365*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.00363*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.00344*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.00243*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.00758*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s5i pulse on channel 1 at 6215ns, square, 4ns, (0.19550*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.19543*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.19532*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.19518*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.19534*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.19535*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.19464*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.19543*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.19548*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.19549*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.19517*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s5q pulse on channel 2 at 6215ns, square, 4ns, (0.00524*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.00843*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.01107*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.00795*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.00760*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.01826*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.00497*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.00263*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.00185*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.01129*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s6i pulse on channel 1 at 6219ns, square, 4ns, (0.23251*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.23243*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.23226*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.23202*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.23212*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.23225*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.23123*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.23241*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.23250*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.23251*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.23195*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s6q pulse on channel 2 at 6219ns, square, 4ns, (0.00598*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.01074*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.01508*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.01332*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.01090*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.02434*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.00663*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.00104*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.00021*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.01600*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s7i pulse on channel 1 at 6223ns, square, 4ns, (0.26975*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.26956*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.26932*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.26900*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.26903*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.26944*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.26799*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.26961*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.26975*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.26975*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.26878*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s7q pulse on channel 2 at 6223ns, square, 4ns, (0.01024*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.01518*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.02016*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.01971*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.01282*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.03079*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.00872*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.00124*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.00145*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.02286*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s8i pulse on channel 1 at 6227ns, square, 4ns, (0.30677*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.30630*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.30629*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.30581*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.30562*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.30643*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.30440*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.30659*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.30676*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.30674*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.30511*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s8q pulse on channel 2 at 6227ns, square, 4ns, (0.01693*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.01702*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.02420*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.02649*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.01436*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.03804*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.01022*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.00059*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.00375*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.03178*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s9i pulse on channel 1 at 6231ns, square, 4ns, (0.34401*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.34337*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.34361*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.34274*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.34242*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.34377*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.34097*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.34384*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.34401*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.34391*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.34131*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s9q pulse on channel 2 at 6231ns, square, 4ns, (0.02093*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.01666*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.02950*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.03301*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.01278*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.04561*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.01092*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.00086*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.00824*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.04302*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s10i pulse on channel 1 at 6235ns, square, 4ns, (0.37926*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.37864*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.37903*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.37776*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.37702*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.37907*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.37564*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.37913*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.37925*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.37913*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.37519*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s10q pulse on channel 2 at 6235ns, square, 4ns, (0.02169*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.01304*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.03361*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.04108*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.01194*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.05221*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.00992*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.00136*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.00972*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.05538*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s11i pulse on channel 1 at 6239ns, square, 4ns, (0.41152*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.41088*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.41143*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.41018*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.40884*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.41141*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.40679*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.41134*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.41149*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.41144*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.40577*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s11q pulse on channel 2 at 6239ns, square, 4ns, (0.02289*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.00840*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.03323*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.04685*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.00954*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.06222*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.01228*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.00491*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.00826*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.06854*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s12i pulse on channel 1 at 6243ns, square, 4ns, (0.43951*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.43860*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.43951*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.43858*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.43741*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.43940*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.43390*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.43917*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.43945*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.43945*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.43282*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s12q pulse on channel 2 at 6243ns, square, 4ns, (0.02830*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.00172*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.02861*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.04291*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.00990*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.07001*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.01734*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.00737*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.00731*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.07639*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s13i pulse on channel 1 at 6247ns, square, 4ns, (0.46059*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.45894*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.46057*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.46006*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.45931*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.46050*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.45473*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.46022*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.46054*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.46046*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.45410*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s13q pulse on channel 2 at 6247ns, square, 4ns, (0.03896*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.00448*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.02221*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.03430*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.00905*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.07324*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.01854*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.00670*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.01083*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.07704*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s14i pulse on channel 1 at 6251ns, square, 4ns, (0.47299*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.47006*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.47272*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.47278*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.47211*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.47295*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.46672*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.47251*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.47295*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.47276*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.46735*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s14q pulse on channel 2 at 6251ns, square, 4ns, (0.05257*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.01618*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.01424*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.02896*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.00608*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.07679*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.02145*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.00659*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.01472*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.07285*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s15i pulse on channel 1 at 6255ns, square, 4ns, (0.47554*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.47105*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.47466*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.47553*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.47474*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.47554*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.46808*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.47491*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.47548*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.47523*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.47083*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s15q pulse on channel 2 at 6255ns, square, 4ns, (0.06520*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.02883*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.00340*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.02762*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.00051*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.08389*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.02450*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.00757*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.01700*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.06674*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s16i pulse on channel 1 at 6259ns, square, 4ns, (0.46797*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.46190*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.46636*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.46790*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.46740*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.46797*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.45853*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.46733*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.46785*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.46734*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.46381*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s16q pulse on channel 2 at 6259ns, square, 4ns, (0.07510*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.03874*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.00811*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.02305*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.00154*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.09353*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.02446*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.01069*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.02432*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.06222*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s17i pulse on channel 1 at 6263ns, square, 4ns, (0.45103*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.44314*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.44880*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.45076*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.45064*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.45103*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.43995*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.45032*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.45068*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.44986*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.44742*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s17q pulse on channel 2 at 6263ns, square, 4ns, (0.08400*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.04484*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.01568*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.01895*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.00249*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.09939*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.02540*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.01784*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.03250*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.05697*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s18i pulse on channel 1 at 6267ns, square, 4ns, (0.42626*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.41714*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.42328*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.42584*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.42597*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.42621*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.41552*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.42555*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.42563*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.42453*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.42335*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s18q pulse on channel 2 at 6267ns, square, 4ns, (0.08769*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.05028*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.01893*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.01559*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.00681*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.09510*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.02456*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.02313*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.03835*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.04969*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s19i pulse on channel 1 at 6271ns, square, 4ns, (0.39569*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.38559*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.39173*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.39526*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.39552*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.39554*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.38564*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.39508*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.39472*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.39408*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.39330*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s19q pulse on channel 2 at 6271ns, square, 4ns, (0.08886*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.05590*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.01863*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.01162*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.01118*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.08863*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.02201*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.02771*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.03571*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.04345*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s20i pulse on channel 1 at 6275ns, square, 4ns, (0.36197*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.35112*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.35702*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.36131*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.36187*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.36164*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.35328*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.36137*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.36075*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.36010*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.36024*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s20q pulse on channel 2 at 6275ns, square, 4ns, (0.08793*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.05961*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.02184*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.00818*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.01535*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.07881*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.02068*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.02968*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.03665*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.03532*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s21i pulse on channel 1 at 6279ns, square, 4ns, (0.32556*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.31478*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.31967*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.32432*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.32549*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.32487*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.31858*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.32490*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.32378*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.32315*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.32429*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s21q pulse on channel 2 at 6279ns, square, 4ns, (0.08308*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.06164*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.02837*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.00698*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.02115*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.06707*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.02067*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.03400*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.03953*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.02874*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s22i pulse on channel 1 at 6283ns, square, 4ns, (0.28823*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.27876*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.28179*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.28622*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.28814*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.28702*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.28299*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.28731*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.28591*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.28540*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.28725*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s22q pulse on channel 2 at 6283ns, square, 4ns, (0.07325*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.06055*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.03399*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.00688*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.02633*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.05471*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.02296*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.03648*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.04025*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.02371*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s23i pulse on channel 1 at 6287ns, square, 4ns, (0.25115*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.24294*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.24469*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.24859*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.25111*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.24954*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.24692*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.25009*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.24846*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.24802*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.25061*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s23q pulse on channel 2 at 6287ns, square, 4ns, (0.06367*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.05658*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.03579*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.00465*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.02840*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.04588*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.02306*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.03662*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.03952*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.01637*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s24i pulse on channel 1 at 6291ns, square, 4ns, (0.21389*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.20678*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.20780*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.21074*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.21385*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.21194*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.21071*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.21276*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.21086*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.21052*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.21367*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s24q pulse on channel 2 at 6291ns, square, 4ns, (0.05468*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.05066*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.03655*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.00391*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.02878*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.03672*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.02191*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.03584*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.03779*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.00967*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s25i pulse on channel 1 at 6295ns, square, 4ns, (0.17759*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.17137*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.17248*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.17416*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.17757*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.17538*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.17523*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.17651*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.17456*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.17374*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.17753*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s25q pulse on channel 2 at 6295ns, square, 4ns, (0.04659*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.04230*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.03474*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.00269*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.02793*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.02883*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.01952*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.03268*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.03674*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.00444*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s26i pulse on channel 1 at 6299ns, square, 4ns, (0.14193*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.13662*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.13789*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.13884*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.14192*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.13973*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.14015*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.14096*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.13936*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.13811*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.14192*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s26q pulse on channel 2 at 6299ns, square, 4ns, (0.03845*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.03363*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.02946*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.00186*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.02491*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.02241*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.01659*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.02690*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.03270*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.00193*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s27i pulse on channel 1 at 6303ns, square, 4ns, (0.10667*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.10236*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.10353*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.10429*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.10666*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.10490*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.10551*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.10588*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.10458*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.10320*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.10667*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s27q pulse on channel 2 at 6303ns, square, 4ns, (0.03002*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.02571*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.02240*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.00138*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.01935*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.01569*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.01301*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.02100*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.02699*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.00001*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s28i pulse on channel 1 at 6307ns, square, 4ns, (0.07218*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.06886*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.06998*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.07051*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.07215*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.07089*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.07149*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.07166*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.07062*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.06956*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.07217*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s28q pulse on channel 2 at 6307ns, square, 4ns, (0.02163*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.01770*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.01542*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.00189*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.01360*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00998*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.00868*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.01493*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.01927*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.00101*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s29i pulse on channel 1 at 6311ns, square, 4ns, (0.03805*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.03612*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.03682*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.03713*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.03801*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.03730*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.03768*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.03781*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.03715*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.03647*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.03804*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s29q pulse on channel 2 at 6311ns, square, 4ns, (0.01196*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.00958*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.00830*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.00158*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.00752*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00524*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.00420*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.00822*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.01083*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.00088*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s30i pulse on channel 1 at 6315ns, square, 4ns, (0.00421*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.00399*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.00406*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.00411*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.00420*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.00413*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00417*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.00419*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.00410*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.00401*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.00421*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s30q pulse on channel 2 at 6315ns, square, 4ns, (0.00135*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.00111*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.00090*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.00024*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.00083*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00057*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.00041*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.00094*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.00128*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.00012*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s31i pulse on channel 1 at 6319ns, square, 4ns, (-0.02956*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.02807*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.02844*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.02889*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.02950*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.02902*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.02926*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.02946*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.02877*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.02795*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.02955*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s31q pulse on channel 2 at 6319ns, square, 4ns, (-0.00926*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.00806*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.00624*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.00192*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.00562*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00417*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.00242*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.00678*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.00963*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.00089*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s32i pulse on channel 1 at 6323ns, square, 4ns, (-0.06362*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.06061*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.06077*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.06205*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.06345*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.06253*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.06293*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.06348*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.06180*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.05970*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.06360*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s32q pulse on channel 2 at 6323ns, square, 4ns, (-0.01933*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.01883*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.01404*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.00466*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.01171*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00936*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.00415*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.01512*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.02199*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.00163*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s33i pulse on channel 1 at 6327ns, square, 4ns, (-0.09796*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.09401*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.09309*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.09524*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.09765*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.09634*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.09697*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.09784*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.09501*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.09138*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.09794*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s33q pulse on channel 2 at 6327ns, square, 4ns, (-0.02754*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.03049*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.02293*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.00774*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.01772*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.01388*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.00494*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.02384*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.03530*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.00192*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s34i pulse on channel 1 at 6331ns, square, 4ns, (-0.13307*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.12847*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.12606*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.12910*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.13243*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.13091*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.13197*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.13301*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.12936*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.12321*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.13307*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s34q pulse on channel 2 at 6331ns, square, 4ns, (-0.03471*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.04262*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.03227*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.01306*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.02391*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.01711*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.00395*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.03123*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.05028*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.00077*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s35i pulse on channel 1 at 6335ns, square, 4ns, (-0.16865*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.16338*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.15915*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.16347*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.16745*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.16621*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.16767*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.16865*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.16406*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.15548*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.16863*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s35q pulse on channel 2 at 6335ns, square, 4ns, (-0.04182*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.05581*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.04147*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.02003*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.02858*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.01808*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.00007*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.03906*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.06534*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.00247*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s36i pulse on channel 1 at 6339ns, square, 4ns, (-0.20461*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.19835*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.19281*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.19832*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.20267*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.20204*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.20388*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.20456*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.19907*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.18777*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.20450*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s36q pulse on channel 2 at 6339ns, square, 4ns, (-0.05023*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.06846*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.05034*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.02806*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.03229*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.01723*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.00450*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.04726*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.08129*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.00652*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s37i pulse on channel 1 at 6343ns, square, 4ns, (-0.24183*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.23475*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.22725*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.23405*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.23932*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.23929*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.24136*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.24165*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.23557*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.22100*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.24150*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s37q pulse on channel 2 at 6343ns, square, 4ns, (-0.05809*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.08269*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.06084*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.03474*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.03494*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.01496*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.00922*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.05465*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.09819*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.01258*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s38i pulse on channel 1 at 6347ns, square, 4ns, (-0.27901*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.27127*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.26076*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.26966*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.27588*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.27632*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.27882*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.27872*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.27199*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.25374*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.27816*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s38q pulse on channel 2 at 6347ns, square, 4ns, (-0.06525*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.09926*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.07162*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.04167*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.03862*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.01031*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.01278*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.06218*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.11602*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.02179*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s39i pulse on channel 1 at 6351ns, square, 4ns, (-0.31620*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.30802*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.29408*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.30455*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.31271*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.31303*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.31618*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.31569*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.30786*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.28617*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.31453*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s39q pulse on channel 2 at 6351ns, square, 4ns, (-0.07143*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.11617*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.08502*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.04686*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.04468*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00327*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.01789*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.07215*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.13449*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.03246*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s40i pulse on channel 1 at 6355ns, square, 4ns, (-0.35306*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.34541*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.32539*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.33942*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.34894*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.34944*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.35306*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.35236*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.34310*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.31958*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.35006*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s40q pulse on channel 2 at 6355ns, square, 4ns, (-0.07309*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.13702*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.09717*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.05379*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.05040*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00139*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.02229*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.08326*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.15007*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.04591*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s41i pulse on channel 1 at 6359ns, square, 4ns, (-0.38759*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.38065*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.35538*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.37314*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.38253*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.38409*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.38756*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.38695*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.37592*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.35193*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.38333*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s41q pulse on channel 2 at 6359ns, square, 4ns, (-0.07303*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.15470*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.10487*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.06247*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.05200*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00545*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.02227*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.09440*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.16239*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.05733*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s42i pulse on channel 1 at 6363ns, square, 4ns, (-0.41906*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.41308*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.38291*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.40314*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.41234*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.41550*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.41893*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.41854*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.40628*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.38235*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.41349*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s42q pulse on channel 2 at 6363ns, square, 4ns, (-0.07054*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.17026*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.11440*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.07475*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.05450*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.01040*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.02089*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.10272*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.17151*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.06813*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s43i pulse on channel 1 at 6367ns, square, 4ns, (-0.44551*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.44035*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.40568*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.42699*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.43649*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.44185*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.44526*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.44508*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.43267*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.40558*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.43867*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s43q pulse on channel 2 at 6367ns, square, 4ns, (-0.06759*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.18411*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.12710*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.08916*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.05694*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.01484*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.01956*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.10616*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.18433*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.07772*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s44i pulse on channel 1 at 6371ns, square, 4ns, (-0.46456*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.46023*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.42277*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.44328*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.45371*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.46093*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.46430*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.46423*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.45257*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.42251*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.45765*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s44q pulse on channel 2 at 6371ns, square, 4ns, (-0.06328*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.19258*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.13902*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.09981*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.05799*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.01574*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.01758*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.10489*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.19314*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.07985*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s45i pulse on channel 1 at 6375ns, square, 4ns, (-0.47458*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.47134*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.43197*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.45124*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.46233*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.47088*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.47434*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.47428*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.46350*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.43126*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.46787*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s45q pulse on channel 2 at 6375ns, square, 4ns, (-0.05540*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.19655*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.14699*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.10714*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.05914*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.01526*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.01701*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.10195*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.19810*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.07956*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s46i pulse on channel 1 at 6379ns, square, 4ns, (-0.47458*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.47208*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.43356*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.45003*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.46042*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.47089*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.47439*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.47433*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.46435*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.43119*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.46840*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s46q pulse on channel 2 at 6379ns, square, 4ns, (-0.04867*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.19302*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.15068*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.11507*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.05911*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.01348*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.01547*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.09803*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.19825*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.07632*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s47i pulse on channel 1 at 6383ns, square, 4ns, (-0.46456*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.46222*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.42456*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.43976*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.44890*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.46012*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.46438*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.46428*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.45452*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.42248*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.45891*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s47q pulse on channel 2 at 6383ns, square, 4ns, (-0.04658*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.18859*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.14977*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.11963*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.06411*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.01298*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.01637*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.09609*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.19321*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.07224*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s48i pulse on channel 1 at 6387ns, square, 4ns, (-0.44551*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.44337*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.40579*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.42109*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.42793*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.44032*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.44532*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.44517*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.43551*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.40643*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.44051*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s48q pulse on channel 2 at 6387ns, square, 4ns, (-0.04357*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.18387*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.14547*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.12390*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.06777*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.01296*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.01738*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.09382*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.18244*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.06651*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s49i pulse on channel 1 at 6391ns, square, 4ns, (-0.41906*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.41713*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.37875*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.39553*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.40043*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.41236*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.41886*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.41880*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.40912*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.38207*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.41460*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s49q pulse on channel 2 at 6391ns, square, 4ns, (-0.04021*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.17933*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.13844*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.12356*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.07461*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.01310*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.01480*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.09072*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.17213*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.06098*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s50i pulse on channel 1 at 6395ns, square, 4ns, (-0.38759*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.38603*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.34799*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.36513*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.36982*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.38081*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.38747*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.38717*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.37832*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.35307*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.38306*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s50q pulse on channel 2 at 6395ns, square, 4ns, (-0.03482*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.17068*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.13003*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.11603*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.07220*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00981*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.01812*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.08427*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.15991*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.05911*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s51i pulse on channel 1 at 6399ns, square, 4ns, (-0.35306*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.35186*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.31538*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.33291*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.33726*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.34640*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.35300*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.35222*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.34412*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.32101*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.34905*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s51q pulse on channel 2 at 6399ns, square, 4ns, (-0.02905*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.15869*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.11757*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.10444*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.06827*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00637*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.02437*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.07893*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.14699*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.05308*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s52i pulse on channel 1 at 6403ns, square, 4ns, (-0.31620*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.31514*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.28089*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.29928*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.30271*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.31008*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.31609*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.31473*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.30836*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.28697*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.31269*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s52q pulse on channel 2 at 6403ns, square, 4ns, (-0.02579*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.14518*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.10204*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.09136*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.06189*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00807*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.03038*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.06997*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.13277*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.04696*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s53i pulse on channel 1 at 6407ns, square, 4ns, (-0.27901*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.27802*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.24573*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.26526*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.26771*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.27346*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.27890*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.27706*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.27226*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.25233*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.27598*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s53q pulse on channel 2 at 6407ns, square, 4ns, (-0.02342*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.13216*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.08649*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.07859*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.05539*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00796*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.03289*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.06098*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.11906*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.04098*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s54i pulse on channel 1 at 6411ns, square, 4ns, (-0.24183*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.24092*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.21189*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.23113*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.23192*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.23669*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.24173*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.23986*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.23569*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.21895*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.23951*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s54q pulse on channel 2 at 6411ns, square, 4ns, (-0.02094*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.11655*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.07113*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.06852*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.04959*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00678*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.03080*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.05415*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.10268*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.03343*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s55i pulse on channel 1 at 6415ns, square, 4ns, (-0.20461*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.20368*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.17853*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.19636*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.19637*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.19996*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.20455*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.20271*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.19943*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.18578*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.20274*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s55q pulse on channel 2 at 6415ns, square, 4ns, (-0.01949*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.09995*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.05749*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.05746*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.04334*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00473*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.02781*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.04572*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.08572*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.02757*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s56i pulse on channel 1 at 6419ns, square, 4ns, (-0.16865*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.16779*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.14586*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.16211*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.16194*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.16486*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.16863*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.16690*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.16411*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.15384*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.16719*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s56q pulse on channel 2 at 6419ns, square, 4ns, (-0.01696*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.08466*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.04649*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.04710*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.03552*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00268*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.02424*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.03885*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.06911*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.02210*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s57i pulse on channel 1 at 6423ns, square, 4ns, (-0.13307*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.13241*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.11410*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.12824*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.12810*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.13004*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.13306*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.13152*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.12911*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.12181*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.13175*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s57q pulse on channel 2 at 6423ns, square, 4ns, (-0.01332*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.06848*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.03555*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.03604*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.02827*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00155*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.02030*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.03224*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.05358*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.01875*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s58i pulse on channel 1 at 6427ns, square, 4ns, (-0.09796*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.09751*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.08292*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.09475*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.09455*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.09561*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.09796*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.09665*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.09480*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.08974*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.09694*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s58q pulse on channel 2 at 6427ns, square, 4ns, (-0.00934*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.05215*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.02487*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.02563*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.02133*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00098*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.01599*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.02467*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.03928*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.01411*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s59i pulse on channel 1 at 6431ns, square, 4ns, (-0.06362*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.06336*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.05320*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.06180*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.06147*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.06198*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.06361*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.06259*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.06150*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.05799*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.06298*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s59q pulse on channel 2 at 6431ns, square, 4ns, (-0.00578*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.03488*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.01512*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.01639*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.01433*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00078*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.01138*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.01627*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.02616*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.00900*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s60i pulse on channel 1 at 6435ns, square, 4ns, (-0.02956*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.02946*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.02449*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.02875*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.02860*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.02878*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.02955*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.02907*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.02855*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.02678*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.02928*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s60q pulse on channel 2 at 6435ns, square, 4ns, (-0.00246*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.01655*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.00688*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.00745*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.00675*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00050*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.00536*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.00764*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.01252*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.00404*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s61i pulse on channel 1 at 6439ns, square, 4ns, (0.00421*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.00420*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.00343*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.00410*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.00408*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.00410*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00421*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.00414*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.00406*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.00379*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.00417*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s61q pulse on channel 2 at 6439ns, square, 4ns, (0.00031*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.00244*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.00097*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.00102*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.00097*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00007*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.00076*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.00112*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.00182*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.00055*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s62i pulse on channel 1 at 6443ns, square, 4ns, (0.03805*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.03798*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.03061*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.03702*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.03695*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.03696*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.03804*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.03741*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.03663*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.03409*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.03777*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s62q pulse on channel 2 at 6443ns, square, 4ns, (0.00226*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.02260*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.00879*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.00906*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.00904*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00056*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.00691*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.01027*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.01688*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.00454*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s63i pulse on channel 1 at 6447ns, square, 4ns, (0.07218*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.07211*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.05754*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.07026*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.07015*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.06989*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.07218*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.07092*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.06925*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.06449*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.07182*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s63q pulse on channel 2 at 6447ns, square, 4ns, (0.00317*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.04357*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.01655*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.01700*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.01805*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00046*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.01340*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.02034*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.03241*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.00714*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s64i pulse on channel 1 at 6451ns, square, 4ns, (0.10667*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.10661*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.08460*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.10387*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.10376*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.10302*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.10667*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.10496*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.10191*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.09522*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.10636*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s64q pulse on channel 2 at 6451ns, square, 4ns, (0.00367*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.06498*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.02431*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.02476*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.02767*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00032*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.01901*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.03151*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.04809*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.00817*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s65i pulse on channel 1 at 6455ns, square, 4ns, (0.14193*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.14185*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.11210*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.13814*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.13790*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.13719*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.14193*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.13976*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.13496*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.12673*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.14169*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s65q pulse on channel 2 at 6455ns, square, 4ns, (0.00495*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.08705*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.03261*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.03359*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.03636*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00081*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.02472*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.04394*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.06390*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.00829*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s66i pulse on channel 1 at 6459ns, square, 4ns, (0.17759*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.17746*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.14065*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.17248*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.17225*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.17186*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.17758*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.17466*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.16828*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.15880*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.17739*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s66q pulse on channel 2 at 6459ns, square, 4ns, (0.00676*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.10842*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.04228*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.04320*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.04472*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00097*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.03209*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.05675*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.07950*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.00838*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s67i pulse on channel 1 at 6463ns, square, 4ns, (0.21389*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.21370*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.17160*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.20718*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.20707*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.20702*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.21389*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.21041*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.20216*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.19163*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.21373*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s67q pulse on channel 2 at 6463ns, square, 4ns, (0.00900*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.12768*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.05316*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.05358*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.05376*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00033*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.03843*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.06985*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.09500*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.00825*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s68i pulse on channel 1 at 6467ns, square, 4ns, (0.25115*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.25093*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.20281*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.24315*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.24183*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.24302*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.25115*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.24737*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.23649*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.22560*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.25100*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s68q pulse on channel 2 at 6467ns, square, 4ns, (0.01044*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.14814*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.06287*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.06778*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.06338*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00035*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.04342*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.08453*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.11037*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.00874*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s69i pulse on channel 1 at 6471ns, square, 4ns, (0.28823*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.28803*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.23322*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.27958*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.27618*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.27923*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.28823*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.28434*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.27053*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.25914*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.28811*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s69q pulse on channel 2 at 6471ns, square, 4ns, (0.01054*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.16936*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.07007*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.08246*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.07144*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00085*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.04716*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.09943*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.12617*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.00825*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s70i pulse on channel 1 at 6475ns, square, 4ns, (0.32556*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.32535*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.26506*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.31640*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.31095*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.31593*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.32556*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.32183*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.30411*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.29448*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.32542*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s70q pulse on channel 2 at 6475ns, square, 4ns, (0.01169*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.18904*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.07668*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.09642*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.07860*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00003*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.04914*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.11621*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.13882*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.00939*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s71i pulse on channel 1 at 6479ns, square, 4ns, (0.36197*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.36166*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.29886*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.35166*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.34569*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.35127*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.36196*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.35858*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.33736*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.33061*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.36184*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s71q pulse on channel 2 at 6479ns, square, 4ns, (0.01488*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.20421*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.08574*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.10730*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.08735*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00157*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.04940*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.13119*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.14736*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.00962*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s72i pulse on channel 1 at 6483ns, square, 4ns, (0.39569*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.39535*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.33087*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.38364*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.37810*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.38341*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.39569*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.39266*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.36972*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.36341*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.39548*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s72q pulse on channel 2 at 6483ns, square, 4ns, (0.01648*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.21702*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.09692*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.11668*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.09785*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00271*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.04888*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.14099*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.15656*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.01313*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s73i pulse on channel 1 at 6487ns, square, 4ns, (0.42626*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.42600*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.35903*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.41314*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.40779*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.41309*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.42626*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.42355*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.39872*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.39349*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.42598*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s73q pulse on channel 2 at 6487ns, square, 4ns, (0.01475*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.22977*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.10495*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.12412*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.10514*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00006*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.04800*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.15073*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.16389*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.01546*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s74i pulse on channel 1 at 6491ns, square, 4ns, (0.45103*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.45095*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.38327*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.43614*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.43249*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.43773*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.45103*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.44879*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.42203*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.41795*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.45070*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s74q pulse on channel 2 at 6491ns, square, 4ns, (0.00889*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.23778*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.11494*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.12801*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.10876*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00066*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.04495*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.15913*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.16956*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.01726*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s75i pulse on channel 1 at 6495ns, square, 4ns, (0.46797*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.46796*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.40061*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.45144*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.45060*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.45419*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.46797*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.46655*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.43719*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.43547*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.46771*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s75q pulse on channel 2 at 6495ns, square, 4ns, (0.00223*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.24188*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.12326*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.12633*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.11270*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00094*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.03641*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.16692*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.17136*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.01552*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s76i pulse on channel 1 at 6499ns, square, 4ns, (0.47554*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.47550*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.40912*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.45742*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.45911*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.46127*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.47553*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.47442*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.44295*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.44571*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.47529*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s76q pulse on channel 2 at 6499ns, square, 4ns, (-0.00587*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.24240*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.13000*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.12390*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.11561*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00259*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.03255*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.17302*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.16577*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.01549*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s77i pulse on channel 1 at 6503ns, square, 4ns, (0.47299*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.47274*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.41102*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.45424*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.45657*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.45863*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.47295*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.47202*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.43984*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.44555*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.47282*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s77q pulse on channel 2 at 6503ns, square, 4ns, (-0.01561*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.23407*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.13187*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.12356*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.11566*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00616*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.03032*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.17395*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.15878*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.01300*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s78i pulse on channel 1 at 6507ns, square, 4ns, (0.46059*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.45986*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.40364*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.44233*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.44414*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.44699*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.46052*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.45983*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.42769*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.43425*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.46043*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s78q pulse on channel 2 at 6507ns, square, 4ns, (-0.02588*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.22185*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.12840*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.12199*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.11111*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00819*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.02640*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.17096*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.15352*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.01207*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s79i pulse on channel 1 at 6511ns, square, 4ns, (0.43951*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.43837*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.38741*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.42163*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.42383*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.42716*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.43942*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.43899*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.40659*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.41423*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.43949*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s79q pulse on channel 2 at 6511ns, square, 4ns, (-0.03165*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.20757*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.12410*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.11636*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.10347*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00892*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.02135*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.16690*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.14691*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.00420*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s80i pulse on channel 1 at 6515ns, square, 4ns, (0.41152*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.41024*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.36342*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.39435*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.39725*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.40103*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.41144*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.41133*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.37910*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.38817*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.41149*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s80q pulse on channel 2 at 6515ns, square, 4ns, (-0.03237*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.19307*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.11764*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.10742*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.09233*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00822*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.01260*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.16009*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.13666*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.00513*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s81i pulse on channel 1 at 6519ns, square, 4ns, (0.37926*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.37792*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.33370*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.36333*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.36577*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.37042*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.37919*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.37918*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.34764*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.35740*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.37918*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s81q pulse on channel 2 at 6519ns, square, 4ns, (-0.03182*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.18022*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.10874*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.10022*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.08137*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00682*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.00773*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.15159*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.12688*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.00760*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s82i pulse on channel 1 at 6523ns, square, 4ns, (0.34401*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.34261*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.30282*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.32850*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.33176*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.33648*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.34400*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.34393*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.31431*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.32290*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.34396*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s82q pulse on channel 2 at 6523ns, square, 4ns, (-0.03105*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.16324*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.10212*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.09098*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.07160*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00205*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.00753*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.13983*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.11865*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.00597*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s83i pulse on channel 1 at 6527ns, square, 4ns, (0.30677*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.30543*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.27053*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.29097*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.29621*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.30003*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.30675*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.30669*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.27945*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.28727*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.30674*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s83q pulse on channel 2 at 6527ns, square, 4ns, (-0.02858*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.14463*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.09715*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.07976*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.06392*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00260*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.00676*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.12654*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.10761*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.00400*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s84i pulse on channel 1 at 6531ns, square, 4ns, (0.26975*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.26859*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.23728*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.25466*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.25986*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.26374*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.26966*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.26969*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.24476*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.25222*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.26972*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s84q pulse on channel 2 at 6531ns, square, 4ns, (-0.02501*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.12832*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.08895*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.07238*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.05662*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00710*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.00584*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.11339*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.09566*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.00371*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s85i pulse on channel 1 at 6535ns, square, 4ns, (0.23251*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.23148*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.20268*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.21876*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.22331*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.22731*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.23227*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.23244*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.21121*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.21711*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.23247*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s85q pulse on channel 2 at 6535ns, square, 4ns, (-0.02186*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.11392*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.07876*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.06475*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.04889*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.01055*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.00562*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.09720*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.08320*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.00377*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s86i pulse on channel 1 at 6539ns, square, 4ns, (0.19550*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.19467*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.16782*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.18328*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.18726*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.19067*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.19515*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.19543*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.17721*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.18257*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.19547*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s86q pulse on channel 2 at 6539ns, square, 4ns, (-0.01797*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.10028*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.06802*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.05615*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.04318*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.01163*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.00532*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.08256*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.06991*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.00304*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s87i pulse on channel 1 at 6543ns, square, 4ns, (0.15972*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.15923*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.13542*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.14969*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.15312*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.15537*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.15940*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.15965*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.14415*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.14886*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.15970*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s87q pulse on channel 2 at 6543ns, square, 4ns, (-0.01257*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.08468*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.05571*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.04543*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.03702*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.01010*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.00469*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.06879*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.05788*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.00247*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s88i pulse on channel 1 at 6547ns, square, 4ns, (0.12424*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.12397*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.10408*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.11632*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.11939*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.12042*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.12397*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.12417*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.11127*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.11540*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.12424*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s88q pulse on channel 2 at 6547ns, square, 4ns, (-0.00822*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.06784*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.04366*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.03438*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.03057*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00828*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.00422*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.05528*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.04603*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.00060*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s89i pulse on channel 1 at 6551ns, square, 4ns, (0.08935*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.08920*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.07384*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.08333*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.08603*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.08623*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.08919*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.08928*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.07954*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.08262*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.08935*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s89q pulse on channel 2 at 6551ns, square, 4ns, (-0.00525*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.05031*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.03224*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.02412*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.02342*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00543*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.00346*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.04071*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.03401*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.00025*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s90i pulse on channel 1 at 6555ns, square, 4ns, (0.05508*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.05502*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.04504*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.05140*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.05311*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.05299*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.05500*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.05500*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.04854*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.05067*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.05507*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s90q pulse on channel 2 at 6555ns, square, 4ns, (-0.00248*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.03169*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.01978*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.01458*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.01502*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00292*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.00294*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.02602*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.02158*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.00070*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s91i pulse on channel 1 at 6559ns, square, 4ns, (0.02109*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.02107*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.01719*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.01971*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.02037*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.02018*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.02107*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.02104*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.01848*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.01929*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.02108*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s91q pulse on channel 2 at 6559ns, square, 4ns, (-0.00091*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.01222*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.00750*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.00546*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.00612*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00098*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.00138*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.01017*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.00853*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.00052*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s92i pulse on channel 1 at 6563ns, square, 4ns, (-0.01264*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.01263*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.01023*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.01182*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.01222*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.01205*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.01263*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.01261*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.01107*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.01149*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.01263*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s92q pulse on channel 2 at 6563ns, square, 4ns, (0.00048*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.00743*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.00449*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.00321*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.00381*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00052*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.00090*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.00610*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.00526*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.00047*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s93i pulse on channel 1 at 6567ns, square, 4ns, (-0.04655*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.04653*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.03753*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.04345*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.04513*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.04416*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.04653*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.04645*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.04068*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.04227*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.04651*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s93q pulse on channel 2 at 6567ns, square, 4ns, (0.00142*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.02754*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.01672*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.01143*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.01474*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00156*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.00308*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.02264*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.01951*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.00197*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s94i pulse on channel 1 at 6571ns, square, 4ns, (-0.08076*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.08075*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.06504*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.07545*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.07850*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.07592*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.08073*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.08062*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.07011*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.07299*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.08066*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s94q pulse on channel 2 at 6571ns, square, 4ns, (0.00102*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.04787*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.02878*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.01894*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.02752*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00217*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.00474*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.04008*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.03456*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.00390*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s95i pulse on channel 1 at 6575ns, square, 4ns, (-0.11544*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.11544*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.09287*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.10794*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.11247*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.10744*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.11543*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.11529*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.09965*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.10367*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.11529*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s95q pulse on channel 2 at 6575ns, square, 4ns, (-0.00046*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.06857*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.04094*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.02601*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.04224*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00170*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.00586*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.05828*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.05078*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.00593*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s96i pulse on channel 1 at 6579ns, square, 4ns, (-0.15082*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.15080*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.12162*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.14173*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.14728*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.13890*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.15081*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.15068*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.12936*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.13489*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.15069*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s96q pulse on channel 2 at 6579ns, square, 4ns, (-0.00205*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.08919*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.05156*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.03247*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.05876*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00091*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.00627*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.07753*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.06746*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.00624*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s97i pulse on channel 1 at 6583ns, square, 4ns, (-0.18654*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.18648*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.15034*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.17550*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.18268*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.17027*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.18654*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.18644*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.15900*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.16643*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.18646*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s97q pulse on channel 2 at 6583ns, square, 4ns, (-0.00468*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.11043*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.06322*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.03773*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.07619*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00094*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.00608*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.09755*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.08424*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.00544*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s98i pulse on channel 1 at 6587ns, square, 4ns, (-0.22319*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.22304*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.17913*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.20984*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.21931*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.20256*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.22319*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.22311*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.18808*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.19816*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.22314*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s98q pulse on channel 2 at 6587ns, square, 4ns, (-0.00830*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.13313*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.07603*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.04142*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.09371*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00141*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.00602*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.12017*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.10269*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.00489*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s99i pulse on channel 1 at 6591ns, square, 4ns, (-0.26046*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.26008*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.20752*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.24385*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.25669*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.23575*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.26045*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.26035*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.21776*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.23189*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.26038*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s99q pulse on channel 2 at 6591ns, square, 4ns, (-0.01401*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.15739*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.09151*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.04415*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.11073*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00267*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.00752*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.14290*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.11859*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.00640*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s100i pulse on channel 1 at 6595ns, square, 4ns, (-0.29739*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.29666*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.23519*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.27816*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.29389*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.26803*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.29735*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.29723*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.24565*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.26716*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.29712*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s100q pulse on channel 2 at 6595ns, square, 4ns, (-0.02074*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.18201*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.10521*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.04547*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.12884*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00492*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.00959*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.16761*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.13064*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.01255*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s101i pulse on channel 1 at 6599ns, square, 4ns, (-0.33484*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.33350*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.26487*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.31257*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.33154*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.30073*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.33475*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.33467*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.27274*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.30398*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.33399*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s101q pulse on channel 2 at 6599ns, square, 4ns, (-0.02992*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.20484*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.12008*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.04685*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.14723*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00747*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.01072*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.19424*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.14040*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.02375*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s102i pulse on channel 1 at 6603ns, square, 4ns, (-0.37070*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.36906*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.29613*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.34408*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.36737*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.33238*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.37052*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.37057*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.29925*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.34013*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.36907*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s102q pulse on channel 2 at 6603ns, square, 4ns, (-0.03490*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.22300*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.13796*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.04965*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.16414*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.01158*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.01000*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.21880*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.14742*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.03472*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s103i pulse on channel 1 at 6607ns, square, 4ns, (-0.40367*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.40173*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.32629*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.37149*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.40094*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.36236*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.40336*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.40356*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.32436*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.37161*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.40173*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s103q pulse on channel 2 at 6607ns, square, 4ns, (-0.03946*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.23765*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.15793*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.04685*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.17789*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.01586*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.00927*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.24029*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.15765*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.03948*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s104i pulse on channel 1 at 6611ns, square, 4ns, (-0.43309*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.43071*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.35332*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.39343*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.43105*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.38859*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.43264*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.43295*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.34892*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.39948*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.43094*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s104q pulse on channel 2 at 6611ns, square, 4ns, (-0.04528*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.25045*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.18104*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.04196*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.19121*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.01968*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.01107*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.25656*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.16728*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.04312*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s105i pulse on channel 1 at 6615ns, square, 4ns, (-0.45607*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.45330*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.37758*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.40925*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.45488*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.40969*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.45528*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.45579*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.37009*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.42062*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.45353*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s105q pulse on channel 2 at 6615ns, square, 4ns, (-0.05024*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.25581*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.20129*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.03294*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.20038*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.02690*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.01596*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.26652*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.17630*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.04806*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s106i pulse on channel 1 at 6619ns, square, 4ns, (-0.47078*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.46756*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.39536*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.41872*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.47024*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.42232*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.46967*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.47024*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.38522*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.43351*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.46784*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s106q pulse on channel 2 at 6619ns, square, 4ns, (-0.05505*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.25559*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.21521*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.02272*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.20805*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.03242*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.02262*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.27063*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.18359*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.05255*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s107i pulse on channel 1 at 6623ns, square, 4ns, (-0.47586*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.47243*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.40481*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.41875*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.47571*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.42354*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.47443*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.47500*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.39155*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.43894*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.47149*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s107q pulse on channel 2 at 6623ns, square, 4ns, (-0.05699*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.25013*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.22603*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.01200*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.21691*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.03683*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.02857*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.27042*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.18376*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.06434*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s108i pulse on channel 1 at 6627ns, square, 4ns, (-0.47078*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.46720*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.40380*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.40875*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.47076*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.41547*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.46925*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.46936*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.38930*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.43488*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.46491*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s108q pulse on channel 2 at 6627ns, square, 4ns, (-0.05801*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.24205*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.23359*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.00496*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.22142*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.03799*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.03656*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.26474*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.18033*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.07413*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s109i pulse on channel 1 at 6631ns, square, 4ns, (-0.45607*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.45206*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.39313*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.38977*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.45606*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.39963*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.45462*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.45417*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.37888*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.42144*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.44847*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s109q pulse on channel 2 at 6631ns, square, 4ns, (-0.06036*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.23120*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.23681*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.00371*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.21977*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.03632*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.04165*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.25388*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.17433*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.08290*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s110i pulse on channel 1 at 6635ns, square, 4ns, (-0.43309*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.42877*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.37568*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.36399*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.43309*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.37488*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.43188*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.43093*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.36288*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.40001*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.42386*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s110q pulse on channel 2 at 6635ns, square, 4ns, (-0.06098*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.21547*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.23468*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.00076*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.21687*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.03237*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.04314*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.23640*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.16601*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.08890*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s111i pulse on channel 1 at 6639ns, square, 4ns, (-0.40367*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.39948*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.35186*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.33243*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.40366*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.34622*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.40267*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.40127*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.34271*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.37204*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.39277*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s111q pulse on channel 2 at 6639ns, square, 4ns, (-0.05797*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.19784*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.22899*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.00202*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.20756*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.02832*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.04395*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.21330*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.15663*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.09314*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s112i pulse on channel 1 at 6643ns, square, 4ns, (-0.37070*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.36682*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.32602*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.29954*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.37070*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.31607*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.36948*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.36822*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.31556*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.34212*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.35825*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s112q pulse on channel 2 at 6643ns, square, 4ns, (-0.05351*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.17644*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.21840*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.00240*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.19370*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.03012*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.04289*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.19453*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.14275*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.09529*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s113i pulse on channel 1 at 6647ns, square, 4ns, (-0.33484*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.33141*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.29790*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.26689*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.33483*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.28317*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.33338*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.33243*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.28517*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.30958*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.32124*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s113q pulse on channel 2 at 6647ns, square, 4ns, (-0.04778*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.15288*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.20220*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.00130*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.17869*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.03121*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.04012*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.17548*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.12758*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.09446*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s114i pulse on channel 1 at 6651ns, square, 4ns, (-0.29739*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.29446*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.26803*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.23370*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.29739*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.24975*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.29587*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.29520*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.25106*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.27542*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.28248*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s114q pulse on channel 2 at 6651ns, square, 4ns, (-0.04161*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.12883*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.18392*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.00080*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.16144*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.03001*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.03598*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.15941*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.11217*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.09298*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s115i pulse on channel 1 at 6655ns, square, 4ns, (-0.26046*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.25779*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.23895*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.20288*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.26044*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.21738*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.25903*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.25832*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.21687*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.24231*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.24481*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s115q pulse on channel 2 at 6655ns, square, 4ns, (-0.03721*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.10365*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.16334*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.00319*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.14348*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.02729*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.03330*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.14425*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.09552*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.08892*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s116i pulse on channel 1 at 6659ns, square, 4ns, (-0.22319*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.22032*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.20792*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.17248*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.22313*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.18551*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.22182*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.22148*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.18332*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.20816*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.20803*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s116q pulse on channel 2 at 6659ns, square, 4ns, (-0.03568*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.08114*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.14164*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.00521*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.12409*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.02468*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.02756*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.12731*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.08052*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.08087*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s117i pulse on channel 1 at 6663ns, square, 4ns, (-0.18654*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.18359*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.17668*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.14325*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.18645*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.15506*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.18543*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.18521*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.15172*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.17459*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.17249*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s117q pulse on channel 2 at 6663ns, square, 4ns, (-0.03303*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.05983*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.11948*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.00591*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.10370*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.02027*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.02224*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.10852*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.06569*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.07102*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s118i pulse on channel 1 at 6667ns, square, 4ns, (-0.15082*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.14799*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.14477*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.11530*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.15068*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.12526*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.15001*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.14986*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.12082*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.14115*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.13862*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s118q pulse on channel 2 at 6667ns, square, 4ns, (-0.02905*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.04228*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.09722*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.00633*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.08399*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.01558*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.01692*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.09026*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.05314*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.05941*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s119i pulse on channel 1 at 6671ns, square, 4ns, (-0.11544*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.11317*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.11186*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.08788*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.11527*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.09600*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.11486*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.11480*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.09220*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.10776*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.10570*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s119q pulse on channel 2 at 6671ns, square, 4ns, (-0.02279*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.02853*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.07486*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.00635*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.06411*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.01152*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.01212*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.06946*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.04140*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.04642*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s120i pulse on channel 1 at 6675ns, square, 4ns, (-0.08076*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.07908*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.07870*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.06133*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.08056*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.06752*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.08046*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.08029*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.06458*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.07537*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.07363*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s120q pulse on channel 2 at 6675ns, square, 4ns, (-0.01635*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.01809*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.05253*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.00564*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.04430*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00687*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.00867*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.04849*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.02900*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.03318*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s121i pulse on channel 1 at 6679ns, square, 4ns, (-0.04655*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.04549*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.04559*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.03528*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.04643*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.03919*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.04646*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.04624*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.03739*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.04347*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.04235*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s121q pulse on channel 2 at 6679ns, square, 4ns, (-0.00990*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.00941*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.03037*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.00334*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.02512*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00288*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.00539*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.02773*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.01666*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.01933*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s122i pulse on channel 1 at 6683ns, square, 4ns, (-0.01264*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.01229*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.01243*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.00958*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.01260*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.01071*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.01262*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.01253*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.01016*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.01177*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.01145*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s122q pulse on channel 2 at 6683ns, square, 4ns, (-0.00295*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.00232*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.00825*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.00098*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.00671*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00063*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.00167*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.00752*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.00461*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.00535*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s123i pulse on channel 1 at 6687ns, square, 4ns, (0.02109*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.02044*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.02080*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.01602*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.02101*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.01805*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.02107*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.02086*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.01699*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.01960*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.01898*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s123q pulse on channel 2 at 6687ns, square, 4ns, (0.00521*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.00347*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.01371*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.00181*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.01092*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00094*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.00314*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.01250*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.00780*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.00919*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s124i pulse on channel 1 at 6691ns, square, 4ns, (0.05508*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.05330*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.05450*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.04185*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.05484*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.04719*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.05501*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.05432*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.04465*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.05109*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.04929*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s124q pulse on channel 2 at 6691ns, square, 4ns, (0.01389*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.00797*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.03580*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.00508*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.02840*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00260*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.00912*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.03224*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.02058*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.02457*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s125i pulse on channel 1 at 6695ns, square, 4ns, (0.08935*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.08660*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.08873*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.06807*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.08894*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.07644*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.08926*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.08795*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.07262*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.08227*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.07939*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s125q pulse on channel 2 at 6695ns, square, 4ns, (0.02201*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.01051*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.05788*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.00853*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.04627*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00398*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.01577*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.05206*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.03486*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.04100*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s126i pulse on channel 1 at 6699ns, square, 4ns, (0.12424*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.12054*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.12376*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.09587*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.12364*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.10660*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.12415*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.12194*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.10010*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.11320*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.10964*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s126q pulse on channel 2 at 6699ns, square, 4ns, (0.03009*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.01098*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.07903*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.01221*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.06381*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00481*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.02381*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.07359*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.05121*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.05844*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s127i pulse on channel 1 at 6703ns, square, 4ns, (0.15972*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.15496*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.15948*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.12426*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.15900*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.13813*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.15967*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.15621*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.12715*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.14413*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.13979*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s127q pulse on channel 2 at 6703ns, square, 4ns, (0.03870*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.00869*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.10035*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.01516*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.08019*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00414*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.03332*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.09667*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.06883*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.07726*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s128i pulse on channel 1 at 6707ns, square, 4ns, (0.19550*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.18968*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.19542*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.15155*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.19471*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.16912*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.19548*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.19096*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.15423*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.17454*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.17017*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s128q pulse on channel 2 at 6707ns, square, 4ns, (0.04732*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.00538*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.12350*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.01757*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.09807*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00285*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.04187*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.12014*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.08807*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.09623*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s129i pulse on channel 1 at 6711ns, square, 4ns, (0.23251*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.22632*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.23250*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.18076*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.23178*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.20020*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.23245*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.22670*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.18217*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.20532*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.20154*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s129q pulse on channel 2 at 6711ns, square, 4ns, (0.05329*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.00145*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.14623*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.01834*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.11823*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00493*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.05161*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.14447*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.10910*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.11593*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s130i pulse on channel 1 at 6715ns, square, 4ns, (0.26975*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.26319*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.26968*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.21157*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.26892*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.23162*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.26963*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.26273*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.20964*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.23584*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.23251*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s130q pulse on channel 2 at 6715ns, square, 4ns, (0.05911*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.00596*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.16734*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.02117*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.13826*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00800*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.06112*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.16976*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.13093*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.13676*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s131i pulse on channel 1 at 6719ns, square, 4ns, (0.30677*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.29973*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.30647*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.24304*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.30554*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.26164*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.30645*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.29803*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.23770*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.26583*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.26388*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s131q pulse on channel 2 at 6719ns, square, 4ns, (0.06531*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.01354*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.18718*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.02739*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.16015*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.01390*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.07267*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.19392*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.15310*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.15643*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s132i pulse on channel 1 at 6723ns, square, 4ns, (0.34401*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.33609*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.34344*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.27414*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.34233*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.29077*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.34331*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.33312*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.26640*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.29636*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.29643*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s132q pulse on channel 2 at 6723ns, square, 4ns, (0.07339*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.01985*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.20782*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.03391*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.18384*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.02194*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.08588*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.21766*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.17468*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.17456*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s133i pulse on channel 1 at 6727ns, square, 4ns, (0.37926*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.37020*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.37854*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.30466*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.37727*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.31664*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.37782*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.36583*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.28984*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.32234*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.32777*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s133q pulse on channel 2 at 6727ns, square, 4ns, (0.08240*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.02325*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.22587*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.03875*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.20875*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.03301*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.10002*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.24460*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.19983*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.19079*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s134i pulse on channel 1 at 6731ns, square, 4ns, (0.41152*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.40125*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.41055*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.33375*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.40940*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.34051*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.40931*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.39560*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.31091*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.34445*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.35702*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s134q pulse on channel 2 at 6731ns, square, 4ns, (0.09136*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.02817*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.24074*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.04169*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.23108*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.04255*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.11336*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.26960*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.22516*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.20466*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s135i pulse on channel 1 at 6735ns, square, 4ns, (0.43951*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.42837*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.43856*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.35959*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.43716*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.36217*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.43615*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.42204*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.33227*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.36188*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.38209*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s135q pulse on channel 2 at 6735ns, square, 4ns, (0.09832*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.02897*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.25272*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.04545*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.24901*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.05428*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.12270*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.28770*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.24942*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.21721*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s136i pulse on channel 1 at 6739ns, square, 4ns, (0.46059*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.45033*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.45974*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.38103*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.45809*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.37790*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.45651*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.44164*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.34887*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.37589*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.39934*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s136q pulse on channel 2 at 6739ns, square, 4ns, (0.09667*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.02796*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.25876*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.04797*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.26332*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.06116*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.13076*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.30072*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.26618*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.22950*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s137i pulse on channel 1 at 6743ns, square, 4ns, (0.47299*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.46338*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.47220*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.39560*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.46991*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.38612*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.46840*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.45328*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.35962*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.38464*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.40754*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s137q pulse on channel 2 at 6743ns, square, 4ns, (0.09486*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.02742*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.25928*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.05395*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.27319*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.06573*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.13513*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.30724*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.27528*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.24007*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s138i pulse on channel 1 at 6747ns, square, 4ns, (0.47554*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.46585*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.47492*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.40257*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.47204*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.38520*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.47085*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.45658*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.36322*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.38520*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.40818*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s138q pulse on channel 2 at 6747ns, square, 4ns, (0.09550*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.02425*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.25314*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.05757*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.27885*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.06659*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.13295*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.30694*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.27885*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.24399*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s139i pulse on channel 1 at 6751ns, square, 4ns, (0.46797*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.45856*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.46750*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.40065*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.46464*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.37597*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.46353*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.44975*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.35658*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.37797*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.40298*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s139q pulse on channel 2 at 6751ns, square, 4ns, (0.09339*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.02106*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.24182*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.05575*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.27865*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.06430*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.12931*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.30306*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.27592*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.23790*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s140i pulse on channel 1 at 6755ns, square, 4ns, (0.45103*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.44238*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.45056*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.38995*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.44782*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.36017*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.44803*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.43356*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.34037*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.36103*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.39152*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s140q pulse on channel 2 at 6755ns, square, 4ns, (0.08796*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.02074*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.22666*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.05376*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.27149*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.05196*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.12433*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.29594*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.27035*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.22393*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s141i pulse on channel 1 at 6759ns, square, 4ns, (0.42626*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.41880*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.42548*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.36916*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.42384*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.33778*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.42469*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.40917*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.31745*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.33910*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.37401*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s141q pulse on channel 2 at 6759ns, square, 4ns, (0.07940*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.02573*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.21311*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.04534*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.26001*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.03652*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.11947*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.28447*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.25828*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.20448*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s142i pulse on channel 1 at 6763ns, square, 4ns, (0.39569*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.38934*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.39457*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.34299*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.39387*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.31025*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.39510*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.37935*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.29341*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.31308*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.35152*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s142q pulse on channel 2 at 6763ns, square, 4ns, (0.07064*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.02978*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.19731*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.03798*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.24560*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.02170*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.11254*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.26549*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.24198*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.18169*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s143i pulse on channel 1 at 6767ns, square, 4ns, (0.36197*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.35621*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.36052*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.31284*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.36060*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.28098*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.36174*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.34652*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.26748*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.28638*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.32552*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s143q pulse on channel 2 at 6767ns, square, 4ns, (0.06431*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.03227*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.18207*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.03146*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.22819*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.01276*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.10461*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.24387*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.22137*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.15829*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s144i pulse on channel 1 at 6771ns, square, 4ns, (0.32556*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.31941*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.32411*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.28116*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.32463*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.25047*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.32553*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.31067*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.23986*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.25838*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.29489*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s144q pulse on channel 2 at 6771ns, square, 4ns, (0.06300*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.03072*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.16413*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.02465*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.20798*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00478*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.09732*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.22013*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.19806*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.13795*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s145i pulse on channel 1 at 6775ns, square, 4ns, (0.28823*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.28219*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.28689*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.24782*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.28765*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.21926*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.28822*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.27438*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.21053*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.22873*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.26223*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s145q pulse on channel 2 at 6775ns, square, 4ns, (0.05870*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.02769*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.14717*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.01823*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.18708*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00151*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.08827*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.19685*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.17538*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.11962*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s146i pulse on channel 1 at 6779ns, square, 4ns, (0.25115*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.24579*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.25009*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.21543*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.25071*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.18989*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.25115*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.23877*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.18229*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.19996*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.22958*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s146q pulse on channel 2 at 6779ns, square, 4ns, (0.05159*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.02299*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.12909*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.01489*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.16437*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00044*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.07788*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.17276*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.15196*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.10184*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s147i pulse on channel 1 at 6783ns, square, 4ns, (0.21389*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.20948*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.21316*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.18176*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.21365*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.16098*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.21383*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.20345*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.15438*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.16993*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.19680*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s147q pulse on channel 2 at 6783ns, square, 4ns, (0.04320*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.01761*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.11274*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.01012*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.14083*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00493*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.06601*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.14804*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.12989*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.08378*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s148i pulse on channel 1 at 6787ns, square, 4ns, (0.17759*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.17390*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.17713*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.15018*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.17749*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.13305*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.17740*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.16909*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.12708*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.13993*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.16418*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s148q pulse on channel 2 at 6787ns, square, 4ns, (0.03602*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.01279*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.09478*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.00575*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.11762*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00818*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.05429*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.12405*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.10934*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.06770*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s149i pulse on channel 1 at 6791ns, square, 4ns, (0.14193*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.13873*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.14162*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.11993*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.14192*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.10534*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.14169*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.13513*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.10105*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.11119*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.13153*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s149q pulse on channel 2 at 6791ns, square, 4ns, (0.02999*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.00944*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.07590*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.00216*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.09513*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00829*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.04341*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.09967*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.08822*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.05332*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s150i pulse on channel 1 at 6795ns, square, 4ns, (0.10667*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.10403*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.10652*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.08992*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.10667*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.07869*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.10642*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.10144*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.07551*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.08346*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.09926*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s150q pulse on channel 2 at 6795ns, square, 4ns, (0.02359*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.00572*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.05739*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.00025*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.07202*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00738*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.03299*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.07535*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.06643*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.03908*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s151i pulse on channel 1 at 6799ns, square, 4ns, (0.07218*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.07035*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.07209*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.06067*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.07217*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.05294*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.07196*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.06855*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.05104*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.05585*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.06739*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s151q pulse on channel 2 at 6799ns, square, 4ns, (0.01616*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.00354*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.03910*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.00078*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.04906*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00566*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.02259*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.05104*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.04572*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.02585*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s152i pulse on channel 1 at 6803ns, square, 4ns, (0.03805*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.03711*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.03802*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.03208*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.03804*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.02777*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.03791*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.03614*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.02715*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.02900*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.03568*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s152q pulse on channel 2 at 6803ns, square, 4ns, (0.00841*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.00149*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.02046*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.00074*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.02600*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00320*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.01190*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.02665*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.02463*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.01322*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s153i pulse on channel 1 at 6807ns, square, 4ns, (0.00421*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.00411*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.00421*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.00357*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.00421*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.00307*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00419*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.00399*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.00305*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.00315*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.00395*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s153q pulse on channel 2 at 6807ns, square, 4ns, (0.00092*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.00012*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.00222*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.00008*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.00288*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00039*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.00135*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.00290*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.00279*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.00146*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s154i pulse on channel 1 at 6811ns, square, 4ns, (-0.02956*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.02890*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.02955*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.02514*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.02955*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.02141*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.02940*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.02790*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.02181*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.02178*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.02772*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s154q pulse on channel 2 at 6811ns, square, 4ns, (-0.00620*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.00057*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.01555*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.00061*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.02038*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00306*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.00977*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.01995*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.01999*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.01027*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s155i pulse on channel 1 at 6815ns, square, 4ns, (-0.06362*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.06246*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.06362*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.05411*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.06361*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.04555*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.06327*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.05999*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.04769*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.04574*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.05960*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s155q pulse on channel 2 at 6815ns, square, 4ns, (-0.01211*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.00056*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.03346*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.00100*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.04442*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00661*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.02117*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.04211*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.04421*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.02224*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s156i pulse on channel 1 at 6819ns, square, 4ns, (-0.09796*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.09631*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.09796*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.08336*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.09796*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.06958*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.09739*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.09199*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.07462*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.06835*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.09157*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s156q pulse on channel 2 at 6819ns, square, 4ns, (-0.01790*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.00042*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.05146*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.00098*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.06896*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.01055*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.03368*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.06347*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.07017*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.03480*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s157i pulse on channel 1 at 6823ns, square, 4ns, (-0.13307*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.13098*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.13307*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.11350*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.13307*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.09453*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.13224*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.12441*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.10285*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.09088*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.12408*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s157q pulse on channel 2 at 6823ns, square, 4ns, (-0.02350*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.00049*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.06947*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.00126*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.09366*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.01483*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.04722*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.08445*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.09721*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.04808*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s158i pulse on channel 1 at 6827ns, square, 4ns, (-0.16865*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.16624*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.16863*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.14309*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.16864*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.12014*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.16754*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.15728*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.13234*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.11317*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.15701*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s158q pulse on channel 2 at 6827ns, square, 4ns, (-0.02840*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.00243*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.08926*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.00153*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.11836*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.01926*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.06088*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.10453*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.12503*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.06156*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s159i pulse on channel 1 at 6831ns, square, 4ns, (-0.20461*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.20219*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.20450*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.17215*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.20457*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.14739*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.20313*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.19067*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.16382*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.13439*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.19070*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s159q pulse on channel 2 at 6831ns, square, 4ns, (-0.03134*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.00662*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.11058*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.00398*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.14192*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.02458*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.07421*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.12258*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.15429*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.07416*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s160i pulse on channel 1 at 6835ns, square, 4ns, (-0.24183*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.23945*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.24166*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.20179*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.24168*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.17567*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.24004*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.22547*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.19540*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.15451*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.22602*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s160q pulse on channel 2 at 6835ns, square, 4ns, (-0.03385*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.00895*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.13327*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.00845*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.16619*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.02934*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.08742*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.14248*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.18603*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.08599*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s161i pulse on channel 1 at 6839ns, square, 4ns, (-0.27901*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.27684*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.27882*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.23041*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.27869*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.20351*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.27688*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.26035*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.22736*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.17353*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.26186*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s161q pulse on channel 2 at 6839ns, square, 4ns, (-0.03476*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.01022*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.15735*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.01341*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.19087*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.03442*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.10032*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.16172*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.21848*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.09631*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s162i pulse on channel 1 at 6843ns, square, 4ns, (-0.31620*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.31427*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.31609*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.25744*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.31559*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.23203*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.31350*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.29536*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.26140*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.19330*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.29804*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s162q pulse on channel 2 at 6843ns, square, 4ns, (-0.03489*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.00830*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.18359*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.01956*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.21481*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.04121*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.11288*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.17790*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.25023*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.10561*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s163i pulse on channel 1 at 6847ns, square, 4ns, (-0.35306*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.35162*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.35303*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.28422*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.35191*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.26009*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.34969*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.33044*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.29729*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.21392*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.33414*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s163q pulse on channel 2 at 6847ns, square, 4ns, (-0.03190*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.00452*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.20945*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.02841*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.23876*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.04869*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.12435*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.19045*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.28087*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.11402*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s164i pulse on channel 1 at 6851ns, square, 4ns, (-0.38759*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.38653*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.38759*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.30954*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.38588*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.28324*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.38431*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.36350*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.33126*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.23619*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.36754*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s164q pulse on channel 2 at 6851ns, square, 4ns, (-0.02866*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.00059*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.23327*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.03636*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.26459*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.05038*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.13451*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.20124*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.30732*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.12307*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s165i pulse on channel 1 at 6855ns, square, 4ns, (-0.41906*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.41828*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.41906*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.33233*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.41662*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.30059*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.41616*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.39357*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.36187*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.25468*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.39767*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s165q pulse on channel 2 at 6855ns, square, 4ns, (-0.02562*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.00193*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.25528*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.04515*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.29199*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.04920*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.14392*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.21133*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.33279*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.13217*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s166i pulse on channel 1 at 6859ns, square, 4ns, (-0.44551*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.44511*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.44545*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.35157*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.44232*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.31564*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.44312*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.41987*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.38827*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.27264*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.42265*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s166q pulse on channel 2 at 6859ns, square, 4ns, (-0.01871*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.00695*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.27364*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.05319*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.31440*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.04601*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.14893*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.21845*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.35234*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.14085*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s167i pulse on channel 1 at 6863ns, square, 4ns, (-0.46456*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.46449*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.46440*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.36378*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.46014*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.32529*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.46282*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.43960*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.40939*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.28418*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.44102*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s167q pulse on channel 2 at 6863ns, square, 4ns, (-0.00835*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.01224*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.28894*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.06396*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.33167*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.04023*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.15024*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.21959*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.36751*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.14602*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s168i pulse on channel 1 at 6867ns, square, 4ns, (-0.47458*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.47458*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.47431*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.36703*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.46891*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.32701*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.47352*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.45000*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.42036*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.29418*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.45089*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s168q pulse on channel 2 at 6867ns, square, 4ns, (0.00234*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.01613*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.30086*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.07315*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.34394*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.03173*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.15076*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.22028*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.37240*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.14807*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s169i pulse on channel 1 at 6871ns, square, 4ns, (-0.47458*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.47449*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.47427*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.36201*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.46838*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.32486*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.47402*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.45020*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.42157*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.30066*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.45040*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s169q pulse on channel 2 at 6871ns, square, 4ns, (0.00948*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.01727*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.30688*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.07650*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.34597*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.02316*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.15015*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.21797*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.36719*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.14955*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s170i pulse on channel 1 at 6875ns, square, 4ns, (-0.46456*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.46431*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.46416*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.35200*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.45897*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.31689*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.46438*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.44138*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.41507*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.29898*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.44045*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s170q pulse on channel 2 at 6875ns, square, 4ns, (0.01547*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.01928*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.30318*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.07190*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.33970*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.01304*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.14492*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.20865*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.35557*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.14774*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s171i pulse on channel 1 at 6879ns, square, 4ns, (-0.44551*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.44508*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.44504*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.33520*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.44046*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.30266*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.44549*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.42326*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.40065*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.29252*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.42103*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s171q pulse on channel 2 at 6879ns, square, 4ns, (0.01946*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.02028*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.29346*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.06686*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.32691*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00395*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.13900*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.19483*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.33602*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.14563*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s172i pulse on channel 1 at 6883ns, square, 4ns, (-0.41906*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.41855*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.41856*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.31441*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.41432*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.28763*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.41906*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.39999*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.37905*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.27911*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.39330*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s172q pulse on channel 2 at 6883ns, square, 4ns, (0.02061*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.02048*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.27705*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.06283*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.30476*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00110*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.12499*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.17869*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.31258*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.14466*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s173i pulse on channel 1 at 6887ns, square, 4ns, (-0.38759*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.38696*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.38715*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.29125*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.38267*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.26494*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.38755*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.37124*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.35375*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.26159*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.36038*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s173q pulse on channel 2 at 6887ns, square, 4ns, (0.02209*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.01855*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.25574*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.06156*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.28291*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00580*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.11139*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.15840*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.28601*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.14267*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s174i pulse on channel 1 at 6891ns, square, 4ns, (-0.35306*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.35218*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.35281*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.26566*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.34845*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.24264*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.35298*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.33854*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.32629*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.23873*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.32638*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s174q pulse on channel 2 at 6891ns, square, 4ns, (0.02494*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.01333*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.23254*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.05685*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.25647*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00773*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.10020*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.13485*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.26011*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.13464*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s175i pulse on channel 1 at 6895ns, square, 4ns, (-0.31620*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.31514*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.31607*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.23945*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.31202*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.22179*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.31612*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.30351*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.29573*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.21425*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.29155*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s175q pulse on channel 2 at 6895ns, square, 4ns, (0.02587*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.00892*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.20651*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.05124*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.22537*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00716*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.08868*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.11192*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.23255*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.12240*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s176i pulse on channel 1 at 6899ns, square, 4ns, (-0.27901*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.27794*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.27898*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.21296*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.27594*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.19812*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.27897*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.26786*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.26358*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.18919*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.25757*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s176q pulse on channel 2 at 6899ns, square, 4ns, (0.02439*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.00418*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.18026*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.04127*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.19645*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00441*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.07810*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.09151*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.20507*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.10726*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s177i pulse on channel 1 at 6903ns, square, 4ns, (-0.24183*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.24079*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.24182*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.18432*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.23980*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.17284*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.24178*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.23271*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.22966*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.16600*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.22445*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s177q pulse on channel 2 at 6903ns, square, 4ns, (0.02237*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.00204*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.15654*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.03129*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.16914*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00458*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.06578*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.07575*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.17585*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.09001*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s178i pulse on channel 1 at 6907ns, square, 4ns, (-0.20461*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.20355*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.20461*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.15499*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.20307*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.14632*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.20456*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.19738*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.19513*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.14270*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.19133*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s178q pulse on channel 2 at 6907ns, square, 4ns, (0.02076*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.00058*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.13358*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.02507*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.14302*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00463*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.05391*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.06155*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.14664*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.07251*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s179i pulse on channel 1 at 6911ns, square, 4ns, (-0.16865*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.16773*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.16863*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.12704*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.16744*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.12079*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.16861*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.16307*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.16119*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.11952*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.15933*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s179q pulse on channel 2 at 6911ns, square, 4ns, (0.01761*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.00217*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.11092*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.02018*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.11770*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00363*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.04301*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.04958*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.11898*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.05529*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s180i pulse on channel 1 at 6915ns, square, 4ns, (-0.13307*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.13245*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.13306*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.09949*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.13223*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.09511*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.13303*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.12871*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.12710*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.09612*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.12720*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s180q pulse on channel 2 at 6915ns, square, 4ns, (0.01288*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.00198*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.08838*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.01498*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.09307*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00350*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.03381*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.03942*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.09203*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.03908*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s181i pulse on channel 1 at 6919ns, square, 4ns, (-0.09796*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.09759*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.09794*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.07268*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.09747*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.06962*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.09791*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.09432*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.09333*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.07203*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.09423*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s181q pulse on channel 2 at 6919ns, square, 4ns, (0.00855*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.00214*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.06568*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.00980*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.06891*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00304*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.02644*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.02975*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.06639*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.02678*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s182i pulse on channel 1 at 6923ns, square, 4ns, (-0.06362*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.06344*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.06360*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.04705*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.06330*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.04529*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.06358*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.06097*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.06043*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.04717*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.06149*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s182q pulse on channel 2 at 6923ns, square, 4ns, (0.00471*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.00161*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.04282*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.00634*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.04468*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00216*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.01815*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.01990*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.04269*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.01630*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s183i pulse on channel 1 at 6927ns, square, 4ns, (-0.02956*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.02950*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.02955*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.02160*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.02941*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.02103*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.02953*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.02824*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.02796*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.02197*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.02867*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s183q pulse on channel 2 at 6927ns, square, 4ns, (0.00186*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.00076*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.02018*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.00292*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.02077*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00120*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.00874*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.00959*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.01977*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.00719*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s184i pulse on channel 1 at 6931ns, square, 4ns, (0.00421*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.00420*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.00421*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.00300*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.00419*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.00297*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00420*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.00401*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.00397*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.00317*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.00408*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s184q pulse on channel 2 at 6931ns, square, 4ns, (-0.00019*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.00009*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.00295*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.00039*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.00298*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00018*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.00128*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.00141*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.00277*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.00103*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s185i pulse on channel 1 at 6935ns, square, 4ns, (0.03805*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.03803*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.03804*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.02637*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.03790*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.02658*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.03801*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.03606*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.03574*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.02877*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.03688*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s185q pulse on channel 2 at 6935ns, square, 4ns, (-0.00100*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.00084*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.02742*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.00333*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.02722*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00164*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.01212*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.01306*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.02490*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.00934*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s186i pulse on channel 1 at 6939ns, square, 4ns, (0.07218*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.07217*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.07216*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.04902*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.07187*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.05002*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.07208*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.06799*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.06728*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.05474*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.06993*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s186q pulse on channel 2 at 6939ns, square, 4ns, (-0.00084*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.00151*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.05298*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.00665*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.05204*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00377*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.02422*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.02615*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.04705*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.01789*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s187i pulse on channel 1 at 6943ns, square, 4ns, (0.10667*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.10667*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.10664*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.07142*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.10618*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.07428*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.10650*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.09990*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.09868*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.08077*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.10321*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s187q pulse on channel 2 at 6943ns, square, 4ns, (-0.00025*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.00244*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.07924*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.01019*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.07656*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00598*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.03739*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.04051*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.06968*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.02695*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s188i pulse on channel 1 at 6947ns, square, 4ns, (0.14193*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.14193*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.14190*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.09395*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.14119*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.09908*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.14173*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.13274*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.13048*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.10620*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.13703*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s188q pulse on channel 2 at 6947ns, square, 4ns, (0.00103*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.00304*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.10638*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.01445*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.10162*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00748*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.05026*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.05586*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.09416*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.03699*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s189i pulse on channel 1 at 6951ns, square, 4ns, (0.17759*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.17759*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.17754*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.11612*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.17645*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.12337*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.17737*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.16620*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.16304*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.13223*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.17071*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s189q pulse on channel 2 at 6951ns, square, 4ns, (0.00048*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.00389*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.13436*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.02009*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.12774*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00885*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.06257*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.07039*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.11854*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.04894*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s190i pulse on channel 1 at 6955ns, square, 4ns, (0.21389*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.21389*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.21385*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.13849*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.21243*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.14834*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.21371*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.19959*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.19565*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.15872*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.20472*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s190q pulse on channel 2 at 6955ns, square, 4ns, (-0.00033*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.00427*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.16300*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.02490*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.15409*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00876*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.07690*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.08643*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.14337*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.06196*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s191i pulse on channel 1 at 6959ns, square, 4ns, (0.25115*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.25115*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.25112*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.16102*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.24947*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.17297*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.25101*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.23390*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.22946*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.18408*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.23964*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s191q pulse on channel 2 at 6959ns, square, 4ns, (-0.00122*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.00397*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.19274*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.02900*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.18209*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00828*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.09147*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.10209*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.17085*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.07516*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s192i pulse on channel 1 at 6963ns, square, 4ns, (0.28823*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.28820*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.28822*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.18347*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.28622*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.19474*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.28812*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.26874*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.26324*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.20811*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.27583*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s192q pulse on channel 2 at 6963ns, square, 4ns, (-0.00359*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.00211*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.22229*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.03391*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.21249*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00775*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.10418*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.11739*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.19941*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.08363*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s193i pulse on channel 1 at 6967ns, square, 4ns, (0.32556*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.32549*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.32556*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.20655*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.32322*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.21660*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.32549*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.30414*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.29571*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.23162*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.31225*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s193q pulse on channel 2 at 6967ns, square, 4ns, (-0.00672*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.00073*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.25165*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.03899*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.24305*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00685*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.11614*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.13619*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.22879*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.09213*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s194i pulse on channel 1 at 6971ns, square, 4ns, (0.36197*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.36179*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.36195*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.22783*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.35937*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.23850*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.36196*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.33788*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.32717*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.25481*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.34817*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s194q pulse on channel 2 at 6971ns, square, 4ns, (-0.01138*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.00343*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.28127*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.04323*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.27228*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00175*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.12984*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.15484*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.25708*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.09898*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s195i pulse on channel 1 at 6975ns, square, 4ns, (0.39569*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.39522*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.39556*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.24881*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.39305*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.25674*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.39568*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.36888*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.35847*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.27932*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.38144*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s195q pulse on channel 2 at 6975ns, square, 4ns, (-0.01930*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.01025*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.30768*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.04570*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.30109*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00336*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.14320*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.16755*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.28028*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.10524*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s196i pulse on channel 1 at 6979ns, square, 4ns, (0.42626*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.42541*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.42586*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.26697*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.42301*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.27359*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.42619*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.39760*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.38884*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.30424*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.41184*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s196q pulse on channel 2 at 6979ns, square, 4ns, (-0.02682*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.01836*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.33230*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.05255*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.32687*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00763*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.15366*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.17465*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.29855*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.10993*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s197i pulse on channel 1 at 6983ns, square, 4ns, (0.45103*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.44988*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.45014*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.28105*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.44763*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.28572*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.45085*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.42068*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.41289*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.32715*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.43648*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s197q pulse on channel 2 at 6983ns, square, 4ns, (-0.03231*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.02832*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.35276*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.05528*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.34900*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.01305*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.16266*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.18152*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.31050*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.11366*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s198i pulse on channel 1 at 6987ns, square, 4ns, (-0.48080*(1-abs(k-0)+abs(1-abs(k-0)))/2+-0.48079*(1-abs(k-1)+abs(1-abs(k-1)))/2+-0.48080*(1-abs(k-2)+abs(1-abs(k-2)))/2+-0.48078*(1-abs(k-3)+abs(1-abs(k-3)))/2+-0.48080*(1-abs(k-4)+abs(1-abs(k-4)))/2+-0.48078*(1-abs(k-5)+abs(1-abs(k-5)))/2+-0.48078*(1-abs(k-6)+abs(1-abs(k-6)))/2+-0.48080*(1-abs(k-7)+abs(1-abs(k-7)))/2+-0.48080*(1-abs(k-8)+abs(1-abs(k-8)))/2+-0.48079*(1-abs(k-9)+abs(1-abs(k-9)))/2+-0.48079*(1-abs(k-10)+abs(1-abs(k-10)))/2+0.46797*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.46692*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.46658*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.29035*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.46424*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.29471*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.46767*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.43712*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.43112*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.34745*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.45371*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s198q pulse on channel 2 at 6987ns, square, 4ns, (-0.00317*(1-abs(k-1)+abs(1-abs(k-1)))/2+0.00085*(1-abs(k-2)+abs(1-abs(k-2)))/2+0.00406*(1-abs(k-3)+abs(1-abs(k-3)))/2+-0.00185*(1-abs(k-4)+abs(1-abs(k-4)))/2+-0.00490*(1-abs(k-5)+abs(1-abs(k-5)))/2+-0.00468*(1-abs(k-6)+abs(1-abs(k-6)))/2+-0.00033*(1-abs(k-7)+abs(1-abs(k-7)))/2+0.00002*(1-abs(k-8)+abs(1-abs(k-8)))/2+-0.00264*(1-abs(k-9)+abs(1-abs(k-9)))/2+0.00293*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.03126*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.03609*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.36701*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.05900*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.36351*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.01684*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.16711*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.18201*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.31348*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.11464*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s199i pulse on channel 1 at 6991ns, square, 4ns, (-0.48080*(1-abs(k-0)+abs(1-abs(k-0)))/2+-0.48074*(1-abs(k-1)+abs(1-abs(k-1)))/2+-0.48080*(1-abs(k-2)+abs(1-abs(k-2)))/2+-0.48070*(1-abs(k-3)+abs(1-abs(k-3)))/2+-0.48074*(1-abs(k-4)+abs(1-abs(k-4)))/2+-0.48064*(1-abs(k-5)+abs(1-abs(k-5)))/2+-0.48045*(1-abs(k-6)+abs(1-abs(k-6)))/2+-0.48080*(1-abs(k-7)+abs(1-abs(k-7)))/2+-0.48080*(1-abs(k-8)+abs(1-abs(k-8)))/2+-0.48077*(1-abs(k-9)+abs(1-abs(k-9)))/2+-0.48070*(1-abs(k-10)+abs(1-abs(k-10)))/2+0.47554*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.47476*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.47299*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.29357*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.47120*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.30234*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.47515*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.44547*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.43993*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.35613*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.46213*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s199q pulse on channel 2 at 6991ns, square, 4ns, (-0.00755*(1-abs(k-1)+abs(1-abs(k-1)))/2+0.00133*(1-abs(k-2)+abs(1-abs(k-2)))/2+0.00993*(1-abs(k-3)+abs(1-abs(k-3)))/2+-0.00785*(1-abs(k-4)+abs(1-abs(k-4)))/2+-0.01231*(1-abs(k-5)+abs(1-abs(k-5)))/2+-0.01835*(1-abs(k-6)+abs(1-abs(k-6)))/2+0.00181*(1-abs(k-7)+abs(1-abs(k-7)))/2+-0.00135*(1-abs(k-8)+abs(1-abs(k-8)))/2+-0.00543*(1-abs(k-9)+abs(1-abs(k-9)))/2+0.00964*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.02713*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.04917*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.37410*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.06410*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.36705*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.01926*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.16640*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.18056*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.31513*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.11214*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s200i pulse on channel 1 at 6995ns, square, 4ns, (-0.48080*(1-abs(k-0)+abs(1-abs(k-0)))/2+-0.48063*(1-abs(k-1)+abs(1-abs(k-1)))/2+-0.48080*(1-abs(k-2)+abs(1-abs(k-2)))/2+-0.48058*(1-abs(k-3)+abs(1-abs(k-3)))/2+-0.48077*(1-abs(k-4)+abs(1-abs(k-4)))/2+-0.48051*(1-abs(k-5)+abs(1-abs(k-5)))/2+-0.47979*(1-abs(k-6)+abs(1-abs(k-6)))/2+-0.48077*(1-abs(k-7)+abs(1-abs(k-7)))/2+-0.48080*(1-abs(k-8)+abs(1-abs(k-8)))/2+-0.48076*(1-abs(k-9)+abs(1-abs(k-9)))/2+-0.48045*(1-abs(k-10)+abs(1-abs(k-10)))/2+0.47299*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.47248*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.46869*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.29047*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.46823*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.30320*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.47253*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.44512*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.43998*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.35605*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.46049*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s200q pulse on channel 2 at 6995ns, square, 4ns, (-0.01285*(1-abs(k-1)+abs(1-abs(k-1)))/2+0.00185*(1-abs(k-2)+abs(1-abs(k-2)))/2+0.01442*(1-abs(k-3)+abs(1-abs(k-3)))/2+-0.00565*(1-abs(k-4)+abs(1-abs(k-4)))/2+-0.01683*(1-abs(k-5)+abs(1-abs(k-5)))/2+-0.03111*(1-abs(k-6)+abs(1-abs(k-6)))/2+0.00538*(1-abs(k-7)+abs(1-abs(k-7)))/2+-0.00152*(1-abs(k-8)+abs(1-abs(k-8)))/2+-0.00589*(1-abs(k-9)+abs(1-abs(k-9)))/2+0.01836*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.02194*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.06366*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.37330*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.06699*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.36303*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.02096*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.15999*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.17362*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.31137*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.10804*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s201i pulse on channel 1 at 6999ns, square, 4ns, (-0.48080*(1-abs(k-0)+abs(1-abs(k-0)))/2+-0.48054*(1-abs(k-1)+abs(1-abs(k-1)))/2+-0.48073*(1-abs(k-2)+abs(1-abs(k-2)))/2+-0.48045*(1-abs(k-3)+abs(1-abs(k-3)))/2+-0.48079*(1-abs(k-4)+abs(1-abs(k-4)))/2+-0.48045*(1-abs(k-5)+abs(1-abs(k-5)))/2+-0.47927*(1-abs(k-6)+abs(1-abs(k-6)))/2+-0.48074*(1-abs(k-7)+abs(1-abs(k-7)))/2+-0.48077*(1-abs(k-8)+abs(1-abs(k-8)))/2+-0.48076*(1-abs(k-9)+abs(1-abs(k-9)))/2+-0.48033*(1-abs(k-10)+abs(1-abs(k-10)))/2+0.46059*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.46032*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.45385*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.27635*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.45565*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.29980*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.46009*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.43513*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.43165*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.34766*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.44888*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s201q pulse on channel 2 at 6999ns, square, 4ns, (-0.01570*(1-abs(k-1)+abs(1-abs(k-1)))/2+0.00839*(1-abs(k-2)+abs(1-abs(k-2)))/2+0.01831*(1-abs(k-3)+abs(1-abs(k-3)))/2+-0.00349*(1-abs(k-4)+abs(1-abs(k-4)))/2+-0.01833*(1-abs(k-5)+abs(1-abs(k-5)))/2+-0.03833*(1-abs(k-6)+abs(1-abs(k-6)))/2+0.00785*(1-abs(k-7)+abs(1-abs(k-7)))/2+-0.00498*(1-abs(k-8)+abs(1-abs(k-8)))/2+-0.00595*(1-abs(k-9)+abs(1-abs(k-9)))/2+0.02114*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.01593*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.07851*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.36848*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.06729*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.34966*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.02138*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.15102*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.16070*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.30212*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.10319*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s202i pulse on channel 1 at 7003ns, square, 4ns, (-0.48080*(1-abs(k-0)+abs(1-abs(k-0)))/2+-0.48047*(1-abs(k-1)+abs(1-abs(k-1)))/2+-0.48049*(1-abs(k-2)+abs(1-abs(k-2)))/2+-0.48029*(1-abs(k-3)+abs(1-abs(k-3)))/2+-0.48071*(1-abs(k-4)+abs(1-abs(k-4)))/2+-0.48047*(1-abs(k-5)+abs(1-abs(k-5)))/2+-0.47904*(1-abs(k-6)+abs(1-abs(k-6)))/2+-0.48068*(1-abs(k-7)+abs(1-abs(k-7)))/2+-0.48069*(1-abs(k-8)+abs(1-abs(k-8)))/2+-0.48074*(1-abs(k-9)+abs(1-abs(k-9)))/2+-0.48026*(1-abs(k-10)+abs(1-abs(k-10)))/2+0.43951*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.43935*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.43066*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.25874*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.43459*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.28995*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.43887*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.41430*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.41638*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.33520*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.42875*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s202q pulse on channel 2 at 7003ns, square, 4ns, (-0.01794*(1-abs(k-1)+abs(1-abs(k-1)))/2+0.01713*(1-abs(k-2)+abs(1-abs(k-2)))/2+0.02222*(1-abs(k-3)+abs(1-abs(k-3)))/2+-0.00910*(1-abs(k-4)+abs(1-abs(k-4)))/2+-0.01782*(1-abs(k-5)+abs(1-abs(k-5)))/2+-0.04109*(1-abs(k-6)+abs(1-abs(k-6)))/2+0.01093*(1-abs(k-7)+abs(1-abs(k-7)))/2+-0.01036*(1-abs(k-8)+abs(1-abs(k-8)))/2+-0.00731*(1-abs(k-9)+abs(1-abs(k-9)))/2+0.02283*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.01179*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.08776*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.35528*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.06556*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.33030*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.02374*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.14671*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.14070*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.28427*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.09667*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s203i pulse on channel 1 at 7007ns, square, 4ns, (-0.48080*(1-abs(k-0)+abs(1-abs(k-0)))/2+-0.48063*(1-abs(k-1)+abs(1-abs(k-1)))/2+-0.48035*(1-abs(k-2)+abs(1-abs(k-2)))/2+-0.48003*(1-abs(k-3)+abs(1-abs(k-3)))/2+-0.48040*(1-abs(k-4)+abs(1-abs(k-4)))/2+-0.48044*(1-abs(k-5)+abs(1-abs(k-5)))/2+-0.47870*(1-abs(k-6)+abs(1-abs(k-6)))/2+-0.48064*(1-abs(k-7)+abs(1-abs(k-7)))/2+-0.48076*(1-abs(k-8)+abs(1-abs(k-8)))/2+-0.48078*(1-abs(k-9)+abs(1-abs(k-9)))/2+-0.48000*(1-abs(k-10)+abs(1-abs(k-10)))/2+0.41152*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.41140*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.40066*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.24092*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.40660*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.27334*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.41084*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.38643*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.39280*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.31497*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.40200*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s203q pulse on channel 2 at 7007ns, square, 4ns, (-0.01289*(1-abs(k-1)+abs(1-abs(k-1)))/2+0.02074*(1-abs(k-2)+abs(1-abs(k-2)))/2+0.02723*(1-abs(k-3)+abs(1-abs(k-3)))/2+-0.01955*(1-abs(k-4)+abs(1-abs(k-4)))/2+-0.01869*(1-abs(k-5)+abs(1-abs(k-5)))/2+-0.04490*(1-abs(k-6)+abs(1-abs(k-6)))/2+0.01223*(1-abs(k-7)+abs(1-abs(k-7)))/2+-0.00647*(1-abs(k-8)+abs(1-abs(k-8)))/2+-0.00455*(1-abs(k-9)+abs(1-abs(k-9)))/2+0.02778*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.01003*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.09390*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.33362*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.06343*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.30762*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.02372*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.14149*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.12269*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.26485*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.08800*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s204i pulse on channel 1 at 7011ns, square, 4ns, (0.48080*(1-abs(k-0)+abs(1-abs(k-0)))/2+0.48064*(1-abs(k-1)+abs(1-abs(k-1)))/2+0.48029*(1-abs(k-2)+abs(1-abs(k-2)))/2+0.47979*(1-abs(k-3)+abs(1-abs(k-3)))/2+0.48001*(1-abs(k-4)+abs(1-abs(k-4)))/2+0.48027*(1-abs(k-5)+abs(1-abs(k-5)))/2+0.47816*(1-abs(k-6)+abs(1-abs(k-6)))/2+0.48060*(1-abs(k-7)+abs(1-abs(k-7)))/2+0.48080*(1-abs(k-8)+abs(1-abs(k-8)))/2+0.48080*(1-abs(k-9)+abs(1-abs(k-9)))/2+0.47966*(1-abs(k-10)+abs(1-abs(k-10)))/2+0.37926*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.37907*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.36766*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.21967*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.37487*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.25125*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.37877*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.35501*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.36392*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.29033*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.37080*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s204q pulse on channel 2 at 7011ns, square, 4ns, (0.01236*(1-abs(k-1)+abs(1-abs(k-1)))/2+-0.02221*(1-abs(k-2)+abs(1-abs(k-2)))/2+-0.03118*(1-abs(k-3)+abs(1-abs(k-3)))/2+0.02755*(1-abs(k-4)+abs(1-abs(k-4)))/2+0.02254*(1-abs(k-5)+abs(1-abs(k-5)))/2+0.05034*(1-abs(k-6)+abs(1-abs(k-6)))/2+-0.01371*(1-abs(k-7)+abs(1-abs(k-7)))/2+0.00216*(1-abs(k-8)+abs(1-abs(k-8)))/2+-0.00044*(1-abs(k-9)+abs(1-abs(k-9)))/2+-0.03310*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.01186*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.09308*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.30916*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.05754*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.28409*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.01910*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.13343*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.10674*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.24402*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.07963*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s205i pulse on channel 1 at 7015ns, square, 4ns, (0.48080*(1-abs(k-0)+abs(1-abs(k-0)))/2+0.48045*(1-abs(k-1)+abs(1-abs(k-1)))/2+0.48004*(1-abs(k-2)+abs(1-abs(k-2)))/2+0.47946*(1-abs(k-3)+abs(1-abs(k-3)))/2+0.47951*(1-abs(k-4)+abs(1-abs(k-4)))/2+0.48026*(1-abs(k-5)+abs(1-abs(k-5)))/2+0.47766*(1-abs(k-6)+abs(1-abs(k-6)))/2+0.48055*(1-abs(k-7)+abs(1-abs(k-7)))/2+0.48079*(1-abs(k-8)+abs(1-abs(k-8)))/2+0.48079*(1-abs(k-9)+abs(1-abs(k-9)))/2+0.47907*(1-abs(k-10)+abs(1-abs(k-10)))/2+0.34401*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.34376*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.33248*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.19929*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.33986*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.22557*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.34370*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.32174*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.33095*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.26258*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.33673*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s205q pulse on channel 2 at 7015ns, square, 4ns, (0.01825*(1-abs(k-1)+abs(1-abs(k-1)))/2+-0.02705*(1-abs(k-2)+abs(1-abs(k-2)))/2+-0.03593*(1-abs(k-3)+abs(1-abs(k-3)))/2+0.03513*(1-abs(k-4)+abs(1-abs(k-4)))/2+0.02286*(1-abs(k-5)+abs(1-abs(k-5)))/2+0.05488*(1-abs(k-6)+abs(1-abs(k-6)))/2+-0.01554*(1-abs(k-7)+abs(1-abs(k-7)))/2+0.00221*(1-abs(k-8)+abs(1-abs(k-8)))/2+-0.00258*(1-abs(k-9)+abs(1-abs(k-9)))/2+-0.04074*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.01317*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.08830*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.28041*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.05329*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.25973*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.01454*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.12176*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.09388*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.22225*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.07042*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s206i pulse on channel 1 at 7019ns, square, 4ns, (0.48080*(1-abs(k-0)+abs(1-abs(k-0)))/2+0.48007*(1-abs(k-1)+abs(1-abs(k-1)))/2+0.48006*(1-abs(k-2)+abs(1-abs(k-2)))/2+0.47930*(1-abs(k-3)+abs(1-abs(k-3)))/2+0.47900*(1-abs(k-4)+abs(1-abs(k-4)))/2+0.48027*(1-abs(k-5)+abs(1-abs(k-5)))/2+0.47709*(1-abs(k-6)+abs(1-abs(k-6)))/2+0.48053*(1-abs(k-7)+abs(1-abs(k-7)))/2+0.48080*(1-abs(k-8)+abs(1-abs(k-8)))/2+0.48076*(1-abs(k-9)+abs(1-abs(k-9)))/2+0.47821*(1-abs(k-10)+abs(1-abs(k-10)))/2+0.30677*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.30645*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.29528*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.17755*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.30285*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.19933*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.30663*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.28653*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.29444*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.23299*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.30033*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s206q pulse on channel 2 at 7019ns, square, 4ns, (0.02654*(1-abs(k-1)+abs(1-abs(k-1)))/2+-0.02667*(1-abs(k-2)+abs(1-abs(k-2)))/2+-0.03793*(1-abs(k-3)+abs(1-abs(k-3)))/2+0.04153*(1-abs(k-4)+abs(1-abs(k-4)))/2+0.02251*(1-abs(k-5)+abs(1-abs(k-5)))/2+0.05962*(1-abs(k-6)+abs(1-abs(k-6)))/2+-0.01602*(1-abs(k-7)+abs(1-abs(k-7)))/2+0.00093*(1-abs(k-8)+abs(1-abs(k-8)))/2+-0.00588*(1-abs(k-9)+abs(1-abs(k-9)))/2+-0.04980*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.01389*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.08316*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.25016*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.04888*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.23318*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00909*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.10957*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.08608*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.19955*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.06248*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s207i pulse on channel 1 at 7023ns, square, 4ns, (0.48080*(1-abs(k-0)+abs(1-abs(k-0)))/2+0.47991*(1-abs(k-1)+abs(1-abs(k-1)))/2+0.48024*(1-abs(k-2)+abs(1-abs(k-2)))/2+0.47903*(1-abs(k-3)+abs(1-abs(k-3)))/2+0.47858*(1-abs(k-4)+abs(1-abs(k-4)))/2+0.48047*(1-abs(k-5)+abs(1-abs(k-5)))/2+0.47656*(1-abs(k-6)+abs(1-abs(k-6)))/2+0.48056*(1-abs(k-7)+abs(1-abs(k-7)))/2+0.48080*(1-abs(k-8)+abs(1-abs(k-8)))/2+0.48066*(1-abs(k-9)+abs(1-abs(k-9)))/2+0.47703*(1-abs(k-10)+abs(1-abs(k-10)))/2+0.26975*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.26938*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.25832*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.15524*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.26627*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.17604*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.26969*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.25246*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.25810*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.20597*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.26393*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s207q pulse on channel 2 at 7023ns, square, 4ns, (0.02926*(1-abs(k-1)+abs(1-abs(k-1)))/2+-0.02328*(1-abs(k-2)+abs(1-abs(k-2)))/2+-0.04123*(1-abs(k-3)+abs(1-abs(k-3)))/2+0.04613*(1-abs(k-4)+abs(1-abs(k-4)))/2+0.01786*(1-abs(k-5)+abs(1-abs(k-5)))/2+0.06375*(1-abs(k-6)+abs(1-abs(k-6)))/2+-0.01526*(1-abs(k-7)+abs(1-abs(k-7)))/2+-0.00120*(1-abs(k-8)+abs(1-abs(k-8)))/2+-0.01151*(1-abs(k-9)+abs(1-abs(k-9)))/2+-0.06013*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.01407*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.07770*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.22060*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.04320*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.20439*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00558*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.09502*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.07841*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.17418*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.05573*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s208i pulse on channel 1 at 7027ns, square, 4ns, (0.48080*(1-abs(k-0)+abs(1-abs(k-0)))/2+0.48001*(1-abs(k-1)+abs(1-abs(k-1)))/2+0.48052*(1-abs(k-2)+abs(1-abs(k-2)))/2+0.47891*(1-abs(k-3)+abs(1-abs(k-3)))/2+0.47797*(1-abs(k-4)+abs(1-abs(k-4)))/2+0.48056*(1-abs(k-5)+abs(1-abs(k-5)))/2+0.47622*(1-abs(k-6)+abs(1-abs(k-6)))/2+0.48064*(1-abs(k-7)+abs(1-abs(k-7)))/2+0.48080*(1-abs(k-8)+abs(1-abs(k-8)))/2+0.48064*(1-abs(k-9)+abs(1-abs(k-9)))/2+0.47565*(1-abs(k-10)+abs(1-abs(k-10)))/2+0.23251*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.23205*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.22163*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.13383*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.22988*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.15336*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.23242*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.21963*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.22168*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.17765*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.22731*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s208q pulse on channel 2 at 7027ns, square, 4ns, (0.02749*(1-abs(k-1)+abs(1-abs(k-1)))/2+-0.01654*(1-abs(k-2)+abs(1-abs(k-2)))/2+-0.04260*(1-abs(k-3)+abs(1-abs(k-3)))/2+0.05208*(1-abs(k-4)+abs(1-abs(k-4)))/2+0.01514*(1-abs(k-5)+abs(1-abs(k-5)))/2+0.06620*(1-abs(k-6)+abs(1-abs(k-6)))/2+-0.01257*(1-abs(k-7)+abs(1-abs(k-7)))/2+-0.00173*(1-abs(k-8)+abs(1-abs(k-8)))/2+-0.01232*(1-abs(k-9)+abs(1-abs(k-9)))/2+-0.07020*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.01462*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.07028*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.19012*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.03482*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.17475*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00629*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.07631*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.07011*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.14999*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.04887*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s209i pulse on channel 1 at 7031ns, square, 4ns, (0.48080*(1-abs(k-0)+abs(1-abs(k-0)))/2+0.48006*(1-abs(k-1)+abs(1-abs(k-1)))/2+0.48070*(1-abs(k-2)+abs(1-abs(k-2)))/2+0.47923*(1-abs(k-3)+abs(1-abs(k-3)))/2+0.47767*(1-abs(k-4)+abs(1-abs(k-4)))/2+0.48067*(1-abs(k-5)+abs(1-abs(k-5)))/2+0.47527*(1-abs(k-6)+abs(1-abs(k-6)))/2+0.48059*(1-abs(k-7)+abs(1-abs(k-7)))/2+0.48077*(1-abs(k-8)+abs(1-abs(k-8)))/2+0.48070*(1-abs(k-9)+abs(1-abs(k-9)))/2+0.47408*(1-abs(k-10)+abs(1-abs(k-10)))/2+0.19550*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.19497*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.18563*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.11387*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.19327*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.13017*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.19542*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.18604*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.18505*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.14829*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.19125*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s209q pulse on channel 2 at 7031ns, square, 4ns, (0.02674*(1-abs(k-1)+abs(1-abs(k-1)))/2+-0.00982*(1-abs(k-2)+abs(1-abs(k-2)))/2+-0.03882*(1-abs(k-3)+abs(1-abs(k-3)))/2+0.05474*(1-abs(k-4)+abs(1-abs(k-4)))/2+0.01115*(1-abs(k-5)+abs(1-abs(k-5)))/2+0.07269*(1-abs(k-6)+abs(1-abs(k-6)))/2+-0.01435*(1-abs(k-7)+abs(1-abs(k-7)))/2+-0.00574*(1-abs(k-8)+abs(1-abs(k-8)))/2+-0.00965*(1-abs(k-9)+abs(1-abs(k-9)))/2+-0.08008*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.01438*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.06133*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.15892*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.02943*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.14586*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00544*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.06007*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.06306*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.12739*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.04055*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s210i pulse on channel 1 at 7035ns, square, 4ns, (-0.48080*(1-abs(k-0)+abs(1-abs(k-0)))/2+-0.47980*(1-abs(k-1)+abs(1-abs(k-1)))/2+-0.48080*(1-abs(k-2)+abs(1-abs(k-2)))/2+-0.47978*(1-abs(k-3)+abs(1-abs(k-3)))/2+-0.47850*(1-abs(k-4)+abs(1-abs(k-4)))/2+-0.48068*(1-abs(k-5)+abs(1-abs(k-5)))/2+-0.47466*(1-abs(k-6)+abs(1-abs(k-6)))/2+-0.48043*(1-abs(k-7)+abs(1-abs(k-7)))/2+-0.48073*(1-abs(k-8)+abs(1-abs(k-8)))/2+-0.48073*(1-abs(k-9)+abs(1-abs(k-9)))/2+-0.47348*(1-abs(k-10)+abs(1-abs(k-10)))/2+0.15972*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.15917*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.15120*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.09369*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.15774*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.10656*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.15969*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.15254*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.15020*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.12046*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.15638*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s210q pulse on channel 2 at 7035ns, square, 4ns, (-0.03096*(1-abs(k-1)+abs(1-abs(k-1)))/2+0.00188*(1-abs(k-2)+abs(1-abs(k-2)))/2+0.03130*(1-abs(k-3)+abs(1-abs(k-3)))/2+-0.04694*(1-abs(k-4)+abs(1-abs(k-4)))/2+-0.01083*(1-abs(k-5)+abs(1-abs(k-5)))/2+-0.07658*(1-abs(k-6)+abs(1-abs(k-6)))/2+0.01897*(1-abs(k-7)+abs(1-abs(k-7)))/2+0.00806*(1-abs(k-8)+abs(1-abs(k-8)))/2+0.00800*(1-abs(k-9)+abs(1-abs(k-9)))/2+0.08357*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.01322*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.05147*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.12936*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.02509*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.11898*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00310*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.04737*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.05434*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.10488*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.03250*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s211i pulse on channel 1 at 7039ns, square, 4ns, (-0.48080*(1-abs(k-0)+abs(1-abs(k-0)))/2+-0.47908*(1-abs(k-1)+abs(1-abs(k-1)))/2+-0.48078*(1-abs(k-2)+abs(1-abs(k-2)))/2+-0.48024*(1-abs(k-3)+abs(1-abs(k-3)))/2+-0.47946*(1-abs(k-4)+abs(1-abs(k-4)))/2+-0.48071*(1-abs(k-5)+abs(1-abs(k-5)))/2+-0.47468*(1-abs(k-6)+abs(1-abs(k-6)))/2+-0.48041*(1-abs(k-7)+abs(1-abs(k-7)))/2+-0.48075*(1-abs(k-8)+abs(1-abs(k-8)))/2+-0.48067*(1-abs(k-9)+abs(1-abs(k-9)))/2+-0.47403*(1-abs(k-10)+abs(1-abs(k-10)))/2+0.12424*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.12380*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.11719*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.07357*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.12256*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.08248*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.12421*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.11868*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.11609*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.09251*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.12199*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s211q pulse on channel 2 at 7039ns, square, 4ns, (-0.04067*(1-abs(k-1)+abs(1-abs(k-1)))/2+-0.00468*(1-abs(k-2)+abs(1-abs(k-2)))/2+0.02318*(1-abs(k-3)+abs(1-abs(k-3)))/2+-0.03581*(1-abs(k-4)+abs(1-abs(k-4)))/2+-0.00945*(1-abs(k-5)+abs(1-abs(k-5)))/2+-0.07646*(1-abs(k-6)+abs(1-abs(k-6)))/2+0.01936*(1-abs(k-7)+abs(1-abs(k-7)))/2+0.00700*(1-abs(k-8)+abs(1-abs(k-8)))/2+0.01130*(1-abs(k-9)+abs(1-abs(k-9)))/2+0.08042*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.01052*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.04126*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.10012*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.02040*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.09291*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00266*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.03675*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.04425*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.08294*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.02354*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s212i pulse on channel 1 at 7043ns, square, 4ns, (-0.48080*(1-abs(k-0)+abs(1-abs(k-0)))/2+-0.47782*(1-abs(k-1)+abs(1-abs(k-1)))/2+-0.48052*(1-abs(k-2)+abs(1-abs(k-2)))/2+-0.48058*(1-abs(k-3)+abs(1-abs(k-3)))/2+-0.47990*(1-abs(k-4)+abs(1-abs(k-4)))/2+-0.48076*(1-abs(k-5)+abs(1-abs(k-5)))/2+-0.47442*(1-abs(k-6)+abs(1-abs(k-6)))/2+-0.48031*(1-abs(k-7)+abs(1-abs(k-7)))/2+-0.48075*(1-abs(k-8)+abs(1-abs(k-8)))/2+-0.48057*(1-abs(k-9)+abs(1-abs(k-9)))/2+-0.47506*(1-abs(k-10)+abs(1-abs(k-10)))/2+0.08935*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.08904*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.08377*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.05330*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.08816*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.05893*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.08932*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.08541*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.08295*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.06542*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.08794*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s212q pulse on channel 2 at 7043ns, square, 4ns, (-0.05344*(1-abs(k-1)+abs(1-abs(k-1)))/2+-0.01645*(1-abs(k-2)+abs(1-abs(k-2)))/2+0.01447*(1-abs(k-3)+abs(1-abs(k-3)))/2+-0.02943*(1-abs(k-4)+abs(1-abs(k-4)))/2+-0.00619*(1-abs(k-5)+abs(1-abs(k-5)))/2+-0.07806*(1-abs(k-6)+abs(1-abs(k-6)))/2+0.02181*(1-abs(k-7)+abs(1-abs(k-7)))/2+0.00670*(1-abs(k-8)+abs(1-abs(k-8)))/2+0.01496*(1-abs(k-9)+abs(1-abs(k-9)))/2+0.07405*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.00744*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.03109*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.07171*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.01451*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.06716*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00246*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.02623*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.03320*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.06086*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.01582*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s213i pulse on channel 1 at 7047ns, square, 4ns, (-0.48080*(1-abs(k-0)+abs(1-abs(k-0)))/2+-0.47626*(1-abs(k-1)+abs(1-abs(k-1)))/2+-0.47992*(1-abs(k-2)+abs(1-abs(k-2)))/2+-0.48079*(1-abs(k-3)+abs(1-abs(k-3)))/2+-0.47999*(1-abs(k-4)+abs(1-abs(k-4)))/2+-0.48080*(1-abs(k-5)+abs(1-abs(k-5)))/2+-0.47326*(1-abs(k-6)+abs(1-abs(k-6)))/2+-0.48016*(1-abs(k-7)+abs(1-abs(k-7)))/2+-0.48074*(1-abs(k-8)+abs(1-abs(k-8)))/2+-0.48049*(1-abs(k-9)+abs(1-abs(k-9)))/2+-0.47604*(1-abs(k-10)+abs(1-abs(k-10)))/2+0.05508*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.05493*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.05135*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.03280*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.05435*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.03583*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.05505*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.05249*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.05092*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.03952*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.05433*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s213q pulse on channel 2 at 7047ns, square, 4ns, (-0.06593*(1-abs(k-1)+abs(1-abs(k-1)))/2+-0.02915*(1-abs(k-2)+abs(1-abs(k-2)))/2+0.00344*(1-abs(k-3)+abs(1-abs(k-3)))/2+-0.02793*(1-abs(k-4)+abs(1-abs(k-4)))/2+-0.00052*(1-abs(k-5)+abs(1-abs(k-5)))/2+-0.08482*(1-abs(k-6)+abs(1-abs(k-6)))/2+0.02477*(1-abs(k-7)+abs(1-abs(k-7)))/2+0.00765*(1-abs(k-8)+abs(1-abs(k-8)))/2+0.01719*(1-abs(k-9)+abs(1-abs(k-9)))/2+0.06748*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.00404*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.01991*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.04424*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.00890*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.04183*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00167*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.01667*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.02100*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.03836*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.00903*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s214i pulse on channel 1 at 7051ns, square, 4ns, (-0.48080*(1-abs(k-0)+abs(1-abs(k-0)))/2+-0.47457*(1-abs(k-1)+abs(1-abs(k-1)))/2+-0.47915*(1-abs(k-2)+abs(1-abs(k-2)))/2+-0.48073*(1-abs(k-3)+abs(1-abs(k-3)))/2+-0.48022*(1-abs(k-4)+abs(1-abs(k-4)))/2+-0.48080*(1-abs(k-5)+abs(1-abs(k-5)))/2+-0.47110*(1-abs(k-6)+abs(1-abs(k-6)))/2+-0.48014*(1-abs(k-7)+abs(1-abs(k-7)))/2+-0.48067*(1-abs(k-8)+abs(1-abs(k-8)))/2+-0.48015*(1-abs(k-9)+abs(1-abs(k-9)))/2+-0.47653*(1-abs(k-10)+abs(1-abs(k-10)))/2+0.02109*(1-abs(k-11)+abs(1-abs(k-11)))/2+0.02105*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.01953*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.01258*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.02078*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.01349*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.02108*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.01998*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.01935*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.01494*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.02083*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s214q pulse on channel 2 at 7051ns, square, 4ns, (-0.07716*(1-abs(k-1)+abs(1-abs(k-1)))/2+-0.03980*(1-abs(k-2)+abs(1-abs(k-2)))/2+-0.00834*(1-abs(k-3)+abs(1-abs(k-3)))/2+-0.02368*(1-abs(k-4)+abs(1-abs(k-4)))/2+0.00158*(1-abs(k-5)+abs(1-abs(k-5)))/2+-0.09610*(1-abs(k-6)+abs(1-abs(k-6)))/2+0.02514*(1-abs(k-7)+abs(1-abs(k-7)))/2+0.01098*(1-abs(k-8)+abs(1-abs(k-8)))/2+0.02499*(1-abs(k-9)+abs(1-abs(k-9)))/2+0.06393*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.00133*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.00797*(1-abs(k-13)+abs(1-abs(k-13)))/2+0.01693*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.00361*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.01621*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.00080*(1-abs(k-17)+abs(1-abs(k-17)))/2+0.00674*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.00839*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.01489*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.00330*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s215i pulse on channel 1 at 7055ns, square, 4ns, (-0.48080*(1-abs(k-0)+abs(1-abs(k-0)))/2+-0.47239*(1-abs(k-1)+abs(1-abs(k-1)))/2+-0.47842*(1-abs(k-2)+abs(1-abs(k-2)))/2+-0.48051*(1-abs(k-3)+abs(1-abs(k-3)))/2+-0.48038*(1-abs(k-4)+abs(1-abs(k-4)))/2+-0.48079*(1-abs(k-5)+abs(1-abs(k-5)))/2+-0.46898*(1-abs(k-6)+abs(1-abs(k-6)))/2+-0.48004*(1-abs(k-7)+abs(1-abs(k-7)))/2+-0.48042*(1-abs(k-8)+abs(1-abs(k-8)))/2+-0.47955*(1-abs(k-9)+abs(1-abs(k-9)))/2+-0.47695*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.01264*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.01262*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.01160*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.00745*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.01243*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.00792*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.01263*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.01190*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.01153*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.00879*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.01251*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s215q pulse on channel 2 at 7055ns, square, 4ns, (-0.08955*(1-abs(k-1)+abs(1-abs(k-1)))/2+-0.04780*(1-abs(k-2)+abs(1-abs(k-2)))/2+-0.01672*(1-abs(k-3)+abs(1-abs(k-3)))/2+-0.02021*(1-abs(k-4)+abs(1-abs(k-4)))/2+0.00266*(1-abs(k-5)+abs(1-abs(k-5)))/2+-0.10594*(1-abs(k-6)+abs(1-abs(k-6)))/2+0.02708*(1-abs(k-7)+abs(1-abs(k-7)))/2+0.01901*(1-abs(k-8)+abs(1-abs(k-8)))/2+0.03464*(1-abs(k-9)+abs(1-abs(k-9)))/2+0.06073*(1-abs(k-10)+abs(1-abs(k-10)))/2+0.00069*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.00502*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.01021*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.00230*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.00985*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00054*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.00425*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.00519*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.00908*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.00183*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s216i pulse on channel 1 at 7059ns, square, 4ns, (0.48080*(1-abs(k-0)+abs(1-abs(k-0)))/2+0.47052*(1-abs(k-1)+abs(1-abs(k-1)))/2+0.47744*(1-abs(k-2)+abs(1-abs(k-2)))/2+0.48033*(1-abs(k-3)+abs(1-abs(k-3)))/2+0.48048*(1-abs(k-4)+abs(1-abs(k-4)))/2+0.48074*(1-abs(k-5)+abs(1-abs(k-5)))/2+0.46868*(1-abs(k-6)+abs(1-abs(k-6)))/2+0.48000*(1-abs(k-7)+abs(1-abs(k-7)))/2+0.48009*(1-abs(k-8)+abs(1-abs(k-8)))/2+0.47885*(1-abs(k-9)+abs(1-abs(k-9)))/2+0.47752*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.04655*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.04651*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.04249*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.02680*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.04568*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.02876*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.04650*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.04360*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.04220*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.03127*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.04611*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s216q pulse on channel 2 at 7059ns, square, 4ns, (0.09891*(1-abs(k-1)+abs(1-abs(k-1)))/2+0.05671*(1-abs(k-2)+abs(1-abs(k-2)))/2+0.02136*(1-abs(k-3)+abs(1-abs(k-3)))/2+0.01759*(1-abs(k-4)+abs(1-abs(k-4)))/2+-0.00768*(1-abs(k-5)+abs(1-abs(k-5)))/2+0.10727*(1-abs(k-6)+abs(1-abs(k-6)))/2+-0.02770*(1-abs(k-7)+abs(1-abs(k-7)))/2+-0.02609*(1-abs(k-8)+abs(1-abs(k-8)))/2+-0.04325*(1-abs(k-9)+abs(1-abs(k-9)))/2+-0.05605*(1-abs(k-10)+abs(1-abs(k-10)))/2+0.00200*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.01902*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.03806*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.00895*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.03660*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00212*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.01630*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.01965*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.03448*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.00641*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s217i pulse on channel 1 at 7063ns, square, 4ns, (0.48080*(1-abs(k-0)+abs(1-abs(k-0)))/2+0.46852*(1-abs(k-1)+abs(1-abs(k-1)))/2+0.47598*(1-abs(k-2)+abs(1-abs(k-2)))/2+0.48027*(1-abs(k-3)+abs(1-abs(k-3)))/2+0.48059*(1-abs(k-4)+abs(1-abs(k-4)))/2+0.48061*(1-abs(k-5)+abs(1-abs(k-5)))/2+0.46858*(1-abs(k-6)+abs(1-abs(k-6)))/2+0.48006*(1-abs(k-7)+abs(1-abs(k-7)))/2+0.47962*(1-abs(k-8)+abs(1-abs(k-8)))/2+0.47884*(1-abs(k-9)+abs(1-abs(k-9)))/2+0.47789*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.08076*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.08073*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.07310*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.04560*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.07913*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.05023*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.08064*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.07534*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.07268*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.05200*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.07999*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s217q pulse on channel 2 at 7063ns, square, 4ns, (0.10797*(1-abs(k-1)+abs(1-abs(k-1)))/2+0.06793*(1-abs(k-2)+abs(1-abs(k-2)))/2+0.02263*(1-abs(k-3)+abs(1-abs(k-3)))/2+0.01412*(1-abs(k-4)+abs(1-abs(k-4)))/2+-0.01358*(1-abs(k-5)+abs(1-abs(k-5)))/2+0.10769*(1-abs(k-6)+abs(1-abs(k-6)))/2+-0.02674*(1-abs(k-7)+abs(1-abs(k-7)))/2+-0.03366*(1-abs(k-8)+abs(1-abs(k-8)))/2+-0.04338*(1-abs(k-9)+abs(1-abs(k-9)))/2+-0.05279*(1-abs(k-10)+abs(1-abs(k-10)))/2+0.00200*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.03432*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.06665*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.01613*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.06324*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00432*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.02908*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.03521*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.06179*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.01111*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s218i pulse on channel 1 at 7067ns, square, 4ns, (0.48080*(1-abs(k-0)+abs(1-abs(k-0)))/2+0.46640*(1-abs(k-1)+abs(1-abs(k-1)))/2+0.47424*(1-abs(k-2)+abs(1-abs(k-2)))/2+0.47992*(1-abs(k-3)+abs(1-abs(k-3)))/2+0.48068*(1-abs(k-4)+abs(1-abs(k-4)))/2+0.48037*(1-abs(k-5)+abs(1-abs(k-5)))/2+0.46926*(1-abs(k-6)+abs(1-abs(k-6)))/2+0.48001*(1-abs(k-7)+abs(1-abs(k-7)))/2+0.47918*(1-abs(k-8)+abs(1-abs(k-8)))/2+0.47833*(1-abs(k-9)+abs(1-abs(k-9)))/2+0.47851*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.11544*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.11544*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.10407*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.06307*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.11285*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.07171*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.11523*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.10751*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.10320*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.07118*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.11417*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s218q pulse on channel 2 at 7067ns, square, 4ns, (0.11680*(1-abs(k-1)+abs(1-abs(k-1)))/2+0.07918*(1-abs(k-2)+abs(1-abs(k-2)))/2+0.02901*(1-abs(k-3)+abs(1-abs(k-3)))/2+0.01087*(1-abs(k-4)+abs(1-abs(k-4)))/2+-0.02039*(1-abs(k-5)+abs(1-abs(k-5)))/2+0.10469*(1-abs(k-6)+abs(1-abs(k-6)))/2+-0.02747*(1-abs(k-7)+abs(1-abs(k-7)))/2+-0.03942*(1-abs(k-8)+abs(1-abs(k-8)))/2+-0.04869*(1-abs(k-9)+abs(1-abs(k-9)))/2+-0.04692*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.00036*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.04997*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.09669*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.02433*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.09047*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00691*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.04204*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.05174*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.09088*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.01706*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s219i pulse on channel 1 at 7071ns, square, 4ns, (0.48080*(1-abs(k-0)+abs(1-abs(k-0)))/2+0.46488*(1-abs(k-1)+abs(1-abs(k-1)))/2+0.47210*(1-abs(k-2)+abs(1-abs(k-2)))/2+0.47897*(1-abs(k-3)+abs(1-abs(k-3)))/2+0.48069*(1-abs(k-4)+abs(1-abs(k-4)))/2+0.47978*(1-abs(k-5)+abs(1-abs(k-5)))/2+0.47049*(1-abs(k-6)+abs(1-abs(k-6)))/2+0.47983*(1-abs(k-7)+abs(1-abs(k-7)))/2+0.47817*(1-abs(k-8)+abs(1-abs(k-8)))/2+0.47724*(1-abs(k-9)+abs(1-abs(k-9)))/2+0.47892*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.15082*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.15070*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.13591*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.08013*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.14709*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.09364*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.15051*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.14109*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.13362*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.08865*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.14908*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s219q pulse on channel 2 at 7071ns, square, 4ns, (0.12270*(1-abs(k-1)+abs(1-abs(k-1)))/2+0.09103*(1-abs(k-2)+abs(1-abs(k-2)))/2+0.04190*(1-abs(k-3)+abs(1-abs(k-3)))/2+0.01031*(1-abs(k-4)+abs(1-abs(k-4)))/2+-0.03123*(1-abs(k-5)+abs(1-abs(k-5)))/2+0.09905*(1-abs(k-6)+abs(1-abs(k-6)))/2+-0.03053*(1-abs(k-7)+abs(1-abs(k-7)))/2+-0.05021*(1-abs(k-8)+abs(1-abs(k-8)))/2+-0.05838*(1-abs(k-9)+abs(1-abs(k-9)))/2+-0.04244*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.00583*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.06537*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.12777*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.03331*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.11822*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00952*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.05327*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.06994*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.12201*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.02281*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s220i pulse on channel 1 at 7075ns, square, 4ns, (0.48080*(1-abs(k-0)+abs(1-abs(k-0)))/2+0.46502*(1-abs(k-1)+abs(1-abs(k-1)))/2+0.47007*(1-abs(k-2)+abs(1-abs(k-2)))/2+0.47744*(1-abs(k-3)+abs(1-abs(k-3)))/2+0.48066*(1-abs(k-4)+abs(1-abs(k-4)))/2+0.47879*(1-abs(k-5)+abs(1-abs(k-5)))/2+0.47206*(1-abs(k-6)+abs(1-abs(k-6)))/2+0.47927*(1-abs(k-7)+abs(1-abs(k-7)))/2+0.47693*(1-abs(k-8)+abs(1-abs(k-8)))/2+0.47609*(1-abs(k-9)+abs(1-abs(k-9)))/2+0.47917*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.18654*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.18615*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.16812*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.09608*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.18188*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.11516*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.18611*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.17544*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.16372*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.10419*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.18433*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s220q pulse on channel 2 at 7075ns, square, 4ns, (0.12218*(1-abs(k-1)+abs(1-abs(k-1)))/2+0.10101*(1-abs(k-2)+abs(1-abs(k-2)))/2+0.05670*(1-abs(k-3)+abs(1-abs(k-3)))/2+0.01147*(1-abs(k-4)+abs(1-abs(k-4)))/2+-0.04392*(1-abs(k-5)+abs(1-abs(k-5)))/2+0.09126*(1-abs(k-6)+abs(1-abs(k-6)))/2+-0.03831*(1-abs(k-7)+abs(1-abs(k-7)))/2+-0.06085*(1-abs(k-8)+abs(1-abs(k-8)))/2+-0.06715*(1-abs(k-9)+abs(1-abs(k-9)))/2+-0.03955*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.01210*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.08082*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.15989*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.04142*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.14675*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.01270*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.06338*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.08940*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.15473*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.02864*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s221i pulse on channel 1 at 7079ns, square, 4ns, (0.48080*(1-abs(k-0)+abs(1-abs(k-0)))/2+0.46509*(1-abs(k-1)+abs(1-abs(k-1)))/2+0.46844*(1-abs(k-2)+abs(1-abs(k-2)))/2+0.47589*(1-abs(k-3)+abs(1-abs(k-3)))/2+0.48072*(1-abs(k-4)+abs(1-abs(k-4)))/2+0.47772*(1-abs(k-5)+abs(1-abs(k-5)))/2+0.47271*(1-abs(k-6)+abs(1-abs(k-6)))/2+0.47877*(1-abs(k-7)+abs(1-abs(k-7)))/2+0.47566*(1-abs(k-8)+abs(1-abs(k-8)))/2+0.47481*(1-abs(k-9)+abs(1-abs(k-9)))/2+0.47978*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.22319*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.22225*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.20167*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.10962*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.21763*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.13737*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.22272*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.21101*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.19441*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.11862*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.22070*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s221q pulse on channel 2 at 7079ns, square, 4ns, (0.12189*(1-abs(k-1)+abs(1-abs(k-1)))/2+0.10831*(1-abs(k-2)+abs(1-abs(k-2)))/2+0.06852*(1-abs(k-3)+abs(1-abs(k-3)))/2+0.00891*(1-abs(k-4)+abs(1-abs(k-4)))/2+-0.05437*(1-abs(k-5)+abs(1-abs(k-5)))/2+0.08783*(1-abs(k-6)+abs(1-abs(k-6)))/2+-0.04414*(1-abs(k-7)+abs(1-abs(k-7)))/2+-0.07010*(1-abs(k-8)+abs(1-abs(k-8)))/2+-0.07566*(1-abs(k-9)+abs(1-abs(k-9)))/2+-0.03134*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.02045*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.09562*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.19441*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.04953*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.17591*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.01453*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.07274*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.10963*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.18906*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.03327*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s222i pulse on channel 1 at 7083ns, square, 4ns, (-0.48080*(1-abs(k-0)+abs(1-abs(k-0)))/2+-0.46482*(1-abs(k-1)+abs(1-abs(k-1)))/2+-0.46712*(1-abs(k-2)+abs(1-abs(k-2)))/2+-0.47373*(1-abs(k-3)+abs(1-abs(k-3)))/2+-0.48072*(1-abs(k-4)+abs(1-abs(k-4)))/2+-0.47643*(1-abs(k-5)+abs(1-abs(k-5)))/2+-0.47366*(1-abs(k-6)+abs(1-abs(k-6)))/2+-0.47827*(1-abs(k-7)+abs(1-abs(k-7)))/2+-0.47400*(1-abs(k-8)+abs(1-abs(k-8)))/2+-0.47324*(1-abs(k-9)+abs(1-abs(k-9)))/2+-0.48031*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.26046*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.25892*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.23625*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.12365*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.25391*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.15935*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.25988*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.24764*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.22556*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.13367*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.25770*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s222q pulse on channel 2 at 7083ns, square, 4ns, (-0.12291*(1-abs(k-1)+abs(1-abs(k-1)))/2+-0.11388*(1-abs(k-2)+abs(1-abs(k-2)))/2+-0.08216*(1-abs(k-3)+abs(1-abs(k-3)))/2+-0.00879*(1-abs(k-4)+abs(1-abs(k-4)))/2+0.06469*(1-abs(k-5)+abs(1-abs(k-5)))/2+-0.08254*(1-abs(k-6)+abs(1-abs(k-6)))/2+0.04924*(1-abs(k-7)+abs(1-abs(k-7)))/2+0.08057*(1-abs(k-8)+abs(1-abs(k-8)))/2+0.08495*(1-abs(k-9)+abs(1-abs(k-9)))/2+0.02175*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.02826*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.10967*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.22924*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.05806*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.20603*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.01731*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.08071*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.13023*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.22354*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.03780*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s223i pulse on channel 1 at 7087ns, square, 4ns, (-0.48080*(1-abs(k-0)+abs(1-abs(k-0)))/2+-0.46396*(1-abs(k-1)+abs(1-abs(k-1)))/2+-0.46696*(1-abs(k-2)+abs(1-abs(k-2)))/2+-0.47151*(1-abs(k-3)+abs(1-abs(k-3)))/2+-0.48074*(1-abs(k-4)+abs(1-abs(k-4)))/2+-0.47481*(1-abs(k-5)+abs(1-abs(k-5)))/2+-0.47442*(1-abs(k-6)+abs(1-abs(k-6)))/2+-0.47789*(1-abs(k-7)+abs(1-abs(k-7)))/2+-0.47259*(1-abs(k-8)+abs(1-abs(k-8)))/2+-0.47040*(1-abs(k-9)+abs(1-abs(k-9)))/2+-0.48065*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.29739*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.29536*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.27106*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.13721*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.28969*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.18185*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.29659*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.28287*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.25758*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.14703*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.29446*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s223q pulse on channel 2 at 7087ns, square, 4ns, (-0.12613*(1-abs(k-1)+abs(1-abs(k-1)))/2+-0.11453*(1-abs(k-2)+abs(1-abs(k-2)))/2+-0.09404*(1-abs(k-3)+abs(1-abs(k-3)))/2+-0.00728*(1-abs(k-4)+abs(1-abs(k-4)))/2+0.07563*(1-abs(k-5)+abs(1-abs(k-5)))/2+-0.07805*(1-abs(k-6)+abs(1-abs(k-6)))/2+0.05284*(1-abs(k-7)+abs(1-abs(k-7)))/2+0.08847*(1-abs(k-8)+abs(1-abs(k-8)))/2+0.09948*(1-abs(k-9)+abs(1-abs(k-9)))/2+0.01203*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.03472*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.12233*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.26384*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.06721*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.23531*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.02179*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.09179*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.14863*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.25850*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.04166*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s224i pulse on channel 1 at 7091ns, square, 4ns, (-0.48080*(1-abs(k-0)+abs(1-abs(k-0)))/2+-0.46282*(1-abs(k-1)+abs(1-abs(k-1)))/2+-0.46711*(1-abs(k-2)+abs(1-abs(k-2)))/2+-0.47033*(1-abs(k-3)+abs(1-abs(k-3)))/2+-0.48076*(1-abs(k-4)+abs(1-abs(k-4)))/2+-0.47333*(1-abs(k-5)+abs(1-abs(k-5)))/2+-0.47477*(1-abs(k-6)+abs(1-abs(k-6)))/2+-0.47750*(1-abs(k-7)+abs(1-abs(k-7)))/2+-0.47209*(1-abs(k-8)+abs(1-abs(k-8)))/2+-0.46786*(1-abs(k-9)+abs(1-abs(k-9)))/2+-0.48076*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.33484*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.33276*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.30613*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.15202*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.32627*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.20555*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.33401*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.31779*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.29039*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.15952*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.33147*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s224q pulse on channel 2 at 7091ns, square, 4ns, (-0.13025*(1-abs(k-1)+abs(1-abs(k-1)))/2+-0.11392*(1-abs(k-2)+abs(1-abs(k-2)))/2+-0.09980*(1-abs(k-3)+abs(1-abs(k-3)))/2+-0.00629*(1-abs(k-4)+abs(1-abs(k-4)))/2+0.08440*(1-abs(k-5)+abs(1-abs(k-5)))/2+-0.07591*(1-abs(k-6)+abs(1-abs(k-6)))/2+0.05621*(1-abs(k-7)+abs(1-abs(k-7)))/2+0.09113*(1-abs(k-8)+abs(1-abs(k-8)))/2+0.11078*(1-abs(k-9)+abs(1-abs(k-9)))/2+0.00654*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.03725*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.13564*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.29834*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.07528*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.26432*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.02348*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.10546*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.16670*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.29440*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.04738*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s225i pulse on channel 1 at 7095ns, square, 4ns, (-0.48080*(1-abs(k-0)+abs(1-abs(k-0)))/2+-0.46136*(1-abs(k-1)+abs(1-abs(k-1)))/2+-0.46663*(1-abs(k-2)+abs(1-abs(k-2)))/2+-0.47008*(1-abs(k-3)+abs(1-abs(k-3)))/2+-0.48076*(1-abs(k-4)+abs(1-abs(k-4)))/2+-0.47283*(1-abs(k-5)+abs(1-abs(k-5)))/2+-0.47557*(1-abs(k-6)+abs(1-abs(k-6)))/2+-0.47721*(1-abs(k-7)+abs(1-abs(k-7)))/2+-0.47139*(1-abs(k-8)+abs(1-abs(k-8)))/2+-0.46516*(1-abs(k-9)+abs(1-abs(k-9)))/2+-0.48080*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.37070*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.36828*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.33961*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.16801*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.36048*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.22953*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.36974*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.35253*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.32325*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.17198*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.36643*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s225q pulse on channel 2 at 7095ns, square, 4ns, (-0.13532*(1-abs(k-1)+abs(1-abs(k-1)))/2+-0.11586*(1-abs(k-2)+abs(1-abs(k-2)))/2+-0.10096*(1-abs(k-3)+abs(1-abs(k-3)))/2+-0.00622*(1-abs(k-4)+abs(1-abs(k-4)))/2+0.08720*(1-abs(k-5)+abs(1-abs(k-5)))/2+-0.07071*(1-abs(k-6)+abs(1-abs(k-6)))/2+0.05865*(1-abs(k-7)+abs(1-abs(k-7)))/2+0.09464*(1-abs(k-8)+abs(1-abs(k-8)))/2+0.12164*(1-abs(k-9)+abs(1-abs(k-9)))/2+0.00004*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.04237*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.14862*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.33045*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.08647*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.29110*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.02678*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.11466*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.18148*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.32840*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.05612*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s226i pulse on channel 1 at 7099ns, square, 4ns, (-0.48080*(1-abs(k-0)+abs(1-abs(k-0)))/2+-0.45871*(1-abs(k-1)+abs(1-abs(k-1)))/2+-0.46613*(1-abs(k-2)+abs(1-abs(k-2)))/2+-0.46971*(1-abs(k-3)+abs(1-abs(k-3)))/2+-0.48064*(1-abs(k-4)+abs(1-abs(k-4)))/2+-0.47219*(1-abs(k-5)+abs(1-abs(k-5)))/2+-0.47618*(1-abs(k-6)+abs(1-abs(k-6)))/2+-0.47731*(1-abs(k-7)+abs(1-abs(k-7)))/2+-0.47040*(1-abs(k-8)+abs(1-abs(k-8)))/2+-0.46335*(1-abs(k-9)+abs(1-abs(k-9)))/2+-0.48075*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.40367*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.40080*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.37074*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.18533*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.39206*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.25219*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.40244*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.38484*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.35241*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.18265*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.39844*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s226q pulse on channel 2 at 7099ns, square, 4ns, (-0.14406*(1-abs(k-1)+abs(1-abs(k-1)))/2+-0.11788*(1-abs(k-2)+abs(1-abs(k-2)))/2+-0.10269*(1-abs(k-3)+abs(1-abs(k-3)))/2+-0.01259*(1-abs(k-4)+abs(1-abs(k-4)))/2+0.09060*(1-abs(k-5)+abs(1-abs(k-5)))/2+-0.06648*(1-abs(k-6)+abs(1-abs(k-6)))/2+0.05782*(1-abs(k-7)+abs(1-abs(k-7)))/2+0.09948*(1-abs(k-8)+abs(1-abs(k-8)))/2+0.12835*(1-abs(k-9)+abs(1-abs(k-9)))/2+-0.00674*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.04802*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.15969*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.35861*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.09610*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.31519*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.03141*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.12185*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.19686*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.35998*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.06476*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s227i pulse on channel 1 at 7103ns, square, 4ns, (-0.48080*(1-abs(k-0)+abs(1-abs(k-0)))/2+-0.45641*(1-abs(k-1)+abs(1-abs(k-1)))/2+-0.46532*(1-abs(k-2)+abs(1-abs(k-2)))/2+-0.46921*(1-abs(k-3)+abs(1-abs(k-3)))/2+-0.48039*(1-abs(k-4)+abs(1-abs(k-4)))/2+-0.47131*(1-abs(k-5)+abs(1-abs(k-5)))/2+-0.47622*(1-abs(k-6)+abs(1-abs(k-6)))/2+-0.47786*(1-abs(k-7)+abs(1-abs(k-7)))/2+-0.46945*(1-abs(k-8)+abs(1-abs(k-8)))/2+-0.46090*(1-abs(k-9)+abs(1-abs(k-9)))/2+-0.48067*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.43309*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.43003*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.39859*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.20086*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.41937*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.27240*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.43129*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.41305*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.37969*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.18663*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.42722*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s227q pulse on channel 2 at 7103ns, square, 4ns, (-0.15119*(1-abs(k-1)+abs(1-abs(k-1)))/2+-0.12103*(1-abs(k-2)+abs(1-abs(k-2)))/2+-0.10492*(1-abs(k-3)+abs(1-abs(k-3)))/2+-0.01992*(1-abs(k-4)+abs(1-abs(k-4)))/2+0.09507*(1-abs(k-5)+abs(1-abs(k-5)))/2+-0.06618*(1-abs(k-6)+abs(1-abs(k-6)))/2+0.05306*(1-abs(k-7)+abs(1-abs(k-7)))/2+0.10387*(1-abs(k-8)+abs(1-abs(k-8)))/2+0.13689*(1-abs(k-9)+abs(1-abs(k-9)))/2+-0.01117*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.05133*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.16939*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.38369*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.10813*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.33670*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.03943*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.13021*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.20832*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.39081*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.07102*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s228i pulse on channel 1 at 7107ns, square, 4ns, (0.48080*(1-abs(k-0)+abs(1-abs(k-0)))/2+0.45557*(1-abs(k-1)+abs(1-abs(k-1)))/2+0.46379*(1-abs(k-2)+abs(1-abs(k-2)))/2+0.46961*(1-abs(k-3)+abs(1-abs(k-3)))/2+0.48004*(1-abs(k-4)+abs(1-abs(k-4)))/2+0.47136*(1-abs(k-5)+abs(1-abs(k-5)))/2+0.47640*(1-abs(k-6)+abs(1-abs(k-6)))/2+0.47847*(1-abs(k-7)+abs(1-abs(k-7)))/2+0.46853*(1-abs(k-8)+abs(1-abs(k-8)))/2+0.45814*(1-abs(k-9)+abs(1-abs(k-9)))/2+0.48060*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.45607*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.45342*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.41861*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.21553*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.44070*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.28955*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.45362*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.43457*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.40059*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.18144*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.44923*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s228q pulse on channel 2 at 7107ns, square, 4ns, (0.15369*(1-abs(k-1)+abs(1-abs(k-1)))/2+0.12676*(1-abs(k-2)+abs(1-abs(k-2)))/2+0.10313*(1-abs(k-3)+abs(1-abs(k-3)))/2+0.02701*(1-abs(k-4)+abs(1-abs(k-4)))/2+-0.09481*(1-abs(k-5)+abs(1-abs(k-5)))/2+0.06489*(1-abs(k-6)+abs(1-abs(k-6)))/2+-0.04732*(1-abs(k-7)+abs(1-abs(k-7)))/2+-0.10794*(1-abs(k-8)+abs(1-abs(k-8)))/2+-0.14588*(1-abs(k-9)+abs(1-abs(k-9)))/2+0.01395*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.04914*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.18103*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.40193*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.11743*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.35237*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.04724*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.13838*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.21801*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.41843*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.07870*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s229i pulse on channel 1 at 7111ns, square, 4ns, (0.48080*(1-abs(k-0)+abs(1-abs(k-0)))/2+0.45661*(1-abs(k-1)+abs(1-abs(k-1)))/2+0.46256*(1-abs(k-2)+abs(1-abs(k-2)))/2+0.46997*(1-abs(k-3)+abs(1-abs(k-3)))/2+0.47979*(1-abs(k-4)+abs(1-abs(k-4)))/2+0.47202*(1-abs(k-5)+abs(1-abs(k-5)))/2+0.47599*(1-abs(k-6)+abs(1-abs(k-6)))/2+0.47918*(1-abs(k-7)+abs(1-abs(k-7)))/2+0.46798*(1-abs(k-8)+abs(1-abs(k-8)))/2+0.45458*(1-abs(k-9)+abs(1-abs(k-9)))/2+0.48058*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.47078*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.46862*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.43056*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.22546*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.45546*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.29896*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.46799*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.44830*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.41547*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.16934*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.46202*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s229q pulse on channel 2 at 7111ns, square, 4ns, (0.15059*(1-abs(k-1)+abs(1-abs(k-1)))/2+0.13118*(1-abs(k-2)+abs(1-abs(k-2)))/2+0.10146*(1-abs(k-3)+abs(1-abs(k-3)))/2+0.03117*(1-abs(k-4)+abs(1-abs(k-4)))/2+-0.09146*(1-abs(k-5)+abs(1-abs(k-5)))/2+0.06782*(1-abs(k-6)+abs(1-abs(k-6)))/2+-0.03941*(1-abs(k-7)+abs(1-abs(k-7)))/2+-0.11028*(1-abs(k-8)+abs(1-abs(k-8)))/2+-0.15661*(1-abs(k-9)+abs(1-abs(k-9)))/2+0.01455*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.04513*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.19041*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.41329*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.11915*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.36368*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.05125*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.14374*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.22141*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.43928*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.09042*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s230i pulse on channel 1 at 7115ns, square, 4ns, (0.48080*(1-abs(k-0)+abs(1-abs(k-0)))/2+0.45806*(1-abs(k-1)+abs(1-abs(k-1)))/2+0.45926*(1-abs(k-2)+abs(1-abs(k-2)))/2+0.46895*(1-abs(k-3)+abs(1-abs(k-3)))/2+0.47951*(1-abs(k-4)+abs(1-abs(k-4)))/2+0.47258*(1-abs(k-5)+abs(1-abs(k-5)))/2+0.47556*(1-abs(k-6)+abs(1-abs(k-6)))/2+0.47977*(1-abs(k-7)+abs(1-abs(k-7)))/2+0.46702*(1-abs(k-8)+abs(1-abs(k-8)))/2+0.45117*(1-abs(k-9)+abs(1-abs(k-9)))/2+0.48064*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.47586*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.47386*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.43477*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.22919*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.46172*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.30207*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.47256*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.45339*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.42103*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.15704*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.46578*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s230q pulse on channel 2 at 7115ns, square, 4ns, (0.14611*(1-abs(k-1)+abs(1-abs(k-1)))/2+0.14229*(1-abs(k-2)+abs(1-abs(k-2)))/2+0.10610*(1-abs(k-3)+abs(1-abs(k-3)))/2+0.03519*(1-abs(k-4)+abs(1-abs(k-4)))/2+-0.08850*(1-abs(k-5)+abs(1-abs(k-5)))/2+0.07077*(1-abs(k-6)+abs(1-abs(k-6)))/2+-0.03139*(1-abs(k-7)+abs(1-abs(k-7)))/2+-0.11428*(1-abs(k-8)+abs(1-abs(k-8)))/2+-0.16616*(1-abs(k-9)+abs(1-abs(k-9)))/2+0.01231*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.04358*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.19343*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.41703*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.11513*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.36769*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.05592*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.14450*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.22176*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.44920*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.09741*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s231i pulse on channel 1 at 7119ns, square, 4ns, (0.48080*(1-abs(k-0)+abs(1-abs(k-0)))/2+0.46141*(1-abs(k-1)+abs(1-abs(k-1)))/2+0.45692*(1-abs(k-2)+abs(1-abs(k-2)))/2+0.46744*(1-abs(k-3)+abs(1-abs(k-3)))/2+0.47930*(1-abs(k-4)+abs(1-abs(k-4)))/2+0.47287*(1-abs(k-5)+abs(1-abs(k-5)))/2+0.47595*(1-abs(k-6)+abs(1-abs(k-6)))/2+0.48019*(1-abs(k-7)+abs(1-abs(k-7)))/2+0.46634*(1-abs(k-8)+abs(1-abs(k-8)))/2+0.44849*(1-abs(k-9)+abs(1-abs(k-9)))/2+0.48071*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.47078*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.46855*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.42879*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.22866*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.45826*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.29524*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.46741*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.44958*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.41783*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.14808*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.45985*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s231q pulse on channel 2 at 7119ns, square, 4ns, (0.13515*(1-abs(k-1)+abs(1-abs(k-1)))/2+0.14964*(1-abs(k-2)+abs(1-abs(k-2)))/2+0.11255*(1-abs(k-3)+abs(1-abs(k-3)))/2+0.03797*(1-abs(k-4)+abs(1-abs(k-4)))/2+-0.08696*(1-abs(k-5)+abs(1-abs(k-5)))/2+0.06812*(1-abs(k-6)+abs(1-abs(k-6)))/2+-0.02425*(1-abs(k-7)+abs(1-abs(k-7)))/2+-0.11703*(1-abs(k-8)+abs(1-abs(k-8)))/2+-0.17327*(1-abs(k-9)+abs(1-abs(k-9)))/2+0.00942*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.04583*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.19437*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.41152*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.10787*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.36671*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.05623*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.13969*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.21692*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.44689*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.10086*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s232i pulse on channel 1 at 7123ns, square, 4ns, (0.48080*(1-abs(k-0)+abs(1-abs(k-0)))/2+0.46416*(1-abs(k-1)+abs(1-abs(k-1)))/2+0.45548*(1-abs(k-2)+abs(1-abs(k-2)))/2+0.46645*(1-abs(k-3)+abs(1-abs(k-3)))/2+0.47848*(1-abs(k-4)+abs(1-abs(k-4)))/2+0.47298*(1-abs(k-5)+abs(1-abs(k-5)))/2+0.47681*(1-abs(k-6)+abs(1-abs(k-6)))/2+0.48059*(1-abs(k-7)+abs(1-abs(k-7)))/2+0.46737*(1-abs(k-8)+abs(1-abs(k-8)))/2+0.44516*(1-abs(k-9)+abs(1-abs(k-9)))/2+0.48079*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.45607*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.45368*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.41457*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.22635*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.44556*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.28219*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.45302*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.43656*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.40635*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.13882*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.44445*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s232q pulse on channel 2 at 7123ns, square, 4ns, (0.12539*(1-abs(k-1)+abs(1-abs(k-1)))/2+0.15398*(1-abs(k-2)+abs(1-abs(k-2)))/2+0.11660*(1-abs(k-3)+abs(1-abs(k-3)))/2+0.04719*(1-abs(k-4)+abs(1-abs(k-4)))/2+-0.08639*(1-abs(k-5)+abs(1-abs(k-5)))/2+0.06184*(1-abs(k-6)+abs(1-abs(k-6)))/2+-0.01427*(1-abs(k-7)+abs(1-abs(k-7)))/2+-0.11283*(1-abs(k-8)+abs(1-abs(k-8)))/2+-0.18167*(1-abs(k-9)+abs(1-abs(k-9)))/2+0.00280*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.04661*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.19009*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.39594*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.09735*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.35829*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.05268*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.13198*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.20707*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.43443*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.10229*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s233i pulse on channel 1 at 7127ns, square, 4ns, (0.48080*(1-abs(k-0)+abs(1-abs(k-0)))/2+0.46578*(1-abs(k-1)+abs(1-abs(k-1)))/2+0.45371*(1-abs(k-2)+abs(1-abs(k-2)))/2+0.46604*(1-abs(k-3)+abs(1-abs(k-3)))/2+0.47740*(1-abs(k-4)+abs(1-abs(k-4)))/2+0.47385*(1-abs(k-5)+abs(1-abs(k-5)))/2+0.47803*(1-abs(k-6)+abs(1-abs(k-6)))/2+0.48080*(1-abs(k-7)+abs(1-abs(k-7)))/2+0.46773*(1-abs(k-8)+abs(1-abs(k-8)))/2+0.44325*(1-abs(k-9)+abs(1-abs(k-9)))/2+0.48075*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.43309*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.43087*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.39310*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.21624*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.42432*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.26774*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.42982*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.41531*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.38842*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.12795*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.42144*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s233q pulse on channel 2 at 7127ns, square, 4ns, (0.11923*(1-abs(k-1)+abs(1-abs(k-1)))/2+0.15910*(1-abs(k-2)+abs(1-abs(k-2)))/2+0.11822*(1-abs(k-3)+abs(1-abs(k-3)))/2+0.05712*(1-abs(k-4)+abs(1-abs(k-4)))/2+-0.08147*(1-abs(k-5)+abs(1-abs(k-5)))/2+0.05155*(1-abs(k-6)+abs(1-abs(k-6)))/2+0.00021*(1-abs(k-7)+abs(1-abs(k-7)))/2+-0.11136*(1-abs(k-8)+abs(1-abs(k-8)))/2+-0.18628*(1-abs(k-9)+abs(1-abs(k-9)))/2+-0.00705*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.04375*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.18175*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.37524*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.08670*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.34041*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.05309*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.12283*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.19156*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.41375*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.09976*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s234i pulse on channel 1 at 7131ns, square, 4ns, (-0.48080*(1-abs(k-0)+abs(1-abs(k-0)))/2+-0.46609*(1-abs(k-1)+abs(1-abs(k-1)))/2+-0.45308*(1-abs(k-2)+abs(1-abs(k-2)))/2+-0.46602*(1-abs(k-3)+abs(1-abs(k-3)))/2+-0.47626*(1-abs(k-4)+abs(1-abs(k-4)))/2+-0.47478*(1-abs(k-5)+abs(1-abs(k-5)))/2+-0.47909*(1-abs(k-6)+abs(1-abs(k-6)))/2+-0.48068*(1-abs(k-7)+abs(1-abs(k-7)))/2+-0.46780*(1-abs(k-8)+abs(1-abs(k-8)))/2+-0.44123*(1-abs(k-9)+abs(1-abs(k-9)))/2+-0.48056*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.40367*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.40176*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.36692*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.20469*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.39635*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.24990*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.40020*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.38731*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.36439*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.11335*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.39284*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s234q pulse on channel 2 at 7131ns, square, 4ns, (-0.11803*(1-abs(k-1)+abs(1-abs(k-1)))/2+-0.16088*(1-abs(k-2)+abs(1-abs(k-2)))/2+-0.11830*(1-abs(k-3)+abs(1-abs(k-3)))/2+-0.06594*(1-abs(k-4)+abs(1-abs(k-4)))/2+0.07587*(1-abs(k-5)+abs(1-abs(k-5)))/2+-0.04049*(1-abs(k-6)+abs(1-abs(k-6)))/2+-0.01058*(1-abs(k-7)+abs(1-abs(k-7)))/2+0.11106*(1-abs(k-8)+abs(1-abs(k-8)))/2+0.19102*(1-abs(k-9)+abs(1-abs(k-9)))/2+0.01533*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.03915*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.16828*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.34792*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.07649*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.31702*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.05277*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.11374*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.17369*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.38743*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.09288*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s235i pulse on channel 1 at 7135ns, square, 4ns, (-0.48080*(1-abs(k-0)+abs(1-abs(k-0)))/2+-0.46672*(1-abs(k-1)+abs(1-abs(k-1)))/2+-0.45182*(1-abs(k-2)+abs(1-abs(k-2)))/2+-0.46533*(1-abs(k-3)+abs(1-abs(k-3)))/2+-0.47581*(1-abs(k-4)+abs(1-abs(k-4)))/2+-0.47576*(1-abs(k-5)+abs(1-abs(k-5)))/2+-0.47988*(1-abs(k-6)+abs(1-abs(k-6)))/2+-0.48045*(1-abs(k-7)+abs(1-abs(k-7)))/2+-0.46836*(1-abs(k-8)+abs(1-abs(k-8)))/2+-0.43938*(1-abs(k-9)+abs(1-abs(k-9)))/2+-0.48015*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.37070*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.36913*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.33737*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.18959*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.36375*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.23028*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.36712*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.35527*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.33633*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.10353*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.36072*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s235q pulse on channel 2 at 7135ns, square, 4ns, (-0.11549*(1-abs(k-1)+abs(1-abs(k-1)))/2+-0.16440*(1-abs(k-2)+abs(1-abs(k-2)))/2+-0.12097*(1-abs(k-3)+abs(1-abs(k-3)))/2+-0.06907*(1-abs(k-4)+abs(1-abs(k-4)))/2+0.06946*(1-abs(k-5)+abs(1-abs(k-5)))/2+-0.02974*(1-abs(k-6)+abs(1-abs(k-6)))/2+-0.01834*(1-abs(k-7)+abs(1-abs(k-7)))/2+0.10866*(1-abs(k-8)+abs(1-abs(k-8)))/2+0.19522*(1-abs(k-9)+abs(1-abs(k-9)))/2+0.02501*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.03416*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.15362*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.31856*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.07149*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.29050*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.05143*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.10585*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.15590*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.35595*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.08547*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s236i pulse on channel 1 at 7139ns, square, 4ns, (-0.48080*(1-abs(k-0)+abs(1-abs(k-0)))/2+-0.46747*(1-abs(k-1)+abs(1-abs(k-1)))/2+-0.44935*(1-abs(k-2)+abs(1-abs(k-2)))/2+-0.46469*(1-abs(k-3)+abs(1-abs(k-3)))/2+-0.47541*(1-abs(k-4)+abs(1-abs(k-4)))/2+-0.47617*(1-abs(k-5)+abs(1-abs(k-5)))/2+-0.48047*(1-abs(k-6)+abs(1-abs(k-6)))/2+-0.48030*(1-abs(k-7)+abs(1-abs(k-7)))/2+-0.46871*(1-abs(k-8)+abs(1-abs(k-8)))/2+-0.43726*(1-abs(k-9)+abs(1-abs(k-9)))/2+-0.47933*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.33484*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.33351*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.30582*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.17005*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.32787*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.20826*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.33127*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.31966*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.30611*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.08943*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.32512*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s236q pulse on channel 2 at 7139ns, square, 4ns, (-0.11244*(1-abs(k-1)+abs(1-abs(k-1)))/2+-0.17105*(1-abs(k-2)+abs(1-abs(k-2)))/2+-0.12341*(1-abs(k-3)+abs(1-abs(k-3)))/2+-0.07181*(1-abs(k-4)+abs(1-abs(k-4)))/2+0.06655*(1-abs(k-5)+abs(1-abs(k-5)))/2+-0.01776*(1-abs(k-6)+abs(1-abs(k-6)))/2+-0.02202*(1-abs(k-7)+abs(1-abs(k-7)))/2+0.10715*(1-abs(k-8)+abs(1-abs(k-8)))/2+0.19993*(1-abs(k-9)+abs(1-abs(k-9)))/2+0.03755*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.02979*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.13636*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.28844*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.06796*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.26219*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.04874*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.09965*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.13570*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.32267*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.08008*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s237i pulse on channel 1 at 7143ns, square, 4ns, (-0.48080*(1-abs(k-0)+abs(1-abs(k-0)))/2+-0.46837*(1-abs(k-1)+abs(1-abs(k-1)))/2+-0.44717*(1-abs(k-2)+abs(1-abs(k-2)))/2+-0.46309*(1-abs(k-3)+abs(1-abs(k-3)))/2+-0.47549*(1-abs(k-4)+abs(1-abs(k-4)))/2+-0.47598*(1-abs(k-5)+abs(1-abs(k-5)))/2+-0.48077*(1-abs(k-6)+abs(1-abs(k-6)))/2+-0.48003*(1-abs(k-7)+abs(1-abs(k-7)))/2+-0.46812*(1-abs(k-8)+abs(1-abs(k-8)))/2+-0.43514*(1-abs(k-9)+abs(1-abs(k-9)))/2+-0.47826*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.29739*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.29648*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.27144*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.15074*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.28980*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.18219*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.29390*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.28215*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.27359*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.07456*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.28806*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s237q pulse on channel 2 at 7143ns, square, 4ns, (-0.10862*(1-abs(k-1)+abs(1-abs(k-1)))/2+-0.17665*(1-abs(k-2)+abs(1-abs(k-2)))/2+-0.12928*(1-abs(k-3)+abs(1-abs(k-3)))/2+-0.07125*(1-abs(k-4)+abs(1-abs(k-4)))/2+0.06793*(1-abs(k-5)+abs(1-abs(k-5)))/2+-0.00498*(1-abs(k-6)+abs(1-abs(k-6)))/2+-0.02720*(1-abs(k-7)+abs(1-abs(k-7)))/2+0.10971*(1-abs(k-8)+abs(1-abs(k-8)))/2+0.20450*(1-abs(k-9)+abs(1-abs(k-9)))/2+0.04936*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.02327*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.12150*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.25635*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.06675*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.23504*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.04540*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.09397*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.11657*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.28789*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.07391*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s238i pulse on channel 1 at 7147ns, square, 4ns, (-0.48080*(1-abs(k-0)+abs(1-abs(k-0)))/2+-0.47038*(1-abs(k-1)+abs(1-abs(k-1)))/2+-0.44311*(1-abs(k-2)+abs(1-abs(k-2)))/2+-0.46223*(1-abs(k-3)+abs(1-abs(k-3)))/2+-0.47519*(1-abs(k-4)+abs(1-abs(k-4)))/2+-0.47587*(1-abs(k-5)+abs(1-abs(k-5)))/2+-0.48080*(1-abs(k-6)+abs(1-abs(k-6)))/2+-0.47984*(1-abs(k-7)+abs(1-abs(k-7)))/2+-0.46724*(1-abs(k-8)+abs(1-abs(k-8)))/2+-0.43520*(1-abs(k-9)+abs(1-abs(k-9)))/2+-0.47672*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.26046*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.26008*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.23625*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.13324*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.25185*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.15708*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.25746*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.24579*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.24063*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.06338*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.25158*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s238q pulse on channel 2 at 7147ns, square, 4ns, (-0.09954*(1-abs(k-1)+abs(1-abs(k-1)))/2+-0.18660*(1-abs(k-2)+abs(1-abs(k-2)))/2+-0.13233*(1-abs(k-3)+abs(1-abs(k-3)))/2+-0.07326*(1-abs(k-4)+abs(1-abs(k-4)))/2+0.06864*(1-abs(k-5)+abs(1-abs(k-5)))/2+0.00189*(1-abs(k-6)+abs(1-abs(k-6)))/2+-0.03035*(1-abs(k-7)+abs(1-abs(k-7)))/2+0.11339*(1-abs(k-8)+abs(1-abs(k-8)))/2+0.20437*(1-abs(k-9)+abs(1-abs(k-9)))/2+0.06253*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.01403*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.10966*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.22380*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.06639*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.20776*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.03943*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.08619*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.09968*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.25263*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.06744*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s239i pulse on channel 1 at 7151ns, square, 4ns, (-0.48080*(1-abs(k-0)+abs(1-abs(k-0)))/2+-0.47219*(1-abs(k-1)+abs(1-abs(k-1)))/2+-0.44084*(1-abs(k-2)+abs(1-abs(k-2)))/2+-0.46287*(1-abs(k-3)+abs(1-abs(k-3)))/2+-0.47451*(1-abs(k-4)+abs(1-abs(k-4)))/2+-0.47645*(1-abs(k-5)+abs(1-abs(k-5)))/2+-0.48075*(1-abs(k-6)+abs(1-abs(k-6)))/2+-0.48001*(1-abs(k-7)+abs(1-abs(k-7)))/2+-0.46632*(1-abs(k-8)+abs(1-abs(k-8)))/2+-0.43656*(1-abs(k-9)+abs(1-abs(k-9)))/2+-0.47551*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.22319*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.22311*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.20158*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.11566*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.21457*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.13360*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.22080*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.20923*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.20711*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.05300*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.21457*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s239q pulse on channel 2 at 7151ns, square, 4ns, (-0.09059*(1-abs(k-1)+abs(1-abs(k-1)))/2+-0.19190*(1-abs(k-2)+abs(1-abs(k-2)))/2+-0.13009*(1-abs(k-3)+abs(1-abs(k-3)))/2+-0.07749*(1-abs(k-4)+abs(1-abs(k-4)))/2+0.06451*(1-abs(k-5)+abs(1-abs(k-5)))/2+0.00676*(1-abs(k-6)+abs(1-abs(k-6)))/2+-0.02762*(1-abs(k-7)+abs(1-abs(k-7)))/2+0.11710*(1-abs(k-8)+abs(1-abs(k-8)))/2+0.20144*(1-abs(k-9)+abs(1-abs(k-9)))/2+0.07112*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.00585*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.09581*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.19088*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.06144*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.17879*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.03259*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.07770*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.08317*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.21680*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.06143*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s240i pulse on channel 1 at 7155ns, square, 4ns, (0.48080*(1-abs(k-0)+abs(1-abs(k-0)))/2+0.47394*(1-abs(k-1)+abs(1-abs(k-1)))/2+0.43933*(1-abs(k-2)+abs(1-abs(k-2)))/2+0.46254*(1-abs(k-3)+abs(1-abs(k-3)))/2+0.47309*(1-abs(k-4)+abs(1-abs(k-4)))/2+0.47672*(1-abs(k-5)+abs(1-abs(k-5)))/2+0.48065*(1-abs(k-6)+abs(1-abs(k-6)))/2+0.48020*(1-abs(k-7)+abs(1-abs(k-7)))/2+0.46613*(1-abs(k-8)+abs(1-abs(k-8)))/2+0.43869*(1-abs(k-9)+abs(1-abs(k-9)))/2+0.47440*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.18654*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.18654*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.16761*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.09889*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.17864*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.11000*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.18472*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.17349*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.17395*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.04449*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.17854*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s240q pulse on channel 2 at 7155ns, square, 4ns, (0.08094*(1-abs(k-1)+abs(1-abs(k-1)))/2+0.19535*(1-abs(k-2)+abs(1-abs(k-2)))/2+0.13126*(1-abs(k-3)+abs(1-abs(k-3)))/2+0.08576*(1-abs(k-4)+abs(1-abs(k-4)))/2+-0.06253*(1-abs(k-5)+abs(1-abs(k-5)))/2+-0.01193*(1-abs(k-6)+abs(1-abs(k-6)))/2+0.02397*(1-abs(k-7)+abs(1-abs(k-7)))/2+-0.11785*(1-abs(k-8)+abs(1-abs(k-8)))/2+-0.19678*(1-abs(k-9)+abs(1-abs(k-9)))/2+-0.07816*(1-abs(k-10)+abs(1-abs(k-10)))/2+0.00028*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.08188*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.15817*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.05370*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.15066*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.02599*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.06855*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.06737*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.18116*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.05403*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s241i pulse on channel 1 at 7159ns, square, 4ns, (0.48080*(1-abs(k-0)+abs(1-abs(k-0)))/2+0.47523*(1-abs(k-1)+abs(1-abs(k-1)))/2+0.43782*(1-abs(k-2)+abs(1-abs(k-2)))/2+0.46082*(1-abs(k-3)+abs(1-abs(k-3)))/2+0.47107*(1-abs(k-4)+abs(1-abs(k-4)))/2+0.47686*(1-abs(k-5)+abs(1-abs(k-5)))/2+0.48053*(1-abs(k-6)+abs(1-abs(k-6)))/2+0.48034*(1-abs(k-7)+abs(1-abs(k-7)))/2+0.46695*(1-abs(k-8)+abs(1-abs(k-8)))/2+0.43771*(1-abs(k-9)+abs(1-abs(k-9)))/2+0.47343*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.15082*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.15077*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.13481*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.08129*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.14416*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.08779*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.14956*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.13942*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.14075*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.03710*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.14429*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s241q pulse on channel 2 at 7159ns, square, 4ns, (0.07294*(1-abs(k-1)+abs(1-abs(k-1)))/2+0.19870*(1-abs(k-2)+abs(1-abs(k-2)))/2+0.13716*(1-abs(k-3)+abs(1-abs(k-3)))/2+0.09622*(1-abs(k-4)+abs(1-abs(k-4)))/2+-0.06145*(1-abs(k-5)+abs(1-abs(k-5)))/2+-0.01602*(1-abs(k-6)+abs(1-abs(k-6)))/2+0.02111*(1-abs(k-7)+abs(1-abs(k-7)))/2+-0.11457*(1-abs(k-8)+abs(1-abs(k-8)))/2+-0.19894*(1-abs(k-9)+abs(1-abs(k-9)))/2+-0.08387*(1-abs(k-10)+abs(1-abs(k-10)))/2+0.00373*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.06762*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.12703*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.04431*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.12263*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.01943*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.05752*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.05417*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.14618*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.04387*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s242i pulse on channel 1 at 7163ns, square, 4ns, (0.48080*(1-abs(k-0)+abs(1-abs(k-0)))/2+0.47632*(1-abs(k-1)+abs(1-abs(k-1)))/2+0.43755*(1-abs(k-2)+abs(1-abs(k-2)))/2+0.45877*(1-abs(k-3)+abs(1-abs(k-3)))/2+0.46957*(1-abs(k-4)+abs(1-abs(k-4)))/2+0.47704*(1-abs(k-5)+abs(1-abs(k-5)))/2+0.48052*(1-abs(k-6)+abs(1-abs(k-6)))/2+0.48046*(1-abs(k-7)+abs(1-abs(k-7)))/2+0.46839*(1-abs(k-8)+abs(1-abs(k-8)))/2+0.43728*(1-abs(k-9)+abs(1-abs(k-9)))/2+0.47365*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.11544*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.11536*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.10285*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.06306*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.11034*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.06596*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.11458*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.10649*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.10778*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.02992*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.11023*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s242q pulse on channel 2 at 7163ns, square, 4ns, (0.06549*(1-abs(k-1)+abs(1-abs(k-1)))/2+0.19931*(1-abs(k-2)+abs(1-abs(k-2)))/2+0.14387*(1-abs(k-3)+abs(1-abs(k-3)))/2+0.10330*(1-abs(k-4)+abs(1-abs(k-4)))/2+-0.06002*(1-abs(k-5)+abs(1-abs(k-5)))/2+-0.01629*(1-abs(k-6)+abs(1-abs(k-6)))/2+0.01820*(1-abs(k-7)+abs(1-abs(k-7)))/2+-0.10855*(1-abs(k-8)+abs(1-abs(k-8)))/2+-0.19989*(1-abs(k-9)+abs(1-abs(k-9)))/2+-0.08264*(1-abs(k-10)+abs(1-abs(k-10)))/2+0.00433*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.05243*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.09670*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.03394*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.09474*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.01408*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.04457*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.04136*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.11150*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.03430*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s243i pulse on channel 1 at 7167ns, square, 4ns, (0.48080*(1-abs(k-0)+abs(1-abs(k-0)))/2+0.47751*(1-abs(k-1)+abs(1-abs(k-1)))/2+0.43763*(1-abs(k-2)+abs(1-abs(k-2)))/2+0.45716*(1-abs(k-3)+abs(1-abs(k-3)))/2+0.46839*(1-abs(k-4)+abs(1-abs(k-4)))/2+0.47705*(1-abs(k-5)+abs(1-abs(k-5)))/2+0.48055*(1-abs(k-6)+abs(1-abs(k-6)))/2+0.48049*(1-abs(k-7)+abs(1-abs(k-7)))/2+0.46958*(1-abs(k-8)+abs(1-abs(k-8)))/2+0.43691*(1-abs(k-9)+abs(1-abs(k-9)))/2+0.47400*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.08076*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.08068*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.07132*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.04506*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.07721*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.04549*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.08018*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.07406*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.07581*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.02237*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.07699*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s243q pulse on channel 2 at 7167ns, square, 4ns, (0.05613*(1-abs(k-1)+abs(1-abs(k-1)))/2+0.19912*(1-abs(k-2)+abs(1-abs(k-2)))/2+0.14892*(1-abs(k-3)+abs(1-abs(k-3)))/2+0.10854*(1-abs(k-4)+abs(1-abs(k-4)))/2+-0.05992*(1-abs(k-5)+abs(1-abs(k-5)))/2+-0.01546*(1-abs(k-6)+abs(1-abs(k-6)))/2+0.01723*(1-abs(k-7)+abs(1-abs(k-7)))/2+-0.10328*(1-abs(k-8)+abs(1-abs(k-8)))/2+-0.20069*(1-abs(k-9)+abs(1-abs(k-9)))/2+-0.08060*(1-abs(k-10)+abs(1-abs(k-10)))/2+0.00344*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.03788*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.06701*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.02367*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.06673*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00963*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.03221*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.02782*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.07760*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.02437*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s244i pulse on channel 1 at 7171ns, square, 4ns, (0.48080*(1-abs(k-0)+abs(1-abs(k-0)))/2+0.47826*(1-abs(k-1)+abs(1-abs(k-1)))/2+0.43924*(1-abs(k-2)+abs(1-abs(k-2)))/2+0.45592*(1-abs(k-3)+abs(1-abs(k-3)))/2+0.46645*(1-abs(k-4)+abs(1-abs(k-4)))/2+0.47706*(1-abs(k-5)+abs(1-abs(k-5)))/2+0.48061*(1-abs(k-6)+abs(1-abs(k-6)))/2+0.48054*(1-abs(k-7)+abs(1-abs(k-7)))/2+0.47043*(1-abs(k-8)+abs(1-abs(k-8)))/2+0.43684*(1-abs(k-9)+abs(1-abs(k-9)))/2+0.47454*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.04655*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.04649*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.04073*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.02663*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.04451*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.02601*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.04625*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.04245*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.04380*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.01384*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.04430*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s244q pulse on channel 2 at 7171ns, square, 4ns, (0.04931*(1-abs(k-1)+abs(1-abs(k-1)))/2+0.19555*(1-abs(k-2)+abs(1-abs(k-2)))/2+0.15266*(1-abs(k-3)+abs(1-abs(k-3)))/2+0.11658*(1-abs(k-4)+abs(1-abs(k-4)))/2+-0.05989*(1-abs(k-5)+abs(1-abs(k-5)))/2+-0.01366*(1-abs(k-6)+abs(1-abs(k-6)))/2+0.01568*(1-abs(k-7)+abs(1-abs(k-7)))/2+-0.09932*(1-abs(k-8)+abs(1-abs(k-8)))/2+-0.20085*(1-abs(k-9)+abs(1-abs(k-9)))/2+-0.07732*(1-abs(k-10)+abs(1-abs(k-10)))/2+0.00233*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.02254*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.03818*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.01365*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.03861*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00528*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.01911*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.01577*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.04445*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.01429*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s245i pulse on channel 1 at 7175ns, square, 4ns, (0.48080*(1-abs(k-0)+abs(1-abs(k-0)))/2+0.47838*(1-abs(k-1)+abs(1-abs(k-1)))/2+0.43940*(1-abs(k-2)+abs(1-abs(k-2)))/2+0.45513*(1-abs(k-3)+abs(1-abs(k-3)))/2+0.46459*(1-abs(k-4)+abs(1-abs(k-4)))/2+0.47620*(1-abs(k-5)+abs(1-abs(k-5)))/2+0.48061*(1-abs(k-6)+abs(1-abs(k-6)))/2+0.48050*(1-abs(k-7)+abs(1-abs(k-7)))/2+0.47040*(1-abs(k-8)+abs(1-abs(k-8)))/2+0.43725*(1-abs(k-9)+abs(1-abs(k-9)))/2+0.47495*(1-abs(k-10)+abs(1-abs(k-10)))/2+-0.01264*(1-abs(k-11)+abs(1-abs(k-11)))/2+-0.01262*(1-abs(k-12)+abs(1-abs(k-12)))/2+-0.01101*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.00728*(1-abs(k-14)+abs(1-abs(k-14)))/2+-0.01211*(1-abs(k-15)+abs(1-abs(k-15)))/2+-0.00704*(1-abs(k-16)+abs(1-abs(k-16)))/2+-0.01255*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.01150*(1-abs(k-18)+abs(1-abs(k-18)))/2+-0.01192*(1-abs(k-19)+abs(1-abs(k-19)))/2+-0.00400*(1-abs(k-20)+abs(1-abs(k-20)))/2+-0.01202*(1-abs(k-21)+abs(1-abs(k-21)))/2)
s245q pulse on channel 2 at 7175ns, square, 4ns, (0.04821*(1-abs(k-1)+abs(1-abs(k-1)))/2+0.19518*(1-abs(k-2)+abs(1-abs(k-2)))/2+0.15500*(1-abs(k-3)+abs(1-abs(k-3)))/2+0.12381*(1-abs(k-4)+abs(1-abs(k-4)))/2+-0.06635*(1-abs(k-5)+abs(1-abs(k-5)))/2+-0.01344*(1-abs(k-6)+abs(1-abs(k-6)))/2+0.01694*(1-abs(k-7)+abs(1-abs(k-7)))/2+-0.09945*(1-abs(k-8)+abs(1-abs(k-8)))/2+-0.19996*(1-abs(k-9)+abs(1-abs(k-9)))/2+-0.07476*(1-abs(k-10)+abs(1-abs(k-10)))/2+0.00068*(1-abs(k-12)+abs(1-abs(k-12)))/2+0.00621*(1-abs(k-13)+abs(1-abs(k-13)))/2+-0.01033*(1-abs(k-14)+abs(1-abs(k-14)))/2+0.00363*(1-abs(k-15)+abs(1-abs(k-15)))/2+0.01050*(1-abs(k-16)+abs(1-abs(k-16)))/2+0.00150*(1-abs(k-17)+abs(1-abs(k-17)))/2+-0.00524*(1-abs(k-18)+abs(1-abs(k-18)))/2+0.00422*(1-abs(k-19)+abs(1-abs(k-19)))/2+0.01199*(1-abs(k-20)+abs(1-abs(k-20)))/2+0.00391*(1-abs(k-21)+abs(1-abs(k-21)))/2)
dark pulse on channel 1 at 7131ns, square, 48ns, 0.480800*(1-abs(k-23)+abs(1-abs(k-23)))/2
init pulse on channel 4 at 2.7630e-06*(1-abs(k-0)+abs(1-abs(k-0)))/2+2.7630e-06*(1-abs(k-1)+abs(1-abs(k-1)))/2+2.7630e-06*(1-abs(k-2)+abs(1-abs(k-2)))/2+2.7630e-06*(1-abs(k-3)+abs(1-abs(k-3)))/2+2.7630e-06*(1-abs(k-4)+abs(1-abs(k-4)))/2+2.7630e-06*(1-abs(k-5)+abs(1-abs(k-5)))/2+2.7630e-06*(1-abs(k-6)+abs(1-abs(k-6)))/2+2.7630e-06*(1-abs(k-7)+abs(1-abs(k-7)))/2+2.7630e-06*(1-abs(k-8)+abs(1-abs(k-8)))/2+2.7630e-06*(1-abs(k-9)+abs(1-abs(k-9)))/2+2.7630e-06*(1-abs(k-10)+abs(1-abs(k-10)))/2+1.9710e-06*(1-abs(k-11)+abs(1-abs(k-11)))/2+1.9710e-06*(1-abs(k-12)+abs(1-abs(k-12)))/2+1.9710e-06*(1-abs(k-13)+abs(1-abs(k-13)))/2+1.9710e-06*(1-abs(k-14)+abs(1-abs(k-14)))/2+1.9710e-06*(1-abs(k-15)+abs(1-abs(k-15)))/2+1.9710e-06*(1-abs(k-16)+abs(1-abs(k-16)))/2+1.9710e-06*(1-abs(k-17)+abs(1-abs(k-17)))/2+1.9710e-06*(1-abs(k-18)+abs(1-abs(k-18)))/2+1.9710e-06*(1-abs(k-19)+abs(1-abs(k-19)))/2+1.9710e-06*(1-abs(k-20)+abs(1-abs(k-20)))/2+1.9710e-06*(1-abs(k-21)+abs(1-abs(k-21)))/2+2.9550e-06*(1-abs(k-22)+abs(1-abs(k-22)))/2+2.9070e-06*(1-abs(k-23)+abs(1-abs(k-23)))/2, square, 2000ns, 1.0
readout pulse on channel 4 at 6153ns, square, 4975ns, 1.0
"""
        return sequence_text.strip()

    def create_example_rabi_sequence(self) -> str:
        """
        Create an example Rabi sequence using the sequence language .

        Returns:
            Sequence text in the sequence language format
        """
        sequence_text = """
sequence: name=rabi_pulsed, type=rabi, duration=1μs, sample_rate=1GHz, repeat=50000

# Define scan variables
variable pulse_duration, start=10ns, stop=200ns, steps=20

# Define the Rabi pulse sequence
# Single microwave pulse with variable duration (will be varied by the scanner)
# Channel 1: IQ modulator I input (microwave pulses)
pi/2 pulse on channel 1 at 0ns, gaussian, 50ns, 1.0

# Channel 2: IQ modulator Q input (for complex microwave pulses)
# For simple Rabi, we can use channel 2 for additional microwave control
# or leave it empty if not needed

# Laser control via Proteus markers:
# ch1_marker1: laser_switch (triggers laser on/off)
# ch2_marker1: counter_trigger (triggers ADwin counting)
"""
        return sequence_text.strip()

    def get_experiment_summary(self) -> Dict[str, Any]:
        """
        Get summary of experiment configuration.

        Returns:
            Dictionary with experiment summary
        """
        return {
            'name': 'ODMR Pulsed Experiment',
            'sequence_name': self.sequence_description.name if self.sequence_description else 'None',
            'scan_points': len(self.scan_sequences),
            'microwave_frequency_ghz': self.microwave_frequency / 1e9,
            'microwave_power_dbm': self.microwave_power,
            'green_laser_power_mw': self.green_laser_power,
            'green_laser_wavelength_nm': self.green_laser_wavelength,
            'delays_ns': {
                'mw': self.mw_delay,
                'green laser': self.green_laser_delay,
                'counter': self.counter_delay
            },
            'output_directory': str(self.output_dir)
        }

    def _update(self):
        pass


class SequencePreviewWindow:
    """Window for previewing sequence scan points."""

    def __init__(self, sequences: List[Sequence], description):
        """Initialize preview window."""
        self.sequences = sequences
        self.description = description
        self.window = None
        self.sequence_builder = SequenceBuilder()
        self.anim = None

    def show(self):
        """Show the preview window."""
        # Create main window
        self.window = tk.Tk()
        self.window.title("ODMR Pulsed Sequence Preview")
        self.window.geometry("800x600")

        # Create notebook for different views
        notebook = ttk.Notebook(self.window)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Tab 1: Sequence overview
        overview_frame = ttk.Frame(notebook)
        notebook.add(overview_frame, text="Overview")
        self._create_overview_tab(overview_frame)

        # Tab 2: Sequence plots
        plots_frame = ttk.Frame(notebook)
        notebook.add(plots_frame, text="Plots")
        self._create_plots_tab(plots_frame)

        # Tab 3: Parameters
        params_frame = ttk.Frame(notebook)
        notebook.add(params_frame, text="Parameters")
        self._create_parameters_tab(params_frame)

        # Show window
        self.window.mainloop()

    def show_and_confirm(self) -> bool:
        """Show the preview with a Proceed/Cancel bar. Returns True to proceed.
        Closing the window (X) counts as Cancel, so hardware is never touched
        unless the user explicitly clicks 'Yes, proceed'."""
        self.proceed = False
        self.window = tk.Tk()
        self.window.title("ODMR Pulsed - Sequence Preview (confirm before hardware)")
        self.window.geometry("820x680")

        notebook = ttk.Notebook(self.window)
        notebook.pack(fill='both', expand=True, padx=10, pady=(10, 0))

        overview_frame = ttk.Frame(notebook)
        notebook.add(overview_frame, text="Overview")
        self._create_overview_tab(overview_frame)

        plots_frame = ttk.Frame(notebook)
        notebook.add(plots_frame, text="Plots")
        self._create_plots_tab(plots_frame)

        params_frame = ttk.Frame(notebook)
        notebook.add(params_frame, text="Parameters")
        self._create_parameters_tab(params_frame)

        bar = ttk.Frame(self.window)
        bar.pack(fill='x', padx=10, pady=10)
        ttk.Label(
            bar,
            text=("This sequence was built in software. "
                  "Proceed and program the hardware?"),
        ).pack(side='left')

        def _yes():
            self.proceed = True
            self.window.quit()

        def _no():
            self.proceed = False
            self.window.quit()

        ttk.Button(bar, text="No, cancel", command=_no).pack(side='right', padx=5)
        ttk.Button(bar, text="Yes, proceed", command=_yes).pack(side='right', padx=5)
        self.window.protocol("WM_DELETE_WINDOW", _no)

        self.window.mainloop()
        try:
            self.window.destroy()
        except Exception:
            pass
        return self.proceed

    def _create_overview_tab(self, parent):
        """Create overview tab."""
        # Sequence info
        info_frame = ttk.LabelFrame(parent, text="Sequence Information", padding=10)
        info_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(info_frame, text=f"Name: {self.description.name}").pack(anchor='w')
        ttk.Label(info_frame, text=f"Total scan points: {len(self.sequences)}").pack(anchor='w')
        ttk.Label(info_frame, text=f"Variables: {len(self.description.variables)}").pack(anchor='w')
        ttk.Label(info_frame, text=f"Pulses per sequence: {len(self.description.pulses)}").pack(anchor='w')

        # Variables info
        if self.description.variables:
            var_frame = ttk.LabelFrame(parent, text="Scan Variables", padding=10)
            var_frame.pack(fill='x', padx=10, pady=5)

            for name, var in self.description.variables.items():
                var_text = f"{var.name}: {var.start_value} to {var.stop_value} ({var.steps} steps)"
                ttk.Label(var_frame, text=var_text).pack(anchor='w')

    def _create_plots_tab(self, parent):
        """Create plots tab."""
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(8, 6))

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.get_tk_widget().pack(fill='both', expand=True)

        # IMPORTANT: store animation on self
        self.anim = self.sequence_builder.animate_scan_sequences(
            self.sequences[:min(5, len(self.sequences))],
            fig=fig,
            ax=ax,
            title="Sequence Preview (First 5 Points)"
        )

        canvas.draw()

    def _create_parameters_tab(self, parent):
        """Create parameters tab."""
        # Parameters info
        params_frame = ttk.LabelFrame(parent, text="Experiment Parameters", padding=10)
        params_frame.pack(fill='x', padx=10, pady=5)

        # This would show the current experiment parameters
        ttk.Label(params_frame, text="Microwave frequency: 2.87 GHz").pack(anchor='w')
        ttk.Label(params_frame, text="Microwave power: -10 dBm").pack(anchor='w')
        ttk.Label(params_frame, text="Laser power: 1.0 mW").pack(anchor='w')
        ttk.Label(params_frame, text="Laser wavelength: 532 nm").pack(anchor='w')
        ttk.Label(params_frame, text="MW delay: 25 ns").pack(anchor='w')
        ttk.Label(params_frame, text="AOM delay: 50 ns").pack(anchor='w')
        ttk.Label(params_frame, text="Counter delay: 15 ns").pack(anchor='w')


# Example usage and testing
if __name__ == "__main__":
    # seq preview part
    """experiment = ODMRPulsedExperiment(name="test_odmr")

    # Set parameters
    experiment.set_microwave_parameters(2.87e9, -10.0, 0.0)
    experiment.set_green_laser_parameters(1.0, 532)
    experiment.set_delay_parameters(0.0, 0.0, 0.0)
    # Create and load example sequence
    experiment.sequence_text = experiment.create_example_odmr_sequence()
    print("Example ODMR Sequence:")
    print(experiment.sequence_text)
    print("\n" + "=" * 50 + "\n")

    if experiment.load_sequence_from_text():
        print("Sequence loaded successfully")
    else:
        print("Failed to load sequence")

    # Build sequences
    if experiment.build_scan_sequences():
        print("Scan sequences built successfully")
    else:
        print("Failed to build scan sequences")
    experiment.show_sequence_preview(10)"""

    # independent testing part
    """experiment = ODMRPulsedExperiment(name="test_odmr")

    # Set parameters
    experiment.set_microwave_parameters(2.87e9, -10.0, 25.0)
    experiment.set_green_laser_parameters(1.0, 532)
    experiment.set_delay_parameters(0.0, 0.0, 0.0)
    # Create and load example sequence
    experiment.sequence_text = experiment.create_example_odmr_sequence()
    print("Example ODMR Sequence:")
    print(experiment.sequence_text)
    print("\n" + "="*50 + "\n")

    if experiment.load_sequence_from_text():
        print("Sequence loaded successfully")
    else:
        print("Failed to load sequence")

    # Build sequences
    if experiment.build_scan_sequences():
       print("Scan sequences built successfully")
    else:
        print("Failed to build scan sequences")
    # Generate AWG sequences
    if experiment.generate_awg_sequences_awg_triggering_adwin_case():
    #if experiment.generate_awg_task_sequences_adwin_triggering_awg_case():
       print("AWG sequences generated successfully")
    else:
        print("Failed to generate AWG sequences")"""

    # full experiment part
    experiment = ODMRPulsedExperiment(name="test_odmr")
    #experiment.filter_wheel.update({'OD': experiment.settings['Filter Wheel OD']})

    # Set parameters
    experiment.set_microwave_parameters(2.3975E9, 16.0, 0.0)
    experiment.set_green_laser_parameters(1.0, 532)
    experiment.set_delay_parameters(0.0, 0.0, 0.0)
    # Create and load example sequence
    # Run experiment
    print("Running experiment...")


    def build_calibration_overrides():
        return {
            "green_laser_delay": experiment.settings['delays']['green_laser_delay'],
            "mw_delay": experiment.settings['delays']['mw_delay'],
            "iq_delay": experiment.settings['delays']['iq_delay'],
            "counter_delay": experiment.settings['delays']['counter_delay'],
            "trigger_delay": experiment.settings['delays']['trigger_delay'],
            "units": "ns"
        }


    # this part is for when we adjust the settings in the gui, the calibration changes accordingly: so no need to change the json
    overrides = build_calibration_overrides()
    experiment.hardware_calibrator.update_calibration_delays(overrides)

    experiment.microwave_power = experiment.settings['microwave']['power']
    """experiment.settings['microwave']['frequency range'] = [2.78e9, 2.781e9, 2.782e9, 2.783e9, 2.784e9, 2.785e9, 2.786e9, 2.787e9, 2.788e9, 2.789e9, 2.79e9, 2.791e9, 2.792e9,
                                                           2.793e9, 2.794e9, 2.795e9, 2.796e9, 2.797e9, 2.798e9, 2.799e9, 2.8e9, 2.801e9, 2.802e9, 2.803e9, 2.804e9, 2.805e9,
                                                           2.806e9, 2.807e9, 2.808e9, 2.809e9, 2.81e9, 2.811e9, 2.812e9, 2.813e9, 2.814e9, 2.815e9, 2.816e9, 2.817e9, 2.818e9]
    experiment.settings['microwave']['frequency range'] = [2.79965e9, 2.79975e9, 2.79985e9, 2.79995e9, 2.80005e9, 2.80015e9, 2.80025e9, 2.80035e9, 2.80045e9, 2.80055e9, 2.80065e9,
                                                           2.80075e9, 2.80085e9, 2.80095e9, 2.80105e9, 2.80115e9, 2.80125e9, 2.80135e9, 2.80145e9, 2.80155e9, 2.80165e9, 2.80175e9,
                                                           2.80185e9, 2.80195e9, 2.80205e9, 2.80215e9, 2.80225e9, 2.80235e9, 2.80245e9, 2.80255e9, 2.80265e9, 2.80275e9, 2.80285e9,
                                                           2.80295e9, 2.80305e9, 2.80315e9, 2.80325e9, 2.80335e9, 2.80345e9, 2.80355e9, 2.80365e9, 2.80375e9, 2.80385e9, 2.80395e9,
                                                           2.80405e9, 2.80415e9, 2.80425e9, 2.80435e9, 2.80445e9, 2.80455e9, 2.80465e9, 2.80475e9, 2.80485e9, 2.80495e9, 2.80505e9,
                                                           2.80515e9, 2.80525e9, 2.80535e9, 2.80545e9, 2.80555e9, 2.80565e9, 2.80575e9, 2.80585e9, 2.80595e9, 2.80605e9, 2.80615e9,
                                                           2.80625e9, 2.80635e9, 2.80645e9, 2.80655e9, 2.80665e9, 2.80675e9, 2.80685e9, 2.80695e9, 2.80705e9, 2.80715e9, 2.80725e9,
                                                           2.80735e9, 2.80745e9, 2.80755e9, 2.80765e9, 2.80775e9, 2.80785e9, 2.80795e9, 2.80805e9, 2.80815e9, 2.80825e9, 2.80835e9,
                                                           2.80845e9, 2.80855e9, 2.80865e9, 2.80875e9, 2.80885e9, 2.80895e9, 2.80905e9, 2.80915e9, 2.80925e9, 2.80935e9, 2.80945e9,
                                                           2.80955e9, 2.80965e9, 2.80975e9, 2.80985e9, 2.80995e9, 2.81005e9, 2.81015e9, 2.81025e9, 2.81035e9, 2.81045e9, 2.81055e9,
                                                           2.81065e9, 2.81075e9, 2.81085e9, 2.81095e9, 2.81105e9, 2.81115e9, 2.81125e9, 2.81135e9, 2.81145e9, 2.81155e9, 2.81165e9,
                                                           2.81175e9, 2.81185e9, 2.81195e9, 2.81205e9, 2.81215e9, 2.81225e9, 2.81235e9]"""

    """experiment.settings['microwave']['frequency range'] = [2.795e9, 2.7954e9, 2.7958e9, 2.796e9, 2.7962e9, 2.7964e9, 2.7966e9, 2.7968e9, 2.797e9, 2.7972e9, 2.7974e9, 2.7976e9,
                                                           2.7978e9, 2.798e9, 2.7982e9, 2.7984e9, 2.7986e9, 2.7988e9, 2.799e9, 2.7992e9, 2.7994e9, 2.7996e9, 2.7998e9, 2.8e9,
                                                           2.8002e9, 2.8004e9, 2.8006e9, 2.8008e9, 2.801e9, 2.8012e9, 2.8014e9, 2.8016e9, 2.8018e9, 2.802e9, 2.8022e9, 2.8024e9,
                                                           2.8026e9, 2.8028e9, 2.803e9, 2.8032e9, 2.8034e9, 2.8036e9, 2.8038e9, 2.804e9, 2.8042e9, 2.8044e9, 2.8046e9, 2.8048e9,
                                                           2.805e9, 2.8052e9, 2.8054e9, 2.8056e9, 2.8058e9, 2.806e9, 2.8062e9, 2.8064e9, 2.8066e9, 2.8068e9, 2.807e9, 2.8072e9,
                                                           2.8074e9, 2.8076e9, 2.8078e9, 2.808e9]"""
    import numpy as np

    center = 2.39871E9
    fine = center + 0.1e6 * np.arange(-43, 44)  # ±4.3 MHz at 0.1 MHz (87 pts)
    coarse = np.arange(2.39371E9, 2.40371E9 + 1, 0.5e6)  # 0.5 MHz grid over full range
    coarse = coarse[(coarse < fine[0] - 0.1e6) | (coarse > fine[-1] + 0.1e6)]  # drop overlap
    freqs = np.sort(np.concatenate([coarse, fine]))  # 128 pts

    #experiment.settings['microwave']['frequency range'] = freqs.tolist()

    experiment.mw_delay = experiment.settings['delays']['mw_delay']
    experiment.green_laser_delay = experiment.settings['delays']['green_laser_delay']
    experiment.counter_delay = experiment.settings['delays']['counter_delay']
    experiment.iq_delay = experiment.settings['delays']['iq_delay']
    experiment.trigger_delay = experiment.settings['delays']['trigger_delay']

    experiment.green_laser_power = experiment.settings['green_laser']['power']
    experiment.green_laser_wavelength = experiment.settings['green_laser']['wavelength']

    # Output paths
    experiment.output_dir = experiment.get_output_dir("odmr_pulsed_output")
    experiment.logger.info("ODMR Pulsed Experiment initialized")

    # Create and load sequence
    source = None
    sequence_loaded = False
    experiment.sequence_text = experiment.create_example_odmr_sequence()
    if experiment.settings["tau_advisor"]["enable"]:
        from tau_advisor import advise_tau
        advise_tau(experiment.sequence_text, preset = experiment.settings["tau_advisor"]["preset"], b_field_G = experiment.settings["tau_advisor"]["b_field_G"],
                   save_path = experiment.settings["tau_advisor"]["save_path"])
    print("Example ODMR Sequence:")
    print(experiment.sequence_text)
    print("\n" + "=" * 50 + "\n")

    if experiment.load_sequence_from_text():
        source = "text"
        print("Sequence loaded successfully")
    else:
        print("Failed to load sequence")

    if sequence_loaded:
        experiment.logger.info(f"Sequence loaded successfully from {source}")
    else:
        experiment.logger.info(f"Failed to load sequence from {source}")

    experiment.count_time = experiment.settings["adwin"]["count_time"] # 300
    experiment.reset_time = experiment.settings["adwin"]["reset_time"] #2000 # was 5320: decreasing it to make ref away from laser falling edge
    experiment.readout_start_ns = experiment.settings["adwin"]["readout_start_ns"] #3002000 #3002000 #502000 # 28312 # calibrated 3000 ns pulse_duration: 6224
    experiment.trigger_duration = experiment.settings["adwin"]["trigger_duration"] #150
    experiment.gate_align_offset_ns = experiment.settings["adwin"]["gate_align_offset_ns"] # 2000 #1650
    # Run experiment
    print("running exp")
    experiment.logger.info("Running experiment...")
    results = experiment.run_experiment_averaged(experiment.settings['microwave']['frequency range'])
    experiment.data = results
    experiment.save_hdf5()
    experiment.cleanup()
    if results['success']:
        experiment.logger.info("Experiment completed successfully!")
        experiment.logger.info(f"Frequencies scanned: {[f / 1e9 for f in results['frequencies']]} GHz")
        experiment.logger.info(
            f"Signal counts shape: {len(results['signal_counts'])} frequencies x {len(results['signal_counts'][0])} points")
    else:
        experiment.logger.info(f"Experiment failed: {results['error']}")
    #experiment.show_sequence_preview(10)
    experiment.logger.info("\nODMR Pulsed Experiment ready!")


    """# Multiharp tester:
    experiment = ODMRPulsedExperiment(name="test_odmr", mode="testing")  # drop mode= for a real run
    # Set parameters
    experiment.set_microwave_parameters(2.86041e9, 12.0, 0.0)
    experiment.set_green_laser_parameters(1.0, 532)
    experiment.set_delay_parameters(0.0, 0.0, 0.0)


    def build_calibration_overrides():
        return {
            "green_laser_delay": experiment.settings['delays']['green_laser_delay'],
            "mw_delay": experiment.settings['delays']['mw_delay'],
            "iq_delay": experiment.settings['delays']['iq_delay'],
            "counter_delay": experiment.settings['delays']['counter_delay'],
            "trigger_delay": experiment.settings['delays']['trigger_delay'],
            "units": "ns"
        }


    # this part is for when we adjust the settings in the gui, the calibration changes accordingly: so no need to change the json
    overrides = build_calibration_overrides()
    experiment.hardware_calibrator.update_calibration_delays(overrides)

    experiment.microwave_power = experiment.settings['microwave']['power']
    experiment.mw_delay = experiment.settings['delays']['mw_delay']
    experiment.green_laser_delay = experiment.settings['delays']['green_laser_delay']
    experiment.counter_delay = experiment.settings['delays']['counter_delay']
    experiment.iq_delay = experiment.settings['delays']['iq_delay']
    experiment.trigger_delay = experiment.settings['delays']['trigger_delay']
    experiment.green_laser_power = experiment.settings['green_laser']['power']
    experiment.green_laser_wavelength = experiment.settings['green_laser']['wavelength']

    # Output paths
    experiment.output_dir = experiment.get_output_dir("odmr_pulsed_output")
    experiment.logger.info("ODMR Pulsed Experiment initialized")
    freqs = experiment.settings['microwave']['frequency range']
    # Create and load sequence
    source = None
    sequence_loaded = False
    experiment.sequence_text = experiment.create_example_odmr_sequence()
    print("Example ODMR Sequence:")
    print(experiment.sequence_text)
    print("\n" + "=" * 50 + "\n")

    if experiment.load_sequence_from_text():
        source = "text"
        print("Sequence loaded successfully")
    else:
        print("Failed to load sequence")

    if sequence_loaded:
        experiment.logger.info(f"Sequence loaded successfully from {source}")
    else:
        experiment.logger.info(f"Failed to load sequence from {source}")

    experiment.count_time = 340  # then 390 then 415
    experiment.reset_time = 6320
    experiment.readout_start_ns = 6000
    experiment.trigger_duration = 150
    experiment.gate_align_offset_ns = 1800  # 1650
    # Run experiment
    print("running exp")
    experiment.logger.info("Running experiment...")
    from multiharp_timing_tester import MultiHarpTimingTester, ChannelConfig, EDGE_RISING

    tester = MultiHarpTimingTester(
        case="init_vs_trigger",
        start_input=2,  # MKR
        stop_input=0,  # DIO26
        start_cfg=ChannelConfig(level_mV=320, edge=EDGE_RISING),
        stop_cfg=ChannelConfig(level_mV=305, edge=EDGE_RISING),
        window_ns=10000.0,  # look 2 µs after each readout edge (preset default)
        output_mode="both",  # "histogram" is fine too
    )
    out = tester.run_with_experiment(experiment, freqs, run_method="run_experiment")
    experiment.timing_result = out["timing"]
    print(out["timing"].summary())  # prints onset + 10–90% rise time
    results = out["odmr"]
    experiment.data = results
    experiment.save_hdf5()
    experiment.cleanup()"""

    """#Multiharp Rabi:
    experiment = ODMRPulsedExperiment(name="test_odmr", mode="testing")  # drop mode= for a real run
    # Set parameters
    experiment.set_microwave_parameters(2.8061e9, 12.0, 0.0)
    experiment.set_green_laser_parameters(1.0, 532)
    experiment.set_delay_parameters(0.0, 0.0, 0.0)


    def build_calibration_overrides():
        return {
            "green_laser_delay": experiment.settings['delays']['green_laser_delay'],
            "mw_delay": experiment.settings['delays']['mw_delay'],
            "iq_delay": experiment.settings['delays']['iq_delay'],
            "counter_delay": experiment.settings['delays']['counter_delay'],
            "trigger_delay": experiment.settings['delays']['trigger_delay'],
            "units": "ns"
        }


    # this part is for when we adjust the settings in the gui, the calibration changes accordingly: so no need to change the json
    overrides = build_calibration_overrides()
    experiment.hardware_calibrator.update_calibration_delays(overrides)

    experiment.microwave_power = experiment.settings['microwave']['power']
    experiment.mw_delay = experiment.settings['delays']['mw_delay']
    experiment.green_laser_delay = experiment.settings['delays']['green_laser_delay']
    experiment.counter_delay = experiment.settings['delays']['counter_delay']
    experiment.iq_delay = experiment.settings['delays']['iq_delay']
    experiment.trigger_delay = experiment.settings['delays']['trigger_delay']
    experiment.green_laser_power = experiment.settings['green_laser']['power']
    experiment.green_laser_wavelength = experiment.settings['green_laser']['wavelength']

    # Output paths
    experiment.output_dir = experiment.get_output_dir("odmr_pulsed_output")
    experiment.logger.info("ODMR Pulsed Experiment initialized")
    freqs = experiment.settings['microwave']['frequency range']
    # Create and load sequence
    source = None
    sequence_loaded = False
    experiment.sequence_text = experiment.create_example_odmr_sequence()
    print("Example ODMR Sequence:")
    print(experiment.sequence_text)
    print("\n" + "=" * 50 + "\n")

    if experiment.load_sequence_from_text():
        source = "text"
        print("Sequence loaded successfully")
    else:
        print("Failed to load sequence")

    if sequence_loaded:
        experiment.logger.info(f"Sequence loaded successfully from {source}")
    else:
        experiment.logger.info(f"Failed to load sequence from {source}")

    experiment.count_time = 300  # then 390 then 415
    experiment.reset_time = 5320 # was 6320: decreasing it to make ref away from laser falling edge
    experiment.readout_start_ns = 6000
    experiment.trigger_duration = 150
    experiment.gate_align_offset_ns = 2000 #1650
    # Run experiment
    print("running exp")
    experiment.logger.info("Running experiment...")
    import numpy as np
    from multiharp_timing_tester import MultiHarpTimingTester, ChannelConfig, EDGE_RISING, EDGE_FALLING

    tester = MultiHarpTimingTester(
    case = "rabi_mw_tagged",
    start_input = 1, start_cfg = ChannelConfig(level_mV=500, edge=EDGE_RISING), # readout marker (~720 mV)
    stop_input = 4, stop_cfg = ChannelConfig(level_mV=320, edge=EDGE_RISING), # SPCM (~640 mV)
    mw_input = 2, mw_cfg = ChannelConfig(level_mV=350, edge=EDGE_RISING), # MW rise (~700 mV)
    mw_fall_input = 3, mw_fall_cfg = ChannelConfig(level_mV=350, edge=EDGE_FALLING),  # MW fall
    trig_input = 0, trig_cfg = ChannelConfig(level_mV=300, edge=EDGE_RISING),  # DO28 (~610 mV)
    rabi_tau_values_ns = np.linspace(4, 3000, 64),  # pass explicitly — do NOT let it infer (units trap)
    )
    out = tester.run_with_experiment(
        experiment, experiment.settings['microwave']['frequency range'],
        run_method="run_experiment_averaged")  # FIX 1: actually run 'averages' times and average
    rabi = out["rabi"]
    experiment.data = out["odmr"]  # FIX 2: hand the ADwin result to save_hdf5 (it saves self.data)
    experiment.save_hdf5()
    experiment.cleanup()"""

    #1. Run IQCalibrationExperiment  →  writes iq_calibration_results.json
    #2. Run ODMRPulsedExperiment     →  reads iq_calibration_results.json at start
    #                                   applies b≈0.89 to channel 2 automatically
    #3. Re-run calibration any time  →  overwrites the JSON
    #                                   next ODMR run picks up the new values

