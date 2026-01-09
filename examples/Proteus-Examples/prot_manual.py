import os
import numpy as np
from math import pi
import random

class ProteusInstrument:
    def __init__(self, connection_type, conn_par = '192.168.2.4', paranoia_level=2):
        # connection_type: "LAN" or "DLL"
        # conn_par: IP address string for LAN or slot number for DLL
        self.connection_type = connection_type
        self.conn_par = conn_par
        self.paranoia_level = paranoia_level
        self.inst = None
        self.admin = None
        self.model = ''
        self.slotNumber = 0
        # connect on construction to mirror MATLAB example flow
        self.connect_to_proteus()

    # ---------- low-level wrappers ----------
    def send_command(self, cmd):
        """ Send SCPI command and perform paranoia checks as in MATLAB wrapper """
        # send command and return string response (RespStr), like inst.SendScpi(...).RespStr
        resp = self.inst.SendScpi(cmd).RespStr
        resp = self._net_str_to_str(resp).strip()

        if self.paranoia_level == 1:
            # *OPC? behavior - in MATLAB it was described as a check
            self.inst.SendScpi('*OPC?')  # ignoring response content here to match example
        elif self.paranoia_level == 2:
            # SYST:ERR? behavior
            err = self.inst.SendScpi('SYST:ERR?').RespStr
            err = self._net_str_to_str(err).strip()
            if not err.startswith('0'):
                raise RuntimeError(f"Instrument SCPI Error: {err}")

        return resp

    def write_binary_data(self, prefix, data_bytes):
        """Wrap inst.WriteBinaryData(prefix, data) call expecting the same return structure"""
        # In MATLAB: res = inst.WriteBinaryData(prefix, myWfm); assert(res.ErrCode == 0)
        res = self.inst.WriteBinaryData(prefix, data_bytes)
        if hasattr(res, 'ErrCode'):
            if res.ErrCode != 0:
                raise RuntimeError(f"WriteBinaryData failed ErrCode={res.ErrCode}")
        return res

    def disconnect(self):
        if self.connection_type.upper() == "LAN":
            # MATLAB used inst.Disconnect()
            try:
                self.inst.Disconnect()
            except Exception:
                pass
        else:
            # MATLAB used admin.CloseInstrument(inst.InstrId); admin.Close();
            try:
                self.admin.CloseInstrument(self.inst.InstrId)
            except Exception:
                pass
            try:
                self.admin.Close()
            except Exception:
                pass

    # ---------- helpers that mirror MATLAB functions ----------
    def _net_str_to_str(self, net_str):
        """ convert .NET string-like object to python string (mirrors netStrToStr) """
        try:
            return str(net_str)
        except Exception:
            return ''

    def identify_model(self):
        idn = self.inst.SendScpi('*IDN?').RespStr
        idn = self._net_str_to_str(idn).strip()
        parts = idn.split(',')
        if len(parts) > 1:
            self.model = parts[1]
        else:
            self.model = ''
        return self.model

    def get_options(self):
        opt = self.inst.SendScpi('*OPT?').RespStr
        opt = self._net_str_to_str(opt).strip()
        if opt == '':
            return []
        return opt.split(',')

    def get_granularity(self, model, options, dacMode):
        flagLowGranularity = False
        for opt in options:
            if 'G1' in opt or 'G2' in opt:
                flagLowGranularity = True

        # NOTE: in the MATLAB snippet there was a temporary override to false in one place.
        # Here we follow the principal logic (no forced temp override).
        granularity = 32
        if 'P258' in model:
            granularity = 32
            if flagLowGranularity:
                granularity = 16
        elif 'P128' in model:
            granularity = 32
            if flagLowGranularity:
                granularity = 16
        elif 'P948' in model:
            if dacMode == 16:
                granularity = 32
                if flagLowGranularity:
                    granularity = 16
            else:
                granularity = 64
                if flagLowGranularity:
                    granularity = 32
        elif 'P908' in model:
            granularity = 64
            if flagLowGranularity:
                granularity = 32

        return granularity

    def get_dac_resolution(self):
        resp = self.inst.SendScpi(':TRAC:FORM?').RespStr
        resp = self._net_str_to_str(resp).strip()
        if 'U8' in resp:
            return 8
        return 16

    def get_channels(self, model, sampleRate):
        """Mirror GetChannels MATLAB function. Returns (chanList, segmList) as Python lists."""
        chanList = []
        segmList = []
        if ('P9484' in model) or ('P2584' in model) or ('P1284' in model):
            if sampleRate <= 2.5e9:
                chanList = [1,2,3,4]
                segmList = [1,2,1,2]
            else:
                chanList = [1,3]
                segmList = [1,1]
        elif ('P9482' in model) or ('P2582' in model) or ('P1282' in model):
            if sampleRate <= 2.5e9:
                chanList = [1,2]
                segmList = [1,2]
            else:
                chanList = [1]
                segmList = [1]
        elif ('P9488' in model) or ('P2588' in model) or ('P1288' in model):
            if sampleRate <= 2.5e9:
                chanList = [1,2,3,4,5,6,7,8]
                segmList = [1,2,1,2,1,2,1,2]
            else:
                chanList = [1,3,5,7]
                segmList = [1,1,1,1]
        elif ('P94812' in model) or ('P25812' in model) or ('P12812' in model):
            if sampleRate <= 2.5e9:
                chanList = [1,2,3,4,5,6,7,8,9,10,11,12]
                segmList = [1,2,1,2,1,2,1,2,1,2,1,2]
            else:
                chanList = [1,3,5,7,9,11]
                segmList = [1,1,1,1,1,1]
        elif 'P9082' in model:
            chanList = [1,2]
            segmList = [1,1]
        elif 'P9084' in model:
            chanList = [1,2,3,4]
            segmList = [1,1,1,1]
        elif 'P9086' in model:
            chanList = [1,2,3,4,5,6]
            segmList = [1,1,1,1,1,1]
        return chanList, segmList

    def my_quantization(self, arr, dacRes, minLevel=0):
        maxLevel = (2 ** dacRes) - 1
        numOfLevels = maxLevel - minLevel + 1
        # MATLAB: retval = round((numOfLevels .* (myArray + 1) - 1) ./ 2);
        retval = np.round((numOfLevels * (arr + 1.0) - 1.0) / 2.0).astype(np.int64)
        retval = retval + minLevel
        retval[retval > maxLevel] = maxLevel
        retval[retval < minLevel] = minLevel
        return retval

    def get_square_wfm(self, samplingRate, numCycles, period, granularity):
        # MATLAB:
        # wfmLength = round(numCycles * period * samplingRate);
        # wfmLength = round(wfmLength / granularity) * granularity;
        wfmLength = int(round(numCycles * period * samplingRate))
        wfmLength = int(round(wfmLength / granularity) * granularity)
        if wfmLength <= 0:
            wfmLength = granularity
        period_samples = wfmLength / numCycles
        idx = np.arange(0, wfmLength)
        # square(sqrWfm * 2 * pi / period) in MATLAB -> use sign(sin(.))
        sqrWfm = np.sign(np.sin(2.0 * pi * idx / period_samples))
        # MATLAB's square returns -1..1; ensure same dtype
        return sqrWfm

    def format_mkr2(self, dac_Mode, mkr1, mkr2):
        # mkr1 and mkr2 are arrays of 0/1 ints
        # MATLAB: mkrData = mkr1 + 2 * mkr2;
        mkr1 = np.asarray(mkr1, dtype=np.uint8)
        mkr2 = np.asarray(mkr2, dtype=np.uint8)
        mkrData = mkr1 + 2 * mkr2
        if dac_Mode == 16:
            # MATLAB: pack pairs: mkrData(1:2:end) + 16 * mkrData(2:2:end)
            a = mkrData[0::2]
            b = mkrData[1::2]
            # if odd length, MATLAB would drop final unmatched element in the indexing behavior
            minlen = min(len(a), len(b))
            packed = (a[:minlen].astype(np.uint8) + (16 * b[:minlen].astype(np.uint8)))
            return packed
        else:
            return mkrData.astype(np.uint8)

    def format_mkr4(self, dac_Mode, m1, m2, m3, m4):
        m1 = np.asarray(m1, dtype=np.uint8)
        m2 = np.asarray(m2, dtype=np.uint8)
        m3 = np.asarray(m3, dtype=np.uint8)
        m4 = np.asarray(m4, dtype=np.uint8)
        mkrData = m1 + 2*m2 + 4*m3 + 8*m4
        if dac_Mode == 16:
            a = mkrData[0::2]
            b = mkrData[1::2]
            minlen = min(len(a), len(b))
            packed = (a[:minlen].astype(np.uint8) + (16 * b[:minlen].astype(np.uint8)))
            return packed
        else:
            return mkrData.astype(np.uint8)

    # ---------- waveform and marker download functions ----------
    def send_wfm_to_proteus(self, samplingRate, channel, segment, myWfm, dacRes, initialize=False):
        # replicate SendWfmToProteus MATLAB function
        if dacRes == 16:
            self.send_command(':TRAC:FORM U16')
        else:
            self.send_command(':TRAC:FORM U8')

        if initialize:
            self.send_command(':TRAC:DEL:ALL')
            self.send_command(f':FREQ:RAST {samplingRate}')

        self.send_command(f':INST:CHAN {channel}')
        self.send_command(f':TRAC:DEF {segment}, {len(myWfm)}')
        self.send_command(f':TRAC:SEL {segment}')

        # Quantize
        q = self.my_quantization(myWfm, dacRes, 0)

        if dacRes == 16:
            # Convert to uint16 then to bytes (little-endian typecast equivalent)
            q16 = q.astype(np.uint16)
            data_bytes = q16.tobytes()  # default little-endian typically matches Matlab typecast in many environments
        else:
            q8 = q.astype(np.uint8)
            data_bytes = q8.tobytes()

        prefix = ':TRAC:DATA 0,'
        self.write_binary_data(prefix, data_bytes)

        if initialize:
            self.send_command(f':SOUR:FUNC:MODE:SEGM {segment}')
            self.send_command(':SOUR:VOLT MAX')
            self.send_command(':OUTP ON')

        return len(data_bytes)

    def send_mkr_to_proteus(self, myMkr):
        prefix = ':MARK:DATA 0,'
        # myMkr should already be uint8 bytes-like
        if isinstance(myMkr, np.ndarray):
            data_bytes = myMkr.tobytes()
        elif isinstance(myMkr, (bytes, bytearray)):
            data_bytes = bytes(myMkr)
        else:
            data_bytes = bytes(myMkr)
        self.write_binary_data(prefix, data_bytes)
        return len(data_bytes)

    # ---------- connection routine (mirrors ConnecToProteus) ----------
    def connect_to_proteus(self):
        # Print PID and initialization messages like MATLAB
        print('INITIALIZING SETTINGS')
        pid = os.getpid()
        print(f'\nProcess ID {pid}\n')

        dll_path = r'C:\Windows\System32\TEPAdmin.dll'
        self.admin = None
        self.slotNumber = 0

        if self.connection_type.upper() == "LAN":
            # replicate: connStr = strcat('TCPIP::',connStr,'::5025::SOCKET');
            conn_string = f'TCPIP::{self.conn_par}::5025::SOCKET'
            # TEProteusInst(connStr, paranoia_level)
            # Expectation: environment provides TEProteusInst
            self.inst = TEProteusInst(conn_string, self.paranoia_level)
            res = self.inst.Connect()
            if not res:
                raise RuntimeError("Failed to connect via LAN")
            self.identify_model()
        else:
            # DLL / PXI path - assumes .NET interop through pythonnet or similar is configured
            # This reproduces the MATLAB flow (NET.addAssembly, create CProteusAdmin, Open, GetSlotIds, etc.)
            import clr  # requires pythonnet if used
            clr.AddReference(dll_path)
            from TaborElec.Proteus.CLI import CProteusAdmin
            # In MATLAB they passed @OnLoggerEvent; here we pass None for simplicity
            self.admin = CProteusAdmin(None)
            rc = self.admin.Open()
            if rc != 0:
                raise RuntimeError("Failed to open admin interface")
            slotIds = self.admin.GetSlotIds()
            # MATLAB used length(size(slotIds)) to get numSlots; adapt:
            try:
                numSlots = len(slotIds)
            except Exception:
                numSlots = 1
            if numSlots <= 0:
                raise RuntimeError("No slots found")
            # choose slot
            sId = slotIds[0]
            if numSlots > 1:
                print(f'\n{numSlots} slots were found\n')
                for n in range(numSlots):
                    sId = slotIds[n]
                    slotInfo = self.admin.GetSlotInfo(sId)
                    if not slotInfo.IsSlotInUse:
                        modelName = slotInfo.ModelName
                        if slotInfo.IsDummySlot and self.conn_par == 0:
                            print(f' * Slot Number:{sId} Model {modelName} [Dummy Slot].')
                        elif self.conn_par == 0:
                            print(f' * Slot Number:{sId} Model {modelName}.')
                # pause(0.1) equivalent
                # select slot
                if self.conn_par == 0:
                    choice = int(input('Enter SlotId '))
                else:
                    choice = int(self.conn_par)
                sId = int(choice)
                slotInfo = self.admin.GetSlotInfo(sId)
                modelName = slotInfo.ModelName
                self.model = self._net_str_to_str(modelName).strip()
            # Connect to selected instrument
            should_reset = True
            self.inst = self.admin.OpenInstrument(sId, should_reset)
            self.slotNumber = sId

# ---------- Example script that mirrors your MATLAB example ----------
def main_example():
    # Clear-like behavior: (not necessary in Python)
    # Print initialization and PID (done in constructor)
    ipAddr = '192.168.2.4'
    pxiSlot = 0
    cType = "LAN"  # "LAN" = VISA or "DLL" = PXI
    if cType == "LAN":
        connPar = ipAddr
    else:
        connPar = pxiSlot
    paranoia_level = 2  # 0, 1 or 2

    proteus = ProteusInstrument(cType, connPar, paranoia_level)
    inst = proteus  # mirror MATLAB variable naming
    admin = proteus.admin
    model = proteus.model
    slotNumber = proteus.slotNumber

    print(f'Connected to: {model}, slot: {slotNumber}')

    # Reset AWG
    proteus.send_command('*CLS;*RST')

    # Get options
    optstr = proteus.get_options()

    samplingRate = 9000e6
    interpol = 4
    dacMode = 16
    if (samplingRate / interpol) > 2.5e9:
        dacMode = 8
        interpol = 1
    if (samplingRate / interpol) < 250e6:
        interpol = 1

    # Get granularity
    granul = proteus.get_granularity(model, optstr, dacMode)
    print(f'\nGranularity = {granul} samples\n')

    # Get Active Channels and Segment #
    chanList, segmList = proteus.get_channels(model, samplingRate / interpol)
    numOfChannels = len(chanList)
    print('Calculating WAVEFORMS')
    minCycles = 1
    period = 1e-7

    print('SETTING AWG')
    # Set sampling rate for AWG to maximum.
    if interpol > 1:
        proteus.send_command(':FREQ:RAST 2.5E9')
        proteus.send_command(f':INT X {interpol}')
    proteus.send_command(f':FREQ:RAST {samplingRate}')

    wfmVolt = 0.5
    wfmOff = 0.0
    mkrVolt = 1.0
    mkrOff = 0.5

    myWfm = None
    for idx_channel in range(numOfChannels):
        channel = idx_channel + 1  # MATLAB 1-based indexing logic for the "mod" usage
        ch_num = chanList[idx_channel]

        # Calculate basic square wave
        if (channel % 4) == 1:
            myWfm = proteus.get_square_wfm(samplingRate / interpol, minCycles, period, granul)

        mkrDiv = 2
        if dacMode == 8:
            mkrDiv = 8

        # myMkr1 length is length(myWfm) / mkrDiv, MATLAB used uint8 zeros
        myMkr1_len = int(len(myWfm) / mkrDiv)
        if myMkr1_len <= 0:
            myMkr1_len = 1
        myMkr1 = np.zeros(myMkr1_len, dtype=np.uint8)
        # Marker 1: sync pulse duration equal to channel number
        # MATLAB did: myMkr1(1:channel) = uint8(1);
        # but ensure not to exceed length
        up_to = min(channel, myMkr1_len)
        myMkr1[0:up_to] = 1

        # Marker 2: random stream bits
        myMkr2 = np.random.rand(myMkr1_len)
        myMkr2 = (myMkr2 > 0.5).astype(np.uint8)

        myMkr = proteus.format_mkr2(dacMode, myMkr1, myMkr2)

        # Select Channel
        proteus.send_command(f':INST:CHAN {ch_num}')
        # DAC Mode set to DIRECT (default)
        proteus.send_command(':SOUR:MODE DIRECT')

        # Segment processing: delete all segments for the bank if segmList(channel) == 1
        seg_for_channel = segmList[idx_channel]
        if seg_for_channel == 1:
            proteus.send_command(':TRAC:DEL:ALL')

        # Waveform Downloading
        print(f'DOWNLOADING WAVEFORM FOR CH{ch_num}')
        proteus.send_wfm_to_proteus(samplingRate, ch_num, seg_for_channel, myWfm, dacMode, False)
        result = proteus.send_mkr_to_proteus(myMkr)
        print('WAVEFORM DOWNLOADED!')

        # Select segment for generation
        print('SETTING AWG OUTPUT')
        proteus.send_command(f':SOUR:FUNC:MODE:SEGM {seg_for_channel}')
        # Output voltage and offset
        proteus.send_command(f':SOUR:VOLT {wfmVolt}')
        proteus.send_command(f':SOUR:VOLT:OFFS {wfmOff}')
        # Activate output and start generation
        proteus.send_command(':OUTP ON')
        proteus.send_command(':MARK:SEL 1')  # Marker1
        proteus.send_command(f':MARK:VOLT:PTOP {mkrVolt}')  # Vpp
        proteus.send_command(f':MARK:VOLT:OFFS {mkrOff}')  # DC Offset
        proteus.send_command(':MARK ON')
        proteus.send_command(':MARK:SEL 2')  # Marker2
        proteus.send_command(f':MARK:VOLT:PTOP {mkrVolt}')
        proteus.send_command(f':MARK:VOLT:OFFS {mkrOff}')
        proteus.send_command(':MARK ON')

        # The new waveform is calculated for the next channel
        if (channel % 4) < 3:
            # Integration
            myWfm = np.cumsum(myWfm)
            # DC removal
            myWfm = myWfm - np.mean(myWfm)
            # Normalization to -1..+1
            max_abs = np.max(np.abs(myWfm))
            if max_abs != 0:
                myWfm = myWfm / max_abs

    # disconnect at the end
    proteus.disconnect()
    print('Done.')

# If user runs this file as script, execute main_example
if __name__ == '__main__':
    main_example()
