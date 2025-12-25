import os
import sys
srcpath = os.path.realpath(r'C:\Users\Duttlab\Downloads\PythonExamples\Examples\SourceFiles')
sys.path.append(srcpath)
from teproteus import TEProteusAdmin as TepAdmin
from tevisainst import TEVisaInst
import numpy as np

sid = 6 #PXI slot of AWT on chassis
admin = TepAdmin() #required to control PXI module
inst = admin.open_instrument(slot_id=sid)
resp = inst.send_scpi_query("*IDN?") # Get the instrument's *IDN
print('connected to: ' + resp) # Print *IDN
inst.send_scpi_cmd('*CLS; *RST')

length = 64
amplitude= 1
channel = 1
# AWG channel
cmd = ':INST:CHAN {0}'.format(channel)
inst.send_scpi_cmd(cmd)
#cmd = ':VOLT {0}'.format(1)
#self.inst.send_scpi_cmd(cmd)
inst.send_scpi_cmd(f":VOLT:OFFS{0}")
sampleRateDAC = 1E9
cmd = ':FREQ:RAST {0}'.format(sampleRateDAC)
inst.send_scpi_cmd(cmd)
cmd = ':TRAC:DEL:ALL'  # Clear CH 1 Memory ######################
inst.send_scpi_cmd(cmd)
cmd = ':INIT:CONT OFF' # play waveform continuously
inst.send_scpi_cmd(cmd)
# scale to 16 bits
max_dac = 65535  # Max Dac
half_dac = max_dac / 2  # DC Level
data_type = np.uint16  # DAC data type
segLen = length
#dacWaveDC = amp * np.ones(segLen)
dacWaveDC = np.full(segLen, amplitude, dtype=float)
dacWaveDC = (dacWaveDC) * half_dac  # scale
dacWaveDC = dacWaveDC.astype(data_type)

segnum = 1
cmd = ':TRAC:DEF {0}, {1}'.format(segnum, len(dacWaveDC))  # memory location and length
inst.send_scpi_cmd(cmd)
cmd = ':TRAC:SEL {0}'.format(segnum)
inst.send_scpi_cmd(cmd)
inst.timeout = 30000  # increase
inst.write_binary_data('*OPC?; :TRAC:DATA', dacWaveDC)  # write, and wait while *OPC completes
inst.timeout = 10000  # return to normal
delay = 100
# Create a Task Table
cmd = f':TASK:COMP:LENG {2}'  # set task table length
inst.send_scpi_cmd(cmd)
cmd = f':TASK:COMP:SEL {1}'  # set task number
inst.send_scpi_cmd(cmd)
cmd = f':TASK:COMP:TYPE:STAR'
inst.send_scpi_cmd(cmd)
cmd = f':TASK:COMP:SEQ {1}'  # Repeat
inst.send_scpi_cmd(cmd)
cmd = f':TASK:COMP:SEGM {1}'
inst.send_scpi_cmd(cmd)
cmd = f':TASK:COMP:NEXT1 {2}'
inst.send_scpi_cmd(cmd)
cmd = f':TASK:COMP:DEL {delay}'
inst.send_scpi_cmd(cmd)
cmd = f':TASK:COMP:IDLE DC'
inst.send_scpi_cmd(cmd)
cmd = f':TASK:COMP:IDLE:LEV {0}'
inst.send_scpi_cmd(cmd)
cmd = f':TASK:COMP:SEL {2}'  # set task number
inst.send_scpi_cmd(cmd)
cmd = f':TASK:COMP:TYPE:END'
inst.send_scpi_cmd(cmd)
cmd = f':TASK:COMP:SEGM {1}'
inst.send_scpi_cmd(cmd)
"""cmd = f':TASK:COMP:NEXT1 {3}'
inst.send_scpi_cmd(cmd)"""
cmd = ':TASK:COMP:WRITE'  # write to FPGA
inst.send_scpi_cmd(cmd)
cmd = ':SOUR:FUNC:MODE TASK'
inst.send_scpi_cmd(cmd)
cmd = ':OUTP ON'
inst.send_scpi_cmd(cmd)
inst.close_instrument()
admin.close_inst_admin()