import os
import sys
import struct
srcpath = os.path.realpath(r'C:\Users\Duttlab\Downloads\PythonExamples\Examples\SourceFiles')
sys.path.append(srcpath)
from teproteus import TEProteusAdmin as TepAdmin
from tevisainst import TEVisaInst
import numpy as np
# testing aom with worst case scenario: longest pulse durations and shortest waiting time:
# from scc papers, longest shelving is 300 ns, longest ionization is 500 ns, short readout: 3ms, and short initialization 1 us
# Connect to instrument(PXI)
sid = 6 #PXI slot of AWT on chassis
admin = TepAdmin() #required to control PXI module
inst = admin.open_instrument(slot_id=sid)
resp = inst.send_scpi_query("*IDN?") # Get the instrument's *IDN
print('connected to: ' + resp) # Print *IDN

# initialize DAC
inst.send_scpi_cmd('*CLS; *RST')

#AWG channel
ch = 3 # everything after relates to CH 3
cmd = ':INST:CHAN {0}'.format(ch)
inst.send_scpi_cmd(cmd)
cmd = ':VOLT {0}'.format(1)
inst.send_scpi_cmd(cmd)

sampleRateDAC = 1.25E9
cmd = ':FREQ:RAST {0}'.format(sampleRateDAC)
inst.send_scpi_cmd(cmd)
cmd = ':TRAC:DEL:ALL' # Clear CH 1 Memory
inst.send_scpi_cmd(cmd)
cmd = ':INIT:CONT ON' # play waveform continuously


#scale to 16 bits
max_dac=65535 # Max Dac
half_dac=max_dac/2 # DC Level
data_type = np.uint16 # DAC data type

segnum = 1
amp = 1
segLen = 30016 # must be a multiple of 64 (corresponds to 400 ns)
dacWaveDC = amp * np.ones(segLen)
dacWaveDC = dacWaveDC * half_dac  # scale
dacWaveDC = dacWaveDC.astype(data_type)
cmd = ':TRAC:DEF {0}, {1}'.format(segnum, len(dacWaveDC)) # memory location and length
inst.send_scpi_cmd(cmd)
cmd = ':TRAC:SEL {0}'.format(segnum)
inst.send_scpi_cmd(cmd)

inst.timeout = 30000 #increase
inst.write_binary_data('*OPC?; :TRAC:DATA', dacWaveDC) # write, and wait while *OPC completes
inst.timeout = 10000 # return to normal

# Create and download a second Segment
segnum = 2
amp = 0
segLen = 1024  # must be a multiple of 64 (corresponds to 400 ns)
dacWaveDC = amp * np.zeros(segLen)
dacWaveDC = dacWaveDC * half_dac  # scale
dacWaveDC = dacWaveDC.astype(data_type)
cmd = ':TRAC:DEF {0}, {1}'.format(segnum, len(dacWaveDC)) # memory location and length
inst.send_scpi_cmd(cmd)
cmd = ':TRAC:SEL {0}'.format(segnum)
inst.send_scpi_cmd(cmd)

inst.timeout = 30000 #increase
inst.write_binary_data('*OPC?; :TRAC:DATA', dacWaveDC) # write, and wait while *OPC completes
inst.timeout = 10000 # return to normal

#Create a Task Table
cmd = ':TASK:COMP:LENG 2' # set task table length
inst.send_scpi_cmd(cmd)
cmd = ':TASK:COMP:SEL 1' # set task 1
inst.send_scpi_cmd(cmd)
cmd = ':TASK:COMP:SEGM 1'
inst.send_scpi_cmd(cmd) # shelving pulse
cmd = ':TASK:COMP:NEXT1 2'
inst.send_scpi_cmd(cmd)
cmd = ':TASK:COMP:SEL 2' # set task 2
inst.send_scpi_cmd(cmd)
cmd = ':TASK:COMP:SEGM 2'
inst.send_scpi_cmd(cmd) # ionization pulse
cmd = ':TASK:COMP:NEXT1 1'
inst.send_scpi_cmd(cmd)
cmd = ':TASK:COMP:WRITE' #write to FPGA
inst.send_scpi_cmd(cmd)
cmd = ':SOUR:FUNC:MODE TASK'
inst.send_scpi_cmd(cmd)
cmd = ':OUTP OFF'
rc = inst.send_scpi_cmd(cmd)