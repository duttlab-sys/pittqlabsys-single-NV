import os
import sys
import struct
srcpath = os.path.realpath(r'C:\Users\Duttlab\Downloads\PythonExamples\Examples\SourceFiles')
sys.path.append(srcpath)
from teproteus import TEProteusAdmin as TepAdmin
from tevisainst import TEVisaInst


import matplotlib.pyplot as plt
import numpy as np

# Connect to instrument(PXI)
sid = 6 #PXI slot of AWT on chassis
admin = TepAdmin() #required to control PXI module
inst = admin.open_instrument(slot_id=sid)
resp = inst.send_scpi_query("*IDN?") # Get the instrument's *IDN
print('connected to: ' + resp) # Print *IDN

# initialize DAC
inst.send_scpi_cmd('*CLS; *RST')


#AWG channel
ch = 1 # everything after relates to CH 1
cmd = ':INST:CHAN {0}'.format(ch) #Everything is now CH one only operation
inst.send_scpi_cmd(cmd)

sampleRateDAC = 1.25E9
cmd = ':FREQ:RAST {0}'.format(sampleRateDAC) 
inst.send_scpi_cmd(cmd)
cmd = ':TRAC:DEL:ALL' # Clear CH 1 Memory
inst.send_scpi_cmd(cmd)
cmd = ':INIT:CONT ON' # play waveform continuously

# Make a waveform
amp = 1  
cycles = 2000
segLen = 4096*4 # must be a multiple of 64
time = np.linspace(0, segLen-1, segLen)
w = 2 * np.pi 
dacWave = amp * np.sin(w*time*cycles/segLen) 
print('Frequency {0} Hz'.format(sampleRateDAC*cycles/segLen))
print('SegLen {0} '.format(segLen))

#scale to 16 bits
max_dac=65535 # Max Dac
half_dac=max_dac/2 # DC Level
data_type = np.uint16 # DAC data type 
dacWave = ((dacWave) + 1.0) * half_dac  
dacWave = dacWave.astype(data_type) 

# Create a waveform memory segment
segnum = 1
cmd = ':TRAC:DEF {0}, {1}'.format(segnum, len(dacWave)) # memory location and length
inst.send_scpi_cmd(cmd)

# Select the segment
cmd = ':TRAC:SEL {0}'.format(segnum)
inst.send_scpi_cmd(cmd)

#Download
inst.timeout = 30000 #increase
inst.write_binary_data('*OPC?; :TRAC:DATA', dacWave) # write, and wait while *OPC completes
inst.timeout = 10000 # return to normal

markerNum = 1
mark = np.zeros(len(dacWave) // 4, np.int8)
mark[0:128] = 17  # 00010001 <- Marker 1 set bit 0 and 4,

# Increase the timeout before writing binary-data:
inst.timeout = 30000
# Send the binary-data with *OPC? added to the beginning of its prefix.
inst.write_binary_data('*OPC?; :MARK:DATA', mark)
# Set normal timeout
inst.timeout = 10000

resp = inst.send_scpi_query(':SYST:ERR?')
print("Marker Download Error = ")
print(resp)

# Select the marker to assign to above trace
cmd = ':MARK:SEL {0}'.format(markerNum)
inst.send_scpi_cmd(cmd)


# Create and download a second Segment
segnum = 2
dacWaveDC = np.ones(segLen)
dacWaveDC = dacWaveDC * half_dac  # scale
dacWaveDC = dacWaveDC.astype(data_type)  
cmd = ':TRAC:DEF {0}, {1}'.format(segnum, len(dacWaveDC)) # memory location and length
inst.send_scpi_cmd(cmd)
cmd = ':TRAC:SEL {0}'.format(segnum)
inst.send_scpi_cmd(cmd)

inst.timeout = 30000 #increase
inst.write_binary_data('*OPC?; :TRAC:DATA', dacWaveDC) # write, and wait while *OPC completes
inst.timeout = 10000 # return to normal

markerNum = 1
mark = np.zeros(len(dacWave) // 4, np.int8)
#mark[0:128] = 17  # 00010001 <- Marker 1 set bit 0 and 4,

# Increase the timeout before writing binary-data:
inst.timeout = 30000
# Send the binary-data with *OPC? added to the beginning of its prefix.
inst.write_binary_data('*OPC?; :MARK:DATA', mark)
# Set normal timeout
inst.timeout = 10000

resp = inst.send_scpi_query(':SYST:ERR?')
print("Marker Download Error = ")
print(resp)

# Select the marker to assign to above trace
cmd = ':MARK:SEL {0}'.format(markerNum)
inst.send_scpi_cmd(cmd)

#Create a Task Table
cmd = ':TASK:COMP:LENG 2' # set task table length
inst.send_scpi_cmd(cmd)
cmd = ':TASK:COMP:SEL 1' # set task 1
inst.send_scpi_cmd(cmd)
cmd = ':TASK:COMP:SEGM 1'
inst.send_scpi_cmd(cmd)
cmd = ':TASK:COMP:NEXT1 2'
inst.send_scpi_cmd(cmd)

cmd = ':TASK:COMP:SEL 2' # set task 2
inst.send_scpi_cmd(cmd)
cmd = ':TASK:COMP:SEGM 2'
inst.send_scpi_cmd(cmd)
cmd = ':TASK:COMP:LOOP 10'
inst.send_scpi_cmd(cmd)
cmd = ':TASK:COMP:NEXT1 1'
inst.send_scpi_cmd(cmd)

cmd = ':TASK:COMP:WRITE' #write to FPGA
inst.send_scpi_cmd(cmd)

cmd = ':SOUR:FUNC:MODE TASK'
inst.send_scpi_cmd(cmd)

cmd = ':MARK:STAT ON'
inst.send_scpi_cmd(cmd)

cmd = ':OUTP ON'
rc = inst.send_scpi_cmd(cmd)