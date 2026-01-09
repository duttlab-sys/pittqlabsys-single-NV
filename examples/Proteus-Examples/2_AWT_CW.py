import os
import sys
import struct
srcpath = os.path.realpath(r'C:\Users\Duttlab\Downloads\PythonExamples\Examples\SourceFiles')
sys.path.append(srcpath)
from teproteus import TEProteusAdmin as TepAdmin
from tevisainst import TEVisaInst


import warnings # this is for GUI warnings
warnings.filterwarnings("ignore")

from tevisainst import TEVisaInst

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.widgets import Button

import keyboard
import time

#init DAC
#Set rates for DAC 
sampleRateDAC = 1.125E9

data_type = np.uint16

# Connect to instrument(PXI)
sid = 3 #PXI slot of AWT on chassis
from teproteus import TEProteusAdmin as TepAdmin
admin = TepAdmin() #required to control PXI module
inst = admin.open_instrument(slot_id=sid) 
resp = inst.send_scpi_query("*IDN?") # Get the instrument's *IDN
print('connected to: ' + resp) # Print *IDN

# initialize DAC
inst.send_scpi_cmd('*CLS; *RST')        

#AWG channel
ch = 1
cmd = ':INST:CHAN {0}'.format(ch)
inst.send_scpi_cmd(cmd)

print('CH I DAC Clk Freq {0}'.format(2.5E9))  # force to max 16 bit
cmd = ':FREQ:RAST {0}'.format(2.5E9)
inst.send_scpi_cmd(cmd)
inst.send_scpi_cmd(':INIT:CONT ON')
inst.send_scpi_cmd(':TRAC:DEL:ALL')

def makeDCData(segLen):
    global dacWave

    max_dac=65535
    half_dac=max_dac/2
    data_type = np.uint16
    
    #Set DC
    dacWave = np.zeros(segLen) + half_dac
    dacWave = dacWave.astype(data_type)
    
def makeOnData(segLen):
    global dacWave

    max_dac=65535-1
    data_type = np.uint16
    
    #Set DC
    dacWave = np.zeros(segLen) + max_dac
    dacWave = dacWave.astype(data_type)

    

def downLoad(segnum):
    global dacWave

    # Select channel
    cmd = ':INST:CHAN {0}'.format(ch)
    inst.send_scpi_cmd(cmd)

    # Define segment
    cmd = ':TRAC:DEF {0}, {1}'.format(segnum, len(dacWave))
    inst.send_scpi_cmd(cmd)

    # Select the segment
    cmd = ':TRAC:SEL {0}'.format(segnum)
    inst.send_scpi_cmd(cmd)

    # Increase the timeout before writing binary-data:
    inst.timeout = 30000
    # Send the binary-data with *OPC? added to the beginning of its prefix.
    inst.write_binary_data('*OPC?; :TRAC:DATA', dacWave)
    # Set normal timeout
    inst.timeout = 10000

    resp = inst.send_scpi_query(':SYST:ERR?')
    print("Trace Download Error = ")
    print(resp)
    
def downLoad_mrkr():
    
    global dacWave

    segnum = 1
    markerNum = 1
    
    # Marker length is a quarter of trace lengh.
    # In this dacWave is 2056
    # Marker Len is 512
    # A marker byte is 8 bits, set bit 0 sets marker 1, set bit 1, sets marker two

    mark = np.zeros(len(dacWave) // 2, np.int8)
    mark[0:256] = 17
    
    # Select channel same as CH you play waveform out on
    cmd = ':INST:CHAN {0}'.format(ch)
    inst.send_scpi_cmd(cmd)

    # Select the segment that you want to assign the marker to
    cmd = ':TRAC:SEL {0}'.format(segnum)
    inst.send_scpi_cmd(cmd)

    # Increase the timeout before writing binary-data:
    inst.timeout = 30000
    # Send the binary-data with *OPC? added to the beginning of its prefix.
    inst.write_binary_data('*OPC?; :MARK:DATA', mark)
    # Set normal timeout
    inst.timeout = 10000

    resp = inst.send_scpi_query(':SYST:ERR?')
    print("Marker Download Error = ")
    print(resp)
    
    # Select the marker
    cmd = ':MARK:SEL {0}'.format(markerNum)
    inst.send_scpi_cmd(cmd)
    
    cmd = ':MARK:STAT ON'
    inst.send_scpi_cmd(cmd)    
    

def setTask():
   
    #Direct RF Output CH
    cmd = ':INST:CHAN {0}'.format(ch)
    inst.send_scpi_cmd(cmd)

    cmd = ':TASK:COMP:LENG 2'
    inst.send_scpi_cmd(cmd)
     
    cmd = ':TASK:COMP:SEL 1' 
    inst.send_scpi_cmd(cmd)
    cmd = ':TASK:COMP:SEGM 1'
    inst.send_scpi_cmd(cmd)
    #cmd = ':TASK:COMP:DTR ON'
    #inst.send_scpi_cmd(cmd)
    cmd = ':TASK:COMP:NEXT1 2'
    inst.send_scpi_cmd(cmd)
    
    cmd = ':TASK:COMP:SEL 2' 
    inst.send_scpi_cmd(cmd)
    cmd = ':TASK:COMP:SEGM 2'
    inst.send_scpi_cmd(cmd)
    cmd = ':TASK:COMP:LOOP 1'
    inst.send_scpi_cmd(cmd)
    cmd = ':TASK:COMP:NEXT1 1'
    inst.send_scpi_cmd(cmd)
    
    cmd = ':TASK:COMP:WRITE'
    inst.send_scpi_cmd(cmd)
    cmd = ':SOUR:FUNC:MODE TASK'
    inst.send_scpi_cmd(cmd)
        

makeOnData(2048)
downLoad(1)

makeDCData(2048)
downLoad(2)

downLoad_mrkr()
setTask()

cmd = ':OUTP ON'
rc = inst.send_scpi_cmd(cmd)



inst.close_instrument()
