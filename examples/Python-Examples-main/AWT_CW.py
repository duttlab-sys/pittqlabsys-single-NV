import os
import sys
import os
import sys
import struct
srcpath = os.path.realpath(r'C:\Users\Duttlab\Downloads\PythonExamples\Examples\SourceFiles')
sys.path.append(srcpath)
from teproteus import TEProteusAdmin as TepAdmin
from tevisainst import TEVisaInst


import warnings # this is for GUI warnings
warnings.filterwarnings("ignore")



import numpy as np

import matplotlib.pyplot as plt
from matplotlib.widgets import Button

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

'''
# Connect to instrument(LAN)


inst_addr = 'TCPIP::192.168.1.15::5025::SOCKET' 
inst = TEVisaInst(inst_addr)
resp = inst.send_scpi_query("*IDN?")
print('connected to: ' + resp)

# initialize DAC
inst.send_scpi_cmd('*CLS; *RST')        

#AWG channel
ch = 3
cmd = ':INST:CHAN {0}'.format(ch)
inst.send_scpi_cmd(cmd)

print('CH I DAC Clk Freq {0}'.format(sampleRateDAC))  # force to max 16 bit
cmd = ':FREQ:RAST {0}'.format(sampleRateDAC)
inst.send_scpi_cmd(cmd)
inst.send_scpi_cmd(':INIT:CONT ON')
inst.send_scpi_cmd(':TRAC:DEL:ALL')

def makeDCData(segLen):
    global dacWaveI
    global dacWaveQ
    
    max_dac=65535
    half_dac=max_dac/2
    data_type = np.uint16
    
    #Set DC
    dacWaveDC = np.zeros(segLen) + half_dac    
    dacWaveI = dacWaveDC.astype(data_type)   
    
    dacWaveDC = np.zeros(segLen) + half_dac       
    dacWaveQ = dacWaveDC.astype(data_type)
    
def makeOnData(segLen):
    global dacWaveI
    global dacWaveQ
    
    max_dac=65535-1
    half_dac=max_dac/2
    data_type = np.uint16
    
    #Set DC
    dacWaveDC = np.zeros(segLen) + max_dac    
    dacWaveI = dacWaveDC.astype(data_type)   
    
    dacWaveDC = np.zeros(segLen) + half_dac      
    dacWaveQ = dacWaveDC.astype(data_type)    
    
    

def downLoad_IQ_DUC(segnum):
    global dacWaveI
    global dacWaveQ

    arr_tuple = (dacWaveI, dacWaveQ)
    dacWaveIQ = np.vstack(arr_tuple).reshape((-1,), order='F')


    # Select channel
    cmd = ':INST:CHAN {0}'.format(ch)
    inst.send_scpi_cmd(cmd)

    # Define segment
    cmd = ':TRAC:DEF {0}, {1}'.format(segnum, len(dacWaveIQ))
    inst.send_scpi_cmd(cmd)

    # Select the segment
    cmd = ':TRAC:SEL {0}'.format(segnum)
    inst.send_scpi_cmd(cmd)

    # Increase the timeout before writing binary-data:
    inst.timeout = 30000
    # Send the binary-data with *OPC? added to the beginning of its prefix.
    inst.write_binary_data('*OPC?; :TRAC:DATA', dacWaveIQ)
    # Set normal timeout
    inst.timeout = 10000

    resp = inst.send_scpi_query(':SYST:ERR?')
    print("Trace Download Error = ")
    print(resp)

    cmd = ':SOUR:MODE DUC'
    resp = inst.send_scpi_cmd(cmd)
    
    cmd = ':SOUR:INT X8'
    resp = inst.send_scpi_cmd(cmd)

    cmd = ':SOUR:IQM ONE'
    resp = inst.send_scpi_cmd(cmd)
    
    sampleRateDACInt = sampleRateDAC * 8
    print('Sample Clk Freq {0}'.format(sampleRateDACInt))
    cmd = ':FREQ:RAST {0}'.format(sampleRateDACInt)
    resp = inst.send_scpi_cmd(cmd)
    
    resp = inst.send_scpi_query(':SYST:ERR?')
    print("IQ Set Error = ")
    print(resp)

    cmd = ':SOUR:NCO:CFR1 {0}'.format(ncoFreq)
    resp = inst.send_scpi_cmd(cmd)
    
    
def downLoad_mrkr():
    
    global dacWaveQ

    segnum = 1
    markerNum = 1
    
    # Marker length is a quarter of trace lengh.
    # In this dacWaveQ is 2056
    # Marker Len is 512
    # A marker byte is 8 bits, set bit 0 sets marker 1, set bit 1, sets marker two
    
    # print(len(dacWaveQ))
    markerLen = len(dacWaveQ) // 2
    markerOnPoints = 256
    # print(markerOnPoints)
    # print(markerLen)
    markerOn = markerOnPoints
    markerOff = markerLen - markerOnPoints
    mark_1 = np.ones(markerOn, np.int8) # add one to this if you want to set market 2, make it three if you want marker one and two on.
    mark_0 = np.zeros(markerOff, np.int8)
    mark = np.concatenate([mark_1, mark_0])
    print(len(dacWaveQ))
    print(len(mark))
    print(mark)
    
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
    cmd = ':TASK:COMP:DTR ON'
    inst.send_scpi_cmd(cmd)
    cmd = ':TASK:COMP:NEXT1 2'
    inst.send_scpi_cmd(cmd)
    
    cmd = ':TASK:COMP:SEL 2' 
    inst.send_scpi_cmd(cmd)
    cmd = ':TASK:COMP:SEGM 2'
    inst.send_scpi_cmd(cmd)
    cmd = ':TASK:COMP:LOOP 5'
    inst.send_scpi_cmd(cmd)
    cmd = ':TASK:COMP:NEXT1 1'
    inst.send_scpi_cmd(cmd)
    
    cmd = ':TASK:COMP:WRITE'
    inst.send_scpi_cmd(cmd)
    cmd = ':SOUR:FUNC:MODE TASK'
    inst.send_scpi_cmd(cmd)
        


makeOnData(2048)
downLoad_IQ_DUC(1)

makeOnData(2048)
downLoad_IQ_DUC(2)

#downLoad_mrkr()
setTask()

cmd = ':OUTP ON'
rc = inst.send_scpi_cmd(cmd)

'''
inst.close_instrument()
