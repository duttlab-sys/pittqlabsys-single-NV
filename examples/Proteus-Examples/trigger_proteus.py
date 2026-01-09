import os
import sys
import struct
import numpy as np
import time
import ipywidgets as widgets
import matplotlib.pyplot as plt
srcpath = os.path.realpath(r'C:\Users\Duttlab\Downloads\PythonExamples\Examples\SourceFiles')
sys.path.append(srcpath)
from teproteus import TEProteusAdmin as TepAdmin
from tevisainst import TEVisaInst
import numpy as np
# testing aom with worst case scenario: longest pulse durations and shortest waiting time:
# Connect to instrument(PXI)
sid = 6 #PXI slot of AWT on chassis
admin = TepAdmin() #required to control PXI module
inst = admin.open_instrument(slot_id=sid)
resp = inst.send_scpi_query("*IDN?") # Get the instrument's *IDN
print('connected to: ' + resp) # Print *IDN

# initialize DAC
inst.send_scpi_cmd('*CLS; *RST')


inst.send_scpi_cmd(':FREQ:RAST 1.25e9')

resp = inst.send_scpi_query(':SYST:ERR?')
print(resp)

model_name = inst.send_scpi_query('SYST:INF:MODel?')
print('Model: {0} '.format(model_name))

# Get number of channels
resp = inst.send_scpi_query(":INST:CHAN? MAX")
print("Number of channels: " + resp)
num_channels = int(resp)

# Get model dependant parameters:

if model_name.startswith('P948'):
    bpp = 2
    max_dac = 65535
    wpt_type = np.uint16
    channels_per_dac = 2
elif model_name.startswith('P908'):
    bpp = 1
    max_dac = 255
    wpt_type = np.uint8
    channels_per_dac = 1
else:
    bpp = 2
    max_dac = 65535
    wpt_type = np.uint16
    channels_per_dac = 2

half_dac = max_dac / 2.0

# Get the maximal number of segments
resp = inst.send_scpi_query(":TRACe:SELect:SEGMent? MAX")
print("Max segment number: " + resp)
max_seg_number = int(resp)

# Get the available memory in bytes of wavform-data (per DDR):
resp = inst.send_scpi_query(":TRACe:FREE?")
arbmem_capacity = (int(resp) // 64) * 64
print("Available memory per DDR: {0:,} wave-bytes".format(arbmem_capacity))

max_seglen = arbmem_capacity // bpp
print('Max segment length: {0:,}'.format(max_seglen))
# Build 3 waveforms

seglen = 4096
cyclelen = seglen
ncycles = seglen / cyclelen
waves = [None for _ in range(3)]

# sin wave:
x = np.linspace(
    start=0, stop=2 * np.pi * ncycles, num=seglen, endpoint=False)
y = (np.sin(x) + 1.0) * half_dac
y = np.round(y)
y = np.clip(y, 0, max_dac)
waves[0] = y.astype(wpt_type)

# triangle wave:
# x = np.linspace(
#   start=0, stop=2 * np.pi * ncycles, num=seglen, endpoint=False)
# y = np.sin(x)
# y = np.arcsin(y)* 2 * half_dac / np.pi + half_dac
# y = np.round(y)
# y = np.clip(y, 0, max_dac)
# waves[1] = y.astype(wpt_type)

# square wave
# x = np.linspace(start=0, stop=seglen, num=seglen, endpoint=False)
# y = np.fmod(x, cyclelen)
# y = (y <= cyclelen / 2) * max_dac
# y = np.round(y)
# y = np.clip(y, 0, max_dac)
# waves[2] = y.astype(wpt_type)
# download 3 waveforms to each DDR

for ichan in range(num_channels):
    if ichan % channels_per_dac == 0:
        channb = ichan + 1
        # Select channel
        cmd = ':INST:CHAN {0}'.format(channb)
        inst.send_scpi_cmd(cmd)
        for iseg in range(1):
            segnum = iseg + 1
            print('Downloading segment {0} of channel {1}'.format(segnum, channb))
            # Define segment
            cmd = ':TRAC:DEF {0}, {1}'.format(segnum, seglen)
            inst.send_scpi_cmd(cmd)

            # Select the segment
            cmd = ':TRAC:SEL {0}'.format(segnum)
            inst.send_scpi_cmd(cmd)
            print(segnum, channb, len(waves[0]), seglen)
            # Send the binary-data:
            inst.write_binary_data(':TRAC:DATA', waves[iseg])

resp = inst.send_scpi_query(':SYST:ERR?')
print(resp)
# Play the first segment in each channel

for ichan in range(num_channels):
    channb = ichan + 1
    # Select channel
    cmd = ':INST:CHAN {0}'.format(channb)
    inst.send_scpi_cmd(cmd)
    # Play the specified segment at the selected channel:
    cmd = ':SOUR:FUNC:MODE:SEGM {0}'.format(1)
    inst.send_scpi_cmd(cmd)

    # Turn on the output of the selected channel:
    inst.send_scpi_cmd(':OUTP ON')

resp = inst.send_scpi_query(':SYST:ERR?')
print(resp)

# Set the thrshhold level of the trigger
inst.send_scpi_cmd(':TRIG:LEV 0.1')

resp = inst.send_scpi_cmd(':INST:ACT {0}'.format(1))
for ichan in range(num_channels):
    channb = ichan + 1
    # Select channel- Trigger source: EXT=TRG1 ,INT=CPU
    cmd = ':INST:CHAN {0}'.format(channb)
    inst.send_scpi_cmd(cmd)
    inst.send_scpi_cmd('TRIG:SOUR:ENAB {}'.format('TRG1'))
    inst.send_scpi_cmd('TRIG:SEL {}'.format('TRG1'))
    inst.send_scpi_cmd('TRIG:STAT ON')

# Define task-table of 3 tasks in each channel.
# The first task shall wait for trigger1.
# In order to

tasklen = 3
for ichan in range(num_channels):
    channb = ichan + 1
    # Select channel
    cmd = ':INST:CHAN {0}'.format(channb)
    inst.send_scpi_cmd(cmd)

    # Compose the task-table rows:
    cmd = ':TASK:COMP:LENG {0}'.format(tasklen)
    inst.send_scpi_cmd(cmd)

    for itask in range(tasklen):
        tasknb = itask + 1
        segnb = itask + 1
        nloops = 2 ** tasknb

        cmd = ':TASK:COMP:SEL {0}'.format(tasknb)
        inst.send_scpi_cmd(cmd)

        inst.send_scpi_cmd(':TASK:COMP:TYPE SING')

        cmd = ':TASK:COMP:SEGM {0}'.format(segnb)
        inst.send_scpi_cmd(cmd)

        cmd = ':TASK:COMP:LOOP {0}'.format(nloops)
        inst.send_scpi_cmd(cmd)

        if 1 == tasknb:
            # Trigger source: EXT=TRG1 ,INT=CPU
            # in case of :TRIG:COUPLE ON need to put INT instead of CPU??
            cmd = ':TASK:COMP:ENAB TRG1'
            inst.send_scpi_cmd(cmd)
        else:
            cmd = ':TASK:COMP:ENAB NONE'
            inst.send_scpi_cmd(cmd)

        if tasklen == tasknb:
            cmd = ':TASK:COMP:NEXT1 1'
            inst.send_scpi_cmd(cmd)
        else:
            cmd = ':TASK:COMP:NEXT1 {0}'.format(tasknb + 1)
            inst.send_scpi_cmd(cmd)

        # Write the task-table
        inst.send_scpi_cmd(':TASK:COMP:WRIT')

        # Set Task-Mode
        inst.send_scpi_cmd(':FUNC:MODE TASK')


resp = inst.send_scpi_query(':SYST:ERR?')
print(resp)


# :TRIG:COUPLE ON will require trigger from master, if single module in use and single channel, it should be off (default)
# If more then 1 channel requires trigger, it should be ON and tasks trigger should be INT instead of CPU
resp = inst.send_scpi_cmd(':TRIG:COUPLE ON')
print(resp)
# for trigger source INT send:
inst.send_scpi_cmd(':INST:CHAN 1')
# trigger command:
inst.send_scpi_cmd('*TRG')

inst.close_instrument()
admin.close_inst_admin()