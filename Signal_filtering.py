# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 16:45:29 2018

@author: rafael
Filtering ECG data

"""

#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.signal import butter, lfilter, cheb2ord, cheby2

## importing data part
ecg_data_frame = pd.read_csv('samples.csv', header = 0)

## filtering part

def cheby_lowpass(wp, ws, fs, gpass, gstop):
    wp = wp/fs
    ws = ws/fs
    order, wn = cheb2ord(wp, ws, gpass, gstop)
    b, a = cheby2(order, gstop, wn)
    return b, a

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a
    
def cheby_lowpass_filter(data, cutoff, fs, gpass, gstop):
    b, a = cheby_lowpass(cutoff[0], cutoff[1], fs, gpass, gstop)
    y = lfilter(b, a, data)
    return y

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


# Filter requirements.
order = 3
band = 35   #bandwidth
disp = 10   #displacement
fs = 360       # sample rate, Hz
cutoff = 50  # desired cutoff frequency of the filter, Hz

cheby_freq = [(cutoff+disp)-band/2, (cutoff+disp)+band/2]

# Get the filter coefficients so we can check its frequency response.
b, a = butter_lowpass(cutoff, fs, order)

# Demonstrate the use of the filter.
# First make some data to be filtered.
t = ecg_data_frame.iloc[:,0]
T = 0.00277778         # seconds
n = t.size # total number of samples


# Filter the data, and plot both the original and filtered signals.
y1 = butter_lowpass_filter(ecg_data_frame.iloc[:,1], cutoff, fs, order)
y2 = butter_lowpass_filter(ecg_data_frame.iloc[:,2], cutoff, fs, order)

z1 = cheby_lowpass_filter(ecg_data_frame.iloc[:,1], cheby_freq, fs, 1, 10)
z2 = cheby_lowpass_filter(ecg_data_frame.iloc[:,2], cheby_freq, fs, 1, 10)

## plotting part
sns.set()

plt.figure(1)
plt.clf() 
ax1 = plt.subplot(2,2,1)
ax1.plot(t, ecg_data_frame.iloc[:,1], linewidth = 2, label = ecg_data_frame.columns[1])
ax1.plot(t, y1, linewidth = 1, label = 'filtered', color = 'red')
ax1.set_title('butterworth 01')
ax1.axis([3, 4.6, -0.7, 0.9])
ax2 = plt.subplot(2,2,2)
ax2.plot(t, ecg_data_frame.iloc[:,2], linewidth = 2, label = ecg_data_frame.columns[2])
ax2.plot(t, y2, linewidth = 1, label = 'filtered', color = 'red')
ax2.set_title('butterworth 02')
ax2.axis([3, 4.6, -0.7, 0.9])

ax3 = plt.subplot(2,2,3)
ax3.plot(t, ecg_data_frame.iloc[:,1], linewidth = 2, label = ecg_data_frame.columns[1])
ax3.plot(t, z1, linewidth = 1, label = 'filtered', color = 'red')
ax3.set_title('chebyshev 01')
ax3.axis([3, 4.6, -0.7, 0.9])
ax4 = plt.subplot(2,2,4)
ax4.plot(t, ecg_data_frame.iloc[:,2], linewidth = 2, label = ecg_data_frame.columns[2])
ax4.plot(t, z2, linewidth = 1, label = 'filtered', color = 'red')
ax4.set_title('chebyshev 02')
ax4.axis([3, 4.6, -0.7, 0.9])

plt.tight_layout()