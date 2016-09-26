from BASS import analyze, samp_entropy_wrapper, poincare_batch, histent_wrapper, moving_statistics, psd_event, ap_entropy_wrapper

import numpy as np
from numpy import NaN, Inf, arange, isscalar, asarray, array
import pandas as pd
import time as t
import sys
import os, errno

#The ekg module. Runs analysis of all aspects pertaining to ekg-signal analysis. Modified from the pleth_analysis module by Ryan Thorpe.

#Change 'breaths' to beats and breath-rate to heart-rate

def ekg_analysis(Data, Settings, Results):
	
	pd.options.display.max_columns = 25
	
	#Run detection
	Data, Settings, Results = analyze(Data, Settings, Results)
	
	#New ekg stuff

	key = Settings['Label']
	start_time = t.clock()
	#create results table
	ekg = pd.DataFrame(columns=['Heartbeats', 'Recording Length (s)','Mean Heartrate', 
								'TTotal mean', 'TTotal std', 'PA Samp Ent', 
								'Intv Samp Ent', 'PA Hist Ent', 'Intv Hist Ent'], 
								index = [key])
	event_type = 'Peaks'
	#total number of heartbeats
	try:
		ekg.ix[key]['Heartbeats'] = Results['Peaks'][key]['Peaks Amplitude'].count()
		
	except:
		ekg.ix[key]['Heartbeats'] = NaN
	
	#length    
	try:
		t_sec = Data['trans'].index[-1]-Data['trans'].index[0]
		ekg.ix[key]['Recording Length (s)'] = t_sec
	except:
		ekg.ix[key]['Recording Length (s)'] = NaN
	#Heartrate   
	try:
		t_min = t_sec/60
		ekg.ix[key]['Mean Heartrate'] = ekg.ix[key]['Heartbeats']/t_min
	except:
		ekg.ix[key]['Mean Heartrate'] = NaN
		
	#RT Modification: deleted area under the curve
	#RT Modification: deleted inspiration    
	#RT Modification: deleted expiration

	#TTOTAL
	try:
		ekg.ix[key]['TTotal mean'] = Results['Peaks'][key]['Intervals'].mean()
		ekg.ix[key]['TTotal std'] = Results['Peaks'][key]['Intervals'].std()
	except:
		ekg.ix[key]['TTotal mean'] = NaN
		ekg.ix[key]['TTotal std'] = NaN
	
	#Shannon Entropy
	try:
		meas = 'Peaks Amplitude'
		Results = samp_entropy_wrapper(event_type, meas, Data, Settings, Results)
		ekg.ix[key]['PA Samp Ent'] = float(Results['Sample Entropy'][meas])
	except:
		ekg.ix[key]['PA Samp Ent'] = NaN
	
	try:
		meas = 'Intervals'
		Results = samp_entropy_wrapper(event_type, meas, Data, Settings, Results)
		ekg.ix[key]['Intv Samp Ent'] = float(Results['Sample Entropy'][meas])
	except:
		ekg.ix[key]['Intv Samp Ent'] = NaN
	
	#poincare
	try:
		meas = 'Peaks Amplitude'
		Results = poincare_batch(event_type, meas, Data, Settings, Results)
		meas = 'Intervals'
		Results = poincare_batch(event_type, meas, Data, Settings, Results)
	except:
		print "Poincare Failed"
	
	#Hist Ent
	try:
		meas = 'all'
		Results = histent_wrapper(event_type, meas, Data, Settings, Results)
		ekg.ix[key]['PA Hist Ent'] = float(Results['Histogram Entropy']['Peaks Amplitude'])
		ekg.ix[key]['Intv Hist Ent'] = float(Results['Histogram Entropy']['Intervals'])
		
	except:
		print "Histogram Entropy Failed"
		ekg.ix[key]['PA Hist Ent'] = NaN
		ekg.ix[key]['Intv Hist Ent'] = NaN
	
	try:
		#Moving Stats
		meas = 'Intervals'
		window = 30 #seconds
		Results = moving_statistics(event_type, meas, window, Data, Settings, Results)
	except:
		pass
	
	#Power Spec Density
	Settings['PSD-Event'] = pd.Series(index = ['Hz','ULF', 'VLF', 'LF','HF','dx'])

	Settings['PSD-Event']['hz'] = 100 #freqency that the interpolation and PSD are performed with.
	Settings['PSD-Event']['ULF'] = 1 #max of the range of the ultra low freq band. range is 0:ulf
	Settings['PSD-Event']['VLF'] = 2 #max of the range of the very low freq band. range is ulf:vlf
	Settings['PSD-Event']['LF'] = 5 #max of the range of the low freq band. range is vlf:lf
	Settings['PSD-Event']['HF'] = 50 #max of the range of the high freq band. range is lf:hf. hf can be no more than (hz/2)
	Settings['PSD-Event']['dx'] = 10 #segmentation for the area under the curve.
	
	meas = 'Intervals'
	scale = 'raw'
	Results = psd_event(event_type, meas, key, scale, Data, Settings, Results) #Run PSD-Event analysis and generate graph
	
	Results['PSD-Event'][key] #Output PSD-Event dataframe
	
	#Approximate Entropy
	meas = 'Intervals'
	Results = ap_entropy_wrapper(event_type, meas, Data, Settings, Results) #Run approx. ent. analysis and generate graph
	Results['Approximate Entropy'] #Output approx. ent. dataframe
	
	#Save cumulative ekg data
	ekg.to_csv(r"%s/%s_EKG.csv"%(Settings['Output Folder'],Settings['Label']))
	end_time = t.clock()
	
	print 'Heart Rate Varibility Analysis Complete: %s sec' %np.round((end_time- start_time), 4)
	#ekg