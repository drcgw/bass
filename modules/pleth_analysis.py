from BASS import analyze, samp_entropy_wrapper, poincare_batch, histent_wrapper, moving_statistics

import numpy as np
from numpy import NaN, Inf, arange, isscalar, asarray, array
import pandas as pd
import time as t
import sys
import os, errno

#The pleth module. Runs analysis of all aspects pertaining to pleth-signal analysis.
def pleth_analysis(Data, Settings, Results):
	pd.options.display.max_columns = 25
	
	#Run detection
	Data, Settings, Results = analyze(Data, Settings, Results)
	
	#New pleth stuff

	key = Settings['Label']
	start_time = t.clock()
	#create results table
	pleth = pd.DataFrame(columns=['Breaths', 'Recording Length (s)','Mean Breath Rate', 
								  'AUC', 'AUC STD', 'Insp Time mean', 'Insp Time std',
								  'Exp Time mean', 'Exp Time std', 'TTotal mean', 'TTotal std',
								  'Apnea Count Per Minute', 'TI Samp Ent', 'TE Samp Ent', 'TTot Samp Ent',
								  'TI Hist Ent', 'TE Hist Ent', 'TTot Hist Ent'], index = [key])
	event_type = 'Bursts'
	#total number of breaths
	try:
		pleth.ix[key]['Breaths'] = Results['Bursts'][key]['Burst Duration'].count()
		
	except:
		pleth.ix[key]['Breaths'] = NaN
	
	#length    
	try:
		t_sec = Data['trans'].index[-1]-Data['trans'].index[0]
		pleth.ix[key]['Recording Length (s)'] = t_sec
	except:
		pleth.ix[key]['Recording Length (s)'] = NaN
	#breath rate   
	try:
		t_min = t_sec/60
		pleth.ix[key]['Mean Breath Rate'] = pleth.ix[key]['Breaths']/t_min
	except:
		pleth.ix[key]['Mean Breath Rate'] = NaN
	#Area under the curve
	try:
		pleth.ix[key]['AUC'] = Results['Bursts'][key]['Burst Area'].mean()
		pleth.ix[key]['AUC STD'] = Results['Bursts'][key]['Burst Area'].std()
	except:
		pleth.ix[key]['AUC'] = NaN
		pleth.ix[key]['AUC'] = NaN
	#inspiration    
	try:
		pleth.ix[key]['Insp Time mean'] = Results['Bursts'][key]['Burst Duration'].mean()
		pleth.ix[key]['Insp Time std'] = Results['Bursts'][key]['Burst Duration'].std()
	except:
		pleth.ix[key]['Insp Time mean'] = NaN
		pleth.ix[key]['Insp Time std'] = NaN
	#Expiration
	try:
		pleth.ix[key]['Exp Time mean'] = Results['Bursts'][key]['Interburst Interval'].mean()
		pleth.ix[key]['Exp Time std'] = Results['Bursts'][key]['Interburst Interval'].std()
	except:
		pleth.ix[key]['Exp Time mean'] = NaN
		pleth.ix[key]['Exp Time std'] = NaN
	#TTOTAL
	try:
		pleth.ix[key]['TTotal mean'] = Results['Bursts'][key]['Total Cycle Time'].mean()
		pleth.ix[key]['TTotal std'] = Results['Bursts'][key]['Total Cycle Time'].std()
	except:
		pleth.ix[key]['TTotal mean'] = NaN
		pleth.ix[key]['TTotal std'] = NaN
	
	#apnea #This is what John Wrote
	#get the differences in time between burst starts
	timedifflst = []
	
	for i in np.arange(0, (len(Results['Bursts'][key]['Burst Start'])-1)):
		timediff = Results['Bursts'][key]['Burst Start'].iloc[i+1]-Results['Bursts'][key]['Burst Start'].iloc[i]
		timedifflst.append(timediff)
		
	#set apnea index as the mean of time differences
	apnea_index = np.array(timedifflst).mean()
	
	#definition of apnea: 120%  above the apnea index
	apnea_thresh = 1.20 * apnea_index
	
	#count apneas: if length of time between Burst Starts is greater than apnea_thresh, +1
	apnea_count = list((timedifflst > apnea_thresh)).count(True)
	
	#normalize by time
	apneas_per_minute = apnea_count/pleth.ix[key]['Recording Length (s)']*60
	
	#add to pleth dataframe
	pleth.ix[key]['Apnea Count Per Minute'] = apneas_per_minute
	
	#Shannon Entropy
	try:
		meas = 'Burst Duration'
		Results = samp_entropy_wrapper(event_type, meas, Data, Settings, Results)
		pleth.ix[key]['TI Samp Ent'] = float(Results['Sample Entropy'][meas])
	except:
		pleth.ix[key]['TI Samp Ent'] = NaN
	
	try:
		meas = 'Interburst Interval'
		Results = samp_entropy_wrapper(event_type, meas, Data, Settings, Results)
		pleth.ix[key]['TE Samp Ent'] = float(Results['Sample Entropy'][meas])
	except:
		pleth.ix[key]['TE Samp Ent'] = NaN
		
	try:
		meas = 'Total Cycle Time'
		Results = samp_entropy_wrapper(event_type, meas, Data, Settings, Results)
		pleth.ix[key]['TTot Samp Ent'] = float(Results['Sample Entropy'][meas])
	except:
		pleth.ix[key]['TTot Samp Ent'] = NaN
	
	#poincare
	try:
		meas = 'Total Cycle Time'
		Results = poincare_batch(event_type, meas, Data, Settings, Results)
		meas = 'Burst Duration'
		Results = poincare_batch(event_type, meas, Data, Settings, Results)
		meas = 'Interburst Interval'
		Results = poincare_batch(event_type, meas, Data, Settings, Results)
	except:
		print "Poincare Failed"
	
	#Hist Ent
	try:
		meas = 'all'
		Results = histent_wrapper(event_type, meas, Data, Settings, Results)
		pleth.ix[key]['TI Hist Ent'] = float(Results['Histogram Entropy']['Burst Duration'])
		pleth.ix[key]['TE Hist Ent'] = float(Results['Histogram Entropy']['Interburst Interval'])
		pleth.ix[key]['TTot Hist Ent'] = float(Results['Histogram Entropy']['Total Cycle Time'])
		
	except:
		print "Histogram Entropy Failed"
		pleth.ix[key]['TI Hist Ent'] = NaN
		pleth.ix[key]['TE Hist Ent'] = NaN
		pleth.ix[key]['TTot Hist Ent'] = NaN
	
	try:
		#Moving Stats
		event_type = 'Bursts'
		meas = 'Total Cycle Time'
		window = 30 #seconds
		Results = moving_statistics(event_type, meas, window, Data, Settings, Results)
	except:
		pass
	
	pleth.to_csv(r"%s/%s_Pleth.csv"%(Settings['Output Folder'],Settings['Label']))
	end_time = t.clock()
	
	print 'Pleth Analysis Complete: %s sec' %np.round((end_time- start_time), 4)
	pleth