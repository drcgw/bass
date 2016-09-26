from bass_functions import *
from modules.pleth_analysis import pleth_analysis
from modules.ekg_analysis import ekg_analysis

class BASS_Dataset(object):
    '''
    Imports dataset as an object
    
    Attributes
    ----------
    Batch: static list
    	Contains all instances of the BASS_Dataset object in order to be referenced by the global runBatch function.
	Data: library
		Instance data
	Settings: library
		Instance settings
	Results: library
		Instance results
    
    Methods
    --------
    run_analysis(settings = self.Settings, analysis_module): function
    	Highest level of the object-oriented analysis pipeline. First syncs the settings of all BASS_Dataset objects 
    	(stored in Batch), then runs the specified analysis module on each one.
    	
    Parameters
    ----------
    inputLocation, label, outputLocation, fileType (default='plain'), timeScale (default='seconds')
    
    Required Input
    --------------
    inputLocation, label, outputFolder
    
    Notes:
    ------
    Analysis must be called after the object is initialize and Settings added if the Settings are to be added manually
    (not via the interactive check and load settings function). Analysis runs according to batch-oriented protocol and
    is specific to the analysis module determined by the "analysis_module" parameter.
    
    '''
    Batch = []
    
    def __init__(self, inputDir, fileName, outputDir, fileType='Plain', timeScale='seconds'):
        self.Data = {}
        self.Settings = {}
        self.Results = {}        
        self.Settings['folder'] = inputDir
        self.Settings['Label'] = fileName
        self.Settings['Output Folder'] = outputDir
        self.Settings['File Type'] = fileType
        self.Settings['Time Scale'] = timeScale
        self.Batch.append(self) #Appends each object instance into a list of datasets (stored as mutable static object in class namespace)
        print "\n############   ", self.Settings['Label'], "   ############\n" #Display object instance label
        self.Data, self.Settings = load_wrapper(self.Data, self.Settings) #Loads data and settings
		
    def run_analysis(self, analysis_mod, settings = None, batch = True):
		'''
		Runs in either single (batch=False) or batch mode. Assuming batch mode, this function first syncs settings of each dataset within 
		Bass_Dataset.Batch to the entered parameter "settings", then runs analysis on each instance within Batch.
		
		Parameters
		----------
		analysis_mod: string
			the name of the BASS_Dataset module which will be used to analyze the batch of datasets
		settings: string or dictionary
			can be entered as the location of a settings file or the actual settings dictionary (default = self.Settings)
		batch: boolean
			determines if the analysis is performed on only the self-instance or as a batch on all object instances (default = True)
		
		Returns
		-------
		None
		
		'''
		
		#Run batch if "batch" is True
		if batch == True:
			#Sets default "settings" to self.Settings
			if settings == None:
				settings = self.Settings
			
			for dataset in BASS_Dataset.Batch:
				#Sync settings to those of a specific object instance
				try:				
					exclusion_list = ['plots folder', 'folder', 
					  'Sample Rate (s/frame)', 'Output Folder', 
					  'Baseline', 'Baseline-Rolling', 'Settings File', 'Time Scale',
					  'Label', 'File Type', 'HDF Key', 'HDF Channel']
					
					for key in settings.keys():
						if key not in exclusion_list:
							dataset.Settings[key] = settings[key]
		
				#Sync settings by passing in the location for a settings file
				except:
					dataset.Settings['Settings File'] = settings
					dataset.Settings = load_settings(dataset.Settings)
				
				print "\n############   ", self.Settings['Label'], "   ############\n" #Display object instance label
				
				if analysis_mod == 'pleth':
					pleth_analysis(dataset.Data, dataset.Settings, dataset.Results)
				elif analysis_mod == 'ekg':
					ekg_analysis(dataset.Data, dataset.Settings, dataset.Results)
		
		#Run a single object if "batch" is False
		else:
			if settings != None:
				#Sync settings to those of a specific object instance
				try:				
					exclusion_list = ['plots folder', 'folder', 
					  'Sample Rate (s/frame)', 'Output Folder', 
					  'Baseline', 'Baseline-Rolling', 'Settings File', 'Time Scale',
					  'Label', 'File Type', 'HDF Key', 'HDF Channel']
					
					for key in settings.keys():
						if key not in exclusion_list:
							self.Settings[key] = settings[key]
		
				#Sync settings by passing in the location for a settings file
				except:
					self.Settings['Settings File'] = settings
					self.Settings = load_settings(self.Settings)
				
			if analysis_mod == 'pleth':
				pleth_analysis(self.Data, self.Settings, self.Results)
			elif analysis_mod == 'ekg':
				ekg_analysis(self.Data, self.Settings, self.Results)



#END OF CODE
print "BASS ready!"