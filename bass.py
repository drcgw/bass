import numpy as np
import scipy
from scipy import signal, fft, arange
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
import pandas as pd
import sys
import matplotlib.patches as patches
import time as t
import sys
import os, errno
from numpy import NaN, Inf, arange, isscalar, asarray, array
from matplotlib.widgets import *
import datetime
from matplotlib import cbook
from scipy.signal import butter, lfilter
from math import log
from scipy.stats.stats import pearsonr
from PIL import Image
from pandas.tools.plotting import autocorrelation_plot
#
#Upload
#Data and Settings. 
#
def load_wrapper(Data, Settings):
    """
    Wrapper that chooses the correct script to load in data files.
    Currently, loads in files from LCPro, ImageJ, SIMA, and Morgan, which are all specalized
    files exported by the CGW or SW labs. 
    Plain calls a simple .txt file and is intended to be more general purpose
    
    Parameters
    ----------
    Data: dictionary
        dictionary containing the pandas dataframes that store the data.
    Settings: dictionary
        dictionary that contains the user's settings.
    
    Returns
    -------
    Data : dictionary
        dictionary containing the pandas dataframes that store the data.
    Settings: dictionary
        dictionary that contains the user's settings.
    
    Notes
    -----
    Add new file loaders at the end of the function by setting up your own gate. 
    """

    #make output folder if it does not exist
    mkdir_p(Settings['Output Folder'])

    try:
        Settings['plots folder'] = Settings['Output Folder'] +"/plots"
        mkdir_p(Settings['plots folder']) #makes a plots folder if does not exist
        print "Made plots folder"
    except:
        try:
            Settings['plots folder'] = Settings['Output Folder'] +"\plots"
            mkdir_p(Settings['plots folder']) #makes a plots folder if does not exist
            print "Made plots folder"
    #LCPro File extraction
    if Settings['File Type'] == 'LCPro':

        Data, Settings, Results = load_RAAIM(Data, Settings, Results)
        events_x, events_y = get_events(Data['original'], Data['ROI parameters'])
        Data['ROI parameters']['Start time (s)'] = Data['ROI parameters']['Time(s)'] - Data['ROI parameters']['attack(s)']
        Data['ROI parameters']['End time (s)'] = Data['ROI parameters']['Time(s)'] + Data['ROI parameters']['decay(s)']
        new_index = [] #create a new, empty list

        for i in np.arange(len(Data['ROI parameters'].index)): #for each index in the original roi_param list, will include duplicates
            new_index.append('Roi'+str(Data['ROI parameters'].index[i])) #reformat name and append it to the empty index list
        Data['ROI parameters'].index = new_index
        Settings['Sample Rate (s/frame)'] = Data['original'].index[1] - Data['original'].index[0]
        print 'Data Loaded'

        

    #ImageJ
    elif Settings['File Type'] == 'ImageJ':
        Data['original'] = pd.read_csv(r'%s/%s_ROI.csv' %(Settings['folder'],Settings['Label']), 
                                       index_col= 'time(s)', sep=',') #load the intensity time series for each roi. should be a text file named exactly 'ROI normalized.txt'
        print "Loaded time series."

        roi_param = pd.read_csv(r'%s/%s_ROI_loc.csv' %(Settings['folder'],Settings['Label']), 
                                index_col=0, sep='\t')#load the parameter list.
        print "Loaded Centroids."

        Data['ROI parameters'] = roi_param
        im = Image.open(r'%s/%s_MaxIntensity.png' %(Settings['folder'], 
                                                    Settings['Label'])) #MUST BE RBG and .png. seriously, I'm not kidding.
        print "Loaded 'rbg.png'"

        new_index = [] #make an empty temp list
        for i in np.arange(len(roi_param.index)): #for each index in roi_loc
            new_index.append('Mean'+str(roi_param.index[i])) #make a string from the index name in the same format as the data
        Data['ROI parameters'].index = new_index

        print 'roi_loc parsed'

        Settings['Sample Rate (s/frame)'] = Data['original'].index[1] - Data['original'].index[0]
        Settings['Graph LCpro events'] = False
        print 'Data Loaded'

    #SIMA
    elif Settings['File Type'] == 'SIMA':
        data = pd.read_csv(r'%s/%s' %(Settings['folder'], Settings['Label']), sep = '\t')
        del data['sequence']
        del data['Unnamed: 0']
        data = data.ix[2:]
        data.index = data['time']
        del data['time']
        Data['original'] = data
        Settings['Graph LCpro events'] = False
        Settings['Sample Rate (s/frame)'] = Data['original'].index[1] - Data['original'].index[0]
        print 'Data Loaded'

    #Plain text, no headers, col[0] is time in seconds
    elif Settings['File Type'] == 'Plain':
        try:
            data = pd.read_csv('%s/%s' %(Settings['folder'], Settings['Label']), sep = '\t', 
                               index_col= 0, header=None)
        except:
            data = pd.read_csv('%s\%s' %(Settings['folder'], Settings['Label']), sep = '\t', 
                               index_col= 0, header=None)
        data.index.name = 'Time(s)'

        new_cols = []

        for i in np.arange(len(data.columns)):
            new_cols.append('Mean'+str(data.columns[i]))
        data.columns = new_cols

        Data['original'] = data
        print 'Data Loaded'
        Settings['Sample Rate (s/frame)'] = Data['original'].index[1] - Data['original'].index[0]
        Settings['Graph LCpro events'] = False


    elif Settings['File Type'] == 'Morgan':
        try:
            data = pd.read_csv('%s/%s' %(Settings['folder'], Settings['Label']), sep = ',', 
                               index_col= 0)
        except:
            data = pd.read_csv('%s\%s' %(Settings['folder'], Settings['Label']), sep = ',', 
                               index_col= 0)
        data.index.name = 'Time(s)'

        #Old morgan data in milliseconds
        new_index = []
        for i in data.index:
            i = round(i, 4)
            if Settings['Milliseconds'] == True:
                i = i/1000
            new_index.append(i)

        data.index = new_index
        new_cols = []

        for i in np.arange(len(data.columns)):
            new_cols.append(str(data.columns[i]))
        data.columns = new_cols

        Data['original'] = data
        print 'Data Loaded'
        Settings['Sample Rate (s/frame)'] = Data['original'].index[1] - Data['original'].index[0]
        Settings['Graph LCpro events'] = False
    else:
        raise ValueError('Not an acceptable file type')
    return Data, Settings

def load_interact():
    '''
    This is the first function in the analysis pipeline. It initalizes the empty dictionaries Data, Settings, and Results.
    It prompts the user to enter the input file path, input file name, and output folder location. 
    Parameters
    ----------
    None
    Returns
    -------
    Data: dictionary
        contains the loaded DataFrame as Data['original'].
    Settings: dictionary
        contains settings used to load file, and sampling rate.
    Notes
    -----
    This function is not used in the Basic version of the notebook, since these things are set in the manual settings block. 
    '''
    
    Data = {}
    Settings = {}
    Results ={}

    Settings['folder']= raw_input('Full File Path to Folder containing file: ')
    Settings['Label'] = raw_input('File Name: ')
    Settings['Output Folder'] = raw_input('Full File Path to Output Folder: ')
    
    #The following settings are temporarily set perminently to this. In the future, file type will be selectable
    Settings['Graph LCpro events'] = False
    Settings['File Type'] = 'Plain' #'LCPro', 'ImageJ', 'SIMA', 'Plain', 'Morgan'
    Settings['Milliseconds'] = False
    
    Data, Settings = load_wrapper(Data, Settings)
    
    return Data, Settings, Results

def mkdir_p(path):
    """"
    This function creates a folder at of the given path, unless the folder already exsists. 
    Parameters
    ----------
    path : string,
        full file path or relative file path.
    Returns
    -------
    None
    Notes
    -----
    This does not check that the file path makes sense or is formatted correctly for OS.
    Examples
    --------
    mkdir_p(User/me/my/path)
    References
    ----------
    .. [1] http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
    """
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise           

def string_eval(string):
    """
    Converts a string to a float, if possible. Otherwise, returns a string.
    Parameters
    ----------
    string : str, 
    Returns
    -------
    string : str
    num : float
    """
    try:
        num = float(string)
        return num
    except ValueError:
        return string

def load_settings(Settings):
    """
    Load a previously saved BASS.py settings file to use for your settings.
    Parameters
    ----------
    Settings: dictionary
        holds settings. this function uses the ['Settings File']. 
    
    Returns
    -------
    Settings: dictionary
        updated Settings dictionary.
    Notes
    -----
    This function calls in a csv and parses the objects inside into the settings dictionary. There are a handful of settings which do not get included because they are unique to each data file. They are in the exclusion_list: 'plots folder', 'folder', 'Sample Rate (s/frame)', 'Output Folder', 'Baseline', 'Baseline-Rolling', 'Settings File', 'Milliseconds', 'Label', 'File Type'. it is simple to add more keys to this list if as changes to the settings file are made.
    Examples
    --------
    Settings['Settings File'] = '~/Users/me/Neuron Modeling/data/IP0_9/Settings.csv'
    Settings = load_settings(Settings)
    """

    settings_temp = pd.read_csv(Settings['Settings File'], index_col=0, header=0, sep=',')
    exclusion_list = ['plots folder', 'folder', 
                      'Sample Rate (s/frame)', 'Output Folder', 
                      'Baseline', 'Baseline-Rolling', 'Settings File', 'Milliseconds',
                      'Label', 'File Type']
    settings_temp = settings_temp.ix[:,0]
    for key, val in settings_temp.iteritems():
        
        if key in exclusion_list:
            continue
        else:
            #print key, val
            Settings[key] = string_eval(val)
            
            if val == 'True':
                Settings[key] = True
            if val == 'False':
                Settings[key] = False
                
    return Settings

def load_settings_interact(Settings):
    """
    Load a previously saved BASS.py settings file to use for your settings.
    Prompts the user for the filepath and file.
    Parameters
    ----------
    Settings: dictionary
        holds settings. 
    
    Returns
    -------
    Settings: dictionary
        updated Settings dictionary. this function adds the ['Settings File']. 
    Notes
    -----
    This function calls in a csv and parses the objects inside into the settings dictionary. 
    There are a handful of settings which do not get included because they are unique to each data file. 
    They are in the exclusion_list: 'plots folder', 'folder', 'Sample Rate (s/frame)', 'Output Folder', 
    'Baseline', 'Baseline-Rolling', 'Settings File', 'Milliseconds', 'Label', 'File Type'. 
    it is simple to add more keys to this list if as changes to the settings file are made.
    Examples
    --------
    Settings = load_settings_interact(Settings)
    """
    
    Settings['Settings File'] = raw_input('Full File path and file for the settings file: ')
    Settings = load_settings(Settings)
    return Settings

def display_settings(Settings):
    '''
    orgqnizes the Settings File into a dataframe, so it can be printed.
    Parameters
    ----------
    location : string
        Full file path and name for the file.
    event_type: string
        Which type of result it is: peaks or bursts.
    Settings: dictionary
        dictionary named Settings.
    Returns
    -------
    Settings_copy: Dataframe
    Notes
    -----
    Compliles the dictionary into a neat dataframe to displays it. If the settings has bot been previously set, nothing will print, since there are no default parameters.
    Examples
    --------
    Settings_display = display_settings(Settings)
    Settings_display
    '''
    Settings_copy = Settings.copy()
    
    if 'Baseline-Rolling' in Settings_copy.keys():
        Settings_copy['Baseline-Rolling'] = True
    Settings_copy = DataFrame.from_dict(Settings_copy, orient='index')
    Settings_copy.columns = ['Value']
    Settings_copy = Settings_copy.sort()
    return Settings_copy

def load_results(location, event_type, Results):
    """
    Loads in a previous master results file.
    Parameters
    ----------
    location : string
        Full file path and name for the file.
    event_type: string
        Which type of result it is: peaks or bursts.
    Results: dictionary
        dictionary named Results, does not need to be empty.
    Returns
    -------
    Results : dictionary
        updated to now contain the master dataframe and the individual DataFrame dictionary.
    Notes
    -----
    Handy way to load in just a previous Results file. 
    Examples
    --------
    Results = {}
    Results = load_results('/my/file/path/Peaks_Results.csv', 'Peaks', Results)
    """
    try:
        temp = pd.read_csv(location, index_col = [0,1])
        
        temp_grouped = temp.groupby(level = 0)
        temp_dict = {}
        for key, df in temp_grouped:
            df.index = df.index.droplevel(0)
            temp_dict[str(key)] = df
        if event_type.lower() == 'bursts':
            Results['Bursts-Master'] = temp
            Results['Bursts'] = temp_dict
        if event_type.lower() == 'peaks':
            Results['Peaks-Master'] = temp
            Results['Peaks'] = temp_dict
        return Results
    except:
        raise OSError('Could not load Results. :(')

###
#
#LCPro Load Block
###
    '''
    This block contains specialty code designed for Sean Wilson. It takes in files that LC_Pro, a module from ImageJ/Fiji that identifies objects and time series events from image stacks.
    This developer does not reccomend this module. It can work for some types of videos, but only if you tweak the parameters.

    Note to future techs in Sean's Lab:
    4/2015
    This developer does not reccomend, under any contitions, LC_pro. It has been proved,repeated and demonstrably, unreliable and it produces deeply skewed results. archived data using it should not be trusted or used.
    if the original videos are available, use another object detection for video software (SIMA is currently what we are testing.)
    '''
def load_RAAIM(Data, Settings, Results):
    '''
    this function takes a path to where all of the LC_pro saved files are. There should be 3 files:
    'ROI normailed.text' - the ROI intensity time series data. Units should be in time (not frame) and relative intensity
    'Parameter List_edit.txt' - this is the events' information file. Duplicate ROIs are expected (since an ROI can have multipule events). The orignal LC_pro file can be loaded, as long as the name is changed to match. 
    'rbg.png' - A still of the video, must be .png. If it is a .tif, it will load, but it will be pseudo colored. it can be just a frame or some averaged measures.
    
    if the files are not named properly or the path is wrong, it will throw a file not found error.
    '''
    data = pd.read_csv(r'%s/ROI normalized.txt' %(Settings['folder']), index_col= 'time(s)', sep='\t') #load the intensity time series for each roi. should be a text file named exactly 'ROI normalized.txt'
    print "Loaded 'ROI normalized.txt'"
    
    roi_param = pd.read_csv(r'%s/Parameter List .txt' %(Settings['folder']), index_col='ROI', sep='\t')#load the parameter list.
    print "Loaded 'Parameter List_edit.txt'"
    
    im = Image.open(r'%s/rgb.png' %(Settings['folder'])) #MUST BE RGB and .png. seriously, I'm not kidding.
    print "Loaded 'rgb.png'"
    
    del data['Unnamed: 0'] #lc_pro outputs a weird blank column named this everytime. I don't know why, but it does. this line deletes it safely.
    
    roi_loc, roi_x, roi_y, data = lcpro_param_parse(roi_param, data , original=True) #use the parameter list to get the x and y location for each ROI
    print "Configured Data"
    
    Settings['plots folder'] = Settings['Output Folder'] +"/plots"
    mkdir_p(Settings['plots folder']) #makes a plots folder inside the path where the data was loaded from
    print "Made plots folder"
    
    Data['original'] = data
    Data['ROI locations'] = roi_loc
    Data['ROI parameters'] = roi_param
    Data['Image'] = im
    
    return Data, Settings, Results
    
def lcpro_param_parse(roi_param, data , original = True):
    '''
    This function takes the Dataframe created by opening the 'Parameter List.txt' from LC_Pro.
    It returns the location data as both a concise list datafram of only locations (roi_loc), an x and y list (roi_x, roi_y). 
    It also changes the names in the roi_loc file to be the same as they are in the data dataframe, which is 
    '''
    roi_loc = roi_param[['X', 'Y']] #make a new dataframe that contains only the x and y coordinates
    roi_loc.drop_duplicates(inplace= True) #roi_param has duplicate keys (rois) because the parameters are based on events, which lc_pro detects. a single roi can have many events. doing it in place like this does cause an error, but don't let it both you none.
    roi_x = roi_loc['X'].tolist() #extract the x column as an array and store it as a value. this is handy for later calculations
    roi_y = roi_loc['Y'].tolist() #extract the y column as an array and store it as a value. this is handy for later calculations
    new_index = [] #make an empty temp list
    for i in np.arange(len(roi_loc.index)): #for each index in roi_loc
        new_index.append('Roi'+str(roi_loc.index[i])) #make a string from the index name in the same format as the data
    roi_loc = DataFrame({'x':roi_x, 'y':roi_y}, index= new_index) #reassign roi_loc to a dataframe with the properly named index. this means that we can use the same roi name to call from either the data or location dataframes
    
    if len(data.columns) != len(new_index) and original == True: #if the number of roi's are the same AND we are using the original file (no roi's have been romved from the edited roi_param)
        sys.exit("The number of ROIs in the data file is not equal to the number of ROIs in the parameter file. That doesn't seem right, so I quit the function for you. Make sure you are loading the correct files, please.")
    
    if original == False: #if it is not the original, then use the roi_loc index to filter only edited roi's.
        data = data[roi_loc.index]
    
    truth = (data.columns == roi_loc.index).tolist() #a list of the bool for if the roi indexes are all the same.
    
    if truth.count(True) != len(data.columns): #all should be true, so check that the number of true are the same.
        sys.exit("The names on data and roi_loc are not identical. This will surely break everything later, so I shut down the program. Try loading these files again.")
    
    return roi_loc, roi_x, roi_y, data

def get_events(data, roi_param):
    '''
    extract the events from the roi_parameter list. It returns them as a pair of dictionaries (x or y data, sored as floats in a list) that use the roi name as the key. 
    duplicate events are ok and expected.
    '''
    
    new_index = [] #create a new, empty list
    
    for i in np.arange(len(roi_param.index)): #for each index in the original roi_param list, will include duplicates
        new_index.append('Roi'+str(roi_param.index[i])) #reformat name and append it to the empty index list
    roi_events = DataFrame(index= new_index) #make an empty data frame using the new_index as the index
    roi_events_time = roi_param['Time(s)'].tolist() #convert time (which is the x val) to a list
    roi_events_amp = roi_param['Amp(F/F0)'].tolist() #conver amplitude (which is the y val) to a list
    roi_events['Time'] = roi_events_time #store it in the events dataframe
    roi_events['Peak Amp'] = roi_events_amp #store is in the events dataframe
    
    events_x = {} #empty dict
    events_y = {} #empty dict
    
    for label in data.columns: #for each roi name in data, initalize the dict by making an empty list for each roi (key) 
        events_x[label] = []
        events_y[label] = []

    for i in np.arange(len(roi_events.index)): #for each event
        key = roi_events.index[i] #get the roi name
        events_x[key].append(roi_events.iloc[i,0]) #use the name to add the event's time data point to the dict
        events_y[key].append(roi_events.iloc[i,1]) #use the name to add the event's amplitude data point to the dict
        
    return events_x, events_y #return the two dictionaries
 
#
#Transform
#wrappers and functions
#
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    Description
    Parameters
    ----------
    param1 : type, shape (N,) optional
        description.
    param2 : type, shape (N,) optional
        description.
    Returns
    -------
    value : type, shape (N) optional
        description.
    Notes
    -----
    more note about usage and philosophy. 
    Examples
    --------
    ?
    References
    ----------
    .. [1] http://wiki.scipy.org/Cookbook/ButterworthBandpass
    .. [2] http://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
    """

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Description
    Parameters
    ----------
    param1 : type, shape (N,) optional
        description.
    param2 : type, shape (N,) optional
        description.
    Returns
    -------
    value : type, shape (N) optional
        description.
    Notes
    -----
    more note about usage and philosophy. 
    Examples
    --------
    ?
    References
    ----------
    .. [1] http://wiki.scipy.org/Cookbook/ButterworthBandpass
    .. [2] http://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def user_input_trans(Settings):
    '''
    this function allows the user to specify and save their transformation settings for later analysis in an interactive way using raw_input().
    The only argument passed in and out is the Settings Dictionary.
    Parameters
    ----------
    Settings: dictionary
        dictionary that contains the user's settings.    
    Returns
    -------
    Settings: dictionary
        dictionary that contains the user's settings.
    Notes
    -----
    Expand on transformation settings choices.
    Examples
    --------
    Settings = user_input_trans(Settings)
    '''
    
    #print "Enter the parameters of the functions you would like to use to transform your data. "
    #print "If you do not want to use a function, enter 'none'"
    
    if 'Linear Fit' in Settings.keys():
        print "Previous Linear Fit R setting: %s" %Settings['Linear Fit']
        if type(Settings['Linear Fit']) == float:
            print "Previous Linear Fit setting: %s" %Settings['Linear Fit']
    Settings = linear_settings(Settings)
    
    
    if 'Bandpass Lowcut' in Settings.keys():
        print "Previous Bandpass setting: %s, %s, %s" %(Settings['Bandpass Lowcut'], 
                                                        Settings['Bandpass Highcut'], 
                                                        Settings['Bandpass Polynomial'])
    lowcut, highcut, poly_band = bandpass_settings()
    Settings['Bandpass Lowcut'] = lowcut
    Settings['Bandpass Highcut'] = highcut
    Settings['Bandpass Polynomial'] = poly_band
    
    if 'Savitzky-Golay Window Size' in Settings.keys():
        print "Previous Savitzky-Golay setting: %s, %s" %(Settings['Savitzky-Golay Window Size'], 
                                                        Settings['Savitzky-Golay Polynomial'])
    window_sav, poly_sav = savgolay_settings()
    Settings['Savitzky-Golay Window Size'] = window_sav
    Settings['Savitzky-Golay Polynomial'] = poly_sav
    
    if window_sav != 'none':
        Settings['Absolute Value'] = True
    
    else:
        if 'Absolute Value' in Settings.keys():
            print "Previous Absolute Value: %s" %Settings['Absolute Value']
        abs_set = abs_settings()
        Settings['Absolute Value'] = abs_set
    
    print "Settings Saved"
    return Settings

def savgolay_settings():
    print "Enter the Savitzky Golay filter settings seperated by a comma. Window size must be odd."
    
    temp_list = raw_input("Savitzky Golay Settings (window, polynomial): ").lower()
    
    temp_list = temp_list.split(',')
    
    if temp_list[0] == "none" or temp_list[0] == "false":
        window = "none"
        poly = "none"
    else:
        window = int(temp_list[0])
        poly = int(temp_list[1])
    return window, poly
    '''
    if len(temp_list) != 2:
        print "ERROR: There should be two parameters separated by a comma."
        print " "
        savgolay_settings()
    if window % 2 == 0:
        print "ERROR: window size must be odd."
        print " "
        savgolay_settings()
    '''
    
def bandpass_settings():
    print "Enter the butterworth bandpass settings seperated by a comma. cuts are in hertz and poly should be an interger."
    
    temp_list = raw_input("Bandpass Settings (lowcut, highcut,polynomial): ").lower()
    
    temp_list = temp_list.split(',')
    
    if temp_list[0] == "none" or temp_list[0] == "false":
        lowcut = "none"
        highcut = "none"
        poly = "none"
    else:
        lowcut = float(temp_list[0])
        highcut = float(temp_list[1])
        poly = int(temp_list[2])
    return lowcut, highcut, poly

def abs_settings():
    print "Enter True or False to turn on or off the absolute value."
    abs_set = raw_input("Absolute Value (True/False): ").lower()
    
    if abs_set == "true":
        abs_set = True
    else:
        abs_set = False
    return abs_set

def linear_settings(Settings):
    print "Enter True or False to turn on or off the linear fit."
    lin_param = raw_input("Linear Fit (True/False): ").lower()
    
    if lin_param == "false" or lin_param == 'none' or lin_param == '':
        Settings['Linear Fit'] = False
        return Settings
    else:
        Settings['Linear Fit'] = float(raw_input('Linear fit R value (0-1): '))
        Settings['Linear Fit-Rolling R'] = float(raw_input('Rolling Window Linear fit R value (0-1): '))
        Settings['Linear Fit-Rolling Window'] = int(raw_input('Rolling Window size (even integer): '))
        Settings['Relative Baseline'] = float(raw_input('Relative Baseline (float): '))
        return Settings
    
def linear_subtraction(data, time_array, R_raw, R_roll, window, b=0):
    '''
    perfrom linear fit and then subtract that line from the input data. This will give the data a slope of 0.
    will check to see which fit (raw or rolling) gives a better fit.
    '''
    #raw fit
    time = arange(len(data)) #arb time array, each val an int

    A = np.vstack([time, np.ones(len(time))]).T #matrix required for solving the lstsq, which we want for the fit
    m, c = np.linalg.lstsq(A, data)[0] # find coef. for the y = m*x + c equation

    count = 0
    ls_tf = []

    r, p = pearsonr(data, ((m*time) +c)) #Raw r val
    
    #rolling fit
    rolling_mean, data_roll, time_roll = baseline_rolling(time_array, 
                                                              np.array(data), 
                                                              window)
    time_r = arange(len(time_roll)) #arb time array, each val an int
    
    A = np.vstack([time_r, np.ones(len(time_roll))]).T #matrix required for solving the lstsq, which we want for the fit
    m_r, c_r = np.linalg.lstsq(A, rolling_mean)[0] # find coef. for the y = m*x + c equation
    
    r_r, p_r = pearsonr(data_roll, ((m_r*time_roll) +c_r+1))
    
    if r>=r_r and r >=R_raw: #raw fit is better than rolling and above thresh
        #print '%s: raw_lin_sub' %label

        for point in data: #this is the line subtraction
            x = (m * count) + c 
            y = (point - x) +b #the plus one is for normalized data, where 1.0 and not 0 is the 'baseline'
            ls_tf.append(y)
            count = count +1
            

        return np.array(ls_tf)
    
    elif r_r >= R_roll: #if rolling is better and above thresh
        #print '%s: rolling_lin_sub' %label
        for point in data: #this is the line subtraction
            x = (m_r * count) + c_r 
            y = (point - x) +b #the plus one is for normalized data, where 1.0 and not 0 is the 'baseline'
            ls_tf.append(y)
            count = count +1
            

        return np.array(ls_tf)
    
    else:
        return data
    
def transform_wrapper(Data, Settings):
    '''
    Iterate over all columns of the original dataframe to transform it.
    Creates a new dataframe called Data['trans']
    '''
    Data['trans'] = DataFrame(index = Data['original'].index)
    for label, column in Data['original'].iteritems():
        data_trans = transformation(column, Settings)
        Data['trans'][label] = data_trans
    return Data, Settings

def transformation(Data, Settings):
    
    data_trans = np.array(Data)
    time_array = Data.index
    if Settings['Linear Fit'] != False:
        data_trans = linear_subtraction(data_trans, time_array,
                                        Settings['Linear Fit'], 
                                        Settings['Linear Fit-Rolling R'],
                                        Settings['Linear Fit-Rolling Window'],
                                        Settings['Relative Baseline'])
        
    if Settings['Bandpass Lowcut'] != 'none':
        data_trans = butter_bandpass_filter(data_trans, Settings['Bandpass Lowcut'], 
                                            Settings['Bandpass Highcut'], 
                                            1/Settings['Sample Rate (s/frame)'], 
                                            order= Settings['Bandpass Polynomial'])
    
    if Settings['Absolute Value'] == True:
        data_trans = abs(data_trans)
    
    if Settings['Savitzky-Golay Window Size'] != 'none':
        data_trans =savitzky_golay(data_trans, Settings['Savitzky-Golay Window Size'], 
                                   Settings['Savitzky-Golay Polynomial'])
    
    return data_trans

def graph_trans(Data):
    '''
    Plots the first column of data from the transformed dataframe
    Parameters
    ----------
    Data: dictionary
        should contain Data['trans']
    Returns
    -------
    None

    Notes
    -----
    Cannot currently call just the transformed graph by Key. Only first column of data is displayed.
    '''
    try:
        
        plt.plot(Data['trans'].index, Data['trans'].ix[:,0], 'k')
        plt.title(r'Transformed Data: %s' %(Data['trans'].ix[:,0].name))
        plt.xlabel('Time (s)')
        plt.ylabel('Relative Amplitude')
        
        plt.show()
    except:
        print "An Error occured: Could not display graph."

#
#Baseline
#
#
def user_input_base(Settings):
    '''
    this function allows the user to specify and save their baseline settings for later analysis in an interactive way using raw_input().
    The only argument passed in and out is the Settings Dictionary.
    Parameters
    ----------
    Settings: dictionary
        dictionary that contains the user's settings.
    
    Returns
    -------
    Settings: dictionary
        dictionary that contains the user's settings.
    Notes
    -----
    What are good values for baseline Settings? There is no one answer. Each user will need to tinker with settings. However, I can offer some discussion of each method and its parameters.

    Static: Under this, no baseline correction is made no baseline generated. Later, the user will select a threshold value (an arbitrary y value) to detect burst boundaries.

    Linear: Linear can be thoughts of as a two part baseline correction. First, the user selects the time segment from which the baseline is calculated. Second, the whole time series is shifted such that the baseline value is now 0. This shift is reflected in all y values down stream (such as peak amplitudes).

    Rolling: 
    Examples
    --------
    Settings = user_input_base(Settings)
    '''
    
    if 'Baseline Type' in Settings.keys():
        print "Previous Baseline Type: %s" %Settings['Baseline Type']
        
    baseline_type = raw_input('Enter Linear, Rolling, or Static: ').lower()
    
    if baseline_type == 'static':
        Settings['Baseline Type'] = baseline_type
        return Settings
    
    elif baseline_type == 'linear':
        print "Enter the start and stop time in seconds of your representitive baseline. Defaults are (0, 1)"
        print "WARNING: Use on aligned time series is not currently supported."
        Settings['Baseline Start'] = float(raw_input('Start (seconds)'))
        if Settings['Baseline Start'] == '':
            Settings['Baseline Start'] = 0
        Settings['Baseline Stop'] = float(raw_input('End (seconds)'))
        if Settings['Baseline Stop'] == '':
            Settings['Baseline Stop'] = 1
        Settings['Baseline Type'] = baseline_type
    
    elif baseline_type == 'rolling':
        
        Settings['Baseline Type'] = baseline_type
        if 'Rolling Baseline Window' in Settings.keys():
            print "Previous Rolling Baseline Window: %s" %Settings['Rolling Baseline Window']
        print "Enter the window size of the rolling baseline in seconds."
        Settings['Rolling Baseline Window'] = float(raw_input('Window size in seconds: '))
        window = float(Settings['Rolling Baseline Window'])
        
        #convert window (s) to index
        window = int((window)/Settings['Sample Rate (s/frame)'])
        
        if window < 2:
            print ("You entered a window size so small it won't run. Give me a larger window size.")
            user_input_base(Settings)
    
    else:
        print 'That was not an acceptable baseline type. Try again.'
        user_input_base(Settings)
    return Settings

def baseline_wrapper(Data, Settings, Results):
    '''
    Wrapper that directs the Data through the baseline functions. There are currently three: Static, Linear, and Rolling.

    Static: this allows the user to define an arbitrary threshold line during burst detection. Therefore, no change is made to the transformed data in this function.
    Linear: This uses a user defined segment of data from the transformed data as the baseline by averaging the points together. Then, the data is shifted by the baseline ammount, such that the baseline is now set at 0.
    Rolling: this generates a rolling baseline from the transformed data. the size of the window is specified by the user. A small window will generate a very course baseline. The size of the window should be larger than the max duration of any event.
    
    Parameters
    ----------
    Data: dictionary
        should contain Data['original'] and Data['trans'].
    Settings: dictionary
        dictionary that contains the user's settings. requires the baseline settings be pre-specified.
    Results: dictionary
        an dictionary named Results.
    
    Returns
    -------
    Data: dictionary
        If linear is used, then Data['shift'] will be added.
    Settings: dictionary
        dictionary that contains the user's settings.
    Results: dictionary
        Updated to contains the following objects:
        Baseline: Single value of the baseline. Only supporeted for single time series. 
        Baseline-Rolling:  DataFrame that contains the rolling baseline for each time series (supported for single and aligned.)
    Notes
    -----
    Chosing the appropreate baseline is important for detecting event boundaries. For the first pass, chosing static may be the most useful to give you an idea of what will work best.
    If you can make your data fundamentally linear (using linear fit/subtraction or another method) then either static or linear will work well for you.
    If your data has a non-linear super structure, then rolling is what you want. Make sure you select a window big enough to not be significantly shifted up by events. 

    Examples
    --------
    ?
    '''
    #check that the baseline settings are set
    if 'Baseline Type' not in Settings.keys():
        Settings = user_input_base(Settings) #if they are not, prompt to select them.


    if Settings['Baseline Type'] == 'static':
        pass #no change in Data is required. gtfo
        return Data, Settings, Results
    
    elif Settings['Baseline Type'] == 'linear':

        Data['shift'] = DataFrame(index = Data['trans'].index)
        baseline = {}
        
        for label, column in Data['trans'].iteritems(): #the problem with aligned time series is that the baserate calculation will be performed for each one, not garunteeing a good shift.
            
            baserate = np.mean(column[Settings['Baseline Start']:Settings['Baseline Stop']]) #average all points to obtain baserate
            baseline[label] = abs(baserate)
            datashift = []

            #may be able to use np.subtract(data,base) instead, but this seems to work correctly.
            for x in column:
                foo = (x-baserate)
                datashift.append(foo)
            Data['shift'][label] = datashift
        Results['Baseline'] = baseline
        return Data, Settings, Results
    
    elif Settings['Baseline Type'] == 'rolling':
        Data['rolling'] = DataFrame()
        Results['Baseline-Rolling'] = DataFrame()
        window = int((Settings['Rolling Baseline Window'])/Settings['Sample Rate (s/frame)'])
        
        if window %2 !=0:
            window = window+1 #if window is odd, there's a problem with baseline_rolling.
        for label, column in Data['trans'].iteritems():
            rolling_mean, data_roll, time_roll = baseline_rolling(Data['trans'].index, 
                                                              np.array(column), 
                                                              window)
            Data['rolling'][label] = data_roll
            Results['Baseline-Rolling'][label] = rolling_mean
        
        Data['rolling'].index = time_roll
        Results['Baseline-Rolling'].index = time_roll
        return Data, Settings, Results    
        
def baseline_rolling(time, data_trans, window):
    '''
    calculates the rolling baseline of a 1d numpy array based on a given window size. 
    To align the rolling baseline along the center, the original data is trimmed, 1/2 baseline from each end.
    '''
    
    rolling_mean = pd.rolling_mean(data_trans, window)
    
    time = time[(window/2):-(window/2)]
    data_trans = data_trans[(window/2):-(window/2)]
    rolling_mean = rolling_mean[window:]
    
    #sanity check
    if len(data_trans) != len(rolling_mean):
        raise Exception("Something has gone horribly wrong. The data arrays are no longer the same size nor aligning. Restart the notebook and try again.")
    return rolling_mean, data_trans, time

def graph_baseline(Data, Settings, Results):
    '''
    Generates a plot displaying the transformed data with the baseline correction applied. 
    Parameters
    ----------
    Data: dictionary
        should contain Data['trans'] (and Data['shift'], if the linear method is selected).
    Settings: dictionary
        dictionary that contains the user's settings.
    Results: dictionary
        an dictionary named Results. If rolling method is being used, the 'Baseline-Rolling' is stored here.
    Returns
    -------
    None

    Notes
    -----
    If Static is chosen, then the trans graph will display again, since no change was made to the data for correct for baseline.
    '''
    
    if Settings['Baseline Type'] == 'linear':
        try:
            plt.plot(Data['shift'].index, Data['shift'].ix[:,0], 'k') #in this instance, baseline = shift
            plt.xlabel('Time (s)')
            plt.ylabel('Amp')
            plt.hlines(0,Data['shift'].index[0],Data['shift'].index[-1],colors='b')
            #plt.xlim(xmin = min(time), xmax = (min(time)+10))
            #plt.ylim(ymin = min(data_baseline), ymax = max(data_baseline))
            plt.title(r'Linear Baseline: %s' %(Data['trans'].ix[:,0].name))

            plt.show()
        except:
            print "An Error Occured and I can't make your graph. Sorry. :("
    
    elif Settings['Baseline Type'] == 'rolling':
        try:
            plt.plot(Data['rolling'].index, Data['rolling'].ix[:,0], 'k')
            plt.xlabel('Time (s)')
            plt.ylabel('Amp')
            plt.plot(Results['Baseline-Rolling'].index, Results['Baseline-Rolling'].ix[:,0], 'b') #in this instance, baseline = rolling average
            plt.title(r'Rolling Baseline: ' %(Data['trans'].ix[:,0].name))
            #plt.xlim(xmin = min(time), xmax = (min(time)+10))
           # plt.ylim(ymin = min(data_baseline), ymax = max(data_baseline))
            
            plt.show()
        except:
            print "An Error Occured and I can't make your graph. Sorry. :("
    
    elif Settings['Baseline Type'] == 'static':
        graph_trans(Data)

#
#Event-Peak Detection
#
#

def peakdet(v, delta, x = None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html
    
    Returns two arrays
    
    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %      
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.
    
    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.
    
    """
    maxtab = []
    mintab = []
       
    if x is None:
        x = arange(len(v))
    
    v = asarray(v)
    
    if len(v) != len(x):
        raise Exception('Input vectors v and x must have same length')
    
    if not isscalar(delta):
        raise Exception('Input argument delta must be a scalar')
    
    if delta <= 0:
        raise Exception('Input argument delta must be positive')
    
    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN
    
    lookformax = True #why set to True in original text?
    
    for i in arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        
        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True
 
    return array(maxtab), array(mintab)

def rrinterval(maxptime): 
    """
    find time from r peak to r peak, called the R-R interval. Input array must be a list of numbers (float).
    """
    if type(maxptime) != list:
        maxptime = maxptime.tolist()
    
    rrint = [] #empty array for ttot to go into
    
    for time in maxptime[1:]: #for each r peak, starting with the second one
        s2time = maxptime.index(time) 
        s2 = maxptime[s2time-1]
        meas = time - s2 #measure the interval by subtracting
        rrint.append(meas) #append the measurement to the ttotal array
    return rrint #return array

def event_peakdet_settings(Data, Settings):
    '''
    this function allows the user to specify and save their peak detection settings for later analysis in an interactive way using raw_input().
    Parameters
    ----------
    Data: dictionary
        should contrain ['trans']
    Settings: dictionary
        dictionary that contains the user's settings.
    Returns
    -------
    Settings: dictionary
        dictionary that contains the user's settings.
    Notes
    -----
    What are good values for peak detection Settings? There is no one answer. Each user will need to tinker with settings. However, I can offer some discussion of each method and its parameters.
    Delta:  
    Examples
    --------
    Settings = user_input_base(Settings)
    '''
    if 'Delta' in Settings.keys():
        print "Previous delta value: %s" %Settings['Delta']
    delta = raw_input("Enter delta value between 0 and %s: " %round(max(Data['trans'].max())-min(Data['trans'].min()),4))
    delta = float(delta)
    Settings['Delta'] = delta
    
    if 'Peak Minimum' in Settings.keys():
        print "Previous Peak minimum value: %s" %Settings['Peak Minimum']
    min_data = min(Data['trans'].min())
    max_data = max(Data['trans'].max())
    peak_min = raw_input("Enter Peak Minimum value between %s and %s: " %(round(min_data,4) ,
                                                                          round(max_data,4)))
    peak_min = float(peak_min)
    Settings['Peak Minimum'] = peak_min
    
    if 'Peak Maximum' in Settings.keys():
        print "Previous Peak Maximum value: %s" %Settings['Peak Maximum']
    peak_max = raw_input("Enter Peak Maximum value between %s and %s: " %(round(peak_min,4) ,
                                                                          round(max_data,4)))
    peak_max = float(peak_max)
    Settings['Peak Maximum'] = peak_max

    return Settings

def event_peakdet(Data, Settings, Results, roi):
    '''
    Wrapper that directs one columns of data, defined by roi name into the correct post processing from peak and valley detection.
    if no peaks or valleys are found, a failur flag is set to True and an empty dataframe is returned.
    peakdet() returns two numpy arrays, mintab and maxtab, which contains the time and amplitude data for peaks and valleys. time is in index, so this function converts it to seconds using the sample rate (in settings). 
    if the baseline settings is linear, then the amplidude is shifted using the baseline value from Results.
    Parameters
    ----------
    Data: series
        pandas series object that contains floats
    Settings: dictionary
        dictionary that contains the user's settings. requires the baseline settings be pre-specified.
    Results: dictionary
        an dictionary named Results, should contain the Baseline or Baseline-Rolling.
    
    Returns
    -------
    results_peaks: dataframe
        Contains the information about each Peak (local maxima). index is peak time (seconds). columns are 'Peaks Amplitude' and 'Intervals'
    results_valleys: dictionary
        dataframe
        Contains the information about each valley (local minima). index is peak time (seconds). columns are 'Valey Amplitude' and 'Intervals'
    failure: bool
        False if all went well, otherwise set to True to alert wrapper above that something didn't happen right.
    Notes
    -----
    While the varible is called Data, it isn't the Data dictionary. this is intended to be called from inside event_peakdet_wrapper 

    Examples
    --------
    roi = Mean1

    results_peaks, results_valleys, failure = event_peakdet(Data[roi], Settings, Results, roi)

    '''
    
    failure = False
    if Settings['Baseline Type'] == 'linear':
        maxtab, mintab = peakdet(np.array(Data), Settings['Delta'], None)
        
        if maxtab.size == 0:
            maxptime = [NaN]
            maxpeaks = [NaN]
            maxptime = [NaN]
            
            results_peaks = DataFrame({'Peaks Amplitude':maxpeaks, 'Intervals':maxpeaks})
            #peak_sum = results_peaks['Peaks Amplitude'].describe()
            print roi, 'has no peaks.'
        else:
            maxtab = np.array(maxtab)
            maxptime = maxtab[:,0] #all of the rows and only the first column are time
            maxptime_true = (np.multiply(maxptime, Settings['Sample Rate (s/frame)'])
                             + Data.index[0]) #get the real time for each peak
            maxpeaks = maxtab[:,1]
            maxpeaks = (np.subtract(maxpeaks,Results['Baseline'][roi]))
            
            results_peaks = DataFrame({'Peaks Amplitude':maxpeaks}, index=maxptime_true)
            
            #filter results
            results_peaks = results_peaks[results_peaks['Peaks Amplitude']>Settings['Peak Minimum']]
            results_peaks = results_peaks[results_peaks['Peaks Amplitude']<=Settings['Peak Maximum']]
            if results_peaks.empty == True:
                failure = True
                print roi, 'has no peaks.'
            else:
                RR = rrinterval(results_peaks.index)
                RR.append(NaN)
                results_peaks['Intervals'] = RR
                results_peaks.index.name = 'Time'
            #Results['Peaks'] = results_peaks
            

            #peak_sum = results_peaks['Peaks Amplitude'].describe()
            
        if mintab.size == 0:
            results_valleys = DataFrame({'Valley Amplitude':[NaN]})
        else:
            mintab = np.array(mintab)
            valleytime = mintab[:,0]
            valleytime = (np.multiply(valleytime, Settings['Sample Rate (s/frame)'])
                             + Data.index[0]) #get the real time for each peak
            valleys = mintab[:,1]
            valleys = (np.subtract(valleys,Results['Baseline'][roi]))
            results_valleys = DataFrame({'Valley Amplitude':valleys}, index = valleytime)

    elif Settings['Baseline Type'] == 'rolling' or Settings['Baseline Type'] == 'static':
        
        maxtab, mintab = peakdet(np.array(Data), Settings['Delta'], None)
        if maxtab.size == 0:
            maxptime = [NaN]
            maxpeaks = [NaN]
            maxptime = [NaN]
            
            results_peaks = DataFrame({'Peaks Amplitude':maxpeaks, 'Intervals':maxpeaks})
            #peak_sum = results_peaks['Peaks Amplitude'].describe()
            print roi, 'has no peaks.'
        else:
            maxtab = np.array(maxtab)
            maxptime = maxtab[:,0] #all of the rows and only the first column are time
            maxptime_true = (np.multiply(maxptime, Settings['Sample Rate (s/frame)'])
                             + Data.index[0]) #get the real time for each peak
            maxpeaks = maxtab[:,1]
            
            results_peaks = DataFrame({'Peaks Amplitude':maxpeaks}, index=maxptime_true)
            
            #filter results
            results_peaks = results_peaks[results_peaks['Peaks Amplitude']>Settings['Peak Minimum']]
            
            if results_peaks.empty == True:
                failure = True
                print roi, 'has no peaks.'
            else:
                RR = rrinterval(results_peaks.index)
                RR.append(NaN)
                results_peaks['Intervals'] = RR
                results_peaks.index.name = 'Time'
            

            #peak_sum = results_peaks['Peaks Amplitude'].describe()
        if mintab.size == 0:
            results_valleys = DataFrame({'Valley Amplitude':[NaN]})
        else:

            mintab = np.array(mintab)
            valleytime = mintab[:,0]
            valleytime = (np.multiply(valleytime, Settings['Sample Rate (s/frame)'])
                             + Data.index[0]) #get the real time for each peak
            valleys = mintab[:,1]
            results_valleys = DataFrame({'Valley Amplitude':valleys}, index = valleytime)

    
    #peak_plot(Data, Settings, Results)
    
    
    return results_peaks, results_valleys, failure

def event_peakdet_wrapper(Data, Settings, Results):
    '''
    The wrapper than handles applying peak detection on an entire dataframe
    '''
    
    peaks_dict = {}
    #peaks_sum = DataFrame()
    valleys_dict = {}
    
    if Settings['Baseline Type'] == 'linear' or Settings['Baseline Type'] == 'static':
        data_temp = Data['trans']
    elif Settings['Baseline Type'] == 'rolling':
        data_temp = Data['rolling']
    
    for label, col in data_temp.iteritems():
        results_peaks, results_valleys, failure = event_peakdet(col, Settings, Results, label)
        
        if failure == False:
            peaks_dict[label] = results_peaks

        elif failure == True:
            pass
            #print label+' had no peaks'
        
        valleys_dict[label] = results_valleys
        
        #peaks_sum[label] = peak_sum
    Results['Peaks'] = peaks_dict
    Results['Valleys'] = valleys_dict
    #Results['Peaks Summary'] = peaks_sum
    try:
        master_peaks = pd.concat(Results['Peaks'])
        Results['Peaks-Master'] = master_peaks
    except:
        Results['Peaks-Master'] = DataFrame()
        print "No Peaks Found in any time series."

    try:
        master_valleys = pd.concat(Results['Valleys'])
        Results['Valleys-Master'] = master_peaks
    except:
        Results['Valleys-Master'] = DataFrame()
        print "No Valleys Found in any time series."

    return Results

#
#Event-Burst Detection
#
#
def event_burstdet_settings(Data, Settings):
    '''
    WRITE DOCSTRING
    '''

    #Threshold
    if 'Threshold' in Settings.keys():
        print "Previous threshold value: %s" %Settings['Threshold']
    
    if Settings['Baseline Type'] == 'static':
        max_data = max(Data['trans'].max())
        min_data = min(Data['trans'].min())
        threshperc = raw_input("Enter a threshold value between %s and %s: " %(round(min_data,4), round(max_data, 4)))
        threshperc = float(threshperc)
        
        Settings['Threshold'] = threshperc
    if Settings['Baseline Type'] == 'linear':
        threshperc = raw_input("Enter a threshold proportion value between 0 and %s: " %round(max(Data['shift'].max()),4))
        threshperc = float(threshperc)
        Settings['Threshold'] = threshperc
        
    if Settings['Baseline Type'] == 'rolling':
        max_data_rolling = max(Data['rolling'].max())
        max_baseline_rolling = max(Results['Baseline-Rolling'].max())
        threshperc = raw_input("Enter a threshold value between 0 and %s: " %round(max_data_rolling-max_baseline_rolling,4))
        threshperc = float(threshperc)
        
        Settings['Threshold'] = threshperc
    
    #inter-event min
    if 'Inter-event interval minimum (seconds)' in Settings.keys():
        print "Previous interval value: %s" %Settings['Inter-event interval minimum (seconds)']
        
    cluster_time = raw_input("Enter the minimum inter-event interval in seconds:")
    cluster_time = float(cluster_time)
    Settings['Inter-event interval minimum (seconds)'] = cluster_time
    
    #burst duration min
    if 'Minimum Burst Duration (s)' in Settings.keys():
        print "Previous Minimum Burst Duration value: %s" %Settings['Minimum Burst Duration (s)']
    minimum_duration = raw_input("Enter the minimum burst duration in seconds:")
    minimum_duration = float(minimum_duration)
    Settings['Minimum Burst Duration (s)'] = minimum_duration
    
    #burst duration max
    if 'Maximum Burst Duration (s)' in Settings.keys():
        print "Previous Maximum Burst Duration value: %s" %Settings['Maximum Burst Duration (s)']
    max_duration = raw_input("Enter the maximum burst duration in seconds:")
    max_duration = float(max_duration)
    Settings['Maximum Burst Duration (s)'] = max_duration

    #Minimum Peak Num
    if 'Minimum Peak Number' in Settings.keys():
        print "Previous Minimum Peaks per Burst number: %s" %Settings['Minimum Peak Number']
    min_peak = raw_input("Enter the minimum number of peaks per burst:")
    min_peak = int(min_peak)
    Settings['Minimum Peak Number'] = min_peak

    #Burst Area
    if 'Burst Area' in Settings.keys():
        print "Previous Burst Area setting: %s" %Settings['Burst Area']
    b_area = raw_input("Do you want to calculate Burst Area? (True/False):")
    b_area = bool(b_area)
    Settings['Burst Area'] = b_area

    #Exclude edges
    if 'Exclude Edges' in Settings.keys():
        print "Previous Exclude Edges setting: %s" %Settings['Exclude Edges']
    ee = raw_input("Do you want to Exclude Edges? (True/False):")
    ee = bool(ee)
    Settings['Exclude Edges'] = ee


    return Settings

def event_burstdet_wrapper(Data, Settings, Results):
    '''
    the wrapper that handles burst detection for a whole data frame. Bursts on the edges are not counted
    '''
    burst_dict = {}
    burst_sum = DataFrame()
    
    if Settings['Baseline Type'] == 'static':
        data_temp = Data['trans']
        time_temp = Data['trans'].index
    elif Settings['Baseline Type'] == 'linear': 
        data_temp = Data['shift']
        time_temp = Data['shift'].index
    elif Settings['Baseline Type'] == 'rolling':
        data_temp = Data['rolling']
        time_temp = Data['rolling'].index
    for roi, col in data_temp.iteritems():
        results_bursts, failure = event_burstdet(col, time_temp, Settings, Results, roi)
        
        if failure == False:
            #burst_sum[roi] = results_bursts.describe()
            burst_dict[roi] = results_bursts
        else:
            pass
            #print roi +'had no bursts'
            
    Results['Bursts'] = burst_dict
    try:
        master_bursts = pd.concat(burst_dict)
        Results['Bursts-Master'] = master_bursts
    except:
        Results['Bursts-Master'] = DataFrame()
        print "No Bursts Found in any time series."
    return Results
    
def event_burstdet(Data, Time, Settings, Results, roi):

    if Settings['Baseline Type'] == 'static': #data[shift]
        bstart, bend, bdur = burstduration_lin(Time, np.array(Data), 
                                               Settings['Threshold'], 
                                               1, 
                                               Settings['Inter-event interval minimum (seconds)'])
    
    elif Settings['Baseline Type'] == 'linear': #data[shift]
        bstart, bend, bdur = burstduration_lin(Time, np.array(Data), 
                                               Results['Baseline'][roi], 
                                               Settings['Threshold'], 
                                               Settings['Inter-event interval minimum (seconds)'])
        
        
    elif Settings['Baseline Type'] == 'rolling': #data[rolling]
        bstart, bend, bdur, b_start_amp, b_end_amp = burstduration_rolling(Time, 
                                                                           np.array(Data), 
                                                                           Results['Baseline-Rolling'][roi], 
                                                                           Settings['Threshold'], 
                                                                           Settings['Inter-event interval minimum (seconds)'])
    failure = False
    if len(bstart) ==0:
        print roi+'has no bursts.'
        bstart = [NaN]
        bend = [NaN]
        bdur = [NaN]
        if Settings['Baseline Type'] == 'rolling':
            b_start_amp=[NaN]
            b_end_amp=[NaN]
        failure = True
        results_bursts = DataFrame()
        return results_bursts, failure
    
    
    results_bursts = DataFrame({'Burst Start': bstart})
    results_bursts.index = bstart
    results_bursts['Burst End'] = bend
    results_bursts['Burst Duration'] = bdur
    
    if Settings['Baseline Type'] == 'rolling':
        results_bursts['Burst Start Amplitude'] = b_start_amp
        results_bursts['Burst End Amplitude'] = b_end_amp
    
    #Filter results
    results_bursts = results_bursts[results_bursts['Burst Duration']>Settings['Minimum Burst Duration (s)']]
    results_bursts = results_bursts[results_bursts['Burst Duration']<=Settings['Maximum Burst Duration (s)']]

    #If there are no bursts, return the empty df and failure flag
    if results_bursts.empty == True:
        print roi+'has no bursts.'
        failure = True
        return results_bursts, failure
    
    #Check for edge events
    try:
        edge = []
        for i in np.arange(len(results_bursts['Burst Start'])):
            if results_bursts['Burst Start'].iloc[i] == Time[0]:#does it start in a burst?
                edge.append(True)
            elif Data[results_bursts['Burst End'].iloc[i]] > Settings['Threshold'] and results_bursts['Burst End'].iloc[i] == Time[-1]:
                edge.append(True)
            else:
                edge.append(False)
        results_bursts['Edge Event'] = edge
    except ValueError:
        #failure = True
        results_bursts['Edge Event'] = [NaN]
    
    #Exclude edge events if setting is true
    if Settings['Exclude Edges'] == True:
        results_bursts = results_bursts[results_bursts['Edge Event'] == False]
    
    #If there are no bursts, return the empty df and failure flag
    if results_bursts.empty == True:
        print roi+'has no bursts.'
        failure = True
        return results_bursts, failure
    
    #interburst interval
    try:
        interburst = interburstinterval(results_bursts['Burst Start'].tolist(), results_bursts['Burst End'].tolist())
        interburst.append(NaN)
        results_bursts['Interburst Interval'] = interburst
    except:
        #failure = True
        results_bursts['Interburst Interval'] = [NaN]
    #Total cycle time from burst start to burst start
    try:
        tot = ttotal(results_bursts['Burst Start'].tolist())
        tot.append(NaN)
        results_bursts['Total Cycle Time'] = tot
    except ValueError:
        #failure = True
        results_bursts['Total Cycle Time'] = [NaN]

    #Peaks per burst
    ppb = []

    for i in np.arange(len(results_bursts['Burst Start'])):
        try:

            peak_df = Results['Peaks'][roi][results_bursts['Burst Start'].iloc[i]:
                                            results_bursts['Burst End'].iloc[i]]
            if peak_df.empty == True:
                ppb.append(0)
            else:
                count = peak_df['Peaks Amplitude'].count()
                ppb.append(count)
        
        except:
            ppb.append(NaN)
    results_bursts['Peaks per Burst'] = ppb
    
    results_bursts = results_bursts[results_bursts['Peaks per Burst']>=Settings['Minimum Peak Number']]

    #If there are no bursts, return the empty df and failure flag
    if results_bursts.empty == True:
        print roi+'has no bursts.'
        failure = True
        return results_bursts, failure
    #Max peak
    #Retrieves the list of peaks in a burst and selects the one with the greatest amplitdue to be the
    #'maximum' peak. used in the attack and decay calculations
    burst_peak_max = []
    burst_peak_id = []
    for i in np.arange(len(results_bursts['Burst Start'])):
        try:

            peak_df = Results['Peaks'][roi][results_bursts['Burst Start'].iloc[i]:
                                            results_bursts['Burst End'].iloc[i]]
            
            if peak_df.empty == True:
                burst_peak_max.append(NaN)
                burst_peak_id.append(NaN)
            else:
                max_peak = peak_df['Peaks Amplitude'].max()
                max_peak_time = peak_df['Peaks Amplitude'].idxmax()

                burst_peak_max.append(max_peak)
                burst_peak_id.append(max_peak_time)
        except:
            burst_peak_max.append(NaN)
            burst_peak_id.append(NaN)
            
    results_bursts['Peak Amp'] = burst_peak_max
    results_bursts['Peak Time'] = burst_peak_id
    
    results_bursts['Attack'] = results_bursts['Peak Time'] - results_bursts['Burst Start']
    results_bursts['Decay'] = results_bursts['Burst End'] - results_bursts['Peak Time']
    
    results_bursts['Intraburst Frequency'] = results_bursts['Peaks per Burst']/results_bursts['Burst Duration']
    #Burst Area
    if Settings['Burst Area'] == True:
        try:
            b_area = burstarea(np.array(Data), 
                                    Time, 
                                    results_bursts['Burst Start'].tolist(), 
                                    results_bursts['Burst End'].tolist())
            results_bursts['Burst Area'] = b_area
        except:
            #failure = True
            results_bursts['Burst Area'] = [NaN]

    return results_bursts, failure

def burstduration_rolling(time, data, baseline_rolling, threshperc, cluster_time):
    """
    threshperc is the percentage of the baseline that the threshold will be above; cluster_time should be changed based on the type of data (default for ECG is 0.006)
    baserate needs to be calculated and passed into this argument
    data should already be transformed, smoothed, and baseline shifted (shifting is technically optional, but it really doesn't matter)
    returns the lists of start times, end times, and duration times
    """
    
    if len(time) != len(data): #logic check, are the number of data points and time points the same?
        raise Exception('You cannot have more time points than there are data points. Get that sorted, buddy.')    
    
    burst_start = [] #empty array for burst start
    burst_end = [] #empty array for burst end
    burst_duration = [] #empty array to calculate burst durration
    b_start_amp = []
    b_end_amp = []
    
    threshold = np.array(baseline_rolling + threshperc) #calculate the point at which a event is considered a peak
    
    burst = False #burst flag, should start not in a burst

    index = -1
    for point in data: #for each data point in the set
        index = index +1
        #print index, "=", t.clock()
        
        if burst == False and point > threshold[index]: #if we are not in a burst already, the value is higher than the threshold, AND the last burst didn't end less than .2 ms ago
            tpoint = time[index] #find the actual time given teh time index
            burst_start.append(tpoint) #add the time at point as the begining of the burst
            b_start_amp.append(threshold[index])
            burst = True #burst flag, we are now in a burst 
        
        if burst == True and  point <= threshold[index]: #if we are in a burst and the point falls below the threshold
            
            if len(burst_end) == 0 or len(burst_start) == 0: #if this is the first end
                tpoint = time[index] #find the actual time given teh time index
                burst_end.append (tpoint) #add the time at point as the end of the burst
                b_end_amp.append(threshold[index])
                burst = False #burst flag, we are now out of the burst
            
            else:
                tpoint = time[index] #find the actual time given teh time index
                burst_end.append (tpoint) #add the time at point as the end of the burst
                b_end_amp.append(threshold[index])
                burst = False #burst flag, we are now out of the burst
                if burst_start[-1] < (burst_end[-2] + cluster_time):#if the new burst is too close to the last one, delete the second to last end and the last start
                    del burst_end[-2]
                    del b_end_amp[-2]
                    del burst_start[-1]
                    del b_start_amp[-1]
    
    if burst == True and len(burst_start) == 1+len(burst_end): #we exit the for loop but are in a burst
        #del burst_start[-1] #delete the last burst start time
        burst_end.append(time[-1]) #last time point is the end of burst
        #del b_start_amp[-1]
        b_end_amp.append(threshold[-1])
    if len(burst_start) != len(burst_end):
        raise Exception('Unexpectedly, the number of burst start times and end times are not equal. Seeing as this is physically impossible, I quit the program for you. Begin hunting for the fatal flaw. Good luck!')
        
    #print t.clock(), "- start duration array"
    for foo in burst_start: #for each burst
        index = burst_start.index(foo)
        duration = burst_end[index] - burst_start[index] #calculate duration by subtracting the start time from the end time, found by indexing
        burst_duration.append(duration) #add the burst duration to the duration list
    #print t.clock(), "-end duration array"
    
    return burst_start, burst_end, burst_duration, b_start_amp, b_end_amp

def burstduration_lin(time, data, baserate, threshperc, cluster_time):
    """
    threshperc is the percentage of the baseline that the threshold will be above; cluster_time should be changed based on the type of data (default for ECG is 0.006)
    baserate needs to be calculated and passed into this argument
    data should already be transformed, smoothed, and baseline shifted (shifting is technically optional, but it really doesn't matter)
    returns the lists of start times, end times, and duration times
    """
    
    if len(time) != len(data): #logic check, are the number of data points and time points the same?
        raise Exception('You cannot have more time points than there are data points. Get that sorted, buddy.')    
    
    burst_start = [] #empty array for burst start
    burst_end = [] #empty array for burst end
    burst_duration = [] #empty array to calculate burst durration
    
    threshold = baserate * threshperc #calculate the point at which a event is considered a peak
    
    burst = False #burst flag, should start not in a burst

    index = -1
    for point in data: #for each data point in the set
        index = index +1
        #print index, "=", t.clock()
        
        if burst == False and point > threshold: #if we are not in a burst already, the value is higher than the threshold, AND the last burst didn't end less than .2 ms ago
            tpoint = time[index] #find the actual time given teh time index
            burst_start.append(tpoint) #add the time at point as the begining of the burst
            burst = True #burst flag, we are now in a burst 
        
        if burst == True and  point <= threshold: #if we are in a burst and the point falls below the threshold
            
            if len(burst_end) == 0 or len(burst_start) == 0: #if this is the first end
                tpoint = time[index] #find the actual time given teh time index
                burst_end.append (tpoint) #add the time at point as the end of the burst
                burst = False #burst flag, we are now out of the burst
            
            else:
                tpoint = time[index] #find the actual time given teh time index
                burst_end.append (tpoint) #add the time at point as the end of the burst
                burst = False #burst flag, we are now out of the burst
                if burst_start[-1] < (burst_end[-2] + cluster_time):#if the new burst is too close to the last one, delete the second to last end and the last start
                    del burst_end[-2]
                    del burst_start[-1]
    
    if burst == True and len(burst_start) == 1+len(burst_end): #we exit the for loop but are in a burst
        #del burst_start[-1] #delete the last burst start time
        burst_end.append(time[-1]) #last time point is the end of burst
    if len(burst_start) != len(burst_end):
        raise Exception('Unexpectedly, the number of burst start times and end times are not equal. Seeing as this is physically impossible, I quit the program for you. Begin hunting for the fatal flaw. Good luck!')
        
    #print t.clock(), "- start duration array"
    for foo in burst_start: #for each burst
        index = burst_start.index(foo)
        duration = burst_end[index] - burst_start[index] #calculate duration by subtracting the start time from the end time, found by indexing
        burst_duration.append(duration) #add the burst duration to the duration list
    #print t.clock(), "-end duration array"
    
    return burst_start, burst_end, burst_duration


def interburstinterval(burst_start, burst_end):
    """
    this function is used to find the inter-burst interval. 
    this is defined as the difference between the last end and the new start time
    Dependent on numpy, burst_start, and burst_end
    """
    
    ibi = []
    
    for end in burst_end[:-1]: #for each end time except the last one
        tindex = burst_end.index(end) #find the start time index
        start = burst_start[tindex+1] #find start time
        ibi.append(start-end) #subtract the old end time from the start time
    
    return ibi

def ttotal(burst_start): 
    """
    find time from start to start, called the interburst interval. Input array must be a list of numbers (float).
    """
    
    ttotal = [] #empty array for ttot to go into
    
    for time in burst_start[1:]: #for each start time, starting with the second one
        s2time = burst_start.index(time) 
        s2 = burst_start[s2time-1]
        meas = time - s2 #measure the interval by subtracting
        ttotal.append(meas) #append the measurement to the ttotal array
    return ttotal #return array

def burstarea(data, time, burst_start, burst_end, dx = 10):
    """
    integral, area under curve of each burst. Use start and end times to split the y values into short lists. 
    need the time array to do this
    """
    from scipy.integrate import simps, trapz #import the integral functions
    
    time = list(time) #time array must be a list for the indexting to work.
    
    burst_area = [] #empty array for the areas to go into
    #count = 0
    for i in np.arange(len(burst_start)): #for each index in the start array
        end = time.index(burst_end[i]) #using the value at each i in the burst_end array, index in the time array to get the time index. this will be the same index # as the data array
        start = time.index(burst_start[i])
        area = trapz(data[start:end], x=time[start:end], dx= dx) #find area using the trapz function, but only 
        burst_area.append(area)
        #count = count + 1
        #print "%s = %s" %(count,area)
    return burst_area

#
#Save
#
#
def Save_Results(Data, Settings, Results):
    '''
    Save function for all files out of SWAN. All files must be present, otherwise it will not save. all files in the same folder with the same name will be saved over, except for the Settings file, which always has a unique name.
    '''
        #Save master files 
    Results['Peaks-Master'].to_csv(r'%s/%s_Peak_Results.csv'
                                   %(Settings['Output Folder'], Settings['Label']))
    Results['Valleys-Master'].to_csv(r'%s/%s_Valley_Results.csv'
                                   %(Settings['Output Folder'], Settings['Label']))
    Results['Bursts-Master'].to_csv(r'%s/%s_Bursts_Results.csv'
                                    %(Settings['Output Folder'], Settings['Label']))

    #Save Master Summary Files
    burst_grouped = Results['Bursts-Master'].groupby(level=0)
    burst_grouped = burst_grouped.describe()
    burst_grouped.to_csv(r'%s/%s_Bursts_Results_Summary.csv'
                                           %(Settings['Output Folder'], Settings['Label']))
    
    peak_grouped = Results['Peaks-Master'].groupby(level=0)
    peak_grouped= peak_grouped.describe()
    peak_grouped.to_csv(r'%s/%s_Peaks_Results_Summary.csv'
                                           %(Settings['Output Folder'], Settings['Label']))

    valley_grouped = Results['Valleys-Master'].groupby(level=0)
    valley_grouped= valley_grouped.describe()
    valley_grouped.to_csv(r'%s/%s_Valley_Results_Summary.csv'
                                           %(Settings['Output Folder'], Settings['Label']))

    #save settings
    Settings_panda = DataFrame.from_dict(Settings, orient='index')
    now = datetime.datetime.now()
    colname = 'Settings: ' + str(now)
    Settings_panda.columns = [colname]
    Settings_panda = Settings_panda.sort()
    Settings_panda.to_csv(r"%s/%s_Settings_%s.csv"%(Settings['Output Folder'], 
                                                 Settings['Label'], 
                                                 now.strftime('%Y_%m_%d__%H_%M_%S')))
    
    print "All results saved to: ", Settings['Output Folder']
    print "Thank you for chosing BASS.py for all your basic analysis needs. Proceed for graphs and advanced analysis."
    print "Analysis Complete"
    
    print "\n--------------------------------------------"

    print "Data Column Names/Keys"
    print "-----"
    for name in Data['original']:
        print name
    print "\n--------------------------------------------"
    print "Available Measurements from Peaks for further analysis:"
    print "-----"
    for label, col in Results['Peaks-Master'].iteritems():
        print label
    print "\n--------------------------------------------"
    print "Available Measurements from Bursts for further analysis:"
    print "-----"
    for label, col in Results['Bursts-Master'].iteritems():
        print label
    
    print "\n---------------------------"
    print '|Event Detection Complete!|'
    print "---------------------------"
    
#
#Line Plots
#general plots as well as graphing functions
#several of the other functions have specific, paired functions, thus their code is not listed here
#

def plot_rawdata(Data):
    figure = plt.plot(Data['original'].index, Data['original'].ix[:,0], 'k')
    plt.title(r'Raw Data %s' %Data['original'].ix[:,0].name)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    DataCursor(figure)
    plt.show(figure) 

def results_timeseries_plot(roi, start, end, Data, Settings, Results):
    '''
    plots a dual line plot of your raw signal and your transformed signal with events overlayed.
    automatically the first 10 seconds of data are displayed. you can pan around inside to find the part you want to see.
    '''
    
    if roi.lower() == 'random':
        rand_int = np.random.randint(len(Data['original'].columns))
        roi = Data['original'].columns[rand_int]
    
    #set the correct data array
    if Settings['Baseline Type'] == 'static':
        data_temp = Data['trans']
    elif Settings['Baseline Type'] == 'linear': 
        data_temp = Data['shift']
    elif Settings['Baseline Type'] == 'rolling':
        data_temp = Data['rolling']

    
    f1 = plt.subplot(211)
    f1.plot(Data['original'].index, Data['original'][roi], 'k')
    f1.set_title(roi +' Raw Data')
    f1.set_ylabel('Amplitude')
    

    f2 = plt.subplot(212, sharex=f1)
    f2.plot(data_temp.index, data_temp[roi], color = 'k', label= 'Time Series') #plot time series
    f2.set_ylim(ymin= min(data_temp.min()), ymax =max(data_temp.max()))
    f2.set_xlim(xmin = start, xmax = end)
    f2.set_ylabel('Relative Amplitude')
    f2.set_title(roi+' Events')
    
    #plot peaks and valleys
    try:
        f2.plot(Results['Peaks'][roi].index, Results['Peaks'][roi]['Peaks Amplitude'], 
                 marker = '^', color = 'b', linestyle = 'None', alpha = 1, label = 'RAIN Peak', markersize = 5)
    except:
        pass
    try:
        f2.plot(Results['Valleys'][roi].index, Results['Valleys'][roi]['Valley Amplitude'], 
                 marker = 'v', color = 'm', linestyle = 'None', alpha = 1, label = 'RAIN Valley', markersize = 5)
    except:
        pass
    #plot bursts
    try:
        if Settings['Baseline Type'] == 'static':
            start_y = []
            end_y = []
            for i in np.arange(len(Results['Bursts'][roi]['Burst Start'])):
                start_y.append(Settings['Threshold'])
            for i in np.arange(len(Results['Bursts'][roi]['Burst End'])):
                end_y.append(Settings['Threshold'])
            f2.plot(Results['Bursts'][roi]['Burst Start'], start_y,
                     marker = 's', color = 'g', linestyle = 'None', alpha = 1, label = 'Burst Start', markersize = 5)
            f2.plot(Results['Bursts'][roi]['Burst End'], end_y,
                     marker = 's', color = 'y', linestyle = 'None', alpha = 1, label = 'Burst End', markersize = 5)
            
        elif Settings['Baseline Type'] == 'linear': 
            start_y = []
            end_y = []
            for i in np.arange(len(Results['Bursts'][roi]['Burst Start'])):
                start_y.append(Results['Baseline'][roi]*Settings['Threshold'])
            for i in np.arange(len(Results['Bursts'][roi]['Burst End'])):
                end_y.append(Results['Baseline'][roi]*Settings['Threshold'])
            f2.plot(Results['Bursts'][roi]['Burst Start'], start_y,
                     marker = 's', color = 'g', linestyle = 'None', alpha = 1, label = 'Burst Start', markersize = 5)
            f2.plot(Results['Bursts'][roi]['Burst End'], end_y,
                     marker = 's', color = 'y', linestyle = 'None', alpha = 1, label = 'Burst End', markersize = 5)


        elif Settings['Baseline Type'] == 'rolling':
            f2.plot(Results['Bursts'][roi]['Burst Start'], Results['Bursts'][roi]['Burst Start Amplitude'],
                     marker = 's', color = 'g', linestyle = 'None', alpha = 1, label = 'Burst Start', markersize = 5)
            f2.plot(Results['Bursts'][roi]['Burst End'], Results['Bursts'][roi]['Burst End Amplitude'],
                     marker = 's', color = 'y', linestyle = 'None', alpha = 1, label = 'Burst End', markersize = 5)
            f2.plot(Results['Baseline-Rolling'][roi].index, Results['Baseline-Rolling'][roi], 'b') #in this instance, baseline = rolling average
            f2.plot(Results['Baseline-Rolling'][roi].index, Results['Baseline-Rolling'][roi]+Settings['Threshold'], 'r')
    
    except:#this would be the case that no bursts were detected
        pass
    try:
        if Settings['Baseline Type'] == 'static':
            f2.hlines(Settings['Threshold'], Data['original'].index[0], Data['original'].index[-1], 'b', label = 'RAIN threshold')

        elif Settings['Baseline Type'] == 'linear': 
            f2.hlines(Results['Baseline'][roi]*Settings['Threshold'], Data['original'].index[0], 
                       Data['original'].index[-1], 'b', label = 'RAIN threshold')
            f2.hlines(0, Data['original'].index[0], Data['original'].index[-1], 'k', label = 'Baseline')

        elif Settings['Baseline Type'] == 'rolling':
            f2.plot(Results['Baseline-Rolling'][roi].index, Results['Baseline-Rolling'][roi], 'b') #in this instance, baseline = rolling average
            f2.plot(Results['Baseline-Rolling'][roi].index, Results['Baseline-Rolling'][roi]+Settings['Threshold'], 'r')
    except:
        pass
    
    plt.show()

def graph_detected_events_save(Data, Settings, Results, roi, lcpro = False):
    '''
    Graph burst and peak results for a given roi. if roi = 'random', then a random roi will be selected.
    There is also the option to overlay LCpro events and boundaries.
    '''
    
    if roi.lower() == 'random':
        rand_int = np.random.randint(len(Data['original'].columns))
        roi = Data['original'].columns[rand_int]
    
    #set the correct data array
    if Settings['Baseline Type'] == 'static':
        data_temp = Data['trans']
    elif Settings['Baseline Type'] == 'linear': 
        data_temp = Data['shift']
    elif Settings['Baseline Type'] == 'rolling':
        data_temp = Data['rolling']

    plt.figure()
    plt.plot(data_temp.index, data_temp[roi], color = 'k', label= 'Time Series') #plot time series
    plt.ylim(ymin= min(data_temp.min()), ymax =max(data_temp.max()))
    plt.xlim(xmin = -1)
    plt.title(roi+' Events')
    
    #plot peaks and valleys
    try:
        plt.plot(Results['Peaks'][roi].index, Results['Peaks'][roi]['Peaks Amplitude'], 
                 marker = '^', color = 'b', linestyle = 'None', alpha = 1, label = 'RAIN Peak', markersize = 5)
    except:
        pass
    try:
        plt.plot(Results['Valleys'][roi].index, Results['Valleys'][roi]['Valley Amplitude'], 
                 marker = 'v', color = 'm', linestyle = 'None', alpha = 1, label = 'RAIN Valley', markersize = 5)
    except:
        pass
    #plot bursts
    try:
        if Settings['Baseline Type'] == 'static':
            start_y = []
            end_y = []
            for i in np.arange(len(Results['Bursts'][roi]['Burst Start'])):
                start_y.append(Settings['Threshold'])
            for i in np.arange(len(Results['Bursts'][roi]['Burst End'])):
                end_y.append(Settings['Threshold'])
            plt.plot(Results['Bursts'][roi]['Burst Start'], start_y,
                     marker = 's', color = 'g', linestyle = 'None', alpha = 1, label = 'Burst Start', markersize = 5)
            plt.plot(Results['Bursts'][roi]['Burst End'], end_y,
                     marker = 's', color = 'y', linestyle = 'None', alpha = 1, label = 'Burst End', markersize = 5)
            
        elif Settings['Baseline Type'] == 'linear': 
            start_y = []
            end_y = []
            for i in np.arange(len(Results['Bursts'][roi]['Burst Start'])):
                start_y.append(Results['Baseline'][roi]*Settings['Threshold'])
            for i in np.arange(len(Results['Bursts'][roi]['Burst End'])):
                end_y.append(Results['Baseline'][roi]*Settings['Threshold'])
            plt.plot(Results['Bursts'][roi]['Burst Start'], start_y,
                     marker = 's', color = 'g', linestyle = 'None', alpha = 1, label = 'Burst Start', markersize = 5)
            plt.plot(Results['Bursts'][roi]['Burst End'], end_y,
                     marker = 's', color = 'y', linestyle = 'None', alpha = 1, label = 'Burst End', markersize = 5)


        elif Settings['Baseline Type'] == 'rolling':
            plt.plot(Results['Bursts'][roi]['Burst Start'], Results['Bursts'][roi]['Burst Start Amplitude'],
                     marker = 's', color = 'g', linestyle = 'None', alpha = 1, label = 'Burst Start', markersize = 5)
            plt.plot(Results['Bursts'][roi]['Burst End'], Results['Bursts'][roi]['Burst End Amplitude'],
                     marker = 's', color = 'y', linestyle = 'None', alpha = 1, label = 'Burst End', markersize = 5)
            plt.plot(Results['Baseline-Rolling'][roi].index, Results['Baseline-Rolling'][roi], 'b') #in this instance, baseline = rolling average
            plt.plot(Results['Baseline-Rolling'][roi].index, Results['Baseline-Rolling'][roi]+Settings['Threshold'], 'r')
    
    except:#this would be the case that no bursts were detected
        pass
    try:
        if Settings['Baseline Type'] == 'static':
            plt.hlines(Settings['Threshold'], Data['original'].index[0], Data['original'].index[-1], 'b', label = 'RAIN threshold')

        elif Settings['Baseline Type'] == 'linear': 
            plt.hlines(Results['Baseline'][roi]*Settings['Threshold'], Data['original'].index[0], 
                       Data['original'].index[-1], 'b', label = 'RAIN threshold')
            plt.hlines(0, Data['original'].index[0], Data['original'].index[-1], 'k', label = 'Baseline')
            
        elif Settings['Baseline Type'] == 'rolling':
            plt.plot(Results['Baseline-Rolling'][roi].index, Results['Baseline-Rolling'][roi], 'b') #in this instance, baseline = rolling average
            plt.plot(Results['Baseline-Rolling'][roi].index, Results['Baseline-Rolling'][roi]+Settings['Threshold'], 'r')
    except:
        pass
    
    #LCPRO events
    if lcpro == True:
        
        if Settings['Baseline Type'] == 'static' or Settings['Baseline Type'] == 'rolling':
            plt.plot(Data['ROI parameters']['Time(s)'].loc[roi], Data['ROI parameters']['Amp(F/F0)'].loc[roi], 
                     marker = 'o', color = 'r', linestyle = 'none', alpha = 0.4, label = 'LCPro Peak', markersize = 10)
        elif Settings['Baseline Type'] == 'linear':
            plt.plot(Data['ROI parameters']['Time(s)'].loc[roi], 
                     (Data['ROI parameters']['Amp(F/F0)'].loc[roi] - Results['Baseline'][roi]), 
                     marker = 'o', color = 'r', linestyle = 'none', alpha = 0.4, label = 'LCPro Peak', markersize = 10)
            
        plt.vlines(Data['ROI parameters']['Start time (s)'].loc[roi], ymin= min(data_temp.min()), 
                   ymax =max(data_temp.max()), color = 'r', label = 'LCpro Boundary')
        plt.vlines(Data['ROI parameters']['End time (s)'].loc[roi], ymin= min(data_temp.min()), 
                   ymax =max(data_temp.max()), color = 'r')
    
    #plt.legend()
    plt.savefig(r'%s/%s.pdf'%(Settings['plots folder'],roi))
    plt.close()

def graph_detected_events_display(Data, Settings, Results, roi, lcpro = False):
    '''
    Graph burst and peak results for a given roi. if roi = 'random', then a random roi will be selected.
    There is also the option to overlay LCpro events and boundaries.
    '''
    
    if roi.lower() == 'random':
        rand_int = np.random.randint(len(Data['original'].columns))
        roi = Data['original'].columns[rand_int]
    
    #set the correct data array
    if Settings['Baseline Type'] == 'static':
        data_temp = Data['trans']
    elif Settings['Baseline Type'] == 'linear': 
        data_temp = Data['shift']
    elif Settings['Baseline Type'] == 'rolling':
        data_temp = Data['rolling']

    plt.figure()
    plt.plot(data_temp.index, data_temp[roi], color = 'k', label= 'Time Series') #plot time series
    plt.ylim(ymin= min(data_temp.min()), ymax =max(data_temp.max()))
    plt.xlim(xmin = -1)
    plt.title(roi+' Events')
    
    #plot peaks and valleys
    try:
        plt.plot(Results['Peaks'][roi].index, Results['Peaks'][roi]['Peaks Amplitude'], 
                 marker = '^', color = 'b', linestyle = 'None', alpha = 1, label = 'RAIN Peak', markersize = 5)
    except:
        pass
    try:
        plt.plot(Results['Valleys'][roi].index, Results['Valleys'][roi]['Valley Amplitude'], 
                 marker = 'v', color = 'm', linestyle = 'None', alpha = 1, label = 'RAIN Valley', markersize = 5)
    except:
        pass
    #plot bursts
    try:
        if Settings['Baseline Type'] == 'static':
            start_y = []
            end_y = []
            for i in np.arange(len(Results['Bursts'][roi]['Burst Start'])):
                start_y.append(Settings['Threshold'])
            for i in np.arange(len(Results['Bursts'][roi]['Burst End'])):
                end_y.append(Settings['Threshold'])
            plt.plot(Results['Bursts'][roi]['Burst Start'], start_y,
                     marker = 's', color = 'g', linestyle = 'None', alpha = 1, label = 'Burst Start', markersize = 5)
            plt.plot(Results['Bursts'][roi]['Burst End'], end_y,
                     marker = 's', color = 'y', linestyle = 'None', alpha = 1, label = 'Burst End', markersize = 5)
            
        elif Settings['Baseline Type'] == 'linear': 
            start_y = []
            end_y = []
            for i in np.arange(len(Results['Bursts'][roi]['Burst Start'])):
                start_y.append(Results['Baseline'][roi]*Settings['Threshold'])
            for i in np.arange(len(Results['Bursts'][roi]['Burst End'])):
                end_y.append(Results['Baseline'][roi]*Settings['Threshold'])
            plt.plot(Results['Bursts'][roi]['Burst Start'], start_y,
                     marker = 's', color = 'g', linestyle = 'None', alpha = 1, label = 'Burst Start', markersize = 5)
            plt.plot(Results['Bursts'][roi]['Burst End'], end_y,
                     marker = 's', color = 'y', linestyle = 'None', alpha = 1, label = 'Burst End', markersize = 5)


        elif Settings['Baseline Type'] == 'rolling':
            plt.plot(Results['Bursts'][roi]['Burst Start'], Results['Bursts'][roi]['Burst Start Amplitude'],
                     marker = 's', color = 'g', linestyle = 'None', alpha = 1, label = 'Burst Start', markersize = 5)
            plt.plot(Results['Bursts'][roi]['Burst End'], Results['Bursts'][roi]['Burst End Amplitude'],
                     marker = 's', color = 'y', linestyle = 'None', alpha = 1, label = 'Burst End', markersize = 5)
            plt.plot(Results['Baseline-Rolling'][roi].index, Results['Baseline-Rolling'][roi], 'b') #in this instance, baseline = rolling average
            plt.plot(Results['Baseline-Rolling'][roi].index, Results['Baseline-Rolling'][roi]+Settings['Threshold'], 'r')
    
    except:#this would be the case that no bursts were detected
        pass
    
    try:
        if Settings['Baseline Type'] == 'static':
            plt.hlines(Settings['Threshold'], Data['original'].index[0], Data['original'].index[-1], 'b', label = 'RAIN threshold')

        elif Settings['Baseline Type'] == 'linear': 
            plt.hlines(Results['Baseline'][roi]*Settings['Threshold'], Data['original'].index[0], 
                       Data['original'].index[-1], 'b', label = 'RAIN threshold')
            plt.hlines(0, Data['original'].index[0], Data['original'].index[-1], 'k', label = 'Baseline')
            
        elif Settings['Baseline Type'] == 'rolling':
            plt.plot(Results['Baseline-Rolling'][roi].index, Results['Baseline-Rolling'][roi], 'b') #in this instance, baseline = rolling average
            plt.plot(Results['Baseline-Rolling'][roi].index, Results['Baseline-Rolling'][roi]+Settings['Threshold'], 'r')
    except:#baseline has not yet been set
        pass

    
    #LCPRO events
    if lcpro == True:
        
        if Settings['Baseline Type'] == 'static' or Settings['Baseline Type'] == 'rolling':
            plt.plot(Data['ROI parameters']['Time(s)'].loc[roi], Data['ROI parameters']['Amp(F/F0)'].loc[roi], 
                     marker = 'o', color = 'r', linestyle = 'none', alpha = 0.4, label = 'LCPro Peak', markersize = 10)
        elif Settings['Baseline Type'] == 'linear':
            plt.plot(Data['ROI parameters']['Time(s)'].loc[roi], 
                     (Data['ROI parameters']['Amp(F/F0)'].loc[roi] - Results['Baseline'][roi]), 
                     marker = 'o', color = 'r', linestyle = 'none', alpha = 0.4, label = 'LCPro Peak', markersize = 10)
            
        plt.vlines(Data['ROI parameters']['Start time (s)'].loc[roi], ymin= min(data_temp.min()), 
                   ymax =max(data_temp.max()), color = 'r', label = 'LCpro Boundary')
        plt.vlines(Data['ROI parameters']['End time (s)'].loc[roi], ymin= min(data_temp.min()), 
                   ymax =max(data_temp.max()), color = 'r')
    
    #plt.legend()
    #plt.savefig(r'%s/%s.pdf'%(Settings['plots folder'],roi))
    #plt.close()
    plt.show()

def average_measurement_plot(event_type, meas, Results):
    """
    Generates a line plot with error bars for a given event measurement. 
    X axis is the names of each time series.
    Parameters
    ----------
    event_type: string
        A string that should be either 'Peaks' or 'Bursts', which will tell the function 
        which results to pull the measurement from.
    meas: string
        A string that specifies which measurement type to use for the figure.
    Results: dictionary
        The dictionary that contains all of the results. this function uses either Results['Peaks-Master'] or Results['Bursts-Master'].
    Returns
    -------
    none
    Notes
    -----
    Displays the figure. does not automatically save out or return the figure object.
    Examples
    -----
    event_type = 'Peaks'
    meas = 'Peaks Amplitude'
    average_measurement_plot(event_type, meas,Results)
    """
    
    if event_type.lower() == 'peaks':
        peak_grouped = Results['Peaks-Master'].groupby(level=0)
        measurement = peak_grouped[meas]
    elif event_type.lower() == 'bursts':
        burst_grouped = Results['Bursts-Master'].groupby(level=0)
        measurement = burst_grouped[meas]
    else:
        raise ValueError('Not an acceptable event type measurement.\n Must be "Peaks" or "Bursts" ')
    
    try:
        plt.errorbar(measurement.mean().index, measurement.mean(), measurement.std(), marker = '^')
        plt.xlabel('Groups')
        plt.ylabel('%s' %(meas))
        plt.title('Average %s %s with standard deviation' %(event_type,meas))
        plt.show()
    except ValueError: #this occurs when the column names are strings that cannot be converted into floats
        temp = np.arange(len(measurement.mean().index))
        plt.errorbar(temp, measurement.mean(), measurement.std(), marker = '^') #use 'order' number instead
        plt.xlabel('Groups (by order number)')
        plt.ylabel('%s' %(meas))
        plt.title('Average %s %s with standard deviation' %(event_type,meas))
        plt.show()
    
class DataCursor(object):
    """A simple data cursor widget that displays the x,y location of a
    matplotlib artist when it is selected."""
    def __init__(self, artists, tolerance=5, offsets=(-20, 20), 
                 template='x: %0.2f\ny: %0.2f', display_all=False):
        """Create the data cursor and connect it to the relevant figure.
        "artists" is the matplotlib artist or sequence of artists that will be 
            selected. 
        "tolerance" is the radius (in points) that the mouse click must be
            within to select the artist.
        "offsets" is a tuple of (x,y) offsets in points from the selected
            point to the displayed annotation box
        "template" is the format string to be used. Note: For compatibility
            with older versions of python, this uses the old-style (%) 
            formatting specification.
        "display_all" controls whether more than one annotation box will
            be shown if there are multiple axes.  Only one will be shown
            per-axis, regardless. 
        """
        self.template = template
        self.offsets = offsets
        self.display_all = display_all
        if not cbook.iterable(artists):
            artists = [artists]
        self.artists = artists
        self.axes = tuple(set(art.axes for art in self.artists))
        self.figures = tuple(set(ax.figure for ax in self.axes))

        self.annotations = {}
        for ax in self.axes:
            self.annotations[ax] = self.annotate(ax)

        for artist in self.artists:
            artist.set_picker(tolerance)
        for fig in self.figures:
            fig.canvas.mpl_connect('pick_event', self)

    def annotate(self, ax):
        """Draws and hides the annotation box for the given axis "ax"."""
        annotation = ax.annotate(self.template, xy=(0, 0), ha='right',
                xytext=self.offsets, textcoords='offset points', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
                )
        annotation.set_visible(False)
        return annotation

    def __call__(self, event):
        """Intended to be called through "mpl_connect"."""
        # Rather than trying to interpolate, just display the clicked coords
        # This will only be called if it's within "tolerance", anyway.
        x, y = event.mouseevent.xdata, event.mouseevent.ydata
        annotation = self.annotations[event.artist.axes]
        if x is not None:
            if not self.display_all:
                # Hide any other annotation boxes...
                for ann in self.annotations.values():
                    ann.set_visible(False)
            # Update the annotation in the current axis..
            annotation.xy = x, y
            annotation.set_text(self.template % (x, y))
            annotation.set_visible(True)
            event.canvas.draw()

def raster(Results):
    """
    Generates the raster plot from the spikes generated in a dataframe
    Parameters
    ----------
    Results: dictionary
        dictionary that holds the results. this function relies on the Results['Peaks'].
    Returns
    -------
    none
    Notes
    -----
    Displays the raster plot of the whole time series. 
    Keys that are stings are converted into order number. 
    """
    n=0
    for key, value in Results['Peaks'].iteritems():
        try:
            key = float(key)
        except:
            legend = key
            key = n
            
        temp_y = []
        for n in np.arange(len(value)):
            temp_y.append(key)
        plt.plot(value.index, temp_y, marker = '.', color = 'k', linestyle = 'None', markersize = 2)
        n+=1
    plt.xlabel('Time (s)')
    plt.ylabel('Groups')
    #plt.ylim(ymin = 2.5)
    #plt.xlim(xmin = 30000, xmax = 45000)
    plt.title('Raster plot')
    #plt.legend()
    plt.show()
    
#
#Event Frequency plots
#
#
def frequency_plot(event_type, meas, key, Data, Settings, Results):
    """
    Generate measurement per minue vs seconds. does not support 'all'
    Parameters
    ----------
    event_type: string
        A string that should be either 'Peaks' or 'Bursts', which will tell the function 
        which results to pull the measurement from.
    meas: string
        A string that specifies which measurement type to use for the figure.
    Data: dictionary
        dictionary containing the pandas dataframes that store the data.
        this function uses the column names of the original data file to organize the 
         Entropy table.
    Settings: dictionary

    Results: dictionary
        The dictionary that contains all of the results. Results['Peaks'] and/or Results['Bursts']
        is used. Results['Peaks-Master'] or Results['Bursts-Master'] are also used to 
    
    Returns
    -------
    NONE
    Notes
    -----
    
    Examples
    --------

    References
    ----------

    """
    
    if event_type.lower() == 'peaks':
        measurement = Results['Peaks']
        columns = Results['Peaks-Master']

    elif event_type.lower() == 'bursts':
        measurement = Results['Bursts']
        columns = Results['Bursts-Master']
    else:
        raise ValueError('Not an acceptable event type measurement.\n Must be "Peaks" or "Bursts" ')

    try:
        freq_list = measurement[key][meas].tolist()
        time_list = measurement[key].index.tolist()

        if freq_list[-1] != freq_list[-1]: #if NaN
            freq_list = freq_list[:-1]
            time_list = time_list[:-1]
        
        freq_min = []
        for i in freq_list:
            freq_min.append(60/i) #conver to per min, instead of sec
        
        plt.plot(time_list, freq_min)
        plt.ylabel('Event Measurement Rate (meas/min.)')
        plt.xlabel('Time (s)')
        plt.xlim(time_list[0], time_list[-1])
        plt.title('Event Freqency- %s: %s' %(event_type, meas))
        plt.show()
    except:
        print "Could not display graph. idk. pick another measurement to graph."
#            
# Poincare plots
#
#
def poincare(data_array):
    """
    Given a 1d array of data, create a Poincare plot along with the SD1 and SD2 parameters
    usees matplotlib.patches.Ellipse to create the fit elipse
    equations are derived from Brennan and http://www.mif.pg.gda.pl/homepages/graff/Tzaa/Guzik_geometry_asymetry.pdf cause OMG THIS MAKES SENSE NOW
    """
    
    x = data_array[:-1]
    y = data_array[1:]
    
    xc = np.mean(x)
    yc = np.mean(y)
    
    #SD1 = np.sqrt((1/len(x)) * sum(((x-y) - np.mean(x-y))^2)/2)
    #SD2 = np.sqrt((1/len(x)) * sum(((x+y) - np.mean(x+y))^2)/2)    
    
    SD1 = 0
    SD2 = 0
    
    for i in np.arange(len(x)):
        
        d1 = np.power(abs((x[i]-xc)-(y[i]-yc))/np.sqrt(2), 2)
        SD1 = SD1 + d1
        
        d2 = np.power(abs((x[i]-xc)+(y[i]-yc))/np.sqrt(2), 2)
        SD2 = SD2 + d2
    
    SD1 = np.sqrt(SD1/len(x))
    SD2 = np.sqrt(SD2/len(x))
    
    return x, y, xc, yc, SD1, SD2 

def poincare_plot(series):
    '''
    input a dataframe column or series to automatically generate a poincare plot. it will also print out the SD1 and SD2 Values.
    '''
    temp_series = series.tolist()
    
    x, y, xc, yc, SD1, SD2 = poincare(temp_series[:-1])

    SD1RR = ('SD1 = ' +str(np.round(SD1,4)))
    SD2RR = ('SD2 = ' +str(np.round(SD2,4)))
    SDRR = (SD1RR, SD2RR)
    ax = plt.subplot(111, aspect = "equal")
    ellipse = patches.Ellipse(xy=(xc, yc), width = SD2, height = SD1, angle = 45, fill = False, color = "r")
    ax.add_artist(ellipse)
    plt.plot(xc, yc, color="r", marker= "+")
    plt.scatter(x, y, color = 'k', marker = '.')
    plt.title('Poincare Plot-%s' %series.name)
    #plt.text(-4,-4,SDRR) #this is commented out because it isn't running. not sure why.
    plt.show()

    print series.name, "results:"
    print SD1RR+' s'
    print SD2RR+' s'

    
#
#Power Spectral Density
#
#

def psd_signal(version, key, Data, Settings, Results):
    '''
    get powerspectral density from signal data
    '''
    sig = Data[version][key].tolist()
    hertz = 2/Settings['Sample Rate (s/frame)']
    Fxx, Pxx = scipy.signal.welch(sig, fs = hertz, window="hanning", 
                                  nperseg=256, noverlap=128, detrend="linear")
    

    plt.plot(Fxx, Pxx)
    plt.xlim(xmin = 0, xmax = hertz/2)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel(r"PSD $(ms^ 2$/Hz)")
    plt.title("PSD-%s: %s" %(version, key))
    plt.show()
    
    FXX = Series(Fxx)
    PXX = Series(Pxx)
    if 'PSD-Signal'not in Results.keys():
        Results['PSD-Signal'] = DataFrame(index = ['ULF', 'VLF', 'LF','HF','LF/HF'])
    
    results_psd = Series(index = ['ULF', 'VLF', 'LF','HF','LF/HF'])

    results_psd['ULF'] = scipy.integrate.simps(PXX[FXX<Settings['PSD-Signal']['ULF']].tolist(), 
                               FXX[FXX<Settings['PSD-Signal']['ULF']].tolist(), 
                               dx =Settings['PSD-Signal']['dx'])
    results_psd['VLF'] = scipy.integrate.simps(PXX[(Settings['PSD-Signal']['ULF']<FXX) & (FXX<=Settings['PSD-Signal']['VLF'])].tolist(),
                               FXX[(Settings['PSD-Signal']['ULF']<FXX) & (FXX<=Settings['PSD-Signal']['VLF'])].tolist(), 
                               dx= Settings['PSD-Signal']['dx'])
    results_psd['LF'] = scipy.integrate.simps(PXX[(Settings['PSD-Signal']['VLF']<FXX) & (FXX<=Settings['PSD-Signal']['LF'])].tolist(),
                              FXX[(Settings['PSD-Signal']['VLF']<FXX) & (FXX<=Settings['PSD-Signal']['LF'])].tolist(), 
                              dx= Settings['PSD-Signal']['dx'])
    results_psd['HF'] = scipy.integrate.simps(PXX[(Settings['PSD-Signal']['LF']<FXX) & (FXX<=Settings['PSD-Signal']['HF'])].tolist(),
                              FXX[(Settings['PSD-Signal']['LF']<FXX) & (FXX<=Settings['PSD-Signal']['HF'])].tolist(), 
                              dx= Settings['PSD-Signal']['dx'])
    results_psd['LF/HF'] = results_psd['LF']/results_psd['HF']
    
    Results['PSD-Signal'][key] = results_psd

    Results['PSD-Signal'].to_csv(r'%s/%s_PSD_Signal.csv'
                                           %(Settings['Output Folder'], Settings['Label']))
    Settings['PSD-Signal'].to_csv(r'%s/%s_PSD_Signal_Settings.csv'
                                           %(Settings['Output Folder'], Settings['Label']))
    return Results

def psd_event(event_type, meas, key, Data, Settings, Results):
    '''
    get powerspectral density from one column's event measurement.
    
    example:
    event_type = 'Peaks'
    meas = 'Intervals'
    key = 'Mean1'
    
    #adopted form Rhenan Bartels Ferreira 
    #https://github.com/RhenanBartels/biosignalprocessing/blob/master/psdRRi.py
    '''
    
    if event_type.lower() == 'peaks':
        measurement = Results['Peaks']
        columns = Results['Peaks-Master']

    elif event_type.lower() == 'bursts':
        measurement = Results['Bursts']
        columns = Results['Bursts-Master']
    else:
        raise ValueError('Not an acceptable event type measurement.\n Must be "Peaks" or "Bursts" ')


    freq_list = measurement[key][meas].tolist()
    time_list = measurement[key].index.tolist()

    if freq_list[-1] != freq_list[-1]: #check if NaN
        freq_list = freq_list[:-1]
        time_list = time_list[:-1]
        

    #evenly spaced array using interpolation hertz        
    
    tx = np.arange(time_list[0], time_list[-1], (1.0/(Settings['PSD-Event']['hz'])))
   
    #interpolate!
    tck = scipy.interpolate.splrep(time_list, freq_list, s = 0)
    
    freq_x = scipy.interpolate.splev(tx, tck, der = 0)
    
    #Number os estimations
    P = int((len(tx) - 256 / 128)) + 1 #AD doesn't know what this does, but i dare not touch a damn thing.
    
    Fxx, Pxx = scipy.signal.welch(freq_x, fs = Settings['PSD-Event']['hz'], window="hanning", 
                                  nperseg=256, noverlap=128, detrend="linear")
    

    plt.plot(Fxx, Pxx)
    plt.xlim(xmin = 0, xmax = Settings['PSD-Event']['hz']/2)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel(r"PSD $(ms^ 2$/Hz)")
    plt.title("PSD-%s: %s-%s" %(key, event_type, meas))
    plt.show()
    
    FXX = Series(Fxx)
    PXX = Series(Pxx)
    if 'PSD-Event'not in Results.keys():
        Results['PSD-Event'] = {}
    
    if key not in Results['PSD-Event'].keys():
        Results['PSD-Event'][key] = DataFrame(index = ['ULF', 'VLF', 'LF','HF','LF/HF'])
    
    results_psd = Series(index = ['ULF', 'VLF', 'LF','HF','LF/HF'])

    results_psd['ULF'] = scipy.integrate.simps(PXX[FXX<Settings['PSD-Event']['ULF']].tolist(), 
                               FXX[FXX<Settings['PSD-Event']['ULF']].tolist(), 
                               dx =Settings['PSD-Event']['dx'])
    results_psd['VLF'] = scipy.integrate.simps(PXX[(Settings['PSD-Event']['ULF']<FXX) & (FXX<=Settings['PSD-Event']['VLF'])].tolist(),
                               FXX[(Settings['PSD-Event']['ULF']<FXX) & (FXX<=Settings['PSD-Event']['VLF'])].tolist(), 
                               dx= Settings['PSD-Event']['dx'])
    results_psd['LF'] = scipy.integrate.simps(PXX[(Settings['PSD-Event']['VLF']<FXX) & (FXX<=Settings['PSD-Event']['LF'])].tolist(),
                              FXX[(Settings['PSD-Event']['VLF']<FXX) & (FXX<=Settings['PSD-Event']['LF'])].tolist(), 
                              dx= Settings['PSD-Event']['dx'])
    results_psd['HF'] = scipy.integrate.simps(PXX[(Settings['PSD-Event']['LF']<FXX) & (FXX<=Settings['PSD-Event']['HF'])].tolist(),
                              FXX[(Settings['PSD-Event']['LF']<FXX) & (FXX<=Settings['PSD-Event']['HF'])].tolist(), 
                              dx= Settings['PSD-Event']['dx'])
    results_psd['LF/HF'] = results_psd['LF']/results_psd['HF']
    
    Results['PSD-Event'][key][meas] = results_psd
    
    temp_psd_master = pd.concat(Results['PSD-Event'])
    temp_psd_master.to_csv(r'%s/%s_PSD_Events.csv'
                                           %(Settings['Output Folder'], Settings['Label']))
    Settings['PSD-Event'].to_csv(r'%s/%s_PSD_Events_Settings.csv'
                                           %(Settings['Output Folder'], Settings['Label']))
    return Results

def PSD_Peaks(Results, results_PSD, Settings_PSD):
    Fxx_rii, Pxx_rii = PSD_rri(Results['Peaks'], hz)
    results_PSD['Peakdet'] = Results_PSD(Fxx_rii, Pxx_rii)
    plt.plot(Fxx_rii, Pxx_rii)
    plt.xlim(xmin = 0, xmax = hz/2)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel(r"PSD $(ms^ 2$/Hz)")
    plt.title("PSD-Peaks")
    plt.show()

def PSD_bursts(Results, results_PSD, Settings_PSD):
    Fxx_tct, Pxx_tct = PSD_tct(Results['Bursts'], hz)
    results_PSD['Burstdet'] = Results_PSD(Fxx_tct, Pxx_tct)
    plt.plot(Fxx_tct, Pxx_tct)
    plt.xlim(xmin = 0, xmax = hz/2)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel(r"PSD $(ms^ 2$/Hz)")
    plt.title("PSD-Bursts")
    plt.show()
    
def PSD_rri(results_peaks, hz):
    '''
    Power spectral density plot of rr intervals. this argument takes the results_peaks dataframe
    '''
    #adopted form Rhenan Bartels Ferreira 
    #https://github.com/RhenanBartels/biosignalprocessing/blob/master/psdRRi.py

    from numpy import arange, cumsum, logical_and
    from scipy.signal import welch
    from scipy.interpolate import splrep, splev

    rri = Results['Peaks']['Intervals'].tolist()
    del rri[-1] #the last value is NaN, so we delete it.
    rri_new = []
    for i in rri:
        bpm = 60/i
        rri_new.append(bpm)
    rri = rri_new
    
    t = Results['Peaks'].index.tolist()
    del t[-1] #must be the same length as rri
    
    #Evenly spaced time array using hz
    tx = arange(t[0], t[-1], 1.0 / hz)

    #Interpolate RRi serie
    tck = splrep(t, rri, s = 0)
    rrix = splev(tx, tck, der=0)

    #Number os estimations
    P = int((len(tx) - 256 / 128)) + 1 #AD doesn't know what this does, but i dare not touch a damn thing.

    #PSD with Welch's Method
    Fxx, Pxx = welch(rrix, fs=hz, window="hanning", nperseg=256, noverlap=128, detrend="linear")

    return Fxx, Pxx

def PSD_tct(results_bursts, hz):
    '''
    Power spectral density plot of total cycle time. this argument takes the results_burst dataframe
    '''
    #adopted form Rhenan Bartels Ferreira 
    #https://github.com/RhenanBartels/biosignalprocessing/blob/master/psdRRi.py

    from numpy import arange, cumsum, logical_and
    from scipy.signal import welch
    from scipy.interpolate import splrep, splev

    tct = Results['Bursts']['Total Cycle Time'].tolist()
    del tct[-1] #the last value is NaN, so we delete it.
    tct_new = []
    for i in tct:
        bpm = 60/i
        tct_new.append(bpm)
    tct = tct_new
    
    t = Results['Bursts']['Burst Start'].tolist()
    del t[-1] #must be the same length as rri
    
    #Evenly spaced time array using hz
    tx = arange(t[0], t[-1], 1.0 / hz)

    #Interpolate RRi serie
    tck = splrep(t, tct, s = 0)
    tctx = splev(tx, tck, der=0)

    #Number os estimations
    P = int((len(tx) - 256 / 128)) + 1 #AD doesn't know what this does, but i dare not touch a damn thing.

    #PSD with Welch's Method
    Fxx, Pxx = welch(tctx, fs=hz, window="hanning", nperseg=256, noverlap=128, detrend="linear")

    return Fxx, Pxx

def Results_PSD(Fxx, Pxx):
    '''
    Generates the result for peakdet interval PSD (PSD_rri)
    '''
    from scipy.integrate import simps

    results_psd = Series(index = ['ULF', 'VLF', 'LF','HF','LF/HF'])
    FXX = Series(Fxx)
    PXX = Series(Pxx)

    results_psd['ULF'] = simps(PXX[FXX<ulf].tolist(), FXX[FXX<ulf].tolist(), dx =dx)
    results_psd['VLF'] = simps(PXX[(ulf<FXX) & (FXX<=vlf)].tolist(),FXX[(ulf<FXX) & (FXX<=vlf)].tolist(), dx= dx)
    results_psd['LF'] = simps(PXX[(vlf<FXX) & (FXX<=lf)].tolist(),FXX[(vlf<FXX) & (FXX<=lf)].tolist(), dx= dx)
    results_psd['HF'] = simps(PXX[(lf<FXX) & (FXX<=hf)].tolist(),FXX[(lf<FXX) & (FXX<=hf)].tolist(), dx= dx)
    results_psd['LF/HF'] = results_psd['LF']/results_psd['HF']
    return results_psd
#
#Entropy
#
#
def histent(data_list):
    """
    Generates the Histogram Entropy value and bin array for a given list of values.
    Parameters
    ----------
    data_list: list
        list of float values.
    Returns
    -------
    HistEntropy: float
        value of the Histogram Entropy of the inputted list.
    binarray: 1d array
        the evenly spaced array of bins based on the extremes and length of the input.
    Notes
    -----
    could probably take other types of array objects
    if all of the samples fall in one bin regardless of the bin size
    means we have the most predictable sitution and the entropy is 0
    if we have uniformly dist function, the max entropy will be 1
    References
    -----
    Thanks to Farina and Irene for the code that this was based on.
    <Get Link or Reference>
    """ 
    NumBin = int(2 * (log(len(data_list), 2)))
    binarray = np.linspace(min(data_list), max(data_list), NumBin)
    no, xo = np.histogram(data_list, binarray); #print no
    no = no.astype(np.float) #or else they are all ints, which do stupid things when being divided

    # normalize the distribution to make area under distribution function unity
    no = no/sum(no); #print no

    # find the bins that have no samples to prevent log(0) in calculation
    no = no[np.nonzero(no)]    

    # if all of the samples fall in one bin regardless of the bin size
    # means we have the most predictable sitution and the entropy is 0
    # if we have uniformly dist function, the max entropy will be 1

    HistEntropy = [-1*(x * log(x, 2))/(log(NumBin,2)) for x in no]
    #print 'HistEntropy array=', HistEntropy
    HistEntropy = sum(HistEntropy)
    #print 'HistEntropy=', HistEntropy
    
    return HistEntropy, binarray

def histent_wrapper(event_type, meas, Data, Settings, Results):
    """
    Calculates histogram entropy value and plots histogram. this is a Wrapper to handle varibles in and out of histent() correctly. Takes two additional parameters that dictate which measurement will be executed. If meas is set to 'all', then all available measurements from the event_type chosen will be calculated iteratevely. 

    Parameters
    ----------
    event_type: string
        A string that should be either 'Peaks' or 'Bursts', which will tell the function 
        which results to pull the measurement from.
    meas: string
        A string that specifies which measurement type to use for the figure.
    Data: dictionary
        dictionary containing the pandas dataframes that store the data.
        this function uses the column names of the original data file to organize the 
        Histogram Entropy table.
    Settings: dictionary

    Results: dictionary
        The dictionary that contains all of the results. Results['Peaks'] and/or Results['Bursts']
        is used. Results['Peaks-Master'] or Results['Bursts-Master'] are also used to 
    
    Returns
    -------
    Results: dictionary
        The dictionary that contains all of the results. Results['Histogram Entropy'], a dataframe is created or added to.
    Notes
    -----
    A histogram is generated as well, and is saved directly to the plots folder stored in Settings['plots folder'].
    There is only one histogram entropy Dataframe, which is updated for each iteration of this function. It is 
    displayed and saved out automatically to Settings['output folder'].
    This function breaks the general rule of 'Three arguments in, Three arguments out.' Mostly because the 'event_type'
    and 'meas' are ment to be temporary varibles anyways. Saving them out doesn't make much sense.

    See histent for more information about what Histogram Entropy is.
    Examples
    --------
    event_type = 'Peaks'
    meas = 'Peaks Amplitude'
    histent_wrapper(event_type, meas, Settings, Results)
    Results['Histogram Entropy']
    """
    
    if 'Histogram Entropy' not in Results.keys():
        Results['Histogram Entropy'] = DataFrame(index = Data['original'].columns)
    
    if event_type.lower() == 'peaks':
        measurement = Results['Peaks']
        columns = Results['Peaks-Master']

    elif event_type.lower() == 'bursts':
        measurement = Results['Bursts']
        columns = Results['Bursts-Master']
    else:
        raise ValueError('Not an acceptable event type measurement.\n Must be "Peaks" or "Bursts" ')
    
    if meas.lower() == 'all':
        for name in columns:

            Results = histent_wrapper(event_type, name, Data, Settings, Results)

        print "All %s measurements analyzed." %(event_type)
        return Results

    else:
        temp_histent = Series(index = Data['original'].columns)
        for key, value in measurement.iteritems():
            
            temp_list = value[meas].tolist() #copy the correct array into a list
            
            try:
                HistEntropy, binarray = histent(temp_list)
                temp_histent[key] = HistEntropy
            except:
                HistEntropy = NaN
                binarray = []
                temp_histent[key] = HistEntropy
            
            try:
                plt.figure(1)
                plt.hist(temp_list,binarray)
                plt.xlabel('%s' %meas)
                plt.ylabel('Count')
                plt.title(r'%s-%s Histogram - %s' %(event_type, meas,key))
                plt.savefig(r'%s/%s-%s Histogram - %s.pdf'%(Settings['plots folder'], event_type,meas, key))
                plt.close()
            except:
                pass
            
        Results['Histogram Entropy'][meas] = temp_histent
        Results['Histogram Entropy'].to_csv(r'%s/%s_Histogram_Entropy.csv'
                                           %(Settings['Output Folder'], Settings['Label']))
        return Results

def ap_entropy_wrapper(event_type, meas, Data, Settings, Results):
    """
    Calculates approximate entropy value. this is a Wrapper to handle varibles in and out of ap_entropy() correctly. Takes two additional parameters that dictate which measurement will be executed. If meas is set to 'all', then all available measurements from the event_type chosen will be calculated iteratevely. 
    For the aprox ent call, M is set to 2 and R is 0.2*std(measurement). these values cannot be changed easily, but can be modified with the source code.
    Parameters
    ----------
    event_type: string
        A string that should be either 'Peaks' or 'Bursts', which will tell the function 
        which results to pull the measurement from.
    meas: string
        A string that specifies which measurement type to use for the figure.
    Data: dictionary
        dictionary containing the pandas dataframes that store the data.
        this function uses the column names of the original data file to organize the 
         Entropy table.
    Settings: dictionary

    Results: dictionary
        The dictionary that contains all of the results. Results['Peaks'] and/or Results['Bursts']
        is used. Results['Peaks-Master'] or Results['Bursts-Master'] are also used to 
    
    Returns
    -------
    Results: dictionary
        The dictionary that contains all of the results. Results['Approximate Entropy'], a dataframe is created or added to.
    Notes
    -----
    There is only one approximate entropy Dataframe, which is updated for each iteration of this function. It is 
    displayed and saved out automatically to Settings['output folder'].
    This function breaks the general rule of 'Three arguments in, Three arguments out.' Mostly because the 'event_type'
    and 'meas' are ment to be temporary varibles anyways. Saving them out doesn't make much sense.

    See ap_entropy for more information about what approximate Entropy is.
    Examples
    --------
    event_type = 'Bursts'
    meas = 'all'
    Results = ap_entropy_wrapper(event_type, meas, Data, Settings, Results)
    Results['Approximate Entropy']

    References
    ----------
    .. [1] Yentes et al., 2013. "The appropriate use of approximate entropy and sample entropy with short data sets." PMID: 23064819
    """
    try:
        import pyeeg
    except ImportError:
        print "You do not have the pyeeg module and cannot use this code." 
    
    if 'Approximate Entropy' not in Results.keys():
        Results['Approximate Entropy'] = DataFrame(index = Data['original'].columns)
    
    if event_type.lower() == 'peaks':
        measurement = Results['Peaks']
        columns = Results['Peaks-Master']

    elif event_type.lower() == 'bursts':
        measurement = Results['Bursts']
        columns = Results['Bursts-Master']
    else:
        raise ValueError('Not an acceptable event type measurement.\n Must be "Peaks" or "Bursts" ')
    
    if meas.lower() == 'all':
        for name in columns:

            Results = ap_entropy_wrapper(event_type, name, Data, Settings, Results)

        print "All %s measurements analyzed." %(event_type)
        return Results

    else:
        temp_apent = Series(index = Data['original'].columns)
        for key, value in measurement.iteritems():
            
            temp_list = value[meas].tolist() #copy the correct array into a list
            if temp_list[-1] != temp_list[-1]: #check if NaN
                temp_list = temp_list[:-1]
            try:
                ap_ent = ap_entropy(temp_list, 2, (0.2*np.std(temp_list)))
                temp_apent[key] = ap_ent
            except:
                ap_ent = NaN
                temp_apent[key] = ap_ent
            
        Results['Approximate Entropy'][meas] = temp_apent
        Results['Approximate Entropy'].to_csv(r'%s/%s_Approximate_Entropy.csv'
                                           %(Settings['Output Folder'], Settings['Label']))
        return Results

def samp_entropy_wrapper(event_type, meas, Data, Settings, Results):
    """
    Calculates approximate entropy value. this is a Wrapper to handle varibles in and out of samp_entropy() correctly. Takes two additional parameters that dictate which measurement will be executed. If meas is set to 'all', then all available measurements from the event_type chosen will be calculated iteratevely. 
    For the sample ent call, M is set to 2 and R is 0.2*std(measurement). these values cannot be changed easily, but can be modified with the source code.
    Parameters
    ----------
    event_type: string
        A string that should be either 'Peaks' or 'Bursts', which will tell the function 
        which results to pull the measurement from.
    meas: string
        A string that specifies which measurement type to use for the figure.
    Data: dictionary
        dictionary containing the pandas dataframes that store the data.
        this function uses the column names of the original data file to organize the 
         Entropy table.
    Settings: dictionary

    Results: dictionary
        The dictionary that contains all of the results. Results['Peaks'] and/or Results['Bursts']
        is used. Results['Peaks-Master'] or Results['Bursts-Master'] are also used to 
    
    Returns
    -------
    Results: dictionary
        The dictionary that contains all of the results. Results['Sample Entropy'], a dataframe is created or added to.
    Notes
    -----
    There is only one Sample entropy Dataframe, which is updated for each iteration of this function. It is 
    displayed and saved out automatically to Settings['output folder'].
    This function breaks the general rule of 'Three arguments in, Three arguments out.' Mostly because the 'event_type'
    and 'meas' are ment to be temporary varibles anyways. Saving them out doesn't make much sense.

    See samp_entropy for more information about what Sample Entropy is.
    Examples
    --------
    event_type = 'Bursts'
    meas = 'all'
    Results = samp_entropy_wrapper(event_type, meas, Data, Settings, Results)
    Results['Sample Entropy']

    References
    ----------
    .. [1] Yentes et al., 2013. "The appropriate use of approximate entropy and sample entropy with short data sets." PMID: 23064819
    """
    try:
        import pyeeg
    except ImportError:
        print "You do not have the pyeeg module and cannot use this code." 
    
    if 'Sample Entropy' not in Results.keys():
        Results['Sample Entropy'] = DataFrame(index = Data['original'].columns)
    
    if event_type.lower() == 'peaks':
        measurement = Results['Peaks']
        columns = Results['Peaks-Master']

    elif event_type.lower() == 'bursts':
        measurement = Results['Bursts']
        columns = Results['Bursts-Master']
    else:
        raise ValueError('Not an acceptable event type measurement.\n Must be "Peaks" or "Bursts" ')
    
    if meas.lower() == 'all':
        for name in columns:

            Results = samp_entropy_wrapper(event_type, name, Data, Settings, Results)

        print "All %s measurements analyzed." %(event_type)
        return Results

    else:
        temp_sampent = Series(index = Data['original'].columns)
        for key, value in measurement.iteritems():
            
            temp_list = value[meas].tolist() #copy the correct array into a list
            if temp_list[-1] != temp_list[-1]:
                temp_list = temp_list[:-1]
            try:
                samp_ent = samp_entropy(temp_list, 2, (0.2*np.std(temp_list)))
                temp_sampent[key] = samp_ent
            except:
                samp_ent = NaN
                temp_sampent[key] = samp_ent
            
        Results['Sample Entropy'][meas] = temp_sampent
        Results['Sample Entropy'].to_csv(r'%s/%s_Sample_Entropy.csv'
                                           %(Settings['Output Folder'], Settings['Label']))
        return Results

#
#Moments
#
#
def moving_statistics(event_type, meas, window, Data, Settings, Results):
    """
    Generates the moving mean, standard deviation, and count for a given measurement.
    Saves out the dataframes of these three results automatically with the window size in the name.
    If meas == 'All', then the function will loop and produce these tables for all measurements. 
    
    Parameters
    ----------
    event_type: string
        A string that should be either 'Peaks' or 'Bursts', which will tell the function 
        which results to pull the measurement from.
    meas: string
        A string that specifies which measurement type to use for the figure.
    window: float
        the size of the window for the moving statistics in seconds.
    Data: dictionary
        dictionary containing the pandas dataframes that store the data.
        this function uses the time index of the analyzed data to window over.
    Settings: dictionary
        dictionary that contains the user's settings.
        this function uses the ['Baseline Type'] and ['Output Folder'] to direct flow
    Results: dictionary
        The dictionary that contains all of the results
    
    Returns
    -------
    Results: dictionary
        the dictionary that contains all of the results. 
        A new dictionary is added called Results['Moving Stats'], which contains the dataframes called things like
        Results['Moving Stats']['Measurement-Count'], Results['Moving Stats']['Measurement-Mean'], and Results['Moving Stats']['Measurement-Std']. These files are both displayed and saved out in the loca
    
    Notes
    -----
    Moving statistics are a handy way to see how measurements are changing over time.
    Choose a window size that is scaled appropreately for your data. The larger the window,
    the coarser the averaging will be. 
    It seems self explaintory why mean and standardeviation are useful, but count is also included.
    This is to be able to calculate frequency in the window, if it is desired: count/window.
    
    Examples
    --------
    event_type = 'Peaks'
    meas = 'all'
    window = 60 #seconds
    Results = moving_statistics(event_type, meas, window, Data, Settings, Results)
    """
    #make Results dictionary if does not exsist
    if 'Moving Stats' not in Results.keys():
        Results['Moving Stats'] = {}
    
    #select and use the correct time 
    if Settings['Baseline Type'] == 'static':
        time_temp = Data['trans'].index
    elif Settings['Baseline Type'] == 'linear': 
        time_temp = Data['shift'].index
    elif Settings['Baseline Type'] == 'rolling':
        time_temp = Data['rolling'].index

    
    if event_type.lower() == 'peaks':
        dictionary = Results['Peaks']
        columns = Results['Peaks-Master']
    elif event_type.lower() == 'bursts':
        dictionary = Results['Bursts']
        columns = Results['Bursts-Master']
    else:
        raise ValueError('Not an acceptable event type measurement.\n Must be "Peaks" or "Bursts" ')
        
    if meas.lower() == 'all':
        
        for name in columns:
            Results = moving_statistics(event_type, name, window, Data, Settings, Results)
        print "All %s measurements analyzed." %(event_type)
        return Results
    
    else:
                
        import math
        #get number
        num = math.trunc(time_temp[-1]/window)
        #create data storage
        sliding_count = DataFrame(index= (np.arange(num)*window))
        sliding_mean = DataFrame(index= (np.arange(num)*window)) 
        sliding_std = DataFrame(index= (np.arange(num)*window))
        
        for key, value in dictionary.iteritems():
        
            temp_count = Series(index= (np.arange(num)*window)) 
            temp_mean = Series(index= (np.arange(num)*window))
            temp_std = Series(index= (np.arange(num)*window))

            for i in (np.arange(num)*window):
                temp_count[i] = value[meas][i:(i+window)].count() #get the count in the window
                temp_mean[i] = value[meas][i:(i+window)].mean() #get the mean in the window
                temp_std[i] = value[meas][i:(i+window)].std() #get the mean in the window

            temp_mean = temp_mean.fillna(0) #temp mean returns NaN for windows with no events. make it zero for graphing
            temp_std = temp_std.fillna(0)
            
            sliding_count[key] = temp_count #store series in results table
            sliding_mean[key] = temp_mean #store series in results table
            sliding_std[key] = temp_std
        
        #my attempt at reordering so the columns are in increaing order
        Results['Moving Stats'][r'%s-Count'%meas] = sliding_count.sort_index(axis = 1)
        Results['Moving Stats'][r'%s-Mean'%meas] = sliding_mean.sort_index(axis = 1) 
        Results['Moving Stats'][r'%s-Std'%meas] = sliding_std.sort_index(axis = 1)
        
        #autosave out these results
        Results['Moving Stats'][r'%s-Count'%meas].to_csv(r'%s/%s_%s_Count.csv'
                                       %(Settings['Output Folder'], Settings['Label'], meas))
        Results['Moving Stats'][r'%s-Mean'%meas].to_csv(r'%s/%s_%s_Mean.csv'
                                       %(Settings['Output Folder'], Settings['Label'], meas))
        Results['Moving Stats'][r'%s-Std'%meas].to_csv(r'%s/%s_%s_std.csv'
                                       %(Settings['Output Folder'], Settings['Label'], meas))
        
        print meas, 'Count'
        print Results['Moving Stats'][r'%s-Count'%meas]
        
        print meas, 'Mean'
        print Results['Moving Stats'][r'%s-Mean'%meas]
        
        print meas, 'Std'
        print Results['Moving Stats'][r'%s-Std'%meas]
        return Results

#
#Pipelines
#
#
def analyze(Data, Settings, Results):
    """
    The pipeline for event detection. Follows the strict '3 arguments in, 3 arguments out'
    rule. These dictionaries must be intalized before this, as well as all of the Settings values.
    Detects bursts and peaks for the data file that is uploaded.
    Settings used are saved out automatically with a time stamp as a receipt of each analysis performed. 

    Parameters
    ----------
    Data: dictionary
        an empty dictionary named Data.
    Settings: dictionary
        dictionary that contains the user's settings.
    Results: dictionary
        an empty dictionary named Results.
    
    Returns
    -------
    Data: dictionary
        Contains the DataFrames with the time series data. Keys define which version of the data it is.
    Settings: dictionary
        dictionary that contains the user's settings.
    Results: dictionary
        Contains the following objects:
            Peaks: dictionary
                keys are the column names from the Data DataFrames. objects are DataFrames that contain information about each peak detected, indexed by peak time.
            Peaks-Master: DataFrame
                multi-indexed DataFrame, created by concatenating all Peaks DataFrames. Column names and peak time are the two indexes. Automatically saved in the Settings['output folder'] location.
            Bursts: dictionary
                keys are the column names from the Data DataFrames. objects are DataFrames that contain information about each burst detected. has an arbitrary index, which can be roughly thought of as burst number.
            Bursts-Master: DataFrame
                multi-indexed DataFrame, created by concatenating all Bursts DataFrames. Column names and burst number are the two indexes. Automatically saved in the Settings['output folder'] location.
            

    Notes
    -----
    This function is the top level function of the bass pipeline. 
    It has a few handy printed outputs, such as how long an analysis took, which step was just completed, lists of which objects contained no peaks or bursts. it also prints a list of key names and analysis measurements, which can be used in further analysis steps.
    
    """
    start = t.clock()
    #Load
    Data, Settings = load_wrapper(Data, Settings)
        
    #transform data
    Data, Settings = transform_wrapper(Data, Settings)
    print 'Transformation completed'

    #set baseline
    Data, Settings, Results = baseline_wrapper(Data, Settings, Results)
    print 'Baseline set completed'

    #run peak detection
    Results = event_peakdet_wrapper(Data, Settings, Results)
    print 'Peak Detection completed'

    #run burst detection
    Results = event_burstdet_wrapper(Data, Settings, Results)
    print 'Burst Detection completed'

    #Save all the graphs
    if Settings['Generate Graphs'] == True:
        for label, col in Data['original'].iteritems():
            graph_detected_events_save(Data, Settings, Results, 
                                  roi = label, lcpro = Settings['Graph LCpro events'])
        print "Graphs Saved"

    #Save master files 
    Results['Peaks-Master'].to_csv(r'%s/%s_Peak_Results.csv'
                                   %(Settings['Output Folder'], Settings['Label']))
    Results['Bursts-Master'].to_csv(r'%s/%s_Bursts_Results.csv'
                                    %(Settings['Output Folder'], Settings['Label']))

    #Save Master Summary Files
    burst_grouped = Results['Bursts-Master'].groupby(level=0)
    burst_grouped = burst_grouped.describe()
    burst_grouped.to_csv(r'%s/%s_Bursts_Results_Summary.csv'
                                           %(Settings['Output Folder'], Settings['Label']))
    
    peak_grouped = Results['Peaks-Master'].groupby(level=0)
    peak_grouped= peak_grouped.describe()
    peak_grouped.to_csv(r'%s/%s_Peaks_Results_Summary.csv'
                                           %(Settings['Output Folder'], Settings['Label']))

    #save settings
    Settings_panda = DataFrame.from_dict(Settings, orient='index')
    now = datetime.datetime.now()
    colname = 'Settings: ' + str(now)
    Settings_panda.columns = [colname]
    Settings_panda = Settings_panda.sort()
    Settings_panda.to_csv(r"%s/%s_Settings_%s.csv"%(Settings['Output Folder'], 
                                                 Settings['Label'], 
                                                 now.strftime('%Y_%m_%d__%H_%M_%S')))

    end = t.clock()
    run_time = end-start
    print "Analysis Complete: ", np.round(run_time,4), " Seconds"
    
    print "\n--------------------------------------------"

    print "Data Column Names/Keys"
    print "-----"
    for name in Data['original']:
        print name
    print "\n--------------------------------------------"
    print "Available Measurements from Peaks for further analysis:"
    print "-----"
    for label, col in Results['Peaks-Master'].iteritems():
        print label
    print "\n--------------------------------------------"
    print "Available Measurements from Bursts for further analysis:"
    print "-----"
    for label, col in Results['Bursts-Master'].iteritems():
        print label
    
    print "\n---------------------------"
    print '|Event Detection Complete!|'
    print "---------------------------"
    return Data, Settings, Results


#END OF CODE
print "BASS ready!"