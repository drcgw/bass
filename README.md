SWAN
====

This is the beta SWAN release from the Dr. Chris G Wilson Neuro-Physiology lab. 
SWAN is the Single Wave Analysis Notebook. This interactive notebook is designed to analyize biomedical time series data, like ECG and Nerve recordings. 

RELEASE log
===
Beta 1.0: 01-27-2015

New Features:
- Load Settings Features 
- Rolling baseline/thresholding 

Updates:
- Refactored the way all data is stored to safer dictionaries than top level varibles. 
- If savitzky-golay is selected, absolute value is automaticaly turned on
- Settings for event detection are now their own blocks, so they can be optionally run (esp in the case of data loaded in). 
- Settings will display previously selected value, if present.
- Histogram Entropy is now safely wrapped. 
- Histogram Entropy data can now be saved. 

Beta 0.2: 11-13-2014

New Features: 
- Power in band for freqency plots, FFT graphs, and tables
- Histogram Entropy

Updates:
- All line plots are now always the same amplitude
- No longer asks for user to input sampling rate, SWAN checks for you
- Fixed bug in gHRV outputs that exported the wrong rr-interval file.

Pre-Beta: 10-16-2014
