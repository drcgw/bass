BASS
====

This is the Biomedical Analysis Software Suite (BASS) Python Module and accompanying IPython notebooks. Each notebook is a variant of a data analytics pipeline that detects and measure events within a time series. The underlying architecture makes it easy to customize the parts of your detection pipeline as well as pick from a menu of analysis options. 

Basic Features
-----------------
Signal Processing: De-trending (linear subtraction), Bandpass filters, Savitzky-Golay Filter

Event Detection: Peaks and boundaries (Bursts)

Event Measurements: Peak amplitude, Peak-Peak Intervals, Duration, Interburst interval, Total Cycle 

Time, Peaks per Burst, Intraburst Frequency, Burst Area, Attack, Decay.

Event Analysis: Descriptive Statistics, Histogram Entropy, Approximate Entropy, Sample entropy, Poincare Plots, Moving averages, Power Spectral Density, Frequency.

How to use
--------------
1. Install Python on your computer. We recommend the Enthought Canopy distribution but Anaconda should work fine also. 

2. Open a terminal window (Mac, Linux) or command prompt (cmd, Windows).

3. Type `ipython notebook` then press `Enter`.

4. The notebook should launch automatically in a web browser window. Make sure that the window is not internet explorer (chrome is ideal, firefox is fine). If the page doesn’t load, look for the line in the terminal window that reads: `The IPython Notebook is running at: http://###.#.#.#:8888/‘`. You can copy and paste the address into the address bar of your web browser.

4. Do not launch from Canopy or other gui-based python environments!

5. Navigate through your folder tree to where you have saved the `Single Wave-Interactive.ipynb` file. Click on it to open it. The `bass.py` and `pyeeg.py` files MUST be in the same folder as the Ipython Notebooks.

6. `Single Wave-Interactive.ipynb` is intended for new and basic users. `Single Wave-Basic.ipynb` is intended for advanced users or users who already know exactly what their analysis settings will be. `Kitchen Sink.ipynb` is intended for developers or superusers.

How to Modify
------------------
Stay tuned, Developers. 

Secret Tricks
----------------
Coming Soon

