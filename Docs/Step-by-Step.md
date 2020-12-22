## MatWAND data processing



Here follows a step by step tutorial on how to use MatWAND to analyze binary files from labchart. 
This tutorial assumes that the LFP/EEG files have been already properly named and placed in a data folder (clock here for more info).

---

1) Initialize analysis.

![Banner](/Images/tutorial/init.png)

---

2) Enter parameters. 

![Banner](/Images/tutorial/input_parameters_gui.png)

In this case our files contain 3 channels each ('bla', 'pfc', 'emg') and we chose to analyze the 'bla' channel.

We also denoted that out files are binary with .adibin extension.

![Banner](/Images/tutorial/file_type.png)

---

2) Convert voltage traces to power spectral density (PSD) using the [fft transform](https://www.mathworks.com/help/signal/ug/power-spectral-density-estimates-using-fft.html) and store to "raw_psd" folder.

+ Automatic separation of .mat files based on comment times.

3) Pre-process the analyzed PSD data  and store to "processed_psd" folder.

4) Plot data from the "processed_psd" folder.




**[<< Back to Main Page](/README.md)**

