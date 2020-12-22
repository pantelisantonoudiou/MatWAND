## MatWAND data processing



Here follows a step by step tutorial on how to use MatWAND to analyze binary files from labchart. 
This tutorial assumes that the LFP/EEG files have been already properly named and placed in a data folder (clock here for more info).

---

1) Initialize analysis.

![Banner](/Images/tutorial/init.png)

---

2) Choose file and analysis parameters. 

![Banner](/Images/tutorial/input_parameters_gui.png)

In this case our files contain 3 channels each ('bla', 'pfc', 'emg'). The channel names need to be separated have to be separated with a semicolon(;) as can be seen in the 'Channel Structure' field. Here we chose to the 'bla' channel as denoted in the 'Channel Analyzed' field.

We also denoted that out files are binary with .adibin extension.

![Banner](/Images/tutorial/file_type.png)

---

2) Convert voltage traces to power spectral density (PSD) using the [fft transform](https://www.mathworks.com/help/signal/ug/power-spectral-density-estimates-using-fft.html) and store to "raw_psd" folder.

+ Automatic separation of .mat files based on comment times.

3) Pre-process the analyzed PSD data  and store to "processed_psd" folder.

4) Plot data from the "processed_psd" folder.




**[<< Back to Main Page](/README.md)**

