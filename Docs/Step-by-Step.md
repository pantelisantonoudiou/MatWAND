## MatWAND data processing

This is a step by step tutorial on how to use MatWAND to analyze LFP/EEG binary files exported from labchart. 
This tutorial assumes that the LFP/EEG binary files have been already properly named and placed in a data folder (clock here for more info).

---

1) Initialize analysis.

![Banner](/Images/tutorial/init.png)

---

2) Choose file and analysis parameters. 

In this example our files contain 3 channels ('bla', 'pfc', 'emg'). The channel names have to be separated with a semicolon( **;** ) as can be seen in the *Channel Structure* field. We chose to analyze the 'bla' channel as denoted in the *Channel Analyzed* field. Only one of the channels from the *Channel Structure* field can be chosen. 

![Banner](/Images/tutorial/input_parameters_gui.png)

We choose our file type from the *File Type* field. In this example we selected *adibin* as we are analyzing binary data from labchart.

![Banner](/Images/tutorial/file_type.png)

---

3) Choose the folder where LFP/EEG binary files are present. 

It is recommended that this folder is under a parent folder. In this example the .adibin are placed under a the parent directory *test_mat*

![Banner](/Images/tutorial/load_raw_data.png)

If 

---



2) Convert voltage traces to power spectral density (PSD) using the [fft transform](https://www.mathworks.com/help/signal/ug/power-spectral-density-estimates-using-fft.html) and store to "raw_psd" folder.

+ Automatic separation of .mat files based on comment times.

3) Pre-process the analyzed PSD data  and store to "processed_psd" folder.

4) Plot data from the "processed_psd" folder.




**[<< Back to Main Page](/README.md)**

