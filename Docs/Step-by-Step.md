## MatWAND Tutorial

This is a step by step tutorial on how to use MatWAND to analyze LFP/EEG binary files exported from labchart. 
This tutorial assumes that the LFP/EEG binary files have been already properly named and placed in a data folder (click here for more info).

---

1) Initialize analysis.

![Banner](/Images/tutorial/init.png)

---

2) Choose file and analysis parameters. 

In this example our files contain 3 channels ('bla', 'pfc', 'emg'). The channel names have to be separated with a semicolon( **;** ) as can be seen in the *Channel Structure* field. We chose to analyze the 'bla' channel as denoted in the *Channel Analyzed* field. Only one of the channels from the *Channel Structure* field can be chosen. 

![Banner](/Images/tutorial/input_parameters_gui.png)

We choose our file type from the *File Type* field. The only file types currently supported are matlab and binary files. In this example we selected *adibin* as we are analyzing binary data from labchart with *single* datatype. The option *other* allows to choose the file extension, scaling and [data type](Docs/Inputs.md) of binary files.

![Banner](/Images/tutorial/file_type.png)

---

3) Choose the data folder where LFP/EEG binary files are present. 

It is recommended that the data folder is under a parent folder. In this example the .adibin are situated in the *raw_data* folder which is placed under the *test_mat* parent directory.

![Banner](/Images/tutorial/load_raw_data.png)

After the folder is loaded a file check is run on the background to determine if the files have been named properly. 
If the file check passes succesfully then the MatWAND status bar will show that the folder was loaded. Click here to see how to name and save files.

![Banner](/Images/tutorial/load_raw_data.png)

---



2) Convert voltage traces to power spectral density (PSD) using the [fft transform](https://www.mathworks.com/help/signal/ug/power-spectral-density-estimates-using-fft.html) and store to "raw_psd" folder.

+ Automatic separation of .mat files based on comment times.

3) Pre-process the analyzed PSD data  and store to "processed_psd" folder.

4) Plot data from the "processed_psd" folder.




**[<< Back to Main Page](/README.md)**

