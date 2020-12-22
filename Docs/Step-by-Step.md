## MatWAND Tutorial

This is a step by step tutorial on how to use MatWAND to analyze LFP/EEG binary files exported from labchart. 
This tutorial assumes that the LFP/EEG binary files have been already properly named and placed in a data folder (click here for more info).

---

#### 1) Initialize analysis.

![Banner](/Images/tutorial/init.png)

---

#### 2) Choose file and analysis parameters. 

- In this example our files contain 3 channels ('bla', 'pfc', 'emg'). 
- The channel names have to be separated with a semicolon( **;** ) as can be seen in the ****Channel Structure*** field. 
- Here the 'bla' channel has been selected for analysis. The channel chosen is selected from ***Channel Analyzed*** field. 
- Only one channel from the ***Channel Structure*** field can be chosen (for this example 'bla', 'pfc', 'emg'). 

![Banner](/Images/tutorial/input_parameters_gui.png)

- We choose our file type from the ***File Type*** field. 
- The only file types currently supported are matlab and binary files. 
- In this example we selected ***adibin*** as we are analyzing binary data from labchart with ***single*** datatype. 
- The option ***other*** allows to choose the file extension, scaling and [data type](/Docs/Inputs.md) of binary files.

![Banner](/Images/tutorial/file_type.png)

---

#### 3) Choose the data folder where LFP/EEG binary files are present. 

- After the properties are chosen, the user will be prompted to choose a data folder. 
- It is recommended that the data folder is under a parent folder as MatWAND will subsequently create an analysis folder. 
- In this example the .adibin are situated in the ***raw_data*** folder which is placed under the ***test_mat*** parent directory.

![Banner](/Images/tutorial/load_raw_data.png)

- After the folder is loaded a file check is run on the background to determine if the files have been named properly. 
- If the file check passes successfully then the MatWAND status bar will show that the folder was loaded. Click [here](/Docs/Inputs.md) to see how to name and arrange files.

![Banner](/Images/tutorial/gui_raw_data_loaded.png)

---

#### 4) Initiate spectral analysis (STFT or Short-time Fourier transform)

- When the *Get FFT* button is toggled the LFP/EEG data will be converted to spectrograms and stored under the analysis folder.
- The analysis folder is named after the chosen channel. In this example ***analysis_bla***.
- Detailed description of the STFT algorithm can be found [here](https://www.mathworks.com/help/signal/ug/power-spectral-density-estimates-using-fft.html).

![Banner](/Images/tutorial/fft_progress.png)

- Before the conversion begins, the user is asked whether to separate files based on comments (provided in matlab files [pipeline](/Images)). 

![Banner](/Images/tutorial/separate_conditions.png)

- After data conversion the user is prompted for time-locked trimming of the files. This is essential for time plots. 
- More details on the trimming procedure can be found [here]().

![Banner](/Images/tutorial/time_lock_trim.png)
---

#### 5) Pre-process analyzed data

- When the STFT conversion is completed the user has to proceed by clicking on ***Power Spectral Density processing***.

![Banner](/Images/tutorial/psd_process.png)



**[<< Back to Main Page](/README.md)**

