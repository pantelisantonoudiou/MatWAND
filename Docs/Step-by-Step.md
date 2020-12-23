## MatWAND Tutorial

- This is a step-by-step tutorial on how to use MatWAND to analyze LFP/EEG data.
- This example will focus on analysis of binary files exported from [labchart](https://www.adinstruments.com/products/labchart).
- This tutorial assumes that the binary files have been properly named and placed in a data folder (click here for more info).

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

- Before the conversion begins, the user is asked whether to separate files based on comments (provided in matlab files [pipeline](/Images/MatWAND_pipeline.svg)). 

![Banner](/Images/tutorial/separate_conditions.png)

- After data conversion the user is prompted for time-locked trimming of the files. This is essential for time plots. 
- More details on the trimming procedure can be found [here]().

![Banner](/Images/tutorial/time_lock_trim.png)

---

#### 5) Pre-process analyzed data (PSD or power spectral density)

- When the STFT conversion is completed the user has to proceed by clicking the ***Power Spectral Density processing*** purple button.
- A menu will pop up with processing options (options are set to ***no*** when initialized).
- ***Change Bin Size***: merges original fft bins (default = 5 seconds) into larger bins (default = 300 seconds when activated).
- ***Normalize***: Normalizes the PSDs based on choice e.g. log(e) or log(10)
- ***Linearize***: Linearize PSD by fitting a two-term exponential (not recommended).
- ***Remove Noise***: Replaces PSD frequency using [pchip](https://www.mathworks.com/help/matlab/ref/pchip.html) (default = 60 +/- 1 Hz when activated).
- ***Remove Outliers***: Replaces outliers that are ***n*** times larger or smaller the median value of all power (default = 5 when activated).

![Banner](/Images/tutorial/psd_process.png)

- The last step in the analysis pipeline is to choose the conditions for plotting. 
- The conditions are automatically detected and displayed based on file naming.

![Banner](/Images/tutorial/cond_choice.png)

---

#### 6) Plot or Export

- spectrograms
- PSDs
- Time Plots
- Summary/Average Plots

---

**[<< Back to Main Page](/README.md)**

